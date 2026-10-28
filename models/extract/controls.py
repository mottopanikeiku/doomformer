"""Negative controls of the static extraction: change one thing in a compiled map, read the map back and list how the
recovered net differs from the source net.

    python models/extract/controls.py --fuzz-seed 0 --index 12 [--workdir DIR]
    python models/extract/controls.py --net models/mnist/mlp.npz --workdir DIR

Each control writes a tampered copy of the .wad and prints: the edit, the differences `Netlist.diff` finds against the
source net, and the structural findings. Exit status 1 when a control is not detected or differs in more places than
the edit explains.
"""
from __future__ import annotations

import argparse
import re
import struct
import sys
import tempfile
from pathlib import Path

from doomformer import extract as ex
from doomformer.kit import MLP
from doomformer.kit import compile as df_compile
from doomformer.mutate import read_lumps, write_lumps


def edit_lines(src: Path, dst: Path, edits: dict[tuple[str, int], callable]) -> None:
    """dst := src with the i-th `kind` block of TEXTMAP replaced by fn(text) for every ((kind, i), fn) in edits."""
    lumps = read_lumps(src)
    out = []
    for name, blob in lumps:
        if name == "TEXTMAP":
            counts: dict[str, int] = {}
            lines = blob.decode().split("\n")
            for n, ln in enumerate(lines):
                kind = ln.split(" ", 1)[0]
                i = counts.get(kind, 0)
                counts[kind] = i + 1
                fn = edits.get((kind, i))
                if fn is not None:
                    lines[n] = fn(ln)
            blob = "\n".join(lines).encode()
        out.append((name, blob))
    write_lumps(dst, out)


def setkv(ln: str, key: str, fn) -> str:
    return re.sub(rf"\b{key} = (-?\d+);", lambda m: f"{key} = {fn(int(m.group(1)))};", ln, count=1)


def lane_info(wad: Path):
    """The extractor's own view of the map: TextMap, dolls per layer (sectors, lines) for choosing what to edit."""
    lumps = ex.read_lumps(wad)
    tm = ex.TextMap(lumps["TEXTMAP"].decode())
    layer_of = {}
    barriers = [s for s in range(tm.n_sectors) if s in tm.nbr and (tm.tag[s] == ex.START_TAG
                                                                    or ex.STAGE_TAG0 < tm.tag[s] < ex.STAGE_TAG0 + 4000)]
    tags = sorted({tm.tag[s] for s in barriers})
    layer_of = {t: i for i, t in enumerate(tags)}
    dolls: dict[int, list[ex.Doll]] = {}
    for b in barriers:
        lane = ex._read_lane(tm, b, layer_of[tm.tag[b]], {})
        if lane is not None:
            dolls.setdefault(lane.doll.layer, []).append(lane.doll)
    return tm, dolls


def line_index(tm: ex.TextMap, a: int, b: int, kinds: set[str], tag: int | None = None) -> int:
    for i in tm.edge.get((a, b) if a < b else (b, a), []):
        e = ex._effect(tm.lines[i])
        if e is not None and e[0] in kinds and (tag is None or e[1] == tag):
            return i
    raise LookupError("no such line")


def controls(net: MLP, wad: Path, work: Path, lane: str = "last") -> list[dict]:
    """Run every control on `wad` (compiled from `net`); returns one record per control. The edited lane of each port is
    the one with the smallest threshold (lane="last": reached by the most inputs) or the largest (lane="first")."""
    tm, dolls = lane_info(wad)
    L = len(dolls)
    last = L - 1
    # an edit site: a light line of a final-layer doll whose target is not the lowest output light (a line that leaves
    # the lowest light without any line would make the extractor drop that row, which is a documented limit)
    order = sorted(dolls[last], key=lambda d: (d.port, d.limit))
    order = order if lane == "first" else sorted(order, key=lambda d: (d.port, -d.limit))
    tmin = min(t for dd in dolls[last] for k, t, _ in dd.lines if k == "light")
    found = None
    for dd in order:
        for a, b in zip(dd.sectors, dd.sectors[1:]):
            for i in tm.edge.get((a, b) if a < b else (b, a), []):
                e = ex._effect(tm.lines[i])
                if e is not None and e[0] == "light" and (found is None or found[2] == tmin) and \
                        (found is None or e[1] != tmin):
                    found = (dd, i, e[1])
        if found is not None and found[2] != tmin:
            break
    d, li, tag_a = found
    chain = d.sectors
    other = tmin if tag_a != tmin else next(t for t in sorted({t for dd in dolls[last] for k, t, _ in dd.lines
                                                              if k == "light"}) if t != tag_a)
    recs = []

    def run(name: str, what: str, edits: dict, expect: int | None = None, expect_what: set | None = None):
        dst = work / f"{name}.wad"
        edit_lines(wad, dst, edits)
        nl = ex.extract(dst)
        diffs = nl.diff(net)
        rec = dict(name=name, edit=what, differences=[str(x) for x in diffs], findings=nl.findings, n=len(diffs))
        ok = bool(diffs or nl.findings)
        if expect is not None and len(diffs) != expect:
            ok = False
        if expect_what is not None and {x.what for x in diffs} != expect_what:
            ok = False
        rec["ok"] = ok
        recs.append(rec)

    sp = tm.lines[li][2]
    run("weight_plus1", f"final layer: line {li} (special {sp}, tag {tag_a}) amount +1",
        {("linedef", li): lambda ln: setkv(ln, "arg1", lambda v: v + 1)}, 1, {"weight"})
    run("sign", f"final layer: line {li} special {sp} -> {ex.LIGHT_LOWER if sp == ex.LIGHT_RAISE else ex.LIGHT_RAISE}",
        {("linedef", li): lambda ln: setkv(ln, "special", lambda v: ex.LIGHT_LOWER if v == ex.LIGHT_RAISE else ex.LIGHT_RAISE)},
        1, {"weight"})
    run("retarget", f"final layer: line {li} tag {tag_a} -> {other}",
        {("linedef", li): lambda ln: setkv(ln, "arg0", lambda v: other)}, 2, {"weight"})
    # gate height: every sector of the lane at the walkway floor moves up by one
    f0 = tm.floor[chain[0]]
    freeze = chain[5]                      # shared by every lane of the layer at this walkway height: left alone
    sec_edits = {("sector", s): (lambda ln: setkv(ln, "heightfloor", lambda v: v + 1))
                 for s in chain if tm.floor[s] == f0 and s != freeze}
    run("gate_height", f"final layer, port {d.port}: the lane's walkway floor {f0} -> {f0 + 1} ({len(sec_edits)} sectors)",
        sec_edits, 1, {"threshold"})
    pad = chain[3]
    run("pad_height", f"final layer, port {d.port}: only the pad sector's floor {f0} -> {f0 + 1}",
        {("sector", pad): lambda ln: setkv(ln, "heightfloor", lambda v: v + 1)}, 1, {"threshold"})
    out_sector = next(s for s in range(tm.n_sectors) if tm.tag[s] == tag_a and tm.isolated[s])
    run("output_bias", f"output light {tag_a}: start level {tm.light[out_sector]} -> {tm.light[out_sector] + 1}",
        {("sector", out_sector): lambda ln: setkv(ln, "lightlevel", lambda v: v + 1)}, 1, {"bias"})
    if L > 1:
        # hidden layer 0: a doll with a plane line and its twin light line
        order0 = sorted(dolls[0], key=lambda d: (d.port, d.limit if lane == "first" else -d.limit))
        h = next(dd for dd in order0 if any(k == "light" for k, _, _ in dd.lines))
        hc = h.sectors
        sites = {}
        for a, b in zip(hc[2:], hc[3:]):
            for kind in ("plane", "light"):
                try:
                    sites.setdefault(kind, line_index(tm, a, b, {kind}))
                except LookupError:
                    pass
        lj = sites["light"]
        if "plane" in sites:                      # opt 0 to 2: a hidden unit's plane line and its twin light line
            pi = sites["plane"]
            run("hidden_plane_line", f"layer 0: the plane line {pi} amount +1 (twin light line unchanged)",
                {("linedef", pi): lambda ln: setkv(ln, "arg2", lambda v: v + 1)}, 1, {"weight"})
            run("hidden_twin_line", f"layer 0: the twin light line {lj} amount +1 (plane line unchanged)",
                {("linedef", lj): lambda ln: setkv(ln, "arg1", lambda v: v + 1)}, 1, {"twin weight"})
            pl_tag = tm.lines[pi][3]
        else:                                     # opt 3: the plane is copied from the light by an ACS script
            run("hidden_light_line", f"layer 0: the unit's light line {lj} amount +1 (the copy script feeds the plane)",
                {("linedef", lj): lambda ln: setkv(ln, "arg1", lambda v: v + 1)}, 1, {"weight"})
            pl_tag = next(t for t in sorted({t for dd in dolls[1] for t in [dd.port]}))
        gate_secs = [s for s in range(tm.n_sectors) if tm.tag[s] == pl_tag and not tm.isolated[s]]
        run("hidden_plane_start", f"layer 0 unit: gate sectors of port {pl_tag}: start height +1",
            {("sector", s): (lambda ln: setkv(ln, "heightfloor", lambda v: v + 1)) for s in gate_secs}, 1, {"bias"})
    scripts = ex.acs_scripts(ex.read_lumps(wad)["BEHAVIOR"])
    if 30 in scripts:                             # opt 3: retarget one copy of layer 0's script (ACS bytecode)
        beh = ex.read_lumps(wad)["BEHAVIOR"]
        off = scripts[30][0]
        calls = [a for sp, a in ex.run_script(beh, off, lights=lambda t: 1000 + t) if sp == ex.FLOOR_MOVE_TO]
        first, second = calls[0][0], calls[1][0]
        pat = b"\x03" + struct.pack("<i", first)
        at = beh.index(pat, off)
        patched = beh[:at] + b"\x03" + struct.pack("<i", second) + beh[at + len(pat):]
        dst = work / "copy_retarget.wad"
        write_lumps(dst, [(n, patched if n == "BEHAVIOR" else b) for n, b in read_lumps(wad)])
        nl = ex.extract(dst)
        diffs = nl.diff(net)
        recs.append(dict(name="copy_retarget", edit=f"ACS script 30: the copy that sets plane {first} sets plane {second} "
                         "instead", differences=[str(x) for x in diffs], findings=nl.findings, n=len(diffs),
                         ok=bool(diffs or nl.findings)))
    return recs


def linedef_index(text: str, special: int, n: int = 0) -> int:
    """Index (among the linedef blocks of a TEXTMAP) of the n-th linedef with this special."""
    k = -1
    for ln in text.split("\n"):
        if ln.startswith("linedef"):
            k += 1
            if re.search(rf"special = {special};", ln):
                if n == 0:
                    return k
                n -= 1
    raise LookupError(f"no linedef with special {special}")


def mechanics_controls(wad: Path, work: Path, net: MLP) -> list[dict]:
    """Controls for the fields that decide whether the dolls run at all: trigger and repeat flags of each line class,
    mover speeds, a teleport destination, the reset script's light tag. The netlist stays equal; a finding is expected."""
    lumps = read_lumps(wad)
    text = dict(lumps)["TEXTMAP"].decode()
    beh = dict(lumps)["BEHAVIOR"]
    recs = []

    def run(name: str, what: str, edits: dict = None, behavior: bytes | None = None, expect: str = ""):
        dst = work / f"{name}.wad"
        if edits is not None:
            edit_lines(wad, dst, edits)
        else:
            write_lumps(dst, [(n, behavior if n == "BEHAVIOR" else b) for n, b in lumps])
        try:
            nl = ex.extract(dst)
        except ex.ExtractError as e:
            recs.append(dict(name=name, edit=what, differences=[], findings=[f"refused: {e}"], n=0, ok=True))
            return
        finds = nl.findings
        ok = bool(finds) and any(expect in f for f in finds) and not nl.diff(net)
        recs.append(dict(name=name, edit=what, differences=[str(x) for x in nl.diff(net)], findings=finds, n=0, ok=ok))

    def drop(flag):
        return lambda ln: ln.replace(f"{flag} = true;", f"{flag} = false;", 1)

    for name, special, flag in (("release_playercross_dropped", ex.LOWER_TO_LOWEST, "playercross"),
                                ("release_repeat_dropped", ex.LOWER_TO_LOWEST, "repeatspecial"),
                                ("weight_repeat_dropped", ex.LIGHT_RAISE, "repeatspecial"),
                                ("script_start_playercross_dropped", ex.EXEC_ALWAYS, "playercross"),
                                ("freeze_playercross_dropped", ex.FLOOR_MOVE_TO, "playercross")):
        try:
            i = linedef_index(text, special)
        except LookupError:
            continue
        run(name, f"linedef {i} (special {special}): {flag} true -> false", {("linedef", i): drop(flag)}, expect="trigger flags")
    for name, special, arg in (("release_speed", ex.LOWER_TO_LOWEST, "arg1"), ("freeze_height", ex.FLOOR_MOVE_TO, "arg2"),
                               ("weight_speed", ex.FLOOR_RAISE, "arg1")):
        try:
            i = linedef_index(text, special)
        except LookupError:
            continue
        run(name, f"linedef {i} (special {special}): {arg} {'-> 1' if arg == 'arg1' else '+1'}", {("linedef", i): lambda ln, a=arg: setkv(ln, a, lambda v: 1 if a == "arg1" else v + 1)},
            expect="line")
    # a teleport destination: the doll-home destination thing gets another id
    tm = ex.TextMap(text)
    k = next(k for k, t in enumerate(tm.things) if t[2] == ex.TELEPORT_DEST and t[3] >= 30000)
    run("reset_destination_moved", f"thing {k} (teleport destination {tm.things[k][3]}): id +1",
        {("thing", k): lambda ln: setkv(ln, "id", lambda v: v + 1)}, expect="destination")
    # the reset script restores another light than it should
    calls = ex.run_script(beh, ex.acs_scripts(beh)[3][0])
    tag = next(a[0] for sp, a in calls if sp == ex.LIGHT_CHANGE_TO)
    pat = b"\x03" + struct.pack("<i", tag)
    at = beh.index(pat, ex.acs_scripts(beh)[3][0])
    run("reset_light_tag", f"ACS script 3: Light_ChangeToValue tag {tag} -> {tag + 1}",
        behavior=beh[:at] + b"\x03" + struct.pack("<i", tag + 1) + beh[at + 5:], expect="reset script")
    # a hidden bias outside the plane contract: the netlist equals the net, the contract fails
    return recs


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--net", help="an MLP .npz (default: a fuzz net)")
    ap.add_argument("--fuzz-seed", type=int, default=0)
    ap.add_argument("--index", type=int, default=12)
    ap.add_argument("--opt", type=int, default=0)
    ap.add_argument("--lane", choices=("first", "last"), default="last",
                    help="edit the lane with the largest (first) or smallest (last) threshold of each port")
    ap.add_argument("--workdir")
    args = ap.parse_args(argv)
    if args.net:
        net, name = MLP.load(args.net), args.net
    else:
        from doomformer import fuzz
        net = fuzz.net_of("mlp", fuzz.gen_case("mlp", args.fuzz_seed, args.index)["spec"])
        name = f"fuzz seed {args.fuzz_seed} index {args.index}"
    work = Path(args.workdir or tempfile.mkdtemp(prefix="extract-controls-"))
    work.mkdir(parents=True, exist_ok=True)
    wad = work / "clean.wad"
    df_compile(net, wad, mode="map", opt=args.opt, limits=False).close()
    nl = ex.extract(wad)
    clean = nl.diff(net)
    print(f"net: {name}, widths {[net.n_in] + [W.shape[0] for W in net.weights]}, opt {args.opt}")
    print(f"clean map: {len(clean)} differences, {len(nl.findings)} findings")
    bad = bool(clean or nl.findings)
    for r in controls(net, wad, work, args.lane) + mechanics_controls(wad, work, net):
        print(f"{r['name']}: {r['edit']}")
        print(f"  {r['n']} difference(s): " + ("; ".join(r["differences"][:4]) or "none"))
        for f in r["findings"][:3]:
            print(f"  finding: {f}")
        print("  detected" if r["ok"] else "  NOT DETECTED AS EXPECTED")
        bad |= not r["ok"]
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
