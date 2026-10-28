"""Mutation harness: how much of a compiled map does the check actually test?

`doomformer fuzz --mutate` compiles a small MLP, then mutates the compiled map one site at a time and runs the net's
inputs through each mutant against the integer reference. A mutant is KILLED when any pass differs from the
reference (or the game fails to settle, crashes or hits the wall-clock limit: a broken map is not a passing map) and
SURVIVES when every pass is exact. Mutation sites:

  TEXTMAP  every linedef special argument (weights, targets, mover speeds, teleports, TIDs, scrollers), special
           numbers (raise <-> lower), sector floor / ceiling heights and light levels, thing ids;
  ACS      every integer literal of the compiled script source (the reset script, the readout pages, the port
           scripts, the motion script), changed by +-1, and the `>> 16` of the plane readouts.

The ACS source is captured from the compiler (the lump is bytecode) and recompiled with acc after the edit.
Survivors get a one-line reason from a rule table (`REASONS`): the rules say which survivors are expected to be
equivalent mutants (a value the reset script overwrites, a speed no pass depends on, ...). A survivor no rule explains
is listed as unexplained; those are the ones to read.

    from doomformer.mutate import run_mutation, demo_net
    report = run_mutation(demo_net(), per_class=20, workers=3)
"""
from __future__ import annotations

import json
import queue
import random
import re
import shutil
import struct
import tempfile
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from . import _mp

LIGHT_RAISE, LIGHT_LOWER, FLOOR_LOWER, FLOOR_RAISE, FLOOR_MOVE_TO, CEIL_LOWER, CEIL_RAISE = 110, 111, 20, 23, 37, 40, 41
LOWER_TO_LOWEST, EXEC_ALWAYS, TELEPORT, SCROLL = 21, 226, 71, 223
PAIRS = {LIGHT_RAISE: LIGHT_LOWER, LIGHT_LOWER: LIGHT_RAISE, FLOOR_RAISE: FLOOR_LOWER, FLOOR_LOWER: FLOOR_RAISE,
         CEIL_RAISE: CEIL_LOWER, CEIL_LOWER: CEIL_RAISE}


# --- WAD lumps ------------------------------------------------------------------------------------------------------
def read_lumps(path) -> list[tuple[str, bytes]]:
    data = Path(path).read_bytes()
    n, directory = struct.unpack("<ii", data[4:12])
    out = []
    for i in range(n):
        offset, size, name = struct.unpack("<ii8s", data[directory + 16 * i:directory + 16 * (i + 1)])
        out.append((name.rstrip(b"\0").decode(), data[offset:offset + size]))
    return out


def write_lumps(path, lumps: list[tuple[str, bytes]]) -> None:
    data, directory = b"", b""
    for name, blob in lumps:
        directory += struct.pack("<ii8s", 12 + len(data), len(blob), name.encode())
        data += blob
    Path(path).write_bytes(struct.pack("<4sii", b"PWAD", len(lumps), 12 + len(data)) + data + directory)


def edit_textmap(src, dst, fn) -> None:
    """Copy the WAD `src` (and its .json) to `dst` with TEXTMAP replaced by fn(text)."""
    lumps = [(n, fn(b.decode()).encode() if n == "TEXTMAP" else b) for n, b in read_lumps(src)]
    write_lumps(dst, lumps)
    shutil.copyfile(Path(src).with_suffix(".json"), Path(dst).with_suffix(".json"))


def tamper_weight(src, dst, delta: int = 1) -> str:
    """dst := src with the first light weight line (Light_RaiseByValue / Light_LowerByValue amount) changed by
    `delta`; returns a description. In compile order the first such line belongs to layer 0, input 0, lowest lane."""
    done = []
    pattern = re.compile(r"(special = (110|111); arg0 = (\d+); arg1 = )(\d+);")

    def repl(m):
        if done:
            return m.group(0)
        old = int(m.group(4))
        name = "Light_RaiseByValue" if m.group(2) == "110" else "Light_LowerByValue"
        done.append(f"{name} tag {m.group(3)}: {old} -> {old + delta}")
        return f"{m.group(1)}{old + delta};"

    edit_textmap(src, dst, lambda text: pattern.sub(repl, text))
    if not done:
        raise ValueError(f"{src}: no light weight line to tamper with")
    return done[0]


# --- the net and its inputs -----------------------------------------------------------------------------------------
def demo_net():
    """The default mutation target: a 3-4-3 two-layer MLP with signed inputs (light AND plane weights)."""
    from .intref import adc_signed
    from .kit import MLP
    rng = np.random.default_rng(7)
    W = [rng.integers(-9, 10, (4, 3)), rng.integers(-9, 10, (3, 4))]
    b = [rng.integers(-5, 6, 4), rng.integers(-5, 6, 3)]
    t0 = np.array([2, 4, 6])
    xs = rng.integers(-12, 13, (64, 3))
    h = adc_signed(xs, t0) @ W[0].T + b[0]
    t1 = np.sort(np.unique(np.clip(h[h >= 1], 1, None)))[::3][:3]
    return MLP(W, b, [t0, t1], signed_input=True)


# --- sites ----------------------------------------------------------------------------------------------------------
@dataclass
class Mutant:
    id: int
    cls: str
    where: str
    kind: str            # "textmap" | "acs"
    line: int            # textmap line index, or ACS source offset of the literal
    old: str
    new: str
    info: dict


_KV = re.compile(r"(\w+) = ([^;]+);")


def _kv(line: str) -> dict[str, str]:
    return {k: v.strip() for k, v in _KV.findall(line)}


def _set(line: str, key: str, value) -> str:
    return re.sub(rf"\b{key} = [^;]+;", f"{key} = {value};", line, count=1)


def textmap_mutants(text: str, meta, start: int = 0) -> list[Mutant]:
    """All single-site TEXTMAP mutants of one compiled map (sectors: +-1 on floor, ceiling, light; linedefs: +-1 on each
    argument, raise <-> lower, effect retargeting)."""
    out: list[Mutant] = []
    lines = text.split("\n")
    port_tags = set(meta.ports.values()) if meta.ports else set()
    gate_tags = port_tags | {int(t) for t in meta.layout.get("tags", {}).values()}
    out_tags = {int(t) for t in re.findall(r"\((\d+)\)", " ".join(meta.layout.get("read_expressions", {}).values()))
                if int(t) >= 20000}
    effect_tags: dict[int, list[int]] = {}
    for ln in lines:
        if ln.startswith("linedef"):
            kv = _kv(ln)
            s = int(kv.get("special", 0))
            if s in PAIRS:
                effect_tags.setdefault(s, []).append(int(kv["arg0"]))
    sector_i = -1

    def add(cls, where, i, new, **info):
        out.append(Mutant(start + len(out), cls, where, "textmap", i, lines[i], new, info))

    for i, ln in enumerate(lines):
        if ln.startswith("sector"):
            sector_i += 1
            kv = _kv(ln)
            tag = int(kv["id"])
            role = ("gate" if tag in gate_tags else "output" if tag in out_tags
                    else f"walk:{kv['texturefloor'].strip(chr(34))}/{kv['lightlevel']}" if tag == 1
                    else "barrier" if tag == 2 or tag >= 60000 else "other")
            for key, cls in (("heightfloor", f"sector-floor-{role}"), ("heightceiling", f"sector-ceiling-{role}"),
                             ("lightlevel", f"sector-light-{role}")):
                for d in (-1, 1):
                    v = int(kv[key]) + d
                    add(cls, f"sector {sector_i} (tag {tag}) {key} {kv[key]} -> {v}", i, _set(ln, key, v),
                        role=role, key=key, tag=tag)
        elif ln.startswith("linedef") and "special" in ln:
            kv = _kv(ln)
            s = int(kv["special"])
            for k in range(5):
                key = f"arg{k}"
                if key not in kv:
                    continue
                for d in (-1, 1):
                    v = int(kv[key]) + d
                    add(_line_class(s, k), f"linedef {i} special {s} {key} {kv[key]} -> {v}", i, _set(ln, key, v),
                        special=s, arg=k)
            if s in PAIRS:
                add("effect-sign", f"linedef {i} special {s} -> {PAIRS[s]}", i, _set(ln, "special", PAIRS[s]),
                    special=s, arg=-1)
                tags = sorted(set(effect_tags[s]))
                other = tags[(tags.index(int(kv["arg0"])) + 1) % len(tags)]
                if other != int(kv["arg0"]):
                    add("effect-target", f"linedef {i} special {s} arg0 {kv['arg0']} -> {other} (another target)", i,
                        _set(ln, "arg0", other), special=s, arg=0)
    return out


def _line_class(s: int, k: int) -> str:
    if s in (LIGHT_RAISE, LIGHT_LOWER):
        return {0: "light-weight-tag", 1: "light-weight"}.get(k, "line-other")
    if s in (FLOOR_RAISE, FLOOR_LOWER, CEIL_RAISE, CEIL_LOWER):
        return {0: "plane-weight-tag", 1: "plane-mover-speed", 2: "plane-weight"}.get(k, "line-other")
    if s == FLOOR_MOVE_TO:
        return {0: "move-to-tag", 1: "move-to-speed", 2: "move-to-height"}.get(k, "line-other")
    if s == LOWER_TO_LOWEST:
        return {0: "barrier-tag", 1: "barrier-speed"}.get(k, "line-other")
    if s == EXEC_ALWAYS:
        return {0: "exec-script", 1: "exec-map", 2: "doll-tid"}.get(k, "line-other")
    if s == TELEPORT:
        return "teleport-tid"
    if s == SCROLL:
        return "scroller"
    return "line-other"


_ACS_SCRIPT = re.compile(r"\bscript\s+(\d+)")
_ACS_STMT = re.compile(r"[^;{}]+;")
_INT = re.compile(r"(?<![\w.])(\d+)(?![\w.])")


def _call_context(stmt: str, pos: int) -> tuple[str, int | None]:
    """(innermost call name, argument index) at offset `pos` of an ACS statement."""
    stack: list[list] = []
    for i, ch in enumerate(stmt[:pos]):
        if ch == "(":
            name = re.search(r"(\w+)\s*$", stmt[:i])
            stack.append([name.group(1) if name else "", 0])
        elif ch == ")" and stack:
            stack.pop()
        elif ch == "," and stack:
            stack[-1][1] += 1
    return (stack[-1][0], stack[-1][1]) if stack else ("", None)


def acs_mutants(source: str, start: int = 0) -> list[Mutant]:
    """Every integer literal of a compiled script, +-1, plus the `>> 16` of plane readouts as `>> 15`."""
    out: list[Mutant] = []
    headers = [(m.start(), int(m.group(1))) for m in _ACS_SCRIPT.finditer(source)]

    def script_at(pos: int) -> int | None:
        cur = None
        for p, n in headers:
            if p <= pos:
                cur = n
        return cur

    for sm in _ACS_STMT.finditer(source):
        stmt = sm.group(0)
        if stmt.lstrip().startswith(("#", "global")) or "script" in stmt.split("(")[0] and "{" in stmt:
            continue
        head = re.sub(r"^\s*(script\s+\d+\s*\([^)]*\)\s*\{\s*)", "", stmt).strip()
        sc = script_at(sm.start())
        for im in _INT.finditer(stmt):
            n = int(im.group(1))
            before = stmt[:im.start(1)]
            if head.startswith("int mo_") or ("[" in before and "]" in stmt[im.start(1):][:3]) \
                    or re.search(r">>\s*$", before):
                continue
            call, arg = _call_context(stmt, im.start(1))
            cls = _acs_class(sc)
            for d in (-1, 1):
                out.append(Mutant(start + len(out), cls, f"script {sc} {call or 'expr'}"
                                  f"{'' if arg is None else f' arg{arg}'}: literal {n} -> {n + d} ({head[:44]})",
                                  "acs", sm.start() + im.start(1), str(n), str(n + d),
                                  dict(script=sc, call=call, arg=arg)))
        if ">> 16" in stmt:
            out.append(Mutant(start + len(out), "acs-readout-shift", f"script {sc}: `>> 16` -> `>> 15` ({head[:40]})",
                              "acs", sm.start() + stmt.index(">> 16") + 3, "16", "15",
                              dict(script=sc, call="shift", arg=None)))
    return out


def _acs_class(sc) -> str:
    if sc is None:
        return "acs-other"
    if 100 <= sc < 200:
        return "acs-readout"
    return {3: "acs-reset", 1: "acs-port-write", 2: "acs-port-write", 4: "acs-tid", 7: "acs-start-barrier",
            8: "acs-motion", 9: "acs-guard"}.get(sc, "acs-other")


# --- reasons for survivors ------------------------------------------------------------------------------------------
def reason(m: Mutant, ctx: dict | None = None) -> str | None:
    """A rule-based one-line reason why a surviving mutant is expected to be equivalent, or None (unexplained).
    The rules read the compiler's design (docs/HOW_IT_WORKS.md); they are expectations, not proofs."""
    c, ctx = m.cls, ctx or {}
    if c.startswith("sector-light-"):
        return "light level of a sector no script reads" if not c.endswith("output") else None
    if c.startswith("sector-floor-walk"):
        return "floor of a walking sector: +-1 unit stays inside the 24-unit step-up slack"
    if c in ("sector-ceiling-output", "sector-floor-output"):
        return "height of an output display sector: only its light level is read"
    if c in ("sector-floor-barrier", "sector-ceiling-barrier"):
        return "a barrier is 13,000 units above the walking floor when closed: 1 unit changes nothing"
    if c.startswith("sector-ceiling-walk"):
        return "ceiling 16,000 units above the 56-unit dolls"
    if c in ("sector-floor-other", "sector-ceiling-other"):
        return "a room or closet no doll's path crosses"
    if c == "sector-ceiling-gate":
        return "ceiling far above the dolls (a floor gate) or the value the reset script restores (a ceiling gate)"
    if c == "sector-floor-gate":
        return "initial height: the reset script and the port writes set it before every pass but the first"
    if c in ("plane-mover-speed", "move-to-speed", "barrier-speed") or (
            m.info.get("call") in ("Floor_MoveToValue", "Ceiling_MoveToValue", "Floor_LowerToLowest",
                                   "Ceiling_LowerToFloor") and m.info.get("arg") == 1):
        return "mover speed 150,000 is about 18,750 units per tic: +-1 changes no move's finish tic"
    closed = c == "move-to-height" or (c == "acs-reset" and m.info.get("call") == "Floor_MoveToValue"
                                       and m.info.get("arg") == 2)
    if closed and re.search(r"\b13000 -> (13001|12999)", m.where):
        return "a closed gate or barrier sits at 13,000: one unit more or less changes no pass"
    if c == "scroller" and m.info.get("arg") in (3, 4):
        return "conveyor speed byte: +-1 leaves the carry direction and the dolls' arrival unchanged"
    if c == "exec-map":
        return "unused map argument of ACS_ExecuteAlways"
    if m.info.get("call", "").startswith("GetSector") and m.info.get("arg") in (1, 2):
        return "x / y point argument of GetSector*Z: a sector's plane is flat"
    if m.info.get("call") == "TeleportOther" and m.info.get("arg") == 2:
        return "fog flag of TeleportOther: only a spawn effect"
    if c == "acs-motion" and ctx.get("n_dolls", 1 << 30) + 1 < 4096:
        return "the motion script runs only on maps with >= 4,096 objects (runtime.settle); this one has fewer"
    return None


# --- running mutants ------------------------------------------------------------------------------------------------
def capture_acs(net, wad: Path, guard: bool = True) -> str:
    """Compile `net` to `wad` and return the ACS source the compiler produced."""
    from . import kit, pure
    seen = []
    orig = pure.compile_acs
    pure.compile_acs = lambda src: (seen.append(src), orig(src))[1]
    try:
        kit.compile(net, wad, mode="map", guard=guard).close()
    finally:
        pure.compile_acs = orig
    return seen[-1]


def apply_acs(source: str, m: Mutant) -> str:
    assert source[m.line:m.line + len(m.old)] == m.old, "ACS source changed"
    return source[:m.line] + m.new + source[m.line + len(m.old):]


def _build_mutant(base: Path, source: str, m: Mutant, dst: Path) -> None:
    lumps = read_lumps(base)
    if m.kind == "textmap":
        def fn(text):
            lines = text.split("\n")
            assert lines[m.line] == m.old
            lines[m.line] = m.new
            return "\n".join(lines)
        lumps = [(n, fn(b.decode()).encode() if n == "TEXTMAP" else b) for n, b in lumps]
    else:
        from .udmf import compile_acs
        code = compile_acs(apply_acs(source, m))
        lumps = [(n, code if n == "BEHAVIOR" else b) for n, b in lumps]
    write_lumps(dst, lumps)
    shutil.copyfile(base.with_suffix(".json"), dst.with_suffix(".json"))


def _run_mutant(job: dict) -> dict:
    """Build and run one mutant; returns {id, killed, how}."""
    from .fleet import open_brain
    from .kit import RangeError
    from .fuzz import net_of
    net = net_of("mlp", job["spec"])
    m = Mutant(**job["mutant"])
    t0 = time.perf_counter()
    with tempfile.TemporaryDirectory(prefix="dfm-") as d:
        wad = Path(d) / "m.wad"
        try:
            _build_mutant(Path(job["base"]), job["source"], m, wad)
        except Exception as e:                                      # e.g. acc rejects the mutated source
            return dict(id=m.id, killed=True, how=f"does not build: {type(e).__name__}: {str(e)[:80]}")
        try:
            brain = open_brain(net, wad)
        except Exception as e:
            return dict(id=m.id, killed=True, how=f"does not start: {type(e).__name__}: {str(e)[:80]}")
        try:
            brain.runner.max_tics = job["max_tics"]
            for x in job["inputs"]:
                try:
                    brain(np.asarray(x))
                except RangeError:
                    continue
                except Exception as e:
                    how = "mismatch" if type(e).__name__ == "MismatchError" else f"{type(e).__name__}"
                    return dict(id=m.id, killed=True, how=how, detail=str(e)[:120],
                                seconds=round(time.perf_counter() - t0, 2))
        finally:
            brain.close()
    return dict(id=m.id, killed=False, how="survived", seconds=round(time.perf_counter() - t0, 2))


def _worker(wid: int, tasks, results):
    _mp.exit_with_parent()
    while True:
        job = tasks.get()
        if job is None:
            return
        results.put(("start", wid, job["mutant"]["id"]))
        try:
            res = _run_mutant(job)
        except BaseException as e:                                  # noqa: BLE001
            res = dict(id=job["mutant"]["id"], killed=True, how=f"harness error {type(e).__name__}: {e}"[:100])
        results.put(("done", wid, res))


def run_jobs(jobs: list[dict], workers: int, timeout: float) -> dict[int, dict]:
    """Run mutant jobs on worker processes; a worker that dies or runs past `timeout` s on one mutant kills it
    (that mutant counts as killed: crash / timeout) and is replaced."""
    ctx = _mp.context()
    tasks, results = ctx.Queue(), ctx.Queue()
    for j in jobs:
        tasks.put(j)
    out: dict[int, dict] = {}
    procs: dict[int, object] = {}
    current: dict[int, tuple[int, float]] = {}
    nxt = 0

    def spawn():
        nonlocal nxt
        p = ctx.Process(target=_worker, args=(nxt, tasks, results), daemon=True)
        with _mp.detached_main():
            p.start()
        procs[nxt] = p
        nxt += 1

    for _ in range(workers):
        spawn()
    try:
        while len(out) < len(jobs):
            try:
                msg = results.get(timeout=1.0)
            except queue.Empty:
                msg = None
            if msg:
                if msg[0] == "start":
                    current[msg[1]] = (msg[2], time.time())
                else:
                    out[msg[2]["id"]] = msg[2]
                    current.pop(msg[1], None)
            for wid, p in list(procs.items()):
                over = wid in current and time.time() - current[wid][1] > timeout
                if (not p.is_alive() or over) and wid in current:
                    mid = current.pop(wid)[0]
                    out.setdefault(mid, dict(id=mid, killed=True, how="timeout" if over else "engine crash"))
                    if p.is_alive():
                        p.terminate()
                    del procs[wid]
                    spawn()
                elif not p.is_alive():
                    del procs[wid]
                    spawn()
    finally:
        for p in procs.values():
            if p.is_alive():
                p.terminate()
    return out


def _deep_inputs(rng, net, n: int) -> list[list[int]]:
    """n random inputs (scales 3, 12, 100, 12,000 per input) that the net's range check accepts."""
    from .kit import RangeError
    out = []
    while len(out) < n:
        s = int(rng.choice([3, 12, 100, 12000]))
        x = rng.integers(-s, s + 1, net.n_in)
        try:
            net.check(x)
        except RangeError:
            continue
        out.append([int(v) for v in x])
    return out


def run_mutation(net, *, per_class: int = 20, seed: int = 0, workers: int = 1, n_inputs: int = 24,
                 timeout: float = 120.0, progress=None) -> dict:
    """Mutate the compiled map of `net` (an MLP) and report the kill rate by class. Returns a JSON-ready dict:
    classes: {class: {sites, sampled, killed, survived}}, survivors: [{class, where, reason}]."""
    from .fuzz import _mlp_inputs, rng_for, spec_of
    from .kit import MLP
    if not isinstance(net, MLP):
        raise TypeError("run_mutation takes a df.MLP")
    rng = random.Random(seed)
    nrng = rng_for(seed, 0, "mlp")
    t0 = time.perf_counter()
    with tempfile.TemporaryDirectory(prefix="dfmut-") as d:
        base = Path(d) / "base.wad"
        source = capture_acs(net, base)
        from .tile import TileMeta
        meta = TileMeta.load(base.with_suffix(".json"))
        text = dict(read_lumps(base))["TEXTMAP"].decode()
        sites = textmap_mutants(text, meta)
        sites += acs_mutants(source, start=len(sites))
        inputs, _ = _mlp_inputs(nrng, net, [], n_inputs)
        from .fleet import open_brain
        brain = open_brain(net, base)                              # the clean map must pass first
        tics = 0
        try:
            for x in inputs:
                brain(np.asarray(x))
                tics = max(tics, brain.info["tics"])
        finally:
            brain.close()
        by_class: dict[str, list[Mutant]] = {}
        for m in sites:
            by_class.setdefault(m.cls, []).append(m)
        picked = []
        for cls in sorted(by_class):
            pool = by_class[cls]
            picked += rng.sample(pool, min(per_class, len(pool)))
        spec = spec_of(net)
        jobs = [dict(mutant=asdict(m), spec=spec, base=str(base), source=source, inputs=inputs,
                     max_tics=max(400, 4 * tics)) for m in picked]
        res = run_jobs(jobs, workers, timeout)
        survivors = [m for m in picked if not res[m.id]["killed"]]
        # a survivor gets a second, larger input set before it is called one
        if survivors:
            more, _ = _mlp_inputs(nrng, net, [np.asarray(x) + 1 for x in inputs], 3 * n_inputs)
            extra = [x for x in more if x not in inputs]
            jobs2 = [dict(j, inputs=extra) for j in jobs if j["mutant"]["id"] in {m.id for m in survivors}]
            res2 = run_jobs(jobs2, workers, timeout)
            for mid, r in res2.items():
                if r["killed"]:
                    res[mid] = dict(r, how=r["how"] + " (on the second input set)")
        # a survivor no rule explains gets 300 more random inputs at four scales before it is listed as unexplained
        alive = [m for m in survivors if not res[m.id]["killed"] and reason(m, dict(n_dolls=meta.n_dolls)) is None]
        if alive:
            deep = _deep_inputs(nrng, net, 300)
            jobs3 = [dict(j, inputs=deep) for j in jobs if j["mutant"]["id"] in {m.id for m in alive}]
            for mid, r in run_jobs(jobs3, workers, timeout).items():
                if r["killed"]:
                    res[mid] = dict(r, how=r["how"] + " (on the deep input set)")
    table = {}
    for cls, pool in sorted(by_class.items()):
        got = [res[m.id] for m in pool if m.id in res]
        table[cls] = dict(sites=len(pool), sampled=len(got), killed=sum(r["killed"] for r in got),
                          survived=sum(not r["killed"] for r in got))
    surv = []
    for m in picked:
        if not res[m.id]["killed"]:
            surv.append(dict(cls=m.cls, where=m.where, reason=reason(m, dict(n_dolls=meta.n_dolls)) or "unexplained"))
    how = {}
    for m in picked:
        if res[m.id]["killed"]:
            k = re.sub(r"\(on the .*", "", res[m.id]["how"]).split(":")[0].strip()
            how[k] = how.get(k, 0) + 1
    tot = len(picked)
    killed = sum(res[m.id]["killed"] for m in picked)
    return dict(net=spec, seed=seed, per_class=per_class, inputs=len(inputs), sites=len(sites), sampled=tot,
                killed=killed, survived=tot - killed, kill_rate=round(killed / tot, 4),
                unexplained=sum(s["reason"] == "unexplained" for s in surv), how_killed=how, classes=table,
                survivors=surv, seconds=round(time.perf_counter() - t0, 1))


def format_report(rep: dict) -> str:
    lines = [f"mutation: {rep['sampled']} of {rep['sites']} sites sampled (<= {rep['per_class']} per class, seed "
             f"{rep['seed']}), {rep['inputs']} inputs each; killed {rep['killed']}, survived {rep['survived']} "
             f"({rep['unexplained']} unexplained); kill rate {100 * rep['kill_rate']:.1f}%",
             f"{'class':26} {'sites':>6} {'sampled':>8} {'killed':>7} {'survived':>9}"]
    for cls, r in rep["classes"].items():
        lines.append(f"{cls:26} {r['sites']:6} {r['sampled']:8} {r['killed']:7} {r['survived']:9}")
    seen = {}
    for s in rep["survivors"]:
        seen.setdefault((s["cls"], s["reason"]), []).append(s["where"])
    if seen:
        lines.append("survivors (class: count, reason, first site):")
        for (cls, why), wh in seen.items():
            lines.append(f"  {cls}: {len(wh)}, {why}; e.g. {wh[0]}")
    return "\n".join(lines)


# --- CLI ------------------------------------------------------------------------------------------------------------
def run_cli(args) -> int:
    from .kit import MLP
    net = MLP.load(args.net) if getattr(args, "net", None) else demo_net()
    rep = run_mutation(net, per_class=args.per_class, seed=args.seed, workers=args.workers)
    print(format_report(rep))
    if args.json:
        Path(args.json).write_text(json.dumps(rep, indent=1) + "\n")
    weight_bearing = ("light-weight", "plane-weight", "effect-sign", "effect-target")
    return 1 if any(rep["classes"].get(c, {}).get("survived") for c in weight_bearing) else 0
