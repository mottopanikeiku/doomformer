"""`doomformer museum`: build, check and export the playable museum map (doomformer/museum.py, docs/museum.md)."""
from __future__ import annotations

import json
import time
from pathlib import Path


def add_parser(sub) -> None:
    m = sub.add_parser("museum", help="the museum: a playable map in which voodoo dolls read a seven-segment digit you "
                                      "light on the floor, and a switch rewires one connection (7 becomes 1)",
                       description="build: write museum.wad (play it in GZDoom or `doomformer play`), its metadata, "
                                   "the network's weights, the reference of every pad pattern and SHA256SUMS. verify: "
                                   "Python stands in for a player (keys only) and checks every READ of the 7 -> "
                                   "MUTATE -> RESTORE tour against the NumPy reference. export: the map plus a "
                                   "self-test script for other ports, with the NumPy lines it must log. shots: PNG "
                                   "views of the hall and the three camera stations.")
    m.add_argument("action", choices=["build", "verify", "export", "shots"])
    m.add_argument("dir", nargs="?", default="museum", help="output directory (build, export, shots; default ./museum)")
    m.add_argument("--wad", metavar="PATH", help="verify, shots: a built museum.wad (default: build one in a "
                                                 "temporary directory)")
    m.add_argument("--digits", action="store_true", help="verify: also light and READ each digit 0 to 9 first")
    m.add_argument("--check", action="store_true", help="export: run the self-test in ViZDoom (console on stdout) "
                                                        "and diff its lines with expected.txt")
    m.add_argument("--low-tags", action="store_true", help="export: every sector tag and TID below 32,768 (Zandronum)")
    m.add_argument("--json", metavar="PATH", help="verify: write the READ records as JSON")


def _segments(bits: int) -> str:
    from .museum import SEGMENTS
    return "".join(SEGMENTS[s] for s in range(7) if bits >> s & 1) or "-"


def _line(label: str, r) -> str:
    cmds = sum(r.audit["commands"].values())
    return (f"{label:24s} pads {_segments(r.bits):7s} w_7a {'-1' if r.flip else '+1'}  DOOM reads {r.digit}  "
            f"scores {r.scores}  EXACT ({r.checked} values; {r.tics} tics; {cmds} readout commands, "
            f"{r.audit['non_readout_commands']} other)")


def _verify(args) -> int:
    from . import museum as mu
    from .kit import MismatchError
    t0 = time.time()
    with mu.Museum(args.wad) as m:
        meta = m.brain.meta
        print(f"museum: {meta.n_dolls:,} voodoo dolls, {meta.n_sectors:,} sectors, {len(meta.outputs)} values per READ "
              f"({m.brain.wad}); built or loaded in {time.time() - t0:.0f} s")
        print("Python stands in for a player: it presses the marine's movement keys to walk onto pads and USE on "
              "switches. During each READ it only advances tics and runs guarded readout scripts.", flush=True)
        records = []
        try:
            if args.digits:
                for d, bits in enumerate(mu.SEGMENT_BITS):
                    m.light(bits)
                    r = m.read()
                    records.append(dict(step=f"digit {d}", **r.__dict__))
                    print(_line(f"digit {d}", r), flush=True)
                    if r.digit != d:
                        print(f"  the reference reads {r.digit} for these pads", flush=True)
            labels = {None: "7", "mutate": "7 after MUTATE", "restore": "7 after RESTORE"}

            def shown(step, r):
                records.append(dict(step=labels[step], **r.__dict__))
                print(_line(labels[step], r), flush=True)

            tour = m.tour(on_read=shown)
        except MismatchError as e:
            print(f"MISMATCH: {e}")
            return 1
    ok = [r.digit for r in tour] == [7, 1, 7]
    print(f"7 -> {tour[1].digit} -> {tour[2].digit}: {'as the reference says' if ok else 'NOT the tour the reference gives'}"
          f"; every value of every READ equal to the NumPy reference of the network as wired at that moment")
    if args.json:
        Path(args.json).write_text(json.dumps(records, indent=1) + "\n")
    return 0 if ok else 1


def _shots(args) -> int:
    from . import museum as mu
    out = Path(args.dir)
    out.mkdir(parents=True, exist_ok=True)
    for path in mu.shots(out, args.wad):
        print(f"wrote {path}")
    return 0


def run(args) -> int:
    from . import museum as mu
    if args.action == "build":
        r = mu.build(Path(args.dir))
        print(f"museum: {r['dolls']:,} voodoo dolls, {r['sectors']:,} sectors, {r['span'][0]:,} x {r['span'][1]:,} "
              f"map units, opt {r['opt']}, {r['values']} values per READ, ACC {r['acc'][:12]}")
        for f, h in r["sha256"].items():
            print(f"{h}  {f}")
        print(f"wrote {r['out']}/ (SHA256SUMS too). Play: gzdoom -iwad freedoom2.wad -file {r['out']}/museum.wad "
              f"(or `doomformer play {r['out']}/museum.wad`)")
        return 0
    if args.action == "verify":
        return _verify(args)
    if args.action == "shots":
        return _shots(args)
    r = mu.export(Path(args.dir), low_tags=args.low_tags)
    print(f"wrote {r['out']}/museum_selftest.wad (sha256 {r['sha256'][:16]}...) and expected.txt ({r['cases']} READs, "
          f"{r['values']} values each, from the NumPy reference){' with every tag below 32,768' if r['low_tags'] else ''}")
    if args.check:
        ok, got = mu.check_export(Path(args.dir))
        print(f"ViZDoom: {len(got)} lines logged, {'identical to' if ok else 'DIFFERENT from'} expected.txt")
        return 0 if ok else 1
    return 0
