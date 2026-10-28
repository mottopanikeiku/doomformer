"""Tie the static claim to the dynamic one on the same map: for each fuzz net, compile once, check
extract(wad).equals(net), then run the net's fuzz inputs in DOOM at tolerance 0 on that same .wad.

    python models/extract/doom_pair.py --seed 0 --indices 1 10 33 --opt 0

A pass is exact when every exposed value equals the integer reference (the usual check of `brain(x)`). Starts one
game at a time. Exit status 1 on a static difference or a DOOM mismatch.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

from doomformer import fuzz
from doomformer.extract import extract
from doomformer.kit import MismatchError, RangeError
from doomformer.kit import compile as df_compile


def main(argv=None) -> int:
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--indices", type=int, nargs="+", required=True)
    ap.add_argument("--opt", type=int, default=0)
    ap.add_argument("--json")
    args = ap.parse_args(argv)
    rows, bad = [], False
    with tempfile.TemporaryDirectory(prefix="dfd-") as d:
        for i in args.indices:
            case = fuzz.gen_case("mlp", args.seed, i)
            net = fuzz.net_of("mlp", case["spec"])
            wad = Path(d) / f"net{i}.wad"
            brain = df_compile(net, wad, mode="map", guard=True, limits=False, opt=args.opt)
            nl = extract(wad)
            diffs = nl.diff(net)
            static_equal = not diffs and not nl.findings
            passes = values = refused = 0
            status = "exact"
            t0 = time.perf_counter()
            with brain:
                for x in case["inputs"]:
                    try:
                        brain(np.asarray(x))
                    except RangeError:
                        refused += 1
                        continue
                    except MismatchError as e:
                        status = f"MISMATCH {e}"[:200]
                        break
                    passes += 1
                    values += int(brain.info["checked"])
            rows.append(dict(index=i, note=case["note"], static_equal=static_equal, doom=status, passes=passes,
                             values_compared=values, refused_inputs=refused, seconds=round(time.perf_counter() - t0, 1)))
            print(f"net {i} ({case['note']}): static {'EQUAL' if static_equal else 'DIFFERENT'}; DOOM {status}: {passes} passes, "
                  f"{values} values at tolerance 0 ({refused} inputs refused by the range check), {rows[-1]['seconds']} s",
                  flush=True)
            bad |= not static_equal or status != "exact" or not passes
    if args.json:
        Path(args.json).write_text(json.dumps(dict(seed=args.seed, opt=args.opt, nets=rows), indent=1) + "\n")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
