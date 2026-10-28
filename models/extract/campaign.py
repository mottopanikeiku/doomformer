"""Static extraction over random nets: extract(compile(net)).equals(net), no game.

    python models/extract/campaign.py --nets 500 --seed 0 --opt 0 --json models/extract/fuzz_opt0.json

Nets come from doomformer.fuzz.gen_case (MLPs). Each net is compiled to a map (the build path of `df.compile(mode="map")`,
no DOOM game starts), read back from the .wad alone and compared with the net. Nets the compiler refuses (an all-zero
layer) are counted apart. Exit status 1 if any net differs.
"""
from __future__ import annotations

import argparse
import json
import sys
import tempfile
import time
from pathlib import Path

from doomformer import fuzz
from doomformer.extract import ExtractError, expected_netlist, extract
from doomformer.kit import compile as df_compile


def invisible(net) -> dict[str, bool]:
    """Which parts of the net the map cannot show (extract.py docstring) this net has."""
    Ws = net.weights
    out = dict(zero_input_column=any(not Ws[0][:, i].any() for i in range(Ws[0].shape[1])), unread_unit_nonzero_row=False,
               unread_unit_zero_row=False, leading_zero_output_row=len(Ws) > 1 and not Ws[-1][0].any())
    for l in range(len(Ws) - 1):
        for j in range(Ws[l].shape[0]):
            if not Ws[l + 1][:, j].any():
                out["unread_unit_nonzero_row" if Ws[l][j].any() else "unread_unit_zero_row"] = True
    return out


def run(nets: int, seed: int, opt: int, edge: bool = False, start: int = 0) -> dict:
    res = dict(kind="mlp", seed=seed, opt=opt, edge=edge, first_index=start, nets=nets, compiled=0, equal=0,
               refused_by_compiler=0, failures=[], layers=0, lanes=0, weights_compared=0, parts=dict(
                   zero_input_column=0, unread_unit_nonzero_row=0, unread_unit_zero_row=0, leading_zero_output_row=0),
               signed=0, relu=0)
    t0 = time.perf_counter()
    with tempfile.TemporaryDirectory(prefix="dfx-") as d:
        for i in range(start, start + nets):
            case = fuzz.gen_case("mlp", seed, i, edge=edge)
            net = fuzz.net_of("mlp", case["spec"])
            wad = Path(d) / "net.wad"
            try:
                df_compile(net, wad, mode="map", limits=False, opt=opt).close()
            except ValueError as e:
                if case["planned"] == "unsupported" and fuzz.UNSUPPORTED_MSG in str(e):
                    res["refused_by_compiler"] += 1
                    continue
                res["failures"].append(dict(index=i, error=f"compile: {e}"))
                continue
            res["compiled"] += 1
            for k, v in invisible(net).items():
                res["parts"][k] += int(v)
            res["signed" if net.signed_input else "relu"] += 1
            try:
                nl = extract(wad)
                diffs = nl.diff(net)
                if diffs or nl.findings:
                    res["failures"].append(dict(index=i, differences=[str(x) for x in diffs[:4]], findings=nl.findings[:4],
                                                note=case["note"]))
                else:
                    res["equal"] += 1
                    res["layers"] += len(nl.layers)
                    res["lanes"] += sum(len(x) for L in nl.layers for x in L.limits)
                    res["weights_compared"] += sum(int(f.size) for L in expected_netlist(net).layers for f in L.fx)
            except ExtractError as e:
                res["failures"].append(dict(index=i, error=f"extract: {e}", note=case["note"]))
            for p in (wad, wad.with_suffix(".json")):
                p.unlink(missing_ok=True)
    res["seconds"] = round(time.perf_counter() - t0, 1)
    res["verdict"] = "ALL EQUAL" if res["equal"] == res["compiled"] and not res["failures"] else "DIFFERENCES"
    return res


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--nets", type=int, default=500)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--opt", type=int, default=0)
    ap.add_argument("--edge", action="store_true")
    ap.add_argument("--json")
    args = ap.parse_args(argv)
    r = run(args.nets, args.seed, args.opt, args.edge, args.start)
    print(f"seed {r['seed']} opt {r['opt']}{' edge' if r['edge'] else ''}: {r['nets']} generated, {r['compiled']} compiled "
          f"({r['refused_by_compiler']} refused: an all-zero layer), {r['equal']} equal, {len(r['failures'])} failures; "
          f"{r['signed']} signed / {r['relu']} relu input; {r['weights_compared']:,} lane weights compared; "
          f"{r['seconds']} s; nets with parts the map cannot show: {r['parts']}")
    for f in r["failures"][:10]:
        print("  FAIL", f)
    if args.json:
        Path(args.json).write_text(json.dumps(r, indent=1) + "\n")
    return 0 if r["verdict"] == "ALL EQUAL" else 1


if __name__ == "__main__":
    sys.exit(main())
