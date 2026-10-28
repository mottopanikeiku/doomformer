"""Measured settle tics against `plan`'s numbers: the geometry bound and the calibrated estimate with its error band.

    python -m demos.settlecheck hello,tiny,mlp24 --opts 0,1,2,3
    python -m demos.settlecheck rand-10_20_12_5-4_3_3-501,induction_onemap --opts 3

A net is a tune reference name (`hello mlp24 mlp16 signed fold tiny softmax arch mnist ...`), a shipped transformer name,
or `rand-SIZES-K-SEED`: a random MLP as `doomformer tune` draws them (`rand-10_20_12_5-4_3_3-501` is 10-20-12-5 with 4, 3, 3
thresholds, seed 501). Each (net, opt) is built, one input runs in DOOM (the settle does not depend on the input), and
the table prints the measured game tics, the geometry estimate and the calibrated estimate with its error. At most one
game at a time; `mnist` at opt 0 to 2 takes minutes."""
from __future__ import annotations

import argparse
import os
import sys
import tempfile
from pathlib import Path


def make(name: str):
    from doomformer import tune
    if name.startswith("rand-"):
        _, sizes, k, seed = name.split("-")[:4]
        return tune._mlp([int(v) for v in sizes.split("_")], [int(v) for v in k.split("_")], int(seed))
    if name in tune.NETS:
        return tune.make_net(name)
    from doomformer.intref import QModel
    return QModel.load(name)


def measure(name: str, opt: int) -> tuple[int, int, int, int]:
    """(measured settle tics, geometry estimate, calibrated estimate, band) of one net at one opt level."""
    from doomformer import cost, tune
    from doomformer.kit import _build
    net = make(name)
    inputs = tune.mnist_digits(1) if name == "mnist" else tune.make_inputs(net, 1, 7)[0][:1]
    with tempfile.TemporaryDirectory(prefix="settlecheck-") as d:
        wad = Path(d) / "m.wad"
        meta = _build(net, wad, opt=opt)
        rows = tune.run_map(net, str(wad), inputs)["rows"]
    if rows[0][0] != "ok":
        raise SystemExit(f"{name} opt {opt}: DOOM did not match the reference ({rows[0]})")
    est = cost.settle_estimate(meta.layout)
    return rows[0][1], meta.layout["estimated_settle_tics"], est["calibrated_tics"], est["tics_band"]


def main(argv=None) -> int:
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("nets", help="comma separated net names")
    ap.add_argument("--opts", default="0,1,2,3")
    a = ap.parse_args(argv)
    print(f"{'net':32s} opt  measured  bound  err    calibrated  err    band")
    for name in a.nets.split(","):
        for opt in (int(v) for v in a.opts.split(",")):
            m, e, c, band = measure(name, opt)
            print(f"{name[:32]:32s} {opt:3d} {m:9,d} {e:6,d} {100 * (e - m) / m:+5.1f}% {c:10,d} {100 * (c - m) / m:+5.1f}% "
                  f"+-{band}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
