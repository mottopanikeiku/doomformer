"""The plane-move bound: how high or low a plane may stand when a doll's move on it starts.

A value that feeds a gate is a plane height (a floor or a ceiling). Dolls move it with Floor_/Ceiling_RaiseByValue and
LowerByValue at speed MOVE_SPEED = 150,000, i.e. speed/8 = 18,750 units per tic in 16.16 fixed point. When a move
starts with the plane above 32,768 - 18,750 = 14,018 (raise) or below -14,018 (lower) the engine's step overflows 32
bits; the move still ends at the right height, but its mover stays busy for 3 tics instead of 1, and a move requested on
the same plane in that window is dropped (probe `mover_busy` in doomformer facts). Measured edges:

    floor, raise:   start 14,018 ok, 14,019 busy        ceiling, raise: start 14,017 ok, 14,018 busy
    floor, lower:   start -14,017 ok, -14,018 busy      ceiling, lower: start -14,018 ok, -14,019 busy

`PLANE_MAX` = 14,018 is the safe bound for both kinds. Dolls cross a plane's lines in any order, so the rule is stated
on partial sums: a plane with bias `b` whose terms add `P` (sum of the raises) and subtract `N` (sum of the lowers)
needs `b + P <= PLANE_MAX` when P > 0 (the last raise starts at most `P - 1` below it) and `b - N >= -PLANE_MAX` when
N > 0. A plane that never rises may start anywhere its final value allows; this is what the engine does, not a margin.
The int16 light accumulators of unconsumed outputs have their own bound (intref.Z_MAX).
"""
from __future__ import annotations

import numpy as np

from .tile import MOVE_SPEED

FIXED_ONE = 65536
PLANE_MAX = 32768 - MOVE_SPEED // 8       # 14,018


def plane_peak(bias, up, down) -> int:
    """Largest |partial sum| that a raise or a lower can start from, over the planes: arrays of bias, P (sum of the
    positive terms) and N (sum of |negative terms|), any shape. Only a plane that has raise terms counts its `b + P`,
    only one with lower terms its `N - b`; a value above PLANE_MAX violates the rule."""
    bias, up, down = (np.asarray(a, dtype=np.int64) for a in (bias, up, down))
    hi = np.where(up > 0, bias + up, 0)
    lo = np.where(down > 0, down - bias, 0)
    return int(max(hi.max(initial=0), lo.max(initial=0)))


def terms_peak(bias, terms) -> int:
    """`plane_peak` for `terms` (..., n_terms) added to `bias` (...): P and N are the sums of its positive and negative
    entries."""
    terms = np.asarray(terms, dtype=np.int64)
    return plane_peak(bias, np.clip(terms, 0, None).sum(-1), -np.clip(terms, None, 0).sum(-1))


def message(name: str, peak: int) -> str:
    return (f"{name}: a plane partial sum reaches {peak} > {PLANE_MAX} (a doll's move that starts there keeps the "
            "engine's mover busy for 3 tics and moves requested on that plane in the window are lost; planes.py)")


def mlp_static_peaks(net) -> list[tuple[str, int]]:
    """Worst case over ALL inputs of the plane partial sums of an MLP's hidden units: [(plane name, peak)] per
    hidden layer. The staircase counts are propagated as intervals (layer 0: [-K, K] or [0, K]; later layers
    [adc(low), adc(high)] of the previous layer's interval), and each input contributes the largest raise and the
    largest lower its count range allows, so this is the exact worst case for layer 0 (the inputs are independent) and
    an upper bound for later layers."""
    from .intref import adc_relu

    lo = np.full(net.n_in, -len(net.thresholds[0]) if net.signed_input else 0, dtype=np.int64)
    hi = np.full(net.n_in, len(net.thresholds[0]), dtype=np.int64)
    out = []
    for l in range(net.n_layers - 1):
        W, b = net.weights[l], net.biases[l]
        wa, wb = W * lo, W * hi                                  # (out, in): the two ends of each term
        up = np.clip(np.maximum(wa, wb), 0, None).sum(1)
        down = -np.clip(np.minimum(wa, wb), None, 0).sum(1)
        out.append((f"-h{l + 1}", plane_peak(b, up, down)))
        h_lo, h_hi = b + np.minimum(wa, wb).sum(1), b + np.maximum(wa, wb).sum(1)
        lo, hi = adc_relu(h_lo, net.thresholds[l + 1]), adc_relu(h_hi, net.thresholds[l + 1])
    return out


def mlp_static_errors(net) -> list[str]:
    return [message(f"{name} (worst case over all inputs)", peak) for name, peak in mlp_static_peaks(net)
            if peak > PLANE_MAX]


def spiking_step_errors(net, v, x) -> list[str]:
    """Plane-move violations of one spiking step from state `v` with inputs `x` (0/1): the potentials v_i and -v_i are
    planes moved by the fire layer (spike reset -theta, leak) and then, after a barrier, by the synapse layer (W of the
    spiking neurons, U of the active inputs). Dolls cross in any order inside a layer."""
    v = np.asarray(v, dtype=np.int64)
    x = np.asarray(x, dtype=np.int64)
    s = (v >= net.theta).astype(np.int64)
    low, high = (v >= net.leak).astype(np.int64), (v <= -net.leak).astype(np.int64)
    fire_up, fire_down = net.leak * high, net.theta * s + net.leak * low
    errs = []
    peak = plane_peak(v, fire_up, fire_down)
    if peak > PLANE_MAX:
        errs.append(message("fire layer of v", peak))
    mid = v - fire_down + fire_up
    terms = np.concatenate([net.W * s[None, :], net.U * x[None, :]], axis=1)
    peak = terms_peak(mid, terms)
    if peak > PLANE_MAX:
        errs.append(message("synapse layer of v", peak))
    return errs


# --- the measurement behind the bound: the reduced net of the fuzz-seed-902 miscompile, DOOM at the boundary ----------
REDUCED_X = [-12000, 5933, 6252, -353, 11483]
REDUCED_THR = [[713, 1423, 3182, 11867], [8540]]


def reduced_net(bias: int, w3: int = 3):
    """The reduced net of the miscompile found by fuzz --edge --seed 902, net 5: the plane -h1_0 starts at -bias, four dolls raise it by
    w3 and nine lower it by 255 at REDUCED_X; the fourth raise starts at -bias + 3 * w3."""
    from .kit import MLP
    return MLP([[[w3, 255, 255, 0, 255]], [[-255]]], [[bias], [185]], REDUCED_THR)


def sweep(biases, w3: int = 3, workdir=None, opt: int = 0) -> list[dict]:
    """Run reduced_net(bias) at REDUCED_X in DOOM for each bias, bypassing the plane check (limits=False and a direct
    readback), and report DOOM's -h1_0 against the reference, the peak the rule computes and whether `check` accepts."""
    import tempfile
    from pathlib import Path

    from .kit import MLPBrain, RangeError
    from .kit import compile as df_compile
    rows = []
    with tempfile.TemporaryDirectory(prefix="dfplanes-") as tmp:
        for b in biases:
            net = reduced_net(int(b), w3)
            peak = terms_peak(net.biases[0], net.reference(REDUCED_X)["activations"][0] * net.weights[0])
            try:
                net.check(REDUCED_X)
                accepted = True
            except RangeError:
                accepted = False
            with df_compile(net, Path(workdir or tmp) / f"b{b}.wad", mode="map", guard=True, limits=False,
                            opt=opt) as brain:
                got = MLPBrain._run(brain, brain._ports(net._x(REDUCED_X)))
            ref = net.reference(REDUCED_X)
            doom, want = got["-h1_0"], -int(ref["hidden"][0][0])
            rows.append(dict(bias=int(b), w3=w3, opt=opt, plane_peak=int(peak), check_accepts=accepted, doom=int(doom),
                             reference=int(want), exact=doom == want))
    return rows


def main(argv=None) -> int:
    import argparse
    import json
    import os
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    p = argparse.ArgumentParser(prog="doomformer verify --plane-sweep", description="DOOM at the plane-move bound: the "
                                "reduced net of fuzz seed 902 with the given biases (check bypassed)")
    p.add_argument("biases", nargs="*", type=int, default=[-12000, -12001, -14009, -14010, -14282])
    p.add_argument("--w3", type=int, default=3, help="the first weight (0: the plane never rises)")
    p.add_argument("--opt", type=int, default=0, help="schedule level of the map (default 0)")
    p.add_argument("--json", metavar="FILE")
    args = p.parse_args(argv)
    rows = sweep(args.biases, args.w3, opt=args.opt)
    print(f"{'bias':>8} {'peak':>7} {'check':>8} {'DOOM -h1_0':>11} {'reference':>10}  result")
    for r in rows:
        print(f"{r['bias']:>8} {r['plane_peak']:>7} {'accepts' if r['check_accepts'] else 'refuses':>8} {r['doom']:>11} "
              f"{r['reference']:>10}  {'exact' if r['exact'] else 'DIFFERS'}")
    if args.json:
        from pathlib import Path
        Path(args.json).write_text(json.dumps(dict(x=REDUCED_X, thresholds=REDUCED_THR, opt=args.opt, rows=rows),
                                              indent=1) + "\n")
    return 0

