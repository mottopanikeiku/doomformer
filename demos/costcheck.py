"""Does `plan`'s predicted_seconds hold? Run random MLPs in DOOM, measure the engine's CPU time, compare.

    python -m demos.costcheck                    # the 11 validation nets, one map and tiles each (about 10 min)
    python -m demos.costcheck --no-map           # tiles only (under 1 min)
    python -m demos.costcheck --refit            # also run the 14 fit nets and print new constants (about 25 min)

For every net it measures the engine process's user+sys CPU time (read from /proc, Linux only) over several passes
with the game already loaded, once for the one-map pass and once for the tile games (the sum over the layers), and
prints it next to `df.plan(net)`'s predicted_seconds. The predictions are the constants of doomformer/cost.py, fitted on
the FIT nets; the VALIDATION nets are other shapes. Wall time on a loaded machine is 1.3 to 2.5 times the CPU time.
Random weights: later layers of a tile pass often settle before the formula's tic count (a prediction above the
measurement); trained nets such as the MNIST one do not.
"""
from __future__ import annotations

import argparse
import itertools
import os

import numpy as np

import doomformer as df
from doomformer.tilefleet import TileGames

TICK = os.sysconf("SC_CLK_TCK")
# (layer sizes, number of staircase thresholds K): every net below is a different shape
FIT = [([4, 6, 3], 8), ([12, 16, 6], 8), ([32, 32, 10], 8), ([64, 64, 10], 8), ([16, 48, 10], 8), ([64, 128, 10], 8),
       ([8, 8, 2, 2], 8), ([24, 40, 12, 6], 8), ([12, 16, 6], 3), ([32, 32, 10], 3), ([64, 64, 10], 3),
       ([16, 48, 10], 3), ([8, 8, 2, 2], 3), ([24, 40, 12, 6], 3)]
VALIDATION = [([8, 12, 4], 3), ([24, 24, 8], 3), ([48, 32, 10], 3), ([100, 32, 10], 3), ([20, 20, 20, 10], 3),
              ([128, 64, 10], 3), ([32, 64, 64, 10], 3), ([200, 50, 10], 3), ([8, 12, 4], 8), ([24, 24, 8], 8),
              ([48, 32, 10], 8)]
WIDE_FIT = [([300, 40, 10], 3), ([500, 64, 10], 3), ([400, 100, 10], 3), ([784, 32, 10], 2), ([600, 48, 16], 2)]


def engine_cpu() -> float:
    """User+sys seconds of this process's child processes (the ViZDoom engines) so far."""
    me, total = os.getpid(), 0.0
    for name in os.listdir("/proc"):
        if name.isdigit():
            try:
                with open(f"/proc/{name}/stat") as f:
                    st = f.read().rsplit(")", 1)[1].split()
            except OSError:
                continue
            if int(st[1]) == me:
                total += (int(st[11]) + int(st[12])) / TICK
    return total


def make(shape, K, seed=0) -> df.MLP:
    rng = np.random.default_rng(seed)
    W = [rng.integers(-3, 4, (b, a)) for a, b in zip(shape, shape[1:])]
    B = [rng.integers(-10, 11, b) for b in shape[1:]]
    T = [np.arange(1, K + 1) * (20 if l == 0 else 12 * max(1, shape[l] // 8)) for l in range(len(W))]
    return df.MLP(W, B, T)


def an_input(net, seed=1):
    rng = np.random.default_rng(seed)
    for _ in range(100):
        x = rng.integers(-100, 101, net.n_in)
        try:
            net.check(x)
            return x
        except df.RangeError:
            pass
    raise RuntimeError("no input within the exact range")


def measure_map(net, x, passes: int) -> float:
    with df.compile(net, mode="map") as brain:
        brain(x)
        c0 = engine_cpu()
        for _ in range(passes):
            brain(x)
        return (engine_cpu() - c0) / passes


def measure_tiles(net, x, passes: int) -> tuple[float, list[dict]]:
    with df.compile(net, mode="tiles") as brain:
        directory = brain.wad
    with TileGames(net, directory) as games:
        games.run(x)
        values = {f"{s}x{i}": int(v) * (-1 if s else 1) for i, v in enumerate(x) for s in ("", "-")}
        per = [dict(cpu=0.0, tics=0) for _ in games.runners]
        for _ in range(passes):
            v = values
            for p, runner in zip(per, games.runners):
                c0 = engine_cpu()
                out, tics, _, _ = runner.run(v)
                p["cpu"] += (engine_cpu() - c0) / passes
                p["tics"] = tics
                v = out
        return sum(p["cpu"] for p in per), per


def nnls(A, y):
    """Least squares with nonnegative coefficients (columns <= 4: try every subset)."""
    best = None
    for k in range(1, A.shape[1] + 1):
        for cols in itertools.combinations(range(A.shape[1]), k):
            c = np.linalg.lstsq(A[:, cols], y, rcond=None)[0]
            if (c >= 0).all():
                r = ((A[:, cols] @ c - y) ** 2).sum()
                if best is None or r < best[0]:
                    full = np.zeros(A.shape[1])
                    full[list(cols)] = c
                    best = (r, full)
    return best[1]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--no-map", action="store_true", help="skip the one-map passes (they are most of the time)")
    ap.add_argument("--refit", action="store_true", help="also run the fit nets and print least-squares constants")
    args = ap.parse_args()
    if not os.path.isdir("/proc/self"):
        raise SystemExit("costcheck reads the engine's CPU time from /proc (Linux)")
    nets = [(s, K, "val") for s, K in VALIDATION] + ([(s, K, "fit") for s, K in FIT + WIDE_FIT] if args.refit else [])
    print(f"{'net':<18}{'K':>2} {'set':<4}{'sectors':>8}{'map tics':>9} {'map pred':>9}{'map meas':>9}{'err':>6}"
          f"{'tile tics':>10}{'pred':>8}{'meas':>8}{'err':>6}  (seconds, engine CPU; load {os.getloadavg()[0]:.0f})", flush=True)
    fit_map, fit_tile = [], []
    errors = {"map": [], "tiles": []}
    for shape, K, role in nets:
        net = make(shape, K)
        x = an_input(net)
        p = df.plan(net)
        m, t = p["map"], p["tiles"]
        mm = None
        if not args.no_map and shape[0] < 300:                 # the wide fit nets (tiles only) have maps of 150k+ sectors
            mm = measure_map(net, x, 12 if m["sectors"] < 30000 else 2)
            if role == "fit":
                fit_map.append((m["sectors"] * m["estimated_settle_tics"], m["dolls"] * m["estimated_settle_tics"], mm))
        tm, per = measure_tiles(net, x, 6)
        if role == "fit":
            for pt, q in zip(t["per_tile"], per):
                fit_tile.append((pt["sectors"] * q["tics"], pt["dolls"] * q["tics"], q["cpu"]))
        else:
            errors["tiles"].append((t["predicted_seconds"] / tm - 1) * 100)
            if mm:
                errors["map"].append((m["predicted_seconds"] / mm - 1) * 100)
        mcols = f"{m['predicted_seconds']:>9.2f}{mm:>9.2f}{(m['predicted_seconds'] / mm - 1) * 100:>+5.0f}%" if mm else " " * 24
        print(f"{'-'.join(map(str, shape)):<18}{K:>2} {role:<4}{m['sectors']:>8,}{m['estimated_settle_tics']:>9,} {mcols}"
              f"{sum(q['tics'] for q in per):>5}/{t['predicted_tics']:<4}{t['predicted_seconds']:>8.3f}{tm:>8.3f}"
              f"{(t['predicted_seconds'] / tm - 1) * 100:>+5.0f}%", flush=True)
    for name, e in errors.items():
        if e:
            e = np.array(e)
            print(f"validation, {name}: {len(e)} nets, rms error {np.sqrt((e ** 2).mean()):.0f}%, "
                  f"largest {e[np.abs(e).argmax()]:+.0f}%")
    if args.refit and fit_map and fit_tile:
        A = np.array([[a, b] for a, b, _ in fit_map])
        y = np.array([c for *_, c in fit_map])
        c = nnls(A / y[:, None], np.ones(len(y)))
        print(f"refit one map: MAP_NS_PER_SECTOR_TIC = {c[0] * 1e9:.1f}, MAP_NS_PER_DOLL_TIC = {c[1] * 1e9:.0f}")
        A = np.array([[a, b, 1.0] for a, b, _ in fit_tile])
        y = np.array([c for *_, c in fit_tile])
        w = 1 / np.maximum(y, 0.02)
        c = nnls(A * w[:, None], y * w)
        print(f"refit one tile: TILE_NS_PER_SECTOR_TIC = {c[0] * 1e9:.1f}, TILE_NS_PER_DOLL_TIC = {c[1] * 1e9:.0f}, "
              f"TILE_RUN_MS = {c[2] * 1e3:.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
