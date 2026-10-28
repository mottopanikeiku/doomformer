"""Time prediction for one pass: `plan()["map"]["predicted_seconds"]` and `["tiles"]["predicted_seconds"]`.

The engine polls every sector every tic (the idle conveyor scrollers, docs/scaling.md), and every walking doll costs
a little more, so the engine time of a pass is

    one map:   tics * (sectors * MAP_NS_PER_SECTOR_TIC + dolls * MAP_NS_PER_DOLL_TIC) / 1e9
    one tile:  tics * (sectors * TILE_NS_PER_SECTOR_TIC + dolls * TILE_NS_PER_DOLL_TIC) / 1e9 + TILE_RUN_MS / 1e3
    tiles:     the sum over the tiles of the net, plus PORT_US per port Python writes (the copying between games)

`tics` of a map is plan's estimated_settle_tics; of a tile it is tile_tics(zone), the settle time of a walk through a
zone of that length rounded up to the 8-tic window TileRunner.settle checks. The constants are least-squares fits of
engine CPU seconds on random MLPs (demos/costcheck.py --refit; docs/scaling.md, "Cost model", has the table and
the error on other nets). The result is engine seconds per pass with the games already loaded, as wall time on an
otherwise idle core; a loaded machine takes 1.3 to 2.5 times as long (measured). A faster or slower CPU scales it;
`sector_tics` (sectors x tics, summed over the tiles) is machine independent.
"""
from __future__ import annotations

import math

MAP_NS_PER_SECTOR_TIC = 102.1
MAP_NS_PER_DOLL_TIC = 1060.0
TILE_NS_PER_SECTOR_TIC = 69.7
TILE_NS_PER_DOLL_TIC = 2603.0
TILE_RUN_MS = 0.24
PORT_US = 18.0                  # Python's send_game_command (measured: 18 us)


def tile_tics(zone: int) -> int:
    """Upper estimate of the settle tics of a tile whose longest zone is `zone` map units long: 30 tics for the start
    sequence and the lane plus the zone at 16 units per tic, rounded up to the 8-tic window of settle. A pass in which
    no doll of the longest block passes its gate settles earlier (random hidden values do; MNIST's do not)."""
    return 8 * math.ceil((30 + zone / 16) / 8)


def map_seconds(sectors: int, tics: int, dolls: int = 0) -> float:
    return tics * (sectors * MAP_NS_PER_SECTOR_TIC + dolls * MAP_NS_PER_DOLL_TIC) / 1e9


def tile_seconds(sectors: int, tics: int, dolls: int = 0, ports: int = 0) -> float:
    return (tics * (sectors * TILE_NS_PER_SECTOR_TIC + dolls * TILE_NS_PER_DOLL_TIC) / 1e9 + TILE_RUN_MS / 1e3
            + ports * PORT_US / 1e6)


def annotate(report: dict, kind: str) -> None:
    """Add predicted_tics, sector_tics and predicted_seconds to the report's map and tiles entries, and `recommended`
    ("map" or "tiles") with `recommended_ratio` (slower / faster predicted seconds). Tiles are predicted for MLPs
    only: a transformer's tile executions per forward depend on the sequence. Without a prediction for both, the
    recommendation is what auto builds (the map when it fits)."""
    m, t = report["map"], report["tiles"]
    if m.get("estimated_settle_tics") and m.get("sectors"):
        n = m["estimated_settle_tics"]
        m["predicted_tics"] = n
        m["sector_tics"] = m["sectors"] * n
        m["predicted_seconds"] = map_seconds(m["sectors"], n, m["dolls"])
    if t is not None and kind == "mlp":
        tics = [tile_tics(p["zone"]) for p in t["per_tile"]]
        t["predicted_tics"] = sum(tics)
        t["sector_tics"] = sum(p["sectors"] * n for p, n in zip(t["per_tile"], tics))
        t["predicted_seconds"] = sum(tile_seconds(p["sectors"], n, p["dolls"], p["ports"])
                                     for p, n in zip(t["per_tile"], tics))
    ok = [(name, o) for name, o in (("map", m), ("tiles", t)) if o is not None and o.get("fits")]
    if not ok:
        return
    timed = sorted((o["predicted_seconds"], name) for name, o in ok if "predicted_seconds" in o)
    if len(timed) == 2:
        report["recommended"] = timed[0][1]
        report["recommended_ratio"] = round(timed[1][0] / timed[0][0], 1)
    else:
        report["recommended"] = ok[0][0]


# --- settle tics: the geometry estimate corrected with measured settles -----------------------------------------------
# pure.compile_pure's `estimated_settle_tics` is an upper bound that the runner's timeout uses; it is off by -2 to +26 percent
# (docs/scheduling.md). The corrected value follows the clock's timeline: per layer, the first release (sync wait),
# the waves after it, the extra chunks of a long clock, the corridor, the clock margin and the copy settle, all at 16
# units per tic, plus a fixed number of tics per layer and for the start. The two fixed numbers are least squares fits
# of measured settles of 44 (net, opt) pairs (8 reference nets x opt 0..3 and MNIST, doomnames_onemap and its softmax
# twin; models/schedule/plan_calibration.json has every row, `python -m doomformer.cost --refit` re-fits).
SETTLE_PER_LAYER, SETTLE_START = 4.78, 29.8
# error band of the corrected value, tics: +-max(SETTLE_BAND_TICS, SETTLE_BAND_FRACTION * tics). Largest errors seen: 20
# tics on the fit rows (10 percent, a 152-tic net), 40 tics on the held-out rows (0.7 percent, induction_onemap at opt 0),
# 4.6 percent at most on the held-out MLPs at opt 3 (models/schedule/plan_calibration.json)
SETTLE_BAND_TICS, SETTLE_BAND_FRACTION = 20, 0.015


def settle_terms(layout: dict) -> dict:
    """Tics of the timeline terms of a one-map layout (`compile_pure(layout_only=True).layout`): wave, chunk, sync,
    corridor, margin, copy, and the layer count."""
    from .pure import Timing
    t = Timing(**layout["timing"]) if "timing" in layout else Timing()
    f = dict(wave=0.0, chunk=0.0, sync=0.0, corr=0.0, margin=0.0, copy=0.0, layers=0)
    for s in layout["stage_stats"]:
        S = s["wave_spacing"]
        chunks = -(-s["waves"] // max(1, 12000 // max(S, t.serial)))
        f["wave"] += (s["waves"] - 1) * S / 16
        f["chunk"] += (chunks - 1) * (S + t.tele_gap + t.chunk_lead) / 16
        f["sync"] += s.get("sync_wait", t.sync_wait) / 16
        f["corr"] += s["corridor_width"] / 16
        f["margin"] += t.clock_margin / 16
        f["copy"] += t.copy_settle / 16 if s.get("copy") else 0
        f["layers"] += 1
    return f


def settle_estimate(layout: dict) -> dict:
    """{"calibrated_tics": n, "tics_band": +-tics, "tics_error_pct": band as percent of n} for a one-map layout."""
    f = settle_terms(layout)
    tics = SETTLE_START + SETTLE_PER_LAYER * f.pop("layers") + sum(f.values())
    n = int(round(tics))
    band = max(SETTLE_BAND_TICS, math.ceil(SETTLE_BAND_FRACTION * n))
    return dict(calibrated_tics=n, tics_band=band, tics_error_pct=round(100 * band / n, 1))


def refit(path=None) -> dict:
    """Least-squares fit of SETTLE_PER_LAYER and SETTLE_START on the "fit" rows of models/schedule/plan_calibration.json
    (measured settle minus the timeline terms = per_layer * layers + start), and the errors of every row, fit and held
    out, under the constants in this module and under the new fit."""
    import json
    from pathlib import Path

    import numpy as np
    path = Path(path) if path else Path(__file__).resolve().parent.parent / "models" / "schedule" / "plan_calibration.json"
    data = json.loads(path.read_text())

    def base(r):
        return sum(v for k, v in r["terms"].items() if k != "layers")

    fit = data["fit"]
    A = np.array([[r["terms"]["layers"], 1.0] for r in fit])
    y = np.array([r["measured"] - base(r) for r in fit])
    (per_layer, start), *_ = np.linalg.lstsq(A, y, rcond=None)

    def errors(rows, pl, st):
        out = []
        for r in rows:
            pred = st + pl * r["terms"]["layers"] + base(r)
            out.append((r["net"], r["opt"], r["measured"], round(pred, 1), 100 * (pred - r["measured"]) / r["measured"],
                        pred - r["measured"]))
        return out

    return dict(per_layer=round(float(per_layer), 2), start=round(float(start), 1),
                fit=errors(fit, SETTLE_PER_LAYER, SETTLE_START), heldout=errors(data["heldout"], SETTLE_PER_LAYER,
                                                                                SETTLE_START))


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="re-fit the settle model of `plan` on models/schedule/plan_calibration.json")
    ap.add_argument("--refit", action="store_true", required=True)
    ap.add_argument("path", nargs="?")
    r = refit(ap.parse_args().path)
    print(f"fitted per layer {r['per_layer']}, start {r['start']} (module: {SETTLE_PER_LAYER}, {SETTLE_START})")
    for label in ("fit", "heldout"):
        rows = r[label]
        rel = [abs(e[4]) for e in rows]
        print(f"{label}: {len(rows)} rows, max |error| {max(rel):.1f} %, max {max(abs(e[5]) for e in rows):.0f} tics, "
              f"median {sorted(rel)[len(rel) // 2]:.1f} %")
        for n, o, m, p, pct, d in rows:
            print(f"  {n:28s} opt {o}  measured {m:5d}  model {p:7.1f}  {pct:+5.1f} %")

