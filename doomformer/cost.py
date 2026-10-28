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
PORT_US = 18.0                  # Python's send_game_command (measured by Perf: 18 us)


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
