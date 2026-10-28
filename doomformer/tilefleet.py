"""Many MLP passes through operation tiles, one process per game set: df.run_many(net, xs, mode="tiles").

This is the multi-map design, not the one-map one. Each worker process holds one resident DOOM game per tile of the
net (one per layer: mlp_l0 ... mlp_l{n-1}) and takes digits one after another. For every input it runs the layers in
order; PYTHON COPIES the integer values of layer l (hidden values and the DOOM-computed negation twins) into the ports
of layer l+1, as TiledMLPBrain does. Only one of a worker's games advances at a time, so `workers` games are active at
once and `workers x layers` engine processes exist (the idle ones only hold memory).

Every pass is checked against the integer reference at tolerance 0, the same values as the one-map path: every hidden
value with its twin and every output of net.reference(x).

Why it is faster than one map: a one-map layer releases its dolls in serial waves (784 waves for MNIST layer 0) and
the engine polls every sector every tic, so cost is about sectors x tics. A tile has the sectors of one layer and the
dolls of a layer run side by side (docs/scaling.md).
"""
from __future__ import annotations

import os
import signal
import time
from pathlib import Path

import numpy as np

from . import _mp
from .kit import MismatchError, ReadoutError


class TileGames:
    """The resident games of an MLP's tiles in this process: run(x) is one checked pass."""

    def __init__(self, net, directory):
        from .runtime import TileRunner
        self.net = net
        self.exposed = net.exposed_names()                    # the names a pass must return (a dropped readout fails)
        self.runners: list = []
        try:
            for l in range(net.n_layers):
                self.runners.append(TileRunner(Path(directory) / f"mlp_l{l}.wad"))
        except BaseException:
            self.close()
            raise

    def run(self, x) -> dict:
        """Forward pass of x through the tiles: returns {"y", "got", "tics"}; raises MismatchError on any difference
        from net.reference(x) (every exposed value, tolerance 0)."""
        net = self.net
        values = {f"{sign}x{i}": int(v) * (-1 if sign else 1) for i, v in enumerate(x) for sign in ("", "-")}
        got, tics = {}, 0
        for runner in self.runners:
            out, t, _, _ = runner.run(values)
            got.update(out)
            tics += t
            values = out
        if set(got) != self.exposed:
            raise ReadoutError(f"tiled MLP pass x={np.asarray(x).tolist()}: the readback names differ from the "
                               f"lowering's (missing {sorted(self.exposed - set(got))[:6]}, unexpected "
                               f"{sorted(set(got) - self.exposed)[:6]})")
        ref = net.reference(x)
        bad = [(n, v, net._expected(n, ref)) for n, v in got.items() if v != net._expected(n, ref)]
        if bad:
            raise MismatchError(f"tiled MLP pass x={np.asarray(x).tolist()}", bad)
        return dict(y=np.array([got[f"y{o}"] for o in range(net.n_out)], dtype=np.int64), checked=len(got), tics=tics)

    def close(self):
        while self.runners:
            self.runners.pop().close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


def _pass(games: TileGames, x) -> tuple:
    """("ok", y, checked, tics, seconds), ("mismatch", mismatches) or ("error", text): fleet._pass's shapes."""
    t0 = time.perf_counter()
    try:
        r = games.run(x)
    except MismatchError as e:
        return ("mismatch", e.mismatches)
    except Exception as e:
        return ("error", f"{type(e).__name__}: {e}")
    return ("ok", r["y"], r["checked"], r["tics"], time.perf_counter() - t0)


def _terminated(signum, frame):
    raise SystemExit(0)                        # unwinds the worker, so `finally` closes its games


def tile_worker(net, directory: str, tasks, results, game: int):
    """fleet._run_local worker: opens the tile games, then runs (index, x) jobs until None."""
    if hasattr(os, "setpgrp"):
        os.setpgrp()                           # the engine processes share this group (orchestrator.stop_processes)
    signal.signal(signal.SIGTERM, _terminated)
    _mp.exit_with_parent()
    games = None
    try:
        try:
            games = TileGames(net, directory)
        except Exception as e:
            results.put((None, game, "error", f"starting the tile games: {type(e).__name__}: {e}"))
            return
        while True:
            job = tasks.get()
            if job is None:
                break
            i, x = job
            results.put((i, game) + _pass(games, x))
    finally:
        if games is not None:
            games.close()
