"""Tile-mode run_many (resident tile games per worker), the engine watchdog and the plan cost model."""
import os
import signal
import subprocess
import sys
import textwrap
import time

import numpy as np
import pytest

import doomformer as df
from doomformer import cost
from doomformer.fleet import run_many
from doomformer.tilefleet import TileGames

NET = dict(weights=[[[2, -1, 1], [-1, 2, 0], [1, 1, -2]], [[1, -2, 1], [2, 1, -1]]],
           biases=[[1, -1, 0], [3, -2]], thresholds=[[1, 3, 5], [1, 2]])


def small():
    return df.MLP(**NET)


def inputs(n, seed=0):
    return np.random.default_rng(seed).integers(-6, 7, size=(n, 3))


def test_tiles_mode_equals_map_mode_and_the_reference():
    net, xs, seen = small(), inputs(6), []
    ys = run_many(net, xs, workers=2, mode="tiles", progress=seen.append)
    assert [e["index"] for e in sorted(seen, key=lambda e: e["index"])] == list(range(6))
    assert {e["checked"] for e in seen} == {len(net.reference(xs[0])["hidden"][0]) * 2 + net.n_out}
    for x, y in zip(xs, ys):
        np.testing.assert_array_equal(y, net.reference(x)["y"])
    for x, y in zip(xs[:2], run_many(net, xs[:2], workers=1, mode="map")):
        np.testing.assert_array_equal(y, net.reference(x)["y"])


def test_tiles_mode_refuses_what_it_cannot_run():
    with pytest.raises(ValueError, match="tiles"):
        run_many(small(), inputs(1), mode="tiles", backend="ray")
    with pytest.raises(ValueError, match="mode"):
        run_many(small(), inputs(1), mode="fast")


def test_tile_games_compare_every_value_at_tolerance_zero():
    net, x = small(), inputs(1)[0]
    with df.compile(net, mode="tiles") as brain:
        directory = brain.wad
    with TileGames(net, directory) as games:
        r = games.run(x)                                           # twice: the second pass reuses the games
        r = games.run(x)
        np.testing.assert_array_equal(r["y"], net.reference(x)["y"])
        real = net.reference

        def off_by_one(x):
            ref = real(x)
            ref["y"] = ref["y"] + np.eye(net.n_out, dtype=np.int64)[0]
            return ref
        net.reference = off_by_one
        with pytest.raises(df.MismatchError):
            games.run(x)


ENGINE_SCRIPT = textwrap.dedent("""
    import sys, time
    import doomformer as df
    from doomformer import _mp
    net = df.MLP([[[1, -1], [2, 1]]], [[0, 0]], [[1, 2]])
    brain = df.compile(net)
    before = _mp.children()
    brain.runner
    print(*sorted(_mp.children() - before), flush=True)
    time.sleep(60)
""")


@pytest.mark.skipif(os.name == "nt", reason="the watchdog needs sh and ps")
@pytest.mark.parametrize("sig", [signal.SIGTERM, signal.SIGKILL])
def test_engine_does_not_outlive_a_killed_parent(sig):
    p = subprocess.Popen([sys.executable, "-c", ENGINE_SCRIPT], stdout=subprocess.PIPE, text=True,
                         env=dict(os.environ, OMP_NUM_THREADS="1"))
    try:
        engine = int(p.stdout.readline().split()[0])
        p.send_signal(sig)
        p.wait(timeout=10)
        end = time.time() + 15
        while time.time() < end and subprocess.run(["ps", "-o", "comm=", "-p", str(engine)], capture_output=True,
                                                   text=True).stdout.strip():
            time.sleep(0.25)
        assert not subprocess.run(["ps", "-o", "comm=", "-p", str(engine)], capture_output=True,
                                  text=True).stdout.strip(), "the engine outlived its killed parent"
    finally:
        p.kill()
        p.wait()


def test_cost_model_is_monotone_and_recommends_tiles_for_a_wide_net():
    assert cost.map_seconds(10_000, 1_000) < cost.map_seconds(20_000, 1_000) < cost.map_seconds(20_000, 2_000)
    rng = np.random.default_rng(0)
    wide = df.MLP([rng.integers(-2, 3, (64, 784)), rng.integers(-2, 3, (10, 64))], [np.zeros(64, np.int64),
                  np.zeros(10, np.int64)], [np.arange(1, 5) * 10, np.arange(1, 5) * 100])
    p = df.plan(wide)
    assert p["map"]["predicted_seconds"] > p["tiles"]["predicted_seconds"]
    assert p["recommended"] == "tiles" and p["mode"] == "map"      # auto still builds one map when it fits
    tiny = df.plan(small())
    assert tiny["map"]["predicted_seconds"] > 0 and tiny["tiles"]["predicted_seconds"] > 0
