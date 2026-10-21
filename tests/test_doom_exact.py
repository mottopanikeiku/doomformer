"""Bit-exact regression tests: small networks compiled to DOOM maps, run in ViZDoom, compared with NumPy.

Each test builds a map, runs it in a real (headless) DOOM game, reads the results back from the game (light levels
and plane heights via ACS) and compares them with an independent integer reference. ~1 minute in total.
Needs ViZDoom and the ACC compiler (built on first use).
"""
from __future__ import annotations

import numpy as np
import pytest

import doomformer.pure as pure
from doomformer.intref import adc_relu, adc_signed
from doomformer.learn import PerceptronRef, build_perceptrons, ports_for
from doomformer.ops import crossbar, neg, output_pair, output_single
from doomformer.runtime import TileRunner
from doomformer.spiking import IntNet, build_spiking, read_step, step_ports
from doomformer.tile import TileSpec, compile_tile


def both(name, v):
    return {name: int(v), neg(name): -int(v)}


def test_tile_crossbar(tmp_path):
    """One tile: z = b + W @ adc_signed(x), including a slow bus (inputs written late)."""
    rng = np.random.default_rng(1)
    thr = np.cumsum(rng.integers(20, 60, 4))
    W, b = rng.integers(-60, 61, (3, 4)), rng.integers(-200, 201, 3)
    spec = TileSpec("xbar", [], [])
    outs = [output_pair(spec, f"z{j}", b[j]) for j in range(3)]
    for i in range(4):
        crossbar(spec, f"x{i}", thr, [(outs[j], W[j, i]) for j in range(3)])
    compile_tile(spec, tmp_path / "xbar.wad")
    r = TileRunner(tmp_path / "xbar.wad")
    try:
        for case in range(6):
            x = rng.integers(-thr[-1] - 40, thr[-1] + 40, 4)
            z = b + W @ adc_signed(x, thr)
            got = r.run({k: v for i in range(4) for k, v in both(f"x{i}", x[i]).items()},
                        write_delay=17 if case % 2 else 0)[0]
            assert [got[f"z{j}"] for j in range(3)] == z.tolist(), f"case {case}, x={x.tolist()}"
    finally:
        r.close()


def test_two_layer_mlp_in_one_map(tmp_path):
    """Layer 0 writes layer 1's gate floors inside the map; no Python between the layers. Checks the outputs and
    the hidden values as layer 1 reads them (its gate-floor heights)."""
    rng = np.random.default_rng(0)
    N, Hd, O = 4, 5, 3
    thr0, thr1 = np.cumsum(rng.integers(20, 50, 3)), np.cumsum(rng.integers(15, 40, 3))
    W1, b1 = rng.integers(-25, 26, (Hd, N)), rng.integers(-30, 31, Hd)
    W2, b2 = rng.integers(-25, 26, (O, Hd)), rng.integers(-30, 31, O)
    s0, s1 = TileSpec("l0"), TileSpec("l1")
    hid = [output_pair(s0, f"h{j}", b1[j]) for j in range(Hd)]
    for i in range(N):
        crossbar(s0, f"x{i}", thr0, [(hid[j], W1[j, i]) for j in range(Hd)])
    out = [output_single(s1, f"y{o}", b2[o]) for o in range(O)]
    for j in range(Hd):
        crossbar(s1, f"h{j}", thr1, [(out[o], W2[o, j]) for o in range(O)], signed=False)
    pure.compile_pure([s0, s1], tmp_path / "mlp.wad", "mlp", readout=[f"-h{j}" for j in range(Hd)])
    r = TileRunner(tmp_path / "mlp.wad", recycle=False)
    try:
        for case in range(4):
            x = rng.integers(-thr0[-1] - 30, thr0[-1] + 30, N)
            h = b1 + W1 @ adc_signed(x, thr0)
            y = b2 + W2 @ adc_relu(h, thr1)
            got = r.run({k: v for i in range(N) for k, v in both(f"x{i}", x[i]).items()})[0]
            assert [-got[f"-h{j}"] for j in range(Hd)] == h.tolist(), f"hidden floors, case {case}"
            assert [got[f"y{o}"] for o in range(O)] == y.tolist(), f"case {case}, x={x.tolist()}"
    finally:
        r.close()


def _train(tmp_path, n_examples: int, cap: int = 100):
    """Online perceptron in one recycled map; returns (#exact examples, n_examples)."""
    rng = np.random.default_rng(0)
    F, K = 6, 3
    teacher = rng.integers(-3, 4, F)
    build_perceptrons(F, 1, K, tmp_path / "learner.wad", cap=cap)
    ref = PerceptronRef(F, 1, K, cap)
    r = TileRunner(tmp_path / "learner.wad", recycle=True)
    exact = 0
    try:
        for _ in range(n_examples):
            x = (rng.random(F) < 0.5).astype(np.int64)
            x[0] = 1
            y = np.array([1 if teacher @ x > 0 else -1])
            s = ref.step(x, y)
            got = r.run(ports_for(x, y))[0]
            exact += got["s0"] == s[0] and [got[f"w0_{i}"] for i in range(F)] == ref.w[0].tolist()
    finally:
        r.close()
    return exact


def test_perceptron_learns_exactly(tmp_path):
    """Weights are ceiling heights that persist between examples; every weight must match after every example."""
    assert _train(tmp_path, 25) == 25


def test_freeze_barriers_are_necessary(tmp_path, monkeypatch):
    """Negative control: without freeze barriers, dolls stuck at weight gates walk on when the learn layer moves
    those gates, and the check must notice."""
    monkeypatch.setattr(pure, "FREEZE", False)
    assert _train(tmp_path, 40) < 40


@pytest.mark.parametrize("leak", [1, 3])
def test_spiking_network_exact(tmp_path, leak):
    """Integer spiking neurons, one map run per time step; every spike and potential, every step."""
    rng = np.random.default_rng(leak)
    n, m = 12, 4
    W = np.where(rng.random((n, n)) < 0.25, rng.integers(-12, 16, (n, n)), 0)
    np.fill_diagonal(W, 0)
    U = np.where(rng.random((n, m)) < 0.4, rng.integers(4, 18, (n, m)), 0)
    net = IntNet([f"n{i}" for i in range(n)], [f"in{k}" for k in range(m)], W, U, theta=30, leak=leak)
    X = (rng.random((20, m)) < 0.4).astype(np.int64)
    V, S = net.run(X)
    meta = build_spiking(net, tmp_path / "spikes.wad")
    r = TileRunner(tmp_path / "spikes.wad", recycle=True)
    try:
        for t in range(len(X)):
            v, s = read_step(r.run(step_ports(X[t], meta))[0], n)
            assert s.tolist() == S[t].tolist(), f"spikes, step {t}"
            assert v.tolist() == V[t + 1].tolist(), f"potentials, step {t}"
        assert S.sum() > 0
    finally:
        r.close()
