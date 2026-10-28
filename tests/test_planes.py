"""The plane-move bound (doomformer/planes.py): a plane that starts a raise above 14,018 (or a lower below -14,018)
keeps the engine's mover busy and loses the next move on it. The reduced net from fuzz seed 902 is the regression
case: DOOM gave 12,254 where the reference says 11,999 and net.check accepted the input."""
from __future__ import annotations

import numpy as np
import pytest

import doomformer as df
from doomformer import facts, planes
from doomformer.intref import check_limits
from doomformer.kit import RangeError
from doomformer.onemap import tiny_model
from doomformer.spiking import IntNet

X = [-12000, 5933, 6252, -353, 11483]
THR = [[713, 1423, 3182, 11867], [8540]]


def reduced(bias: int, w3: int = 3) -> df.MLP:
    """The reduced net: one hidden unit whose plane -h1_0 starts at -bias and rises 4 * w3 and falls 2295 on X."""
    return df.MLP([[[w3, 255, 255, 0, 255]], [[-255]]], [[bias], [185]], THR)


def doom(net: df.MLP, x, tmp_path):
    with df.compile(net, tmp_path / "net.wad", mode="map", guard=True, limits=False) as brain:
        return brain(np.asarray(x))


def test_the_replay_net_is_refused_with_the_bound_named(tmp_path):
    net = reduced(-14282)
    with pytest.raises(RangeError, match="14294 > 14018"):                     # per input: b + P = 14282 + 12
        net.check(X)
    with pytest.raises(RangeError, match=r"worst case over all inputs.*> 14018"):
        df.compile(net, tmp_path / "a.wad", mode="map")
    with pytest.raises(RangeError, match="14018"):
        with df.compile(net, tmp_path / "b.wad", mode="map", limits=False) as brain:    # the opt-out keeps check(x)
            brain(np.asarray(X))
    net.check(X, planes=False)                                                 # tile mode moves no planes: accepted


def test_doom_is_exact_up_to_the_bound_and_refused_beyond(tmp_path):
    """The rule is the engine's edge, conservative by one move: bias -14,006 puts the partial sum at 14,018 (accepted,
    exact), -14,007 at 14,019 (refused). DOOM itself is still exact at -14,009 (the 4th raise starts at 14,018) and
    loses a 255 move at -14,010 (it starts at 14,019). A plane that never rises may start beyond the bound: weight 3 -> 0
    at bias -14,282 is accepted and exact."""
    ok = reduced(-14006)
    ok.check(X)
    assert doom(ok, X, tmp_path).tolist() == ok.reference(X)["y"].tolist()
    with pytest.raises(RangeError, match="plane partial sum"):
        reduced(-14007).check(X)
    rows = {r["bias"]: r for r in planes.sweep([-14009, -14010], workdir=tmp_path)}          # check bypassed
    assert rows[-14009]["exact"] and not rows[-14009]["check_accepts"]
    assert not rows[-14010]["exact"] and rows[-14010]["doom"] - rows[-14010]["reference"] == 255
    no_raise = reduced(-14282, w3=0)
    no_raise.check(X)
    assert doom(no_raise, X, tmp_path).tolist() == no_raise.reference(X)["y"].tolist()


def test_the_engine_fact_mover_busy_holds():
    rows = facts.run(only=["mover_busy"])
    assert [r.ok for r in rows] == [True], rows[0].observed


def test_static_bound_is_sound_for_random_nets():
    """If the all-inputs bound accepts a net, check(x) never refuses an input for its planes; and the all-inputs peak
    is at least the peak of every input."""
    rng = np.random.default_rng(5)
    accepted = refused = 0
    for _ in range(150):
        n_in, mid = int(rng.integers(3, 7)), int(rng.integers(1, 5))
        W0 = rng.integers(-255, 256, (mid, n_in)) * (rng.random((mid, n_in)) < 0.8)
        W1 = rng.integers(-50, 51, (2, mid))
        W0[0, 0] = W0[0, 0] or 7
        W1[0, 0] = W1[0, 0] or 1
        b0 = rng.integers(8000, 13001, mid) * rng.choice([-1, 1], mid)
        net = df.MLP([W0, W1], [b0, [0, 0]], [rng.integers(1, 40, 4).cumsum(), [1000]], signed_input=True)
        peaks = dict(planes.mlp_static_peaks(net))
        static_ok = not net.plane_errors()
        for _ in range(12):
            x = rng.integers(-12000, 12001, n_in)
            try:
                net.check(x)
            except RangeError as e:
                if "plane partial sum" in str(e):
                    refused += 1
                    assert not static_ok
                    assert int(str(e).split("reaches ")[1].split(" ")[0]) <= peaks["-h1"]
                continue
            accepted += 1
        assert static_ok == (peaks["-h1"] <= planes.PLANE_MAX)
    assert accepted and refused


def test_spiking_step_is_checked_against_the_plane_bound():
    net = IntNet(["a", "b"], ["u"], [[0, 0], [3000, 0]], [[0], [0]], theta=1, leak=0)
    assert planes.spiking_step_errors(net, [0, 0], [0]) == []
    assert planes.spiking_step_errors(net, [1, 0], [0]) == []                 # b's plane starts at 0, raises by 3000
    assert planes.spiking_step_errors(net, [1, 12000], [0])                   # b: 12,000 - 1 + 3,000 > 14,018
    assert planes.spiking_step_errors(net, [0, 12000], [0]) == []             # a does not spike: no raise


def test_transformer_limits_include_the_plane_bound():
    m = tiny_model()
    seqs = [[0, 1, 2]]
    assert check_limits(m, seqs, planes=True) == []
    m.p["L0.bup"] = np.full_like(m.p["L0.bup"], 255)                           # every hidden unit counts its top level
    m.p["L0.thr_m"] = np.array([1, 2])
    m.p["L0.Wdn"] = np.ones_like(m.p["L0.Wdn"])                                # so z2 has raise terms
    m.p["L0.bdn"] = np.full_like(m.p["L0.bdn"], 14015)                         # and starts high (also a weight-range error)
    errs = check_limits(m, seqs, planes=True)
    assert any("plane partial sums of L0.z2" in e for e in errs), errs
    assert not any("plane partial sums" in e for e in check_limits(m, seqs))   # tile mode: no planes
    with pytest.raises(RangeError):
        df.compile(m, mode="map")
