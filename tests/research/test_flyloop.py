"""The closed fly loop (demos/flyloop.py): the PRNG/input replica, and one small real-DOOM loop."""
from __future__ import annotations

import numpy as np

from demos import flyloop as F
from doomformer.spiking import IntNet


def _tiny_net() -> IntNet:
    """Sugar lines drive MN9_left (its spikes thrust him toward the food); bitter lines drive an inhibitor."""
    W = np.array([[0, -20, 0], [0, 0, 0], [5, 0, 0]])
    U = np.array([[8, 8, 0, 0], [0, 0, 6, 6], [0, 0, 0, 0]])
    return IntNet(["MN9_left", "h0", "h1"], ["s0", "s1", "b0", "b1"], W, U, theta=10, leak=1,
                  meta=dict(input_groups={"sugar": [0, 1], "bitter": [2, 3]},
                            input_p_per_step={"sugar": 0.5, "bitter": 0.5}))


def test_lehmer_is_the_minimal_standard_without_overflow():
    x = 1
    for _ in range(10000):
        x = F.lehmer(x)
    assert x == 1043618065                      # Park & Miller (1988): the 10,000th value from seed 1
    rng = np.random.default_rng(0)
    for x in [1, 2, F.SCHRAGE_Q - 1, F.SCHRAGE_Q, F.SCHRAGE_Q + 1, F.M31 - 2] + rng.integers(1, F.M31 - 1, 2000).tolist():
        assert F.lehmer(x) == F.LEHMER_A * x % F.M31
        hi, lo = divmod(x, F.SCHRAGE_Q)
        assert max(F.LEHMER_A * lo, F.SCHRAGE_R * hi) < 2 ** 31      # Schrage: every intermediate fits in int32


def test_input_stream_does_not_depend_on_what_he_senses():
    net = _tiny_net()
    groups, line_group, thr = F.groups_of(net)
    spec = F.LoopSpec(wad="", seed=12345, period=1, calibration={}, stations=[], groups=groups,
                      line_group=line_group, thresholds=thr, outputs={}, status=[])
    a, b = F.Replica(net, spec), F.Replica(net, spec)
    on = []
    for t in range(4000):
        xa, _, sa = a.inputs(0b01)              # sugar only
        xb, _, sb = b.inputs(0b00 if t % 2 else 0b11)
        assert sa == sb                        # every line draws every step
        assert xa[2:].sum() == 0
        if t % 2 == 0:
            assert (xb[:2] == xa[:2]).all()    # same draws, gated by the group mask
        else:
            assert xb.sum() == 0
        on.append(xa[:2])
    assert abs(np.mean(on) - 0.5) < 0.02


def test_closed_loop_in_one_doom_game(tmp_path):
    """The ACS clock runs every step; sugar -> MN9 thrusts -> he reaches the food, eats, and sugar stops. Every
    spike/potential exact, the in-map PRNG equal to the replica, and Python sent only readout commands."""
    net = _tiny_net()
    spec = F.compile_flyloop(net, tmp_path / "tiny_loop.wad", stations=(F.Station("sugar", 30, 3),), seed=7)
    run = F.LoopRun(spec, net)
    masks, food = [], []
    try:
        for _ in range(16):
            rec = run.step()
            masks.append(rec["status"]["mask"])
            food.append(rec["status"]["food0"])
    finally:
        run.close()
    assert run.stats["steps"] == run.stats["exact"] == run.stats["prng_ok"] == run.stats["settled"] == 16
    assert not run.audit()["forbidden"]
    eaten = food.index(0)              # the step in whose compute window his last lunge reached the food ...
    assert 1 in masks[:eaten] and set(masks[eaten + 1:]) == {0}    # ... sugar was sensed before, never after
