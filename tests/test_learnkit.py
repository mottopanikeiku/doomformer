"""df.Perceptron / Learner: the in-map perceptron through the public toolkit API, in real headless DOOM games.

Every `learner.step` compares scores, action, verdicts and ALL weights (and their negated twins) read back from the game
with the integer twin; these tests check that it does so (negative controls included), that predictions do not learn, that
caps bind, and that the world/actions hooks are driven by DOOM's own decisions. At most one game at a time.
"""
from __future__ import annotations

import numpy as np
import pytest

import doomformer as df
import doomformer.pure as pure
from doomformer.learn import PerceptronRef
from doomformer.learnkit import Learner, Perceptron

LEFT, RIGHT_TAG = 11, 12
LIT, DIM = 255, 40


def rand_example(rng, n: int, heads: int, labels=(-1, 0, 1)):
    x = (rng.random(n) < 0.5).astype(np.int64)
    x[0] = 1
    return x, rng.choice(labels, heads)


def test_validation_and_plan():
    for bad in (dict(n_features=0), dict(n_features=3, heads=0), dict(n_features=3, clip=0),
                dict(n_features=3, cap=2, clip=3), dict(n_features=3, cap=5000)):
        with pytest.raises(ValueError):
            Perceptron(**bad)
    with pytest.raises(TypeError):
        Perceptron(2.5)
    p = Perceptron(4, 2, cap=7, clip=2)
    plan = p.plan()
    assert plan["fits"] and plan["dolls"] > 0 and plan["sectors"] > plan["dolls"]
    assert p.action_names() == ["pick0", "pick1", "pick2", "right0", "right1", "wrong0", "wrong1"]
    ref = p.reference()
    assert isinstance(ref, PerceptronRef) and (ref.n_feat, ref.heads, ref.K, ref.cap) == (4, 2, 2, 7)
    assert not ref.w.any()
    with pytest.raises(KeyError):
        p.compile(actions={"y0": []})
    with pytest.raises(TypeError):
        p.compile(actions={"pick0": [3]})


def test_multi_head_steps_are_exact_and_weights_come_from_doom(tmp_path):
    p = Perceptron(4, 3, cap=100, clip=3)
    rng = np.random.default_rng(4)
    with p.compile(tmp_path / "p.wad") as learner:
        assert isinstance(learner, Learner) and learner.weights.shape == (3, 4)
        assert not learner.weights.any()                             # read from the game before any run
        seen = set()
        for _ in range(30):
            x, y = rand_example(rng, 4, 3)
            scores, action = learner.step(x, y)                      # raises MismatchError on any difference
            assert scores.shape == (3,) and 0 <= action <= 3
            np.testing.assert_array_equal(learner.weights, learner.twin.w)
            assert (action == 0) == bool((scores <= 0).all())
            if action:
                assert scores[action - 1] >= 1 and (scores[:action - 1] <= 0).all()
            seen.add(action)
            assert learner.info["checked"] == 3 + 1 + 4 + 6 + 2 * 12
        assert len(seen) >= 3 and learner.info["steps"] == 30 and 0 < learner.info["lessons"] <= 30
        w = learner.weights.copy()
        steps = learner.info["steps"]
        s1, a1 = learner.predict([1, 1, 0, 1])                    # no label: no lesson
        np.testing.assert_array_equal(learner.weights, w)
        s2, a2 = learner.predict([1, 1, 0, 1])
        np.testing.assert_array_equal(s1, s2)
        assert a1 == a2 and learner.info["steps"] == steps + 2
        np.testing.assert_array_equal(s1, np.clip(w, -3, 3) @ np.array([1, 1, 0, 1]))
        for bad_x, bad_y in (([1, 0, 2, 0], [0, 0, 0]), ([1, 0, 0], [0, 0, 0]), ([1, 0, 0, 0], [0, 0]),
                             ([1, 0, 0, 0], [0, 2, 0]), ([0.5, 0, 0, 0], [0, 0, 0])):
            with pytest.raises((ValueError, TypeError)):
                learner.step(bad_x, bad_y)
        assert learner.info["steps"] == steps + 2                    # rejected before DOOM ran


def test_learns_a_rule_and_the_cap_binds(tmp_path):
    """3-feature majority + bias, with 15% label noise and a tiny cap so the cap gate has to stop lessons."""
    p = Perceptron(4, 1, cap=2, clip=2)
    rng = np.random.default_rng(7)
    with p.compile(tmp_path / "cap.wad") as learner:
        for n in range(60):
            x = np.array([1, *rng.integers(0, 2, 3)])
            y = 1 if x[1:].sum() >= 2 else -1
            if rng.random() < 0.15:
                y = -y
            learner.step(x, [y])
        w = learner.weights
        assert np.abs(w).max() == 2                                   # a weight sat on the cap and stayed in range
        assert (np.abs(w) <= 2).all()
        np.testing.assert_array_equal(w, learner.twin.w)


def test_learns_the_majority_rule(tmp_path):
    p = Perceptron(4, 1)
    rng = np.random.default_rng(3)
    rows = [np.array([1, a, b, c]) for a in (0, 1) for b in (0, 1) for c in (0, 1)]
    with p.compile(tmp_path / "maj.wad") as learner:
        acc = []
        for _ in range(8):
            order = rng.permutation(len(rows))
            hits = 0
            for k in order:
                x = rows[k]
                y = 1 if x[1:].sum() >= 2 else -1
                _, action = learner.step(x, [y])
                hits += (action == 1) == (y == 1)
            acc.append(hits / len(rows))
        assert acc[-1] == 1.0 and acc[0] < 1.0
        # the final classifier, from DOOM's own scores (no lesson), is the majority function
        for x in rows:
            s, a = learner.predict(x)
            assert (a == 1) == (x[1:].sum() >= 2)


def test_mismatch_is_raised_for_a_wrong_reference(tmp_path):
    class WrongRule(PerceptronRef):
        def mistake(self, y, s):                                      # y*s < 0 instead of <= 0
            return bool(y) and y * s < 0

    p = Perceptron(2, 1)
    with p.compile(tmp_path / "bad.wad") as learner:
        learner.twin = WrongRule(2, 1, p.clip, p.cap)
        with pytest.raises(df.MismatchError) as err:                  # s = 0, y = -1: DOOM lowers w0, the twin does not
            learner.step([1, 0], [-1])
        names = {n for n, _, _ in err.value.mismatches}
        assert {"w0_0", "-w0_0"} <= names


def test_bigger_maps_stay_exact(tmp_path):
    """78 and 77 dolls: sizes where the old 58,000-unit-wide layout lost updates (docs/engine_limits.md)."""
    for n, heads in ((5, 3), (8, 2)):
        rng = np.random.default_rng(n)
        with Perceptron(n, heads).compile(tmp_path / f"big{n}.wad") as learner:
            for _ in range(25):
                x, y = rand_example(rng, n, heads, labels=(-1, 1))
                learner.step(x, y)


@pytest.mark.parametrize("flag,value", [("FREEZE", False), ("LAND_LEN", 0)])
def test_broken_maps_are_caught(tmp_path, monkeypatch, flag, value):
    """Negative control: maps compiled without freeze barriers / landing pads compute wrong numbers, and the learner
    notices within a few steps."""
    monkeypatch.setattr(pure, flag, value)
    p = Perceptron(4, 2, cap=5, clip=3)
    rng = np.random.default_rng(0)
    with p.compile(tmp_path / f"{flag}.wad") as learner:
        with pytest.raises(df.MismatchError):
            for _ in range(8):
                x, y = rand_example(rng, 4, 2, labels=(-1, 1))
                learner.step(x, y)


def world(m) -> str:
    kw = dict(floor=0, ceil=160, ctex="CEIL5_1", ftex="FLAT23", wall="STARTAN3")
    m.rect(-1024, -3072, -288, -2048, light=150, **kw)
    m.rect(-288, -3072, 512, -2560, light=DIM, tag=LEFT, **kw)
    m.rect(-288, -2560, 512, -2048, light=DIM, tag=RIGHT_TAG, **kw)
    m.thing(-768, -2560, 1, angle=0)
    return f"script 201 (void) {{ o1 = GetSectorLightLevel({LEFT}); o2 = GetSectorLightLevel({RIGHT_TAG}); }}"


def test_world_actions_are_fired_by_dooms_decisions(tmp_path):
    """pick0 / pick1 set the two lamps through Light_ChangeToValue lines that only the dolls cross."""
    acts = {"pick1": [df.act.light(LEFT, LIT), df.act.light(RIGHT_TAG, DIM)],
            "pick0": [df.act.light(RIGHT_TAG, LIT), df.act.light(LEFT, DIM)]}
    p = Perceptron(2, 1)
    with p.compile(tmp_path / "w.wad", world=world, actions=acts) as learner:
        assert learner.run_script(201, read=2) == [DIM, DIM]
        x = [1, 1]
        learner.step(x, [1])                                          # s = 0: action 0 -> pick0 -> lamp B
        assert learner.run_script(201, read=2) == [DIM, LIT]
        learner.step(x, [1])                                          # w = [1, 1]: s = 2 -> action 1 -> lamp A
        assert learner.run_script(201, read=2) == [LIT, DIM]
        learner.step(x, [-1])                                         # still predicts 1 (a mistake); w -> [0, 0]
        assert learner.run_script(201, read=2) == [LIT, DIM]
        np.testing.assert_array_equal(learner.weights, [[0, 0]])
        learner.step(x, [-1])                                         # s = 0: action 0 -> lamp B
        assert learner.run_script(201, read=2) == [DIM, LIT]
