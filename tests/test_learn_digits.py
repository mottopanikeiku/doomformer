"""The multiclass in-map perceptron (doomformer/learn_digits.py): tiny maps in real headless DOOM games.

Four layers of dolls (scores, pair comparisons, winner, update) run in one map; after every step all outputs and ALL
weights are compared with the NumPy twin. Negative controls: one update line that moves its ceiling by 2 must diverge
at the step where the twin first executes it; a host write to a weight must be rejected by the audit. At most one game
at a time; the whole file is about 15 s.
"""
from __future__ import annotations

import numpy as np
import pytest

import doomformer as df
from doomformer import learn_digits as ld

C, N = 4, 6


def stream_of(steps: int, seed: int = 0, n: int = N, C: int = C):
    rng = np.random.default_rng(seed)
    protos = rng.integers(0, 2, (C, n))
    for _ in range(steps):
        t = int(rng.integers(C))
        x = (protos[t] ^ (rng.random(n) < 0.2)).astype(np.int64)
        x[0] = 1                                              # a bias feature
        yield x, t


def textbook(x_t, C, n, clip):
    """The multiclass perceptron written from the definition (no pair counting): returns the weights after each step."""
    w = np.zeros((C, n), dtype=np.int64)
    out = []
    for x, t in x_t:
        s = (np.clip(w, -clip, clip) + clip) @ x
        p = int(np.argmax(s))
        if p != t:
            w[t] += x
            w[p] -= x
        out.append(w.copy())
    return out


def test_twin_is_the_textbook_perceptron_and_the_comparator_is_argmax():
    pairs = list(stream_of(200, seed=3, n=9, C=5))
    ref = ld.DigitRef(9, 5)
    for (x, t), w in zip(pairs, textbook(pairs, 5, 9, ld.CLIP)):
        got = ref.step(x, t)
        assert np.array_equal(ref.w, w)
        z = [got[f"z{c}"] for c in range(5)]
        assert sum(z) == 1 and z.index(1) == int(np.argmax([got[f"s{c}"] for c in range(5)]))
    rng = np.random.default_rng(0)                            # ties go to the lowest index, like np.argmax
    for _ in range(300):
        s = rng.integers(0, 4, 7)
        _, wins, z = ref.__class__(1, 7).comparator(s)
        assert z.sum() == 1 and int(z.argmax()) == int(np.argmax(s))


def test_validation_and_plan():
    for bad in (dict(n_features=0), dict(n_features=3, classes=1), dict(n_features=3, clip=0)):
        with pytest.raises(ValueError):
            ld.DigitPerceptron(**bad)
    with pytest.raises(TypeError):
        ld.DigitPerceptron(2.5)
    p = ld.DigitPerceptron(N, C)
    plan = p.plan()
    assert plan["fits"] and plan["dolls"] == 2 * C * N + C * (C - 1) // 2 + 2 * C + 4     # + one clock doll per layer
    assert plan["values"] == C + C * (C - 1) // 2 + 3 * C + C * N     # s, -d, -wins, z, q, weights
    assert not p.reference().w.any()


@pytest.mark.parametrize("opt", [3, 0])
def test_tiny_run_is_exact_in_doom_and_python_only_writes_inputs(tmp_path, opt):
    """opt 3 (the default) has an ACS copy bus between the layers, opt 0 has none: both are exact on this map."""
    p = ld.DigitPerceptron(N, C)
    steps = list(stream_of(10 if opt == 3 else 4))
    with p.compile(tmp_path / "d.wad", opt=opt) as learner:
        assert not learner.weights.any()                      # a blank brain: every ceiling read from the game is 0
        for (x, t), w in zip(steps, textbook(steps, C, N, ld.CLIP)):
            r = learner.step(x, t)
            assert np.array_equal(r.weights, w) and np.array_equal(learner.twin.w, w)
            assert r.checked == C + C * (C - 1) // 2 + 3 * C + C * N
            assert r.predicted == int(np.argmax(r.scores)) and r.correct == (r.predicted == t)
        assert learner.info["steps"] == len(steps) and learner.info["lessons"] >= 1
        a = learner.audit()
        assert a["input_writes"] == len(steps) * (N + C) and a["other"] == 0
        assert a["guards"] == a["readouts"] > 0                # a readout guard before every page read
        assert np.array_equal(learner.weights, w)             # a fresh readout of the ceilings
        with pytest.raises(ValueError):
            learner.step(np.array([2] * N), 0)
        with pytest.raises(ValueError):
            learner.step(np.ones(N, dtype=int), C)


@pytest.mark.parametrize("line", ld.TAMPER_LINES)
def test_one_changed_update_line_diverges_at_the_predicted_step(tmp_path, line):
    steps = list(stream_of(14, seed=1))
    found = [(c, i, ld.first_crossing(steps, N, C, c, i, line)) for c in range(C) for i in range(N)]
    c, i, k = next(f for f in found if f[2] is not None and f[2] >= 2)      # a line crossed late enough to see exactness first
    with ld.DigitPerceptron(N, C).compile(tmp_path / "t.wad", tamper=(c, i, line)) as learner:
        for j, (x, t) in enumerate(steps):
            if j < k:
                learner.step(x, t)                            # exact until the tampered line is first crossed
                continue
            with pytest.raises(df.MismatchError) as e:
                learner.step(x, t)
            assert [name for name, _, _ in e.value.mismatches] == [f"w{c}_{i}"]    # nothing else of the 454-odd differs
            _, got, want = e.value.mismatches[0]
            assert got - want == (-1 if line == "lower" else 1) and learner.info["steps"] - 1 == k
            break
    with pytest.raises(ValueError):
        ld.DigitPerceptron(N, C).compile(tmp_path / "bad.wad", tamper=(C, 0, line))


def test_a_host_write_to_a_weight_is_rejected(tmp_path):
    with ld.DigitPerceptron(N, C).compile(tmp_path / "a.wad") as learner:
        x, t = next(stream_of(1))
        learner.step(x, t)
        tag = learner.meta.layout["tags"]["ceil:w1_2"]
        assert tag not in set(learner.meta.ports.values())
        with pytest.raises(RuntimeError, match="rejected"):
            learner.runner.game.send_game_command(f"puke -2 {tag} 5 0")
        with pytest.raises(RuntimeError, match="rejected"):
            learner.runner.game.new_episode()
        a = learner.audit()
        assert a["other"] == 2
        assert (learner.twin.w == learner.weights).all()      # nothing reached the engine


def test_learned_weights_run_through_the_forward_path_in_doom(tmp_path):
    w = np.random.default_rng(2).integers(-5, 6, (C, N))
    x = np.random.default_rng(3).integers(0, 2, (3, N))
    winners = ld.doom_forward(w, x)
    assert winners == list(((np.clip(w, -ld.CLIP, ld.CLIP) + ld.CLIP) @ x.T).argmax(0))
