"""Generation without Python: the fixed-point iteration equals greedy generation, and one map writes a whole text
with the level repeating its own passes (Python only reads), every pass exact; two negative controls."""
import numpy as np
import pytest

from doomformer import intref, selfgen
from doomformer.intref import QModel
from doomformer.kit import MismatchError


def random_model(seed: int, T: int = 4, V: int = 5, attn: str = "relu") -> QModel:
    """Random small weights (both signs, many ties), relu or softmax attention."""
    rng = np.random.default_rng(seed)
    d, dm, H = 4, 5, 1 if attn == "relu" else 2
    w = lambda shape: rng.integers(-2, 3, shape, dtype=np.int64)
    p = dict(tok_emb=w((V, d)), pos_emb=w((T, d)), thr_x0=np.array([1, 2]), Wun=w((V, d)), bun=w(V))
    for k in "qkv":
        p["L0.W" + k], p["L0.b" + k] = w((d, d)), w(d)
        p["L0.thr_" + k] = np.array([1, 3])
    p.update({"L0.thr_a": np.array([1, 2, 4]), "L0.Wo": w((d, d)), "L0.r1": w(d), "L0.bo": w(d),
              "L0.thr_x1": np.array([1, 2]), "L0.Wup": w((dm, d)), "L0.bup": w(dm), "L0.thr_m": np.array([1, 2]),
              "L0.Wdn": w((d, dm)), "L0.r2": w(d), "L0.bdn": w(d), "L0.thr_x2": np.array([1, 2])})
    if attn == "relu":
        p["L0.thr_o"] = np.tile([1, 3], (T, 1))
    return QModel("$ABCDEFGH"[:V], T, d, H, dm, 2, 3, 2, 1, p, attn=attn, R=1 if attn == "relu" else 2)


def test_fixed_point_iteration_is_greedy_generation():
    rng = np.random.default_rng(0)
    for trial in range(300):
        T, V = int(rng.integers(3, 8)), int(rng.integers(3, 9))
        m = random_model(trial, T, V, ("relu", "softmax")[trial % 2])
        prompt = "".join(m.vocab[i] for i in rng.integers(1, V, int(rng.integers(0, T - 1))))
        passes = selfgen.iterate(m, prompt)
        assert len(passes) == T - 1 - len(prompt)
        for k, p in enumerate(passes):
            assert p.written[:1 + len(prompt)] == p.tokens[:1 + len(prompt)] == selfgen.initial_tokens(m, prompt)[
                :1 + len(prompt)]
            assert p.winners == [int(np.argmax(p.logits[t])) for t in selfgen.positions(m, prompt)]
            if k + 1 < len(passes):
                assert passes[k + 1].tokens == p.written
        assert selfgen.text(m, passes[-1].written, prompt) == intref.generate(m, prompt, temperature=0)


SEED = 32     # T=4, empty prompt: every pass changes the context ([0,0,0,0] -> [0,4,4,1] -> [0,4,2,2] -> [0,4,2,3])


def test_dolls_write_the_text(tmp_path):
    m = random_model(SEED)
    ref = selfgen.iterate(m)
    assert [p.written for p in ref] == [[0, 4, 4, 1], [0, 4, 2, 2], [0, 4, 2, 3]]
    assert not intref.check_limits(m, [p.tokens for p in ref])
    meta = selfgen.build(m, tmp_path / "selfgen.wad")
    assert [s["name"] for s in meta.layout["stage_stats"]][-2:] == ["argmax.compare", "argmax.winner"]
    assert meta.ports == {}                                         # nothing for the host to write
    with selfgen.LoopRun(m, tmp_path / "selfgen.wad") as run:
        wrote = run.run()
        audit = run.audit()
    assert wrote == intref.generate(m, "") == "DBC"
    assert [p["tokens"] for p in run.passes] == [p.written for p in ref]
    assert all(p["checked"] == len(meta.outputs) for p in run.passes)
    assert audit["non_readout_commands"] == 0 and audit["forbidden"] == {}
    assert set(audit["commands"]) <= run.game.allowed


def test_negative_controls(tmp_path):
    m = random_model(SEED)
    # one winner line writes the wrong token: pass 1's read of the token ports catches it
    selfgen.build(m, tmp_path / "wrong.wad", miswrite={(0, 4): 1})
    with selfgen.LoopRun(m, tmp_path / "wrong.wad") as run, pytest.raises(MismatchError, match="tok:1"):
        run.next_pass()
    # a pass read while the dolls still compute is not exact
    selfgen.build(m, tmp_path / "early.wad")
    with selfgen.LoopRun(m, tmp_path / "early.wad") as run:
        while run.status()["phase"] != selfgen.PH_COMPUTE:
            run.game.advance_action(4)
        run.game.advance_action(400)
        status = run.status()
        assert status["phase"] == selfgen.PH_COMPUTE
        with pytest.raises(MismatchError):
            run.check(1, run.runner.read(), status)
