"""The induction task and its shipped model: generator, held-out guarantees, baselines, model vs reference (no DOOM)."""
import numpy as np
import pytest

from doomformer import induction as I
from doomformer import intref


def test_rule_matches_targets_and_prefix_is_distinct():
    s = I.test_sets()["test"]
    for row, tgt, n in zip(s["tok"][:500], s["tgt"][:500], s["length"][:500]):
        rule = I.rule(row.tolist())
        assert all(rule[t] == tgt[t] for t in range(len(row)) if tgt[t] != I.IGNORE)
        queried = [t for t in range(len(row)) if tgt[t] != I.IGNORE]
        assert queried == [t for t in range(n) if rule[t] is not None]
        assert I.answer(row[:n].tolist()) == rule[n - 1]


def test_training_stream_excludes_every_held_out_sequence():
    held = I.held_pairs()
    tok, tgt, length = I.train_batch(np.random.default_rng(3), 3000)
    assert not np.isin(I.row_hash(tok), I.excluded()).any()
    pos = np.arange(tok.shape[1] - 1)[None]
    adjacent = (pos >= 1) & (pos <= length[:, None] - 2)
    assert not (held[tok[:, :-1], tok[:, 1:]] & adjacent).any()          # no held pair next to each other in training
    sets = I.test_sets()
    hashes = [set(I.row_hash(s["tok"]).tolist()) for s in sets.values()]
    assert all(len(a & b) == 0 for i, a in enumerate(hashes) for b in hashes[i + 1:])
    u = sets["unseen"]
    q = np.where(u["tgt"] != I.IGNORE, u["tgt"], 0)
    assert (held[u["tok"], q] | (u["tgt"] == I.IGNORE)).all()           # every queried pair of "unseen" is held out


def test_count_tables_and_heuristics_fail_the_rule_does_not():
    train = I.train_batch(np.random.default_rng(5), 60_000)[:2]
    res = I.baselines(train, I.test_sets()["test"])
    for name in ("unigram", "bigram table", "trigram table"):
        assert res[name][0] < 0.12
    assert max(v[0] for k, v in res.items() if "rule" not in k) < 0.35
    assert res["the rule (look back, exact)"] == (1.0, 1.0)
    unseen = I.baselines(train, I.test_sets()["unseen"])
    assert unseen["bigram table"][0] == 0.0                              # the table has never seen a queried pair


def test_shipped_model_induces_on_held_out_sequences():
    m = intref.QModel.load("induction")
    assert intref.check_limits(m, [r[:n].tolist() for r, n in zip(I.test_sets(m.T)["test"]["tok"][:60],
                                                                 I.test_sets(m.T)["test"]["length"][:60])]) == []
    hit = tot = 0
    for name in ("test", "unseen"):
        s = I.test_sets(m.T)[name]
        for row, tgt, n in zip(s["tok"][:150], s["tgt"][:150], s["length"][:150]):
            pred = intref.forward(m, row[:n].tolist()).argmax(-1)
            sup = tgt[:n] != I.IGNORE
            hit += int((pred[sup] == tgt[:n][sup]).sum())
            tot += int(sup.sum())
    assert hit / tot > 0.9


def test_torch_int64_forward_equals_intref():
    pytest.importorskip("torch")
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "training"))
    import train_algo

    m = intref.QModel.load("induction")
    s = I.test_sets(m.T)["test"]
    lg = train_algo.int_logits(m, s["tok"][:120]).numpy()
    for i, n in enumerate(s["length"][:120]):
        assert np.array_equal(lg[i, :n], intref.forward(m, s["tok"][i, :n].tolist()))
