"""The induction circuit tools (doomformer.circuit) on the shipped induction model, reference only. The same cells run
exact in DOOM through `python -m demos.circuit --doom-rows N` (heavy: a 10-token context is tens of seconds per run)."""
from __future__ import annotations

import numpy as np
import pytest

from doomformer import circuit as C
from doomformer import induction as I
from doomformer import intervene as iv
from doomformer import shipped
from doomformer.intref import QModel, forward


@pytest.fixture(scope="module")
def model():
    try:
        return QModel.load(shipped.resolve("induction"))
    except Exception as e:                                       # the model ships with the induction package
        pytest.skip(f"no shipped induction model: {e}")


@pytest.mark.parametrize("kind", ["answer", "source"])
def test_pairs_follow_the_task_rule(kind):
    pairs = C.induction_pairs(400, seed=1, kind=kind)
    assert len(pairs) == 400
    for p in pairs:
        assert I.answer(p.clean) == p.answer and I.answer(p.corrupt) == p.distractor and p.answer != p.distractor
        assert sorted(p.clean) == sorted(p.corrupt)                       # a swap keeps the tokens
        assert sum(a != b for a, b in zip(p.clean, p.corrupt)) == 2
        named = [pos for role, ps in p.roles.items() for pos in ps]
        assert set(named) == set(range(len(p.clean)))                     # the roles cover every position


def test_circuit_identities_and_recovery(model):
    pairs = C.induction_pairs(40, seed=2)
    data = C.mean_data(pairs)
    full = C.faithfulness(model, pairs, [(l, h) for l in range(model.L) for h in range(model.H)], range(model.L), data)
    none = C.faithfulness(model, pairs, [], [], data)
    assert full["faithfulness"] == 1.0 and none["faithfulness"] == 0.0
    assert full["ld_circuit"] == full["ld_full"] and none["ld_circuit"] == none["ld_none"]
    res = C.role_recovery(model, pairs, [("L1.x2", None), ("L1.ao", 0), ("L1.ao", 1), ("L0.x2", None)])
    rec = dict(zip(res["cells"], res["recovery"]))
    q = res["roles"].index("query")
    assert abs(rec[("L1.x2", None)][q] - 1.0) < 1e-9                 # the last position's residual is the logits
    assert rec[("L1.ao", 0)][q] > 0.2 and rec[("L1.ao", 1)][q] > 0.2  # both induction heads carry the answer
    assert abs(rec[("L0.x2", None)][q]) < 0.01                      # layer 0 at the query position hardly differs
    assert rec[("L1.ao", 0)][res["roles"].index("bos")] == 0.0      # the last layer's output at BOS feeds nothing


def test_circuit_plan_mean_ablates_everything_else(model):
    pairs = C.induction_pairs(20, seed=3)
    data = C.mean_data(pairs)
    plan = iv.compile_plan(model, None, C.circuit_plan(model, {(1, 0)}, {0}, data))     # keep L1.H0 and the L0 MLP
    acts = []
    for seq in data:
        tr = {}
        forward(model, seq, tr)
        acts.append(tr)

    def mean_at(key, pos, cols):
        have = [a[key][pos, cols] for a in acts if len(a[key]) > pos]
        return (2 * np.sum(have, 0) + len(have)) // (2 * len(have))               # rounded half up

    run = {}
    forward(model, pairs[0].clean, run, interventions=plan)
    dh = model.dh
    for pos in range(len(pairs[0].clean)):
        assert np.array_equal(run["L0.ao"][pos], mean_at("L0.ao", pos, slice(None)))        # both L0 heads
        assert np.array_equal(run["L1.ao"][pos, dh:], mean_at("L1.ao", pos, slice(dh, None)))   # L1.H1
        assert np.array_equal(run["L1.m"][pos], mean_at("L1.m", pos, slice(None)))          # the L1 MLP
    assert plan.by_key.keys() == {"L0.ao", "L1.ao", "L1.m"}                                   # nothing else touched
