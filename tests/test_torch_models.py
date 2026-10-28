"""Transformer.from_torch: a torch GPT (doomformer.torch_models) lowered to a QModel whose integer forward equals the
torch integer forward exactly; refusals; determinism; the three-stage recipe runs."""
from __future__ import annotations

import numpy as np
import pytest

import doomformer as df
from doomformer import intref

torch = pytest.importorskip("torch")
tm = pytest.importorskip("doomformer.torch_models")

VOCAB = "$abcdefg"
T = 6


def _data(n=40, seed=0):
    g = torch.Generator().manual_seed(seed)
    rows = [[0] + torch.randint(1, len(VOCAB), (int(torch.randint(2, T, (1,), generator=g)),), generator=g).tolist()
            for _ in range(n)]
    rows = [r[:T] for r in rows]
    return tm.pack(rows, [r[1:] + [0] for r in rows], T)


def _gpt(**kw):
    return tm.GPT(len(VOCAB), T, 16, 2, 24, 3, 3, 3, kw.pop("L", 2), 64.0, kw.pop("seed", 1), **kw)


@pytest.mark.parametrize("kw", [dict(), dict(attn="softmax", R=2, norm="rms", Rn=3, mlp="reglu"),
                                dict(attn="relu", norm="rms", Rn=2, mlp="relu")])
def test_lowered_qmodel_forward_equals_the_torch_integer_forward(kw):
    data = _data()
    net = _gpt(**kw)
    tm.calibrate(net, data, seed=0)
    net.quant = net.hard = True
    qm = df.Transformer.from_torch(net, VOCAB)
    assert isinstance(qm, intref.QModel) and qm.attn == net.attn and qm.norm == net.norm and qm.mlp == net.mlp
    net.eval()
    rows = data[0].tolist()
    with torch.no_grad():
        for r, v in zip(rows[:12], data[2][:12]):
            n = int(v.sum())
            logits, _, _ = net(torch.tensor([r[:n]]))
            assert np.array_equal(logits[0].numpy().astype(np.int64), intref.forward(qm, r[:n]))


def test_from_torch_is_deterministic_and_exports_integers():
    data = _data()
    outs = []
    for _ in range(2):
        net = _gpt()
        tm.calibrate(net, data, seed=0)
        outs.append(df.Transformer.from_torch(net, VOCAB))
    a, b = outs
    assert all(np.array_equal(a.p[k], b.p[k]) and a.p[k].dtype == np.int64 for k in a.p)
    assert all(np.abs(v).max() <= intref.W_MAX for k, v in a.p.items() if ".thr" not in k and not k.startswith("thr"))


def test_from_torch_refusals_name_the_problem():
    net = _gpt()
    with pytest.raises(TypeError, match="lowers a doomformer.torch_models.GPT, not a FloatGPT"):
        df.Transformer.from_torch(tm.FloatGPT(len(VOCAB), T, 16, 2, 24, 1), VOCAB)
    with pytest.raises(TypeError, match="not a Linear"):
        df.Transformer.from_torch(torch.nn.Linear(2, 2), VOCAB)
    with pytest.raises(ValueError, match="never been calibrated"):
        df.Transformer.from_torch(net, VOCAB)
    with pytest.raises(ValueError, match="vocab has 3 symbols"):
        df.Transformer.from_torch(net, "abc", calib_tokens=_data())
    qm = df.Transformer.from_torch(net, VOCAB, calib_tokens=_data())            # calibrates, then lowers
    assert qm.T == T and qm.L == 2


def test_fit_runs_the_three_stages_and_leaves_an_integer_model():
    data = _data(60)
    net = _gpt(L=1)
    res = tm.fit(net, data, data, steps=(4, 4, 4), lrs=(0.01, 0.005, 0.002), bs=16, seed=0)
    assert net.quant and net.hard and set(res) == {"float_dev_acc", "dev_int_acc", "dev_int_ce", "worst"}
    qm = df.Transformer.from_torch(net, VOCAB)
    with torch.no_grad():
        r = data[0][0].tolist()[:int(data[2][0].sum())]
        logits, _, _ = net(torch.tensor([r]))
    assert np.array_equal(logits[0].numpy().astype(np.int64), intref.forward(qm, r))


def test_lowered_tiny_gpt_is_exact_in_doom_tiles():
    net = tm.GPT(len(VOCAB), T, 8, 1, 8, 2, 2, 2, 1, 64.0, 1)
    data = _data(30)
    tm.calibrate(net, data, seed=0)
    net.quant = net.hard = True
    qm = df.Transformer.from_torch(net, VOCAB)
    with df.compile(qm, mode="tiles", workers=1) as brain:
        logits = brain.logits([0, 3, 5])                            # raises MismatchError on any difference
        assert np.array_equal(logits, intref.forward(qm, [0, 3, 5])) and brain.info["checked"] > 0
