"""Quantisation tooling (doomformer.quantize, doomformer.qat, the torch walker in doomformer.interop): MSE calibration,
equalisation, BatchNorm folding, AvgPool/MaxPool lowering, GELU/SiLU staircases and quantisation-aware training."""
from __future__ import annotations

import hashlib

import numpy as np
import pytest

import doomformer as df
from doomformer import interop, quantize

torch = pytest.importorskip("torch")
nn = torch.nn


def _rand_layers(seed=3, dims=(6, 9, 7, 4)):
    rng = np.random.default_rng(seed)
    layers = [(rng.normal(0, 1.2, (dims[i + 1], dims[i])), rng.normal(0, 0.4, dims[i + 1])) for i in range(len(dims) - 1)]
    return layers, rng.normal(0, 1.5, (120, dims[0]))


def _digest(net) -> str:
    h = hashlib.sha256()
    for a in [*net.weights, *net.biases, *net.thresholds, net.in_scale]:
        h.update(np.ascontiguousarray(a).tobytes())
    return h.hexdigest()[:16]


# --- defaults are unchanged ------------------------------------------------------------------------------------------
def test_default_quantisation_is_bit_identical_to_v0_5_0():
    layers, calib = _rand_layers()
    net = df.MLP.from_numpy(layers, calib, levels=5)
    assert _digest(net) == "917d5a97eae9147c"                      # computed with the v0.5.0 code
    same = df.MLP.from_numpy(layers, calib, levels=5, calibrate="max", equalise=False)
    assert _digest(same) == _digest(net)


def test_unknown_calibration_is_refused():
    layers, calib = _rand_layers()
    with pytest.raises(ValueError, match="calibrate"):
        df.MLP.from_numpy(layers, calib, calibrate="percentile")


# --- MSE step and equalise -------------------------------------------------------------------------------------------
def test_relu_step_mse_beats_the_max_step_on_a_heavy_tail():
    rng = np.random.default_rng(0)
    h = np.concatenate([np.abs(rng.normal(0, 10, 5000)), [60.0]])          # one outlier sets the max step
    K = 4

    def err(D):
        r = np.maximum(h, 0)
        return ((r - np.clip(np.round(r / D), 0, K) * D) ** 2).mean()

    d_max, d_mse = quantize.relu_step(h, K, "max"), quantize.relu_step(h, K, "mse")
    assert d_max == 60.0 / K and 0.25 * d_max <= d_mse < d_max and err(d_mse) < 0.5 * err(d_max)


def test_equalise_keeps_the_float_function_and_equalises_the_ranges():
    layers, calib = _rand_layers(seed=5, dims=(8, 12, 10, 3))
    layers[0] = (layers[0][0] * np.linspace(0.3, 3, 12)[:, None], layers[0][1])     # unequal unit ranges
    fl = interop.normalize_layers(layers)
    eq = quantize.equalise_layers(fl, calib)
    for x in calib[:20]:
        a, b = x, x
        for (W, bb, act), (W2, b2, _) in zip(fl, eq):
            a, b = act(a @ W.T + bb), act(b @ W2.T + b2)
        np.testing.assert_allclose(a, b, atol=1e-9)
    h = calib @ fl[0][0].T + fl[0][1]
    h2 = calib @ eq[0][0].T + eq[0][1]
    q = lambda v: np.array([np.quantile(v[v[:, j] > 0, j], 0.99) for j in range(v.shape[1])])   # noqa: E731
    assert q(h2).std() < 0.5 * q(h).std()


def test_mse_with_equalise_lowers_the_error_on_a_deep_net():
    rng = np.random.default_rng(1)
    dims = (10, 24, 24, 24, 24, 5)
    layers = [(rng.normal(0, 1, (dims[i + 1], dims[i])) * (rng.lognormal(0, 0.7, dims[i + 1])[:, None]),
               rng.normal(0, 0.3, dims[i + 1])) for i in range(len(dims) - 1)]
    calib = rng.normal(0, 1, (400, dims[0]))
    errs = {}
    for cal in ("max", "mse"):
        net = df.MLP.from_numpy(layers, calib, levels=4, calibrate=cal)
        errs[cal] = net.quantization_report(calib)["mean_abs_error"]
    assert errs["mse"] < errs["max"]


# --- BatchNorm folding -----------------------------------------------------------------------------------------------
def _randomise_bn(model, seed=0):
    g = torch.Generator().manual_seed(seed)
    for m in model.modules():
        if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
            m.running_mean.copy_(torch.randn(m.num_features, generator=g, dtype=torch.float64) * 0.5)
            m.running_var.copy_(torch.rand(m.num_features, generator=g, dtype=torch.float64) + 0.5)
            m.weight.data = torch.rand(m.num_features, generator=g, dtype=torch.float64) + 0.5
            m.bias.data = torch.randn(m.num_features, generator=g, dtype=torch.float64) * 0.3
    return model.eval()


@pytest.mark.parametrize("build", [
    lambda: nn.Sequential(nn.Conv2d(1, 3, 3, padding=1), nn.BatchNorm2d(3), nn.ReLU(), nn.Conv2d(3, 2, 3, stride=2, padding=1),
                          nn.BatchNorm2d(2), nn.ReLU(), nn.Flatten(), nn.Linear(2 * 9, 4)),
    lambda: nn.Sequential(nn.Conv2d(1, 3, 3, padding=1), nn.ReLU(), nn.BatchNorm2d(3), nn.Conv2d(3, 2, 3, padding=1),
                          nn.ReLU(), nn.BatchNorm2d(2), nn.Flatten(), nn.Linear(2 * 36, 4)),          # BN after ReLU, padded conv next
    lambda: nn.Sequential(nn.Flatten(), nn.BatchNorm1d(36), nn.Linear(36, 8), nn.BatchNorm1d(8), nn.Tanh(), nn.Linear(8, 4),
                          nn.BatchNorm1d(4)),                                                          # input BN, final BN
    lambda: nn.Sequential(nn.Conv1d(1, 3, 3, padding=1), nn.BatchNorm1d(3), nn.ReLU(), nn.Flatten(), nn.Linear(3 * 6, 3)),
])
def test_batchnorm_folds_exactly_in_float(build):
    torch.manual_seed(0)
    model = _randomise_bn(build().double())
    shape = (1, 6) if isinstance(model[0], nn.Conv1d) else (1, 6, 6)
    X = torch.randn(40, *shape).double()
    net = df.MLP.from_torch(model, X, levels=8)
    np.testing.assert_allclose(net.float_forward(X.numpy()), model(X).detach().numpy(), atol=1e-9)
    assert net.n_layers >= 2 or isinstance(model[0], nn.Flatten)


def test_batchnorm_refusals():
    X = torch.randn(30, 1, 6, 6).double()
    train_bn = nn.Sequential(nn.Conv2d(1, 2, 3), nn.BatchNorm2d(2), nn.ReLU(), nn.Flatten(), nn.Linear(32, 2)).double()
    with pytest.raises(ValueError, match="training mode"):
        df.MLP.from_torch(train_bn, X)
    for bad, msg in [(nn.Sequential(nn.Conv2d(1, 2, 3), nn.ReLU(), nn.LayerNorm([2, 4, 4]), nn.Flatten(), nn.Linear(32, 2)), "LayerNorm"),
                     (nn.Sequential(nn.Conv2d(1, 2, 3), nn.GroupNorm(1, 2), nn.ReLU(), nn.Flatten(), nn.Linear(32, 2)), "GroupNorm"),
                     (nn.Sequential(nn.Conv2d(1, 2, 3), nn.ReLU(), nn.Flatten(), nn.Linear(32, 2), nn.BatchNorm1d(2), nn.ReLU()), "last"),
                     (nn.Sequential(nn.Conv2d(1, 2, 3), nn.ReLU(), nn.BatchNorm2d(2), nn.MaxPool2d(2), nn.Flatten(),
                                    nn.Linear(8, 2)), "BatchNorm")]:
        with pytest.raises(ValueError, match=msg):
            df.MLP.from_torch(bad.double().eval(), X)


# --- AvgPool: an exact linear layer ----------------------------------------------------------------------------------
@pytest.mark.parametrize("pool,shape", [
    (nn.AvgPool2d(2), (2, 6, 6)), (nn.AvgPool2d(3, stride=2, padding=1), (2, 7, 7)),
    (nn.AvgPool2d(3, stride=2, padding=1, count_include_pad=False), (2, 7, 7)), (nn.AvgPool2d((2, 3)), (1, 6, 6)),
    (nn.AvgPool1d(3, stride=2), (2, 9)), (nn.AdaptiveAvgPool2d(1), (3, 5, 5)), (nn.AdaptiveAvgPool2d((2, 3)), (2, 5, 7)),
    (nn.AdaptiveAvgPool1d(4), (2, 10)),
])
def test_avg_pool_matrix_equals_torch(pool, shape):
    torch.manual_seed(1)
    x = torch.randn(3, *shape).double()
    out = pool(x)
    layers = interop.torch_layers(nn.Sequential(pool, nn.Flatten(), nn.Linear(int(np.prod(out.shape[1:])), 2)).double(), shape)
    P, pb, act = layers[0]
    assert act.kind == "identity" and P.shape == (int(np.prod(out.shape[1:])), int(np.prod(shape)))
    np.testing.assert_allclose(x.reshape(3, -1).numpy() @ P.T + pb, out.reshape(3, -1).numpy(), atol=1e-12)


def test_cnn_with_avgpool_and_batchnorm_quantises_and_matches_float():
    torch.manual_seed(2)
    model = _randomise_bn(nn.Sequential(nn.Conv2d(1, 4, 3, padding=1), nn.BatchNorm2d(4), nn.ReLU(), nn.AvgPool2d(2),
                                        nn.Conv2d(4, 4, 3, padding=1), nn.BatchNorm2d(4), nn.ReLU(), nn.AdaptiveAvgPool2d(1),
                                        nn.Flatten(), nn.Linear(4, 3)).double())
    X = torch.rand(80, 1, 8, 8).double()
    net = df.MLP.from_torch(model, X, levels=16, calibrate="mse")
    np.testing.assert_allclose(net.float_forward(X.numpy()), model(X).detach().numpy(), atol=1e-9)
    rep = net.quantization_report(X.numpy())
    assert rep["argmax_agreement"] > 0.8 and rep["in_range"] == len(X)


# --- MaxPool: exact, by comparisons ----------------------------------------------------------------------------------
def _pooled_counts(ref, K):
    """Counts recovered from the replicated layer (unit (i, k) = [h_i >= t_k]) and the pooled counts layer 2 gives."""
    a1 = ref["activations"][1]                        # (N, n*K) indicator units
    a2 = ref["activations"][2]                        # (N, n_out*K)
    return a1.reshape(len(a1), -1, K).sum(-1), a2.reshape(len(a2), -1, K).sum(-1)


@pytest.mark.parametrize("order", ["act_then_pool", "pool_then_act"])
def test_maxpool_is_exact_on_the_integer_counts(order):
    torch.manual_seed(3)
    pool_act = [nn.ReLU(), nn.MaxPool2d(2)] if order == "act_then_pool" else [nn.MaxPool2d(2), nn.ReLU()]
    model = nn.Sequential(nn.Conv2d(1, 3, 3, padding=1), *pool_act, nn.Flatten(), nn.Linear(3 * 9, 4)).double().eval()
    X = torch.rand(60, 1, 6, 6).double()
    K = 3
    net = df.MLP.from_torch(model, X, levels=K)
    assert net.n_layers == 3                                          # conv as [h_i >= t_k] units, pooled units, Linear
    np.testing.assert_allclose(net.float_forward(X.numpy()), model(X).detach().numpy(), atol=1e-12)
    ref = net.reference(net.encode(X.numpy()))
    counts, pooled = _pooled_counts(ref, K)
    # independent numpy: the counts of the staircase on h_i, then a max over every 2x2 window
    h = ref["hidden"][0][:, ::K]                                      # unit (i, 0) carries h_i itself (bias not lowered)
    t = net.thresholds[1][0] + (net.biases[0][0] - net.biases[0][:K])  # t_k recovered from the lowered biases
    np.testing.assert_array_equal(counts, (h[:, :, None] >= t[None, None, :]).sum(-1))
    win = counts.reshape(len(X), 3, 6, 6)
    expect = win.reshape(len(X), 3, 3, 2, 3, 2).max(axis=(3, 5)).reshape(len(X), -1)
    np.testing.assert_array_equal(pooled, expect)
    assert (pooled.max() <= K) and (pooled.max() > 0)


@pytest.mark.parametrize("pool,H", [
    (nn.MaxPool2d(3, stride=2, padding=1), 5), (nn.MaxPool2d((2, 3)), 6), (nn.MaxPool2d(2, stride=1), 5),
    (nn.MaxPool2d(2, dilation=2, stride=2), 6), (nn.Sequential(nn.MaxPool2d(2), nn.MaxPool2d(2)), 8)])
def test_maxpool_windows_match_torch_on_the_integer_counts(pool, H):
    """Overlapping, padded, dilated, rectangular and composed windows: the pooled integer counts equal torch's own
    max pool applied to the integer counts."""
    torch.manual_seed(4)
    C, K = 2, 4
    conv = nn.Conv2d(1, C, 3, padding=1)
    n_out = int(np.prod(nn.Sequential(conv, pool)(torch.zeros(1, 1, H, H)).shape[1:]))
    model = nn.Sequential(conv, nn.ReLU(), pool, nn.Flatten(), nn.Linear(n_out, 3)).double().eval()
    X = torch.rand(40, 1, H, H).double()
    net = df.MLP.from_torch(model, X, levels=K)
    np.testing.assert_allclose(net.float_forward(X.numpy()), model(X).detach().numpy(), atol=1e-12)
    ref = net.reference(net.encode(X.numpy()))
    counts, pooled = _pooled_counts(ref, K)
    expect = pool(torch.tensor(counts.reshape(len(X), C, H, H), dtype=torch.float64)).reshape(len(X), -1).numpy()
    np.testing.assert_array_equal(pooled, expect)
    assert net.quantization_report(X.numpy())["in_range"] == len(X)


def test_maxpool_without_a_staircase_is_refused_precisely():
    X = torch.randn(20, 1, 6, 6).double()
    with pytest.raises(ValueError, match=r"unsupported MaxPool2d.*staircase of a sum"):
        df.MLP.from_torch(nn.Sequential(nn.Conv2d(1, 2, 3), nn.MaxPool2d(2), nn.Flatten(), nn.Linear(8, 2)).double(), X)
    with pytest.raises(ValueError, match="MaxPool2d: ceil_mode"):
        df.MLP.from_torch(nn.Sequential(nn.Conv2d(1, 2, 3), nn.ReLU(), nn.MaxPool2d(2, ceil_mode=True), nn.Flatten(),
                                        nn.Linear(8, 2)).double(), X)


def test_maxpool_net_is_exact_in_doom(tmp_path):
    torch.manual_seed(5)
    model = nn.Sequential(nn.Conv1d(1, 2, 3, padding=1), nn.ReLU(), nn.MaxPool1d(2), nn.Flatten(), nn.Linear(6, 3)).double().eval()
    X = torch.rand(50, 1, 6).double()
    net = df.MLP.from_torch(model, X, levels=[2, 3])
    plan = df.plan(net)
    with df.compile(net, tmp_path / "pool.wad") as brain:
        assert (brain.info["dolls"], brain.info["sectors"]) == (plan["dolls"], plan["sectors"])
        for x in net.encode(X[:2].numpy()):
            np.testing.assert_array_equal(brain(x), net.reference(x)["y"])


# --- GELU / SiLU -----------------------------------------------------------------------------------------------------
def test_gelu_silu_follow_the_monotone_envelope_with_documented_error():
    for name, bound in (("gelu", 0.0855), ("gelu_tanh", 0.0855), ("silu", 0.1395)):
        act = interop.resolve_act(name)
        env = interop.monotone_envelope(act.fn, (-3.0, 4.0))
        z = np.linspace(-3, 4, 4001)
        assert (np.diff(env(z)) >= -1e-12).all()
        assert np.abs(env(z) - act.fn(z)).max() <= bound
        st = quantize.fit_staircase(act, z, 12)
        assert (np.diff(st) >= 0).all() and st.max_error() <= st.step / 2 + 1e-9
        assert np.abs(st.value(z) - act.fn(z)).max() <= bound + st.step / 2 + 1e-9


def test_gelu_silu_through_from_torch():
    torch.manual_seed(6)
    for act in (nn.GELU(), nn.GELU(approximate="tanh"), nn.SiLU()):
        model = nn.Sequential(nn.Linear(5, 12), act, nn.Linear(12, 3)).double().eval()
        X = torch.randn(200, 5).double()
        net = df.MLP.from_torch(model, X, levels=[16, 24])
        rep = net.quantization_report(X.numpy())
        assert rep["mean_abs_error"] < 0.05 * rep["float_max_abs"]
        assert net.float_acts[0].monotone is False


def test_gelu_net_round_trips_through_save_load(tmp_path):
    layers, calib = _rand_layers()
    layers[0] = (*layers[0], "gelu")
    net = df.MLP.from_numpy(layers, calib, levels=6)
    net.save(tmp_path / "g.npz")
    back = df.MLP.load(tmp_path / "g.npz")
    np.testing.assert_allclose(back.float_forward(calib), net.float_forward(calib), atol=0)


# --- quantisation-aware training -------------------------------------------------------------------------------------
def _blobs(n=400, seed=0):
    g = torch.Generator().manual_seed(seed)
    y = torch.randint(0, 3, (n,), generator=g)
    centres = torch.tensor([[2.0, 0, 0, 1], [-1.5, 2, 0, 0], [0, -2, 2, -1]])
    return centres[y] + 0.7 * torch.randn(n, 4, generator=g), y


def test_qat_fit_returns_a_model_from_torch_lowers_with_a_small_gap():
    from doomformer import qat

    torch.manual_seed(0)
    x, y = _blobs()
    model = nn.Sequential(nn.Linear(4, 16), nn.ReLU(), nn.Linear(16, 16), nn.ReLU(), nn.Linear(16, 3))
    opt = torch.optim.Adam(model.parameters(), 1e-2)
    for _ in range(150):
        opt.zero_grad()
        nn.functional.cross_entropy(model(x), y).backward()
        opt.step()
    loader = [(x[i:i + 50], y[i:i + 50]) for i in range(0, 400, 50)]
    q = qat.fit(model, loader, levels=4, steps=120)
    assert isinstance(q[0], qat.InputStair) and sum(isinstance(m, qat.Stair) for m in q.modules()) == 2
    assert not q.training
    net = df.MLP.from_torch(q, x.numpy())
    assert net.n_layers == 3 and [len(t) for t in net.thresholds] == [4, 4, 4]
    gap = qat.lowering_gap(q, net, x, y.numpy(), float_model=model)
    assert gap["argmax_agreement"] > 0.9 and gap["accuracy_integer"] > 0.85 and gap["accuracy_float"] > 0.9
    # the staircases carried over: the integer thresholds are the trained ones in units of the layer's float scale
    st = [m for m in q.modules() if isinstance(m, qat.Stair)][0]
    ratio = net.thresholds[1] / (st.thr.numpy().astype(np.float64))
    np.testing.assert_allclose(ratio, ratio[0], rtol=0.02)


def test_qat_levels_validation_and_cnn_with_batchnorm_and_pool():
    from doomformer import qat

    torch.manual_seed(1)
    model = nn.Sequential(nn.Conv2d(1, 3, 3, padding=1), nn.BatchNorm2d(3), nn.ReLU(), nn.MaxPool2d(2), nn.Flatten(),
                          nn.Linear(3 * 9, 3))
    x = torch.rand(64, 1, 6, 6)
    y = torch.randint(0, 3, (64,))
    with pytest.raises(ValueError, match="levels"):
        qat.fit(model, [(x, y)], levels=[4, 4, 4], steps=2)
    bn = model[1]
    q = qat.fit(model, [(x, y)], levels=[4, 3], steps=6)
    assert torch.equal(q[1][1].running_mean, bn.running_mean)           # BatchNorm trained with frozen statistics
    net = df.MLP.from_torch(q.double(), x.double().numpy())
    assert net.input_shape == (1, 6, 6) and len(net.thresholds[0]) == 4


def test_qat_equalise_keeps_the_float_function_across_batchnorm_pool_and_flatten():
    import copy

    from doomformer import qat

    torch.manual_seed(7)
    model = nn.Sequential(nn.Conv2d(1, 4, 3, padding=1), nn.BatchNorm2d(4), nn.ReLU(), nn.MaxPool2d(2),
                          nn.Conv2d(4, 5, 3, padding=1), nn.ReLU(), nn.AvgPool2d(2), nn.Flatten(), nn.Linear(5 * 4, 6),
                          nn.ReLU(), nn.Linear(6, 3)).double()
    bn = model[1]
    bn.running_mean.normal_()
    bn.running_var.uniform_(0.5, 2)
    bn.weight.data.uniform_(0.5, 2)
    body = qat._convert(copy.deepcopy(model.eval()), [4] * 4, "identity", [])      # staircases not calibrated: float ReLU
    x = torch.rand(200, 1, 8, 8).double()
    before = body(x).detach().clone()
    assert qat._equalise(nn.Sequential(nn.Identity(), body), x) == 3
    torch.testing.assert_close(body(x), before, atol=1e-12, rtol=0)


def test_qat_fit_refuses_one_shot_and_empty_loaders():
    from doomformer import qat

    model = nn.Sequential(nn.Flatten(), nn.Linear(8, 4), nn.ReLU(), nn.Linear(4, 2))
    batch = (torch.rand(16, 8), torch.randint(0, 2, (16,)))
    with pytest.raises(ValueError, match="one-shot"):
        qat.fit(model, (b for b in [batch] * 3), levels=4, steps=5)
    with pytest.raises(ValueError, match="no batches"):
        qat.fit(model, [], levels=4, steps=5)
    with pytest.raises(ValueError, match="no batches"):
        qat.fit(model, [], levels=4, steps=5, calib_x=batch[0])


def test_non_finite_calibration_rows_are_refused_with_their_index():
    layers, calib = _rand_layers()
    for bad in (np.nan, np.inf):
        x = calib.copy()
        x[7, 2] = bad
        for cal in ("max", "mse"):
            with pytest.raises(ValueError, match=r"row 7, feature 2"):
                df.MLP.from_numpy(layers, x, calibrate=cal)
    model = nn.Sequential(nn.Flatten(), nn.Linear(64, 8), nn.ReLU(), nn.Linear(8, 3)).eval()
    x = np.random.default_rng(0).random((20, 1, 8, 8))
    x[3, 0, 2, 2] = np.nan
    with pytest.raises(ValueError, match="row 3, feature 18"):
        df.MLP.from_torch(model, x, levels=4, calibrate="mse")


def test_layernorm_error_does_not_point_at_a_missing_converter():
    model = nn.Sequential(nn.Linear(4, 4), nn.LayerNorm(4), nn.ReLU(), nn.Linear(4, 2))
    with pytest.raises(ValueError, match="LayerNorm is not supported") as e:
        df.MLP.from_torch(model.eval(), torch.randn(10, 4))
    assert "Transformer.from_torch" not in str(e.value)
