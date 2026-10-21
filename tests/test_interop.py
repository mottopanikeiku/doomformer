"""Float interop (doomformer.interop + MLP.from_numpy/from_torch/predict/quantization_report): conv layers lowered to
exact linear layers (equal to torch's conv in float), monotone activations as staircases, and quantised nets run in a
real headless DOOM game whose every number must equal the integer reference."""
from __future__ import annotations

import numpy as np
import pytest

import doomformer as df
from doomformer import interop


# --- staircase ------------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("fn,rng_,K", [(np.tanh, (-3, 3), 8), (interop._sigmoid, (-6, 6), 5),
                                       (lambda z: np.clip(z, 0, 6), (-2, 9), 6), (lambda z: np.maximum(z, 0), (-1, 4), 4)])
def test_staircase_follows_fn_within_half_a_step(fn, rng_, K):
    st = interop.staircase(fn, rng_, K)
    assert isinstance(st, interop.Staircase) and len(st) == K and (np.diff(st) >= 0).all()
    assert rng_[0] <= st.min() and st.max() <= rng_[1]
    assert st.max_error() <= st.step / 2 + 1e-9               # the documented approximation bound
    z = np.linspace(*rng_, 300)
    np.testing.assert_array_equal(st.counts(z), (z[:, None] >= np.asarray(st)[None]).sum(1))   # = DOOM's adc_relu
    assert st.value(rng_[0]) == st.lo and np.isclose(st.value(rng_[1]), st.lo + K * st.step)


def test_staircase_rejects_bad_functions():
    with pytest.raises(ValueError, match="nondecreasing"):
        interop.staircase(np.sin, (0, 6), 4)
    with pytest.raises(ValueError, match="constant"):
        interop.staircase(lambda z: np.zeros_like(z), (0, 1), 4)
    with pytest.raises(ValueError):
        interop.staircase(np.tanh, (2, 1), 4)
    with pytest.raises(ValueError):
        interop.staircase(np.tanh, (0, 1), 0)


def test_staircase_of_a_step_function_puts_the_threshold_at_the_jump():
    st = interop.staircase(lambda z: (z >= 1.5) * 1.0 + (z >= 2.5) * 1.0, (0, 4), 2)
    np.testing.assert_allclose(st, [1.5, 2.5], atol=1e-9)
    assert st.max_error() == 0.0


def test_staircase_int_thresholds_shift_to_positive():
    st = interop.staircase(np.tanh, (-3, 3), 4)
    t, shift = st.int_thresholds(0.1)
    assert shift == 1 - int(np.rint(np.asarray(st).min() / 0.1)) and t.min() == 1
    np.testing.assert_array_equal(t - shift, np.rint(np.asarray(st) / 0.1))
    df.MLP([[[1, 1]]], [[0]], [t])                            # a valid threshold set for the integer MLP


# --- conv lowering equals torch (float) -----------------------------------------------------------------------------
@pytest.mark.filterwarnings("ignore:Using padding='same'")
def test_conv2d_lowering_equals_torch_conv():
    torch = pytest.importorskip("torch")
    torch.manual_seed(0)
    cfgs = [dict(), dict(padding=1), dict(stride=2, padding=1), dict(dilation=2, padding=2), dict(padding="same"),
            dict(groups=2, padding=1), dict(padding="valid", stride=(1, 2)), dict(bias=False)]
    for cfg in cfgs:
        conv = torch.nn.Conv2d(4, 6, (3, 2), **cfg).double()
        x = torch.randn(5, 4, 7, 6).double()
        M, b, shape = interop.conv2d_matrix(conv.weight.detach().numpy(),
                                            None if conv.bias is None else conv.bias.detach().numpy(), (4, 7, 6),
                                            conv.stride, conv.padding, conv.dilation, conv.groups)
        want = conv(x).detach()
        assert tuple(want.shape[1:]) == shape, cfg
        np.testing.assert_allclose(x.reshape(5, -1).numpy() @ M.T + b, want.reshape(5, -1).numpy(), atol=1e-12,
                                   err_msg=str(cfg))
        assert (M == 0).mean() > 0.5                          # sparse: most taps absent


def test_conv1d_lowering_equals_torch_conv():
    torch = pytest.importorskip("torch")
    torch.manual_seed(1)
    for cfg in [dict(), dict(padding=2), dict(stride=2), dict(dilation=2, padding="same"), dict(groups=3)]:
        conv = torch.nn.Conv1d(3, 6, 3, **cfg).double()
        x = torch.randn(4, 3, 11).double()
        M, b, shape = interop.conv1d_matrix(conv.weight.detach().numpy(), conv.bias.detach().numpy(), (3, 11),
                                            conv.stride, conv.padding, conv.dilation, conv.groups)
        want = conv(x).detach()
        assert tuple(want.shape[1:]) == shape, cfg
        np.testing.assert_allclose(x.reshape(4, -1).numpy() @ M.T + b, want.reshape(4, -1).numpy(), atol=1e-12)


def test_conv_lowering_validates_shapes():
    with pytest.raises(ValueError):
        interop.conv2d_matrix(np.zeros((2, 3, 3, 3)), None, (1, 5, 5))      # 3 input channels expected
    with pytest.raises(ValueError):
        interop.conv2d_matrix(np.zeros((2, 1, 7, 7)), None, (1, 5, 5))      # kernel does not fit
    with pytest.raises(ValueError):
        interop.conv2d_matrix(np.zeros((2, 1, 3, 3)), None, (1, 5, 5), stride=2, padding="same")


# --- float models with activations, from_numpy ---------------------------------------------------------------------
def float_net(seed=7):
    rng = np.random.default_rng(seed)
    return [(rng.normal(0, 0.8, (6, 3)), rng.normal(0, 0.3, 6)), (rng.normal(0, 0.8, (5, 6)), rng.normal(0, 0.3, 5)),
            (rng.normal(0, 0.8, (3, 5)), rng.normal(0, 0.3, 3))], rng.normal(0, 1, (300, 3))


@pytest.mark.parametrize("act", ["relu6", "sigmoid", "tanh", "hardtanh", ("clamp", -0.5, 2.0), np.tanh])
def test_from_numpy_activations_approximate_the_float_model(act):
    (l0, l1, l2), calib = float_net()
    net = df.MLP.from_numpy([(*l0, act), (*l1, "tanh"), l2], calib, levels=[16, 24, 24])
    f = net.float_forward(calib)
    # the stored float model is the one passed in, with exact float activations (no staircases)
    name = act if isinstance(act, str) else act[0] if isinstance(act, tuple) else "tanh"
    fa = {"relu6": lambda z: np.clip(z, 0, 6), "sigmoid": interop._sigmoid, "tanh": np.tanh,
          "hardtanh": lambda z: np.clip(z, -1, 1), "clamp": lambda z: np.clip(z, -0.5, 2.0)}[name]
    h = np.tanh(fa(calib @ l0[0].T + l0[1]) @ l1[0].T + l1[1])
    np.testing.assert_allclose(f, h @ l2[0].T + l2[1], atol=1e-12)
    rep = net.quantization_report(calib)
    assert rep["warnings"] == [] and rep["in_range"] == len(calib)
    assert rep["max_abs_error"] < 0.2 * rep["float_max_abs"] and rep["argmax_agreement"] >= 0.88
    assert [len(t) for t in net.thresholds] == [16, 24, 24]
    coarse = df.MLP.from_numpy([(*l0, act), (*l1, "tanh"), l2], calib, levels=[8, 12, 12]).quantization_report(calib)
    assert coarse["mean_abs_error"] > 1.3 * rep["mean_abs_error"]        # more levels, better approximation
    assert all((t >= 1).all() and (np.diff(t) >= 0).all() for t in net.thresholds)


def test_from_numpy_staircase_hidden_values_stay_in_range_with_shift():
    (l0, l1, l2), calib = float_net()
    net = df.MLP.from_numpy([(*l0, "sigmoid"), (*l1, "tanh"), l2], calib, levels=6)
    ref = net.reference(net.encode(calib))
    for h in ref["hidden"]:
        assert np.abs(h).max() <= 12000
    # sigmoid / tanh thresholds start below zero, so the staircase is moved up: the integer hidden value is h + shift
    assert (net.biases[0] != np.rint(l0[1] / net.in_scale.mean())).any()


def test_given_staircase_is_used_as_is():
    (l0, l1, l2), calib = float_net()
    st = interop.staircase(np.tanh, (-2, 2), 7)
    net = df.MLP.from_numpy([(*l0, st), l1[:2] + ("relu",), l2], calib, levels=4)
    assert len(net.thresholds[1]) == 7 and len(net.thresholds[2]) == 4
    # integer thresholds are the float thresholds / s, shifted: their spacing follows the staircase
    t = net.thresholds[1]
    assert np.argmax(np.diff(t)) == np.argmax(np.diff(np.asarray(st))) or np.ptp(np.diff(t)) <= 2


def test_identity_activation_merges_exactly_and_last_activation_is_refused():
    (l0, l1, l2), calib = float_net()
    merged = interop.normalize_layers([(*l0, "identity"), (*l1, "identity"), l2])
    assert len(merged) == 1
    x = calib[:20]
    want = (x @ l0[0].T + l0[1]) @ l1[0].T + l1[1]
    np.testing.assert_allclose((x @ merged[0][0].T + merged[0][1]), want @ l2[0].T + l2[1], atol=1e-12)
    with pytest.raises(ValueError, match="last layer"):
        df.MLP.from_numpy([l0, (*l1, "sigmoid")], calib)
    with pytest.raises(ValueError, match="unknown activation"):
        df.MLP.from_numpy([(*l0, "swishy"), l1], calib)
    with pytest.raises(ValueError, match="not nondecreasing"):
        df.MLP.from_numpy([(*l0, np.sin), l1], calib)
    with pytest.raises(ValueError, match="levels"):
        df.MLP.from_numpy([l0, l1], calib, levels=[4, 4, 4])


def test_default_relu_path_is_unchanged_by_activation_support():
    (l0, l1, l2), calib = float_net()
    a = df.MLP.from_numpy([l0, l1, l2], calib, levels=8)
    b = df.MLP.from_numpy([(*l0, "relu"), (*l1, None), (*l2, None)], calib, levels=[8, 8, 8])
    for u, v in zip(a.weights + a.biases + a.thresholds, b.weights + b.biases + b.thresholds):
        np.testing.assert_array_equal(u, v)


# --- from_torch ------------------------------------------------------------------------------------------------------
def torch_conv_model(act, seed=3):
    torch = pytest.importorskip("torch")
    nn = torch.nn
    torch.manual_seed(seed)
    return nn.Sequential(
        nn.Sequential(nn.Conv2d(1, 3, 3, padding=1), act, nn.Dropout(0.3)),
        nn.Sequential(nn.Conv2d(3, 2, 3, stride=2), nn.ReLU()),
        nn.Flatten(), nn.Linear(2 * 2 * 2, 6), nn.Identity(), nn.Tanh(), nn.Linear(6, 4)).double().eval()


@pytest.mark.parametrize("name", ["ReLU", "ReLU6", "Hardtanh", "Sigmoid", "Tanh"])
def test_from_torch_conv_models_float_model_equals_torch(name):
    torch = pytest.importorskip("torch")
    model = torch_conv_model(getattr(torch.nn, name)())
    X = torch.rand(120, 1, 5, 5).double()
    net = df.MLP.from_torch(model, X, levels=6)
    assert net.n_in == 25 and net.input_shape == (1, 5, 5) and net.n_layers == 4 and net.n_out == 4
    np.testing.assert_allclose(net.float_forward(X.numpy()), model(X).detach().numpy(), atol=1e-12)
    np.testing.assert_allclose(net.float_forward(X[0].numpy()), model(X[:1]).detach().numpy()[0], atol=1e-12)
    rep = net.quantization_report(X.numpy())
    assert rep["warnings"] == [] and rep["argmax_agreement"] >= 0.6
    np.testing.assert_array_equal(net.encode(X[:3].numpy()), net.encode(X[:3].numpy().reshape(3, -1)))
    flat = df.MLP.from_torch(model, X.reshape(120, 25), levels=6, input_shape=(1, 5, 5))
    np.testing.assert_array_equal(flat.weights[0], net.weights[0])


def test_from_torch_conv1d_and_single_linear():
    torch = pytest.importorskip("torch")
    nn = torch.nn
    torch.manual_seed(2)
    model = nn.Sequential(nn.Conv1d(2, 3, 3, padding=1), nn.Hardtanh(-2, 2), nn.Flatten(), nn.Linear(24, 3)).double().eval()
    X = torch.randn(80, 2, 8).double()
    net = df.MLP.from_torch(model, X, levels=6)
    np.testing.assert_allclose(net.float_forward(X.numpy()), model(X).detach().numpy(), atol=1e-12)
    lin = nn.Linear(3, 2).double()
    one = df.MLP.from_torch(lin, torch.randn(10, 3).double())
    assert one.n_layers == 1


def test_from_torch_refuses_what_it_cannot_do():
    torch = pytest.importorskip("torch")
    nn = torch.nn
    X = torch.randn(30, 1, 6, 6).double()
    for bad, msg in [(nn.Sequential(nn.Conv2d(1, 2, 3, padding_mode="reflect", padding=1), nn.ReLU(), nn.Flatten(),
                                    nn.Linear(2 * 36, 2)), "padding_mode"),
                     (nn.Sequential(nn.Conv2d(1, 2, 3), nn.ReLU(), nn.Linear(2 * 16, 2)), "Flatten"),
                     (nn.Sequential(nn.Conv2d(1, 2, 3), nn.MaxPool2d(2), nn.Flatten(), nn.Linear(8, 2)), "unsupported"),
                     (nn.Sequential(nn.Conv2d(1, 2, 3), nn.ReLU(), nn.ReLU(), nn.Flatten(), nn.Linear(32, 2)), "two activations"),
                     (nn.Sequential(nn.Conv2d(1, 2, 3), nn.ReLU(), nn.Flatten(), nn.Linear(32, 2), nn.Sigmoid()), "last layer"),
                     (nn.Sequential(nn.Conv2d(1, 2, 3), nn.Dropout(0.5), nn.Flatten(), nn.Linear(32, 2)), "training mode"),
                     (nn.Sequential(nn.Flatten(), nn.Linear(36, 2), nn.Flatten(0, 1)), "Flatten")]:
        with pytest.raises(ValueError, match=msg):
            df.MLP.from_torch(bad.double(), X)
    with pytest.raises(ValueError, match="C,...|input_shape"):
        df.MLP.from_torch(nn.Sequential(nn.Conv2d(1, 2, 3), nn.ReLU(), nn.Flatten(), nn.Linear(32, 2)).double(),
                          X.reshape(30, 36))


# --- persistence, report, predict -----------------------------------------------------------------------------------
def test_save_load_keeps_input_shape_and_builtin_activations(tmp_path):
    torch = pytest.importorskip("torch")
    model = torch_conv_model(torch.nn.Sigmoid())
    X = torch.rand(60, 1, 5, 5).double()
    net = df.MLP.from_torch(model, X, levels=4)
    net.save(tmp_path / "c.npz")
    back = df.MLP.load(tmp_path / "c.npz")
    assert back.input_shape == (1, 5, 5)
    np.testing.assert_allclose(back.float_forward(X.numpy()), net.float_forward(X.numpy()), atol=0)
    assert back.quantization_report(X.numpy()) == net.quantization_report(X.numpy())
    custom = df.MLP.from_numpy([(*float_net()[0][0], np.tanh), float_net()[0][1]], float_net()[1])
    custom.save(tmp_path / "k.npz")
    assert df.MLP.load(tmp_path / "k.npz").float_layers is None         # a callable cannot be persisted
    np.testing.assert_array_equal(df.MLP.load(tmp_path / "k.npz").weights[0], custom.weights[0])


def test_report_warns_when_a_layer_saturates_the_gate_range():
    # one hidden unit with weight 200 per staircase level: 60 levels of x -> 12,000 exactly, 61 -> over
    thr = np.arange(1, 62)
    net = df.MLP([[[200]], [[1]]], [[0], [0]], [thr, [1]], signed_input=False)
    rr = net.range_report(np.array([[10], [60]]))
    assert rr["layers"][0]["gate_peak"] == 12000 and rr["layers"][0]["rows_over"] == 0
    assert any(w.startswith("h1") and "uses 100%" in w for w in rr["warnings"])
    rr = net.range_report(np.array([[10], [61]]))
    assert rr["layers"][0]["gate_peak"] == 12200 and rr["layers"][0]["rows_over"] == 1
    assert any("h1" in w and "saturates" in w for w in rr["warnings"])
    with pytest.raises(df.RangeError):
        net.check(np.array([61]))
    quiet = net.range_report(np.array([[3]]))
    assert quiet["warnings"] == [] and quiet["layers"][0]["gate_usage"] == pytest.approx(600 / 12000)
    assert rr["layers"][0]["worst_case"] == 12200
    # a quantised net's layers can never leave their ranges (from_numpy bounds the worst case), so near-full
    # usage is not reported; inputs 200x beyond the calibration data leave the INPUT range, and that is.
    (l0, l1, l2), calib = float_net()
    q = df.MLP.from_numpy([l0, l1, l2], calib, levels=8)
    assert all(d["worst_case"] <= 12000 for d in q.range_report(q.encode(calib))["layers"][:2])
    rep = q.quantization_report(calib * 200)
    assert rep["in_range"] < len(calib) and any(w.startswith("input") and "saturates" in w for w in rep["warnings"])
    assert q.quantization_report(calib)["warnings"] == []


def test_report_has_per_layer_usage_for_quantised_nets():
    (l0, l1, l2), calib = float_net()
    rep = df.MLP.from_numpy([l0, l1, l2], calib).quantization_report(calib)
    assert [d["layer"] for d in rep["layers"]] == ["h1", "h2", "y"]
    assert all(0 < d["partial_sum_usage"] < 1 for d in rep["layers"])
    assert all(0 <= d["top_level_fraction"] <= 1 and d["gate_usage"] < 1 for d in rep["layers"][:2])
    assert rep["input_range"]["usage"] < 1 and "gate_peak" not in rep["layers"][-1]


def test_predict_requires_its_own_brain():
    net = df.MLP([[[1, -1], [-1, 1]]], [[0, 0]], [[1, 2]])
    other = df.MLP([[[1, -1], [-1, 1]]], [[0, 0]], [[1, 2]])

    class FakeBrain:
        def __init__(self, n):
            self.net = n

        def __call__(self, x):
            return net.reference(x)["y"]

    assert net.predict(FakeBrain(net), np.array([5, 0])) == 0
    assert net.predict(FakeBrain(net), np.array([[5, 0], [0, 5]])).tolist() == [0, 1]
    with pytest.raises(ValueError, match="this MLP"):
        net.predict(FakeBrain(other), np.array([5, 0]))


# --- DOOM: the integer model of a lowered conv net is exact ----------------------------------------------------------
def test_quantised_conv_net_is_exact_in_doom_and_predict_is_the_argmax(tmp_path):
    torch = pytest.importorskip("torch")
    nn = torch.nn
    torch.manual_seed(4)
    model = nn.Sequential(nn.Conv1d(1, 2, 3, padding=1), nn.Hardtanh(-1, 1), nn.Flatten(), nn.Linear(12, 5), nn.Tanh(),
                          nn.Linear(5, 3)).double().eval()
    X = torch.rand(100, 1, 6).double()
    net = df.MLP.from_torch(model, X, levels=[2, 3, 3])
    assert net.input_shape == (1, 6) and net.n_layers == 3 and any((b != 0).any() for b in net.biases)
    plan = df.plan(net)
    xs = net.encode(X[:4].numpy())
    with df.compile(net, tmp_path / "conv.wad") as brain:
        assert (brain.info["dolls"], brain.info["sectors"]) == (plan["dolls"], plan["sectors"])
        for x in xs:
            ref = net.reference(x)
            np.testing.assert_array_equal(brain(x), ref["y"])           # raises MismatchError on any difference
            assert brain.info["checked"] == net.n_out + 2 * sum(len(h) for h in ref["hidden"])
            assert net.predict(brain, x) == int(np.argmax(ref["y"]))
        assert net.predict(brain, xs).tolist() == np.argmax(net.reference(xs)["y"], axis=1).tolist()
        assert net.predict(brain, X[4].numpy(), encode=True) == int(np.argmax(net.reference(net.encode(X[4].numpy()))["y"]))


def test_numpy_conv2d_with_staircased_sigmoid_is_exact_in_doom(tmp_path):
    rng = np.random.default_rng(6)
    K = rng.normal(0, 1, (2, 1, 2, 2))
    M, b, shape = interop.conv2d_matrix(K, rng.normal(0, 0.3, 2), (1, 3, 3), padding=1)
    layers = [(M, b, "sigmoid"), (rng.normal(0, 0.5, (3, M.shape[0])), rng.normal(0, 0.2, 3))]
    calib = rng.uniform(-1, 1, (60, 1, 3, 3))
    net = df.MLP.from_numpy(layers, calib, levels=[2, 3])
    assert shape == (2, 4, 4) and net.n_in == 9 and net.input_shape == (1, 3, 3)
    with df.compile(net, tmp_path / "c2.wad") as brain:
        for x in calib[:3]:
            ref = net.reference(net.encode(x))
            np.testing.assert_array_equal(brain(net.encode(x)), ref["y"])
