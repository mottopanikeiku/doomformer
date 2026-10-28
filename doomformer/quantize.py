"""The quantiser behind `MLP.from_numpy` / `MLP.from_torch`: float layers -> an integer `MLP`.

Everything here is NumPy and runs before DOOM sees a number; DOOM then computes the integer net exactly. What the
options do, in order:

* ``calibrate="max"`` (default): the staircase step of a ReLU layer is ``max(h) / K`` over the calibration rows and
  the input step of a feature is ``max|x| / K``.
* ``calibrate="mse"``: the step is the one with the least squared error between the unit's float value and its
  staircase on the calibration rows, searched over ``a * max / K`` for ``a`` in 0.25..1 (61 steps): the same
  `step_size` rule as training/train_mnist.py. Other activations (sigmoid, tanh, GELU, ...) search the clipped
  range of the staircase over quantiles of the calibration pre-activations, scored the same way.
* ``equalise``: before quantising, every ReLU hidden unit is rescaled (incoming weights and bias times c, outgoing
  weights divided by c, factors within 1/4..4) towards the same 99th percentile of its positive activations. The
  float function is unchanged (ReLU is positively homogeneous); a layer has ONE staircase step, equal ranges make it
  fit every unit. Skipped for layers with a max pool and for staircases whose thresholds you fixed.
* ``MaxPool`` after (or before) a monotone activation is lowered exactly, by comparisons: the pooled count is
  ``max_i a_i = sum_k [some a_i >= k]``. The hidden layer is emitted once per level k with its bias lowered by the
  distance between the thresholds (so unit (i, k) is ``[h_i >= t_k]``, one doll), then a layer of single-threshold
  units ``[sum_i u_(i,k) >= 1]`` and the next layer reads the sum over k. Exact against the integer activations;
  the cost is K times the lines of the pooled layer.
"""
from __future__ import annotations

import numpy as np

from .interop import Act, MaxPool, Staircase, level_list, monotone_envelope, normalize_layers, staircase
from .intref import PORT_MAX, W_MAX, Z_MAX, adc_relu, adc_signed

INPUT_SUBSTEPS = 16        # integer input units per staircase level
CALIBRATIONS = ("max", "mse")
MSE_GRID = np.linspace(0.25, 1.0, 61)             # step = a * max / K (training/train_mnist.py step_size)
RANGE_QUANTILES = (0.0, 0.0005, 0.001, 0.0025, 0.005, 0.01, 0.02, 0.05)
EQUALISE_Q = 0.99
MSE_SAMPLE = 400_000


def _thin(v: np.ndarray, n: int = MSE_SAMPLE) -> np.ndarray:
    v = np.asarray(v, dtype=np.float64).ravel()
    return v if len(v) <= n else v[::-(-len(v) // n)]


def relu_step(h: np.ndarray, K: int, calibrate: str = "max", floor: float = 1.0) -> float:
    """The staircase step D of a ReLU layer with K levels (thresholds (k - 1/2) D) for the pre-activations h:
    max(floor, max h) / K, or with calibrate="mse" the D = a * max / K (a in 0.25..1, 61 steps) with the least squared
    error between relu(h) and clip(round(relu(h) / D), 0, K) * D. Ties go to the smaller step. floor is 1 for the
    integer pre-activations of the quantiser (a step below one integer unit is meaningless), tiny for floats."""
    top = max(floor, float(np.max(h, initial=0))) / K
    if calibrate != "mse":
        return top
    r = np.maximum(_thin(h), 0)
    errs = [(float(((r - np.clip(np.round(r / (a * top)), 0, K) * (a * top)) ** 2).mean()), a * top) for a in MSE_GRID]
    return min(errs)[1]


def input_steps(cx: np.ndarray, K: int, signed: bool, calibrate: str = "max") -> np.ndarray:
    """The input staircase step per feature: max|x| / K, or with calibrate="mse" the a * max|x| / K (a in 0.25..1)
    with the least squared error of clip(round(x / D), -K or 0, K) * D against x. 0 becomes 1 (a constant-zero
    feature)."""
    top = np.abs(cx).max(0) / K
    top[top == 0] = 1.0
    if calibrate != "mse":
        return top
    x = cx[:: max(1, len(cx) // 20000)]
    lo = -K if signed else 0
    best, best_a = None, None
    for a in MSE_GRID:
        D = a * top
        e = ((x - np.clip(np.round(x / D), lo, K) * D) ** 2).mean(0)
        if best is None:
            best, best_a = e, np.full(len(top), a)
        else:
            better = e < best
            best, best_a = np.where(better, e, best), np.where(better, a, best_a)
    return top * best_a


def fit_staircase(act: Act, z: np.ndarray, K: int, calibrate: str = "max", eps: float = 1e-9) -> Staircase:
    """A K-level staircase for the activation `act` over the pre-activation values z (float): the range is
    (min z, max z) (widened by eps if z is constant), or with calibrate="mse" the quantile range (p, 1 - p) for the p
    in RANGE_QUANTILES with the least squared error between act(z) and the staircase value. Activations that are not
    monotone (GELU, SiLU) follow their monotone envelope over that range."""
    z = np.asarray(z, dtype=np.float64)

    def build(lo: float, hi: float) -> Staircase:
        if not hi > lo:
            hi = lo + eps
        return staircase(act.fn if act.monotone else monotone_envelope(act.fn, (lo, hi)), (lo, hi), K)

    if calibrate != "mse":
        return build(float(z.min()), float(z.max()))
    zs = _thin(z)
    fz = np.asarray(act.fn(zs), dtype=np.float64)
    best, best_st, first = None, None, None
    for p in RANGE_QUANTILES:
        lo, hi = (float(zs.min()), float(zs.max())) if p == 0 else (float(np.quantile(zs, p)),
                                                                     float(np.quantile(zs, 1 - p)))
        if not hi > lo:
            continue
        try:
            st = build(lo, hi)
        except ValueError as e:
            first = first or e
            continue
        err = float(((fz - st.value(zs)) ** 2).mean())
        if best is None or err < best:
            best, best_st = err, st
    if best_st is None:
        raise first or ValueError("the pre-activations are constant")
    return best_st


def plan_staircase(act: Act, h: np.ndarray, s: float, K: int, calibrate: str = "max"
                   ) -> tuple[np.ndarray, int, float, float]:
    """The integer staircase for the hidden pre-activations h (int, calibration rows) whose float unit is s:
    (thresholds >= 1 nondecreasing, shift added to h, offset = float value of count 0, unit = float value per count)."""
    if act.kind == "relu":
        D = relu_step(h, K, calibrate)
        t = np.maximum(1, np.round((np.arange(1, K + 1) - 0.5) * D)).astype(np.int64)
        return np.maximum.accumulate(t), 0, 0.0, D * s
    if act.kind == "staircase":
        st = act.stair
    else:
        try:
            st = fit_staircase(act, h * s, K, calibrate, eps=s)
        except ValueError as e:
            raise ValueError(f"activation {act.name}: {e}") from e
    t, shift = st.int_thresholds(s)
    return np.maximum.accumulate(t), shift, st.lo, st.step


def equalise_layers(fl, cx: np.ndarray, q: float = EQUALISE_Q):
    """Rescale the ReLU hidden units of the float layers `fl` ([(W, b, Act)], from `normalize_layers`) towards the
    same q-quantile of their positive activations on the calibration rows cx (see the module docstring). Returns a
    new list; the function it computes is the same. Units with at most 10 positive activations keep their scale."""
    fl = [(W.copy(), b.copy(), act) for W, b, act in fl]
    a = cx
    for l in range(len(fl) - 1):
        W, b, act = fl[l]
        h = a @ W.T + b
        if act.kind == "relu" and act.pool is None:
            pos = np.where(h > 0, h, np.nan)
            enough = (h > 0).sum(0) > 10
            m = np.zeros(h.shape[1])
            if enough.any():
                m[enough] = np.nanquantile(pos[:, enough], q, axis=0)
            live = m > 0
            if live.any():
                c = np.ones_like(m)
                c[live] = np.clip(m[live].mean() / m[live], 0.25, 4.0)
                fl[l] = (W * c[:, None], b * c, act)
                W2, b2, act2 = fl[l + 1]
                fl[l + 1] = (W2 / c[None, :], b2, act2)
                h = h * c
        a = act(h)
    return fl


def pool_layers(pool: MaxPool, t: np.ndarray, Wi: np.ndarray, bi: np.ndarray):
    """The integer layers that compute `pool` over a hidden layer with integer weights Wi (n, in), biases bi (already
    including the staircase shift) and thresholds t (K, nondecreasing): returns (W1, b1, thr1, W2, b2) where layer 1
    has n*K units (i, k) = [h_i >= t_k] (shared threshold t_0; the bias is lowered by t_k - t_0) and layer 2 has
    n_out*K units [some i in window j has level >= k], threshold 1. The pooled count is the sum of layer 2's
    units over k (the next layer's weights are repeated K times, see `quantise`)."""
    K, n = len(t), Wi.shape[0]
    W1 = np.repeat(Wi, K, axis=0)
    b1 = np.repeat(bi, K) - np.tile(t - t[0], n)
    W2 = np.zeros((pool.n_out * K, n * K), dtype=np.int64)
    for j, g in enumerate(pool.groups):
        for k in range(K):
            W2[j * K + k, g * K + k] = 1
    return W1, b1, np.array([t[0]], dtype=np.int64), W2, np.zeros(pool.n_out * K, dtype=np.int64)


def quantise(cls, layers, calib_x, levels=8, *, input_shape=None, calibrate: str = "max", equalise: bool | None = None,
             input_stair=None):
    """Quantise float layers into an integer `cls` (= df.MLP): see `MLP.from_numpy` for the full description.
    input_stair: (step per feature, levels, signed) fixing the input staircase (a qat.InputStair), else it is
    calibrated from calib_x."""
    if calibrate not in CALIBRATIONS:
        raise ValueError(f"calibrate must be one of {CALIBRATIONS}, got {calibrate!r}")
    fl = normalize_layers(layers)
    Ks = level_list(levels, len(fl))
    cx = np.asarray(calib_x, dtype=np.float64)
    n_in = fl[0][0].shape[1]
    shape = tuple(cx.shape[1:]) if cx.ndim > 2 else None
    if input_shape is not None:
        shape = tuple(int(i) for i in input_shape)
        if int(np.prod(shape)) != n_in:
            raise ValueError(f"input_shape {shape} has {int(np.prod(shape))} values, the first layer takes {n_in}")
    if cx.ndim < 2 or not len(cx):
        raise ValueError(f"calib_x must be (N, {n_in})")
    cx = cx.reshape(len(cx), -1)
    if cx.shape[1] != n_in:
        raise ValueError(f"calib_x must be (N, {n_in})")
    if not np.isfinite(cx).all():
        bad = np.argwhere(~np.isfinite(cx))[0]
        raise ValueError(f"calib_x has a non-finite value (NaN or inf) at row {int(bad[0])}, feature {int(bad[1])} "
                         f"(flat index): {int((~np.isfinite(cx)).sum())} in all; clean the calibration set")
    if input_stair is None:
        K = Ks[0]
        signed = bool((cx < 0).any())
        D_in = input_steps(cx, K, signed, calibrate)
    else:
        D_in, K, signed = np.asarray(input_stair[0], dtype=np.float64), int(input_stair[1]), bool(input_stair[2])
        if D_in.shape != (n_in,) or not (D_in > 0).all() or K < 1:
            raise ValueError("input_stair needs a positive step per input feature and levels >= 1")
    if equalise is None:
        equalise = calibrate == "mse"
    work = equalise_layers(fl, cx) if equalise else fl
    in_scale = D_in / INPUT_SUBSTEPS
    thr = [np.maximum(1, np.round((np.arange(1, K + 1) - 0.5) * INPUT_SUBSTEPS)).astype(np.int64)]
    a = np.rint(cx / in_scale).astype(np.int64)
    a = adc_signed(a, thr[0]) if signed else adc_relu(a, thr[0])
    unit, off = D_in, np.zeros(n_in)                  # float value of an activation: off + unit * a, per unit
    kin, expand = K, 1                                # largest count of an input unit; input columns per unit (pooled)
    Ws, bs, s = [], [], 1.0
    for l, (W, b, act) in enumerate(work):
        last = l == len(work) - 1
        limit = Z_MAX if last else PORT_MAX
        Wu = W * unit
        b_eff = b + W @ off
        amax = np.full(W.shape[1], kin, dtype=np.int64)
        s = max(np.abs(Wu).max(initial=0) / W_MAX,
                float((np.abs(b_eff) + kin * np.abs(Wu).sum(1)).max(initial=0)) / limit, 1e-12)
        while True:
            Wi, bi = np.rint(Wu / s).astype(np.int64), np.rint(b_eff / s).astype(np.int64)
            ok = np.abs(Wi).max(initial=0) <= W_MAX
            if ok and not last:
                h = a @ Wi.T + bi
                t, shift, st_off, st_unit = plan_staircase(act, h, s, Ks[l + 1], calibrate)
                spread = int(t[-1] - t[0]) if act.pool is not None else 0     # the lowered biases of a pooled layer
                ok = ((np.abs(bi + shift) + np.abs(Wi) @ amax).max(initial=0) + spread <= limit
                      and int(t.max()) <= PORT_MAX)
            elif ok:
                ok = (np.abs(bi) + np.abs(Wi) @ amax).max(initial=0) <= limit
            if ok:
                break
            s *= 1.01
        Wcols = np.repeat(Wi, expand, axis=1) if expand > 1 else Wi      # a pooled input unit feeds `expand` columns
        expand = 1
        if last:
            Ws.append(Wcols)
            bs.append(bi)
            break
        bi = bi + shift
        h = h + shift
        a = adc_relu(h, t)
        if act.pool is None:
            Ws.append(Wcols)
            bs.append(bi)
            thr.append(t)
            unit, off = np.full(W.shape[0], st_unit), np.full(W.shape[0], st_off)
            kin = len(t)
        else:                                                    # exact max pool, see pool_layers
            W1, b1, t1, W2, b2 = pool_layers(act.pool, t, Wcols, bi)
            Ws += [W1, W2]
            bs += [b1, b2]
            thr += [t1, np.array([1], dtype=np.int64)]
            a = act.pool(a)
            unit, off = np.full(act.pool.n_out, st_unit), np.full(act.pool.n_out, st_off)
            kin, expand = len(t), len(t)
    net = cls(Ws, bs, thr, signed_input=signed)
    net.in_scale, net.out_scale = in_scale, float(s)
    net.float_layers, net.float_acts = [(W, b) for W, b, _ in fl], [act for _, _, act in fl]
    net.input_shape = shape if shape is not None and len(shape) > 1 else None
    return net
