"""Float-model interop: staircases for monotone activations, conv layers as exact linear layers, torch walking.

Nothing here talks to DOOM. This module prepares *float* models for `df.MLP.from_numpy` / `from_torch`
(doomformer.kit), which turn them into an integer MLP; DOOM then computes that integer MLP and every number is
checked against an independent integer reference. What each part does, exactly:

* Staircase activations are an APPROXIMATION. DOOM's activation is the count of passed gates,
  ``adc_relu(z, thr) = #{k : z >= thr[k]}``: a non-decreasing step function with K steps. `staircase(fn, x_range, levels)`
  chooses K thresholds so that ``lo + step * count(z)`` follows the float function ``fn`` (K equal output steps, each
  threshold at the first z whose ``fn(z)`` reaches the middle of its step). The next layer's weights absorb ``step`` and
  its bias absorbs ``lo`` (`MLP.from_numpy`), so ``fn`` never has to be evaluated by DOOM. `Staircase.max_error()`
  measures how far the staircase is from ``fn`` on its range; outside the range the staircase is constant.
* Convolutions are LOWERED, not approximated: `conv2d_matrix` / `conv1d_matrix` return the dense matrix (mostly zero)
  whose product with the flattened input is the convolution (zero padding included). In float it equals torch's
  conv up to rounding of the sums. It is then quantised like any Linear layer; the zero taps cost nothing in DOOM
  (no gate lane, no line is emitted for a zero weight). Eval-mode BatchNorm is folded into the layer it follows (or
  the next one) and AvgPool / AdaptiveAvgPool are dense averaging matrices: exact in float. MaxPool next to a
  monotone activation is lowered exactly on the staircase counts (doomformer/quantize.py).
* GELU and SiLU are not monotone: their staircase follows the monotone envelope of the function over the calibrated
  range (`monotone_envelope`), an extra error of half the depth of the dip.
* Python does all of this before DOOM sees anything.
"""
from __future__ import annotations

import itertools
import math
from collections.abc import Callable
from dataclasses import dataclass, field, replace

import numpy as np


# --- staircases ----------------------------------------------------------------------------------------------------
class Staircase(np.ndarray):
    """The thresholds (float, in the pre-activation domain, nondecreasing, length K) of a K-level staircase.

    It is an ndarray of the thresholds with extra attributes: ``lo`` (output value at count 0), ``step`` (output
    value of one more passed threshold), ``x_range``, ``fn``. ``value(z) = lo + step * counts(z)`` is the float
    function the staircase stands for; ``counts(z) = #{k : z >= thr[k]}`` is what DOOM's gates produce."""

    lo: float
    step: float
    x_range: tuple[float, float]
    fn: Callable | None

    def __new__(cls, thresholds, lo: float, step: float, x_range, fn=None):
        obj = np.asarray(thresholds, dtype=np.float64).view(cls)
        obj.lo, obj.step, obj.x_range, obj.fn = float(lo), float(step), (float(x_range[0]), float(x_range[1])), fn
        return obj

    def __array_finalize__(self, obj):
        for name in ("lo", "step", "x_range", "fn"):
            setattr(self, name, getattr(obj, name, None))

    def counts(self, z) -> np.ndarray:
        """#{k : z >= thr[k]} for every z."""
        return np.searchsorted(np.asarray(self), np.asarray(z, dtype=np.float64), side="right")

    def value(self, z) -> np.ndarray:
        """The float value the staircase stands for: lo + step * counts(z)."""
        return self.lo + self.step * self.counts(z)

    def max_error(self, n: int = 4001) -> float:
        """max |fn(z) - value(z)| over n evenly spaced z in x_range (needs fn)."""
        if self.fn is None:
            raise ValueError("this staircase has no fn to compare with")
        z = np.linspace(*self.x_range, n)
        return float(np.abs(np.asarray(self.fn(z), dtype=np.float64) - self.value(z)).max())

    def int_thresholds(self, scale: float) -> tuple[np.ndarray, int]:
        """Integer thresholds for a pre-activation integer h = z / scale: (round(thr / scale) + shift, shift).
        shift >= 0 is the smallest integer that makes every threshold >= 1 (DOOM's thresholds are positive); the
        caller must add `shift` to the bias that produces h (the gate then sees h + shift)."""
        t = np.rint(np.asarray(self) / scale).astype(np.int64)
        shift = max(0, 1 - int(t.min(initial=1)))
        return t + shift, shift


def staircase(fn: Callable, x_range: tuple[float, float], levels: int) -> Staircase:
    """Thresholds of a `levels`-step staircase approximating a monotone nondecreasing function.

    fn: vectorised (numpy array in, array out), nondecreasing (checked on 257 points; a ValueError names the
    problem). x_range = (lo, hi): the pre-activation range the staircase covers; for z below lo the staircase equals
    its level 0, above hi its top level. The output range [fn(lo), fn(hi)] is cut into `levels` equal steps; the k-th
    threshold is the smallest z in [lo, hi] (found by bisection to ~1e-12 of the range) at which fn reaches the middle
    of step k, so the staircase value is within half a step of fn wherever fn is continuous and inside the range.
    Discontinuous fn put their threshold at the jump.

    Returns a `Staircase` (an ndarray of float thresholds with .lo/.step/.value/.counts/.max_error). To use the
    thresholds in an integer MLP, either hand them to `MLP.from_numpy` (activation per layer, see there), or turn them
    into integers yourself with ``Staircase.int_thresholds(scale)``, which also reports the bias shift needed to make
    them positive.

        st = df.staircase(np.tanh, (-3, 3), 8)       # 8 thresholds, st.lo = tanh(-3), st.step = (tanh(3)-tanh(-3))/8
    """
    K = int(levels)
    lo, hi = float(x_range[0]), float(x_range[1])
    if K < 1:
        raise ValueError("levels must be >= 1")
    if not (math.isfinite(lo) and math.isfinite(hi) and lo < hi):
        raise ValueError(f"x_range must be finite with lo < hi, got {x_range!r}")
    xs = np.linspace(lo, hi, 257)
    ys = _eval(fn, xs)
    if (np.diff(ys) < -1e-9 * max(1.0, float(np.abs(ys).max()))).any():
        raise ValueError("fn is not nondecreasing on x_range: a staircase of dolls can only count thresholds crossed "
                         "(use a monotone activation, or restrict x_range to a monotone part)")
    y_lo, y_hi = float(ys[0]), float(ys[-1])
    if not y_hi > y_lo:
        raise ValueError(f"fn is constant on x_range {x_range!r} (fn(lo) == fn(hi) == {y_lo}); nothing to approximate")
    step = (y_hi - y_lo) / K
    targets = y_lo + (np.arange(1, K + 1) - 0.5) * step
    a, b = np.full(K, lo), np.full(K, hi)                  # invariant: fn(b) >= target; fn(a) < target or a == lo
    for _ in range(100):
        m = (a + b) / 2
        ok = _eval(fn, m) >= targets
        b, a = np.where(ok, m, b), np.where(ok, a, m)
    return Staircase(b, y_lo, step, (lo, hi), fn)


def _eval(fn: Callable, z: np.ndarray) -> np.ndarray:
    y = np.asarray(fn(z), dtype=np.float64)
    if y.shape != z.shape or not np.isfinite(y).all():
        raise ValueError("fn must map an array to a finite array of the same shape")
    return y


# --- activation specs ----------------------------------------------------------------------------------------------
def _sigmoid(z):
    z = np.asarray(z, dtype=np.float64)
    e = np.exp(-np.abs(z))
    return np.where(z >= 0, 1.0 / (1.0 + e), e / (1.0 + e))


@dataclass(frozen=True, eq=False)
class MaxPool:
    """A max pool over the units of an activation: output unit j is the maximum of the input units `groups[j]`
    (index arrays into the flat input, ascending, no repeats; windows at an edge simply have fewer taps). `n_in` is
    the number of input units. Applying it to the integer counts of a staircase is exact (see
    `MLP.from_numpy`: pooled counts are computed by comparisons, `max_i a_i = sum_k [some a_i >= k]`)."""
    groups: tuple[np.ndarray, ...]
    n_in: int

    @property
    def n_out(self) -> int:
        return len(self.groups)

    def __call__(self, h):
        h = np.asarray(h)
        return np.stack([h[..., g].max(-1) for g in self.groups], -1)

    def then(self, outer: "MaxPool") -> "MaxPool":
        """The pool that first applies `self`, then `outer` (a max of maxes is a max over the union of the windows)."""
        if outer.n_in != self.n_out:
            raise ValueError(f"pool of {outer.n_in} units applied to {self.n_out} pooled units")
        return MaxPool(tuple(np.unique(np.concatenate([self.groups[i] for i in g])) for g in outer.groups), self.n_in)


@dataclass(frozen=True, eq=False)
class Act:
    """One activation of a float model. kind: 'relu' (the calibrated ReLU staircase), 'identity' (merged into the
    next Linear), 'fn' (any monotone function, staircase over the calibrated range) or 'staircase' (thresholds fixed
    by the caller). spec: a JSON-able description for save/load, None when the activation cannot be persisted.
    monotone=False marks a function that is not nondecreasing (GELU, SiLU): its staircase follows the monotone
    envelope of fn over the calibrated range (`monotone_envelope`), an approximation with a measured error.
    pool: a MaxPool applied after the activation (`pool_first=False`) or before it (`pool_first=True`, the order of
    a model that pools its linear outputs and then activates them); for a monotone activation the two are the same
    function, and a staircase commutes with max exactly."""
    kind: str
    name: str
    fn: Callable | None = None
    spec: list | None = None
    stair: Staircase | None = field(default=None, repr=False)
    monotone: bool = True
    pool: MaxPool | None = field(default=None, repr=False)
    pool_first: bool = False

    def __call__(self, z):
        z = np.asarray(z, dtype=np.float64)
        if self.pool is not None and self.pool_first:
            z = self.pool(z)
        if self.kind == "relu":
            y = np.maximum(z, 0)
        elif self.kind == "identity":
            y = z
        else:
            y = np.asarray(self.fn(z), dtype=np.float64)
        return self.pool(y) if self.pool is not None and not self.pool_first else y

    @property
    def needs_calibration(self) -> bool:
        return self.kind in ("relu", "fn")


def with_pool(act: Act, pool: MaxPool, pool_first: bool) -> Act:
    """`act` followed by (or preceded by) `pool`; a second pool on the same activation is composed with the first."""
    if act.pool is not None:
        if act.pool_first != pool_first:
            raise ValueError("MaxPool before and after the same activation: put the activation between the pools")
        pool = act.pool.then(pool)
    return replace(act, pool=pool, pool_first=pool_first, spec=None)


def monotone_envelope(fn: Callable, x_range: tuple[float, float], n: int = 4097) -> Callable:
    """A nondecreasing function close to `fn` on x_range: the midpoint of the running maximum from the left and the
    running minimum from the right of fn on n grid points (the best uniform approximation by a nondecreasing
    function, linearly interpolated). Its error against fn on the grid is half the largest drop that fn makes on
    the way up (GELU on (-3, 4): 0.085, SiLU: 0.14); the error is zero where fn is already monotone."""
    xs = np.linspace(float(x_range[0]), float(x_range[1]), n)
    ys = _eval(fn, xs)
    g = (np.maximum.accumulate(ys) + np.minimum.accumulate(ys[::-1])[::-1]) / 2
    return lambda z: np.interp(np.asarray(z, dtype=np.float64), xs, g)


def gelu(z, approximate: str = "none"):
    """GELU: z Phi(z) (approximate="none", the erf form) or the tanh approximation (as torch.nn.GELU)."""
    z = np.asarray(z, dtype=np.float64)
    if approximate == "tanh":
        return 0.5 * z * (1.0 + np.tanh(math.sqrt(2.0 / math.pi) * (z + 0.044715 * z ** 3)))
    if approximate != "none":
        raise ValueError(f"approximate must be 'none' or 'tanh', got {approximate!r}")
    return 0.5 * z * (1.0 + np.vectorize(math.erf, otypes=[np.float64])(z / math.sqrt(2.0)))


def silu(z):
    """SiLU / swish: z sigmoid(z)."""
    z = np.asarray(z, dtype=np.float64)
    return z * _sigmoid(z)


RELU = Act("relu", "relu", spec=["relu"])
IDENTITY = Act("identity", "identity", spec=["identity"])


def _hardtanh(lo: float, hi: float, name: str) -> Act:
    lo, hi = float(lo), float(hi)
    if not lo < hi:
        raise ValueError(f"{name}: need min < max, got {lo}, {hi}")
    return Act("fn", f"{name}({lo:g}, {hi:g})", fn=lambda z: np.clip(z, lo, hi), spec=["hardtanh", lo, hi])


def resolve_act(a) -> Act | None:
    """Turn an activation spec into an Act (None stays None = "the default for that position").

    Accepted: "relu", "identity" (or "linear"), "relu6", "sigmoid", "tanh", "hardtanh" (= clamp to [-1, 1]),
    "gelu", "gelu_tanh", "silu" (not monotone: the staircase follows `monotone_envelope`),
    ("hardtanh", lo, hi) / ("clamp", lo, hi), a `Staircase` (thresholds fixed by you, see `staircase`), or any
    vectorised monotone nondecreasing callable (staircase over the calibrated pre-activation range)."""
    if a is None or isinstance(a, Act):
        return a
    if isinstance(a, Staircase):
        if a.fn is None:
            raise ValueError("a Staircase used as an activation needs its fn (use df.staircase(fn, ...))")
        return Act("staircase", f"staircase({len(a)})", fn=a.fn, stair=a)
    if isinstance(a, str):
        key = a.lower()
        if key == "relu":
            return RELU
        if key in ("identity", "linear"):
            return IDENTITY
        if key == "relu6":
            return Act("fn", "relu6", fn=lambda z: np.clip(z, 0, 6), spec=["relu6"])
        if key == "sigmoid":
            return Act("fn", "sigmoid", fn=_sigmoid, spec=["sigmoid"])
        if key == "tanh":
            return Act("fn", "tanh", fn=np.tanh, spec=["tanh"])
        if key == "hardtanh":
            return _hardtanh(-1.0, 1.0, "hardtanh")
        if key == "gelu":
            return Act("fn", "gelu", fn=gelu, spec=["gelu"], monotone=False)
        if key == "gelu_tanh":
            return Act("fn", "gelu_tanh", fn=lambda z: gelu(z, "tanh"), spec=["gelu_tanh"], monotone=False)
        if key in ("silu", "swish"):
            return Act("fn", "silu", fn=silu, spec=["silu"], monotone=False)
        raise ValueError(f"unknown activation {a!r} (relu, identity, relu6, sigmoid, tanh, hardtanh, gelu, "
                         "gelu_tanh, silu, ('clamp', lo, hi), a Staircase or a monotone callable)")
    if isinstance(a, (tuple, list)) and len(a) == 3 and a[0] in ("hardtanh", "clamp"):
        return _hardtanh(a[1], a[2], a[0])
    if callable(a):
        return Act("fn", getattr(a, "__name__", "fn"), fn=a)
    raise TypeError(f"cannot use {a!r} as an activation")


def act_from_spec(spec) -> Act:
    """Inverse of Act.spec (used by MLP.load)."""
    return resolve_act(spec[0] if len(spec) == 1 else tuple(spec))


def normalize_layers(layers) -> list[tuple[np.ndarray, np.ndarray, Act]]:
    """[(W, b) | (W, b, act), ...] -> [(W float (out, in), b float (out,), Act)] with identity activations merged
    into the next layer (exact in float: W2 (W1 x + b1) + b2) and the last activation required to be identity.
    A missing / None activation means ReLU between layers and nothing after the last layer."""
    out: list[tuple[np.ndarray, np.ndarray, Act]] = []
    n = len(layers)
    if not n:
        raise ValueError("need at least one layer")
    for i, item in enumerate(layers):
        if len(item) not in (2, 3):
            raise ValueError(f"layer {i}: expected (W, b) or (W, b, activation)")
        W = np.asarray(item[0], dtype=np.float64)
        b = np.asarray(item[1], dtype=np.float64).reshape(-1)
        if W.ndim != 2 or b.shape != (W.shape[0],):
            raise ValueError(f"layer {i}: W {W.shape} / b {b.shape} do not form a Linear layer")
        act = resolve_act(item[2] if len(item) == 3 else None)
        if i == n - 1:
            if act is not None and act.kind != "identity":
                raise ValueError(f"the last layer has activation {act.name!r}: DOOM returns the raw sums of the last "
                                 "layer, apply the activation in Python to decode(y)")
            act = IDENTITY
        elif act is None:
            act = RELU
        if out and out[-1][2].kind == "identity":
            W1, b1, _ = out.pop()
            if W.shape[1] != W1.shape[0]:
                raise ValueError(f"layer {i}: {W.shape[1]} inputs, previous layer has {W1.shape[0]} outputs")
            W, b = W @ W1, W @ b1 + b
        out.append((W, b, act))
    for i in range(len(out)):
        W, _, act = out[i]
        if act.pool is not None and act.pool.n_in != W.shape[0]:
            raise ValueError(f"layer {i}: a pool of {act.pool.n_in} units on a layer with {W.shape[0]} outputs")
        if i and out[i][0].shape[1] != (out[i - 1][2].pool.n_out if out[i - 1][2].pool is not None else out[i - 1][0].shape[0]):
            raise ValueError(f"layer {i}: {out[i][0].shape[1]} inputs, previous layer has "
                             f"{out[i - 1][2].pool.n_out if out[i - 1][2].pool is not None else out[i - 1][0].shape[0]} outputs")
    return out


def level_list(levels, n: int) -> list[int]:
    """`levels`: an int (every staircase) or n ints (the input staircase, then one per hidden activation)."""
    if isinstance(levels, (int, np.integer)):
        ks = [int(levels)] * n
    else:
        ks = [int(k) for k in levels]
        if len(ks) != n:
            raise ValueError(f"levels: this net has {n} staircases (the input, then one per hidden layer); got {len(ks)}")
    if min(ks) < 1:
        raise ValueError("levels must be >= 1")
    return ks


# --- convolutions as linear layers ---------------------------------------------------------------------------------
def _pair(v, name: str) -> tuple[int, int]:
    if isinstance(v, (int, np.integer)):
        return int(v), int(v)
    v = tuple(int(i) for i in v)
    if len(v) != 2:
        raise ValueError(f"{name} must be an int or a pair")
    return v


def _one(v, name: str) -> int:
    if isinstance(v, (int, np.integer)):
        return int(v)
    v = tuple(int(i) for i in v)
    if len(v) != 1:
        raise ValueError(f"{name} must be an int or a 1-tuple")
    return v[0]


def conv2d_matrix(weight, bias, in_shape, stride=1, padding=0, dilation=1, groups: int = 1):
    """The convolution torch.nn.Conv2d(weight, bias, stride, padding, dilation, groups) as a dense matrix.

    weight: (C_out, C_in // groups, kh, kw); in_shape: (C_in, H, W); padding: int, pair, "valid" or "same" (zeros).
    Returns (M, b, out_shape) with M (C_out*Ho*Wo, C_in*H*W) float64 (almost all zeros) and b (C_out*Ho*Wo,) the bias
    repeated per output position, so that  M @ x.reshape(-1) + b  equals conv2d(x).reshape(-1) (same row-major
    flattening as torch.nn.Flatten) up to float rounding. Padding taps are simply absent (zero padding)."""
    w = np.asarray(weight, dtype=np.float64)
    if w.ndim != 4:
        raise ValueError(f"weight must be (C_out, C_in/groups, kh, kw), got {w.shape}")
    Co, Cg, kh, kw = w.shape
    C, H, Wd = (int(i) for i in in_shape)
    if groups < 1 or C != Cg * groups or Co % groups:
        raise ValueError(f"weight {w.shape} with groups={groups} does not fit {C} input channels")
    bvec = np.zeros(Co) if bias is None else np.asarray(bias, dtype=np.float64).reshape(-1)
    if bvec.shape != (Co,):
        raise ValueError(f"bias must have {Co} entries")
    (sy, sx), (dy, dx) = _pair(stride, "stride"), _pair(dilation, "dilation")
    if isinstance(padding, str):
        if padding == "valid":
            before, total = (0, 0), (0, 0)
        elif padding == "same":
            if (sy, sx) != (1, 1):
                raise ValueError('padding="same" needs stride 1 (as in torch)')
            total = (dy * (kh - 1), dx * (kw - 1))
            before = (total[0] // 2, total[1] // 2)
        else:
            raise ValueError(f"padding must be an int, a pair, 'valid' or 'same', got {padding!r}")
    else:
        p = _pair(padding, "padding")
        before, total = p, (2 * p[0], 2 * p[1])
    Ho = (H + total[0] - dy * (kh - 1) - 1) // sy + 1
    Wo = (Wd + total[1] - dx * (kw - 1) - 1) // sx + 1
    if Ho < 1 or Wo < 1:
        raise ValueError(f"kernel {kh}x{kw} does not fit the {H}x{Wd} input")
    M = np.zeros((Co * Ho * Wo, C * H * Wd))
    oy, ox = np.arange(Ho)[:, None], np.arange(Wo)[None, :]
    per_group = Co // groups
    for ky in range(kh):
        for kx in range(kw):
            iy = oy * sy - before[0] + ky * dy + np.zeros_like(ox)
            ix = ox * sx - before[1] + kx * dx + np.zeros_like(oy)
            ok = (iy >= 0) & (iy < H) & (ix >= 0) & (ix < Wd)
            iy, ix, pos = iy[ok], ix[ok], (oy * Wo + ox)[ok]
            for co in range(Co):
                g = co // per_group
                for cl in range(Cg):
                    v = w[co, cl, ky, kx]
                    if v != 0.0:
                        M[co * Ho * Wo + pos, ((g * Cg + cl) * H + iy) * Wd + ix] += v
    return M, np.repeat(bvec, Ho * Wo), (Co, Ho, Wo)


def conv1d_matrix(weight, bias, in_shape, stride=1, padding=0, dilation=1, groups: int = 1):
    """`conv2d_matrix` for torch.nn.Conv1d: weight (C_out, C_in // groups, k), in_shape (C_in, L); returns
    (M, b, (C_out, L_out))."""
    w = np.asarray(weight, dtype=np.float64)
    if w.ndim != 3:
        raise ValueError(f"weight must be (C_out, C_in/groups, k), got {w.shape}")
    C, L = (int(i) for i in in_shape)
    pad = padding if isinstance(padding, str) else (0, _one(padding, "padding"))
    M, b, (Co, _, Lo) = conv2d_matrix(w[:, :, None, :], bias, (C, 1, L), (1, _one(stride, "stride")), pad,
                                       (1, _one(dilation, "dilation")), groups)
    return M, b, (Co, Lo)


# --- pooling as windows --------------------------------------------------------------------------------------------
def _axis_windows(length: int, k: int, stride: int, pad: int, dil: int):
    """Per output position of one axis: the valid input indices of its window (padded taps are absent)."""
    n = (length + 2 * pad - dil * (k - 1) - 1) // stride + 1
    if n < 1:
        raise ValueError(f"a pool window of {k} does not fit an axis of {length}")
    wins = []
    for o in range(n):
        pos = o * stride - pad + np.arange(k) * dil
        wins.append(pos[(pos >= 0) & (pos < length)])
    return wins


def _grid_windows(shape: tuple[int, ...], axes: list[list[np.ndarray]]):
    """Windows over a (C, *spatial) row-major flat tensor: one ascending index array per (channel, output position),
    channel major, positions row-major. `axes[d][o]` = the input indices of output position o along spatial axis d."""
    sp = shape[1:]
    S = int(np.prod(sp))
    steps = [int(np.prod(sp[d + 1:])) for d in range(len(sp))]
    outs = []
    for combo in itertools.product(*[range(len(w)) for w in axes]):
        idx = np.zeros(1, dtype=np.int64)
        for d, o in enumerate(combo):
            idx = (idx[:, None] + axes[d][o][None, :] * steps[d]).ravel()
        outs.append(idx)
    return [np.sort(g + c * S) for c in range(shape[0]) for g in outs]


def _tuple(v, d: int, name: str) -> tuple[int, ...]:
    if isinstance(v, (int, np.integer)):
        return (int(v),) * d
    v = tuple(int(i) for i in v)
    if len(v) == 1:
        v = v * d
    if len(v) != d:
        raise ValueError(f"{name} must be an int or {d} ints")
    return v


def pool_windows(shape, kernel, stride=None, padding=0, dilation=1):
    """The windows of torch.nn.MaxPool1d/2d / AvgPool1d/2d over a (C, *spatial) sample (flat row-major, as
    nn.Flatten): (out_shape, [index array of each output unit], kernel area). Padded taps are absent from the
    index arrays; the kernel area is the divisor of count_include_pad averaging."""
    shape = tuple(int(i) for i in shape)
    d = len(shape) - 1
    k = _tuple(kernel, d, "kernel_size")
    st = _tuple(kernel if stride is None else stride, d, "stride")
    pad, dil = _tuple(padding, d, "padding"), _tuple(dilation, d, "dilation")
    axes = [_axis_windows(shape[1 + i], k[i], st[i], pad[i], dil[i]) for i in range(d)]
    out_shape = (shape[0], *[len(a) for a in axes])
    return out_shape, _grid_windows(shape, axes), int(np.prod(k))


def adaptive_avg_windows(shape, output_size):
    """The windows of nn.AdaptiveAvgPool1d/2d (start = floor(o L / n), end = ceil((o + 1) L / n))."""
    shape = tuple(int(i) for i in shape)
    d = len(shape) - 1
    size = (output_size,) * d if output_size is None or isinstance(output_size, (int, np.integer)) else tuple(output_size)
    out = _tuple(tuple(shape[1 + i] if s is None else s for i, s in enumerate(size)), d, "output_size")
    axes = []
    for i in range(d):
        L, n = shape[1 + i], out[i]
        axes.append([np.arange((o * L) // n, -((-(o + 1) * L) // n)) for o in range(n)])
    return (shape[0], *out), _grid_windows(shape, axes)


def avg_pool_matrix(n_in: int, groups, divisors=None) -> np.ndarray:
    """The dense matrix whose product with the flat input averages every window (`divisors`: the number to divide
    by, default the window's size)."""
    P = np.zeros((len(groups), n_in))
    for j, g in enumerate(groups):
        P[j, g] = 1.0 / (len(g) if divisors is None else divisors[j])
    return P


# --- torch nn.Sequential -> float layers ---------------------------------------------------------------------------
def _leaves(model):
    import torch

    if isinstance(model, torch.nn.Sequential):
        for m in model:
            yield from _leaves(m)
    else:
        yield model


def _bn_affine(m, shape: tuple[int, ...]) -> tuple[np.ndarray, np.ndarray]:
    """An eval-mode BatchNorm as y = a * x + c per unit of a `shape` sample (a, c flat, row-major)."""
    import torch
    nn = torch.nn

    kind = type(m).__name__
    if m.training:
        raise ValueError(f"{kind} is in training mode: call model.eval() before quantising (the folded statistics "
                         "are the running ones)")
    if m.running_mean is None or m.running_var is None:
        raise ValueError(f"{kind} has no running statistics (track_running_stats=False): nothing to fold")
    if isinstance(m, nn.BatchNorm2d) and len(shape) != 3:
        raise ValueError(f"BatchNorm2d needs a (C, H, W) sample, the tensor is {shape}")
    if isinstance(m, nn.BatchNorm1d) and len(shape) not in (1, 2):
        raise ValueError(f"BatchNorm1d needs a (n,) or (C, L) sample, the tensor is {shape}")
    if shape[0] != m.num_features:
        raise ValueError(f"{kind}({m.num_features}) does not match the {shape[0]} features/channels of {shape}")
    mean, var = m.running_mean.detach().cpu().double().numpy(), m.running_var.detach().cpu().double().numpy()
    gamma = m.weight.detach().cpu().double().numpy() if m.weight is not None else np.ones(m.num_features)
    beta = m.bias.detach().cpu().double().numpy() if m.bias is not None else np.zeros(m.num_features)
    a = gamma / np.sqrt(var + m.eps)
    c = beta - mean * a
    per = int(np.prod(shape[1:]))
    return np.repeat(a, per), np.repeat(c, per)


def torch_model(model, input_shape: tuple[int, ...]):
    """`torch_layers` plus what the model says about its input: (layers, input_stair) where input_stair is None or
    (step per input feature (flat float array), levels, signed) from a leading `doomformer.qat.InputStair`."""
    import torch
    nn = torch.nn
    from . import qat

    shape = tuple(int(i) for i in input_shape)
    layers: list[list] = []                          # [W, b, Act | None] (None: no activation yet)
    pending: list = [None]                           # (a, c): a BatchNorm after an activation, folded into the next layer
    pool: list = [None]                              # (MaxPool, name): a max pool seen before its activation
    input_stair = None
    n_seen = 0

    def weight_layer(W, b, out_shape):
        nonlocal shape
        if pool[0] is not None:
            raise ValueError(f"unsupported {pool[0][1]}: followed by a layer with no activation in between (a max "
                             "over linear outputs is not a staircase of a sum: put a monotone activation, e.g. ReLU, "
                             "right before or after the pool)")
        if pending[0] is not None:
            a, c = pending[0]
            if a.shape[0] != W.shape[1]:
                raise ValueError("internal: BatchNorm and layer sizes differ")
            W, b = W * a[None, :], b + W @ c
            pending[0] = None
        layers.append([W, b, None])
        shape = out_shape

    def activate(act: Act, m):
        if not layers:
            raise ValueError(f"{type(m).__name__} before the first Linear/Conv: the inputs are quantised by the input "
                             "staircase, put the activation after a layer")
        if layers[-1][2] is not None or pending[0] is not None:
            first = layers[-1][2].name if layers[-1][2] is not None else "BatchNorm"
            raise ValueError(f"two activations in a row ({first}, {type(m).__name__})")
        if pool[0] is not None:
            act = with_pool(act, pool[0][0], True)
            pool[0] = None
        layers[-1][2] = act

    def max_pool(mp, name: str):
        nonlocal shape
        if not layers:
            raise ValueError(f"unsupported {name} before the first Linear/Conv")
        if pending[0] is not None:
            raise ValueError(f"unsupported {name} after a BatchNorm that follows an activation: put the BatchNorm "
                             "before the activation (it folds into the layer) or after the pool")
        d = len(shape) - 1
        if d not in (1, 2) or (d == 1 and isinstance(mp, nn.MaxPool2d)) or (d == 2 and isinstance(mp, nn.MaxPool1d)):
            raise ValueError(f"{name} does not fit the {shape} sample")
        if mp.ceil_mode or mp.return_indices:
            raise ValueError(f"{name}: ceil_mode and return_indices are not supported")
        out_shape, groups, _ = pool_windows(shape, mp.kernel_size, mp.stride, mp.padding, mp.dilation)
        mpool = MaxPool(tuple(np.unique(g) for g in groups), int(np.prod(shape)))
        shape = out_shape
        if layers[-1][2] is not None:                        # activation, then pool
            layers[-1][2] = with_pool(layers[-1][2], mpool, False)
        elif pool[0] is not None:
            pool[0] = (pool[0][0].then(mpool), name)
        else:                                                # linear outputs, pooled before an activation
            pool[0] = (mpool, name)

    def avg_pool(am, name: str, adaptive: bool):
        d = len(shape) - 1
        if d not in (1, 2):
            raise ValueError(f"{name} needs a (C, L) or (C, H, W) sample, the tensor is {shape}")
        n_in = int(np.prod(shape))
        if adaptive:
            out_shape, groups = adaptive_avg_windows(shape, am.output_size)
            div = None
        else:
            if am.ceil_mode:
                raise ValueError(f"{name}: ceil_mode is not supported")
            out_shape, groups, area = pool_windows(shape, am.kernel_size, am.stride, am.padding, 1)
            div = None
            if getattr(am, "divisor_override", None) is not None:
                div = [am.divisor_override] * len(groups)
            elif getattr(am, "count_include_pad", True):
                div = [area] * len(groups)
        weight_layer(avg_pool_matrix(n_in, groups, div), np.zeros(len(groups)), out_shape)

    def batch_norm(m):
        a, c = _bn_affine(m, shape)
        if layers and layers[-1][2] is None and pool[0] is None:     # after a linear layer: fold into its rows
            W, b, _ = layers[-1]
            layers[-1] = [W * a[:, None], b * a + c, None]
        elif pool[0] is not None:
            raise ValueError(f"unsupported {type(m).__name__} between a MaxPool and its activation")
        else:                                                        # after an activation / at the input: fold forward
            if pending[0] is not None:
                pa, pc = pending[0]
                a, c = a * pa, a * pc + c
            pending[0] = (a, c)

    for m in _leaves(model):
        if isinstance(m, qat.InputStair):
            if n_seen or layers:
                raise ValueError("InputStair must be the first module")
            if int(np.prod(m.step.shape)) != int(np.prod(shape)):
                raise ValueError(f"InputStair has {int(np.prod(m.step.shape))} features, the sample {shape}")
            input_stair = (m.step.detach().cpu().double().numpy().reshape(-1).copy(), int(m.levels), bool(m.signed))
            n_seen += 1
            continue
        n_seen += 1
        if isinstance(m, nn.Linear):
            if len(shape) != 1:
                raise ValueError(f"Linear after a {len(shape)}-D tensor {shape}: add nn.Flatten() first")
            if shape[0] != m.in_features:
                raise ValueError(f"Linear expects {m.in_features} features, the tensor has {shape[0]}")
            b = m.bias.detach().cpu().double().numpy() if m.bias is not None else np.zeros(m.out_features)
            weight_layer(m.weight.detach().cpu().double().numpy(), b, (m.out_features,))
        elif isinstance(m, (nn.Conv1d, nn.Conv2d)):
            d = 1 if isinstance(m, nn.Conv1d) else 2
            if len(shape) != d + 1:
                raise ValueError(f"{type(m).__name__} needs a {d + 1}-D (C,...) sample, the tensor is {shape} "
                                 "(pass input_shape or a calib_x with the sample shape)")
            if m.padding_mode != "zeros":
                raise ValueError(f"{type(m).__name__}: only padding_mode='zeros' is supported")
            if shape[0] != m.in_channels:
                raise ValueError(f"{type(m).__name__} expects {m.in_channels} channels, the tensor has {shape[0]}")
            fn = conv1d_matrix if d == 1 else conv2d_matrix
            b = None if m.bias is None else m.bias.detach().cpu().double().numpy()
            M, bb, out_shape = fn(m.weight.detach().cpu().double().numpy(), b, shape, m.stride, m.padding, m.dilation,
                                  m.groups)
            weight_layer(M, bb, out_shape)
        elif isinstance(m, nn.Flatten):
            if (m.start_dim, m.end_dim) not in ((1, -1), (1, len(shape))):
                raise ValueError("Flatten: only the default start_dim=1, end_dim=-1 is supported")
            shape = (int(np.prod(shape)),)
        elif isinstance(m, nn.Identity):
            pass
        elif isinstance(m, nn.Dropout):
            if m.training and m.p > 0:
                raise ValueError("Dropout is in training mode (p > 0): call model.eval() before quantising")
        elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
            batch_norm(m)
        elif isinstance(m, (nn.MaxPool1d, nn.MaxPool2d)):
            max_pool(m, type(m).__name__)
        elif isinstance(m, (nn.AvgPool1d, nn.AvgPool2d)):
            avg_pool(m, type(m).__name__, False)
        elif isinstance(m, (nn.AdaptiveAvgPool1d, nn.AdaptiveAvgPool2d)):
            avg_pool(m, type(m).__name__, True)
        elif isinstance(m, qat.Stair):
            activate(m.as_act(), m)
        elif isinstance(m, nn.ReLU):
            activate(RELU, m)
        elif isinstance(m, nn.ReLU6):
            activate(resolve_act("relu6"), m)
        elif isinstance(m, nn.Hardtanh):
            activate(_hardtanh(m.min_val, m.max_val, "hardtanh"), m)
        elif isinstance(m, nn.Sigmoid):
            activate(resolve_act("sigmoid"), m)
        elif isinstance(m, nn.Tanh):
            activate(resolve_act("tanh"), m)
        elif isinstance(m, nn.GELU):
            activate(resolve_act("gelu_tanh" if m.approximate == "tanh" else "gelu"), m)
        elif isinstance(m, nn.SiLU):
            activate(resolve_act("silu"), m)
        elif isinstance(m, (nn.LayerNorm, nn.GroupNorm, nn.InstanceNorm1d, nn.InstanceNorm2d)):
            raise ValueError(f"unsupported module {type(m).__name__}: its mean and variance are computed per input, "
                             "so it is not a fixed affine map and cannot be folded (BatchNorm in eval mode can). "
                             "The transformer path (Transformer.from_torch) has an integer RMSNorm")
        else:
            raise ValueError(f"unsupported module {type(m).__name__}: supported are Linear, Conv1d, Conv2d, Flatten, "
                             "Identity, Dropout (eval), BatchNorm1d/2d (eval), AvgPool1d/2d, AdaptiveAvgPool1d/2d, "
                             "MaxPool1d/2d (next to a monotone activation) and the activations ReLU, ReLU6, "
                             "Hardtanh, Sigmoid, Tanh, GELU, SiLU")
    if not layers:
        raise ValueError("the model has no Linear/Conv layer")
    if pool[0] is not None:
        raise ValueError(f"unsupported {pool[0][1]}: no activation follows it (a max over linear outputs is not a "
                         "staircase of a sum: put a monotone activation, e.g. ReLU, right before or after the pool)")
    if pending[0] is not None:
        raise ValueError("a BatchNorm after the last activation has no layer to fold into: put it before the activation")
    return [(W, b, IDENTITY if a is None else a) for W, b, a in layers], input_stair


def torch_layers(model, input_shape: tuple[int, ...]) -> list[tuple[np.ndarray, np.ndarray, Act]]:
    """Walk an nn.Sequential (nested allowed) or a single supported module into float layers [(W, b, Act)].

    Supported: Linear, Conv1d, Conv2d (lowered with conv2d_matrix; padding_mode must be 'zeros'), Flatten (default
    start_dim=1, end_dim=-1), Identity, Dropout (must be in eval mode or p == 0: it does nothing then),
    BatchNorm1d/2d in eval mode (folded into the preceding Linear/Conv, or into the next layer when it follows an
    activation), AvgPool1d/2d and AdaptiveAvgPool1d/2d (an exact linear layer), MaxPool1d/2d next to a monotone
    activation (see `MLP.from_numpy`), the activations ReLU, ReLU6, Hardtanh, Sigmoid, Tanh, GELU, SiLU, and the
    `doomformer.qat` modules. A Linear/Conv without an activation after it is returned with the identity activation
    (normalize_layers merges it into the next layer). Anything else raises ValueError.
    input_shape: the per-sample shape the model is fed, e.g. (1, 5, 5); (n,) for flat inputs."""
    return torch_model(model, input_shape)[0]
