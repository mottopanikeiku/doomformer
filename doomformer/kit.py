"""The doomformer toolkit: build integer networks, compile them into ONE DOOM map, run them, check every number.

    import doomformer as df
    net = df.MLP(weights=[W0, W1], biases=[b0, b1], thresholds=[t0, t1])
    with df.compile(net) as brain:
        y = brain(x)            # computed by DOOM, compared with net.reference(x); raises df.MismatchError

What DOOM computes vs what Python does
--------------------------------------
DOOM (ViZDoom 1.3.2's ZDoom engine) computes every integer of the network's forward pass inside one map:
voodoo dolls (one per staircase level of each input unit) walk through gates whose floor heights are the values,
and every doll that passes crosses lines that raise or lower other sectors' floors/lights by its weight. Hidden
values are floor heights that the next layer's gates read directly; outputs accumulate in sector light levels.
Python compiles the map (geometry + ACS), writes the inputs into the input gates, drops the start barriers, waits
until the dolls stop, reads the values back through ACS readout scripts and compares them with an independent
integer reference (numpy, `intref`), tolerance 0. A check is a per-run comparison, not a proof for all inputs.
Spiking nets: Python writes each step's input spikes; the potentials stay in the map between steps.
Transformers: Python writes the token ids and picks the next token from the DOOM-computed logits (intref.choose).

`MLP.from_numpy` / `from_torch` quantise a float model into an integer `MLP`: that is an approximation of the float
model (see `quantization_report`); DOOM is then exact with respect to the integer model, not the float one.
Conv1d/Conv2d layers are lowered to exact linear layers (so are eval-mode BatchNorm and AvgPool; MaxPool is lowered
exactly on the staircase counts) and ReLU6/Hardtanh/Sigmoid/Tanh/GELU/SiLU (any monotone function) become staircases
(approximations): see `doomformer.interop`, `doomformer.quantize` (calibrate="mse") and docs/interop.md; `doomformer.qat`
fine-tunes a torch model with the staircases before it is lowered.

Limits (intref's contract): every value written into a gate (inputs, hidden values) within +-12,000 (16.16
fixed-point plane heights), every partial sum of an accumulation within +-30,000 (dolls cross in any order;
light levels are int16), weights within +-255.

The world hook
--------------
`df.compile(net, world=fn)` calls ``fn(m)`` with the brain's `udmf.MapBuilder` after the brain is laid out;
``fn`` adds the real player's surroundings and returns extra ACS source (a str, may be empty). MapBuilder basics:

* ``r = m.rect(x0, y0, x1, y1, floor=0, ceil=128, light=160, tag=0, ftex=..., ctex=..., wall=...)`` adds an
  axis-aligned rectangular sector and returns its index. Rectangles must not overlap; edges they share become
  two-sided lines, the others solid walls (checked here: overlapping the brain raises ValueError).
* ``m.link(a, b, special=..., arg0=..., playercross=True, repeatspecial=True)`` puts a line special on every line
  between rects a and b (``playeruse=True`` for a switch-like use).
* ``m.thing(x, y, type, angle=0, id=tid)`` places a thing (1 = player start, 3004 = zombieman, ...).
* the world MUST place a player start (``m.thing(x, y, 1)``): the brain's dolls are earlier player-1 starts, the
  last one becomes the real player and the others his voodoo dolls (checked here).
* build inside the safe world area ``df.plan(net)["map"]["world_area"]`` = [x0, y0, x1, y1] (also
  ``brain.meta.layout["world_area"]``): x in [-16000, 16000], y from the brain's top - 32,000 up to -2048 (the brain
  occupies y >= 0, its output lights a few rows below y = -512, and a closet for the conveyor scrollers sits at
  (-4096, -4096)). The whole map must span <= 32,000 units per axis (checked here): beyond that ViZDoom's 16.16
  coordinate differences wrap and the engine puts things in the wrong sector, e.g. the player on the brain's
  conveyors (docs/engine_limits.md).
* free numbers: sector tags 3..999, TIDs 1..9999, ACS scripts >= 200 (see `act`). The ACS may use the brain's
  globals ``o1..o60`` (ViZDoom USER1..USER60) to report values: ``brain.run_script(200, read=3)``.

`actions={output_name: [specials]}` attach line specials (see `act`) to an output: every doll that adds a
POSITIVE weight to it crosses them, i.e. they fire once per positive contribution unit, not on the final value.
Note: the run only ends once every DoomPlayer (dolls and the player) has stood still for 8 tics (on maps with more
dolls than ViZDoom lists objects, 4,096, an ACS script in the map checks every doll; see runtime.TileRunner.settle).
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
import uuid
from collections.abc import Callable
from pathlib import Path

import numpy as np

from .acc import cache_dir
from .interop import IDENTITY, RELU, Act, act_from_spec, torch_model
from .intref import PORT_MAX, W_MAX, Z_MAX, QModel, adc_relu, adc_signed, check_limits, choose, forward
from .onemap import (build_onemap, compare_trace, exposed_names, logits_from_readout, token_ports,
                     transformer_layers)
from .ops import crossbar, output_pair, output_single
from .planes import PLANE_MAX, mlp_static_errors, spiking_step_errors, terms_peak
from .planes import message as plane_message
from .pure import MAX_SPAN, compile_pure, map_bounds, world_area
from .quantize import quantise
from .spiking import IntNet, build_spiking, read_step, step_ports
from .tile import TileMeta, TileSpec
from .udmf import MapBuilder

SpikingNet = IntNet
Transformer = QModel          # Transformer.load(path or shipped model name), see doomformer.shipped


def limit_sequences(m: QModel, n: int = 24, seed: int = 0) -> list[list[int]]:
    """The sequences `compile` range-checks a transformer on: every single token after BOS, plus `n` seeded random
    sequences of random length up to T and `n` of length T (tokens in the vocabulary)."""
    rng = np.random.default_rng(seed)
    V = len(m.vocab)
    seqs = [[0, t] for t in range(V)] if m.T >= 2 else [[0]]
    seqs += [[0] + [int(t) for t in rng.integers(0, V, int(rng.integers(0, m.T)))] for _ in range(n)]
    seqs += [[int(t) for t in rng.integers(0, V, m.T)] for _ in range(n)]
    return seqs


class RangeError(ValueError):
    """A value would leave the range DOOM's mechanisms represent exactly (the intref limits)."""


class ReadoutError(RuntimeError):
    """A pass read back a different set of names than the lowering exposes (a readout was dropped or invented)."""


class MismatchError(RuntimeError):
    """DOOM's readback differs from the integer reference. `.mismatches`: [(name, doom, reference)]."""

    def __init__(self, what: str, mismatches: list[tuple[str, int, int]]):
        self.mismatches = mismatches
        shown = ", ".join(f"{n}: DOOM {d} != reference {r}" for n, d, r in mismatches[:8])
        more = f" (+{len(mismatches) - 8} more)" if len(mismatches) > 8 else ""
        super().__init__(f"{what}: {len(mismatches)} value(s) differ: {shown}{more}")


def _ints(a, what: str, ndim: int | tuple[int, ...]) -> np.ndarray:
    arr = np.asarray(a)
    if arr.dtype == object or not (np.issubdtype(arr.dtype, np.integer) or np.issubdtype(arr.dtype, np.floating)
                                   or arr.dtype == bool):
        raise TypeError(f"{what}: expected integers, got dtype {arr.dtype}")
    if np.issubdtype(arr.dtype, np.floating) and not np.array_equal(arr, np.round(arr)):
        raise TypeError(f"{what}: expected integers, got non-integral floats (quantise first, e.g. MLP.from_numpy)")
    if arr.ndim not in ((ndim,) if isinstance(ndim, int) else ndim):
        raise ValueError(f"{what}: expected {ndim}-D, got shape {arr.shape}")
    return arr.astype(np.int64)


def _partial_sum_bound(a: np.ndarray, W: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Largest |partial sum| of b + W @ a when the terms arrive in any order (bias first)."""
    terms = a[..., None, :] * W
    hi = b + np.clip(terms, 0, None).sum(-1)
    lo = b + np.clip(terms, None, 0).sum(-1)
    return np.maximum(np.abs(hi), np.abs(lo))


class MLP:
    """Integer MLP of any depth >= 1, computed by DOOM exactly as written here.

        a_0     = adc_signed(x, thr_0) if signed_input else adc_relu(x, thr_0)
        h_{l+1} = b_l + W_l @ a_l
        a_l     = adc_relu(h_l, thr_l)                    for l >= 1
        output  = last h (no activation)

    adc_relu(z, thr) = #{k: z >= thr[k]}, adc_signed(z, thr) = adc_relu(z, thr) - adc_relu(-z, thr) (intref).
    weights[l]: int (out, in), biases[l]: int (out,), thresholds[l]: positive nondecreasing ints <= 12,000
    (len(thresholds) == len(weights); thresholds[0] is the input staircase). In DOOM each threshold of each input
    unit is one voodoo doll (two with signed_input); value names: inputs ``x{i}``, hidden ``h{l}_{j}``
    (l = 1..n-1, their negations ``-h{l}_{j}``), outputs ``y{o}``.
    """

    def __init__(self, weights, biases, thresholds, signed_input: bool = True):
        n = len(weights)
        if n < 1 or len(biases) != n or len(thresholds) != n:
            raise ValueError("need len(weights) == len(biases) == len(thresholds) >= 1")
        self.weights = [_ints(W, f"weights[{l}]", 2) for l, W in enumerate(weights)]
        self.biases = [_ints(b, f"biases[{l}]", 1) for l, b in enumerate(biases)]
        self.thresholds = [_ints(t, f"thresholds[{l}]", 1) for l, t in enumerate(thresholds)]
        self.signed_input = bool(signed_input)
        for l, (W, b, t) in enumerate(zip(self.weights, self.biases, self.thresholds)):
            if b.shape != (W.shape[0],):
                raise ValueError(f"biases[{l}] shape {b.shape} != ({W.shape[0]},)")
            if l and W.shape[1] != self.weights[l - 1].shape[0]:
                raise ValueError(f"weights[{l}] has {W.shape[1]} inputs, layer {l - 1} has "
                                 f"{self.weights[l - 1].shape[0]} outputs")
            if not len(t) or (t < 1).any() or (t > PORT_MAX).any() or (np.diff(t) < 0).any():
                raise RangeError(f"thresholds[{l}] must be nonempty, positive, nondecreasing, <= {PORT_MAX}")
            if np.abs(W).max(initial=0) > W_MAX:
                raise RangeError(f"weights[{l}]: |w| reaches {int(np.abs(W).max())} > {W_MAX}")
        self.in_scale: np.ndarray | None = None      # set by from_numpy/from_torch: x_float ~ x_int * in_scale
        self.out_scale: float | None = None          # y_float ~ y_int * out_scale
        self.float_layers: list[tuple[np.ndarray, np.ndarray]] | None = None
        self.float_acts: list[Act] | None = None     # activation after each float layer (last: identity)
        self.input_shape: tuple[int, ...] | None = None   # per-sample shape (C, H, W) etc. if not flat

    # --- shape and names ---------------------------------------------------------------------------------------
    @property
    def n_layers(self) -> int:
        return len(self.weights)

    @property
    def n_in(self) -> int:
        return self.weights[0].shape[1]

    @property
    def n_out(self) -> int:
        return self.weights[-1].shape[0]

    def _out_name(self, l: int, j: int) -> str:
        return f"y{j}" if l == self.n_layers - 1 else f"h{l + 1}_{j}"

    def value_names(self) -> list[str]:
        """Every output name a map of this net can carry (for `actions`)."""
        names = [f"{s}h{l}_{j}" for l in range(1, self.n_layers) for j in range(self.weights[l - 1].shape[0])
                 for s in ("", "-")]
        return names + [f"y{o}" for o in range(self.n_out)]

    # --- reference ---------------------------------------------------------------------------------------------
    def _x(self, x) -> np.ndarray:
        x = _ints(x, "x", (1, 2))
        if x.shape[-1] != self.n_in:
            raise ValueError(f"x has {x.shape[-1]} features, the net takes {self.n_in}")
        return x

    def reference(self, x) -> dict:
        """Integer forward pass (numpy). x: (n_in,) or (batch, n_in) ints.
        Returns dict(y=..., hidden=[h_1..h_{n-1}], activations=[a_0..a_{n-1}])."""
        x = self._x(x)
        a = adc_signed(x, self.thresholds[0]) if self.signed_input else adc_relu(x, self.thresholds[0])
        acts, hidden = [a], []
        for l, (W, b) in enumerate(zip(self.weights, self.biases)):
            h = a @ W.T + b
            if l == self.n_layers - 1:
                return dict(y=h, hidden=hidden, activations=acts)
            hidden.append(h)
            a = adc_relu(h, self.thresholds[l + 1])
            acts.append(a)
        raise AssertionError("unreachable")

    def check(self, x, planes: bool = True) -> None:
        """Raise RangeError if x would make any gate-fed value leave +-12,000, any partial sum leave +-30,000 or, in
        one map (planes=True), a hidden plane's partial sums leave the engine's plane-move bound (planes.py: +-14,018).
        This is the contract: every input check accepts gives DOOM == reference in every exposed value. Tile mode has
        no doll-moved planes, so tile brains and run_many(mode="tiles") call it with planes=False."""
        x = self._x(x)
        errs = []
        if np.abs(x).max(initial=0) > PORT_MAX:
            errs.append(f"|x| reaches {int(np.abs(x).max())} > {PORT_MAX} (inputs are written into gate floors)")
        r = self.reference(x)
        for l, (W, b) in enumerate(zip(self.weights, self.biases)):
            a = r["activations"][l]
            worst = int(_partial_sum_bound(a, W, b).max(initial=0))
            name = "y" if l == self.n_layers - 1 else f"h{l + 1}"
            if worst > Z_MAX:
                errs.append(f"partial sums of {name} reach {worst} > {Z_MAX}")
            if l < self.n_layers - 1 and np.abs(r["hidden"][l]).max(initial=0) > PORT_MAX:
                errs.append(f"|{name}| reaches {int(np.abs(r['hidden'][l]).max())} > {PORT_MAX} (feeds gates)")
            if planes and l < self.n_layers - 1:
                peak = terms_peak(b, a[..., None, :] * W)
                if peak > PLANE_MAX:
                    errs.append(plane_message(f"-{name}", peak))
        if errs:
            raise RangeError("; ".join(errs))

    def plane_errors(self) -> list[str]:
        """The plane-move bound over ALL inputs (interval worst case, planes.py); `compile` raises RangeError with
        these unless limits=False. `check(x)` applies the same bound to the activations x produces."""
        return mlp_static_errors(self)

    def _expected(self, name: str, ref: dict) -> int:
        sign = -1 if name.startswith("-") else 1
        base = name.lstrip("-")
        if base[0] == "y":
            return sign * int(ref["y"][int(base[1:])])
        l, j = base[1:].split("_")
        return sign * int(ref["hidden"][int(l) - 1][int(j)])

    # --- lowering to DOOM --------------------------------------------------------------------------------------
    def _stages(self) -> list[TileSpec]:
        """One pure layer per weight matrix: hidden layers emit value pairs whose
        negations become the next layer's gate floors; the last layer accumulates into lights."""
        stages, src = [], [f"x{i}" for i in range(self.n_in)]
        for l, (W, b) in enumerate(zip(self.weights, self.biases)):
            if not W.any():
                raise ValueError(f"weights[{l}] is all zero: layer {l} would be the constant biases and has no dolls "
                                 "to carry them (the compiler cannot lower it); fold the constants into the next "
                                 "layer's biases or give the layer a nonzero weight")
            spec = TileSpec(f"mlp_l{l}")
            emit = output_single if l == self.n_layers - 1 else output_pair
            outs = [emit(spec, self._out_name(l, j), int(b[j])) for j in range(W.shape[0])]
            for i, name in enumerate(src):
                crossbar(spec, name, self.thresholds[l], [(outs[j], int(W[j, i])) for j in range(W.shape[0])],
                         signed=l == 0 and self.signed_input)
            stages.append(spec)
            src = [self._out_name(l, j) for j in range(W.shape[0])]
        return stages

    def exposed_names(self) -> set[str]:
        """The names a map of this net reads back (every light and plane height the lowering exposes): the set a
        pass must return, so a dropped readout fails the check instead of shrinking it."""
        return {name for s in self._stages() for name, _ in s.outputs}

    # --- persistence -------------------------------------------------------------------------------------------
    def save(self, path) -> None:
        """Write an .npz. The float model (and its activations) is stored only when every activation is one of the
        built-ins (relu, relu6, sigmoid, tanh, hardtanh/clamp, identity): a callable or a Staircase cannot be
        persisted, and then a loaded net has no float model (integer model, scales and input_shape are kept)."""
        arrs = dict(n_layers=self.n_layers, signed_input=self.signed_input)
        for l in range(self.n_layers):
            arrs |= {f"W{l}": self.weights[l], f"b{l}": self.biases[l], f"thr{l}": self.thresholds[l]}
        if self.in_scale is not None:
            arrs |= dict(in_scale=self.in_scale, out_scale=self.out_scale)
        if self.input_shape is not None:
            arrs["input_shape"] = np.array(self.input_shape, dtype=np.int64)
        specs = [a.spec for a in self.float_acts] if self.float_acts is not None else []
        if self.float_layers is not None and all(s is not None for s in specs):
            for l, (W, b) in enumerate(self.float_layers):
                arrs |= {f"Wf{l}": W, f"bf{l}": b}
            arrs["float_acts"] = np.array(json.dumps(specs))
        with open(path, "wb") as f:              # np.savez would append .npz to a str path without it
            np.savez(f, **arrs)

    @classmethod
    def load(cls, path) -> "MLP":
        with np.load(path) as z:
            n = int(z["n_layers"])
            net = cls([z[f"W{l}"] for l in range(n)], [z[f"b{l}"] for l in range(n)],
                      [z[f"thr{l}"] for l in range(n)], signed_input=bool(z["signed_input"]))
            if "in_scale" in z:
                net.in_scale, net.out_scale = z["in_scale"].astype(float), float(z["out_scale"])
            if "input_shape" in z:
                net.input_shape = tuple(int(i) for i in z["input_shape"])
            if "Wf0" in z:
                net.float_layers = [(z[f"Wf{l}"], z[f"bf{l}"]) for l in range(n)]
                specs = json.loads(str(z["float_acts"])) if "float_acts" in z else [["relu"]] * (n - 1) + [["identity"]]
                net.float_acts = [act_from_spec(s) for s in specs]
        return net

    # --- float interop -----------------------------------------------------------------------------------------
    @classmethod
    def from_numpy(cls, layers, calib_x, levels=8, *, input_shape=None, calibrate: str = "max",
                   equalise: bool | None = None, input_stair=None) -> "MLP":
        """Quantise the float MLP  f(x) = W_{n-1} act(... act(W_0 x + b_0) ...) + b_{n-1}  into an integer MLP.

        layers: [(W (out, in), b (out,)) | (W, b, activation), ...] floats (torch Linear layout). The activation after
        a layer is ReLU unless given: "relu", "identity" (merged into the next layer), "relu6", "sigmoid", "tanh",
        "hardtanh", "gelu", "gelu_tanh", "silu", ("clamp", lo, hi) / ("hardtanh", lo, hi), a monotone nondecreasing
        callable, or a `df.staircase` result (thresholds chosen by you); the last layer has none (DOOM returns its raw
        sums; apply a final sigmoid/softmax to `decode(y)` in Python). See `interop.staircase` for how an activation
        becomes thresholds.
        calib_x: (N, in) typical inputs, or (N, *input_shape) (flattened row-major; `input_shape` is kept on the net
        so `encode` / `float_forward` accept that shape). To feed a convolution use `interop.conv2d_matrix` /
        `conv1d_matrix` (exact dense matrix) or `from_torch`.
        levels: staircase levels K per sign, one int for all staircases, or one int per staircase (the input's, then
        one per hidden layer; a `Staircase` activation brings its own length). The result approximates f; DOOM is
        exact w.r.t. the integer model.
        calibrate: "max" (default; the numbers below) or "mse": the ReLU step of a layer (and the input step of a
        feature) is the a * max / K, a in 0.25..1, with the least squared staircase error on the calibration rows
        (the `step_size` rule of training/train_mnist.py), and the range of any other staircase is the quantile
        range with the least squared error. equalise: rescale the ReLU hidden units to equal 99th percentiles of
        their positive activations before quantising (the float function is unchanged); default: on iff
        calibrate == "mse". input_stair: (step per feature, levels, signed) fixes the input staircase (what
        `from_torch` takes from a `qat.InputStair`). See doomformer/quantize.py.

        * input, per feature i: level size D_i = max|calib_x[:, i]| / K, in_scale_i = D_i / 16,
          encode(x) = round(x / in_scale), thr_0 = round((k - 1/2) * 16), k = 1..K, so
          a_0 ~ clip(round(x / D), -K, K). signed_input iff calib_x has a negative entry (else relu input).
        * layer l maps activations a_l, which stand for the float values off_l + u_l * a_l (off_0 = 0, u_0 = D;
          after a hidden staircase off and u are that staircase's `lo` and `step`), to an integer h with float
          unit s: W_int = round(W_l u_l / s), b_int = round((b_l + W_l off_l) / s) (the offset of the previous
          staircase is folded into the bias). s starts at the float bound
          max(max|W_l u_l| / 255, max_rows(|b'| + K sum_j |W_l u_l|) / limit) and grows in 1 % steps until
          |W_int| <= 255 and, for every row,
          |b_int + shift| + sum_j |W_int[:, j]| max|a_l,j| <= 12,000 (hidden) or 30,000 (output): a worst case over ALL
          activations, so every input whose encoding fits +-12,000 is in range.
        * ReLU staircase: D_l = max(1, max over calib of h_l) / K in integer units, thr_l = round((k - 1/2) D_l)
          (made >= 1 and nondecreasing), i.e. a ~ clip(round(relu(h) / D_l), 0, K); u_l = D_l s, off_l = 0.
        * any other activation: a K-level `staircase(fn, (min h s, max h s), K)` over the calibrated pre-activation
          range (a given `Staircase` is used as is); its thresholds are divided by s and rounded; if that makes
          some < 1 the whole staircase is moved up by `shift` and `shift` is added to this layer's integer
          biases, so the integer hidden value (and the gate DOOM sees) is h + shift. off_l = staircase.lo,
          u_l = staircase.step. The approximation error of such a staircase is up to half a step of the output
          range per hidden unit (`staircase(...).max_error()`) plus rounding of the weights. A function that is not
          monotone (GELU, SiLU) is replaced by its monotone envelope first (`interop.monotone_envelope`): an extra
          error of half the depth of its dip (GELU 0.085, SiLU 0.14).
        * a `MaxPool` (an Act with a pool, from `from_torch`) after a staircase is computed by comparisons on the
          integer counts, exactly (module doc of doomformer/quantize.py); it adds layers (K times the pooled
          layer's lines).
        * output: decode(y) = y * out_scale with out_scale = s of the last layer (one scalar, so the argmax of
          the integer outputs is the argmax of the decoded ones).
        """
        return quantise(cls, layers, calib_x, levels, input_shape=input_shape, calibrate=calibrate, equalise=equalise,
                        input_stair=input_stair)

    @classmethod
    def from_torch(cls, model, calib_x, levels=8, *, input_shape=None, calibrate: str = "max",
                   equalise: bool | None = None) -> "MLP":
        """`from_numpy` for a torch nn.Sequential (nested Sequentials allowed) or a single supported module.

        Supported: Linear, Conv1d, Conv2d (lowered to exact dense linear layers, see `interop.conv2d_matrix`;
        zero padding only), Flatten, Identity, Dropout (must be in eval mode), BatchNorm1d/2d (eval mode: folded into
        the preceding Linear/Conv, or into the next layer when it follows an activation), AvgPool1d/2d and
        AdaptiveAvgPool1d/2d (an exact linear layer), MaxPool1d/2d next to a monotone activation (exact, by
        comparisons), and as activations ReLU, ReLU6, Hardtanh, Sigmoid, Tanh, GELU, SiLU (staircases, an
        approximation: see `from_numpy`; GELU and SiLU follow their monotone envelope) and the `doomformer.qat`
        modules (a model from `qat.fit` keeps its trained staircases and input quantiser; `levels` then only
        applies to staircases the model does not define). A layer without an activation after it is merged into the
        next one. The last layer cannot have an activation. LayerNorm/GroupNorm raise ValueError. calib_x:
        (N, *sample_shape) (a tensor or an array), e.g. (N, 1, 5, 5) for a Conv2d model; if it is flat give
        `input_shape`. `calibrate` and `equalise` are those of `from_numpy`.
        Same approximation caveat as from_numpy: the integer model approximates the torch model, DOOM computes the
        integer model exactly."""
        import torch

        if isinstance(calib_x, torch.Tensor):
            calib_x = calib_x.detach().cpu().double().numpy()
        cx = np.asarray(calib_x, dtype=np.float64)
        if cx.ndim < 2:
            raise ValueError("calib_x must be (N, *sample_shape)")
        sample = tuple(int(i) for i in input_shape) if input_shape is not None else tuple(cx.shape[1:])
        if int(np.prod(sample)) != int(np.prod(cx.shape[1:])):
            raise ValueError(f"input_shape {sample} does not match calib_x samples of shape {cx.shape[1:]}")
        layers, input_stair = torch_model(model, sample)
        return quantise(cls, layers, cx.reshape(len(cx), *sample), levels, input_shape=sample, calibrate=calibrate,
                        equalise=equalise, input_stair=input_stair)

    def _flat(self, x_float) -> np.ndarray:
        """x with the sample shape (input_shape) flattened row-major; flat x passes through."""
        x = np.asarray(x_float, dtype=np.float64)
        k = len(self.input_shape) if self.input_shape else 0
        if k > 1 and x.ndim >= k and x.shape[x.ndim - k:] == self.input_shape:
            x = x.reshape(x.shape[:x.ndim - k] + (-1,))
        return x

    def encode(self, x_float) -> np.ndarray:
        """Float inputs -> the integer inputs of this quantised net: round(x / in_scale). x may have the sample shape
        (e.g. (C, H, W) or (N, C, H, W)); the result is flat (n_in,) / (N, n_in)."""
        if self.in_scale is None:
            raise ValueError("encode needs a net made by from_numpy/from_torch")
        return np.rint(self._flat(x_float) / self.in_scale).astype(np.int64)

    def decode(self, y_int) -> np.ndarray:
        """Integer outputs -> floats: y * out_scale."""
        if self.out_scale is None:
            raise ValueError("decode needs a net made by from_numpy/from_torch")
        return np.asarray(y_int, dtype=np.float64) * self.out_scale

    def float_forward(self, x_float) -> np.ndarray:
        """The original float model (numpy) this net was quantised from, activations included (conv layers appear as
        their lowered matrices; for torch models this equals model(x) up to float rounding)."""
        if self.float_layers is None:
            raise ValueError("no float model attached (made by from_numpy/from_torch, or saved with built-in "
                             "activations only)")
        a = self._flat(x_float)
        acts = self.float_acts or [RELU] * (len(self.float_layers) - 1) + [IDENTITY]
        for (W, b), act in zip(self.float_layers, acts):
            a = act(a @ W.T + b)
        return a

    def predict(self, brain, x, encode: bool = False):
        """argmax over the outputs DOOM computed for x: `brain(x)` (verified against the integer reference, raises
        df.MismatchError on any difference) followed by numpy's argmax in PYTHON (first index wins ties). x: integer
        inputs as for `brain(x)` ((n_in,) -> int, (B, n_in) -> int array); encode=True first applies `self.encode`
        to float inputs. `brain` must have been compiled from this very net."""
        if getattr(brain, "net", None) is not self:
            raise ValueError("predict needs a brain compiled from this MLP (df.compile(net))")
        y = brain(self.encode(x) if encode else x)
        return int(np.argmax(y)) if y.ndim == 1 else np.argmax(y, axis=-1)

    def range_report(self, x_int) -> dict:
        """How much of DOOM's exact ranges the integer inputs x_int (n_in,) / (N, n_in) use, per layer (numpy only).

        input: peak |x| against the +-12,000 gate range. layers: one dict per weight layer (name h1.. / y) with
        partial_sum_peak and partial_sum_usage (worst |partial sum| over any crossing order against the +-30,000 int16
        accumulator), for hidden layers gate_peak and gate_usage (|h| against +-12,000: the value written into the
        next layer's gates) and top_level_fraction (share of staircase outputs sitting on the last step: that
        staircase is saturated), worst_case (max over rows of |b| + sum |W| * max activation: a bound on |h| and
        on every partial sum for ANY input) and rows_over (rows that `check` / DOOM would refuse because of this layer).
        warnings: one line per input or layer that exceeds a range (those rows raise RangeError before DOOM runs) or
        uses >= 90 % of it while other inputs could still leave it (worst_case above the range); a layer whose
        worst_case fits the range can never leave it, however close the data comes, so it is not reported."""
        x = np.atleast_2d(self._x(x_int))
        ref = self.reference(x)
        n = len(x)
        warnings = []

        def usage(name, peak, limit, what, worst=None):
            if worst is not None and worst <= limit:
                return
            if peak > limit:
                warnings.append(f"{name}: {what} reaches {peak} > +-{limit:,}: saturates the exact range, DOOM cannot "
                                "represent it (those inputs raise RangeError)")
            elif peak >= 0.9 * limit:
                warnings.append(f"{name}: {what} uses {100 * peak / limit:.0f}% of the +-{limit:,} range: other inputs "
                                "may be refused (RangeError)")

        peak_in = int(np.abs(x).max(initial=0))
        usage("input", peak_in, PORT_MAX, "|x| (written into gate floors)")
        layers = []
        for l, (W, b) in enumerate(zip(self.weights, self.biases)):
            last = l == self.n_layers - 1
            name = "y" if last else f"h{l + 1}"
            h = ref["y"] if last else ref["hidden"][l]
            ps = _partial_sum_bound(ref["activations"][l], W, b)
            worst = int((np.abs(b) + np.abs(W) @ np.full(W.shape[1], len(self.thresholds[l]))).max(initial=0))
            d = dict(layer=name, partial_sum_peak=int(ps.max(initial=0)),
                     partial_sum_usage=float(ps.max(initial=0)) / Z_MAX, worst_case=worst)
            over = (ps > Z_MAX).any(-1)
            usage(name, d["partial_sum_peak"], Z_MAX, "a partial sum", worst)
            if not last:
                gp = int(np.abs(h).max(initial=0))
                d |= dict(gate_peak=gp, gate_usage=gp / PORT_MAX,
                          top_level_fraction=float((ref["activations"][l + 1] == len(self.thresholds[l + 1])).mean()))
                over = over | (np.abs(h) > PORT_MAX).any(-1)
                usage(name, gp, PORT_MAX, "|h| (written into the next layer's gates)", worst)
            d["rows_over"] = int(np.count_nonzero(over))
            layers.append(d)
        return dict(n=n, input=dict(peak=peak_in, usage=peak_in / PORT_MAX), layers=layers, warnings=warnings)

    def quantization_report(self, x_float) -> dict:
        """How well decode(reference(encode(x))) approximates the float model on x_float (N, n_in) (or (N, *input_shape)):
        max/mean abs error, the float outputs' max |value| for scale, argmax agreement (if n_out > 1) and how many rows
        pass `check`; plus the `range_report` of the encoded inputs under "input_range" / "layers" / "warnings"
        (a warning appears when the input or a layer exceeds, or comes within 10 % of, the +-12,000 gate range or the
        +-30,000 accumulator range and some other input could leave it). Pure numpy; DOOM is not involved."""
        x = np.atleast_2d(self._flat(x_float))
        f = self.float_forward(x)
        xi = self.encode(x)
        q = self.decode(self.reference(xi)["y"])
        err = np.abs(q - f)
        ok = 0
        for row in xi:
            try:
                self.check(row)
                ok += 1
            except RangeError:
                pass
        rep = dict(n=len(x), max_abs_error=float(err.max()), mean_abs_error=float(err.mean()),
                   float_max_abs=float(np.abs(f).max()), in_range=ok)
        if self.n_out > 1:
            rep["argmax_agreement"] = float((q.argmax(1) == f.argmax(1)).mean())
        rr = self.range_report(xi)
        rep |= dict(input_range=rr["input"], layers=rr["layers"], warnings=rr["warnings"])
        return rep


# --- compile -------------------------------------------------------------------------------------------------------
def _kind(net) -> str:
    if isinstance(net, MLP):
        return "mlp"
    if isinstance(net, IntNet):
        return "spiking"
    if isinstance(net, QModel):
        return "transformer"
    raise TypeError(f"df.compile/plan take df.MLP, df.SpikingNet or df.Transformer, not {type(net).__name__}")


def _build(net, wad: Path, world=None, actions=None, layout_only: bool = False, guard: bool = False, opt=0,
           timing=None) -> TileMeta:
    kind = _kind(net)
    if kind == "mlp":
        stages = net._stages()
        consumed = {key.split(":", 1)[1] for s in stages for key in s.ports}
        readout = [name for s in stages for name, _ in s.outputs if name in consumed]
        return compile_pure(stages, wad, "mlp", readout=readout, world=world, actions=actions,
                            layout_only=layout_only, guard=guard, opt=opt, timing=timing)
    if kind == "spiking":
        if guard:
            raise ValueError("guard is supported for MLP and transformer maps, not spiking nets")
        if opt:
            raise ValueError("opt > 0 is supported for MLP and transformer maps, not spiking nets (their potentials "
                             "are state the schedule analysis does not model)")
        return build_spiking(net, wad, "spiking", world=world, actions=actions, layout_only=layout_only)
    return build_onemap(net, wad, world=world, actions=actions, layout_only=layout_only, guard=guard, opt=opt,
                        timing=timing)


def _check_actions(net, actions: dict | None):
    if not actions:
        return
    kind = _kind(net)
    if kind == "mlp":
        valid = set(net.value_names())
    elif kind == "spiking":
        valid = set(net.names)
    else:
        valid = {name for s in transformer_layers(net) for name, _ in s.outputs}
    for name, specials in actions.items():
        if name not in valid:
            raise KeyError(f"actions: {name!r} is not an output of this {kind} net "
                           f"(e.g. {sorted(valid)[:6]})")
        if not isinstance(specials, (list, tuple)) or not all(isinstance(s, dict) and "special" in s
                                                              for s in specials):
            raise TypeError(f"actions[{name!r}] must be a list of dicts with 'special' (see df.act)")


def _overlaps(boxes: np.ndarray, r) -> bool:
    return bool(((boxes[:, 0] < r[2]) & (r[0] < boxes[:, 2]) & (boxes[:, 1] < r[3]) & (r[1] < boxes[:, 3])).any())


def _world_hook(world: Callable[[MapBuilder], str]) -> Callable[[MapBuilder], str]:
    """Run the user's world builder and reject the three silent ways it can break the brain: no player start,
    overlapping sectors and a map wider or taller than MAX_SPAN (where ViZDoom misplaces things)."""

    def hook(m: MapBuilder) -> str:
        n_rects, n_things = len(m.rects), len(m.things)
        area = world_area(map_bounds(m))
        src = world(m)
        if not isinstance(src, str):
            raise TypeError("world(m) must return ACS source as a str (may be empty)")
        starts = [i for i, t in enumerate(m.things) if t["type"] == 1]
        if not starts or starts[-1] < n_things:
            raise ValueError("world must place the player start (m.thing(x, y, 1)); otherwise the last voodoo "
                             "doll becomes the player and the brain breaks")
        cx, cy = m.closet_origin
        k = len(m.wall_props)                     # the conveyor-scroller closet compile_pure adds next
        old = np.array([(r.x0, r.y0, r.x1, r.y1) for r in m.rects[:n_rects]]
                       + [(cx + 96 * k, cy, cx + 96 * k + 64, cy + 64)], dtype=np.int64)
        new = [(r.x0, r.y0, r.x1, r.y1) for r in m.rects[n_rects:]]
        for i, r in enumerate(new):
            if _overlaps(old, r) or (i and _overlaps(np.array(new[:i]), r)):
                raise ValueError(f"world rect {r} overlaps the brain, its closets or another world rect "
                                 f"(build inside the world area {area}, away from (-4096, -4096))")
        if new:
            boxes = np.array(new, dtype=np.int64)
            lo = np.minimum(old[:, :2].min(0), np.minimum.accumulate(boxes[:, :2]))
            hi = np.maximum(old[:, 2:].max(0), np.maximum.accumulate(boxes[:, 2:]))
            over = np.flatnonzero(((hi - lo) > MAX_SPAN).any(1))
            if over.size:
                i = int(over[0])
                w, h = (hi - lo)[i].tolist()
                raise ValueError(f"world rect {new[i]} makes the map span {w} x {h} units; beyond {MAX_SPAN} per "
                                 "axis ViZDoom's 16.16 coordinate differences wrap and things land in the wrong "
                                 "sector (the player rides the brain's conveyors; docs/engine_limits.md). Build the "
                                 f"world inside {area} = [x0, y0, x1, y1] (df.plan(net)['map']['world_area'])")
        return src

    return hook


def plan(net, *, mode: str = "auto", workers: int = 1, opt: int = 0) -> dict:
    """Report both map and tile geometry without compiling ACS or starting games.

    Top-level dolls/sectors/fits/span retain their historical one-map meaning. `map`
    and `tiles` hold per-mode counts; `mode` is the selected execution mode.
    `map` also holds the brain map's bounds, span [width, height] and world_area
    (where a world hook may build); fits=False (with a reason) when the map would
    span more than 32,000 units per axis (docs/engine_limits.md). `tiles` holds
    the largest tile's span and fits=False (with a reason) when any tile map
    would span more than 32,000 units or hold 20,000+ dolls. When the selected
    mode cannot be built, the report carries a top-level `reason` (in auto:
    neither one map nor tiles fit) and df.compile refuses it with ValueError.
    Auto builds one map whenever the map fits and selects tiles (MLP/Transformer) only when it does not and the
    tiles do; estimated_settle_tics is a geometry estimate, not a wall-time prediction. Stateful spiking nets remain
    map-only. opt: the one-map schedule level (see compile); estimated_settle_tics is for that schedule.
    """
    from .tiled import AUTO_SETTLE_TICS, tile_plan
    if mode not in ("auto", "map", "tiles"):
        raise ValueError("mode must be 'auto', 'map' or 'tiles'")
    if isinstance(workers, bool) or not isinstance(workers, int) or workers < 1:
        raise ValueError("workers must be a positive integer")
    kind = _kind(net)
    if mode == "tiles" and kind == "spiking":
        raise ValueError("spiking nets require mode='map' (state stays in one game)")
    try:
        meta = _build(net, Path("layout_only.wad"), layout_only=True, opt=opt)
        layout = meta.layout
        map_option = dict(dolls=meta.n_dolls, sectors=meta.n_sectors, fits=bool(layout["fits"]),
                          span=layout["span"], bounds=layout["bounds"], world_area=layout["world_area"],
                          tiles=1, estimated_settle_tics=layout["estimated_settle_tics"])
        from .cost import settle_estimate
        map_option.update(settle_estimate(layout))     # calibrated_tics, tics_band, tics_error_pct (doomformer/cost.py)
        if not layout["fits"]:
            map_option["reason"] = (f"the map would span {layout['span'][0]} x {layout['span'][1]} units, beyond "
                                    f"the {MAX_SPAN}-unit limit per axis (docs/engine_limits.md)")
    except ValueError as exc:
        if "corridor width" not in str(exc):
            raise
        map_option = dict(dolls=None, sectors=None, fits=False, span=None, bounds=None, world_area=None, tiles=1,
                          estimated_settle_tics=None, reason=str(exc))
    tiles = tile_plan(net) if kind != "spiking" else None
    selected = mode
    if mode == "auto":                    # map first: in one map nothing but inputs and outputs crosses into Python
        selected = "tiles" if (not map_option["fits"] and tiles is not None and tiles["fits"]) else "map"
    report = dict(dolls=map_option["dolls"], sectors=map_option["sectors"], fits=map_option["fits"],
                  span=map_option["span"], mode=selected, workers=workers, map=map_option, tiles=tiles,
                  auto_settle_tics=AUTO_SETTLE_TICS)
    from .cost import annotate
    annotate(report, kind)                # predicted_seconds per mode, `recommended` (doomformer/cost.py)
    option = map_option if selected == "map" else tiles
    if not option["fits"]:                # the selected mode cannot be built: say why (both reasons in auto)
        report["reason"] = (f"fits neither one map ({map_option['reason']}) nor tiles ({tiles['reason']})"
                            if mode == "auto" and tiles is not None else option["reason"])
    return report


def compile(net, wad=None, *, world: Callable[[MapBuilder], str] | None = None,
            actions: dict[str, list[dict]] | None = None, resolution=None, configure=None,
            mode: str = "auto", workers: int = 1, backend: str = "local", address: str | None = None,
            guard: bool = False, limits: bool = True, opt: int = 0) -> "Brain":
    """Build `net` into one map or verified operation tiles and return its Brain.

    wad: output path (a .json with the map's metadata is written next to it); default: a content-addressed file
    in ~/.cache/doomformer/brains (DOOMFORMER_CACHE overrides the root). world: see the module docstring.
    actions: {output_name: [line specials]} (df.act). For an MLP the names are ``y{o}``/``h{l}_{j}``/``-h{l}_{j}``,
    for a spiking net neuron names (they fire once per spike), for a transformer onemap value names such as
    ``logits:t:j``. resolution/configure are passed to runtime.TileRunner when the game starts (lazily, on first
    use of `brain.runner`).

    mode='auto' builds one map whenever the map fits (plan's map 'fits') and falls back to tiles only when it
    does not (one-map runs can be slow: see plan's estimated_settle_tics); 'map'
    always requests the original one-game implementation; 'tiles' uses Python's
    integer-copy bus between games, with at most one live game per worker.
    Tiles support MLP and Transformer, not stateful spiking nets. World/actions
    and runner configuration require a map (auto preserves map mode when any
    such hooks are supplied). In tiles mode wad names a directory; a .wad suffix
    becomes .tiles. backend="ray" runs the tile games as Ray actors (one game each) on the cluster at `address`
    (default: a running cluster, else a local Ray); it needs mode="tiles" (for many map passes use df.run_many).
    guard=True adds a readout guard (one map only): every read first fills the USER slots with a fresh nonce, so a
    readout page that never runs fails loudly instead of repeating the last pass's values (compile_pure's `guard`).
    limits=True: compile checks the exact range before building and raises RangeError, limits=False skips these checks
    (a pass out of range then fails at run time as a DOOM-vs-reference mismatch, or is refused by net.check). Transformers:
    the parameters and `limit_sequences(net)` against intref.check_limits (sampled sequences, not every token sequence).
    One-map MLPs: the plane-move bound (planes.py, +-14,018) for the worst case over all inputs (`MLP.plane_errors`);
    net.check(x) applies it per input, always. Spiking nets: the state is not bounded at compile time, every
    `SpikingBrain.step` checks the step it is about to run. Tile mode has no doll-moved planes and skips the bound.
    opt (one map; MLP and transformer): 0 is the original schedule, byte for byte; 1 adds the schedule relaxations
    that hold for every input (mutually exclusive dolls and light targets share waves, graph-coloured waves); 2 adds
    the tuned timing constants of doomformer/tuning.json. Tiles have no waves and ignore it. docs/scheduling.md.
    See docs/scaling.md for counts, measurements and limits."""
    kind = _kind(net)
    if limits and kind == "transformer":
        errs = check_limits(net, limit_sequences(net), planes=mode == "map")
        if errs:
            raise RangeError("transformer outside the exact range (limits=False skips this check): "
                             + "; ".join(errs[:6]) + (f" (+{len(errs) - 6} more)" if len(errs) > 6 else ""))
    if mode not in ("auto", "map", "tiles"):
        raise ValueError("mode must be 'auto', 'map' or 'tiles'")
    if isinstance(workers, bool) or not isinstance(workers, int) or workers < 1:
        raise ValueError("workers must be a positive integer")
    if backend not in ("local", "ray"):
        raise ValueError("backend must be 'local' or 'ray'")
    if backend == "ray" and mode != "tiles":
        raise ValueError("backend='ray' requires mode='tiles' (one map is one game; df.run_many spreads map passes "
                         "over many games)")
    if backend == "ray":
        from .fleet import ray_module
        ray_module()                                      # ImportError with the install command, before building
    map_hooks = (world is not None or actions is not None or resolution is not None or configure is not None
                 or guard)
    if mode == "auto" and not map_hooks:
        report = plan(net, workers=workers, opt=opt)
        if "reason" in report:
            raise ValueError(f"{kind}: {report['reason']}")
        selected = report["mode"]
    else:
        selected = "map" if mode == "auto" else mode
    if selected == "tiles":
        if kind == "spiking":
            raise ValueError("spiking nets require mode='map' (state stays in one game)")
        if map_hooks:
            raise ValueError("world/actions/resolution/configure/guard require mode='map', not multiple tile games")
        from .tiled import compile_tiles
        return compile_tiles(net, wad, workers=workers, backend=backend, address=address)
    if limits and kind == "transformer" and mode != "map":     # mode "map" was checked with the plane bound above
        errs = check_limits(net, limit_sequences(net), planes=True)
        if errs:
            raise RangeError("transformer outside the exact range (limits=False skips this check): "
                             + "; ".join(errs[:6]) + (f" (+{len(errs) - 6} more)" if len(errs) > 6 else ""))
    if limits and kind == "mlp":
        errs = net.plane_errors()
        if errs:
            raise RangeError("MLP outside the one-map plane range for some inputs (limits=False skips this check; "
                             "net.check(x) still refuses the inputs that violate it; mode='tiles' has no planes): "
                             + "; ".join(errs[:6]) + (f" (+{len(errs) - 6} more)" if len(errs) > 6 else ""))
    _check_actions(net, actions)
    hook = _world_hook(world) if world is not None else None

    def build(path: Path) -> TileMeta:
        return _build(net, path, world=hook, actions=actions, guard=guard, opt=opt)

    if wad is not None:
        path = Path(wad)
        path.parent.mkdir(parents=True, exist_ok=True)
        meta = build(path)
    else:
        d = cache_dir() / "brains"
        d.mkdir(parents=True, exist_ok=True)
        tmp = d / f".tmp-{os.getpid()}-{uuid.uuid4().hex[:8]}.wad"
        try:
            meta = build(tmp)
            digest = hashlib.sha256(tmp.read_bytes() + tmp.with_suffix(".json").read_bytes()).hexdigest()[:16]
            path = d / f"{_kind(net)}-{digest}.wad"
            os.replace(tmp.with_suffix(".json"), path.with_suffix(".json"))
            os.replace(tmp, path)
        finally:
            for p in (tmp, tmp.with_suffix(".json")):
                p.unlink(missing_ok=True)
    brain = {"mlp": MLPBrain, "spiking": SpikingBrain, "transformer": TransformerBrain}[_kind(net)](
        net, path, meta, resolution=resolution, configure=configure)
    brain.limits = limits
    brain.info.update(mode="map", workers=1, tiles=1)
    return brain


class Brain:
    """A compiled map plus (lazily) a headless ViZDoom game running it. Use as a context manager or call close().

    info: kind, wad, dolls, sectors, runs, tics (game tics of the last run), checked (values compared last run).
    runner: the live runtime.TileRunner (runner.game is the vizdoom.DoomGame); started on first use. Runs reuse
    the loaded map: ACS script 3 sends the dolls home and restores the brain's values between runs; the world
    (things, the player, sectors you built) is never reset by the brain."""

    kind = "brain"

    def __init__(self, net, wad: Path, meta: TileMeta, resolution=None, configure=None):
        self.net, self.meta = net, meta
        self._runner, self._resolution, self._configure = None, resolution, configure
        self.info = dict(kind=self.kind, wad=str(wad), dolls=meta.n_dolls, sectors=meta.n_sectors, runs=0,
                         tics=None, checked=None)

    @property
    def wad(self) -> Path:
        return Path(self.info["wad"])

    @property
    def runner(self):
        if self._runner is None:
            from .runtime import TileRunner
            kw = {} if self._resolution is None else dict(resolution=self._resolution)
            self._runner = TileRunner(self.wad, configure=self._configure, recycle=True, **kw)
        return self._runner

    def _run(self, ports: dict[str, int]) -> dict[str, int]:
        got, tics, _, _ = self.runner.run(ports)
        self.info["runs"] += 1
        self.info["tics"] = tics
        return got

    def run_script(self, script: int, *args: int, read: int = 0) -> list[int]:
        """Run ACS script `script` (e.g. a world readout writing o1..oN), advance one tic, return USER1..USER{read}."""
        import vizdoom as vzd
        g = self.runner.game
        g.send_game_command(f"puke -{int(script)} " + " ".join(str(int(a)) for a in args))
        g.advance_action(1)
        return [int(g.get_game_variable(getattr(vzd.GameVariable, f"USER{i}"))) for i in range(1, read + 1)]

    def close(self):
        if self._runner is not None:
            self._runner.close()
            self._runner = None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def __repr__(self):
        return f"<{type(self).__name__} {self.info['dolls']} dolls, {self.info['sectors']} sectors, {self.wad}>"

    def record(self, inputs, path, view: str = "topdown", every: int = 8, fps: int = 30, **kw):
        """Run `inputs` through this brain as `brain(x)` / `brain.step(x)` / `brain.generate(p)` would (every pass is
        verified against the integer reference) and film it to a .gif or .mp4: see doomformer/record.py and
        docs/recording.md. Returns a `doomformer.record.Recording`; frames are drawn from state that is only read."""
        from .record import record
        return record(self, inputs, path, view, every, fps, **kw)


class MLPBrain(Brain):
    """brain(x) -> y: one forward pass in DOOM, every exposed value checked against net.reference(x)."""

    kind = "mlp"

    _exposed: set[str] | None = None                       # names a pass must return (MLP.exposed_names)

    def _ports(self, x: np.ndarray) -> dict[str, int]:
        ports = {}
        for key in self.meta.ports:
            name = key.split(":", 1)[1]                    # "x{i}" or "-x{i}"
            v = int(x[int(name.lstrip("-")[1:])])
            ports[name] = -v if name.startswith("-") else v
        return ports

    def run(self, x, check: bool = False) -> dict[str, int]:
        """One pass; returns the raw readback {name: int}: lights y{o} and h{l}_{j}, gate floors -h{l}_{j}.
        check=True compares every value with net.reference(x) and raises MismatchError on any difference."""
        x = self.net._x(x)
        if x.ndim != 1:
            raise ValueError("run takes one input vector")
        self.net.check(x)
        got = self._run(self._ports(x))
        if check:
            ref = self.net.reference(x)
            expected = self.net.exposed_names() if self._exposed is None else self._exposed
            self._exposed = expected
            if set(got) != expected:
                raise ReadoutError(f"MLP pass x={x.tolist()}: the readback names differ from the lowering's "
                                   f"(missing {sorted(expected - set(got))[:6]}, unexpected "
                                   f"{sorted(set(got) - expected)[:6]})")
            bad = [(n, v, self.net._expected(n, ref)) for n, v in got.items() if v != self.net._expected(n, ref)]
            self.info["checked"] = len(got)
            if bad:
                raise MismatchError(f"MLP pass x={x.tolist()}", bad)
        return got

    def __call__(self, x) -> np.ndarray:
        x = self.net._x(x)
        if x.ndim == 2:
            return np.stack([self(row) for row in x])
        got = self.run(x, check=True)
        return np.array([got[f"y{o}"] for o in range(self.net.n_out)], dtype=np.int64)


class SpikingBrain(Brain):
    """brain.step(x) -> (v, s): one spiking time step in DOOM (potentials persist in the map as plane heights),
    every spike and potential checked against SpikingNet.step. brain.v is the reference state."""

    kind = "spiking"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.v = np.zeros(self.net.n, dtype=np.int64)

    def step(self, x_spikes) -> tuple[np.ndarray, np.ndarray]:
        x = _ints(x_spikes, "x_spikes", 1)
        if x.shape != (len(self.net.inputs),) or ((x != 0) & (x != 1)).any():
            raise ValueError(f"x_spikes must be {len(self.net.inputs)} values in {{0, 1}}")
        try:
            v_ref, s_ref = self.net.step(self.v, x)
        except OverflowError as e:
            raise RangeError(str(e)) from e
        if errs := spiking_step_errors(self.net, self.v, x):
            raise RangeError("; ".join(errs))
        got = self._run(step_ports(x, self.meta))
        v, s = read_step(got, self.net.n)
        ref = {}                                          # every readback: spikes s, their twins -s, potentials -v
        for i in range(self.net.n):
            ref |= {f"s{i}": int(s_ref[i]), f"-s{i}": -int(s_ref[i]), f"-v{i}": -int(v_ref[i])}
        bad = [(n, val, ref[n]) for n, val in got.items() if val != ref[n]]
        self.v = v_ref
        self.info["checked"] = len(got)
        if bad:
            raise MismatchError(f"spiking step {self.info['runs'] - 1}", bad)
        return v, s


class TransformerBrain(Brain):
    """The whole integer transformer (embedding, attention, MLP, unembedding) in one map (onemap). Python writes the
    token ids and reads every exposed value; sampling the next token happens in Python (intref.choose).
    `generate` adds up info["checked"] over its passes into info["checked_total"] (also in tiles mode)."""

    kind = "transformer"

    _exposed: set[str] | None = None                       # names a pass must return (onemap.exposed_names)
    limits = True                                          # df.compile(limits=False) turns the per-pass range check off

    def logits(self, tokens) -> np.ndarray:
        """Logits [len(tokens), vocab] from DOOM; every exposed preactivation, score and logit is compared with
        intref.forward (MismatchError on any difference). Before DOOM runs, this sequence is range-checked
        (intref.check_limits with the plane-move bound, RangeError): `compile` checked only kit.limit_sequences."""
        tokens = [int(t) for t in tokens]
        if self.limits and (errs := check_limits(self.net, [tokens], planes=True)):
            raise RangeError("; ".join(errs[:6]) + (f" (+{len(errs) - 6} more)" if len(errs) > 6 else ""))
        ports = token_ports(self.net, tokens)
        trace = {}
        expected = forward(self.net, tokens, trace)
        got = self._run(ports)
        if self._exposed is None:
            self._exposed = exposed_names(self.net)
        if set(got) != self._exposed:
            raise ReadoutError(f"transformer pass {tokens}: the readback names differ from the lowering's (missing "
                               f"{sorted(self._exposed - set(got))[:6]}, unexpected "
                               f"{sorted(set(got) - self._exposed)[:6]})")
        try:
            self.info["checked"] = compare_trace(got, trace, len(tokens))
        except AssertionError as e:                       # compare_trace: "name: DOOM a, intref b"
            name, rest = str(e).split(": DOOM ", 1)
            doom, ref = rest.split(", intref ")
            raise MismatchError(f"transformer pass {tokens}", [(name, int(doom), int(ref))]) from e
        lg = logits_from_readout(self.net, got, len(tokens))
        if not np.array_equal(lg, expected):
            bad = [(f"logits:{t}:{j}", int(lg[t, j]), int(expected[t, j])) for t, j in zip(*np.nonzero(lg != expected))]
            raise MismatchError(f"transformer logits {tokens}", bad)
        return lg

    def generate(self, prompt: str = "", temperature: float = 0.0, seed: int = 0, max_new: int | None = None) -> str:
        """intref.generate with every step's logits computed (and checked) by DOOM; same sampling and RNG."""
        m = self.net
        bad = [c for c in prompt if c not in m.vocab]
        if bad:
            raise ValueError(f"prompt characters {bad} not in the vocabulary {m.vocab!r}")
        toks = [0] + [m.vocab.index(c) for c in prompt]
        rng = np.random.default_rng(seed)
        max_new = max_new if max_new is not None else m.T - len(toks)
        for _ in range(max_new):
            if len(toks) >= m.T:
                break
            lg = self.logits(toks)
            self.info["checked_total"] = self.info.get("checked_total", 0) + self.info["checked"]
            nxt = choose(lg[-1], temperature, rng)
            if nxt == 0:
                break
            toks.append(nxt)
        return "".join(m.vocab[t] for t in toks[1:])


# --- play ----------------------------------------------------------------------------------------------------------
def play(wad_or_brain, *, resolution=None) -> None:
    """Open the map in a visible ViZDoom window in spectator mode (you walk around; nothing is checked). Needs a
    display. The brain stays idle (its start barriers are closed) unless you start it yourself."""
    if isinstance(wad_or_brain, Brain):
        wad = wad_or_brain.wad
    elif isinstance(wad_or_brain, (str, os.PathLike)):
        wad = Path(wad_or_brain)
    else:
        raise TypeError(f"play takes a .wad path or a Brain, not {type(wad_or_brain).__name__}")
    if not wad.is_file():
        raise FileNotFoundError(f"no such file: {wad}")
    if sys.platform.startswith("linux") and not (os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")):
        raise RuntimeError("df.play needs a display (no DISPLAY or WAYLAND_DISPLAY set); run it on a desktop, "
                           "or use brain(...) headless")
    import vizdoom as vzd

    from .runtime import FREEDOOM2, init_game
    g = vzd.DoomGame()
    g.set_doom_game_path(FREEDOOM2)
    g.set_doom_scenario_path(str(wad))
    g.set_doom_map("MAP01")
    g.set_screen_resolution(resolution or vzd.ScreenResolution.RES_800X600)
    g.set_window_visible(True)
    g.set_mode(vzd.Mode.SPECTATOR)
    g.set_episode_timeout(10**9)
    init_game(g)
    try:
        while not g.is_episode_finished():
            g.advance_action()
    except (vzd.ViZDoomIsNotRunningException, vzd.ViZDoomUnexpectedExitException):   # the window was closed
        pass
    finally:
        g.close()

