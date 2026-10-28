# Float-model interop: convolutions, pooling, BatchNorm, activations, quantisation-aware training

How to get a *float* model (PyTorch or NumPy) into DOOM, what is exact on the way, what is approximated, and who
does what. Code: [`doomformer/interop.py`](../doomformer/interop.py) (walking torch modules, staircases, pooling
windows), [`doomformer/quantize.py`](../doomformer/quantize.py) (the quantiser: MSE steps, equalise, max pool),
[`doomformer/qat.py`](../doomformer/qat.py) (quantisation-aware training) and the `MLP` class in
[`doomformer/kit.py`](../doomformer/kit.py); example [`examples/11_conv_in_doom.py`](../examples/11_conv_in_doom.py);
tests [`tests/test_interop.py`](../tests/test_interop.py), [`tests/test_quant.py`](../tests/test_quant.py); measured
results [`models/quant/EVAL.md`](../models/quant/EVAL.md).

> **What is real here.** Everything on this page happens in Python *before* DOOM sees a number: the conv layers are
> rewritten as linear layers, the activations are replaced by staircases, the weights are rounded to integers. The
> result is an integer `df.MLP`. DOOM then computes that **integer** model, and every number DOOM exposes (outputs and
> every hidden value) is read back and compared with an independent NumPy integer reference with tolerance 0
> (`df.MismatchError` otherwise). DOOM does **not** compute your float model: the integer model is an approximation
> of it, and the quantisation report measures how good. The final `argmax` of `MLP.predict` is taken by Python.

## Who does what

| step | who |
|---|---|
| walk the torch module / take your NumPy layers, lower convs to matrices, pick staircase thresholds, round weights | Python (`interop.py`, `MLP.from_numpy`) |
| input quantisation (`net.encode`), writing the integers into the input gates | Python |
| staircase activations (counting passed gates), every weighted sum, every layer chaining | **DOOM** (voodoo dolls, gate sectors, line specials; see [BUILD_YOUR_OWN](BUILD_YOUR_OWN.md#concepts)) |
| ACS level scripts | move the dolls home between passes and copy sector heights/lights into `USER` variables for readback; they compute no network arithmetic |
| compare with the integer reference | Python (NumPy integers, independent of the map) |
| argmax of the outputs (`MLP.predict`), decoding to floats (`net.decode`) | Python |

## What is exact and what is not

| piece | status |
|---|---|
| conv (1-D and 2-D, stride, zero padding, dilation, groups) rewritten as a linear layer | **exact in float**: the matrix applied to the flattened input equals `torch.nn.Conv*` (tested for 13 configurations against float64 torch, difference <= 1e-12, i.e. rounding of the sums). Zero padding is absent taps, `padding_mode` other than `zeros` is refused. |
| rounding the (lowered) weights to integers, scaling | **approximation** (per-layer scale `s`, like every Linear layer) |
| ReLU as a K-level staircase | **approximation** (levels are cut calibrated to your data) |
| ReLU6 / Hardtanh / clamp / Sigmoid / Tanh / any monotone function as a K-level staircase | **approximation**, up to half an output step per unit for inputs inside the calibrated range; saturation (constant) outside it |
| GELU / SiLU as a K-level staircase | **approximation**: the function is not monotone, so the staircase follows its monotone envelope (the best nondecreasing uniform fit on the calibrated range), which is off by half the depth of the dip (GELU 0.085, SiLU 0.14 in output units) on top of the half-step above |
| BatchNorm (eval) folded into a layer, AvgPool as a matrix | **exact in float** (1e-9 against float64 torch); only the rounding of the folded weights is an approximation |
| MaxPool after a staircase | **exact**: the integer net computes the maximum of the staircase counts (`tests/test_quant.py` checks it against a NumPy window maximum and against torch's own `MaxPool2d` on the counts); the float model it approximates is the same up to the staircase |
| `calibrate="mse"`, `equalise` | change which approximation is made (the step with the least squared error; rescaled units), not whether |
| DOOM running the integer model | **exact**, checked per pass, tolerance 0 |
| `predict` = `argmax(brain(x))` | Python's `numpy.argmax` of DOOM's integers (first index wins ties) |

## `MLP.from_torch(model, calib_x, levels=8, *, input_shape=None, calibrate="max", equalise=None)`

`model`: an `nn.Sequential` (nested `Sequential`s are fine) or a single supported module.

| module | handling |
|---|---|
| `Linear` | a layer |
| `Conv1d`, `Conv2d` | lowered to a dense (mostly zero) layer with `interop.conv1d_matrix` / `conv2d_matrix`; `stride`, `padding` (int, pair, `"same"`, `"valid"`), `dilation`, `groups`, `bias=False`; `padding_mode='zeros'` only |
| `Flatten` | row-major flatten, default `start_dim=1, end_dim=-1` only (this is the ordering the lowered matrices use) |
| `Identity`, `Dropout` | skipped; `Dropout` must be in `eval()` mode (or `p=0`), else `ValueError` (silent mis-quantisation is worse than an error) |
| `ReLU`, `ReLU6`, `Hardtanh(min, max)`, `Sigmoid`, `Tanh`, `GELU`, `SiLU` | staircases, see below (GELU/SiLU follow their monotone envelope) |
| `BatchNorm1d`, `BatchNorm2d` (eval mode) | folded: into the preceding Linear/Conv/AvgPool rows (`a = gamma / sqrt(var + eps)`, `c = beta - mean a`), or, when it follows an activation, into the columns and bias of the next layer (also for a padded conv, because padding taps are absent from the lowered matrix). Exact in float (tested to 1e-9 against float64 torch). `track_running_stats=False` and training mode are refused. |
| `AvgPool1d`, `AvgPool2d`, `AdaptiveAvgPool1d`, `AdaptiveAvgPool2d` | an exact linear layer (a dense averaging matrix, `count_include_pad`, `divisor_override`, padding, stride), merged into the next layer like any layer without an activation |
| `MaxPool1d`, `MaxPool2d` (kernel, stride, padding, dilation) next to a monotone activation, before or after it | **exact on the staircase counts**, by comparisons: see "MaxPool" below. `ceil_mode` and `return_indices` are refused. |
| `qat.InputStair`, `qat.Stair` | the input quantiser and staircases a `qat.fit` model was trained with: see "Quantisation-aware training" |

A layer without an activation after it is merged with the next layer (`W2 (W1 x + b1) + b2`, exact in float): a
`Conv -> Flatten -> Linear` stack is two layers. The *last* layer cannot have an activation (DOOM returns its raw
sums; apply `sigmoid`/`softmax` to `net.decode(y)` in Python). Anything else raises `ValueError` naming the module
and why: `LayerNorm`/`GroupNorm`/`InstanceNorm` (their mean and variance are computed per input, so they are not a
fixed affine map; the integer transformers have their own RMSNorm), `MaxPool` with no staircase next to it (a max
over linear outputs is not a staircase of a sum), `BatchNorm` in training mode, `ceil_mode` pools.

`calib_x` is `(N, *sample_shape)`, e.g. `(N, 1, 5, 5)`, or flat with `input_shape=(1, 5, 5)`. The net remembers
`input_shape`, so `net.encode(x)` and `net.float_forward(x)` accept `(C, H, W)` / `(N, C, H, W)` and the integer
inputs of the brain are the flattened `(n_in,)` vectors.

`levels`: one int for every staircase, or one int per staircase (the input's first, then one per hidden layer of the
float model, i.e. one per activation: a `Conv -> ReLU -> MaxPool` counts once). Every level of every unit is a voodoo
doll, so levels multiply the size of the map directly: a binary glyph needs `levels=[1, ...]` for its input. The
input's levels apply to every pixel: 8x8 digits with values 0..16 are lossless at `levels=[16, K, ...]` and lose
information at `levels=K` (measured below).

`MLP.from_numpy(layers, calib_x, levels=8, *, input_shape=None, calibrate="max", equalise=None, input_stair=None)` is the
same for float matrices:
`layers = [(W, b), (W, b, activation), ...]`; the default activation is ReLU between layers. `activation` is one of
`"relu"`, `"identity"`, `"relu6"`, `"sigmoid"`, `"tanh"`, `"hardtanh"`, `"gelu"`, `"gelu_tanh"`, `"silu"`,
`("clamp", lo, hi)` / `("hardtanh", lo, hi)`, a vectorised monotone nondecreasing callable (checked on 257 points,
`ValueError` otherwise), or a `df.staircase(...)` result. To feed a convolution use
`interop.conv2d_matrix(weight, bias, in_shape, stride, padding, dilation, groups)` -> `(M, b, out_shape)` (and
`conv1d_matrix`).

## Activations as staircases: `df.staircase`

DOOM's activation is a count: lane `k` passes iff `z >= thr[k]`, so `adc_relu(z, thr) = #{k : z >= thr[k]}`, a
nondecreasing step function with K steps. `df.staircase(fn, x_range, levels)` chooses K thresholds for any monotone
nondecreasing `fn`:

```python
st = df.staircase(np.tanh, (-3, 3), 8)        # a Staircase: ndarray of 8 float thresholds
st.lo, st.step                                # tanh(-3), (tanh(3) - tanh(-3)) / 8
st.value(z)                                   # lo + step * counts(z): what the staircase stands for
st.counts(z)                                  # #{k : z >= thr[k]}: what DOOM's gates produce
st.max_error()                                # max |fn - value| on x_range
t, shift = st.int_thresholds(scale)           # integer thresholds for h = z / scale, plus the bias shift to make them >= 1
```

The output range `[fn(lo), fn(hi)]` is cut into K equal steps; threshold `k` is the smallest `z` (by bisection) where
`fn` reaches the middle of step `k`. Hence the staircase is within half a step of `fn` on `x_range` (measured
`max_error`, which matches `step / 2` to the printed digits):

| K | tanh on (-3, 3) | sigmoid on (-6, 6) | relu6 on (-2, 9) |
|---|---|---|---|
| 2 | 0.498 | 0.249 | 1.499 |
| 4 | 0.249 | 0.124 | 0.750 |
| 8 | 0.124 | 0.062 | 0.375 |
| 16 | 0.062 | 0.031 | 0.188 |

Inside `from_numpy`, the staircase is placed over the calibrated pre-activation range (the range your calibration data
produces in the integer model), unless you pass a `Staircase` (then yours is used, with your own `x_range`).
Its output is a count `a` standing for the float value `lo + step * a`; the next layer's weights absorb `step` and its
bias absorbs `lo` (`b + W @ lo`), so DOOM never has to know about the offset. For ReLU, `lo = 0` and the range is
`0..max h`.

**The shift.** DOOM's thresholds are positive. Sigmoid/Tanh/Hardtanh thresholds start below zero, so the quantiser
moves the whole staircase up by `shift` integer units and adds `shift` to that layer's integer biases: the hidden value
DOOM stores in the gate (and `reference` returns, and `brain` compares) is `h + shift`, not `h`. This only changes
the offset of a quantity nobody reads except the next layer's gates.

Errors accumulate through layers; measured on a random 3-layer float net (3-6-5-3, tanh in layer 2, first activation as
named, random seed 7, 300 calibration rows, mean abs error of decoded outputs; max error in parentheses, float outputs reach
about 1.3-2.4): `levels=[8, 12, 12]` gives 0.067-0.12 (0.28-0.66); `[16, 24, 24]` gives 0.036-0.059 (0.16-0.30), i.e. the error
roughly halves when the levels double. Use `quantization_report` on your own data before spending DOOM time.

## Calibration: `calibrate="max"` and `"mse"`, `equalise`

`from_numpy` / `from_torch` take `calibrate="max"` (default, unchanged: the integer nets and hashes of v0.5.0) or
`"mse"`, and `equalise=None` (on exactly when `calibrate == "mse"`).

* **Max** sets a ReLU layer's staircase step to `max(h) / K` over the calibration rows and an input feature's step to
  `max|x| / K`. One rare large value stretches every level.
* **MSE** picks the step `a * max / K`, `a` in 0.25..1 (61 steps), with the least squared error between `relu(h)` and
  the staircase `clip(round(relu(h) / D), 0, K) D` on the calibration rows: the `step_size` rule of
  [`training/train_mnist.py`](../training/train_mnist.py), now in the library (`quantize.relu_step`). The same search
  runs per input feature. Sigmoid, Tanh, GELU and the other `fn` staircases search the clipped range of the
  staircase over the quantiles `(p, 1 - p)`, `p` in {0, 0.05 %, 0.1 %, 0.25 %, 0.5 %, 1 %, 2 %, 5 %}, scored the same way
  (`quantize.fit_staircase`).
* **Equalise** rescales every ReLU hidden unit (incoming weights and bias times `c`, outgoing weights divided by
  `c`, `c` within 1/4..4) towards the same 99th percentile of its positive activations on the calibration rows. The
  float function is unchanged (ReLU is positively homogeneous; tested to 1e-9), but a layer has ONE staircase step, and
  equal ranges make it fit every unit. It is skipped for layers that carry a max pool and for staircases you fixed.

Measured effect, 8x8 digits, uniform `levels=K`, test accuracy in percent, three seeds, from
[`models/quant/EVAL.md`](../models/quant/EVAL.md) (`python training/quant_digits.py --aggregate`):

| net, K = all levels | float | PTQ max (v0.5.0) | PTQ mse | QAT |
|---|---|---|---|---|
| CNN (1,954 parameters), K = 2 | 95.9 | 79.8 | 92.4 | 92.5 |
| CNN, K = 4 | 95.9 | 92.7 | 95.1 | 94.8 |
| CNN, K = 8 | 95.9 | 95.0 | 95.8 | 95.7 |
| depth-6 MLP (width 48), K = 2 | 95.9 | 45.5 | 90.3 | 91.9 |
| depth-6 MLP, K = 4 | 95.9 | 88.9 | 93.7 | 93.8 |
| depth-6 MLP, K = 8 | 95.9 | 94.6 | 95.5 | 95.3 |
| BatchNorm + MaxPool + AvgPool CNN, K = 4 | 97.8 | 93.9 | 96.0 | 96.4 |
| BatchNorm + MaxPool + AvgPool CNN, K = 8 | 97.8 | 97.0 | 97.3 | 97.6 |

The test set has 1,797 digits (standard error 0.5 points); the three nets and their training are in
`training/quant_digits.py`. The input of `levels=K` is cut to K levels too; with `levels=[16, K, ...]` the input keeps
its 17 pixel values and PTQ mse gains 0.6 to 1.1 points at K = 4.

## MaxPool: exact, by comparisons

The integer activations of a staircase layer are counts `a_i` in `0..K`, and `max_i a_i = sum_k [some a_i >= k]`. For a
`Conv -> ReLU -> MaxPool` (or `Conv -> MaxPool -> ReLU`, the same function: a staircase commutes with max) the pooled
layer is emitted once per level `k`, with unit `(i, k)` = `[h_i >= t_k]` (the bias lowered by `t_k - t_0` so that one
shared threshold serves all levels, one doll per unit); a second layer with units `[sum over the window of u_(i,k) >= 1]`
(threshold 1, one doll each) follows, and the next layer reads the sum over `k`. Nothing but staircases of sums is used,
so the engine primitives are the ones every MLP uses, and the result is the exact maximum of the integer counts:
`tests/test_quant.py` compares it with an independent NumPy window maximum (2x2) and with torch's own `MaxPool2d`
applied to the integer counts (overlapping 3x3 stride 2 with padding, dilated, rectangular and composed windows), and
runs a pooled 1-D net in DOOM at tolerance 0. Cost, measured with `df.plan` on
`Conv2d(1>8, 3x3, p1) -> ReLU -> Pool(2) -> Flatten -> Linear(128>10)` on 8x8 input:

| K | pool | layers | dolls | sectors |
|---|---|---|---|---|
| 2 | MaxPool2d(2) | 3 | 1,411 | 55,590 |
| 2 | AvgPool2d(2) | 2 | 1,154 | 42,916 |
| 4 | MaxPool2d(2) | 3 | 2,819 | 173,100 |
| 4 | AvgPool2d(2) | 2 | 2,306 | 85,282 |
| 8 | MaxPool2d(2) | 3 | 5,632 | 145,560 as tiles (no longer one map) |
| 8 | AvgPool2d(2) | 2 | 4,610 | 170,064 |

Max pooling costs about one extra layer of `K * n_out` single-threshold units and `K` times the lines of the pooled
layer. A MaxPool right after the input (no layer before it) and a MaxPool between a BatchNorm and its activation are
refused with a message; an AvgPool right after the input is accepted (it is a linear layer and merges into the next one).

## Quantisation-aware training: `doomformer.qat`

```python
from doomformer import qat                          # also df.qat; needs torch
qmodel = qat.fit(model, loader, levels=4, steps=3000)          # model: a trained nn.Sequential, loader: (x, y) batches
net = df.MLP.from_torch(qmodel, calib_x)                       # carries the trained staircases and the input quantiser
qat.lowering_gap(qmodel, net, x_test, y_test, float_model=model)   # accuracy float / staircase model / integer, agreement
```

`fit` copies the model, puts a `qat.Stair` (K levels, `lo + step * #{k : h >= thr[k]}`, exactly what DOOM counts) in
place of every ReLU / ReLU6 / Hardtanh / Sigmoid / Tanh / GELU / SiLU and a `qat.InputStair` (per-feature step,
`clip(round(x / D), -K, K) D`) in front, and trains with a straight-through gradient (the forward pass is the
staircase, the backward pass the identity: the recipe of `training/train_mnist.py`; `ste="clip"` uses the gradient of the
clipped activation). The steps of all staircases are set from calibration rows before training and again every
`refresh` steps (default `steps / 15`) by the MSE rule above; Adam, cosine learning rate from `lr` (5e-4) to `lr / 50`.
BatchNorm layers train with frozen running statistics so that the folded layer is the trained one. `levels`, `calib_x`
(default: the first loader batches), `loss_fn` (default cross-entropy / MSE), `distill` (opt-in knowledge
distillation from the float model), `signed_input`, `seed` are arguments; see the docstring.

`MLP.from_torch(qmodel, ...)` reads thresholds, offset and step of every `Stair` and the step of the `InputStair`, so
the integer net counts the same staircases the model was trained with. What differs is the rounding of the weights to
`|w| <= 255` (and of the thresholds) per layer and the integer input encoding (`round(x / (D / 16))` before the input
staircase, which differs from the float one only within 1/32 of a level boundary). `lowering_gap` measures it with the
NumPy integer reference: on the digits nets (84 rows of EVAL.md) the integer net and the torch staircase model give the
same class on 98.8 % of the test digits on average (96.3 % to 100 %), and their accuracies differ by 0.26 points on
average (largest 1.5, signed mean -0.03: noise, not bias).

Measured (EVAL.md, three seeds, writer-disjoint test set, mean accuracy in percent): on ReLU Linear/Conv nets QAT and
PTQ-mse are within the seed spread of each other at K = 3, 4 and 8 (CNN K = 4: 95.1 PTQ mse, 94.8 QAT; depth-6 MLP K = 4:
93.7 and 93.8); QAT is ahead at K = 2 for the MLP (91.9 against 90.3), and behind PTQ mse on the depth-6 MLP when the
training set is small (1,258 rows, K = 4: 92.5 against 93.2). That does not carry over to every net: on nets with
BatchNorm and several MaxPools, or with GELU, QAT was ahead in EVAL.md's extra rows `cnn_mp` and `mlp_gelu` (see there for
the K = 2, 4, 8 numbers) and in separate runs (BatchNorm + 2 MaxPool CNN: +5.3 points at K = 2, +0.7 at
K = 4; GELU MLP: +1.9 at K = 2, +0.7 at K = 4). Try `calibrate="mse"` first; try QAT for low K or for nets with max pools
or GELU, and it gives a torch model with the same staircases to compare the integer net with.
`equalise=True` (off by default) rescales the channels behind every ReLU of the torch model towards equal 99th percentiles
before training and at every refresh, as the recipe of `training/train_mnist.py` does (function-preserving; skipped next
to a BatchNorm after the ReLU or a grouped conv). On 18 validation splits (K = 4) it moved QAT from 95.2 % to 95.3 %
(PTQ mse 95.3 %): +0.8 on the writer-split depth-6 MLP, nothing elsewhere (`models/quant/qat_equalise_validation.log`).
The sweep of the other QAT settings (steps, learning rate, distillation) on a validation split is in
`models/quant/sweep_validation.log`.

## `MLP.predict(brain, x)`

```python
y_class = net.predict(brain, x_int)                       # int
classes = net.predict(brain, X_int)                       # (B,) array
net.predict(brain, x_float, encode=True)                  # applies net.encode first
```

`brain(x)` runs DOOM and checks it, then Python takes `numpy.argmax` of the integer outputs (first maximum on ties).
`brain` must have been compiled from this very `net`; otherwise `ValueError`.

## `quantization_report` and `range_report`

`net.quantization_report(x_float)` returns `n`, `max_abs_error`, `mean_abs_error`, `float_max_abs`, `in_range`,
`argmax_agreement` and:

* `input_range`: `peak` of `|x_int|` and `usage` against the +-12,000 gate range,
* `layers`: per weight layer (`h1`, `h2`, ..., `y`): `partial_sum_peak` / `partial_sum_usage` (worst partial sum in any
  crossing order against the +-30,000 accumulator), for hidden layers `gate_peak` / `gate_usage` (`|h|` against
  +-12,000), `top_level_fraction` (share of staircase outputs on the last step), `worst_case` (bound on `|h|` for
  **any** input: `max_row |b| + sum |W| * K`) and `rows_over` (rows that `check` would refuse),
* `warnings`: a line for the input or a layer that **exceeds** its range ("saturates the exact range ... RangeError")
  or uses >= 90 % of it *while its worst case is above the range*. A layer whose `worst_case` fits can never leave
  the range however close the data comes (`from_numpy` makes every layer fit by construction), so it is not warned
  about.

`net.range_report(x_int)` is the same for any integer MLP including hand-built ones (no float model needed). All of
it is NumPy; DOOM is not involved.

## Example 11: a conv net reads a glyph in DOOM

`python examples/11_conv_in_doom.py [--glyphs 12] [--seed 0] [--numpy]`: 7x7 pixel arrows (up/down/left/right, shifted by
up to one pixel, 3 % of the pixels flipped). Model: `Conv2d(1->3, 3x3) -> Hardtanh(-1, 1) -> Flatten -> Linear(75 -> 4)`,
trained with PyTorch (or NumPy with `--numpy`), quantised with `levels=[1, 3]`.

Measured on one shared aarch64 machine (`nice -n 10`, one DOOM process), seed 0:

| | PyTorch | NumPy (`--numpy`) |
|---|---|---|
| accuracy on 2,000 fresh glyphs, float model / integer model (reference arithmetic, not DOOM) | 0.955 / 0.945 | 0.957 / 0.949 |
| `df.plan` | 276 dolls, 5,938 sectors, fits | same |
| 12 glyph passes in DOOM | 13 s, **12/12 exact** | 14 s, **12/12 exact** |
| DOOM class == truth / float class == truth (12 test glyphs) | 11/12 / 11/12 | 12/12 / 12/12 |

"Exact" = DOOM's four outputs *and every hidden value* equal the integer reference (tolerance 0, `MismatchError`
otherwise, and the script exits non-zero). Disagreements between float and integer classes come from quantisation.
The script prints float, integer and DOOM class per glyph. With PyTorch it also asserts that the lowered float model
equals `model(x)` (difference < 1e-9).

## Transformers: `Transformer.from_torch`

```python
from doomformer import torch_models as tm
net = tm.GPT(V, T, d, H, dm, K, KA, KM, L, attn="relu", norm="none", mlp="relu")    # a torch module
tm.fit(net, train, dev, steps=(2000, 2500, 2000))                                  # (tok, tgt, valid) tensors, see tm.pack
qm = df.Transformer.from_torch(net, vocab)                                         # a QModel, range-checked
```

`doomformer/torch_models.py` holds the torch model the training scripts always used (moved out of `training/train_brain.py`,
which now imports it, so the shipped models and the library share one implementation). `GPT` mirrors
`intref.forward` exactly: token + position embedding, per layer q/k/v staircases, causal attention, output projection with a
residual gain, MLP with a residual gain, unembedding. Variants: attention `relu` (counts of passed score thresholds,
`thr_o[t]` folds 1/(t+1)) or `softmax` (quantised exponential with a null slot, division as a count of comparisons);
norm `none` or `rms` (integer RMSNorm before q/k/v and the MLP); MLP `relu` or `reglu` (gated). Any other value of
`attn`, `norm` or `mlp` raises `ValueError` naming the supported ones.

`tm.fit` is the three-stage recipe: calibrate the thresholds from activation quantiles; a float stage (every staircase a
sum of sigmoids, weights unrounded); act-QAT (hard staircases with a sigmoid surrogate gradient); full QAT (weights,
biases, embeddings, residual gains and thresholds rounded with a straight-through gradient, the best dev integer
cross-entropy kept). After it the module IS the integer model (`quant = hard = True`: every tensor is an integer-valued
float), and `Transformer.from_torch(net, vocab)` exports the rounded parameters as the `QModel` the training scripts write
to `models/`. It raises with the reason for a `FloatGPT` or any other module (a transformer is a chain of staircases
whose thresholds, residual gains and rounded weights are trained together, so a float transformer cannot be lowered by
rounding), for a GPT that was never calibrated or trained (`calib_tokens=` calibrates it first, which is plain
post-training rounding and can be far off; a calibration set with no valid position or a single distinct token raises), for
a wrong `vocab` length and for a lowered model that leaves DOOM's exact ranges (`intref.check_limits`: |w| <= 255, thresholds
in 1..12,000, gate-fed values within +-12,000, every partial sum within +-30,000, on the model's limit sequences).
`tm.save(net, path)` / `tm.load(path)` write and restore a GPT (configuration, parameters, threshold units, calibration
flag); the restored model lowers to the same parameters (same sha256, `tests/test_torch_models.py`).

What is exact: the lowered `QModel`'s `intref.forward` equals the torch integer forward bit for bit (`tests/test_torch_models.py`
for three variants, 100 of 100 held-out names on the trained models), and DOOM equals `intref.forward` at tolerance 0
(`df.compile(qm, mode="tiles")`, `brain.logits`). What is approximation: everything between the float model and the
integers (staircases, rounding, `thr_o`, the integer RMS). The integer logits are scaled by one learned float
(`exp(net.lg)`, saved in `info.json`) to give probabilities; that scale is applied in Python and never enters DOOM.
`models/quant/TRANSFORMER.md` has the measured perplexities against n-gram baselines and the DOOM run.

Limits: there is no cap on the number of levels, layers or width in `from_torch`; the limits are the range checks above, and
the size of the DOOM maps (`df.plan`). The shipped models and the training scripts use K <= 7, L <= 4, d <= 64; a net with
K = 9 and L = 5 lowers and passes `check_limits` (tested), and a K = 9 net was reported to run exact in DOOM in a separate run
that I did not repeat. No attention variant or norm beyond the ones above, and no import of a transformer trained
elsewhere: its weights would have to be trained through these staircases. On the level names the model does not beat its
5-gram baseline (`models/quant/TRANSFORMER.md`, which also has a second corpus).

## Limits

* The integer MLP stores dense weight matrices. A lowered conv is `(C_out*Ho*Wo) x (C_in*H*W)` ints, mostly zero; zero
  weights cost nothing in DOOM (no line, no doll work) but do cost memory in NumPy and in `.npz` files (the example's is 75 x 49; a 28x28 image
  with a 3x3 conv of 8 channels and no padding is 5,408 x 784 = 4.2 M entries).
* Every staircase level of every unit is a voodoo doll; sizes and times are in
  [BUILD_YOUR_OWN](BUILD_YOUR_OWN.md#limits-and-measured-sizes). `df.plan(net)` counts them without building anything.
* Staircase activations are monotone nondecreasing functions. GELU and SiLU are not, so they are replaced by their
  monotone envelope first (extra error 0.085 / 0.14, see above); Softmax inside the network is not supported. A final
  softmax or sigmoid is applied in Python to the decoded outputs.
* `MaxPool` multiplies the lines of the pooled layer by its K (see "MaxPool"), `AvgPool` and BatchNorm cost nothing
  (they are merged into the weights). The NumPy side stores the pool layer as a dense `(n_out K) x (n K)` int64 matrix:
  512 x 2,048 (8 MB) for the 8x8x8 conv above at K = 4, and 6,272 x 25,088 (1.26 GB of int64 for the matrix alone) for a
  28x28x8 conv with 2x2 pooling at K = 4. A separate run of that net measured a peak RSS of 3.2 to 3.7 GB in `from_torch`
  (about 6 s), and `df.plan` took 96 s and reported that it fits neither one map (corridor width over the 32,000 span
  limit) nor tiles; I have not run it myself.
* `save` stores the float model only if every activation is a built-in one (`relu`, `relu6`, `sigmoid`, `tanh`,
  `hardtanh`/`clamp`, `gelu`, `gelu_tanh`, `silu`, `identity`) and no pool is attached; a callable, a `Staircase`
  (including the staircases of a `qat.fit` model) or a pooled layer cannot be persisted, and a net loaded from such a
  file has an integer model and scales but no `float_forward`.
* The calibration data defines the staircase ranges and the weight scale: inputs far outside it saturate (clamp) rather
  than extrapolate. Inputs whose encoding leaves +-12,000 are refused (`RangeError`) before DOOM runs.
