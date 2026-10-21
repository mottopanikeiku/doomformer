# Float-model interop: convolutions, activations, argmax

How to get a *float* model (PyTorch or NumPy) into DOOM, what is exact on the way, what is approximated, and who
does what. Code: [`doomformer/interop.py`](../doomformer/interop.py) and the `MLP` class in
[`doomformer/kit.py`](../doomformer/kit.py); example [`examples/11_conv_in_doom.py`](../examples/11_conv_in_doom.py);
tests [`tests/test_interop.py`](../tests/test_interop.py).

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
| DOOM running the integer model | **exact**, checked per pass, tolerance 0 |
| `predict` = `argmax(brain(x))` | Python's `numpy.argmax` of DOOM's integers (first index wins ties) |

## `MLP.from_torch(model, calib_x, levels=8, *, input_shape=None)`

`model`: an `nn.Sequential` (nested `Sequential`s are fine) or a single supported module.

| module | handling |
|---|---|
| `Linear` | a layer |
| `Conv1d`, `Conv2d` | lowered to a dense (mostly zero) layer with `interop.conv1d_matrix` / `conv2d_matrix`; `stride`, `padding` (int, pair, `"same"`, `"valid"`), `dilation`, `groups`, `bias=False`; `padding_mode='zeros'` only |
| `Flatten` | row-major flatten, default `start_dim=1, end_dim=-1` only (this is the ordering the lowered matrices use) |
| `Identity`, `Dropout` | skipped; `Dropout` must be in `eval()` mode (or `p=0`), else `ValueError` (silent mis-quantisation is worse than an error) |
| `ReLU`, `ReLU6`, `Hardtanh(min, max)`, `Sigmoid`, `Tanh` | staircases, see below |

A layer without an activation after it is merged with the next layer (`W2 (W1 x + b1) + b2`, exact in float): a
`Conv -> Flatten -> Linear` stack is two layers. The *last* layer cannot have an activation (DOOM returns its raw
sums; apply `sigmoid`/`softmax` to `net.decode(y)` in Python). Anything else (`MaxPool`, `BatchNorm`, ...) raises
`ValueError` naming the module. Pooling is not lowered because max is not a monotone staircase of a sum; fold
`BatchNorm` into the preceding layer yourself.

`calib_x` is `(N, *sample_shape)`, e.g. `(N, 1, 5, 5)`, or flat with `input_shape=(1, 5, 5)`. The net remembers
`input_shape`, so `net.encode(x)` and `net.float_forward(x)` accept `(C, H, W)` / `(N, C, H, W)` and the integer
inputs of the brain are the flattened `(n_in,)` vectors.

`levels`: one int for every staircase, or one int per staircase (the input's first, then one per hidden layer: the
number the result has as `net.n_layers`). Every level of every unit is a voodoo doll, so levels multiply the size of
the map directly: a binary glyph needs `levels=[1, ...]` for its input.

`MLP.from_numpy(layers, calib_x, levels=8, *, input_shape=None)` is the same for float matrices:
`layers = [(W, b), (W, b, activation), ...]`; the default activation is ReLU between layers. `activation` is one of
`"relu"`, `"identity"`, `"relu6"`, `"sigmoid"`, `"tanh"`, `"hardtanh"`, `("clamp", lo, hi)` / `("hardtanh", lo, hi)`,
a vectorised monotone nondecreasing callable (checked on 257 points, `ValueError` otherwise), or a `df.staircase(...)`
result. To feed a convolution use `interop.conv2d_matrix(weight, bias, in_shape, stride, padding, dilation, groups)` ->
`(M, b, out_shape)` (and `conv1d_matrix`).

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

## Limits

* The integer MLP stores dense weight matrices. A lowered conv is `(C_out*Ho*Wo) x (C_in*H*W)` ints, mostly zero; zero
  weights cost nothing in DOOM (no line, no doll work) but do cost memory in NumPy and in `.npz` files (the example's is 75 x 49; a 28x28 image
  with a 3x3 conv of 8 channels and no padding is 5,408 x 784 = 4.2 M entries).
* Every staircase level of every unit is a voodoo doll; sizes and times are in
  [BUILD_YOUR_OWN](BUILD_YOUR_OWN.md#limits-and-measured-sizes). `df.plan(net)` counts them without building anything.
* Staircase activations must be monotone nondecreasing (no GELU/SiLU/Softmax inside the network). A final softmax or
  sigmoid is applied in Python to the decoded outputs.
* `save` stores the float model only if every activation is a built-in one (`relu`, `relu6`, `sigmoid`, `tanh`,
  `hardtanh`/`clamp`, `identity`); a callable or `Staircase` cannot be persisted, and a net loaded from such a file has
  an integer model and scales but no `float_forward`.
* The calibration data defines the staircase ranges and the weight scale: inputs far outside it saturate (clamp) rather
  than extrapolate. Inputs whose encoding leaves +-12,000 are refused (`RangeError`) before DOOM runs.
