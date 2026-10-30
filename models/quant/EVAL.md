# models/quant: post-training quantisation against quantisation-aware training on 8x8 digits

This study measures how much accuracy a float PyTorch model loses when `df.MLP.from_torch` turns it into an integer net
with K-level staircases. It compares three methods:

* PTQ max: the v0.5.0 calibration.
* PTQ mse: `calibrate="mse"`, an MSE step plus equalisation.
* QAT: `df.qat.fit`.

Accuracies are those of the integer net, computed by the NumPy integer reference (`net.reference`). DOOM matches that
reference at tolerance 0 ([last section](#the-lowered-nets-in-doom)).
Code: [`training/quant_digits.py`](../../training/quant_digits.py), [`doomformer/quantize.py`](../../doomformer/quantize.py),
[`doomformer/qat.py`](../../doomformer/qat.py). Docs: [`docs/interop.md`](../../docs/interop.md).

## Commands

```
python training/quant_digits.py --protocol writer --seed 0     # also seeds 1, 2, and --protocol mixed (4 to 8 min each on one thread of a busy machine)
python training/quant_digits.py --protocol writer --seed 0 --models cnn mlp6 --ks 2 3 --conventions uniform   # also seeds 1, 2
python training/quant_digits.py --aggregate                    # results.json, results_table.md
python training/quant_digits.py --doom 5 --models mlp6 cnn cnn_bn_pool --protocol writer --seed 0 --workers 3 --k 4   # doom_<model>.json
python training/quant_digits.py --sweep --protocol writer --seed 0 --models cnn     # QAT settings on a validation split, also mlp6, seed 1, mixed: sweep_validation.log
python training/quant_digits.py --sweep-equalise --protocol writer --seed 0 --models cnn   # qat.fit(equalise=) on a validation split: qat_equalise_validation.log
```

Everything is seeded, and torch runs on one thread. Two runs of the same command give identical accuracies.
`run_<protocol>_<seed>.json` hold every number below. `results_table.md` is the aggregate.

## Data and protocols

The data is UCI optical digits: 8x8 pixels with values 0..16, inputs `x / 16`. The script downloads the files to
`build/optdigits/` and checks sha256.

| protocol | train | test | standard error near 95 % |
|---|---|---|---|
| `writer`: optdigits.tra against optdigits.tes, different writers | 3,823 | 1,797 | 0.5 points |
| `mixed`: optdigits.tes split 70/30 at random | 1,258 | 539 | 0.9 points |

Each protocol runs three seeds. A seed sets the initial weights, the shuffling and, for `mixed`, the split. Cells are
`mean (min-max)` in percent.

## Models

* `cnn`: Conv(1>8, 3x3, p1) ReLU Conv(8>8, 3x3, s2, p1) ReLU Flatten Linear(128>10), 1,954 parameters.
* `mlp6`: Linear 64>48 ReLU, four times Linear 48>48 ReLU, Linear 48>10.
* `cnn_bn_pool`: Conv(1>8) BatchNorm ReLU MaxPool(2) Conv(8>16) BatchNorm ReLU AvgPool(2) Flatten Linear(64>10).
* `cnn_mp`: Conv(1>8) BatchNorm ReLU MaxPool(2) Conv(8>16) BatchNorm ReLU MaxPool(2) Flatten Linear(64>10).
* `mlp_gelu`: Linear 64>64 GELU, Linear 64>64 GELU, Linear 64>10.

`cnn_mp` and `mlp_gelu` run on `writer` only, with `--models cnn_mp mlp_gelu --ks 2 4 8 --conventions uniform`.

## Setup

* Float training: Adam, one-cycle lr 3e-3, 60 epochs, batch 64, no augmentation.
* `levels`: `all K` is `levels=K`, which also cuts the input pixels to K levels. `input 16, hidden K` is
  `levels=[16, K, ...]`, which keeps all 17 input values at the cost of 64 x 16 input dolls.
* QAT: `qat.fit(model, loader, levels, steps=15 epochs)` with the library defaults: lr 5e-4, straight-through gradient.
* Calibration rows: the first 1,000 training rows.
* `dolls`: `df.plan` for one map, or the tile total for nets that do not fit one map.

No hyperparameter is tuned on the test rows. The QAT defaults come from a sweep of 18 settings on a 20 % validation
split of the training rows (`--sweep`, log in `sweep_validation.log`). No fixed setting beats PTQ mse by more than 0.3
points on average, which is inside the noise of 767 validation digits.

## Results

### Protocol `writer`: train 3,823, test 1,797, seeds 0, 1, 2 (mean, min-max in percent; integer accuracy by the NumPy reference)

| model | float | K | levels | PTQ max | PTQ mse | QAT | QAT torch | dolls |
|---|---|---|---|---|---|---|---|---|
| cnn | 95.9 (95.6-96.2) | 2 | all 2 | 79.8 (76.0-82.1) | 92.4 (91.7-93.5) | 92.5 (92.4-92.8) | 92.6 (92.2-92.8) | 1,411 (map) |
| cnn | 95.9 (95.6-96.2) | 3 | all 3 | 89.7 (87.8-90.8) | 94.4 (93.8-94.8) | 94.5 (94.2-94.8) | 94.5 (94.0-94.8) | 2,115 (map) |
| cnn | 95.9 (95.6-96.2) | 4 | all 4 | 92.7 (91.4-93.4) | 95.1 (94.9-95.3) | 94.8 (94.5-95.2) | 94.8 (94.7-95.1) | 2,819 (map) |
| cnn | 95.9 (95.6-96.2) | 4 | input 16, hidden 4 | 93.2 (92.7-93.5) | 95.8 (95.7-95.9) | 95.2 (94.4-95.5) | 95.3 (94.7-95.9) | 3,587 (map) |
| cnn | 95.9 (95.6-96.2) | 8 | all 8 | 95.0 (94.6-95.2) | 95.8 (95.5-96.0) | 95.7 (95.6-95.8) | 95.6 (95.6-95.6) | 5,635 (map) |
| cnn | 95.9 (95.6-96.2) | 8 | input 16, hidden 8 | 95.6 (95.4-95.8) | 95.7 (95.5-95.9) | 95.6 (95.2-95.9) | 95.7 (95.2-96.2) | 6,147 (map) |
| mlp6 | 95.9 (95.6-96.3) | 2 | all 2 | 45.5 (39.8-49.8) | 90.3 (90.0-90.5) | 91.9 (91.4-92.4) | 92.2 (91.8-93.1) | 614 (map) |
| mlp6 | 95.9 (95.6-96.3) | 3 | all 3 | 78.3 (75.7-80.7) | 93.8 (93.6-94.2) | 93.6 (93.3-94.0) | 93.8 (93.4-94.2) | 918 (map) |
| mlp6 | 95.9 (95.6-96.3) | 4 | all 4 | 88.9 (86.3-91.5) | 93.7 (93.0-94.3) | 93.8 (93.4-94.0) | 94.0 (93.4-94.3) | 1,222 (map) |
| mlp6 | 95.9 (95.6-96.3) | 4 | input 16, hidden 4 | 89.7 (87.6-91.9) | 94.8 (94.4-95.1) | 94.4 (94.0-94.9) | 94.2 (93.5-94.7) | 1,990 (map) |
| mlp6 | 95.9 (95.6-96.3) | 8 | all 8 | 94.6 (94.0-95.2) | 95.5 (95.2-95.9) | 95.3 (94.5-96.1) | 95.3 (94.8-95.9) | 2,438 (map) |
| mlp6 | 95.9 (95.6-96.3) | 8 | input 16, hidden 8 | 94.5 (94.2-94.8) | 95.3 (94.9-95.5) | 95.4 (95.0-95.6) | 95.4 (94.9-95.7) | 2,950 (map) |
| cnn_bn_pool | 97.8 (97.6-97.9) | 4 | all 4 | 93.9 (92.5-95.2) | 96.0 (95.5-96.3) | 96.4 (95.8-96.9) | 96.2 (95.5-96.5) | 3,844 (map) |
| cnn_bn_pool | 97.8 (97.6-97.9) | 4 | input 16, hidden 4 | 94.5 (94.3-94.8) | 96.6 (96.5-96.8) | 96.8 (96.0-97.2) | 96.8 (96.3-97.1) | 4,608 (tiles) |
| cnn_bn_pool | 97.8 (97.6-97.9) | 8 | all 8 | 97.0 (96.7-97.4) | 97.3 (97.1-97.7) | 97.6 (97.5-97.8) | 97.5 (97.1-97.9) | 7,680 (tiles) |
| cnn_bn_pool | 97.8 (97.6-97.9) | 8 | input 16, hidden 8 | 96.8 (96.6-97.1) | 97.4 (97.1-97.6) | 97.8 (97.7-97.9) | 97.6 (97.3-97.8) | 8,192 (tiles) |
| cnn_mp | 98.3 (98.0-98.6) | 2 | all 2 | 79.3 (77.0-81.4) | 88.3 (85.5-90.0) | 92.8 (91.8-94.0) | 92.7 (91.2-94.2) | 2,053 (map) |
| cnn_mp | 98.3 (98.0-98.6) | 4 | all 4 | 93.6 (93.0-94.0) | 96.4 (96.1-96.5) | 96.9 (96.4-97.3) | 97.2 (96.9-97.6) | 4,101 (map) |
| cnn_mp | 98.3 (98.0-98.6) | 8 | all 8 | 97.1 (96.8-97.2) | 97.8 (97.2-98.2) | 98.0 (97.9-98.1) | 98.0 (97.9-98.1) | 8,192 (tiles) |
| mlp_gelu | 96.2 (96.0-96.4) | 2 | all 2 | 90.7 (90.0-91.2) | 92.5 (92.4-92.5) | 93.1 (92.8-93.3) | 93.1 (92.9-93.2) | 387 (map) |
| mlp_gelu | 96.2 (96.0-96.4) | 4 | all 4 | 94.1 (93.6-94.4) | 95.3 (95.1-95.4) | 95.4 (95.2-95.6) | 95.4 (95.3-95.6) | 771 (map) |
| mlp_gelu | 96.2 (96.0-96.4) | 8 | all 8 | 95.7 (95.5-95.9) | 96.1 (96.0-96.1) | 96.0 (95.9-96.2) | 96.0 (96.0-96.2) | 1,539 (map) |

### Protocol `mixed`: train 1,258, test 539, seeds 0, 1, 2 (mean, min-max in percent; integer accuracy by the NumPy reference)

| model | float | K | levels | PTQ max | PTQ mse | QAT | QAT torch | dolls |
|---|---|---|---|---|---|---|---|---|
| cnn | 96.4 (94.8-97.4) | 4 | all 4 | 94.2 (92.6-95.5) | 95.4 (93.1-96.8) | 95.9 (94.2-96.8) | 96.0 (94.4-97.0) | 2,819 (map) |
| cnn | 96.4 (94.8-97.4) | 4 | input 16, hidden 4 | 94.1 (91.5-96.1) | 95.8 (94.8-96.8) | 95.9 (94.2-96.7) | 96.2 (94.2-97.6) | 3,587 (map) |
| cnn | 96.4 (94.8-97.4) | 8 | all 8 | 95.5 (94.2-96.5) | 96.2 (94.1-97.2) | 95.9 (94.4-96.8) | 95.9 (94.1-97.0) | 5,635 (map) |
| cnn | 96.4 (94.8-97.4) | 8 | input 16, hidden 8 | 96.2 (94.4-97.2) | 96.2 (94.8-97.0) | 96.2 (94.4-97.4) | 96.1 (94.8-97.2) | 6,147 (map) |
| mlp6 | 95.8 (95.2-96.3) | 4 | all 4 | 84.7 (78.8-92.0) | 93.2 (92.4-93.9) | 92.5 (91.3-93.3) | 92.6 (91.1-94.1) | 1,222 (map) |
| mlp6 | 95.8 (95.2-96.3) | 4 | input 16, hidden 4 | 86.2 (82.2-92.4) | 93.8 (92.8-95.2) | 93.6 (92.6-94.1) | 93.5 (91.7-94.4) | 1,990 (map) |
| mlp6 | 95.8 (95.2-96.3) | 8 | all 8 | 93.2 (91.1-94.4) | 95.1 (94.1-96.1) | 94.7 (93.9-95.7) | 95.1 (94.4-95.9) | 2,438 (map) |
| mlp6 | 95.8 (95.2-96.3) | 8 | input 16, hidden 8 | 93.9 (92.6-95.2) | 95.2 (94.4-96.1) | 94.8 (93.9-96.1) | 95.0 (93.7-95.9) | 2,950 (map) |
| cnn_bn_pool | 98.6 (98.5-98.7) | 4 | all 4 | 94.3 (94.1-94.8) | 96.8 (95.7-98.1) | 97.1 (96.5-97.6) | 97.2 (96.5-97.6) | 3,844 (map) |
| cnn_bn_pool | 98.6 (98.5-98.7) | 4 | input 16, hidden 4 | 96.6 (95.7-97.2) | 97.1 (96.5-97.6) | 97.8 (97.4-98.0) | 97.5 (97.0-97.8) | 4,608 (tiles) |
| cnn_bn_pool | 98.6 (98.5-98.7) | 8 | all 8 | 97.4 (96.8-98.0) | 97.9 (97.8-98.0) | 97.9 (97.6-98.1) | 97.7 (97.4-98.0) | 7,680 (tiles) |
| cnn_bn_pool | 98.6 (98.5-98.7) | 8 | input 16, hidden 8 | 98.1 (98.0-98.3) | 98.2 (98.1-98.3) | 98.1 (97.8-98.5) | 98.1 (97.6-98.7) | 8,192 (tiles) |

### QAT minus PTQ mse on nets with max pools or GELU (`writer`, points)

| model | K | mean | seeds 0, 1, 2 |
|---|---|---|---|
| cnn_mp | 2 | +4.6 | +2.7, +6.3, +4.8 |
| cnn_mp | 4 | +0.5 | +0.3, +0.4, +0.8 |
| cnn_mp | 8 | +0.2 | +0.8, 0.0, -0.1 |
| mlp_gelu | 2 | +0.65 | +0.8, +0.7, +0.4 |
| mlp_gelu | 4 | +0.2 | +0.1, +0.3, +0.1 |
| mlp_gelu | 8 | -0.04 | +0.1, 0.0, -0.2 |

## Reading the tables

* PTQ max fails at low K. One rare large activation sets the step of a whole layer.
* PTQ mse recovers most of the loss with no training. At K = 8 the three methods are within 1 point of each other and
  of float.
* On ReLU Linear/Conv nets QAT is level with PTQ mse, inside the seed spread. It is ahead at K = 2 and on nets with max
  pools or GELU. Try `calibrate="mse"` first.
* `input 16, hidden 4` is 0.6 to 1.1 points above `all 4` for PTQ mse on `writer`. On the CNN it costs 768 more dolls.
* The integer net and the torch staircase model it came from agree on 98.8 % of test digits on average. The gap comes
  from rounding weights to `|w| <= 255`, and it has no sign bias.
* Targets were CNN K = 4 at least 95 % and depth-6 MLP K = 4 at least 93 %. PTQ mse meets both on the `writer` mean.
  Not every seed and method meets them.

## The lowered nets in DOOM

`--doom 5` runs the K = 4 QAT nets of `writer` seed 0 on the first five test digits, as one map each. Every pass
compares every exposed value with the NumPy integer reference at tolerance 0. Values per pass are the outputs plus
both signs of every hidden value. Wall times are on a shared machine with three games running and say nothing about
speed.

| model | K | digits | all values exact | values per pass | tics per pass | dolls | sectors | mode | DOOM class | integer class | label | float class | integer test accuracy | wall |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| cnn | 4 | 5 | yes | 1290 | 3960 | 2,819 | 159,945 | map | 01834 | 01834 | 01234 | 01134 | 95.16 % | 313 s, 3 games |
| cnn_bn_pool | 4 | 5 | yes | 5642 | 6968 | 3,844 | 312,442 | map | 01234 | 01234 | 01234 | 01234 | 96.55 % | 930 s, 3 games |
| mlp6 | 4 | 5 | yes | 490 | 5528 | 1,222 | 114,694 | map | 01834 | 01834 | 01234 | 01134 | 93.38 % | 282 s, 3 games |

DOOM equals the integer reference on every value of every pass. On digit 2 of `cnn` and `mlp6` the integer net reads 8
where the float model reads 1. That error is quantisation, not DOOM. `cnn_bn_pool` folds BatchNorm into both
convolutions and lowers the max pool to comparisons: 4 integer layers in one map.

## Limits

* Three seeds, 8x8 digits, five small architectures, one corpus. Differences below 1 point between methods are not
  established.
* The results say nothing about much larger nets or MNIST-size inputs.
* A max pool over a 28x28x8 conv output needs a dense 6,272 x 25,088 matrix at K = 4. `df.plan` reports that such a
  net fits neither one map nor tiles, so a LeNet-5 on MNIST does not run in DOOM.
* The data comes from the UCI files. The scikit-learn loader is not used.
* Transformers are covered in [TRANSFORMER.md](TRANSFORMER.md).
