# models/quant: post-training quantisation against quantisation-aware training on 8x8 digits

What this measures: how much accuracy a float PyTorch model loses when `df.MLP.from_torch` turns it into an integer
net with K-level staircases, with the v0.5.0 calibration (PTQ max), the new `calibrate="mse"` (MSE step + equalise,
PTQ mse) and `df.qat.fit` (QAT), and that the lowered nets run in DOOM exactly. The accuracies are those of the integer
net, computed by the NumPy integer reference (`net.reference`), which DOOM matches at tolerance 0 (last section).
Code: [`training/quant_digits.py`](../../training/quant_digits.py), [`doomformer/quantize.py`](../../doomformer/quantize.py),
[`doomformer/qat.py`](../../doomformer/qat.py); docs: [`docs/interop.md`](../../docs/interop.md).

## Commands

```
python training/quant_digits.py --protocol writer --seed 0     # also seeds 1, 2, and --protocol mixed (4 to 8 min each on one thread of a busy machine)
python training/quant_digits.py --protocol writer --seed 0 --models cnn mlp6 --ks 2 3 --conventions uniform   # also seeds 1, 2
python training/quant_digits.py --aggregate                    # results.json, results_table.md
python training/quant_digits.py --doom 5 --models mlp6 cnn cnn_bn_pool --protocol writer --seed 0 --workers 3 --k 4   # doom_<model>.json
python training/quant_digits.py --sweep --protocol writer --seed 0 --models cnn     # QAT settings on a validation split, also mlp6, seed 1, mixed: sweep_validation.log
python training/quant_digits.py --sweep-equalise --protocol writer --seed 0 --models cnn   # qat.fit(equalise=) on a validation split: qat_equalise_validation.log
```

Everything is seeded (initial weights, shuffling, split, torch on one thread): a second run of
`--protocol writer --seed 0 --models cnn --ks 4 --conventions uniform` gave the identical accuracies to the last digit.
`run_<protocol>_<seed>.json` hold every number below; `results_table.md` is the aggregate.

## Data and protocols

UCI optical digits (optdigits.tra 3,823 rows, optdigits.tes 1,797 rows; 8x8 pixels with values 0..16; the script
downloads them to `build/optdigits/` and checks sha256). The 1,797 rows of optdigits.tes are the rows of
`sklearn.datasets.load_digits` (the first row matches); scikit-learn is not installed here, so that equality is from its
documentation, not checked. Inputs are `x / 16`.

* `writer`: train on optdigits.tra, test on optdigits.tes (different writers; the harder, standard split). Test size
  1,797: one point is 18 digits, the standard error of an accuracy near 95 % is 0.5 points.
* `mixed`: the 1,797 rows of optdigits.tes split 70/30 at random (seeded), like the usual scikit-learn digits split.
  Test size 539: standard error 0.9 points.

Three seeds each (initial weights, shuffling and, for `mixed`, the split change). Cells are `mean (min-max)` in percent.

## Models

* `cnn`: Conv(1>8, 3x3, p1) ReLU Conv(8>8, 3x3, s2, p1) ReLU Flatten Linear(128>10), 1,954 parameters.
* `mlp6`: Linear 64>48 ReLU, four times Linear 48>48 ReLU, Linear 48>10: six Linear layers.
* `cnn_bn_pool`: Conv(1>8) BatchNorm ReLU MaxPool(2) Conv(8>16) BatchNorm ReLU AvgPool(2) Flatten Linear(64>10): the
  layers `from_torch` refused in v0.5.0.

Float training: Adam, one-cycle lr 3e-3, 60 epochs, batch 64, no augmentation. `levels` is `all K` (`levels=K`: every
staircase including the input's 0..16 pixel values cut to K levels) or `input 16,
hidden K` (`levels=[16, K, ...]`: the input keeps all 17 pixel values; 64 x 16 input dolls instead of 64 x K).
QAT: `qat.fit(model, loader, levels, steps=15 epochs)` (900 steps for 3,823 rows, 300 for 1,258) with the library
defaults (lr 5e-4, straight-through identity gradient, MSE steps refreshed every `steps / 15`). No hyperparameter was
tuned on the test rows. A sweep of the QAT settings on a 20 % validation split of the training rows (steps 300, 1,000,
3,000 x lr 1e-4, 3e-4, 1e-3 x `distill` 0, 0.5: 18 settings; CNN and depth-6 MLP, `writer` and `mixed`, two seeds: 8
splits, K = 4; `python training/quant_digits.py --sweep ...`, log in `sweep_validation.log`) gave 93.0 to 94.9 % (mean
94.2 %) for QAT and 94.6 % for PTQ mse on the same splits. Marginal means: `distill` 0.5 +0.2 points, lr 3e-4 against
1e-3 +0.3, steps no effect. No setting beat PTQ mse by more than 0.3 points (the best, 3,000 steps / lr 1e-4 /
distill 0.5, 94.9 %), which is inside the noise of 767 validation digits, so the defaults stayed.
Calibration rows: the first 1,000 training rows. `dolls` is `df.plan` for one map or, for nets that do not fit one map,
the tile total.

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

## Reading the tables

* **PTQ max is the v0.5.0 behaviour** and it is the one that fails at low K: depth-6 MLP at K = 4, 88.9 % against a float
  95.9 % (an independent run on a scikit-learn split measured 75.9 % against 96.0 %), CNN 92.7 % against 95.9 %, and 45.5 % / 79.8 %
  at K = 2. One rare large activation sets the step of a whole layer.
* **PTQ mse (MSE step + equalise) recovers most of it with no training**: depth-6 MLP K = 4 93.7 %, CNN K = 4 95.1 %
  (`writer`), BatchNorm + pool CNN 96.0 % against a float 97.8 %. The effect shrinks with K: at K = 8 the three
  methods are within 1 point of each other and of float.
* **QAT is not ahead of PTQ mse on these small nets** except where the staircase is coarsest or the net has the most
  structure: K = 2 depth-6 MLP 91.9 % against 90.3 %, BatchNorm + pool CNN 96.4 % against 96.0 % at K = 4 (`writer`). At
  K = 3, 4 and 8 the two are inside the seed spread on `cnn` and `mlp6`, and on `mixed` (1,258 training rows) QAT is
  behind PTQ mse on `mlp6` (K = 4: 92.5 % against 93.2 %). `calibrate="mse"` is the first thing to try; QAT adds a torch
  model with the same staircases (`QAT torch`) to compare the integer net with.
* **The input levels cost accuracy at `levels=K`**: `input 16, hidden 4` is 0.6 to 1.1 points above `all 4` for PTQ mse on
  the three nets (`writer`), for +768 input dolls on the CNN.
* **The lowering gap** (integer net against the torch staircase model it came from, 84 rows of the tables): same class on
  98.8 % of the test digits on average (96.3 % to 100 %), accuracy differing by 0.26 points on average (largest 1.5) and
  −0.03 points signed, so the weight and threshold rounding is noise, not a bias. It comes from rounding the weights to
  at most 8 bits (`|w| <= 255`) per layer: units whose pre-activation lies within the rounding noise of a threshold
  change level, and at K = 4 a level is a quarter of the range.
* **Targets** (CNN K = 4 at least 95 %, depth-6 MLP K = 4 at least 93 %, `levels=K`): `writer`, mean of three
  seeds: CNN 95.1 % (PTQ mse; seeds 94.9 to 95.3; QAT 94.8 %), depth-6 MLP 93.7 % (PTQ mse; seeds 93.0 to 94.3; QAT
  93.8 %). Met on the mean, not on every seed or every method (CNN QAT 94.8 %, one PTQ mse seed at 94.9 %). `mixed`: CNN
  95.4 % (PTQ mse) and 95.9 % (QAT), depth-6 MLP 93.2 % (PTQ mse) and 92.5 % (QAT): the MLP target is not met by QAT
  there, and one `mixed` CNN seed is at 93.1 % (PTQ mse). With `levels=[16, 4, 4]` the CNN is at 95.8 % (PTQ mse,
  `writer`, every seed 95.7 % or above). An independent run lost 8.9 points on its CNN at K = 4 (97.6 to 88.7 with PTQ
  max, its own split and training); here PTQ max loses 3.2 points (95.9 to 92.7) and PTQ mse 0.8.

## The lowered nets in DOOM

The K = 4 QAT nets of the `writer` seed 0 rows (`levels=4`; the integer test accuracies in the table equal the `qat`
entries of `run_writer_0.json`), the first five test digits of optdigits.tes each, `df.run_many(net, digits, workers=3,
mode="map")`: every pass runs the whole net as one map and compares every exposed value with the NumPy integer
reference at tolerance 0 (`MismatchError` otherwise; all five passes of every net completed). Values per pass =
outputs plus both signs of every hidden value. Wall times are on a shared 16-core machine under a load average of about
35 with three games running, so they say nothing about speed.

| model | K | digits | all values exact | values per pass | tics per pass | dolls | sectors | mode | DOOM class | integer class | label | float class | integer test accuracy | wall |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| cnn | 4 | 5 | yes | 1290 | 3960 | 2,819 | 159,945 | map | 01834 | 01834 | 01234 | 01134 | 95.16 % | 313 s, 3 games |
| cnn_bn_pool | 4 | 5 | yes | 5642 | 6968 | 3,844 | 312,442 | map | 01234 | 01234 | 01234 | 01234 | 96.55 % | 930 s, 3 games |
| mlp6 | 4 | 5 | yes | 490 | 5528 | 1,222 | 114,694 | map | 01834 | 01834 | 01234 | 01134 | 93.38 % | 282 s, 3 games |

Reading: DOOM equals the integer reference on all values of all passes; the argmax of DOOM's outputs is the integer net's class by construction, and the integer net differs from the float class on digit 2 of `cnn` and `mlp6` (it reads 8 where the float model says 1; the label is 2): that is quantisation, not DOOM. The `cnn` pass has the same 1,290 values and 3,960 tics as an independent run of the same architecture at K = 4. The `cnn_bn_pool` net contains BatchNorm folded into both convolutions, a max pool (lowered by comparisons) and an average pool: 4 integer layers, 3,844 dolls, one map.

## Limits and not done

* Three seeds, 8x8 digits, three small architectures, one corpus. The test sets are small (standard error 0.5 and 0.9
  points); differences below 1 point between methods are not established.
* `from_torch` quantises with its own calibration rows; the numbers do not say how it behaves on much larger nets, wider
  value ranges or MNIST-size inputs. A max pool over a 28x28x8 conv output needs a dense `(n_out K) x (n K)` matrix of
  1.3 GB (computed from the shapes, not run): a LeNet-5 with BatchNorm and MaxPool on full MNIST is not in this study
  for that reason.
* The scikit-learn path was not run (not installed). Everything here uses the UCI files directly.
* `Transformer.from_torch` is not part of this change: no torch transformer is converted. The transformers of `models/`
  are still trained by `training/train_qat.py` and its siblings, outside `from_torch`.
