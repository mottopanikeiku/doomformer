# DOOM learns

Two learners keep their weights as ceiling heights of sectors in one DOOM map and change them with voodoo dolls. After every
example, Python reads every exposed value and compares it with a NumPy integer twin at tolerance 0.

| | what it learns | weights | decision | page |
|---|---|---|---|---|
| `doomformer learn digits` | to read 8x8 digits from a blank brain, 10 classes | 370 | the dolls take the argmax of 10 scores | [below](#doom-learns-to-read-digits) |
| `df.Perceptron` | binary yes/no heads (the arena marine learns to aim) | 2 to 64 | the dolls pick the first head with a score of 1 or more | [`df.Perceptron`](#dfperceptron-binary-heads) |

## DOOM learns to read digits

```sh
doomformer learn digits --steps 1500        # about 80 minutes on a loaded machine; one line per step, then the held-out accuracy
doomformer learn digits --classes 4 --steps 300     # 5 minutes: digits 0 to 3, a map with a quarter of the sectors
doomformer learn digits --steps 40 --tamper 0 1 raise            # negative control: diverges at the predicted step
```

The learner is an online multiclass perceptron. Its 10 x 37 = 370 weights are the ceiling heights of 370 sectors and start at 0.
The 37 inputs are the 36 centre pixels of an 8x8 digit, each 1 where the grey level is at least 8, and a constant 1. One training
example is one run of one map. Python writes the 37 inputs and the label (a one-hot) into gate heights, advances the clock and
reads. The dolls do the rest, in four layers of the same map with nothing from Python between them:

| layer | dolls | what the dolls do |
|---|---|---|
| forward | 370 | weight (c, i) is a ceiling. For every pixel that is on, a doll walks a corridor of 6 ceiling gates (thresholds -2 to 3) under it and crosses lines that add 1 to the score `s_c` for each gate it fits under, and +-1 to the 9 differences `d_cb` that involve class c. So `s_c = sum_i x_i (3 + clamp(w_ci, -3, 3))` |
| compare | 45 | one doll per class pair a < b passes iff `s_a >= s_b` (the gate reads `d_ab = s_a - s_b + 1 >= 1`) and adds 1 to `wins_a` and -1 to `wins_b`; `W_c = #{j > c: s_c >= s_j} - #{j < c: s_j >= s_c} + c` (the comparator of `df.with_argmax`) |
| winner | 20 | a doll per class passes iff `W_c >= 9`: the first maximal score, as `np.argmax`. It lights `z_c` and lowers `q_c`. A second doll per class, behind the label's gate, raises `q_c`. `q_c = 1 + y_c - z_c` |
| learn | 370 | for every pixel that is on: the doll lowers `w_ci` by 1, then crosses a gate under `q_c >= 1` and raises it by 1, then a gate under `q_c >= 2` and raises it by 1. Net change `y_c - z_c` |

The learn layer is the perceptron rule: on a mistake, +x to the weights of the true class and -x to the weights of the predicted
class; nothing when the prediction is right. The weights are `state` (`doomformer/pure.py`): they survive the rewind between
examples and are never written by Python.

### What is checked, and what Python does

After every step Python reads 455 values and compares each with `DigitRef`, a NumPy integer twin, at tolerance 0: the 10 scores, the
45 pair differences, the 10 win counts, the 10 winner lights, the 10 update signals `q` and all 370 weights (ceiling heights read
through the game). The prediction is DOOM's winner light, not Python's argmax. A difference stops the run
(`df.MismatchError`, with the names and values). The twin exists three times and all three agree: `DigitRef` (the pair
comparator of the dolls), `learn_digits.plain_perceptron` (the textbook rule, used for the comparison below) and a short
textbook function in `tests/test_learn_digits.py`.

Python's part, counted by `StepAudit` (every console command is classified; anything else raises before it reaches the engine):
input writes (`puke -1|-2` to an input port: the 36 pixels, the bias and the label's 10 one-hot values, 47 per step), a rewind
(`puke -3`: the ACS script sends the dolls home and restores the non-state values), a start (`puke -7`: drops the start
barriers) and readouts (`puke -100..`). A write to a weight sector is rejected, so the weights change only by what the dolls
do (the test sends one and expects the rejection). In the 1,500-step run: 70,500 input writes (1,500 steps x 47), 1,499 rewinds,
1,500 starts, 0 motion checks, 12,000 readout commands (8 pages per step), 0 other.

Negative controls (`--tamper CLASS FEATURE LINE`, `tests/test_learn_digits.py`): the map is built with one of the three update
lines of one weight moving its ceiling by 2 instead of 1. The twin says which step first crosses that line (`lower`: the pixel is
on; `keep`: on and `q >= 1`; `raise`: on and the true class was not predicted). DOOM must be exact before it and differ at
exactly that step in exactly that weight. `models/digits/tamper_lower.log`, `tamper_keep.log` and `tamper_raise.log`: caught at
the predicted step (19, 19 and 14).

### What it is not

* The rule is Python's design. The dolls execute a perceptron update that the author wrote as geometry; DOOM does not find a rule.
  The labels come from the data set and reach the map as Python's input writes. It is supervised, online, one pass over a
  random stream with repeats (a fresh permutation of the 1,297 training digits per pass).
* The weights are bounded by construction: each is an integer, a score sees it clamped to [-3, 3], and the update has no
  saturation, so `|w|` grows by at most 1 per step. The learner stops at 9,999 steps (plane values stay inside the exact range
  of the map). The clamp makes the learner differ from the textbook perceptron (see the table).
* The default schedule (`opt 3`) has an ACS script copy each layer's light accumulators into the next layer's gate heights
  (docs/scheduling.md). The script adds and compares nothing, and it never writes a weight. `--opt 0`, `1` and `2` build the
  maps without the copy bus; they are exact on the 4 x 6 test map and on one step each of the 10 x 37 map, and they need one
  wave per forward doll: 1,448 tics per step on 4 x 6 and 20,472 tics on 10 x 37 (measured, below). Hence the default.
* The accuracy is that of a perceptron on 36 binary pixels: 10 % for the blank brain and 71.2 % on 500 held-out digits after
  1,500 steps (this stream), between 68 % and 82 % at the 16 checkpoints from 750 to 1,500 examples. It does not reach 90 %. The clamp
  costs a few points against the textbook rule (table below). The whole 8x8 image (65 features) costs twice the time per step.

### Results (one game, `doomformer learn digits --steps 1500 --every 50 --seed 0`)

One run, seed 0, `models/digits/learn_1500.log` (every step) and `learn_1500.json`: **1,500 steps, every one exact**. After each
step the 455 values (10 scores, 45 differences, 10 win counts, 10 winner lights, 10 update signals, 370 weights) equal the
twin. 445 steps made a mistake and changed weights (on average 33 weights); 1,055 did not.

| examples seen | DOOM weights (read from the ceilings) | NumPy, same stream and clamp | NumPy, textbook perceptron (no clamp), same stream |
|---|---|---|---|
| 0 (blank) | 10.0 % | 10.0 % | 10.0 % |
| 50 | 43.4 % | 43.4 % | 40.0 % |
| 100 | 46.6 % | 46.6 % | 55.0 % |
| 200 | 65.0 % | 65.0 % | 64.8 % |
| 300 | 62.8 % | 62.8 % | 64.2 % |
| 500 | 64.4 % | 64.4 % | 71.8 % |
| 750 | 75.6 % | 75.6 % | 75.2 % |
| 1,000 | 72.4 % | 72.4 % | 76.0 % |
| 1,250 | 79.2 % | 79.2 % | 85.4 % |
| 1,500 | 71.2 % | 71.2 % | 81.0 % |

Held-out accuracy of 500 digits that the learner never trains on (50 per class). The first column scores the weights read from
DOOM (NumPy forward pass over the 370 ceilings); the second is the independent NumPy code of the same learner and agrees at
every one of the 30 checkpoints from 50 to 1,500 examples (`curve identical to DOOM's: True`). The online accuracy over the stream (the prediction
before each update) is 70.3 %. The curve of the last iterate is noisy: from 200 examples on it moves by up to 13 points between checkpoints
(`learn_1500.png`, an asset of the v0.6.0 release, draws it with the learned weights as 6x6 pictures; `models/digits/figure.py` rebuilds it).

The one DOOM stream is seed 0, run once. Without DOOM, the same learner on 20 other streams (`models/digits/numpy_seeds.py`,
`numpy_seeds.txt`) reaches 74.8 % +- 5.3 (range 63.0 to 82.6) at 1,500 examples, and the textbook perceptron without the clamp
78.4 % +- 4.8; seed 0 is below the mean for the clamped learner (71.2 %).

The learned weights also run through the repository's ordinary forward path: `df.MLP` with `df.with_argmax`, compiled to a
separate map, the first 100 held-out digits, every value of every pass checked exactly (32 s). DOOM's winner equals NumPy's
argmax for 100 of 100 digits, and 71 of the 100 are the right digit. The 4-class run (`--classes 4 --steps 300`,
`models/digits/learn_4class_300.log`) is also exact in all 300 steps: held-out accuracy 83.5 % on 200 digits of 0 to 3
(textbook NumPy 89.5 %; checkpoints between 70.5 % and 94.5 %).

### Cost

Measured by `models/digits/cost_curve.py` (`cost_curve.txt`): the map for C classes and n features, 4 steps per row (1 step
for opt 0 to 2 at 10 x 37), all exact; one DOOM game; machine load average 21 at the start and higher later, so every time is
a loaded one.

| classes x features | opt | dolls | sectors | game tics per step | seconds per step |
|---|---|---|---|---|---|
| 4 x 6 | 3 | 66 | 2,196 | 296 | 0.08 |
| 4 x 37 | 3 | 314 | 12,240 | 296 | 0.62 |
| 10 x 12 | 3 | 309 | 15,117 | 304 | 0.75 |
| 10 x 25 | 3 | 569 | 30,327 | 304 | 1.45 |
| **10 x 37** (the 1,500-step run) | 3 | 809 | 44,367 | 304 | 2.14 (3.23 over the run, load 30 to 40) |
| 10 x 65 (the whole 8x8 image and a bias) | 3 | 1,369 | 77,127 | 304 | 4.30 |
| 4 x 6 | 0 | 66 | 2,207 | 1,448 | 0.44 |
| 4 x 6 | 2 | 66 | 2,207 | 1,312 | 0.34 |
| 10 x 37 | 2 | 809 | 44,729 | 20,304 | 152.9 |
| 10 x 37 | 1 | 809 | 44,729 | 20,448 | 174.9 |
| 10 x 37 | 0 | 809 | 44,735 | 20,472 | 202.9 |

At opt 3 a step takes 296 to 304 tics whatever the size (each layer is one wave; the corridors set the time), and the seconds
follow the sector count (about 120 sectors per weight at 10 classes): 10 x 37 takes 456,000 tics for 1,500 steps. At opt 0 to 2 the 10 x 37
forward layer needs one wave per forward doll (370 waves, docs/scheduling.md), so a step is 20,000 tics. Bigger runs are limited
by time, not by exactness: the run stopped at 1,500 steps because it takes 80 minutes, and no step differed. The learner
refuses more than 9,999 steps (the weight range). A 2,000-step run is about 110 minutes at the load of this run; the whole 8x8 image
(10 x 65) would take 4.3 s per step or more. `learn_digits.py` takes any (classes, features); the `df.Perceptron` maps of the
section below cost 0.07 to 1.8 s per step at 8 to 32 features and 1 to 2 heads.

### Data

`models/digits/digits8x8.npz`: 1,797 digits of the UCI "Optical Recognition of Handwritten Digits" test set (Alpaydin and Kaynak
1998, CC BY 4.0, via scikit-learn), 500 held out (50 per class, stored in the file), 1,297 for training. See
[`models/digits/README.md`](../models/digits/README.md).

# `df.Perceptron`: binary heads

A perceptron whose **weights are ceiling heights in a DOOM map** and whose learning rule is executed by voodoo dolls.
You feed it labelled examples one at a time. After every example DOOM's weights have changed (they live in the map and
are never reloaded). Every number DOOM produces is read back from the game and compared, with tolerance 0, against an
independent integer reference.

```python
import numpy as np, doomformer as df
from doomformer.learnkit import Perceptron        # (also exported as df.Perceptron)

spec = Perceptron(n_features=3, heads=1)           # weights start at 0
with spec.compile() as learner:                    # builds the map; the game starts on the first step
    scores, action = learner.step([1, 1, 0], [+1]) # x in {0,1}^3, one label per head in {-1, 0, +1}
    print(scores, action, learner.weights)         # all three come from DOOM
```

`examples/12_doom_learns_live.py` is the full demo (a demon on 7 spots, "shoot iff left of centre", then the teacher flips the
rule and DOOM relearns; a traffic light in the map follows DOOM's verdicts).

## What DOOM, the ACS scripts and Python each do

**One `step(x, y)` is one run of the map.** The map has two layers of voodoo dolls (`doomformer/learn.py`,
`doomformer/learnkit.py`):

| who | what |
|---|---|
| **DOOM, forward layer** | For each head `h` and active feature `i`, a doll walks a corridor of `clip` ceiling gates under the weight's ceiling (and one under the negated twin). Every gate it fits under adds +1 (resp. -1) to the head's score floor. Result: `s_h = sum_i x_i * clamp(w_hi, -clip, clip)`. |
| **DOOM, learn layer** | Reading the scores of *this* step (before any update): **action dolls** choose the action (see below); **verdict dolls** light the `right{h}` / `wrong{h}` outputs; **update dolls** apply the perceptron rule: for an active feature `i` of head `h`, a doll passes the cap gate (`y*w < cap`), the label gate (`y_h = +1` or `-1`) and the mistake gate (`y_h * s_h <= 0`), then crosses a `Ceiling_RaiseByValue` / `Ceiling_LowerByValue` line on the weight's ceiling (and its twin) with amount 1. Weights persist from step to step; they never leave `[-cap, cap]`. |
| **ACS level scripts** | Only plumbing: scripts 1/2 write Python's inputs into gate heights (`Floor_/Ceiling_MoveToValue`), script 3 rewinds the dolls and non-state values between steps (state, i.e. the weights, is left alone), script 4 tags each doll as it starts, script 7 starts layer 0, script 8 (the motion check) compares doll positions between windows so Python knows when every doll has stopped, scripts 100+ copy light levels and plane heights into `USER1..60` for Python to read. **No ACS script adds, compares or updates any value of the network.** Your own `world` scripts are yours. |
| **Python** | Builds the map, writes `x` and `y` into the map before a run, waits for the dolls to settle, reads everything back, and compares it with the twin (`PerceptronRef`: plain NumPy integers). In the example, Python also *teaches* (it knows the rule, it produces the labels) and moves the demon (scenery). |

Nothing presented as computed by DOOM is computed by Python: the twin is only what DOOM is checked against.

### The learning rule, exactly

With `s_h` the score before the update and `y_h in {-1, 0, +1}`:

```
s_h      = sum_i x_i * clip(w_hi, -clip, clip)
mistake  = (y_h != 0) and (y_h * s_h <= 0)                 # s = 0 counts as a mistake for either label
w_hi    <- clip(w_hi + y_h * x_i, -cap, cap)   for every active i, only when `mistake`
```

`y_h = 0` means "no lesson for this head". Features must be 0 or 1 (a doll is either there or not); a bias is just a feature
that is always 1. `cap` bounds the weights (a gate in the map stops lessons that would leave `[-cap, cap]`); `clip`
(default 3) bounds how much one weight can contribute to a score (the forward corridor has `clip` gates).

### The action DOOM chooses

The learn layer's action dolls read the scores and pick `action`:

```
action = 1 + (index of the first head with s_h >= 1)      or 0 if no head has s_h >= 1
```

computed from the scores *before* this step's lesson. With one head, `action == 1` means "predict yes" (`s >= 1`). The
dolls add `action` to the `act` light and 1 to exactly one `pick{k}` light.

Verdicts (per head, only when `y_h != 0`): `right{h} = 1` iff `y_h = +1 and s_h >= 1` or `y_h = -1 and s_h <= 0`, else
`wrong{h} = 1`. This differs from the mistake rule: with `s = 0` and `y = -1` the verdict is *right* but the rule
still updates, because the perceptron rule demands a margin.

## API

```python
spec = Perceptron(n_features, heads=1, cap=100, clip=3)
spec.plan()            # dict(dolls, sectors, fits): counts the map without building it
spec.reference()       # a fresh PerceptronRef twin (all weights 0)
spec.action_names()    # keys valid in compile(actions=...)
learner = spec.compile(wad=None, world=None, actions=None, resolution=None, configure=None)

scores, action = learner.step(x, y)     # one example; raises df.MismatchError on any difference
scores, action = learner.predict(x)     # step with all labels 0: no lesson, still a full checked run
learner.weights                         # (heads, n_features) read from the game (ceiling heights), not from the twin
learner.twin                            # the PerceptronRef (advances every step)
learner.info                            # dolls, sectors, runs, tics, steps, lessons, checked
learner.run_script(n, ..., read=k)      # your world scripts, as for any df.Brain
learner.close()                         # or use `with`
```

`Learner` is a `df.Brain` subclass (`kind == "perceptron"`), so `runner`, `run_script`, `close` and the context manager work
as in [`BUILD_YOUR_OWN.md`](BUILD_YOUR_OWN.md). `wad=None` keeps the map in the content-addressed cache; an explicit path
keeps it (and its `.json`) where you want it.

### What every step checks

After each run `step` reads back and compares, with tolerance 0: the scores `s{h}`, `act`, every `pick{k}`, every
`right{h}` / `wrong{h}`, and **all** weights `w{h}_{i}` *and* their negated twins `-w{h}_{i}` (the ceilings the forward
layer reads). That is `heads + 1 + (heads + 1) + 2*heads + 2*heads*n_features` values (`learner.info["checked"]`). A
difference raises `df.MismatchError` whose `.mismatches` lists `(name, DOOM, reference)`; the twin has advanced by then,
so after a mismatch the learner is only good for diagnosis. Inputs are validated *before* DOOM runs (`ValueError` /
`TypeError`).

### Hooks: `world` and `actions`

`world=` and `actions=` work as in `df.compile`. The outputs you can attach `df.act` specials to are
`pick0` (no head positive), `pick1..pickH` (head `k-1` is the first positive), `right{h}` and `wrong{h}`. Each receives +1
from exactly one doll per step when its condition holds, so its specials fire once per such step. **That doll** crosses
them: the decision is DOOM's. They fire on the *decision*, in the learn layer, before the weights have finished
updating. In example 12 `right0` sets a green lamp bright (and the red one dim) and `wrong0` the reverse; Python then
*reads* the lamps' light levels through a world script and checks them against the reference verdict.

## Limits and measured numbers (arm64, one DOOM process, other jobs running)

* Step time: 8 features x 1 head (40 dolls) about 75 ms; 19 x 2
  (165 dolls) about 650 ms; 32 x 2 (269 dolls, 7,862 sectors) about 1.8 s; all 30 steps of each were exact. A step is a full
  run of the map; there is no batching.
* Map size: `plan()` reports dolls and sectors, e.g. `Perceptron(8, 1)` = 40 dolls / 1,096 sectors, `Perceptron(5, 3)` = 78 / 2,137.
  The whole map (layout, closets and your world) must stay within 32,000 units per axis (`doomformer/pure.py`). A wider
  map makes ViZDoom's sector lookups wrong at some positions ([`engine_limits.md`](engine_limits.md)). In a
  58,000-unit layout that shows up as *lost weight updates* (DOOM keeps the old weight) in maps of about 75+ dolls, and
  the per-step check catches it. The same configurations are exact under the current layout
  (`tests/test_learnkit.py::test_bigger_maps_stay_exact`).
  `cap <= 1000` is enforced by the API (the default is 100; the demos stay within +-16).
* Integers only: features 0/1, labels -1/0/+1, weights in `[-cap, cap]`; each score satisfies `|s| <= n_features * clip`.
* If the map is built wrong (negative controls: no freeze
  barriers, no landing pads, a twin with the wrong mistake rule), the learner raises `MismatchError`.
  `tests/test_learnkit.py` builds those broken maps and checks that they are caught within a few steps.

## Examples and tests

* `examples/12_doom_learns_live.py`: ~9 s, exits nonzero on any mismatch, if the lamps do not match the verdicts, or if accuracy
  does not rise / drop on the flip / recover; writes `examples/out/12_doom_learns_live.png`.
* `tests/test_learnkit.py`: validation, multi-head exactness with `predict`, cap binding, a learned majority rule, a wrong twin,
  broken maps (negative controls), larger maps, and the `world`/`actions` hooks.
* The same machinery runs a random-teacher check with negative controls, and the arena marine that learns to aim (two
  heads, 19 features).
* `tests/test_learn_digits.py` (about 15 s): the twin against a textbook perceptron, validation, a tiny 4-class map exact in
  DOOM at opt 3 and opt 0 with the audit counts, blank ceilings read from the game, the three tamper lines diverging at the
  predicted step in exactly one weight, a host write to a weight rejected, and learned weights through the forward path.
