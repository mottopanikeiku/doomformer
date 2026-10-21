# DOOM learns: `df.Perceptron`

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
