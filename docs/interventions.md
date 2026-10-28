# Causal interventions

`ablate=` zeroes units. The intervention API adds four more modes, per position and per head, and `df.patch_recovery`
turns single-cell patches into a recovery table. Every intervened run executes on voodoo dolls in the tile maps and is
compared with the equally intervened NumPy integer reference at tolerance 0.

```python
import doomformer as df
from doomformer import intervene as iv
from doomformer.intref import QModel, forward

m = QModel.load("doomplayer")                       # 1 layer, d=16, 2 heads, T=3
clean, corrupt = [0, 12, 23], [0, 16, 23]           # BOS, bearing i (aligned) or m (12 to 20 degrees left), distance 1
logits = forward(m, corrupt, interventions=[
    iv.resample("L0.v", clean, heads=[1], pos=[1]),          # head 1's value at position 1 from the clean run
    iv.scale("L0.x1", 1, 2, pos=[2]),                        # halve x1 at position 2
    iv.zero("L0.ao", units=[0, 3]),                          # zero two units (doors in DOOM)
])
table = df.patch_recovery(m, clean, corrupt, doom=True, workers=3)   # every (site, position, head) row, run in DOOM
print(table.text())
```

## Modes and sites

A site is an integer activation of the model, named as in `ablate=`: `x0`, `L{l}.ao`, `L{l}.x1`, `L{l}.m`, `L{l}.x2`, and
three more, `L{l}.q`, `L{l}.k`, `L{l}.v` (the counts after the q, k and v staircases). `ao`, `q`, `k` and `v` have `H`
heads of `dh` consecutive units: `heads=[h]` selects units `h*dh` to `(h+1)*dh - 1`. Give `pos=[...]` and `units=[...]` or
`heads=[...]`; the default is every position and every unit.

| mode | the activation becomes |
|---|---|
| `zero(site)` | 0 |
| `set_value(site, v)` | `v` (an int, or an array broadcast to [positions, units]) |
| `mean(site, clean, by="position")` | the rounded (half up) mean of that activation over the clean token lists, per position (`by="all"`: over all positions) |
| `resample(site, clean, src_pos=None)` | the activation of the clean run `clean` (token list) at `src_pos` (default: the same position) |
| `scale(site, num, den=1)` | `trunc(x * num / den)`, clipped to the site's range |

`set`, `mean` and `resample` are resolved to fixed integers before the run (the donor runs are plain `intref.forward`
calls); `scale` reads the run's own activation. `forward(..., interventions=[...])` applies them where `ablate=` zeroes
(before any consumer reads the activation), in the order given; `ablate=` and `interventions=` can be combined, the
zeros of `ablate=` come first. `intref.generate` takes the same argument. Positions past the end of a run are skipped,
so a prefix run agrees with the full run (causal). A count a staircase cannot output (outside its range, or a level
skipped because two thresholds are equal) raises `ValueError` in both the reference and DOOM. The gated MLP (`mlp="reglu"`)
supports `zero` at `m` only (m = g * u is a product of two gates). `ValueError` also for an unknown site, a layer the model
does not have, `units` and `heads` together, and units outside the site.

## How DOOM runs them

`DoomTransformer(model, tile_dir, interventions=[...])` (`use(...)` changes them between runs; `brain.intervene(...)` on a
tile brain from `df.compile(net, mode="tiles")`). The tile engine is the only place where Python sits between layers, so it
applies an intervention in one of two ways, and says which:

- **Doors.** `zero` at `x0`, `ao`, `x1`, `m`, `x2` closes the zone doors of the unit's blocks in every consuming tile,
  the mechanism of `ablate=`, now per position. Python tells the game which doors to close, then the dolls compute.
- **Bus patch.** Every other case (`set`, `mean`, `resample`, `scale`, `zero` at `q`, `k`, `v`, any mix at one site) is a
  patch of the exposed value. The dolls compute the pre-activation `z` of a tile output (`z0`, `zq`, `zk`, `zv`, `zo`, `z1`,
  `zm`, `z2`) and Python reads it back. For every unit whose activation changes, Python computes the staircase count of that
  integer, applies the intervention to the count, and writes `z'` (and `-z'`) into the ports of the consuming tiles, where
  `z'` is the smallest-magnitude integer with the new count (a threshold). The consumers (crossbars, staircases, products) are
  dolls. For softmax attention the `ao` of a unit is the count of the `P`/`N` comparison lanes the `out` tile reads; Python
  writes the lanes of the new count (`P_k = 1` for `k < v`, or `N_k = 1` for `k < -v`, the others 0).

What `check_tokens` compares at tolerance 0 for an intervened run: every exposed tensor and its negation twin (the
pre-activations the dolls computed, which depend on every patch upstream), the logits, and, for each patched cell, that the
count of the written value equals the reference's patched activation. The readbacks of patched cells are the values the
dolls computed before Python overwrote them. What it does not cover: that the patch value itself is the right one. A wrong
patch value gives an internally consistent wrong run; `tests/test_interventions.py` shows it: DOOM runs a patch one level
off, the check against the intended reference fails, and DOOM's logits differ from the intended reference's by exactly the
difference the reference predicts for the two patches (`predicted == got`), tensor by tensor.

Prefix reuse: `engine.forward(tokens, reuse=True)` keeps the longest held prefix with the same tokens and the same
interventions at those positions (each position stores a signature of the plan up to it) and runs only the rest; the whole
context is still checked.

Models: relu attention, softmax attention, RMSNorm and the gated MLP all run (`tests/test_interventions.py` and a scratch
sweep of every mode on `arch_model` in the three architectures of `tests/test_arch.py`: all exact). Python-written values are
the patched cells only.

## One map

A one-map brain has no host between its layers, so the only way in is the map itself.
`iv.fold(model, interventions)` returns a copy of the model with constant interventions compiled into its weights: zero, `set`
and `mean(by="all")` at every position, at `x0`, `ao`, `x1`, `m`, `x2` (not `scale`, not per-position values, not `q`/`k`/`v`,
not RMSNorm models). The consumers of an intervened unit read a fixed count, so column `u` of each consumer weight is zeroed and
`W[:, u] * v` moves into the consumer's bias (the residual weights `r1`, `r2` likewise). The folded model's pre-activations,
scores and logits equal the intervened reference's; `tests/test_interventions.py` compiles one folded tiny model into one map
and compares all exposed values with tolerance 0. The intervention is part of the compiled map; there is no runtime door.

The marine's own level takes the same route: `python -m demos.brainlevel --lesion 'x0:0;L0.ao:7;L0.x1:7;L0.m:0;L0.x2:0'`
folds the lesion (`x0` unit 0, `L0.ao` 7, `L0.x1` 7, `L0.m` 0, `L0.x2` 0: one of the 12 five-neuron pacifists of `doomplayer_onemap`
that the exhaustive search over its 48 neurons finds, none of 1 to 4 neurons exists: 3 minutes for sizes up to 5 on one core under
load) into `doomplayer_onemap` and builds the arena map around the lesioned transformer. `fold` equals the ablated reference on
all 86 observations (logits equal). In the level, `--decisions 14` ran 14 of 14 decisions EXACT (7,560 values at tolerance 0;
the level's clock sensed, the dolls decided, Python sent 178 console commands, all readout scripts, 0 others; 501 s with one game
under load). The lesion changed the chosen action in 0 of these 14 decisions (the marine only turned right: the demon was 60 to
180 degrees to his right and never in front), so this run shows the lesioned brain computing exactly inside his level, not a
changed behaviour. The lesion is a different compiled map, not a door closed while the level runs; a runtime door in a one-map
level is not built (the
doors of the tile engine exist only in tile maps).

## Patching with recovery

`df.patch_recovery(model, clean, corrupt, sites, answer=None, direction="denoising", doom=False)` runs the corrupt tokens
and writes in the clean run's activation of one cell (a site, a position, units or a head), one cell per row. Metric:
`logit[a] - logit[b]` at the last position (default: the argmax of the clean run and of the corrupt run). Recovery:
`(metric(patched) - metric(corrupt)) / (metric(clean) - metric(corrupt))`; `direction="noising"` patches corrupt into
clean and reports `(metric(clean) - metric(patched)) / (metric(clean) - metric(corrupt))`. `doom=True` runs every row in DOOM,
`doom=N` the N rows with the largest recovery, an iterable the rows with those indices. `patching.grid(m, ...)` builds the
row list (`per_head=True`, `per_unit=True`, `positions=[...]`). Rows are single-cell patches; path patching is not provided.

```sh
python -m doomformer.patching doomplayer --clean i1 --corrupt m1 --answer FL --per-head --pos 1,2 --doom all --workers 3
python -m doomformer.patching doomplayer_xl --clean i1 --corrupt m1 --answer FL --per-head --pos 1,2 --site L0.v --site L1.ao --doom all --workers 3
```

`doomplayer`, clean `i1` (aim aligned, near: fires), corrupt `m1` (12 to 20 degrees left: turns left), metric
`logit[F] - logit[L]` = 2,881 (clean) and -3,072 (corrupt). Rows with a nonzero recovery (the other 16 of the 24 rows recover
0 and are in the log); all 24 rows ran in DOOM:

| site | position | cells | cells changed by the patch | recovery | in DOOM |
|---|---|---|---|---|---|
| `x0` | 1 | all units | 15 | 1.000 | EXACT, 546 values |
| `L0.v` | 1 | head 0 | 8 | 0.698 | EXACT, 539 values |
| `L0.v` | 1 | head 1 | 6 | 0.829 | EXACT, 537 values |
| `L0.ao` | 2 | head 0 | 6 | 0.698 | EXACT, 537 values |
| `L0.ao` | 2 | head 1 | 6 | 0.829 | EXACT, 537 values |
| `L0.x1` | 2 | all units | 14 | 1.000 | EXACT, 545 values |
| `L0.m` | 2 | all units | 12 | 0.911 | EXACT, 543 values |
| `L0.x2` | 2 | all units | 13 | 1.000 | EXACT, 544 values |

The bearing token enters through the values at position 1 of both heads, and each head's output at the last position carries
0.70 and 0.83 of the logit difference (the two sum above 1: the heads are redundant). Patching q, k, x0 at
position 2, or x1, m, x2, ao at position 1 recovers nothing. 24 of 24 rows EXACT in 320 s wall time with
3 games under load average 43 (log: [`models/interventions/patch_doomplayer.txt`](../models/interventions/patch_doomplayer.txt)).

`doomplayer_xl` (2 layers, d=32, 4 heads), same pair, metric 3,997 (clean) and -4,147 (corrupt), the rows of `L0.v` and `L1.ao`
(positions 1, 2, every head), all 16 ran in DOOM, 16 of 16 EXACT in 718 s wall time (3 games, same load): `L0.v` position 1 head
0 recovers 0.565 and head 3 0.573, `L1.ao` position 2 head 2 0.435, head 0 0.159, head 1 0.099, the rest 0.

## The induction circuit

`models/induction` (2 layers, d=16, 2 heads, softmax attention, MLP 32, T=10; 99.95% of the queried tokens of the 4,000 held-out
sequences of `doomformer/induction.py` are right, `models/induction/train_info.json`) copies the token that followed the first occurrence of the current token. Two pairs of sequences
test two paths (`doomformer/circuit.py`): "answer" pairs swap the answer token with another prefix token (every position keeps
its previous token, so keys and attention patterns are the same in both runs: this tests what is copied), "source" pairs
swap the first occurrence of the query token with another prefix token (the heads must find a new source: this tests how it is
found). Positions are named by role (`query` the last token, `src` j the first occurrence, `ans` j+1, `dist`/`src2`/`ans2` the
swapped positions, `bos`, `rest`). Mean recovery of one cell patched from clean into corrupt, 200 pairs of each kind from the
held-out sequences (seed 0), `python -m demos.circuit` (about 1.5 min on one core under load); cells below 0.05 everywhere
are omitted:

| cell | answer pairs | source pairs |
|---|---|---|
| `L1.ao` head 0, head 1 at `query` | 0.55, 0.47 | 0.52, 0.47 |
| `L1.x1`, `L1.x2` at `query` | 1.00, 1.00 | 1.00, 1.00 |
| `L1.v` head 0, head 1 at `ans` | 0.56, 0.47 | below 0.05 |
| `L0.x1`, `L0.x2` at `ans` | 1.01, 1.01 | 0.50, 0.50 (and 0.49 at `ans2`) |
| `L0.ao` head 0, head 1 at `ans`, `ans2` | below 0.05 | 0.22, 0.06 at `ans`; 0.35, 0.13 at `ans2` |
| `L0.v` head 0, head 1 at `src`, `src2` | below 0.05 | 0.23, 0.06 at `src`; 0.36, 0.13 at `src2` |
| `L1.k` head 0, head 1 at `ans`, `ans2` | below 0.05 | 0.21, 0.19 at `ans`; 0.20, 0.16 at `ans2` |
| `L0.m` at `ans` / `L1.m` at `query` | 0.37 / 0.37 | 0.01 / 0.35 |

Reading: the two layer-1 heads put the answer into the last position (their outputs carry 0.55 and 0.47 of the logit
difference, together 1.02: they are redundant), they read it from the value at `ans` (0.56 and 0.47), and the layer-1 keys at
the positions after a candidate source decide where they look. The source pairs show who writes that key: the layer-0 value of
head 0 at the moved token (`src`, `src2`: 0.23 and 0.36) reaches the position after it through head 0's output (`ans`, `ans2`:
0.22 and 0.35). Head 0 of layer 0 carries most of the previous-token effect; head 1 about a third of it. This matches
`python training/analyze_induction.py models/induction` (2,760 queried tokens: the layer-0 heads put 59.5% and 53.0% of their
attention weight on position t-1, the layer-1 heads 93.6% and 94.0% on position j+1). Single cells recover part of the effect because the circuit is redundant.

Circuit search (`demos.circuit`): every subset of the 4 heads and 2 MLP layers is kept and everything else is mean-ablated at
every position (head outputs and MLP hidden units replaced by their rounded mean over the 200 clean sequences, per position).
Share of the logit difference restored, best subset of each size: 0 components 0.000, 1 (L0.MLP) 0.001, 2 (L0.H0, L1.H0) 0.083,
3 (+ L0.MLP) 0.347, 4 (+ L1.MLP) 0.580, 5 (L0.H0, L1.H0, L1.H1, both MLPs) 0.874 (top-1 0.94), all 6 components 1.000. No
proper subset reaches 0.9: under mean ablation every head and every MLP layer is needed. The circuit is the model.

In DOOM, pair 0 of each kind (an 8-token context; 3 games, load average above 30; logs in
[`models/interventions/`](../models/interventions/): `circuit_doom2.txt`, `circuit_doom3.txt`, `circuit_ref.txt`):

- `python -m demos.circuit --doom-rows 4` patches the four cells with the largest recovery of each kind: 8 of 8 rows EXACT,
  110 patched cells, about 9,270 values compared per run (tolerance 0), 15 to 110 s per run. Answer pairs: `L0.x1` and `L0.x2` at
  `ans` (1.021; 14 and 12 cells), `L1.x1` and `L1.x2` at `query` (1.000; 15 cells each). Source pairs: `L1.x1` and `L1.x2` at
  `query` (1.000), `L1.ao` head 0 at `query` (0.613; 8 cells), `L0.x1` at `ans` (0.282; 15 cells).
- `python -m demos.circuit --doom-circuits 2` runs the best proper subset (L0.H0, L1.H0, L1.H1, both MLPs: L0.H1 mean-ablated
  at every position) and the all-ablated model on two pairs: 4 of 4 runs EXACT, 807 patched cells (44 to 46 for the circuit,
  354 to 363 for all-ablated), 37,831 values at tolerance 0, 9.8 min wall time.

The same command on `models/induction_relu` (relu attention, same shape; reference only, 1.7 min, 200 pairs per kind, seed 0):
the layer-1 heads carry 0.57 and 0.44 of the logit difference at `query` and read it from the value at `ans` (0.57, 0.44), as above.
The previous-token role moves to layer-0 head 1 (source pairs: its value at `src`/`src2` 0.47/0.48, its output at `ans`/`ans2`
0.47/0.48), and head 0 only matters for answer pairs (0.34). Mean-ablating everything outside heads and MLPs is gentler here:
the best 4 components (all four heads) restore 0.810, adding the layer-1 MLP gives 0.949 (top-1 0.99, full model 1.00), and the
layer-0 MLP is the only component outside the 0.9 circuit. Which of the two layer-0 heads is the previous-token head differs
between the two trained models. These relu numbers were not run in DOOM.

## Lesions

`demos.lobotomy --minimal --null` and `--names` use the exact batch evaluator `doomformer/lesion.py` for the minimal pacifist,
the hit rates per site, the shuffled-weight null and the name model; the numbers and commands are in
[DEMOS.md, Lesion statistics](DEMOS.md#lesion-statistics). The minimal lesion runs in DOOM as before (`ablate=`, doors).

## Limits

- Interventions are on the integer activations the tile engine exposes. `fold` is the only route into a one-map brain and
  it is a compile-time change. There is no in-map runtime door or switch for one-map models.
- `scale` and the per-position values are not foldable; the gated MLP supports `zero` at `m` only.
- A row takes 6 to 21 s for `doomplayer` and 25 to 77 s for `doomplayer_xl` (3 games, load average 43; the tile maps are the
  cost, not the interventions).
- The patch value is Python's choice and is checked only through its effects (above).
- Rows are single cells. Path patching is not provided; a path patch is a set of resolved values you can pass as
  `set_value` arrays.
