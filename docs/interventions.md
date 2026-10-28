# Causal interventions

`ablate=` zeroes units. The intervention API adds more modes, per position and per head, `df.patch_recovery` turns
single-cell patches into a recovery table, `df.probe` fits a rank-1 direction to steer with, and a one-map level can close
doors on its own clock. Every intervened run executes on voodoo dolls and is compared with the equally intervened NumPy
integer reference at tolerance 0.

What runs where, in one sentence: Python rewrites tile ports and the dolls compute the rest. In tile mode Python sits
between the layers, closes doors or writes the patched values into the ports, and the dolls do every product, sum and
staircase; in a one-map level there is no Python between layers, and the level's own clock closes the doors
([One map](#one-map)).

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
| `add_direction(site, u, alpha=1)` | `x + alpha * u`, clipped to the site's range; `u` is an integer vector with one entry per selected unit |
| `project_out(site, u)` | `x - c * u` with `c = trunc((x . u) / (u . u))`, clipped; afterwards `|x' . u| < u . u`, so the component is removed up to one integer step |

`set`, `mean` and `resample` are resolved to fixed integers before the run (the donor runs are plain `intref.forward`
calls); `scale`, `add_direction` and `project_out` read the run's own activation. `forward(..., interventions=[...])` applies them where `ablate=` zeroes
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
- **Bus patch.** Every other case (`set`, `mean`, `resample`, `scale`, `add_direction`, `project_out`, `zero` at `q`, `k`, `v`, any mix at one site) is a
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

A one-map brain has no host between its layers, so the way in is the map itself. There are two routes.

**Runtime lane doors: the level closes them.** `iv.lane_doors(model, ablate)` names the doors for a set of units, and
`build_onemap(..., doors=...)` (`compile_pure(doors=...)`) builds them: every consumer lane that reads such a unit (a doll walks
to a gate whose height is the unit's pre-activation; one lane per threshold, position and consumer) gets a door in its pad, a
sector with the unit's door tag (UDMF `moreids`) whose ceiling stands 128 units above the walkway. Lowering that ceiling to the
floor stops the lane's doll, the lane adds nothing, and the unit is zero-ablated at every consumer: the same arithmetic as
`ablate=`. Maps built without `doors=` are byte for byte the same as before (sha256 of the wad and json of a tiny one-map
brain at opt 0 and opt 3, compared with the v0.5.0 tree). The default map has no doors; a map with doors differs from it in the
ceiling heights and the extra id of those pads.

The closing is level logic, with two trigger kinds, both compiled into the level's clock script. A decision counter:
`python -m demos.brainlevel --lesion SPEC --lesion-at N`; at the top of decision N, right after the rewind has taken every doll
home, the clock lowers the armed ceilings (`Ceiling_LowerByValue`), sets a flag the status readout shows, and from then on the
same map computes the masked brain. A sensed event: `--lesion-when SPEC`, where SPEC is `near` (the nearest live demon is within
200 units: the distance token the clock just sensed is '0' or '1'), `ahead` (the bearing token is 'i', in the sights) or both
joined by `+`. Right after the clock's own `bl_sense()` of each decision (after the rewind) the script tests the condition on
`bl_bearing` / `bl_distance`, the tokens it feeds the transformer; the first time it holds it lowers the doors and latches
(`bl_doors`): they stay closed whatever the tokens do later. The all-or-none limit below applies to both.
Python never sends either trigger: it advances tics and runs readout scripts, as in the healthy run. Per decision it checks every
exposed value against the healthy reference before the doors close and against the masked reference (`forward(..., ablate=lesion)`)
from then on, and checks the flag the level reports: for the counter it must equal `decision >= N`; for the sensed event Python
recomputes the condition from the sensed tokens (the ones it has already checked against `policy.observe`) and the latch must
equal "the condition held at this or an earlier decision". The doors close after the rewind because of a probe on a
tiny model: doors closed right after a run, without a rewind, gave a next run that differed from the masked reference in 23
values; with a rewind between the run and the closing every value matched
(`tests/test_interv2.py::test_lane_doors_close_between_decisions_and_match_the_masked_reference`). The cause is probably a
doll that stayed blocked at its gate inside the pad sector when the ceiling came down (not measured).

The run: `doomplayer_onemap`, the lesion `x0:0;L0.ao:7;L0.x1:7;L0.m:0;L0.x2:0` (one of the 12 five-neuron pacifists of the
exhaustive search over its 48 neurons: none of 1 to 4 neurons exists; 78 lanes carry its doors), seed 2, closing before
decision 3. How the seed and the decision were chosen: healthy runs of seeds 1, 2, 3 (10 decisions each, `--opt 3`, no lesion;
`models/interventions/brainlevel_healthy_seed{1,2,3}_opt3.txt`):
seeds 1 and 3 turn right for all 10 decisions, seed 2 has the demon in his sights at decision 3 and the
healthy brain fires (1 kill). The pacifist turns that context, and every other 'in sights' context, to TURN RIGHT (reference scan
over `policy.all_contexts()`), so the door closes in the middle of the fight, on the decision where the healthy brain pulls the
trigger. Result of `python -m demos.brainlevel --seed 2 --decisions 12 --opt 3 --lesion 'x0:0;L0.ao:7;L0.x1:7;L0.m:0;L0.x2:0'
--lesion-at 3` (log: [`models/interventions/brainlevel_lesion_at3_seed2_opt3.txt`](../models/interventions/brainlevel_lesion_at3_seed2_opt3.txt)):

| decisions | doors | what he sees | healthy brain | the level's brain | values at tolerance 0 |
|---|---|---|---|---|---|
| 1, 2 | open | demon 100 to 180 degrees right, then 12 to 20 left | TURN RIGHT, TURN LEFT | the same | 540 each, EXACT |
| 3 | closed before it | demon in his sights, far | FIRE | TURN RIGHT | 540, EXACT against the masked reference |
| 4 to 12 | closed | the aim swings between 'in sights' and 12 to 20 left | FIRE on every 'in sights' | TURN RIGHT there, TURN LEFT otherwise | 540 each, EXACT |

12 of 12 decisions EXACT (6,480 values), the action differs from the healthy brain's on the decisions 3, 5, 7, 9 and 11, 0 kills
(the healthy run of the same seed kills a demon at decision 3), Python sent 154 console commands, all readout scripts, 0 others;
73 s wall at 6.1 s per decision (`--opt 3`: 738 compute tics per decision, load average about 20). The same command with the
default `--opt 0` ([`..._opt0.txt`](../models/interventions/brainlevel_lesion_at3_seed2_opt0.txt)): 12 of 12 EXACT, the same
actions, 4,302 compute tics, 494 s (41 s per decision). Control: the same map with the doors armed and never closed
(`--lesion-at 99`, 6 decisions, [`brainlevel_doors_never_closed_seed2_opt3.txt`](../models/interventions/brainlevel_doors_never_closed_seed2_opt3.txt)):
6 of 6 EXACT against the healthy reference, the marine fires and kills at decision 3.

The sensed event, same lesion, same map and `--opt 3` (logs in `models/interventions/`, linked below). How the runs were chosen:
the reference scan over `policy.all_contexts()` gives, for this lesion, every 'in sights' context FIRE to TURN RIGHT, and
changes the action in 14 of the 34 contexts whose distance token is '0' or '1' (mostly FIRE to TURN LEFT or TURN RIGHT). For
`ahead`, seed 2 again, because its healthy run meets the first 'in sights' demon at decision 3. For `near`, the first healthy run
with a near demon among the logged runs is seed 4 (`models/brainlevel_opt/opt3_seed4_50.log`: first 'near' at decision 22,
bearing 4 to 7 degrees left, healthy FIRE), and the scan turns that context to TURN LEFT.

| run | event | doors open for | action differs from the healthy brain's at | result |
|---|---|---|---|---|
| seed 2, `--lesion-when ahead`, 12 decisions ([log](../models/interventions/brainlevel_lesion_when_ahead_seed2_opt3.txt)) | in the sights, decision 3 | decisions 1, 2 | 3, 5, 7, 9, 11 | 12 of 12 EXACT, 6,480 values, 0 kills, 0 non-readout commands |
| seed 4, `--lesion-when near`, 30 decisions ([log](../models/interventions/brainlevel_lesion_when_near_seed4_opt3.txt)) | first near demon, decision 22 | decisions 1 to 21 | 22, 24, 27, 28, 29, 30 | 30 of 30 EXACT, 16,200 values, 0 kills (the healthy run kills one at decision 22), 0 non-readout commands |
| seed 2, `--lesion-when near`, 12 decisions ([log](../models/interventions/brainlevel_lesion_when_never_seed2_opt3.txt)) | never (no demon within 200 units in these decisions) | all 12 | none | 12 of 12 EXACT against the healthy reference, the marine fires and kills at decision 3, 0 actions changed |

Before the event every value equals the healthy reference with the doors open, from the event decision on every value equals the
masked reference. In the second run the event comes 21 decisions in, when a demon has walked up to the marine; the decision
number was not chosen in advance. The flip is on the decision where the event is sensed: the clock senses the tokens, the
doors close, and the dolls compute the masked brain on those same tokens. 5.9 s per decision (738 compute tics); 71 s, 176 s and
74 s wall, load average 5 to 20. The 'never' control only shows the doors stay open when the condition is false in these 12
decisions; the condition is a function of the two sensed tokens, nothing else.

What this does not show: the closing is all-or-none (one clock line lowers every armed door; one tag per unit exists, so a
staggered or conditional closing is more ACS lines, not run); doors only zero (`set`, `mean`, `scale`, directions need the value
and are not available in a level); relu attention, no norm and the relu MLP only; which units can close is decided when the map
is built (the doors are in it), when they close is decided by the level; one seed and one lesion; the demon and the marine's
arena are the level's, not a benchmark.

**Fold: a different map.** `iv.fold(model, interventions)` returns a copy of the model with constant interventions compiled
into its weights: zero, `set` and `mean(by="all")` at every position, at `x0`, `ao`, `x1`, `m`, `x2` (not `scale`, directions,
per-position values, `q`/`k`/`v`, RMSNorm models). The consumers of an intervened unit read a fixed count, so column `u` of each
consumer weight is zeroed and `W[:, u] * v` moves into the consumer's bias (the residual weights `r1`, `r2` likewise). The
folded model's pre-activations, scores and logits equal the intervened reference's; `tests/test_interventions.py` compiles one
folded tiny model into one map and compares all exposed values with tolerance 0. `python -m demos.brainlevel --lesion SPEC`
without `--lesion-at` uses it: the lesion is part of the compiled map and there is no runtime event. Its earlier run
(`--decisions 14`, seed 4, [`brainlevel_lesion14.txt`](../models/interventions/brainlevel_lesion14.txt)) was 14 of 14 EXACT (7,560
values) and changed 0 actions, because the demon stayed 60 to 180 degrees to his right.

## Patching with recovery

`df.patch_recovery(model, clean, corrupt, sites, answer=None, direction="denoising", doom=False)` runs the corrupt tokens
and writes in the clean run's activation of one cell (a site, a position, units or a head), one cell per row. Metric:
`logit[a] - logit[b]` at the last position (default: the argmax of the clean run and of the corrupt run). Recovery:
`(metric(patched) - metric(corrupt)) / (metric(clean) - metric(corrupt))`; `direction="noising"` patches corrupt into
clean and reports `(metric(clean) - metric(patched)) / (metric(clean) - metric(corrupt))`. `doom=True` runs every row in DOOM,
`doom=N` the N rows with the largest recovery, an iterable the rows with those indices. `patching.grid(m, ...)` builds the
row list (`per_head=True`, `per_unit=True`, `positions=[...]`). Rows are single-cell patches; path patching is not provided.

Rows that rewrite a whole site where the metric reads carry a `flag` (`row["flag"]`, a `[...]` note in `table.text()`) and
are not findings. The rule: a patch of every unit of a site (every head together for `ao`, `q`, `k`, `v`). `tautological`:
the last layer's `x1` or `x2` at the last position, where the metric reads: the logits are a function of that one vector, so
writing the donor's copy restores the donor's logits and the recovery is 1 by construction. `whole residual`: every unit of
`x0`, `x1` or `x2` at any other layer or position: the patch writes everything that position holds, so it bounds what a
smaller patch there can recover and says where the information is, not which units carry it. `whole site`: every unit of
`ao`, `m`, `q`, `k` or `v` at the last position: it rewrites the site's whole contribution at the read position (`L0.m` of a
one-layer model, 0.911; induction `L1.m` at `query`, 0.37), near-tautological. The same rows at an earlier position are not
flagged: they say which position carries the effect. A patch of some units, or of one head, is never flagged.
`table.headline` lists the unflagged rows; counts and headline numbers use them only, and `doom=N` picks the N rows with the
largest recovery among the unflagged first.

```sh
python -m doomformer.patching doomplayer --clean i1 --corrupt m1 --answer FL --per-head --pos 1,2 --doom all --workers 3
python -m doomformer.patching doomplayer_xl --clean i1 --corrupt m1 --answer FL --per-head --pos 1,2 --site L0.v --site L1.ao --doom all --workers 3
```

`doomplayer`, clean `i1` (aim aligned, near: fires), corrupt `m1` (12 to 20 degrees left: turns left), metric
`logit[F] - logit[L]` = 2,881 (clean) and -3,072 (corrupt). Rows with a nonzero recovery (the other 16 of the 24 rows recover
0 and are in the log); all 24 rows ran in DOOM. Four of the eight rows rewrite a whole site and are flagged:

| site | position | cells | cells changed by the patch | recovery | in DOOM |
|---|---|---|---|---|---|
| `x0` | 1 | all units | 15 | 1.000 | EXACT, 546 values; whole residual |
| `L0.v` | 1 | head 0 | 8 | 0.698 | EXACT, 539 values |
| `L0.v` | 1 | head 1 | 6 | 0.829 | EXACT, 537 values |
| `L0.ao` | 2 | head 0 | 6 | 0.698 | EXACT, 537 values |
| `L0.ao` | 2 | head 1 | 6 | 0.829 | EXACT, 537 values |
| `L0.x1` | 2 | all units | 14 | 1.000 | EXACT, 545 values; tautological |
| `L0.m` | 2 | all units | 12 | 0.911 | EXACT, 543 values; whole site |
| `L0.x2` | 2 | all units | 13 | 1.000 | EXACT, 544 values; tautological |

The headline rows are the four unflagged ones (heads of `L0.v` and `L0.ao`). The bearing token enters through the values at position 1 of both heads, and
each head's output at the last position carries 0.70 and 0.83 of the logit difference (the two sum above 1: the heads are
redundant). Patching q, k, x0 at position 2, or x1, m, x2, ao at position 1 recovers nothing. 24 of 24 rows EXACT in 320 s wall
time with 3 games under load average 43 (log: [`models/interventions/patch_doomplayer.txt`](../models/interventions/patch_doomplayer.txt);
the log predates the flags, the recoveries and DOOM results are unchanged).

`doomplayer_xl` (2 layers, d=32, 4 heads), same pair, metric 3,997 (clean) and -4,147 (corrupt), the rows of `L0.v` and `L1.ao`
(positions 1, 2, every head), all 16 ran in DOOM, 16 of 16 EXACT in 718 s wall time (3 games, same load): `L0.v` position 1 head
0 recovers 0.565 and head 3 0.573, `L1.ao` position 2 head 2 0.435, head 0 0.159, head 1 0.099, the rest 0.

## A planted circuit

`doomformer/groundtruth.py` builds, rather than trains, a one-layer integer transformer whose algorithm is known: on
`[BOS, a, b, c]` (digits 0 to 3) the logits at the last position one-hot `(b + c) mod 4`. The weights are 0 and 1 (saved in
`models/planted`; `python -m doomformer.groundtruth --save DIR` writes them). Head 0 attends to position `t - 1` (the query
one-hots `t - 1`, the key one-hots `s`) and copies that position's token one-hot into a block of the residual stream called
PREV; the current token reaches the MLP through the residual weight `r1`; MLP unit `4a + b` fires when PREV is `a` and the
current token is `b` and adds 1 to the output dimension `(a + b) mod 4`. Head 1 is a decoy: it attends to `t - 2`, copies that
token, and its output goes to dimensions that nothing reads. The planted structure is therefore: the token at `t - 1` reaches
`t` only through head 0, the token at `t` only through the residual stream, and every combination happens in the MLP. The model
gives the algorithm's answer on all 64 contexts and passes `check_limits`.

`planted_cells(kind, clean, corrupt)` derives from this construction alone which cells (site, position, unit) a
clean-into-corrupt patch can recover, for pairs that change `b` ("prev") or `c` ("cur"). `check_recovery` patches every unit
and every head of every site and position singly (`patch_recovery`, denoising) for every such pair and compares the set of
cells with nonzero recovery with the planted set. `python -m doomformer.groundtruth` (full run, reference only: 1.5 min on one
core under load):

| pairs | cells tested | planted | nonzero recovery | disagreements |
|---|---|---|---|---|
| "prev": 192 (every `a`, `b`, `c`, and a changed `b`) | 104,448 | 2,688 | 2,688 | 0 |
| "cur": 192 | 104,448 | 1,536 | 1,536 | 0 |
| "decoy": 192 (a changed `a`, read only by head 1) | head 1's output changes in 192 pairs | | | the logits change in 0 |

The recovery table over the "prev" pairs (levels: every single unit of the site, or one head; the other rows recover 0 in every
pair):

| site | position | level | planted cells per pair | mean recovery of the nonzero cells |
|---|---|---|---|---|
| `x0` | 2 | unit | 2 (the old and the new token) | 0.50 |
| `L0.v` | 2 | head 0 | the head | 1.00 |
| `L0.v` | 2 | unit | 2 | 0.50 |
| `L0.ao` | 3 | head 0 | the head | 1.00 |
| `L0.ao` | 3 | unit | 2 | 0.50 |
| `L0.x1` | 3 | unit | 2 | 0.50 |
| `L0.m` | 3 | unit | 2 (units `4b + c` and `4b' + c`) | 0.50 |
| `L0.x2` | 3 | unit | 2 | 0.50 |

Head 1 recovers nothing at any site or position, `q` and `k` recover nothing (the patterns do not change), and the "cur" pairs
recover through `x0`, `x1`, `m`, `x2` at position 3 only: no head. A single unit recovers 0.50 because the corrupt run keeps its
own active unit next to the patched one (a tie in the logits); a whole head or site recovers 1.00. The decoy is inert by
construction (its output is written where nothing reads), so "head 1 recovers 0" tests that patching does not invent a role for an
active head; it does not test a hard case. Mean-ablation circuit search over the three components (`circuit_check`, the 64
contexts): keeping H0 and the MLP restores 1.000 of the logit difference (top-1 1.000), keeping H1 and the MLP 0.000, any
subset without both H0 and the MLP 0.000: the smallest faithful circuit is 2 of 3 components and it is the planted one.

In DOOM (log: [`models/interventions/planted_circuit.txt`](../models/interventions/planted_circuit.txt)): `--doom 3 --doom-off 10`
runs, for 3 "prev" and 3 "cur" pairs, every planted row (units and heads: 14 per "prev" pair, 8 per "cur" pair) and 10 unplanted
rows chosen at random, in tile maps with Python patching the bus: 126 of 126 rows EXACT at tolerance 0 (564 to 566 values each).
`--doom-circuit 64` runs the circuit-only model (H0 and the MLP kept, H1 mean-ablated at every position) and the all-ablated
model on every one of the 64 contexts: 128 of 128 runs EXACT at tolerance 0, 72,448 values, 256 patched cells (42 min wall
time with 2 games under load average about 35; log:
[`planted_circuit_doom64.txt`](../models/interventions/planted_circuit_doom64.txt)). The faithful subset below the whole model
(2 of 3 components, faithfulness 1.000) is therefore verified in DOOM on all 64 contexts, not on a sample; the reference
(`circuit_check`) gives the answer `(b + c) mod 4` for the circuit on 64 of 64 contexts and for the all-ablated model on none.

Limits: the circuit is tiny, hand-built and for one algorithm; it tests the recovery method against a known answer, it says
nothing about trained models, where the induction circuit below is the harder case. The expected set comes from the
construction by hand-written rules; the agreement is the check on those rules and on patching together.

### The same circuit with a backup head

`planted_model(backup=True)` (`python -m doomformer.groundtruth --backup`, saved in `models/planted_backup`) makes head 1 a
backup of head 0: it also attends to `t - 1`, copies that token and writes it to PREV too (the staircase saturates, so either
head alone gives the same `x1`). Everything else is as above. The planted rule extends (`planted_cells(..., backup=True)`):
a single patch into one head's `v` or `ao` leaves the other head writing the corrupt token `b'`, so PREV holds both tokens and
the logits tie; the unit of the new token `b` in that head (`b` in head 0, `8 + b` in head 1) recovers 0.50, the unit of `b'`
patched to 0 recovers 0 (the other head still writes it), and the cells both heads share (`x0` at 2, `x1`, `m`, `x2` at 3)
are as before. Log: [`planted_backup.txt`](../models/interventions/planted_backup.txt).

| | one head, decoy head 1 | backup head 1 |
|---|---|---|
| single-cell patches, 192 "prev" pairs: planted / nonzero / disagreements | 2,688 / 2,688 / 0 | 3,072 / 3,072 / 0 |
| `L0.v` at 2, `L0.ao` at 3, head 0 alone | 1.00 | 0.50 |
| the same, head 1 alone | 0 | 0.50 |
| both heads patched together | 1.00 | 1.00 |
| zero-ablating head 0 alone, top-1 on the 64 contexts | 0.000 | 1.000 |
| zero-ablating head 1 alone | 1.000 | 1.000 |
| circuit search: faithful subsets | H0 + MLP only | H0 + MLP and H1 + MLP, 1.000 each |

Single-cell patching and single-head knock-out are misleading here: each head gets half of the credit, neither head looks
necessary, and the table reads "two heads that each do half" when the truth is "either head does all of it". Only the joint patch
(1.00), the circuit search (two minimal circuits of 2 of 3 components, `H0 + H1` alone 0) and the ablation of both heads show
the redundancy; nothing in the search ranks the two circuits, and the 0.50 values carry no information about which head the
model "uses" (both do). The same reading applies to the trained models: the two heads of `doomplayer` recover 0.70 and 0.83 of
the logit difference, together more than 1, and the induction heads recover about half each; those numbers are credit splitting
under redundancy, not a ranking of heads.

DOOM (tile maps, 3 games, Python patching the bus): `--doom 2 --doom-off 8` runs, for 2 "prev" and 2 "cur" pairs, every planted
row (16 per "prev" pair, 8 per "cur" pair) and 8 random unplanted rows: 80 of 80 rows EXACT at tolerance 0. `--doom-circuit 8`
runs both minimal circuits (H0 + MLP with H1 mean-ablated, H1 + MLP with H0 mean-ablated) and the all-ablated model on the
first 8 contexts: 16 of 16 runs EXACT for each circuit (9,072 values, 48 patched cells each). Sample sizes are smaller than for
the decoy model (8 contexts, not 64).

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
| `L1.x1`, `L1.x2` at `query` | 1.00, 1.00 (tautological) | 1.00, 1.00 (tautological) |
| `L1.v` head 0, head 1 at `ans` | 0.56, 0.47 | below 0.05 |
| `L0.x1`, `L0.x2` at `ans` | 1.01, 1.01 (whole residual) | 0.50, 0.50 (and 0.49 at `ans2`) (whole residual) |
| `L0.ao` head 0, head 1 at `ans`, `ans2` | below 0.05 | 0.22, 0.06 at `ans`; 0.35, 0.13 at `ans2` |
| `L0.v` head 0, head 1 at `src`, `src2` | below 0.05 | 0.23, 0.06 at `src`; 0.36, 0.13 at `src2` |
| `L1.k` head 0, head 1 at `ans`, `ans2` | below 0.05 | 0.21, 0.19 at `ans`; 0.20, 0.16 at `ans2` |
| `L0.m` at `ans` / `L1.m` at `query` | 0.37 / 0.37 (the second is a whole site) | 0.01 / 0.35 (the second is a whole site) |

Flagged rows rewrite a whole site (`doomformer.patching.row_flag`): `L1.x1`, `L1.x2` at `query` are the last
layer's residual at the position the metric reads (tautological, 1 by construction), `L0.x1`, `L0.x2` at `ans` carry everything
position `ans` holds after layer 0 (whole residual: the answer reaches the last position through `ans`, not which units carry
it), and `L1.m` at `query` rewrites every MLP unit at the read position (whole site; `L0.m` at `ans` is not at the read position
and is not flagged). They are not findings and are left out of the readings below and of the DOOM row choice (log of the
reference run with the flags: [`circuit_ref_flags.txt`](../models/interventions/circuit_ref_flags.txt), marks T, W, S).

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

In DOOM. First run (pair 0 of each kind, an 8-token context; 3 games, load average above 30; logs in
[`models/interventions/`](../models/interventions/): `circuit_doom2.txt`, `circuit_doom3.txt`, `circuit_ref.txt`; these runs
predate the flags and included the whole-residual cells):

- `python -m demos.circuit --doom-rows 4` patches the four cells with the largest recovery of each kind: 8 of 8 rows EXACT,
  110 patched cells, about 9,270 values compared per run (tolerance 0), 15 to 110 s per run. Answer pairs: `L0.x1` and `L0.x2` at
  `ans` (1.021; 14 and 12 cells), `L1.x1` and `L1.x2` at `query` (1.000; 15 cells each). Source pairs: `L1.x1` and `L1.x2` at
  `query` (1.000), `L1.ao` head 0 at `query` (0.613; 8 cells), `L0.x1` at `ans` (0.282; 15 cells).
- Ten pairs per kind, unflagged cells only: `python -m demos.circuit --doom-rows 2 --doom-pairs 10` patches, on pairs 0 to 9 of
  the 200 (seed 0), the two unflagged cells with the largest mean recovery of the kind. Answer pairs, `L1.v` head 0 at `ans` and
  `L1.ao` head 0 at `query` (mean recovery 0.56 and 0.55 over the 200): 20 of 20 runs EXACT, 6 to 8 patched cells per run, 146
  patched cells and 209,206 values at tolerance 0 in total, 99 to 294 s per run (one game, load average 5 to 35; log
  [`circuit_doom_10pairs.txt`](../models/interventions/circuit_doom_10pairs.txt)). Source pairs (the `--doom-kind source` run), `L1.ao` head 0 and head 1 at `query` (mean recovery 0.52 and 0.47): 20 of 20
  EXACT, 5 to 8 patched cells per run, 147 patched cells and 209,207 values in total, 49 to 82 s per run (3 games; log
  [`circuit_doom_10pairs_source.txt`](../models/interventions/circuit_doom_10pairs_source.txt)). Together 40 of 40 runs on 10
  pairs per kind (not the 200: a run takes 50 to 300 s); the 200-pair means stay reference numbers. The cells chosen are the largest
  unflagged ones, which are the layer-1 heads and their values, not the previous-token path of layer 0.
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

## Steering

`df.probe(model, site, contexts, labels, pos=...)` (`doomformer/steer.py`) fits one direction: the integer activations of a site at
one position are averaged per class (two labels) and the direction is the difference of the class means, scaled and rounded to
integers with largest entry `scale` (default 4). The fit is floating point and deterministic for a given dataset; the
direction it returns is an integer vector, and `probe.add(alpha)` / `probe.project()` are the interventions `add_direction` /
`project_out` (integer arithmetic, clipped to the site's range), applied in the reference and in DOOM. `probe.accuracy` is the
fit accuracy of a midpoint threshold along the direction, not held out.

`python -m demos.steer` (`doomplayer`, reference tables in seconds; `--doom 5 --workers 2` adds DOOM runs): the probe separates the
5 contexts whose bearing is 'i' (aim aligned) from the 5 whose bearing is 'm' (12 to 20 degrees left), fitted at `x0` position 1,
`L0.ao` position 2 and `L0.x1` position 2 (the last position); fit accuracy 1.00 at all three on those 10 contexts. Steering
adds alpha times the direction at the same site and position of each of the 86 contexts of `policy.all_contexts()`. Four of the
five 'm' contexts do not fire in the healthy policy (point blank already fires). Log:
[`models/interventions/steer_doomplayer.txt`](../models/interventions/steer_doomplayer.txt).

| site | alpha | 'm' contexts that fire (of 4) | contexts whose action changes (of 86) | of them now F | random directions: 'm' fire / changed | held-out: 'm' at distances 3, 4 that fire (probe fitted on distances 0 to 2) |
|---|---|---|---|---|---|---|
| `x0`, 1 | 1 | 1 | 16 | 14 | 0.10 / 10.1 | 0 of 2 |
| `x0`, 1 | 2 | 4 | 39 | 35 | 0.40 / 28.4 | 2 of 2 |
| `x0`, 1 | 3 | 4 | 44 | 34 | 0.39 / 39.8 | 2 of 2 |
| `L0.ao`, 2 | 2 | 4 | 44 | 44 | 0.38 / 22.5 | 2 of 2 |
| `L0.ao`, 2 | 3 | 4 | 57 | 57 | 0.61 / 34.7 | 2 of 2 |
| `L0.x1`, 2 | 2 | 3 | 27 | 27 | 0.14 / 15.4 | 1 of 2 |
| `L0.x1`, 2 | 3 | 4 | 45 | 45 | 0.28 / 29.2 | 2 of 2 |

(200 random integer directions per cell, entries in -4..4 with one entry set to +-4, mean over the directions; alpha 1 at
`L0.ao` and `L0.x1` fires 1 of 4 and changes 22 and 13 contexts.) Subtracting the direction (alpha -2) turns the 'i' contexts at
`x0` from FIRE to TURN LEFT at distances 1 to 4 and at `L0.ao` at all five distances. `project_out` of the direction changes 2 of
the 86 contexts at each of the three sites (at `x0` and `L0.ao` one 'i' context stops firing, at `L0.x1` none does): the policy
does not depend on this one direction alone.

**Controls.** Random directions are a weak null (log: the same `steer_doomplayer.txt`): a direction fitted on data is not a random vector. Four controls at alpha 2,
200 draws each, same site, position and alpha ("m fire" is the mean number of the four 'm' contexts that fire; P is the share
of draws that fire at least as many as the real probe):

| site | direction | m fire | P(>= real) | contexts changed (of 86) | end in F |
|---|---|---|---|---|---|
| `x0`, 1 | real probe ('i' vs 'm') | 4 | | 39 | 35 |
| | random, largest entry 4 | 0.41 | 0.000 | 30.6 | 5.1 |
| | norm-matched random (probe entries permuted, random signs) | 0.29 | 0.005 | 22.4 | 4.2 |
| | shuffled-label probe ('i' vs 'm' labels permuted) | 1.80 | 0.450 | 30.8 | 15.8 |
| | 'fire vs not fire' probe (all 86 contexts) | 4 | | 49 | 46 |
| `L0.ao`, 2 | real probe | 4 | | 44 | 44 |
| | random, largest entry 4 | 0.46 | 0.010 | 23.2 | 9.8 |
| | norm-matched random | 0.30 | 0.000 | 19.1 | 7.0 |
| | shuffled-label probe | 1.29 | 0.195 | 27.5 | 16.2 |
| | 'fire vs not fire' probe | 3 | | 62 | 62 |
| `L0.x1`, 2 | real probe | 3 | | 27 | 27 |
| | random, largest entry 4 | 0.12 | 0.005 | 14.7 | 4.0 |
| | norm-matched random | 0.08 | 0.000 | 10.1 | 3.1 |
| | shuffled-label probe | 0.97 | 0.210 | 20.3 | 12.7 |
| | 'fire vs not fire' probe | 4 | | 62 | 62 |

Corrected conclusion. The real direction beats random and norm-matched random vectors (P at most 0.01). It does not beat a probe
fitted on shuffled labels: 45% of the shuffled-label probes at `x0`, 19.5% at `L0.ao` and 21% at `L0.x1` fire at least as many
'm' contexts, and their mean is 1.8 of 4, 1.3 and 1.0. A 'fire vs not fire' probe fires 4, 3 and 4 of the 4
and changes more contexts (49, 62 and 62 of 86). So 4 of 4 'm' contexts firing is not evidence
that the direction encodes "bearing is 'i'": a direction fitted on any labels of these contexts fires a good share of them
too, and in the real runs 35 of the 39 changed contexts at `x0` (all 44 at `L0.ao`) end in FIRE, so the probe is mostly a push toward
FIRE that happens to reach the 'm' contexts. (Shuffled-label numbers from a separate run, 1.74 of 4 and 43% reaching 4 of 4 at `x0`,
1.31 and 22% at `L0.ao`, agree with these within the draw noise.) What is specific: at `x0`, position 1, the activation is the
staircase count of the token embedding, so the direction is, up to scale and rounding, the count difference of the tokens 'i' and 'm'; and every steered
run is exact in DOOM, which does not depend on what the direction means. The earlier reading of this table ("4 of 4 against
0.4 for random") compared with the weakest control.

In DOOM: for each of the three sites, the four 'm' contexts at alpha 2, one other context ('a' bearing, distance 0), and one
`project_out` on an 'i' context, 18 runs in tile maps with Python patching the bus: 18 of 18 EXACT at tolerance 0, 226 patched
cells, 9,784 values (7 to 20 s per run, 2 games under load average about 25). The DOOM logits give the same action as the
steered reference in every run (for example `L0.x1` alpha 2 context 'm4' stays TURN LEFT in both).

Limits: one probe family (difference of means, one direction, two classes), one model, a fit and a test on the same 10 contexts
except the held-out column (2 contexts at distances 3 and 4); the flip rates are over the policy's 86 observations, not game
play. Steering in a one-map level is not available: a direction needs the activation's value and a level has no host to read
it (only zero lane doors exist there).

## Lesions

`demos.lobotomy --minimal --null` and `--names` use the exact batch evaluator `doomformer/lesion.py` for the minimal pacifist,
the hit rates per site, the shuffled-weight null and the name model; the numbers and commands are in
[DEMOS.md, Lesion statistics](DEMOS.md#lesion-statistics). The minimal lesion runs in DOOM as before (`ablate=`, doors).

## Limits

- Interventions are on the integer activations the tile engine exposes. In a one-map level the only runtime intervention is a
  lane door that zeroes a unit (`--lesion-at`): relu attention, no norm, the relu MLP, doors chosen when the map is built, closing
  decided by the level's clock. `set`, `mean`, `scale`, directions and per-position values need a host and exist in tile mode
  only (or as `fold`, a different map, for the constant ones).
- Python rewrites tile ports and the dolls compute the rest: in tile mode Python writes the patched values (bus patches) and
  chooses which doors close; the dolls do every product, sum and staircase, and the check compares every exposed value with the
  reference at tolerance 0. The patch value itself is Python's choice, checked only through its effects (above).
- The gated MLP supports `zero` at `m` only.
- A row takes 6 to 21 s for `doomplayer` and 25 to 77 s for `doomplayer_xl` (3 games, load average 43; the tile maps are the
  cost, not the interventions).
- Rows are single cells. Path patching is not provided; a path patch is a set of resolved values you can pass as
  `set_value` arrays.
