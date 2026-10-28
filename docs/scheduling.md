# Scheduling: waves, opt levels, the checker and the tuner

One map runs a layer's dolls through a clock. This page is the compiler's schedule: what it must prevent, the four
`opt` levels, the algorithm of each pass and why it is exact, the static checker (`doomformer lint`), the timing
tuner (`doomformer tune`) with its failure edges, and measured results. **`opt=3` puts an ACS copy bus between the
layers:** a script reads each layer's light levels and writes them into the next layer's gate heights, so the claim
"no ACS between layers" holds at `opt=0`, 1 and 2 (the default is 0) and does not hold at `opt=3`. The script adds,
compares and chooses nothing; the dolls and their lines still do every sum, threshold, comparison and choice, and
every value is still read back and checked. `opt=0` is the original schedule and builds the same bytes as before.
The validation record (every pass of every campaign, with the command that made it) is in
[`models/schedule/`](../models/schedule): `summary.md`, `runs.csv`, `commands.txt`.

## What the schedule has to prevent

A dense layer is a crossbar: every doll that passes its lane gate crosses one line per target, and each line adds
its weight to a sector (a light level, or the floor or ceiling of a gate that a later layer reads). The engine
gives two facts (HOW_IT_WORKS.md, "Engine facts"):

* a plane mover (floor or ceiling special) aimed at a sector in the same tic as another one is dropped: the second
  one processed is lost (measured by the checker's probe: two dolls, same sector, same tic: one move lost; one tic
  apart: both land);
* a light level takes every same-tic hit.

So dolls that can write the same plane must cross their lines in different tics. The layer's clock doll walks past
release lines `S` map units apart (about 16 units per tic, so S = 64 is 4 tics); each line lowers one wave's sync
barrier, and the dolls behind it start together. Dolls of one wave cross their (aligned) effect lines in the same tic,
dolls of consecutive waves `S / 16` tics apart. The original scheduler gave each doll the first wave in which none of
its targets was taken (first fit), kept `S = 64` and the clock constants `SYNC_WAIT = 400`, `CLOCK_MARGIN = 512`.
That is correct and slow: on MNIST 91 percent of the 5,696 tics are the 1,296 waves of layers 0 and 1 (one wave per
input doll, because every pixel doll hits all 64 hidden sectors).

## Levels

| `opt` | passes | what it assumes |
|---|---|---|
| 0 | none: the original schedule | byte-identical to v0.4.1 output (sha256 table below) |
| 1 | `exclusive`, `lights`, `colour` | only facts that hold for every input |
| 2 | opt 1 + `tuned` | timing constants from `doomformer/tuning.json` and a per-layer sync wait from the arrival model; margins over measured failure edges |
| 3 | opt 2 + `copy` | light accumulators and an ACS copy at each layer end (below); the "no ACS between layers" claim changes |

`df.compile(net, opt=N)`, `df.plan(net, opt=N)`, `df.run_many(..., opt=N)`, and `--opt N` on `mnist`, `draw`,
`demo`, `hello` and `plan`. A set of pass names (`opt={"exclusive", "lights"}`) selects single passes; `plan --explain`
and the ablation rows below use that. Spiking nets, learners and agents stay at `opt=0`. Tile maps have no waves.

### Opt 0 hashes

sha256 of the `.wad` and `.json` that `df.compile(net)` (opt 0, no flags) builds. The first five rows were built with
v0.4.1 (92c3480) and with v0.5.0 (4ace1d2) and equal the working tree's; `induction_onemap` did not exist in v0.4.1 and
its hash is the v0.5.0 one (all rows: `models/schedule/opt0_identity.json`). Check them with
`doomformer tune --hashes` (builds the six maps, no game, about 30 s; exit 1 on a difference); `--update-hashes`
rewrites [`doomformer/opt0_hashes.json`](../doomformer/opt0_hashes.json). A change of a default build means an opt 0
byte changed: the table must then change in the same commit, with the reason.

| model | sectors | `.wad` sha256 | `.json` sha256 |
|---|---|---|---|
| `tiny_model` | 12,620 | `3e64746d0a4fbde309d74652e9357c98127d930634455294635bc729d0a34145` | `d173ac4c70a33d63d1ed6a4385b71e897a7602b4f801906ad000fdbbe045faad` |
| `doomnames_onemap` | 95,448 | `5efd4969ad8d3070ee93890f837aefac1a15321f67dc03bae99e98683764d9e6` | `38eadf65757913998326c6f27626c51b796c635fc4b53116c844273f7609812b` |
| `doomnames_onemap_softmax` | 123,066 | `09efc21ede5b3bf8dba006a81fba66944b879aa9b696c55c8014b9f304c1fee6` | `d65fc5147e1259a27600e76b882413f96aac585866c6ab30f6f84e6652f0c0b9` |
| `doomplayer_onemap` | 37,109 | `4afa39dfdfbab0caf556b84341cae729faecab70d932c6864de60fa1d872aace` | `b6be0839b9a53b72dda9524fc3e0c242c8a4508e3d5cf2660abc0c2e3f372999` |
| `induction_onemap` (v0.5.0) | 244,890 | `dbc68ef6267279c673d7b1ec3e20aecafbd4e098810348ea095dccc59946fb05` | `6e7adc9d6ae9af81b097469640f43937da8dab2ea0a827ef4f05966bfdadcd85` |
| MNIST MLP (`models/mnist/mlp.npz`) | 158,550 | `e1c36707eef84c65c755e9ac2454c5e48de97841d18d96412a274d7632128e7c` | `875b756bf9fe487a8dcba665e976f1b195ef176d67ff49c7beeebb9a1aa7ee49` |

## The passes

Code: `doomformer/pure.py` (`compile_pure`, `Timing`, `opt_passes`), `doomformer/waves.py` (the conflict graph and its
colouring), `doomformer/explain.py` (`plan --explain`).

**Conflict graph.** One layer's dolls are the vertices. Two dolls are adjacent when they share a *plane* target (a
floor or ceiling tag that a later layer reads as a gate, or an action special) and can both reach an effect line.
A wave is a colour class; the layer needs as many waves as colours. Layers whose dolls share no plane target
(components of the graph) are coloured separately and reuse the same wave numbers.

**`exclusive`.** A doll reaches an effect only if its lane gate and every stage gate up to its first effect pass.
Each gate is an interval of one value: a floor gate on the port that carries `v` (or `-v`) opens iff `port <= limit`,
a ceiling gate iff `port >= limit`, so each gate bounds `v` from one side. Two dolls whose intervals for one value
are disjoint can never both pass, whatever the input: one-hot lookups (`ops.equals`, a token embedding has one doll
per token), the plus and minus dolls of a signed read (`x >= t` and `-x >= t'` with `t, t' >= 1`), equality dolls.
They share a wave. The analysis uses only values that are constant while the layer runs: a value the layer itself
writes (a `state` value that its own dolls update) gives no interval. It assumes the two ports of one value carry `v`
and `-v`, which every writer in the toolkit guarantees (`MLPBrain._ports`, `token_ports`, the accumulator pairs).

**`lights`.** A light level takes every hit: two lines that add to one light in one tic both land
(`tests/test_waves.py::test_light_hits_in_one_tic_are_not_lost_but_plane_moves_are`: two dolls released together
add 2 to a light, while the second plane move on the twin is lost). Light targets therefore never conflict. Action
targets keep their conflicts (a user's special may move a plane).

**`colour`.** DSATUR (Brelaz 1979) colours the conflict graph of each component: next the vertex with most distinct
neighbour colours, lowest free colour. It is exact on cliques, bipartite graphs and odd cycles. A comparison layer
(`onemap.argmax_layers`, `drawpad.with_argmax`) has `k (k - 1) / 2` dolls with two targets each, the line graph of
K_k; a component with that shape is coloured by the circle method (round-robin tournament), `k - 1` colours for even
`k` and `k` for odd `k`, the minimum (an edge colouring of K_k needs that many). First fit gives 15 waves for
`k = 10`, DSATUR and the circle method 9; for every `k` from 2 to 20 the result is the minimum
(`tests/test_waves.py::test_comparison_layer_is_coloured_optimally`). Each layer also reports a clique lower bound
(`stage_stats[i]["lower_bound"]`); `waves == lower_bound` in every layer of the shipped transformers at opt 1.

**`tuned`.** The constants of `Timing` (serial, clock_margin, corr_gap, tele_gap, chunk_lead) come from
`doomformer/tuning.json`, the failure-edge table of `doomformer tune`. The first release of each layer follows the
slowest lane doll: `sync_wait = lane_arrival_units(stairs, drop) + sync_margin`, with `lane_arrival_units` taken from
the checker's exact doll model (125 units on a flat lane, `16 sqrt(2 d) + 32` when the first corridor floor lies `d`
units below the lane, which makes the doll fall). The old constant 400 is that number for no layer in particular.

**`copy`** (opt 3). In every layer each consumed value is accumulated in ONE light level (a value and its negation,
built by `ops.output_pair`, share the light of the positive name; the negated twin gets no effect lines). All of a
layer's dolls then hit only lights: one wave. At the layer's end the clock crosses a line that runs an ACS script
(`ACS_ExecuteAlways`, script `30 + layer`), which reads each light and writes it into the gate planes that read it
with `Floor_MoveToValue` / `Ceiling_MoveToValue` (the engine's special takes a magnitude and a sign flag, so the script
splits the sign off; for the negated plane it flips the flag). `Timing.copy_settle` units later the freeze lines and
the next layer's start barrier follow. Values with `actions`, or in `state`, or whose twin is not an exact negation
keep their plane accumulators. Names are unchanged: every plane the next layer reads is read back and compared, and
so is every light nobody consumes. **What this costs in wording:** the one-map claim "no ACS between layers" becomes
"an ACS script copies each layer's light levels into the next layer's gate heights; it adds, compares and chooses
nothing; the sums, thresholds, comparisons and the choice are the dolls' and lines'". A copy that were wrong shows as
a plane that differs from the reference. `opt=3` is the only level whose maps contain an ACS script between layers.

### A latent opt=0 bug these tools found

`doomformer fuzz` (net 591, seed 0) found a net whose output was wrong at every `opt`: an MLP with signed input and
thresholds `[5748, 8058]`. The barrier closes at height 13,000 and opens by lowering to the lane's walkway at 18,750
units per tic. A walkway between -5774 and -5751 needs two mover tics and the first one stops 1 to 24 units above it:
the doll beside the barrier steps up onto it, rides it down, and arrives about 7 tics late. Its plane move then lands
in the tic of the next wave's doll and is dropped. `SchedCheck`'s model reproduced it (engine trace: z = -5750 for
two tics, then a fall to -5772), a second minimal net (`[5740, 11999]`, `x = 12000`) showed the same, and
`pure._barrier_height` now picks, per layer, a closed height at which no walkway sees a first-tic stop within a step
(`tests/test_waves.py::test_late_barrier_float_regression`; with the old constant both nets fail). It applies at every
`opt`; no shipped model's walkways are in that window, so their maps are byte-identical.

## The checker: `doomformer lint WAD`, `df.check_schedule(build)`

`doomformer/schedule.py` is a translation validation of the schedule. It reads only the compiled map (the TEXTMAP
of the `.wad` and the port keys of its `.json`), rebuilds every lane, gate, sync barrier, effect cell and clock chain,
and replays each doll tic by tic in the engine's 16.16 fixed point (carry push, friction, step-up and fall, move
splitting, teleports that keep speed). Predicted line-crossing tics equal the engine's on every clock line, lane cell
and sync arrival it was compared against (hello: 66 lines, 74 cells, 7 arrivals; `tiny_model`: 163 lines, 687 cells,
198 arrivals). It reports

* a **hazard**: two dolls that can both pass their gates and cross plane-moving lines (or actions) of one sector less
  than the measured minimum apart (1 tic: in the same tic the second move is lost, 1 tic apart both land; probe in
  the checker's log), with a **witness**: the port values that make both pass, and an input vector when the ports are
  network inputs. Dolls with disjoint gate intervals, and light targets, are not hazards;
* a **margin** below its minimum: sync wait (release minus the latest arrival, at least 4 tics), serial, clock margin
  (last write against the next layer's read and against the freeze line), teleport clearance, and at opt 3 the copy
  line against the last light hit and the settle after it.

`doomformer lint WAD [--net NET] [--json]` exits 1 on any hazard or margin. Seconds on the loaded machine (opt
0/1/2/3): MNIST 27/27/23/13, doomnames_onemap 17/36/17/13, softmax 27/41/20/16, doomplayer_onemap 10/17/8/5,
`tiny_model` 3/5/3/2. `opt` 0 and 1 are clean on all five shipped models.

Tests (`tests/test_schedule.py`): a tightened constant (`serial={0: 8}`) is flagged and its witness mismatches in
DOOM, while the same input is exact and unflagged at opt 0 and opt 1; with the old barrier height restored
(`pure._barrier_height = CLOSED`) net 591 is flagged with witness `x0 = -8058` and DOOM fails at -8058 and -9000;
at opt 3 `clock_margin=0` puts the copy line before the last light hit and DOOM mismatches. On an independent
24-32-16-8 net the checker gives 117 / 55 / 0 hazards at S = 16 / 24 / 32 and DOOM is exact on 11/30, 25/30, 30/30
inputs; every DOOM failure is on the value the hazards name.

Findings on the shipped settings: MNIST at opt 3 with `clock_margin = 144` is flagged (copy line 2 tics before the last
hit) and fails in DOOM (114 values wrong from layer 1); `doomplayer_onemap` at opt 2 with 144 is flagged (freeze line
1 tic early; not run in DOOM). Both led to the shipped values below.

### Soundness campaign: `doomformer lint --fuzz`

The question is whether a map the checker calls clean can still be wrong in DOOM (a false negative). The campaign
([`doomformer/lintfuzz.py`](../doomformer/lintfuzz.py), results in
[`models/schedule/lintfuzz-seed0/`](../models/schedule/lintfuzz-seed0): `summary.md`, `summary.json`, one row per map in
`lintfuzz.jsonl`, `commands.txt`) builds a map, takes the checker's verdict, then runs 16 inputs of the net in DOOM
and compares every exposed value with the integer reference at tolerance 0. The nets are `doomformer fuzz`'s generator
(`fuzz.gen_case`: 1 to 4 layers, signed and unsigned inputs, hidden values placed on thresholds and at the +-12,000
range edge) and, for about one map in seven, mid-size reference-style MLPs (up to 16-24-24-8). Inputs are the
fuzzer's boundary-seeking set: thresholds and thresholds +-1, extremes, zero, random. Net `i` of a seed and its
constants are drawn from `SeedSequence([seed, i, ...])`, so a row reproduces with `--only INDEX:OPT`.

* **random**: per net and opt level 1, 2, 3, each timing constant is replaced with probability 1/2 (at least one) by a
  uniform draw from a range around its failure edge (serial 8 to 72, clock_margin 0 to 200, sync wait 0 to 160 or
  sync_margin -64 to 96, corr_gap, tele_gap, chunk_lead, copy_settle; opt 3 clock_margin 0 to 320).
* **edge**: per net, opt level and constant (serial, clock_margin) with the others shipped, bisect the smallest value
  the checker calls clean and run it in DOOM; also run one unit below it. A false negative here would be an exact
  boundary miss.

| campaign | nets | maps scored | false negatives | flagged, DOOM wrong | flagged, DOOM exact | clean, exact |
|---|---|---|---|---|---|---|
| random, opt 1, 2, 3 | 983 | 2,947 | **0** | 278 | 935 | 1,734 |
| edge, opt 1, 2, 3 | 280 | 2,450 | **0** | 542 | 232 | 1,676 |

The random campaign covers 1,000 net indices (51 maps unsupported: a zero weight layer the compiler refuses; 2 build
errors) and ran in full; the edge campaign was set to 500 nets and stopped after 282 because of machine load
(`--nets 500` resumes it). Totals: 5,397 maps, 97,549 DOOM passes, 0 false negatives.

Conservative rate (flagged maps that DOOM ran exactly): 77 percent in the random campaign, but 920 of the 935 are
maps flagged only for the `sync_wait` margin (a release less than 4 tics after the slowest lane doll can arrive; the
tuner finds no failure on small nets down to sync_margin -64). Without that flag the rate is 15 of 210 flagged maps
(7 percent). At the edge, the value one unit below the checker's clean value fails in DOOM on 66 percent of the
clock_margin maps and 73 percent of the serial maps, so the checker's boundary is within a unit of the true one on
about two thirds of the nets and conservative on the rest.

What this does not show: nets are small (at most 24 inputs and 4 layers), 16 inputs per map can miss an input-specific
hazard, only serial and clock_margin were probed at the boundary (the others only by random draws), nets with gated
corridor layers (transformers) are not in the campaign (they are in the tuner's reference runs), and `copy_settle`,
`corr_gap`, `tele_gap` and `chunk_lead` rarely matter for nets this small. The checker is a model of the engine; an
engine fact outside the model would be found only by DOOM runs like these.

```
doomformer lint --fuzz --nets 1000 --seed 0 --workers 6 --fuzz-out models/schedule/lintfuzz-seed0   # 11 minutes at 6 games on a shared machine
doomformer lint --fuzz --edge --nets 500 --seed 0 --workers 6 --fuzz-out models/schedule/lintfuzz-seed0
doomformer lint --fuzz --only 1:1 --seed 9                   # one row again (net 1, opt 1)
```

## The tuner: `doomformer tune`, `doomformer/tuning.json`

`doomformer tune` finds, with the oracle (every exposed value against the NumPy reference, tolerance 0), the value
below which each timing constant starts to fail on reference nets, and writes the table to `doomformer/tuning.json`
with the commands and input seeds that made it. Game tics are simulated, so pass or fail does not depend on machine
load. `doomformer tune --check` re-runs the shipped values. The edge is the smallest value for which every net passes
every input at it and at all tested larger values. Inputs: extremes, boundary vectors (`x = t + d`, `d` in
0, 1, 10, 23, 24, the values that make a doll step onto its lane gate and arrive late) and random vectors.

| constant | meaning | default (opt 0) | largest failing | edge | shipped | how |
|---|---|---|---|---|---|---|
| `serial` | clock distance between wave releases | 64 | 28 (24-32-16-8) | 29 (1.8 tics) | 40 (2.5 tics) | 1.25 x edge, up to 8 |
| `sync_margin` | release past the slowest predicted lane arrival | (sync_wait 400) | none down to -64 | <= -32 | 64 (4 tics) | the checker's minimum |
| `clock_margin` (opt 2) | next layer's line past the last effect cell | 512 | 72 (tiny_model 6 of 33; the other constants shipped) | 80 (5 tics; 72 with the others at their defaults) | 192 (12 tics, 2.4 x the edge) | model bound (176) + 16, over the rule's 160; re-scanned 2026-10-28 (`docs/tuning.md`) |
| `clock_margin` (opt 3) | copy line past the last effect cell | 512 | 128 (MNIST, 24-32-16-8) | <= 192 | 256 | max(1.25 x 192, 256) |
| `corr_gap`, `tele_gap`, `chunk_lead`, `copy_settle` | | 64, 64, 16, 64 | not measured | | defaults | |

Measured edges depend on the nets scanned (hello, signed 6-10-4, 24-32-16-8, tiny_model, a softmax fixture, 200-wave
fold net, MNIST for opt 3); the coarse scan steps were 32 units, bisected to 8. The first release of a layer used a
constant `sync_wait` of 25 tics; random inputs showed an edge of 30 units on small nets, which is void: dolls that step
onto their lane gate (port 1 to 24 above the walkway) arrive 7 to 9 tics later, and layers whose first corridor floor is
below the lane fall first. The shipped rule is therefore the model's `lane_arrival_units` per layer plus `sync_margin`.

## `plan`'s settle estimate and its error band

`doomformer plan MODEL --opt N` prints the game tics of one pass as `about N game tics per pass, +-X %`. The geometry
counter's `estimated_settle_tics` (still in `df.plan(...)["map"]`, and what the runner's timeout uses) is an upper
bound that ran from -10 to +28 percent off the measured settle (median 6 percent; at opt 3 +22 percent on `doomnames_onemap`, +12 on MNIST);
its error is almost the same at every opt level for one net, so a per-level factor does not fix it. The printed value
follows the clock's timeline instead (`doomformer/cost.py`, `settle_estimate`): per layer the first release (sync
wait), the waves after it `(waves - 1) * S`, the extra chunks of a long clock, the corridor, the clock margin and, at
opt 3, the copy settle, all at 16 map units per tic, plus 4.78 tics per layer and 29.8 tics for the start. The two
fixed numbers are the only fitted values (least squares on 44 measured settles: the 8 reference nets of the tuner
and MNIST, `doomnames_onemap` and its softmax twin at opt 0 to 3). A fit with one intercept per opt level gave 32.3,
30.4, 28.1 and 28.5 tics, so the levels do not need their own constant.

Validation on nets that were not in the fit: four random MLPs (10-20-12-5, 40-16-8, 6-30-30-4 and 16-16-16-16-4) and
the trained `induction_onemap` and `doomplayer_onemap`, each at opt 0 to 3 (24 settles, measured after the model form
and constants were fixed):

| | fit rows (44) | held-out rows (24) |
|---|---|---|
| calibrated: median \|error\| | 0.7 % | 0.7 % |
| calibrated: largest \|error\| | 10.1 % (the 152-tic `fold` net at opt 3, 15 tics) | 4.6 % (the MLPs at opt 3, 8 to 11 tics); 0.7 % on the transformers (40 tics at most) |
| geometry bound: median \|error\| | 6.6 % | 5.4 % |
| geometry bound: range | -10.5 % to +27.6 % | -0.4 % to +27.2 % |

The band is `+-max(20 tics, 1.5 %)`; no row of either set is outside it. It holds for nets like these (MLPs of up to
four layers, small and trained transformers, one or two clock chunks per layer); a net with very different geometry
(many gated corridor layers, hundreds of chunks) is not covered by the held-out rows. `plan` leaves the predicted seconds
on the geometry bound, because the cost model's constants were fitted on it (`docs/scaling.md`). Every row, with its
timeline terms, is in [`models/schedule/plan_calibration.json`](../models/schedule/plan_calibration.json);
`python -m doomformer.cost --refit` re-fits and prints the errors; `python -m demos.settlecheck NETS --opts 0,1,2,3`
measures new nets in DOOM and prints bound, calibrated value and error side by side.

## Results

Estimated settle tics come from the geometry counter (`doomformer plan MODEL --opt N --explain`); measured tics are what
`TileRunner.settle` returned, constant per model because the schedule does not depend on the input. Every pass in the
table was exact: every exposed value equals the NumPy integer reference at tolerance 0. Opt 2 and 3 use the final
`tuning.json` (serial 40, sync_margin 64, clock_margin 192 at opt 2 and 256 at opt 3, the rest defaults). Counts are
passes (inputs, prompts or digits); commands are in `models/schedule/commands.txt`, per-pass rows in `runs.csv`.

| model | opt 0 | opt 1 | opt 2 | opt 3 |
|---|---|---|---|---|
| `hello` 4-6-5-3 | 488 | 384 | 240 (5/5) | 184 (5/5) |
| 24-32-16-8 MLP, K=3 | 1,440 | 960 (30/30) | 608 (30/30) | 200 (30/30) |
| `tiny_model` | 1,800 | 1,048 (20/20) | 736 (20/20) | 512 (20/20) |
| `doomnames_onemap` | 3,488 (2/2) | 1,512 (24/24) | 1,128 (20 prompts + generate, 24/24) | 544 (20 prompts + generate, 24/24) |
| `doomnames_onemap_softmax` | 4,144 (1/1) | 2,344 (10/10) | 1,928 (10/10) | 592 (10/10) |
| MNIST map | 5,696 | 5,672 (20/20) | 3,560 (76/76 so far, see below) | 272 (300/300; 290 read right; DOOM's digit equals `models/mnist/doom_1000.json` on all 300; re-run with the final constants: 100/100 exact, 99 right, [log](../models/schedule/revalidate/mnist-opt3-100.log)) |
| DrawPad READ pass | 5,705 | 5,681 (10/10) | 3,569 (10/10) | 273 (31/31, digits 0-9 walked with three seeds; every walked digit read right; re-run with the final constants: 30/30 walked, 30 right, 30 exact, [log](../models/schedule/revalidate/draw-opt3-30walked.log)) |
| `selfgen --loop`, `doomnames_onemap` | 4,000 | 2,080 (5/5) | 1,568 (5/5) | 768 (5/5); the text equals `intref.generate` at every level |

* MNIST at opt 2: the campaign (100 digits planned, one game; rows in [`models/schedule/`](../models/schedule/)) was still running when this table
  was written; 76 digits were done and all exact. An earlier 4-digit run gave the same 3,560 tics. The 3,500-tic target
  is missed (3,548 estimated, 3,560 measured). Opt 3 reaches 272.
* MNIST at opt 1 gains only the comparison layer (15 to 9 waves): layers 0 and 1 are full cliques of 784 and 512 dolls,
  1,296 waves, and only shorter spacing helps. At opt 2 the spacing is 40 units; at opt 3 there are no waves.
* `doomnames_onemap` target (at most 1,700 tics): 1,512 at opt 1.
* The opt 0 rows are the v0.4.1 figures (the 24-32-16-8 row is 1,440; MNIST 5,696 is the measured
  settle of `doomformer mnist`).
* Runs under earlier constants are in `runs.csv` labelled `provisional`: opt 3 with clock_margin 144 lost values on
  MNIST (114), DrawPad and the 24-32-16-8 MLP (0/30), and opt 2 with a constant sync_wait of 64 lost one value per
  prompt on `doomnames_onemap` (0/3). They are the reason for the final constants.

## What remains and what was not tested

* The checker's minimums for the freeze line and the copy line's own tic are reasoned or measured on one net.
  The doomplayer opt-2 flag was not run in DOOM.
* The corridor-spread idea (spacing a corridor layer's waves by the spread of one target's cells instead of the whole
  corridor) was built and removed: the checker found 688 certain hazards on the softmax model, because dolls move slower
  than 16 units per tic through corridors. Corridor layers keep `S = span + corr_gap`.
* `corr_gap`, `tele_gap`, `chunk_lead` and `copy_settle` ship their defaults, unmeasured.
* opt 3 was not scanned for `sync_margin`. Nothing was run on macOS. `doomplayer_onemap` at opt 2 was not run in DOOM
  as a level (the tuner's `--check` does not include it); at opt 3 the arena level of `demos/brainlevel.py` ran it 50
  of 50 exact (`models/brainlevel_opt/`).
* `doomformer tune --check` (opt 2 and opt 3, every reference net and both trained models) ran end to end:
  331 of 331 and 169 of 169 passes exact (`docs/tuning.md`).
* The clock_margin edge (opt 2) was re-scanned with 30 random inputs per net; edge 80 with the other constants shipped
  (72 fails on `tiny_model`), shipped 192. The serial edge and the opt 3 clock_margin edge keep their earlier scans
  (10 random inputs per net or fewer, others at their defaults for serial).
* MNIST at opt 2: the revalidation with the final constants ran 100 digits planned on 2 games and was stopped by its time
  box; the digits it finished are in `models/schedule/revalidate/mnist-opt2-100.log`. The 76 of 76 above and the
  4-digit run are the earlier evidence. MNIST at opt 3 (100 of 100), DrawPad (30 of 30 walked) and `selfgen --loop` ran
  to completion.
* Twin elision (`--no-twins`) was not built; see below.

## Twin elision: not done, and why

A count of the layout showed that dropping the twin accumulators nothing consumes would cut MNIST from 158,550 to 89,344
sectors by layout count. At opt 3 the copy pass already does that: a value and its negation share one light, the negated
twin gets no effect lines, and MNIST has 88,146 sectors at opt 3 (158,538 at opt 2; `doomformer plan models/mnist
--opt 3`; per layer 58,509 + 28,679 + 701 + 126). What remains at opt 3 is one effect cell per (doll, target) weight:
784 x 64 + 512 x 45 cells and their lanes. So `--no-twins` could only help opt 1 and 2 (it would drop the positive hidden
value's plane accumulator, which nothing reads, and with it that value's check), and it does not bring the engine time per
MNIST pass at opt 3 (3.35 s measured in a review run at load 10) below 0.5 s: that needs 6.7 times fewer
sector-tics, and the sector count at opt 3 is already the twin-free one. Not built, not measured.
