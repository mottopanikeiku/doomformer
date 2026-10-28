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
| 0 | none: the original schedule | byte-identical to v0.4.1 output (hashes below) |
| 1 | `exclusive`, `lights`, `colour` | only facts that hold for every input |
| 2 | opt 1 + `tuned` | timing constants from `doomformer/tuning.json` and a per-layer sync wait from the arrival model; margins over measured failure edges |
| 3 | opt 2 + `copy` | light accumulators and an ACS copy at each layer end (below); the "no ACS between layers" claim changes |

`df.compile(net, opt=N)`, `df.plan(net, opt=N)`, `df.run_many(..., opt=N)`, and `--opt N` on `mnist`, `draw`,
`demo`, `hello` and `plan`. A set of pass names (`opt={"exclusive", "lights"}`) selects single passes; `plan --explain`
and the ablation rows below use that. Spiking nets, learners and agents stay at `opt=0`. Tile maps have no waves.

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
at opt 3 `clock_margin=0` puts the copy line before the last light hit and DOOM mismatches. On the Compiler review's
24-32-16-8 net the checker gives 117 / 55 / 0 hazards at S = 16 / 24 / 32 and DOOM is exact on 11/30, 25/30, 30/30
inputs; every DOOM failure is on the value the hazards name.

Findings on the shipped settings: MNIST at opt 3 with `clock_margin = 144` is flagged (copy line 2 tics before the last
hit) and fails in DOOM (114 values wrong from layer 1); `doomplayer_onemap` at opt 2 with 144 is flagged (freeze line
1 tic early; not run in DOOM). Both led to the shipped values below.

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
| `clock_margin` (opt 2) | next layer's line past the last effect cell | 512 | 71 (tiny_model) | 72 (4.5 tics) | 192 (12 tics) | model bound (176) + 16, over the rule's 144 |
| `clock_margin` (opt 3) | copy line past the last effect cell | 512 | 128 (MNIST, 24-32-16-8) | <= 192 | 256 | max(1.25 x 192, 256) |
| `corr_gap`, `tele_gap`, `chunk_lead`, `copy_settle` | | 64, 64, 16, 64 | not measured | | defaults | |

Measured edges depend on the nets scanned (hello, signed 6-10-4, 24-32-16-8, tiny_model, a softmax fixture, 200-wave
fold net, MNIST for opt 3); the coarse scan steps were 32 units, bisected to 8. The first release of a layer used a
constant `sync_wait` of 25 tics; random inputs showed an edge of 30 units on small nets, which is void: dolls that step
onto their lane gate (port 1 to 24 above the walkway) arrive 7 to 9 tics later, and layers whose first corridor floor is
below the lane fall first. The shipped rule is therefore the model's `lane_arrival_units` per layer plus `sync_margin`.

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
| MNIST map | 5,696 | 5,672 (20/20) | 3,560 (76/76 so far, see below) | 272 (300/300; 290 read right; DOOM's digit equals `models/mnist/doom_1000.json` on all 300) |
| DrawPad READ pass | 5,705 | 5,681 (10/10) | 3,569 (10/10) | 273 (31/31, digits 0-9 walked with three seeds; every walked digit read right) |
| `selfgen --loop`, `doomnames_onemap` | 4,000 | 2,080 (5/5) | 1,568 (5/5) | 768 (5/5); the text equals `intref.generate` at every level |

* MNIST at opt 2: the campaign (100 digits planned, one game; rows in [`models/schedule/`](../models/schedule/)) was still running when this table
  was written; 76 digits were done and all exact. An earlier 4-digit run gave the same 3,560 tics. The 3,500-tic target
  is missed (3,548 estimated, 3,560 measured). Opt 3 reaches 272.
* MNIST at opt 1 gains only the comparison layer (15 to 9 waves): layers 0 and 1 are full cliques of 784 and 512 dolls,
  1,296 waves, and only shorter spacing helps. At opt 2 the spacing is 40 units; at opt 3 there are no waves.
* `doomnames_onemap` target (at most 1,700 tics): 1,512 at opt 1.
* The opt 0 rows are the v0.4.1 figures (the 24-32-16-8 row is the Compiler review's 1,440; MNIST 5,696 is the measured
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
* opt 3 was not scanned for `sync_margin`. `doomformer tune --check` was not run end to end, nothing was run on macOS,
  and `doomplayer_onemap` at opt 2 was not run in DOOM.
* Scans used 10 random inputs per net (not 30) with steps of 32 units bisected to 8.
