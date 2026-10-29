# Changelog

Releases are tagged on GitHub: https://github.com/mottopanikeiku/doomformer/releases. Every version keeps the rule
that each value DOOM computes is checked against the NumPy integer reference with tolerance 0. The films and images the
README links to are assets of the v0.1.0 release.

## Unreleased

- Build pinned ACC with dynamic Linux runtimes, without requiring `glibc-static`. `doctor` now builds missing ACC
  and compiles an ACS probe instead of mistaking tools on PATH for a working compiler.
- Guard readback by default, including spiking and tile maps. Poison and check USER slots before each page, not
  only the first: a dropped guard or page fails even on identical consecutive pages or repeated inputs.
  One-map `guard=False` remains an explicit historical opt-out. Default WAD and metadata hashes change because
  the guard script and its metadata are present; geometry and arithmetic do not change.
- `compile_pure` maps are guarded too: the digit learner (`StepAudit` counts the guard as its own class), the
  machines on both clocks (the doll clock's ACS lump keeps the guard script), self-generation (including the
  until-EOS map's boot-time check of the prompt's state values), `df.Agent` and the brain-level demos. A guarded read
  takes two tics per page (`TileMeta.read_tics`); HOLD windows are sized from it
  and keep their old length wherever the guarded pages fit the old margin (the machines and the brain-level demos:
  their documented tics per step do not change; the shipped self-generation maps get longer windows). Documented
  command counts grow by one guard per page read. The fly loop keeps the unguarded readback:
  its 9 guarded pages do not fit its 16-tic HOLD window of world time ([verification](docs/verification.md)).
- Bring the film site's DrawPad numbers in line with the committed README (273 tics at opt 3; 5,705 at opt 0).
  No new timing measurements.
- The museum map ([docs/museum.md](docs/museum.md)): `doomformer museum build` writes museum.wad, which plays without
  Python. You light a digit on seven floor pads and press READ; 136 voodoo dolls compute a hand-written network (each
  digit's score counts the segments that agree with it) and light the answer. MUTATE rewires the connection from segment
  a to digit 7 inside the running level (`SetLineSpecial` on its two weight lines), and the same 7 then reads as 1.
  `museum verify` plays the 7, MUTATE, RESTORE tour headless and checks every value of every READ; `museum export`
  writes a self-test map for other ports with the NumPy lines it must log. Hashes, weights and the reference of all 128
  pad patterns are in `models/museum`. Tested only in ViZDoom.

## v0.7.1 (2026-10-28)

Fixes for what the v0.7.0 review found. Default builds do not change.

**Static extraction** ([docs/verification.md](docs/verification.md))
- The extractor said EQUAL on maps that DOOM computes wrong (a dropped `playercross` or `repeatspecial` flag, a moved teleport
  destination, a changed light tag in the reset script, a plane starting at 16,001). It now reads the trigger and repeat flags
  of every line class, mover speeds on every line, every teleport destination, the reset script (every light, plane, barrier
  and freeze walkway restored once, every doll sent home), scripts 1, 2, 4, 7 and the copy scripts, and the readout pages, and
  it runs the plane contract on the recovered net. What it does not read prints as `NOT CHECKED` (vertex coordinates and
  timing, geometry that is not a lane, scripts 8 and 9, thing angles, whether the engine executes the map).
- Single-edit mutation score of the extractor (`models/extract/mutation_score.py`, 3-4-3 net): 23,622 edits at opt 0 of which
  17,906 are detected, 19,754 at opt 3 of which 14,890; the 98 undetected edits run in DOOM on 12 inputs were all exact.
  484 of 490 random nets are equal with the contract ok (the other 6 break the contract and compile only with `limits=False`).

**Soundness** ([docs/verification.md](docs/verification.md))
- Mutation labels are conditioned: `sector-ceiling-gate` and `sector-floor-gate` hold only on nets whose planes stay within
  +-10,000 (on a plane-edge net the 16,000 -> 15,999 mutant dies). Of the 117 input gaps, 4 die on the deep set.
- `fuzz --edge` draws plane starts from 15,000 to 16,000 on about 5 % of nets (10 in 10,000 before) and covers spiking nets.
  Seeds 326047183 (opt 0, 1,000 nets), 1732841896 (opt 3, 500), 959462705 (tiles, 500): 14,782 passes; 200 spiking nets
  (seed 317167311): 12,794 passes; 0 mismatches. The weekly `fuzz.yml` ran on v0.7.0 (run 40747096084) and is cited.
  `verify --plane-sweep` prints the seven rows of the table by default.

**Schedule and cost** ([docs/scheduling.md](docs/scheduling.md), [docs/scaling.md](docs/scaling.md))
- `lint` models the first-tic stop of a barrier that the engine steps onto only when it stands more than 8 units above the
  walkway (`LOWERING_ABSORBED`): with the v0.4.1 barrier height restored it now flags 66 of the 66 failing maps of the
  reviewer's 100 nets (65 of 67 before; same nets), the current compiler's 200 maps stay clean, and the first 150 nets of the
  random campaign give 449 identical verdicts and 1 new conservative flag.
- Re-scans at 30 or more inputs: serial edge 24 (shipped 40, 1.7x), opt 3 `clock_margin` edge 160 (shipped 256, 1.6x);
  `tuning.json` is unchanged.
- `doomformer bench stack`: with half the scrollers the game tics double and a pass costs 1.8 times more; with one per sector
  the pass raises `MismatchError`. The 1.3 times of the idle table is not reachable. `plan` seconds on nets of other K ran
  0 to 40 % above the prediction (fitted and validated on K3 and K8 nets); `plan` says so. Tile throughput is given as steady
  state (81 and 880 digits a minute at 1 and 16 workers) and with game start (79 and 552). Stale numbers corrected (rms 13 %,
  90 ns per sector-tic, 2.84 s and 5.7 times, the 88 s header).

**Machines** ([docs/machine.md](docs/machine.md))
- `subleq --latch` prints the violation line only when the trace ended at a refused step; `--blind` works with `--latch`
  (`ownwrite.sq` on the doll clock: exact, latch up, level halted). The restricted form of SUBLEQ is stated plainly: plain
  SUBLEQ is universal with unbounded memory, and no argument is given for the restricted form.

**Interventions** ([docs/interventions.md](docs/interventions.md))
- The steering null at `x0` (two distinct vectors) is dropped as degenerate; new nulls: random in the span of the data,
  same-norm permuted, probes from random contexts. The probe beats those and not a shuffled-label or a 'fire vs not fire' probe.
- `circuit_check` reports per-context agreement beside the ratio of means. A planted head with negative interference
  (`planted_model(negative=True)`): the ratio of means reads 1.000 while the logit difference is right in 32 of 64 contexts;
  24 of 24 rows exact in DOOM. An AND variant looks like the OR variant to single-cell patching. 'No compact circuit' in the
  induction model holds for mean ablation only (corrupt-run baseline: 0.991 with four components, reference only).

**Quantisation** ([docs/interop.md](docs/interop.md), [models/quant/TRANSFORMER.md](models/quant/TRANSFORMER.md))
- `tm.save` and `tm.load`; `GPT(norm="layer")`, bad sizes and an all-zero calibration set raise `ValueError`; the documented
  K <= 7 and L <= 4 are what the shipped models use, not enforced limits (a K = 9, L = 5 net lowers and passes `check_limits`).
  A second corpus (Python reference prose, one seed): integer 6.52, float stage 6.13, plain float GPT 5.57, 5-gram 4.85.

**Other engines** ([docs/engine_limits.md](docs/engine_limits.md))
- The Zandronum tag failure has a cause: `short tag` in the ZA_3.2.1 source (`r_defs.h:709`, `p_udmf.cpp:1312`,
  `p_spec.cpp:274`). A statement without an artifact is removed; the UZDoom speed trial is re-run on the shipped map; one
  table gives the row counts per port.

**Player** ([docs/DEMOS.md](docs/DEMOS.md))
- The live-demon result is stated with its scenario (stationary marine, three demons 560 units away, first damage after a
  median 128 tics); the frozen-run divergence is traced to the number of dormant tics, not the dolls.

## v0.7.0 (2026-10-28)

Default builds do not change. The default SUBLEQ map is new in v0.6.0 and changes here (the violation latch): its sha256
is now `286b198b...` (ACS clock) and `eecb2053...` (doll clock).

**Static extraction** ([docs/verification.md](docs/verification.md))
- `doomformer verify --extract WAD --net NET` and `doomformer extract --dump WAD` (`doomformer/extract.py`): the net is read
  back out of a compiled one-map MLP from the `.wad` alone (TEXTMAP and the ACS copy scripts; the sidecar JSON and the
  source net are not read) and compared with the source: layer widths, every weight, bias and threshold, signed or relu
  input. The shipped MNIST net (784-64-45-10-10; 73,766 lane weights, 129 biases, 1,351 lane thresholds) comes back equal at
  opt 0 to 3; so do 490 of 490 random nets at opt 0 (10 refused for an all-zero layer) and 199 of 199 at each of opt 1, 2,
  3. Negative controls (weight +1, sign flip, retargeted line, gate height, output bias, hidden plane line and start, a
  retargeted copy script) differ exactly at the edited place. Two last-layer threshold edits that the DOOM check passed on
  its 61 inputs are shown by the extractor. Tile maps are not read. It shows what the map says, not that the engine runs it
  so. Evidence in `models/extract/`.

**Machines** ([docs/machine.md](docs/machine.md))
- A violation latch in the SUBLEQ map: 18 dolls raise `viol` when the store address falls on one of the executing
  instruction's three cells, and the level halts at that step; `machine subleq FILE --latch`. 28 of 400 random programs
  wrote their own cells (three of them were computed wrongly in v0.6.0) and all 28 were flagged at the interpreter's step;
  doll clock 9 of 9. The borrow-first update that would make the step safe was not built.
- `models/subleq/loop.sq`: 103 steps, same map, exact on the ACS clock (173,712 tics) and the doll clock (300,496 tics).
  A bubble sort of 4 or 5 values does not fit in 18 cells.

**Live demons** ([docs/DEMOS.md](docs/DEMOS.md), `models/live_demons/`)
- `python -m demos.brainlevel --live` (code in `demos/brainlive.py`): the demons stay awake while his dolls compute; the level
  samples the sensor once per decision and the action lands 773 tics later. Over 90 lives (seeds 4, 5, 6; every decision
  exact, 48,600 values) he makes 0 kills and is dead after a median 191 tics. A Python control with the same policy and
  delay makes 0 kills too, and 3.3 kills a life at a delay of 3 tics (frozen demons: 22). Sweep of the delay in the docs.
  Frozen dolls were run to 163 decisions only, not to a finished life.

**Transformers** ([docs/interop.md](docs/interop.md), [models/quant/TRANSFORMER.md](models/quant/TRANSFORMER.md))
- `df.Transformer.from_torch` and `doomformer/torch_models.py` (the integer-faithful GPT and its three-stage staircase
  recipe, moved out of `training/train_brain.py`; `training/quant_transformer.py`). A 24k-parameter level-name model:
  integer perplexity 7.75, float stage 7.34, plain float GPT 7.54, 5-gram 5.84. Three held-out names exact in DOOM tile mode
  (56,390 dolls). It does not convert a transformer trained elsewhere.

**Interventions** ([docs/interventions.md](docs/interventions.md))
- `demos.brainlevel --lesion-when near|ahead`: the lane doors close the first time the tokens the level senses satisfy a
  condition, latched; seed 4, `near`: closes at decision 22, 30 of 30 exact; seed 2, `ahead`: decision 3, 12 of 12 exact; a
  never-true control keeps the healthy actions. The 25-pair induction rows did not finish; 10 pairs per kind stand.

**Other engines** ([docs/engine_limits.md](docs/engine_limits.md#other-ports))
- The doll-clock counter machine is exported with an ACS logger and run in the other engines (`doomformer machine multiply 7 9
  --clock dolls --export DIR [--low-tags]`): the 464 steps match the interpreter's log byte for byte in GZDoom (OpenGL and
  Vulkan), LZDoom and UZDoom; a 50-step multiply and a SUBLEQ add are exact in ZDoom 2.8.1 and in Zandronum with tags below
  32,768; Zandronum with default tags stalls at step 1. The same-tic probe needs 16-unit cells in every engine. Correction:
  no engine honours `ConsoleCommand("quit")`, ViZDoom included.

**Cost** ([docs/scaling.md](docs/scaling.md), `models/bench/`)
- `doomformer bench idle | pass | tiles` (`doomformer/bench.py`): the idle-conveyor table by scroller stack (4, 2, 1, 0), the
  engine CPU per pass of the shipped MNIST net, and a tile throughput sweep (`--workers 1,2,4,8,16`). On a quiet machine
  (load average 1 to 2.6): MNIST in one map 73.2 s at `--opt 0` and 2.84 s at `--opt 3` (NumPy 47.6 us); tiles 81 digits a
  minute with 1 worker and 880 with 16 (every pass exact), 0.72 s of engine CPU per digit with 1 and 0.92 s with 16. An idle
  MNIST opt 3 map costs 10.77 ms a tic and a running pass 10.44, so the scrollers are the whole pass.
- The `plan` seconds are refitted on the quiet machine (map 89.9 ns per sector-tic and 966 ns per doll-tic; tiles 67.1 ns and
  2,254 ns): rms error on 11 held-out nets 13 % for one map (was 15 %) and 22 % for tiles (was 32 %); MNIST at `--opt 0`
  21 % over (was 37 %), at `--opt 3` 1 % under.

## v0.6.1 (2026-10-28)

Fixes for the faults found after v0.6.0. Default builds do not change, except that `df.compile` and `net.check` refuse
two more kinds of net (see Soundness) and `--opt 0` is documented as equal to v0.4.1 outside the window of the v0.5.0
barrier-height fix (2 of 300 random nets differ; the six shipped models do not).

**Soundness** ([docs/verification.md](docs/verification.md))
- Plane starts are bounded too. A plane that starts above 16,000 (the gate sector's ceiling) is exact on the first pass and
  wrong on every later one (`MLP([[[255,255,255,255]],[[-255]]], [[-16001],[0]], ...)`: DOOM 11,920 against 11,921);
  a floor lowered through a third mover step from far below reads a wrapped height (start -29,820: DOOM 4,605, reference
  12,000). `net.check`, `df.compile` and `check_limits(planes=True)` now require every plane to start in [-15,000, 16,000]
  on top of the move bound of +-14,018; an MLP bias must lie in [-16,000, 15,000]. `fuzz --edge` draws wide one-way planes;
  3,000 fresh nets (opt 0, opt 3, tiles; seeds 2109258237, 19503025, 1794520407): 22,682 passes, 0 mismatches.
- The mutation survivor classes were run on a deep input set (246 inputs). Walking-floor mutants die there (the pinned case:
  sector 231 floor -26 to -27, exact at x = 0 and 3, wrong at x = 2), so those 117 survivors are labelled input gaps, not
  equivalent. `mutation_v2.json`: 1,004 sites, 297 killed, 590 survivors with a rule, 117 input gaps, 0 unexplained. The
  superseded 301/703/33 table and a false sentence about +-1 changes are removed.
- Chunk records of `fuzz` carry `start` and the source hash; `fuzz --edge` is documented as MLP-only. Corrected numbers:
  `fuzz --nets 40 --seed 0` makes 306 passes (5,946 values, 60 inputs refused); the 100 transformers are 99 ok and 1
  unsupported.

**Schedule** ([docs/scheduling.md](docs/scheduling.md))
- `doomformer lint --net MODEL` checks the plane contract of an MLP; a net whose plane peaks at 15,003 is flagged at every
  opt (it built, linted clean and failed in DOOM at opt 2 before). A bare WAD gets a note that it cannot be checked.
- The lint campaign's 0 false negatives are for the compiler as shipped. An injected old barrier-height bug was caught on 65
  of 67 failing maps; the missed family (signed input, thresholds [5728, 5736]) is documented as not covered.
- `induction_onemap` is flagged at opt 2 and 3 (layer 9, sync wait 2 tics against 4) and exact in DOOM: documented as a
  conservative flag and added to `tune --check` (3 prompts each, exact). `copy_settle < 1` at opt 3 raises a `ValueError`
  that names it. `demos/costcheck.py --shipped` prints engine CPU per pass of the shipped MNIST net.
- Corrected text: the doll share of a tic is 8 % at MNIST opt 0, 14 % at opt 3 and 24 % for 12-16-6 K3; an idle tic is 5,
  16 and 32 times cheaper without scrollers; tiles against one map are 83 times in engine CPU and 125 times in wall time;
  the 12-16-6 net is three builds; `plan` seconds are fitted under load (102 s against 75 s on a quiet machine).

**Other engines** ([docs/engine_limits.md](docs/engine_limits.md#other-ports))
- CI on macOS failed in `test_facts`: on macOS arm64 (GitHub macos-14, ViZDoom 1.3.2 wheel) the coordinate wrap did not
  occur in 8 of the 54 probe cases. `facts.py` now expects the observed values there for those rows and says so;
  `models/ports/facts_macos14.json` and `facts_ubuntu_x86.json` hold both tables. The 32,000-unit rule is kept everywhere.
- Quit text recounted from the artifacts (Zandronum prints it in 0 of 53 logs); the Zandronum tag failure is stated as an
  observation (tags from 32,768 up are not found; probe `tag_wrap`), cause unknown. 28 probes, 55 cases; 53 rows per port:
  ZDoom 53, Zandronum 50, GZDoom family 44 matching.

**Machines** ([docs/machine.md](docs/machine.md))
- `machine ... --clock dolls --blind`: no command from the first tic to the end, then one read; `multiply 7 9` (371,780
  tics) and `subleq selfmod` (40,500 tics) exact. Four independent SUBLEQ programs added (`models/subleq/*.sq`).
- Corrected: the own-cell guard is host-side and the map computes a wrong value on a refused program (3 of 14 random ones);
  `--clock dolls` covers SUBLEQ; plane writes per program are 49 to 64; the ACS lump also holds script 8 and, for SUBLEQ,
  scripts 1 and 2.

**Quantisation** ([models/quant/EVAL.md](models/quant/EVAL.md))
- `qat.fit` refuses one-shot and empty loaders instead of hanging; non-finite calibration values raise `ValueError` with
  their index; the LayerNorm error no longer points at a converter that does not exist.
- QAT is no better than MSE calibration on ReLU Linear and Conv nets and ahead on nets with several max pools or GELU
  (`cnn_mp`: +4.6 at 2 levels, +0.5 at 4; `mlp_gelu`: +0.65 and +0.2). A 28x28 MaxPool net needs 3.2 to 3.7 GB (a separate run).

**Interventions** ([docs/interventions.md](docs/interventions.md))
- Steering controls: the 'i' against 'm' direction beats random and norm-matched random directions and does not beat
  shuffled-label probes (45, 19.5 and 21 % of 200 draws fire at least as many); a 'fire vs not fire' probe does as well.
- A planted circuit with a backup head: single patches recover 0.5 each, two minimal circuits, 80 of 80 DOOM rows exact.
  `whole site` rows (every unit of a site at the read position) are flagged and left out of the headline.

**Player** ([docs/drawpad.md](docs/drawpad.md))
- The `draw` banner and HUD say about 8 s (273 to 275 tics measured) and keep the 163 s comparison in the note on what
  `--opt 3` changes. In the seed 4 run his dolls choose FIRE first at decision 22, five times in all.

**Learning** ([docs/learning.md](docs/learning.md))
- The 20 NumPy streams include the DOOM stream (seeds 0 to 19). The clamp of the forward corridor (6 gates, scores see
  weights in [-3, 3]) is explained: about 10 points on this stream, 3.6 on average.

## v0.6.0 (2026-10-28)

Default builds do not change: `df.compile`, `plan` and every command without a new flag build the same maps as v0.5.0 (and
as v0.4.1 outside the window of the v0.5.0 barrier-height fix, see v0.6.1; `doomformer tune --hashes` compares the sha256
of the opt 0 `.wad` and `.json` of six models). Two defaults change
by design: `net.check` and `df.compile` now refuse nets and inputs that start a plane move beyond the engine's bound (see
Soundness), and `doomformer draw` builds its map with `--opt 3` (the Python API default stays `opt=0`).

**Soundness** ([docs/verification.md](docs/verification.md))
- Fixed the contract: `net.check(x)` accepted an input on which DOOM was wrong (`fuzz --edge --seed 902`, net 5: DOOM 12,254,
  reference 11,999). A plane a doll raises from above 14,018 (or lowers from below -14,018) keeps the engine's mover busy for
  3 tics and a move on that plane in the window is lost (32,768 - 150,000/8 = 14,018). `net.check`, `df.compile` (MLP: all
  inputs; transformer: `check_limits(planes=True)`), the spiking step and the per-pass transformer check now enforce it;
  tile mode has no planes and is unchanged. `limits=False` skips the compile-time check. Probe `doomformer facts --only
  mover_busy`; measurement `doomformer verify --plane-sweep`. Default builds are byte-identical; only the checks changed.
- `df.coverage(brain_or_net)` and `doomformer verify --coverage`: which of the values the reference defines a pass compares
  (MNIST: 248 of 248 read back and compared), measured by changing each readback value by 1; tests for MLP, spiking and
  transformer in one map and in tiles, with a negative control.
- `doomformer fuzz --tiles`, `--seed random`, plane-edge nets; `doomformer fuzz --mutate --tile K | --learning | --controls`;
  survivor rules with aggregate controls; a CI fuzz with a printed random seed on every push and weekly
  (`.github/workflows/fuzz.yml`).
- Campaign v2 (`models/verify/v2/`): 15,000 `fuzz --edge` nets over ten fresh seeds (6,000 at opt 0, 5,000 at opt 3, 4,000
  as tile maps): 113,605 passes, 1,987,632 values, 0 mismatches; 300 spiking nets and 100 transformers, 0 mismatches. A
  target of 10 seeds of 5,000 nets was cut to 15,000 nets because the machine was loaded.
- Mutation survivors are explained by rules (`mutate.reason`) with aggregate controls: 296 of 1,004 sites killed and 708
  survivors (297 and 707 after the deep set of v0.6.1), 0 without a rule; tile 0, tile 1 and the learning map were mutated
  too (27.4, 24.3 and 20.6 % killed; 8, 9 and
  74 survivors without a rule).

**Interventions** ([docs/interventions.md](docs/interventions.md))
- A runtime door in a one-map level: `python -m demos.brainlevel --lesion SPEC --lesion-at N` builds the healthy brain with a
  door in every lane that reads a lesioned unit (`build_onemap(doors=)`, `intervene.lane_doors`), and the level's own clock
  closes them before decision N. Python only reads. Seed 2, `--opt 3`, 12 of 12 decisions EXACT (6,480 values), the action flips
  from FIRE to TURN RIGHT at decision 3 and on every later 'in sights' decision; `--opt 0` the same. Default builds are
  byte-identical (`doors=None`).
- `add_direction` and `project_out` interventions, `df.probe` (difference-of-means direction, `doomformer/steer.py`) and
  `python -m demos.steer`: 18 of 18 steered runs EXACT in DOOM; the flip rates and the controls are in the docs.
- A compiled ground-truth circuit (`doomformer/groundtruth.py`, `models/planted`): patching recovery finds the planted cells
  and no others (2 x 104,448 single-cell patches, 0 disagreements), 126 of 126 DOOM rows EXACT.
- `patch_recovery` flags rows that patch the whole residual stream (`tautological`: last layer at the last position, 1 by
  construction; `whole residual`) and leaves them out of the headline rows; `demos.circuit --doom-pairs K` runs the DOOM rows
  on K pairs per kind.

**Machines** ([docs/machine.md](docs/machine.md))
- `df.SubleqMachine` / `doomformer machine subleq`: a SUBLEQ machine in one map. 18 memory cells of -64 to 63 hold
  instructions and data; addresses are read and written through multiplexers of equality gates; programs rewrite their own
  instructions. Add, multiply 7 x 9 (20 steps), a self-modifying sum (the first instruction's operand changes each round)
  and a fill program (its store address changes) run on one compiled map (one sha256), every step's pc and cells and every
  intermediate value equal to a Python SUBLEQ interpreter at tolerance 0 (`models/subleq/`). Negative control
  `--tamper ADDR`: a retargeted decode line diverges at the predicted step.
- `--clock dolls` (counter machines and SUBLEQ): no OPEN script. A loop doll and line specials restart the level, rewind it
  (TeleportOther, Floor_MoveToValue, Light_ChangeToValue lines) and stop it at a halt gate; the ACS lump keeps the readout pages
  and the stillness check. `multiply 7 9`, `gcd 48 18`, `fib 20`, `add 3 4` and the SUBLEQ programs exact
  (`models/dollclock/`). Engine fact found: line specials crossed in the same tic by one doll do not all act
  (`models/dollclock/probe_same_tic.py`). Default builds and the default clock are unchanged.

**Quantisation tooling** ([docs/interop.md](docs/interop.md), [models/quant/EVAL.md](models/quant/EVAL.md))
- `MLP.from_numpy` / `from_torch(calibrate="mse", equalise=)`: the MSE-optimal staircase step and the unit equalisation of
  `training/train_mnist.py`, now in the library (`doomformer/quantize.py`). Defaults are unchanged: the integer nets of
  `from_numpy` / `from_torch` are bit-identical to v0.5.0 (checked against the v0.5.0 code, `tests/test_quant.py`).
- `from_torch` folds eval-mode BatchNorm1d/2d, lowers AvgPool1d/2d and AdaptiveAvgPool1d/2d as matrices, lowers MaxPool1d/2d
  exactly on the staircase counts (comparisons only; K times the lines of the pooled layer) and takes GELU and SiLU as
  staircases of their monotone envelope (extra error 0.085 / 0.14). LayerNorm/GroupNorm still raise, with the reason.
- `doomformer.qat` (`df.qat.fit`, `Stair`, `InputStair`, `lowering_gap`): quantisation-aware fine-tuning with a
  straight-through staircase; `from_torch` lowers the result with the trained staircases.
- 8x8 digits, three seeds, writer-disjoint split (`training/quant_digits.py`): depth-6 MLP at K = 4 88.9 % (PTQ max, the v0.5.0
  behaviour) -> 93.7 % (PTQ mse) / 93.8 % (QAT), float 95.9 %; CNN 92.7 -> 95.1 / 94.8 %; BatchNorm + MaxPool + AvgPool CNN
  93.9 -> 96.0 / 96.4 %. QAT is not ahead of PTQ mse beyond the seed spread except at K = 2. The three nets at K = 4 run in
  DOOM, 5 digits each, every value exact (one map each). `Transformer.from_torch` is not part of this change.

**Schedule trust** ([docs/scheduling.md](docs/scheduling.md), [docs/tuning.md](docs/tuning.md), results in `models/schedule/`)
- `doomformer lint --fuzz` (`doomformer/lintfuzz.py`): a soundness campaign of the static checker. 1,000 random nets x opt
  1, 2, 3 x random timing constants drawn around the failure edges, 16 boundary-seeking inputs per map in DOOM at tolerance
  0: 2,947 maps scored, 0 false negatives; `--edge` runs the smallest clean value of serial and clock_margin per net and
  one below it: 2,450 maps, 0 false negatives. Seeds, commands and the conservative rate are in
  `models/schedule/lintfuzz-seed0/`.
- `plan` prints the game tics of a pass with an error band (`about 276 game tics per pass, +-7.2 %`), calibrated on 44
  measured settles and checked on 24 held-out ones (`doomformer/cost.py`, `models/schedule/plan_calibration.json`,
  `python -m demos.settlecheck`). `estimated_settle_tics` is unchanged. `plan` accepts a model directory (`models/mnist`)
  and `mnist` as well as an `.npz`.
- `doomformer tune --hashes` checks the sha256 of the opt 0 `.wad` and `.json` of six models against
  `doomformer/opt0_hashes.json` (equal to v0.4.1 and v0.5.0); default builds are byte-identical.
- Corrected the `clock_margin` edge: with the other constants shipped, the failure edge on `tiny_model` is 80, not the 72 that v0.5.0
  listed (72 failed 6 of 33 inputs); shipped 192 is 2.4 times the true edge. `doomformer tune --shipped-others` reproduces
  it. Re-checks with the shipped table: `tune --check` opt 2 331 of 331 and opt 3 169 of 169 exact, MNIST at opt 3 100 of
  100 exact (99 read right, 272 tics), DrawPad 30 of 30 walked digits at opt 3, `selfgen --loop` at opt 2 and 3. MNIST at
  opt 2 was stopped after 19 digits, all exact (the 76 of 76 of v0.5.0 stays the evidence).

**Learning** ([docs/learning.md](docs/learning.md))
- `doomformer learn digits` (`doomformer/learn_digits.py`, `learn_cli.py`): a blank multiclass perceptron in one map. Its 370
  weights (10 classes x 36 centre pixels of 8x8 digits and a bias) are ceiling heights; four layers of dolls compute the 10
  scores, take the argmax (the comparator of `df.with_argmax`) and apply the perceptron update. After each of 1,500 steps all
  455 exposed values and every weight equal a NumPy integer twin (`models/digits/learn_1500.log`); held-out accuracy 10 % to
  71.2 % (textbook NumPy perceptron, same stream: 81.0 %). Python writes the pixels and the label and reads; `StepAudit`
  counts every console command and rejects a write to a weight. `--tamper` builds one update line moving by 2 and the run
  diverges at the predicted step (three lines, logs in `models/digits/`). Data: UCI optical digits, CC BY 4.0
  (`models/digits/`). The default schedule is opt 3 (ACS copy bus); opt 0 to 2 took 20,000 tics per step and were exact on
  one step each.

**Player** ([docs/drawpad.md](docs/drawpad.md))
- `doomformer draw` builds the map with `--opt 3` by default: a READ takes 273 game tics (8 s of real-time DOOM) instead of
  5,705 (163 s); `--opt 0` is the old map. The banner and the HUD countdown say how long the READ takes. A 7 drawn through
  real X key events (Xvfb and XTest) reads EXACT at `--opt 3`, 248 values, and a CLEAR pressed during a second READ is
  reported, not checked (`models/drawpad_opt3/`).
- `python -m demos.brainlevel --opt N`: the tic budget is calibrated on the clock-free twin at the chosen level. 50
  decisions at seed 4 and `--opt 3` are 50 of 50 EXACT (27,000 values), choose the same 50 actions as `--opt 0`, and take
  774 game tics (22 s) per decision against 4,338 (124 s) (`models/brainlevel_opt/`).

**Other engines** ([docs/engine_limits.md](docs/engine_limits.md#other-ports), `models/ports/`)
- `.github/workflows/ports.yml` (manual dispatch): on a GitHub x86_64 runner it installs ZDoom 2.8.1, Zandronum 3.2.1,
  GZDoom 4.14.2, LZDoom 3.88b and UZDoom 5.0.3, runs the 50 exported probe maps and two `hello` brains in each under xvfb
  and diffs the `FACT` lines with `expected.txt`. ZDoom matches on 52 of 52 rows, Zandronum on 50 (it cannot find sector
  tags above 32,767), the GZDoom family on 43 (planes are doubles there, and its UDMF parser refuses coordinates beyond
  +-32,768; the coordinate wrap is still there). The logged values of the `hello` brain equal the NumPy reference byte for
  byte in all of them (Zandronum with `hello_tags15`).
- `doomformer facts --export` also writes `hello/` and `hello_tags15/`; new probes `mover_busy`, `wrap_shifted`,
  `line_floor_speed` and `tag_range` (27 probes, 50 maps).

## v0.5.0 (2026-10-28)

Default builds do not change: `df.compile`, `plan` and every command without a new flag build the same maps as v0.4.0 and
v0.4.1 (hashes of the one-map WADs of `doomnames_onemap`, `doomnames_onemap_softmax` and `doomplayer_onemap` compared
between the v0.4.1 tree and this one: identical; `docs/scheduling.md` records six models). One compiler bug is fixed at
every level (see Verification): a net whose walkways fall in a narrow window now gets a different barrier height, so its
map differs; no shipped model is in that window.

**Schedule levels** ([docs/scheduling.md](docs/scheduling.md), [docs/tuning.md](docs/tuning.md))
- `df.compile(net, opt=0|1|2|3)`, `df.plan(net, opt=)`, `run_many(opt=)` and `--opt N` on `mnist`, `draw`, `demo`, `hello`
  and `plan`. `opt=0` is the default and the old schedule. opt 1: dolls that can never pass their gates together share
  a wave (conflict graph colouring in `doomformer/waves.py`). opt 2: timing constants from `doomformer/tuning.json`
  and a per-layer sync wait. opt 3: light accumulators plus an ACS script that copies each layer's lights into the next
  layer's gate heights, so "no ACS between layers" holds for opt 0 to 2 only.
- Settle tics (estimated or measured, all runs exact): `hello` 488 / 384 / 240 / 160 for opt 0 / 1 / 2 / 3, `tiny_model`
  1,800 / 1,048 / 648 / 744, `doomnames_onemap` 3,488 / 1,512 / 1,136 / 768, MNIST 5,676 estimated at opt 0 and 3,548 at
  opt 2. MNIST at opt 3 measured 328 tics with `clock_margin` 512; the shipped opt 3 constants are in the tuning table.
- `doomformer tune` measures the failure edge of each timing constant against the oracle and writes
  `doomformer/tuning.json`; `tune --check` re-runs the shipped values. The table is partial: `serial`, `sync_margin` and
  `clock_margin` are measured, `corr_gap`, `tele_gap`, `chunk_lead` and `copy_settle` ship their defaults.
- `doomformer lint WAD` / `df.check_schedule`: a static checker that replays every doll tic by tic and reports plane moves
  of one sector closer than 1 tic, with a witness input, and timing margins below their minimum. `opt` 0 and 1 are clean
  on the five shipped one-map models. `doomformer plan --explain` prints per-layer tics and sectors.
- `doomformer mnist --tiles`: the MNIST net as four tile maps, Python copying the layer values between the games.
  All 10,000 test digits: 10,000 of 10,000 exact (248 values each, tolerance 0), 9,716 read correctly (97.16 %), 216 game
  tics per digit, 1,390 s on 6 workers (`models/mnist/doom_tiles_10000.json`). This is the tile design, not the one-map
  design; the one-map run of 1,000 digits stays `doom_1000.json`.
- `doomformer plan` gains a cost model (`doomformer/cost.py`: 102.1 ns per sector-tic and 1.06 us per doll-tic for one
  map; 69.7 ns, 2.6 us and 0.24 ms per run for tiles). `python -m demos.costcheck` validates it: one map 11 nets with
  root-mean-square error 18 %, tiles 11 nets with 22 %.

**Verification** ([docs/verification.md](docs/verification.md), results in `models/verify/`)
- `doomformer fuzz`: random MLPs, transformers and spiking nets compiled and run in DOOM against the reference. Campaign:
  2,000 MLPs (276,594 values), 500 range-edge MLPs, 300 each at opt 1 and opt 3, 100 transformers (all 8 attention,
  norm and MLP variants, 54,675 values), 100 spiking nets: 0 mismatches. `--plant NAME` patches one lowering function
  and the run must fail (`gate-gt` fails on 10 of 12 nets).
- What fuzz found, now fixed: a barrier that needs two mover tics left a doll one tic late and one plane move was lost
  (MLP net 591 of seed 0, edge net 246 of seed 1); a gated MLP unit with an all-zero `Wdn` column allocated a port that
  was read back from a sector that does not exist; `check_limits` raised on empty level sets.
- `doomformer fuzz --mutate`: 1,004 of 5,578 map mutation sites run, 301 killed. Every class that carries a weight, a
  sign, a target, a doll identity or an output's initial value is killed; the survivors are mostly timing constants,
  ceilings, lights of walking sectors and unread scripts, each with a stated reason.
- `doomformer verify NET`: every input class of a small MLP (`--mode classes`) or every input (`--mode all`). `--validate 20`
  checks `classes` against `all` on 20 nets: all 20 clean maps verified by both, a map with one weight line changed by +1 fails
  both on 20 of 20.
- Guards: the set of names read back must equal the set the lowering exposes; `df.compile(net, guard=True)` makes a
  readout page that never runs fail instead of returning stale values; `df.compile(transformer)` checks parameter and
  sequence ranges and raises `RangeError` before building.
- `doomformer facts`: the engine facts as 23 probes in 46 maps, PASS or FAIL per row; `facts --export DIR` writes the
  ACS-only maps.

**Causal interventions** ([docs/interventions.md](docs/interventions.md))
- `doomformer/intervene.py`: zero, set, mean, resample and scale at a named site (`x0`, `ao`, `x1`, `m`, `x2`, and now `q`,
  `k`, `v`), position and unit or head, in `intref.forward(..., interventions=[...])` and `DoomTransformer`. DOOM runs
  each one and `check_tokens` compares every tensor, patched cell and the logits with the equally intervened reference at
  tolerance 0; a wrong patch value fails by the amount the reference predicts.
- `df.patch_recovery` and `python -m doomformer.patching`: a recovery table per (site, position, head) with chosen rows
  run in DOOM. `doomplayer`: 24 of 24 rows exact; `doomplayer_xl`: 16 of 16 selected rows exact. `iv.fold` compiles a
  constant intervention into the weights.
- `python -m demos.circuit`: patching recovery per head and role and a circuit search on the induction model. Under mean
  ablation every head and every MLP layer is needed, so the circuit is the model. In DOOM: 8 of 8 patched rows exact and 4
  of 4 circuit runs exact.
- Lesion statistics (`demos.lobotomy --minimal --null`, `--names`): the 7-neuron pacifist is a 4-in-3,000 pick (64 of
  11,440 seven-subsets of `x0`); five neurons are the exhaustive minimum over all 96; 355 of the 416 single neurons change
  the DEMON CANYON sample, so DEMON DEATH is not selective. DEMOS.md and commands.md are corrected.

**Induction** (`models/induction*`, `doomformer/induction.py`, `training/train_algo.py`, [docs/DEMOS.md](docs/DEMOS.md))
- A transformer trained on a task a count table cannot solve: after a random prefix of distinct tokens, the answer is the
  token that followed the first occurrence of the current token. On held-out sequences and on a set whose queried token pairs
  never occur in training: integer L=2 d=16 K=7 0.9995 and 0.9995, float GPT of the same size 0.9991 and 0.9981, bigram
  table 0.069 and 0.000. Relu and softmax attention, K=7 and K=4, and a one-map model (`models/induction*/EVAL.md`).
- `python -m demos.induction --heldout 25`: 25 of 25 held-out sequences exact in DOOM tile maps, 520,749 values at tolerance 0.

**Machines and generation**
- `df.Machine` / `doomformer machine` ([docs/machine.md](docs/machine.md)): a counter machine in one map. Registers and a
  one-hot program counter are plane heights, the program is geometry, a halt latch ends the run, and Python compares the pc
  and every register after every step with an interpreter. Programs: add, multiply, power, gcd, fib, pow2, collatz, primes.
  `machine multiply 7 9` gives 63 in 464 steps with 24,180 values checked; `machine pow2 40` gives 1,099,511,627,776 in 942
  steps; `--tamper K` makes the trace diverge at the predicted step. `df.StoredMachine` loads two programs into one compiled
  map as initial state.
- `demo --one-map --loop --until-eos` (`doomformer/selfhalt.py`): a halt latch set by the winner at EOS ends the loop
  inside the map; one map file serves every prompt of one length.
- `python -m doomformer` runs the `doomformer` command.
- `docs/related_work.md`: prior work and how Doomformer differs, from pages opened on 2026-10-27.
- Package: `doomformer/tuning.json` and the models are in the wheel; `demos/` is not.
- Tests: 328 collected (v0.4.1: 211).

## v0.4.1 (2026-10-27)

- The README is rewritten: the proof first (walk a digit, DOOM reads it), then what is checked, how it works, the
  commands, limits and licensing. It says per demo what DOOM computes and what Python does, names the picked samples and
  the training-set scope of each result, and credits earlier work (counter.wad, conway.wad, the DOOM calculator).
- New docs: `docs/README.md` (index of the documentation), `docs/commands.md` (every command with options, run time and
  sample output), `docs/DEMOS.md` "Run it yourself", `demos/README.md`, `CHANGELOG.md`, `CITATION.cff`, `SECURITY.md`,
  `CODE_OF_CONDUCT.md`, a pull request template and an issue chooser config. `docs/HOW_IT_WORKS.md` gains an
  "In the network, in the map" table and "Who does what, per demo"; `docs/BUILD_YOUR_OWN.md` and `docs/drawpad.md` are
  corrected.
- The research demos run from a clone: `python -m demos.brainlevel`, `demos.aimlearn`, `demos.flyloop` and
  `demos.lobotomy`, each with `--seed`, a count and (except the lobotomy) `--gif PATH`. They print one line per decision
  or step with the number of values checked, and end with what Python did. `demos/` is not part of the pip package.
- CLI polish:
  - `doomformer --version`; every subcommand has help text; `hello --cases` rejects 0; `plan` of a missing model file
    says so; the `model` argument lists the shipped model names.
  - `mnist --image` prints the 28x28 picture DOOM will read before the run, warns when it does not look like one centred
    digit (too little ink, too much ink, more than 3 blobs), and exits 2 with a message for a blank picture, a text file
    or an unreadable image.
  - `--png` and `--record` paths are checked for writability before any DOOM work starts and exit 2 with
    "cannot write PATH: reason".
  - Missing build tools (git, cmake, a C compiler) are named with an install hint before the ACC build starts, and
    `doctor` reports them and exits 1.
  - `ask` prints the expected tile runs and time for the number of games before it starts (fitted to two logged runs:
    653 s with 6 games and 1,210 s with 3), and long runs print a heartbeat every 30 s.
  - The `mnist` summary says "Python wrote the 784 pixels, released the dolls and compared"; the `--loop` help says
    Python advances the clock.
- Tests: 211 collected, up from 187. New: `tests/test_cli_polish.py`, `tests/research/test_demo_cli.py` (each demo at tiny
  settings in real DOOM games), a spectator-mode check in `tests/test_drawpad.py`.
- Release assets: v0.4.0 carries the README images (`banner_wide.png`, `ladder.png`, `pipeline.png`) and the film of the
  self-writing map (`selfgen.mp4`, `selfgen.gif`, `poster_selfgen.png`). The other films stay on v0.1.0. The film site
  adds that film and an anchor per film.
- `pyproject.toml`: authors, Python 3.11 to 3.13 and Console classifiers, Documentation, Demo and Changelog URLs.

## v0.4.0 (2026-10-22)

- RMSNorm (`"norm": "rms"`), computed by the dolls: the sum of squares from a staircase whose lane k adds 2k - 1, the
  integer RMS from a fixed staircase with thresholds d*k^2, and the division by it from the same count of comparisons
  that softmax attention uses.
- Gated MLP (`"mlp": "reglu"`): the gate's dolls climb the up value's multiplier corridor, so the product never exists as
  a value; every gate they pass adds a column of the down matrix.
- `models/doomnames_modern`: RMSNorm, softmax attention and the gated MLP, 43 tile maps, 6,890 voodoo dolls. It wrote
  MILL in 170 tile runs with 62,000 values checked, all exact. 80.47 % next-character accuracy on the 136-name corpus
  against 82.06 % for the plain model (ablations in its `EVAL.md`).
- Generation without Python in one map: `doomformer demo --one-map --loop`. One map holds the transformer and a
  comparator at every position; the winner's doll moves the next position's token input to its letter and the level
  rewinds and runs the next pass by itself. Python advances the clock, runs read-only readouts (audit: 1,704 readout
  commands, 0 others) and checks every value of every pass. `doomnames_onemap` wrote DAN (5,875 dolls),
  `doomnames_onemap_softmax` wrote DAGE (6,536 dolls), both equal to greedy generation.
- ReLU-attention models give the same logits as before the change (`tests/test_softmax.py::test_relu_models_unchanged`).

## v0.3.0 (2026-10-22)

- Colab notebook `examples/doomformer_colab.ipynb`: installs doomformer, lets you draw a digit for DOOM to read, runs
  `mnist`, the softmax demo, and films a small net's voodoo dolls. The `notebook` workflow runs it headlessly on every
  push to main.
- `doomformer demo --one-map --softmax` runs `models/doomnames_onemap_softmax`: a whole transformer with softmax
  attention in one map, no Python between layers; 4,209 voodoo dolls, 123,066 sectors, 4,144 game tics, about 2 minutes
  per letter. Accuracy matches the ReLU-attention `doomnames_onemap` within seed noise (validation 55/228 for both).
- `training/train_brain.py --attn softmax`.
- CI jobs get a 75-minute limit (macOS took 33 minutes for v0.2.0).

## v0.2.0 (2026-10-22)

- Softmax attention computed by the dolls (`"attn": "softmax"`): the exponential as a staircase with thresholds
  c + sigma*ln(k+1), the division by the sum as a count of comparisons R*num - l*den >= 0, one null slot so a query
  whose scores are all below the first threshold attends to nothing. No new engine feature.
- `models/doomnames_softmax` (`doomformer demo --softmax`, trained with `training/train_qat.py --attn softmax --R 2`):
  81.13 % next-character accuracy as an integer model against 82.06 % for the ReLU model. It wrote FINAL INEALL in 494
  tile runs on 8 games, 106,834 values checked, all exact.
- Worker processes close their DOOM games when the program that started them is killed.

## v0.1.0 (2026-10-20)

First public release.

- The compiler (`doomformer/`): integer MLPs, spiking networks and transformers become one DOOM map or a set of tile
  maps; ViZDoom runs them; every value is read back and compared with an integer reference (`df.MismatchError`).
- `doomformer draw`: walk a digit on a 28 x 28 floor, the level reads it. `doomformer mnist`: the MNIST network
  (784 -> 64 -> 45 comparisons -> 10 win counts -> one-hot) is one map with 1,355 voodoo dolls; 966 of 1,000 test digits
  read correctly, all 1,000 bit-exact against the reference (`models/mnist/doom_1000.json`), and 30 of 30 digits the
  marine walked.
- `doomformer demo`: a transformer writes a DOOM level name.
- Research demos (`demos/`): the marine's policy transformer inside the level he fights in, a 106-map transformer that
  wrote DEMON CANYON, a one-game learner, a fruit-fly circuit derived from FlyWire that drives the marine, in-map perceptron
  learning.
- Twelve runnable examples (`examples/`), the guide `docs/BUILD_YOUR_OWN.md`, the engine notes `docs/engine_limits.md`.
- Release assets: the images and films the README and docs link to (listed in `SHA256SUMS`).
