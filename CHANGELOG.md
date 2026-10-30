# Changelog

Releases are tagged on GitHub: https://github.com/mottopanikeiku/doomformer/releases. In every version, each value DOOM
computes is checked against the NumPy integer reference at tolerance 0. The films and images the README links to are
assets of the v0.1.0 release.

## Unreleased

- **The museum map** ([docs/museum.md](docs/museum.md)). `doomformer museum build` writes a map you play without Python.
  Light a digit on seven pads and press READ. MUTATE rewires one connection inside the running level, and a 7 reads as 1.
  `museum verify` checks every READ. `museum export` writes a self-test map for other ports.
- **Ports.** The museum self-test is exact in GZDoom, LZDoom, UZDoom and ZDoom 2.8.1. Zandronum needs the low-tag build
  (run 40907926589).
- **Guarded readback by default**, for one-map, spiking and tile maps. Before each page the map writes a poison value,
  and the page must replace it. Default build hashes change. Geometry and arithmetic do not. `guard=False` opts out.
  The fly loop stays unguarded, since 9 guarded pages do not fit its 16-tic window ([verification](docs/verification.md)).
- **ACC build.** ACC builds with dynamic Linux runtimes, so `glibc-static` is not needed. `doomformer doctor` builds a
  missing ACC and compiles a probe.
- The film site's DrawPad numbers match the README (273 tics at opt 3, 5,705 at opt 0).

## v0.7.1 (2026-10-28)

Fixes. Default builds do not change.

**Verification** ([docs/verification.md](docs/verification.md))
- The extractor now reads trigger and repeat flags, mover speeds, teleport destinations, the reset script and the readout
  pages. It runs the plane contract on the recovered net. What it does not read prints as `NOT CHECKED`.
- `models/extract/mutation_score.py` scores the extractor against single map edits.
- Mutation labels `sector-ceiling-gate` and `sector-floor-gate` hold only for planes within +-10,000.
- `fuzz --edge` draws plane starts near the 16,000 bound more often and covers spiking nets.
- `verify --plane-sweep` prints its table by default.

**Schedule and cost** ([docs/scheduling.md](docs/scheduling.md), [docs/scaling.md](docs/scaling.md))
- `lint` models the first-tic stop of a lowering barrier (`LOWERING_ABSORBED`).
- `doomformer bench stack` measures scroller stacks. `plan` warns that its fit comes from K3 and K8 nets.
- Tile throughput is given both as steady state and with game start. Stale cost numbers are corrected.

**Other changes**
- `subleq --latch` prints the violation line only on a refused step. `--blind` works with `--latch`
  ([docs/machine.md](docs/machine.md)).
- New steering nulls and per-context agreement in `circuit_check` ([docs/interventions.md](docs/interventions.md)).
- `tm.save` and `tm.load`. Bad sizes, `GPT(norm="layer")` and an all-zero calibration set raise `ValueError`
  ([docs/interop.md](docs/interop.md)).
- The Zandronum tag failure is traced to `short tag` in its source
  ([docs/engine_limits.md](docs/engine_limits.md#other-ports)).
- The live-demon result states its scenario ([docs/DEMOS.md](docs/DEMOS.md)).

## v0.7.0 (2026-10-28)

Default builds do not change. The default SUBLEQ map changes (violation latch).

**Static extraction** ([docs/verification.md](docs/verification.md#static-extraction))
- `doomformer verify --extract WAD --net NET` and `doomformer extract --dump WAD` read the net back out of a one-map
  `.wad` and compare it with the source. Evidence in `models/extract/`.

**Machines** ([docs/machine.md](docs/machine.md))
- `machine subleq FILE --latch`: dolls raise `viol` when a program writes a cell of the executing instruction, and the
  level halts.
- `models/subleq/loop.sq`: a 103-step loop, exact on both clocks.

**Player and interventions**
- `python -m demos.brainlevel --live`: the demons stay awake while his dolls compute
  ([docs/DEMOS.md](docs/DEMOS.md#live-demons-what-the-latency-costs-him)).
- `demos.brainlevel --lesion-when near|ahead` closes the lane doors when a sensed condition first holds
  ([docs/interventions.md](docs/interventions.md)).

**Transformers** ([docs/interop.md](docs/interop.md))
- `df.Transformer.from_torch` and `doomformer/torch_models.py`: the integer-faithful GPT and its training recipe.
  It does not convert a transformer trained elsewhere.

**Other engines** ([docs/engine_limits.md](docs/engine_limits.md#other-ports))
- `machine ... --clock dolls --export DIR` runs the doll-clock machines in other engines with an ACS logger.
- Correction: no engine honours `ConsoleCommand("quit")`, ViZDoom included.

**Cost** ([docs/scaling.md](docs/scaling.md))
- `doomformer bench idle | pass | tiles` measures engine CPU per pass and tile throughput.
- `plan` seconds are refitted on a quiet machine.

## v0.6.1 (2026-10-28)

Fixes. Default builds do not change. `df.compile` and `net.check` refuse two more kinds of net.

**Soundness** ([docs/verification.md](docs/verification.md#the-plane-move-bound))
- Every plane must start in [-15,000, 16,000]. An MLP bias must lie in [-16,000, 15,000].
- Walking-floor mutation survivors are relabelled input gaps. `mutation_v2.json` has 0 unexplained.
- `fuzz` chunk records carry `start` and the source hash.

**Schedule** ([docs/scheduling.md](docs/scheduling.md))
- `doomformer lint --net MODEL` checks the plane contract of an MLP.
- `induction_onemap` at opt 2 and 3 is a documented conservative flag. `copy_settle < 1` at opt 3 raises `ValueError`.
- `demos/costcheck.py --shipped` prints engine CPU per pass of the MNIST net.

**Other engines** ([docs/engine_limits.md](docs/engine_limits.md#other-ports))
- `facts.py` expects the observed coordinate values on macOS arm64. Both tables are in `models/ports/`.

**Machines** ([docs/machine.md](docs/machine.md))
- `machine ... --clock dolls --blind`: no command from the first tic to the end, then one read.
- Four SUBLEQ programs in `models/subleq/*.sq`.

**Quantisation** ([models/quant/EVAL.md](models/quant/EVAL.md))
- `qat.fit` refuses one-shot and empty loaders. Non-finite calibration values raise `ValueError` with their index.

**Interventions** ([docs/interventions.md](docs/interventions.md))
- Steering controls added. A planted circuit with a backup head. `whole site` rows are flagged.

## v0.6.0 (2026-10-28)

Default builds do not change (`doomformer tune --hashes`). By design, `net.check` and `df.compile` refuse plane moves
beyond the engine's bound, and `doomformer draw` builds with `--opt 3`.

**Soundness** ([docs/verification.md](docs/verification.md))
- Plane move bound of +-14,018 enforced by `net.check`, `df.compile` and the per-pass checks. `limits=False` skips it.
- `df.coverage` and `doomformer verify --coverage` show which reference values a pass compares.
- `fuzz --tiles`, `--seed random`, plane-edge nets, and a CI fuzz on every push and weekly (`.github/workflows/fuzz.yml`).
- Mutation survivors are explained by rules (`mutate.reason`). Campaign results in `models/verify/v2/`.

**Interventions** ([docs/interventions.md](docs/interventions.md))
- `demos.brainlevel --lesion SPEC --lesion-at N`: lane doors that the level's own clock closes.
- `add_direction`, `project_out`, `df.probe` and `python -m demos.steer`.
- A compiled ground-truth circuit (`doomformer/groundtruth.py`) to test patching recovery.
- `patch_recovery` flags rows that patch the whole residual stream.

**Machines** ([docs/machine.md](docs/machine.md))
- `df.SubleqMachine` / `doomformer machine subleq`: a SUBLEQ machine with 18 cells in one map.
- `--clock dolls`: the level clocks itself, with no OPEN script.

**Quantisation** ([docs/interop.md](docs/interop.md))
- `from_torch(calibrate="mse", equalise=)` in the library (`doomformer/quantize.py`). Defaults are unchanged.
- `from_torch` handles BatchNorm, average and max pooling, GELU and SiLU.
- `doomformer.qat`: quantisation-aware fine-tuning with a straight-through staircase.

**Schedule** ([docs/scheduling.md](docs/scheduling.md), [docs/tuning.md](docs/tuning.md))
- `doomformer lint --fuzz`: a soundness campaign of the static checker. Results in `models/schedule/`.
- `plan` prints game tics per pass with an error band and accepts a model directory.
- `doomformer tune --hashes` checks opt 0 builds of six models against `doomformer/opt0_hashes.json`.
- Correction: the `clock_margin` failure edge on `tiny_model` is 80, not 72.

**Learning** ([docs/learning.md](docs/learning.md#doom-learns-to-read-digits))
- `doomformer learn digits`: a blank multiclass perceptron that learns in one map.

**Player** ([docs/drawpad.md](docs/drawpad.md))
- `doomformer draw` defaults to `--opt 3`: a READ takes 273 game tics instead of 5,705.
- `demos.brainlevel --opt N` calibrates the tic budget per level.

**Other engines** ([docs/engine_limits.md](docs/engine_limits.md#other-ports))
- `.github/workflows/ports.yml` runs the probes and `hello` brains in five other ports.
- New probes `mover_busy`, `wrap_shifted`, `line_floor_speed` and `tag_range`.

## v0.5.0 (2026-10-28)

Default builds do not change. A compiler bug in barrier height is fixed. No shipped model is affected.

**Schedule levels** ([docs/scheduling.md](docs/scheduling.md), [docs/tuning.md](docs/tuning.md))
- `--opt 0|1|2|3` on the compiler and commands. `opt=0` is the default and the old schedule.
- `doomformer tune` measures timing-constant failure edges and writes `doomformer/tuning.json`.
- `doomformer lint WAD` / `df.check_schedule`: a static schedule checker. `plan --explain` prints per-layer cost.
- `doomformer mnist --tiles`: all 10,000 test digits exact ([docs/scaling.md](docs/scaling.md#all-10000-mnist-test-digits-in-doom)).
- `doomformer plan` gains a cost model ([docs/scaling.md](docs/scaling.md#cost-model)).

**Verification** ([docs/verification.md](docs/verification.md))
- `doomformer fuzz`: random nets run in DOOM against the reference. `--plant NAME` must make it fail.
- Fixed bugs found by fuzzing: a late doll at a two-tic barrier, a phantom port read, `check_limits` on empty level sets.
- `doomformer fuzz --mutate`, `doomformer verify NET` and readback guards.
- `doomformer facts`: the engine facts as probe maps, PASS or FAIL per row.

**Interventions** ([docs/interventions.md](docs/interventions.md))
- `doomformer/intervene.py`: zero, set, mean, resample and scale at a named site, checked in DOOM.
- `df.patch_recovery`, `iv.fold` and `python -m demos.circuit`.
- Lesion statistics ([docs/DEMOS.md](docs/DEMOS.md#lesion-statistics)).

**Induction** ([docs/DEMOS.md](docs/DEMOS.md#induction-a-transformer-that-learned-an-algorithm-modelsinduction))
- A transformer trained on an induction task a count table cannot solve, run in DOOM.

**Machines and other**
- `df.Machine` / `doomformer machine`: a counter machine in one map ([docs/machine.md](docs/machine.md)).
- `demo --one-map --loop --until-eos`: the map stops itself at EOS.
- `python -m doomformer`, `docs/related_work.md`, `tuning.json` and models in the wheel.

## v0.4.1 (2026-10-27)

- README rewritten. New docs: `docs/README.md`, `docs/commands.md`, `CITATION.cff`, `SECURITY.md` and others.
- Research demos run from a clone: `demos.brainlevel`, `demos.aimlearn`, `demos.flyloop`, `demos.lobotomy`.
- CLI: `--version`, help text, input checks for `mnist --image`, writable-path checks, build-tool hints in `doctor`.
- `ask` prints expected run time and a heartbeat.
- Release assets for README images and the self-writing map film.
- `pyproject.toml` metadata.

## v0.4.0 (2026-10-22)

- RMSNorm (`"norm": "rms"`) and gated MLP (`"mlp": "reglu"`), computed by the dolls.
- `models/doomnames_modern`: RMSNorm, softmax attention and gated MLP in tile maps.
- `doomformer demo --one-map --loop`: generation without Python between passes
  ([docs/HOW_IT_WORKS.md](docs/HOW_IT_WORKS.md#generation-without-python)).

## v0.3.0 (2026-10-22)

- Colab notebook `examples/doomformer_colab.ipynb`, run headless in CI.
- `doomformer demo --one-map --softmax`: a softmax transformer in one map.
- `training/train_brain.py --attn softmax`.
- CI jobs get a 75-minute limit.

## v0.2.0 (2026-10-22)

- Softmax attention computed by the dolls (`"attn": "softmax"`)
  ([docs/HOW_IT_WORKS.md](docs/HOW_IT_WORKS.md#softmax-attention)).
- `models/doomnames_softmax` (`doomformer demo --softmax`).
- Worker processes close their DOOM games when the parent is killed.

## v0.1.0 (2026-10-20)

First public release.

- The compiler (`doomformer/`): integer MLPs, spiking nets and transformers become DOOM maps, checked value by value.
- `doomformer draw` and `doomformer mnist`: DOOM reads a digit.
- `doomformer demo`: a transformer writes a DOOM level name.
- Research demos in `demos/`, examples in `examples/`, guide `docs/BUILD_YOUR_OWN.md`.
- Release assets listed in `SHA256SUMS`.
