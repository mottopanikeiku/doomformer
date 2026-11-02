# Changelog

Releases are tagged on [GitHub](https://github.com/mottopanikeiku/doomformer/releases). Every version compares the values
read from DOOM with the NumPy integer reference, at tolerance 0. The original films and images linked from the README
are v0.1.0 assets; the self-writing map film is on v0.4.0.

## Unreleased

- The [museum map](docs/museum.md) runs without Python after `doomformer museum build`: you light a digit on seven
  pads and press READ, and MUTATE rewires one connection inside the running level so that a 7 reads as 1, while
  `museum verify` checks every READ and `museum export` writes a self-test map for other ports.
- The museum self-test matches the reference in GZDoom, LZDoom, UZDoom and ZDoom 2.8.1, while Zandronum needs the
  low-tag build.
- `doomformer draw --build DIR` writes the DrawPad map as a standalone WAD that GZDoom plays without Python, and
  `draw --export DIR` writes its self-test, which inks the digits 0 to 9, reads each one and logs all 248 values so
  that other ports can be checked against the NumPy reference ([drawpad](docs/drawpad.md#in-gzdoom-without-python)).
- Readback guards are enabled by default for one-map, spiking and tile maps, so before each page the map writes a
  poison value that the page must replace. This changes default build hashes while geometry and arithmetic stay the
  same, `guard=False` opts out, and the fly loop stays unguarded because 9 guarded pages exceed its 16-tic window
  ([verification](docs/verification.md)).
- ACC links dynamic Linux runtimes, which removes the `glibc-static` requirement, and `doomformer doctor` builds a
  missing ACC and compiles a probe.
- The 1 s deadline in `test_hung_worker_is_killed_at_the_deadline` starts after the healthy worker returns its result,
  which removes a dependency on runner speed that made CI flaky, and `TilePool` is unchanged.

## v0.7.1 (2026-10-28)

Bug fixes; default builds are unchanged.

**Verification** ([docs/verification.md](docs/verification.md))
- The extractor reads trigger and repeat flags, mover speeds, teleport destinations, the reset script and the readout
  pages. It checks the recovered network's plane limits. Fields it does not read print as `NOT CHECKED`.
- `models/extract/mutation_score.py` scores how often the extractor catches single map edits.
- Mutation labels `sector-ceiling-gate` and `sector-floor-gate` apply only to planes within +-10,000.
- `fuzz --edge` samples plane starts near the 16,000 bound more often and covers spiking nets.
- `verify --plane-sweep` prints its table by default.

**Schedule and cost** ([docs/scheduling.md](docs/scheduling.md), [docs/scaling.md](docs/scaling.md))
- `lint` models the first-tic stop of a lowering barrier (`LOWERING_ABSORBED`).
- `doomformer bench stack` measures scroller stacks. `plan` warns that its fit comes from K3 and K8 nets.
- Report tile throughput both at steady state and including game startup. Correct outdated cost figures.

**Other changes**
- `subleq --latch` prints the violation line only on a refused step. `--blind` works with `--latch`
  ([docs/machine.md](docs/machine.md)).
- Add null controls for steering and per-context agreement in `circuit_check` ([docs/interventions.md](docs/interventions.md)).
- Add `tm.save` and `tm.load`. Invalid sizes, `GPT(norm="layer")` and an all-zero calibration set raise `ValueError`
  ([docs/interop.md](docs/interop.md)).
- Trace the Zandronum tag failure to `short tag` in its source
  ([docs/engine_limits.md](docs/engine_limits.md#other-ports)).
- Document the live-demon test scenario ([docs/DEMOS.md](docs/DEMOS.md)).

## v0.7.0 (2026-10-28)

Default neural-network builds are unchanged. The default SUBLEQ map gains a violation latch.

**Static extraction** ([docs/verification.md](docs/verification.md#static-extraction))
- `doomformer verify --extract WAD --net NET` and `doomformer extract --dump WAD` recover the network from a one-map
  `.wad` and compare it with the source. Results are in `models/extract/`.

**Machines** ([docs/machine.md](docs/machine.md))
- `machine subleq FILE --latch`: dolls raise `viol` and halt the level when a program writes a cell of the instruction
  being executed.
- `models/subleq/loop.sq`: a 103-step loop, exact on both clocks.

**Player and interventions**
- `python -m demos.brainlevel --live`: the demons stay awake while his dolls compute
  ([docs/DEMOS.md](docs/DEMOS.md#live-demons-what-the-latency-costs-him)).
- `demos.brainlevel --lesion-when near|ahead` closes lane doors when a sensed condition first becomes true
  ([docs/interventions.md](docs/interventions.md)).

**Transformers** ([docs/interop.md](docs/interop.md))
- Add `df.Transformer.from_torch` and `doomformer/torch_models.py` for GPT models trained with the supplied
  integer-compatible recipe. Transformers trained elsewhere are unsupported.

**Other engines** ([docs/engine_limits.md](docs/engine_limits.md#other-ports))
- `machine ... --clock dolls --export DIR` runs the doll-clock machines in other engines with an ACS logger.
- Correction: no engine honours `ConsoleCommand("quit")`, ViZDoom included.

**Cost** ([docs/scaling.md](docs/scaling.md))
- `doomformer bench idle | pass | tiles` measures engine CPU per pass and tile throughput.
- Refit `plan` timing estimates on a quiet machine.

## v0.6.1 (2026-10-28)

Bug fixes; default builds are unchanged. `df.compile` and `net.check` reject two more kinds of network.

**Soundness** ([docs/verification.md](docs/verification.md#the-plane-move-bound))
- Require every plane to start in [-15,000, 16,000] and MLP biases to lie in [-16,000, 15,000].
- Classify surviving walking-floor mutations as gaps in the inputs tested. None remain unexplained in `mutation_v2.json`.
- Include `start` and the source hash in `fuzz` chunk records.

**Schedule** ([docs/scheduling.md](docs/scheduling.md))
- `doomformer lint --net MODEL` checks the plane contract of an MLP.
- `induction_onemap` at opt 2 and 3 is a documented conservative flag. `copy_settle < 1` at opt 3 raises `ValueError`.
- `demos/costcheck.py --shipped` prints engine CPU per pass of the MNIST net.

**Other engines** ([docs/engine_limits.md](docs/engine_limits.md#other-ports))
- Update coordinate expectations in `facts.py` for macOS arm64. Both platform tables are in `models/ports/`.

**Machines** ([docs/machine.md](docs/machine.md))
- `machine ... --clock dolls --blind`: no command from the first tic to the end, then one read.
- Four SUBLEQ programs in `models/subleq/*.sq`.

**Quantisation** ([models/quant/EVAL.md](models/quant/EVAL.md))
- `qat.fit` rejects one-shot and empty loaders. Non-finite calibration values raise `ValueError` with their index.

**Interventions** ([docs/interventions.md](docs/interventions.md))
- Add steering controls and a planted circuit with a backup head; flag `whole site` rows.

## v0.6.0 (2026-10-28)

Default builds are unchanged (`doomformer tune --hashes`). `net.check` and `df.compile` reject plane moves beyond the
engine's bound. `doomformer draw` now builds with `--opt 3`.

**Soundness** ([docs/verification.md](docs/verification.md))
- Enforce the +-14,018 plane-move bound in `net.check`, `df.compile` and the per-pass checks. `limits=False` skips it.
- `df.coverage` and `doomformer verify --coverage` show which reference values a pass compares.
- Add `fuzz --tiles`, `--seed random`, plane-edge networks and CI fuzzing on every push and weekly
  (`.github/workflows/fuzz.yml`).
- Rules in `mutate.reason` explain surviving mutations. Results are in `models/verify/v2/`.

**Interventions** ([docs/interventions.md](docs/interventions.md))
- `demos.brainlevel --lesion SPEC --lesion-at N` closes lane doors using the level's own clock.
- Add `add_direction`, `project_out`, `df.probe` and `python -m demos.steer`.
- Add a compiled ground-truth circuit (`doomformer/groundtruth.py`) to test patching recovery.
- `patch_recovery` flags rows that patch the whole residual stream.

**Machines** ([docs/machine.md](docs/machine.md))
- `df.SubleqMachine` / `doomformer machine subleq`: a SUBLEQ machine with 18 cells in one map.
- `--clock dolls`: the level clocks itself, with no OPEN script.

**Quantisation** ([docs/interop.md](docs/interop.md))
- `from_torch(calibrate="mse", equalise=)` in the library (`doomformer/quantize.py`). Defaults are unchanged.
- `from_torch` handles BatchNorm, average and max pooling, GELU and SiLU.
- `doomformer.qat`: quantisation-aware fine-tuning with a straight-through staircase.

**Schedule** ([docs/scheduling.md](docs/scheduling.md), [docs/tuning.md](docs/tuning.md))
- `doomformer lint --fuzz` tests the static checker against running maps. Results are in `models/schedule/`.
- `plan` prints game tics per pass with an error band and accepts a model directory.
- `doomformer tune --hashes` checks opt 0 builds of six models against `doomformer/opt0_hashes.json`.
- Correct the `clock_margin` failure edge on `tiny_model` from 72 to 80.

**Learning** ([docs/learning.md](docs/learning.md#doom-learns-to-read-digits))
- `doomformer learn digits`: a blank multiclass perceptron that learns in one map.

**Player** ([docs/drawpad.md](docs/drawpad.md))
- `doomformer draw` defaults to `--opt 3`: a READ takes 273 game tics instead of 5,705.
- `demos.brainlevel --opt N` calibrates the tic budget per level.

**Other engines** ([docs/engine_limits.md](docs/engine_limits.md#other-ports))
- `.github/workflows/ports.yml` runs the probes and `hello` brains in five other ports.
- New probes `mover_busy`, `wrap_shifted`, `line_floor_speed` and `tag_range`.

## v0.5.0 (2026-10-28)

Fix the compiler's barrier-height bug. Default builds and shipped models are unaffected.

**Schedule levels** ([docs/scheduling.md](docs/scheduling.md), [docs/tuning.md](docs/tuning.md))
- Add `--opt 0|1|2|3` to the compiler and commands. `opt=0` keeps the original schedule and remains the default.
- `doomformer tune` measures when timing constants fail and writes `doomformer/tuning.json`.
- Add `doomformer lint WAD` / `df.check_schedule` for static schedule checks. `plan --explain` prints per-layer cost.
- All 10,000 MNIST test digits match the reference with `doomformer mnist --tiles`
  ([docs/scaling.md](docs/scaling.md#all-10000-mnist-test-digits-in-doom)).
- Add a cost model to `doomformer plan` ([docs/scaling.md](docs/scaling.md#cost-model)).

**Verification** ([docs/verification.md](docs/verification.md))
- `doomformer fuzz` runs random networks in DOOM against the reference. `--plant NAME` must make it fail.
- Fix bugs found by fuzzing: a late doll at a two-tic barrier, a phantom port read and `check_limits` on empty level sets.
- Add `doomformer fuzz --mutate`, `doomformer verify NET` and readback guards.
- `doomformer facts` runs engine-fact probe maps and reports PASS or FAIL for each row.

**Interventions** ([docs/interventions.md](docs/interventions.md))
- `doomformer/intervene.py` applies zero, set, mean, resample and scale at a named site, checked in DOOM.
- Add `df.patch_recovery`, `iv.fold` and `python -m demos.circuit`.
- Lesion statistics ([docs/DEMOS.md](docs/DEMOS.md#lesion-statistics)).

**Induction** ([docs/DEMOS.md](docs/DEMOS.md#induction-a-transformer-that-learned-an-algorithm-modelsinduction))
- Run a transformer in DOOM after training it on an induction task that a count table cannot solve.

**Machines and other**
- `df.Machine` / `doomformer machine`: a counter machine in one map ([docs/machine.md](docs/machine.md)).
- `demo --one-map --loop --until-eos` lets the map stop itself at EOS.
- Add `python -m doomformer` and `docs/related_work.md`. Include `tuning.json` and models in the wheel.

## v0.4.1 (2026-10-27)

- Rewrite the README. Add `docs/README.md`, `docs/commands.md`, `CITATION.cff` and other documentation.
- Run research demos from a clone: `demos.brainlevel`, `demos.aimlearn`, `demos.flyloop`, `demos.lobotomy`.
- Add CLI `--version`, help text, `mnist --image` input validation, writable-path checks and build-tool hints in `doctor`.
- `ask` prints the expected run time and a heartbeat.
- Add release assets for README images and the self-writing map film.
- Update `pyproject.toml` metadata.

## v0.4.0 (2026-10-22)

- Compute RMSNorm (`"norm": "rms"`) and a gated MLP (`"mlp": "reglu"`) with the dolls.
- Add `models/doomnames_modern`: RMSNorm, softmax attention and a gated MLP in tile maps.
- `doomformer demo --one-map --loop` generates without Python between passes
  ([docs/HOW_IT_WORKS.md](docs/HOW_IT_WORKS.md#generation-without-python)).

## v0.3.0 (2026-10-22)

- Add `examples/doomformer_colab.ipynb` and run the notebook headless in CI.
- `doomformer demo --one-map --softmax` runs a softmax transformer in one map.
- Add `training/train_brain.py --attn softmax`.
- Set a 75-minute limit for CI jobs.

## v0.2.0 (2026-10-22)

- Compute softmax attention with the dolls (`"attn": "softmax"`)
  ([docs/HOW_IT_WORKS.md](docs/HOW_IT_WORKS.md#softmax-attention)).
- Add `models/doomnames_softmax` (`doomformer demo --softmax`).
- Worker processes close their DOOM games when the parent is killed.

## v0.1.0 (2026-10-20)

First public release.

- Compile integer MLPs, spiking networks and transformers into DOOM maps with `doomformer/`, checked value by value.
- Read digits in DOOM with `doomformer draw` and `doomformer mnist`.
- Generate a DOOM level name with a transformer using `doomformer demo`.
- Add research demos in `demos/`, examples in `examples/` and the guide in `docs/BUILD_YOUR_OWN.md`.
- List release assets in `SHA256SUMS`.
