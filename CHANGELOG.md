# Changelog

Releases are tagged on GitHub: https://github.com/mottopanikeiku/doomformer/releases. Every version keeps the rule
that each value DOOM computes is checked against the NumPy integer reference with tolerance 0. The films and images the
README links to are assets of the v0.1.0 release.

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
