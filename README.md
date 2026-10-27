<p align="center">
  <img src="https://github.com/mottopanikeiku/doomformer/releases/download/v0.4.0/banner_wide.png" alt="DOOMFORMER: the marine fights a horde of demons under a hellish sky" width="100%">
</p>

Doomformer compiles integer neural networks into ZDoom maps and runs them in ViZDoom. Voodoo dolls (the copies of the
player that DOOM spawns for extra player-1 starts) do the arithmetic, and sector light levels and plane heights hold the
sums. Python reads every value back from the running game and compares it with a NumPy integer reference, tolerance 0.
One value off by one and the run fails. It is slow: about 65,000 times slower than NumPy on a small MLP, and a handwritten
digit is 163 seconds of real-time DOOM.

[Install](#install-and-run) · [Proof](#proof-walk-a-digit-doom-reads-it) · [How it works](#how-it-works) · [Films](https://mottopanikeiku.github.io/doomformer/) · [Docs](docs/README.md)

<a href="https://github.com/mottopanikeiku/doomformer/actions/workflows/tests.yml"><img src="https://github.com/mottopanikeiku/doomformer/actions/workflows/tests.yml/badge.svg" alt="tests"></a>
<a href="https://colab.research.google.com/github/mottopanikeiku/doomformer/blob/main/examples/doomformer_colab.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open in Colab"></a>
<a href="LICENSE"><img src="https://img.shields.io/badge/license-MIT-blue" alt="MIT"></a>

<p align="center">
  <img src="https://github.com/mottopanikeiku/doomformer/releases/download/v0.4.0/ladder.png" alt="A number is a crowd of voodoo dolls. A weight is a line on the floor. A sum is a light level or a height. An activation is a staircase. Attention is a ceiling. Every value is read back and compared with NumPy: tolerance 0." width="760">
</p>

## Install and run

```sh
pip install git+https://github.com/mottopanikeiku/doomformer   # Python 3.11+, Linux or macOS
doomformer hello             # 2 s: a small MLP, every value checked
doomformer draw --digit 7    # 1.5 to 2 min: the marine walks a 7 and the level reads it
```

`doomformer doctor` checks git, cmake, ViZDoom and the display. The first run clones and builds ACC, the ACS script
compiler (a few seconds to a minute), so `git`, `cmake`, a C compiler and network access to github.com must be there. The [Colab
notebook](https://colab.research.google.com/github/mottopanikeiku/doomformer/blob/main/examples/doomformer_colab.ipynb)
needs no setup, but you draw on a browser canvas and `doomformer mnist --image` feeds it to DOOM, with no marine.

## Proof: walk a digit, DOOM reads it

<p align="center">
  <img src="https://github.com/mottopanikeiku/doomformer/releases/download/v0.1.0/drawpad.gif" alt="DrawPad: the marine walks a 7 across a 28x28 floor canvas, presses READ, and the winning voodoo doll lights a 7 on a 7-segment display" width="720">
</p>

Holding FIRE, the marine inks a 3×3 brush of floor cells under him. READ makes the level write the 784 pixels into the
network, and 1,355 voodoo dolls in the same map compute it. With `--digit`, Python stands in for you: it presses the
marine's keys and USE once, then only advances the clock and reads.

```text
          ▄▄▄▄▄▄▄▄▄▄▄▄▄▄
       ▄████████████████
       ████▀▀     ▀████▀
       ▀▀▀        ▄███▀
                  ████
                  ███
                  ███
                  ███
                 ████
                 ███▀
                 ████
                 ▀███

DOOM reads: 7  (right)   -- the winner's doll lit it on the 7-segment display
EXACT: all 248 values DOOM computed [...] equal the NumPy integer reference (tolerance 0)
READ pass: 5,705 game tics = 163 s of real-time DOOM at 35 tics/s, 79 s wall [...]
```

Abridged at `[...]`; [docs/commands.md](docs/commands.md#doomformer-draw---digit-7) has the whole output. Without `--digit`
the command opens a window and you walk and draw yourself; DOOM then runs in real time and a READ takes 2 min 43 s.

The dolls compute 784 pixels → 64 hidden units → 45 pairwise score comparisons → 10 win counts. Only the digit that wins all
nine of its comparisons gets a doll through, and the line that doll crosses starts the level script that lights the digit on
the display. Python reads back the 784 pixels, the hidden layer, every comparison and the answer and checks all 248 values
(119 values, each read with its negated twin, plus 10 outputs).

![eight MNIST test digits and what DOOM read: 7 2 1 0 4 9 6 3, all exact](https://github.com/mottopanikeiku/doomformer/releases/download/v0.1.0/mnist.png)

EXACT means DOOM equals the integer network. It says nothing about whether the digit is right. On the first 100 MNIST test
digits DOOM reads 99 correctly, and 966 of 1,000 sampled from all 10,000; every result is exact. The integer network
scores 97.16 % on all 10,000 (the float model scores 97.58 %). The 30 walked digits (0 to 9, three scripted stroke
variants each) all read correctly and exact, and the network is also trained on that stroke distribution: the first
version of this demo, a 784-32-10 network trained on MNIST alone, reads only 41.08 % of such strokes. Through real key
events only a 7 and a 1 were tried, with an earlier 48-unit network. [Details](docs/drawpad.md) ·
[film](https://mottopanikeiku.github.io/doomformer/#drawpad)

## What is checked

<p align="center">
  <img src="https://github.com/mottopanikeiku/doomformer/releases/download/v0.4.0/pipeline.png" alt="Pipeline: your network becomes integers and a DOOM map, ViZDoom runs it and Python reads the values back (light levels and plane heights). A NumPy integer reference, which never sees the map, computes the same values. Every value is compared with tolerance 0: EXACT or MismatchError." width="800">
</p>

Python reads sector light levels and floor and ceiling heights from the running game and compares each with an
independent NumPy integer implementation of the same network. The check is integer equality over the runs that were
made, and says nothing about inputs that were not run. The tests attack it: raising one weight line in a small compiled
map from 5 to 6 must fail by exactly +1 on exactly the inputs whose dolls cross that line, and a readout that echoes a
value Python wrote must fail (`pytest tests/test_tamper.py`, three tests, about 15 s).

Who does what:

- The dolls do every multiply-add, activation and comparison. ACS level scripts address cells, move input gates, rewind
  the dolls, sense the world and copy results out. None of the network's arithmetic runs in ACS. The toolkit's
  `df.Agent` is the one exception: it takes its argmax in ACS ([docs/agents.md](docs/agents.md)).
- `mnist`: Python writes the 784 pixel ports and decides when the dolls have stopped; the dolls pick the digit.
- `demo` and `ask`: Python writes the token ids and decides when the dolls have stopped, samples each letter from the
  logits DOOM computed and, when a model is cut into several maps, copies integers from one game to the next.
  `demo --one-map` is one map with no Python between layers; Python writes the tokens and samples the letter after each
  pass.
- `draw` and his brain is the level: Python only advances the clock and runs readout scripts (in `draw`, from READ on).
  The level supplies the inputs (the 784 pixels, the sensed tokens) and the dolls pick the digit or the action.
- `demo --one-map --loop`: Python only advances the clock and runs readout scripts. The dolls pick each letter and write
  it into the next position.
- The fly loop: Python only advances the clock and runs readout scripts, after calibrating the step period on a separate
  game that it does write. The level draws the input spikes and the output spikes fire the thrust lines.

In those runs a wrapper (`ReadoutOnly`) makes any other call raise, and the commands are counted: 58 in a headless
`draw` pass, 710 in the film run of his brain, 0 others.

The maps use ZDoom's UDMF format and ACS, so they need ZDoom (here
[ViZDoom](https://github.com/Farama-Foundation/ViZDoom) 1.3.2 with Freedoom assets), not the 1993 executable.
`net.quantization_report(x)` measures how far an integer model is from the float model it came from. EXACT does not.

## How it works

[docs/HOW_IT_WORKS.md](docs/HOW_IT_WORKS.md) maps each part of a network to a part of the map: a value to a group of
dolls, a weight to a line, an activation to a staircase, attention to a corridor of ceiling gates, ablation to a door.

<table>
<tr>
<td width="50%" valign="top"><img src="https://github.com/mottopanikeiku/doomformer/releases/download/v0.1.0/crossbar.png" alt="voodoo dolls crossing lava weight lines"><br>A matrix multiply. Each doll is one unit of an activation, and each lava line it crosses adds a weight.</td>
<td width="50%" valign="top"><img src="https://github.com/mottopanikeiku/doomformer/releases/download/v0.1.0/ride.png" alt="voodoo dolls jammed under attention gates"><br>Attention. Dolls squeeze under ceiling gates that another live value has lowered. Counting who gets through is a multiply.</td>
</tr>
</table>

The engine has sharp edges and the compiler works around them. Two floor moves aimed at one sector in the same tic: one is
silently lost. A teleport line cancels the lines a doll crossed earlier in that tic. Maps wider than about 34,000 units
put things in the wrong sector. [Engine facts](docs/HOW_IT_WORKS.md#engine-facts) lists each finding with the probe or
test behind it.

Computers inside DOOM maps exist already. [counter.wad](https://www.wad-archive.com/wad/8d0de31d8b392fc2d7fd817e933b3cf43bca799a) (2006) is a 4-bit counter built from voodoo dolls and conveyor belts,
[conway.wad](https://github.com/pwiecz/conway.wad) runs Game of Life in a Boom map, and [Danny Spencer's DOOM
calculator](https://github.com/nukep/doom-calculator) is an adding machine. Doomformer adds a compiler from integer
networks to ZDoom maps, a measured list of engine behaviours that break exactness, and a tolerance-0 check against a NumPy
reference.

## Commands

| Command | What DOOM does | Time | Skill level |
|---|---|---|---|
| `doomformer hello` | a small MLP, 5 random inputs | 2 s | I'm too young to die |
| `doomformer draw --digit 7` | the marine walks a 7, the level reads it | 1.5 to 2 min | |
| `doomformer mnist` | reads 4 test digits, or your own image | 1.5 to 3 min | Hey, not too rough |
| `doomformer demo` | a transformer writes a level name | 15 to 60 s | Hurt me plenty |
| `doomformer demo --nightmare` | the 2-layer model, 106 tile maps | 3.5 to 7 min | Nightmare! |
| `doomformer ask "what does iddqd do"` | answers from 189 DOOM facts | about 11 min | Ultra-Violence |

`hello` runs a 4-6-5-3 integer MLP. `mnist --image my7.png` reads your own digit. `demo` is a 1-layer transformer trained on
136 official level names, cut into 38 tile maps; `--nightmare` is the 2-layer model (56,390 dolls); `ask` is a 2-layer
transformer trained on 189 DOOM facts. Variants of `demo` add `--softmax` (attention computed by the dolls),
`--model doomnames_modern` (RMSNorm and a gated MLP computed by the dolls), `--one-map` (one forward pass of a smaller model
in one map) and `--one-map --loop` (one map writes a whole name); [docs/commands.md](docs/commands.md) has every command
with sample output.

Times are from an idle 16-core arm64 Linux machine running up to 8 games at once; `hello` and `draw` use one game. With 2
games, `demo` took about 1 minute and `demo --softmax` about 3.5 minutes on a busy machine, and `ask` took 20 minutes on 3
games.

- "Not in its training names" is a low bar: 29 of 30 unselected samples of the 2-layer model at temperature 200 are new, and the default `demo`
  writes one of its 136 training names for 6 of 10 seeds (the output says which). The default seed (20) is picked.
- DEMON CANYON, the name `--nightmare` wrote, is a hand-picked sample (seed 2, temperature 150; the number of seeds tried
  before it is not recorded). The 2-layer model was trained on 11,341 of 12,934 cleaned level names.
- `ask` gets 94.7 % of its training paraphrases right and 35.4 % of new wordings of the same facts. "what does iddqd do" is a
  training question and answers GOD MODE. A question outside the 189 facts gets an answer that is exact and meaningless.

## Bigger experiments

<p align="center">
  <img src="https://github.com/mottopanikeiku/doomformer/releases/download/v0.1.0/hero.gif" alt="His brain is the level: the marine fights a demon while 1,345 dots on the right, voodoo dolls of him, run his policy transformer" width="720">
</p>

**His brain is the level.** A level script senses the nearest demon, 1,345 voodoo dolls of the marine run his policy
transformer and pick the action, and the level turns him and pulls the trigger. Python advances the clock and reads. 50 of
50 decisions exact (seed 4). The policy imitates a scripted aimbot that sees exact demon positions. The demons are frozen
while he thinks, and the film time-lapses the thinking. [Film](https://mottopanikeiku.github.io/doomformer/#brain) ·
[code](demos/brainlevel.py)

Run the experiments from a clone (`pip install -e .`, then `python -m demos.<name>`). [demos/README.md](demos/README.md)
lists each command, its time and what it prints:

- `python -m demos.brainlevel`: 5 decisions in about 4 min; `--decisions 50 --seed 4` is the 50 of 50 run, about 31 min.
- `python -m demos.aimlearn`: 2 lives in about 3 min; `--lives 5 --seed 4` takes about 11 min.
- `python -m demos.flyloop`: 40 steps in about 3 min; `--steps 900` runs to the end at step 169, about 13 min.
- `python -m demos.lobotomy`: the lesion scan and 3 decisions in DOOM, about 1.5 min.

<table>
<tr>
<td width="50%" valign="top"><a href="https://mottopanikeiku.github.io/doomformer/#trilogy"><img src="https://github.com/mottopanikeiku/doomformer/releases/download/v0.1.0/poster_trilogy.png" alt="A transformer names a level: film poster"></a><br><b>A transformer names a level.</b> The 2-layer name model writes DEMON CANYON in 1,690 tile runs; a Python generator builds the level.<br><sub>One hand-picked sample (seed 2). Not a corpus name.</sub></td>
<td width="50%" valign="top"><a href="https://mottopanikeiku.github.io/doomformer/#doom-plays-doom"><img src="https://github.com/mottopanikeiku/doomformer/releases/download/v0.1.0/poster_doom_beats_doom.png" alt="A marine plays the level it named: film poster"></a><br><b>A marine plays the level it named.</b> A 1-layer policy on 55 games walks that level: doors, two kills, exit. 102 decisions, all exact.<br><sub>Python gives the compass and enemy positions. Seed 118 of 120, picked.</sub></td>
</tr>
<tr>
<td width="50%" valign="top"><a href="https://mottopanikeiku.github.io/doomformer/#doom-learns"><img src="https://github.com/mottopanikeiku/doomformer/releases/download/v0.1.0/poster_one_game.png" alt="A perceptron learns inside one game: film poster"></a><br><b>A perceptron learns inside one game.</b> 38 ceiling-height weights, updated by dolls from an aimbot's labels. The current code gives 4, 22, 15, 20 and 16 kills over 5 lives, 815 of 815 decisions exact.<br><sub>The film is an earlier run on an older map layout: 4, 24, 26, 22, 24 kills, 931/931 exact, seed 4. Python bins features and gives labels.</sub></td>
<td width="50%" valign="top"><a href="https://mottopanikeiku.github.io/doomformer/#lobotomy"><img src="https://github.com/mottopanikeiku/doomformer/releases/download/v0.1.0/poster_lobotomy.png" alt="Lesions: film poster"></a><br><b>Lesions.</b> A 7-pellet shotgun blast closes 7 neurons as doors and the marine stops firing: 0 kills over 74 decisions, all exact.<br><sub>The blast was picked from a scan of 3,000 random blasts.</sub></td>
</tr>
<tr>
<td width="50%" valign="top"><a href="https://mottopanikeiku.github.io/doomformer/#fruit-fly"><img src="https://github.com/mottopanikeiku/doomformer/releases/download/v0.1.0/poster_fly_alive.png" alt="A fruit-fly circuit: film poster"></a><br><b>A fruit-fly circuit.</b> 255 neurons derived from the FlyWire connectome drive the marine's lunge, grooming twitch and takeoff; refuse is silence. 169/169 steps exact.<br><sub>Simplified integer dynamics, synthetic input spikes, firing rates 0.6 to 2.7 times the float model's, one seed.</sub></td>
<td width="50%" valign="top"><a href="https://mottopanikeiku.github.io/doomformer/#one-map"><img src="https://github.com/mottopanikeiku/doomformer/releases/download/v0.1.0/poster_onemap_shil_raw.png" alt="A transformer in one map: film poster"></a><br><b>A transformer in one map.</b> No Python between layers: 3,548 dolls write SHIL, 2,680/2,680 exposed values exact.<br><sub>A 1-layer, d=8 model. Python samples the letter. Time-lapse.</sub></td>
</tr>
<tr>
<td width="50%" valign="top"><a href="https://mottopanikeiku.github.io/doomformer/#writes-itself"><img src="https://github.com/mottopanikeiku/doomformer/releases/download/v0.4.0/poster_selfgen.png" alt="Generation without Python: film poster"></a><br><b>Generation without Python.</b> The winner's doll sets the next token and the level rewinds itself; Python only advances the clock and reads. It wrote DAN in 5 passes (5,875 dolls).<br><sub>One model and the default prompt; the film time-lapses the thinking. ViZDoom lists at most 4,096 objects, so 1,779 dolls are not drawn ([engine limit](docs/engine_limits.md#object-list-vizdoom-lists-at-most-4096-objects)).</sub></td>
<td width="50%" valign="top"><a href="https://mottopanikeiku.github.io/doomformer/#full-brain"><img src="https://github.com/mottopanikeiku/doomformer/releases/download/v0.1.0/poster_full_brain.png" alt="The 4-layer transformer: film poster"></a><br><b>A 4-layer transformer.</b> It writes UNHOLY STRONGHOLD over 10,166 tile instances and 1.9 million game tics.<br><sub>Instances run on a pool of games, not 10,166 games at once.</sub></td>
</tr>
</table>

[The showreel](https://mottopanikeiku.github.io/doomformer/#showreel) covers all of them. [docs/DEMOS.md](docs/DEMOS.md)
has the numbers, the scope of each check and the list of picked seeds, and [docs/FLY.md](docs/FLY.md) covers the fly.

## Build your own

Compile a network into a map and let its outputs open doors, flash lights or shove the marine. This runs as pasted:

```python
import numpy as np
import doomformer as df

# 2 inputs -> [a AND b, a OR b] -> 2 outputs, all integers
net = df.MLP(
    weights=[np.array([[1, 1], [1, 1]]), np.array([[1, 0], [0, 1]])],
    biases=[np.array([-1, 0]), np.array([0, 0])],
    thresholds=[np.array([1]), np.array([1])],
    signed_input=False,
)
with df.compile(net) as brain:
    for x in ([0, 0], [1, 0], [1, 1]):
        # computed by voodoo dolls; raises df.MismatchError on any difference from net.reference
        print(x, "->", brain(np.array(x)).tolist())
    brain.record([np.array([0, 1]), np.array([1, 1])], "dolls.gif")   # film the dolls doing it
```

`df.MLP.from_torch` quantises a PyTorch model ([example 02](examples/02_pytorch_to_doom.py)). The `world` and `actions`
hooks put rooms in the map and attach doors, lights and thrusts to the outputs ([example
03](examples/03_brain_opens_doors.py)). The same API takes spiking networks, transformers, Conv1d and Conv2d, and learners
that update their weights inside the map. `df.with_argmax` lets the dolls pick the winning output, and `mode="tiles"`
spreads a model that does not fit one map over a pool of games (`doomformer plan` tells you which case you have). The guide
is [docs/BUILD_YOUR_OWN.md](docs/BUILD_YOUR_OWN.md), and there are twelve one-file [examples](examples). Built something
odd? [Show and tell](https://github.com/mottopanikeiku/doomformer/issues/new/choose).

## Many games at once

One map is one game, and one game uses one core. `df.run_many` gives each input to the next idle game, checks every pass
and returns the outputs in order, on your machine or on a Ray cluster you start. Nothing is hosted. 1,000 MNIST test
digits in 14 games on one 16-core machine took 2 hours: 966 read correctly, all 1,000 exact, 5,696,000 game tics or 45
hours of real-time DOOM. Every result is in [`models/mnist/doom_1000.json`](models/mnist/doom_1000.json).

```sh
pip install "doomformer[ray]"
ray start --head                  # other machines: ray start --address=HEAD:6379
doomformer mnist --all --ray --record mnist.json   # all 10,000 digits
```

Budget one core and about 1.3 GB of RAM per game (measured peak 1.24 GB). [docs/scaling.md](docs/scaling.md#scaling-out-many-games-many-machines)
has the measurements.

## Limits

- DOOM is slow. One pass of a 12→16→6 MLP takes 726 ms in DOOM and 11.1 µs in NumPy, about 65,000 times slower (30 timed
  runs, [Speed](docs/HOW_IT_WORKS.md#speed)); a fly-circuit step is about 73,000 times slower.
- Attention is integer attention: ReLU-style in most models, softmax computed by the dolls in three. No model has
  LayerNorm (`doomnames_modern` has an integer RMSNorm), and none is a drop-in float transformer. Weights are trained
  offline in PyTorch with quantisation-aware training.
- `--one-map` runs a smaller model than `demo` (1 layer, d=8, 6 positions), because the 136-name model needs 116,968 dolls
  and does not fit one map.
- The marine's policies imitate a scripted aimbot that sees exact positions. Python bins them into 86 possible observations
  and supplies the labels. There is no reward-driven RL and no learning from pixels.
- Showcased seeds, lesions and names are picked. [Seeds and selection](docs/DEMOS.md#seeds-and-selection) lists each pick.

## What's in the box

| Path | What it holds |
|---|---|
| [`doomformer/`](doomformer) | the toolkit: map compiler, DOOM runtime, exact checks, CLI |
| [`demos/`](demos) | research code: his brain is the level, the walker and level generator, the fly circuit, the learning arena |
| [`training/`](training) | quantisation-aware training for every model, with the corpora and their sources |
| [`models/`](models) | every trained model with its training log and evaluation |
| [`examples/`](examples) | twelve one-file examples, most under 15 s |
| [`tests/`](tests) | 211 tests, 92 of them on real headless DOOM; CI runs the full suite on Linux and macOS |
| [`docs/`](docs/README.md) | how it works, every demo with its numbers, the engine facts, the toolkit guide, [every command with sample output](docs/commands.md) |
| [`CHANGELOG.md`](CHANGELOG.md) | what each release added |

## Licensing

Code and docs: [MIT](LICENSE). Data keeps its own licence ([THIRD_PARTY.md](THIRD_PARTY.md)): the level-name lists are Doom
Wiki (CC BY-SA 4.0) and id Software's source (GPL-2.0), the 100 sample MNIST digits are CC BY-SA 3.0, and the fruit-fly
circuits in `models/fly` are derived from the FlyWire connectome (CC BY-NC 4.0, non-commercial). Freedoom graphics are
BSD-licensed. DOOM is a trademark of id Software / ZeniMax; this project is not affiliated with them.
