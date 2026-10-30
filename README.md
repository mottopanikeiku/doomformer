<p align="center">
  <img src="https://github.com/mottopanikeiku/doomformer/releases/download/v0.4.0/banner_wide.png" alt="DOOMFORMER: the marine fights a horde of demons under a hellish sky" width="100%">
</p>

Doomformer compiles integer neural networks into ZDoom maps. Voodoo dolls, the copies of the player that DOOM spawns
for extra player starts, do the arithmetic. Light levels and floor heights hold the sums. Python reads the values back
from the running game and compares each one with a NumPy integer reference. One value off by one and the run fails.

It is slow. A small MLP runs about 23,000 times slower than in NumPy, and reading one handwritten digit takes 8 seconds
of DOOM time.

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
doomformer draw --digit 7    # 15 s: the marine walks a 7 and the level reads it
```

`doomformer doctor` checks your setup and builds ACC, the ACS compiler, if it is missing. That needs git, cmake and a C
compiler. On Fedora: `sudo dnf install git cmake gcc gcc-c++ make`. The [Colab notebook](https://colab.research.google.com/github/mottopanikeiku/doomformer/blob/main/examples/doomformer_colab.ipynb)
needs no setup: you draw on a browser canvas and DOOM reads it.

## Proof: walk a digit, DOOM reads it

<p align="center">
  <img src="https://github.com/mottopanikeiku/doomformer/releases/download/v0.1.0/drawpad.gif" alt="DrawPad: the marine walks a 7 across a 28x28 floor canvas, presses READ, and the winning voodoo doll lights a 7 on a 7-segment display" width="720">
</p>

The marine holds FIRE to ink the floor under him, then presses READ. The level writes the 784 pixels into a network,
and 1,355 voodoo dolls in the same map compute the answer. The doll of the winning digit crosses a line that lights the
digit on a display.

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
READ pass: 273 game tics = 8 s of real-time DOOM at 35 tics/s, 3 s wall [...]
```

The network is 784 pixels → 64 hidden units → 45 pairwise comparisons → 10 win counts. Python reads back 248 values
and compares each one. With `--digit`, Python presses the marine's keys for you and then only reads. Without it, you
walk and draw yourself. [Full output](docs/commands.md#doomformer-draw---digit-7)

EXACT means DOOM equals the integer network. It does not mean the digit is right:

| run | right | exact |
|---|---|---|
| first 100 MNIST test digits | 99 | 100 |
| 1,000 sampled test digits, one map each | 966 | 1,000 |
| all 10,000 test digits, tile maps | 9,716 (97.16 %) | 10,000 |

The float model scores 97.58 %. The network is also trained on the strokes the scripted marine walks. A version trained
on MNIST alone reads only 41 % of them. [Details](docs/drawpad.md) · [Film](https://mottopanikeiku.github.io/doomformer/#drawpad)

![eight MNIST test digits and what DOOM read: 7 2 1 0 4 9 6 3, all exact](https://github.com/mottopanikeiku/doomformer/releases/download/v0.1.0/mnist.png)

## The museum: light a digit, rewire one connection

`doomformer museum build` writes `museum.wad`. You can play it in GZDoom with no Python at all.

Step on seven floor pads to light a digit, then press READ. 136 voodoo dolls run a tiny hand-written network and light
the answer. Press MUTATE and the level rewires one connection while it runs. Now the same 7 reads as a 1. RESTORE puts
it back.

```text
$ doomformer museum verify
7                        pads abc     w_7a +1  DOOM reads 7  scores [4, 6, 3, 5, 4, 3, 2, 7, 3, 4]  EXACT (148 values; 744 tics; [...])
7 after MUTATE           pads abc     w_7a -1  DOOM reads 1  scores [4, 6, 3, 5, 4, 3, 2, 5, 3, 4]  EXACT (148 values; 744 tics; [...])
7 after RESTORE          pads abc     w_7a +1  DOOM reads 7  scores [4, 6, 3, 5, 4, 3, 2, 7, 3, 4]  EXACT (148 values; 744 tics; [...])
```

A self-test version of the map gives the exact values in GZDoom, LZDoom, UZDoom and ZDoom 2.8.1. Zandronum needs a
build with low sector tags. [Details](docs/museum.md)

## What is checked

<p align="center">
  <img src="https://github.com/mottopanikeiku/doomformer/releases/download/v0.4.0/pipeline.png" alt="Pipeline: your network becomes integers and a DOOM map, ViZDoom runs it and Python reads the values back (light levels and plane heights). A NumPy integer reference, which never sees the map, computes the same values. Every value is compared with tolerance 0: EXACT or MismatchError." width="800">
</p>

Every run compares each value the map exposes with the NumPy reference. That proves equality for the inputs that were
run, nothing more. The rest of the evidence attacks the check:

- **Tampering.** Raise one weight line by 1 and the check fails on exactly the inputs that cross it.
- **Guarded readback.** Before each readout page the map writes a poison value, and the page must overwrite it. A page
  that never runs cannot pass off old values as new.
- **Fuzzing.** `doomformer fuzz` compiles random nets and compares every value. 20,000 nets at the range limits gave
  0 mismatches. CI fuzzes a fresh seed on every push.
- **Static extraction.** `doomformer verify --extract` reads the weights back out of a compiled map, without DOOM, and
  compares them with the source net.
- **Mutation.** Change one line of a map and see whether the check notices. 297 of 1,004 mutants are caught. None of
  the survivors is proven harmless.
- **Coverage.** The MNIST map exposes 248 values. Its 903 staircase counts and 1,351 gate comparisons are checked only
  through the sums they feed.

Fuzzing and targeted tests found four bugs in the compiler's own maps. All are fixed. [Verification](docs/verification.md)

| run | Python | voodoo dolls |
|---|---|---|
| `mnist`, `df.compile` | writes the inputs, waits, reads | every sum, activation, comparison and the choice |
| `demo`, `ask` (tile maps) | writes tokens, copies values between maps, samples letters | every sum and activation |
| `draw`, `--loop`, the marine's brain, the fly | advances the clock and reads; any other call raises | everything, including the choice |
| `machine` | loads the program, then advances the clock and reads | program counter and registers |

ACS level scripts move gates, rewind the dolls, sense the world and copy results out. Two do more: `df.Agent` takes an
argmax in ACS, and `--opt 3` copies each layer's values into the next layer with a script.
[Per demo](docs/HOW_IT_WORKS.md#who-does-what-per-demo)

The maps use UDMF and ACS, so they need ZDoom. This repo runs [ViZDoom](https://github.com/Farama-Foundation/ViZDoom)
1.3.2 with Freedoom assets.

## How it works

Each part of a network becomes a part of the map. A value is a group of dolls and a weight is a line. An activation is
a staircase, and attention is a corridor of ceiling gates. [How it works](docs/HOW_IT_WORKS.md)

<table>
<tr>
<td width="50%" valign="top"><img src="https://github.com/mottopanikeiku/doomformer/releases/download/v0.1.0/crossbar.png" alt="voodoo dolls crossing lava weight lines"><br>A matrix multiply. Each doll is one unit of an activation, and each lava line it crosses adds a weight.</td>
<td width="50%" valign="top"><img src="https://github.com/mottopanikeiku/doomformer/releases/download/v0.1.0/ride.png" alt="voodoo dolls jammed under attention gates"><br>Attention. Dolls squeeze under ceiling gates that another value has lowered. Counting who gets through is a multiply.</td>
</tr>
</table>

The engine has sharp edges. A floor mover drops a second request while the first is still moving. A teleport cancels
the lines a doll crossed earlier in the same tic. A thing far from a BSP node can land in the wrong sector, so every map
fits in 32,000 units. `doomformer facts` re-runs 28 probes for these in about 10 s. [Engine facts](docs/HOW_IT_WORKS.md#engine-facts)

The probe maps, a small MLP and a counter machine also ran in stock ZDoom, Zandronum, GZDoom, LZDoom and UZDoom. The
MLP and the machine match the reference in all five. Zandronum needs a build with low sector tags, and some probes
differ in GZDoom, which keeps plane heights as doubles. [Other ports](docs/engine_limits.md#other-ports)

Computers in DOOM maps exist already. [counter.wad](https://www.wad-archive.com/wad/8d0de31d8b392fc2d7fd817e933b3cf43bca799a)
(2006) counts with voodoo dolls, [conway.wad](https://github.com/pwiecz/conway.wad) runs Game of Life, and [Danny
Spencer's calculator](https://github.com/nukep/doom-calculator) adds numbers. Doomformer adds a compiler for neural
networks, a list of engine behaviours that break exactness, and the tolerance-0 check. [Related work](docs/related_work.md)

## Commands

| Command | What DOOM does | Time | Skill level |
|---|---|---|---|
| `doomformer hello` | a small MLP, 5 random inputs | 2 s | I'm too young to die |
| `doomformer draw --digit 7` | the marine walks a 7, the level reads it | 15 s | |
| `doomformer mnist` | reads 4 test digits, or your own image | 1.5 to 3 min | Hey, not too rough |
| `doomformer demo` | a transformer writes a level name | 15 to 60 s | Hurt me plenty |
| `doomformer demo --nightmare` | the 2-layer model, 106 tile maps | 3.5 to 7 min | Nightmare! |
| `doomformer ask "what does iddqd do"` | answers from 189 DOOM facts | about 11 min | Ultra-Violence |
| `doomformer machine subleq all` | a SUBLEQ machine runs three programs, one rewriting itself | about 5 min | |
| `doomformer learn digits --classes 4 --steps 300` | a blank perceptron learns 8×8 digits inside one map | about 5 min | |
| `doomformer museum build` | a playable map that reads a digit, no Python needed | 1 s | |

Times are from a 16-core arm64 Linux machine. [docs/commands.md](docs/commands.md) lists every command with sample
output.

Some outputs are picked. DEMON CANYON, the name `--nightmare` writes, is a hand-picked sample. The default `demo` seed is
picked too, and the model writes one of its training names for 6 of 10 nearby seeds. `ask` answers 94.7 % of its
training questions and 35.4 % of new wordings. A question outside its 189 facts gets an exact, meaningless answer.

## Bigger experiments

<p align="center">
  <img src="https://github.com/mottopanikeiku/doomformer/releases/download/v0.1.0/hero.gif" alt="His brain is the level: the marine fights a demon while 1,345 dots on the right, voodoo dolls of him, run his policy transformer" width="720">
</p>

**His brain is the level.** A level script senses the nearest demon. 1,345 voodoo dolls of the marine run his policy
transformer and pick an action, and the level turns him and fires. Python only advances the clock and reads. 50 of 50
decisions are exact. The policy imitates a scripted aimbot. [Film](https://mottopanikeiku.github.io/doomformer/#brain) ·
[Code](demos/brainlevel.py)

He is not playable. A decision takes 774 game tics (22 s), and the demons are frozen while he thinks. With them awake
(`--live`) he dies in 5.5 s without a kill. [Live demons](docs/DEMOS.md)

Run the experiments from a clone with `pip install -e .`. [demos/README.md](demos/README.md) has times and sample
output.

| command | what it shows |
|---|---|
| `python -m demos.brainlevel` | his brain is the level |
| `python -m demos.brainlevel --live` | the same brain against awake demons |
| `python -m demos.brainlevel --lesion-when near` | the level closes neuron doors when a demon comes close |
| `python -m demos.aimlearn` | a perceptron learns to aim inside one game |
| `python -m demos.flyloop` | a fruit-fly circuit drives the marine |
| `python -m demos.lobotomy` | lesions that make him a pacifist |

<table>
<tr>
<td width="50%" valign="top"><a href="https://mottopanikeiku.github.io/doomformer/#trilogy"><img src="https://github.com/mottopanikeiku/doomformer/releases/download/v0.1.0/poster_trilogy.png" alt="A transformer names a level: film poster"></a><br><b>A transformer names a level.</b> The 2-layer name model writes DEMON CANYON, and a Python generator builds the level.<br><sub>One hand-picked sample.</sub></td>
<td width="50%" valign="top"><a href="https://mottopanikeiku.github.io/doomformer/#doom-plays-doom"><img src="https://github.com/mottopanikeiku/doomformer/releases/download/v0.1.0/poster_doom_beats_doom.png" alt="A marine plays the level it named: film poster"></a><br><b>A marine plays the level it named.</b> A 1-layer policy walks it: doors, two kills, exit. 102 decisions, all exact.<br><sub>Python gives the compass and enemy positions. Seed 118 of 120, picked.</sub></td>
</tr>
<tr>
<td width="50%" valign="top"><a href="https://mottopanikeiku.github.io/doomformer/#doom-learns"><img src="https://github.com/mottopanikeiku/doomformer/releases/download/v0.1.0/poster_one_game.png" alt="A perceptron learns inside one game: film poster"></a><br><b>A perceptron learns inside one game.</b> 38 ceiling-height weights, updated by dolls from an aimbot's labels.<br><sub>The film is an older run. Python bins the features and gives the labels.</sub></td>
<td width="50%" valign="top"><a href="https://mottopanikeiku.github.io/doomformer/#lobotomy"><img src="https://github.com/mottopanikeiku/doomformer/releases/download/v0.1.0/poster_lobotomy.png" alt="Lesions: film poster"></a><br><b>Lesions.</b> A 7-pellet blast closes 7 neurons and the marine stops firing. 74 decisions, all exact.<br><sub>Picked from 3,000 random blasts.</sub></td>
</tr>
<tr>
<td width="50%" valign="top"><a href="https://mottopanikeiku.github.io/doomformer/#fruit-fly"><img src="https://github.com/mottopanikeiku/doomformer/releases/download/v0.1.0/poster_fly_alive.png" alt="A fruit-fly circuit: film poster"></a><br><b>A fruit-fly circuit.</b> 255 neurons derived from the FlyWire connectome drive the marine. 169 of 169 steps exact.<br><sub>Simplified dynamics, synthetic input spikes.</sub></td>
<td width="50%" valign="top"><a href="https://mottopanikeiku.github.io/doomformer/#one-map"><img src="https://github.com/mottopanikeiku/doomformer/releases/download/v0.1.0/poster_onemap_shil_raw.png" alt="A transformer in one map: film poster"></a><br><b>A transformer in one map.</b> No Python between layers. 3,548 dolls write SHIL.<br><sub>A 1-layer, d=8 model. Python samples the letter.</sub></td>
</tr>
<tr>
<td width="50%" valign="top"><a href="https://mottopanikeiku.github.io/doomformer/#writes-itself"><img src="https://github.com/mottopanikeiku/doomformer/releases/download/v0.4.0/poster_selfgen.png" alt="Generation without Python: film poster"></a><br><b>Generation without Python.</b> The winner's doll writes the next token and the level rewinds itself. It wrote DAN.<br><sub>One model and the default prompt.</sub></td>
<td width="50%" valign="top"><a href="https://mottopanikeiku.github.io/doomformer/#full-brain"><img src="https://github.com/mottopanikeiku/doomformer/releases/download/v0.1.0/poster_full_brain.png" alt="The 4-layer transformer: film poster"></a><br><b>A 4-layer transformer.</b> It writes UNHOLY STRONGHOLD over 10,166 tile runs.<br><sub>Tile runs share a pool of games.</sub></td>
</tr>
</table>

[docs/DEMOS.md](docs/DEMOS.md) has the numbers, the scope of each check and the [picked seeds](docs/DEMOS.md#seeds-and-selection).

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

`df.MLP.from_torch` converts a PyTorch MLP or CNN ([example 02](examples/02_pytorch_to_doom.py)). The `world` and
`actions` hooks add rooms and wire outputs to doors, lights and thrusts ([example 03](examples/03_brain_opens_doors.py)).
The same API takes spiking networks, transformers and learners. The guide is [docs/BUILD_YOUR_OWN.md](docs/BUILD_YOUR_OWN.md),
and there are twelve one-file [examples](examples). Built something odd? [Show and tell](https://github.com/mottopanikeiku/doomformer/issues/new/choose).

## Many games at once

One map runs in one game on one core. `df.run_many` hands inputs to a pool of games, on your machine or on a Ray cluster
you start. For speed, a network can also be split into tile maps, one per layer, with Python copying the values between
them.

| run | digits | right | exact | time |
|---|---|---|---|---|
| one map, 14 games | 1,000 | 966 | 1,000 | 2 h |
| 4 tile maps, 6 games | 10,000 | 9,716 | 10,000 | 23 min |

```sh
doomformer mnist --tiles --all --workers 6         # all 10,000 digits as tile maps, about 23 min
pip install "doomformer[ray]"
ray start --head                  # other machines: ray start --address=HEAD:6379
doomformer mnist --all --ray --record mnist.json   # all 10,000 digits, one map each, on a cluster
```

Budget one core and 1.3 GB of RAM per game. `doomformer plan` predicts the time of a model and picks a mode.
[Scaling](docs/scaling.md#scaling-out-many-games-many-machines)

## More

| what | command | doc |
|---|---|---|
| Faster schedules: MNIST in one map from 5,696 to 272 tics | `--opt 1`, `2`, `3` | [scheduling](docs/scheduling.md) |
| Static schedule checker and timing tuner | `doomformer lint`, `tune` | [scheduling](docs/scheduling.md) |
| A counter machine: 7 × 9 = 63 in 464 checked steps | `doomformer machine multiply 7 9` | [machine](docs/machine.md) |
| A SUBLEQ machine with self-modifying code, 18 cells | `doomformer machine subleq all` | [machine](docs/machine.md) |
| A blank perceptron learns digits in the map: 71 % held-out (textbook rule: 81 %) | `doomformer learn digits` | [learning](docs/learning.md) |
| PyTorch import with BatchNorm, pooling and QAT; this repo's GPT module | `MLP.from_torch`, `Transformer.from_torch` | [interop](docs/interop.md) |
| A transformer that learned induction: 0.9995 held-out, exact in DOOM | `python -m demos.induction` | [induction](models/induction/EVAL.md) |
| Activation patching, lesions and steering, checked in DOOM | `intref.forward(interventions=...)` | [interventions](docs/interventions.md) |
| Engine time per pass and throughput | `doomformer bench` | [scaling](docs/scaling.md) |

## Limits

- **Slow.** A small MLP runs about 23,000 times slower than NumPy. MNIST in one map runs 1.5 million times slower at the
  default schedule. Most of the cost is idle conveyor scrollers, and removing it needs an engine change.
  [Speed](docs/HOW_IT_WORKS.md#speed)
- **Only the runs are checked.** Inputs that were not run are not covered. Some map types are not fuzzed.
  [Not covered](docs/verification.md#not-covered)
- **Integer models.** Attention is integer attention. No model has LayerNorm. `Transformer.from_torch` takes only this
  repo's GPT module, trained with its recipe.
- **One engine.** The compiler is validated in ViZDoom 1.3.2. Other engines ran only small cases.
- **No reinforcement learning.** The marine's policies imitate a scripted aimbot. Python supplies its observations and
  labels.
- **Finite machines.** The machines have bounded memory. Nothing here claims DOOM is Turing complete.
- **Picked samples.** Showcased seeds, names and lesions are picked. [Seeds and selection](docs/DEMOS.md#seeds-and-selection)

## What's in the box

| Path | What it holds |
|---|---|
| [`doomformer/`](doomformer) | the toolkit: map compiler, DOOM runtime, exact checks, CLI |
| [`demos/`](demos) | research code: his brain is the level, the walker and level generator, the fly circuit, the learning arena |
| [`training/`](training) | quantisation-aware training for every model, with the corpora and their sources |
| [`models/`](models) | every trained model with its training log and evaluation |
| [`examples/`](examples) | twelve one-file examples, most under 15 s |
| [`tests/`](tests) | 479 tests with torch installed, 175 of them on real headless DOOM; CI runs them on Linux and macOS |
| [`docs/`](docs/README.md) | how it works, every demo with its numbers, the engine facts, the toolkit guide, [every command](docs/commands.md) |
| [`CHANGELOG.md`](CHANGELOG.md) | what each release added |

## Licensing

Code and docs: [MIT](LICENSE). Data keeps its own licence ([THIRD_PARTY.md](THIRD_PARTY.md)): the level-name lists are
Doom Wiki (CC BY-SA 4.0) and id Software's source (GPL-2.0), the 100 sample MNIST digits are CC BY-SA 3.0, and the
fruit-fly circuits in `models/fly` are derived from the FlyWire connectome (CC BY-NC 4.0, non-commercial). Freedoom
graphics are BSD-licensed. DOOM is a trademark of id Software / ZeniMax; this project is not affiliated with them.
