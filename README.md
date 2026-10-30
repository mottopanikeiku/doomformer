<p align="center">
  <img src="https://github.com/mottopanikeiku/doomformer/releases/download/v0.4.0/banner_wide.png" alt="DOOMFORMER: the marine fights a horde of demons under a hellish sky" width="100%">
</p>

<p align="center">
  <b>Neural networks computed by voodoo dolls inside DOOM maps.</b><br>
  Every value is read back and checked against NumPy. Tolerance 0.
</p>

<p align="center">
  <a href="https://github.com/mottopanikeiku/doomformer/actions/workflows/tests.yml"><img src="https://github.com/mottopanikeiku/doomformer/actions/workflows/tests.yml/badge.svg" alt="tests"></a>
  <a href="https://github.com/mottopanikeiku/doomformer/releases"><img src="https://img.shields.io/github/v/release/mottopanikeiku/doomformer?color=8b0000" alt="release"></a>
  <img src="https://img.shields.io/badge/python-3.11%2B-blue" alt="Python 3.11+">
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-MIT-blue" alt="MIT"></a>
  <a href="https://colab.research.google.com/github/mottopanikeiku/doomformer/blob/main/examples/doomformer_colab.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open in Colab"></a>
</p>

<p align="center">
  <a href="#quick-start">Quick start</a> ·
  <a href="#walk-a-digit-doom-reads-it">Walk a digit</a> ·
  <a href="#the-museum-rewire-one-connection">The museum</a> ·
  <a href="#how-it-works">How it works</a> ·
  <a href="#what-is-checked">What is checked</a> ·
  <a href="https://mottopanikeiku.github.io/doomformer/">Films</a> ·
  <a href="docs/README.md">Docs</a>
</p>

| **10,000 / 10,000** | **1,355** | **0** | **4** |
|:---:|:---:|:---:|:---:|
| MNIST test digits exact | voodoo dolls in one MNIST map | mismatches in 20,000 fuzzed nets | other DOOM engines run the museum exactly |

<p align="center">
  <img src="https://github.com/mottopanikeiku/doomformer/releases/download/v0.4.0/ladder.png" alt="A number is a crowd of voodoo dolls. A weight is a line on the floor. A sum is a light level or a height. An activation is a staircase. Attention is a ceiling. Every value is read back and compared with NumPy: tolerance 0." width="760">
</p>

## Quick start

```sh
pip install git+https://github.com/mottopanikeiku/doomformer   # Python 3.11+, Linux or macOS
doomformer hello             # 2 s: a small MLP, every value checked
doomformer draw --digit 7    # 15 s: the marine walks a 7 and the level reads it
```

`doomformer doctor` checks your setup. It builds ACC, the ACS compiler, if it is missing. That needs git, cmake and a
C compiler. On Fedora: `sudo dnf install git cmake gcc gcc-c++ make`.

> [!TIP]
> No install needed for the [Colab notebook](https://colab.research.google.com/github/mottopanikeiku/doomformer/blob/main/examples/doomformer_colab.ipynb).
> Draw a digit in your browser and DOOM reads it.

## Walk a digit, DOOM reads it

<p align="center">
  <img src="https://github.com/mottopanikeiku/doomformer/releases/download/v0.1.0/drawpad.gif" alt="DrawPad: the marine walks a 7 across a 28x28 floor canvas, presses READ, and the winning voodoo doll lights a 7 on a 7-segment display" width="720">
</p>

The marine holds FIRE and inks the floor. Then he presses READ.

1,355 voodoo dolls in the same map run an MNIST network. The winning digit's doll crosses a line. That line lights the
digit on the display.

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

Python reads back 248 values and checks every one. With `--digit`, Python presses the keys for you. Without it, you
walk and draw yourself. [Full output](docs/commands.md#doomformer-draw---digit-7)

> [!NOTE]
> EXACT means DOOM got the same numbers as the integer network. It does not mean the digit is right.

| run | right | exact |
|---|---|---|
| first 100 MNIST test digits | 99 | 100 |
| 1,000 sampled test digits, one map each | 966 | 1,000 |
| all 10,000 test digits, tile maps | 9,716 (97.16 %) | 10,000 |

The float model scores 97.58 %. The network also trained on the strokes the scripted marine walks.
[Details](docs/drawpad.md) · [Film](https://mottopanikeiku.github.io/doomformer/#drawpad)

![eight MNIST test digits and what DOOM read: 7 2 1 0 4 9 6 3, all exact](https://github.com/mottopanikeiku/doomformer/releases/download/v0.1.0/mnist.png)

## The museum: rewire one connection

<p align="center">
  <img src="https://github.com/mottopanikeiku/doomformer/releases/download/v0.7.1/museum_mutate.png" alt="Two views of one doll lane. Before MUTATE, DOOM reads 7 and the doll has passed its gate. After MUTATE, DOOM reads 1 and the same voodoo doll stands at its gate." width="800">
</p>

> [!TIP]
> **Play it without Python.** Download [museum.wad](https://github.com/mottopanikeiku/doomformer/releases/download/v0.7.1/museum.wad)
> ([sha256](https://github.com/mottopanikeiku/doomformer/releases/download/v0.7.1/museum_SHA256SUMS.txt)) and the free
> [Freedoom](https://freedoom.github.io/download.html) IWAD. Then run `gzdoom -iwad freedoom2.wad -file museum.wad`.

Step on seven floor pads to light a digit. Press READ. 136 voodoo dolls run a tiny network and light the answer.

Press MUTATE and the level rewires one connection while it runs. The same 7 now reads as 1. You can watch it happen:
one doll stops at its gate.

```text
$ doomformer museum verify
7                        pads abc     w_7a +1  DOOM reads 7  scores [4, 6, 3, 5, 4, 3, 2, 7, 3, 4]  EXACT (148 values; 744 tics; [...])
7 after MUTATE           pads abc     w_7a -1  DOOM reads 1  scores [4, 6, 3, 5, 4, 3, 2, 5, 3, 4]  EXACT (148 values; 744 tics; [...])
7 after RESTORE          pads abc     w_7a +1  DOOM reads 7  scores [4, 6, 3, 5, 4, 3, 2, 7, 3, 4]  EXACT (148 values; 744 tics; [...])
```

A self-test version runs exactly in GZDoom, LZDoom, UZDoom and ZDoom 2.8.1. [Details](docs/museum.md)

## How it works

Every part of a network becomes a part of the map:

| in a neural network | in the DOOM map |
|---|---|
| a value of 5 | 5 voodoo dolls |
| a weight | a line on the floor; each doll that crosses it adds the weight |
| a sum | a sector's light level or floor height |
| an activation (ReLU) | a staircase of gates the dolls must climb |
| attention | dolls squeezing under ceilings that another value lowered |
| a lesion | a closed door |

One layer, from dolls to dolls:

```mermaid
flowchart LR
    A["input value<br/>= a crowd of dolls"] --> B["weight lines<br/>each crossing adds w"]
    B --> C["the sum<br/>= a light level or height"]
    C --> D["staircase of gates<br/>= the activation"]
    D --> E["next layer's dolls"]
```

<table>
<tr>
<td width="50%" valign="top"><img src="https://github.com/mottopanikeiku/doomformer/releases/download/v0.1.0/crossbar.png" alt="voodoo dolls crossing lava weight lines"><br><b>A matrix multiply.</b> Each doll is one unit of a value. Each lava line it crosses adds a weight.</td>
<td width="50%" valign="top"><img src="https://github.com/mottopanikeiku/doomformer/releases/download/v0.1.0/ride.png" alt="voodoo dolls jammed under attention gates"><br><b>Attention.</b> Dolls squeeze under ceiling gates. Counting who gets through is a multiply.</td>
</tr>
</table>

The engine has sharp edges, and the compiler works around them:

- A floor mover ignores a second request while it is still moving.
- A teleport cancels lines a doll crossed earlier in the same tic.
- Things far from a BSP node can land in the wrong room, so every map fits in 32,000 units.

`doomformer facts` checks these in about 10 s. [Engine facts](docs/HOW_IT_WORKS.md#engine-facts) ·
[Engine features it needs](docs/HOW_IT_WORKS.md#engine-features-it-needs) ·
[Other ports](docs/engine_limits.md#other-ports)

Computers in DOOM maps are not new. [counter.wad](https://www.wad-archive.com/wad/8d0de31d8b392fc2d7fd817e933b3cf43bca799a)
counts with voodoo dolls, [conway.wad](https://github.com/pwiecz/conway.wad) runs Game of Life, and
[Danny Spencer's calculator](https://github.com/nukep/doom-calculator) adds numbers. New here: a compiler for neural
networks, and an exact check on every value. [Related work](docs/related_work.md)

## What is checked

<p align="center">
  <img src="https://github.com/mottopanikeiku/doomformer/releases/download/v0.4.0/pipeline.png" alt="Pipeline: your network becomes integers and a DOOM map, ViZDoom runs it and Python reads the values back (light levels and plane heights). A NumPy integer reference, which never sees the map, computes the same values. Every value is compared with tolerance 0: EXACT or MismatchError." width="800">
</p>

Every run compares each value the map exposes with NumPy. That proves the inputs that ran, nothing more. So the repo
also attacks its own check:

| attack | result |
|---|---|
| **Tamper** with one weight line | the check fails on exactly the inputs that cross it |
| **Fuzz** 20,000 random nets at the range limits | 0 mismatches; CI fuzzes a fresh seed on every push |
| **Mutate** one line of a map, 1,004 times | 297 caught; no survivor is proven harmless |
| **Extract** the weights back out of the map, without DOOM | equal to the source net |
| **Guard** every readout page with a poison value | a page that never runs cannot pass old values |

The MNIST map exposes 248 values. It does not expose its 903 staircase counts and 1,351 gate comparisons. Those are
checked only through the sums they feed. Fuzzing and tests found four bugs in the compiler. All are fixed.
[Verification](docs/verification.md)

<details>
<summary><b>Who computes what</b></summary>

| run | Python | voodoo dolls |
|---|---|---|
| `mnist`, `df.compile` | writes the inputs, waits, reads | every sum, activation, comparison and the choice |
| `demo`, `ask` (tile maps) | writes tokens, copies values between maps, samples letters | every sum and activation |
| `draw`, `--loop`, the marine's brain, the fly | only advances the clock and reads | everything, including the choice |
| `machine` | loads the program, then advances the clock and reads | program counter and registers |
| `museum`, played in GZDoom | nothing | every sum, comparison and the choice |

ACS level scripts move gates, rewind the dolls, sense the world and copy results out. Two do more. `df.Agent` takes an
argmax in ACS. `--opt 3` copies each layer's values into the next layer with a script.
[Per demo](docs/HOW_IT_WORKS.md#who-does-what-per-demo)

The maps need ZDoom (UDMF and ACS). This repo runs [ViZDoom](https://github.com/Farama-Foundation/ViZDoom) 1.3.2 with
Freedoom assets.

</details>

## Commands

| Command | What DOOM does | Time | Skill level |
|---|---|---|---|
| `doomformer hello` | a small MLP, 5 random inputs | 2 s | I'm too young to die |
| `doomformer draw --digit 7` | the marine walks a 7, the level reads it | 15 s | Hey, not too rough |
| `doomformer museum build` | a playable map that reads a digit | 1 s | Hey, not too rough |
| `doomformer mnist` | reads 4 test digits, or your own image | 1.5 to 3 min | Hurt me plenty |
| `doomformer demo` | a transformer writes a level name | 15 to 60 s | Hurt me plenty |
| `doomformer machine subleq all` | a tiny computer runs three programs | about 5 min | Ultra-Violence |
| `doomformer learn digits --classes 4 --steps 300` | a blank brain learns digits inside the map | about 5 min | Ultra-Violence |
| `doomformer ask "what does iddqd do"` | answers from 189 DOOM facts | about 11 min | Ultra-Violence |
| `doomformer demo --nightmare` | the 2-layer model, 106 tile maps | 3.5 to 7 min | Nightmare! |

Times are from a 16-core arm64 Linux machine. [Every command](docs/commands.md)

Some outputs are picked. DEMON CANYON, the name `--nightmare` writes, is one hand-picked sample. `ask` gets 94.7 % of
its training questions right, and 35.4 % of new wordings.

## Bigger experiments

<p align="center">
  <img src="https://github.com/mottopanikeiku/doomformer/releases/download/v0.1.0/hero.gif" alt="His brain is the level: the marine fights a demon while 1,345 dots on the right, voodoo dolls of him, run his policy transformer" width="720">
</p>

**His brain is the level.** The level senses the nearest demon. 1,345 voodoo dolls of the marine run his brain and pick
an action. The level turns him and fires. Python only steps the clock and reads. 50 of 50 decisions are exact.
[Film](https://mottopanikeiku.github.io/doomformer/#brain) · [Code](demos/brainlevel.py)

> [!WARNING]
> He is not playable. One decision takes 22 s of game time, and the demons are frozen while he thinks. With the demons
> awake (`--live`), he dies in 5.5 s without a kill. [Live demons](docs/DEMOS.md#live-demons-what-the-latency-costs-him)

| run it from a clone | what you see |
|---|---|
| `python -m demos.brainlevel` | his brain is the level |
| `python -m demos.brainlevel --live` | the same brain against awake demons |
| `python -m demos.brainlevel --lesion-when near` | the level closes neuron doors when a demon gets close |
| `python -m demos.aimlearn` | a perceptron learns to aim inside one game |
| `python -m demos.flyloop` | a fruit-fly circuit drives the marine |
| `python -m demos.lobotomy` | lesions that turn him into a pacifist |

<table>
<tr>
<td width="50%" valign="top"><a href="https://mottopanikeiku.github.io/doomformer/#trilogy"><img src="https://github.com/mottopanikeiku/doomformer/releases/download/v0.1.0/poster_trilogy.png" alt="A transformer names a level: film poster"></a><br><b>A transformer names a level.</b> It writes DEMON CANYON. A Python generator builds the level.<br><sub>One hand-picked sample.</sub></td>
<td width="50%" valign="top"><a href="https://mottopanikeiku.github.io/doomformer/#doom-plays-doom"><img src="https://github.com/mottopanikeiku/doomformer/releases/download/v0.1.0/poster_doom_beats_doom.png" alt="A marine plays the level it named: film poster"></a><br><b>A marine plays the level it named.</b> Doors, two kills, exit. 102 decisions, all exact.<br><sub>Python gives the compass. Seed 118 of 120, picked.</sub></td>
</tr>
<tr>
<td width="50%" valign="top"><a href="https://mottopanikeiku.github.io/doomformer/#doom-learns"><img src="https://github.com/mottopanikeiku/doomformer/releases/download/v0.1.0/poster_one_game.png" alt="A perceptron learns inside one game: film poster"></a><br><b>A perceptron learns inside one game.</b> Its 38 weights are ceiling heights. Dolls raise and lower them.<br><sub>Python gives the labels.</sub></td>
<td width="50%" valign="top"><a href="https://mottopanikeiku.github.io/doomformer/#lobotomy"><img src="https://github.com/mottopanikeiku/doomformer/releases/download/v0.1.0/poster_lobotomy.png" alt="Lesions: film poster"></a><br><b>Lesions.</b> A shotgun blast closes 7 neurons. The marine stops firing.<br><sub>Picked from 3,000 random blasts.</sub></td>
</tr>
<tr>
<td width="50%" valign="top"><a href="https://mottopanikeiku.github.io/doomformer/#fruit-fly"><img src="https://github.com/mottopanikeiku/doomformer/releases/download/v0.1.0/poster_fly_alive.png" alt="A fruit-fly circuit: film poster"></a><br><b>A fruit-fly circuit.</b> 255 neurons from the FlyWire connectome drive the marine.<br><sub>Simplified dynamics.</sub></td>
<td width="50%" valign="top"><a href="https://mottopanikeiku.github.io/doomformer/#one-map"><img src="https://github.com/mottopanikeiku/doomformer/releases/download/v0.1.0/poster_onemap_shil_raw.png" alt="A transformer in one map: film poster"></a><br><b>A transformer in one map.</b> No Python between layers. 3,548 dolls write SHIL.<br><sub>A small model.</sub></td>
</tr>
<tr>
<td width="50%" valign="top"><a href="https://mottopanikeiku.github.io/doomformer/#writes-itself"><img src="https://github.com/mottopanikeiku/doomformer/releases/download/v0.4.0/poster_selfgen.png" alt="Generation without Python: film poster"></a><br><b>Generation without Python.</b> The winner's doll writes the next letter. The level rewinds itself.<br><sub>One model, one prompt.</sub></td>
<td width="50%" valign="top"><a href="https://mottopanikeiku.github.io/doomformer/#full-brain"><img src="https://github.com/mottopanikeiku/doomformer/releases/download/v0.1.0/poster_full_brain.png" alt="The 4-layer transformer: film poster"></a><br><b>A 4-layer transformer.</b> It writes UNHOLY STRONGHOLD over 10,166 tile runs.<br><sub>Runs share a pool of games.</sub></td>
</tr>
</table>

[All demos, numbers and picked seeds](docs/DEMOS.md)

## Build your own

Compile a network into a map. Let its outputs open doors, flash lights or shove the marine. This runs as pasted:

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

`df.MLP.from_torch` converts a PyTorch model ([example 02](examples/02_pytorch_to_doom.py)). The `world` and `actions`
hooks wire outputs to doors, lights and thrusts ([example 03](examples/03_brain_opens_doors.py)).
[Guide](docs/BUILD_YOUR_OWN.md) · [12 examples](examples) · [Show and tell](https://github.com/mottopanikeiku/doomformer/issues/new/choose)

## Many games at once

One map runs in one game, on one core. `df.run_many` spreads inputs over many games, or over a Ray cluster. A network
can also be split into tile maps, one per layer. Then Python copies values between them.

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

Budget one core and 1.3 GB of RAM per game. [Scaling](docs/scaling.md#scaling-out-many-games-many-machines)

## More

| what | try | read |
|---|---|---|
| Faster schedules: MNIST in one map from 5,696 to 272 tics | `--opt 3` | [scheduling](docs/scheduling.md) |
| A counter machine: 7 × 9 = 63 in 464 checked steps | `doomformer machine multiply 7 9` | [machine](docs/machine.md) |
| A SUBLEQ machine with self-modifying code | `doomformer machine subleq all` | [machine](docs/machine.md) |
| A blank brain learns digits in the map: 71 % held-out | `doomformer learn digits` | [learning](docs/learning.md) |
| PyTorch import with BatchNorm, pooling and QAT | `MLP.from_torch` | [interop](docs/interop.md) |
| A transformer that learned induction, exact in DOOM | `python -m demos.induction` | [induction](models/induction/EVAL.md) |
| Patching, lesions and steering, checked in DOOM | `intref.forward(interventions=...)` | [interventions](docs/interventions.md) |
| Speed per pass and throughput | `doomformer bench` | [scaling](docs/scaling.md) |

## Limits

> [!IMPORTANT]
> - **Slow.** A small MLP runs about 23,000 times slower than NumPy. Most of the time goes to idle conveyor belts.
>   [Speed](docs/HOW_IT_WORKS.md#speed)
> - **Only the runs are checked.** Inputs that never ran are not covered. [Not covered](docs/verification.md#not-covered)
> - **Integer models.** Attention is integer attention. No model has LayerNorm.
> - **ViZDoom first.** Other DOOM engines ran only small maps.
> - **No reinforcement learning.** The marine copies a scripted aimbot.
> - **Finite machines.** Memory is bounded. Nothing here claims DOOM is Turing complete.
> - **Picked samples.** Showcased seeds and names are picked. [The list](docs/DEMOS.md#seeds-and-selection)

## What's in the box

| Path | What it holds |
|---|---|
| [`doomformer/`](doomformer) | the toolkit: map compiler, DOOM runtime, exact checks, CLI |
| [`demos/`](demos) | research code: the marine's brain, the level generator, the fly circuit, the learning arena |
| [`training/`](training) | training scripts for every model, with the data and sources |
| [`models/`](models) | every trained model with its training log and evaluation |
| [`examples/`](examples) | twelve one-file examples, most under 15 s |
| [`tests/`](tests) | 496 tests; CI runs them against real headless DOOM on Linux and macOS |
| [`docs/`](docs/README.md) | how it works, every demo with its numbers, the engine facts, the guide, [every command](docs/commands.md) |
| [`CHANGELOG.md`](CHANGELOG.md) | what each release added |

## Licensing

Code and docs: [MIT](LICENSE). Data keeps its own licence ([THIRD_PARTY.md](THIRD_PARTY.md)). The level-name lists come
from the Doom Wiki (CC BY-SA 4.0) and id Software's source (GPL-2.0). The 100 sample MNIST digits are CC BY-SA 3.0. The
fruit-fly circuits in `models/fly` come from the FlyWire connectome (CC BY-NC 4.0, non-commercial). Freedoom graphics
are BSD-licensed. DOOM is a trademark of id Software / ZeniMax. This project is not affiliated with them.
