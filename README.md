<p align="center">
  <img src="https://github.com/mottopanikeiku/doomformer/releases/download/v0.4.0/banner_wide.png" alt="DOOMFORMER: the marine fights a horde of demons under a hellish sky" width="100%">
</p>

# Doomformer

I built a compiler that turns neural networks into DOOM maps. Voodoo dolls do the arithmetic: weights are lines on the floor, sums are light levels, and activations are staircases.

<p align="center">
  <img src="https://github.com/mottopanikeiku/doomformer/releases/download/v0.1.0/drawpad.gif" alt="The marine draws a 7 on the floor. Voodoo dolls run the network and light a 7 on the display." width="720">
</p>

DOOM reads handwritten digits with 97% accuracy on MNIST. Across all 10,000 test digits, every value read back from the maps matched the NumPy integer network exactly.

[Watch the films](https://mottopanikeiku.github.io/doomformer/) · [Play the museum](#play-the-museum) · [How it works](#how-it-works) · [Docs](docs/README.md)

## Quick start

Python 3.11+ on Linux or macOS. In a virtual environment:

```sh
pip install git+https://github.com/mottopanikeiku/doomformer
doomformer doctor
doomformer hello
```

`hello` compiles a small MLP into a map, runs five inputs through it and prints `5/5 EXACT`. No game window or DOOM purchase needed: ViZDoom includes the free Freedoom assets.

The first run builds ACC, the level-script compiler. Install git, cmake, make and a C/C++ compiler first. On Fedora: `sudo dnf install git cmake gcc gcc-c++ make`. From a clone, use `pip install .` instead of the first command.

Or open the [Colab notebook](https://colab.research.google.com/github/mottopanikeiku/doomformer/blob/main/examples/doomformer_colab.ipynb) and draw a digit in your browser.

## Play the museum

Download [museum.wad](https://github.com/mottopanikeiku/doomformer/releases/download/v0.7.1/museum.wad) and [Freedoom](https://freedoom.github.io/download.html), then open the map in GZDoom:

```sh
gzdoom -iwad freedoom2.wad -file museum.wad
```

Step on seven floor pads to draw a digit, then press READ. A hand-written network of 136 voodoo dolls lights the answer. Press MUTATE and one connection changes from +1 to −1: the same 7 now reads as 1. Press RESTORE to put it back.

<p align="center">
  <img src="https://github.com/mottopanikeiku/doomformer/releases/download/v0.7.1/museum_mutate.png" alt="Before and after rewiring one connection: a doll that passed its gate now stops, changing the answer from 7 to 1." width="800">
</p>

The map runs without Python. Its self-test also passes in GZDoom, LZDoom, UZDoom and ZDoom 2.8.1. [Museum guide](docs/museum.md)

## Draw a digit

```sh
doomformer draw --digit 7
```

The marine holds FIRE to ink a 28×28 floor canvas, then presses READ. In the same map, 1,355 voodoo dolls run the MNIST network. The winning digit's doll crosses the line that lights the display. With `--digit`, Python presses the keys; run `doomformer draw` to walk and draw yourself.

| MNIST test run | Correct digits | Exact network computations |
|---|---:|---:|
| First 100 digits, one map | 99 / 100 | 100 / 100 |
| 1,000 sampled digits, one map | 966 / 1,000 | 1,000 / 1,000 |
| All 10,000 digits, four tile maps | **9,716 / 10,000** | **10,000 / 10,000** |

Each pass checks 248 values against the integer reference with zero tolerance. The full test-set accuracy is 97.16%; the float model scores 97.58%. In tile mode, Python copies values between maps. DrawPad keeps the whole network in one map.

[Results and logs](models/mnist/EVAL.md) · [DrawPad guide](docs/drawpad.md) · [Running many games](docs/scaling.md)

## How it works

A voodoo doll is an extra player body in the map. Conveyor belts push the dolls across lines that change the level. The compiler arranges those lines and sectors into a network:

| Network part | Map part |
|---|---|
| A value of 5 | 5 voodoo dolls |
| A weight | A line on the floor; each doll that crosses it adds the weight |
| A sum | A sector's light level or floor height |
| An activation (ReLU) | A staircase of gates the dolls must climb |
| Attention | Dolls squeezing under ceilings that another value lowered |
| A lesion | A closed door |

<table>
<tr>
<td width="50%" valign="top"><img src="https://github.com/mottopanikeiku/doomformer/releases/download/v0.1.0/crossbar.png" alt="Voodoo dolls crossing weight lines"><br>A matrix multiply: each crossing adds a weight.</td>
<td width="50%" valign="top"><img src="https://github.com/mottopanikeiku/doomformer/releases/download/v0.1.0/ride.png" alt="Voodoo dolls under attention ceilings"><br>Attention: counting the dolls that fit under a ceiling multiplies two values.</td>
</tr>
</table>

ACS level scripts write inputs, move gates, rewind dolls and copy results out. Python reads the resulting light levels and heights and compares them with NumPy. A mismatch raises an error. The tests also change weights and timing deliberately to check that errors are caught.

The maps use ZDoom's UDMF and ACS features. The Python package runs them in ViZDoom 1.3.2. [Compiler details](docs/HOW_IT_WORKS.md) · [Verification](docs/verification.md)

Computers in DOOM have a long history: [counter.wad](https://www.wad-archive.com/wad/8d0de31d8b392fc2d7fd817e933b3cf43bca799a) counts with voodoo dolls, [conway.wad](https://github.com/pwiecz/conway.wad) runs Game of Life, and [Danny Spencer's calculator](https://github.com/nukep/doom-calculator) adds numbers. Doomformer compiles neural networks into this machinery. [Related work](docs/related_work.md)

## His brain is the level

<p align="center">
  <img src="https://github.com/mottopanikeiku/doomformer/releases/download/v0.1.0/hero.gif" alt="The marine fights a demon while 1,345 voodoo dolls run his policy transformer in the same level" width="720">
</p>

The level senses the nearest demon. Then 1,345 voodoo dolls of the marine run his policy transformer, choose an action, turn him and fire. Python advances the clock and reads the results. All 50 decisions in the film match the integer reference.

From a clone:

```sh
python -m demos.brainlevel
```

The demons pause while he thinks: one decision takes 22 seconds of game time. With `--live`, they keep moving and kill him before he can act. The policy learned to copy a scripted aimbot.

[Watch the film](https://mottopanikeiku.github.io/doomformer/#brain) · [Demo code](demos/brainlevel.py)

The other films include a transformer naming a level, a marine playing that level, a perceptron learning inside one game, and a fruit-fly circuit driving the marine. I selected the generated names, the walker's seed 118 from 120 seeds, and the lesion demos from neuron scans, including the pacifist from 3,000 random blasts. [Films, results and selection details](docs/DEMOS.md)

## Build your own

This network computes AND and OR. The dolls run both layers; Python checks the values.

```python
import numpy as np
import doomformer as df

net = df.MLP(
    weights=[np.array([[1, 1], [1, 1]]), np.array([[1, 0], [0, 1]])],
    biases=[np.array([-1, 0]), np.array([0, 0])],
    thresholds=[np.array([1]), np.array([1])],
    signed_input=False,
)
with df.compile(net) as brain:
    for x in ([0, 0], [1, 0], [1, 1]):
        print(x, "->", brain(np.array(x)).tolist())
```

`df.MLP.from_torch` imports a PyTorch network. The `world` and `actions` hooks connect outputs to doors, lights and thrusts. `brain.record(...)` films the dolls computing.

[Build your own guide](docs/BUILD_YOUR_OWN.md) · [Examples](examples) · [Command reference](docs/commands.md)

## Working with it

- These are small integer models. The compiler supports MLPs, spiking networks and transformers, including integer softmax and RMSNorm; it has no LayerNorm.
- Conveyor belts make inference slow. A map is useful for seeing and changing a computation, rather than speeding it up.
- Exact checks cover the values exposed by each map on the inputs run. Internal staircase counts are checked through the sums they feed. [Details](docs/verification.md)

## Licensing

Code and docs are [MIT](LICENSE). Data and assets keep their own licences, including CC BY-NC 4.0 for the FlyWire circuits. See [THIRD_PARTY.md](THIRD_PARTY.md). DOOM is a trademark of id Software / ZeniMax; this project is unaffiliated.
