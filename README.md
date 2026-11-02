<p align="center">
  <img src="https://github.com/mottopanikeiku/doomformer/releases/download/v0.4.0/banner_wide.png" alt="DOOMFORMER: the marine fights a horde of demons under a hellish sky" width="100%">
</p>

# Doomformer

Doomformer compiles integer neural networks into DOOM maps, where voodoo dolls do the arithmetic. A doll that crosses a line adds that line's weight to a light level or a floor height.

<p align="center">
  <img src="https://github.com/mottopanikeiku/doomformer/releases/download/v0.1.0/drawpad.gif" alt="The marine draws a 7 on the floor. Voodoo dolls run the network and light a 7 on the display." width="720">
</p>

The marine draws a 7 on the floor and presses READ, and 1,355 voodoo dolls in the same map run an MNIST network and light the answer. I also ran the whole 10,000-digit test set through DOOM, split into four tile maps with Python passing values between them, and it got 97.2% right while every value the game computed matched the NumPy integer reference.

The films are on the [project page](https://mottopanikeiku.github.io/doomformer/), the guides start at [docs/](docs/README.md), and there is a [Colab notebook](https://colab.research.google.com/github/mottopanikeiku/doomformer/blob/main/examples/doomformer_colab.ipynb) if you'd rather not install anything.

## Install

Python 3.11+ on Linux or macOS, plus git, cmake, make and a C/C++ compiler (the first run builds ACC, the level-script compiler).

```sh
pip install git+https://github.com/mottopanikeiku/doomformer
doomformer doctor
doomformer hello
```

`hello` compiles a small MLP into a map, runs five inputs through it and prints `5/5 EXACT`, and it needs neither a game window nor a copy of DOOM because ViZDoom ships the free Freedoom assets.

## Draw a digit

```sh
doomformer draw --digit 7
```

The marine holds FIRE to ink a 28×28 floor canvas and then presses READ. The whole network lives in that one map, and the winning digit's doll crosses the line that lights the display. Without `--digit`, `doomformer draw` lets you walk and draw yourself, which needs a display.

| MNIST test run | Correct | Exact |
|---|---:|---:|
| First 100 digits, one map | 99 / 100 | 100 / 100 |
| 1,000 sampled digits, one map | 966 / 1,000 | 1,000 / 1,000 |
| All 10,000 digits, four tile maps | **9,716 / 10,000** | **10,000 / 10,000** |

A digit counts as exact when all 248 values DOOM exposes for it equal the integer reference. The float version of the network scores 97.6%, so rounding it to integers costs under half a point, and the logs are in [models/mnist/EVAL.md](models/mnist/EVAL.md).

`doomformer draw --build drawpad` writes the same map to `drawpad/drawpad.wad`, or you can download [drawpad.wad](https://github.com/mottopanikeiku/doomformer/releases/download/v0.7.2/drawpad.wad) (85 MB). It plays in GZDoom with Freedoom and no Python (`gzdoom -iwad freedoom2.wad -file drawpad.wad`), so you can ink the canvas, press READ yourself and watch the display light up about eight seconds later.

## Play the museum

Grab [museum.wad](https://github.com/mottopanikeiku/doomformer/releases/download/v0.7.1/museum.wad) and [Freedoom](https://freedoom.github.io/download.html), then:

```sh
gzdoom -iwad freedoom2.wad -file museum.wad
```

Step on seven floor pads to draw a digit and press READ, and a hand-written network of 136 voodoo dolls lights the answer. When you press MUTATE, one connection flips from +1 to −1 and the same 7 reads as 1 until RESTORE puts it back. The map runs without Python, and its self-test passes in GZDoom, LZDoom, UZDoom and ZDoom 2.8.1.

<p align="center">
  <img src="https://github.com/mottopanikeiku/doomformer/releases/download/v0.7.1/museum_mutate.png" alt="Before and after rewiring one connection: a doll that passed its gate now stops, changing the answer from 7 to 1." width="800">
</p>

## How it works

A voodoo doll is an extra player body in the map that conveyor belts push across lines that change the level, and the compiler arranges those lines and sectors into a network.

| Network | Map |
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

ACS level scripts write inputs, move gates, rewind dolls and copy results out, while Python reads the light levels and heights and raises an error when one differs from NumPy. The tests also break weights and timing on purpose to make sure that check notices.

On a shared 16-core arm64 Linux machine, one pass of a 12-16-6 toy network takes 726 ms in DOOM, from writing the inputs to reading the results back, against 11 µs in NumPy, but in exchange you can walk around inside it and shut doors on its neurons. The maps use ZDoom's UDMF format and ACS scripts and run in ViZDoom 1.3.2.

Doom computers go back a while: [counter.wad](https://www.wad-archive.com/wad/8d0de31d8b392fc2d7fd817e933b3cf43bca799a) counts with voodoo dolls, [conway.wad](https://github.com/pwiecz/conway.wad) runs Game of Life, and [Danny Spencer's calculator](https://github.com/nukep/doom-calculator) adds. Doomformer compiles neural networks onto the same machinery, and the pages on [how it works](docs/HOW_IT_WORKS.md), [verification](docs/verification.md) and [related work](docs/related_work.md) go into the details.

## His brain is the level

<p align="center">
  <img src="https://github.com/mottopanikeiku/doomformer/releases/download/v0.1.0/hero.gif" alt="The marine fights a demon while 1,345 voodoo dolls run his policy transformer in the same level" width="720">
</p>

The level senses the nearest demon, and then 1,345 voodoo dolls run the marine's policy transformer, which picks an action, turns him and fires. Python only advances the clock and reads results, and all 50 decisions in the film match the integer reference of a policy that learned to copy a scripted aimbot.

```sh
python -m demos.brainlevel   # from a clone
```

The demons pause while he thinks, because one decision takes 22 seconds of game time at `--opt 3`, and when they keep moving (the `--live` flag) they kill him before he can act.

The other films show a transformer naming a level, a marine playing that level, a perceptron learning inside one game and a fruit-fly circuit driving the marine. They are picked runs, so [docs/DEMOS.md](docs/DEMOS.md) names each film's seed and has the tables over all evaluated seeds.

## Build your own

The dolls run both layers of this small network, which computes AND and OR, and Python checks every value.

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

`df.MLP.from_torch` imports a PyTorch network, and the compiler handles integer MLPs, spiking nets and transformers (softmax and RMSNorm included, no LayerNorm). The `world` and `actions` hooks wire outputs to doors, lights and thrusts, `brain.record(...)` films the dolls computing, and the [build guide](docs/BUILD_YOUR_OWN.md) and [examples](examples) are the place to start.

To hack on it, run `pip install -e ".[test,train]"` and then `pytest -q tests`. Most tests boot real headless DOOM games, so new compiler behavior needs a test that runs in DOOM too.

## License

Code and docs are [MIT](LICENSE), while data and assets keep their own licenses, including CC BY-NC 4.0 for the FlyWire circuits ([THIRD_PARTY.md](THIRD_PARTY.md)). DOOM is a trademark of id Software / ZeniMax, and this project is not affiliated with them.
