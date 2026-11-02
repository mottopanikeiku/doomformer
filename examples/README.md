# Examples

Each example is a standalone script that uses the installed `doomformer` package to compile a network into a DOOM map,
run headless ViZDoom and compare the exposed values with an integer reference, exiting non-zero on a mismatch, and
[`BUILD_YOUR_OWN.md`](../docs/BUILD_YOUR_OWN.md) documents the API.

## Run

```sh
pip install git+https://github.com/mottopanikeiku/doomformer      # needs git, cmake and a C compiler on first use
doomformer demo                                                    # a quick tour first: a transformer writes a name in DOOM
curl -O https://raw.githubusercontent.com/mottopanikeiku/doomformer/main/examples/01_xor.py
python 01_xor.py                                                   # any example runs from any directory
```

You can also clone the repository (`git clone https://github.com/mottopanikeiku/doomformer`, `pip install -e .`) and run
`python examples/01_xor.py`, which needs no display. PyTorch is optional because examples 02 and 11 also work with
NumPy, and images and GIFs go in `out/` beside the script. `EXACT` compares DOOM with the integer model, while example
02 separately measures the error introduced by quantising its float model.

| # | script | what it does | runtime (measured) |
|---|---|---|---|
| 01 | [`01_xor.py`](01_xor.py) | A hand-written integer MLP computes XOR (and 3-bit parity) in DOOM, all inputs, every one compared with the integer reference. | <1 s |
| 02 | [`02_pytorch_to_doom.py`](02_pytorch_to_doom.py) | Train a float MLP (PyTorch, or NumPy if it is not installed) on "is the point inside the circle?", quantise it, run test points in DOOM; prints float vs integer vs DOOM. | ~10 s (NumPy), ~20 s (PyTorch) |
| 03 | [`03_brain_opens_doors.py`](03_brain_opens_doors.py) | Two switches into a tiny net; its AND output opens a door and its OR output turns on the room light, inside the same map (`world` + `actions` hooks). | <1 s |
| 04 | [`04_spiking_disco.py`](04_spiking_disco.py) | An 8-neuron spiking ring oscillator flashes the lights of a hall as its spikes run around; one DOOM run per time step. Saves a GIF. | ~1 s |
| 05 | [`05_one_map_transformer.py`](05_one_map_transformer.py) | `df.Transformer.load("doomnames_onemap")` writes a name in one map, with exact checks per step. `--tiny` uses a smaller toy model. | ~90 s per letter, ~8 s with `--tiny` |
| 06 | [`06_shove_the_marine.py`](06_shove_the_marine.py) | An MLP reads a 2D "wind" and shoves the marine with `ThrustThing` lines; prints his position before and after. | <1 s |
| 07 | [`07_autonomous_turret.py`](07_autonomous_turret.py) | `df.Agent`: a hand-written MLP turret that senses, decides and shoots demons with no host decisions; ACS sensors and an in-map clock feed it, ACS argmax + actuators act ([`docs/agents.md`](../docs/agents.md)). | ~2 s |
| 08 | [`08_your_own_agent.py`](08_your_own_agent.py) | A copyable template: two sensors, your weights, your actuators. | ~2 s |
| 09 | [`09_record_a_gif.py`](09_record_a_gif.py) | `brain.record(...)`: a GIF of the dolls computing XOR, result stamped EXACT only after the check ([`docs/recording.md`](../docs/recording.md)). | ~5 s |
| 10 | [`10_big_model_tiles.py`](10_big_model_tiles.py) | `df.compile(..., mode="tiles")`: the 1-layer d=16 name model across DOOM worker games (Python copies the bus) ([`docs/scaling.md`](../docs/scaling.md)). | ~25 s |
| 11 | [`11_conv_in_doom.py`](11_conv_in_doom.py) | A trained Conv2d net reads 7x7 arrow glyphs; the conv is lowered exactly, Hardtanh is a 3-level staircase; torch or NumPy ([`docs/interop.md`](../docs/interop.md)). | ~35 s (12 glyphs) |
| 12 | [`12_doom_learns_live.py`](12_doom_learns_live.py) | `df.Perceptron`: DOOM learns a rule online (weights are ceiling heights), the teacher flips the rule and it relearns; lamps show right/wrong ([`docs/learning.md`](../docs/learning.md)). | ~9 s |
| nb | [`doomformer_colab.ipynb`](doomformer_colab.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mottopanikeiku/doomformer/blob/main/examples/doomformer_colab.ipynb) | Nothing to install: draw a digit in the browser and DOOM reads it, two MNIST test digits, the softmax-attention name demo, and a GIF of the dolls computing XOR. Outside Colab the drawing cell uses a shipped test digit; [`notebook.yml`](../.github/workflows/notebook.yml) runs it headlessly. | ~5.5 min on 2 cores, plus drawing |

### 01: XOR

This is the XOR network's floor plan, where the orange lines add weights.

![the XOR map from above](https://github.com/mottopanikeiku/doomformer/releases/download/v0.1.0/01_xor_map.png)

### 02: PyTorch to DOOM

The integer model's decision region is orange; the true circle is white. Test points run through DOOM are green
when classified correctly and red otherwise. DOOM matches both integer-model mistakes near the boundary.

![decision map](https://github.com/mottopanikeiku/doomformer/releases/download/v0.1.0/02_circle.png)

### 03: A network that opens a door

From left to right, (a, b) = (0, 0), (1, 0), (1, 1), and the light follows OR while the door follows AND.

![three screenshots](https://github.com/mottopanikeiku/doomformer/releases/download/v0.1.0/03_brain_opens_doors.png)

### 04: Spiking disco

Two spikes chase each other around a ring of 8 neurons; each spike lights its hall panel.

![GIF of the disco hall](https://github.com/mottopanikeiku/doomformer/releases/download/v0.1.0/04_spiking_disco.gif)

### 05: The transformer in one map

This is the whole transformer's map from above, where every orange tick is a weight line, from the full model run,
while `--tiny` writes `out/05_onemap_tiny.png`.

![transformer map](https://github.com/mottopanikeiku/doomformer/releases/download/v0.1.0/05_onemap.png)

### 06: Shove the marine

The top row shows the marine before and the bottom row after the winds (0,0), (2,0), (0,-1) and (-2,2), and in this
room one kick moved him about 64 units and two kicks about 127.

![shoved marine](https://github.com/mottopanikeiku/doomformer/releases/download/v0.1.0/06_shove_the_marine.png)

### 09: Record a GIF

![the XOR dolls marching, then EXACT](https://github.com/mottopanikeiku/doomformer/releases/download/v0.1.0/09_record_a_gif.gif)

### 12: DOOM learns live

![accuracy per epoch, before and after the rule flips](https://github.com/mottopanikeiku/doomformer/releases/download/v0.1.0/12_doom_learns_live.png)
