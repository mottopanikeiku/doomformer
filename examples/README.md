# Examples

Small scripts, each one file that needs only the installed `doomformer` package (no sibling imports; the trained
models load by name). Each builds a network, compiles it into a DOOM map, runs it in headless ViZDoom,
prints what DOOM computed next to the integer reference, and exits non-zero on any mismatch. Read
[`docs/BUILD_YOUR_OWN.md`](../docs/BUILD_YOUR_OWN.md) for the concepts and the API.

## Run

```sh
pip install git+https://github.com/mottopanikeiku/doomformer      # needs git, cmake and a C compiler on first use
doomformer demo                                                    # a quick tour first: a transformer writes a name in DOOM
curl -O https://raw.githubusercontent.com/mottopanikeiku/doomformer/main/examples/01_xor.py
python 01_xor.py                                                   # any example runs from any directory
```

Or clone the repository (`git clone https://github.com/mottopanikeiku/doomformer`, `pip install -e .`) and run
`python examples/01_xor.py`; the research code in `demos/` needs the checkout. No display is needed.
PyTorch is optional (examples 02 and 11 fall back to NumPy). Images and GIFs are written to an `out/` directory next to
the example file. "EXACT" always means DOOM's integers equal the integer model's integers; it says nothing about how well an
integer model matches a float model (example 02 measures that separately).

| # | script | what it does | runtime (measured) |
|---|---|---|---|
| 01 | [`01_xor.py`](01_xor.py) | A hand-written integer MLP computes XOR (and 3-bit parity) in DOOM, all inputs, every one compared with the integer reference. | <1 s |
| 02 | [`02_pytorch_to_doom.py`](02_pytorch_to_doom.py) | Train a float MLP (PyTorch, or NumPy if it is not installed) on "is the point inside the circle?", quantise it, run test points in DOOM; prints float vs integer vs DOOM. | ~10 s (NumPy), ~20 s (PyTorch) |
| 03 | [`03_brain_opens_doors.py`](03_brain_opens_doors.py) | Two switches into a tiny net; its AND output opens a door and its OR output turns on the room light, inside the same map (`world` + `actions` hooks). | <1 s |
| 04 | [`04_spiking_disco.py`](04_spiking_disco.py) | An 8-neuron spiking ring oscillator flashes the lights of a hall as its spikes run around; one DOOM run per time step. Saves a GIF. | ~1 s |
| 05 | [`05_one_map_transformer.py`](05_one_map_transformer.py) | A trained transformer (`df.Transformer.load("doomnames_onemap")`) writes a name inside one map, every step verified bit-exactly. **Slow: ~90 s per letter**; `--tiny` is a fast toy model. | ~90 s per letter, ~8 s with `--tiny` |
| 06 | [`06_shove_the_marine.py`](06_shove_the_marine.py) | An MLP reads a 2D "wind" and shoves the marine with `ThrustThing` lines; prints his position before and after. | <1 s |
| 07 | [`07_autonomous_turret.py`](07_autonomous_turret.py) | `df.Agent`: a hand-written MLP turret that senses, decides and shoots demons with no host decisions; ACS sensors and an in-map clock feed it, ACS argmax + actuators act ([`docs/agents.md`](../docs/agents.md)). | ~2 s |
| 08 | [`08_your_own_agent.py`](08_your_own_agent.py) | A copyable template: two sensors, your weights, your actuators. | ~2 s |
| 09 | [`09_record_a_gif.py`](09_record_a_gif.py) | `brain.record(...)`: a GIF of the dolls computing XOR, result stamped EXACT only after the check ([`docs/recording.md`](../docs/recording.md)). | ~5 s |
| 10 | [`10_big_model_tiles.py`](10_big_model_tiles.py) | `df.compile(..., mode="tiles")`: the 1-layer d=16 name model across DOOM worker games (Python copies the bus) ([`docs/scaling.md`](../docs/scaling.md)). | ~25 s |
| 11 | [`11_conv_in_doom.py`](11_conv_in_doom.py) | A trained Conv2d net reads 7x7 arrow glyphs; the conv is lowered exactly, Hardtanh is a 3-level staircase; torch or NumPy ([`docs/interop.md`](../docs/interop.md)). | ~35 s (12 glyphs) |
| 12 | [`12_doom_learns_live.py`](12_doom_learns_live.py) | `df.Perceptron`: DOOM learns a rule online (weights are ceiling heights), the teacher flips the rule and it relearns; lamps show right/wrong ([`docs/learning.md`](../docs/learning.md)). | ~9 s |

### 01: XOR

One line of what DOOM sees: the floor plan of the XOR network (orange = lines that add a weight).

![the XOR map from above](https://github.com/mottopanikeiku/doomformer/releases/download/v0.1.0/01_xor_map.png)

### 02: PyTorch to DOOM

Integer model's decision map (orange), true circle (white), and the test points that went through DOOM (green = classified
right, red = wrong). The two red points are the integer model's mistakes near the edge, and DOOM reproduces them exactly.

![decision map](https://github.com/mottopanikeiku/doomformer/releases/download/v0.1.0/02_circle.png)

### 03: A network that opens a door

Left to right: (a, b) = (0, 0), (1, 0), (1, 1). The light follows OR, the door follows AND.

![three screenshots](https://github.com/mottopanikeiku/doomformer/releases/download/v0.1.0/03_brain_opens_doors.png)

### 04: Spiking disco

Two spikes chase each other around a ring of 8 neurons; each spike lights its hall panel.

![GIF of the disco hall](https://github.com/mottopanikeiku/doomformer/releases/download/v0.1.0/04_spiking_disco.gif)

### 05: The transformer in one map

The whole transformer's map from above (every orange tick is a weight line). Full model run shown; `--tiny` writes
`out/05_onemap_tiny.png`.

![transformer map](https://github.com/mottopanikeiku/doomformer/releases/download/v0.1.0/05_onemap.png)

### 06: Shove the marine

Top row: before; bottom row: after. Winds (0,0), (2,0), (0,-1), (-2,2). One kick moved him ~64 units (two kicks ~127).
That distance is measured for this room and force. DOOM's integer arithmetic does not guarantee it.

![shoved marine](https://github.com/mottopanikeiku/doomformer/releases/download/v0.1.0/06_shove_the_marine.png)

### 09: Record a GIF

![the XOR dolls marching, then EXACT](https://github.com/mottopanikeiku/doomformer/releases/download/v0.1.0/09_record_a_gif.gif)

### 12: DOOM learns live

![accuracy per epoch, before and after the rule flips](https://github.com/mottopanikeiku/doomformer/releases/download/v0.1.0/12_doom_learns_live.png)


## Show us yours

Built something strange? Open a **Show and tell** issue at <https://github.com/mottopanikeiku/doomformer/issues/new/choose>
with a link, a screenshot or GIF and the bit-exact output (see [`CONTRIBUTING.md`](../CONTRIBUTING.md)).
