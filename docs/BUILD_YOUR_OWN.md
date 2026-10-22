# Build your own DOOM-computed network

This is the manual for `import doomformer as df`: take an integer neural network (an MLP, a spiking net, a
transformer), compile it into a DOOM map, run the map in ViZDoom and check every number DOOM produces against an
independent integer reference. You can also add rooms, doors, lights and scripts to the same map and let the
network's arithmetic drive them.

> **What is real here.** DOOM does the integer arithmetic of the network: sums of weights, staircase activations,
> products inside the transformer's attention, spike thresholds. Python (a) builds the map, (b) writes the *inputs*
> into the map before a run, (c) waits, (d) reads the results back and (e) compares them with a NumPy integer
> reference. For the transformer, Python also turns logits into the next token (argmax or seeded sampling).
> "Bit-exact" always means: DOOM's integers equal the integer model's integers. Quantising a float model to integers
> is an approximation (see [Quantising a float model](#quantising-a-float-model)). A pass takes seconds (see
> [Limits](#limits-and-measured-sizes)).

Contents: [Concepts](#concepts) · [Install and first run](#install-and-first-run) · [The API](#the-api) ·
[Hooks: world and actions](#hooks-world-and-actions) · [Limits and measured sizes](#limits-and-measured-sizes) ·
[What Python does vs what DOOM does](#what-python-does-vs-what-doom-does) · [Recipes](#recipes) ·
[Troubleshooting](#troubleshooting) · [Share your project](#share-your-project)

## Concepts

**Voodoo dolls are units.** In DOOM a map can contain extra "player 1" starts. The game spawns the *last* one as the
real player and the earlier ones as inert copies, voodoo dolls. A doll obeys the same physics as the player: it
walks, it is blocked by walls and steps taller than 24 units, and when it crosses a line with a special, that special
fires. The compiler places one doll per lane of the network (3,548 for the one-map transformer, a few for XOR) and
makes each walk down its own corridor.

**Staircase thresholds are activations.** A value that feeds an activation is written into the height of a *gate*
sector's floor (a "port"). The value's doll can only pass if the gate is low enough: a lane with threshold θ passes
iff `value >= θ`. A staircase of K lanes with thresholds `θ_1 <= ... <= θ_K` therefore counts how many thresholds
the value clears: `adc_relu(z, thr) = #{k : z >= thr[k]}`, and `adc_signed(z, thr) = adc_relu(z, thr) - adc_relu(-z, thr)`
(see `doomformer/intref.py`). Different threshold sets give you ReLU, clamping, rounding, scaling or a plain
step. That count is the activation, expressed as the number of dolls that get through.

**Crossbars are lines.** A doll that gets through then walks across one line per output unit. Each line's special
adds the weight to that output: a `Light_RaiseByValue` / `Light_LowerByValue` on an accumulator sector (whose light
level is a saturating int16), or, inside a pure map, a floor/ceiling move on the *next layer's gate*. Summing
`w * count` over all dolls gives `b + W @ a`. Outputs also keep a negated twin so that later layers can test `>=`
and `<=` with the one gate kind.

**Ports and readout.** *Ports* are the gate sectors Python writes the input values into. *Readout* is how results come
back: output sectors' light levels (and, for the hidden values you want to check, the heights of their gate planes) are copied by small
ACS scripts into ViZDoom's `USER1..USER60` variables, which Python reads. In a pure (one-map) network no Python happens between
layers: the layers are chained by dolls moving the next layer's gates, and a clock doll releases each layer's waves
in order.

**One map vs tiles.** `df.compile` builds *one map* (`doomformer/pure.py`) by default: the whole network, layer
after layer, in a single game. Only the inputs and the outputs cross the Python boundary, so the whole network has
to fit the map (see [Limits](#limits-and-measured-sizes)). Tile mode (`doomformer/tile.py`, [scaling.md](scaling.md))
splits big models into one small map per operation and copies the integers between games in Python.

## Install and first run

```sh
pip install git+https://github.com/mottopanikeiku/doomformer      # needs git, cmake and a C compiler on first use
doomformer doctor                                                  # python / vizdoom / ACC / Freedoom check
doomformer hello                                                   # a tiny MLP, 5 random inputs, run in DOOM, EXACT per input
doomformer demo                                                    # a trained transformer writes a level name in DOOM (~20 s with 8 DOOM games)
```

`doomformer hello` prints per-input `EXACT` and the number of dolls and sectors, and exits nonzero on any mismatch.
The first compile also builds the ACS compiler ([ACC](https://github.com/rheit/acc), pinned commit) into
`~/.cache/doomformer/`. See [Troubleshooting](#troubleshooting) if that fails.

`doomformer demo` loads the shipped `doomnames` transformer in `tiles` mode and has a pool of DOOM games (one per core, at
most 8) write a whole name; every exposed tile tensor is compared exactly with the integer reference, and the summary says
whether the name is in the 136-name training corpus. `demo --one-map` runs the smaller `doomnames_onemap` inside ONE map
(about 85 s per letter), `demo --one-map --softmax` its softmax-attention sibling `doomnames_onemap_softmax` (about 120 s
per letter), `demo --fast` a random-weight toy in one map (about 6 s), `--png PATH` saves a top-down picture.

The trained transformers (`doomnames_onemap`, `doomnames`, ...) ship inside the package and load by name:
`df.Transformer.load("doomnames_onemap")` works from any directory (`df.available_models()` lists the names; a path to
a model directory works too). Then run the examples in [`examples/`](../examples/README.md): each is one file that needs
only the installed package.

## The API

### `df.MLP`: an integer MLP

```python
import numpy as np, doomformer as df

net = df.MLP(
    weights=[np.array([[1, 1], [1, 1]]), np.array([[1, -1]])],   # (out, in) int arrays, any depth >= 1
    biases=[np.array([0, -1]), np.array([0])],
    thresholds=[np.array([1]), np.array([1])],                   # positive nondecreasing ints, one staircase per layer
    signed_input=False,
)
```

Exact semantics (this is the contract DOOM must reproduce):

```
a_0     = adc_signed(x, thr_0)   if signed_input else adc_relu(x, thr_0)
h_{l+1} = b_l + W_l @ a_l
a_l     = adc_relu(h_l, thr_l)   for l >= 1
output  = h_last                 (raw sums, no activation)
```

* `net.reference(x)` returns `dict(y=..., hidden=[...])`, computed with NumPy integers. It is the independent reference.
* `net.check(x)` raises `df.RangeError` if a gate-fed value leaves ±12,000 or a partial sum leaves ±30,000 (`df.MLP(...)` itself rejects
  weights above ±255 and thresholds outside 1..12,000).
* `net.save(path)` / `df.MLP.load(path)` use an `.npz`.

### Quantising a float model

```python
net = df.MLP.from_torch(torch_sequential, calib_x, levels=8)        # Linear, Conv1d/2d, ReLU, Hardtanh, Sigmoid, Tanh, ...
net = df.MLP.from_numpy([(W1, b1), (W2, b2)], calib_x, levels=8)    # float weights, ReLU between layers (or per-layer activations)
x_int = net.encode(x_float);  y_float = net.decode(net.reference(x_int)["y"])
print(net.quantization_report(x_float))                             # argmax agreement / max abs error vs the float model
```

The integer MLP is an **approximation** of the float model (rounded weights, `levels`-step staircase
activations). DOOM then reproduces the *integer* model exactly. `quantization_report` measures the distance between
the integer and the float model and says nothing about DOOM, which is checked separately. See
`examples/02_pytorch_to_doom.py`.

**Let DOOM pick the answer.** `df.with_argmax(net)` appends three layers: dolls compare every pair of output scores,
count each output's wins, and only the first maximal score's doll gets through (np.argmax's tie rule). The new net's
outputs are the one-hot of the argmax, so `actions` on `y{i}` fire for the winner only: open the winner's door, light
its lamp. If the score layer is too large for the comparator's limits it is scaled down first (`net.score_scale`);
check the accuracy afterwards with the reference. `doomformer mnist` and `doomformer draw` use it
([drawpad.md](drawpad.md)).

### `df.compile`: build, load, run

```python
with df.compile(net) as brain:         # wad=None: a content-addressed file in ~/.cache/doomformer/brains
    y = brain(np.array([1, 0]))        # one pass in DOOM; compares y AND every hidden value DOOM exposes with net.reference
print(brain.info)                      # kind, wad, dolls, sectors, runs, tics (of the last run), checked (values compared)
```

`brain(x)` takes **integers** (use `net.encode(x_float)` for nets from `from_numpy` / `from_torch`) and also accepts a batch
`(B, n_in)`. It raises `df.MismatchError` (listing `(name, DOOM, reference)` triples in `.mismatches`) on any difference,
and `df.RangeError` *before* running if a value would leave the safe ranges. `brain.run(x, check=False)` returns the raw
readback dictionary (`y{o}`, `h{l}_{j}`, `-h{l}_{j}`) without comparing. `brain.close()` (or the `with` block) shuts the game
down. The game is started lazily on the first pass.

`brain.runner` is the live `doomformer.runtime.TileRunner` (`brain.runner.game` is the `vizdoom.DoomGame`), for screenshots
(`game.get_state().screen_buffer`) and game variables; `brain.run_script(n, *args, read=k)` runs one of your world's ACS
scripts and returns `USER1..USER{k}`. `df.compile(..., resolution=vzd.ScreenResolution.RES_320X240, configure=fn)` forwards
a ViZDoom resolution and a `fn(game)` hook that can add game variables before the game starts. `wad="my.wad"` keeps the map
(and a `my.json` next to it) where you want it.

`df.plan(net)` returns `dict(dolls=..., sectors=..., fits=..., span=...)` **without** building anything (it counts the geometry the
compiler would emit; `fits` is false when the map, world included, would exceed 32,000 units on either axis, see
[`engine_limits.md`](engine_limits.md)). `df.plan(net)["map"]["world_area"]` is the rectangle where your world may go. Use it
first when your net is big.

### `df.SpikingNet`: integer leaky integrate-and-fire neurons

```python
net = df.SpikingNet(names, inputs, W, U, theta=10, leak=0)     # = doomformer.spiking.IntNet
with df.compile(net) as brain:
    v, s = brain.step(x_spikes)        # one time step in DOOM; potentials v and spikes s verified against IntNet.step
```

`s_t = [v_t >= theta]`, `v_{t+1} = v_t - theta*s_t - leak*[v_t >= leak] + leak*[v_t <= -leak] + W @ s_t + U @ x_t`.
The potentials live inside the map between steps, so `brain.step` is *stateful*. Run `net.run(X)` for the reference trajectory.
`|v|` must stay within ±12,000.

### `df.Transformer`: the one-map transformer

```python
t = df.Transformer.load("doomnames_onemap")                    # a shipped model name, or a QModel directory path
with df.compile(t) as brain:
    toks = [0] + [t.vocab.index(c) for c in "SH"]              # token 0 is the BOS/EOS marker
    logits = brain.logits(toks)                                # verified: every exposed pre-activation and the logits
    name = brain.generate("SH", temperature=0, seed=0, max_new=1)
```

Every step of `generate` is verified exactly like `intref.generate` (same greedy/seeded sampling). The sampling
itself (softmax and the random draw) is Python's. It takes about 90 s per letter for `doomnames_onemap`
(`examples/05_one_map_transformer.py`).

A config with `"attn": "softmax"` (and an integer `"R"`) gives softmax attention: the dolls compute the exponentials and
divide by their sum, in one-map and tile mode alike. `models/doomnames_softmax` is a trained example in tiles
(`doomformer demo --softmax`, trained by `training/train_qat.py --attn softmax --R 2`), `models/doomnames_onemap_softmax`
one in one map (`doomformer demo --one-map --softmax`, trained by `training/train_brain.py --attn softmax --R 1`), and
[HOW_IT_WORKS.md](HOW_IT_WORKS.md#softmax-attention) has the arithmetic.

## Hooks: world and actions

Both hooks go to `df.compile(net, world=..., actions=...)` and are passed straight to `compile_pure`.

### `world(m) -> str`

A callable that receives the brain's `doomformer.udmf.MapBuilder` (`from doomformer.udmf import MapBuilder` if you want
the type) after the brain is laid out, and returns extra ACS source (a `str`, `""` for none). Use it to add rooms,
things and scripts next to the brain:

* `m.rect(x0, y0, x1, y1, floor=0, ceil=128, light=160, tag=0, ftex=, ctex=, wall=, facing=)` adds an axis-aligned
  rectangular sector and returns its index. Rects must not overlap; edges two rects share become two-sided lines (a
  rect may touch part of a longer neighbour edge), free edges become solid walls. `facing="DOOR1"` sets the upper/lower
  texture seen *from the neighbours* (the front of a door). Textures are Freedoom's (`FLOOR4_8`, `STARTAN3`, `DOOR1`,
  `DOORTRAK`, `COMPBLUE`, `CEIL5_1`, `FLAT14`, ... all used in the examples).
* `m.link(a, b, special=..., arg0=..., playercross=True, repeatspecial=True)` puts a line special on every line between
  rects `a` and `b` (`playeruse=True` for a switch).
* `m.thing(x, y, type, angle=0, id=tid)` places a thing by doomednum (`1` player start, `2006` BFG9000, `48` tall
  techno column, `3004` zombieman, ...).
* **The world must place a player start (`m.thing(x, y, 1)`).** The brain's voodoo dolls are earlier player-1 starts and
  the *last* one becomes the real player, so yours must come last; `df.compile` checks that you placed one.
* **Where to build.** The brain occupies `y >= 0` (rows from x = -16,000 to 16,000, stacking north), its output lights sit
  a few rows below `y = -512`, and a closet for the conveyor scrollers is at `(-4096, -4096)`. The whole map must stay
  within **32,000 units per axis** (beyond that ViZDoom's 16.16 coordinate differences wrap: the player gets carried by the
  brain's conveyors and doll updates are lost, see [`engine_limits.md`](engine_limits.md)). Build inside
  `df.plan(net)["map"]["world_area"]` (for a small net about x -16,000..16,000, y -31,960..-2,048; the examples use y
  around -2,700). `df.compile` raises `ValueError` if a world rect overlaps the brain, the closet or another world rect,
  or pushes the map past the limit.
* **Free numbers:** sector tags **3..999**, thing TIDs **1..9999**, ACS script numbers **>= 200**. The brain uses tags 1, 2 and
  >= 1,000, TIDs >= 10,000, scripts 1-8 and 100+ and the ACS globals `mo_x`, `mo_y`, `mo_z` (the motion check). The
  `df.act` helpers reject numbers outside these ranges.
* The returned ACS is appended to the map's script. It may use the brain's globals `o1..o60` (ViZDoom `USER1..USER60`) to
  report values (the readout scripts use the same variables, so run your readout script between passes, not during one):
  `brain.run_script(201, read=2)` starts script 201, advances one tic and returns `[USER1, USER2]`
  (see `examples/03_*`, `04_*`, `06_*`).
* **The world persists across passes.** Between passes ACS script 3 rewinds *only* the brain (dolls home, biases restored,
  gates closed, output lights reset). Doors that were opened, lights that were changed, things, the player's position and his
  velocity are **not** reset. `examples/03_*` (door stays open, light stays on) and `examples/06_*` (the script that puts the
  marine back in the middle) show both sides of that.
* Do not rely on a pass being the only thing happening: the run only ends once every `DoomPlayer` (dolls and the player) has
  stood still for 8 tics, so a player who is still sliding delays the end of the pass.

### `actions = {output_name: [special, ...]}`

Each special is a dict `dict(special=<ZDoom line special number>, arg0=..., ..., arg4=...)`. Every voodoo doll that adds a
**positive** weight to that output also crosses these specials, so they fire:

* once **per positive contribution unit**: a doll of an input unit's staircase lane that passed its gate and has a weight > 0
  onto that output. An output that receives weight +2 from a unit whose staircase value is 3 fires its actions **3 times**
  (one per active lane). It does not fire 6 times, and it does not fire once because the output is positive. In
  `examples/06_*` a wind of +2 produces 2 kicks and the marine moves about twice as far as for 1 kick.
* **not on the final value.** If one unit pushes +1 and another pushes -5, the positive one still fired. Negative weights never
  fire anything. Design actions to be idempotent (set a light, open a door) or to be counters on purpose (thrust).
* **during the pass**, as real game events, while the network is still computing; for a spiking net the keys are **neuron names**
  and the specials fire once in every step in which that neuron spikes. For an MLP the valid keys are the value names
  `y{o}` (outputs), and also hidden `h{l}_{j}` / their twins `-h{l}_{j}`; for a transformer onemap value names such as `logits:t:j`.

Helpers in `df.act` (each returns a plain dict; `tests/test_kit.py::test_act_helpers_fire_in_doom` fires all four in DOOM):

| helper | ZDoom special | effect |
|---|---|---|
| `df.act.thrust_player(angle, force, tid, nolimit=False)` | `ThrustThing` (72) | kicks the thing with that TID; `angle` in **degrees** (0 east, 90 north, 180 west, 270 south), `force` in units/tic; ZDoom caps the speed at 30 unless `nolimit` |
| `df.act.door_open(tag, speed=16)` | `Door_Open` (11) | opens the door sector(s) with that tag (to 4 below the lowest neighbouring ceiling) and leaves them open |
| `df.act.light(tag, value)` | `Light_ChangeToValue` (112) | sets the light level of sectors with that tag |
| `df.act.script(n, *args)` | `ACS_ExecuteAlways` (226) | starts world script `n` (>= 200) with up to three int arguments; its activator is the doll |

Any other ZDoom special can be written by hand as a dict. The activator of every special is a voodoo doll, never the player.
Specials that act on "the activator" would act on the machinery, so pass explicit tags/TIDs.

A line special with a **nonzero TID** crossed by a doll acts on the thing with that TID (the real player), and leaves the
doll alone. That is why `thrust_player` takes a TID, and why the player needs one. Give it in the world's ACS with
`script 200 ENTER { Thing_ChangeTID(0, 900); }` (voodoo dolls never run ENTER scripts) and let the game tick once
(`brain.runner.game.advance_action(3)`) before the first pass.

## Limits and measured sizes

Hard limits (enforced by `net.check` / `RangeError`, and by `intref.check_limits` for transformers):

| quantity | limit | why |
|---|---|---|
| any value that feeds a gate (inputs, hidden pre-activations, spike potentials) | \|z\| <= 12,000 | gate heights are 16.16 fixed point; spans >= 32,768 overflow |
| thresholds | positive, nondecreasing, <= 12,000 | they are heights |
| every partial sum of an accumulation | \|sum\| <= 30,000 | the accumulator is a saturating int16 sector light and dolls cross in arbitrary order |
| weights (`intref`) | \|w\| <= 255 | `W_MAX` |
| one map's extent, world included | 32,000 x 32,000 map units | `df.plan(net)["fits"]`, `["span"]`; see [`engine_limits.md`](engine_limits.md) |

Size and speed, measured on one machine: aarch64, 16 cores shared with other jobs, one core per run, `nice -n 10`,
`OMP_NUM_THREADS=1`, headless ViZDoom 1.3.2, load average about 1. Test nets have random integer weights in [-3, 3],
biases in [-3, 3], signed input, two weight layers and thresholds `4, 8, ..., 4K` (`K` staircase lanes per unit, each
lane one voodoo doll). Inputs are random in +-4K. "Time per pass" is the mean wall time of `brain(x)`, including the
exact comparison with the reference, for passes after the first. The first pass also loads the map. Every pass below
was **exact**, including every hidden value.

| net (in, hidden, out), K | dolls | sectors | build | time per pass | game tics per pass | peak RAM |
|---|---|---|---|---|---|---|
| (2, 2, 1), K=1 | 8 | 104 | 0.01 s | 0.01 s | 200 | |
| (4, 8, 3), K=4 | 66 | 1,355 | 0.05 s | 0.07 s | 424 | |
| (16, 16, 10), K=4 | 194 | 6,806 | 0.25 s | 0.76 s | 960 | |
| (16, 32, 10), K=8 | 514 | 23,866 | 0.93 s | 7.5 s | 2,264 | |
| (32, 64, 10), K=8 | 1,026 | 77,678 | 3.2 s | 40.7 s | 4,360 | |
| (64, 128, 10), K=8 | 2,050 | 275,418 | 11.8 s | 209.5 s (first pass 236 s) | 8,544 | 2.0 GB |

`df.plan` only (counted, not built or run), under the 32,000-unit limit:

| net (in, hidden, out), K | dolls | sectors | span | fits | plan time |
|---|---|---|---|---|---|
| (128, 256, 10), K=8 | 4,098 | 1,030,814 | 31,768 x 13,256 | yes | 9.9 s |
| (784, 128, 10), K=8 | 13,570 | 3,122,886 | 31,888 x 31,448 | yes | 128 s |
| (256, 512, 10), K=8 | 8,194 | 3,990,314 | 31,860 x 35,384 | **no** | 67 s |

`fits` covers geometry only. Maps with more than 4,095 dolls also meet ViZDoom's 4,096-object limit (see
[`engine_limits.md`](engine_limits.md)), and load time and memory are separate questions: the 275k-sector map above
already needs 2 GB and 3.5 minutes per pass.

The shipped one-map transformer `doomnames_onemap` (1 layer, d=8, 1 head, MLP 16, context 6) is 3,548 dolls and
95,448 sectors. It builds in about 4 s, and one forward pass (one letter) takes about 85-96 s (3,488 game tics). A
5-step generation took 423 s. `onemap.tiny_model()` is 593 dolls, 12,620 sectors, 1,800 tics, a few seconds per
forward. The bigger `doomnames` does not fit one map: `df.plan` counts 116,968 dolls and 15.2 million sectors.

From about 20,000 sectors upwards a pass cost 3-25 ms of wall time per game tic in these runs. Doubling both widths
from (32, 64, 10) to (64, 128, 10) doubled the game tics (4,360 to 8,544), multiplied the sectors by 3.5 and the time
per pass by 5. Every staircase lane of every unit is one doll (the dolls column counts lanes), so `levels` (or the
length of your threshold arrays) multiplies the size directly. Plan first, and try fewer levels before shrinking your
layers. Tiny nets (hundreds of sectors) are effectively instant, which is why the examples run in about a second.

## What Python does vs what DOOM does

| step | who |
|---|---|
| build the map, compile ACS, write the WAD | Python (`pure.compile_pure`, ACC) |
| write the input values into the port sectors | Python (`puke` commands to ACS scripts), once per pass |
| staircase activations, sums of weights, products in attention, spike/leak/reset | **DOOM** (dolls, lines, light levels, plane heights) |
| softmax attention (`attn="softmax"`): the exponentials and the division by their sum | **DOOM** (a log-spaced staircase, then a count of comparisons) |
| chaining layers, waiting between them | **DOOM** (clock doll, barriers), no Python in between |
| world actions (doors, lights, thrusts) | **DOOM**, as part of the same pass |
| read outputs and hidden values back | Python (`puke` readout scripts into `USER` variables) |
| the integer reference | Python/NumPy (independent implementation of the semantics) |
| comparison, `MismatchError` | Python |
| sampling the next token (softmax over the logits, the random draw) | Python (seeded, same code as the reference) |
| quantising a float model | Python, before DOOM sees anything |

## Recipes

* **Run any quantised PyTorch MLP:** `df.MLP.from_torch(model, calib_x)` -> `df.compile` -> `brain(net.encode(x))`.
* **A network that opens a door / switches a light:** `world` builds the room, `actions` attach `df.act.door_open` /
  `df.act.light` to an output; make the actions idempotent. `examples/03_brain_opens_doors.py`.
* **A network that moves the player:** give the player a TID, `df.act.thrust_player(angle, force, tid)` per direction,
  recentre him with a script between passes. `examples/06_shove_the_marine.py`.
* **Light show from a spiking net:** neuron names as `actions` keys, `df.act.light(tag, 255)` on the neuron's own panel and
  `df.act.light(prev_tag, low)` to switch the previous one off. `examples/04_spiking_disco.py`.
* **Walk around the circuit:** `df.play(brain)` or `doomformer play x.wad` opens the map in a visible ViZDoom window in
  spectator mode (needs a display; the brain stays idle, nothing is computed). `df.record.topdown(wad)` draws any brain WAD from above
  as a PIL image without a display (used for the pictures of examples 01 and 05).
* **Hand-write a function:** staircases make any monotone step easy: parity of `s` ones is
  `[s>=1] - [s>=2] + [s>=3]`. `examples/01_xor.py`.
* **Plan before building:** `df.plan(net)` (or `doomformer plan model_dir_or_mlp.npz`); if `fits` is false, or the dolls/sectors are more than you
  can wait for, shrink the net or use fewer staircase levels.

## Troubleshooting

* **ACC not found / the ACC build fails.** `doomformer doctor` shows what it found. Lookup order: `$DOOMFORMER_ACC` (an `acc` binary
  with its include files nearby), the repository's `third_party/acc/build/acc` (only looked for in a source
  checkout; skipped when `DOOMFORMER_ACC_IGNORE_REPO=1`), then `~/.cache/doomformer/acc/<rev>/build/acc`, which is built once on first use with
  `git clone https://github.com/rheit/acc` + `git checkout <pinned rev>` + `cmake`. For that you need `git`, `cmake` and a C
  compiler (`gcc`/`clang`) on `PATH` and network access to github.com (each step has a time limit); the error names whichever is
  missing or failed. A pip install of doomformer does not bundle ACC; offline, build it elsewhere and point `DOOMFORMER_ACC` at it.
* **`MismatchError`.** Run `net.check(x)` first. A `RangeError` means a value left ±12,000 / ±30,000. If the check is
  clean, the difference between DOOM and the reference is real: save the map (`df.compile(net, wad="bug.wad")`), the input and the error text and open an issue.
* **Slow or running out of memory with a big net.** Check `df.plan(net)`. Millions of sectors need many GB and minutes to load.
* **`df.play` / `doomformer play` raises "needs a display".** It opens a visible window, so it needs `$DISPLAY` or `$WAYLAND_DISPLAY`. On a
  headless machine everything else works: `brain(...)` renders offscreen, and `brain.runner.game.get_state().screen_buffer` gives you
  screenshots (examples 03, 04 and 06).
* **`RuntimeError: acc failed: ...`.** Your world's ACS did not compile; the message is ACC's output. When you build the source in an
  f-string, double the braces (`{{`, `}}`), and number your scripts >= 200.
* **`ValueError: world rect ... overlaps ...` / "... span ..." / "world must place the player start".** Build inside
  `df.plan(net)["map"]["world_area"]`, away from `(-4096, -4096)`, and call `m.thing(x, y, 1)` for the player; see
  [the world hook](#hooks-world-and-actions).
* **The player sits somewhere unexpected.** The *last* `thing(type=1)` is the player; earlier ones are brain dolls.

## Going further

| want to | read | example |
|---|---|---|
| an agent that senses, decides and acts entirely in the map (`df.Agent`, `df.sense`) | [`agents.md`](agents.md) | `07_autonomous_turret.py`, `08_your_own_agent.py` |
| a GIF/MP4 of the dolls computing (`brain.record`) | [`recording.md`](recording.md) | `09_record_a_gif.py` |
| models too big for one map (`df.compile(..., mode="tiles")`) | [`scaling.md`](scaling.md) | `10_big_model_tiles.py` |
| many inputs through one map in many DOOM games, here or on a Ray cluster (`df.run_many`) | [`scaling.md`](scaling.md#scaling-out-many-games-many-machines) | |
| conv layers, other activations, `df.staircase` | [`interop.md`](interop.md) | `11_conv_in_doom.py` |
| learning inside DOOM (`df.Perceptron`) | [`learning.md`](learning.md) | `12_doom_learns_live.py` |
| the engine's hard limits (map extent, object count) | [`engine_limits.md`](engine_limits.md) | |

## Share your project

Open a **Show and tell** issue at <https://github.com/mottopanikeiku/doomformer/issues/new/choose> with a link to your
repository, a screenshot or GIF and the output of `doomformer doctor`. A good project repo has a script that prints
`EXACT` per pass, the `.wad` (or the script that builds it), and one sentence that states what DOOM computed and what
Python did. Pull requests that add an example to `examples/` are welcome.
