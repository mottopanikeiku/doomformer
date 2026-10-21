# Recording a run: `brain.record(...)`

`brain.record(inputs, "run.gif")` runs your inputs through the brain exactly like `brain(x)` (or `brain.step(x)`,
`brain.generate(prompt)`) and, while DOOM computes, films it: the compiled map seen from above with every voodoo doll as a
dot, and/or the player's first-person view. The output is a small `.gif` or `.mp4`. [`examples/09_record_a_gif.py`](../examples/09_record_a_gif.py)
films XOR.

```python
import numpy as np, doomformer as df

with df.compile(net) as brain:                                    # net: any df.MLP / df.SpikingNet / df.Transformer
    rec = brain.record([[0, 0], [0, 1], [1, 0], [1, 1]], "xor.gif")     # view='topdown', every=8, fps=30
    print(rec)          # <Recording xor.gif: 104 frames, 0.19 MB, 4 pass(es), 768 tics, 20 values checked, verified=True>
    rec.results         # what brain(x) returned for each input, as it always does
```

`rec.verified` is `True` only if every recorded pass passed the exact comparison. If one does not, `record` still writes the
file (the failing pass ends on a red MISMATCH banner, never EXACT) and then raises `df.MismatchError`; a `df.RangeError`
is raised before anything runs, as for a normal call, and nothing is written.

## What each part does

| step | who |
|---|---|
| the network's arithmetic, the walking of the dolls, every number that is later compared | **DOOM** (the compiled map; ACS level scripts only copy values to `USER` variables and do the readout) |
| the exact comparison with the integer reference, `MismatchError` | Python (`brain(x)`, which `record` calls unchanged) |
| sampling: ViZDoom's object list (positions of every `DoomPlayer`) and, for first-person, the screen buffer, every `every` tics | Python, **reading only**: `record` never sends a command to the game while a pass runs |
| the first-person image | ViZDoom's own renderer (this is what DOOM draws) |
| the top-down image, the legend, the text, the dots, the file | Python (Pillow, imageio), drawn from the positions DOOM reported |

Because the sampling is a wrapper around the runner's settle loop (`TileRunner.settle`, which already looks at the dolls every
8 tics), a recorded pass has the same tics and the same outputs as an unrecorded one. `tests/test_record.py` checks it: it
records four XOR passes, then runs the same four passes on a fresh unrecorded brain and asserts equal outputs and an
equal number of game tics (768 in total).

What is shown on screen, and when:

* **Input** ("Python writes x = [1, 0] into the 2 input ports"): from the first frame of each pass.
* **Running**: "DOOM is computing: tic N" and `result: pending`. No output and no EXACT yet, because the readback and
  the comparison happen only after the pass.
* **After the pass**, only if the comparison passed: the values DOOM read back (MLP `y`, spiking `spikes`/`v`, the transformer's top
  logits of the last position) and `EXACT: n values read back from DOOM == the integer reference (tolerance 0)`, held for
  `hold` seconds. `n` is `brain.info["checked"]` of that pass (every exposed value, hidden ones included).
* For `brain.generate`, each forward pass is its own pass in the film, and the last one adds `generated 'AC' (Python picks each
  letter from DOOM's logits)`: the choice of the letter is Python's (argmax/seeded sampling), the logits are DOOM's.

## Arguments

```python
brain.record(inputs, path, view="topdown", every=8, fps=30, *, width=640, colors=None, hold=1.5,
             camera_script=None, title=None, **generate_kwargs)
```

* `inputs`: **MLP**: one integer vector, or a list / 2-D array of them (one pass each). **Spiking net**: one spike vector or a
  `(steps, n_in)` array (one DOOM step each; the state carries over from the brain's current state like `brain.step`).
  **Transformer**: a prompt `str` (runs `brain.generate`; `temperature=`, `seed=`, `max_new=` are passed on) or a list of token
  ids (one `brain.logits` pass).
* `path`: `.gif` (a palette shared by the whole file, only the changed region stored per frame; imageio) or `.mp4` (H.264,
  yuv420p, needs `imageio-ffmpeg`, already a dependency of the package).
* `view`: `"topdown"`, `"firstperson"` or `"both"` (side by side, map 60% / screen 40% of the width for a 4:3 screen).
* `every`: DOOM tics between frames; a positive multiple of 8 (the settle loop inspects the dolls every 8 tics and `record`
  does not change that). Default 8. `fps`: playback speed of the file; at the default a frame is 8 tics = 0.23 s of game
  time shown for 1/30 s.
* `width`, `colors`: the knobs for size. A frame is `width` pixels wide (heights follow); `colors` is the GIF palette size
  (default 64 for top-down, 128 when the first-person image is in the frame). `hold`: seconds a finished pass stays on screen.
* `camera_script`: see [Camera](#camera-first-person).

Only a **one-map brain** can be recorded: `df.compile(net)` gives you one whenever the net fits one map (and always with
`world=`, `actions=` or `mode="map"`); a tile-mode brain runs several games and `record` raises `TypeError` for it.

## The top-down view

* Lines: every linedef of the compiled WAD with both ends at `y >= 0` (the brain; your world, the player's room and the output
  lights are at `y < 0` and are not drawn). Walls and ordinary two-sided lines are grey-blue; **lines that carry any special**
  (weights, barriers, teleports, scripts) are orange. Shapes are to scale and a map much wider than it is tall is cut
  into bands stacked top to bottom, like lines of text (so a corridor can continue on the band below). Same drawing as
  `df.record.topdown(wad)` (a still PIL image; that one always draws every line); above ~40,000 linedefs a recording
  draws a density image instead of one line at a time.
* Dots: **every voodoo doll** (the real player is excluded), including each layer's clock doll (the one that walks the long corridors and releases the layer's
  waves). Colour = layer, named in the legend under the title (`layer 1`, `layer 2` for MLPs; the stage names such as
  `embed`, `L0.qkv` for a transformer; `fire`, `synapses` for a spiking net). A dim dot waits at home, a bright dot has already moved in this pass, a **white
  dot with a tail is moving**: it moved more than 0.5 map units since the previous frame, and the tail goes back to where it was.
  The colours come from the compile-time layout; if its doll count does not match the map's, the legend is dropped and all dots share one colour.
* Frames are samples. A doll crosses its corridor in a few tics, so with `every=8` a walking doll shows up only in the frames
  that catch it; the tail is what shows where it came from.

## Camera (first-person)

Without more, `firstperson` is what the real player sees from where he stands (ViZDoom's screen buffer; the weapon sprite is part
of it). For a camera your world places a camera thing and an ACS script that makes the player's view follow it:

```python
def world(m):
    ...                                                           # your rooms
    m.thing(-700, -2688, 1)                                        # the player
    m.thing(-300, -2688, 9025, angle=180, id=77)                   # SecurityCamera with TID 77
    return "script 202 (int cam) { if (cam) { ChangeCamera(77, 0, 0); } else { ChangeCamera(0, 0, 0); } }"

with df.compile(net, world=world) as brain:
    brain.record(x, "cam.gif", view="both", camera_script=202)
```

`record` calls `brain.run_script(202, 1)` before the first pass and `run_script(202, 0)` after the last one (also if a pass failed).
ACS `ChangeCamera` is DOOM; `record` only presses your script. Number the script >= 200, as for every world script. The test
`test_mp4_and_view_both_with_a_world_camera` records with and without the camera and asserts the two first-person views differ.
The default screen is 160x120; pass `resolution=vzd.ScreenResolution.RES_320X240` to `df.compile` for a sharper picture (it is
resized to the frame either way).

## Size and speed (measured)

Measured on one shared aarch64 machine, one core, `nice -n 10`. File sizes are from `record` runs, times are wall clock.

| film | frames | file |
|---|---|---|
| XOR, 4 passes, top-down, `width=800`, `every=8` (example 09) | 104 | GIF 0.24 MB, MP4 0.12 MB |
| 8-neuron spiking ring in example 04's hall (20 dolls), first-person from a 320x240 screen, `width=480`, `every=16`, 10 steps | 140 | GIF 0.44 MB |
| tiny one-map transformer (593 dolls, 12,620 sectors), one forward pass, `width=640`, `every=64` | 31 | GIF 0.10 MB |
| the same pass, `every=8` (225 samples) | 225 | MP4 0.39 MB |

Sampling costs wall time: one tiny-transformer forward pass takes 4.8 s unrecorded, 5.7 s recorded at `every=64` and 9.1 s at
`every=8` (including drawing and encoding). The frames of all passes are kept in memory as positions (about 8 bytes per doll per
sample) until the end; for a long `generate` raise `every`. A GIF's size is dominated by how many frames there are and how much
changes between them; lower `width`, raise `every`, or use `.mp4`.

## Limits

* The film shows positions of dolls at sampled tics, drawn by Python. The EXACT line (the exact comparison of
  `brain(x)`) is what shows the numbers are right.
* `record` identifies the real player as the last `DoomPlayer` in ViZDoom's object list (the compiler guarantees the player start is the
  last one) and draws the rest; the layer colouring needs the layout's doll counts to add up to the map's.
* GIF frame times are stored in 10 ms units. With `fps=30` frames last 30 ms in a GIF; the MP4 is exactly 30 fps.
* Sampling during a pass relies on `TileRunner.settle` (a wrapper is installed on the runner instance for the duration of the call and
  removed afterwards); a brain whose runs do not go through it cannot be recorded.
