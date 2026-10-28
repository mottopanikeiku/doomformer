# Scaling from one map to operation tiles

```python
import doomformer as df

if __name__ == "__main__":       # tile workers are separate processes: keep the guard when you run this as a script
    net = df.Transformer.load("doomnames")
    print(df.plan(net, mode="auto", workers=2))
    with df.compile(net, mode="tiles", workers=2) as brain:
        name = brain.generate("F", max_new=4)  # FEAR, including the EOS decision
        print(name, brain.info)
```

`compile` and `plan` accept `mode="auto" | "map" | "tiles"` and a positive integer
`workers` (default 1). Tile mode supports integer Transformers and MLPs. Stateful
spiking networks remain map-only: their potentials must stay inside one game.

## What runs where

**Map:** the full network is one map. DOOM dolls cross weight lines, move the next
layer's gates and release the layer clocks. Python writes inputs, waits, reads
results and independently checks them. No Python integer bus between layers.

**Tiles:** `model_tiles` builds a WAD for each operation kind; `orchestrator.TilePool`
schedules executions in up to `workers` processes. DOOM dolls compute each tile's
integer sums, staircase activations and attention products. ACS scripts move the
input gate planes, reset dolls/accumulator lights and expose the sector lights
through USER variables. **Python copies these read-back integers between games**,
so the network does not run autonomously in one DOOM world. ACS does not evaluate
the neural network. Python also schedules attention slots and samples the next
token from the DOOM-computed logits, exactly as `intref.choose`.

The toolkit bounds each worker to **one live DOOM game** (`max_live=1`), and
switching tiles closes/reloads games. `workers=2` therefore means at most two live
DOOM games. Compilation also uses at most the requested number of workers.

Every `brain.logits(tokens)` / `brain(tokens)` in tile Transformer mode checks all
exposed preactivations, causal scores, weighted sums and logits against a fresh
independent NumPy integer `intref.forward`, with **tolerance 0**. It separately
checks each DOOM-read negation twin. Upper-triangular, noncausal scores are not
with `(name, DOOM, reference)` triples. Each check covers the inputs of that call
and the integer model only, with no claim about other inputs or about the
pre-quantisation floating-point model.

For generation, a matching token prefix reuses only the already-read DOOM bus
values. Each new position executes its tiles; the entire requested prefix is
checked again before its logits are returned. A changed/shortened context resets
the host's token/attention cache. The inherited Brain generation loop checks each
step and keeps the same greedy/seeded sampling semantics as map mode.

For MLPs there is one tile per layer. Python copies raw hidden values and their
DOOM-computed negations, **not** activations; the next tile's gates compute the
staircase. `brain(x)` and batches return verified integer outputs; `brain.run(x,
check=True)` returns and checks all hidden/output readbacks. As in map mode,
`run(x)` defaults to raw readback without comparison. Range checks run before
execution.

## Automatic selection and planning

Without map hooks, `mode="auto"` first plans the one-map geometry and **builds one map whenever it fits**, because
then nothing but the inputs and the outputs crosses into Python. It selects tiles for an MLP or Transformer only
when the one-map layout does not fit (it would span more than 32,000 units on either axis, see
[engine_limits.md](engine_limits.md)) and the tile plan does. A one-map pass can be slow; plan's
`estimated_settle_tics` tells you roughly how slow, and `mode="tiles"` is always available on request.

The estimate is a deterministic geometry heuristic. It is a count of game tics, not seconds ([Cost model](#cost-model)
turns it into seconds). On 35 passes of 25 random MLPs (0.5k to 275k sectors) it was between 1.2 % below and 3.9 %
above the tics the engine needed.
The formula: for each stage it adds

```
ceil((waves * wave_spacing + corridor_width + 400 + 512) / 16) + 16
```

and sums stages. Here 400 and 512 are the pure compiler's sync/clock distances;
16 units/tic is the nominal conveyor speed; the final 16 tics is a stage margin.
Clock folding, acceleration and inactive gates are not modelled. The estimate guides routing; it does not bound
settling (a pass that needs more than the larger of 20,000 and twice the estimate raises). It says nothing about which
mode is faster: `plan`'s `recommended` does, from predicted seconds.

`plan` emits neither WAD nor ACS and starts no game. It reports:

- `map`: dolls, sectors, `fits`, `span` ([width, height]), `bounds`, `world_area`, `tiles=1`,
  `estimated_settle_tics`, and for a map that fits `predicted_tics`, `sector_tics` (sectors x tics) and
  `predicted_seconds`;
- `tiles`: summed dolls/sectors across **distinct WADs**, number of WADs, largest
  tile's dolls/sectors, `span` (the largest tile extent per axis) and `fits` (every tile map spans at most 32,000
  units per axis and holds fewer than 20,000 dolls; otherwise a `reason` names the tiles), `per_tile` (name, sectors,
  dolls, ports, outputs and longest zone of every tile) and, for an MLP, `predicted_tics` (summed over the tiles),
  `sector_tics` and `predicted_seconds` (the sum over the layers, one input, games loaded; Transformers get none: their
  tile executions per forward depend on the sequence);
- `mode`: selected mode; `workers`; `auto_settle_tics=2000` (informational: the estimate above which a one-map pass is
  slow; auto does not switch on it);
- `recommended`: `"map"` or `"tiles"`, the one with the smaller predicted seconds when both fit and both are
  predicted (an MLP), else what auto builds; `recommended_ratio`: slower over faster predicted seconds. Auto still
  builds one map whenever it fits, so the recommendation is advice: `mnist --tiles` and `df.run_many(..., mode="tiles")`
  act on it.

Top-level `dolls`, `sectors`, `fits`, `span` always describe the **map** layout, even
when the selected `mode` is `tiles`; use `report[report["mode"]]` for that mode's
counts. The tile count is not execution count: attention tiles run repeatedly for
heads, dimensions and positions. Counts come from the actual geometry emitters
using a counting sink. `fits` is a layout/resource check and runs no engine. It is
false (with a `reason`) when the map would span more than 32,000 units on either
axis, where ViZDoom starts to put things in the wrong sector ([engine_limits.md](engine_limits.md)).
If an individual pure corridor exceeds 32,000 units, the map option reports
`fits=False`, a reason and unavailable counts/estimate (`None`). When the selected mode does not fit, the report
has a top-level `reason` (auto: neither one map nor tiles fit) and `df.compile` raises `ValueError` instead of
building; `mode="tiles"` with a tile that does not fit raises too.

World/actions/resolution/configure hooks require a single map. Auto keeps map
mode when such hooks are supplied, the one exception to its routing rule.
Explicit tiles with these hooks raises an error rather than discarding them.
Spiking auto always stays map-only. Map mode uses one worker regardless of the
requested worker count.

## Cost model

What a pass costs follows from one law: the engine visits every sector on every tic, so

```
engine seconds per pass = tics x (sectors x about 100 ns + dolls x about 1 us)        (one map)
```

The sector term is flat in activity: an idle map costs what a running one does, and 500 extra idle dolls on a
16,384-sector grid cost 18 us per tic (a probe from a review run, not repeated here). `plan` uses it (`doomformer/cost.py`):
`predicted_seconds` is the engine time of one pass with the games loaded, on an otherwise idle core of the machine of
the fit; `sector_tics` (sectors x tics) does not depend on the machine.

**Fit and validation.** `python -m demos.costcheck` (validation nets, about 10 min; `--refit` also the fit nets)
measures engine CPU seconds (user+sys of the ViZDoom child, from `/proc`) over 6 to 12 passes with the game loaded.
Random MLPs, one map: 14 fit nets of 2.2k to 275k sectors give 102.1 ns per sector-tic and 1.06 us per doll-tic
(root-mean-square error 19 % on the fit nets). Tiles: 32 tile runs of 9 nets plus 5 wider nets (up to 156k sectors)
give 69.7 ns per sector-tic, 2.6 us per doll-tic and 0.24 ms per run; a tile's tics are `tile_tics(zone)` (30 tics plus
the longest zone at 16 units per tic, rounded up to the 8-tic settle window). The validation nets were not used for
the fit (`NET K` is the layer sizes and the number of staircase thresholds; measured 2026-10-27 at load average 25
to 30, so the machine was shared, the CPU seconds carry cache contention):

| net | one map: sectors, tics | predicted s | measured CPU s | error | tiles: tics pred/meas | predicted s | measured CPU s | error |
|---|---|---:|---:|---:|---|---:|---:|---:|
| 8-12-4 K3 | 2,239, 527 | 0.17 | 0.13 | +28 % | 88/88 | 0.014 | 0.013 | +6 % |
| 24-24-8 K3 | 9,596, 1,062 | 1.29 | 1.42 | -9 % | 96/80 | 0.045 | 0.038 | +17 % |
| 48-32-10 K3 | 22,619, 1,739 | 4.73 | 5.28 | -10 % | 96/80 | 0.088 | 0.078 | +12 % |
| 100-32-10 K3 | 44,733, 2,987 | 15.85 | 18.06 | -12 % | 96/104 | 0.167 | 0.173 | -4 % |
| 20-20-20-10 K3 | 10,207, 1,259 | 1.64 | 2.11 | -23 % | 136/96 | 0.045 | 0.033 | +34 % |
| 128-64-10 K3 | 103,821, 4,059 | 47.16 | 41.44 | +14 % | 112/120 | 0.349 | 0.375 | -7 % |
| 32-64-64-10 K3 | 54,071, 2,647 | 16.24 | 16.33 | -1 % | 184/120 | 0.189 | 0.120 | +58 % |
| 200-50-10 K3 | 128,181, 5,612 | 81.49 | 76.91 | +6 % | 104/112 | 0.425 | 0.448 | -5 % |
| 8-12-4 K8 | 5,924, 1,087 | 0.92 | 1.20 | -23 % | 88/88 | 0.033 | 0.030 | +11 % |
| 24-24-8 K8 | 25,522, 2,502 | 8.05 | 10.28 | -22 % | 96/96 | 0.101 | 0.100 | +1 % |
| 48-32-10 K8 | 60,232, 4,299 | 31.11 | 40.31 | -23 % | 96/104 | 0.191 | 0.195 | -2 % |

One map: 11 nets, root-mean-square error 18 %, largest +28 %, 10 of 11 within 25 %. Tiles: root-mean-square error 22 %,
largest +58 %, 9 of 11 within 25 %. The tile tic formula is an upper estimate: in a random net the later layers settle
before it (the 32-64-64-10 row, 184 predicted against 120 measured); the shipped MNIST tiles run exactly at it
(216 tics predicted and measured). The one-map tic estimate itself (`estimated_settle_tics`) was between 1.2 % below
and 3.9 % above the measured tics on all 35 passes. Shipped MNIST: predicted 99 to 102 s one map (measured 104 s wall
per digit, one game at load 25, 2 digits; 64 to 117 s in the older runs) and 0.81 s tiles (measured 0.88 to 0.92 s CPU,
0.83 s mean wall per pass in the 10,000-digit campaign).

What the model leaves out: wall time on a loaded machine (1.3 to 2.5 times the CPU time in these runs: a 64-64-10 map
pass took 14.2 s CPU and 27.0 s wall at load 41), game start (0.3 s for 4k sectors, 7 to 13 s for 54k to 103k), compile
time, memory (about 5.2 KB per sector plus 20 MB per engine), and other CPUs: constants fitted on arm64 under load are
not a promise for yours. Transformers get no tile prediction: their tile executions per forward depend on the sequence.

**The idle conveyor tax.** The sector term is the conveyor: 87 to 96 % of the 32-32-10 map's sectors carry the
conveyor tag, and each carries 4 stacked Boom carry scrollers (`tile.CONVEYOR`), which the engine updates every tic whether or
not a doll stands there. Reproduced on one real map per size by recompiling it with `tile.CONVEYOR["stack"]` set to
4, 2, 1 and 0 and advancing 300 idle tics (engine CPU is read in 10 ms ticks, so the stack-0 cost is shown as wall
time; maps with stack 0 do not run, this is the idle cost only; load 28):

| net (sectors) | stack 4 | stack 2 | stack 1 | stack 0 |
|---|---:|---:|---:|---:|
| 12-16-6 (2,628) | 267 us/tic (101 ns/sector-tic) | 233 us | 200 us | under 55 us |
| 32-32-10 (10,570) | 1,533 us (145 ns) | 1,233 us | 967 us | 96 us |
| 64-64-10 (36,088) | 4,000 us (111 ns) | 3,267 us | 2,467 us | 124 us |

(CPU microseconds per tic; stack 0 is wall time, the CPU timer is too coarse there.) The first scroller costs about 65 to
80 ns per sector, each further one 10 to 25 ns; with none the idle map costs 3 to 9 ns per sector-tic. Taking all four
scrollers away makes an idle tic 20 to 40 times cheaper. A doll needs the carry to move, so the scrollers cannot simply
go; a conveyor that sleeps while no doll stands on it would be a change in the engine, which this repository does not
make (not built, not tested).

## One map versus tiles: the price

A one-map layer releases its dolls in serial waves (a wave is 64 map units, about 4 tics, and dolls whose target
sectors collide share none): MNIST layer 0 has 784 waves, layer 1 512. A tile has the sectors of one layer and runs
every doll of it side by side. For the shipped MNIST net, from `plan` and the runs:

| | one map | 4 tiles |
|---|---:|---:|
| sectors | 158,550 | 111,112 (100,828 + 9,510 + 652 + 122) |
| game tics per digit | 5,696 | 216 (72 + 64 + 40 + 40) |
| sector-tics | 903 million | 7.9 million |
| engine CPU per digit | about 100 s (predicted 99 to 102) | 0.88 to 0.92 s; layer 0 is 0.72 to 0.77 s |
| wall per digit | 104 s (one game, load 25) | 0.83 s mean (campaign, 6 workers) |
| Python between layers | none | copies 64 hidden values and their twins into layer 1, and so on |

Tiles are about 100 to 125 times faster on this net. The price is the design: with tiles, Python carries the integer
layer values from one game to the next (it copies them, and the check compares every value), while in one map nothing
but the 784 inputs and the readback crosses. Use one map to show that no Python runs between layers, tiles to push many
inputs through.

The gap grows with width: measured CPU, one map over tiles, 12-16-6 16x (K3), 32-32-10 56x, 64-64-10 83x, and at K8 (more
thresholds, more dolls) 57x, 184x and 280x.

## The 65,000 times, as a range

The README's "about 65,000 times slower than NumPy" is one number for the 12-16-6 toy (726 ms in DOOM against 11.1 us). The
ratio depends on the net. `MLP.reference(x)` takes 13.2 us (4-6-3), 14.0 us (12-16-6), 15.6 us (32-32-10), 19.2 us (64-64-10)
and 47.8 us (784-64-45-10-10 MNIST, one thread, `timeit`, loaded machine). Engine CPU of one pass against that:

| net | one map | tiles |
|---|---:|---:|
| 12-16-6, K3 (3,925 sectors) | 0.32 s, about 23,000x | 0.020 s, about 1,400x |
| 32-32-10, K3 | 3.19 s, about 200,000x | 0.057 s, about 3,600x |
| 64-64-10, K3 | 14.2 s, about 740,000x | 0.172 s, about 8,900x |
| MNIST 784-64-45-10-10 (shipped) | about 100 s, about 2 million x (wall 104 s) | 0.9 s CPU, about 18,000x (wall 0.83 s) |
| MNIST, quiet machine (load 3 to 4, 2026-10-28; NumPy 47.7 us) | 74.9 s, about 1.6 million x at `--opt 0`; 2.78 s, about 58,000x at `--opt 3` | not measured |

So the range is about 20,000x to 65,000x for the smallest MLPs, 200,000x at 32-32-10, and one to two million times for MNIST
in one map at the default schedule (58,000 times at `--opt 3`); with tiles the same net is about 18,000 times slower than the NumPy reference. A ratio against a 14 us call is
mostly a statement about how little the reference does. `plan` predicts 100 s for the one-map MNIST pass at `--opt 0`, a
third above the 74.9 s of the quiet run: the cost model was fitted under load.

Three tables use a 12-16-6 net: the timer table in [HOW_IT_WORKS.md](HOW_IT_WORKS.md#speed) (202 dolls, 6,521 sectors, wall
time with the input writes and the readback), the range table above (K3, 3,925 sectors, engine CPU) and the conveyor table
(2,628 sectors, idle cost only). They are different builds of that shape and were not re-run on one net, so their sector
counts and the 65,000x and 23,000x are not interchangeable. The one-map MNIST pass is quoted as 104 s (wall, one game at
load 25, the campaign), 88 s (the `mnist` sample output in [commands.md](commands.md#doomformer-mnist)) and 78.8 s of
engine CPU (load 10, a later review run): the same map at different loads.

## Measured results

Measured on 2026-10-19 on Linux arm64 with ViZDoom 1.3.2, `OMP_NUM_THREADS=1`,
`nice -n 10`, headless, tolerance zero. Wall times include lazy
game/worker startup but exclude compilation unless labelled. Each figure is a
single run with cached tile builds reused.

Geometry from `df.plan`:

| Model | Mode | Dolls | Sectors | Distinct WADs | Fits | Estimated map tics |
|---|---|---:|---:|---:|---|---:|
| `onemap.tiny_model()` | map | 593 | 12,620 | 1 | yes | 1,937 |
| same | tiles | 214 | 2,766 | 12 | yes | |
| `models/doomnames_onemap` (d=8) | map | 3,548 | 95,448 | 1 | yes | 3,592 |
| same | tiles | 728 | 11,498 | 18 | yes | |
| `models/doomnames` (L=1, d=16, T=16) | map | 116,968 | 15,190,214 | 1 | **no** | 100,244 |
| same | tiles | 10,462 | 135,912 | 38 | yes | |

The d=16 model's largest tile has 448 dolls and 5,252 sectors. Its planned map
layout does not fit, so it has no map runtime.

A `tiny_model()` forward on tokens `[0, 1]` in both modes:

| Mode | Compile/build wall | Forward wall | Tile executions | Game tics | Checked integers | Exact logits |
|---|---:|---:|---:|---:|---:|---|
| map | 0.413 s | 5.214 s | 1 map pass | 1,800 | 148 | yes |
| tiles, 2 workers | 0.080 s | 2.447 s | 23 | 968 summed | 148 | yes |

Each row is `df.compile(m, mode=mode, workers=2)` followed by `brain.logits([0, 1])`
and `np.array_equal` against `intref.forward`, run sequentially for
`mode in ("map", "tiles")`.

The trained-model example:

```sh
OMP_NUM_THREADS=1 nice -n 10 .venv/bin/python examples/10_big_model_tiles.py
```

Output: `text="FEAR"`, `reference="FEAR"`, `exact=true`,
`workers=2`, `compile_seconds=0.192`, `generate_seconds=19.994`,
`total_seconds=20.187`, `runs=4`, `tile_runs=140`, `checked=1650`.
Generation starts from prompt `F`, computes five token positions (BOS plus FEAR),
and verifies the EOS decision. `checked` is the last call's number of comparisons.
`tics=5952` in the same output is the **last request's summed tile tics**, not
the whole generation's. The shipped default finishes the short name in
under 60 s on this machine.

Focused test command (`tests/test_tiled.py` only):

```sh
OMP_NUM_THREADS=1 nice -n 10 .venv/bin/python -m pytest -q tests/test_tiled.py
```

Result: **4 passed in 29.69 s**.

## Brain metadata and limitations

Both modes are `Brain` instances with context-manager cleanup. `info["runs"]`
counts API forward requests, `checked` comparisons on the last checked request,
and `tics` the last request's game tics (summed across tiles in tile mode).
Tile brains add `mode`, `workers`, `backend` (`"local"` or `"ray"`), `tiles`, and cumulative `tile_runs`.
`brain.wad` is the tile **directory**; a supplied `name.wad` becomes `name.tiles`.
Without a supplied path, content-addressed tiles live under the normal
`DOOMFORMER_CACHE`/`~/.cache/doomformer/brains` root.

A tile brain does not have one `runner`, one `run_script` world, or one playable
WAD. Those single-world APIs, world/actions hooks and recording through the
single-game runner require `mode="map"`. For inspection, a Transformer exposes
`brain.engine` (the existing `DoomTransformer`); both tile brain kinds expose
`brain.pool.log`. Worker processes use multiprocessing **spawn**, so scripts
starting games must use an `if __name__ == "__main__":` guard, as example 10 does.

The same safe integer ranges still apply: gate-fed values within ±12,000,
partial sums within ±30,000, weights within ±255. Tiles lift whole-map packing
pressure, not per-tile geometry, integer-range or total-compute limits. More
workers increase parallel opportunity only where independent tile tasks exist;
serial dependencies and game reloads remain real costs.

## Scaling out: many games, many machines

One map is one game. To push many inputs through the same map, `df.run_many` spreads independent passes over many
games: on this machine (worker processes) or on a Ray cluster of any size. Nothing is hosted; you run the games.

```python
import numpy as np

import doomformer as df

if __name__ == "__main__":       # local workers are spawned processes: keep the guard in scripts
    net = df.MLP([[[2, -1], [-1, 2]], [[1, -2]]], [[1, -1], [3]], [[1, 3], [1, 2]])
    xs = np.random.default_rng(0).integers(-4, 5, size=(12, 2))
    ys = df.run_many(net, xs, workers=3)                          # 3 DOOM games, every pass checked
    print(np.array_equal(np.stack(ys), net.reference(xs)["y"]))   # True
```

`run_many(net, inputs, *, workers=None, backend="local", address=None, world=None, actions=None, progress=None, mode="map", opt=0)`:

- compiles the map once, in the calling process (`world`/`actions` as in `df.compile`);
- starts `workers` games, each holding the map in one live DOOM game, and gives every input to the next idle game;
- checks every pass exactly as `brain(x)` does for an MLP (every hidden value and output, 248 values per MNIST
  digit) and as `brain.logits(tokens)` does for a Transformer (inputs are token lists, outputs the logits);
- returns the outputs in input order;
- range-checks every input before any game starts (`RangeError`/`ValueError` naming the input index);
- raises `df.MismatchError` on the first difference, with `.index` set to the input's index (the message starts
  `input 4: ...`), and `RuntimeError` naming the input when a pass fails;
- calls `progress(info)` after every checked pass, with `index`, `done`, `total`, `game`, `output`, `checked`,
  `tics`, `seconds` (the pass's wall time) and `retried`.

Spiking nets are refused: their potentials stay in one game between steps. `workers` defaults to
`min(CPU count, len(inputs))` locally and to the cluster's CPU count (at most one game per input) on Ray.

Local workers close their games and exit within about a second when the calling process dies without closing them
(`kill -9`, a crash). `doomformer` treats SIGTERM like Ctrl+C, so `kill` and `timeout` close every game as well.
A game started by plain `df.compile(net)` in your own script has its own guard: when the Python process dies (SIGTERM,
`kill -9`, a crash) without `close()`, a small `sh` loop started next to each engine (`doomformer._mp.guard_engine`)
kills the engine within a second. Before this, an engine (0.5 to 1.2 GB for a big map) kept running after `kill`.
Measured with a one-map brain and `kill -TERM` and `kill -KILL` of the parent: no engine and no guard process left
(`tests/test_tiles_many.py`). The guard needs `sh` and `ps` (Linux, macOS) and does nothing on Windows; use
`with df.compile(net) as brain:` to close the game on normal exit.

### Tile games in `run_many`: `mode="tiles"`

`df.run_many(net, xs, mode="tiles", workers=W)` (an MLP, local backend; `mode="auto"` takes `plan(net)["recommended"]`)
is the multi-map design. Every worker process holds one resident game per layer (`workers x layers` engines, only one
per worker computes at a time) and takes inputs one after another. For an input it runs layer 0, **Python copies the
integer values of the layer (hidden values and their DOOM-computed negation twins) into the ports of layer 1**, and so
on; nothing is loaded again between inputs. The check is the one of `mode="map"`, at tolerance 0: every hidden value,
its twin and every output equal `net.reference(x)`, and the set of names read back must equal the lowering's
(`MLP.exposed_names`). `progress` gets the same fields; `tics` is summed over the layers. The code is
`doomformer/tilefleet.py` (`TileGames`, one worker's games; `tile_worker`).

The command line: `doomformer mnist --tiles [--workers N]`. Its header and its summary line say that this is the
multi-map design and that Python copied the layer values between the games.

Scaling curve of `doomformer mnist --tiles --digits 96 --workers W --seed 0`, measured 2026-10-27 with other jobs on the
machine (load average 23 at the start of the 1-worker run, 31 at the end of the 8-worker run; `OMP_NUM_THREADS=1`; wall
time includes compiling and starting `4 x W` games, about 3 s for one worker); 96/96 digits exact each time:

| workers (games computing) | engine processes | wall | digits per minute | mean pass | peak engine memory |
|---:|---:|---:|---:|---:|---:|
| 1 | 4 | 91.5 s | 63 | 0.89 s | 0.42 GB |
| 2 | 8 | 50.1 s | 115 | 0.96 s | 0.77 GB |
| 4 | 16 | 36.5 s | 158 | 1.27 s | 1.47 GB |
| 8 | 32 | 30.4 s | 189 | 1.89 s | 2.79 GB |

The curve is flat early because 96 digits are few (the start of `4 x W` games is a larger part of the wall) and because the
machine was loaded: the mean pass grew from 0.89 s to 1.89 s as the 8 games shared cores with other jobs. The 10,000-digit
run below (6 workers, a quieter machine) reached 432 digits per minute with a mean pass of 0.83 s. One worker alone took
0.89 s per digit. The curve was not measured beyond 8 workers or on an idle machine.

### All 10,000 MNIST test digits in DOOM

`doomformer mnist --tiles --all --workers 6 --record models/mnist/doom_tiles_10000.json` (2026-10-27, 6 workers = 24 engine
processes, 6 computing at once): **10,000/10,000 exact** (248 values each, tolerance 0), **9,716 read correctly (97.16 %)**,
216 game tics per digit, 1,390 s (23.2 minutes) wall, 431.7 digits per minute, mean pass 0.83 s (0.77 to 1.73 s), no digit
retried. DOOM's one-hot output equals the NumPy integer reference's on all 10,000 digits, so the 97.16 % is the integer
net's accuracy (`models/mnist/EVAL.md`) reproduced by DOOM, not a new number. The file has one line per digit like
`doom_1000.json`, plus `mode`, `tiles`, `reference_match` and `accuracy_percent` in its header. Check it without DOOM:
rebuild the digits with `doomformer.cli.mnist_test_set()` and compare each row's `y` with
`net.reference(net.encode(x))["y"]` (all 10,000 rows match). Python wrote and copied the integers between the games; this
run is not the one-map design. The one-map MNIST run of 1,000 digits is `doom_1000.json`.

The command line uses it: `doomformer mnist --digits N --workers W [--tiles] [--ray [ADDRESS]] [--all] [--record PATH]`.
Up to 100 digits come from the shipped test digits; `--digits` above 100 samples from, and `--all` reads, the
10,000 MNIST test digits, downloaded once from `https://ossci-datasets.s3.amazonaws.com/mnist/` into
`~/.cache/doomformer/mnist` and checked against their sha256 digests. Runs of more than 8 digits draw the first 4,
print a progress line at most every 30 s and end with the summary (correct/N, exact/N, digits per minute, games).
`--record PATH` writes the run and one JSON line per digit (DOOM's one-hot, label, checked values, tics, seconds,
game). While it runs, `--record PATH` also writes `PATH.partial`, one line per finished digit, so a killed run keeps what
it read; the file is removed when the run ends normally.

### The Ray backend

```sh
pip install "doomformer[ray]"
```

`backend="ray"` runs one Ray actor per game, each reserving one CPU (`num_cpus=1`). `address` is passed to
`ray.init`; without it, `run_many` joins a running cluster (`address="auto"`) and otherwise starts a local Ray for
the call and shuts it down afterwards. If the process already called `ray.init`, that connection is used and kept.

- The map is compiled on the calling machine and sent to the actors as bytes through Ray's object store (`ray.put`
  once for all actors). Each actor writes it into a temporary directory on its own machine, so the nodes need no
  shared filesystem. Every node needs the same Python minor version, the same doomformer and Ray versions, and
  `vizdoom==1.3.2`; it does not need the ACS compiler.
- An actor that dies (process killed, node lost) costs only the input it was running: that input runs once more on
  another game, and its `progress` info has `retried=True`. A second death of the same input raises `RuntimeError`.
  Dead actors are not replaced; when none is left the call raises `RuntimeError`.
- More `workers` than free CPUs is allowed: the extra actors wait for CPUs and take inputs once they start.
- Without Ray installed, the error is `ImportError: backend="ray" needs Ray: pip install "doomformer[ray]"`.

Tile mode runs on Ray too: `df.compile(net, mode="tiles", backend="ray", workers=N, address=...)` backs the
`TilePool` with N actors instead of N local processes. The task protocol and the zero-tolerance checks are the same;
the tile WADs (all `.wad`/`.json` files of the tile directory) are shipped as bytes. As with a dead local worker, a
dead tile actor makes the pool raise `RuntimeError` and refuse further runs. Tile tasks stay with the game that holds
their tile, so every actor must start: keep `workers` at or below the cluster's free CPUs (a task queued on an actor
that never starts waits until the pool's 900 s deadline). `backend="ray"` requires `mode="tiles"`; for many passes
of one map use `run_many`.

### A Ray cluster across machines

On the head machine:

```sh
pip install "doomformer[ray]"
ray start --head --port=6379 --num-cpus=8      # 8 games on this machine
```

On every other machine:

```sh
pip install "doomformer[ray]"
ray start --address=HEAD_IP:6379 --num-cpus=8
```

`ray start --head` prints the `ray start --address=...` line for the other machines. Ray 2.59 enables token
authentication by default and keeps the head's token in `~/.ray/auth_token`; the other machines need the same token
(see Ray's token authentication guide), or every machine sets `RAY_AUTH_MODE=disabled` before `ray start`.
`ray status` lists the CPUs and memory that joined. Then, on any machine of the cluster:

```sh
doomformer mnist --all --workers 32 --ray --record mnist-10000.json
```

or `df.run_many(net, xs, backend="ray", address="auto")` in Python. `ray stop` on every machine ends the cluster.
Set `--num-cpus` to the number of games a machine should run: one per core, with about 1.3 GB of RAM each for the
MNIST map (measured peak 1.24 GB).

On one machine, start Ray once with `ray start --head` before running several `run_many(backend="ray")` calls at
the same time. Without it, the first call starts a local Ray, the later calls join it through `address="auto"`, and
they lose their games when the first call ends and shuts that Ray down (observed: the second process exited with no
result).

### What one game costs

Measured on the previous MNIST map (784-48-10: 1,227 dolls, 122,955 sectors; 5,176 game tics per digit). The shipped
784-64-10 map has 1,355 dolls and 158,550 sectors, takes 5,696 game tics per digit and peaks at 1.23 GB per game.

| Item | Measured |
|---|---|
| compile the map (once per call) | 4.8 s and 5.6 s |
| WAD + metadata shipped to Ray actors | 118.2 MB |
| `ray.put` of those bytes (once) | 0.37 s |
| an actor writing them to its disk | 0.04 to 0.05 s |
| local game start (load the map) | 2.32 s |
| Ray: 1 actor from creation to a loaded game | 3.2 s |
| Ray: 8 actors from creation to 8 loaded games | 6.48 s |
| starting a local Ray (`ray.init`) | 3.82 s |
| one pass, one game | 64 to 117 s, depending on machine load |
| engine process memory | 319 MB resident, 656 MB peak while loading |
| local worker process memory | 50 MB |
| Ray actor process memory | 317 MB resident, 429 MB peak (it receives the WAD bytes) |

One game uses one core. Budget one core and about 1.3 GB of RAM per game for the shipped MNIST map (measured peak 1.24 GB).

### Measured throughput

Measured on 2026-10-21 on Linux arm64 (16 cores, 61 GB) with ViZDoom 1.3.2 and Ray 2.59.0,
`OMP_NUM_THREADS=1 nice -n 10`, while other DOOM runs shared the machine (load average 11 to 35). Each row is one
`doomformer mnist --digits N --workers W --seed S [--ray]` run; wall time includes compiling the map, starting Ray
(Ray rows) and starting the games. These rows ran the previous 784-48-10 map; every digit was exact (216 values each,
tolerance 0).

| Backend | Games | Digits | Wall | Digits per minute | Mean pass (s) | Read correctly |
|---|---:|---:|---:|---:|---:|---:|
| local | 4 | 8 | 144.7 s | 3.32 | 67.0 | 8/8 |
| local | 4 | 8 | 147.3 s | 3.26 | 68.6 | 8/8 |
| Ray | 4 | 8 | 173.9 s | 2.76 | 76.4 | 8/8 |
| Ray | 4 | 8 | 172.0 s | 2.79 | 76.4 | 8/8 |
| local | 8 | 16 | 163.0 s | 5.89 | 76.4 | 15/16 |
| Ray | 8 | 16 | 191.5 s | 5.01 | 86.1 | 15/16 |
| Ray | 8 | 40 | 521.1 s | 4.61 | 96.5 | 39/40 |

Run one after the other, the Ray passes were slower, but so was the machine: when the two backends run the same 6
digits at the same time with 3 games each, the mean pass is 78.0 s on Ray and 78.3 s locally (wall 168.9 s and
168.5 s; that Ray was started before the timer). A game runs at the same speed in either backend; Ray adds its
start (about 4 s for a local Ray, 3 to 7 s for the actors) and lets the games run on more machines.

With the shipped 784-64-10 map, `doomformer mnist --digits 1000 --seed 0 --workers 14 --record ...` (local, 14 games,
while other jobs shared the machine): 7,420.9 s, 8.09 digits per minute, mean pass 103.1 s (83 to 153 s), 966/1,000
read correctly, 1,000/1,000 exact (248 values each), no retries, largest process 1.24 GB. The record is
[`models/mnist/doom_1000.json`](../models/mnist/doom_1000.json).

Tile mode, `models/doomnames` writing `FEAR` from `F` (the request of example 10; 140 tile runs, 118.1 MB of tile
WADs shipped to every Ray actor; Ray started before the timer, actor start and shipping included):

| Backend | Workers | Generate wall | Text, exact |
|---|---:|---:|---|
| local | 2 | 26.9 s | FEAR, yes |
| Ray | 2 | 23.7 s | FEAR, yes |
| local | 8 | 12.8 s | FEAR, yes |
| Ray | 8 | 13.55 s | FEAR, yes |

Focused test command: `python -m pytest -q tests/test_fleet.py` (the three Ray tests skip without Ray).
