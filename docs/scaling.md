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

The estimate is a deterministic geometry heuristic, **not measured wall time**.
For each stage it adds

```
ceil((waves * wave_spacing + corridor_width + 400 + 512) / 16) + 16
```

and sums stages. Here 400 and 512 are the pure compiler's sync/clock distances;
16 units/tic is the nominal conveyor speed; the final 16 tics is a stage margin.
Clock folding, acceleration, inactive gates and engine costs are not modelled, so
the estimate guides routing and neither bounds settling nor predicts that tiles are
faster. Force either mode when comparing.

`plan` emits neither WAD nor ACS and starts no game. It reports:

- `map`: dolls, sectors, `fits`, `span` ([width, height]), `bounds`, `world_area`, `tiles=1`,
  `estimated_settle_tics`;
- `tiles`: summed dolls/sectors across **distinct WADs**, number of WADs, largest
  tile's dolls/sectors, `span` (the largest tile extent per axis) and `fits` (every tile map spans at most 32,000
  units per axis and holds fewer than 20,000 dolls; otherwise a `reason` names the tiles);
- `mode`: selected mode; `workers`; `auto_settle_tics=2000` (informational: the estimate above which a one-map pass is
  slow; auto does not switch on it).

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

`run_many(net, inputs, *, workers=None, backend="local", address=None, world=None, actions=None, progress=None)`:

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

The command line uses it: `doomformer mnist --digits N --workers W [--ray [ADDRESS]] [--all] [--record PATH]`.
Up to 100 digits come from the shipped test digits; `--digits` above 100 samples from, and `--all` reads, the
10,000 MNIST test digits, downloaded once from `https://ossci-datasets.s3.amazonaws.com/mnist/` into
`~/.cache/doomformer/mnist` and checked against their sha256 digests. Runs of more than 8 digits draw the first 4,
print a progress line at most every 30 s and end with the summary (correct/N, exact/N, digits per minute, games).
`--record PATH` writes the run and one JSON line per digit (DOOM's one-hot, label, checked values, tics, seconds,
game).

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
