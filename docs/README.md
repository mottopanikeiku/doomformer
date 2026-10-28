# Documentation

Start with [BUILD_YOUR_OWN.md](BUILD_YOUR_OWN.md) if you want to compile a network, or [HOW_IT_WORKS.md](HOW_IT_WORKS.md)
if you want to know what the dolls do. The repository front page is the [README](../README.md); runnable scripts are in
[`examples/`](../examples) and the release history is in the [changelog](../CHANGELOG.md).

## Using the toolkit

| Page | Content |
|---|---|
| [commands.md](commands.md) | Every `doomformer` command with its options, run time and what it checks |
| [BUILD_YOUR_OWN.md](BUILD_YOUR_OWN.md) | The `import doomformer as df` manual: compile an MLP, a spiking net or a transformer, add rooms, doors and lights, what DOOM computes and what Python does |
| [interop.md](interop.md) | Getting a float PyTorch or NumPy model into DOOM: convolutions, activations, argmax, what is exact and what is approximated |
| [recording.md](recording.md) | `brain.record(...)`: film a run as a GIF or MP4 |
| [scaling.md](scaling.md) | Tile maps, many games in parallel, many machines with Ray, the cost model behind `plan` (`python -m demos.costcheck` validates it), all 10,000 MNIST digits in tiles |
| [agents.md](agents.md) | `df.Agent`: sensors, an in-map clock, the network and actuators in one map |
| [learning.md](learning.md) | `df.Perceptron`: weights stored as ceiling heights, learning executed by dolls |
| [scheduling.md](scheduling.md) | `df.compile(opt=0..3)`: waves, the four schedule levels, the static checker (`doomformer lint`), `plan --explain`, settle tics per level |
| [tuning.md](tuning.md) | `doomformer tune`: the failure edge of each timing constant measured against the oracle, the table behind `opt` 2 and 3, `tune --check` |
| [machine.md](machine.md) | `df.Machine`: a counter machine in one map (registers as plane heights, the program as geometry or as initial state), what is and is not claimed, limits, comparison with counter.wad, conway.wad and the DOOM calculator |
| [interventions.md](interventions.md) | `df.intervene` and `df.patch_recovery`: zero, set, mean, resample and scale at a named site, position and head; how DOOM runs them (doors or a bus patch), what the check covers, activation patching with recovery executed by dolls, the induction circuit, limits |

## How it works

| Page | Content |
|---|---|
| [HOW_IT_WORKS.md](HOW_IT_WORKS.md) | Buses, staircases, crossbars, softmax attention, RMSNorm, the gated MLP, generation without Python, speed |
| [engine_limits.md](engine_limits.md) | Map extent (32,000 units per axis), the 4,096-object list, and how to run the engine-fact probes (`doomformer facts`) |
| [related_work.md](related_work.md) | Prior work and tooling this repo builds on or differs from, with the pages that were opened |
| [verification.md](verification.md) | What the exactness check compares, translation validation, `doomformer fuzz` (differential fuzzing, planted bugs, mutation kill rates) and `doomformer verify` (every input class of a small net), fuzz and mutation results, what is not covered |

## Demos and results

| Page | Content |
|---|---|
| [DEMOS.md](DEMOS.md) | Every demo with its numbers, what DOOM computes and what Python does, seeds and selection; [Run it yourself](DEMOS.md#run-it-yourself) has the command for each research demo |
| [`demos/README.md`](../demos/README.md) | The research code and how to start each demo from a clone: `python -m demos.brainlevel`, `aimlearn`, `flyloop`, `lobotomy`, `induction` and `circuit` |
| [drawpad.md](drawpad.md) | `doomformer draw`: the map that holds an MNIST network and the room it reads from |
| [FLY.md](FLY.md) | The fruit-fly circuit that steers the marine |
| [fly_circuit.md](fly_circuit.md) | How the FlyWire circuits were made, how they compare with the float model, where they deviate |
