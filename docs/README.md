# Documentation

New here: [BUILD_YOUR_OWN.md](BUILD_YOUR_OWN.md) to compile a network, [HOW_IT_WORKS.md](HOW_IT_WORKS.md) to see how
DOOM computes it. Scripts are in [`examples/`](../examples). Release history is in the [changelog](../CHANGELOG.md).

## Using the toolkit

| Page | Content |
|---|---|
| [commands.md](commands.md) | Every `doomformer` command, its options and sample output |
| [BUILD_YOUR_OWN.md](BUILD_YOUR_OWN.md) | The `import doomformer as df` manual |
| [interop.md](interop.md) | PyTorch and NumPy import, quantisation |
| [recording.md](recording.md) | `brain.record(...)`: film a run as GIF or MP4 |
| [scaling.md](scaling.md) | Tile maps, parallel games, Ray, the cost model behind `plan` |
| [agents.md](agents.md) | `df.Agent`: sensors, clock, network and actuators in one map |
| [learning.md](learning.md) | `doomformer learn digits` and `df.Perceptron` |
| [scheduling.md](scheduling.md) | `opt` levels, waves, `doomformer lint` |
| [tuning.md](tuning.md) | `doomformer tune`: the timing constants and their failure edges |
| [machine.md](machine.md) | `df.Machine` and `df.SubleqMachine`: counter and SUBLEQ machines in one map |
| [interventions.md](interventions.md) | `df.intervene` and `df.patch_recovery`: ablation and activation patching in DOOM |

## How it works

| Page | Content |
|---|---|
| [HOW_IT_WORKS.md](HOW_IT_WORKS.md) | The mechanism and the engine facts |
| [engine_limits.md](engine_limits.md) | Map extent, the object list, the fact probes, other ports |
| [related_work.md](related_work.md) | Prior work |
| [verification.md](verification.md) | The exactness check, fuzzing, mutation, what is not covered |

## Demos and results

| Page | Content |
|---|---|
| [DEMOS.md](DEMOS.md) | Every demo with its results, seeds and [commands](DEMOS.md#run-it-yourself) |
| [`demos/README.md`](../demos/README.md) | The research code and how to start each demo |
| [drawpad.md](drawpad.md) | `doomformer draw`: the MNIST map and its drawing room |
| [museum.md](museum.md) | `doomformer museum`: a playable map that reads a digit, and a switch that rewires one connection |
| [FLY.md](FLY.md) | The fruit-fly circuit that steers the marine |
| [fly_circuit.md](fly_circuit.md) | How the FlyWire circuits are built and where they deviate |
