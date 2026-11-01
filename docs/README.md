# Documentation

Start with [Build your own](BUILD_YOUR_OWN.md) to compile a network, or [How DOOM computes](HOW_IT_WORKS.md) for the
mechanism. Runnable scripts are in [`examples/`](../examples); release changes are in the [changelog](../CHANGELOG.md).

## Using the toolkit

| Page | Content |
|---|---|
| [Commands](commands.md) | CLI options and sample output |
| [Build your own](BUILD_YOUR_OWN.md) | Networks, `df.compile`, world hooks and actions |
| [Model import](interop.md) | PyTorch and NumPy conversion, calibration and quantisation-aware training |
| [recording.md](recording.md) | `brain.record(...)`: film a run as GIF or MP4 |
| [scaling.md](scaling.md) | Tile maps, parallel games, Ray, the cost model behind `plan` |
| [Agents](agents.md) | `df.Agent`: sensors, network, clock and actuators in one map |
| [learning.md](learning.md) | `doomformer learn digits` and `df.Perceptron` |
| [scheduling.md](scheduling.md) | `opt` levels, waves, `doomformer lint` |
| [tuning.md](tuning.md) | `doomformer tune`: the timing constants and their failure edges |
| [machine.md](machine.md) | `df.Machine` and `df.SubleqMachine`: counter and SUBLEQ machines in one map |
| [interventions.md](interventions.md) | `df.intervene` and `df.patch_recovery`: ablation and activation patching in DOOM |

## How it works

| Page | Content |
|---|---|
| [How DOOM computes](HOW_IT_WORKS.md) | Gates, crossbars, attention and engine arithmetic |
| [Engine limits](engine_limits.md) | Coordinates, object readout and other DOOM ports |
| [Related work](related_work.md) | Earlier map computers, integer transformers and translation validation |
| [Verification](verification.md) | Exact comparison, fuzzing, map mutation, input coverage and static extraction |

## Demos and results

| Page | Content |
|---|---|
| [Demos](DEMOS.md) | Results, [sample selection](DEMOS.md#seeds-and-selection) and [commands](DEMOS.md#run-it-yourself) |
| [`demos/README.md`](../demos/README.md) | Research demos and entry points |
| [drawpad.md](drawpad.md) | `doomformer draw`: the MNIST map and its drawing room |
| [museum.md](museum.md) | `doomformer museum`: a playable map that reads a digit, and a switch that rewires one connection |
| [FLY.md](FLY.md) | A FlyWire-derived spiking circuit that steers the marine |
| [fly_circuit.md](fly_circuit.md) | Circuit construction and simplifications |
