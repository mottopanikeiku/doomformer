# Docs

Start with [Build your own](BUILD_YOUR_OWN.md) to compile a network, or [How DOOM computes](HOW_IT_WORKS.md) for the mechanism. Runnable scripts are in [`examples/`](../examples), release notes in the [changelog](../CHANGELOG.md).

## Using the toolkit

| Page | Covers |
|---|---|
| [Commands](commands.md) | CLI options and sample output |
| [Build your own](BUILD_YOUR_OWN.md) | Networks, `df.compile`, world hooks and actions |
| [Model import](interop.md) | PyTorch and NumPy conversion, calibration, quantisation-aware training |
| [Recording](recording.md) | `brain.record(...)`: film a run as GIF or MP4 |
| [Scaling](scaling.md) | Tile maps, parallel games, Ray, the cost model behind `plan` |
| [Agents](agents.md) | `df.Agent`: sensors, network, clock and actuators in one map |
| [Learning](learning.md) | `doomformer learn digits` and `df.Perceptron` |
| [Scheduling](scheduling.md) | `opt` levels, waves, `doomformer lint` |
| [Tuning](tuning.md) | `doomformer tune`: timing constants and where they fail |
| [Machines](machine.md) | `df.Machine` and `df.SubleqMachine`: counter and SUBLEQ machines in one map |
| [Interventions](interventions.md) | `df.intervene` and `df.patch_recovery`: ablation and activation patching |

## How it works

| Page | Covers |
|---|---|
| [How DOOM computes](HOW_IT_WORKS.md) | Gates, crossbars, attention and engine arithmetic |
| [Engine limits](engine_limits.md) | Coordinates, object readout and other DOOM ports |
| [Related work](related_work.md) | Earlier map computers, integer transformers, translation validation |
| [Verification](verification.md) | Exact comparison, fuzzing, map mutation, input coverage, static extraction |

## Demos and results

| Page | Covers |
|---|---|
| [Demos](DEMOS.md) | Results, seeds and commands |
| [`demos/README.md`](../demos/README.md) | Research demos and entry points |
| [DrawPad](drawpad.md) | `doomformer draw`: the MNIST map and its drawing room |
| [Museum](museum.md) | `doomformer museum`: a playable map that reads a digit, and a switch that rewires one connection |
| [Fly circuit](FLY.md) | A FlyWire-derived spiking circuit that steers the marine |
| [FlyWire circuits](fly_circuit.md) | How the circuits were built and where they differ from the float model |
