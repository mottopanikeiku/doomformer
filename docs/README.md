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
| [scaling.md](scaling.md) | Tile maps, many games in parallel, many machines with Ray |
| [agents.md](agents.md) | `df.Agent`: sensors, an in-map clock, the network and actuators in one map |
| [learning.md](learning.md) | `df.Perceptron`: weights stored as ceiling heights, learning executed by dolls |

## How it works

| Page | Content |
|---|---|
| [HOW_IT_WORKS.md](HOW_IT_WORKS.md) | Buses, staircases, crossbars, softmax attention, RMSNorm, the gated MLP, generation without Python, speed |
| [engine_limits.md](engine_limits.md) | Map extent (32,000 units per axis) and the 4,096-object list |

## Demos and results

| Page | Content |
|---|---|
| [DEMOS.md](DEMOS.md) | Every demo with its numbers, what DOOM computes and what Python does, seeds and selection; [Run it yourself](DEMOS.md#run-it-yourself) has the command for each research demo |
| [`demos/README.md`](../demos/README.md) | The research code and how to start each demo from a clone |
| [drawpad.md](drawpad.md) | `doomformer draw`: the map that holds an MNIST network and the room it reads from |
| [FLY.md](FLY.md) | The fruit-fly circuit that steers the marine |
| [fly_circuit.md](fly_circuit.md) | How the FlyWire circuits were made, how they compare with the float model, where they deviate |
