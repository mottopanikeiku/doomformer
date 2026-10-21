# models/fly: data notice

Two integer spiking networks derived from the **FlyWire connectome** (v783 public release):
- `feeding.json`: 200 neurons, 3,794 synaptic connections of the adult *Drosophila* central brain, plus 64 gustatory
  receptor neurons (23 sugar, 41 bitter) as input lines.
- `brain.json`: 255 neurons, 5,052 connections, 200 input lines (23 sugar, 40 bitter gustatory, 40 LC4 looming,
  96 Johnston's organ, 1 P9) for three behaviours: feeding (MN9), escape (Giant Fiber), antennal grooming (aDN1).

Weights are signed synapse counts × one global scale (optionally one inhibitory factor), rounded and clipped to ±200.

**Licence:** the FlyWire public release data is CC BY-NC 4.0 (https://home.flywire.ai/guidelines,
https://creativecommons.org/licenses/by-nc/4.0/). These files are adaptations: selected neurons, pruned (weak
synapses dropped), integerized. Non-commercial use only.

**Attribution:** Dorkenwald et al. 2024, *Neuronal wiring diagram of an adult brain*, Nature 634:124
(https://doi.org/10.1038/s41586-024-07558-y); Schlegel et al. 2024, *Whole-brain annotation and multi-connectome cell
typing of Drosophila*, Nature 634:139 (https://doi.org/10.1038/s41586-024-07686-5); synapses: Zheng et al. 2018,
Buhmann et al. 2021, Heinrich et al. 2018; neurotransmitter signs: Eckstein et al. 2024, Cell
(https://doi.org/10.1016/j.cell.2024.03.016); model and neuron ids: Shiu et al. 2024, *A Drosophila computational brain
model reveals sensorimotor processing*, Nature 634:210 (https://doi.org/10.1038/s41586-024-07763-9), as packaged in
https://github.com/eonsystemspbc/fly-brain.

Both files are built with `demos/fly.py`. Method, numbers and deviations from the float model:
[`docs/fly_circuit.md`](../../docs/fly_circuit.md).
