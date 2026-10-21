# Side quest: a fruit-fly circuit in DOOM

The circuit code is in [`demos/fly.py`](../demos/fly.py) and [`demos/flyloop.py`](../demos/flyloop.py), and the circuits
are in [`models/fly/`](../models/fly).

The doll machinery is not specific to transformers. `doomformer/spiking.py` runs integer spiking neurons, so it can run a
circuit taken from the FlyWire connectome of *Drosophila*. Three steps, from host-driven to closed loop.

## THE FLY IS ALIVE

**[`fly_alive.mp4`](https://github.com/mottopanikeiku/doomformer/releases/download/v0.1.0/fly_alive.mp4)** (71 s): the
255-neuron FlyWire-derived circuit of `models/fly/brain.json` runs a closed sensorimotor loop **inside one live DOOM
map**. The host makes no sensory-input or motor-action decisions (`demos/flyloop.py`).

- Clock: an ACS script in the map runs every 1 ms step of the circuit. It rewinds the dolls, senses, draws the input
  spikes, releases the dolls and waits the compute period: 689 tics, the worst-case settle of 615 tics measured on a
  clock-free twin map plus 12 %. Then it holds for readout and opens a 16-tic world window. Every step runs in DOOM, and
  nothing is fast-forwarded.
- Senses: level logic in ACS decides which input groups are on from the game state: standing at the food, nukage under
  him, a dusty patch, a demon in sight within 512 units and ±45° of his heading. Each input line's spike is a Bernoulli
  draw from a Park–Miller generator written in ACS. Python replicates it to check and never supplies it.
- Muscles: each output neuron's spike doll crosses a line special aimed at the marine's TID. MN9 → `ThrustThing` toward
  the food (a lunge), Giant Fiber → backwards `ThrustThing` + `ThrustThingZ` (a takeoff), aDN1 → a grooming twitch that
  clears the dust. A probe map shows that a voodoo doll crossing a TID-targeted special moves the real player instead of
  itself and leaves the computation exact.
- Python: builds the map and calibrates the period on a separate twin game (whose state it does write). During the run it
  only advances tics and runs readout scripts, which copy values into ACS globals that nothing in the map reads. It writes
  no input, potential or action and makes no decision. Its console commands are counted, and none other than readout was
  sent. A wrapper around every public game API call found no host writes over 50 live steps. The check replays the
  circuit with `IntNet.step` and the replicated PRNG, conditioned on the sensory mask the map reports each step. That
  makes it an independent neural reference. It does not independently model the sensing rules.

| station | sensed | output spikes | what his body did |
|---|---|---|---|
| sugar on a counter | sugar, 35 steps | MN9 4 | 4 lunges, 84 units: **ate it** 34 ms after onset; the sugar input stops |
| dusty patch | antennal touch, 38 steps | aDN1 1 | grooming twitch: **dust cleared** 37 ms after onset |
| sugar + bitter in nukage | sugar + bitter, 59 steps | none | **refused**: the food stays on its counter |
| platform facing a demon pen | looming, 7 steps | Giant Fiber 1 | **takeoff** (sampled peak 75–77 units, depending on the run's sampling) into the pit 14 ms after onset; the looming stops |

169 steps, **169/169 bit-exact** (every spike and potential of 255 neurons). The PRNG state matched every step, the dolls
were at rest in every readout window, and the action lines fired exactly when the output neurons spiked. The map has
1,222 voodoo dolls and 31,615 sectors, and a step takes 728 tics and ~4–5.5 s of wall time. An independent re-run
reproduced the same table. Level design, not fly: the sensing rules, the walkway and station gates, and which special
each output drives. The input trains are Bernoulli draws with no sensory physics, and one seed is reported.
[`tests/research/test_flyloop.py`](../tests/research/test_flyloop.py) checks the PRNG replica and runs a small closed
loop in DOOM.

## THE FLY MARINE (host-driven)

In this version Python is in the loop: it walks him, generates the input spike trains and applies host thresholds to the
output spikes. It has its own feeding-only circuit and shows the circuit-building method.

**DOOM executes a FlyWire-derived integer taste-to-feeding circuit.** The marine carries 200 simulated neurons selected
from the FlyWire connectome (v783): the circuit around MN9, the proboscis motor neuron, in the setting of the whole-brain
model of [Shiu et al. 2024](https://doi.org/10.1038/s41586-024-07763-9). 64 synthetic taste-neuron input trains drive it.
It is a 200-neuron subcircuit with simplified dynamics and claims no equivalence with the 138,639-neuron Shiu model or
with any embodied fly. DOOM runs it as **spiking neurons**. Each neuron's membrane potential is a floor height, a spike
is a voodoo doll getting through a gate, and each synapse is a line it crosses that raises or lowers a target's
potential. One run of the map is **1 ms of model time**, which is far more than 1 ms of wall time (see
[Speed](HOW_IT_WORKS.md#speed)).

Scripted or done by the host:
- His walking and the motion trajectories are scripted (Python). A host threshold on the circuit's MN9 spikes selects
  whether the eat or the avoid trajectory runs.
- A label decides what he "tastes". Each snack is predefined as `sugar` or `sugar+bitter`. When his legs stop 40–140
  units short of it, the 23 sugar input lines (and, for `sugar+bitter`, the 41 bitter lines) receive Bernoulli spike
  trains (200 Hz at 1 ms steps). There is no physical gustatory sensing.
- The decision rule is the host's: he eats when **three spikes summed over the two MN9 outputs** have occurred, and
  otherwise refuses at the end of the 120 ms tasting window.
- Rest between snacks is fast-forwarded. After the stimulus DOOM keeps stepping while the reference says a neuron will
  still spike (the "afterglow", 1–5 steps in this run). The reference then runs the remaining silent leak decay (103,
  1,130 and 126 model steps), and the **host writes the resulting fixed-point potentials into the map**. That is a
  reference→engine state injection, and the bit-exact claim does not cover those steps.

Result on seed 1: sugar → the eat threshold is reached after **44 ms of model time** → he eats. Sugar + bitter → no MN9
spike in 120 ms → he refuses. Sugar → 47 ms → he eats. **219/219 DOOM steps are bit-exact** over the active and
afterglow steps (every spike and every potential of all 200 neurons, each step, vs NumPy). One map holds his corridor and
the circuit: 861 voodoo dolls, 23,914 sectors, 3,794 synapses.

How the circuit was made (`demos/fly.py`):
- Float model: a NumPy re-implementation of the Shiu LIF model on the **full** FlyWire brain (138,639 neurons) reproduces
  the qualitative on/off result of the paper. Sugar input drives MN9 (123.8/71.6 Hz left/right, 5 seeds, 1 s), and adding
  bitter input silences it (0 Hz). Absolute rates were not compared with the paper's.
- Subcircuit: from the 506 non-input neurons that fire in the float model, the 200 with the most activity-weighted
  influence on MN9. The 64 taste neurons (23 sugar + 41 bitter) are external input lines and never simulated. Synapses
  with fewer than 4 contacts are dropped.
- Integer weights: signed contact counts × a globally calibrated scale (0.392857), rounded and **clipped to ±200**, with
  no per-neuron gains. A grid search (432 candidates, 45 valid) against the float model's behaviour chose the global
  parameters (gain, threshold 100, leak 1), so the model is calibrated.
- Integer model vs the float brain: on 10 fresh seeds it keeps the on/off logic (sugar 10/10 fire, sugar + bitter 0/10,
  silence without input). It is not rate-accurate. Sugar MN9 spikes per 120 ms are 25.9/17.8 (integer, seeds 0–9;
  26.2/18.1 on seeds 16000–16009) against 12.8/6.6 in the float full-brain model, about 2–2.7× as many.
- Simplifications: linear leak, delta synapses, 1 ms steps, no refractory period, synaptic delay of one step (instead of
  1.8 ms).
- Method details and the full tables: [`fly_circuit.md`](fly_circuit.md). The circuit's parameters are also in
  `models/fly/feeding.json`'s metadata.

### A day in the life: three behaviours in one circuit

`models/fly/brain.json` is one 255-neuron FlyWire-derived circuit for three of the fly's behaviours. It has 200
externally stimulated input lines: 23 sugar, 40 bitter, 40 LC4 looming, 96 Johnston's organ, 1 P9. Its neurons are
selected jointly by influence on the three behaviours' output neurons. The integerization is the same plain one: one
global scale, inhibitory factor 0.7, no per-neuron gains. In the float full-brain model, LC4 alone drives the Giant Fiber
escape neurons (140/144 Hz at 200 Hz input; P9 is not needed). Johnston's organ input drives aDN1_right (antennal
grooming), while aDN1_left stayed silent in the reported float-model trials, so grooming uses aDN1_right only. In the
integer circuit, for 120-step trials from rest at the stated input rates, sugar, LC4 and Johnston's-organ input each
drive only their own output. Sugar + bitter, bitter alone, P9 alone and no input leave all monitored outputs silent.
Every off-diagonal output count is 0 across 10 seeds per stimulus, also on fresh seeds 7000–7009 and in `tests/`. Rates
are not matched: escape fires ~1.6× and grooming ~0.6× the float model's spikes.

One corridor, four stations, seed 1:
- sugar → 3rd MN9 spike at 36 ms → **eats**
- dust (puffs spawned around him, Johnston's-organ trains) → 2nd aDN1 spike at 68 ms → **grooms** (a head wiggle)
- sugar + bitter → MN9 silent for 120 ms → **refuses**
- a demon charges and is frozen 200 units away, LC4 trains → 3rd Giant Fiber spike at 14 ms → **takes off** (launched
  760 units back and 78 units up)

**253/253 DOOM steps bit-exact** (every spike and every potential of all 255 neurons). One map: 1,222 voodoo dolls,
32,819 sectors, 4.9 s of wall time per 1 ms step. The host-side caveats of the feeding version apply: scripted
locomotion and actions, synthetic spike trains, host thresholds, rest fast-forward.

The connectome data is CC BY-NC 4.0 (FlyWire), see [Licensing](../README.md#licensing). The engine underneath,
`doomformer/spiking.py`, runs integer leaky integrate-and-fire networks with one shared threshold and leak, binary inputs
and bounded potentials. A random 180-neuron, 5,102-synapse network is 40/40 steps exact, and its `leak` negative control
is caught.
