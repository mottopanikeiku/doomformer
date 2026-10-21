# Autonomous agents: the brain and the world in one map

`Agent` compiles an integer MLP or Transformer, an ACS sensor/decision clock, and
actuators into one map. Once the live game starts, Python **only advances tics and
runs allowlisted readout scripts**. It never supplies network inputs, chooses an
action for the game, presses buttons, resets the live network or starts its clock.

```python
from doomformer.agent import Agent, Actuator
from doomformer import sense
from doomformer.kit import MLP

net = MLP([[[1], [-1], [0]]], [[-8, 8, 1]], [list(range(1, 18))],
          signed_input=False)

def world(m):
    m.rect(-1024, -3072, 0, -2048, floor=0, ceil=128, tag=20)
    m.thing(-112, -2560, 3002, id=901)  # sensed monsters share this TID
    m.thing(-512, -2560, 1, angle=45)   # final player start is the marine
    return ""

with Agent(net, sensors=[sense.bearing()],
           actuators={0: Actuator.turn(640), 1: Actuator.turn(-640),
                      2: Actuator.fire()},
           world=world, world_tics=4) as agent:
    records = agent.run(32)
    print(records[-1], agent.audit())
```

This snippet uses ordinary-health demons. The acceptance turret in
[`07_autonomous_turret.py`](../examples/07_autonomous_turret.py) sets its demon's
health to **one**, a level-design choice.
[`08_your_own_agent.py`](../examples/08_your_own_agent.py) is a complete, small,
copyable two-sensor template.

## Who computes what?

| Component | Responsibility |
|---|---|
| Python, before the live game | Build geometry and ACS; measure timing on a separate twin; construct the independent integer reference |
| DOOM engine queries | Sample actor health, ammo, positions, angles, distance, visibility and sector membership |
| ACS level scripts | Iterate tagged actors, select the nearest live monster, bin sensor values, write input ports, rewind/release dolls, wait, read output sectors, select argmax, apply the selected actuator |
| DOOM dolls and geometry | All network arithmetic, thresholds, hidden values, Transformer embedding/attention/MLP/logits |
| Python, during the live game | Advance tics, read instrumented samples and all exposed network values, independently replicate sensor selection/binning and network computation, compare with tolerance **zero**, reject mismatches |

Room layout, spawn positions, monster health, optional freezing, sensing rules,
the clock, pistol, argmax and actuators are **level design / ACS**. The network
computes only the action **scores**. The examples are hand-written integer
networks; none is a trained policy.

The audit covers the live autonomous game. Calibration uses an isolated,
clock-free twin and intentionally writes its ports; those writes do not reach the
live map and are not included in its command count. The twin closes before the
live game opens (at most one DOOM process per Agent).

## Inputs and exact sensor semantics

A sensor is a `sense.Sensor` built by one of these constructors. Sensors sample
once, before that decision's network computation. Health and position may change
while computing, especially when `freeze_monsters=False`: verification always
uses the **instrumented sample-time values**, not a later world snapshot.

| Constructor | Integer result |
|---|---|
| `bearing(tokens=False)` | Nearest live monster's relative-bearing bin 0..16; no live monster = 17 |
| `distance(tokens=False)` | Nearest live monster's distance bin 0..4, edges 100/200/350/550 map units; no monster = 4 |
| `health(scale=1, offset=0)` | `trunc_toward_zero((health - offset) / scale)` |
| `ammo(scale=1, offset=0)` | Same expression on the integer `Clip` inventory |
| `x_bin(width=64, origin=0)` | `trunc_toward_zero((floor(X) - origin) / width)` |
| `y_bin(width=64, origin=0)` | Same on Y; floor is the arithmetic right shift of the raw 16.16 coordinate |
| `monster_count()` | Count of actors with `monster_tid` and sampled health > 0 |
| `sector_flag(tag)` | `int(ThingCountSector(T_NONE, player_tid, tag) != 0)` |
| `line_of_sight()` | `CheckSight(player_tid, nearest_tid, 0)` converted to 0/1; no live monster = 0 |
| `constant(value)` | The specified integer, e.g. Transformer BOS token 0 |

Nearest is selected by `(VectorLength(dx,dy), signed relative VectorAngle)`:
smaller distance, then smaller signed bearing; exact equality retains the first
actor in the engine iterator. Dead actors are recorded but not selected.
Distances are raw 16.16 engine query results; each distance edge is inclusive in
the higher bin.

Bearing edges in degrees are
`-100,-60,-35,-20,-12,-7,-4,-2,2,4,7,12,20,35,60,100`.
The implementation reuses `brainacs.sensor_acs`'s integer bin helpers. Outside a
12-angle-unit boundary margin it uses the engine's signed 16-bit `VectorAngle`;
inside it uses a fixed-point cross product with the edge, using the same Taylor
rotation as brainacs. The wrap boundary is decided by the sign of the rotated Y
coordinate. These are **fixed-point ACS semantics**, an approximation of
real-number trigonometry. Python independently replicates the Taylor products,
selection, edge tests and bins. The exact integer formulas are in
`sense.nearest_replica` and `brainacs.sensor_acs`.

`tokens=True` returns the existing `policy.VOCAB` token indexes instead of raw
bin indexes: bearing starts at `VOCAB.index('a')`, distance at
`VOCAB.index('0')`; absent bearing is `VOCAB.index('?')`.

The sample records each actor's relative X/Y, engine `VectorLength` and
`VectorAngle` results, `CheckSight`, and health. Python checks selection and
binning **from those primitive queries**. It does not independently reimplement
DOOM's collision visibility, sector lookup, health system, distance function or
angle lookup. Health/ammo/sector queries are independently transformed but the
underlying query is trusted as game state, just as a network readout is trusted
as an observation of the engine. Kills are engine statistics, not network outputs.

## MLP and Transformer wiring

MLPs require one sensor per input. The usual MLP staircase activation contract
still applies: an input bin is not automatically one-hot encoded.

Transformers accept one to `net.T` sensor tokens. For the policy vocabulary a
common wiring is:

```python
sensors = [sense.constant(0), sense.bearing(tokens=True),
           sense.distance(tokens=True)]
actuators = {'L': Actuator.turn(640), 'R': Actuator.turn(-640),
             'F': Actuator.fire()}
```

Vocabulary characters select logits at the last supplied sensor position.
Integers mean an output index (`yN` for an MLP; last-position vocabulary index for
a Transformer). Explicit `yN` and `logits:t:j` names are also accepted. Only final
network outputs are action scores. Unused Transformer positions are padded with
zero **in ACS**, and Python verifies **all exposed positions including padding**.
Token sensors must produce valid vocabulary indexes, not arbitrary health values.

## Action semantics and timing

`actuators` is an insertion-ordered mapping of chosen output to one actuator or a
nonempty sequence. ACS selects **the largest score among these entries only**.
Ties choose the **first mapping entry**, not the lowest output index. All the
winning entry's actuators execute once in listed order.

* `Actuator.turn(delta)`: add a signed integer number of 1/65536 turns to the
  player's current angle, modulo 65536. Positive turns left. `640` is 3.515625°.
* `Actuator.fire()`: give the ScriptFireToken. ScriptPistol fires from the current
  facing with no horizontal random spread, consuming Clip ammo. Normal engine
  weapon state/recovery, vertical aiming and hit/damage behavior still apply.
  A token does not guarantee a hit, or an immediate shot if the weapon is not
  ready. The token inventory holds one, so shots do not queue.
* `Actuator.thrust(angle, force)`: explicit player-TID thrust; angle is integer
  degrees rounded to 256 engine directions, force is 0..255, ordinary speed cap.
* `Actuator.door(tag, speed=16)`: `Door_Open`, tag 3..999, opening speed 1..255.
* `Actuator.script(number, *args)`: `ACS_ExecuteAlways` for script 200..29999,
  up to three signed integer arguments, with the real player as activator.

The loop is: rewind → sample → write ports → compute → argmax → actuate →
`world_tics` awake tics → hold for readout. `run(steps)` returns after those
complete actions; each record includes pre-action inputs, network values,
selected action, scores, checked count, sample instrumentation and post-action
kill count. Repeated `run()` calls continue the same game. Zero steps starts the
game but does not advance a decision.

Default `freeze_monsters=True` makes actors with `monster_tid` dormant during
compute/readout and wakes them for the action interval. Dormant monsters cannot
be shot, so give a firing action enough awake tics for its weapon state.
`world_tics=20` is the default; the tiny examples use 4 after a 70-tic boot.
The player is **not** frozen, nor are arbitrary world scripts paused.

With `period=None`, the clock-free twin opens both input polarities, runs its
layer clocks, and measures settling. The live compute period is that measurement
plus `max(64, ceil(0.2 * settled_tics))`. This calibration does not bound the
settling time of arbitrary custom worlds, but every live result is still
verified. An explicit `period` skips the measurement; a too-short period can
raise `MismatchError`. World hooks execute for both builds and should be
deterministic and not change network geometry based on invocation count.

## Readout-only audit and limits

The live game's `ReadoutOnly` wrapper accepts only the exact generated network
readout commands, Agent status, and sample pages. It rejects port writes,
resets, arbitrary console commands, `set_action`, `make_action`, `new_episode`
and other mutation APIs. Attempts are counted before rejection. `audit()`
returns `commands`, `forbidden`, and `non_readout_commands`. Tests also wrap the
game with an independent outer allowlist, so the audit does not rest on an
internally maintained counter alone.

ACS argmax reads sector expressions directly; it never borrows the USER slots
Python uses for readout. Thus status polling during computation cannot race the
action selector. Readout holds budget for all sensor and network pages; step and
phase are checked again after reading. Do not advance the game independently
between `run()` calls: missing an autonomous decision is an error, not silently
skipped verification.

* World hooks use the normal toolkit layout/last-player checks. Tags 3..999 and
  world thing TIDs 1..9999 are available, except Agent's player and monster IDs.
* Defaults: `player_tid=900`, `monster_tid=901`, `max_monsters=64`. The capacity
  counts corpses as well as live actors. Overflow raises `RangeError`; worlds
  that respawn indefinitely must remove old corpses or increase the capacity.
* Agent reserves scripts **30000..30003**, `ag_*`, `AG_PLAYER`, and temporary
  TIDs **50000..50000+max_monsters-1**. Do not create actors with those temporary
  TIDs or reuse them for world logic. Avoid networks whose doll/home TID ranges
  overlap these reserved temporary IDs; this is a small/medium-map API.
* Input port values must remain within ±12,000; Transformer tokens must be
  within the vocabulary. Out-of-range samples stop the ACS clock and raise.
  Existing network partial-sum/gate restrictions still apply.
* Large neural networks still have the ordinary map-size and runtime costs.
  Whole zero-weight layers remain unsupported by the underlying pure compiler
  (it reports a layer with no dolls).
* A custom world/script/`Actuator(source)` is trusted level code, not sandboxed
  ACS. Do not overwrite sensor arrays, network ports, or USER variables.
* Exactness says nothing about inference speed, aiming skill, or the
  correctness of engine physics.

## Measured acceptance

Measured with ViZDoom 1.3.2, `OMP_NUM_THREADS=1`, `nice -n 10`:

```text
.venv/bin/python -m pytest -q tests/test_agent.py
10 passed in 24.42s

.venv/bin/python examples/07_autonomous_turret.py
32 steps, every one checked exact (a mismatch raises); kills=1; non-readout commands=0; period=232; calibration={'settled_tics': 168, 'margin': 64}
```

The turret command completed in 2.4 seconds. Example 08 returned eight exact
steps, one kill and zero non-readout commands. These figures cover
`tests/test_agent.py` and examples 07 and 08 only. The tests cover all sensors,
nearest selection and visibility through an open/closed sector, every built-in
actuator, MLP hidden values, Transformer padded positions, insertion-order ties,
sequential runs, independent outer API auditing, rejected writes, deliberately
corrupted readback and an undersized compute period.
