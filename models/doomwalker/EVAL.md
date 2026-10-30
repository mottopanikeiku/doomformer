# doomwalker: the marine that walks DEMON CANYON

DOOM named the level DEMON CANYON, `demos/levelgen.py` built it, and this model plays it in DOOM. Every number below
comes from a log in this directory unless marked otherwise.

## What it is

doomwalker is a tiny integer transformer that plays a generated DOOM level from five tokens of observation.

| part | value |
|---|---|
| model | **1 layer, d=8, 1 head, MLP 16, K=7, context 5** (`config.json`) |
| size | 1,242 integer weights, 16 DOOM tile maps, 2,150 voodoo dolls |
| observation | `observe -> [$, compass, enemy bearing, enemy distance, flag]` |
| action | argmax over the 7 action tokens `L R l r W F U` |
| scenario | a `levelgen` level, ViZDoom skill 2, pistol start, **no god mode, no notarget** |
| control | 6 game tics per decision, buttons `[forward delta, turn delta, ATTACK, USE]` |
| training levels | 170 names from `training/data/levels.json`, none of them evaluation levels |
| training data | **4,080 episodes, 390,100 decisions** |
| training | `training/train_walker.py`, 12,000 act-QAT + 4,000 full-QAT steps |

The policy lives in `demos/walker.py` and follows the contract of `policy.py`. The expert, `walker.expert_index`, is a
scripted speed-runner with exact knowledge of objects and route. It follows the compass, presses USE in door and exit
zones and shoots only when blocked. The data holds 2,040 expert episodes with action noise and 2 DAgger rounds of 1,020
episodes, 70 % driven by the integer learner and relabelled by the expert. The loss covers the action logits only.

**The compass.** The route bearing is host-provided navigation, like a compass. The harness calls
`walker.set_level(info)` with the return value of `levelgen.build_level`, and `observe` turns route and pose into the
bearing of a look-ahead waypoint, in 17 bins. Near a visible, useful pickup the compass points at the pickup instead.

The brain never sees the map and chooses every action. Enemy bearing and distance come from monsters visible on screen.
The module keeps a little host memory (route progress, previous action, last positions). That memory only feeds the
flag token and makes no decisions.

## Acceptance

Every check passes. The integer model and its DOOM run agree with the reference on every value.

| check | result | log |
|---|---|---|
| `check_limits == []` | **OK** on 5,848 unique contexts | `train_log.txt` |
| torch-int forward == `intref.forward` | **0 mismatches in 5,848 / 5,848** contexts | `train_info.json` |
| decisions that disagree with their context's majority label | 0 of 390,100 | `train_info.json: ambiguity` |
| agreement with the expert, train / dev / test | 99.97 % / 99.96 % / 99.94 % | `train_log.txt` |
| agreement on states the integer model visits in the eval | **99.99 %** | `eval_walker.txt` |
| DOOM brain, 8 real DEMON CANYON contexts | 540/540 values exact, same buttons: **ALL BIT-EXACT** | `check_walker_doom.txt` |
| DOOM brain speed, `DoomTransformer(workers=12)` | **1.16 s per decision**, first call 4.3 s | `check_walker_doom.txt` |
| full episode, seed 118 | 102/102 decisions bit-exact, 8,670 tile runs, 646,530 game tics, 221 s wall | `ouroboros_seed118.json` |

The 5,848 contexts are the 2,940 in the dataset plus every context `observe` can produce (17 compass x 86 enemy x 4
flags). They cover all dataset decisions. Zero disagreement means the expert is a function of the tokens, so the
learner can match it exactly. The 8 DOOM contexts cover turn, veer, walk, use and a visible enemy.

### Held-out levels (integer model through `intref`, seeds 9000..., same seeds for all three policies)

DEMON CANYON runs 40 episodes and each of the 10 held-out names runs 8. None is in training. Skill 2, at most 600
decisions per episode (`eval_walker.txt`, `eval_walker.json`).

| | policy | episodes | exit reached | route progress | kills | deaths | decisions / episode |
|---|---|---|---|---|---|---|---|
| **DEMON CANYON** | **integer model** | 40 | **0.85** (34/40) | **0.96** | 0.40 | 6 | 98.8 |
| | expert | 40 | 0.85 | 0.96 | 0.40 | 6 | 98.8 |
| | random | 40 | 0.00 | 0.11 | 0.03 | 40 | 108.6 |
| 10 other held-out levels | integer model | 80 | 0.57 (46/80) | 0.88 | 0.84 | 34 | 93.1 |
| | expert | 80 | 0.57 | 0.88 | 0.85 | 34 | 93.1 |
| | random | 80 | 0.00 | 0.17 | 1.02 | 80 | 166.2 |

Integer model exit rate per held-out level, 8 episodes each:

| level | exit rate |
|---|---|
| BAD ARCHEOLOGY | 1.00 |
| WHY THEY DIE | 1.00 |
| SILENCED PROGRESSION | 1.00 |
| DEATHWIND CITY | 0.88 |
| MORIOR INVICTUS | 0.62 |
| TRUE GRIT | 0.62 |
| SHIFTING SANDS | 0.50 |
| SPACE JUNK | 0.12 |
| THE BARRACKS | 0.00 |
| SOFTWARE | 0.00 |

The integer model reproduces the expert almost action for action. Per level the numbers are identical except SOFTWARE,
where one episode differs by one kill. At skill 1 (no log file) the exit rates are DEMON CANYON 0.82 and the 10 other
levels 0.76, the same as the expert.

The student cannot beat the expert, so the limits are the expert's:

* The expert is a stealth speed-runner. Gunfire wakes every monster in earshot, and every fighting variant did worse
  (expert only, no log file).
* It shoots only when blocked, hence 0.4 kills per DEMON CANYON episode.
* Deaths happen where monsters wall the route: exit-room guards, demons in corridors, hitscan guards. THE BARRACKS,
  SOFTWARE and SPACE JUNK are near-unwinnable for a pistol runner. No episode times out.
* Pickups barely matter. A larger pickup detour lowers the success rate.
* DAgger and action noise protect against drift. On the training levels, noisy learner episodes exit 43 % of the time
  and the noise-free expert 66 % (`train_log.txt`).

### Model size (all trained on the same 390,100 decisions, same recipe)

| model | weights | test agreement | exit rate DEMON CANYON / 10 held-out | DOOM wall per decision (12 workers) |
|---|---|---|---|---|
| L1 d16 H2 dm32 K7 (`train_policy.py` size) | 3,386 | 99.96 % | 0.85 / 0.57 | 2.05 s (140 tile executions) |
| **L1 d8 H1 dm16 K7 (shipped)** | **1,242** | 99.94 % | **0.85 / 0.57** | **1.16 s** (85 tile executions) |
| L1 d8 H1 dm8 K7 | 1,106 | 99.76 % | 0.85 / 0.53 | not run |

d8/dm8 loses accuracy on left turns (98.4 %). Smaller K fails on a shorter setup (195k decisions, d8 dm8): K=5 reaches
97.2 % and K=3 80.2 % train agreement.

## Seed 118 through the DOOM brain

Configuration: `--level "DEMON CANYON" --seed 118 --decisions 140 --workers 12`, 640x480 with HUD,
`make_game(seed=118)` then `new_episode()`. The integer model reaches the exit in **102 decisions** (612 game tics),
kills 2 monsters and arrives with 53 health. The DOOM brain plays the same episode bit-exact in all 102 decisions
(`ouroboros_seed118.json`, log `ouroboros_seed118_log.txt`).

`film_dc_seeds.txt` lists seeds 1..120 in this configuration. 84/120 reach the exit, fewer than in the headless eval,
because the two configurations seed the RNG at different points. A seed reproduces only in the configuration it was
found in.

| kills at exit | seeds |
|---|---|
| 0 | 46 |
| 1 | 37 (for example 102, 87, 14, 79) |
| 2 | seed 118 only |

## Training

```sh
.venv/bin/python training/train_walker.py --out models/doomwalker --d 8 --dm 16 --levels 170 --dagger 2
```

Training took 1,053 s on 4 threads + 8 collection processes. The dataset is cached in
`training/data/doomwalker_dataset.npz`.
