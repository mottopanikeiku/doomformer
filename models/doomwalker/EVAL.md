# doomwalker: the marine that walks DEMON CANYON

*DOOM named a level (DEMON CANYON), DOOM built it (`demos/levelgen.py`), now DOOM plays it.*
Every number below comes from a log in this directory unless marked otherwise.

## What it is

| | |
|---|---|
| model | **1 layer, d=8, 1 head, MLP 16, K=7, context 5** (`config.json`), 1,242 integer weights, 16 DOOM tile maps (2,150 voodoo dolls) |
| policy interface | `demos/walker.py` (same contract as `policy.py`): `observe -> [$, compass, enemy bearing, enemy distance, flag]`, `act` = argmax over the 7 action tokens `L R l r W F U` |
| scenario | a `levelgen` level, ViZDoom skill 2, pistol start, **no god mode, no notarget**, 6 game tics per decision, buttons `[forward delta, turn delta, ATTACK, USE]` |
| expert | `walker.expert_index`: a scripted speed-runner with exact info (objects, route). Follows the compass, turns only as far as needed (keeps walking while it veers), presses USE in door/exit zones (every other decision, USE is edge-triggered), shoots only when blocked |
| training data | 170 level names from `training/data/levels.json` (never DEMON CANYON, never the 10 evaluation names), **4,080 episodes, 390,100 decisions**: 2,040 expert episodes with action noise (random actions held 1-3 decisions, eps 0-0.2) + 2 DAgger rounds (1,020 episodes each, 70 % driven by the integer learner, expert relabels) |
| training | `training/train_walker.py` (QAT recipe of `train_policy.py`, 12,000 act-QAT + 4,000 full-QAT steps, loss on the action logits only) |

**The compass.** The route bearing is host-provided navigation info, like a compass: the harness calls
`walker.set_level(info)` (`levelgen.build_level`'s return value) and `observe` turns the route polyline and the marine's
pose into the bearing of a look-ahead waypoint, in 17 bins. Near a visible, useful pickup (armor/health within 120
units, weapons within 700, no monster within 300) the compass points at the pickup instead. The brain never sees the map
and chooses every action. Enemy bearing/distance come from ViZDoom objects restricted to monsters in the label buffer
(= visible on screen). The module keeps a little host memory (progress along the route, previous action, last positions).
It only feeds the observation (flag `1/2`: door-or-exit zone and whether the previous action was USE; flag `3`:
blocked) and never makes decisions.

## Acceptance

| check | result |
|---|---|
| `check_limits == []` | **OK** on 5,848 unique contexts = every context in the dataset (2,940) plus every context `observe` can produce (17 compass x 86 enemy x 4 flags), `train_log.txt` |
| torch-int forward == `intref.forward` | **0 mismatches in 5,848 / 5,848** contexts (covering all 390,100 dataset decisions) |
| expert is a function of the tokens | 0 of 390,100 decisions disagree with the majority label of their context (`train_info.json: ambiguity`), so the learner can match it exactly |
| agreement with the expert (intref) | train 99.97 %, dev 99.96 %, test (held-out episodes) 99.94 %; on the states the integer model visits in the eval: **99.99 %** |
| DOOM brain on real in-game contexts | `check_walker_doom.txt`: 8 real DEMON CANYON contexts (turn, veer, walk, use, enemy visible), every tensor 10/10 exact (540/540 values), logits equal to `intref`, same buttons: **ALL BIT-EXACT**, `DoomTransformer(workers=12)`: **1.16 s per decision** (first call 4.3 s warm-up) |
| full episode through DOOM | `ouroboros_seed118.json`: seed 118, 102 decisions, 102/102 bit-exact, 8,670 tile runs, 646,530 game tics, 221 s wall incl. video |

### Held-out levels (integer model through `intref`, seeds 9000..., same seeds for all three policies)

`DEMON CANYON` (40 episodes) and the 10 held-out names (8 episodes each), none of them in training, skill 2,
at most 600 decisions per episode (`eval_walker.txt`, `eval_walker.json`):

| | policy | episodes | exit reached | route progress | kills | deaths | decisions / episode |
|---|---|---|---|---|---|---|---|
| **DEMON CANYON** | **integer model** | 40 | **0.85** (34/40) | **0.96** | 0.40 | 6 | 98.8 |
| | expert | 40 | 0.85 | 0.96 | 0.40 | 6 | 98.8 |
| | random | 40 | 0.00 | 0.11 | 0.03 | 40 | 108.6 |
| 10 other held-out levels | integer model | 80 | 0.57 (46/80) | 0.88 | 0.84 | 34 | 93.1 |
| | expert | 80 | 0.57 | 0.88 | 0.85 | 34 | 93.1 |
| | random | 80 | 0.00 | 0.17 | 1.02 | 80 | 166.2 |

Per level (integer model exit rate, 8 episodes): BAD ARCHEOLOGY 1.00, WHY THEY DIE 1.00, SILENCED PROGRESSION 1.00,
DEATHWIND CITY 0.88, MORIOR INVICTUS 0.62, TRUE GRIT 0.62, SHIFTING SANDS 0.50, SPACE JUNK 0.12, THE BARRACKS 0.00,
SOFTWARE 0.00 (full table in `eval_walker.txt`). The integer model reproduces the expert almost action for action: the
per-level numbers are identical except SOFTWARE (one episode differs by one kill).

At skill 1 (damage halved, same model, same seeds; no log file): DEMON CANYON 0.82 (0.82 expert),
10 other held-out levels 0.76 (0.76 expert).

The model reaches the DEMON CANYON exit in 34/40 episodes (85 %), the same as its expert. The student cannot beat the
expert, because the expert sees only what the tokens contain, so the limits are the expert's:

* The expert is a stealth speed-runner. Gunfire wakes every monster in earshot, and every variant that stops to fight did
  worse (expert only, 40 held-in levels + DEMON CANYON; no log file). Fighting every visible monster inside 45 degrees
  reached the DEMON CANYON exit 0 % of the time, opportunistic fire while walking 38 %, fighting anything within 40-64
  units 30-87 % (never better than 87 %). It shoots only when blocked (pushing without moving), hence 0.4 kills per
  DEMON CANYON episode and 0.84 on the other levels. On the seed-118 episode below, 1-2 kills is typical, mostly when a
  monster blocks the way.
* Deaths happen where monsters wall the route: the hell knight / baron / cacodemon guards of the exit room, demons in
  corridors, and levels full of hitscan guards. Chaingunners and shotgunners on THE BARRACKS shoot a runner at range
  (0/8), and SOFTWARE is 0/8. With SPACE JUNK (0.12), these three of the ten held-out levels are near-unwinnable for a
  pistol runner. No episode times out.
* Pickups barely matter: a larger pickup detour lowered the success rate (detours cost time and drag monsters along).
* DAgger and action noise therefore mostly protect against drift: learner episodes with noise exit 43 % of the time, the
  noise-free expert 66 %, on the training levels (`train_log.txt`).

### Model size (all trained on the same 390,100 decisions, same recipe)

| model | weights | test agreement | exit rate DEMON CANYON / 10 held-out | DOOM wall per decision (12 workers) |
|---|---|---|---|---|
| L1 d16 H2 dm32 K7 (`train_policy.py` size) | 3,386 | 99.96 % | 0.85 / 0.57 | 2.05 s (140 tile executions) |
| **L1 d8 H1 dm16 K7 (shipped)** | **1,242** | 99.94 % | **0.85 / 0.57** | **1.16 s** (85 tile executions) |
| L1 d8 H1 dm8 K7 | 1,106 | 99.76 % | 0.85 / 0.53 | not run |

Smaller K failed on a shorter run (195k decisions, 3,000 + 1,000 steps, d8 dm8): K=5 97.2 %, K=3 80.2 % train agreement.
The shipped model matches the expert's closed-loop numbers exactly; d8/dm8 loses accuracy on left turns (98.4 %).

## Seed 118 through the DOOM brain

Configuration: `--level "DEMON CANYON" --seed 118 --decisions 140 --workers 12`, 640x480 with HUD,
`make_game(seed=118)` then `new_episode()`. The integer model reaches the exit in **102 decisions**
(612 game tics), kills 2 monsters on the way and arrives with 53 health. The same episode played with the DOOM brain:
102/102 decisions bit-exact, exit reached, 2 kills, 221 s wall (`ouroboros_seed118.json`, log
`ouroboros_seed118_log.txt`). The first call of the DOOM brain takes ~4 s (warm-up). At the steady 1.16 s per decision,
the 102 decisions cost about two minutes of brain time.

`film_dc_seeds.txt` lists every seed 1..120 in this configuration. 84/120 reach the exit (70 %, lower than the 85 % of
the headless eval). The two configurations produce different trajectories from the same seed, probably because monsters become
visible at different times at 640x480 and the RNG stream differs. 46 of them
finish with 0 kills, 37 with 1 kill, **only seed 118 with 2 kills**. Alternatives with a single kill: seed 102 (95 decisions, health
94), 87 (97, 78), 14 (98, 62), 79 (98, 60). A seed reproduces only in the configuration it was found in:
`make_game` calls `set_seed` before `init`, like the harness, while the headless `play_episode` seeds before `new_episode` and
plays a different stream.

## Training

```sh
.venv/bin/python training/train_walker.py --out models/doomwalker --d 8 --dm 16 --levels 170 --dagger 2
```

Training took 1,053 s on 4 threads + 8 collection processes. The dataset is cached in
`training/data/doomwalker_dataset.npz`.
