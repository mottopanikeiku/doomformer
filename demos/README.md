# Demos

These scripts run the larger experiments: a marine whose brain is his level, a learner, a fly circuit and more. Each
one plays real DOOM and checks every value it computes.

<p align="center"><img src="https://github.com/mottopanikeiku/doomformer/releases/download/v0.1.0/one_game.gif" alt="the marine fights in his arena while his own voodoo dolls compute his next move" width="720"></p>

`demos/` is not part of the pip package. Run each demo from a clone, from the repository root.

```sh
git clone https://github.com/mottopanikeiku/doomformer && cd doomformer
pip install -e .
python -m demos.brainlevel
```

> [!NOTE]
> Each demo compares every value DOOM exposes with an integer reference, at tolerance 0. It prints one line per
> decision or step, then the totals and what Python did.

Results and their scope are in [docs/DEMOS.md](../docs/DEMOS.md).

| command | default run | what it prints | documented run | docs |
|---|---|---|---|---|
| `python -m demos.brainlevel` | 5 decisions, 4 min | per decision: what the level sensed, the action the dolls chose, EXACT over 540 values. Then the readout-only audit | `--decisions 50 --seed 4`: 50/50 exact, 5 kills, 31 min | [his brain is the level](../docs/DEMOS.md#his-brain-is-the-level) |
| `python -m demos.aimlearn` | 2 lives, 261 decisions, 3 min | per decision: aimbot label, the dolls' action, scores, whether weights changed, EXACT over 41 values. Kills per life | `--lives 5 --seed 4`: 815/815 exact, 11 min with `--gif` | [the other demos](../docs/DEMOS.md#the-other-demos) |
| `python -m demos.flyloop` | 40 steps, 3 min | per step: input and output spikes, position, EXACT over 511 values. A station table | `--steps 900 --seed 1`: 169/169 exact, 13 min | [the fly is alive](../docs/FLY.md#the-fly-is-alive) |
| `python -m demos.lobotomy` | 3 decisions in DOOM, 1.3 min | scan counts, the blast's personality, kills, lesioned decisions EXACT in DOOM | `--decisions 26`: every observation the lesion changes, about 6 min | [lesion statistics](../docs/DEMOS.md#lesion-statistics) |
| `python -m demos.circuit` | reference tables, 1.6 min | patching recovery per head and role, the best circuit. `--doom-rows N --doom-circuits M` runs them in DOOM | `--doom-rows 4`: 8/8 exact, 16 min. `--doom-circuits 2`: 4/4 exact, 10 min | [interventions.md](../docs/interventions.md#the-induction-circuit) |
| `python -m demos.induction` | 1 held-out sequence, 42 to 310 s | every value per tile run, DOOM's answer and the rule's | `--heldout 25 --workers 3`: 25/25 exact, 2,545 s | [induction](../docs/DEMOS.md#induction-a-transformer-that-learned-an-algorithm-modelsinduction) |

Times are one run each on a shared 16-core aarch64 Linux machine. Up to three DOOM games ran, one core per game
(`OMP_NUM_THREADS=1`). An idle machine is faster. Repeat runs give the same decisions, kills and exact counts.

> [!TIP]
> Add `--gif run.gif` to any demo except `lobotomy` to get a small looping GIF of the game and its dolls.


## Reading a run

- `EXACT (N values ...)` is one decision or step whose N readouts all equal the reference. On a difference
  `brainlevel` raises, and `aimlearn`, `flyloop` and `lobotomy` print `MISMATCH` and exit 1.
- `brainlevel` and `flyloop` end with an audit line. Their game handle raises on any console command or game call
  outside the readout scripts, so `non-readout: 0` means Python did nothing else.
- In `aimlearn`, Python bins what he sees, supplies the aimbot's labels and presses the buttons the dolls chose. The
  dolls compute the scores, the action and every weight update. This is imitation of a scripted expert.
- The 5 default `brainlevel` decisions are all TURN RIGHT. The first demon is far to his right. He first fires at
  decision 22 of the documented run (`models/brainlevel_opt/*.log`).
- `brainlevel --live` keeps the demons awake while the dolls compute. He dies before his first action lands, see
  [live demons](../docs/DEMOS.md#live-demons-what-the-latency-costs-him).
- The `aimlearn` film kills (4, 24, 26, 22, 24) do not reproduce with the current compiler. It gives 4, 22, 15, 20, 16,
  all exact. The aimbot gives 9, 26, 26, 25, 26 under both. See [run it yourself](../docs/DEMOS.md#run-it-yourself).

## Options

<details><summary><b>All flags</b></summary>

| flag | meaning |
|---|---|
| `--seed S` | `brainlevel`, `aimlearn`: game RNG, default 4. `flyloop`: in-map spike-train seed, default 1. `lobotomy`: first episode seed, default 9000 |
| `--decisions N` | `brainlevel`: decisions (default 5). `aimlearn`: stop after N. `lobotomy`: observations run in DOOM (default 3) |
| `--lives N` | `aimlearn`: lives (default 2). `brainlevel`: lives the clock plays (default 5) |
| `--steps N` | `flyloop`: 1 ms model steps (default 40). The run ends when the last station is done |
| `--gif PATH` | looping GIF under 3 MB: the screen next to the map, every doll a dot. Not for `lobotomy` |
| `--period N` | `brainlevel`, `flyloop`: tics per decision or step. Default: a clock-free twin map's settle time plus a margin |
| `--opt N` | `brainlevel`: schedule level, as `doomformer.compile(opt=N)`, default 0. See the table below |
| `--live` | `brainlevel`: demons awake while the dolls compute. `--opt` defaults to 3. Not with `--lesion*` or `--gif` |
| `--driver {brain,aimbot,blank}` | `aimlearn`: who presses the buttons |
| `--workdir DIR` | keep the compiled maps (default: a temporary directory) |
| `--blasts N`, `--pellets N`, `--episodes N`, `--model NAME`, `--workers N` | `lobotomy`: random blasts (3,000), neurons per blast (7), episodes for kill counts (20), model (`doomplayer`), DOOM games (1) |
| `--minimal`, `--null`, `--names` | `lobotomy` statistics instead of the scan: minimal pacifist (`--minimal`), shuffled copies (`--null`), name model (`--names`) |
| `--max-size N`, `--sample-size N`, `--pool-size N` | `lobotomy --minimal`: exhaustive size limit above 16 units (6), random subsets per larger size (20,000), largest pooled size (5) |
| `--null-seeds N`, `--tau T`, `--stat-seed S`, `--json PATH` | shuffled copies (20), KL temperature (150), subset seed (0), JSON output |
| `--names-model NAME`, `--names-limit N` | `lobotomy --names`: name model (`doombrain`), first N validation names (0: all) |

</details>

GIF sizes: 0.22 MB for 2 `brainlevel` decisions, 0.82 MB for the full fly run, 2.37 MB for 5 learner lives. The film
used `--period 4160` for `brainlevel`.

| `brainlevel --opt`, seed 4, 50 decisions | 0 | 3 |
|---|---|---|
| compute budget, tics | 4,302 | 738 |
| wall time per decision | 54.8 s | 6.0 s |

Both levels give the same 50 actions, all exact (logs: [`models/brainlevel_opt/`](../models/brainlevel_opt/)).

## Sample output

These are the default runs, trimmed with `[...]`.

<details><summary><b><code>python -m demos.brainlevel</code></b>, 224 s.</summary>

```text
calibration (a clock-free twin of the map): the dolls settle in 3824 tics; compute budget 4302 tics; 36s
one map: his arena + his transformer + the action argmax (1,345 voodoo dolls of him, 37,410 sectors); built in 1.3s
decision   1/5  sees enemy 100-180 deg right, distant  his dolls chose TURN RIGHT (logits L/R/F [-350, 612, -347])  EXACT (540 values: tokens, every transformer scalar, logits, comparator, action; 4302 compute tics; 38.0s)  kills 0, bullets 26
[...]
5/5 decisions EXACT (2,700 values checked, tolerance 0); the level fired 0 shots and 0 demons died
Python sent 70 console commands, all readout scripts (non-readout: 0); it wrote no token, port or action and pressed no button. The level sensed the demon, the dolls computed and chose, the level turned him or fired.
wall 186s, 37.2s per decision (4302 compute tics + 32 hold + 4 action tics each)
```

</details>

<details><summary><b><code>python -m demos.aimlearn</code></b>, 180 s.</summary>

Prints the first 10 decisions, then every 25th.

```text
one map: his arena + a perceptron that learns to aim (156 voodoo dolls of him, 4579 sectors); built in 0.1s
decision    1  life 1  sees enemy 100-180 deg right  aimbot says R  his dolls chose L  scores [0, 0]  learned: weights changed  EXACT (41 values: 2 scores, the action, 38 weights; 1152 tics, 0.7s)
[...]
life 1: 4 kills (killed), 71 decisions, agrees with the aimbot 70%, 71/71 exact so far, 48s
[...]
life 2: 22 kills (out of bullets), 190 decisions, agrees with the aimbot 87%, 261/261 exact so far, 179s
261/261 decisions EXACT (41 values each, 10701 in all; tolerance 0); weights changed in 58 of them; kills per life [4, 22]
```

</details>

<details><summary><b><code>python -m demos.flyloop</code></b>, 193 s.</summary>

Prints every 10th step and every step with an output spike.

```text
one map: his level + 255 fly neurons (1,222 voodoo dolls of him, 31,615 sectors); calibration {'worst_settle': 615, 'spikes': 255, 'period': 689}; step = 4 rewind + 3 write + 689 compute + 16 hold + 16 world = 728 tics; built in 8.1s
[...]
step   32  station 1  senses sugar          2 input spikes,   7 neurons spike  output {'MN9_left': 1, 'MN9_right': 1}  pos (-1360,-6145,0) health 100  EXACT (255 potentials + 255 spikes + PRNG state; 32/32, 146s)
[...]
station 1 sugar  senses (steps) sugar 35 | output spikes MN9_left 2, MN9_right 2 | 84 units north, ATE the health bonus | 39 steps
40/40 steps EXACT (511 values each: every spike, every potential, the PRNG state; tolerance 0), PRNG 40/40, dolls at rest in the hold window 40/40, action lines vs output spikes: 0 differ; scenario complete: False
Python decisions in the loop: 0. Console commands Python sent: 444, all readout scripts (non-readout: 0); it wrote no port, potential or input and moved nothing.
```

</details>

<details><summary><b><code>python -m demos.lobotomy</code></b>, 80 s.</summary>

```text
scan of 86 observations (33s): 16 of 96 single neurons and 2531 of 3000 7-pellet blasts change at least one action; 4 of the blasts make a pacifist
the lesion: {'x0': [0, 2, 3, 5, 7, 11, 14]} -> PACIFIST (never fires); actions on the 86 observations {'R': 58, 'L': 28} (healthy {'R': 33, 'F': 19, 'L': 34})
20 episodes (the integer reference picks the actions, seeds 9000..): healthy 23.45 kills, lesioned 0.00 kills (3s)
decision  1/3  sees enemy 12-20 deg right, point blank  healthy brain: F, lesioned brain in DOOM: R  EXACT (531 values, tensors, negation twins, logits, action; 14.1s)
[...]
3/3 lesioned decisions EXACT in DOOM (1,593 values, tolerance 0); the doors changed the chosen action in 3 of them
```

</details>
