# Demos

The research code behind the "Bigger experiments" in the [top-level README](../README.md). `demos/` is not part of the pip
package: run each demo from a clone, from the repository root.

```sh
git clone https://github.com/mottopanikeiku/doomformer && cd doomformer
pip install -e .
python -m demos.brainlevel
```

Every demo runs real headless DOOM games, checks every value DOOM exposes against an integer reference with tolerance 0,
and prints one line per decision or step, then the totals and what Python did. None of them uses film code or needs
anything outside the repository. Each takes `--seed`, one count (`--decisions`, `--lives` or `--steps`) and, except the
lobotomy, `--gif PATH`.

| demo | command | what it proves | default run | documented run | docs |
|---|---|---|---|---|---|
| His brain is the level | `python -m demos.brainlevel` | The marine's policy transformer (1,345 voodoo dolls) and the argmax over his three actions run in the same map as the arena he fights in. The map's own clock senses the demon, the dolls decide, the level turns him or fires. Python only advances tics and reads, and every decision's 540 values equal the integer reference. | 5 decisions, 4 min (measured) | `--decisions 50 --seed 4`: 50/50 exact, 5 kills, 26 to 21 bullets. 31 min (measured) | [DEMOS.md](../docs/DEMOS.md#his-brain-is-the-level) |
| DOOM learns | `python -m demos.aimlearn` | A blank perceptron in his own map, with its 38 weights as ceiling heights, learns to aim from an aimbot. Both scores, the action and all 38 weights equal an integer NumPy perceptron every decision. | 2 lives, 261 decisions, 3 min (measured) | `--lives 5 --seed 4`: 815/815 exact, 11 min (measured, with `--gif`). The kills per life differ from the film, see below | [DEMOS.md](../docs/DEMOS.md#the-other-demos) |
| The fruit-fly circuit | `python -m demos.flyloop` | A 255-neuron FlyWire circuit and the marine's level share one map. An in-map clock senses, draws the input spikes, runs the circuit and holds. Every neuron's spike and potential equal the reference every step. Python sends only readout commands. | 40 steps, 3 min (measured) | `--steps 900 --seed 1`: 169/169 exact, ends by itself, 13 min (measured) | [FLY.md](../docs/FLY.md#the-fly-is-alive) |
| Lobotomy | `python -m demos.lobotomy` | A 7-pellet blast on the 1-layer `doomplayer` makes him a pacifist. The scan of all single neurons and 3,000 random blasts runs on all 86 observations. Some lesioned decisions run on voodoo dolls with the doors closed and equal the lesioned reference. | 3 decisions in DOOM, 1.3 min (measured) | `--decisions 26` runs every observation where the lesion changes the action, about 6 min (estimate: 14 s each) | [DEMOS.md](../docs/DEMOS.md#the-other-demos) |

Times are wall clock on a shared 16-core aarch64 Linux machine with up to three DOOM games and other jobs running, one core
per game (`OMP_NUM_THREADS=1`). An idle machine is faster. Each time is one run. Repeat runs here gave the same
decisions, kills and exact counts.

## Reading a run

- `EXACT (N values ...)` is one decision or step whose N DOOM readouts all equal the reference. On a difference,
  `brainlevel` raises at once; `aimlearn`, `flyloop` and `lobotomy` print `MISMATCH` and exit with code 1.
- `his brain is the level` and `flyloop` end with an audit line. Their game handle raises on every console command and
  game call outside the readout scripts, so the count of non-readout commands is the number of times Python did anything
  else. It is 0.
- `aimlearn` is a different claim. Python bins what he sees into features, supplies the aimbot's labels and presses the
  buttons the dolls chose. The dolls compute the scores, the action and every weight update. The labels come from exact
  positions the aimbot reads, so this is imitation of a scripted expert. It is no reward-driven learning.
- The decisions in the default `brainlevel` run are all turns: the first demon is far away and 100 to 180 degrees to his
  right, and he first fires at decision 22 of the documented run (seed 4).

## Options

| flag | meaning |
|---|---|
| `--seed S` | the game's own RNG (demon spawns). `brainlevel` and `aimlearn`: default 4, the documented seed. `flyloop`: the seed of the in-map spike-train generator, default 1. `lobotomy`: the first episode seed of the kill counts, default 9000 |
| `--decisions N` | `brainlevel`: decisions to run (default 5). `aimlearn`: stop after N decisions. `lobotomy`: observations to run in DOOM (default 3) |
| `--lives N` | `aimlearn`: lives to play (default 2). `brainlevel`: lives the level's clock plays (default 5) |
| `--steps N` | `flyloop`: 1 ms model steps (default 40). The run ends by itself when the last station is done |
| `--gif PATH` | a looping GIF of the run, under 3 MB: ViZDoom's screen next to the compiled map with every voodoo doll a dot (white while it moves), and one caption line. Built from `doomformer.record` pieces and read-only samples of the running game. Measured sizes: 0.22 MB for 2 `brainlevel` decisions, 0.82 MB for the full fly run, 2.37 MB for the 5-life learner |
| `--period N` | `brainlevel`, `flyloop`: tics the dolls get per decision or step. The default is measured on a clock-free twin map plus a margin; the film used 4,160 for `brainlevel` and gets the same 50 decisions |
| `--driver {brain,aimbot,blank}` | `aimlearn`: who presses the buttons. `aimbot` gives 9, 26, 26, 25, 26 kills in 5 lives of seed 4, as documented |
| `--workdir DIR` | keep the compiled maps (default: a temporary directory) |
| `--blasts N`, `--pellets N`, `--episodes N`, `--model NAME`, `--workers N` | `lobotomy`: random blasts to scan (3,000), neurons per blast (7), episodes for the kill counts (20), the model (`doomplayer`) and DOOM games for the tile run (1) |

## Measured against the documented numbers

| claim | this tree |
|---|---|
| His brain is the level, seed 4, 50 decisions: 50/50 exact, 5 kills, 5 shots, 26 to 21 bullets | the same, with the default budget (4,302 tics) and with `--period 4160`; identical decisions in both |
| Fly: 169/169 exact. Sugar 35 steps, MN9 4 spikes, ate. Dust 38 steps, aDN1 1 spike, dust cleared. Sugar and bitter 59 steps, no output, refused. Looming 7 steps, Giant Fiber 1 spike, takeoff into the pit with a sampled peak of 75 to 77 units | the same: 169/169, 35, MN9 4 and 84 units north, 38 and aDN1 1, 59 and none, 7 and Giant Fiber 1 with a peak of 75 |
| Lobotomy: 16 of 96 single neurons, 2,531 of 3,000 blasts change an action, the blast is a pacifist with 0 kills | the same: 16 of 96, 2,531 of 3,000, PACIFIST, 0.00 kills against 23.45 for the healthy brain over 20 episodes (seeds 9000 to 9019) |
| One-game learner, seed 4, film run: kills 4, 24, 26, 22, 24 and 931/931 exact; aimbot baseline 9, 26, 26, 25, 26 | `--lives 5 --seed 4`: kills 4, 22, 15, 20, 16 and 815/815 exact (634 s with `--gif`). `--driver aimbot`: 9, 26, 26, 25, 26 and 1,061/1,061 exact. The first life matches (4 kills in 71 decisions); lives 2 to 5 do not. See below |

The learner's exactness holds in every run. The film's kill counts come from a run made with the map compiler as it was on
18 October (a re-run of that snapshot here reproduces them). The current compiler lays the learner map out differently (4,579
sectors, 1,152 settle tics per decision, against 4,539 and 1,136) and the game trajectory differs from life 2 on; the public
`v0.1.0` tag gives the current numbers too. Details in [DEMOS.md](../docs/DEMOS.md#run-it-yourself).
