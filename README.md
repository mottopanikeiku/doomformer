<p align="center">
  <img src="https://github.com/mottopanikeiku/doomformer/releases/download/v0.4.0/banner_wide.png" alt="DOOMFORMER: the marine fights a horde of demons under a hellish sky" width="100%">
</p>

Doomformer compiles integer neural networks into ZDoom maps and runs them in ViZDoom. Voodoo dolls (the copies of the
player that DOOM spawns for extra player-1 starts) do the arithmetic, and sector light levels and plane heights hold the
sums. Python reads every value back from the running game and compares it with a NumPy integer reference, tolerance 0.
One value off by one and the run fails. It is slow: about 23,000 times slower than NumPy on a small MLP (engine time; 65,000
times with a timer around the whole call), and a handwritten digit takes 163 seconds of real-time DOOM in one map at the
default schedule and 8 seconds at `--opt 3`.

[Install](#install-and-run) · [Proof](#proof-walk-a-digit-doom-reads-it) · [How it works](#how-it-works) · [Films](https://mottopanikeiku.github.io/doomformer/) · [Docs](docs/README.md)

<a href="https://github.com/mottopanikeiku/doomformer/actions/workflows/tests.yml"><img src="https://github.com/mottopanikeiku/doomformer/actions/workflows/tests.yml/badge.svg" alt="tests"></a>
<a href="https://colab.research.google.com/github/mottopanikeiku/doomformer/blob/main/examples/doomformer_colab.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open in Colab"></a>
<a href="LICENSE"><img src="https://img.shields.io/badge/license-MIT-blue" alt="MIT"></a>

<p align="center">
  <img src="https://github.com/mottopanikeiku/doomformer/releases/download/v0.4.0/ladder.png" alt="A number is a crowd of voodoo dolls. A weight is a line on the floor. A sum is a light level or a height. An activation is a staircase. Attention is a ceiling. Every value is read back and compared with NumPy: tolerance 0." width="760">
</p>

## Install and run

```sh
pip install git+https://github.com/mottopanikeiku/doomformer   # Python 3.11+, Linux or macOS
doomformer hello             # 2 s: a small MLP, every value checked
doomformer draw --digit 7    # 15 s: the marine walks a 7 and the level reads it
```

`doomformer doctor` checks git, cmake, ViZDoom and the display. The first run clones and builds ACC, the ACS script
compiler (a few seconds to a minute), so `git`, `cmake`, a C compiler and network access to github.com must be there. The [Colab
notebook](https://colab.research.google.com/github/mottopanikeiku/doomformer/blob/main/examples/doomformer_colab.ipynb)
needs no setup, but you draw on a browser canvas and `doomformer mnist --image` feeds it to DOOM, with no marine.

## Proof: walk a digit, DOOM reads it

<p align="center">
  <img src="https://github.com/mottopanikeiku/doomformer/releases/download/v0.1.0/drawpad.gif" alt="DrawPad: the marine walks a 7 across a 28x28 floor canvas, presses READ, and the winning voodoo doll lights a 7 on a 7-segment display" width="720">
</p>

Holding FIRE, the marine inks a 3×3 brush of floor cells under him. READ makes the level write the 784 pixels into the
network, and 1,355 voodoo dolls in the same map compute it. With `--digit`, Python stands in for you: it presses the
marine's keys and USE once, then only advances the clock and reads.

```text
          ▄▄▄▄▄▄▄▄▄▄▄▄▄▄
       ▄████████████████
       ████▀▀     ▀████▀
       ▀▀▀        ▄███▀
                  ████
                  ███
                  ███
                  ███
                 ████
                 ███▀
                 ████
                 ▀███

DOOM reads: 7  (right)   -- the winner's doll lit it on the 7-segment display
EXACT: all 248 values DOOM computed [...] equal the NumPy integer reference (tolerance 0)
READ pass: 273 game tics = 8 s of real-time DOOM at 35 tics/s, 3 s wall [...]
```

Abridged at `[...]`; [docs/commands.md](docs/commands.md#doomformer-draw---digit-7) has the whole output. `draw` builds its
map with `--opt 3`, so a READ takes 273 game tics; at `--opt 0`, the original schedule, it takes 5,705 (163 s of real-time
DOOM). Without `--digit` the command opens a window and you walk and draw yourself; DOOM then runs in real time.

The dolls compute 784 pixels → 64 hidden units → 45 pairwise score comparisons → 10 win counts. Only the digit that wins all
nine of its comparisons gets a doll through, and the line that doll crosses starts the level script that lights the digit on
the display. Python reads back the 784 pixels, the hidden layer, every comparison and the answer and checks all 248 values
(119 values, each read with its negated twin, plus 10 outputs).

![eight MNIST test digits and what DOOM read: 7 2 1 0 4 9 6 3, all exact](https://github.com/mottopanikeiku/doomformer/releases/download/v0.1.0/mnist.png)

EXACT means DOOM equals the integer network. It says nothing about whether the digit is right. On the first 100 MNIST test
digits DOOM reads 99 correctly, and 966 of 1,000 sampled from all 10,000; every result is exact. The integer network
scores 97.16 % on all 10,000 (the float model scores 97.58 %). The 30 walked digits (0 to 9, three scripted stroke
variants each) all read correctly and exact at `--opt 3`, and the network is also trained on that stroke distribution: the
first version of this demo, a 784-32-10 network trained on MNIST alone, reads only 41.08 % of such strokes. Through real X
key events a drawn 7 read exact at the default `--opt 3`; a 7 and a 1 were tried at `--opt 0` with an earlier network.
[Details](docs/drawpad.md) · [film](https://mottopanikeiku.github.io/doomformer/#drawpad)

## What is checked

<p align="center">
  <img src="https://github.com/mottopanikeiku/doomformer/releases/download/v0.4.0/pipeline.png" alt="Pipeline: your network becomes integers and a DOOM map, ViZDoom runs it and Python reads the values back (light levels and plane heights). A NumPy integer reference, which never sees the map, computes the same values. Every value is compared with tolerance 0: EXACT or MismatchError." width="800">
</p>

Python reads sector light levels and floor and ceiling heights from the running game and compares each with an
independent NumPy integer implementation of the same network. The check is integer equality over the runs that were
made, and says nothing about inputs that were not run. The tests attack it: raising one weight line in a small compiled
map from 5 to 6 must fail by exactly +1 on exactly the inputs whose dolls cross that line, and a readout that echoes a
value Python wrote must fail (`pytest tests/test_tamper.py`, three tests, about 15 s).

`doomformer fuzz --nets 40` compiles random nets, runs them in DOOM and compares every value (306 passes, 0 mismatches,
10 s; a planted compiler bug is found). `fuzz --edge` puts weights, biases and inputs at the range limits: 15,000 nets
over ten fresh seeds, at `--opt 0`, at `--opt 3` and as tile maps, made 113,605 passes and compared 1,987,632 values with
no mismatch (the range check refused 47,686 of the inputs it drew, about 30 %), and 3,000 more with the generator widened
to plane starts near the bounds below made 22,682 passes with no mismatch. Earlier campaigns ran 2,000 random MLPs
(15,374 passes, 276,594 values), 300 spiking nets and 100 random transformers over all 8 attention, norm and MLP variants
(99 ran; one all-zero matrix is unsupported), after the scheduling fixes. `fuzz --edge` draws MLPs only. CI fuzzes 60 nets
with a fresh, printed seed on every push and a longer campaign every Monday. The fuzzer and independent runs found four
faults in this repo's own maps: two scheduling bugs and two plane limits. A plane move that starts more than 14,018
units from zero keeps its mover busy for three tics and drops the next request on that plane, and a plane that starts
above 16,000 is exact on the first pass and wrong on later ones, so DOOM disagreed with the reference on nets that
`net.check` had accepted. `check` and `compile` now refuse them: planes start inside [-15,000, 16,000] and move within
±14,018 (`doomformer verify --plane-sweep`, [docs/verification.md](docs/verification.md)).

`doomformer verify` runs every input class of a small MLP (the classes agree with every input on 20 of 20 nets, under an
assumption stated in the same page). `df.coverage` counts what is compared: on MNIST all 248 exposed values, which are
129 values the reference defines and 119 twins, and the count falls when a readout name is dropped. The map does not
expose the 903 staircase counts and 1,351 gate comparisons of the same net; they are checked through the sums they feed.
`doomformer verify --extract net.wad --net net.npz` reads the net back out of a compiled map without starting DOOM and
compares every weight, bias and threshold with the source net: the shipped MNIST net (784-64-45-10-10) comes back equal
from its map alone at `--opt 0` to 3, and so do 490 of 490 random nets at `--opt 0` and 199 of 199 at each of `--opt 1` to
3. A changed weight line shows as exactly that weight, and two last-layer threshold edits that the DOOM check passed on its
input sample show too. It shows what the map says; that DOOM runs the map as read is what the exactness checks test.
A mutation harness changes one line of a compiled map: of the 1,004 sites run (of 5,578) 297 are killed, all weight, sign
and target classes among them. Of the 707 survivors 590 match a rule (a redundant scroller, a freeze line a forward pass
never needs) that aggregate controls support, and 117 are input gaps (walking floors that only some inputs expose); none
is proven equivalent. `doomformer lint --fuzz` compared the static
schedule checker with DOOM on 5,397 maps: 0 false negatives. The method is translation validation (Pnueli, Siegel and
Singerman, 1998): a test of each compile, not a proof that the compiler is right.

Who does what:

- The dolls do every multiply-add, activation and comparison. ACS level scripts address cells, move input gates, rewind
  the dolls, sense the world and copy results out. None of the network's arithmetic runs in ACS. The toolkit's
  `df.Agent` is the one exception: it takes its argmax in ACS ([docs/agents.md](docs/agents.md)).
- `mnist`: Python writes the 784 pixel ports and decides when the dolls have stopped; the dolls pick the digit.
- `demo` and `ask`: Python writes the token ids and decides when the dolls have stopped, samples each letter from the
  logits DOOM computed and, when a model is cut into several maps, copies integers from one game to the next.
  `demo --one-map` is one map with no Python between layers; Python writes the tokens and samples the letter after each
  pass.
- `draw` and his brain is the level: Python only advances the clock and runs readout scripts (in `draw`, from READ on).
  The level supplies the inputs (the 784 pixels, the sensed tokens) and the dolls pick the digit or the action.
- `demo --one-map --loop`: Python only advances the clock and runs readout scripts. The dolls pick each letter and write
  it into the next position.
- The fly loop: Python only advances the clock and runs readout scripts, after calibrating the step period on a separate
  game that it does write. The level draws the input spikes and the output spikes fire the thrust lines.
- `--opt 3` (the default of `draw`; elsewhere `--opt 0`, which builds the bytes of v0.5.0 and, outside the window of the
  v0.5.0 barrier-height fix, of v0.4.1): an ACS script copies
  each layer's light levels into the next layer's gate heights. It adds and compares nothing, and the dolls still do the
  sums, thresholds, comparisons and the choice, but "no ACS between layers" holds at `--opt 0` to 2 only.
- `doomformer machine`: the dolls run the program counter and the registers. By default the loop, the rewind and the halt
  check are an ACS script, and a stored program is loaded by the host before the first tic. With `--clock dolls` a loop
  doll and line specials restart, rewind and stop the level, and ACS only reads values and checks that the dolls have
  stopped ([docs/machine.md](docs/machine.md)).
- `learn digits`: Python writes the pixels and the label of each example, advances the clock and reads; the dolls compute
  the scores, the argmax and the weight update ([docs/learning.md](docs/learning.md)).
- `python -m demos.brainlevel --lesion-at N` or `--lesion-when near|ahead`: the level's own clock closes the doors of the
  lesioned units before decision N, or the first time the tokens it senses satisfy the condition; Python only reads.

In the `draw`, his brain, loop, fly and machine runs a wrapper (`ReadoutOnly`) makes any other call raise, and the commands
are counted: 58 in a headless `draw` pass, 710 in the film run of his brain, 154 in the lesion run, 0 others.

The maps use ZDoom's UDMF format and ACS, so they need ZDoom (here
[ViZDoom](https://github.com/Farama-Foundation/ViZDoom) 1.3.2 with Freedoom assets), not the 1993 executable.
`net.quantization_report(x)` measures how far an integer model is from the float model it came from. EXACT does not.

## How it works

[docs/HOW_IT_WORKS.md](docs/HOW_IT_WORKS.md) maps each part of a network to a part of the map: a value to a group of
dolls, a weight to a line, an activation to a staircase, attention to a corridor of ceiling gates, ablation to a door.

<table>
<tr>
<td width="50%" valign="top"><img src="https://github.com/mottopanikeiku/doomformer/releases/download/v0.1.0/crossbar.png" alt="voodoo dolls crossing lava weight lines"><br>A matrix multiply. Each doll is one unit of an activation, and each lava line it crosses adds a weight.</td>
<td width="50%" valign="top"><img src="https://github.com/mottopanikeiku/doomformer/releases/download/v0.1.0/ride.png" alt="voodoo dolls jammed under attention gates"><br>Attention. Dolls squeeze under ceiling gates that another live value has lowered. Counting who gets through is a multiply.</td>
</tr>
</table>

The engine has sharp edges and the compiler works around them. A plane mover refuses a second request on the same plane
for as long as the first is still moving, and the later one is lost. A teleport line cancels the lines a doll crossed
earlier in that tic. A thing more than 32,767 units from a BSP node can be filed under the wrong sector, so every map
fits 32,000 units. These are ZDoom 2.8.1 behaviours measured in ViZDoom 1.3.2. The 32,000-unit rule is needed on Linux; on
macOS arm64 the wrap did not occur in the probe maps (cause not investigated), and the rule is kept everywhere. The exported
probe maps and the `hello` brain also ran in stock ZDoom 2.8.1 (53 of 53 rows match), Zandronum 3.2.1 (50 of 53: it cannot
find sector tags above 32,767) and GZDoom 4.14.2, LZDoom and UZDoom (44 of 53: planes are doubles there and the UDMF parser
refuses coordinates beyond ±32,768; the coordinate wrap stays). The `hello` brain's logged values equal the NumPy reference
in ZDoom, GZDoom, LZDoom and UZDoom, and in Zandronum once its tags are moved below 32,768. The doll-clock counter machine
(`multiply 7 9`, 464 steps, 235,567 tics) and a SUBLEQ add also ran in GZDoom, LZDoom and UZDoom with every step's pc and
registers equal to the interpreter's log; ZDoom 2.8.1 and Zandronum (tags below 32,768) ran a 50-step multiply and the
SUBLEQ add exactly, and stalled Zandronum with default tags at step 1 ([Other ports](docs/engine_limits.md#other-ports)).
`doomformer facts` re-runs 28 probes (55 cases) with PASS or FAIL per row (about 10 s), and [Engine
facts](docs/HOW_IT_WORKS.md#engine-facts) lists each finding with the probe or test behind it.

Computers inside DOOM maps exist already. [counter.wad](https://www.wad-archive.com/wad/8d0de31d8b392fc2d7fd817e933b3cf43bca799a) (2006) is a 4-bit counter built from voodoo dolls and conveyor belts,
[conway.wad](https://github.com/pwiecz/conway.wad) runs Game of Life in a Boom map, and [Danny Spencer's DOOM
calculator](https://github.com/nukep/doom-calculator) is an adding machine. Doomformer adds a compiler from integer
networks to ZDoom maps, a measured list of engine behaviours that break exactness, and a tolerance-0 check against a NumPy
reference. [docs/related_work.md](docs/related_work.md) lists what was opened and what each entry shares and does not.

## Commands

| Command | What DOOM does | Time | Skill level |
|---|---|---|---|
| `doomformer hello` | a small MLP, 5 random inputs | 2 s | I'm too young to die |
| `doomformer draw --digit 7` | the marine walks a 7, the level reads it | 15 s | |
| `doomformer mnist` | reads 4 test digits, or your own image | 1.5 to 3 min | Hey, not too rough |
| `doomformer demo` | a transformer writes a level name | 15 to 60 s | Hurt me plenty |
| `doomformer demo --nightmare` | the 2-layer model, 106 tile maps | 3.5 to 7 min | Nightmare! |
| `doomformer ask "what does iddqd do"` | answers from 189 DOOM facts | about 11 min | Ultra-Violence |
| `doomformer machine subleq all` | a SUBLEQ machine runs three programs on one map, one of them rewriting itself | about 5 min | |
| `doomformer learn digits --classes 4 --steps 300` | a blank perceptron learns 8×8 digits inside one map | about 5 min | |

`hello` runs a 4-6-5-3 integer MLP, `demo` is a 1-layer transformer trained on 136 official level names cut into 38 tile
maps, `--nightmare` is the 2-layer model (56,390 dolls), and `ask` is a 2-layer transformer trained on 189 DOOM facts.
[docs/commands.md](docs/commands.md) has every command and variant with sample output: `mnist --image` and `--tiles`,
`demo --softmax`, `--model doomnames_modern`, `--one-map`, `--loop`, `--opt`, `fuzz`, `verify`, `lint`, `tune`, `facts`,
`machine`, `learn`.

Times are from an idle 16-core arm64 Linux machine running up to 8 games at once; `hello` and `draw` use one game. With 2
games, `demo` took about 1 minute and `demo --softmax` about 3.5 minutes on a busy machine, and `ask` took 20 minutes on 3
games.

- "Not in its training names" is a low bar: 29 of 30 unselected samples of the 2-layer model at temperature 200 are new, and the default `demo`
  writes one of its 136 training names for 6 of 10 seeds (the output says which). The default seed (20) is picked.
- DEMON CANYON, the name `--nightmare` wrote, is a hand-picked sample (seed 2, temperature 150; the number of seeds tried
  before it is not recorded). The 2-layer model was trained on 11,341 of 12,934 cleaned level names.
- `ask` gets 94.7 % of its training paraphrases right and 35.4 % of new wordings of the same facts. "what does iddqd do" is a
  training question and answers GOD MODE. A question outside the 189 facts gets an answer that is exact and meaningless.

## Bigger experiments

<p align="center">
  <img src="https://github.com/mottopanikeiku/doomformer/releases/download/v0.1.0/hero.gif" alt="His brain is the level: the marine fights a demon while 1,345 dots on the right, voodoo dolls of him, run his policy transformer" width="720">
</p>

**His brain is the level.** A level script senses the nearest demon, 1,345 voodoo dolls of the marine run his policy
transformer and pick the action, and the level turns him and pulls the trigger. Python advances the clock and reads. 50 of
50 decisions exact (seed 4). A decision takes 4,338 game tics (124 s of real-time DOOM); with `--opt 3` the same 50
decisions pick the same 50 actions, all exact, in 774 tics (22 s) each. In that run his dolls turn for 21 decisions while
the first demon is far away, first choose FIRE at decision 22, and choose it five times in all; five demons die. The policy
imitates a scripted aimbot that sees exact demon positions. The demons are frozen while he thinks, and with them awake
(`--live`, below) he is dead after 5.5 s with 0 kills; the film time-lapses the thinking.
[Film](https://mottopanikeiku.github.io/doomformer/#brain) · [code](demos/brainlevel.py)

Run the experiments from a clone (`pip install -e .`, then `python -m demos.<name>`). [demos/README.md](demos/README.md)
lists each command, its time and what it prints:

- `python -m demos.brainlevel`: 5 decisions in about 4 min; `--decisions 50 --seed 4` is the 50 of 50 run, about 31 min;
  with `--opt 3` the 50 take about 5 min on a loaded machine.
- `python -m demos.brainlevel --seed 2 --decisions 12 --opt 3 --lesion 'x0:0;L0.ao:7;L0.x1:7;L0.m:0;L0.x2:0' --lesion-at 3`:
  from decision 3 the level's clock closes the doors of 5 units and the marine turns where the healthy brain fires; 12 of
  12 decisions exact, about 2 min.
- `python -m demos.brainlevel --live --lives 3`: the demons stay awake while his dolls compute; the level samples the
  sensor once per decision and the action lands 773 tics later. Over 90 lives (seeds 4, 5, 6, every decision exact) he makes
  0 kills and is dead after a median 191 tics (5.5 s), before his first action lands; the same arena frozen gives 22 kills
  a life. A Python copy of the same policy with the same delay does the same (0 kills) and with a 3-tic delay makes 3.3
  kills a life, so the delay is the cause, not the dolls. `python -m demos.brainlive sweep` prints the delay sweep.
- `python -m demos.aimlearn`: 2 lives in about 3 min; `--lives 5 --seed 4` takes about 11 min.
- `python -m demos.flyloop`: 40 steps in about 3 min; `--steps 900` runs to the end at step 169, about 13 min.
- `python -m demos.lobotomy`: the lesion scan and 3 decisions in DOOM, about 1.5 min.

<table>
<tr>
<td width="50%" valign="top"><a href="https://mottopanikeiku.github.io/doomformer/#trilogy"><img src="https://github.com/mottopanikeiku/doomformer/releases/download/v0.1.0/poster_trilogy.png" alt="A transformer names a level: film poster"></a><br><b>A transformer names a level.</b> The 2-layer name model writes DEMON CANYON in 1,690 tile runs; a Python generator builds the level.<br><sub>One hand-picked sample (seed 2). Not a corpus name.</sub></td>
<td width="50%" valign="top"><a href="https://mottopanikeiku.github.io/doomformer/#doom-plays-doom"><img src="https://github.com/mottopanikeiku/doomformer/releases/download/v0.1.0/poster_doom_beats_doom.png" alt="A marine plays the level it named: film poster"></a><br><b>A marine plays the level it named.</b> A 1-layer policy on 55 games walks that level: doors, two kills, exit. 102 decisions, all exact.<br><sub>Python gives the compass and enemy positions. Seed 118 of 120, picked.</sub></td>
</tr>
<tr>
<td width="50%" valign="top"><a href="https://mottopanikeiku.github.io/doomformer/#doom-learns"><img src="https://github.com/mottopanikeiku/doomformer/releases/download/v0.1.0/poster_one_game.png" alt="A perceptron learns inside one game: film poster"></a><br><b>A perceptron learns inside one game.</b> 38 ceiling-height weights, updated by dolls from an aimbot's labels. The current code gives 4, 22, 15, 20 and 16 kills over 5 lives, 815 of 815 decisions exact.<br><sub>The film is an earlier run on an older map layout: 4, 24, 26, 22, 24 kills, 931/931 exact, seed 4. Python bins features and gives labels.</sub></td>
<td width="50%" valign="top"><a href="https://mottopanikeiku.github.io/doomformer/#lobotomy"><img src="https://github.com/mottopanikeiku/doomformer/releases/download/v0.1.0/poster_lobotomy.png" alt="Lesions: film poster"></a><br><b>Lesions.</b> A 7-pellet shotgun blast closes 7 neurons as doors and the marine stops firing: 0 kills over 74 decisions, all exact.<br><sub>Picked from 3,000 random blasts, 4 of which were pacifists: 64 of the 11,440 seven-subsets of the 16 `x0` units are, and 5 neurons is the minimum.</sub></td>
</tr>
<tr>
<td width="50%" valign="top"><a href="https://mottopanikeiku.github.io/doomformer/#fruit-fly"><img src="https://github.com/mottopanikeiku/doomformer/releases/download/v0.1.0/poster_fly_alive.png" alt="A fruit-fly circuit: film poster"></a><br><b>A fruit-fly circuit.</b> 255 neurons derived from the FlyWire connectome drive the marine's lunge, grooming twitch and takeoff; refuse is silence. 169/169 steps exact.<br><sub>Simplified integer dynamics, synthetic input spikes, firing rates 0.6 to 2.7 times the float model's, one seed.</sub></td>
<td width="50%" valign="top"><a href="https://mottopanikeiku.github.io/doomformer/#one-map"><img src="https://github.com/mottopanikeiku/doomformer/releases/download/v0.1.0/poster_onemap_shil_raw.png" alt="A transformer in one map: film poster"></a><br><b>A transformer in one map.</b> No Python between layers: 3,548 dolls write SHIL, 2,680/2,680 exposed values exact.<br><sub>A 1-layer, d=8 model. Python samples the letter. Time-lapse.</sub></td>
</tr>
<tr>
<td width="50%" valign="top"><a href="https://mottopanikeiku.github.io/doomformer/#writes-itself"><img src="https://github.com/mottopanikeiku/doomformer/releases/download/v0.4.0/poster_selfgen.png" alt="Generation without Python: film poster"></a><br><b>Generation without Python.</b> The winner's doll sets the next token and the level rewinds itself; Python only advances the clock and reads. It wrote DAN in 5 passes (5,875 dolls).<br><sub>One model and the default prompt; the film time-lapses the thinking. ViZDoom lists at most 4,096 objects, so 1,779 dolls are not drawn ([engine limit](docs/engine_limits.md#object-list-vizdoom-lists-at-most-4096-objects)).</sub></td>
<td width="50%" valign="top"><a href="https://mottopanikeiku.github.io/doomformer/#full-brain"><img src="https://github.com/mottopanikeiku/doomformer/releases/download/v0.1.0/poster_full_brain.png" alt="The 4-layer transformer: film poster"></a><br><b>A 4-layer transformer.</b> It writes UNHOLY STRONGHOLD over 10,166 tile instances and 1.9 million game tics.<br><sub>Instances run on a pool of games, not 10,166 games at once.</sub></td>
</tr>
</table>

[The showreel](https://mottopanikeiku.github.io/doomformer/#showreel) covers all of them. [docs/DEMOS.md](docs/DEMOS.md)
has the numbers, the scope of each check and the list of picked seeds, and [docs/FLY.md](docs/FLY.md) covers the fly.

## Build your own

Compile a network into a map and let its outputs open doors, flash lights or shove the marine. This runs as pasted:

```python
import numpy as np
import doomformer as df

# 2 inputs -> [a AND b, a OR b] -> 2 outputs, all integers
net = df.MLP(
    weights=[np.array([[1, 1], [1, 1]]), np.array([[1, 0], [0, 1]])],
    biases=[np.array([-1, 0]), np.array([0, 0])],
    thresholds=[np.array([1]), np.array([1])],
    signed_input=False,
)
with df.compile(net) as brain:
    for x in ([0, 0], [1, 0], [1, 1]):
        # computed by voodoo dolls; raises df.MismatchError on any difference from net.reference
        print(x, "->", brain(np.array(x)).tolist())
    brain.record([np.array([0, 1]), np.array([1, 1])], "dolls.gif")   # film the dolls doing it
```

`df.MLP.from_torch` quantises a PyTorch model ([example 02](examples/02_pytorch_to_doom.py)), and the `world` and
`actions` hooks put rooms in the map and attach doors, lights and thrusts to the outputs ([example
03](examples/03_brain_opens_doors.py)). The same API takes spiking networks, transformers, Conv1d and Conv2d, and
learners that update their weights inside the map; `mode="tiles"` spreads a model over a pool of games (`doomformer
plan` tells you which case you have). The guide is [docs/BUILD_YOUR_OWN.md](docs/BUILD_YOUR_OWN.md), and there are
twelve one-file [examples](examples). Built something odd? [Show and
tell](https://github.com/mottopanikeiku/doomformer/issues/new/choose).

## Many games at once

One map is one game, and one game uses one core. `df.run_many` gives each input to the next idle game, checks every pass
and returns the outputs in order, on your machine or on a Ray cluster you start. Nothing is hosted. 1,000 MNIST test
digits in 14 games on one 16-core machine took 2 hours: 966 read correctly, all 1,000 exact, 5,696,000 game tics or 45
hours of real-time DOOM ([`models/mnist/doom_1000.json`](models/mnist/doom_1000.json)).

Tiles are the multi-map design: four tile maps, one per layer, in resident games, with Python copying the layer values
from one game to the next. `doomformer mnist --tiles --all --workers 6` read all 10,000 test digits in 1,390 s: 10,000
of 10,000 exact (248 values each), 9,716 right (97.16 %), DOOM's one-hot equal to NumPy's on every digit
([`models/mnist/doom_tiles_10000.json`](models/mnist/doom_tiles_10000.json)). A digit takes 216 game tics and 0.83 s
there against 5,696 tics and about 104 s in one map. `doomformer plan` predicts seconds (rms error 13 % for one map, 22
% for tiles, on 11 held-out nets; 21 % over for MNIST at `--opt 0`), prints game tics with an error band (276 ± 7.2 % for
MNIST at `--opt 3`, 272 measured) and recommends one. `doomformer bench tiles` reads MNIST at 81 digits per minute with 1
worker and 880 with 16, every pass exact, at 0.72 s of engine CPU per digit with 1 worker and 0.92 s with 16 (100 shipped
digits, twice; 16 cores).

```sh
doomformer mnist --tiles --all --workers 6         # all 10,000 digits as tile maps, about 23 min
pip install "doomformer[ray]"
ray start --head                  # other machines: ray start --address=HEAD:6379
doomformer mnist --all --ray --record mnist.json   # all 10,000 digits, one map each, on a cluster
```

Budget one core and about 1.3 GB of RAM per game (measured peak 1.24 GB).
[docs/scaling.md](docs/scaling.md#scaling-out-many-games-many-machines) has the measurements.

## What else it does

- **Schedules.** `--opt 1`, `2` and `3` (`df.compile(net, opt=N)`) shorten the one-map schedule: `doomnames_onemap` goes
  from 3,488 game tics to 1,512, 1,128 and 544, MNIST from 5,696 to 272 at `--opt 3` (100 of 100 exact and 99 right with
  the final constants, 300 of 300 before the last retune), and a DrawPad READ from 5,705 to 273 (30 of 30 walked digits
  right and exact). `--opt 2` on MNIST ran on 76 digits (3,560 tics). The ACS copy at `--opt 3` is under "Who does what".
  `doomformer lint WAD` flags plane moves that can share a tic (0 false negatives on 5,397 fuzzed maps of the compiler as
  shipped; an injected old bug was caught on 65 of 67 failing maps), `doomformer lint --net MODEL` also checks the plane
  bound, and `doomformer tune` measures the timing edges ([docs/scheduling.md](docs/scheduling.md)).
- **A counter machine and two stored-program machines.** `doomformer machine multiply 7 9` is a ROM counter machine (the
  program is the map's geometry): it prints 63 after 464 steps (39 dolls), the program counter and every register checked at
  each step; `fib 20`, `gcd`, `primes` and `pow2 40` run too. `--clock dolls` takes the ACS loop script out (a loop doll
  and line specials restart and rewind the level) and `--blind` reads only once at the end. `doomformer machine subleq all`
  runs a SUBLEQ machine with 18 memory cells of -64 to 63, operands as addresses and programs that rewrite their own
  instructions: add, 7 × 9 = 63 and a self-modifying sum run on one map (one sha256), 1,624 game tics a step, every step
  compared with a Python interpreter at tolerance 0. Changing one line of the ROM or the decode makes the run diverge at
  the predicted step. A program that writes the three cells of the instruction it is executing sets a violation latch in
  the map, which halts the level at that step (`--latch`; all 28 of 400 random programs that do so were flagged at the
  interpreter's step); without it the map computes a wrong value there. A 103-step loop runs exact on both clocks, and a
  bubble sort does not fit in 18 cells. Memory and registers are bounded plane heights, so these are finite-state machines
  and no claim that DOOM is Turing complete
  ([docs/machine.md](docs/machine.md)). `demo --one-map --loop --until-eos` lets the end-of-text winner stop the loop.
- **A learner that starts blank.** `doomformer learn digits --steps 1500`: a 10-class perceptron with 370 weights, all 0,
  learns 8×8 digits inside one map. The weights are ceiling heights; the dolls compute the scores, the argmax and the
  update, and Python writes each example's pixels and label. After each of the 1,500 steps all 455 exposed values, every
  weight included, equal a NumPy twin at tolerance 0 (about 80 min on a loaded machine). Held-out accuracy goes from 10 %
  to 71.2 % on 500 digits; a textbook perceptron without the weight clamp reaches 81.0 % on the same stream. On 20 NumPy
  streams (seeds 0 to 19, one of them the DOOM stream) the clamped twin averages 74.8 % ± 5.3. The forward corridor has 6
  gates, so a score only sees weights in [-3, 3] and larger ones keep moving unseen; that costs about 10 points on this
  stream and 3.6 on average. The update rule and the labels are Python's ([docs/learning.md](docs/learning.md)).
- **Quantisation.** `df.MLP.from_torch(model, calib_x, levels=4, calibrate="mse")` also folds BatchNorm, lowers AvgPool and
  MaxPool (MaxPool by comparisons, exact on the staircase counts) and takes GELU and SiLU as staircases; `df.qat.fit` adds
  quantisation-aware fine-tuning. On 8×8 digits at 4 levels a depth-6 MLP goes from 88.9 % with max calibration to 93.7 %
  with MSE calibration (float 95.9 %) and a CNN from 92.7 % to 95.1 % (mean of 3 seeds, [`models/quant/EVAL.md`](models/quant/EVAL.md));
  QAT is no better than MSE calibration on ReLU Linear and Conv nets, and ahead on nets with several max pools or GELU
  (BatchNorm + 2 MaxPool CNN: +4.6 points at 2 levels, +0.5 at 4). The lowered nets ran exact in DOOM on 5 digits each.
  `df.Transformer.from_torch` lowers a `torch_models.GPT` trained with the staircase recipe: a 24k-parameter level-name
  model is 5.6 % worse in perplexity than its float stage (7.75 against 7.34), ran exact in DOOM on 3 held-out names, and is
  behind a 5-gram baseline (5.84). It does not convert a transformer trained elsewhere.
- **A model that learned an algorithm.** `models/induction`, a 2-layer integer transformer, is trained on induction over
  random tokens, which no count table can solve. Held-out accuracy is 0.9995 against 0.9991 for a float GPT of the same
  size, 0.069 for a bigram table and 0.233 for a 4-gram table; on pairs never seen in training it is 0.9995 against
  0.000 for the tables. `python -m demos.induction --heldout 25` runs 25 sequences in DOOM: 25 of 25 exact, 520,749
  values. One seed per row, and a d=16 ReLU K=4 run failed to learn (log kept).
- **Interventions.** `intref.forward(interventions=...)` and DOOM runs patch a value (zero, set, mean, resample, scale, add
  or remove a direction) at a site, position and head, and every value is compared with the equally patched reference:
  24 of 24 activation-patching rows of `doomplayer` and 40 of 40 patched runs of the induction model (10 pairs per kind)
  ran exact in DOOM. In tile mode Python reads each pre-activation the dolls computed and writes the patched value into the
  next tile's ports, and the dolls compute everything downstream; exactness does not check that the patch value was the
  right one. Patching recovers the planted head of a hand-built 1-layer model, `(b + c) mod 4`, with 0 disagreements over
  384 pairs; with a backup head single patches split 0.5 and 0.5 and two circuits remain. Rows that patch the whole
  residual or a whole site at the read position are flagged and left out of the headline numbers. Steering `doomplayer`
  along a probe direction changes 39 of 86 contexts, beats random directions and does not beat shuffled-label probes. The
  lesion statistics corrected the demos above: DEMON DEATH is not a selective lesion, since 355 of the 416 single neurons
  change that sample ([docs/interventions.md](docs/interventions.md)).

## Limits

- DOOM is slow. Engine time of one pass against the NumPy reference (14 to 48 µs) is about 23,000 times for a 12→16→6
  MLP, 1.5 million times for MNIST in one map (73.2 s of engine CPU), 60,000 times for it at `--opt 3` (2.84 s) and about
  15,000 times for MNIST in tiles
  ([docs/scaling.md](docs/scaling.md#the-65000-times-as-a-range)). A one-map pass costs what the idle map costs: the
  conveyor scrollers are the whole 2.84 s of MNIST at `--opt 3` (`doomformer bench idle`), and one scroller per sector
  would buy at most 1.3 times, so 0.5 s per pass needs an engine change. With a timer around the whole call (write inputs,
  run, read back) the 12→16→6 MLP is 726 ms against 11.1 µs, about 65,000 times (30 runs,
  [Speed](docs/HOW_IT_WORKS.md#speed)), and a fly-circuit step is about 73,000 times.
- Attention is integer attention: ReLU-style in most models, softmax computed by the dolls in three. No model has
  LayerNorm (`doomnames_modern` has an integer RMSNorm), and none is a drop-in float transformer. Transformers are trained
  offline in PyTorch with quantisation-aware training (`training/`); `MLP.from_torch` converts MLPs and CNNs, and
  `Transformer.from_torch` only the `torch_models.GPT` module trained with that recipe.
- The compiler is validated in ViZDoom 1.3.2. The probe maps, the `hello` brain and the doll-clock counter machine ran in
  other engines (stock ZDoom, Zandronum, GZDoom, LZDoom, UZDoom); no network larger than `hello` did, and in the GZDoom
  family plane arithmetic differs, so the plane limits of this repo are ZDoom's. The compiler emits sector tags above 32,767,
  which Zandronum cannot find.
- The checks cover the nets and inputs that were run. The fuzzers do not cover learning maps, more than 4,096 objects, or
  transformers in tile mode ([Not covered](docs/verification.md#not-covered)).
- `--one-map` runs a smaller model than `demo` (1 layer, d=8, 6 positions), because the 136-name model needs 116,968 dolls
  and does not fit one map.
- The marine's policies imitate a scripted aimbot that sees exact positions. Python bins them into 86 possible observations
  and supplies the labels. There is no reward-driven RL and no learning from pixels.
- Showcased seeds, lesions and names are picked. [Seeds and selection](docs/DEMOS.md#seeds-and-selection) lists each pick.

## What's in the box

| Path | What it holds |
|---|---|
| [`doomformer/`](doomformer) | the toolkit: map compiler, DOOM runtime, exact checks, CLI |
| [`demos/`](demos) | research code: his brain is the level, the walker and level generator, the fly circuit, the learning arena |
| [`training/`](training) | quantisation-aware training for every model, with the corpora and their sources |
| [`models/`](models) | every trained model with its training log and evaluation |
| [`examples/`](examples) | twelve one-file examples, most under 15 s |
| [`tests/`](tests) | 465 tests with torch installed (fewer without it), 172 of them on real headless DOOM (4 run only with `DOOMFORMER_HEAVY=1` or take minutes); CI runs the suite on Linux and macOS |
| [`docs/`](docs/README.md) | how it works, every demo with its numbers, the engine facts, the toolkit guide, [every command with sample output](docs/commands.md) |
| [`CHANGELOG.md`](CHANGELOG.md) | what each release added |

## Licensing

Code and docs: [MIT](LICENSE). Data keeps its own licence ([THIRD_PARTY.md](THIRD_PARTY.md)): the level-name lists are Doom
Wiki (CC BY-SA 4.0) and id Software's source (GPL-2.0), the 100 sample MNIST digits are CC BY-SA 3.0, and the fruit-fly
circuits in `models/fly` are derived from the FlyWire connectome (CC BY-NC 4.0, non-commercial). Freedoom graphics are
BSD-licensed. DOOM is a trademark of id Software / ZeniMax; this project is not affiliated with them.
