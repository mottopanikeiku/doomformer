# Commands

Every command prints what DOOM computed and the result of the exact check. The README has the short table; this page has
the commands one by one with real sample output. Output is from the commands as shipped; where lines are shortened a
`[...]` marks the cut, and progress lines (`... 15 s, 108 tile runs`) are left out. The `demo` sample ran on 4 games,
the `--nightmare`, `--one-map` and `ask` samples on 6, 1 and 6 games (see the line `in N parallel DOOM games`; the
`model:` lines of `--nightmare` and `--one-map` and the `ask` banner were updated by hand to the current wording of the
code).

Times are from an idle 16-core arm64 Linux machine running up to 8 games at once. `hello`, `draw` and the one-map commands
use one game. With fewer games the tile-map commands take proportionally longer: on 2 games `demo` took about 1 minute and
`demo --softmax` about 3.5 minutes on a busy machine, and `ask` took 20 minutes on 3 games (13 for the first letter).

| command | what it runs | time |
|---|---|---|
| [`doomformer hello`](#doomformer-hello) | a 4-6-5-3 integer MLP on 5 random inputs | 2 s |
| [`doomformer draw --digit 7`](#doomformer-draw---digit-7) | the marine walks a 7 on the DrawPad floor, the level reads it | 1.5 to 2 min |
| [`doomformer mnist`](#doomformer-mnist) | reads MNIST test digits, or your own with `--image`, the whole net in ONE map | 1.5 to 3 min per digit |
| `doomformer mnist --tiles` | the same net as 4 tile maps in resident games, Python copying the layer values between them (the multi-map design); `--all` reads the 10,000 test digits | about 1 s per digit, 23 min for 10,000 on 6 workers |
| `doomformer mnist --opt 3` | the one-map net with the compiler's fastest schedule: an ACS script copies each layer's light levels into the next layer's gates (`--opt 0` is the default and the v0.4.0 map) | 272 game tics per digit (8 s for two digits on 2 games) |
| [`doomformer demo`](#doomformer-demo) | a 1-layer transformer writes a level name, over 38 tile maps | 15 to 60 s |
| [`doomformer demo --softmax`](#softmax-attention-modern-and-the-flags) | softmax attention computed by the dolls | 1 to 3 min |
| [`doomformer demo --model doomnames_modern`](#softmax-attention-modern-and-the-flags) | RMSNorm and a gated MLP computed by the dolls | about 35 s |
| [`doomformer demo --nightmare`](#doomformer-demo---nightmare) | the 2-layer model, 106 tile maps | 3.5 to 7 min |
| [`doomformer demo --one-map`](#one-map-and---loop) | one forward pass of a smaller model in one map | about 90 s per letter |
| [`doomformer demo --one-map --loop`](#one-map-and---loop) | one map writes a whole name | 17 to 21 min |
| [`doomformer demo --one-map --loop --until-eos --prompt DAN`](#one-map-and---loop) | the map stops at the end-of-text token; one map per prompt length | 4 min (one pass, loaded machine) |
| [`doomformer ask "what does iddqd do"`](#doomformer-ask) | a 2-layer transformer answers from 189 facts | about 11 min |
| `doomformer plan MODEL` | counts dolls and sectors of a shipped model without building it | 20 s to 4 min |
| `doomformer doctor` | checks Python, ViZDoom, the ACS compiler and the Freedoom IWAD | about 1 s |
| `doomformer facts` | runs the engine facts as 46 tiny probe maps in ViZDoom, PASS or FAIL per row (`--json`, `--only NAME`, `--list`, `--export DIR [--check]`) | about 15 s |
| [`doomformer fuzz --nets 40`](verification.md) | random nets compiled and run in DOOM against the integer reference (`--kind mlp\|spiking\|transformer`, `--edge`, `--replay FILE`, `--plant BUG`, `--mutate`) | `--nets 40`: 10 s |
| [`doomformer verify NET`](verification.md) | every input class of a small MLP's whole domain (`--mode classes`) or every input (`--mode all`), tampered maps fail (`--wad`) | seconds to minutes |
| [`doomformer machine multiply 7 9`](#doomformer-machine) | a counter machine in one map: registers and pc are plane heights, every step checked | 17 s |
| [`doomformer tune --check`](tuning.md) | builds the reference nets with the committed opt=2 timing (`doomformer/tuning.json`) and checks fresh inputs; without `--check` it re-measures where each timing constant fails | minutes (`--check`) to hours |
| `doomformer lint WAD` | checks the wave schedule of a compiled one-map brain against the engine model of `doomformer/schedule.py`: plane writes that can share a tic (with a witness input) and timing margins; exit 1 on a failure (`--json`, `--net MODEL`) | 1 to 45 s (MNIST 27 s, loaded machine) |

## `doomformer hello`

A 4-6-5-3 integer MLP compiled into one map, five random inputs, every value checked. The first run also builds the ACS
compiler (about 10 s on a cold cache).

```text
tiny integer MLP 4-6-5-3 compiled into one DOOM map: 60 voodoo dolls, 1222 sectors
input 1/5  x=[-71, -82, 64, 37] -> y=[11, 4, -15]  EXACT (3 outputs + 22 hidden readouts equal the integer reference; 488 game tics)
...
5/5 EXACT
```

## `doomformer draw --digit 7`

The DrawPad level: a 28×28 floor canvas, a READ switch and a 7-segment display in one map with 1,355 voodoo dolls.
`--digit D --seed S` makes the marine walk digit D (built-in stroke variant S) and press READ. Without `--digit` it opens a
window and you walk and draw (arrow keys, hold Ctrl to draw, Space on the READ switch). How it works:
[drawpad.md](drawpad.md).

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
EXACT: all 248 values DOOM computed (hidden layer, 45 score comparisons and 10 win counts, each with its twin, and the one-hot answer) equal the NumPy integer reference computed from the 784 pixel ports DOOM holds (tolerance 0); one display firing
pixels: the 784 ports equal the level's canvas (the tiles the pen inked)
pixels: the 784 ports equal the 3x3 brush stamped at every position the marine had with FIRE held (1 of them within 0.1 units of a tile edge, counted for either tile)
READ pass: 5,705 game tics = 163 s of real-time DOOM at 35 tics/s, 79 s wall | Python sent 58 commands, 29 readout scripts each followed by the ack script it waits for (0 waited more than one advance), and nothing else (0 rejected)
```

## `doomformer mnist`

The same network as DrawPad on MNIST test digits, 4 by default, one DOOM game per digit. `--digits N` samples N test digits
(up to 100 come from the shipped ones), `--all` reads all 10,000, `--workers`, `--ray` and `--record PATH` are described in
[scaling.md](scaling.md). `--image my7.png` reads your own: a photo or drawing of one dark digit on light paper, or light on
dark. EXACT compares DOOM with the integer network. It does not say that the image was a digit.

```text
$ doomformer mnist --digits 2 --seed 1

          ███
          ████                      ▄▄▄█████████▄▄
           ████                     ▀█████▀▀▀▀▀████
            ████                             ▄▄███▀
             ███▄                       ▄▄▄███████▄▄
             ████                       ████▀▀▀▀▀▀█████
              ████                                  ▀███
               ████                      ▄▄         ▄██▀
               ▀████                     █████▄▄██████▀
                ▀██                        ▀▀▀▀▀▀▀▀▀▀

DOOM reads: 1  right            DOOM reads: 3  right
test #46                        test #51

2/2 read correctly | 2/2 EXACT: every value DOOM computed (248 per digit: hidden layer, comparisons, win counts and
the one-hot answer) equals the NumPy integer reference (tolerance 0); Python wrote the 784 pixels, released the dolls and compared | 5,696 game tics per digit | 88 s
```

### `--tiles`, `--opt` and `plan`

`--tiles` runs the same network as four tile maps, one per layer, in resident games (each worker holds the four), and
Python copies the integer values of each layer into the next tile's game: the multi-map design. The same 248 values per
digit are compared at tolerance 0. `--all --workers 6` read all 10,000 test digits: 10,000 of 10,000 exact, 9,716 right
(97.16 %), 216 game tics (summed over the tiles) and 0.83 s per digit, 1,390 s in all, 431.7 digits per minute, and DOOM's
one-hot equals NumPy's on every digit ([`models/mnist/doom_tiles_10000.json`](../models/mnist/doom_tiles_10000.json)).
Sample (2 workers, a loaded machine):

```text
$ doomformer mnist --digits 2 --seed 1 --tiles --workers 2
DOOM reads handwritten digits. The network (784 pixels -> 64 -> 45 score comparisons -> 10 win counts -> one-hot) runs as 4 TILE maps, one per layer: 1,351 voodoo dolls, 111,112 sectors in all. [...]
reading 2 digits with 2 workers; each holds the 4 tile games of the network (8 engine processes, 2 computing at once) (about 1.5 s per digit per worker) ...

[the two digits, as above]

DOOM reads: 1  right            DOOM reads: 3  right
test #46                        test #51

2/2 read correctly | 2/2 EXACT: every value DOOM computed (248 per digit: hidden layer, comparisons, win counts and the one-hot answer) equals the NumPy integer reference (tolerance 0); Python wrote the 784 pixels, copied the layer values between the 4 tile games and compared | 216 game tics per digit (mean) summed over the tiles | 4 s | 30.2 digits per minute in 2 tile workers
```

`--opt N` (0 is the default, 1 to 3) picks the one-map schedule ([scheduling.md](scheduling.md)); `--opt 0` builds the
v0.4.0 maps byte for byte. At `--opt 3` an ACS script copies each layer's light levels into the next layer's gate
heights. It adds and compares nothing, and the dolls still do the sums, thresholds, comparisons and the choice, but the
claim "no ACS between layers" holds at `--opt 0` to 2 only. MNIST drops from 5,696 to 272 game tics:

```text
$ doomformer mnist --digits 2 --seed 1 --opt 3 --workers 2
DOOM reads handwritten digits. The whole network (784 pixels -> 64 -> 45 score comparisons -> 10 win counts -> one-hot) is ONE DOOM map:
1,355 voodoo dolls, 88,146 sectors; every ink pixel releases a doll that crosses the weight lines of the hidden layer, and dolls compare every pair of digit scores, so DOOM itself picks the digit.
reading 2 digits in 2 DOOM games at once [...]

[the two digits, as above]

DOOM reads: 1  right            DOOM reads: 3  right
test #46                        test #51

2/2 read correctly | 2/2 EXACT: every value DOOM computed (248 per digit: hidden layer, comparisons, win counts and the one-hot answer) equals the NumPy integer reference (tolerance 0); Python wrote the 784 pixels, released the dolls and compared | 272 game tics per digit (mean) | 8 s | 14.3 digits per minute in 2 DOOM games
```

`doomformer plan MODEL --opt N --explain` counts dolls and sectors without building the map, predicts the seconds of a pass
(rms error 18 % for one map and 22 % for tiles on 11 held-out nets) and recommends one map or tiles; `--explain` adds the
schedule per layer and what each pass saves:

```text
$ doomformer plan models/mnist/mlp.npz --explain --opt 3
counting dolls and sectors without building the map (large models take a few minutes) ...
MLP models/mnist/mlp.npz: 1355 voodoo dolls, 88146 sectors, fits in one map: yes (span 31,969 x 5,192 units, limit 32,000 per axis)
  one map: about 304 game tics per pass (estimate), predicted 3.2 s per pass
  tiles: 4 tile maps, 1351 dolls in total, about 216 game tics summed, predicted 806 ms per pass
  predicted seconds: engine time of one pass with the games loaded (about 102 ns per sector per tic, error about +-20%, docs/scaling.md); a loaded or slower machine takes longer
  recommendation: tiles is faster by about 3.9x (df.run_many(net, xs, mode='tiles'); `doomformer mnist --tiles` for the MNIST net); one map keeps Python out of the loop between layers
  df.compile(mode='auto') would use: map

schedule of the one-map build, opt ['colour', 'copy', 'exclusive', 'lights', 'tuned']: 304 estimated game tics [...]

layer                   dolls  waves  bound    S    tics   sectors  dominated by
mlp_l0                    784      1      1   40      85    58,509  corridor length (40%)
mlp_l1                    512      1      1   40      81    28,679  corridor length (37%)
mlp_l2                     45      1      1   40      72       701  corridor length (29%)
mlp_l3                     10      1      1   40      66       126  corridor length (28%)

passes, cumulative (estimated settle tics, geometry counter):
                                        tics   delta   sectors    delta   waves
baseline (opt 0)                       5,676           158,550            1,312
+ exclusive dolls share waves          5,676      +0   158,550       +0   1,312
+ light targets never conflict         5,676      +0   158,550       +0   1,312
+ DSATUR / circle-method colouring     5,652     -24   158,544       -6   1,306
+ tuned timing constants               3,560  -2,092   158,538       -6   1,306
+ copy bus (one wave per layer) <-       304  -3,256    88,146  -70,392       4
bound: clique lower bound of the layer's conflict graph (waves cannot go below it); empty at opt 0.
```

## `doomformer demo`

A 1-layer transformer (`doomnames`, d=16, 2 heads, 136 official level names) writes a name. The 38 tile maps run in a pool
of games; Python copies the integers between games and samples each letter from the logits DOOM computed. `--seed N` gives
other names, `--letters N`, `--prompt TEXT`, `--temperature T` and `--png PATH` do what they say.

```text
A transformer writes a DOOM level name; every number it computes is computed by DOOM.
model: the shipped doomnames: 1 layer, d=16, 2 heads, context 16, ReLU attention, trained on 136 DOOM level names
sampling: temperature 180, seed 20
38 tile maps (10,462 voodoo dolls, 135,912 sectors) built in 0 s; running them in 4 DOOM game processes ...

DOOM wrote: 'GHEER'   (not one of its training names)
  174 tile runs in 4 parallel DOOM games | 10,462 voodoo dolls in the tile maps | 6,902 values read back and compared with the NumPy integer reference (tolerance 0) | 22 s
EXACT: every value DOOM exposed equals the integer reference, and the reference writes the same name
```

"Not in its training names" is a low bar. At the default temperature the model writes one of its 136 training names for 6 of
10 seeds (seeds 20 to 29, [`models/doomnames_softmax/EVAL.md`](../models/doomnames_softmax/EVAL.md)); the output says which.
Seed 20 is the hand-picked sample above.

### Softmax attention, modern and the flags

`--softmax` runs the same model trained with softmax attention. The dolls compute the exponentials (a staircase with
log-spaced steps) and divide by their sum (a count of comparisons); [HOW_IT_WORKS.md](HOW_IT_WORKS.md#softmax-attention)
has the construction.

```text
DOOM wrote: 'FINAL INEALL'   (not one of its training names)
  494 tile runs [...] | 4,886 voodoo dolls in the tile maps |
  106,834 values read back and compared with the NumPy integer reference (tolerance 0) [...]
EXACT: every value DOOM exposed equals the integer reference, and the reference writes the same name
```

The `[...]` cuts the number of games and the wall time, which depend on the machine (161 s on 3 games on a busy one).

`--model doomnames_modern` adds RMSNorm and a gated MLP (ReGLU) to the softmax attention. The dolls compute the sums of
squares (as sums of odd numbers), the integer square root (a staircase at d·k²) and the division by it. It wrote MILL, which
is one of its training names, in 170 tile runs on 6 games in 34 s, all 62,000 values exact.

## `doomformer demo --nightmare`

The 2-layer, 24,279-parameter transformer, trained on 11,341 of 12,934 cleaned real DOOM level names, compiled into 106
maps.

```text
A transformer writes a DOOM level name; every number it computes is computed by DOOM.
model: the shipped doombrain: 2 layers, d=32, 4 heads, context 24, ReLU attention
sampling: temperature 150, seed 2
106 tile maps (56,390 voodoo dolls, 785,460 sectors) built in 10 s; running them in 6 DOOM game processes ...

DOOM wrote: 'DEMON CANYON'
  1,690 tile runs in 6 parallel DOOM games | 56,390 voodoo dolls in the tile maps | 120,666 values read back and compared with the NumPy integer reference (tolerance 0) | 210 s
EXACT: every value DOOM exposed equals the integer reference, and the reference writes the same name
```

DEMON CANYON is a hand-picked sample, three edits from DEMON BAYOU in the corpus; it is not in the 12,934 names. Unselected
samples are in [`models/doombrain/eval_report.txt`](../models/doombrain/eval_report.txt): 29 of 30 samples at temperature
200 are new names too.

Close the doors in front of one neuron (`L1.m` unit 33) in every consuming tile and the same brain writes DEMON DEATH at
this seed. The lesion is not selective: 355 of the 416 single neurons change the name at temperature 150,
seed 2 (343 at seed 3, 326 at seed 7, 368 for greedy decoding), and `L1.m` 33 ranks 324th of 416 by teacher-forced logit KL
on the validation names. Three neurons write DEMON DEATH at this seed (`L0.x1` 20, `L1.x1` 29, `L1.m` 33); the numbers and
the command (`python -m demos.lobotomy --names`) are in [DEMOS.md](DEMOS.md#lesion-statistics). In the NumPy reference:

```sh
python -c "import doomformer as df;from doomformer import intref as i;n=df.Transformer.load('doombrain');print(i.generate(n,'',temperature=150,seed=2),'|',i.generate(n,'',temperature=150,seed=2,ablate={'L1.m':[33]}))"
# DEMON CANYON | DEMON DEATH
```

In DOOM the same doors are closed with `DoomTransformer(..., ablate=...)`; the CLI has no flag for it (zero, set, mean, resample
and scale per position and head: [interventions.md](interventions.md); activation patching with recovery:
`python -m doomformer.patching MODEL --clean ... --corrupt ... --doom N`). Four games wrote
DEMON DEATH in 181 s on a busy machine, and a longer check compared 8,280 exposed values of the final context with the ablated reference, all equal:

```python
import doomformer as df
from doomformer import intref
from doomformer.orchestrator import DoomTransformer

if __name__ == "__main__":
    net = df.Transformer.load("doombrain")
    ab = {"L1.m": [33]}
    with df.compile(net, mode="tiles", workers=4) as brain:
        eng = DoomTransformer(net, brain.wad, workers=4, max_live=1, ablate=ab)
        print(eng.generate("", temperature=150, seed=2), "| reference:",
              intref.generate(net, "", temperature=150, seed=2, ablate=ab))
        eng.close()
```

## One map, and `--loop`

`--one-map` runs a different, smaller model, `doomnames_onemap` (1 layer, d=8, 6 positions, trained on level names of up to 5
letters), because the 136-name model needs 116,968 dolls and does not fit one map. Embedding, attention, MLP and logits run
with no Python between layers; Python writes the token and samples the letter after each pass.

```text
A transformer's forward pass, computed inside ONE DOOM map.
model: the shipped doomnames_onemap: 1 layer, d=8, 1 head, context 6, ReLU attention, trained on DOOM level names of up to 5 letters
map built in 5 s: 3,548 voodoo dolls cross lava lines in 95,448 sectors
DOOM is running 1 forward pass(es), about 85 s each on one core (3,500 game tics of dolls walking) ...

DOOM wrote: 'D'
  3,548 voodoo dolls | 95,448 sectors | 176 values read back and compared with the NumPy integer reference (tolerance 0) | 3,488 game tics per pass | 89 s
EXACT: every value DOOM exposed equals the integer reference, and the reference writes the same text
```

`--one-map --loop` builds one map that writes a whole name. At every position the dolls compare all 30 logits, the winner's
doll moves the next position's token input to its letter, and the level rewinds and runs the next pass by itself. Python only
advances the clock and reads: every value of every pass is checked, and the final text has to equal greedy generation.
Add `--softmax` for the softmax model. Measured on one machine (empty prompt, two games at once; from
[HOW_IT_WORKS.md](HOW_IT_WORKS.md#generation-without-python)):

| model | dolls | passes | tics per pass | wall time per pass | values checked | wrote |
|---|---|---|---|---|---|---|
| `doomnames_onemap` | 5,875 | 5 | 4,000 | 186 to 190 s | 29,490 | DAN |
| `doomnames_onemap_softmax` | 6,536 | 5 | 4,656 | 237 to 259 s | 30,510 | DAGE |

The Python side was audited: 1,704 and 1,912 readout commands, nothing else.

`--until-eos` (with `--loop`) adds a halt latch per position: the winner's doll sets it to 1 for EOS, and the level
leaves its loop after the pass whose newest token is EOS. The map is the same file for every prompt of one length;
Python writes the prompt's tokens as initial state before the first pass, then only reads
([HOW_IT_WORKS.md](HOW_IT_WORKS.md#stopping-at-eos)). `--prompt DAN` stopped after pass 1 of at most 2 (4,000 tics,
3,020 values checked, 237 s with the build, load average 33 to 43); `--fast --loop --until-eos --prompt A` takes
about 10 s.

## `doomformer ask`

A 2-layer transformer (d=32, 4 heads) trained on 189 facts about DOOM; the question is the prompt and the answer comes out
letter by letter. One answer is thousands of short DOOM runs (5,320 tile runs for GOD MODE) spread over at most 8 games, and
every value is checked. GOD MODE took 653 s on 6 games and 1,210 s on 3.

```text
You ask DOOM: WHAT DOES IDDQD DO?
DOOM answers with a 2-layer transformer (d=32, 4 heads) trained on 189 facts about DOOM, split into tile maps over 6 DOOM games.
The first letter runs all 20 positions of the question through the network (3,160 tile runs): about 7 min with 6 DOOM games. Each later letter adds one position: about 28 s. (Fitted to two timed runs on a loaded machine; a quiet one was up to twice as fast.)
DOOM is reading your question ...
DOOM says: GOD MODE

  5,320 tile runs in 6 DOOM games | 313,248 values read back and compared with the NumPy integer reference (tolerance 0) | 653 s
EXACT: every value DOOM exposed equals the integer reference, and the reference gives the same answer
```

| you ask | DOOM answers | in the training facts? |
|---|---|---|
| what does iddqd do | GOD MODE | yes, a training paraphrase of fact 52 |
| what is love | INVULNERABILITY | no |
| what is a voodoo doll | BERSERK | no |
| can it run doom | SIXTY SECONDS | no |

The first row is a logged DOOM run. The other answers are the NumPy reference's; `ask` exits with `MISMATCH` when DOOM
differs from the reference, so the answer is DOOM's whenever it prints EXACT. The model gets 94.7 % (606 of 640) of its
training paraphrases right and 35.4 % (67 of 189) of new wordings of the same facts
([`models/doomoracle/EVAL_REPORT.md`](../models/doomoracle/EVAL_REPORT.md)). Outside the 189 facts the answer is exact
and meaningless.

## `doomformer machine`

A counter machine compiled into one map ([machine.md](machine.md)): the program counter and the registers are plane heights
changed by voodoo dolls, the program is geometry, a halt latch stops the run. Python advances the clock, reads, and compares the pc and
every register after every step with its interpreter (tolerance 0). `doomformer machine list` shows the programs: `add a b`,
`multiply a b`, `power a n`, `gcd a b`, `fib n` and `pow2 n` (multi-limb registers), `collatz n`, `primes N`, and `stored` (one
map, two programs loaded as initial state). `--tamper K` changes one line of instruction K in the compiled map and expects the
run to diverge at the step the interpreter predicts; `--tamper-what target` retargets a pc line instead of a register amount.

```text
multiply: 7 * 9
  9 instructions, registers a b c t; the interpreter halts after 464 steps with c = 63 (Python computes 63)
  map: 39 voodoo dolls, 578 sectors, span 17,129 x 4,136
  [...]
c = 63 (expected 63)
  464 steps | 39 dolls | 578 sectors | 126,340 game tics (216 to 216 per step) | 16 s
  audit: 9,610 readout commands, 0 other commands or game calls, 0 state loads (the map starts with the program and the arguments baked in)
  clock: the level's ACS script (rewind, barrier release, settle check, halt-latch read); data path and program counter: dolls
EXACT: the pc and every register of all 464 steps equal the interpreter's (24,180 values, tolerance 0); the halt latch stopped the level
```

Run times (one game, a loaded machine): `add 3 4` 1 s, `multiply 7 9` 17 s, `gcd 48 18` 37 s, `power 2 6` 66 s, `collatz 7` 76 s, `fib 20` 106 s.

## `doomformer fuzz`, `verify`, `facts`, `lint`

Checks of the checker ([verification.md](verification.md), [scheduling.md](scheduling.md)). `fuzz` compiles random nets,
runs them in DOOM and compares every value with the integer reference:

```text
$ doomformer fuzz --nets 40 --workers 2
  [10/40] net 8 mlp ok (8 passes, 184 values)
  [...]
fuzz: 40 mlp nets, 304 passes, 5,914 values compared at tolerance 0, 0 mismatches, 0 errors, 0 unsupported (all-zero layer), 0 refused, 61 inputs refused by the range check; seed 0, 9.8 s
```

`--kind transformer|spiking`, `--edge` (weights at +-255, values at the range limits), `--opt N`, `--replay FILE` and
`--mutate` (change one line of a compiled map and count the kills) are described in verification.md, and `--plant BUG` must
exit 1: a planted compiler bug is found. `verify NET` runs every input class of a small MLP (`--mode all`: every input of
a box) and prints `VERIFIED` only if all runs are exact; it is for MLPs, not transformers.

```text
$ doomformer verify and_or.npz --domain 0 1 --mode all
and_or.npz: 2 inputs, 2 layer(s), unsigned input
mode all: every input of [0, 1]^2 (4 inputs) the net accepts: 4 runs; 0 candidates refused by the range check
VERIFIED: 24 values exact against the integer reference (tolerance 0), 0.2 s
```

`facts` runs the engine facts as small probe maps in ViZDoom and prints PASS or FAIL per row (ZDoom 2.8.1 behaviour in
ViZDoom 1.3.2; GZDoom and Zandronum are untested, `--export DIR` writes the maps for other ports):

```text
$ doomformer facts
engine: ViZDoom 1.3.2 (ZDoom 2.8.1)

PASS  light_saturates    1/1 cases    0.1s  [acs]
PASS  light_same_tic     1/1 cases    0.2s  [acs]
PASS  mover_same_tic     1/1 cases    0.1s  [acs]
PASS  mover_active_lock  1/1 cases    0.1s  [acs]
PASS  mover_speed        1/1 cases    0.1s  [acs]
[...]
23/23 facts hold (46/46 cases) on ViZDoom 1.3.2 (ZDoom 2.8.1) in 7 s
```

`lint WAD` reads a compiled one-map brain and replays every doll tic by tic: it reports plane moves that can share a tic
(with a witness input) and timing margins below their minimum, and exits 1 on either.
