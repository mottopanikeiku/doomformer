# DrawPad: walk a digit, the level reads it

`doomformer draw` is one DOOM map that holds a trained MNIST network *and* the room it reads from. You walk on a
28 x 28 floor canvas holding FIRE, the tiles under you light up, you press the READ switch, and the map's voodoo
dolls compute the whole network, compare the ten digit scores with each other and light the winning digit on a
7-segment display built from sectors. Python does not pick the digit. It checks afterwards that every number DOOM
produced equals the NumPy integer reference.

```sh
doomformer draw                  # you play in a window (needs a display)
doomformer draw --digit 7        # headless: Python presses the marine's keys to walk a built-in 7, then READ
doomformer mnist                 # the same network on MNIST test digits, also decided by DOOM
```

## How to play (`doomformer draw`)

The map builds (about 10 s), a DOOM window opens on the level, and the terminal prints the controls:

| key | does |
|---|---|
| arrow keys, W / S | walk; Left / Right turn |
| mouse | turns (moving it forward or back does not walk) |
| A / D | step sideways |
| Shift | run |
| Ctrl or the left mouse button | FIRE: hold it while you walk on the canvas and the tiles you walk over turn blue (3 x 3 brush) |
| Space or E | USE: press a switch |
| Esc | DOOM's menu: frees the mouse, pauses the game |

The marine starts facing north. In front of him stand two switches on podiums; the right-hand one is READ, the
left-hand one CLEAR, and the lane between them leads onto the canvas. The 7-segment display rises behind the canvas.

* **Draw**: walk over the canvas with FIRE held. Walking backwards keeps the stroke in view.
* **READ**: walk up to the right-hand switch and press USE. The HUD counts the seconds (`DOOM is reading your
  digit: 37 s`). After about 5,705 tics (163 s at DOOM's 35 tics/s) the winner's doll lights the digit on the
  display and the HUD shows `DOOM reads 7`. The terminal prints the canvas, the digit and the exact check once
  every doll and the marine stand still.
* **While DOOM reads** (from READ until the display lights) the pen, READ and CLEAR are off; the switches answer
  `DOOM is still reading: wait for the display`.
* **CLEAR**: the left-hand switch wipes the canvas and the display. Then draw the next digit.
* **Quit**: close the window (Esc first, the window holds the mouse), choose Quit Game in DOOM's menu, or press
  Ctrl+C in the terminal. The terminal prints `the DOOM window closed` or `stopped`.

READ or CLEAR pressed after the display lit but before every doll stood still cuts the pass short: the terminal
says that READ was not checked, and a new READ is checked as usual.

## The level

| part | what it is |
|---|---|
| canvas | 28 x 28 sectors of 32 x 32 map units (the marine is 32 wide), one per MNIST pixel; the 20 x 20 box MNIST centres its digits in is lit a little brighter. Each tile sector holds a `SecActEnter` thing whose special starts ACS script 201 with the tile's (row, column). |
| pen | the script ignores every activator but the player (his TID, set by the ENTER script; voodoo dolls carry their own TIDs) and, while FIRE is held, stamps a 3 x 3 block of pixels around the tile into the level's canvas array, lights those tiles and changes their floor texture. An OPEN script does the same every tic FIRE is held, so pressing FIRE without moving paints too. Walkway sectors around the canvas reset "the tile under the marine". The pen is off from READ until the display lights. |
| READ | a USE switch on a 32-high podium south of the canvas (too high to step on, low enough to look over). It runs the brain's own scripts: 3 (rewind: dolls home, barriers up, biases restored, the same rewind that runs between any two passes), then script 1 once per pixel (the port writer `TileRunner.start` uses: `Floor_MoveToValue` on the pixel's gate, -16 for ink, 0 for paper), waits 4 tics, then 7 (drop layer 0's start barriers), then shows the seconds since on the HUD every second until the display lights. Refused (a HUD note) from READ until the display lights. |
| CLEAR | the other podium: script 3, wipe the canvas array, its lights and textures, write 0 to every pixel port, switch the display off. Refused like READ. |
| display | a staircase of 14 x 8 sector cells north of the canvas (cells 96 wide and 40 deep, each row 64 higher than the one in front, so it faces the marine); the cells of the seven segments carry tags 10..16. The action of output `y_d` is `ACS_ExecuteAlways(206, d)`: the doll that crosses it lights the segments of `d` (light 255 and a bright floor) and shows `DOOM reads d` in DOOM's big font. |
| STATUS, CANVAS, pixel pages, ACK | readout scripts 207, 208, 210..223 copy the level's state, its canvas rows and the 784 pixel port heights into ACS globals 1..60 (ViZDoom's `USER1..60`); script 209 adds 1 to ACS global 0 (ViZDoom's map reward). See "Readouts". |

`doomformer.drawpad.world` builds it through the ordinary `df.compile(net, world=..., actions=...)` hooks. Nothing in
`pure.py` is special-cased.

## The network: DOOM picks the digit

`models/mnist/mlp.npz` is `784 -> 64 -> 45 -> 10 -> 10`:

```
a0 = [pixel >= 8]           (ink encodes to 16)              784 dolls
h1 = b + W a0,  a1 = 8-level staircase                       64 x 8 = 512 dolls
d_ij = s_i - s_j + 1 for the 45 pairs i < j                  (s = the 10 digit scores, never formed themselves)
c_ij = [d_ij >= 1] = [s_i >= s_j]                            45 dolls
w_i  = sum_{j>i} c_ij - sum_{j<i} c_ji + i                   = #{j > i: s_i >= s_j} + #{j < i: s_i > s_j}
z_i  = [w_i >= 9]                                            10 dolls; exactly one passes: the first maximal score
y    = I z                                                   one-hot; its doll fires the display action
```

`df.with_argmax(net)` (in `doomformer/drawpad.py`) turns any integer MLP whose last layer gives scores into these
winner layers. Each `d_ij` row is one linear combination of the layer before the scores (`W_i - W_j`,
`b_i - b_j + 1`), so nothing is re-read or re-quantised between the scores and the comparison. Because `d_ij` feeds a
gate, every row must satisfy `|b_i - b_j + 1| + sum_k |W_ik - W_jk| * 8 <= 12,000` and `|W_ik - W_jk| <= 255`.
`with_argmax` multiplies the score layer by the largest factor (0.5 % steps) that fits: x0.2545 for the shipped
net. DOOM then computes the argmax of those rescaled integer scores, lowest index on ties, exactly as `np.argmax`.
`tests/test_drawpad.py` checks 9,000 random score vectors with many ties, a rescaled random net, and a DOOM run in
which only the winner's action fires, once.

The doll counts above are lanes; the map adds one clock doll per layer (1,355 dolls in all).

## Who does what

| step | DOOM (dolls, lines, sectors) | level ACS (`drawpad.level_acs`) | Python |
|---|---|---|---|
| drawing | the marine moves; sector actions fire on the tile he enters | stamps the 3x3 brush into its canvas array, lights the tiles | `draw`: runs the STATUS readout after every tic it advances, to see READ. `draw --digit`: presses the marine's movement keys and FIRE (stands in for you), records his position each tic |
| READ | the USE line runs the level script | runs the brain's scripts 3, 1 (x784), 7; counts seconds on the HUD | `--digit`: presses USE once, then releases every key |
| the network | every sum, staircase, pairwise comparison, win count and the one-hot | nothing | nothing (the game handle is `ReadoutOnly`: any write raises) |
| the answer | the winner's doll crosses its `y_d` action line | lights the segments of `d` (a 10-entry lookup) | nothing |
| after the pass | | readout scripts copy sector heights / lights / its own status into `USER1..60` | advances tics until every doll and the marine stand still, runs readout scripts, compares every value with `MLP.reference` of the 784 pixels read back from the ports (`df.MismatchError` on any difference), checks that the display fired once for the digit whose output is 1, that the ports equal the canvas the level held when the pass began, and (`--digit`) that they equal the brush stamped along the recorded path |

The level's ACS does no network arithmetic: it addresses tiles (`row * 28 + col`), sets lights and floors, writes
0 or 16 into ports, looks up which segments a digit has, and divides the tics since READ by 35 for the HUD.

## Readouts

A readout is a level script that copies values into ACS globals 1..60, which ViZDoom reports as `USER1..USER60`.
Python sends it as a console command (`puke -207`), sends the ACK script right after it (`puke -209`: ACS global 0
plus 1, which ViZDoom reports as the map reward in 16.16 fixed point), advances one tic at a time until the reward
has gone up by exactly one, and then reads `USER1..n` (`DrawPad._users`). The two commands reach ZDoom between the
same two tics and run in the same tic, and nothing else writes those globals, so once the ACK has run the values are
the readout's. `ReadoutOnly` allows the ACK command and `get_total_reward`.

Why: in spectator mode (`doomformer draw`) the engine keeps real time, so one `advance_action(1)` runs one or more
tics (1-10 measured), and a command sent between two advances can run on a tic that the next advance does not reach.
The old readout (command, one advance, read) then returned the previous readout's values. That was the glitch seen
once under heavy load: a STATUS read that gets the rows of a blank canvas from the readout before is all zero, and a
brain page read the same way holds the page before, which fails the exact check. Measured during READ passes in
spectator mode under Xvfb, readouts alternating STATUS (`Timer()` in USER9) and CANVAS (all zero on a blank canvas),
so a stale read shows either way:

| readout | load average | readouts | stale |
|---|---|---|---|
| command, one advance, read | 17 | 300 | 32 (20 all-zero STATUS, 12 CANVAS holding the STATUS before) |
| command, one advance, read, 0-40 ms host delays | 17 | 800 | 1 |
| command, one advance, read | 31-34 (4 busy loops added) | 1,200 | 0 |
| readout + ACK, read once the ACK ran | 17-34 | 2,300 | 0 |

How often a plain read is stale depends on timing (engine load against host speed): 32 of 300 in one run, 0 of
1,200 in another. One case: the CANVAS command sent after the state of tic 3,015 had not run when the state of tic
3,017 was read. In PLAYER mode (`draw --digit`, the tests, every other doomformer game) one advance is one tic: 600
STATUS readouts all showed episode time minus `Timer()` = 1, and no ACK of a headless pass needed more than one
advance. In a keyboard session at load average 39, 5 of the 23 readouts of one pass waited more than one advance for
their ACK: the old readout would have read those 5 before their scripts ran. Another keyboard session (load average
14-21, two busy loops added) counted 505 readouts in `watch` and one READ pass: each of its 486 STATUS readouts
showed a later `Timer()` than the one before, and the pass was exact. The old repeat-until-two-runs-agree workaround
is gone; a readout runs once.
`tests/test_drawpad.py` checks the protocol against a stand-in engine whose commands land up to two tics late, and
300 readouts in a real spectator game during a READ pass (needs Xvfb).

## Measurements

One machine (aarch64, 16 cores shared with other jobs), ViZDoom 1.3.2, `OMP_NUM_THREADS=1`. The `draw` rows ran
next to other jobs at load average 9-39; the `mnist` row ran six games at once next to other jobs. The keyboard rows
were measured with the previous 784-48-10 net (1,227 dolls, 5,185 tics per READ).

| what | measured |
|---|---|
| DrawPad map | 1,355 voodoo dolls, 159,467 sectors (brain 158,550 + world 917), 31,912 x 5,768 map units (limit 32,000 per axis); built in 7-14 s |
| `doomformer draw --digit d --seed s`, d = 0..9, s = 0..2, headless, 6 games at once | **30/30 read right**, **30/30 EXACT** (248 values each), pixel ports equal to the level's canvas 30/30 and to the brush along the recorded path 30/30 (0-11 positions per digit within 0.1 units of a tile edge) |
| one headless READ pass | 5,705 game tics from the READ press until every doll stood still = 163 s of real-time DOOM at 35 tics/s; 76-99 s wall; 29 readouts, each with its ACK (58 commands), none waited more than one advance, nothing else (0 rejected) |
| walking (stand-in player) | 243-449 tics per digit, then 74-155 tics to the READ podium |
| peak memory | 1.23 GB per game process (`/usr/bin/time -v`, Python included) |
| `doomformer draw` played under Xvfb with real X key and mouse events (XTest), the window and HUD captured from the X display (previous net) | a 7 drawn with the arrow keys, `,` `.` and Ctrl, READ with Space: **DOOM reads 7, EXACT**, 5,288 tics, 175 s wall (load average 39), 23 readouts, 5 of which waited more than one advance for their ACK. Esc opens DOOM's menu and releases the mouse; Esc closes it. CLEAR with E. A 1 drawn with W/A/S/D and Ctrl: **DOOM reads 1, EXACT**, 5,181 tics, 148 s. CLEAR pressed 26 s into that READ is refused and the pass is checked. Closing the window prints `the DOOM window closed`, exit 0 |
| all keyboard sessions (previous net) | 9 READs in 5 sessions, **9 EXACT**, 5,181-5,288 tics each. Three drawn 7s read as 7 and three drawn 1s as 1; three blank canvases read as 8 (keys sent before the level ran, fixed since: the controls print once it runs; a scripted walk that missed the canvas; one READ pressed on purpose) |
| quitting | Ctrl+C during a READ: `stopped`, exit 130, no process left in its process group; Quit Game in DOOM's menu: `the DOOM window closed`, exit 0; New Game in DOOM's menu does nothing |
| `doomformer mnist --digits 100 --workers 6 --seed 0` (the 100 shipped test digits) | **99/100 read right** (test #62 read as 5, it is a 9), **100/100 EXACT**, 5,696 game tics per digit (mean), 1,555 s in all (about 93 s per digit per game) |
| `doomformer mnist --digits 1000 --seed 0 --workers 14` (sampled from the 10,000 test digits) | **966/1,000 read right**, **1,000/1,000 EXACT**, 5,696 game tics per digit, 7,421 s in all, 8.1 digits per minute; every result in [`models/mnist/doom_1000.json`](../models/mnist/doom_1000.json) |
| 784-32-10 net, np.argmax in Python | 98/100, all exact; 1,042 dolls, 64,105 sectors, 4,408 tics, about 40 s per digit |

`--seed` seeds `[seed, digit]`. The SimWalker predicts the DOOM result: it reads all 30 walks above right with the
shipped net, and 28/30 with the previous 784-48-10 net (the seed-0 0 read as 7, the seed-2 8 as 4), which read the
seed-0 0 as 7 in DOOM too.

## Training (`training/train_mnist.py`)

784 -> 64 -> 10, binarised pixels. Every batch holds 128 MNIST training digits (half of them randomly shifted,
scaled, rotated, sheared, thickened or thinned) and 128 synthetic *walked* digits, drawn from 450,000 painted with the
canvas brush: 150,000 `walked_strokes` (the three `drawpad.DIGIT_STROKES` variants per digit, 14-21 pixels tall,
stretched, rotated, slanted, wobbled and placed) and 75,000 `varied_strokes` (those and 3 to 6 more variants per
digit: narrow, open and overlapping 0s, curly 2s, open 4s, single-stroke 8s; stroke ends cut or extended; 13-22 pixels
tall and wider ranges of the rest). Each is painted along the strokes (`rasterise`) and at every tic of a
`SimWalker`, which walks them with the stand-in player's controller and DOOM's measured player physics
(`walked_pixels`). 20 float epochs, then 15 quantisation-aware epochs with the 8-level staircase
clip(round(relu(h) / D), 0, 8) and a straight-through gradient (lr 5e-4, cosine decay to 1e-5). Before each of them
`equalise` rescales hidden units to equal 99th percentiles of their activations (the float model is unchanged) and
`step_size` picks the step D with the least squared staircase error on 5,000 calibration digits.

The step size matters most. The previous recipe set D from the largest activation (max / 8), which left most
activations on the lowest levels: in one run the float model read 98.76 % of the SimWalker test digits and its
staircase 97.72 %. 48 hidden units with this recipe (40,000 to 150,000 varied digits) reach 98.10-98.56 % on the
walked test digits (strokes) over three runs; 64 reach 98.70 % (seed 0) and 99.08 % (seed 1). The shipped net is 64
wide, seed 0.
`python training/train_mnist.py` reproduces the shipped file byte for byte (checked with `cmp`).

| model | MNIST test (10,000) | walked digits, strokes (5,000) | walked digits, SimWalker (5,000) |
|---|---|---|---|
| first version, 784-32-10 trained on MNIST alone: `mlp.npz`, np.argmax of the scores | 95.44 % | 41.08 % | 41.58 % |
| first version, 784-32-10 + `with_argmax` (score layer x0.4244) | 95.41 % | 41.44 % | 41.86 % |
| previous 784-48-10 + `with_argmax` (score layer x0.2828) | 95.86 % | 97.16 % | 97.14 % |
| shipped score net (784-64-10, np.argmax of the scores) | 97.21 % | 98.72 % | 98.90 % |
| shipped `mlp.npz` (winner layers, x0.2545; DOOM's one-hot) | 97.16 % | 98.70 % | 98.92 % |

All accuracies are the NumPy integer reference (`MLP.reference`); DOOM computes that model exactly. The walked test
digits use another random stream than training. The SimWalker paints within 0 to 6 pixels of what the level paints
when the stand-in player walks the same strokes in DOOM (20 digits checked).

## Notes

* **The stand-in player.** In `draw --digit` Python presses the marine's movement keys (ViZDoom's delta buttons, a
  pure-pursuit controller) and FIRE, and the report says so. Python is read-only from the READ press on.
* **Pixels vs the walked path.** After each READ the 784 pixel ports are compared with the brush stamped at every
  tic position the marine had with FIRE held. The level's sector action can put the marine in the neighbouring tile
  while his reported position is 0.021 units past the tile edge (seen twice). Positions within 0.1 units of an edge
  therefore count for either tile (`Walker.explains`), and the report gives their number.
* **Played with a keyboard.** `doomformer draw` under Xvfb with real X key and mouse events (XTest), screenshots of
  the X display: see "Measurements".
