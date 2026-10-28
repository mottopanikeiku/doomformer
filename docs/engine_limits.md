# Engine limits: map extent (32,000 units per axis) and the object list (4,096 objects)

ViZDoom 1.3.2 embeds ZDoom 2.8.1 (the binary's version string reads `1.3.2 (ZDOOM 2.8.1+)`), and this page describes that
engine. GZDoom and Zandronum were not run (no build for the arm64 Linux machine the maps were written on); a port that
keeps coordinates in doubles may not show the wrap at all. The probe maps in `doomformer facts --export` are meant for
checking that (section "Running the probes elsewhere"). ZDoom 2.8.1 keeps map coordinates in 16.16 fixed point. A coordinate *difference* of more than
32,767 units does not fit and wraps around. The sector lookup does exactly that subtraction. For every BSP node on its
path, `R_PointInSubsector` calls

```c
inline int R_PointOnSide (fixed_t x, fixed_t y, const node_t *node)
{
	return DMulScale32 (y-node->y, node->dx, node->x-x, node->dy) > 0;
}
```

([`src/vizdoom/src/r_utility.h`](https://github.com/Farama-Foundation/ViZDoom/blob/1.3.0/src/vizdoom/src/r_utility.h),
ViZDoom 1.3.0 source; the installed 1.3.2 was not checked against it). When a thing is more than 32,767 units from a partition
line's origin, `y-node->y` or `node->x-x` overflows, the side test answers wrongly, and the engine puts the thing in
the wrong sector. A map can only produce such a difference if it is wider or taller than 32,767 units. Whether it
actually happens depends on the BSP: the thing has to meet such a node on its walk down the tree.

## What it does to a brain

A probe builds real brains with the shipping emitter (`pure.compile_pure`) next to a plain (tag 0) 2048 x 2048 room
holding the player. To build maps the compiler now refuses, it moves the brain's rows and lifts the span guard. In each
map it measures how far the idle player moves in 35 tics and how many forward passes are exact. A pass is exact when
every light (doll line crossings) and every hidden gate floor (gate reads) equals the integer reference. It also runs
25 training steps of a DOOM-trained perceptron (`learnkit`), whose state dolls rewrite weight heights every step, each
checked against its twin. The probe builds 78 maps in 73 s.

| case | what is moved | span (units) | player drift in 35 idle tics | exact passes |
|---|---|---|---|---|
| x | room east of the brain (3 nets) | 30,000 / 32,000 / 32,700 / 33,000 / 33,500 / 34,000 | 0 for all | all |
| x | | 35,000 | 422.5 (`repro`), 0 (`mine`, `mlp3`) | all |
| x | | 36,000 / 40,000 | 422.5 (`repro`, `mine`), 0 (`mlp3`) | all |
| x | | 46,000 | 422.5 for all three | all |
| y | brain rows from y = 16,000, room south (3 nets) | 30,000 ... 33,500 | 0 for all | all |
| y | | 34,000 ... 46,000 | 422.5 (`repro`, `mine`), 62.6 (`mlp3`) | all |
| far | 3-layer MLP lanes moved east, room at x = -30,000 | 31,000 ... 40,000 | 0 | 3/3 each |
| far | | 46,000 | 422.5 | 3/3 |
| row | tiny transformer, rows 32,000 / 58,000 / 60,000 wide, room central or far west | 31,884 ... 61,929 | 0 | 2/2 each (every exposed scalar) |
| learn | `Perceptron(5, 3)`, 78 dolls: rows 32,000 / 58,000 wide | 31,762 / 57,505 | 0 / 0 | 25/25 steps / 2/25 (step 3: weight `w2_4` stayed 0, twin -1) |
| learn | `Perceptron(8, 2)`, 77 dolls: rows 32,000 / 58,000 wide | 31,945 / 57,150 | 0 / 0 | 25/25 steps / 0/25 (step 1: 6 values differ) |

What the table shows:

* The player is carried at the speed of the brain's conveyor scrollers (422.5 units in 35 tics) or stopped by a wall
  after a while (62.6). The engine has put him in one of the brain's conveyor strips, though he stands in a tag-0 room.
  It happened from 34,000 (y) and 35,000 (x) units on, and at no tested span of 33,500 or less. Those thresholds belong
  to these BSPs: the wrap itself starts at a coordinate difference of 32,768 (see "The wrap by itself" below), and the
  same room drifts with one net and not with another. A span below 33,500 that did not drift is luck of the tree.
* Doll arithmetic goes wrong too. Forward passes stayed exact in every map the probe built, including lanes up to
  62,000 units from the player's room. That agrees with the earlier one-map results, verified exact at about 58,000
  units wide. The learners did not stay exact: in the 58,000-wide layout DOOM lost weight updates within 1 to 3 steps,
  and the same nets are exact for all 25 steps in the 32,000-wide layout. Maps of about 75 or more dolls show the same
  in the learner tests (`tests/test_learnkit.py::test_bigger_maps_stay_exact`). An exact forward pass in a wide map is
  luck of the BSP, so the rule covers the whole map.
* The settle check ends a run only after every DoomPlayer ViZDoom reports, the real player included, has stood still
  for 8 tics. A drifting player can therefore keep a run from ever settling. ViZDoom reports at most 4,096 of them
  (see the object list section below).

## The wrap by itself

`doomformer facts --only wrap_lone_room --only wrap_balanced` isolates the wrap from the brains. Each map holds plain
rooms with distinct floor heights; an ACS script spawns a `MapSpot` at each room's centre and reads
`GetActorFloorZ`, which is the floor of the sector the engine filed the thing under. The rows below come from
`doomformer facts` on ViZDoom 1.3.2; the unbalanced layout is 150 rooms (256 units square, floor 0) in x = 0..5,856,
y = 0..3,856 plus one lone room (floor 64) whose west edge is at the x given.

| layout | lone room west edge x | the lone room is found in |
|---|---|---|
| unbalanced | 28,000 / 32,000 / 32,700 | its own sector |
| unbalanced | 32,800 / 33,000 / 34,000 | a neighbour's sector |
| unbalanced | 36,000 | its own sector again (wrap luck) |
| unbalanced, west | -30,000 | its own sector |
| unbalanced, west | -32,000 | a neighbour's sector |
| balanced: 100 rooms in a row, 60,000 / 64,000 units wide, x or y | | all 100 rooms found |
| balanced: 65,000 units wide, y | | 1 of 100 rooms filed under another sector |

What this says, and what it does not:

* The hard limit is the arithmetic: a node difference of 32,768 or more does not fit. Where the first miss falls is
  luck of the BSP. The balanced row, whose tree is symmetric about the middle of the map, filed every room correctly up
  to 64,000 units of width; the lone-room layout misfiled a room 32,800 units out. I did not dump either tree. A room
  at 32,700 (its centre is at 32,828) was still found, and the room at -30,000 was found though it lies 36,000 units
  west of the other rooms' east edge (x = 5,856): a difference above the limit does not misfile every time.
* The 32,000-unit rule keeps every difference under 32,768 whatever the tree. "Fine below 33,500" is what these
  brains' BSPs happened to allow, and the earlier "wider than about 34,000 units" is where those trees first failed.
* The probe has one room per spawn, so it does not show what happens to a doll that walks across a misfiled
  region; the table of the 78 brain maps above does.

## The rule

Every map the toolkit builds stays within `MAX_SPAN = 32,000` units per axis (`tile.MAX_SPAN`, imported by `pure`),
world included. That leaves a 767-unit margin below the wrap. One-map builds (`pure.compile_pure`: MLP, transformer
onemap, spiking, perceptron, agent, brainlevel) pack their layout to fit; tile maps (`tile.compile_tile`, used by
`mode="tiles"`) are checked (last bullet).

* Layout: layer strips fold into rows in x = [-16,000, 16,000] (`X_ORIGIN`, `X_END`), stacked north from y = 0 to at
  most `Y_END = 27,904`. Output lights sit a few rows below y = -512 and the scroller closets at (-4096, -4096) and
  eastwards. Packing only moves strips; the timing rules (waves, sync barriers, clock teleports) and the arithmetic do
  not depend on where a strip lies.
* Too large: if the rows would pass `Y_END`, a single corridor would be wider than a row, or the finished map would
  span more than 32,000 on either axis, then `layout["fits"]` and `df.plan(net)["fits"]` are False, plan gives a
  `reason`, and compiling raises `ValueError` instead of writing a map. In auto mode plan then picks tiles where it can.
* World area: `df.plan(net)["map"]["world_area"]` (and `brain.meta.layout["world_area"]`) is the rectangle
  [x0, y0, x1, y1] = [-16,000, top - 32,000, 16,000, -2,048], where `top` is the brain's northern edge. Any world
  built inside it keeps the map within the limit. The world hook (`df.compile(net, world=fn)`) rejects a world rect
  that pushes the map past 32,000 units and names this area. `compile_pure` checks the finished map again, so
  callers that pass `world=` directly (brainlevel, flyloop, agent) are covered as well.
* Metadata: `layout["bounds"]`, `layout["span"]` ([width, height]), `layout["max_span"]` and `layout["world_area"]`.
  `df.plan` reports `span` at top level and `span`, `bounds` and `world_area` under `map`.
* Tile maps: `compile_tile` measures the finished tile map (decorations included) and raises `ValueError` beyond the
  limit; `layout_only` reports `layout["bounds"]`, `layout["span"]` and `layout["fits"]` instead. Tile packing is not
  changed: a too-large tile is refused, not repacked. `tiled.tile_plan` (`df.plan(net)["tiles"]`) reports the largest
  tile span per axis and `fits` = every tile within the limit and under 20,000 dolls, with a `reason` naming the tiles
  that are not. A check of zone and output coordinates against ±32,768 alone would call MLP 1600x256 a fit (its
  layer-0 tile is 32,488 units wide) and pick tiles for MLP 2048x512. `compile_tiles` refuses a plan that does not fit.
  In auto mode `df.plan` picks tiles only if they fit; if neither
  one map nor tiles fit, the report's `reason` says so and `df.compile` raises. `tests/test_span.py` pins the
  MLP 1600x256 case. Measured with `df.plan` and `layout_only`; MLP NxH is layer 0 H x N and layer 1 4 x H, weights random
  in -3..3, thresholds [5, 10], one map without a world:

  | net | tile_plan | largest tile (w x h) | one map (w x h) | auto |
  |---|---|---|---|---|
  | MLP 1024x256 | fits | 24,424 x 19,312 | 31,771 x 20,072, fits | map |
  | MLP 1600x256 | **does not fit** (`mlp_l0`) | 32,488 x 23,952 | 31,965 x 28,904, fits | map |
  | MLP 2048x512 | does not fit (`mlp_l0`) | 54,264 x 29,568 | 31,771 x 61,976, does not fit | neither: `df.compile` raises |
  | `models/doomnames` | fits | 8,416 x 5,808 | 31,989 x 919,832, does not fit | tiles |
  | `models/doomplayer` | fits | 6,100 x 4,128 | 31,993 x 21,992, fits | map |

  Larger nets, same procedure (`tile_plan` only): MLP 4096x512 has a largest tile of 108,576 x 29,568 and MLP 3000x1000
  one of 142,704 x 30,720; neither fits.

Sizes under the packed layout (`layout_only` builds with the worlds named; the unpacked layout was 58,000 wide):

| model | dolls | sectors | span (w x h) | world area y range |
|---|---|---|---|---|
| tiny transformer | 593 | 12,620 | 31,884 x 4,520 | -31,576 ... -2,048 |
| `models/doomnames_onemap` | 3,548 | 95,448 | 31,947 x 6,728 | -29,368 ... -2,048 |
| `models/doomplayer_onemap` + arena + dolls' argmax | 1,345 | 37,410 | 31,962 x 7,928 | -30,856 ... -2,048 |
| `models/doomplayer` (d=16) + arena | 8,825 | 528,366 | 31,993 x 24,680 | -14,104 ... -2,048 |
| fly circuit (`models/fly/brain.json`) + fly_alive level | 1,222 | 31,615 | 31,855 x 8,664 | -29,944 ... -2,048 |

The earlier unpacked maps (one-map SHIL at 57,962 units, brainlevel at 57,948, fly_alive at 57,832, the d=16
`models/doomplayer` at 57,978) exceed the limit. Their recorded results were checked exact and stand. The toolkit no
longer produces such maps.

## Object list: ViZDoom lists at most 4,096 objects

ViZDoom 1.3's game state copies at most `VIZ_MAX_OBJECTS = 4096` objects
([`viz_defines.h`](https://github.com/Farama-Foundation/ViZDoom/blob/1.3.0/src/vizdoom/src/viz_defines.h)). It
walks the sectors in index order and lists the things in each one
([`VIZ_GameStateUpdateObjects`](https://github.com/Farama-Foundation/ViZDoom/blob/1.3.0/src/vizdoom/src/viz_game.cpp));
once the list is full, everything after it is cut off. On the d=16 map below `get_state().objects` returns 4,096
DoomPlayers out of 8,826. In a brain map those are the dolls of the earliest layers, because their sectors come first.

A settle check that watches only that list stops at the first 8 tics in which the listed dolls happen to stand still,
while unlisted dolls (later layers, the clock dolls) are still working.

The toolkit therefore checks motion inside the map. Every one-map brain (`pure.compile_pure`) and every tile map
(`tile.compile_tile`) carries ACS script 8 (`tile.motion_acs`, map arrays `mo_x`, `mo_y` and `mo_z`). The script
compares every doll and the player with where they were at its previous run: dolls by the TIDs they give themselves on
their first step (`DOLL_TID0 + i`), the player as the script's activator. It writes the number that moved to USER1 and
the number of dolls without their TID to USER2. `TileRunner.settle` then works as follows:

* If the map has fewer dolls than the list can hold and the list is not full, settling is unchanged: every DoomPlayer
  in the list stood still for 8 tics. The 202-doll MLP takes 992 tics per pass and the tiny transformer settles at
  1,800 tics, the same as with the list alone.
* If the map has 4,095 dolls or more, or the list comes back full, the script also runs on the last tic of every
  8-tic window. Settling then also needs the listed DoomPlayers unchanged, the script to have sampled the previous
  window as well, and zero dolls moved and zero dolls without their TID.
* A map without the script and with too many dolls to list raises `RuntimeError` ("recompile it") instead of settling.
* The settle timeout defaults to twice the layout's settle estimate (`layout["estimated_settle_tics"]`, also what
  `df.plan` reports), with a floor of 20,000 tics, so big maps need no hand-set budget.

The regression test is `tests/test_settle.py`. Its map has 4,204 dolls and 46,242 sectors. 4,201 of the dolls wait at
their sync barrier while the unlisted clock doll walks to the release line; then a second layer reads the first
layer's result:

| settle check | tics | `-h`, `y` for x = 1 / 0 / 1 (want -1, 1 / 0, 0 / -1, 1) |
|---|---|---|
| objects list + motion script | 184 each run | exact all three runs |
| objects list only (forced with `VIZ_MAX_OBJECTS` patched to 10^9) | 40, 24, 40 | 0, 0 every run: nothing computed |

### The d=16 `models/doomplayer`

The map holds the d=16 model in its arena: 8,825 dolls, 528,366 sectors, 31,993 x 24,680 units. It builds in 20 s and
loads in 41 s (3.6 GB resident). [`models/doomplayer/onemap_measure.json`](../models/doomplayer/onemap_measure.json)
records the run with the motion check: dolls, sectors, build and load time, and the decision with its 29,110 tics and
1,860.863 s. Plan's settle estimate for this map is 28,976 tics.

With the list-only check:

* The run stopped with "not settled after 20000 tics" after 1,260 s.
* With that cap lifted, the run "settled" falsely at 20,992 tics (1,281 s) with 471 of 969 exposed values wrong, from
  the `wsum` layer on.
* The 58,000-wide layout gave the identical 471 mismatches.
* With a fixed budget instead (start, advance 34,000 tics, read), one forward was exact: 2,065 s for 34,000 tics,
  16.5 tics/s.

With the motion check, one decision runs exactly with no hand-set budget:

* It settled at 29,088 tics, close to the 28,976-tic estimate.
* The decision, write and read included, took 29,110 tics and 1,861 s (15.6 tics/s).
* All 969 exposed values (every preactivation, attention score, negation and logit) and the action (L) match the integer
  reference, for input tokens [0, 21, 26].
* The two tic rates come from different runs with other jobs running, so they do not isolate what the motion
  script costs.

## Running the probes elsewhere

`doomformer facts` (23 probes, 46 maps; 12 s and 18 s in two runs on a 16-core machine at load average 30 to 40, ViZDoom 1.3.2) is the executable form of the rows in the Engine
facts table of [`HOW_IT_WORKS.md`](HOW_IT_WORKS.md) and of this page. Most probes are ACS-only: the map's own script
builds the situation, waits, and prints one line with `Log`:

```
FACT light_saturates up1=30160 up2=32767 down=-32768
```

`doomformer facts --export DIR` writes the 42 ACS-only maps as `<case>.wad`, the lines they must print as
`expected.txt`, and `RUN.txt` with the command lines. Four probes need ViZDoom's Python side and are not exported:
`puke` (console command), `object_list` (ViZDoom's state), `advance_action` and `turn_key` (ViZDoom's action handling).
`doomformer facts --export DIR --check` runs each exported WAD in a fresh ViZDoom process with the engine console on
stdout and compares the `FACT` line with `expected.txt`: 42 of 42 match. That tests the export path, not another port.

Command lines for another port (not run: no GZDoom or Zandronum build is available on the arm64 Linux machine this was
written on):

```
gzdoom -iwad freedoom2.wad -file light_saturates.wad +map MAP01 -nosound -stdout | grep '^FACT '
for w in *.wad; do gzdoom -iwad freedoom2.wad -file "$w" +map MAP01 -nosound -stdout | grep '^FACT '; done > observed.txt
diff expected.txt observed.txt
```

`freedoom2.wad` ships in the `vizdoom` package directory. If a build prints nothing on stdout, write its console to a file and grep that; check every flag against the port's own help, none was run. Every map ends with `ConsoleCommand("quit")`; if a port refuses
it, stop the run after the line appears. Which rows could differ, as a guess to test and not a finding:

* the 16.16 rows (`plane_span`, `wrap_lone_room`, `wrap_balanced`, `mover_speed`) in a port that computes in doubles;
* the doll-physics rows (`carry_stack`, `doll_steps`, `gate_opens`, `doll_fall`, `teleport_cancels`) if a port changes the
  step check, conveyor or fall arithmetic; their numbers are exact for ViZDoom 1.3.2, so a difference of a unit counts
  as a difference;
* `voodoo_tid` and `self_tag`, if a port copies the start's TID to the doll or runs ENTER scripts for dolls.

A port that differs on a row is a finding about the port. The compiled brains stay checked against the integer
reference only in ViZDoom 1.3.2.

## Queued console commands: about 880 per tic crash the engine process

Reported by the tile-runner work and reproduced here with a one-map script (`script 5 (int a, int b, int c)` counting its
starts) and `game.send_game_command("puke -5 1 <i> 0")` called N times before `advance_action(3)`. N = 400, 800 and 860
run all N starts; N = 880 and N = 1,000 end with `ViZDoomErrorException` ("Unexpected ViZDoom instance crash"; the engine
writes `vizdoom-crash.log` and looks for `gdb` and `xmessage`). The limit is a count, not bytes: 3,000 plain `puke -5`
commands with no arguments ran 3,000 starts, and 1,000 commands with two arguments ran, so the trigger depends on the
command form (not explored further; the three-argument form is the one the port writes use). `runtime.TileRunner.start`
therefore steps one tic after every 512 port writes. This is not a `doomformer facts` probe: a crash opens a
dialog on desktops that have `xmessage`, and was run once by hand (Linux arm64, ViZDoom 1.3.2).

