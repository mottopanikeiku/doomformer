# Engine limits: map extent (32,000 units per axis) and the object list (4,096 objects)

ViZDoom 1.3.2 embeds ZDoom 2.8.1 (the binary's version string reads `1.3.2 (ZDOOM 2.8.1+)`), and this page describes that
engine. The probe maps of `doomformer facts --export` were also run in GZDoom 4.14.2, LZDoom 3.88b, UZDoom 5.0.3, ZDoom
2.8.1 and Zandronum 3.2.1 on x86_64 Linux (section "Other ports"): the coordinate wrap is still there in the GZDoom
family, its plane arithmetic is not 16.16, its UDMF parser refuses coordinates beyond +-32,768, and Zandronum cannot find
sectors with tags above 32,767. The doll-clock counter machine ran exactly in every port tried: all 464 steps in the GZDoom
family, 50 steps in ZDoom 2.8.1 and in Zandronum with tags below 32,768, which does not start it with the default tags. ZDoom 2.8.1 keeps map coordinates in 16.16 fixed point. A coordinate *difference* of more than
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

## Other ports

`doomformer facts` (28 probes, 55 cases; about 10 s on a 16-core machine: 8 s in two runs here at low load, 13 s at load
average 28; 11.5 s on a GitHub ubuntu-latest runner and 20 s on macos-14; ViZDoom 1.3.2) is the executable
form of the rows in the Engine facts table of [`HOW_IT_WORKS.md`](HOW_IT_WORKS.md) and of this page. Most probes are
ACS-only: the map's own script builds the situation, waits, and prints one line with `Log`:

```
FACT light_saturates up1=30160 up2=32767 down=-32768
```

`doomformer facts --export DIR` writes the 51 ACS-only maps as `<case>.wad`, the lines they must print as
`expected.txt`, and `RUN.txt` with the command lines. Four probes need ViZDoom's Python side and are not exported:
`puke` (console command), `object_list` (ViZDoom's state), `advance_action` and `turn_key` (ViZDoom's action handling).
`doomformer facts --export DIR --check` runs each exported WAD in a fresh ViZDoom process with the engine console on
stdout and compares the `FACT` line with `expected.txt`: 51 of 51 match. The values in `expected.txt` are what ViZDoom
printed, so that check alone does not test another engine.

The export also writes two whole brains, `DIR/hello/` and `DIR/hello_tags15/` (`doomformer/ports.py`). Each is the
network of `doomformer hello` (MLP 4-6-5-3, seed 0) in one map with an ACS driver script instead of the Python runner:
the script writes the input ports, drops the start barriers, waits until every voodoo doll has stood still for 8 tics,
and logs one `FACT hello run=k ...` line with `run=k` and the 25 values the Python runner reads (3 outputs, 11 hidden
preactivations, 11 negated gate floors), then rewinds the map (script 3) and does the next input, 5
inputs in all (the 5 of `doomformer hello`). `expected.txt` of these two directories comes from the NumPy reference
(`MLP.reference`); no engine produced them. `hello_tags15` is the same brain with the stage and freeze tag bases moved
below 32,768 (see the Zandronum row below). In ViZDoom both print the 5 lines byte for byte
(`tests/test_ports.py`). `DIR/same_tic/` holds four small maps of the doll clock's same-tic probe (see "The machine"), and
the machines themselves are a separate export, `doomformer machine PROGRAM ARGS --clock dolls --export DIR`.

CPU use of these runs: nothing in them runs in parallel except `ports_run.py`'s `--jobs` (one engine per WAD). A ViZDoom
game is one engine process with two threads, one of them an idle SDL timer, and the engine runs at full speed on one core;
`pytest tests/test_ports.py` took 7.7 s wall and 5.9 s user at load 1.8 (6.2 s user with 20 busy loops competing) on the
16-core machine with `OMP_NUM_THREADS=1`. A user time of several times the wall time (3 min 28 s user in 9.5 s wall at
load 21) was seen once elsewhere; it was not reproduced here and no source of that CPU time was found.

### What was run

Runs of `.github/workflows/ports.yml` (GitHub Actions, 2026-10-28): the probe and hello rows ran three times before
the machine was added (run ids 40666944010, 40668754701 and 40690850348, the same matrix for the 52 rows they share);
`models/ports/` holds run 40724317546, the workflow exactly as shipped (all 6 jobs and the matrix job green), on branch
`ports-experiment` at v0.6.1 plus the working tree of this package; the machine rows of the runs 40711456775 and
40718931214 before it agree with it. The museum rows and `matrix.md` are from run 40907926589 (2026-10-29, branch
`loop/museum-ports`, the same Ubuntu and Mesa versions). `ubuntu-latest` was Ubuntu 26.04.1 (4 vCPUs, x86_64, Mesa 26.0.8
llvmpipe for OpenGL and lavapipe for Vulkan, X server Xvfb, no sound). Every port gets the
same 51 probe WADs, the 4 same-tic WADs and the two hello directories from `doomformer facts --export out` of the same
checkout, Freedoom's
`freedoom2.wad` as IWAD, and the runner `.github/scripts/ports_run.py`, which starts one engine per WAD (3 at a time),
ends it one second after the case's `FACT` lines appeared (limit 90 s per probe, 300 s for hello, 900 s for the museum)
and diffs them with `expected.txt`. The
full tables and logs are in `models/ports/` (`matrix.md`, one directory per port with `results.json`, `observed.txt`,
`summary.md` per set and a banner with the engine's version lines).

| port | version, source | video |
|---|---|---|
| GZDoom | 4.14.2, the `.deb` of the GitHub release (`gh release download -R ZDoom/gzdoom -p 'gzdoom_*_amd64.deb'`) | OpenGL 4.5 (`+vid_preferbackend 0`) and Vulkan (`1`) |
| LZDoom | 3.88b, apt package `lzdoom` of `debian.drdteam.org` | OpenGL |
| UZDoom | 5.0.3, apt package `uzdoom` of `debian.drdteam.org` | OpenGL (`+vid_preferbackend 0`) |
| ZDoom | 2.8.1 (the version ViZDoom embeds), apt package `zdoom` of `debian.drdteam.org` | software renderer |
| Zandronum | 3.2.1 (250804-2140), apt package `zandronum` of `debian.drdteam.org` | OpenGL |

Ubuntu's own apt has none of them (`apt-cache policy gzdoom zandronum zdoom` lists no candidate). The DRD Team server
refused the runners for minutes in some of the 14 runs of this work, so the workflow retries the download and caches the
`.deb` files.

Command line of the GZDoom rows (`{iwad}`, `{wad}` and `{work}` are filled in per WAD; `stdbuf` is needed because the
engine block-buffers stdout into a pipe, and the runner then sees the line late):

```
xvfb-run -n N -s "-screen 0 800x600x24" stdbuf -oL -eL gzdoom -iwad {iwad} -file {wad} +map MAP01 -stdout \
  -nosound -nomusic -noautoload -nojoy -width 320 -height 200 -config {work}/gzdoom.ini -savedir {work}/save \
  -logfile {work}/log.txt +vid_preferbackend 0        # environment: HOME={work} LIBGL_ALWAYS_SOFTWARE=1
```

ZDoom and Zandronum took `<port> -iwad {iwad} -file {wad} +map MAP01 -stdout -nosound` (the same `stdbuf` and
`xvfb-run`). The whole matrix is `gh workflow run ports.yml --ref <branch>` (about 15 to 30 minutes with the machine
rows). `RUN.txt` in the export
has the commands for a stranger's machine. No port ended itself after the map's `ConsoleCommand("quit")`: the runner
had to end every process, and `results.json` says `killed after FACT line` for every row of ZDoom and Zandronum and for
all but the 5 refused maps of each GZDoom-family port (those never start and say `killed after the map was refused`).
The logs differ in what they print. ZDoom 2.8.1 prints `ZDoom doesn't support execution of console commands from
scripts` in 59 of 59 logs, GZDoom, LZDoom and UZDoom print their own version of that line in 54 or 55 of 59 or 60 (the
refused maps never get that far; `grep -rli "support execution" logs`), Zandronum prints no such line in any of 61 and
its process was still running one second after the `FACT` line. ViZDoom's engine refuses the command as well (it
prints `ViZDoom doesn't support execution of console commands from scripts`), so the quit at the end of a map ends
nothing anywhere; the `--check` run and the tests bound the number of tics instead.

### Results

One table with every count (`models/ports/matrix.md` has the rows; each cell is PASS / DIFF / map refused, from the
`results.json` files). A port's total is the sum of what it ran: the 51 probe WADs, the 4 same-tic WADs, the 2 hello
brains, the machine rows it could run (`multiply 7 9` only where the shim speeds the engine up; `multiply 2 3` and
`subleq add 3 4` everywhere; the two low-tag machine rows only for Zandronum) and the museum's self-test map
(`doomformer museum export`; Zandronum also gets its low-tag build). The museum rows come from run 40907926589. Every
other cell of that run's matrix is identical to run 40724317546, and `models/ports/matrix.md` is that run's matrix
artifact. [museum.md](museum.md#ports) has the details.

| port | 51 probe maps | 4 same-tic | 2 hello | machine rows | museum | all rows run |
|---|---|---|---|---|---|---|
| ZDoom 2.8.1 | 51 / 0 / 0 | 4 / 0 | 2 / 0 | 2 rows: 2 PASS | 1 PASS | 60 rows: 60 PASS |
| Zandronum 3.2.1 | 49 / 2 / 0 (`tag_range`, `tag_wrap`) | 2 / 2 (default-tag cells 4 and 16 differ) | 1 / 1 (`hello` differs) | 4 rows: 2 PASS (low tags), 2 DIFF (default tags) | 2 rows: PASS (low tags), no line in 900 s (default tags) | 63 rows: 55 PASS, 7 DIFF, 1 no line |
| GZDoom 4.14.2 (OpenGL and Vulkan, same rows), LZDoom 3.88b, UZDoom 5.0.3 | 42 / 4 / 5 (`mover_speed`, `mover_busy`, `plane_span_32768`, `plane_span_33000`; `wrap_lone_room_32700` ... `_36000` refused) | 4 / 0 | 2 / 0 | 3 rows: 3 PASS | 1 PASS | 61 rows: 52 PASS, 4 DIFF, 5 refused |

The 51 probe maps plus the 2 hello brains (53 rows) give the numbers the README uses: ZDoom 53 PASS, Zandronum 50 PASS
and 3 DIFF, the GZDoom family 44 PASS, 4 DIFF and 5 refused.

The 43 rows that hold in all six runs are 40 probe maps, the two `same_tic_*_tags15` maps and `hello_tags15`: `light_saturates`,
`light_same_tic`, `mover_same_tic`, `mover_active_lock`, `plane_span_32767`, `plane_span_32000`, `voodoo_tid`,
`carry_stack`, the four `setcurrent_*`, `doll_steps`, `gate_opens`, `self_tag`, both `line_special_tid_*`, the six
`line_order_teleport_*`, `doll_fall`, `dormant`, `line_floor_speed`, the four `wrap_lone_room` rows inside +-32,768, all
five `wrap_balanced_*` and the five `wrap_shifted_*` rows. They cover conveyors (stacked carry scrollers), the 24-unit
step, gate sectors, falls, teleports ordered against other line specials, line specials with a TID, light levels and the
TID handling of voodoo dolls, in GZDoom as in ZDoom.

The doll arithmetic of a whole brain runs elsewhere too. The 5 lines of the `hello` brain (5 x `run=k` plus 25 values,
130 fields) equal the NumPy integer reference byte for byte in GZDoom 4.14.2 (OpenGL and Vulkan), LZDoom 3.88b, UZDoom
5.0.3 and ZDoom 2.8.1, and `hello_tags15` does so in all six runs, Zandronum 3.2.1 included. A port takes 72 to 77 s
of game time for the 5 inputs (35 tics per second, about 15 s per input). In each of them the voodoo dolls walked the
conveyor lanes, crossed the gate and effect lines and left the lights and plane heights the reference computes. The
other probes only suggest that; this run measures it.

### Differences

Observed values, from `models/ports/matrix.md` (identical in the four GZDoom-family runs):

| row | ViZDoom 1.3.2, ZDoom 2.8.1, Zandronum | GZDoom family |
|---|---|---|
| `mover_speed` | `t1_300000=9464`, `t1_500000=-6072`, `t3_500000=-12144` | `12000` for these three and the other nine values |
| `mover_busy` | `fr_14019=12`, `fl_m14018=-12`, `cr_14018=12`, `cl_m14019=-12`: the second request is dropped | `-243`, `243`, `-243`, `243`: accepted, like the five other keys |
| `plane_span_32768`, `plane_span_33000` | `moved=0` (the player does not move) | `moved=1` |
| `wrap_lone_room_32700`, `_32800`, `_33000`, `_34000`, `_36000` | `floor=64, 0, 0, 0, 64` | no `FACT` line: map refused, `Value 32956.000000 out of range for a coordinate 'X'. Valid range is [-32768 .. 32768]` (the first vertex out of range: 32,956 for the 32,700 map, the west edge x for the others) |

* The three plane rows (`mover_speed`, `mover_busy`, `plane_span`) are the 16.16 fixed-point rows. GZDoom's
  `secplane_t` holds its plane distance as `double D` and its floor mover keeps `double m_Speed`
  ([`r_defs.h`](https://github.com/ZDoom/gzdoom/blob/g4.14.2/src/gamedata/r_defs.h),
  [`a_floor.h`](https://github.com/ZDoom/gzdoom/blob/g4.14.2/src/playsim/mapthinkers/a_floor.h), tag g4.14.2). A mover
  step of 18,750 units, a speed of 300,000 or a floor-to-ceiling span of 32,768 does not overflow there, so no step is
  lost, no height wraps and the player is not frozen. LZDoom and UZDoom print the same values; their source was not
  read. The plane bound of +-14,018 units (`doomformer/planes.py`) and the 32,768-unit span rule are ZDoom 2.8.1 limits
  (ViZDoom, ZDoom and Zandronum agree on them); `hello` stays exact in all of them.
* The five refused rows come from a parser limit. GZDoom's UDMF parser rejects a coordinate outside
  [-32768, 32768] ([`udmf.cpp`](https://github.com/ZDoom/gzdoom/blob/g4.14.2/src/maploader/udmf.cpp), line 277), and
  these maps put the lone room's east edge at 32,956 or farther. Whether GZDoom keeps the wrap was therefore asked with
  a follow-up probe, `wrap_shifted`: 150 rooms in x = -6,000 ... -144 and one lone room, every coordinate inside the
  limit. In ViZDoom a lone room starting at 26,000 or 28,000 is found, one at 30,000, 31,000 or 32,000 is filed under a
  neighbour. GZDoom, LZDoom and UZDoom print the same five values. `FLevelLocals::PointInSubsector` converts the point
  to `fixed_t` and walks the nodes with `R_PointOnSide`, which computes
  `DMulScale(y - node->y, node->dx, node->x - x, node->dy, 32)` on 32-bit values
  ([`p_maputl.cpp`](https://github.com/ZDoom/gzdoom/blob/g4.14.2/src/playsim/p_maputl.cpp),
  [`r_utility.h`](https://github.com/ZDoom/gzdoom/blob/g4.14.2/src/rendering/r_utility.h)), the same subtraction as in
  ZDoom 2.8.1. `wrap_balanced_y_65000` (`bad=1`) agrees too. The map extent rule (32,000 units per axis) therefore holds
  for the GZDoom family; what the family adds is the 32,768 coordinate limit of the parser. The maps carry no node
  lump, so each engine builds its own BSP at load; the five rows come out the same, which says nothing about other
  maps.
* Zandronum 3.2.1 differs on five of the 57 probe, same-tic and hello rows (`tag_range`, `tag_wrap`, `hello`, the two
  default-tag `same_tic` rows), on the machine rows and on the default-tag museum row (no line in 900 s; its low-tag
  build is exact). The probe, same-tic, hello and machine rows share one observation: sector tags from 32,768 up are not found.
  `tag_range` has sectors with tags 1,000, 20,000, 32,767, 32,768, 40,000, 60,000, 64,004 and 65,535 and reads each
  floor with `GetSectorFloorZ(tag, 0, 0)`. ViZDoom, ZDoom and the GZDoom family find all of them. Zandronum finds
  1,000, 20,000 and 32,767 and returns `0` (no such sector) for the five tags from 32,768 up, so the boundary lies
  exactly between 32,767 and 32,768. A line special whose `arg0` is such a tag (`line64010`) lowers nothing. `tag_wrap`
  (sectors with tags 40,000 and 65,541) adds: Zandronum finds the first one by looking up -25,536 (40,000 - 65,536) and
  the second by looking up 5 (65,541 - 65,536), and does not find either by its own tag. The cause is in the source of
  the release tag `ZA_3.2.1` (`https://foss.heptapod.net/zandronum/zandronum-stable`, files read at that tag): the
  sector's tag is a 16-bit field, `short tag;` (`src/r_defs.h:709`); the UDMF parser stores `sec->tag =
  (short)CheckInt(key);` for the `id` key (`src/p_udmf.cpp:1312`); and the lookup compares that field with an `int`,
  `while (start >= 0 && sectors[start].tag != tag)` (`src/p_spec.cpp:274`, in `P_FindSectorFromTag(int tag, int start)`,
  which the ACS `GetSectorFloorZ` and the line specials use). A tag of 40,000 is stored as -25,536 and never equals 40,000;
  65,541 is stored as 5. (The binary run here reports version 3.2.1, build 250804-2140; that it was built from exactly this tag was not checked,
  but the probe results are what `short` predicts.) The compiler's tags reach 64,0xx (port tags from
  1,000, light tags from 20,000, stage barriers from 60,000, freeze barriers 64,000 and wave tags above them), so in
  `hello` the clock doll's wave lines act on sectors Zandronum does not find. The input ports (tags from 1,000) and the
  start barrier (tag 2) are found, so the dolls of the first wave walk; the later waves are never released, the hidden
  values stay at their biases and all five runs log the same line. (One-off diagnostic scripts, not shipped, showed the
  first wave's dolls ending where ViZDoom's do and the wave barriers' floors unreadable.) With `hello_tags15` (stage
  base 25,000, freeze base 30,000, no tag above 32,767) the same brain matches in Zandronum. A brain with enough waves
  to pass tag 32,767 from that base would fail again; the compiler itself still emits the 60,000 and 64,000 bases, and
  only the low-tag exports move them (`ports.build_hello(low_tags=True)`, `machine ... --low-tags --export` and
  `museum export --low-tags`).

### The machine

The hello brain has 60 dolls. The doll-clock counter machine is the larger test. `doomformer machine multiply 7 9
--clock dolls` is one map with 43 voodoo dolls and 721 sectors whose loop is a loop doll and line specials (the default
build has no OPEN script; its ACS lump holds the readout pages and the stillness script only). It runs 464 steps, 235,567
game tics. `doomformer machine PROGRAM ARGS --clock dolls --export DIR` (`doomformer/ports_machine.py`) writes
`DIR/machine.wad`, the same map plus one more ACS script, a logger (script 40, OPEN) that exists only in the export. The
logger sends nothing and does what the Python runner does: it waits until the counter `pass` equals k, waits until no doll
has moved for 8 tics, and then logs in one tic every value the runner reads (56 per step: the pc one-hots, the registers,
the fire values, their negations, `pass` and `done`) as `FACT machine step=k pc0=0 pc1=1 ... r0=7 ...`. Step 0 is the
initial state. `DIR/expected.txt` holds the 465 lines (26,040 values) from the machine's interpreter (`Program.want`); no
engine produced them. A step that does not come within 6,000 tics logs `FACT machine step=k stalled pass=P` and stops the
logger, so a stalled engine says where. `subleq add 3 4` is exported the same way (5 lines of 314 values, 713 dolls,
17,718 sectors; the logger first writes the program's initial memory with the calls the host's loader makes).
`--low-tags` builds with every sector tag and TID below 32,768 (stage and freeze tag bases 25,000 and 30,000, loop
barrier 31,000, teleport TIDs from 32,000), which the default build never does.

ViZDoom prints the 465 lines byte for byte (sha256 of the observed and of the expected file are both `0e72d853e1c1...`;
80 s for 1,000,000 tics with the console on stdout); the `--low-tags` build prints the same lines, and `subleq add 3 4`
prints its 5 (`tests/test_ports.py`).

The 464-step run takes 1.9 hours at 35 tics per second, so the workflow runs the engine behind
`.github/scripts/ports_timeshim.c`, an `LD_PRELOAD` library that runs the engine's clock 200 times faster (it scales
`clock_gettime`, `gettimeofday`, `nanosleep` and the like; game tics, ACS `Delay` and everything the map computes count
tics and do not change). Three other ways did nothing (run 40704056556, not kept in `models/`): `gzdoom +i_timescale 20` (76 s for `hello`, as
without), `faketime` (the same), and the first shim runs, whose `LD_PRELOAD` `stdbuf` replaced (on this runner `stdbuf` is the Rust coreutils
one); the shim therefore line-buffers stdout itself and the machine runs do not use `stdbuf`. The shim does not speed up
ZDoom 2.8.1 and Zandronum: their 50-step `multiply 2 3` (about 25,000 tics) took 721 and 722 s, which is 35 tics per
second, real time. For those two the workflow runs `multiply 2 3` and `subleq add 3 4` instead of `multiply 7 9`
(ZDoom and Zandronum were not run on `multiply 7 9`; there is no partial result). UZDoom 5.0.3 needs `TIMESHIM_SPEED=20`:
re-run on the shipped `subleq add 3 4` map (run 40747178775, `models/ports/uzdoom/speed_subleq_add/`), x1 printed 2 of 5
lines in 150 s, x20 all 5 in 83 s, and x200 printed step 0 and nothing after it in 150 s, with no `stalled` line either;
the cause was not investigated.

| row | ZDoom 2.8.1 | Zandronum 3.2.1 | GZDoom 4.14.2 GL | GZDoom Vulkan | LZDoom 3.88b | UZDoom 5.0.3 (x20) |
|---|---|---|---|---|---|---|
| `same_tic_cell4`, dolls not home (ViZDoom: 14) | 14 | 11 | 14 | 14 | 14 | 14 |
| `same_tic_cell16` (ViZDoom: 0) | 0 | 11 | 0 | 0 | 0 | 0 |
| both `same_tic_*_tags15` | as ViZDoom | as ViZDoom | as ViZDoom | as ViZDoom | as ViZDoom | as ViZDoom |
| `multiply 7 9`, 465 lines | not run | not run | PASS, 183 s | PASS, 279 s | PASS, 71 s | PASS, 343 s |
| `multiply 2 3`, 51 lines | PASS, 721 s | stalled at step 1; `--low-tags`: PASS, 722 s | PASS, 24 s | PASS, 33 s | PASS, 9 s | PASS, 43 s |
| `subleq add 3 4`, 5 lines | PASS, 302 s | stalled at step 1; `--low-tags`: PASS, 304 s | PASS, 57 s | PASS, 82 s | PASS, 36 s | PASS, 75 s |

What the table says:

* GZDoom (OpenGL and Vulkan), LZDoom and UZDoom ran the whole 464 steps of `multiply 7 9` and print the interpreter's 465
  lines byte for byte: 26,040 values equal at every step boundary. The voodoo-doll clock (the loop doll walking a chain
  of line specials that raise and lower barriers, teleport every doll home and reset every port and light) runs there
  exactly, with GZDoom's double-precision planes. That agrees with the differences above: the four rows where GZDoom
  differs (`mover_speed` at speeds of 300,000 and more, `mover_busy`, `plane_span` at 32,768 and more) are outside the
  plane bound of +-14,018 and the 32,768-unit span that the machine's maps stay within. That is a reading of the two
  tables together, not a separate test.
* ZDoom 2.8.1 (the engine ViZDoom embeds) and, with its tags moved, Zandronum ran `multiply 2 3` (50 steps, 51 lines) and
  `subleq add 3 4` (4 steps) exactly. The `multiply 7 9` run was not done there: 1.9 hours per port at the engines' own
  clock. No run diverged from the interpreter, and none stalled except Zandronum's with the default tags.
* The same-tic rows reproduce in every port that runs the clock. In ZDoom 2.8.1 two line specials that one doll crosses
  in the same tic do not both act (`models/dollclock/probe_same_tic.py`: with 4-unit chain cells 14 of 24 dolls are
  not teleported home by tic 500, with 16-unit cells none). The GZDoom family does the same, 14 and 0, so the doll clock's
  16-unit cell (`dollclock.CELL`) is needed in all of them, and the loss is not an artefact of ViZDoom's engine or of
  16.16 arithmetic.
* Zandronum with the default tags does not start the machine: step 0 prints, then `FACT machine step=1 stalled pass=0`
  after 6,000 tics (`multiply 7 9`, `multiply 2 3` and `subleq add 3 4` alike), and the same-tic WADs print 11 dolls not
  home for both cell widths. That is the tag observation above again (the loop barrier has tag 90,000, the wave
  barriers 64,004 and up); with `--low-tags` the machine and the same-tic rows equal ViZDoom's. Whether Zandronum runs
  `multiply 7 9` exactly with low tags was not run for the time reason above.
* Time: 71 s (LZDoom) to 343 s (UZDoom at x20) for the 235,567 tics with the shim at x200, 690 to 3,300
  tics per second including the engine's start; ZDoom and Zandronum 12 minutes for 25,000 tics.

### The wrap is a Linux behaviour of the ViZDoom wheel

On macOS arm64 the wrap did not occur in the rows that show it on Linux. GitHub Actions, `macos-14`, ViZDoom 1.3.2
wheel, `doomformer facts --json` (run 40690058123; `models/ports/facts_macos14.json`) against `ubuntu-latest` x86_64 in
the same run (`facts_ubuntu_x86.json`): 8 of 54 cases differ, all of them wrap rows, and every one of them reports the
lone room or the balanced rooms as found in their own sector.

| row | Linux (x86_64 here, arm64 on the development machine) | macOS arm64 |
|---|---|---|
| `wrap_lone_room_-32000`, `_32800`, `_33000`, `_34000` | `floor=0` (filed under a neighbour) | `floor=64` (found) |
| `wrap_balanced_y_65000` | `bad=1` | `bad=0` |
| `wrap_shifted_30000`, `_31000`, `_32000` | `floor=0` | `floor=64` |

The other 46 cases are identical on both platforms (the plane, doll, light, TID and line rows included), and every wrap
row that is found on Linux is found on macOS. The cause was not investigated. `facts.py` therefore expects the found
values when `sys.platform == "darwin"` for exactly these eight rows, and the probe texts say so
(`facts_macos14_platform_aware.json`, from run 40690838494 with the platform-aware `facts.py`, has all 55 cases passing on
macOS, and `pytest tests/test_facts.py` passed there and on ubuntu-latest in the same run). The consequence for the compiler: the
32,000-unit rule (`MAX_SPAN`) is needed on Linux and kept everywhere; macOS does not show the failure, but nothing
here shows that it is safe to exceed the limit there, because only these probe maps were tried.

### Limits of this result

* The five ports are one engine family (ZDoom) in three branches: ZDoom 2.8.1 with Zandronum (2.8.1-based), and GZDoom
  with LZDoom and UZDoom (same source lineage). The results of the GZDoom three are not independent of each other. No
  Boom-family port (dsda-doom, PrBoom+) was tried: they have no ACS and no UDMF, and the maps need both. No Windows
  build and no macOS build of another port was run (ViZDoom itself ran on macOS, see above). Nothing here ran on real graphics hardware; the video is software OpenGL or Vulkan.
* The probe `expected.txt` files stay ViZDoom's Linux output. Stock ZDoom 2.8.1 prints all 51 of those lines and
  Zandronum 49, so the probes are not an artefact of ViZDoom's build; only `hello`'s expected lines come from an
  independent source (NumPy).
* ViZDoom's patched ZDoom runs the maps without the 35 Hz wait; the ports above run in real time, so
  one probe takes up to 16 s (median 2.3 to 11 s, by port) and the jobs took 4 min to 6 min with the install.
* The transformer, MNIST and the learners were not run in another port, nor was `multiply 7 9` in ZDoom 2.8.1 or
  Zandronum. Their rows use the same lines, plane movers and doll scheduling as `hello` and the machine; that is an
  argument and was not measured. The machine ran on x86_64 Linux only.
* The follow-up probes `wrap_shifted`, `line_floor_speed`, `tag_range` and `tag_wrap` were written after the first GZDoom run
  and are part of `doomformer facts` and of the export. ViZDoom 1.3.2 crashed the engine process in both attempts at
  a `wrap_shifted`-style map with the lone room at x = 32,512 (150 rooms in -6,000 ... -144), so that value is not in the
  list; the crash was not investigated.

## Queued console commands: about 880 per tic crash the engine process

Reported by the tile-runner work and reproduced here with a one-map script (`script 5 (int a, int b, int c)` counting its
starts) and `game.send_game_command("puke -5 1 <i> 0")` called N times before `advance_action(3)`. N = 400, 800 and 860
run all N starts; N = 880 and N = 1,000 end with `ViZDoomErrorException` ("Unexpected ViZDoom instance crash"; the engine
writes `vizdoom-crash.log` and looks for `gdb` and `xmessage`). The limit is a count, not bytes: 3,000 plain `puke -5`
commands with no arguments ran 3,000 starts, and 1,000 commands with two arguments ran, so the trigger depends on the
command form (not explored further; the three-argument form is the one the port writes use). `runtime.TileRunner.start`
therefore steps one tic after every 512 port writes. This is not a `doomformer facts` probe: a crash opens a
dialog on desktops that have `xmessage`, and was run once by hand (Linux arm64, ViZDoom 1.3.2).

