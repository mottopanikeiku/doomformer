# The museum map

I wanted one map that someone can download and play without Python, and in which they can see that the voodoo dolls do
the arithmetic. In the museum you light a digit on a seven-segment pattern of floor pads, press READ, and the dolls work
out which digit it is. A second switch rewires one connection of the network inside the running level, and then the
same 7 reads as a 1. A third switch puts the connection back.

The network is small and written by hand (no training), so every weight can be checked by eye.

## The network

There are seven inputs, one per segment a to g, and x_s is 1 for a lit pad and 0 for a dark one. Each digit d has one
score neuron:

    s_d = b_d + sum_s w_ds * x_s,   w_ds = +1 if digit d lights segment s, else -1,   b_d = 7 - (segments of d)

So s_d counts the segments on which the pads agree with digit d: 7 means the pads show exactly that digit. A 7 (pads a,
b and c) gives the scores 4, 6, 3, 5, 4, 3, 2, 7, 3, 4 for the digits 0 to 9. Then come the same winner layers as in
DrawPad (`drawpad.with_argmax`): 45 pair comparators, 10 win counts and 10 winners. The lowest digit wins a tie, as in
`np.argmax`. The winner's doll crosses the line that lights its digit on the display. `doomformer.museum.reader()` builds
this network and `models/museum/museum_net.json` holds its weights, biases and thresholds as the compiler read them.

In the map, each score is a light level (s_d) and a floor height (-s_d). The doll of segment s walks one lane and
crosses two lines per digit: one adds w_ds to the light, the other adds -w_ds to the floor. That is 140 weight lines
for the 70 weights. Each READ exposes 140 values (scores, comparisons, win counts, winners and the one-hot answer,
hidden values with their negated twins).

## What MUTATE changes

MUTATE changes the connection from segment a to digit 7 from +1 to -1. It does that with ACS `SetLineSpecial` on the two
weight lines of that connection, which the build gives the line ids 1 and 2: the light line goes from
`Light_RaiseByValue(s_7, 1)` to `Light_LowerByValue(s_7, 1)`, the floor line from `Floor_LowerByValue(-s_7, 1)` to
`Floor_RaiseByValue(-s_7, 1)`. These are the only line specials the level ever changes. RESTORE sets them back.

With a, b and c lit, s_7 drops from 7 to 5 while s_1 stays 6, so DOOM reads 1. Of the 128 pad patterns, the flip changes
the answer for 8: four that read 7 before and 1 after (a, a+b, a+c, a+b+c) and four that read 7 before and 0 after
(`models/museum/reference.json` lists every pattern's scores and digit with and without the flip).

I checked that the rewired map is exactly the network with that one weight changed, without running DOOM: the static
extractor (`doomformer extract`) recovers the netlist from the built WAD and it equals the network's. After I replace the
two specials in the WAD the way `SetLineSpecial` does, the recovered netlist equals the network with w_7a = -1, and
against the original network it differs only in layer 0, row 7, column 0 (the weight and its twin)
(`tests/test_museum.py`).

## The hall

North of the start spot are the pads, laid out as a big seven-segment digit on the floor, then a scoreboard and then
the seven-segment display, a staircase of cells rising to the north wall. The switches are on the south wall, behind
you. From west to east: camera A,
camera B, camera C, CLEAR, READ, MUTATE and RESTORE. Standing in front of one shows what it does. The MUTATE podium is
brighter while the connection is rewired.

- Stepping onto a pad switches it on or off. The pads lock while DOOM is reading.
- READ runs the network's own scripts: 3 (rewind the dolls), 1 once per segment (write the port), then 7 (start). Then
  only the dolls work. About 744 game tics later (21 s of real-time DOOM) the winner's doll lights the display.
- After a READ the HUD lists the ten scores, and the scoreboard raises one bar per digit to 16 units per score point.
  Both are copies of the score lights, made by the level after the display has lit. They compute nothing.
- The camera stations put your view inside the circuit: camera A in segment a's input lane, near the two lines MUTATE
  rewires, and cameras B and C in the lanes of the sixth score doll of digit 7 and of digit 1 (that doll passes its
  gate only when the score is at least 6). Moving brings your own view back. In my frames (`museum shots`) these lanes
  are narrow, tall corridors and no doll is in view, so the cameras show where the circuit is, not what it computes.
  The scores are easier to inspect on the HUD and the scoreboard.

The map has 136 voodoo dolls and 3,975 sectors and spans 31,898 x 4,360 map units. It is built with the original
schedule (`opt 0`), so no ACS runs between the layers.

## Commands

    doomformer museum build museum            # museum.wad, museum.json, museum_net.json, reference.json, SHA256SUMS
    gzdoom -iwad freedoom2.wad -file museum/museum.wad
    doomformer play museum/museum.wad         # the same in a ViZDoom window
    doomformer museum verify                  # headless: Python plays the 7 -> MUTATE -> RESTORE tour, every READ checked
    doomformer museum export DIR --check      # the self-test map and its expected lines, run in ViZDoom
    doomformer museum shots DIR               # PNG frames of the hall and the three cameras, before and after MUTATE

A build from the same source gives the same four files, byte for byte. `models/museum/SHA256SUMS` holds their hashes and
`tests/test_museum.py` rebuilds and compares them. The hashes depend on the pinned ACC revision, which compiles the
level's scripts.

## How I checked it

`doomformer museum verify` has Python stand in for a player. It presses only the marine's movement keys to walk onto
pads and USE on switches. During each READ it only advances tics and runs readout scripts, through
`readout.ReadoutOnly`. Every readout is guarded: the map's script 9 first writes the poison value 100,003 into the
USER variables, Python waits until it sees the poison, then runs the readout and waits until the poison is gone. A
readout that never runs therefore raises instead of returning old values. Python then compares all 140 values, the
seven ports (against the lit pads) and the display (one firing, for the right digit) with the NumPy reference of the
network as it is wired at that moment, which it learns from the level's own flag.

`models/museum/verify.log` is the run behind the numbers here. It lights and reads each digit 0 to 9 and then does the
tour: 7, MUTATE, 7 (DOOM reads 1), RESTORE, 7 (DOOM reads 7 again). Every READ was exact, with 148 values checked and
no command other than readouts during a READ.

`doomformer museum export` adds one script to the map, for other ports. Without a player or Python, it lights the
pads for each digit 0 to 9, then rewires and reads a 7, then restores and reads a 7. It waits for the display and for the
dolls to stand still, and logs one line per READ with the digit and all 140 values. `expected.txt` holds those 12 lines
as the NumPy reference computes them. In ViZDoom the 12 logged lines were identical to expected.txt
(`models/museum/selftest.log`).

## Ports

I ran the museum only in ViZDoom 1.3.2 (ZDoom 2.8.1 inside), headless. I did not run it in GZDoom, LZDoom, UZDoom, ZDoom
or Zandronum. The map uses UDMF (namespace "zdoom"), ACS, voodoo dolls, Boom scrollers, sector actions (SecActEnter),
`SetLineSpecial`, `ChangeCamera` and SecurityCamera things. The brain and machine maps built the same way ran exact in
GZDoom 4.14.2, LZDoom, UZDoom, ZDoom 2.8.1 and, with every sector tag below 32,768, Zandronum 3.2.1
([engine_limits.md](engine_limits.md)). That makes the museum likely to work in them, but I have not shown it. The
export is the way to check a port: run `museum_selftest.wad` and compare its `FACT museum` lines with expected.txt
(`--low-tags` builds it with every tag below 32,768 for Zandronum).

## Limits

- The network is hand-written for clean seven-segment patterns. It is not a learned model, and its answer for a messy
  pattern is just the digit with the most agreeing segments.
- A READ checks DOOM against the integer network. It does not check that the network's answer is "right".
- The scoreboard, the HUD scores and the display's segment pattern are drawn by level scripts from values the dolls
  computed. The arithmetic itself is only in the dolls and their lines.
- The release WAD is a build output, not a file in the repository. `doomformer museum build` reproduces it, and
  `models/museum/SHA256SUMS` says what it should hash to.
