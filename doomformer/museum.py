"""The museum: a playable DOOM level whose voodoo-doll circuit reads a seven-segment digit, and lets you rewire it.

One map (MAP01) holds a small integer network and the hall it reads from. Nothing in it is trained; every weight is
written down here:

  input      seven floor pads laid out as a seven-segment digit (segments a..g). Stepping onto a pad toggles it. READ
             writes x_s = 1 (lit) or 0 (dark) into the network's seven input ports, the gate floors -x_s.
  scores     one neuron per digit d:  s_d = b_d + sum_s w_ds x_s,  w_ds = +1 if digit d lights segment s, else -1,
             b_d = 7 - (segments of d). s_d counts the segments on which the pads agree with d (7 = a perfect match).
             It is a light level and a floor height in the map. The doll of each input lane adds w_ds to the light
             of s_d and -w_ds to its negated floor by crossing one line each: 140 weight lines for the 70 weights.
  argmax     drawpad.with_argmax: 45 pair comparators, 10 win counts and 10 winners (np.argmax: lowest digit wins a
             tie). The winner's doll crosses the line that lights digit d on the seven-segment display.

MUTATE (a switch) rewires one genuine connection: the two weight lines of w_7a (segment a -> digit 7), which add +1 to
s_7's light and -1 to its floor, become lines that add -1 and +1. That is ACS SetLineSpecial on the two lines' ids,
the only line specials the level ever changes; after it the map's netlist is the compiler's netlist for w_7a = -1.
Lit a, b and c (a 7): s_7 drops from 7 to 5 and s_1 = 6 wins, so DOOM reads 1. RESTORE puts the two lines back.

Inspecting: the scoreboard south of the display raises one bar per digit to the score DOOM holds (the level copies
the score lights into the bars after each READ: a readout, nothing is computed there), the HUD lists the ten scores,
and three camera stations put the view in the circuit's doll lanes (ChangeCamera to a SecurityCamera): A, segment a's
input lane near the two lines MUTATE rewires; B and C, at the start of the lanes of digit 7's and digit 1's sixth score
dolls, looking at their gates (a doll passes iff its score >= 6: after MUTATE digit 7's stops there). Moving returns
the view.

Python's part. `build` writes the map and its records; nobody needs Python to play it (docs/museum.md: ports). The
headless check (`Museum.verify`, `doomformer museum verify`) has Python stand in for a player: it presses the marine's
keys to walk onto pads and press switches, then watches each READ through readout.ReadoutOnly (tic advances and
readout scripts only) and compares every value DOOM computed (140 network values, 7 ports, the display) with the
NumPy integer reference of the network as rewired at that moment. Every readout is guarded: the map's script 9
poisons the USER variables first and the readout must replace the poison (a script that never ran raises). The
export (`export`) adds a self-test script that drives the same level routines without a player and logs every value,
so other ports can be diffed against the NumPy lines without Python.
"""
from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from . import drawpad
from .drawpad import SEGMENT_BITS, Walker

SEGMENTS = "abcdefg"
FLIP = (7, 0)               # (digit, segment): the connection MUTATE rewires, segment a -> digit 7
POISON = 100_003            # what script 9 writes into USER1..60 before a readout (no light or height reaches it)
GUARD_SCRIPT = 9            # compile_pure(guard=True)'s poison script
PLAYER_TID = 900
CAM_TID0 = 950              # SecurityCamera TIDs 950..952 (stations A, B, C)
LID_LIGHT, LID_PLANE = 1, 2  # line ids of w_7a's two weight lines (light s_7, floor -s_7)
PAD_TAG0, SEG_TAG0, BAR_TAG0, LAMP_TAG = 10, 20, 30, 40   # world sector tags: pads a..g, display, bars 0..9, lamp
(S_ENTER, S_PAD, S_READ, S_CLEAR, S_SHOW, S_MUTATE, S_RESTORE, S_CAM, S_HINT, S_STATUS, S_PORTS,
 S_SELFTEST) = range(200, 212)
IDLE, WRITING, THINKING, SHOWN = 0, 1, 2, 3                  # mu_phase
STATUS_NAMES = ("phase", "reads", "digit", "fires", "bits", "flip", "think_start", "timer", "clears", "mutations")
OPT = 0                     # the original schedule: no ACS between layers (docs/scheduling.md)
SECACT_ENTER, EXEC_ALWAYS, SECURITY_CAMERA = 9998, 226, 9025
LIGHT_RAISE, LIGHT_LOWER, FLOOR_RAISE, FLOOR_LOWER = 110, 111, 23, 20
ACK_TICS = 64               # a readout must show its result within this many tics (spectator mode keeps real time)

# --- the hall (world area: y <= -2048; x and y in map units; the marine is 32 wide) ---------------------------------
WEST, EAST = -768, 768
D_ROWS, D_COLS, D_W, D_D, D_RISE = 14, 8, 96, 40, 64      # the display: drawpad's staircase of cells
D_X0, D_S = -D_COLS * D_W // 2, -2720
NORTH = D_S + D_ROWS * D_D                                 # -2160
BAR_W, BAR_S, BAR_N, BAR_UNIT = 72, -2880, -2800, 16       # scoreboard: bar d's floor = 16 * s_d
BAR_X0 = -5 * BAR_W
SEG_T, SEG_L, PAD_TOP, PAD_X = 64, 256, -2976, 0          # floor pads: thickness, length, top edge, centre
PODIUM_S, PODIUM_N, FRONT_N = -4224, -4192, -4128          # podium row on the south wall, the hint cells north of it
SOUTH = PODIUM_S
START = (0, -4000)
PODIUMS = ("cam_a", "cam_b", "cam_c", "clear", "read", "mutate", "restore")
POD_W, POD_GAP = 96, 64
POD_TEX = dict(cam_a="SW1TEK", cam_b="SW1TEK", cam_c="SW1TEK", clear="SW1GRAY", read="SW1COMP", mutate="SW1SKULL",
               restore="SW1GARG")
ROOM = dict(ceil=1024, ctex="F_SKY1", wall="STARTAN3")
WALK = dict(floor=0, light=176, ftex="FLOOR4_8", **ROOM)
PAD_OFF, PAD_ON = (128, "FLAT23"), (255, "CEIL5_1")
SEG_OFF, SEG_ON, SEG_BG = (112, "FLAT23"), (255, "CEIL5_1"), (48, "FLAT5_1")
COLUMN = 320                # walking columns x = +-COLUMN pass the pads without touching them


def podium_x(k: int) -> tuple[int, int]:
    x0 = -(len(PODIUMS) * POD_W + (len(PODIUMS) - 1) * POD_GAP) // 2 + k * (POD_W + POD_GAP)
    return x0, x0 + POD_W


def pad_rects() -> list[tuple[int, int, int, int]]:
    """(x0, y0, x1, y1) of the pads a..g: a seven-segment digit on the floor, corners and holes left as walkway."""
    t, w = SEG_T, SEG_L
    xl, x0, x1, xr = PAD_X - w // 2 - t, PAD_X - w // 2, PAD_X + w // 2, PAD_X + w // 2 + t
    yt = PAD_TOP
    ya, ym1 = yt - t, yt - t - w
    ym0, yd1 = ym1 - t, ym1 - t - w
    yd0 = yd1 - t
    return [(x0, ya, x1, yt), (x1, ym1, xr, ya), (x1, yd1, xr, ym0), (x0, yd0, x1, yd1), (xl, yd1, x0, ym0),
            (xl, ym1, x0, ya), (x0, ym0, x1, ym1)]


def pad_centre(s: int) -> tuple[float, float]:
    x0, y0, x1, y1 = pad_rects()[s]
    return (x0 + x1) / 2, (y0 + y1) / 2


# --- the network -------------------------------------------------------------------------------------------------------
def score_net(flip: bool = False):
    """The score layer alone (df.MLP, relu input x in {0, 1}): s_d = b_d + sum_s w_ds x_s, then an identity read-out
    of the clipped scores. flip=True: w_7a = -1 instead of +1 (what MUTATE does to the map)."""
    from .kit import MLP
    lit = np.array([[(SEGMENT_BITS[d] >> s) & 1 for s in range(7)] for d in range(10)], dtype=np.int64)
    W = 2 * lit - 1
    if flip:
        d, s = FLIP
        W[d, s] = -W[d, s]
    b = 7 - lit.sum(1)
    return MLP([W, np.eye(10, dtype=np.int64)], [b, np.zeros(10, dtype=np.int64)],
               [np.array([1]), np.arange(1, 8)], signed_input=False)


def reader(flip: bool = False):
    """The network the level computes: score_net plus drawpad.with_argmax (pairs, wins, winners, one-hot y)."""
    net = drawpad.with_argmax(score_net(flip))
    if net.score_scale != 1.0:
        raise AssertionError("the score layer must fit with_argmax unscaled")
    return net


def bits_of(x) -> int:
    return sum(int(v) << s for s, v in enumerate(x))


def x_of(bits: int) -> np.ndarray:
    return np.array([(bits >> s) & 1 for s in range(7)], dtype=np.int64)


def reference(bits: int, flip: bool = False) -> dict:
    """What DOOM must compute for the lit pads `bits` (bit s = segment s): scores, winner and every exposed value."""
    net = reader(flip)
    ref = net.reference(x_of(bits))
    names = sorted(net.exposed_names())
    return dict(bits=bits, flip=flip, scores=[int(v) for v in ref["hidden"][0]], digit=int(np.argmax(ref["y"])),
                values={n: net._expected(n, ref) for n in names})


def reference_table() -> dict:
    """Every input (all 128 pad patterns), with and without the flip: the scores and the digit DOOM must show."""
    rows = []
    for flip in (False, True):
        for bits in range(128):
            r = reference(bits, flip)
            rows.append(dict(bits=bits, segments="".join(SEGMENTS[s] for s in range(7) if bits >> s & 1),
                             flip=flip, scores=r["scores"], digit=r["digit"]))
    changed = [r["bits"] for r, f in zip(rows[:128], rows[128:]) if r["digit"] != f["digit"]]
    return dict(flip=dict(digit=FLIP[0], segment=SEGMENTS[FLIP[1]], weight=[1, -1]),
                patterns_whose_digit_the_flip_changes=changed, rows=rows)


# --- the level -------------------------------------------------------------------------------------------------------
def world(ports: list[int], score_lights: list[int], flip_lines: dict, selftest: tuple | None = None):
    """The museum's world hook for df.compile: the hall (pads, display, scoreboard, podiums) and the level's ACS.
    ports: tag of input port x_s; score_lights: tag of s_d's light; flip_lines: the two lines MUTATE rewires
    (`flip_specials`); selftest: export only, (cases, planned meta) of the self-test script (`selftest_acs`)."""

    def build(m) -> str:
        bands = []                                    # (y0, y1, [(x0, x1, kind)]) north to south, kind: walk/...
        cells = []
        for r in range(D_ROWS):
            y = D_S + (D_ROWS - 1 - r) * D_D
            row = [(WEST, D_X0, "walk")]
            for c in range(D_COLS):
                s = drawpad.segment(r, c)
                row.append((D_X0 + c * D_W, D_X0 + (c + 1) * D_W, ("seg", r, s)))
            row.append((D_X0 + D_COLS * D_W, EAST, "walk"))
            bands.append((y, y + D_D, row))
        bands.append((BAR_N, D_S, [(WEST, EAST, "walk")]))
        bars = [(WEST, BAR_X0, "walk")] + [(BAR_X0 + d * BAR_W, BAR_X0 + (d + 1) * BAR_W, ("bar", d))
                                            for d in range(10)] + [(BAR_X0 + 10 * BAR_W, EAST, "walk")]
        bands.append((BAR_S, BAR_N, bars))
        bands.append((PAD_TOP, BAR_S, [(WEST, EAST, "walk")]))
        pads = pad_rects()
        ys = sorted({y for p in pads for y in (p[1], p[3])}, reverse=True)
        for y1, y0 in zip(ys, ys[1:]):
            row, xs = [], WEST
            for s, (px0, py0, px1, py1) in sorted(enumerate(pads), key=lambda e: e[1][0]):
                if py0 <= y0 and py1 >= y1:
                    if px0 > xs:
                        row.append((xs, px0, "walk"))
                    row.append((px0, px1, ("pad", s)))
                    xs = px1
            row.append((xs, EAST, "walk"))
            bands.append((y0, y1, row))
        bands.append((FRONT_N, ys[-1], [(WEST, EAST, "walk")]))
        front = [(WEST, podium_x(0)[0], "walk")]
        pods = [(WEST, podium_x(0)[0], "walk")]
        for k in range(len(PODIUMS)):
            a, b = podium_x(k)
            if k:
                gap = (podium_x(k - 1)[1], a)
                front.append((*gap, "walk"))
                pods.append((*gap, "walk"))
            front.append((a, b, ("front", k)))
            pods.append((a, b, ("podium", k)))
        front.append((podium_x(len(PODIUMS) - 1)[1], EAST, "walk"))
        pods.append((podium_x(len(PODIUMS) - 1)[1], EAST, "walk"))
        bands.append((PODIUM_N, FRONT_N, front))
        bands.append((PODIUM_S, PODIUM_N, pods))
        fronts, podiums = {}, {}
        for y0, y1, row in bands:
            for x0, x1, kind in row:
                if kind == "walk":
                    r = m.rect(x0, y0, x1, y1, **WALK)
                elif kind[0] == "seg":
                    _, r_, s = kind
                    light, tex = SEG_BG if s is None else SEG_OFF
                    r = m.rect(x0, y0, x1, y1, floor=(D_ROWS - r_) * D_RISE, light=light, ftex=tex,
                               tag=0 if s is None else SEG_TAG0 + s, facing="STARTAN3", **ROOM)
                elif kind[0] == "bar":
                    r = m.rect(x0, y0, x1, y1, floor=0, light=200, ftex="CEIL5_1", tag=BAR_TAG0 + kind[1],
                               facing="COMPBLUE", **ROOM)
                elif kind[0] == "pad":
                    r = m.rect(x0, y0, x1, y1, floor=0, light=PAD_OFF[0], ftex=PAD_OFF[1], tag=PAD_TAG0 + kind[1],
                               **ROOM)
                    m.thing((x0 + x1) / 2, (y0 + y1) / 2, SECACT_ENTER, special=EXEC_ALWAYS, arg0=S_PAD,
                            arg2=kind[1])
                elif kind[0] == "front":
                    r = m.rect(x0, y0, x1, y1, **WALK)
                    fronts[kind[1]] = r
                    m.thing((x0 + x1) / 2, (y0 + y1) / 2, SECACT_ENTER, special=EXEC_ALWAYS, arg0=S_HINT,
                            arg2=kind[1])
                else:
                    k = kind[1]
                    lamp = PODIUMS[k] == "mutate"
                    tex = POD_TEX[PODIUMS[k]]
                    r = m.rect(x0, y0, x1, y1, floor=32, light=200, ftex="CEIL5_1", facing=(tex, tex),
                               tag=LAMP_TAG if lamp else 0, **ROOM)
                    podiums[k] = r
                cells.append(r)
        for k, pod in podiums.items():
            name = PODIUMS[k]
            script, arg = {"clear": (S_CLEAR, 0), "read": (S_READ, 0), "mutate": (S_MUTATE, 0),
                           "restore": (S_RESTORE, 0)}.get(name, (S_CAM, k))
            m.link(fronts[k], pod, special=EXEC_ALWAYS, arg0=script, arg2=arg, playeruse=True, repeatspecial=True)
        m.thing(*START, 1, angle=90)                  # the player: the LAST player-1 start
        return level_acs(ports, score_lights, flip_lines) + (selftest_acs(*selftest) if selftest else "")

    return build


CAMERA_TEXT = (
    "Camera A: segment a's doll lane. It crosses one line per digit and plane; MUTATE rewires two of them.",
    "Camera B: digit 7's sixth score doll. It passes its gate only if s7 >= 6.",
    "Camera C: digit 1's sixth score doll. It passes its gate only if s1 >= 6.")
HINT_TEXT = dict(
    cam_a=CAMERA_TEXT[0] + " USE: look. Move: back.", cam_b=CAMERA_TEXT[1] + " USE: look. Move: back.",
    cam_c=CAMERA_TEXT[2] + " USE: look. Move: back.",
    clear="CLEAR: all pads dark, the brain rewound.",
    read="READ: the level writes the lit pads into the circuit; the voodoo dolls compute and light the answer.",
    mutate="MUTATE: rewire segment a -> digit 7 from +1 to -1 (two weight lines).",
    restore="RESTORE: put segment a -> digit 7 back to +1.")


def level_acs(ports: list[int], score_lights: list[int], flip_lines: dict) -> str:
    """The level's scripts (numbers >= 200; see the module docstring and docs/museum.md). Only READ writes the brain:
    the network's own scripts 3 (rewind), 1 (one port write per segment) and 7 (start), as TileRunner.start does."""
    fl = flip_lines
    hints = " ".join(f'if (k == {k}) HudMessageBold(s:"{HINT_TEXT[p]}"; HUDMSG_FADEOUT, 3, CR_WHITE, 0.5, 0.85, 3.0, 1.0);'
                     for k, p in enumerate(PODIUMS))
    cams = " ".join(f'if (k == {k}) HudMessageBold(s:"{CAMERA_TEXT[k]}"; HUDMSG_FADEOUT, 3, CR_GOLD, 0.5, 0.85, 6.0, 1.0);'
                    for k in range(3))
    scores = ", ".join(f's:"{" " if d else ""}{d}:", d:GetSectorLightLevel({t})' for d, t in enumerate(score_lights))
    status = " ".join(f"o{k + 1} = {e};" for k, e in enumerate(
        ("mu_phase", "mu_reads", "mu_digit", "mu_fires", "mu_bits", "mu_flip", "mu_t0", "Timer()", "mu_clears",
         "mu_mutations")))
    port_page = " ".join(f"o{s + 1} = GetSectorFloorZ({t}, 0, 0) >> 16;" for s, t in enumerate(ports))

    def rewire(flipped: bool) -> str:
        a, b = (fl["flipped"] if flipped else fl["original"])
        return (f"SetLineSpecial({LID_LIGHT}, {a[0]}, {a[1]}, {a[2]}, {a[3]}, 0, 0); "
                f"SetLineSpecial({LID_PLANE}, {b[0]}, {b[1]}, {b[2]}, {b[3]}, 0, 0);")

    return f"""
// --- the museum level (doomformer/museum.py) ---
int mu_port[7] = {{ {", ".join(map(str, ports))} }};
int mu_seg[10] = {{ {", ".join(map(str, SEGMENT_BITS))} }};
int mu_score[10] = {{ {", ".join(map(str, score_lights))} }};
int mu_phase; int mu_reads; int mu_digit = -1; int mu_fires; int mu_bits; int mu_flip; int mu_t0; int mu_clears;
int mu_mutations;

function void mu_pads(int bits) {{
  int s;
  for (s = 0; s < 7; s++) {{
    if (bits & (1 << s)) {{ Light_ChangeToValue({PAD_TAG0} + s, {PAD_ON[0]}); ChangeFloor({PAD_TAG0} + s, "{PAD_ON[1]}"); }}
    else {{ Light_ChangeToValue({PAD_TAG0} + s, {PAD_OFF[0]}); ChangeFloor({PAD_TAG0} + s, "{PAD_OFF[1]}"); }}
  }}
  mu_bits = bits;
}}
function void mu_segments(int bits) {{
  int k;
  for (k = 0; k < 7; k++) {{
    if (bits & (1 << k)) {{ Light_ChangeToValue({SEG_TAG0} + k, {SEG_ON[0]}); ChangeFloor({SEG_TAG0} + k, "{SEG_ON[1]}"); }}
    else {{ Light_ChangeToValue({SEG_TAG0} + k, {SEG_OFF[0]}); ChangeFloor({SEG_TAG0} + k, "{SEG_OFF[1]}"); }}
  }}
}}
function void mu_bars(int shown) {{          // the scoreboard copies each score light (a readout; nothing is computed)
  int d;
  for (d = 0; d < 10; d++) Floor_MoveToValue({BAR_TAG0} + d, 32, shown * {BAR_UNIT} * GetSectorLightLevel(mu_score[d]), 0);
}}
function int mu_busy(void) {{ return mu_phase == {WRITING} || mu_phase == {THINKING}; }}
function void mu_wait(void) {{
  HudMessageBold(s:"DOOM is still reading: wait for the display"; HUDMSG_PLAIN, 2, CR_ORANGE, 0.5, 0.2, 2.0);
}}
function void mu_rewire(int flipped) {{        // the only line specials the level changes: w_7a's two weight lines
  if (flipped) {{ {rewire(True)} }}
  else {{ {rewire(False)} }}
  mu_flip = flipped; mu_mutations++;
  if (flipped) Light_ChangeToValue({LAMP_TAG}, 255); else Light_ChangeToValue({LAMP_TAG}, 200);
}}
script {S_ENTER} ENTER {{
  Thing_ChangeTID(0, {PLAYER_TID}); ClearInventory();
  HudMessageBold(s:"The museum: step on the floor segments to light a digit, then press READ (south wall).";
                 HUDMSG_FADEOUT, 1, CR_GOLD, 0.5, 0.12, 6.0, 1.0);
}}
script {S_PAD} (int s) {{                    // SecActEnter of pad s: the player only, never a voodoo doll
  if (ActivatorTID() != {PLAYER_TID}) terminate;
  if (mu_busy()) {{ mu_wait(); terminate; }}
  mu_pads(mu_bits ^ (1 << s));
}}
script {S_READ} (void) {{                    // the brain's own scripts 3, 1 (once per segment) and 7
  int s;
  if (mu_busy()) {{ mu_wait(); terminate; }}
  mu_phase = {WRITING}; mu_reads++; mu_digit = -1; mu_fires = 0; mu_segments(0); mu_bars(0);
  ACS_ExecuteAlways(3, 0);
  Delay(3);
  for (s = 0; s < 7; s++) ACS_ExecuteAlways(1, 0, mu_port[s], (mu_bits >> s) & 1, (mu_bits >> s) & 1);
  Delay(4);
  mu_t0 = Timer(); mu_phase = {THINKING};
  ACS_ExecuteAlways(7, 0);
  while (mu_phase == {THINKING}) {{
    HudMessageBold(s:"DOOM is reading the lit segments: ", d:(Timer() - mu_t0) / 35, s:" s";
                   HUDMSG_PLAIN, 1, CR_GOLD, 0.5, 0.12, 2.0);
    Delay(35);
  }}
}}
script {S_CLEAR} (void) {{
  int s;
  if (mu_busy()) {{ mu_wait(); terminate; }}
  ACS_ExecuteAlways(3, 0);
  for (s = 0; s < 7; s++) ACS_ExecuteAlways(1, 0, mu_port[s], 0, 0);
  mu_pads(0); mu_digit = -1; mu_fires = 0; mu_segments(0); mu_bars(0); mu_phase = {IDLE}; mu_clears++;
  HudMessageBold(s:"pads cleared"; HUDMSG_FADEOUT, 1, CR_WHITE, 0.5, 0.12, 1.0, 1.0);
}}
script {S_SHOW} (int d) {{                   // fired by the winner's doll crossing its y_d line
  mu_fires++; mu_digit = d; mu_phase = {SHOWN};
  mu_segments(mu_seg[d]); mu_bars(1);
  SetFont("BIGFONT");
  HudMessageBold(s:"DOOM reads ", d:d; HUDMSG_FADEOUT, 1, CR_GREEN, 0.5, 0.12, 30.0, 1.0);
  SetFont("SMALLFONT");
  HudMessageBold(s:"segments that agree, per digit (the score lights): ", {scores};
                 HUDMSG_FADEOUT, 4, CR_WHITE, 0.5, 0.2, 30.0, 1.0);
}}
script {S_MUTATE} (void) {{
  if (mu_busy()) {{ mu_wait(); terminate; }}
  if (mu_flip) {{ HudMessageBold(s:"already rewired: RESTORE puts it back"; HUDMSG_FADEOUT, 2, CR_ORANGE, 0.5, 0.2, 2.0, 1.0); terminate; }}
  mu_rewire(1);
  HudMessageBold(s:"rewired: segment a -> digit 7 is now -1 (it was +1). READ again."; HUDMSG_FADEOUT, 2, CR_RED, 0.5, 0.2, 6.0, 1.0);
}}
script {S_RESTORE} (void) {{
  if (mu_busy()) {{ mu_wait(); terminate; }}
  if (!mu_flip) {{ HudMessageBold(s:"nothing to restore"; HUDMSG_FADEOUT, 2, CR_ORANGE, 0.5, 0.2, 2.0, 1.0); terminate; }}
  mu_rewire(0);
  HudMessageBold(s:"restored: segment a -> digit 7 is +1 again"; HUDMSG_FADEOUT, 2, CR_GREEN, 0.5, 0.2, 6.0, 1.0);
}}
script {S_CAM} (int k) {{                    // camera stations: the circuit itself; moving returns the view
  if (ActivatorTID() != {PLAYER_TID}) terminate;
  ChangeCamera({CAM_TID0} + k, 0, 1);
  {cams}
}}
script {S_HINT} (int k) {{
  if (ActivatorTID() != {PLAYER_TID}) terminate;
  {hints}
}}
script {S_STATUS} (void) {{ {status} }}
script {S_PORTS} (void) {{ {port_page} }}
"""


# --- the circuit's lines and cameras: found in the compiled map, recorded in its metadata -------------------------
def _tag(meta, name: str) -> int:
    return int(re.search(r"Get\w+\((\d+)", meta.layout["read_expressions"][name]).group(1))


def flip_specials(meta) -> dict:
    """The two weight lines of w_7a as (special, arg0, arg1, arg2) before and after the rewiring, from the layout:
    Light_RaiseByValue(light of s_7, 1) and Floor_LowerByValue(floor -s_7, speed, 1), and the opposite pair."""
    from .tile import MOVE_SPEED
    light, plane = _tag(meta, f"h1_{FLIP[0]}"), _tag(meta, f"-h1_{FLIP[0]}")
    return dict(original=[(LIGHT_RAISE, light, 1, 0), (FLOOR_LOWER, plane, MOVE_SPEED, 1)],
                flipped=[(LIGHT_LOWER, light, 1, 0), (FLOOR_RAISE, plane, MOVE_SPEED, 1)])


def _lane(tm, gate: int) -> list[int]:
    """The sectors of the doll lane through `gate`, west to east (a lane is a path of sectors)."""
    out = []
    for first in sorted(tm.nbr.get(gate, ()), key=lambda s: tm.x0[s]):
        path, prev, cur = [], gate, first
        while True:
            path.append(cur)
            nxt = [s for s in tm.nbr.get(cur, ()) if s != prev]
            if len(nxt) != 1:
                break
            prev, cur = cur, nxt[0]
        out.append(path)
    west, east = out
    return west[::-1] + [gate] + east


def _lane_of_gate(tm, tag: int, limit: int) -> list[int]:
    """The lane whose gate sector has tag `tag` and the floor of a gate with that limit."""
    from .tile import STEP_UP
    gates = [s for s in range(tm.n_sectors) if tm.tag[s] == tag]
    lanes = [_lane(tm, g) for g in gates]
    lanes = [ln for ln in lanes if any(tm.floor[s] == limit - STEP_UP for s in ln)]
    if len(lanes) != 1:
        raise AssertionError(f"museum: {len(lanes)} lanes with gate tag {tag} and limit {limit}")
    return lanes[0]


def locate(text: str, meta) -> dict:
    """In the compiled TEXTMAP: the indices of w_7a's two weight lines (in segment a's input lane) and three camera
    spots (x, y, angle): A looks west along segment a's lane from just east of those lines; B and C stand in the home
    sectors of the sixth score dolls of digit 7 and of digit 1 (where they start) and look east at their gates."""
    from .extract import TextMap
    tm = TextMap(text)
    spec = flip_specials(meta)
    lane = _lane_of_gate(tm, meta.ports["floor:-x0"], -1)
    found = {}
    for a, b in zip(lane, lane[1:]):
        for i in tm.edge.get((a, b) if a < b else (b, a), []):
            ln = tm.lines[i]
            key = (ln[2], ln[3], ln[4], ln[5])
            for which, want in zip(("light", "plane"), spec["original"]):
                if key == want:
                    found.setdefault(which, []).append((i, float(tm.x1[a])))
    if sorted(found) != ["light", "plane"] or any(len(v) != 1 for v in found.values()):
        raise AssertionError(f"museum: expected one light and one plane line of w_7a in segment a's lane, got {found}")
    (li, lx), (pi, px) = found["light"][0], found["plane"][0]
    cams = [_spot(tm, lane, max(lx, px) + CAM_BACK)]
    for d in (FLIP[0], 1):                    # from the doll's home, looking east at its gate (passes iff s_d >= 6)
        ln = _lane_of_gate(tm, _tag(meta, f"-h1_{d}"), -6)
        home = ln[0]
        cams.append((float(tm.x0[home]) + 12.0, float((tm.y0[home] + tm.y1[home]) / 2), 0))
    return dict(lines=dict(light=li, plane=pi), cameras=cams)


CAM_BACK = 128              # a camera stands this far east of what it shows, looking west


def _spot(tm, lane: list[int], x: float) -> tuple[float, float, int]:
    """(x, y, 180): a camera in the lane at x (inside the lane's last sector if the lane ends before x)."""
    inside = [s for s in lane if tm.x0[s] <= x <= tm.x1[s]]
    s = inside[0] if inside else lane[-1]
    x = min(x, float(tm.x1[s]) - 8)
    return float(x), float((tm.y0[s] + tm.y1[s]) / 2), 180


def finish(wad: Path, meta) -> dict:
    """Give w_7a's two weight lines their ids (LID_LIGHT, LID_PLANE) and add the three SecurityCameras, in the
    compiled WAD's TEXTMAP. Nothing else changes: no other line, sector or script. Returns what `locate` found."""
    from .machine import _wad_bytes, _wad_lumps
    lumps = _wad_lumps(Path(wad).read_bytes())
    tm = next(i for i, (n, _) in enumerate(lumps) if n == "TEXTMAP")
    text = lumps[tm][1].decode()
    where = locate(text, meta)
    lines = text.split("\n")
    idx = [i for i, ln in enumerate(lines) if ln.startswith("linedef {")]
    for which, lid in (("light", LID_LIGHT), ("plane", LID_PLANE)):
        k = idx[where["lines"][which]]
        if " id = " in lines[k]:
            raise AssertionError("museum: the weight line already has an id")
        lines[k] = lines[k].replace("linedef { ", f"linedef {{ id = {lid}; ", 1)
    things = [i for i, ln in enumerate(lines) if ln.startswith("thing {")]
    cams = [f"thing {{ x = {x:.1f}; y = {y:.1f}; type = {SECURITY_CAMERA}; angle = {a}; id = {CAM_TID0 + k}; "
            f"height = 40; skill1 = true; skill2 = true; skill3 = true; skill4 = true; skill5 = true; single = true; "
            f"coop = true; dm = true; }}" for k, (x, y, a) in enumerate(where["cameras"])]
    lines[things[0]:things[0]] = cams                 # before every player start: the last start stays the player's
    lumps[tm] = ("TEXTMAP", "\n".join(lines).encode())
    Path(wad).write_bytes(_wad_bytes(lumps))
    return where


# --- building ---------------------------------------------------------------------------------------------------------
_NO_LINES = dict(original=[(0, 0, 0, 0)] * 2, flipped=[(0, 0, 0, 0)] * 2)


def _actions() -> dict:
    from . import act
    return {f"y{d}": [act.script(S_SHOW, d)] for d in range(10)}


def plan(opt: int = OPT):
    """The compiled map's metadata from a layout-only pass (no ACS, no file): its port and light tags."""
    from .kit import _build
    return _build(reader(), Path("layout_only.wad"), world=world([0] * 7, [0] * 10, _NO_LINES), actions=_actions(),
                  layout_only=True, guard=True, opt=opt)


def compile_museum(wad: Path, *, selftest: bool = False, opt: int = OPT, configure=None, resolution=None):
    """Build the museum map into `wad` (its .json next to it) and return its kit.MLPBrain (the game starts on first
    use). selftest=True: the export, with the self-test script (`selftest_acs`)."""
    from . import kit
    planned = plan(opt)
    ports = [planned.ports[f"floor:-x{s}"] for s in range(7)]
    lights = [_tag(planned, f"h1_{d}") for d in range(10)]
    lines = flip_specials(planned)
    wad = Path(wad)
    wad.parent.mkdir(parents=True, exist_ok=True)
    hook = world(ports, lights, lines, (selftest_cases(), planned) if selftest else None)
    brain = kit.compile(reader(), wad, world=hook, actions=_actions(), mode="map", guard=True, opt=opt,
                        configure=configure, resolution=resolution)
    if (brain.meta.ports, brain.meta.layout["read_expressions"]) != (planned.ports, planned.layout["read_expressions"]):
        raise AssertionError("museum: the planned and the built map number the ports or lights differently")
    where = finish(wad, brain.meta)
    brain.meta.layout["museum"] = dict(
        flip=dict(digit=FLIP[0], segment=FLIP[1], line_ids=[LID_LIGHT, LID_PLANE], specials=lines,
                  line_indices=[where["lines"]["light"], where["lines"]["plane"]]),
        cameras=where["cameras"], ports=ports, score_lights=lights, selftest=selftest)
    brain.meta.save(wad.with_suffix(".json"))
    return brain


def sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


BUILD_FILES = ("museum.wad", "museum.json", "museum_net.json", "reference.json")


def build(out: Path, *, opt: int = OPT) -> dict:
    """`doomformer museum build DIR`: DIR/museum.wad and museum.json (the map and its metadata), museum_net.json (the
    weights, biases and thresholds the compiler read), reference.json (every pad pattern with and without the
    rewiring: the scores and the digit) and SHA256SUMS of the four. Returns what the CLI prints."""
    from .acc import REV
    out = Path(out)
    brain = compile_museum(out / "museum.wad", opt=opt)
    net = brain.net
    (out / "museum_net.json").write_text(json.dumps(dict(
        note="df.MLP with relu input; weights[l] is (out, in); doomformer.museum.reader() returns it",
        weights=[w.tolist() for w in net.weights], biases=[b.tolist() for b in net.biases],
        thresholds=[t.tolist() for t in net.thresholds], signed_input=net.signed_input), indent=1) + "\n")
    (out / "reference.json").write_text(json.dumps(reference_table(), indent=1) + "\n")
    sums = {f: sha256(out / f) for f in BUILD_FILES}
    (out / "SHA256SUMS").write_text("".join(f"{h}  {f}\n" for f, h in sums.items()))
    m = brain.meta
    return dict(out=str(out), dolls=m.n_dolls, sectors=m.n_sectors, span=m.layout["span"], opt=opt, acc=REV,
                values=len(m.outputs), sha256=sums, estimated_settle_tics=m.layout["estimated_settle_tics"])


# --- the Python-free self-test (export) ---------------------------------------------------------------------------
LOG_TAG, CASE = "FACT", "museum"


def selftest_cases() -> list[tuple[int, bool]]:
    """(lit pads, rewired) per READ of the self-test: the digits 0..9, then MUTATE and a 7, then RESTORE and a 7."""
    return [(b, False) for b in SEGMENT_BITS] + [(SEGMENT_BITS[7], True), (SEGMENT_BITS[7], False)]


def expected_lines(names: list[str], cases=None) -> list[str]:
    """The self-test's FACT lines as the NumPy reference computes them (no engine)."""
    out = []
    for k, (bits, flip) in enumerate(cases or selftest_cases()):
        r = reference(bits, flip)
        out.append(f"{LOG_TAG} {CASE} case={k} flip={int(flip)} bits={bits} digit={r['digit']} fires=1"
                   + "".join(f" {n}={r['values'][n]}" for n in names))
    return out


def selftest_acs(cases, meta) -> str:
    """Script S_SELFTEST (OPEN, export only). Per case: set the pads (mu_pads, what stepping on them does), rewire if
    the case says so (mu_rewire, what MUTATE and RESTORE do), start READ (the switch's own script), wait until it
    counted this READ, the display fired and no doll moved for 8 tics, then log one line: the digit, the number of
    display firings and every value the Python check reads (the map's readout expressions, sorted by name). It
    writes no port and moves nothing itself."""
    from .tile import DOLL_TID0
    reads = meta.layout["read_expressions"]
    names = sorted(meta.outputs)
    n = meta.n_dolls
    log = (f's:"{LOG_TAG} {CASE} case=", d:k, s:" flip=", d:mu_flip, s:" bits=", d:mu_bits, s:" digit=", d:mu_digit, '
           f's:" fires=", d:mu_fires' + "".join(f', s:" {nm}=", d:{reads[nm]}' for nm in names))
    return f"""
// --- the self-test (export only, never in a default build) ---
int st_bits[{len(cases)}] = {{ {", ".join(str(b) for b, _ in cases)} }};
int st_flip[{len(cases)}] = {{ {", ".join(str(int(f)) for _, f in cases)} }};
int st_x[{n}]; int st_y[{n}]; int st_z[{n}];
function int st_moved(void) {{
  int moved = 0;
  for (int i = 0; i < {n}; i++) {{
    int tid = {DOLL_TID0} + i;
    if (!IsTIDUsed(tid)) {{ moved++; continue; }}
    int px = GetActorX(tid); int py = GetActorY(tid); int pz = GetActorZ(tid);
    if (px != st_x[i] || py != st_y[i] || pz != st_z[i]) {{ moved++; st_x[i] = px; st_y[i] = py; st_z[i] = pz; }}
  }}
  return moved;
}}
function void st_log(int k) {{ Log({log}); }}
script {S_SELFTEST} OPEN {{
  int k; int t; int m;
  Delay(8);
  for (k = 0; k < {len(cases)}; k++) {{
    if (st_flip[k] != mu_flip) mu_rewire(st_flip[k]);
    mu_pads(st_bits[k]);
    ACS_ExecuteAlways({S_READ}, 0);
    t = 0;
    while (mu_reads < k + 1 || mu_phase != {SHOWN}) {{
      Delay(1); t++;
      if (t > 40000) {{ Log(s:"{LOG_TAG} {CASE} case=", d:k, s:" timeout"); terminate; }}
    }}
    st_moved();
    do {{ Delay(8); m = st_moved(); }} while (m != 0);
    st_log(k);
  }}
  Delay(2);
  ConsoleCommand("quit");
}}
"""


def export(out: Path, *, opt: int = OPT, low_tags: bool = False) -> dict:
    """`doomformer museum export DIR`: DIR/museum_selftest.wad (the museum plus the self-test script) and
    DIR/expected.txt (its FACT lines from the NumPy reference). low_tags: every sector tag and TID below 32,768
    (Zandronum; ports_machine.tags_below_2_15)."""
    import contextlib

    from .ports_machine import _check_low, tags_below_2_15
    out = Path(out)
    with tags_below_2_15() if low_tags else contextlib.nullcontext():
        brain = compile_museum(out / "museum_selftest.wad", selftest=True, opt=opt)
    if low_tags:
        _check_low(out / "museum_selftest.wad")
    lines = expected_lines(sorted(brain.meta.outputs))
    (out / "expected.txt").write_text("\n".join(lines) + "\n")
    return dict(out=str(out), cases=len(lines), values=len(brain.meta.outputs), low_tags=low_tags,
                sha256=sha256(out / "museum_selftest.wad"))


def check_export(out: Path, timeout: int = 900) -> tuple[bool, list[str]]:
    """Run DIR/museum_selftest.wad in a fresh ViZDoom process with the console on stdout (facts' runner) and compare
    its FACT lines with expected.txt. Returns (all equal, the lines it logged)."""
    import os
    import subprocess
    import sys
    out = Path(out)
    want = [ln for ln in (out / "expected.txt").read_text().splitlines() if ln]
    r = subprocess.run([sys.executable, "-m", "doomformer.ports_machine", "--run-vizdoom",
                        str(out / "museum_selftest.wad"), "400000"], capture_output=True, text=True,
                       timeout=timeout, env=dict(os.environ, OMP_NUM_THREADS="1"))
    got = [ln.strip() for ln in (r.stdout + r.stderr).splitlines() if ln.strip().startswith(f"{LOG_TAG} {CASE} ")]
    return got == want, got


# --- playing it headless: Python stands in for the player -------------------------------------------------------
@dataclass
class Read:
    """One verified READ: what DOOM computed and how it was checked."""
    bits: int                  # the pads the ports hold (bit s = segment s)
    flip: bool                 # w_7a rewired (the level's flag) when the READ ran
    digit: int                 # the digit the winner's doll lit on the display
    scores: list[int]          # s_0..s_9 as DOOM holds them (the score lights)
    checked: int               # values compared with the NumPy reference (all equal, else MismatchError)
    tics: int                  # game tics from the READ press until every doll stood still
    status: dict
    audit: dict                # ReadoutOnly: every console command Python sent during the READ


class Museum(Walker):
    """The museum map and its live game, headless: `light` walks the marine onto pads, `press` presses a switch with
    USE, `verify` watches a READ through ReadoutOnly and compares every value with the reference. wad: a built
    museum.wad (its museum.json next to it); default: build one into `workdir` (a temporary directory)."""

    BUTTONS = drawpad.DrawPad.BUTTONS

    def __init__(self, wad=None, *, workdir=None, resolution=None, screen: bool = False):
        """screen=True: RGB frames with the HUD messages (for `shots`)."""
        import tempfile

        from . import kit
        from .tile import TileMeta
        self._tmp, self.screen = None, screen
        if wad is None:
            if workdir is None:
                self._tmp = tempfile.TemporaryDirectory(prefix="doomformer-museum-")
                workdir = self._tmp.name
            self.brain = compile_museum(Path(workdir) / "museum.wad", configure=self._configure,
                                        resolution=resolution)
        else:
            meta = TileMeta.load(Path(wad).with_suffix(".json"))
            if "museum" not in meta.layout or meta.layout["museum"]["selftest"]:
                raise ValueError(f"{wad}: not a museum.wad from `doomformer museum build`")
            self.brain = kit.MLPBrain(reader(), Path(wad), meta, resolution=resolution, configure=self._configure)
        self.path: list[tuple[float, float, bool]] = []
        self._game = None
        self._on_pad: int | None = None               # the pad the marine stands on (he leaves it the way he came)

    def _configure(self, g):
        import vizdoom as vzd
        for b in self.BUTTONS:
            g.add_available_button(getattr(vzd.Button, b))
        for v in ("POSITION_X", "POSITION_Y", "ANGLE", "VELOCITY_X", "VELOCITY_Y"):
            g.add_available_game_variable(getattr(vzd.GameVariable, v))
        if self.screen:
            g.set_screen_format(vzd.ScreenFormat.RGB24)
            g.set_render_messages(True)

    @property
    def game(self):
        if self._game is None:
            self._game = self.brain.runner.game
            self._game.advance_action(2)                    # ENTER: the player gets his TID
        return self._game

    def close(self):
        self.brain.close()
        if self._tmp is not None:
            self._tmp.cleanup()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    # --- observing: every readout is guarded ---------------------------------------------------------------------
    def _users(self, g, script: int, n: int) -> list[int]:
        """Run readout script `script` and return USER1..n as it wrote them. First script 9 writes POISON into every
        USER variable and the host waits until it sees it; then the readout runs and the host waits until it replaced
        the poison. A guard or readout that never runs raises (after ACK_TICS tics) instead of returning old values."""
        from .runtime import USER

        def until(test, what):
            for _ in range(ACK_TICS):
                g.advance_action(1)
                vals = [int(g.get_game_variable(USER[i])) for i in range(n)]
                if test(vals):
                    return vals
            raise RuntimeError(f"museum: {what} did not run within {ACK_TICS} tics")

        g.send_game_command(f"puke -{GUARD_SCRIPT} {POISON}")
        until(lambda v: all(x == POISON for x in v), f"the readout guard before script {script}")
        g.send_game_command(f"puke -{script}")
        return until(lambda v: all(x != POISON for x in v), f"readout script {script}")

    def status(self, g=None) -> dict:
        """The level's state (one readout): phase, reads, digit (-1: none), fires, bits (lit pads), flip, ..."""
        return dict(zip(STATUS_NAMES, self._users(g or self.game, S_STATUS, len(STATUS_NAMES))))

    def ports(self, g=None) -> list[int]:
        """The seven input ports' floor heights (-x_s) as they are in the map."""
        return self._users(g or self.game, S_PORTS, 7)

    def read_brain(self, g) -> dict[str, int]:
        """Every value the brain's readout pages expose (scripts 100.., one guarded readout each)."""
        out = {}
        for pi, page in enumerate(self.brain.meta.pages):
            out.update(zip(page, self._users(g, 100 + pi, len(page))))
        return out

    def state(self) -> tuple[float, float, float, float, float]:
        import vizdoom as vzd
        g = self.game
        return tuple(float(g.get_game_variable(getattr(vzd.GameVariable, v))) for v in (
            "POSITION_X", "POSITION_Y", "ANGLE", "VELOCITY_X", "VELOCITY_Y"))

    # --- the stand-in player ---------------------------------------------------------------------------------------
    def tic(self, forward: float = 0, strafe: float = 0, turn: float = 0, fire: bool = False, use: bool = False):
        g = self.game
        g.set_action([forward, strafe, turn, float(fire), float(use)])
        g.advance_action(1)
        x, y, _ = self.where()
        self.path.append((x, y, bool(fire)))

    SOUTH_LANE = -3800                                   # walking lanes that pass every pad without touching one

    def _go(self, target: tuple[float, float], via_pad: int | None = None):
        """Walk to `target` over the safe lanes (x = +-COLUMN, y = SOUTH_LANE); with via_pad the last leg enters pad
        via_pad from its side and the next walk leaves it the same way."""
        x, y, _ = self.where()
        pts = [(x, y)]
        if self._on_pad is not None:
            pts.append(self._approach(self._on_pad))
        elif y < self.SOUTH_LANE:
            pts.append((x, self.SOUTH_LANE))
        end = self._approach(via_pad) if via_pad is not None else (target[0], self.SOUTH_LANE)
        sx = pts[-1][0]
        if not (abs(sx) == COLUMN and sx == end[0]):
            pts += [(sx, self.SOUTH_LANE), (end[0], self.SOUTH_LANE)]
        pts += [end, target]
        self.walk(pts)
        self._on_pad = via_pad

    @staticmethod
    def _approach(s: int) -> tuple[float, float]:
        """Where the marine turns into pad s: on the column beside it (east for a, b, c, d, g; west for e, f)."""
        return (-COLUMN if s in (4, 5) else COLUMN), pad_centre(s)[1]

    def light(self, bits: int) -> None:
        """Walk onto every pad whose state differs from `bits` (each step onto a pad toggles it)."""
        have = self.status()["bits"]
        for s in range(7):
            if (have ^ bits) >> s & 1:
                self._go(pad_centre(s), via_pad=s)
        got = self.status()["bits"]
        if got != bits:
            raise RuntimeError(f"museum: the pads show {got:07b}, wanted {bits:07b}")

    def press(self, name: str) -> None:
        """Walk to the podium `name` (PODIUMS), face it and press USE for one tic, then let go of every key."""
        a, b = podium_x(PODIUMS.index(name))
        self._go(((a + b) / 2, PODIUM_N + 28))
        self.turn_to(270)
        self.tic(use=True)
        self.game.set_action([0.0] * len(self.BUTTONS))

    # --- the READ: observe only ------------------------------------------------------------------------------------
    def verify(self, reads: int) -> Read:
        """Watch READ number `reads` through ReadoutOnly (tic advances and readout scripts, nothing else) until every
        doll and the marine stood still, then compare every value DOOM computed with the NumPy integer reference of
        the network as the level is wired now (its flip flag): the 140 network values, the 7 ports against the lit
        pads, and the display (it fired once, for the reference's winner). Raises df.MismatchError on a difference."""
        from .kit import MismatchError, ReadoutError
        from .readout import ReadoutOnly, readout_commands
        from .tile import MOTION_SCRIPT
        runner, raw = self.brain.runner, self.game
        g = ReadoutOnly(raw, readout_commands(len(runner.meta.pages), f"puke -{GUARD_SCRIPT} {POISON}",
                                              *(f"puke -{s}" for s in (S_STATUS, S_PORTS, MOTION_SCRIPT))))
        runner.game = g
        tic0 = raw.get_episode_time()
        try:
            for _ in range(64):
                st = self.status(g)
                if st["reads"] == reads and st["phase"] >= THINKING:
                    break
            else:
                raise RuntimeError(f"READ {reads} did not start the brain: {st}")
            bits = st["bits"]
            runner.settle()
            tics = int(g.get_episode_time() - tic0)
            got = self.read_brain(g)
            ports = self.ports(g)
            st = self.status(g)
        finally:
            runner.game = raw
        net = reader(bool(st["flip"]))
        if set(got) != net.exposed_names():
            raise ReadoutError(f"museum READ {reads}: read {len(got)} values, the lowering exposes "
                               f"{len(net.exposed_names())}")
        x = [-v for v in ports]
        bad = [(f"x{s}", v, (bits >> s) & 1) for s, v in enumerate(x) if v != (bits >> s) & 1]
        if st["reads"] != reads or st["bits"] != bits:
            bad.append(("pads changed during the READ", st["bits"], bits))
        ref = net.reference(x_of(bits))
        bad += [(k, v, net._expected(k, ref)) for k, v in got.items() if v != net._expected(k, ref)]
        want = int(np.argmax(ref["y"]))
        if st["fires"] != 1 or st["digit"] != want:
            bad.append(("display", st["digit"], want))
        if bad:
            raise MismatchError(f"museum READ {reads}", bad)
        return Read(bits=bits, flip=bool(st["flip"]), digit=st["digit"], scores=[got[f"h1_{d}"] for d in range(10)],
                    checked=len(got) + len(x) + 1, tics=tics, status=st, audit=g.audit())

    def read(self) -> Read:
        """Press READ and verify the pass."""
        reads = self.status()["reads"] + 1
        self.press("read")
        return self.verify(reads)

    def tour(self, bits: int = SEGMENT_BITS[7], on_read=None) -> list[Read]:
        """The museum's experiment: light `bits` (default the 7: a, b, c), READ; MUTATE, READ; RESTORE, READ. Every
        READ is verified against the network as wired at that moment; returns the three Reads."""
        out = []
        self.light(bits)
        for step in (None, "mutate", "restore"):
            if step:
                flips = self.status()["mutations"]
                self.press(step)
                if self.status()["mutations"] != flips + 1:
                    raise RuntimeError(f"museum: {step.upper()} did not rewire the connection")
            out.append(self.read())
            if on_read:
                on_read(step, out[-1])
        return out

    def look(self, name: str) -> np.ndarray:
        """Press camera station `name` (cam_a, cam_b, cam_c), let the view settle and return the frame (screen=True)."""
        self.press(name)
        self.tic()
        self.tic()
        return np.asarray(self.game.get_state().screen_buffer)

    def frame(self) -> np.ndarray:
        """The marine's view from the start spot, facing north (screen=True)."""
        self._go(START)
        self.turn_to(90)
        self.tic()
        return np.asarray(self.game.get_state().screen_buffer)


def shots(out: Path, wad=None) -> list[Path]:
    """`doomformer museum shots DIR`: PNG frames of the experiment, headless. Light a 7 and READ; the hall and the
    three camera stations; MUTATE and READ; the hall and the cameras again. Every READ is verified as in `verify`."""
    import vizdoom as vzd
    from PIL import Image
    out = Path(out)
    paths = []

    def save(name: str, img: np.ndarray):
        paths.append(out / f"{name}.png")
        Image.fromarray(img).save(paths[-1])

    with Museum(wad, resolution=vzd.ScreenResolution.RES_640X480, screen=True) as m:
        m.light(SEGMENT_BITS[7])
        for when in ("wired", "rewired"):
            if when == "rewired":
                m.press("mutate")
            m.read()
            save(f"hall_{when}", m.frame())
            for cam in ("cam_a", "cam_b", "cam_c"):
                save(f"{cam}_{when}", m.look(cam))
    return paths

