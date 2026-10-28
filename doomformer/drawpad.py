"""DrawPad: draw a digit by walking on a DOOM floor; the level reads it with voodoo dolls and shows the answer.

One map holds the MNIST MLP *and* the room it reads from (`world`): a 28 x 28 canvas of floor tiles (TILE units each,
one sector per pixel), a READ and a CLEAR switch, and a stepped 7-segment display north of the canvas.

* Drawing. Every tile sector holds a SecActEnter thing whose special starts script TILE with the tile's (row, col):
  the script ignores every activator but the player (his TID, given by the ENTER script; voodoo dolls have their
  own TIDs) and, while the player holds FIRE, stamps a BRUSH x BRUSH block of pixels into the canvas array and
  lights those tiles. A PEN loop (OPEN script) stamps the tile he stands on on every tic FIRE is held, so pressing
  FIRE without moving paints too. FIRE up = pen up. The pen is off from READ until the display lights.
* READ (a USE switch) runs the brain's own scripts: 3 (rewind the dolls, as between any two passes), then script 1
  once per pixel port (the same port writer TileRunner uses: Floor_MoveToValue on the gate, -16 for ink, 0 for
  paper), then 7 (drop the start barriers), and shows the seconds since on the HUD. The level does all of it;
  nothing comes from Python. READ and CLEAR are refused (a HUD note) until the display lights.
* The network ends in winner layers (`with_argmax`): dolls compare every pair of scores and count wins, so exactly
  one output y_d is 1. Its doll crosses an ACS_ExecuteAlways line (the `actions` of y_d) that lights digit d on the
  7-segment display. CLEAR rewinds the brain, clears the canvas and the display.

Python's part during a READ pass (`DrawPad.verify`): advance tics and run readout scripts only, through
doomformer.readout.ReadoutOnly (any other game call raises). Afterwards it reads back the 784 pixel ports exactly as
the level wrote them, recomputes every value with the NumPy integer reference from those pixels, and raises
df.MismatchError on any difference. Every readout script is followed by the ACK script and read once ACK has run
(`DrawPad._users`): in spectator mode a command can run tics after the advance that should have run it.

Before READ, headless runs (`doomformer draw --digit`) have Python stand in for a human: `Walker` presses the
marine's keys to walk `walked_strokes` (randomised DIGIT_STROKES). `SimWalker` replays the same keys with DOOM's
measured player physics in Python; training/train_mnist.py paints walked training digits with it (`walked_pixels`).
`doomformer draw` (spectator=True) opens a window where a person plays and `watch` checks every READ.
See docs/drawpad.md.
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass

import numpy as np

from .intref import PORT_MAX, W_MAX

SIZE = 28                 # canvas pixels per side (MNIST)
TILE = 32                 # map units per pixel (the marine is 32 wide)
BRUSH = 3                 # stamp BRUSH x BRUSH pixels around the tile under the marine (MNIST-thick strokes)
SPEED, LOOKAHEAD = 8.0, 24.0  # stand-in walker: map units per tic, pursuit lookahead (map units)
EDGE = 0.1                    # map units: a position this close to a tile edge may count for either tile (explains)
# measured: a MOVE_*_DELTA value V adds V/32 (forward) or V/33.1 (strafe) units/tic to the velocity before
# ZDoom's friction (x 0.90625 per tic); |V| <= 50 stays in that linear range
FRICTION, ACCEL_FWD, ACCEL_SIDE, V_MAX = 0.90625, 1 / 32, 1 / 33.1, 50.0


# --- winner layers: DOOM picks the digit ------------------------------------------------------------------------------
def with_argmax(net):
    """The integer MLP `net` (last layer: n scores s = b + W a) extended so that DOOM outputs the one-hot of the
    first maximal score (np.argmax: lowest index wins ties):

        d_ij = s_i - s_j + 1  for i < j     (one row per pair, computed from a, never from re-read scores)
        c_ij = [d_ij >= 1] = [s_i >= s_j]   (staircase, threshold [1])
        w_i  = sum_{j>i} c_ij - sum_{j<i} c_ji + i   = #{j>i: s_i >= s_j} + #{j<i: s_i > s_j}
        z_i  = [w_i >= n-1]                 (staircase, threshold [n-1]): 1 only for the first maximum
        y    = I z                          (identity output layer: `actions` on y_i fire for the winner only)

    d is a gate-fed value, so every pair row must fit |b_i - b_j + 1| + sum_k |W_ik - W_jk| * K <= 12,000 (K = levels
    of the staircase feeding the scores) and |W_ik - W_jk| <= 255. If the score layer does not fit, W and b are
    multiplied by the largest f (in 0.5 % steps) for which rint(f W), rint(f b) fit; the result then picks the argmax
    of those rescaled integer scores (`net.score_scale` = f, 1.0 when unchanged). Returns a new df.MLP with
    n_layers + 3 weight matrices; the input encoding (in_scale, input_shape) is kept."""
    from .kit import MLP
    W, b = net.weights[-1].astype(np.int64), net.biases[-1].astype(np.int64)
    n, K = W.shape[0], len(net.thresholds[-1])
    if n < 2:
        raise ValueError("with_argmax needs at least two scores")
    I, J = np.triu_indices(n, 1)                       # pairs i < j, lexicographic

    def pair_layer(f):
        Wq, bq = np.rint(W * f).astype(np.int64), np.rint(b * f).astype(np.int64)
        Wd, bd = Wq[I] - Wq[J], bq[I] - bq[J] + 1
        ok = np.abs(Wd).max(initial=0) <= W_MAX and (np.abs(bd) + np.abs(Wd).sum(1) * K).max() <= PORT_MAX
        return ok, Wd, bd

    f = 1.0
    ok, Wd, bd = pair_layer(f)
    while not ok:
        f *= 0.995
        ok, Wd, bd = pair_layer(f)
    Wwin = np.zeros((n, len(I)), dtype=np.int64)
    Wwin[I, np.arange(len(I))] = 1
    Wwin[J, np.arange(len(I))] = -1
    out = MLP(net.weights[:-1] + [Wd, Wwin, np.eye(n, dtype=np.int64)],
              net.biases[:-1] + [bd, np.arange(n, dtype=np.int64), np.zeros(n, dtype=np.int64)],
              net.thresholds + [np.array([1]), np.array([n - 1])], signed_input=net.signed_input)
    out.in_scale, out.input_shape = net.in_scale, net.input_shape
    out.out_scale = 1.0 if net.in_scale is not None else None      # the one-hot is its own float value
    out.score_scale = f
    return out


# --- digits drawn as walked strokes -----------------------------------------------------------------------------------
def _arc(cx, cy, rx, ry, a0, a1, n=None):
    """Points of an ellipse arc in the 20-unit-tall stroke frame (x right, y DOWN), angles in degrees, counter-
    clockwise as seen on screen for a1 > a0."""
    n = n or max(4, int(abs(a1 - a0) / 15))
    a = np.radians(np.linspace(a0, a1, n + 1))
    return [(cx + rx * math.cos(t), cy - ry * math.sin(t)) for t in a]


def _strokes() -> dict[int, list[list[list[tuple[float, float]]]]]:
    """DIGIT_STROKES[d][variant] = strokes; a stroke is a polyline in a frame 20 units tall and about 12 wide (x right,
    y down); the pen is lifted between strokes."""
    A = _arc
    return {
        0: [[A(6, 10, 6, 10, 90, 450)], [A(6, 10, 5.5, 10, 90, -290)], [A(6, 10, 4.5, 10, 100, 470)]],
        1: [[[(6, 0), (6, 20)]], [[(3, 4), (6.5, 0), (6.5, 20)]],
            [[(3, 4), (6.5, 0), (6.5, 20), (3.5, 20), (9.5, 20)]]],
        2: [[A(6, 5.5, 5.5, 5.5, 160, -20) + [(0.5, 20), (12, 20)]],
            [A(6, 5, 5, 5, 170, -40) + [(1, 20), (12, 19.5)]],
            [[(1, 2), (5, 0), (10, 1), (11, 5), (1, 20), (12, 20)]]],
        3: [[A(5.5, 5, 5, 5, 150, -90) + A(5.5, 15, 5.5, 5, 90, -150)],
            [[(1, 0), (11, 0)] + A(5.5, 14, 5.5, 6, 90, -150)],
            [A(6, 5, 5, 5, 160, -90) + A(6, 15, 6, 5, 90, -160)]],
        4: [[[(7, 0), (0, 13), (12, 13)], [(9, 4), (9, 20)]],
            [[(9, 20), (9, 0), (0, 13), (12, 13)]],
            [[(2, 0), (1, 11), (12, 11)], [(9, 0), (9, 20)]]],
        5: [[[(11, 0), (2, 0), (1.5, 9)] + A(5.5, 14, 5.5, 6, 130, -150)],
            [[(2, 0), (1.5, 9)] + A(5.5, 14, 5.5, 6, 130, -150), [(2, 0), (11, 0)]],
            [[(10, 0), (2, 0), (2, 8)] + A(6, 13.5, 5, 6.5, 120, -160)]],
        6: [[[(9, 0), (5, 2), (2, 6), (0.5, 12)] + A(6, 14.5, 5.5, 5.5, 180, 520)],
            [[(10, 1), (4, 3), (1, 10)] + A(6, 14, 5, 6, 180, 530)],
            [[(8, 0), (2, 10)] + A(6, 15, 4.5, 5, 200, 540)]],
        7: [[[(0, 0), (12, 0), (4, 20)]],
            [[(0, 0), (12, 0), (5, 20)], [(3, 10), (11, 10)]],
            [[(0, 2), (1, 0), (12, 0), (8, 8), (6, 20)]]],
        8: [[A(6, 5, 5, 5, 30, 270) + A(6, 15, 5.5, 5, 90, -270) + A(6, 5, 5, 5, -90, 30)],
            [A(6, 5, 4.5, 5, 90, 450), A(6, 15, 5.5, 5, 90, 450)],
            [[(11, 1), (6, 0), (1, 2), (1, 6), (11, 14), (11, 18), (6, 20), (1, 18), (1, 14), (11, 6), (11, 2)]]],
        9: [[A(6, 5, 5, 5, 0, 360) + [(11, 5), (11, 20)]],
            [A(6, 5, 5, 5, -10, 350) + [(11, 5), (9, 20)]],
            [A(6, 5.5, 5, 5.5, 0, 360) + [(11, 5.5), (10, 15), (7, 20), (2, 19)]]],
    }


DIGIT_STROKES = _strokes()


def _densify(p: np.ndarray, step: float) -> np.ndarray:
    seg = np.linalg.norm(np.diff(p, axis=0), axis=1)
    s = np.concatenate([[0.0], np.cumsum(seg)])
    if s[-1] == 0:
        return p[:1]
    t = np.linspace(0, s[-1], max(2, int(math.ceil(s[-1] / step)) + 1))
    return np.stack([np.interp(t, s, p[:, 0]), np.interp(t, s, p[:, 1])], 1)


def walked_strokes(digit: int, rng: np.random.Generator, variant: int | None = None) -> list[np.ndarray]:
    """One randomised drawing of `digit` as strokes in canvas pixel coordinates ((col, row) floats, (0, 0) the
    north-west corner of the canvas): a DIGIT_STROKES variant scaled to 14-21 pixels tall, stretched, rotated,
    slanted, wobbled (smooth noise) and placed off-centre by up to 3 pixels, the way a marine walking it might."""
    variants = DIGIT_STROKES[int(digit)]
    v = variants[rng.integers(len(variants)) if variant is None else variant]
    h = rng.uniform(14, 21) / 20
    sx, th, shear = h * rng.uniform(0.8, 1.25), math.radians(rng.uniform(-12, 12)), rng.uniform(-0.25, 0.25)
    rot = np.array([[math.cos(th), -math.sin(th)], [math.sin(th), math.cos(th)]])
    out = []
    for stroke in v:
        p = np.asarray(stroke, dtype=np.float64) - (6, 10)
        p = np.stack([p[:, 0] * sx + shear * p[:, 1] * h, p[:, 1] * h], 1) @ rot.T
        p = _densify(p, 0.5)
        knots = max(2, int(len(p) / 8) + 2)                              # smooth wobble, about 0.6 px
        noise = rng.normal(0, 0.6, (knots, 2))
        t = np.linspace(0, knots - 1, len(p))
        p = p + np.stack([np.interp(t, np.arange(knots), noise[:, k]) for k in (0, 1)], 1)
        out.append(p)
    allp = np.concatenate(out)
    lo, hi = allp.min(0), allp.max(0)
    centre = np.array([SIZE / 2, SIZE / 2]) + rng.uniform(-3, 3, 2)
    shift = np.clip(centre - (lo + hi) / 2, 1.5 - lo, SIZE - 1.5 - hi)
    return [p + shift for p in out]


def stamp(tiles) -> np.ndarray:
    """The canvas after stamping the BRUSH x BRUSH brush at every (row, col) in `tiles` (outside cells clipped):
    exactly what the level's TILE/PEN scripts do. Returns 784 0/1 pixels (row-major, row 0 = north)."""
    img = np.zeros((SIZE, SIZE), dtype=np.uint8)
    r0 = (BRUSH - 1) // 2
    for r, c in tiles:
        if 0 <= r < SIZE and 0 <= c < SIZE:
            img[max(0, r - r0):r - r0 + BRUSH, max(0, c - r0):c - r0 + BRUSH] = 1
    return img.reshape(-1)


def rasterise(strokes) -> np.ndarray:
    """Pixels of strokes ((col, row) canvas coordinates) as the canvas paints them: every tile the pen passes over
    (polylines sampled every 0.2 pixel) stamped with the brush."""
    tiles = set()
    for p in strokes:
        q = _densify(np.asarray(p, dtype=np.float64), 0.2)
        tiles |= {(int(math.floor(y)), int(math.floor(x))) for x, y in q}
    return stamp(tiles)


# --- the level --------------------------------------------------------------------------------------------------------
X0, Y1 = -448, -2816        # north-west corner of the canvas; pixel (row r, col c) is x X0+32c.., y Y1-32(r+1)..
X1, Y0 = X0 + SIZE * TILE, Y1 - SIZE * TILE
WEST, EAST, SOUTH, NORTH = -576, 576, -3904, -2080      # the whole world (inside df.plan's world area, y <= -2048)
START = (0, -3840)          # the marine starts here facing north, the canvas and the display in front of him
PODIUM_Y = (-3776, -3744)   # two 32-high podiums (too high to step on, low enough to look over); switches south
READ_X, CLEAR_X = (64, 192), (-192, -64)
GUIDE = range(4, 24)        # MNIST's 20 x 20 box, lit a little brighter on the canvas
SKY = 1024
# 7-segment display: a staircase of cells (rows rise towards the north) so it faces the canvas
B_ROWS, B_COLS, B_W, B_D, B_RISE = 14, 8, 96, 40, 64
B_X0, B_S = -B_COLS * B_W // 2, -2640                  # display x0 and its south (front) edge
SEGMENT_BITS = (0x3F, 0x06, 0x5B, 0x4F, 0x66, 0x6D, 0x7D, 0x07, 0x7F, 0x6F)    # abcdefg = bits 0..6
PLAYER_TID = 900
TILE_TAG0, SEG_TAG0 = 100, 10                        # world tags: 100..883 pixels, 10..16 segments a..g
S_ENTER, S_TILE, S_LEAVE, S_PEN, S_READ, S_CLEAR, S_SHOW, S_STATUS, S_CANVAS, S_ACK = range(200, 210)   # level scripts
S_PIXELS = 210              # pixel port readout pages 210..223
IDLE, WRITING, THINKING, SHOWN = 0, 1, 2, 3          # dp_phase
PAPER_LIGHT, GUIDE_LIGHT, INK_LIGHT = 104, 136, 255
PAPER, INK, WALK = "FLAT1_1", "FLAT14", "FLOOR4_8"
SEG_OFF, SEG_ON, SEG_BG = (112, "FLAT23"), (255, "CEIL5_1"), (48, "FLAT5_1")
SECACT_ENTER, ACS_EXECUTE_ALWAYS = 9998, 226
STATUS_NAMES = ("phase", "reads", "digit", "fires", "ink", "row", "col", "think_start", "timer", "clears")
ACK_TICS = 64               # PLAYER mode: tics a readout may wait for its ack before the readout counts as lost
CONTROLS = ("bind w +forward", "bind s +back", "bind a +moveleft", "bind d +moveright", "bind e +use",
            "m_forward 0")    # spectator window: WASD and E besides DOOM's arrows, Ctrl, Space; the mouse only turns


def segment(r: int, c: int) -> int | None:
    """7-segment index (0..6 = a..g) of display cell (row r from the top, col c), None for background. Segments are
    two cells thick and leave the corners dark, like a real display."""
    if 2 <= c < B_COLS - 2:
        return {0: 0, 1: 0, 6: 6, 7: 6, 12: 3, 13: 3}.get(r)
    if c < 2:
        return 5 if 2 <= r < 6 else 4 if 8 <= r < 12 else None
    return 1 if 2 <= r < 6 else 2 if 8 <= r < 12 else None


def pixel_ports(net, ports: dict[str, int] | None = None) -> tuple[list[int], int]:
    """(tag of pixel i's input port for i in 0..783, the port value for ink). The net must read 784 relu inputs
    (`signed_input=False`) whose ink encodes to one integer, so every pixel is one floor port holding -x (pure:
    floor gate fed -x); `ports` (meta.ports of a compiled map) defaults to compile_pure's numbering (PORT_TAG0 + order
    of the first layer's ports)."""
    from .tile import PORT_TAG0
    if net.n_in != SIZE * SIZE or net.signed_input or net.n_out != 10:
        raise ValueError("DrawPad needs a 784-pixel relu-input net with 10 outputs (models/mnist/mlp.npz)")
    ink = net.encode(np.ones(net.n_in)) if net.in_scale is not None else np.ones(net.n_in, dtype=np.int64)
    if (ink != ink[0]).any() or ink[0] <= 0 or (net.encode(np.zeros(net.n_in)) != 0).any():
        raise ValueError("DrawPad needs every pixel's ink to encode to the same positive integer and paper to 0")
    if ports is None:
        keys = net._stages()[0].ports
        ports = {k: PORT_TAG0 + i for i, k in enumerate(keys)}
    tags = [ports.get(f"floor:-x{i}") for i in range(net.n_in)]
    missing = [i for i, t in enumerate(tags) if t is None]
    if missing or len(ports) != net.n_in:
        raise ValueError(f"DrawPad needs a weight on every pixel; pixels {missing[:8]} have no input port")
    return tags, int(ink[0])


MEASURED_READ_TICS = {0: 5705, 3: 273}     # READ press to every doll still, shipped net (docs/drawpad.md, Measurements)


def world(tags: list[int], ink: int, read_s: int | None = None):
    """The DrawPad world hook for df.compile: canvas, walkways, podium switches, the 7-segment display and the
    level's ACS (see the module docstring)."""

    def build(m) -> str:
        room = dict(ceil=SKY, ctex="F_SKY1", wall="STARTAN3")
        walk = dict(floor=0, light=160, ftex=WALK, **room)
        walkways = []
        for r in range(SIZE):
            for c in range(SIZE):
                x, y = X0 + c * TILE, Y1 - (r + 1) * TILE
                lit = GUIDE_LIGHT if r in GUIDE and c in GUIDE else PAPER_LIGHT
                m.rect(x, y, x + TILE, y + TILE, floor=0, light=lit, tag=TILE_TAG0 + r * SIZE + c, ftex=PAPER, **room)
                m.thing(x + TILE // 2, y + TILE // 2, SECACT_ENTER, special=ACS_EXECUTE_ALWAYS, arg0=S_TILE, arg2=r,
                        arg3=c)
        walkways += [m.rect(WEST, Y0, X0, Y1, **walk), m.rect(X1, Y0, EAST, Y1, **walk),
                     m.rect(WEST, Y1, EAST, B_S, **walk)]
        bx1 = B_X0 + B_COLS * B_W
        walkways += [m.rect(WEST, B_S, B_X0, NORTH, **walk), m.rect(bx1, B_S, EAST, NORTH, **walk)]
        for r in range(B_ROWS):                       # row 0 = top of the digit = farthest north and highest
            y = B_S + (B_ROWS - 1 - r) * B_D
            for c in range(B_COLS):
                s = segment(r, c)
                light, tex = SEG_BG if s is None else SEG_OFF
                m.rect(B_X0 + c * B_W, y, B_X0 + (c + 1) * B_W, y + B_D, floor=(B_ROWS - r) * B_RISE, light=light,
                       tag=0 if s is None else SEG_TAG0 + s, ftex=tex, ceil=SKY, ctex="F_SKY1", wall="STARTAN3",
                       facing="STARTAN3")
        p0, p1 = PODIUM_Y
        walkways += [m.rect(WEST, SOUTH, EAST, p0, **walk), m.rect(WEST, p1, EAST, Y0, **walk)]
        xs = [WEST, CLEAR_X[0], CLEAR_X[1], READ_X[0], READ_X[1], EAST]
        south = walkways[-2]
        for k, (a, b) in enumerate(zip(xs, xs[1:])):
            if k in (1, 3):                           # the podiums: CLEAR (west) and READ (east)
                pod = m.rect(a, p0, b, p1, floor=32, light=200, ftex="CEIL5_1", facing=("SW1COMP", "SW1COMP")
                             if k == 3 else ("SW1GRAY", "SW1GRAY"), **room)
                m.link(south, pod, special=ACS_EXECUTE_ALWAYS, arg0=S_READ if k == 3 else S_CLEAR, playeruse=True,
                       repeatspecial=True)
            else:
                walkways.append(m.rect(a, p0, b, p1, **walk))
        for w in walkways:
            r = m.rects[w]
            m.thing((r.x0 + r.x1) / 2, (r.y0 + r.y1) / 2, SECACT_ENTER, special=ACS_EXECUTE_ALWAYS, arg0=S_LEAVE)
        m.thing(*START, 1, angle=90)                   # the player: the LAST player-1 start
        return level_acs(tags, ink, read_s)

    return build


def level_acs(tags: list[int], ink: int, read_s: int | None = None) -> str:
    """The level's scripts (numbers >= 200): ENTER, TILE/LEAVE (sector actions), PEN, READ, CLEAR, SHOW (fired by the
    winner doll), STATUS, CANVAS, ACK (dp_ack++, ACS global 0, which ViZDoom reports as the map reward: the host sends
    it after every readout script and reads only once it has moved) and the pixel readout pages (S_PIXELS + p:
    o1..o60 := floor heights of 60 pixel ports)."""
    from .tile import PAGE
    of_about = f" of about {read_s} s" if read_s else ""
    n, r0 = SIZE * SIZE, (BRUSH - 1) // 2
    pages = []
    for p in range(-(-n // PAGE)):
        body = " ".join(f"o{k + 1} = GetSectorFloorZ({tags[i]}, 0, 0) >> 16;"
                        for k, i in enumerate(range(p * PAGE, min(n, (p + 1) * PAGE))))
        pages.append(f"script {S_PIXELS + p} (void) {{ {body} }}")
    status = " ".join(f"o{k + 1} = {e};" for k, e in enumerate(
        ("dp_phase", "dp_reads", "dp_digit", "dp_fires", "dp_ink", "dp_r", "dp_c", "dp_t0", "Timer()", "dp_clears")))
    return f"""
// --- DrawPad level (doomformer/drawpad.py) ---
int dp_px[{n}];
int dp_port[{n}] = {{ {", ".join(map(str, tags))} }};
int dp_seg[10] = {{ {", ".join(map(str, SEGMENT_BITS))} }};
int dp_phase; int dp_reads; int dp_digit = -1; int dp_fires; int dp_ink; int dp_r = -1; int dp_c = -1;
int dp_t0; int dp_clears;
global int 0:dp_ack;

function int dp_paper(int i) {{
  int r = i / {SIZE}; int c = i % {SIZE};
  if (r >= {GUIDE.start} && r < {GUIDE.stop} && c >= {GUIDE.start} && c < {GUIDE.stop}) return {GUIDE_LIGHT};
  return {PAPER_LIGHT};
}}
function void dp_paint(int r, int c) {{
  int dr; int dc; int i;
  for (dr = -{r0}; dr < {BRUSH - r0}; dr++)
    for (dc = -{r0}; dc < {BRUSH - r0}; dc++)
      if (r + dr >= 0 && r + dr < {SIZE} && c + dc >= 0 && c + dc < {SIZE}) {{
        i = (r + dr) * {SIZE} + c + dc;
        if (!dp_px[i]) {{
          dp_px[i] = 1; dp_ink++;
          Light_ChangeToValue({TILE_TAG0} + i, {INK_LIGHT}); ChangeFloor({TILE_TAG0} + i, "{INK}");
        }}
      }}
}}
function int dp_row(int r) {{             // canvas row r as bits (readout only)
  int c; int bits = 0;
  for (c = 0; c < {SIZE}; c++) if (dp_px[r * {SIZE} + c]) bits |= 1 << c;
  return bits;
}}
function void dp_segments(int bits) {{
  int k;
  for (k = 0; k < 7; k++) {{
    if (bits & (1 << k)) {{
      Light_ChangeToValue({SEG_TAG0} + k, {SEG_ON[0]}); ChangeFloor({SEG_TAG0} + k, "{SEG_ON[1]}");
    }} else {{
      Light_ChangeToValue({SEG_TAG0} + k, {SEG_OFF[0]}); ChangeFloor({SEG_TAG0} + k, "{SEG_OFF[1]}");
    }}
  }}
}}
function int dp_busy(void) {{ return dp_phase == {WRITING} || dp_phase == {THINKING}; }}
function void dp_wait(void) {{
  HudMessageBold(s:"DOOM is still reading: wait for the display"; HUDMSG_PLAIN, 2, CR_ORANGE, 0.5, 0.2, 2.0);
}}
script {S_ENTER} ENTER {{ Thing_ChangeTID(0, {PLAYER_TID}); ClearInventory(); }}
script {S_TILE} (int r, int c) {{          // SecActEnter of pixel (r, c): the player only, never a voodoo doll
  if (ActivatorTID() != {PLAYER_TID}) terminate;
  dp_r = r; dp_c = c;
  if (!dp_busy() && (GetPlayerInput(-1, INPUT_BUTTONS) & BT_ATTACK)) dp_paint(r, c);
}}
script {S_LEAVE} (void) {{ if (ActivatorTID() == {PLAYER_TID}) {{ dp_r = -1; dp_c = -1; }} }}
script {S_PEN} OPEN {{                     // FIRE held while standing on the canvas paints too
  while (TRUE) {{
    if (dp_r >= 0 && !dp_busy() && (GetPlayerInput(0, INPUT_BUTTONS) & BT_ATTACK)) dp_paint(dp_r, dp_c);
    Delay(1);
  }}
}}
script {S_READ} (void) {{                  // the READ switch: the brain's own scripts 3, 1 (per pixel) and 7
  int i;
  if (dp_busy()) {{ dp_wait(); terminate; }}
  dp_phase = {WRITING}; dp_reads++; dp_digit = -1; dp_fires = 0; dp_segments(0);
  ACS_ExecuteAlways(3, 0);
  Delay(3);
  for (i = 0; i < {n}; i++) {{
    if (dp_px[i]) ACS_ExecuteAlways(1, 0, dp_port[i], {ink}, 1);
    else ACS_ExecuteAlways(1, 0, dp_port[i], 0, 0);
  }}
  Delay(4);
  dp_t0 = Timer(); dp_phase = {THINKING};
  ACS_ExecuteAlways(7, 0);
  while (dp_phase == {THINKING}) {{        // what the player sees until the winner's doll lights the display
    HudMessageBold(s:"DOOM is reading your digit: ", d:(Timer() - dp_t0) / 35, s:" s{of_about}";
                   HUDMSG_PLAIN, 1, CR_GOLD, 0.5, 0.12, 2.0);
    Delay(35);
  }}
}}
script {S_CLEAR} (void) {{                 // the CLEAR switch: rewind the brain, wipe canvas, ports and display
  int i;
  if (dp_busy()) {{ dp_wait(); terminate; }}
  ACS_ExecuteAlways(3, 0);
  for (i = 0; i < {n}; i++) {{
    if (dp_px[i]) {{
      dp_px[i] = 0; Light_ChangeToValue({TILE_TAG0} + i, dp_paper(i)); ChangeFloor({TILE_TAG0} + i, "{PAPER}");
    }}
    ACS_ExecuteAlways(1, 0, dp_port[i], 0, 0);
  }}
  dp_ink = 0; dp_digit = -1; dp_fires = 0; dp_segments(0); dp_phase = {IDLE}; dp_clears++;
  HudMessageBold(s:"canvas cleared"; HUDMSG_FADEOUT, 1, CR_WHITE, 0.5, 0.12, 1.0, 1.0);
}}
script {S_SHOW} (int d) {{                 // fired by the winner's doll crossing its y_d line
  dp_fires++; dp_digit = d; dp_phase = {SHOWN};
  dp_segments(dp_seg[d]);
  SetFont("BIGFONT");
  HudMessageBold(s:"DOOM reads ", d:d; HUDMSG_FADEOUT, 1, CR_GREEN, 0.5, 0.12, 30.0, 1.0);
}}
script {S_STATUS} (void) {{ {status} }}
script {S_CANVAS} (void) {{ {" ".join(f"o{r + 1} = dp_row({r});" for r in range(SIZE))} }}
script {S_ACK} (void) {{ dp_ack++; }}     // sent after every readout script; the host reads once dp_ack moved
{chr(10).join(pages)}
"""


# --- playing it -------------------------------------------------------------------------------------------------------
def canvas_xy(col: float, row: float) -> tuple[float, float]:
    """Map coordinates of canvas point (col, row) (pixel units, (0, 0) = north-west corner)."""
    return X0 + col * TILE, Y1 - row * TILE


def tile_of(x: float, y: float) -> tuple[int, int]:
    """(row, col) of the canvas pixel containing map point (x, y) (may lie outside 0..27)."""
    return int(math.floor((Y1 - y) / TILE)), int(math.floor((x - X0) / TILE))


class ReadInterrupted(RuntimeError):
    """READ or CLEAR was pressed after the display lit but before every doll stood still: the pass is not checked."""


@dataclass
class Read:
    """One verified READ: what DOOM computed and how it was checked."""
    pixels: np.ndarray            # the 784 inputs read back from the pixel ports (0/1)
    digit: int                    # DOOM's answer: the digit the winner's doll lit on the display
    y: list[int]                  # the one-hot output DOOM computed (light levels y0..y9)
    checked: int                  # values compared with the NumPy integer reference (all equal, else MismatchError)
    tics: int                     # game tics from the READ press until every doll stood still
    wall: float                   # seconds of wall time for those tics
    status: dict                  # the level's STATUS after the pass
    audit: dict                   # ReadoutOnly: every console command Python sent during the pass
    canvas: bool                  # pixels == the level's canvas array when the brain started (what the pen painted)
    rasterised: bool | None       # pixels == the brush stamped along the recorded path (None: no path recorded)
    edge_positions: int           # pen-down positions within EDGE of a tile edge (may count for either tile)
    readouts: int                 # readout scripts Python ran during the pass (each confirmed by the ack script)
    readout_waits: int            # readouts whose ack took more than one advance (spectator mode keeps real time)


def route(a, b) -> list[tuple[float, float]]:
    """Waypoints from map point a to b that pass the podium row through the free lane between the podiums."""
    gap = (CLEAR_X[1] + READ_X[0]) / 2
    south, north = (gap, PODIUM_Y[0] - 28), (gap, PODIUM_Y[1] + 24)
    if a[1] < PODIUM_Y[1] and b[1] > PODIUM_Y[0]:
        return [tuple(a), south, north, tuple(b)]
    if a[1] > PODIUM_Y[0] and b[1] < PODIUM_Y[1]:
        return [tuple(a), north, south, tuple(b)]
    return [tuple(a), tuple(b)]


class Walker:
    """The stand-in player's walking (pure pursuit with a velocity controller, view kept facing north while drawing).
    Subclasses provide tic() (one tic with these keys held, appending (x, y, FIRE) to self.path) and state()
    (x, y, angle, vx, vy). DrawPad presses the keys in DOOM; SimWalker integrates the same keys with DOOM's
    player physics in Python, which is how training paints walked digits the way the level would."""

    path: list

    def state(self) -> tuple[float, float, float, float, float]:
        raise NotImplementedError

    def tic(self, forward: float = 0, strafe: float = 0, turn: float = 0, fire: bool = False, use: bool = False):
        raise NotImplementedError

    def where(self) -> tuple[float, float, float]:
        return self.state()[:3]

    def expected_pixels(self) -> np.ndarray:
        """The brush stamped at every recorded tic position with FIRE held: what the canvas should hold."""
        return stamp({tile_of(x, y) for x, y, pen in self.path if pen})

    def explains(self, pixels, edge: float = EDGE) -> tuple[bool, int]:
        """(pixels equal the brush stamped along the recorded path, number of pen-down positions within `edge` map
        units of a tile edge). Such a position may count for either tile: in DrawPad runs the level's sector action
        twice had the marine in the neighbouring tile while his reported position was 0.021 units past the edge."""
        must, may, near = set(), set(), 0
        for x, y, pen in self.path:
            if not pen:
                continue
            rows = {math.floor((Y1 - y + d) / TILE) for d in (-edge, 0, edge)}
            cols = {math.floor((x - X0 + d) / TILE) for d in (-edge, 0, edge)}
            cand = {(r, c) for r in rows for c in cols}
            if len(cand) == 1:
                must |= cand
            else:
                near += 1
            may |= cand
        lo, hi, px = stamp(must), stamp(may), np.asarray(pixels)
        return bool(((px >= lo) & (px <= hi)).all()), near

    # --- walking: what a stand-in player does with the keys ----------------------------------------------------------
    def turn_to(self, angle: float):
        """Turn in place to face `angle` degrees (90 = north)."""
        for _ in range(8):
            d = (angle - self.where()[2] + 180) % 360 - 180
            if abs(d) < 0.5:
                return
            self.tic(turn=-d)

    def stop(self, fire: bool = False, max_tics: int = 40):
        """Release the movement keys until the marine has (nearly) stopped."""
        for _ in range(max_tics):
            if math.hypot(*self.state()[3:]) < 0.05:
                return
            self.tic(fire=fire)

    def walk(self, points, fire: bool = False, speed: float = SPEED):
        """Walk through map points [(x, y), ...] (pure pursuit; moving and strafing, the view direction is kept), FIRE
        held or not."""
        pts = _densify(np.asarray(points, dtype=np.float64), 4.0)
        k = 0
        for _ in range(20 * len(pts) + 400):
            x, y, angle, vx, vy = self.state()
            while k < len(pts) - 1 and math.hypot(pts[k][0] - x, pts[k][1] - y) < LOOKAHEAD:
                k += 1
            dx, dy = pts[k][0] - x, pts[k][1] - y
            dist = math.hypot(dx, dy)
            if k == len(pts) - 1 and dist < 2:
                break
            s = min(speed, max(1.0, 0.35 * dist)) / max(dist, 1e-9)
            ax, ay = dx * s / FRICTION - vx, dy * s / FRICTION - vy      # velocity change wanted this tic
            a = math.radians(angle)
            fwd = (ax * math.cos(a) + ay * math.sin(a)) / ACCEL_FWD     # the key values that give that change
            side = (ax * math.sin(a) - ay * math.cos(a)) / ACCEL_SIDE
            self.tic(forward=float(np.clip(fwd, -V_MAX, V_MAX)), strafe=float(np.clip(side, -V_MAX, V_MAX)),
                     fire=fire)
        self.stop(fire)

    def draw(self, strokes):
        """Walk `strokes` (canvas coordinates, see walked_strokes) with FIRE held, pen up in between."""
        self.turn_to(90)
        for p in strokes:
            pts = [canvas_xy(c, r) for c, r in np.asarray(p)]
            self.walk(route(self.where()[:2], pts[0]))
            self.tic()
            self.tic(fire=True)
            self.walk(pts, fire=True)
            self.tic(fire=True)
            self.tic()


class SimWalker(Walker):
    """Walker in Python: the measured DOOM player physics (key value V adds V * ACCEL_FWD / ACCEL_SIDE units/tic of
    thrust along the view / to its right, the move uses velocity + thrust, then FRICTION; ZDoom zeroes a velocity
    below 1/16 unit/tic without input). A model of DOOM for training data, not a substitute for it."""

    def __init__(self, x: float = START[0], y: float = START[1], angle: float = 90.0):
        self.x, self.y, self.angle, self.vx, self.vy = float(x), float(y), float(angle), 0.0, 0.0
        self.path = []

    def state(self):
        return self.x, self.y, self.angle, self.vx, self.vy

    def tic(self, forward=0, strafe=0, turn=0, fire=False, use=False):
        self.angle = (self.angle - turn) % 360
        a = math.radians(self.angle)
        tx = forward * ACCEL_FWD * math.cos(a) + strafe * ACCEL_SIDE * math.sin(a)
        ty = forward * ACCEL_FWD * math.sin(a) - strafe * ACCEL_SIDE * math.cos(a)
        self.vx, self.vy = self.vx + tx, self.vy + ty
        self.x, self.y = self.x + self.vx, self.y + self.vy
        self.vx, self.vy = self.vx * FRICTION, self.vy * FRICTION
        if not (forward or strafe) and math.hypot(self.vx, self.vy) < 1 / 16:
            self.vx = self.vy = 0.0
        self.path.append((self.x, self.y, bool(fire)))


def walked_pixels(strokes) -> np.ndarray:
    """The canvas a SimWalker leaves after drawing `strokes` from the start spot: walked_strokes painted the way the
    level paints a stand-in player's walk (tile under the marine on every tic with FIRE held, 3x3 brush)."""
    w = SimWalker()
    w.draw(strokes)
    return w.expected_pixels()


class DrawPad(Walker):
    """The DrawPad level compiled with `net` (default: the shipped models/mnist/mlp.npz) and its live game.

    Headless (default): Python stands in for a human and presses the marine's keys (`draw`, `press`); spectator=True
    opens a window where you play (`watch`). Either way a READ pass is watched only through ReadoutOnly (`verify`).
    opt: the brain's schedule level, as df.compile (0 to 3)."""

    BUTTONS = ("MOVE_FORWARD_BACKWARD_DELTA", "MOVE_LEFT_RIGHT_DELTA", "TURN_LEFT_RIGHT_DELTA", "ATTACK", "USE")
    HUMAN = ("MOVE_FORWARD", "MOVE_BACKWARD", "MOVE_LEFT", "MOVE_RIGHT", "TURN_LEFT", "TURN_RIGHT", "SPEED")

    def __init__(self, net=None, wad=None, *, spectator: bool = False, resolution=None, opt: int = 0):
        from . import act, kit, shipped
        self.net = net if net is not None else kit.MLP.load(shipped.data("mnist/mlp.npz"))
        self.tags, self.ink = pixel_ports(self.net)
        self.spectator = spectator
        actions = {f"y{d}": [act.script(S_SHOW, d)] for d in range(10)}
        tics = MEASURED_READ_TICS.get(opt) if net is None else None   # measured on the shipped net; else the plan
        if tics is None:
            tics = kit.plan(self.net, mode="map", opt=opt)["map"]["estimated_settle_tics"]
        self.read_s = max(1, round(tics / 35))
        self.brain = kit.compile(self.net, wad, world=world(self.tags, self.ink, self.read_s), actions=actions,
                                 resolution=resolution, configure=self._configure, opt=opt)
        if pixel_ports(self.net, self.brain.meta.ports)[0] != self.tags:
            raise AssertionError("the compiled map numbered the pixel ports differently from the level's ACS")
        self.path: list[tuple[float, float, bool]] = []      # (x, y, FIRE held) after every tic of drawing
        self.readouts = 0          # readout scripts run (each confirmed by its ack)
        self.readout_waits = 0     # readouts whose ack needed more than one advance_action(1)
        self._acks = 0             # dp_ack as last seen
        self._game = None

    def _configure(self, g):
        import vizdoom as vzd
        for b in self.BUTTONS + (self.HUMAN if self.spectator else ()):
            g.add_available_button(getattr(vzd.Button, b))
        for v in ("POSITION_X", "POSITION_Y", "ANGLE", "VELOCITY_X", "VELOCITY_Y"):
            g.add_available_game_variable(getattr(vzd.GameVariable, v))
        g.set_render_messages(True)
        if self.spectator:
            from .runtime import _engine_home
            g.set_doom_config_path(str(_engine_home() / "_drawpad.ini"))      # its key bindings stay out of others'
            g.set_window_visible(True)
            g.set_mode(vzd.Mode.SPECTATOR)
            g.set_render_hud(True)

    @property
    def game(self):
        if self._game is None:
            self._game = self.brain.runner.game
            if self.spectator:
                for cmd in CONTROLS:
                    self._game.send_game_command(cmd)
            self._game.advance_action(2)                    # ENTER: the player gets his TID
            self._acks = self._ack(self._game)
        return self._game

    @property
    def info(self) -> dict:
        return self.brain.info

    def close(self):
        self.brain.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    # --- observing (allowed during a READ pass) --------------------------------------------------------------------
    @staticmethod
    def _ack(g) -> int:
        """dp_ack: ACS global 0, which ViZDoom reports as the map reward in 16.16 fixed point."""
        return int(round(g.get_total_reward() * 65536))

    def _users(self, g, script: int, n: int) -> list[int]:
        """Run readout script `script` once and return USER1..n as it wrote them.

        The ack script S_ACK is sent right after it. Both commands enter the engine together, run in the same tic, and
        nothing else writes USER1..n, so once dp_ack has moved the values are the script's. In spectator mode the
        engine keeps real time: one advance_action(1) runs one or more tics, and a command sent between two advances
        can land on a tic that only runs after the next state update. Reading right after one advance then returned
        the previous readout's values. In PLAYER mode the ack arrives after exactly one tic (ACK_TICS bounds it); in
        spectator mode the host waits as long as it takes (an open menu pauses the game)."""
        from .runtime import USER
        want = self._acks + 1
        g.send_game_command(f"puke -{script}")
        g.send_game_command(f"puke -{S_ACK}")
        advances = 0
        while True:
            g.advance_action(1)
            advances += 1
            got = self._ack(g)
            if got == want:
                break
            if got != self._acks or (advances >= ACK_TICS and not self.spectator):
                raise RuntimeError(f"readout script {script}: ack {got} after {advances} tics, expected {want}")
        self._acks = want
        self.readouts += 1
        self.readout_waits += advances > 1
        return [int(g.get_game_variable(USER[i])) for i in range(n)]

    def status(self, g=None) -> dict:
        """The level's state (one readout): phase, reads, digit (-1 = none), fires, ink, row/col under the marine, ..."""
        return dict(zip(STATUS_NAMES, self._users(g or self.game, S_STATUS, len(STATUS_NAMES))))

    def read_brain(self, g) -> dict[str, int]:
        """Every value the brain's readout pages expose (scripts 100.., as TileRunner.read, one acked readout each)."""
        out = {}
        for pi, page in enumerate(self.brain.meta.pages):
            out.update(zip(page, self._users(g, 100 + pi, len(page))))
        return out

    def ports(self, g=None) -> np.ndarray:
        """The 784 pixel port heights as they are in the map (-ink where the level wrote ink, 0 for paper)."""
        from .tile import PAGE
        n = SIZE * SIZE
        out = []
        for p in range(-(-n // PAGE)):
            out += self._users(g or self.game, S_PIXELS + p, min(PAGE, n - p * PAGE))
        return np.array(out, dtype=np.int64)

    def canvas(self, g=None) -> np.ndarray:
        """The level's canvas array (what TILE/PEN painted so far; one readout): 784 0/1 pixels, row 0 = north."""
        rows = self._users(g or self.game, S_CANVAS, SIZE)
        return np.array([(r >> c) & 1 for r in rows for c in range(SIZE)], dtype=np.uint8)

    def state(self) -> tuple[float, float, float, float, float]:
        """(x, y, angle, vx, vy) of the marine (game variables)."""
        import vizdoom as vzd
        g = self.game
        return tuple(float(g.get_game_variable(getattr(vzd.GameVariable, v))) for v in (
            "POSITION_X", "POSITION_Y", "ANGLE", "VELOCITY_X", "VELOCITY_Y"))

    def tic(self, forward: float = 0, strafe: float = 0, turn: float = 0, fire: bool = False, use: bool = False):
        """One game tic with these keys held (Python standing in for a human); records where the marine ended up."""
        g = self.game
        g.set_action([forward, strafe, turn, float(fire), float(use)])
        g.advance_action(1)
        x, y, _ = self.where()
        self.path.append((x, y, bool(fire)))

    def press(self, which: str = "read"):
        """Walk to the READ (or CLEAR) podium, face it and press USE for one tic, then let go of every key."""
        a, b = READ_X if which == "read" else CLEAR_X
        self.walk(route(self.where()[:2], ((a + b) / 2, PODIUM_Y[0] - 28)))
        self.turn_to(90)
        self.tic(use=True)
        self.game.set_action([0.0] * len(self.BUTTONS))

    def watch(self, on_read, on_start=None) -> None:
        """Spectator mode: you play. The level's STATUS is read after every advance (one readout). When a READ starts,
        on_start(status) is called, the pass is verified (`verify`) and on_read(Read) is called; a pass cut short by
        another READ or by CLEAR is reported with on_read(ReadInterrupted). Returns when the window is closed."""
        import vizdoom as vzd
        g = self.game
        try:
            last = self.status()["reads"]
            while not g.is_episode_finished():
                st = self.status()
                if st["reads"] == last:
                    continue
                last = st["reads"]
                if on_start is not None:
                    on_start(st)
                try:
                    on_read(self.verify(last))
                except ReadInterrupted as e:          # a new READ is picked up by the next status
                    on_read(e)
        except (vzd.ViZDoomIsNotRunningException, vzd.ViZDoomUnexpectedExitException):    # the window was closed
            pass

    # --- the READ pass: observe only ------------------------------------------------------------------------------
    def verify(self, reads: int, on_tic=None, every: int = 8) -> Read:
        """Watch the READ pass number `reads` through ReadoutOnly (advance tics, readout scripts; nothing else) until
        every doll and the marine stood still, then compare every value DOOM computed with the NumPy integer
        reference computed from the 784 pixel ports DOOM holds. Raises df.MismatchError on any difference.
        on_tic(pad, tic) is called every `every` tics (films)."""
        from .kit import MismatchError
        from .readout import ReadoutOnly, readout_commands
        from .tile import MOTION_SCRIPT, PAGE
        runner, raw = self.brain.runner, self.game
        n_pix = -(-SIZE * SIZE // PAGE)
        g = ReadoutOnly(raw, readout_commands(len(runner.meta.pages), *(f"puke -{s}" for s in (
            S_STATUS, S_CANVAS, S_ACK, MOTION_SCRIPT)), *(f"puke -{S_PIXELS + p}" for p in range(n_pix))))
        runner.game, self._game = g, g
        t0, tic0, n0, w0 = time.time(), raw.get_episode_time(), self.readouts, self.readout_waits
        again = ReadInterrupted(f"READ {reads} not checked: READ was pressed again before every doll stood still")
        try:
            for _ in range(64):
                st = self.status(g)
                if st["reads"] > reads:
                    raise again
                if st["reads"] == reads and st["phase"] >= THINKING:
                    break
            else:
                raise RuntimeError(f"READ {reads} did not start the brain: {st}")
            canvas, clears = self.canvas(g), st["clears"]      # the pen is off from READ until the display lights
            runner.settle(on_tick=(lambda t: on_tic(self, t)) if on_tic else None, check_every=every)
            tics = g.get_episode_time() - tic0
            wall = time.time() - t0
            got = self.read_brain(g)
            ports = self.ports(g)
            st = self.status(g)
        finally:
            runner.game, self._game = raw, raw
        if st["reads"] != reads:
            raise again
        if st["clears"] != clears:
            raise ReadInterrupted(f"READ {reads} not checked: CLEAR was pressed before every doll stood still")
        x = -ports
        bad = [(f"x{i}", int(v), 0) for i, v in enumerate(x) if v not in (0, self.ink)]
        if bad:
            raise MismatchError("pixel ports hold neither ink nor paper", bad)
        self.net.check(x)
        ref = self.net.reference(x)
        bad = [(k, v, self.net._expected(k, ref)) for k, v in got.items() if v != self.net._expected(k, ref)]
        y = [got[f"y{d}"] for d in range(10)]
        if st["fires"] != 1 or st["digit"] < 0 or y[st["digit"]] != 1:
            bad.append(("display", st["digit"], int(np.flatnonzero(ref["y"])[0])))
        if bad:
            raise MismatchError(f"DrawPad READ {reads}", bad)
        pixels = (x == self.ink).astype(np.uint8)
        drawn = any(p[2] for p in self.path)
        same, near = self.explains(pixels) if drawn else (None, 0)
        return Read(pixels=pixels, digit=st["digit"], y=y, checked=len(got), tics=tics, wall=wall, status=st,
                    audit=g.audit(), canvas=bool(np.array_equal(pixels, canvas)), rasterised=same,
                    edge_positions=near, readouts=self.readouts - n0, readout_waits=self.readout_waits - w0)
