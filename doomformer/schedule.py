"""Static schedule checker for one-map brains: translation validation of the wave schedule against an engine model.

check_schedule(build) (CLI: `doomformer lint WAD`) reads only the compiled map: the TEXTMAP of the .wad (sectors,
two-sided special lines, things, carry scrollers) and the port keys of its .json (layout.tags). It uses no compiler
objects, so any map pure.compile_pure built can be checked.

Reconstruction. Every player-1 start but the last is a voodoo doll; its lane is the contiguous row of sectors east of
it. Port sectors are gates: a floor gate passes iff value <= walk + 24, a ceiling gate iff value >= walk + 56, walk =
the floor west of the gate (tile.Gate). The first non-port, non-freeze sector more than 24 above its west neighbour is
the start barrier; the next one after the first gate is the sync barrier (its tag is the wave). Cells are the special
lines after the sync barrier: Light_* 110-112 are no target, floor/ceiling movers target (kind, tag), other specials are
actions keyed by (special, args). Dolls without gates are clocks: LOWER_TO_LOWEST lines release a wave (arg0 = a sync
tag) or start the next layer (arg0 = a start tag), Floor_MoveToValue lines raise freeze sectors, Teleport_NoFog continues
at the destination thing, ACS_ExecuteAlways(30..99) is the opt-3 copy line. Layers follow the next-layer lines from the
one start tag no line lowers (ACS script 7).

Engine model (Track.run): every doll is simulated tic by tic in 16.16 fixed point as ZDoom 2.8 (ViZDoom 1.3.2) moves a
voodoo doll. Carry push = the sum of the sector tag's Boom carry scrollers (4 x 255: 97,536/65,536 units/tic), averaged
over touching conveyor sectors whose floor is not below the doll; moves in steps of at most radius - 1; P_SlideMove
stops 1/32 of a step short of a blocking line; friction 0xE800 on the floor only, no push and no friction in the air;
gravity 1 unit/tic^2, doubled on the first tic off a ledge but not when a floor lowers under the doll; step up <= 24
and 56 units of headroom, both against the current z; a line is crossed when the centre passes it. A barrier lowered at
tic r stands at F - 18,750 (t - r) for the dolls of tic t > r (they tick before the mover). Teleport_NoFog keeps the
velocity, and the steps of that tic left after the crossing continue from the destination. Tic 0 is the first tic after
TileRunner.start (script 7 lowers layer 0's start barriers); layer s starts when clock s-1 crosses its next line. A lane
is simulated from rest at its start barrier to its sync barrier (arrival: the first tic it stands blocked there) and
from rest at the sync barrier from its wave's release on. Data dependence: every gate is assumed passed, and each floor
gate's step height ranges over 0..24 as far as the gates before it allow (0: at or below the walkway, the 8-unit gate is
bridged; 1..24: the doll steps onto the gate, loses its push and falls back); cell tics are intervals over these
variants and over REST_BAND.

Checks. A hazard is two cells of one plane (or action) target whose tic intervals come closer than SEPARATION (or the
mover's lifetime) and whose gate boxes intersect: no value needs disjoint intervals ('kind:v' and 'kind:-v' are v and
its negation; values the same layer writes are left out). certain: both tics exact, every input passing both gates
loses a move. Witness: the value nearest 0 in each interval; layer-0 values are inputs (an x vector for MLP names x0..);
deeper values are searched by random inputs and hill climbing on net.reference or intref.forward when a network is
given. Margins, named after the Timing field that sets them: serial (crossbar release spacing >= SEPARATION), corr_gap
(corridor layers: fewest tics between two waves' writes of one plane >= SEPARATION), sync_wait (release minus the latest
arrival >= SYNC_SETTLE), clock_margin (freeze lines after the last doll left a freeze sector; every gate read at least
the mover's lifetime after an earlier layer's last write of it; opt 3: the copy line not before the tic of the layer's
last effect line, gate reads >= COPY_WRITE tics after the copy line), tele_gap (a teleport >= 1 tic after its chunk's
last line), copy_settle (freeze/next line not before the copy line).

Measurements (ViZDoom 1.3.2, shared 16-core machine at load 30+; the probes behind them are listed with the validation
record in models/schedule/):
  SEPARATION = 1 tic: two dolls (weights 1 and 2) write one floor plane, and separately one ceiling plane, released with
    serial S = 0..48 units; plus a corridor variant where the later-spawned doll crosses first (S = 64..192). Same tic:
    the second-processed move is lost (plane 1 instead of 3); 1, 2 or 3 tics apart: both land, in both orders.
  Motion: every tic of every doll of a hello pass matches Track.run within 0.01 units (61 dolls, 491 tics). Predicted
    crossing tics equal the engine's for every clock line, lane cell and sync arrival of hello (66 clock lines, 74 cells,
    7 arrivals), tiny_model tokens [1, 3, 2] (163, 687, 198), hello with serial 3000/1500 (70 clock lines, 5 teleports)
    and net591 at x = -9000 and 9000.
  REST_BAND = 1 unit: the engine's slide stops a doll 1, 2, 4 or 8 units short at its first stop and it crawls on; final
    rest offsets were 0 (755 dolls) or -1 unit (10, after a 28-30 unit drop) in 4 recordings. SYNC_SETTLE = 4 tics: the
    longest crawl from the predicted arrival to the final rest (198 tiny_model dolls).
  Barrier float (Verify's fuzzer, net591): a barrier needing two mover tics whose first tic stops 1..24 units above the
    walkway (walkways -5774..-5751 under CLOSED = 13,000) is stepped onto; the floor drops away and the doll floats ~8
    tics without push. Reproduced exactly (certain hazard at tic 68, witness x0 = -8058; DOOM fails for x <= -8058).
    pure._barrier_height now picks barrier heights without such a stop. The engine steps onto the stopped barrier only
    when it stands MORE than 8 units above its final height (one lane net per stop height 1..26 above the walkway, the
    doll's cell crossing against the model: stops 1..8 cross 3 tics before a model that floats, 9..24 float and match);
    LOWERING_ABSORBED = 8 encodes it. A stop of 1..8 units therefore never floats and is not a hazard.
  Cross-check: the tuner's mlp24 (24-32-16-8, K = 3, seed 7) at opt 1 with serial S on every layer gets 186 / 92 / 0 hazards
    for S = 16 / 24 / 32 units; DOOM was exact on 17/30, 25/30, 30/30 inputs.
  opt 3: copy_settle 4 and 16 units exact (hello, 6 inputs each); clock_margin 0, 32, 64 units put the copy line 8, 6,
    3 tics before the last light hit and DOOM fails on every input.

Not covered or not exact:
  * Rest positions after the slide shortfall are bounded (REST_BAND, SYNC_SETTLE), not simulated.
  * A doll that arrives at its sync barrier later than SYNC_SETTLE before its release is reported (sync_wait), not
    simulated: it walks on without stopping. Tuned hello at sync_wait 64 with step-up inputs was exact on 6/6 inputs
    while that margin read -3 tics, so this margin is a precondition of the model, not a proven failure.
  * Layer-0 dolls are assumed at rest against their start barriers when TileRunner.start returns (true in the recorded
    first and second runs).
  * Lanes with more than max_variants step-height combinations: only the extreme heights are simulated (noted).
  * Mover lifetimes of non-ByValue floor/ceiling specials are bounded by 32,768 units at their speed; actions are held to
    SEPARATION like planes (not measured); lights are never hazards (they take same-tic hits).
  * The copy script's start latency (opt 3) is not measured: COPY_WRITE assumes 1 tic plus 2 mover tics, as pure.py.
  * Dolls blocked by things or world geometry, maps not built by pure.compile_pure, tile maps.
"""
from __future__ import annotations

import bisect
import itertools
import json
import re
import struct
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

# --- engine model: ZDoom 2.8 as built into ViZDoom 1.3.2, 16.16 fixed point --------------------------------------------
FRAC = 1 << 16
RADIUS, STEP, HEIGHT = 16 * FRAC, 24 * FRAC, 56 * FRAC   # player radius, MaxStepHeight, height
FRICTION = 0xE800            # ORIG_FRICTION: velocity *= 0.90625 per tic on the floor
STOPSPEED = 0x1000           # velocities below this stop
GRAVITY = FRAC               # sv_gravity 800 * 0.00125: 1 unit/tic^2 (doubled on the first tic off a ledge)
SLIDE_FUDGE = 0x800          # P_SlideMove stops 1/32 of the move short of a blocking line
MAXMOVE = RADIUS - FRAC       # P_XYMovement moves in steps of at most radius - 1
CARRY_FACTOR = 0x1800        # 0.09375: Boom carry scroller push per unit of scroll speed
BIG = 10 ** 9
LOWERING_ABSORBED = 8     # a lowering barrier that stops 8 or fewer units above its final height is not stepped onto (engine)
STEP_UP_UNITS, DOLL_H_UNITS = 24, 56     # tile.Gate: floor passes iff value <= walk + 24, ceiling iff >= walk + 56

TELEPORT_NOFOG, EXEC_ALWAYS, LOWER_TO_LOWEST, SCROLL_FLOOR, FLOOR_MOVE_TO = 71, 226, 21, 223, 37
THING_PLAYER1, THING_DEST = 1, 14
COPY_SCRIPTS = range(30, 100)                  # opt 3: ACS_ExecuteAlways(30 + layer) on a clock runs the copy bus
LIGHT = {110, 111, 112}                        # Light_RaiseByValue/LowerByValue/ChangeToValue: immediate, no thinker
FLOOR = {20, 21, 22, 23, 24, 25, 28, 35, 36, 37, 66, 67, 68, 200, 238, 240, 242}
CEIL = {10, 11, 12, 13, 14, 40, 41, 42, 43, 45, 47, 69, 192, 193, 194, 201, 252, 253, 254}
BY_VALUE = {20: 1, 23: 1, 40: 1, 41: 1, 35: 8, 36: 8}      # special -> distance multiplier of arg2 (speed arg1/8)


# --- TEXTMAP --------------------------------------------------------------------------------------------------------
def _field(key: str, num: str = r"-?\d+", optional: bool = False) -> str:
    look = rf"(?=[^}}]*?\b{key}\s*=\s*({num})\s*;)"
    return rf"(?:{look})?" if optional else look


_FLOAT = r"[-+]?[\d.]+(?:[eE][-+]?\d+)?"
_VERTEX = re.compile(rb"\bvertex\s*\{" + (_field("x", _FLOAT) + _field("y", _FLOAT)).encode())
_SIDEDEF = re.compile(rb"\bsidedef\s*\{" + _field("sector").encode())
_SECTOR = re.compile(rb"\bsector\s*\{" + (_field("heightfloor", optional=True) + _field("heightceiling", optional=True)
                                           + _field("id", optional=True)).encode())
_LINEDEF = re.compile(rb"\blinedef\s*\{" + "".join(
    [_field("v1"), _field("v2"), _field("sidefront"), _field("sideback", optional=True),
     _field("special", optional=True)] + [_field(f"arg{i}", optional=True) for i in range(5)]).encode())
_THING = re.compile(rb"\bthing\s*\{" + (_field("x", _FLOAT) + _field("y", _FLOAT) + _field("type")
                                         + _field("id", optional=True)).encode())


def _ints(rows: list, width: int, default: int = 0) -> np.ndarray:
    a = np.array(rows, dtype="S24").reshape(-1, width)
    a[a == b""] = str(default).encode()
    return a.astype(np.int64)


def textmap(wad) -> bytes:
    """The TEXTMAP lump of the first map in a UDMF .wad."""
    data = Path(wad).read_bytes()
    magic, n, ofs = struct.unpack_from("<4sii", data)
    if magic not in (b"PWAD", b"IWAD"):
        raise ValueError(f"{wad}: not a WAD file")
    for i in range(n):
        pos, size, name = struct.unpack_from("<ii8s", data, ofs + 16 * i)
        if name.rstrip(b"\0") == b"TEXTMAP":
            return data[pos:pos + size]
    raise ValueError(f"{wad}: no TEXTMAP lump (only UDMF maps are supported)")


@dataclass
class MapData:
    """Sectors as axis-aligned boxes (doomformer maps are made of rectangles), the special lines between two sectors,
    the carry push of each sector tag and the things."""
    x0: np.ndarray
    y0: np.ndarray
    x1: np.ndarray
    y1: np.ndarray
    floor: np.ndarray
    ceil: np.ndarray
    tag: np.ndarray
    lines: dict            # (sector a, sector b), a < b -> (special, (arg0..arg4))
    push: dict             # sector tag -> carry push per tic (fixed point) of its Scroll_Floor carry scrollers
    things: np.ndarray     # rows x, y, type, id

    @classmethod
    def parse(cls, text: bytes) -> "MapData":
        v = np.array(_VERTEX.findall(text), dtype="S32").reshape(-1, 2).astype(np.float64)
        side_sector = _ints(_SIDEDEF.findall(text), 1)[:, 0]
        sec = _ints(_SECTOR.findall(text), 3)
        ln = _ints(_LINEDEF.findall(text), 10, default=-1)
        things = np.array(_THING.findall(text), dtype="S32").reshape(-1, 4)
        things[things == b""] = b"0"
        things = things.astype(np.float64)
        n = len(sec)
        x0, y0 = np.full(n, np.inf), np.full(n, np.inf)
        x1, y1 = np.full(n, -np.inf), np.full(n, -np.inf)
        lx = np.stack([v[ln[:, 0], 0], v[ln[:, 1], 0]], 1)
        ly = np.stack([v[ln[:, 0], 1], v[ln[:, 1], 1]], 1)
        front = side_sector[ln[:, 2]]
        back = np.where(ln[:, 3] >= 0, side_sector[np.maximum(ln[:, 3], 0)], -1)
        for owners, keep in ((front, slice(None)), (back[back >= 0], back >= 0)):
            np.minimum.at(x0, owners, lx[keep].min(1))
            np.minimum.at(y0, owners, ly[keep].min(1))
            np.maximum.at(x1, owners, lx[keep].max(1))
            np.maximum.at(y1, owners, ly[keep].max(1))
        lines, push = {}, {}
        special = ln[:, 4]
        args = np.maximum(ln[:, 5:10], 0)
        for i in np.flatnonzero(special > 0):
            sp, a = int(special[i]), tuple(int(x) for x in args[i])
            if sp == SCROLL_FLOOR and a[2] in (1, 2):              # carry (or scroll and carry): Boom conveyor
                push[a[0]] = push.get(a[0], 0) + ((((a[3] - 128) << 16) // 32) * CARRY_FACTOR >> 16)
            if back[i] >= 0:
                f, b = int(front[i]), int(back[i])
                lines[(min(f, b), max(f, b))] = (sp, a)
        return cls(x0, y0, x1, y1, sec[:, 0], sec[:, 1], sec[:, 2], lines, push, things)


class Rows:
    """Sectors sorted into rows of equal [y0, y1], west to east, for walking a doll's lane eastwards."""

    def __init__(self, md: MapData):
        self.md = md
        order = np.lexsort((md.x0, md.y1, md.y0))
        self.order = order
        ys = np.stack([md.y0[order], md.y1[order]], 1)
        new_row = np.ones(len(order), dtype=bool)
        new_row[1:] = (ys[1:] != ys[:-1]).any(1)
        self.row_start = np.flatnonzero(new_row)
        self.row_y = ys[self.row_start]
        self.row_end = np.append(self.row_start[1:], len(order))
        brk = np.ones(len(order), dtype=bool)                 # brk[i]: sorted sector i is the last of its run
        brk[:-1] = new_row[1:] | (md.x1[order][:-1] != md.x0[order][1:])
        self.run_end = np.flatnonzero(brk)
        self.sx0 = md.x0[order]

    def find(self, x: float, y: float) -> int | None:
        """Position in sorted order of the sector containing (x, y)."""
        for r in np.flatnonzero((self.row_y[:, 0] < y) & (y < self.row_y[:, 1])):
            a, b = self.row_start[r], self.row_end[r]
            i = a + int(np.searchsorted(self.sx0[a:b], x, side="right")) - 1
            if i >= a and x < self.md.x1[self.order[i]]:
                return int(i)
        return None

    def chain(self, x: float, y: float) -> list[int]:
        """Sector indices from the one containing (x, y) east to the end of its contiguous run."""
        i = self.find(x, y)
        if i is None:
            return []
        end = self.run_end[np.searchsorted(self.run_end, i)]
        return [int(s) for s in self.order[i:end + 1]]


# --- doll motion ----------------------------------------------------------------------------------------------------
class Track:
    """The sectors a doll walks east through, merged where nothing differs, in fixed point. A sector's floor may be
    dynamic (a barrier lowered at some tic) or a port (a value)."""

    def __init__(self, md: MapData, chain: list[int], dynamic: set[int] = frozenset()):
        self.raw = chain
        self.at = {}                  # raw sector -> merged index
        x0, x1, fl, ce, pu, keep = [], [], [], [], [], []
        for s in chain:
            props = (int(md.floor[s]), int(md.ceil[s]), md.push.get(int(md.tag[s]), 0))
            if fl and s not in dynamic and keep[-1] and (fl[-1], ce[-1], pu[-1]) == props and x1[-1] == md.x0[s]:
                x1[-1] = int(md.x1[s])
            else:
                x0.append(int(md.x0[s]))
                x1.append(int(md.x1[s]))
                fl.append(props[0])
                ce.append(props[1])
                pu.append(props[2])
                keep.append(s not in dynamic)
            self.at[s] = len(x0) - 1
        self.x0 = [a * FRAC for a in x0]
        self.x1 = [a * FRAC for a in x1]
        self.floor = [a * FRAC for a in fl]
        self.ceil = [a * FRAC for a in ce]
        self.push = pu

    def run(self, x: int, t0: int, t_end: int, floors: dict | None = None, ceils: dict | None = None,
            stop: int | None = None, halt: int | None = None, give_up: int = 64,
            v0: int = 0) -> tuple[list[int], list[int], int]:
        """Positions at the end of tics t0..t of a doll on the floor at fixed-point x with velocity v0 at the end of tic
        t0, until tic t_end, until it is past `stop`, blocked east of `halt` or blocked for `give_up` tics. floors:
        merged index -> fixed height, or (height, target, speed, tic) for a floor lowered from `height` towards
        `target` by `speed` per tic from tic `tic` on (LOWER_TO_LOWEST). ceils: merged index -> fixed height. Returns
        (positions, waits, final velocity): waits lists the first tic of every stretch of tics in which the doll stood
        blocked on the floor."""
        floors, ceils = floors or {}, ceils or {}
        x0, x1, push = self.x0, self.x1, self.push
        base_f, base_c = self.floor, self.ceil

        def fl(i, t):
            f = floors.get(i)
            if f is None:
                return base_f[i]
            if isinstance(f, tuple):
                h, target, speed, r = f
                if t <= r:
                    return h
                v = max(target, h - speed * (t - r))
                return target if v - target <= LOWERING_ABSORBED * FRAC else v
            return f

        def ce(i):
            return ceils.get(i, base_c[i])

        def touching(px):
            return range(bisect.bisect_right(x1, px - RADIUS), bisect.bisect_left(x0, px + RADIUS))

        out = [x]
        z = max(fl(i, t0) for i in touching(x))
        floorz, vx, vz, blocked, waits = z, v0, 0, False, []
        stop = BIG * FRAC if stop is None else stop
        for t in range(t0 + 1, t_end + 1):
            near = touching(x)
            # a floor that moved under the doll at the end of the last tic (a lowering barrier) changes floorz but not
            # z: the doll falls from where it stood, without the doubled first tic of walking off a ledge (measured)
            floorz = max(fl(i, t) for i in near)
            z = max(z, floorz)
            q = [push[i] for i in near if push[i] and z <= fl(i, t)]
            if q:
                vx += sum(q) // len(q)
            oldfloorz, start, moved = floorz, x, True
            steps = 1 + vx // MAXMOVE if vx > MAXMOVE else 1      # P_XYMovement splits moves above radius - 1
            for k in range(1, steps + 1):
                nx = start + vx * k // steps
                near = touching(min(nx, x1[-1] - 1))
                fz = max(fl(i, t) for i in near)
                cz = min(ce(i) for i in near)
                wall = nx + RADIUS > x1[-1]                      # the strip ends in a one-sided wall
                if wall or fz - z > STEP or cz - fz < HEIGHT or cz - max(z, fz) < HEIGHT:
                    bad = [i for i in near if fl(i, t) - z > STEP or ce(i) - max(z, fz) < HEIGHT]
                    contact = x0[bad[0]] - RADIUS if bad else (x1[-1] - RADIUS if wall else x)
                    one = vx // steps
                    if one > 0 and contact > x:                  # P_SlideMove: up to 1/32 of the step short
                        frac = ((contact - x) << 16) // one - SLIDE_FUDGE
                        if frac > 0:
                            x += (one * frac) >> 16
                    moved = False
                    break
                x, floorz = nx, fz
                z = max(z, fz)                                   # step up
            if not moved:
                vx = 0
                floorz = max(fl(i, t) for i in touching(x))
            if z <= floorz:
                vx = 0 if vx < STOPSPEED else (vx * FRICTION) >> 16
            if z != floorz or vz:
                z += vz
                if z > floorz:
                    vz -= 2 * GRAVITY if (vz == 0 and oldfloorz > floorz and z == oldfloorz) else GRAVITY
                if z <= floorz:
                    z, vz = floorz, 0
            stuck = not moved and z <= floorz                    # blocked, standing on the floor
            if stuck and not blocked:
                waits.append(t)
            blocked = stuck
            out.append(x)
            if x > stop or (blocked and ((halt is not None and x > halt) or t - waits[-1] >= give_up)):
                break
        return out, waits, vx

    def key(self, x: int, floors: dict, ceils: dict, t0: int, reach: int) -> tuple:
        """Everything a run from x at tic t0 depends on up to `reach` (fixed point, east of x), relative to x, to the
        floor under x and to t0: runs with equal keys move alike."""
        base = max(self.floor[i] for i in range(len(self.x0)) if self.x0[i] < x + RADIUS and self.x1[i] > x - RADIUS)
        geo = []
        for i in range(len(self.x0)):
            if self.x1[i] <= x - RADIUS or self.x0[i] >= x + reach:
                continue
            f = floors.get(i, self.floor[i])
            f = (f[0] - base, f[1] - base, f[2], f[3] - t0) if isinstance(f, tuple) else f - base
            geo.append((self.x0[i] - x, self.x1[i] - x, f, ceils.get(i, self.ceil[i]) - base, self.push[i]))
        return tuple(geo)


# --- the schedule the map encodes -----------------------------------------------------------------------------------
@dataclass
class Gate:
    """A port sector in a lane: the doll passes iff lo <= var <= hi (var = the port's value name without its sign)."""
    sector: int
    x0: int                  # west edge, map units
    key: str                 # port key "floor:-x0" / "ceil:v"
    kind: str
    walk: int                # floor of the sector west of the gate: the doll's walking floor there
    var: str
    lo: int
    hi: int
    pos: int                 # chain position


@dataclass
class Cell:
    """A special line a lane doll crosses after its sync barrier."""
    x: int
    special: int
    args: tuple
    target: tuple | None     # ("floor"|"ceil", tag) for a plane mover, ("action", special, args), None for a light
    ticks: int               # plane mover lifetime in tics
    gates: int               # how many of the lane's gates lie west of it


@dataclass
class Lane:
    doll: int
    x: float
    y: float
    chain: list[int]
    start: int               # chain position of the start barrier
    sync: int | None         # chain position of the sync barrier
    gates: list[Gate]
    cells: list[Cell]
    freeze: list[int]        # chain positions of freeze sectors
    layer: int = -1


@dataclass
class ClockLine:
    x: int
    special: int
    args: tuple
    segment: int
    role: str = ""           # release | next | freeze | teleport | copy | other


@dataclass
class Clock:
    doll: int
    segments: list           # [(chain, start x or None for "waits at its start barrier")]
    start: int               # chain position (segment 0) of its start barrier
    lines: list[ClockLine]
    layer: int = -1


def _mover_ticks(special: int, args: tuple) -> int:
    speed = max(args[1], 1)
    if special in BY_VALUE:
        return max(1, -(-(args[2] * BY_VALUE[special] * 8) // speed))
    return max(1, -(-(32768 * 8) // speed))       # to-value/to-neighbour movers: bounded by the 16.16 height range


def _target(special: int, args: tuple) -> tuple | None:
    if special in LIGHT:
        return None
    if special in FLOOR:
        return ("floor", args[0])
    if special in CEIL:
        return ("ceil", args[0])
    return ("action", special, args)


def _interval(name: str, kind: str, limit: int) -> tuple[str, int, int]:
    """(var, lo, hi) for a gate on the port carrying `name` (maybe '-v'): floor passes iff value <= limit, ceiling iff
    value >= limit (tile.Gate)."""
    neg = name.startswith("-")
    var = name.lstrip("-")
    if kind == "floor":
        return (var, -limit, BIG) if neg else (var, -BIG, limit)
    return (var, -BIG, -limit) if neg else (var, limit, BIG)


class Layout:
    """Layers, clocks and lanes of a compiled one-map brain, read from its TEXTMAP and the port keys of its .json."""

    def __init__(self, md: MapData, meta: dict):
        self.md = md
        self.meta = meta
        layout = meta.get("layout", {})
        self.port_key = {int(t): k for k, t in (layout.get("tags") or meta["ports"]).items()}
        self.inputs = set(meta["ports"])
        rows = self.rows = Rows(md)
        things = md.things
        starts = np.flatnonzero(things[:, 2] == THING_PLAYER1)
        self.player = int(starts[-1]) if len(starts) else None
        self.dests = {int(t[3]): (float(t[0]), float(t[1])) for t in things[things[:, 2] == THING_DEST] if t[3]}
        lowered = set()                    # tags a LOWER_TO_LOWEST line lowers
        moved_to = set()                   # tags a Floor_MoveToValue line on a clock raises (freeze barriers)
        for sp, a in md.lines.values():
            if sp == LOWER_TO_LOWEST:
                lowered.add(a[0])
            elif sp == FLOOR_MOVE_TO:
                moved_to.add(a[0])
        self.freeze_tags = moved_to
        self.lanes: list[Lane] = []
        self.clocks: list[Clock] = []
        self.skipped: list[int] = []
        for d in starts[:-1]:
            x, y = float(things[d, 0]), float(things[d, 1])
            chain = rows.chain(x, y)
            if not chain:
                self.skipped.append(int(d))
                continue
            self._read_doll(int(d), x, y, chain)
        self._layers(lowered)

    def line(self, a: int, b: int):
        return self.md.lines.get((min(a, b), max(a, b)))

    def barrier(self, chain: list[int], k: int) -> bool:
        """Chain position k is a closed barrier: a non-port sector too high to step onto from its west neighbour."""
        md, s = self.md, chain[k]
        return (k > 0 and int(md.tag[s]) not in self.port_key and int(md.tag[s]) not in self.freeze_tags
                and md.floor[s] - md.floor[chain[k - 1]] > STEP_UP_UNITS)

    def _read_doll(self, d: int, x: float, y: float, chain: list[int]):
        md = self.md
        bars = [k for k in range(len(chain)) if self.barrier(chain, k)]
        ports = [k for k, s in enumerate(chain) if int(md.tag[s]) in self.port_key]
        if not bars:
            self.skipped.append(d)
            return
        if not ports:                                       # a clock: follow its teleports
            segments, lines, seen = [(chain, None)], [], set()
            seg = 0
            while True:
                ch = segments[seg][0]
                tele = None
                for a, b in zip(ch, ch[1:]):
                    ln = self.line(a, b)
                    if ln:
                        lines.append(ClockLine(int(md.x1[a]), ln[0], ln[1], seg))
                        if ln[0] == TELEPORT_NOFOG:
                            tele = ln[1][0]
                            break
                if tele is None or tele in seen or tele not in self.dests:
                    break
                seen.add(tele)
                tx, ty = self.dests[tele]
                nxt = self.rows.chain(tx, ty)
                if not nxt:
                    break
                segments.append((nxt, tx))
                seg += 1
            self.clocks.append(Clock(d, segments, bars[0], lines))
            return
        start = bars[0]
        sync = next((k for k in bars if k > ports[0]), None)
        gates = []
        for k in ports:
            s = chain[k]
            key = self.port_key[int(md.tag[s])]
            kind, name = key.split(":", 1)
            walk = int(md.floor[chain[k - 1]])
            limit = walk + STEP_UP_UNITS if kind == "floor" else walk + DOLL_H_UNITS
            var, lo, hi = _interval(name, kind, limit)
            gates.append(Gate(s, int(md.x0[s]), key, kind, walk, var, lo, hi, k))
        cells = []
        if sync is not None:
            for k in range(sync, len(chain) - 1):
                ln = self.line(chain[k], chain[k + 1])
                if ln:
                    xk = int(md.x1[chain[k]])
                    cells.append(Cell(xk, ln[0], ln[1], _target(*ln), _mover_ticks(*ln),
                                      sum(1 for g in gates if g.x0 < xk)))
        freeze = [k for k, s in enumerate(chain) if int(md.tag[s]) in self.freeze_tags]
        self.lanes.append(Lane(d, x, y, chain, start, sync, gates, cells, freeze))

    def _layers(self, lowered: set):
        md = self.md
        tag_of = lambda chain, k: int(md.tag[chain[k]])
        clock_of = {}
        for c in self.clocks:
            clock_of.setdefault(tag_of(c.segments[0][0], c.start), c)
        sync_tags = {tag_of(l.chain, l.sync) for l in self.lanes if l.sync is not None}
        for c in self.clocks:
            for ln in c.lines:
                if ln.special == LOWER_TO_LOWEST:
                    ln.role = "release" if ln.args[0] in sync_tags else ("next" if ln.args[0] in clock_of else "other")
                elif ln.special == FLOOR_MOVE_TO and ln.args[0] in self.freeze_tags:
                    ln.role = "freeze"
                elif ln.special == TELEPORT_NOFOG:
                    ln.role = "teleport"
                elif ln.special == EXEC_ALWAYS and ln.args[0] in COPY_SCRIPTS:
                    ln.role = "copy"
                else:
                    ln.role = "other"
        nexts = {ln.args[0] for c in self.clocks for ln in c.lines if ln.role == "next"}
        first = [t for t in clock_of if t not in nexts]
        if len(first) != 1:
            raise ValueError(f"cannot order the layers: {len(first)} clocks start without a previous layer "
                             f"(start tags {sorted(first)})")
        order, tag = [], first[0]
        while tag in clock_of and tag not in order:
            order.append(tag)
            nxt = [ln.args[0] for ln in clock_of[tag].lines if ln.role == "next"]
            tag = nxt[0] if nxt else None
        self.layer_tags = order
        layer = {t: i for i, t in enumerate(order)}
        for t, c in clock_of.items():
            c.layer = layer.get(t, -1)
        for l in self.lanes:
            l.layer = layer.get(tag_of(l.chain, l.start), -1)
        self.layer_clock = [clock_of[t] for t in order]


# --- predicted tics -------------------------------------------------------------------------------------------------
CEIL_OPEN = 16000 * FRAC          # a ceiling gate the doll passes under (its value is never below walk + 56 then)
RUN_LIMIT = 20000                 # tics a single run may take before the prediction gives up
# measured in ViZDoom 1.3.2 (module docstring, "Measurements"):
SEPARATION = 1        # tics between two plane moves on one sector so that neither is lost (same tic: one is lost)
REST_BAND = FRAC      # released dolls rest at face - push/32 or up to 1 unit further west
SYNC_SETTLE = 4       # tics from the predicted arrival at a sync barrier until the doll rests there
TELE_MIN = 1          # a clock teleport must come at least one tic after the chunk's last release line
COPY_SETTLE = 0       # opt 3: the freeze/next line may share the copy line's tic (copy_settle 4 and 16 units exact);
COPY_WRITE = 3        # what matters is that the next layer reads its gates after copy tic + 1 (script) + 2 (movers)
COPY_LIGHTS = 0       # opt 3: a light hit in the copy line's tic counts when the layer's lane dolls think before its
#                       clock (spawned earlier, as pure.py emits them); else 1. 24-32-16-8 MLP: -1 tic 0/6 exact, 0 6/6


@dataclass
class LaneTiming:
    arrival: tuple | None = None          # (lo, hi): first tic blocked at the sync barrier, over the gate variants
    release: int | None = None            # tic the clock crosses this lane's release line
    cells: list = field(default_factory=list)       # (lo, hi) crossing tic per cell, None if never reached
    reads: list = field(default_factory=list)       # (lo, hi) tic each gate is first consulted
    freeze_exit: int | None = None        # last tic its box overlaps one of its freeze sectors (if it passes them)
    variants: int = 0                     # gate step-height combinations simulated after the sync barrier
    capped: bool = False                  # too many combinations: only the extremes were simulated


class Timeline:
    """Predicted tic of every clock line and lane cell, counted from tic 0 = the tic TileRunner.start drops layer 0's
    start barriers (ACS script 7 acts like a line crossed at tic 0). values: {value name: int} predicts one input
    exactly (gates blocked or passed by the given values); without it every gate is passed and each floor gate's
    step height ranges over what the gates before it allow (intervals)."""

    def __init__(self, layout: Layout, values: dict | None = None, max_variants: int = 256):
        self.lo = layout
        self.md = layout.md
        self.values = values
        self.max_variants = max_variants
        speeds = [a[1] for sp, a in self.md.lines.values() if sp == LOWER_TO_LOWEST]
        self.speed0 = (max(speeds) if speeds else 150000) * FRAC // 8
        self.cache: dict = {}
        self.layer_start: list[int] = []
        self.clock: list[list] = []                    # per layer: crossing tic per clock line (None if never)
        self.release: dict[int, int] = {}              # wave tag -> release tic
        self.lane: dict[int, LaneTiming] = {}          # lane index -> timing
        by_layer: dict[int, list[int]] = {}
        for i, l in enumerate(layout.lanes):
            by_layer.setdefault(l.layer, []).append(i)
        t = 0
        for s, clock in enumerate(layout.layer_clock):
            self.layer_start.append(t)
            tics = self._clock(clock, t)
            self.clock.append(tics)
            for ln, tic in zip(clock.lines, tics):
                if ln.role == "release" and tic is not None:
                    self.release[ln.args[0]] = tic
            for i in by_layer.get(s, []):
                self.lane[i] = self._lane(layout.lanes[i], t)
            nxt = [tic for ln, tic in zip(clock.lines, tics) if ln.role == "next"]
            if s + 1 < len(layout.layer_clock):
                if not nxt or nxt[0] is None:
                    raise ValueError(f"layer {s}: the clock never reaches the line that starts layer {s + 1}")
                t = nxt[0]

    # helpers
    def _rest(self, chain: list[int], k: int) -> int:
        """Fixed-point x of a doll resting against the barrier at chain position k (P_SlideMove leaves 1/32 of a push)."""
        md = self.md
        push = md.push.get(int(md.tag[chain[k - 1]]), 0)
        return int(md.x0[chain[k]]) * FRAC - RADIUS - push // 32

    def _drop(self, chain: list[int], k: int, tic: int, speed: int) -> tuple:
        md = self.md
        f = int(md.floor[chain[k]])
        around = [int(md.floor[chain[j]]) for j in (k - 1, k + 1) if 0 <= j < len(chain)]
        return (f * FRAC, min([f] + around) * FRAC, speed, tic)

    def _run(self, tr: Track, x: int, t0: int, floors: dict, ceils: dict, stop: int | None = None,
             halt: int | None = None, v0: int = 0) -> tuple[list[int], list[int], int]:
        reach = (stop if stop is not None else halt) - x + 2 * RADIUS + 64 * FRAC
        key = (tr.key(x, floors, ceils, t0, reach), (stop or 0) - x, (halt or 0) - x, stop is None, v0)
        hit = self.cache.get(key)
        if hit is None:
            pos, waits, v = tr.run(x, t0, t0 + RUN_LIMIT, floors, ceils, stop=stop, halt=halt, v0=v0)
            hit = self.cache[key] = ([p - x for p in pos], [w - t0 for w in waits], v)
        return [p + x for p in hit[0]], [w + t0 for w in hit[1]], hit[2]

    @staticmethod
    def _first(pos: list[int], t0: int, x: int) -> int | None:
        """Tic in which the doll's centre first passes x (P_PointOnLineSide: crossing means x' > X >= x)."""
        i = bisect.bisect_right(pos, x)
        return t0 + i if i < len(pos) else None

    def _read(self, g: Gate, pos: list[int], t0: int, waits: list[int]) -> int | None:
        """Tic the doll's box first reaches the gate sector (it passes then, or is blocked from then on)."""
        edge = g.x0 * FRAC - RADIUS
        r = self._first(pos, t0, edge)
        if r is None and waits and edge - 2 * FRAC < pos[-1] <= edge:
            r = waits[-1]
        return r

    def _speed_of(self, tag: int) -> int:
        for sp, a in self.md.lines.values():
            if sp == LOWER_TO_LOWEST and a[0] == tag:
                return a[1] * FRAC // 8
        return self.speed0

    def _clock(self, clock: Clock, t0: int) -> list:
        md = self.md
        tics: list = [None] * len(clock.lines)
        t, x, v, carry = t0, None, 0, 0
        for seg, (chain, sx) in enumerate(clock.segments):
            idx = [i for i, ln in enumerate(clock.lines) if ln.segment == seg]
            if not idx:
                break
            if seg == 0:
                k = clock.start
                tr = Track(md, chain, {chain[k]})
                x = self._rest(chain, k)
                floors = {tr.at[chain[k]]: self._drop(chain, k, t, self._speed_of(int(md.tag[chain[k]])))}
            else:
                tr = Track(md, chain)
                x, floors = int(round(sx * FRAC)) + carry, {}
            stop = clock.lines[idx[-1]].x * FRAC
            pos, _, v = self._run(tr, x, t, floors, {}, stop=stop, v0=v)
            for i in idx:                               # lines behind the start (the home line) are never crossed
                tics[i] = self._first(pos, t, clock.lines[i].x * FRAC) if clock.lines[i].x * FRAC > x else None
            last = tics[idx[-1]]
            if last is None or clock.lines[idx[-1]].role != "teleport":
                break
            # Teleport_NoFog keeps the velocity, and the steps of that tic's move left after the step that crossed the
            # line continue from the destination (P_XYMovement moves in steps of at most radius - 1; measured)
            j = last - t
            d = pos[j] - pos[j - 1]
            steps = 1 + d // MAXMOVE if d > MAXMOVE else 1
            k = next(k for k in range(1, steps + 1) if pos[j - 1] + d * k // steps > stop)
            carry, t = d - d * k // steps, last
        return tics

    def _gate_options(self, lane: Lane, upto: int) -> list[list]:
        """Per gate (index < upto): the floor values (fixed) or ceilings to simulate."""
        out, box = [], {}
        for g in lane.gates[:upto]:
            lo, hi = box.get(g.var, (-BIG, BIG))
            lo, hi = max(lo, g.lo), min(hi, g.hi)
            box[g.var] = (lo, hi)
            if self.values is not None:
                v = self._value(g)
                out.append([("floor", v * FRAC)] if g.kind == "floor" else [("ceil", v * FRAC)])
                continue
            if g.kind == "ceil":
                out.append([("ceil", CEIL_OPEN)])
                continue
            neg = g.key.split(":", 1)[1].startswith("-")
            plo, phi = (-hi, -lo) if neg else (lo, hi)      # range of the port value itself
            hs = sorted({max(0, h) for h in range(max(plo - g.walk, -1), min(phi - g.walk, STEP_UP_UNITS) + 1)}
                        ) if phi - g.walk >= -1 and plo - g.walk <= STEP_UP_UNITS else []
            if phi - g.walk < 0:
                hs = [0]
            out.append([("floor", (g.walk + h) * FRAC) for h in hs])
        return out

    def _value(self, g: Gate) -> int:
        name = g.key.split(":", 1)[1]
        if name in self.values:
            return int(self.values[name])
        v = int(self.values[name.lstrip("-")])
        return -v if name.startswith("-") else v

    def _lane(self, lane: Lane, t_start: int) -> LaneTiming:
        md, lt = self.md, LaneTiming()
        chain, ks = lane.chain, lane.sync
        if ks is None:
            return lt
        lt.reads = [None] * len(lane.gates)
        k0 = lane.start
        pre = [i for i, g in enumerate(lane.gates) if g.pos < ks]
        post = [i for i, g in enumerate(lane.gates) if g.pos > ks]
        # phase A: start barrier -> sync barrier
        sub = chain[:ks + 1]
        dyn = {chain[k0], chain[ks]} | {lane.gates[i].sector for i in pre}
        tr = Track(md, sub, dyn)
        x = self._rest(chain, k0)
        floors0 = {tr.at[chain[k0]]: self._drop(chain, k0, t_start, self._speed_of(int(md.tag[chain[k0]])))}
        halt = int(md.x0[chain[ks]]) * FRAC - RADIUS - 8 * FRAC
        opts = self._gate_options(lane, len(pre))
        arrivals, exits = [], []

        for combo in itertools.product(*opts) if opts else [()]:
            floors, ceils = dict(floors0), {}
            for i, (kind, v) in zip(pre, combo):
                (floors if kind == "floor" else ceils)[tr.at[lane.gates[i].sector]] = v
            pos, waits, _ = self._run(tr, x, t_start, floors, ceils, halt=halt)
            for i in pre:
                r = self._read(lane.gates[i], pos, t_start, waits)
                if r is not None:
                    lt.reads[i] = (min(r, lt.reads[i][0]), max(r, lt.reads[i][1])) if lt.reads[i] else (r, r)
            if pos[-1] > halt and waits:
                arrivals.append(waits[-1])
                exits += [self._first(pos, t_start, int(md.x1[chain[k]]) * FRAC + RADIUS - 1)
                          for k in lane.freeze if k < ks]
        if not arrivals:                           # blocked before its sync barrier for the given values
            return lt
        lt.arrival = (min(arrivals), max(arrivals))
        wave = int(md.tag[chain[ks]])
        lt.release = self.release.get(wave)
        if lt.release is None:
            return lt
        # phase B: from rest at the sync barrier to the end of the lane
        sub = chain[ks - 1:]
        dyn = {chain[ks]} | {lane.gates[i].sector for i in post}
        tr = Track(md, sub, dyn)
        x = self._rest(chain, ks)
        floors0 = {tr.at[chain[ks]]: self._drop(chain, ks, lt.release, self._speed_of(wave))}
        ends = [c.x for c in lane.cells] + [lane.gates[i].x0 for i in post] + \
               [int(md.x1[chain[k]]) + RADIUS for k in lane.freeze if k > ks]
        stop = max(ends) * FRAC + REST_BAND if ends else x
        opts = self._gate_options(lane, len(lane.gates))[len(pre):]
        combos = list(itertools.product(*opts)) if opts else [()]
        if len(combos) > self.max_variants:
            lt.capped = True
            ext = [sorted({o[0], o[-1]}, key=lambda c: c[1]) for o in opts]
            combos = list(itertools.product(*ext))
        lt.variants = len(combos)
        lt.cells = [None] * len(lane.cells)
        for combo in combos:
            floors, ceils = dict(floors0), {}
            for i, (kind, v) in zip(post, combo):
                (floors if kind == "floor" else ceils)[tr.at[lane.gates[i].sector]] = v
            pos, waits, _ = self._run(tr, x, lt.release, floors, ceils, stop=stop)
            # the doll rests at the canonical x or up to REST_BAND further west: crossing X from there is crossing
            # X + REST_BAND from the canonical rest
            for j, c in enumerate(lane.cells):
                t, u = self._first(pos, lt.release, c.x * FRAC), self._first(pos, lt.release, c.x * FRAC + REST_BAND)
                if t is not None:
                    u = t if u is None else u
                    lt.cells[j] = (min(t, lt.cells[j][0]), max(u, lt.cells[j][1])) if lt.cells[j] else (t, u)
            for i in post:
                r = self._read(lane.gates[i], pos, lt.release, waits)
                if r is not None:
                    lt.reads[i] = (min(r, lt.reads[i][0]), max(r, lt.reads[i][1])) if lt.reads[i] else (r, r)
            exits += [self._first(pos, lt.release, int(md.x1[chain[k]]) * FRAC + RADIUS + REST_BAND)
                      for k in lane.freeze if k > ks]
        exits = [e for e in exits if e is not None]
        lt.freeze_exit = max(exits) if exits else None
        return lt


# --- the check ------------------------------------------------------------------------------------------------------
def _box(lane: Lane, upto: int, skip: set) -> dict:
    """{var: (lo, hi)}: what the gates west of a cell demand of the values (values in `skip` change during the layer
    and are left out)."""
    box: dict = {}
    for g in lane.gates[:upto]:
        if g.var in skip:
            continue
        lo, hi = box.get(g.var, (-BIG, BIG))
        box[g.var] = (max(lo, g.lo), min(hi, g.hi))
    return box


def _joint(a: dict, b: dict) -> dict | None:
    """Intersection of two boxes, None when some value would need disjoint intervals (the dolls exclude each other)."""
    out = dict(a)
    for var, (lo, hi) in b.items():
        l0, h0 = out.get(var, (-BIG, BIG))
        lo, hi = max(lo, l0), min(hi, h0)
        if lo > hi:
            return None
        out[var] = (lo, hi)
    return out if all(lo <= hi for lo, hi in out.values()) else None


def _pick(lo: int, hi: int, cap: int = 12000) -> int:
    """The value nearest 0 in [lo, hi], inside +-cap (the port range) when the interval allows it."""
    v = min(max(0, lo), hi)
    return int(max(min(v, max(lo, min(hi, cap))), min(hi, max(lo, -cap))))


@dataclass
class ScheduleReport:
    """Result of check_schedule. ok: no hazard and every margin at or above its measured minimum. hazards: dicts with
    layer, target (plane kind and tag), value (the port value name moved), dolls (two descriptions), tics (each
    doll's predicted crossing interval), separation (fewest tics between them), needed (tics they must be apart),
    witness. margins: dicts with name (a Timing field), layer, tics, minimum, ok, detail."""
    wad: str
    ok: bool
    hazards: list
    margins: list
    layers: list
    hazard_count: int = 0
    notes: list = field(default_factory=list)
    seconds: float = 0.0
    plane_errors: list = field(default_factory=list)       # planes.py bound, MLP given with net= (else not checked)

    def to_dict(self) -> dict:
        """JSON-ready dict (the measured minimums under "minimums")."""
        d = json.loads(json.dumps(asdict(self), default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o)))
        d["minimums"] = dict(separation=SEPARATION, sync_settle=SYNC_SETTLE, tele_gap=TELE_MIN,
                             copy_settle=COPY_SETTLE, rest_band_units=REST_BAND / FRAC)
        return d

    def __str__(self) -> str:
        out = [f"schedule check of {self.wad}: {'OK' if self.ok else 'FAILED'} ({self.hazard_count} hazards, "
               f"{sum(not m['ok'] for m in self.margins)} margins below their minimum; {self.seconds:.1f} s)"]
        out += [f"PLANE BOUND: {e}" for e in self.plane_errors]
        out.append("layer  dolls  waves  plane cells  first release  last effect  next layer")
        for L in self.layers:
            out.append(f"{L['layer']:>5}  {L['dolls']:>5}  {L['waves']:>5}  {L['plane_cells']:>11}  "
                       f"{_t(L['first_release']):>13}  {_t(L['last_effect']):>11}  {_t(L['next']):>10}")
        out.append("margin         layer   tics  minimum  ok   detail")
        worst: dict = {}
        for m in self.margins:                        # the tightest margin of each check and layer
            key = (m["name"], m["layer"], m["check"])
            if key not in worst or m["tics"] - m["minimum"] < worst[key]["tics"] - worst[key]["minimum"]:
                worst[key] = m
        for m in sorted(worst.values(), key=lambda m: (m["layer"], m["name"], m["check"])):
            ok = "yes" if m["ok"] else "NO "
            out.append(f"{m['name']:<14} {m['layer']:>5}  {m['tics']:>5}  {m['minimum']:>7}  {ok:<3}  {m['detail']}")
        for h in self.hazards[:20]:
            a, b = h["dolls"]
            out.append(f"HAZARD layer {h['layer']}: {h['target'][0]} plane tag {h['target'][1]} ({h['value']}): "
                       f"doll {a['doll']} at tics {_iv(h['tics'][0])} and doll {b['doll']} at {_iv(h['tics'][1])}, "
                       f"{h['separation']} tics apart (need {h['needed']})")
            out.append(f"  {'certain' if h['certain'] else 'possible'}; witness: {h['witness']['text']}")
        if self.hazard_count > 20:
            out.append(f"... {self.hazard_count - 20} more hazards")
        out += [f"note: {n}" for n in self.notes]
        return "\n".join(out)


def _t(v) -> str:
    return "-" if v is None else str(v)


def _iv(iv) -> str:
    return str(iv[0]) if iv[0] == iv[1] else f"{iv[0]}-{iv[1]}"


class _Checker:
    def __init__(self, layout: Layout, timeline: Timeline, net=None, witnesses: int = 8):
        self.lo, self.tl, self.net, self.max_witnesses = layout, timeline, net, witnesses
        self.md = layout.md
        self.notes: list[str] = []
        lay = layout
        self.written: dict[int, set] = {}            # layer -> value names its plane cells move
        self.writers: dict[int, list] = {}           # port tag -> [(layer, lo, hi, ticks)]
        for i, lane in enumerate(lay.lanes):
            lt = timeline.lane.get(i)
            for j, c in enumerate(lane.cells):
                if c.target and c.target[0] in ("floor", "ceil"):
                    key = lay.port_key.get(c.target[1])
                    if key:
                        self.written.setdefault(lane.layer, set()).add(key.split(":", 1)[1].lstrip("-"))
                    if lt and lt.cells and lt.cells[j]:
                        self.writers.setdefault(c.target[1], []).append((lane.layer, *lt.cells[j], c.ticks))
        # corridor layers: some lane has a gate after its sync barrier
        self.corridor = {l.layer: True for l in lay.lanes if l.sync is not None and any(g.pos > l.sync for g in l.gates)}

    def lane_text(self, i: int) -> dict:
        lane = self.lo.lanes[i]
        md = self.md
        return dict(lane=i, doll=lane.doll, x=lane.x, y=lane.y, wave=int(md.tag[lane.chain[lane.sync]]),
                    gates=[f"{g.key} ({'<=' if g.kind == 'floor' else '>='} "
                           f"{g.walk + (STEP_UP_UNITS if g.kind == 'floor' else DOLL_H_UNITS)})" for g in lane.gates])

    def hazards(self) -> tuple[list, int, list]:
        lay, tl = self.lo, self.tl
        events: dict = {}
        for i, lane in enumerate(lay.lanes):
            lt = tl.lane.get(i)
            if not lt or not lt.cells:
                continue
            skip = self.written.get(lane.layer, set())
            for j, c in enumerate(lane.cells):
                if c.target is None or lt.cells[j] is None:
                    continue
                events.setdefault(c.target, []).append((lt.cells[j][0], lt.cells[j][1], c.ticks, i, j,
                                                        _box(lane, c.gates, skip)))
        found, count, corr = [], 0, []
        for target, evs in events.items():
            evs.sort()
            for a in range(len(evs)):
                lo_a, hi_a, k_a, i_a, j_a, box_a = evs[a]
                for b in range(a + 1, len(evs)):
                    lo_b, hi_b, k_b, i_b, j_b, box_b = evs[b]
                    need = max(k_a, k_b, SEPARATION)
                    if lo_b - hi_a >= need:
                        break
                    gap = max(0, lo_b - hi_a, lo_a - hi_b)
                    if gap >= need:
                        continue
                    joint = box_a if i_a == i_b else _joint(box_a, box_b)
                    if joint is None:
                        continue                          # mutually exclusive gates: never both pass
                    count += 1
                    if len(found) < 10000:
                        found.append(dict(layer=lay.lanes[i_a].layer, target=target, value=self.value_of(target),
                                          dolls=[self.lane_text(i_a), self.lane_text(i_b)],
                                          tics=[(lo_a, hi_a), (lo_b, hi_b)], separation=gap, needed=need,
                                          certain=lo_a == hi_a and lo_b == hi_b, box=joint))
            corr += self._corr(evs)
        # certain hazards first: both crossing tics are exact, so every input that passes both gates loses a move;
        # the others collide for some rest positions of the dolls (REST_BAND) or some gate step heights
        found.sort(key=lambda h: (not h["certain"], h["separation"] - h["needed"], h["layer"]))
        found = found[:1000]
        for h in found[:self.max_witnesses]:
            h["witness"] = self.witness(h["box"])
        for h in found[self.max_witnesses:]:
            h["witness"] = dict(text="(not computed; raise `witnesses`)", found=False)
        return found, count, corr

    def _corr(self, evs) -> list:
        """Fewest tics between two non-exclusive writes of one plane by dolls of different waves in corridor layers."""
        out = []
        lay = self.lo
        for a in range(len(evs) - 1):
            lo_a, hi_a, _, i_a, _, box_a = evs[a]
            la = lay.lanes[i_a]
            if not self.corridor.get(la.layer):
                continue
            for b in range(a + 1, min(len(evs), a + 64)):
                lo_b, hi_b, _, i_b, _, box_b = evs[b]
                lb = lay.lanes[i_b]
                if lb.layer != la.layer or lb.chain[lb.sync] == la.chain[la.sync]:
                    continue
                if self.md.tag[lb.chain[lb.sync]] == self.md.tag[la.chain[la.sync]] or _joint(box_a, box_b) is None:
                    continue
                out.append((la.layer, max(0, lo_b - hi_a, lo_a - hi_b)))
                break
        return out

    def value_of(self, target) -> str:
        if target[0] in ("floor", "ceil"):
            key = self.lo.port_key.get(target[1])
            return key.split(":", 1)[1] if key else f"tag {target[1]}"
        return f"action {target[1]}{target[2]}"

    # witnesses
    def witness(self, box: dict) -> dict:
        inputs = {k.split(":", 1)[1].lstrip("-") for k in self.lo.inputs}
        values = {var: _pick(lo, hi) for var, (lo, hi) in box.items()}
        deep = sorted(v for v in values if v not in inputs)
        cond = ", ".join(f"{v}={values[v]}" for v in sorted(values)) or "any input"
        if not deep:
            vec = self._vector(values, inputs)
            text = f"inputs {cond}" + (f"; x = {vec}" if vec is not None else "")
            return dict(found=True, values=values, x=vec, text=text)
        if self.net is None:
            return dict(found=False, values=values, text=f"needs {cond}; {', '.join(deep)} are not inputs and no "
                                                          f"network was given to search for an input")
        hit = self._search(box)
        if hit is None:
            return dict(found=False, values=values, text=f"needs {cond}; no input found by random search and "
                                                          f"hill climbing on the network's reference")
        kind, inp, vals = hit
        return dict(found=True, values=vals, **{kind: inp}, text=f"{kind} = {inp} (gives {cond})")

    def _vector(self, values: dict, inputs: set) -> list | None:
        idx = {}
        for v in inputs:
            if not (v.startswith("x") and v[1:].isdigit()):
                return None
            idx[v] = int(v[1:])
        if not idx:
            return None
        x = [0] * (max(idx.values()) + 1)
        for v, i in idx.items():
            x[i] = int(values.get(v, 0))
        return x

    def _evaluate(self):
        """(sampler, values(input) -> {var: int}, kind) for the attached network, or None."""
        net = self.net
        rng = np.random.default_rng(0)
        if hasattr(net, "weights") and hasattr(net, "reference"):          # kit.MLP
            def sample(scale):
                return rng.integers(-scale, scale + 1, net.n_in)

            def values(x):
                ref = net.reference(x)
                out = {f"x{i}": int(v) for i, v in enumerate(x)}
                for l, h in enumerate(ref["hidden"], 1):
                    out.update({f"h{l}_{j}": int(v) for j, v in enumerate(h)})
                out.update({f"y{j}": int(v) for j, v in enumerate(ref["y"])})
                return out
            return sample, values, "x", lambda x: [int(v) for v in x], lambda x, s: _mutate_vec(x, s, rng)
        if hasattr(net, "vocab") and hasattr(net, "T"):                     # intref.QModel
            from .intref import forward

            def sample(scale):
                return [0] + [int(rng.integers(1, len(net.vocab))) for _ in range(net.T - 1)]

            def values(tokens):
                trace: dict = {}
                forward(net, list(tokens), trace)
                out = {f"tok:{t}": int(v) for t, v in enumerate(tokens)}
                for k, arr in trace.items():
                    a = np.asarray(arr)
                    for idx in np.ndindex(a.shape):
                        out[":".join([k, *map(str, idx)])] = int(a[idx])
                return out
            return sample, values, "tokens", list, lambda x, s: _mutate_tokens(x, net, rng)
        return None

    def _search(self, box: dict, budget: int = 3000):
        ev = self._evaluate()
        if ev is None:
            return None
        sample, values, kind, show, mutate = ev

        def score(inp):
            try:
                if hasattr(self.net, "check"):
                    self.net.check(np.asarray(inp))
                v = values(inp)
            except (ValueError, KeyError, IndexError):
                return None, None
            s = 0
            for var, (lo, hi) in box.items():
                if var not in v:
                    return None, None
                s += max(0, lo - v[var], v[var] - hi)
            return s, v

        best, best_s = None, None
        for n in range(budget):
            inp = sample([10, 100, 1000][n % 3]) if best is None or n % 4 == 0 else mutate(best, best_s)
            s, v = score(inp)
            if s is None:
                continue
            if best_s is None or s <= best_s:
                best, best_s = inp, s
                if s == 0:
                    return kind, show(inp), {var: v[var] for var in box}
        return None


def _mutate_vec(x, s, rng):
    y = np.array(x).copy()
    i = rng.integers(len(y))
    y[i] += int(rng.integers(-max(1, s), max(1, s) + 1))
    return y


def _mutate_tokens(x, net, rng):
    y = list(x)
    y[int(rng.integers(1, len(y)))] = int(rng.integers(1, len(net.vocab)))
    return y


def _margins(ck: _Checker, corr: list) -> tuple[list, list]:
    lay, tl, md = ck.lo, ck.tl, ck.md
    margins, layers = [], []

    def add(name, layer, tics, minimum, detail, check):
        margins.append(dict(name=name, layer=layer, check=check, tics=int(tics), minimum=int(minimum),
                            ok=bool(tics >= minimum), detail=detail))

    copy_tic = {}
    for s, clock in enumerate(lay.layer_clock):
        tics = tl.clock[s]
        roles = [(ln, t) for ln, t in zip(clock.lines, tics) if t is not None]
        rel = sorted(t for ln, t in roles if ln.role == "release")
        nxt = next((t for ln, t in roles if ln.role == "next"), None)
        frz = [t for ln, t in roles if ln.role == "freeze"]
        copy = next((t for ln, t in roles if ln.role == "copy"), None)
        lanes = [i for i, l in enumerate(lay.lanes) if l.layer == s]
        cells = [(lay.lanes[i].cells[j], tl.lane[i].cells[j]) for i in lanes if tl.lane[i].cells
                 for j in range(len(tl.lane[i].cells)) if tl.lane[i].cells[j] is not None]
        last = max((iv[1] for _, iv in cells), default=None)
        layers.append(dict(layer=s, dolls=len(lanes), waves=len({md.tag[lay.lanes[i].chain[lay.lanes[i].sync]]
                                                                 for i in lanes if lay.lanes[i].sync is not None}),
                           plane_cells=sum(1 for c, _ in cells if c.target and c.target[0] != "action"),
                           first_release=rel[0] if rel else None, last_effect=last, next=nxt,
                           corridor=bool(ck.corridor.get(s)), copy=copy))
        # serial / corr_gap: wave release spacing, or same-plane spacing of corridor waves
        gaps = [b - a for a, b in zip(rel, rel[1:])]
        mine = [g for L, g in corr if L == s]
        if ck.corridor.get(s) and (mine or gaps):
            add("corr_gap", s, min(mine) if mine else min(gaps), SEPARATION,
                "fewest tics between two waves' writes of one plane" if mine else
                f"fewest tics between two of {len(rel)} wave releases", "spacing")
        elif gaps:
            add("serial", s, min(gaps), SEPARATION, f"fewest tics between two of {len(rel)} wave releases", "spacing")
        # sync_wait: every doll rests at its sync barrier before its wave is released
        waits = [tl.lane[i].release - tl.lane[i].arrival[1] for i in lanes
                 if tl.lane[i].arrival is not None and tl.lane[i].release is not None]
        if waits:
            add("sync_wait", s, min(waits), SYNC_SETTLE, "release minus the latest arrival at a sync barrier", "arrival")
        # tele_gap: a clock teleport never shares a tic with the release line before it
        for i, (ln, t) in enumerate(zip(clock.lines, tics)):
            if ln.role == "teleport" and t is not None:
                before = [u for l2, u in zip(clock.lines[:i], tics[:i]) if l2.segment == ln.segment
                          and l2.role != "other" and u is not None]
                if before:
                    add("tele_gap", s, t - max(before), TELE_MIN, f"teleport at tic {t} after the chunk's last line",
                        "teleport")
        # clock_margin: freeze lines after the last doll left a freeze sector
        exits = [tl.lane[i].freeze_exit for i in lanes if tl.lane[i].freeze_exit is not None]
        if frz and exits:
            add("clock_margin", s, min(frz) - max(exits), 0, "first freeze line minus the last doll leaving a "
                                                                  "freeze sector", "freeze")
        if copy is not None:
            copy_tic[s] = copy
            lights = [iv[1] for c, iv in cells]
            if lights:
                # same tic counts only when every lane doll of the layer thinks before the clock (spawned earlier)
                first = all(lay.lanes[i].doll < clock.doll for i in lanes)
                add("clock_margin", s, copy - max(lights), COPY_LIGHTS if first else COPY_LIGHTS + 1,
                    "copy line minus the last effect line", "copy")
            after = [t for t in frz + ([nxt] if nxt is not None else [])]
            if after:
                add("copy_settle", s, min(after) - copy, COPY_SETTLE, "freeze/next line minus the copy line", "settle")
    # clock_margin: every gate is read after the last plane move that writes it has finished (read after write)
    written = set(ck.writers)
    raw: dict = {}
    for i, lane in enumerate(lay.lanes):
        lt = tl.lane.get(i)
        if not lt:
            continue
        for g, r in zip(lane.gates, lt.reads):
            if r is None:
                continue
            tag = int(md.tag[g.sector])
            for (w, lo, hi, ticks) in ck.writers.get(tag, []):
                if w < lane.layer:
                    slack = r[0] - hi - ticks
                    if w not in raw or slack < raw[w][0]:
                        raw[w] = (slack, r[0] - hi, ticks, g.key, lane.layer)
            if tag not in written and g.key not in lay.inputs:             # opt 3: written by a copy script
                src = [s for s in copy_tic if s < lane.layer]
                if src:
                    w = max(src)
                    slack = r[0] - copy_tic[w] - COPY_WRITE
                    if w not in raw or slack < raw[w][0]:
                        raw[w] = (slack, r[0] - copy_tic[w], COPY_WRITE, g.key, lane.layer)
    for w, (slack, tics, ticks, key, r) in sorted(raw.items()):
        add("clock_margin", w, tics, ticks, f"layer {r} reads {key} {tics} tics after layer {w}'s last write of it",
            "read")
    return margins, layers


def check_schedule(build, *, wad=None, net=None, witnesses: int = 8, max_variants: int = 256) -> ScheduleReport:
    """Check the schedule of a compiled one-map brain against the engine model of this module.

    build: what df.compile returns (a Brain), a .wad path (its .json next to it), or a TileMeta from compile_pure
    together with wad=. net: the network, to search inputs for witnesses whose gates read hidden values (a Brain
    brings its own). Returns a ScheduleReport; report.ok is False on any hazard or margin below its minimum."""
    import time
    t0 = time.time()
    if hasattr(build, "meta") and hasattr(build, "wad"):              # a Brain
        if build.info.get("mode") == "tiles" or build.info.get("tiles", 1) != 1:
            raise ValueError("check_schedule checks one-map brains; tile brains have no waves")
        wad, net = Path(build.wad), net if net is not None else getattr(build, "net", None)
    elif hasattr(build, "layout") and hasattr(build, "ports"):         # a TileMeta
        if wad is None:
            raise ValueError("a TileMeta does not know its .wad: pass wad=")
        wad = Path(wad)
    else:
        wad = Path(build)
        if wad.suffix == ".json":
            wad = wad.with_suffix(".wad")
    meta = json.loads(wad.with_suffix(".json").read_text())
    if "stage_stats" not in meta.get("layout", {}):
        raise ValueError(f"{wad}: not a one-map brain (its .json has no layout.stage_stats)")
    md = MapData.parse(textmap(wad))
    layout = Layout(md, meta)
    tl = Timeline(layout, max_variants=max_variants)
    ck = _Checker(layout, tl, net=net, witnesses=witnesses)
    hazards, count, corr = ck.hazards()
    margins, layers = _margins(ck, corr)
    notes = list(ck.notes)
    capped = sum(1 for lt in tl.lane.values() if lt.capped)
    if capped:
        notes.append(f"{capped} lanes had more than {max_variants} gate step-height combinations; only the extreme "
                     "heights were simulated for them")
    if layout.skipped:
        notes.append(f"{len(layout.skipped)} player starts are neither lanes nor clocks (ignored)")
    plane_errors: list = []
    if net is not None and hasattr(net, "plane_errors"):
        plane_errors = list(net.plane_errors())            # the plane-move contract of planes.py, worst case over inputs
    else:
        notes.append("the plane-move bound (planes.py: a plane that starts a move beyond +-14,018 keeps its mover busy "
                     "and later moves on it are lost) is not checked: it depends on the net's weights, which the map "
                     "does not hold. Give net= (`lint --net MODEL`, MLPs) to check it")
    ok = count == 0 and all(m["ok"] for m in margins) and not plane_errors
    return ScheduleReport(str(wad), ok, hazards, margins, layers, count, notes, time.time() - t0, plane_errors)
