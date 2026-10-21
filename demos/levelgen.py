"""Procedural DOOM level from a name: rooms + corridors on a grid, built with `udmf.MapBuilder`.

    info = build_level("THE COURTYARD", Path("build/levels/the_courtyard.wad"))

The name seeds everything (sha256 -> random.Random) and picks a texture theme from its words.  The WAD holds a
UDMF map (MAP01) and a MAPINFO lump whose `map MAP01 "<NAME>"` makes ZDoom show the name
on the automap / level start.  The returned dict also carries the player's route through the room graph
(`route`, a polyline) so a flythrough can be scripted.
"""
from __future__ import annotations

import hashlib
import heapq
import math
import random
import re
import struct
from dataclasses import dataclass, field
from pathlib import Path

from doomformer.udmf import MapBuilder

PITCH = 768            # grid cell size
MIN_ROOMS, MAX_ROOMS = 8, 20
ROUTE_CAP = 5300       # centre-to-centre units from start to exit (~5.8k walked): fits the 18 s flythrough
SPEC_FLICKER, SPEC_STROBE_FAST, SPEC_STROBE_SLOW, SPEC_GLOW, SPEC_FIRE = 1, 2, 3, 8, 17
SPEC_NUKAGE, SPEC_HELLSLIME = 7, 5

# --------------------------------------------------------------------------------------------- themes
# style = (wall, floor, ceiling); every tuple was picked from contact sheets of freedoom2.wad's textures.
THEMES: dict[str, dict] = {
    "techbase": dict(
        keys=("BASE", "COMPLEX", "FACTORY", "TECH", "LAB", "STATION", "REACTOR", "PLANT", "COMMAND", "BUNKER",
              "CONTROL", "OUTPOST", "WAREHOUSE", "CITY", "GRID", "CORE", "NUKE", "SECTOR", "SILO", "HANGAR", "DOCK",
              "REFINERY", "POWER", "MACHINE", "SYSTEM", "TERMINAL", "FACILITY", "ARSENAL", "MINE SHAFT"),
        styles=[("STARTAN3", "FLOOR0_1", "CEIL3_5"), ("STARG3", "FLOOR0_3", "CEIL3_5"), ("GRAY7", "FLOOR0_6", "CEIL3_5"),
                ("BROWN1", "FLAT5_5", "CEIL3_1"), ("TEKGREN2", "FLAT1", "CEIL3_5"), ("STARGR1", "FLOOR0_3", "CEIL5_1")],
        accent=["TEKWALL4", "COMPBLUE", "METAL2", "PIPE2"],
        corridor=[("GRAY1", "FLOOR0_3", "CEIL5_1"), ("STARGR1", "FLOOR0_6", "CEIL3_5"), ("BROWN96", "FLAT5_5", "CEIL3_1")],
        outdoor_floor=["FLOOR4_8", "FLAT1", "FLOOR0_3"], outdoor_wall=["STARG3", "GRAY7", "STARTAN3"],
        doors=[("DOOR1", 64, 72), ("DOOR3", 64, 72), ("BIGDOOR2", 128, 128), ("SPCDOOR1", 64, 128)],
        doortrak="DOORTRAK", step="STEP1", lift="SUPPORT2", switch="SW1COMP", lamp="TLITE6_5",
        hazard=[("NUKAGE1", SPEC_NUKAGE)], sky="SKY1", outdoor=0.15, dark=0.25, lamps=0.7, pillar_style=0,
        tiers=[[(3004, 5), (9, 2)], [(9, 3), (3001, 3), (3004, 2)], [(3001, 3), (3002, 3), (65, 3), (58, 1)],
               [(65, 3), (3002, 2), (3005, 2), (66, 1)]],
        guard=[(3005, 1), (65, 3), (3002, 2)],
        deco=[2028, 85, 86, 48, 2035, 2035], hang=[]),
    "hell": dict(
        keys=("HELL", "BLOOD", "FIRE", "DEMON", "INFERNO", "FLESH", "DEATH", "DOOM", "UNHOLY", "SKULL", "FLAME", "BURN",
              "LAVA", "DAMN", "SATAN", "EVIL", "SIN", "SCREAM", "DREAD", "TORTURE", "GORE", "ABYSS", "CURSE", "WRATH",
              "AGONY", "SOUL", "BONE", "DARK", "BLACK", "RED", "DIABOL", "BRIMSTONE", "SULFUR", "HADES", "OBLIVION"),
        styles=[("MARBLE1", "FLOOR7_2", "CEIL5_3"), ("STONE2", "FLOOR6_1", "RROCK03"), ("SKIN2", "SFLR6_4", "FLAT10"),
                ("GSTONE1", "FLOOR7_2", "CEIL5_3"), ("BIGBRIK1", "RROCK07", "CEIL5_1"), ("REDWALL", "FLAT5_3", "FLAT10")],
        accent=["SKULWALL", "FIREMAG1", "FIRELAV3", "MARBFAC2", "GSTVINE1"],
        corridor=[("STONE3", "MFLR8_4", "RROCK03"), ("BIGBRIK2", "FLOOR6_1", "CEIL5_1"), ("ASHWALL2", "MFLR8_4", "CEIL5_1")],
        outdoor_floor=["MFLR8_4", "RROCK05", "FLOOR6_1"], outdoor_wall=["SP_ROCK1", "STONE2", "ROCKRED1"],
        doors=[("BIGDOOR7", 128, 128), ("BIGDOOR5", 128, 128), ("BIGDOOR6", 128, 112)],
        doortrak="DOORTRAK", step="STEP2", lift="SUPPORT3", switch="SW1HOT", lamp=None,
        hazard=[("BLOOD1", 0), ("LAVA1", SPEC_HELLSLIME)], sky="SKY3", outdoor=0.2, dark=0.45, lamps=0.0, pillar_style=1,
        tiers=[[(3001, 4), (3004, 3), (9, 1)], [(3001, 3), (3002, 3), (3006, 2), (9, 2)],
               [(3002, 3), (3005, 2), (69, 2), (3001, 2), (66, 1)], [(3003, 2), (69, 3), (66, 2), (67, 2), (3005, 2)]],
        guard=[(3003, 2), (69, 2), (3005, 1)],
        deco=[44, 45, 46, 55, 56, 57, 70, 35, 29, 2035], hang=[49, 50, 51, 52, 53, 59, 60, 61, 62, 63, 25, 26, 27, 28]),
    "caverns": dict(
        keys=("CAVE", "CAVERN", "PIT", "MINE", "ROCK", "TUNNEL", "DEPTH", "CRYPT", "SWAMP", "MARSH", "MOUNTAIN", "GROTTO",
              "CANYON", "CAVITY", "BURROW", "DEN", "LAIR", "HOLLOW", "CHASM", "VALLEY", "RAVINE", "UNDER", "CATACOMB",
              "BOG", "FEN", "DUNGEON", "QUARRY", "SHAFT", "GORGE"),
        styles=[("ROCK4", "RROCK11", "RROCK10"), ("ROCK5", "MFLR8_2", "RROCK17"), ("BSTONE1", "RROCK12", "RROCK16"),
                ("AROCK3", "MFLR8_3", "RROCK03"), ("TANROCK5", "FLAT5_5", "RROCK18"), ("MC6", "RROCK20", "RROCK19")],
        accent=["SP_ROCK1", "A-VINES", "BROWNGRN", "AROCK2"],
        corridor=[("ROCK5", "RROCK10", "RROCK17"), ("TANROCK5", "MFLR8_2", "RROCK10"), ("BSTONE2", "RROCK11", "RROCK16")],
        outdoor_floor=["GRASS1", "GRASS2", "RROCK19"], outdoor_wall=["ROCK4", "ROCK5", "AROCK3"],
        doors=[("BIGDOOR6", 128, 112), ("DOOR1", 64, 72), ("BIGDOOR3", 128, 128)],
        doortrak="DOORTRAK", step="STEP3", lift="SUPPORT2", switch="SW1ROCK", lamp=None,
        hazard=[("NUKAGE1", SPEC_NUKAGE), ("FWATER1", 0)], sky="SKY2", outdoor=0.3, dark=0.4, lamps=0.0, pillar_style=2,
        tiers=[[(3004, 4), (3001, 3), (9, 1)], [(3001, 4), (3002, 2), (3006, 2), (58, 1)],
               [(3002, 3), (58, 2), (3005, 2), (9, 2), (3001, 2)], [(3005, 3), (3002, 3), (66, 2), (58, 2), (3003, 1)]],
        guard=[(3005, 2), (3002, 2), (66, 1)],
        deco=[47, 47, 54, 43, 34, 2035, 70], hang=[]),
    "gothic": dict(
        keys=("COURT", "CASTLE", "KEEP", "GARDEN", "TOWER", "GATE", "TEMPLE", "CHAPEL", "CHURCH", "CATHEDRAL",
              "ABBEY", "FORT", "PALACE", "CLOISTER", "MONASTERY", "SANCTUM", "SHRINE", "YARD", "PLAZA", "SQUARE",
              "HALL", "MANOR", "ARENA", "CITADEL", "RAMPART", "BASTION", "GRAVEYARD", "TOMB", "BRIDGE", "THRONE"),
        styles=[("STONE2", "FLAT1", "CEIL3_5"), ("BRICK5", "FLOOR5_3", "CEIL3_1"), ("MARBLE2", "FLOOR0_6", "CEIL3_2"),
                ("BIGBRIK2", "FLAT5_4", "CEIL5_2"), ("GSTONE1", "FLOOR7_2", "CEIL5_3"), ("STONE3", "FLOOR5_1", "CEIL5_1")],
        accent=["GSTLION", "MARBFAC3", "GSTVINE1", "SLADWALL"],
        corridor=[("STONE3", "FLOOR5_1", "CEIL5_1"), ("BIGBRIK2", "FLAT5_4", "CEIL5_2"), ("BRICK5", "FLOOR5_3", "CEIL3_1")],
        outdoor_floor=["FLAT5_4", "FLOOR5_3", "GRASS1"], outdoor_wall=["STONE2", "BRICK5", "GSTONE1"],
        doors=[("BIGDOOR5", 128, 128), ("BIGDOOR6", 128, 112), ("DOOR1", 64, 72)],
        doortrak="DOORTRAK", step="STEP2", lift="SUPPORT2", switch="SW1MARB", lamp=None,
        hazard=[("FWATER1", 0), ("LAVA1", SPEC_HELLSLIME)], sky="SKY1", outdoor=0.4, dark=0.25, lamps=0.0, pillar_style=2,
        tiers=[[(3004, 4), (9, 2), (3001, 2)], [(9, 3), (3001, 3), (65, 2)], [(3001, 2), (3002, 3), (65, 2), (69, 1)],
               [(69, 3), (3002, 2), (66, 2), (3003, 2), (65, 2)]],
        guard=[(69, 2), (3003, 1), (65, 2)],
        deco=[35, 44, 45, 46, 30, 31, 32, 33, 54, 2028, 2035], hang=[]),
}
THEME_ORDER = ["techbase", "hell", "caverns", "gothic"]
MONSTER_TYPES = {3004, 9, 3001, 3002, 58, 3006, 3005, 3003, 65, 66, 67, 68, 69, 64, 71, 84, 7, 16}
WEAPONS = {2001: "shotgun", 2002: "chaingun", 2003: "rocket launcher", 2004: "plasma rifle", 2005: "chainsaw"}


def name_seed(name: str) -> int:
    return int.from_bytes(hashlib.sha256(" ".join(name.upper().split()).encode()).digest()[:8], "big")


def pick_theme(name: str, seed: int) -> str:
    words = re.findall(r"[A-Z]+", name.upper())
    scores = {t: 0 for t in THEMES}
    for t, th in THEMES.items():
        for k in th["keys"]:
            for w in words:
                if w == k or w.startswith(k) or (len(k) >= 5 and k in w):
                    scores[t] += 1
    best = max(scores.values())
    if best:
        return max(THEME_ORDER, key=lambda t: scores[t])
    return THEME_ORDER[seed % len(THEME_ORDER)]


def slugify(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_") or "level"


# --------------------------------------------------------------------------------------------- geometry helpers
def snap(v: float, g: int = 8) -> int:
    return int(round(v / g)) * g


def subtract(p, h):
    """Rectangle p minus rectangle h (h inside p) as up to four rectangles."""
    x0, y0, x1, y1 = p
    hx0, hy0, hx1, hy1 = h
    out = []
    if hx0 > x0:
        out.append((x0, y0, hx0, y1))
    if hx1 < x1:
        out.append((hx1, y0, x1, y1))
    if hy0 > y0:
        out.append((hx0, y0, hx1, hy0))
    if hy1 < y1:
        out.append((hx0, hy1, hx1, y1))
    return out


def carve(outer, holes):
    pieces = [outer]
    for h in holes:
        nxt = []
        for p in pieces:
            ix0, iy0, ix1, iy1 = max(p[0], h[0]), max(p[1], h[1]), min(p[2], h[2]), min(p[3], h[3])
            if ix0 >= ix1 or iy0 >= iy1:
                nxt.append(p)
            else:
                nxt += subtract(p, (ix0, iy0, ix1, iy1))
        pieces = nxt
    return pieces


def contact(a, b):
    """None-overlap contact test for (x0, y0, x1, y1) tuples: 'overlap', 'edge' (shared segment) or None."""
    ox = min(a[2], b[2]) - max(a[0], b[0])
    oy = min(a[3], b[3]) - max(a[1], b[1])
    if ox > 0 and oy > 0:
        return "overlap"
    if (ox == 0 and oy > 0) or (oy == 0 and ox > 0):
        return "edge"
    return None


def rect_point_dist(r, x, y):
    dx = max(r[0] - x, 0, x - r[2])
    dy = max(r[1] - y, 0, y - r[3])
    return math.hypot(dx, dy)


def polyline_dist(poly, x, y):
    best = 1e9
    for (ax, ay), (bx, by) in zip(poly, poly[1:]):
        dx, dy = bx - ax, by - ay
        L2 = dx * dx + dy * dy
        t = 0 if L2 == 0 else max(0, min(1, ((x - ax) * dx + (y - ay) * dy) / L2))
        best = min(best, math.hypot(x - (ax + t * dx), y - (ay + t * dy)))
    return best


def rect_polyline_dist(r, poly):
    best = 1e9
    for (ax, ay), (bx, by) in zip(poly, poly[1:]):
        n = max(1, int(math.hypot(bx - ax, by - ay) // 8))
        for i in range(n + 1):
            t = i / n
            best = min(best, rect_point_dist(r, ax + t * (bx - ax), ay + t * (by - ay)))
            if best < 1:
                return best
    return best


def weighted(rng, table):
    total = sum(w for _, w in table)
    k = rng.uniform(0, total)
    for v, w in table:
        k -= w
        if k <= 0:
            return v
    return table[-1][0]


# --------------------------------------------------------------------------------------------- model
@dataclass
class Room:
    idx: int
    cell: tuple[int, int]
    x0: int
    y0: int
    x1: int
    y1: int
    floor: int = 0
    height: int = 128
    kind: str = "normal"      # normal | outdoor | dark | bright
    light: int = 160
    special: int = 0
    style: tuple = ("STARTAN3", "FLOOR0_1", "CEIL3_5")
    wall_tex: str = ""
    flat_tex: str = ""
    ceil_tex: str = ""
    holes: list = field(default_factory=list)       # dicts: kind, rect
    pieces: list = field(default_factory=list)      # rect indices of the room proper
    rects: list = field(default_factory=list)       # every rect index belonging to the room (incl. holes)
    attach: dict = field(default_factory=lambda: {"w": [], "e": [], "s": [], "n": []})
    depth: int = 0

    @property
    def cx(self):
        return (self.x0 + self.x1) // 2

    @property
    def cy(self):
        return (self.y0 + self.y1) // 2

    @property
    def box(self):
        return (self.x0, self.y0, self.x1, self.y1)


@dataclass
class Edge:
    idx: int
    a: int
    b: int
    axis: str                # 'x' (rooms side by side) or 'y'
    lo: int                  # room index at the smaller coordinate
    hi: int
    p0: int = 0              # corridor extent along the axis
    p1: int = 0
    c0: int = 0              # corridor cross-axis extent
    c1: int = 0
    kind: str = "flat"       # flat | door | stairs | lift
    door: tuple | None = None
    tree: bool = True
    rects: list = field(default_factory=list)

    @property
    def length(self):
        return self.p1 - self.p0

    @property
    def width(self):
        return self.c1 - self.c0

    @property
    def cmid(self):
        return (self.c0 + self.c1) // 2


def stairs_plan(length: int, d: int):
    """(steps, riser, depth) for a flat-lead + steps staircase over `length`, or None if it does not fit."""
    d = abs(d)
    if d == 0:
        return 0, 0, 0
    for s in (16, 8):
        if d % s:
            continue
        n = d // s
        depth = min(48, (length - 16) // n // 8 * 8)
        if depth >= 24:
            return n, s, depth
    return None


# --------------------------------------------------------------------------------------------- generator
class _Retry(Exception):
    pass


class Generator:
    def __init__(self, name: str, seed: int, attempt: int, theme: str | None):
        self.name = name
        self.rng = random.Random(seed * 1000 + attempt)
        self.theme_name = theme or pick_theme(name, seed)
        self.T = THEMES[self.theme_name]
        self.b = MapBuilder()
        self.rooms: list[Room] = []
        self.edges: list[Edge] = []
        self.tag = 100
        self.doors: list[dict] = []
        self.lifts: list[dict] = []
        self.owner: dict[int, str] = {}       # rect index -> area label
        self.rect_info: dict[int, str] = {}   # rect index -> role
        self.route: list[tuple[int, int]] = []
        self.alcove: dict | None = None
        self.counts = {"monsters": 0, "items": 0, "decor": 0, "weapons": []}

    # ---- layout
    def layout(self):
        rng = self.rng
        n = rng.randint(MIN_ROOMS, MAX_ROOMS)
        cols, rows = [(4, 3), (4, 3), (5, 3), (5, 3), (5, 4), (6, 4)][min(5, (n - 8) // 2)]
        cols, rows = (cols, rows) if rng.random() < 0.7 else (rows + 1, cols - 1) if n <= 12 else (cols, rows)
        while cols * rows < n + 2:
            cols += 1
        cells = {(rng.randrange(cols), rng.randrange(rows))}
        while len(cells) < n:
            c = rng.choice(sorted(cells))
            d = rng.choice([(1, 0), (-1, 0), (0, 1), (0, -1)])
            nc = (c[0] + d[0], c[1] + d[1])
            if 0 <= nc[0] < cols and 0 <= nc[1] < rows:
                cells.add(nc)
        sizes = [96, 128, 160, 192, 224, 256, 320]
        for i, c in enumerate(sorted(cells)):
            hw, hh = rng.choice(sizes), rng.choice(sizes)
            if hw * hh < 160 * 160:
                hw = max(hw, 160)
            ccx, ccy = c[0] * PITCH + PITCH // 2, c[1] * PITCH + PITCH // 2
            self.rooms.append(Room(i, c, ccx - hw, ccy - hh, ccx + hw, ccy + hh))
        by_cell = {r.cell: r for r in self.rooms}
        # spanning tree (randomised Prim) + a few extra loops
        adj = []
        for r in self.rooms:
            for d in ((1, 0), (0, 1)):
                o = by_cell.get((r.cell[0] + d[0], r.cell[1] + d[1]))
                if o:
                    adj.append((r.idx, o.idx))
        if len({*(a for a, _ in adj), *(b for _, b in adj)}) != len(self.rooms):
            raise _Retry("disconnected cells")
        seen = {rng.choice(self.rooms).idx}
        tree, rest = [], list(adj)
        while len(seen) < len(self.rooms):
            cand = [e for e in rest if (e[0] in seen) != (e[1] in seen)]
            if not cand:
                raise _Retry("disconnected")
            e = rng.choice(cand)
            rest.remove(e)
            tree.append(e)
            seen |= set(e)
        loops = [e for e in rest if rng.random() < 0.3]
        for a, b in tree + loops:
            ra, rb = self.rooms[a], self.rooms[b]
            if ra.cell[1] == rb.cell[1]:
                axis, (lo, hi) = "x", ((ra, rb) if ra.x0 < rb.x0 else (rb, ra))
            else:
                axis, (lo, hi) = "y", ((ra, rb) if ra.y0 < rb.y0 else (rb, ra))
            self.edges.append(Edge(len(self.edges), a, b, axis, lo.idx, hi.idx, tree=(a, b) in tree))
        for e in self.edges:
            lo, hi = self.rooms[e.lo], self.rooms[e.hi]
            if e.axis == "x":
                e.p0, e.p1 = lo.x1, hi.x0
                olo, ohi = max(lo.y0, hi.y0) + 16, min(lo.y1, hi.y1) - 16
            else:
                e.p0, e.p1 = lo.y1, hi.y0
                olo, ohi = max(lo.x0, hi.x0) + 16, min(lo.x1, hi.x1) - 16
            w = rng.choice([64, 96, 128])
            w = min(w, (ohi - olo) // 16 * 16)
            e.c0 = snap(rng.uniform(olo, ohi - w), 16)
            e.c0 = max(olo, min(e.c0, ohi - w))
            e.c1 = e.c0 + w
            sides = ("e", "w") if e.axis == "x" else ("n", "s")
            lo.attach[sides[0]].append((e.c0, e.c1))
            hi.attach[sides[1]].append((e.c0, e.c1))

    # ---- rooms: kinds, heights, floors
    def decorate_rooms(self):
        rng, T = self.rng, self.T
        for r in self.rooms:
            style = rng.choice(T["styles"])
            r.style = style
            r.height = rng.choice([128, 128, 160, 160, 192, 256])
            roll = rng.random()
            area = (r.x1 - r.x0) * (r.y1 - r.y0)
            if roll < T["outdoor"] and area >= 192 * 192 * 2:
                r.kind, r.height = "outdoor", 256
            elif roll < T["outdoor"] + T["dark"]:
                r.kind = "dark"
            elif roll > 0.85:
                r.kind = "bright"
            r.wall_tex, r.flat_tex, r.ceil_tex = style
            if r.kind == "outdoor":
                r.wall_tex = rng.choice(T["outdoor_wall"])
                r.flat_tex = rng.choice(T["outdoor_floor"])
                r.ceil_tex = "F_SKY1"
                r.light = rng.choice([232, 240, 252])
            elif r.kind == "dark":
                r.light = rng.choice([144, 152, 160, 168])
            elif r.kind == "bright":
                r.light = rng.choice([224, 236, 248])
            else:
                r.light = rng.choice([184, 192, 200, 208])
            if rng.random() < 0.18:
                r.wall_tex = rng.choice(T["accent"])
        # floors: breadth-first over the spanning tree, each tree edge must fit a staircase or a lift
        self.rooms[0].floor = 0
        placed = {0}
        deltas = [0, 0, 0, 0, 16, -16, 32, -32, 48, -48, 64, -64]
        while len(placed) < len(self.rooms):
            for e in self.edges:
                if not e.tree or ((e.a in placed) == (e.b in placed)):
                    continue
                src, dst = (e.a, e.b) if e.a in placed else (e.b, e.a)
                opts = [d for d in deltas if -48 <= self.rooms[src].floor + d <= 112]
                d = rng.choice(opts)
                self.rooms[dst].floor = self.rooms[src].floor + d
                self.rooms[dst].depth = self.rooms[src].depth + 1
                placed.add(dst)
        # edge kinds
        doors_left, lifts_left = rng.randint(2, 5), rng.randint(1, 2)
        keep = []
        for e in self.edges:
            lo, hi = self.rooms[e.lo], self.rooms[e.hi]
            d = hi.floor - lo.floor
            if d == 0:
                if doors_left > 0 and rng.random() < 0.45:
                    tex = rng.choice(T["doors"])
                    e.door = tex
                    e.c1 = e.c0 + tex[1]
                    ohi = (min(lo.y1, hi.y1) - 16) if e.axis == "x" else (min(lo.x1, hi.x1) - 16)
                    olo = (max(lo.y0, hi.y0) + 16) if e.axis == "x" else (max(lo.x0, hi.x0) + 16)
                    if tex[1] > ohi - olo:
                        e.door = None
                    else:
                        if e.c1 > ohi:
                            e.c0, e.c1 = ohi - tex[1], ohi
                        e.kind = "door"
                        doors_left -= 1
            elif abs(d) >= 32 and lifts_left > 0 and rng.random() < 0.5 and e.length >= 96:
                e.kind = "lift"
                lifts_left -= 1
            else:
                plan = stairs_plan(e.length, d)
                if plan is None:
                    if e.tree:
                        raise _Retry("stairs do not fit")
                    continue                      # drop this loop edge
                e.kind = "stairs"
            keep.append(e)
        self.edges = keep
        for i, e in enumerate(self.edges):
            e.idx = i
        # rebuild attach intervals after door resizing
        for r in self.rooms:
            r.attach = {"w": [], "e": [], "s": [], "n": []}
        for e in self.edges:
            sides = ("e", "w") if e.axis == "x" else ("n", "s")
            self.rooms[e.lo].attach[sides[0]].append((e.c0, e.c1))
            self.rooms[e.hi].attach[sides[1]].append((e.c0, e.c1))

    # ---- route: start, exit and the walk between them
    def mouth(self, e: Edge, room_idx: int):
        """(mouth point on the room wall, unit vector pointing into the room)."""
        lo = room_idx == e.lo
        if e.axis == "x":
            return ((e.p0, e.cmid), (-1, 0)) if lo else ((e.p1, e.cmid), (1, 0))
        return ((e.cmid, e.p0), (0, -1)) if lo else ((e.cmid, e.p1), (0, 1))

    def plan_route(self):
        rooms = self.rooms
        nbr: dict[int, list[tuple[int, Edge]]] = {r.idx: [] for r in rooms}
        for e in self.edges:
            nbr[e.a].append((e.b, e))
            nbr[e.b].append((e.a, e))

        def dijkstra(s):
            dist, prev = {s: 0.0}, {}
            pq = [(0.0, s)]
            while pq:
                d, u = heapq.heappop(pq)
                if d > dist.get(u, 1e18):
                    continue
                for v, e in nbr[u]:
                    nd = d + math.hypot(rooms[u].cx - rooms[v].cx, rooms[u].cy - rooms[v].cy)
                    if nd < dist.get(v, 1e18):
                        dist[v], prev[v] = nd, (u, e)
                        heapq.heappush(pq, (nd, v))
            return dist, prev

        best = None
        for s in range(len(rooms)):
            dist, prev = dijkstra(s)
            for t, d in dist.items():
                if best is None or d > best[0]:
                    best = (d, s, t)
        self.start_room = best[1]
        sr = rooms[self.start_room]
        if sr.kind == "dark":
            sr.kind, sr.light = "normal", 196
        # exit room: farthest first among rooms a ~18 s flythrough can reach (ROUTE_CAP), needs a free wall for the switch
        dist, prev = dijkstra(self.start_room)
        for t in sorted(dist, key=lambda k: (dist[k] > ROUTE_CAP, -dist[k] if dist[k] <= ROUTE_CAP else dist[k])):
            if t == self.start_room:
                continue
            alc = self._alcove_spot(rooms[t])
            if alc:
                self.exit_room, self.alcove = t, alc
                break
        else:
            raise _Retry("no exit alcove")
        path, u = [], self.exit_room
        while u != self.start_room:
            pu, e = prev[u]
            path.append((u, e))
            u = pu
        path.reverse()
        self.path = path                       # [(room, edge used to arrive)]
        pts = [(rooms[self.start_room].cx, rooms[self.start_room].cy)]
        cur = self.start_room
        self.door_points = []
        for nxt, e in path:
            (ma, na), (mb, nb) = self.mouth(e, cur), self.mouth(e, nxt)
            pts += [(ma[0] + na[0] * 64, ma[1] + na[1] * 64), ma, mb, (mb[0] + nb[0] * 64, mb[1] + nb[1] * 64),
                    (rooms[nxt].cx, rooms[nxt].cy)]
            if e.kind == "door":
                self.door_points.append(((ma[0] + mb[0]) // 2, (ma[1] + mb[1]) // 2))
            cur = nxt
        a = self.alcove
        px, py = a["approach"]
        nx, ny = a["normal"]
        pts += [(px + nx * 64, py + ny * 64), (px, py)]
        self.route = pts

    def _alcove_spot(self, r: Room):
        rng = self.rng
        cands = []
        for side in ("w", "e", "s", "n"):
            lo, hi = ((r.y0, r.y1) if side in "we" else (r.x0, r.x1))
            for p in range(lo + 32, hi - 32 - 64 + 1, 16):
                if all(p + 64 + 24 <= a0 or p - 24 >= a1 for a0, a1 in r.attach[side]):
                    cands.append((side, p))
        if not cands:
            return None
        side, p = rng.choice(cands)
        mid = p + 32
        if side == "w":
            rect, appr, normal = (r.x0 - 16, p, r.x0, p + 64), (r.x0 + 40, mid), (1, 0)
        elif side == "e":
            rect, appr, normal = (r.x1, p, r.x1 + 16, p + 64), (r.x1 - 40, mid), (-1, 0)
        elif side == "s":
            rect, appr, normal = (p, r.y0 - 16, p + 64, r.y0), (mid, r.y0 + 40), (0, 1)
        else:
            rect, appr, normal = (p, r.y1, p + 64, r.y1 + 16), (mid, r.y1 - 40), (0, -1)
        return dict(room=r.idx, side=side, rect=rect, approach=appr, normal=normal)

    # ---- room interiors: pillars, daises, pits, light panels
    def make_holes(self):
        rng, T = self.rng, self.T
        for r in self.rooms:
            w, h = r.x1 - r.x0, r.y1 - r.y0
            if r.idx == self.start_room and rng.random() < 0.5:
                budget = 1
            else:
                budget = rng.choice([0, 1, 1, 2, 2, 3, 4]) if w * h >= 256 * 256 else rng.choice([0, 0, 1])
            tries = 0
            while len(r.holes) < budget and tries < 40:
                tries += 1
                kind = rng.choice(["pillar", "pillar", "dais", "pit", "lamp", "lamp"])
                if kind == "lamp" and not T["lamps"] and r.kind != "outdoor":
                    kind = "pillar"
                if kind == "lamp" and r.kind == "outdoor":
                    continue
                if kind == "pit" and (r.kind != "dark" or rng.random() < 0.3) and rng.random() < 0.6:
                    kind = "dais"
                sz = {"pillar": (32, 64), "dais": (96, 192), "pit": (96, 192), "lamp": (64, 128)}[kind]
                hw, hh = rng.randrange(sz[0], sz[1] + 1, 16), rng.randrange(sz[0], sz[1] + 1, 16)
                if kind == "pillar":
                    hh = hw if rng.random() < 0.6 else hh
                if w < hw + 128 or h < hh + 128:
                    continue
                x0 = rng.randrange(r.x0 + 48, r.x1 - 48 - hw + 1, 16)
                y0 = rng.randrange(r.y0 + 48, r.y1 - 48 - hh + 1, 16)
                rect = (x0, y0, x0 + hw, y0 + hh)
                if any(contact((rect[0] - 40, rect[1] - 40, rect[2] + 40, rect[3] + 40), o["rect"]) for o in r.holes):
                    continue
                if kind in ("pillar", "pit") and rect_polyline_dist(rect, self.route) < 56:
                    continue
                if rect_polyline_dist(rect, self.route) < 24:
                    continue
                r.holes.append(dict(kind=kind, rect=rect))
            # corner pillars in big rooms (symmetry reads as architecture)
            if r.kind != "dark" and w >= 384 and h >= 384 and rng.random() < 0.35 and not r.holes:
                for sx in (r.x0 + 56, r.x1 - 56 - 32):
                    for sy in (r.y0 + 56, r.y1 - 56 - 32):
                        rect = (sx, sy, sx + 32, sy + 32)
                        if rect_polyline_dist(rect, self.route) >= 56:
                            r.holes.append(dict(kind="pillar", rect=rect))

    # ---- build rects
    def new_tag(self):
        self.tag += 1
        return self.tag

    def add(self, area, role, box, **kw):
        i = self.b.rect(*box, **kw)
        self.owner[i] = area
        self.rect_info[i] = role
        return i

    def build_rooms(self):
        rng, T = self.rng, self.T
        for r in self.rooms:
            area = f"R{r.idx}"
            floor, ceil = r.floor, r.floor + r.height
            special = 0
            if r.kind == "dark" and rng.random() < 0.5:
                special = rng.choice([SPEC_FLICKER, SPEC_FLICKER, SPEC_FIRE, SPEC_STROBE_SLOW])
            elif r.kind == "bright" and rng.random() < 0.2:
                special = SPEC_GLOW
            elif self.theme_name == "hell" and r.kind != "outdoor" and rng.random() < 0.2:
                special = SPEC_GLOW
            r.special = special
            common = dict(floor=floor, ceil=ceil, ftex=r.flat_tex, ctex=r.ceil_tex, wall=r.wall_tex)
            holes = [h["rect"] for h in r.holes]
            for box in carve(r.box, holes):
                r.pieces.append(self.add(area, "room", box, light=r.light, special=special, **common))
            for h in r.holes:
                k, box = h["kind"], h["rect"]
                if k == "pillar":
                    top = ceil if r.kind != "outdoor" else floor + 128
                    i = self.add(area, "pillar", box, floor=top, ceil=top, light=r.light, ftex=r.flat_tex,
                                 ctex=r.ceil_tex, wall=r.wall_tex, facing=(None, rng.choice(T["accent"] + [r.wall_tex])))
                elif k == "dais":
                    i = self.add(area, "dais", box, floor=floor + 16, ceil=ceil, light=min(255, r.light + 24),
                                 special=special, ftex=rng.choice(T["outdoor_floor"] if r.kind == "outdoor" else
                                                                    [s[1] for s in T["styles"]]),
                                 ctex=r.ceil_tex, wall=r.wall_tex, facing=(None, T["step"]))
                elif k == "pit":
                    ftex, hspec = rng.choice(T["hazard"])
                    i = self.add(area, "pit", box, floor=floor - 16, ceil=ceil, light=min(255, r.light + 40),
                                 special=hspec, ftex=ftex, ctex=r.ceil_tex, wall=r.wall_tex)
                else:                                                   # lamp: light panel in the ceiling
                    lamp = T["lamp"] or r.ceil_tex
                    sp = SPEC_FLICKER if rng.random() < 0.12 else 0
                    i = self.add(area, "lamp", box, floor=floor, ceil=ceil, light=min(255, r.light + 80), special=sp,
                                 ftex=r.flat_tex, ctex=lamp, wall=r.wall_tex)
                r.rects.append(i)
            r.rects += r.pieces

    def build_corridors(self):
        rng, T = self.rng, self.T
        for e in self.edges:
            lo, hi = self.rooms[e.lo], self.rooms[e.hi]
            area = f"C{e.idx}"
            style = rng.choice(T["corridor"])
            wall, ftex, ctex = style
            light = rng.choice([160, 168, 176, 188])
            ch = rng.choice([96, 128, 128])
            fl, fh = lo.floor, hi.floor
            if lo.kind == "outdoor" and hi.kind == "outdoor" and e.kind in ("flat", "stairs"):
                wall, ftex, ctex = hi.wall_tex, hi.flat_tex, "F_SKY1"
                light, ch = hi.light, 192
            cc = max(fl, fh) + ch
            d = fh - fl

            def piece(p0, p1, floor, ceil, role, **kw):
                if e.axis == "x":
                    box = (p0, e.c0, p1, e.c1)
                else:
                    box = (e.c0, p0, e.c1, p1)
                kw.setdefault("light", light)
                kw.setdefault("ftex", ftex)
                kw.setdefault("ctex", ctex)
                kw.setdefault("wall", wall)
                i = self.add(area, role, box, floor=floor, ceil=ceil, **kw)
                e.rects.append(i)
                return i

            L = e.length
            if e.kind == "flat":
                piece(e.p0, e.p1, fl, cc, "corridor")
            elif e.kind == "stairs":
                n, s, depth = stairs_plan(L, d)
                lead = (L - n * depth) // 2 // 8 * 8
                tail = L - n * depth - lead
                sg = 1 if d > 0 else -1
                face = (None, T["step"])
                x = e.p0
                if lead:
                    piece(x, x + lead, fl, cc, "corridor", facing=face)
                x += lead
                for k in range(n):
                    piece(x, x + depth, fl + sg * s * (k + 1), cc, "step", facing=face,
                          light=light + (8 if k % 2 else 0))
                    x += depth
                if tail:
                    piece(x, x + tail, fh, cc, "corridor", facing=face)
            elif e.kind == "lift":
                low_is_lo = fl < fh
                lift_len = 64
                top = max(fl, fh)
                low = min(fl, fh)
                if low_is_lo:
                    pl = piece(e.p0, e.p0 + lift_len, low, top + 96, "lift", wall=T["lift"], facing=(None, T["lift"]),
                               light=light + 16)
                    ph = piece(e.p0 + lift_len, e.p1, top, cc, "corridor", facing=(None, T["lift"]))
                    room_side = lo
                else:
                    ph = piece(e.p0, e.p1 - lift_len, top, cc, "corridor", facing=(None, T["lift"]))
                    pl = piece(e.p1 - lift_len, e.p1, low, top + 96, "lift", wall=T["lift"], facing=(None, T["lift"]),
                               light=light + 16)
                    room_side = hi
                tag = self.new_tag()
                self.b.rects[pl].tag = tag
                self.lifts.append(dict(edge=e.idx, rect=pl, tag=tag, low=low, top=top))
                e.lift = (pl, ph, room_side.idx)
            elif e.kind == "door":
                tex, dw, dh = e.door
                cdh = max(ch, dh)
                cc2 = max(fl, fh) + cdh
                dlen, flank = 16, 16
                mid_len = L - dlen - 2 * flank
                a_len = max(16, (mid_len // 2) // 8 * 8)
                x = e.p0
                piece(x, x + a_len, fl, cc2, "corridor")
                x += a_len
                pre = piece(x, x + flank, fl, fl + dh, "flank", light=light - 16)
                x += flank
                tag = self.new_tag()
                dr = piece(x, x + dlen, fl, fl, "door", light=light - 24, tag=tag, wall=T["doortrak"], facing=tex)
                x += dlen
                post = piece(x, x + flank, fl, fl + dh, "flank", light=light - 16)
                x += flank
                piece(x, e.p1, fl, cc2, "corridor")
                self.doors.append(dict(edge=e.idx, rect=dr, tag=tag, tex=tex, rects=(pre, dr, post)))
        # the alcove with the exit switch
        a = self.alcove
        room = self.rooms[a["room"]]
        i = self.add("X", "alcove", a["rect"], floor=room.floor, ceil=room.floor + 128, light=min(255, room.light + 48),
                     ftex=room.flat_tex, ctex=room.ceil_tex if room.kind != "outdoor" else self.T["styles"][0][2],
                     wall=room.wall_tex)
        a["index"] = i
        far = {"w": "w", "e": "e", "s": "s", "n": "n"}[a["side"]]
        self.b.edge(i, far, line=dict(special=243, playeruse=True, repeatspecial=False),
                    sidedef=dict(texturemiddle=self.T["switch"]))

    def wire_specials(self):
        b = self.b
        for d in self.doors:
            pre, dr, post = d["rects"]
            props = dict(special=12, arg0=d["tag"], arg1=64, arg2=105, playeruse=True, playeruseback=True,
                         repeatspecial=True)
            b.link(dr, pre, **props)
            b.link(dr, post, **props)
        for lf in self.lifts:
            e = self.edges[lf["edge"]]
            pl, ph, room_idx = e.lift
            props = dict(special=64, arg0=lf["tag"], arg1=48, arg2=105, playercross=True, repeatspecial=True)
            for ri in self.rooms[room_idx].pieces:
                if contact(self._box(ri), self._box(pl)) == "edge":
                    b.link(ri, pl, **props)

    def _box(self, i):
        r = self.b.rects[i]
        return (r.x0, r.y0, r.x1, r.y1)

    def validate(self):
        boxes = [self._box(i) for i in range(len(self.b.rects))]
        allowed = {("X", f"R{self.alcove['room']}")}
        for e in self.edges:
            allowed |= {(f"R{e.lo}", f"C{e.idx}"), (f"R{e.hi}", f"C{e.idx}")}
        allowed |= {(b_, a_) for a_, b_ in allowed}
        n = len(boxes)
        for i in range(n):
            for j in range(i + 1, n):
                c = contact(boxes[i], boxes[j])
                if not c:
                    continue
                oi, oj = self.owner[i], self.owner[j]
                if c == "overlap":
                    raise _Retry(f"overlap {oi} {oj}")
                if oi != oj and (oi, oj) not in allowed:
                    raise _Retry(f"unintended contact {oi} {oj} {boxes[i]} {boxes[j]}")

    # ---- things
    def place_things(self):
        rng, T, b = self.rng, self.T, self.b
        route = self.route
        taken: list[tuple[float, float]] = []
        rooms = self.rooms
        maxdepth = max(1, max(r.depth for r in rooms))

        def spot(room: Room, margin=36, avoid_route=96, spacing=56, tries=40):
            cands = [i for i in room.pieces] + [i for i in room.rects if self.rect_info[i] in ("dais", "lamp")]
            for _ in range(tries):
                i = rng.choice(cands)
                rc = b.rects[i]
                if rc.x1 - rc.x0 < 2 * margin + 8 or rc.y1 - rc.y0 < 2 * margin + 8:
                    continue
                x = rng.uniform(rc.x0 + margin, rc.x1 - margin)
                y = rng.uniform(rc.y0 + margin, rc.y1 - margin)
                if polyline_dist(route, x, y) < avoid_route:
                    continue
                if any(math.hypot(x - tx, y - ty) < spacing for tx, ty in taken):
                    continue
                taken.append((x, y))
                return snap(x, 4), snap(y, 4)
            return None

        def face(x, y, room):
            return rng.choice(range(0, 360, 45))

        start = rooms[self.start_room]
        sx, sy = route[0]
        ang = math.degrees(math.atan2(route[1][1] - sy, route[1][0] - sx)) % 360
        b.thing(sx, sy, 1, int(round(ang)))
        for k, (dx, dy) in enumerate([(40, 0), (-40, 0), (0, 40)], start=2):
            if all(rect_point_dist(h["rect"], sx + dx, sy + dy) > 24 for h in start.holes):
                b.thing(sx + dx, sy + dy, k, int(round(ang)))
        taken.append((sx, sy))

        # monsters
        for r in rooms:
            if r.idx == self.start_room:
                continue
            area = (r.x1 - r.x0) * (r.y1 - r.y0)
            t = r.depth / maxdepth
            n = int(area / 70000) + rng.choice([0, 1, 1, 2])
            if r.kind == "dark":
                n += 1
            if r.idx == self.exit_room:
                n += 2
            n = min(n, 6)
            for _ in range(n):
                p = spot(r)
                if not p:
                    continue
                if r.idx == self.exit_room and rng.random() < 0.6:
                    mt = weighted(rng, T["guard"])
                else:
                    tier = min(3, int(t * 3.4 + rng.uniform(-0.4, 0.5)))
                    mt = weighted(rng, T["tiers"][max(0, tier)])
                kw = dict(ambush=True) if rng.random() < 0.45 else {}
                b.thing(p[0], p[1], mt, face(*p, r), **kw)
                self.counts["monsters"] += 1
        # weapons: shotgun early, chaingun mid, rocket launcher late, plasma at the end
        by_depth = sorted((r for r in rooms if r.idx != self.start_room), key=lambda r: r.depth)
        wp_plan = [(2001, 0.1), (2002, 0.45), (2003, 0.7), (2004, 0.95)]
        for typ, frac in wp_plan:
            r = by_depth[min(len(by_depth) - 1, int(frac * len(by_depth)))]
            p = spot(r, margin=32, avoid_route=24)
            if p:
                b.thing(p[0], p[1], typ, 0)
                self.counts["weapons"].append(WEAPONS[typ])
        # items
        for r in rooms:
            if r.idx == self.start_room:
                if rng.random() < 0.8:
                    p = spot(r, avoid_route=24)
                    if p:
                        b.thing(p[0], p[1], 2011, 0)
                        self.counts["items"] += 1
                continue
            t = r.depth / maxdepth
            n = rng.choice([1, 1, 2, 3])
            for _ in range(n):
                p = spot(r, margin=24, avoid_route=16, spacing=40)
                if not p:
                    continue
                roll = rng.random()
                if roll < 0.28:
                    typ = 2011 if t < 0.6 else 2012
                elif roll < 0.5:
                    typ = rng.choice([2007, 2008, 2007, 2010 if t > 0.5 else 2008, 2047 if t > 0.8 else 2007])
                elif roll < 0.68:
                    typ = rng.choice([2014, 2015])
                elif roll < 0.8:
                    typ = 2018 if t < 0.7 else 2019
                elif roll < 0.9:
                    typ = 2012
                else:
                    typ = 8 if t > 0.4 else 2007
                if typ in (2014, 2015):                       # bonuses come in little groups
                    for k in range(rng.randint(2, 4)):
                        b.thing(p[0] + 16 * (k % 2) * 2 - 8, p[1] + 16 * (k // 2) * 2 - 8, typ, 0)
                        self.counts["items"] += 1
                else:
                    b.thing(p[0], p[1], typ, 0)
                    self.counts["items"] += 1
        # decorations
        for r in rooms:
            n = rng.choice([1, 2, 3, 4]) if r.kind != "outdoor" else rng.choice([2, 3, 4, 5])
            for _ in range(n):
                typ = rng.choice(T["deco"])
                if T["hang"] and rng.random() < 0.35 and r.kind != "outdoor":
                    typ = rng.choice(T["hang"])
                p = spot(r, margin=44, avoid_route=96, spacing=48)
                if p:
                    b.thing(p[0], p[1], typ, 0)
                    self.counts["decor"] += 1
        # light for the player's first steps: a few torches in corridors would be lost on `taken`; skip

    # ---- the whole thing
    def generate(self):
        self.layout()
        self.decorate_rooms()
        self.plan_route()
        self.make_holes()
        self.build_rooms()
        self.build_corridors()
        self.wire_specials()
        self.validate()
        self.place_things()


def mapinfo_text(name: str, theme: dict) -> str:
    safe = name.replace('"', "'")
    return (f'map MAP01 "{safe}"\n{{\n\tsky1 = "{theme["sky"]}"\n\tnext = "EndGameC"\n}}\n')


def write_wad(path: Path, lumps: list[tuple[str, bytes]]):
    data, directory, ofs = b"", b"", 12
    for name, blob in lumps:
        directory += struct.pack("<ii8s", ofs, len(blob), name.encode())
        data += blob
        ofs += len(blob)
    Path(path).write_bytes(struct.pack("<4sii", b"PWAD", len(lumps), 12 + len(data)) + data + directory)


def build_level(name: str, path: Path, seed: int | None = None, theme: str | None = None) -> dict:
    """Generate the level for `name` into `path` (a PWAD for freedoom2.wad, map MAP01). Deterministic in (name, seed)."""
    name = " ".join(name.upper().split())
    seed = name_seed(name) if seed is None else seed
    last = None
    for attempt in range(60):
        g = Generator(name, seed, attempt, theme)
        try:
            g.generate()
            break
        except _Retry as exc:
            last = exc
    else:
        raise RuntimeError(f"level generation failed for {name!r}: {last}")
    textmap = g.b.textmap()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    write_wad(path, [("MAPINFO", mapinfo_text(name, g.T).encode()), ("MAP01", b""), ("TEXTMAP", textmap.encode()),
                     ("ENDMAP", b"")])
    kinds = {}
    for e in g.edges:
        kinds[e.kind] = kinds.get(e.kind, 0) + 1
    xs = [r.x0 for r in g.rooms] + [r.x1 for r in g.rooms]
    ys = [r.y0 for r in g.rooms] + [r.y1 for r in g.rooms]
    return dict(
        name=name, seed=seed, attempt=attempt, theme=g.theme_name, path=(p.relative_to(R).as_posix() if (p := Path(path).resolve()).is_relative_to(R := Path(__file__).resolve().parents[1]) else str(path)), wad_bytes=path.stat().st_size,
        rooms=len(g.rooms), corridors=len(g.edges), corridor_kinds=kinds, sectors=len(g.b.rects),
        monsters=g.counts["monsters"], items=g.counts["items"], decor=g.counts["decor"], weapons=g.counts["weapons"],
        outdoor_rooms=sum(r.kind == "outdoor" for r in g.rooms), sky=g.T["sky"],
        flicker_sectors=sum(1 for r in g.b.rects if r.special in (SPEC_FLICKER, SPEC_FIRE, SPEC_STROBE_SLOW, SPEC_GLOW)),
        doors=[dict(tex=d["tex"], axis=g.edges[d["edge"]].axis, edge=d["edge"]) for d in g.doors],
        door_points=g.door_points, lifts=len(g.lifts),
        start=dict(room=g.start_room, x=g.route[0][0], y=g.route[0][1]), exit_room=g.exit_room,
        exit_switch=dict(approach=g.alcove["approach"], normal=g.alcove["normal"]),
        route=g.route, bounds=(min(xs), min(ys), max(xs), max(ys)),
        floor_range=(min(r.floor for r in g.rooms), max(r.floor for r in g.rooms)),
    )
