"""Generic DOOM compute tile.

A tile has input *ports* and *outputs*. A port is a set of thin gate sectors sharing a tag; the bus
moves their floor ("floor" ports) or ceiling ("ceil" ports) to an integer verbatim. An output is an
accumulator sector whose light level starts at a bias and is changed by line specials.

Gates (8 units thin, voodoo dolls are 32 wide and 56 tall):
  floor gate: passes iff value <= limit. The doll walks on floor limit-24 and can step onto the gate
              only if the gate floor is at most 24 higher. Open gates are bridged, so no pits.
  ceil gate : passes iff value >= limit. The doll walks on floor limit-56 and fits under the gate
              only if the gate ceiling leaves 56 units.
A doll never falls inside a corridor: ceiling-gate thresholds rise along it, so the floor climbs
24-unit stairs (instant step-ups) instead of dropping (airborne dolls would mis-read gates).

A tile is a list of *blocks*:
  lanes  : one doll per lane, each behind its own ADC gate (thermometer code of a port value);
           a passing doll is teleported into the block's shared zone
  stages : the shared zone; each stage = optional gate followed by effect lines
           (Light_RaiseByValue / Light_LowerByValue on output accumulators).
Crossbar block   = lanes + one ungated stage           -> adds count * w
Multiplier block = lanes + a corridor of ceiling gates  -> adds count_a * count_b * w
Equality block   = lane gate (v <= t) + stage gate (-v <= -t)
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Literal

from .udmf import MapBuilder, compile_acs, write_wad

LANE_W = 40          # doll diameter is 32
STEP_UP = 24         # Doom step-up limit
DOLL_H = 56          # player height
GATE_LEN = 8         # thinner than a doll -> open floor gates are bridged
FX_LEN = 4           # spacing of effect lines
STAIR_LEN = 36       # longer than a doll, so it never straddles two risers
POST_LEN = 40        # flat section after a gate: doll diameter 32 + margin
# Heights are 16.16 fixed point: every floor/ceiling difference the engine computes must stay < 32768.
# Port values and gate limits live in [-PORT_MAX, PORT_MAX]; closed gates and ceilings sit just above.
PORT_MAX = 12000
CLOSED = 13000       # floor-gate floor before the bus writes the port
CEIL = 16000
CONV_TAG = 1
START_TAG = 2       # per-lane start barriers: closed at reset, dropped to the walking floor once gates settled
ZONE_X = 224        # x offset of a block's shared zone from its lane area
LANE_LEN = 152      # home | barrier | pad | ADC gate | post | dead end
PORT_TAG0 = 1000
OUT_TAG0 = 20000
PAGE = 60            # ViZDoom USER1..USER60
CONVEYOR = dict(xmove=255, ymove=128, stack=4)
DOLL_TID0 = 10000    # dolls tag themselves DOLL_TID0 + lane on their first step
HOME_TID0 = 30000    # teleport spot on each lane's pad, used by the reset script
DOOR_TAG0 = 50000    # per-block door at the zone entrance (open; lower it to ablate the block)
Y_COLUMN = 28000     # wrap blocks into a new column beyond this height
# Mover speed (1/8 unit per tic): 150000 = 18,750 units per tic, so any port move finishes in 1 tic from heights within
# +-12,000. The engine adds speed to the plane height in 16.16 fixed point; a move that starts above 32,768 - 18,750 =
# 14,018 (or below -14,018) keeps its mover busy for 3 tics, so the compiler keeps every plane inside that bound (planes.py).
MOVE_SPEED = 150000
MOTION_SCRIPT = 8   # ACS (motion_acs): counts the dolls and the player that moved since its previous run
GUARD_SCRIPT = 9    # ACS (guard_acs): fills every USER slot with its argument; TileRunner.read runs it before each page
GUARD_NONCE = 100_003     # poison outside the light/height range; touches USER slots, never model state
MAX_SPAN = 32000    # map extent limit per axis for every map (brain, tile, world): docs/engine_limits.md
Kind = Literal["floor", "ceil"]


def motion_acs(n_dolls: int) -> str:
    """ACS script MOTION_SCRIPT: compares every doll (TIDs DOLL_TID0 .. DOLL_TID0 + n_dolls - 1, which the dolls give
    themselves on their first step) and the activator (the player, when puked) with its position at the script's
    previous run; writes the number that moved to o1 (USER1) and the number of dolls without their TID to o2 (USER2).
    runtime.TileRunner.settle uses it where ViZDoom's object list cannot show every doll."""
    n = n_dolls + 1
    return f"""int mo_x[{n}];
int mo_y[{n}];
int mo_z[{n}];
script {MOTION_SCRIPT} (void) {{
  int moved = 0;
  int untagged = 0;
  for (int i = 0; i < {n}; i++) {{
    int tid = 0;
    if (i < {n_dolls}) {{
      tid = {DOLL_TID0} + i;
      if (!IsTIDUsed(tid)) {{ untagged++; continue; }}
    }}
    int x = GetActorX(tid);
    int y = GetActorY(tid);
    int z = GetActorZ(tid);
    if (x != mo_x[i] || y != mo_y[i] || z != mo_z[i]) {{ moved++; mo_x[i] = x; mo_y[i] = y; mo_z[i] = z; }}
  }}
  o1 = moved;
  o2 = untagged;
}}"""


def guard_acs() -> str:
    """ACS script GUARD_SCRIPT(nonce): writes the nonce to o1..o60 (USER1..60). On a guarded map TileRunner.read runs it
    with GUARD_NONCE before each readout page and checks that the page replaced the poison."""
    fill = " ".join(f"o{i} = nonce;" for i in range(1, PAGE + 1))
    return f"script {GUARD_SCRIPT} (int nonce) {{ {fill} }}"


@dataclass(frozen=True)
class Gate:
    port: int
    limit: int
    kind: Kind = "floor"          # floor: passes iff value <= limit ; ceil: passes iff value >= limit

    @property
    def walk_floor(self) -> int:
        return self.limit - STEP_UP if self.kind == "floor" else self.limit - DOLL_H


@dataclass(frozen=True)
class Effect:
    out: int
    w: int


@dataclass
class Stage:
    gate: Gate | None
    effects: list[Effect]


@dataclass
class Block:
    lanes: list[Gate]
    stages: list[Stage]


@dataclass
class TileSpec:
    name: str
    ports: list[str] = field(default_factory=list)            # "floor:<value>" / "ceil:<value>"
    outputs: list[tuple[str, int]] = field(default_factory=list)   # (name, bias)
    blocks: list[Block] = field(default_factory=list)

    def __post_init__(self):
        self._port_idx = {n: i for i, n in enumerate(self.ports)}

    def port(self, value: str, kind: Kind) -> int:
        key = f"{kind}:{value}"
        if key not in self._port_idx:
            self._port_idx[key] = len(self.ports)
            self.ports.append(key)
        return self._port_idx[key]

    def output(self, name: str, bias: int = 0) -> int:
        self.outputs.append((name, int(bias)))
        return len(self.outputs) - 1

    @property
    def n_dolls(self) -> int:
        return sum(len(b.lanes) for b in self.blocks)


@dataclass
class TileMeta:
    name: str
    ports: dict[str, int]          # "kind:value" -> sector tag
    outputs: dict[str, int]        # output name -> sector tag
    pages: list[list[str]]         # readout pages: output names in USER1.. order
    n_dolls: int
    n_sectors: int
    layout: dict = field(default_factory=dict)   # geometry for cameras/switches (see compile_tile)

    def save(self, path: Path):
        Path(path).write_text(json.dumps(asdict(self)))

    @classmethod
    def load(cls, path: Path) -> "TileMeta":
        return cls(**json.loads(Path(path).read_text()))

    @property
    def read_tics(self) -> int:
        """Game tics one TileRunner.read takes: a tic per page, plus a guard tic before each page when guarded."""
        return len(self.pages) * (2 if "guard_script" in self.layout else 1)


def _fx_line(e: Effect) -> dict:
    return dict(special=110 if e.w > 0 else 111, arg0=OUT_TAG0 + e.out, arg1=abs(e.w),
                playercross=True, repeatspecial=True)


def _gate_rect(m: MapBuilder, g: Gate, x0, y0, x1, y1, group) -> int:
    floor = CLOSED if g.kind == "floor" else g.walk_floor
    return m.rect(x0, y0, x1, y1, floor=floor, ceil=CEIL if g.kind == "floor" else g.walk_floor,
                  tag=PORT_TAG0 + g.port, group=group, light=224, ftex="FLAT23", wall="METAL")


def _live_stages(block: Block) -> list[Stage]:
    return [s for s in block.stages if s.gate is not None or any(e.w for e in s.effects)]


def _zone_floor(stages: list[Stage]) -> int:
    """Walking floor at a zone's entry: the first gate's (an ungated first stage runs on it too), else 0."""
    return next((st.gate.walk_floor for st in stages if st.gate is not None), 0)


def _zone_len(stages: list[Stage]) -> int:
    """Length of a block's shared zone; mirrors the geometry emitted in compile_tile."""
    floor = _zone_floor(stages)
    x = 48 + GATE_LEN                                   # entry + door
    for st in stages:
        if st.gate is not None:
            while st.gate.walk_floor - floor > STEP_UP:
                floor += STEP_UP
                x += STAIR_LEN
            floor = st.gate.walk_floor
            x += STAIR_LEN + GATE_LEN + POST_LEN
        x += FX_LEN * sum(1 for e in st.effects if e.w)
    return x + 64


def compile_tile(spec: TileSpec, wad: Path, decorate=None, *, layout_only: bool = False) -> TileMeta:
    """Compile `spec` into `wad`. `decorate(builder, layout) -> str` may add geometry/things to the map
    and return extra ACS source; `layout` lists every block's zone, lane area and door tag, the output
    sectors and the player start. Every block has an open door (ceiling) at its zone entrance, tag
    DOOR_TAG0 + block index; lowering it stops that block's dolls (used for ablation).
    layout_only counts the same emitted geometry without compiling ACS or writing files (then layout["fits"] says
    whether the map would be built). A map (decorations included) wider or taller than MAX_SPAN raises ValueError."""
    limits = [g.limit for b in spec.blocks for g in b.lanes] + \
             [s.gate.limit for b in spec.blocks for s in b.stages if s.gate is not None]
    if limits and max(abs(v) for v in limits) > PORT_MAX:
        raise ValueError(f"{spec.name}: gate limit {max(limits, key=abs)} outside +-{PORT_MAX}")
    blocks = [(bi, b, _live_stages(b)) for bi, b in enumerate(spec.blocks)]
    blocks = [(bi, b, st) for bi, b, st in blocks if b.lanes and st]
    layout: dict = dict(blocks=[], outputs=[], player=[-256, -128], motion_script=MOTION_SCRIPT)
    # wrap blocks into columns so the whole tile is roughly 4:3 (it is shown on an automap)
    col_w = ZONE_X + max((_zone_len(st) for _, _, st in blocks), default=0) + 512
    total_h = sum(len(b.lanes) * LANE_W for _, b, _ in blocks)
    max_h = max((len(b.lanes) * LANE_W for _, b, _ in blocks), default=0)
    y_col = min(Y_COLUMN, max(max_h, int((total_h * col_w * 0.75) ** 0.5)))
    n_out_rows = (len(spec.outputs) + 63) // 64
    if layout_only:
        from .pure import _GeometryCounter
        m = _GeometryCounter(closet_origin=(0, -256 - n_out_rows * 72 - 160))
    else:
        m = MapBuilder(closet_origin=(0, -256 - n_out_rows * 72 - 160))
    group, tid = 0, 1
    homes: list[int] = []
    x_col, y, col_len = 0, 0, 0
    for bi, block, stages in blocks:
        n = len(block.lanes)
        h = n * LANE_W
        if y + h > y_col:
            x_col += col_len + 512
            y, col_len = 0, 0
        # ---- lanes: home | registration line | home | start barrier | pad | ADC gate | post | teleport | dead end
        # A doll pressed against a closed gate that then opens to a small step-up (1..24 units) never climbs
        # it, so dolls must never wait at an ADC gate: they wait at the start barrier, which is dropped to
        # exactly their walking floor (step 0) only after every port has been written and has settled.
        for li, g in enumerate(block.lanes):
            group += 1
            lane = len(homes)
            ly = y + li * LANE_W
            cy = ly + LANE_W // 2
            f = g.walk_floor
            kw = dict(ceil=CEIL, group=group, tag=CONV_TAG)
            home = m.rect(x_col, ly, x_col + 24, ly + LANE_W, floor=f, light=192, ftex="FLOOR4_8", **kw)
            home2 = m.rect(x_col + 24, ly, x_col + 48, ly + LANE_W, floor=f, light=192, ftex="FLOOR4_8", **kw)
            # the doll tags itself (Thing_ChangeTID on the activator) so the reset script can find it
            m.link(home, home2, special=226, arg0=4, arg1=0, arg2=DOLL_TID0 + lane, playercross=True,
                   repeatspecial=True)
            m.rect(x_col + 48, ly, x_col + 56, ly + LANE_W, floor=CLOSED, ceil=CEIL, group=group, tag=START_TAG,
                   light=224, ftex="FLAT23", wall="SUPPORT3")
            m.rect(x_col + 56, ly, x_col + 96, ly + LANE_W, floor=f, light=192, ftex="FLOOR4_8", **kw)
            _gate_rect(m, g, x_col + 96, ly, x_col + 96 + GATE_LEN, ly + LANE_W, group)
            post = m.rect(x_col + 104, ly, x_col + 128, ly + LANE_W, floor=f, light=192, **kw)
            dead = m.rect(x_col + 128, ly, x_col + LANE_LEN, ly + LANE_W, floor=f, light=128, **kw)
            m.link(post, dead, special=71, arg0=tid, playercross=True, repeatspecial=True)   # Teleport_NoFog
            m.thing(x_col + 16, cy, 1)                                      # voodoo doll
            m.thing(x_col + 16, cy, 14, id=HOME_TID0 + lane)                 # where the reset sends it
            m.thing(x_col + ZONE_X + 24, cy, 14, id=tid)                     # zone entry
            homes.append(lane)
            tid += 1
        # ---- shared zone: entry | per stage: [stairs] pre | gate | effect lines | runout
        group += 1
        x = x_col + ZONE_X
        floor = _zone_floor(stages)
        kw = dict(ceil=CEIL, group=group, tag=CONV_TAG)
        prev = m.rect(x, y, x + 48, y + h, floor=floor, light=176, **kw)
        x += 48
        door_tag = DOOR_TAG0 + bi
        prev = m.rect(x, y, x + GATE_LEN, y + h, floor=floor, ceil=CEIL, group=group, tag=door_tag, light=176,
                      ftex="FLAT23", wall="DOOR3")
        door_x = [x, x + GATE_LEN]
        x += GATE_LEN
        for st in stages:
            if st.gate is not None:
                target = st.gate.walk_floor
                if target < floor:
                    raise ValueError(f"{spec.name}: corridor would descend {floor}->{target}")
                while target - floor > STEP_UP:
                    floor += STEP_UP
                    prev = m.rect(x, y, x + STAIR_LEN, y + h, floor=floor, light=176, **kw)
                    x += STAIR_LEN
                floor = target
                m.rect(x, y, x + STAIR_LEN, y + h, floor=floor, light=176, **kw)
                prev = _gate_rect(m, st.gate, x + STAIR_LEN, y, x + STAIR_LEN + GATE_LEN, y + h, group)
                x += STAIR_LEN + GATE_LEN
            for e in st.effects:
                if not e.w:
                    continue
                cur = m.rect(x, y, x + FX_LEN, y + h, floor=floor, light=200, ftex="FLAT1_1", **kw)
                m.link(prev, cur, **_fx_line(e))
                prev = cur
                x += FX_LEN
            if st.gate is not None:
                # flat run-off: the doll must fully clear the gate before the next riser, otherwise
                # its box straddles gate and riser and the gate is judged against the higher floor
                prev = m.rect(x, y, x + POST_LEN, y + h, floor=floor, light=176, **kw)
                x += POST_LEN
        m.rect(x, y, x + 64, y + h, floor=floor, light=160, **kw)            # runout
        col_len = max(col_len, x + 64 - x_col)
        layout["blocks"].append(dict(index=bi, lanes=[x_col, y, x_col + LANE_LEN, y + h],
                                     zone=[x_col + ZONE_X, y, x + 64, y + h], door_tag=door_tag, door_x=door_x,
                                     n_lanes=n, entry_floor=_zone_floor(stages),
                                     lane_port=spec.ports[block.lanes[0].port]))
        y += h
    # ---- output accumulators (display sectors below the machine) and the real player
    for oi, (name, bias) in enumerate(spec.outputs):
        ox, oy = (oi % 64) * 72, -256 - (oi // 64) * 72
        m.rect(ox, oy, ox + 64, oy + 64, floor=0, ceil=128, light=bias, tag=OUT_TAG0 + oi, ftex="FLAT14",
               wall="COMPBLUE")
        layout["outputs"].append([name, ox, oy, ox + 64, oy + 64])
    m.rect(-384, -256, -128, 0, floor=0, ceil=128)
    m.thing(-256, -128, 1)                                                  # last player-1 start = player
    m.scroll_carry(CONV_TAG, **CONVEYOR)
    extra_acs = decorate(m, layout) if decorate else ""
    bounds = [min(r.x0 for r in m.rects), min(r.y0 for r in m.rects),
              max(r.x1 for r in m.rects), max(r.y1 for r in m.rects)]
    span = [bounds[2] - bounds[0], bounds[3] - bounds[1]]
    layout.update(bounds=bounds, span=span, max_span=MAX_SPAN, fits=max(span) <= MAX_SPAN)
    if not layout["fits"] and not layout_only:
        raise ValueError(f"{spec.name}: the tile map spans {span[0]} x {span[1]} units (bounds {bounds}), beyond the "
                         f"{MAX_SPAN}-unit limit per axis where ViZDoom misplaces things (docs/engine_limits.md)")

    names = [n for n, _ in spec.outputs]
    pages = [names[i:i + PAGE] for i in range(0, len(names), PAGE)]
    decl = "\n".join(f"global int {i}:o{i};" for i in range(1, PAGE + 1))
    page_scripts = []
    for pi, page in enumerate(pages):
        body = " ".join(f"o{i + 1} = GetSectorLightLevel({OUT_TAG0 + pi * PAGE + i});" for i in range(len(page)))
        page_scripts.append(f"script {100 + pi} (void) {{ {body} }}")
    page_scripts.append(guard_acs())
    layout["guard_script"] = GUARD_SCRIPT
    floor_tags = [PORT_TAG0 + i for i, p in enumerate(spec.ports) if p.startswith("floor:")]
    ceil_tags = [PORT_TAG0 + i for i, p in enumerate(spec.ports) if p.startswith("ceil:")]
    reset = "\n  ".join(
        [f"TeleportOther({DOLL_TID0 + h}, {HOME_TID0 + h}, 0);" for h in homes]
        + [f"Light_ChangeToValue({OUT_TAG0 + oi}, {bias});" for oi, (_, bias) in enumerate(spec.outputs)]
        + [f"Floor_MoveToValue({t}, {MOVE_SPEED}, {CLOSED}, 0);" for t in floor_tags]
        + [f"Ceiling_LowerToFloor({t}, {MOVE_SPEED});" for t in ceil_tags]
        + [f"Floor_MoveToValue({START_TAG}, {MOVE_SPEED}, {CLOSED}, 0);"])
    acs = f"""#include "zcommon.acs"
{decl}
script 1 (int tag, int h, int neg) {{ Floor_MoveToValue(tag, {MOVE_SPEED}, h, neg); }}
script 2 (int tag, int h, int neg) {{ Ceiling_MoveToValue(tag, {MOVE_SPEED}, h, neg); }}
script 3 (void) {{
  {reset}
}}
script 4 (int tid) {{ Thing_ChangeTID(0, tid); }}
script 5 (int tag) {{ Ceiling_LowerToFloor(tag, {MOVE_SPEED}); }}
script 6 (int tag) {{ Ceiling_MoveToValue(tag, {MOVE_SPEED}, {CEIL}, 0); }}
script 7 (void) {{ Floor_LowerToLowest({START_TAG}, {MOVE_SPEED}); }}
{chr(10).join(page_scripts)}
{motion_acs(len(homes))}
{extra_acs}
"""
    wad = Path(wad)
    if not layout_only:
        write_wad(wad, m.textmap(), compile_acs(acs))
    meta = TileMeta(name=spec.name, ports={p: PORT_TAG0 + i for i, p in enumerate(spec.ports)},
                    outputs={n: OUT_TAG0 + i for i, n in enumerate(names)}, pages=pages,
                    n_dolls=spec.n_dolls,
                    n_sectors=m.sector_count if layout_only else len(m.rects), layout=layout)
    if not layout_only:
        meta.save(wad.with_suffix(".json"))
    return meta
