"""Follow-up probes written after the first run of the exported maps in GZDoom (docs/engine_limits.md, "Other ports").

GZDoom's UDMF parser refuses a coordinate outside [-32768, 32768], so the `wrap_lone_room` maps whose lone room ends
east of 32,768 do not load there. `wrap_shifted` keeps every coordinate inside that range and still puts 36,000 or more
units between the lone room and the west edge of the cluster: the 150 rooms stand in x = -6,000 .. -144 instead of
0 .. 5,856. Imported at the end of facts.py, so `doomformer facts` runs these and `--export` writes them.
"""
from __future__ import annotations

from .facts import ROOM_CEIL, Case, ProbeMap, probe

_SHIFT = -6000
_SHIFTED = {26000: 64, 28000: 64, 30000: 0, 31000: 0, 32000: 0}      # floor read at the lone room (64 = found)


def _shifted(x: int) -> Case:
    m = ProbeMap()
    for i in range(15):
        for j in range(10):
            m.rect(_SHIFT + i * 400, j * 400, _SHIFT + i * 400 + 256, j * 400 + 256, floor=0, ceil=ROOM_CEIL)
    m.player = (_SHIFT + 128, 128)
    m.rect(x, 0, x + 256, 256, floor=64, ceil=ROOM_CEIL)
    s = f"""  Spawn("MapSpot", {x + 128}.0, 128.0, 64.0, 900);
  r_floor = GetActorFloorZ(900) >> 16;"""
    return Case(f"wrap_shifted_{x}", m, ["floor"], s, dict(floor=_SHIFTED[x]))


@probe("wrap_shifted", "the wrap with every coordinate inside +-32,768: 150 rooms in x = -6,000 .. -144 and one lone room; "
       "a lone room starting at 26,000 or 28,000 is found in its own sector, at 30,000, 31,000 and 32,000 (36,000 or more "
       "units from the cluster's west edge) it is filed under a neighbour (ViZDoom 1.3.2: a lone room at 32,512 crashed "
       "the engine process, so it is not in the list)", "engine_limits.md: Other ports")
def _wrap_shifted():
    return [_shifted(x) for x in _SHIFTED]


# ---- line speed argument ------------------------------------------------------------------------------------------

from .facts import CARRY, TAG_SCRIPTS, _lane, base_map                         # noqa: E402

_SPEEDS = (150000, 255, 240, 100, 0)            # 0: control lane without the special


@probe("line_floor_speed", "Floor_LowerToLowest (special 21) on a line crossed by a voodoo doll, with the speed argument "
       "150,000 (the compiler's MOVE_SPEED), 255, 240 and 100: the 24-unit step sector next to the doll's lane drops to its "
       "neighbour's floor (0); without the special (s0) it stays at 24", "engine_limits.md: Other ports")
def _line_floor_speed():
    m = base_map()
    for k, speed in enumerate(_SPEEDS):
        ids = _lane(m, 0, 1024 + 128 * k, 500 + k, [(32, 0, 256, 300), (64, 24, 256, 50 + k), (64, 0, 256, 0)], stub_tag=300)
        if speed:
            m.link(ids[0], ids[1], special=21, arg0=50 + k, arg1=speed, playercross=True, repeatspecial=True)
    m.scroll_carry(300, stack=4, **CARRY)
    s = "  Delay(80);\n" + "".join(f"  r_s{v} = GetSectorFloorZ({50 + k}, 0, 0) >> 16;\n" for k, v in enumerate(_SPEEDS))
    return [Case("line_floor_speed", m, [f"s{v}" for v in _SPEEDS], s, dict(_LINE_SPEED), extra=TAG_SCRIPTS)]


_LINE_SPEED = {"s150000": 0, "s255": 0, "s240": 0, "s100": 0, "s0": 24}


# ---- sector tags above 32,767 ---------------------------------------------------------------------------------------

from .facts import box                                                         # noqa: E402

_TAGS = (1000, 20000, 32767, 32768, 40000, 60000, 64004, 65535)


@probe("tag_range", "sector tags up to 65,535 (the compiler uses 1,000 ... 64,0xx): GetSectorFloorZ(tag) finds the sector "
       "with that tag, and a doll-crossed Floor_LowerToLowest line with a tag above 32,767 as arg0 acts on it "
       "(line64010 = -8, the neighbour's floor)", "engine_limits.md: Other ports")
def _tag_range():
    m = base_map()
    for i, t in enumerate(_TAGS):
        box(m, 256 * i, 0, floor=8 * (i + 1), ceil=256, tag=t)
    ids = _lane(m, 0, 1024, 500, [(32, 0, 256, 300), (64, 24, 256, 64010), (64, -8, 256, 0)], stub_tag=300)
    m.link(ids[0], ids[1], special=21, arg0=64010, arg1=150000, playercross=True, repeatspecial=True)
    m.scroll_carry(300, stack=4, **CARRY)
    s = "".join(f"  r_t{t} = GetSectorFloorZ({t}, 0, 0) >> 16;\n" for t in _TAGS)
    s += "  Delay(80);\n  r_line64010 = GetSectorFloorZ(64010, 0, 0) >> 16;\n"
    keys = [f"t{t}" for t in _TAGS] + ["line64010"]
    return [Case("tag_range", m, keys, s, {**{f"t{t}": 8 * (i + 1) for i, t in enumerate(_TAGS)}, "line64010": -8},
                 extra=TAG_SCRIPTS)]
