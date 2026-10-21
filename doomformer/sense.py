"""Integer sensors for :class:`doomformer.agent.Agent`.

ACS samples engine primitives once per step; Python replicates the integer
selection/binning from that recorded sample, never from a later game state.
VectorAngle, VectorLength, CheckSight and sector membership are engine queries,
not neural-network arithmetic (nor Python-reimplemented collision geometry).
"""
from __future__ import annotations

from dataclasses import dataclass
import math

from . import policy
from .brainacs import MARGIN, NO_ENEMY_TOKENS


@dataclass(frozen=True)
class Sensor:
    """A built-in sensor specification; use the constructors below."""
    kind: str
    scale: int = 1
    offset: int = 0
    tag: int = 0
    tokens: bool = False

    def __post_init__(self):
        if self.kind not in {"bearing", "distance", "health", "ammo", "x", "y", "count", "sector", "sight", "constant"}:
            raise ValueError(f"unknown sensor {self.kind!r}")
        if not isinstance(self.scale, int) or not 1 <= self.scale <= 32767:
            raise ValueError("sensor scale must be an integer in 1..32767")
        if not isinstance(self.offset, int) or not -12000 <= self.offset <= 12000:
            raise ValueError("sensor offset must be an integer in -12000..12000")
        if self.kind == "sector" and not 3 <= self.tag <= 999:
            raise ValueError("world sector tags must be in 3..999")

    def expression(self, raw: str) -> str:
        if self.kind == "bearing":
            return "ag_bearing" if self.tokens else f"(ag_bearing - {policy.VOCAB.index('a')})"
        if self.kind == "distance":
            return "ag_distance" if self.tokens else f"(ag_distance - {policy.VOCAB.index('0')})"
        if self.kind == "count":
            return "ag_alive"
        if self.kind == "sight":
            return "ag_sight"
        if self.kind == "constant":
            return str(self.offset)
        if self.kind == "sector":
            return f"({raw} != 0)"
        value = f"({raw} >> 16)" if self.kind in {"x", "y"} else raw
        return f"(({value} - ({self.offset})) / {self.scale})"

    def primitive(self, player_tid: int) -> str:
        return {
            "health": f"GetActorProperty({player_tid}, APROP_Health)",
            "ammo": f'CheckActorInventory({player_tid}, "Clip")',
            "x": f"GetActorX({player_tid})", "y": f"GetActorY({player_tid})",
            "sector": f"ThingCountSector(T_NONE, {player_tid}, {self.tag})",
        }.get(self.kind, "0")

    def replica(self, raw: int, nearest: dict) -> int:
        if self.kind == "bearing":
            return nearest["bearing"] - (0 if self.tokens else policy.VOCAB.index('a'))
        if self.kind == "distance":
            return nearest["distance"] - (0 if self.tokens else policy.VOCAB.index('0'))
        if self.kind == "count":
            return nearest["count"]
        if self.kind == "sight":
            return nearest["sight"]
        if self.kind == "constant":
            return self.offset
        if self.kind == "sector":
            return int(raw != 0)
        value = (raw >> 16) if self.kind in {"x", "y"} else raw
        value -= self.offset
        return (1 if value >= 0 else -1) * (abs(value) // self.scale)


def bearing(*, tokens=False):
    """Nearest live monster bearing: bins 0..16, no monster 17; policy token IDs if tokens=True.

    Edges (degrees): -100,-60,-35,-20,-12,-7,-4,-2,2,4,7,12,20,35,60,100.
    Uses brainacs's fixed-point edge correction, not ideal floating-point angles.
    """
    return Sensor("bearing", tokens=tokens)


def distance(*, tokens=False):
    """Nearest live monster distance bin 0..4 (edges 100,200,350,550); absent=4."""
    return Sensor("distance", tokens=tokens)


def health(scale=1, offset=0):
    """trunc_toward_zero((sampled integer health - offset) / scale)."""
    return Sensor("health", scale, offset)


def ammo(scale=1, offset=0):
    """trunc_toward_zero((sampled Clip inventory - offset) / scale)."""
    return Sensor("ammo", scale, offset)


def x_bin(width=64, origin=0):
    """trunc_toward_zero((floor(sampled player X) - origin) / width)."""
    return Sensor("x", width, origin)


def y_bin(width=64, origin=0):
    """trunc_toward_zero((floor(sampled player Y) - origin) / width)."""
    return Sensor("y", width, origin)


def monster_count():
    """Number of actors with the Agent's monster_tid and sampled health > 0."""
    return Sensor("count")


def sector_flag(tag):
    """1 iff ThingCountSector(T_NONE, player_tid, tag) is nonzero at sample time."""
    return Sensor("sector", tag=tag)


def line_of_sight():
    """CheckSight(player, nearest live monster, 0), or 0 when none is alive."""
    return Sensor("sight")


def constant(value):
    """An integer constant, useful for a transformer's BOS token."""
    return Sensor("constant", offset=value)


def _mul(a, b):
    return (a * b) >> 16


def _trig(angle):
    k = ((angle + 8192) >> 14) & 3
    r = ((angle + 8192) & 16383) - 8192
    x = r * 6 + ((r * 18559) >> 16)
    x2 = _mul(x, x)
    sn = _mul(x, 65536 - _mul(x2, 10923 - _mul(x2, 546 - _mul(x2, 13))))
    cs = 65536 - _mul(x2, 32768 - _mul(x2, 2731 - _mul(x2, 91)))
    return ((cs, sn), (-sn, cs), (-cs, -sn), (sn, -cs))[k]


def nearest_replica(angle, candidates):
    """Independent integer selection and binning of instrumented engine queries.

    Candidate rows are dx, dy (16.16), VectorLength (16.16), signed
    VectorAngle-relative-to-player (16-bit turns), CheckSight, and health.
    Equal distance chooses smaller signed bearing; an exact tie keeps first.
    """
    live = [row for row in candidates if row[5] > 0]
    if not live:
        return dict(bearing=NO_ENEMY_TOKENS[0], distance=NO_ENEMY_TOKENS[1], count=0, sight=0)
    dx, dy, dist, b, sight, _ = min(live, key=lambda row: (row[2], row[3]))
    c, s = _trig(angle)
    rx, ry = _mul(dx, c) + _mul(dy, s), _mul(dy, c) - _mul(dx, s)
    base = policy.VOCAB.index('a')
    if b >= 32768 - MARGIN or b <= -32768 + MARGIN:
        bt = base + (len(policy.BEARING_EDGES) if ry >= 0 else 0)
    else:
        bt = base
        for edge in policy.BEARING_EDGES:
            threshold = math.ceil(edge * 65536 / 360)
            c = round(math.cos(math.radians(edge)) * 65536)
            s = round(math.sin(math.radians(edge)) * 65536)
            bt += int(b > threshold + MARGIN or
                      (b >= threshold - MARGIN and _mul(ry, c) - _mul(rx, s) >= 0))
    dt = policy.VOCAB.index('0') + sum(dist >= edge << 16 for edge in policy.DIST_EDGES)
    return dict(bearing=bt, distance=dt, count=len(live), sight=int(sight != 0))
