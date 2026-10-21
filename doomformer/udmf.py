"""Tiny UDMF map builder: axis-aligned rectangular sectors that tile the plane.

Shared rectangle edges become two-sided lines automatically; per-pair line
properties (specials, blocking) are attached with `link`.
"""
from __future__ import annotations

import bisect
import struct
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

from . import acc


@dataclass
class Rect:
    x0: int
    y0: int
    x1: int
    y1: int
    floor: int
    ceil: int
    light: int = 160
    tag: int = 0
    special: int = 0
    ftex: str = "FLOOR0_1"
    ctex: str = "F_SKY1"
    wall: str = "STARTAN3"
    blocking: bool = False   # every two-sided line on this rect's border is impassable
    group: int | None = None  # lines between rects of different (non-None) groups are impassable
    facing: str | tuple | None = None  # upper/lower texture on neighbouring sides facing this rect: str or (top, bottom)
    sector: dict = field(default_factory=dict)  # extra UDMF sector properties (lightcolor, fadecolor, ...)

    def cw_edges(self):
        a, b, c, d = (self.x0, self.y0), (self.x0, self.y1), (self.x1, self.y1), (self.x1, self.y0)
        return [(a, b), (b, c), (c, d), (d, a)]  # clockwise (y up): interior on the right


@dataclass
class MapBuilder:
    rects: list[Rect] = field(default_factory=list)
    things: list[dict] = field(default_factory=list)
    links: dict[frozenset, dict] = field(default_factory=dict)
    wall_props: dict[int, list[dict]] = field(default_factory=dict)
    closet_origin: tuple[int, int] = (-8192, -8192)
    edge_props: dict[tuple[int, str], tuple[dict, dict]] = field(default_factory=dict)

    def rect(self, *args, **kw) -> int:
        self.rects.append(Rect(*args, **kw))
        return len(self.rects) - 1

    def link(self, a: int, b: int, **props):
        """Properties for every line segment shared by rects a and b."""
        self.links[frozenset((a, b))] = props

    def edge(self, rect: int, side: str, line: dict | None = None, sidedef: dict | None = None):
        """Extra properties for the lines on one border of `rect` (side: 'w' x0, 'e' x1, 's' y0, 'n' y1).

        `line` is merged into the linedef, `sidedef` into rect's own sidedef of that line."""
        self.edge_props[(rect, side)] = (line or {}, sidedef or {})

    def thing(self, x: float, y: float, type: int, angle: int = 0, **kw):
        self.things.append(dict(x=float(x), y=float(y), type=type, angle=angle, **kw))

    def static_specials(self, props: list[dict]):
        """Init-time line specials (scrollers etc.) on the walls of off-map closets, 4 per closet."""
        for i in range(0, len(props), 4):
            k = len(self.wall_props)
            cx, cy = self.closet_origin
            r = self.rect(cx + 96 * k, cy, cx + 96 * k + 64, cy + 64, floor=0, ceil=64)
            self.wall_props[r] = list(props[i:i + 4])

    def scroll_carry(self, tag: int, xmove: int, ymove: int, stack: int):
        """Conveyor: `stack` additive Boom carry scrollers on sectors tagged `tag` (speed byte 128 = 0)."""
        self.static_specials([dict(special=223, arg0=tag, arg1=0, arg2=1, arg3=xmove, arg4=ymove)] * stack)

    def _segments(self):
        points = {p for r in self.rects for e in r.cw_edges() for p in e}
        on_x: dict[int, list[int]] = {}   # x -> sorted ys of points on that vertical line
        on_y: dict[int, list[int]] = {}   # y -> sorted xs of points on that horizontal line
        for x, y in points:
            on_x.setdefault(x, []).append(y)
            on_y.setdefault(y, []).append(x)
        for lst in (*on_x.values(), *on_y.values()):
            lst.sort()
        segs: dict[tuple, list] = {}
        for ri, r in enumerate(self.rects):
            for p, q in r.cw_edges():
                vertical = p[0] == q[0]
                axis = 1 if vertical else 0
                lo, hi = sorted((p[axis], q[axis]))
                fixed = p[0] if vertical else p[1]
                line = (on_x if vertical else on_y)[fixed]
                cuts = line[bisect.bisect_left(line, lo):bisect.bisect_right(line, hi)]
                pts = [(fixed, c) if vertical else (c, fixed) for c in cuts]
                if p[axis] > q[axis]:
                    pts.reverse()
                for s, t in zip(pts, pts[1:]):
                    segs.setdefault(tuple(sorted((s, t))), []).append((ri, s, t))
        return segs

    def textmap(self) -> str:
        out = ['namespace = "zdoom";']
        vidx: dict[tuple, int] = {}
        verts, lines, sides = [], [], []
        pending = {r: list(p) for r, p in self.wall_props.items()}

        def v(p):
            if p not in vidx:
                vidx[p] = len(verts)
                verts.append(p)
            return vidx[p]

        def side(ri, upper_lower_tex=True, other=None, extra=None):
            r = self.rects[ri]
            top = bottom = r.wall
            f = self.rects[other].facing if other is not None else None
            if isinstance(f, str):
                top = bottom = f
            elif f:
                top, bottom = f[0] or top, f[1] or bottom
            sides.append(dict(sector=ri, texturemiddle="-" if upper_lower_tex else r.wall,
                              texturetop=top, texturebottom=bottom) | (extra or {}))
            return len(sides) - 1

        def edge_name(ri, s, t):
            r = self.rects[ri]
            if s[0] == t[0]:
                return "w" if s[0] == r.x0 else "e"
            return "s" if s[1] == r.y0 else "n"

        for key, owners in self._segments().items():
            if len(owners) > 2:
                raise ValueError(f"overlapping sectors at {key}: {owners}")
            (ra, s, t) = owners[0]
            line = dict(v1=v(s), v2=v(t))
            ea = self.edge_props.get((ra, edge_name(ra, s, t)), ({}, {}))
            if len(owners) == 1:
                line.update(sidefront=side(ra, False, extra=ea[1]), blocking=True)
                if pending.get(ra):
                    line.update(pending[ra].pop(0))
                line.update(ea[0])
            else:
                rb, s2, t2 = owners[1]
                eb = self.edge_props.get((rb, edge_name(rb, s2, t2)), ({}, {}))
                line.update(sidefront=side(ra, other=rb, extra=ea[1]), sideback=side(rb, other=ra, extra=eb[1]),
                            twosided=True)
                line.update(self.links.get(frozenset((ra, rb)), {}))
                line.update(ea[0])
                line.update(eb[0])
                a, b = self.rects[ra], self.rects[rb]
                if a.blocking or b.blocking or (a.group is not None and b.group is not None and a.group != b.group):
                    line["blocking"] = True
            lines.append(line)

        for x, y in verts:
            out.append(f"vertex {{ x = {x:.1f}; y = {y:.1f}; }}")
        for ln in lines:
            out.append("linedef { " + " ".join(f"{k} = {_fmt(val)};" for k, val in ln.items()) + " }")
        for sd in sides:
            out.append("sidedef { " + " ".join(f"{k} = {_fmt(val)};" for k, val in sd.items()) + " }")
        for r in self.rects:
            extra = "".join(f" {k} = {_fmt(val)};" for k, val in r.sector.items())
            out.append(
                f'sector {{ heightfloor = {r.floor}; heightceiling = {r.ceil}; texturefloor = "{r.ftex}"; '
                f'textureceiling = "{r.ctex}"; lightlevel = {r.light}; id = {r.tag}; special = {r.special};{extra} }}')
        flags = {f"skill{i}": True for i in range(1, 6)} | dict(single=True, coop=True, dm=True)
        for t in self.things:
            out.append("thing { " + " ".join(f"{k} = {_fmt(val)};" for k, val in (t | flags).items()) + " }")
        return "\n".join(out) + "\n"


def _fmt(val) -> str:
    if isinstance(val, bool):
        return "true" if val else "false"
    if isinstance(val, str):
        return f'"{val}"'
    return str(val)


def compile_acs(source: str) -> bytes:
    """Compile ACS source with acc (located or built by `acc.ensure()`) and return the object code."""
    exe = acc.ensure()
    with tempfile.TemporaryDirectory() as d:
        src, obj = Path(d, "map.acs"), Path(d, "map.o")
        src.write_text(source)
        r = subprocess.run([str(exe), "-i", str(acc.include_dir(exe)), str(src), str(obj)], capture_output=True,
                           text=True)
        if r.returncode or not obj.exists():
            raise RuntimeError(f"acc failed:\n{r.stdout}\n{r.stderr}")
        return obj.read_bytes()


def write_wad(path: Path, textmap: str, behavior: bytes, mapname: str = "MAP01",
              extra_lumps: list[tuple[str, bytes]] | None = None):
    lumps = list(extra_lumps or []) + [
        (mapname, b""), ("TEXTMAP", textmap.encode()), ("BEHAVIOR", behavior), ("ENDMAP", b"")]
    data, directory, ofs = b"", b"", 12
    for name, blob in lumps:
        directory += struct.pack("<ii8s", ofs, len(blob), name.encode())
        data += blob
        ofs += len(blob)
    Path(path).write_bytes(struct.pack("<4sii", b"PWAD", len(lumps), 12 + len(data)) + data + directory)
