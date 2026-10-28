"""Static extraction: read a compiled MLP map and recover the network it encodes.

`extract(wad)` reads one file, the map's .wad: its UDMF TEXTMAP (sectors, linedefs with their special numbers, arguments
and tags, things) and its BEHAVIOR lump (ACS bytecode). It does not read the source net, the .json that `compile`
writes next to the map, or the lowering code's layout tables. The result is a `Netlist`: per layer the gates of every
doll lane, the weight lines each doll crosses and the accumulators' start values. `Netlist.diff(net)` compares it with
the netlist that a `kit.MLP` implies (`expected_netlist`, derived here from the net's arithmetic alone), and
`Netlist.to_mlp()` reads the layers back as an MLP when every lane agrees. `extract(compile(net)).equals(net)` is
the static counterpart of the exactness check: it shows which weights, biases and thresholds the map holds, where the
exactness check shows that DOOM computed the same values for the inputs that were run.

What the map says is not what the engine does. This module encodes what each special does by its own reading of the
engine facts (docs/HOW_IT_WORKS.md, "Engine facts"); it is not a model of ZDoom. A map that is read as net N here is
exact in DOOM only as far as those readings hold, which the exactness checks (fuzz, verify) test.

Engine semantics used (one reading each):
  gate     a doll standing on the walkway of floor f steps onto a sector whose floor is at most f + 24 (STEP_UP) and
           is blocked by a higher one: a floor gate whose floor is the port value v passes iff v <= f + 24. The
           gate limit is that f + 24; the lane's threshold is t = -limit (ports carry -x, or x for the negative
           staircase, so v <= -t is x >= t).
  lines    a doll crossing a line with a special (playercross) runs it once:
             Light_RaiseByValue / Light_LowerByValue (110/111)   light level of tag +-arg1
             Floor_RaiseByValue / Floor_LowerByValue (23/20)     floor height of tag +-arg2 (mover speed arg1)
             Ceiling_RaiseByValue / Ceiling_LowerByValue (41/40) ceiling height of tag +-arg2
  sums     the final value of a light or a plane is its start value plus the sum of every crossed line's amount.
  barriers Floor_LowerToLowest (21) on a barrier tag releases the dolls behind it (the barrier floor goes to the lower
           neighbour's); ACS_ExecuteAlways (226) runs a script; Floor_MoveToValue (37) freezes a finished layer's
           walkway; Teleport_NoFog (71) and TeleportOther (76) move a doll to the thing with the destination's id.

What is read (every item is a finding when it differs from what the compiler writes; `Netlist.checks` and `findings`):
  * every linedef with a special: its class, tag, every argument (unused ones must be 0), mover speed (150,000 to
    262,144), and its trigger flags (exactly playercross and repeatspecial on a two-sided line); plain two-sided lines
    carry no flags; weight lines outside any lane; the conveyor scrollers; blocking two-sided lines.
  * ACS scripts 1, 2, 3, 4, 7, the copy scripts and the readout pages are run or decoded and compared with the TEXTMAP:
    the reset script restores every light, internal plane and barrier to its start value, once, and sends every doll
    to a destination thing in its own home sector; script 7 releases layer 0; the pages read every light and internal
    plane once.
  * lane and clock geometry: sector specials, the conveyor tag, steps and headroom, barriers that block closed and open
    to a walkable step, closed sector boundaries, one doll thing per home plus the player, thing flags, teleport
    destinations inside a lane sector.
  * the plane contract (planes.plane_errors) on the recovered net.
`Netlist.not_checked` and `NOT_CHECKED` list what is not read; EQUAL is a statement about the arithmetic netlist, the
plane contract and the items above, not about timing or about anything on that list.

Encoding conventions the compiler uses and this module relies on (a violation raises `ExtractError`, or is listed in
`Netlist.findings` when the net is still readable):
  * a doll lane is a chain of sectors along x: home, home2, start barrier, pad, gate, freeze sector, ... wave
    barrier, effect cells. Layer s's start barrier has tag 2 (s = 0) or 60000 + s.
  * the gate sector's tag is the port; the walkway floor of the lane (home, home2, pad) is gate limit - 24.
  * accumulators that no doll reads are isolated sectors with a tag; their light level is the start value. A plane
    that feeds a later layer is the floor of that layer's gate sectors, and its start height is the gate floor.
  * ports and accumulators are numbered in order of first use: input ports in pairs (-x_i, x_i) for a signed input
    layer, units in ascending port tag within a layer, output lights in ascending tag. The map holds no names, so
    "signed input" is a reading of that pairing: a relu net whose input columns come in negated pairs reads as a
    signed net; `equals` compares in the port form of the net under its own `signed_input`, so it is not affected.
  * each hidden unit j of a layer writes the light of h_j and the plane of -h_j in consecutive lines of a lane
    (opt 3: the light only, and an ACS script copies it into the plane).
Not recoverable from a map: inputs whose weights are all zero (no lane exists), hidden units with no outgoing weight
whose row is also zero, and leading all-zero output rows of a net with hidden layers (their lights cannot be told
from a hidden layer's); `expected_netlist` drops the same things, and the multiset of the remaining unattributed
lights' start values is compared.
"""
from __future__ import annotations

import re
import struct
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

# --- engine facts, copied (with the compiler's names) so that extraction does not import the lowering code ------------
STEP_UP = 24                     # Doom's step-up limit (tile.STEP_UP)
CEIL = 16000                     # ceiling of a floor gate (tile.CEIL)
MOVE_SPEED_MIN = 150000          # plane mover speed that reaches any port height in one tic (tile.MOVE_SPEED)
CONV_TAG = 1                     # conveyor sectors (tile.CONV_TAG)
START_TAG = 2                    # layer 0's start barrier (tile.START_TAG)
STAGE_TAG0 = 60000               # start barrier of layer s >= 1 is STAGE_TAG0 + s (pure.STAGE_TAG0)
FLOOR_RAISE, FLOOR_LOWER, LOWER_TO_LOWEST, FLOOR_MOVE_TO = 23, 20, 21, 37
CEIL_RAISE, CEIL_LOWER, EXEC_ALWAYS, SCROLL_FLOOR = 41, 40, 226, 223
LIGHT_RAISE, LIGHT_LOWER = 110, 111
LIGHT_CHANGE_TO, TELEPORT_OTHER, CEIL_MOVE_TO, CEIL_LOWER_TO_FLOOR, TELEPORT = 112, 76, 47, 254, 71
PLAYER1_START = 1

EFFECTS = {LIGHT_RAISE: ("light", +1), LIGHT_LOWER: ("light", -1), FLOOR_RAISE: ("plane", +1),
           FLOOR_LOWER: ("plane", -1), CEIL_RAISE: ("cplane", +1), CEIL_LOWER: ("cplane", -1)}


class ExtractError(ValueError):
    """The map is not a compiled MLP in the form this module reads (the message says which convention fails)."""


# --- WAD and TEXTMAP --------------------------------------------------------------------------------------------------
def read_lumps(path) -> dict[str, bytes]:
    """Lumps of the first map of a PWAD by name (TEXTMAP, BEHAVIOR, ...)."""
    data = Path(path).read_bytes()
    if data[:4] not in (b"PWAD", b"IWAD"):
        raise ExtractError(f"{path}: not a WAD")
    n, directory = struct.unpack("<ii", data[4:12])
    out: dict[str, bytes] = {}
    for i in range(n):
        offset, size, name = struct.unpack("<ii8s", data[directory + 16 * i:directory + 16 * (i + 1)])
        out.setdefault(name.rstrip(b"\0").decode(), data[offset:offset + size])
    if "TEXTMAP" not in out:
        raise ExtractError(f"{path}: no TEXTMAP lump (a UDMF map is required)")
    return out


_BLOCK = re.compile(r"^(\w+) \{ (.*) \}$")
_KV = re.compile(r'(\w+) = ("[^"]*"|[^;]*);')
TWOSIDED, BLOCKING, PLAYERCROSS, REPEAT = 1, 2, 4, 8
_LINE_KEYS = {"v1", "v2", "sidefront", "sideback", "special", "arg0", "arg1", "arg2", "arg3", "arg4"}
_FLAG_BITS = {"twosided": TWOSIDED, "blocking": BLOCKING, "playercross": PLAYERCROSS, "repeatspecial": REPEAT}


class TextMap:
    """The parts of a UDMF map that extraction reads: per sector floor, ceiling, light, tag; per linedef its two
    sectors, special, arguments and flags; things."""

    def __init__(self, text: str):
        vx, vy, raw_lines, sidedefs, sectors, things = [], [], [], [], [], []
        self.bits = dict(_FLAG_BITS)                 # every boolean linedef key of this map -> bit
        self.extra_line_keys: set[str] = set()       # numeric linedef keys the reader does not interpret (id, alpha, ...)
        for ln in text.splitlines():
            m = _BLOCK.match(ln)
            if m is None:
                continue
            kind, f = m.group(1), dict(_KV.findall(m.group(2)))
            if kind == "vertex":
                vx.append(float(f["x"]))
                vy.append(float(f["y"]))
            elif kind == "linedef":
                flags = 0
                for k, v in f.items():
                    if k in _LINE_KEYS:
                        continue
                    if v in ("true", "false"):
                        if v == "true":
                            flags |= self.bits.setdefault(k, 1 << len(self.bits))
                    else:
                        self.extra_line_keys.add(k)
                raw_lines.append((int(f["v1"]), int(f["v2"]), int(f["sidefront"]), int(f.get("sideback", -1)),
                                  int(f.get("special", 0)), *(int(f.get(f"arg{i}", 0)) for i in range(5)), flags))
            elif kind == "sidedef":
                sidedefs.append(int(f["sector"]))
            elif kind == "sector":
                sectors.append((int(f["heightfloor"]), int(f["heightceiling"]), int(f["lightlevel"]), int(f.get("id", 0)),
                                f.get("moreids", "").strip('"'), int(f.get("special", 0))))
            elif kind == "thing":
                things.append((float(f["x"]), float(f["y"]), int(f["type"]), int(f.get("id", 0)),
                               frozenset(k for k, v in f.items() if v == "true")))
        if not sectors or not raw_lines:
            raise ExtractError("TEXTMAP has no sectors or no linedefs")
        if any(not 0 <= r[2] < len(sidedefs) or not -1 <= r[3] < len(sidedefs) or not 0 <= r[0] < len(vx)
               or not 0 <= r[1] < len(vx) for r in raw_lines) or any(not 0 <= sd < len(sectors) for sd in sidedefs):
            raise ExtractError("a linedef or sidedef refers to a vertex, sidedef or sector that does not exist")
        self.floor = [s[0] for s in sectors]
        self.ceil = [s[1] for s in sectors]
        self.light = [s[2] for s in sectors]
        self.tag = [s[3] for s in sectors]
        self.special = [s[5] for s in sectors]
        self.things = things
        self.n_sectors = len(sectors)
        # linedefs: (sector_front, sector_back or -1, special, a0..a4, flags)
        self.lines = [(sidedefs[a], sidedefs[b] if b >= 0 else -1, sp, a0, a1, a2, a3, a4, fl)
                      for (_, _, a, b, sp, a0, a1, a2, a3, a4, fl) in raw_lines]
        # sector bounding boxes from the linedefs' vertices
        vx_a, vy_a = np.array(vx), np.array(vy)
        v1 = np.array([r[0] for r in raw_lines])
        v2 = np.array([r[1] for r in raw_lines])
        front = np.array([sidedefs[r[2]] for r in raw_lines])
        back = np.array([sidedefs[r[3]] if r[3] >= 0 else -1 for r in raw_lines])
        self.x0 = np.full(self.n_sectors, np.inf)
        self.y0 = np.full(self.n_sectors, np.inf)
        self.x1 = np.full(self.n_sectors, -np.inf)
        self.y1 = np.full(self.n_sectors, -np.inf)
        for secs, a, b in ((front, v1, v2), (back, v1, v2)):
            keep = secs >= 0
            s, a, b = secs[keep], a[keep], b[keep]
            for vs in (a, b):
                np.minimum.at(self.x0, s, vx_a[vs])
                np.minimum.at(self.y0, s, vy_a[vs])
                np.maximum.at(self.x1, s, vx_a[vs])
                np.maximum.at(self.y1, s, vy_a[vs])
        # every sector's boundary must close: each vertex is met by an even number of its boundary lines
        nv = len(vx)
        has_back = back >= 0
        keys = np.concatenate([front.astype(np.int64) * nv + v1, front.astype(np.int64) * nv + v2,
                               back[has_back].astype(np.int64) * nv + v1[has_back],
                               back[has_back].astype(np.int64) * nv + v2[has_back]])
        uniq, cnt = np.unique(keys, return_counts=True)
        self.open_sectors = sorted({int(k // nv) for k in uniq[cnt % 2 == 1]})
        # adjacency over two-sided lines between different sectors
        self.edge: dict[tuple[int, int], list[int]] = {}
        self.nbr: dict[int, set[int]] = {}
        self.isolated = np.ones(self.n_sectors, dtype=bool)
        for i, (a, b, *_rest) in enumerate(self.lines):
            self.isolated[a] = False
            if b < 0:
                continue
            self.isolated[b] = False
            if a != b:
                self.edge.setdefault((a, b) if a < b else (b, a), []).append(i)
                self.nbr.setdefault(a, set()).add(b)
                self.nbr.setdefault(b, set()).add(a)
        # a sector with only one-sided lines has no two-sided neighbour; recompute "isolated" accordingly
        self.isolated = np.array([s not in self.nbr for s in range(self.n_sectors)])

    def lines_between(self, a: int, b: int) -> list[tuple]:
        return [self.lines[i] for i in self.edge.get((a, b) if a < b else (b, a), [])]

    def flag_names(self, mask: int) -> set[str]:
        return {k for k, b in self.bits.items() if mask & b}

    def things_at(self, sector: int, type_: int) -> list[tuple]:
        return [t for t in self.things if t[2] == type_ and self.x0[sector] <= t[0] <= self.x1[sector]
                and self.y0[sector] <= t[1] <= self.y1[sector]]

    def things_in(self, sector: int, type_: int) -> int:
        n = 0
        for x, y, t, *_ in self.things:
            if t == type_ and self.x0[sector] <= x <= self.x1[sector] and self.y0[sector] <= y <= self.y1[sector]:
                n += 1
        return n


# --- the netlist --------------------------------------------------------------------------------------------------------
@dataclass
class Doll:
    """One lane: its gate and the weight lines it crosses (kind, tag, signed amount) in lane order."""
    layer: int
    port: int
    limit: int
    lines: list[tuple[str, int, int]]
    sectors: list[int]
    wave: int | None = None
    freeze: int | None = None
    tid: int | None = None
    wave_i: int | None = None
    n_lines: int = 0


@dataclass
class Layer:
    """One layer in port form. `ports[p]` is the tag of gate port p; lane k of port p has gate limit `limits[p][k]`
    and adds `fx[p][k, r]` to row r. `bias[r]` is row r's start value (hidden: the h value, the plane starts at its
    negation). `dead` lists units nothing reads, as (bias, row weights per port and lane); `unattached` the start values
    of lights no doll writes and that no row owns."""
    index: int
    ports: list[int]
    limits: list[list[int]]
    fx: list[np.ndarray]
    bias: np.ndarray
    row_tags: list[int]
    dead: list[tuple[int, tuple]] = field(default_factory=list)
    final: bool = False
    col_ids: list[int] | None = None          # expected_netlist only: the net's own column and row numbers
    row_ids: list[int] | None = None
    fxl: list[np.ndarray] | None = None       # extract only: the same weights read from each hidden unit's twin light

    @property
    def n_rows(self) -> int:
        return len(self.bias)


@dataclass
class Diff:
    """One place where two netlists differ. layer/row/col/lane are 0-based in the net's own numbering when known."""
    what: str
    layer: int
    row: int | None
    col: int | None
    lane: int | None
    got: object
    want: object
    text: str

    def __str__(self) -> str:
        return self.text


@dataclass
class Netlist:
    layers: list[Layer]
    unattached: list[int]
    findings: list[str] = field(default_factory=list)
    checks: dict = field(default_factory=dict)
    source: str = ""
    not_checked: list[str] = field(default_factory=list)

    # --- readings ---------------------------------------------------------------------------------------------
    def signed_reading(self) -> bool:
        """True when layer 0's ports come in (-x, x) pairs with equal gate limits and negated lines: the structure a
        signed input layer has. A relu input layer of negated column pairs has the same structure (see the module
        docstring); `diff` does not depend on this reading."""
        L = self.layers[0]
        if len(L.ports) % 2:
            return False
        for p in range(0, len(L.ports), 2):
            if L.limits[p] != L.limits[p + 1] or not np.array_equal(L.fx[p], -L.fx[p + 1]):
                return False
        return True

    def lane_consistent(self) -> bool:
        """Every lane of a port has the same weights, and every port of a layer the same lane limits."""
        for L in self.layers:
            for p in range(len(L.ports)):
                if (L.fx[p] != L.fx[p][:1]).any():
                    return False
            if len({tuple(x) for x in L.limits}) > 1:
                return False
        return True

    def to_mlp(self):
        """The recovered `kit.MLP` (a layer count, widths, weights, biases, thresholds, signed or relu input). Raises
        ExtractError when the lanes of a port disagree, the layers' thresholds differ between ports, or a signed reading
        is not exact. Inputs and units without a lane are not in the result (module docstring)."""
        from .kit import MLP
        if not self.lane_consistent():
            raise ExtractError("lanes disagree: the map is not an MLP (use Netlist.diff for the places)")
        signed = self.signed_reading()
        Ws, bs, thr = [], [], []
        for L in self.layers:
            cols = np.stack([f[0] for f in L.fx], axis=1) if L.fx else np.zeros((L.n_rows, 0), dtype=np.int64)
            if L.index == 0 and signed:
                cols = cols[:, 0::2]
            Ws.append(cols.astype(np.int64))
            bs.append(np.asarray(L.bias, dtype=np.int64))
            thr.append(np.array(sorted(-x for x in L.limits[0]), dtype=np.int64))
        return MLP(Ws, bs, thr, signed_input=signed)

    # --- comparison ------------------------------------------------------------------------------------------------
    def diff(self, net) -> list[Diff]:
        """Every difference between this netlist and the one `net` implies (empty: equal). Differences are in the net's
        numbering: layer l, row j of W_l, column i of W_l (for layer 0 of a signed net also the sign of the port)."""
        return diff_netlists(self, expected_netlist(net))

    def equals(self, net) -> bool:
        return not self.findings and not self.diff(net)

    # --- dump ----------------------------------------------------------------------------------------------------
    def dump(self, full: bool = False, limit_rows: int = 6, limit_cols: int = 12) -> str:
        """The recovered netlist per layer: thresholds, biases, weights (the + port of a signed input, lane 0)."""
        out = [f"netlist of {self.source}" if self.source else "netlist"]
        signed = self.signed_reading()
        for L in self.layers:
            kind = "output layer" if L.final else ("input layer" if L.index == 0 else "hidden layer")
            sets = sorted({tuple(sorted(-v for v in x)) for x in L.limits})
            ports = L.ports[0::2] if (L.index == 0 and signed) else L.ports
            idx = list(range(0, len(L.ports), 2)) if (L.index == 0 and signed) else list(range(len(L.ports)))
            W = np.stack([L.fx[p][0] for p in idx], axis=1) if idx else np.zeros((L.n_rows, 0), dtype=np.int64)
            lane_ok = all((f == f[:1]).all() for f in L.fx)
            sign = (", signed input (ports -x, x)" if signed else ", relu input (ports -x)") if L.index == 0 else ""
            out.append(f"layer {L.index} ({kind}{sign}): {len(ports)} inputs, {L.n_rows} rows, "
                       f"{len(L.ports)} gate ports, {sum(len(x) for x in L.limits)} lanes")
            out.append("  thresholds " + ("; ".join(str(list(t)) for t in sets[:3])
                                           + (f" (+{len(sets) - 3} more sets)" if len(sets) > 3 else ""))
                       + (" per input" if len(sets) == 1 else ""))
            out.append(f"  bias {_clip(L.bias.tolist(), None if full else 24)}")
            out.append(f"  weights {W.shape[0]} x {W.shape[1]}, {int((W != 0).sum())} nonzero"
                       + ("" if lane_ok else "; LANES OF A PORT DISAGREE (first lane shown)"))
            rows = W.shape[0] if full else min(limit_rows, W.shape[0])
            for r in range(rows):
                out.append(f"    row {r}: " + _clip(W[r].tolist(), None if full else limit_cols))
            if rows < W.shape[0]:
                out.append(f"    ... {W.shape[0] - rows} more rows")
            if L.dead:
                out.append(f"  units nothing reads: {len(L.dead)} (bias {[b for b, _ in L.dead][:8]})")
        if self.unattached:
            out.append(f"unattributed lights (start values): {_clip(sorted(self.unattached), 16)}")
        for k, v in self.checks.items():
            out.append(f"check {k}: {v}")
        for f in self.findings:
            out.append(f"FINDING: {f}")
        for item in self.not_checked:
            out.append(f"NOT CHECKED: {item}")
        return "\n".join(out)


def _clip(values: list, n: int | None) -> str:
    if n is None or len(values) <= n:
        return str(values)
    return str(values[:n])[:-1] + f", ... {len(values) - n} more]"


# --- expected netlist of a net --------------------------------------------------------------------------------------
def _lane_order(limits: list[int], fx_rows: np.ndarray):
    """Lane order of one port: ascending limit, ties by their weight vector (identical lanes are interchangeable)."""
    keys = [(limits[k], tuple(int(v) for v in fx_rows[k])) for k in range(len(limits))]
    return sorted(range(len(keys)), key=keys.__getitem__)


def expected_netlist(net) -> Netlist:
    """The netlist a `kit.MLP` implies, from its arithmetic alone: lane k of an input unit with threshold t is a gate
    of limit -t; a signed first layer has two ports per input (-x with the weights w, x with -w); a hidden unit that no
    later weight reads is a dead unit (its row is kept when nonzero); inputs with all-zero weights and unreadable rows
    are dropped (module docstring)."""
    Ws = [np.asarray(W, dtype=np.int64) for W in net.weights]
    bs = [np.asarray(b, dtype=np.int64) for b in net.biases]
    L = len(Ws)
    layers: list[Layer] = []
    unattached: list[int] = []
    cols_live: list[list[int]] = [[i for i in range(W.shape[1]) if W[:, i].any()] for W in Ws]
    for l, W in enumerate(Ws):
        last = l == L - 1
        thr = np.asarray(net.thresholds[l], dtype=np.int64)
        limits_k = [-int(t) for t in thr]
        cols = cols_live[l]
        if last:
            first_nz = next((j for j in range(W.shape[0]) if W[j].any()), W.shape[0])
            keep = list(range(first_nz if L > 1 else 0, W.shape[0]))
            if L > 1:
                unattached += [int(bs[l][j]) for j in range(first_nz)]
            dead_rows: list[int] = []
        else:
            live = set(cols_live[l + 1])
            keep = [j for j in range(W.shape[0]) if j in live]
            dead_rows = [j for j in range(W.shape[0]) if j not in live]
            for j in keep:
                if not W[j].any():
                    unattached.append(int(bs[l][j]))
            for j in dead_rows:
                if not W[j].any():
                    unattached += [int(bs[l][j]), -int(bs[l][j])]
        port_cols: list[tuple[int, int]] = []              # (column of W, sign)
        for i in cols:
            for sign in ((1, -1) if l == 0 and net.signed_input else (1,)):
                port_cols.append((i, sign))
        limits, fxs = [], []
        for i, sign in port_cols:
            rows = np.array([[sign * int(W[j, i]) for j in keep]] * len(limits_k), dtype=np.int64).reshape(len(limits_k), len(keep))
            order = _lane_order(limits_k, rows)
            limits.append([limits_k[k] for k in order])
            fxs.append(rows[order])
        dead = []
        for j in dead_rows:
            if not W[j].any():
                continue
            lanes = []
            for i, sign in port_cols:
                row = np.array([sign * int(W[j, i])] * len(limits_k), dtype=np.int64).reshape(len(limits_k), 1)
                order = _lane_order(limits_k, row)
                lanes.append(tuple(int(row[k, 0]) for k in order))
            dead.append((int(bs[l][j]), tuple(lanes)))
        layer = Layer(l, list(range(len(port_cols))), limits, fxs, bs[l][keep].astype(np.int64), list(range(len(keep))),
                      sorted(dead), last)
        layer.col_ids = [i for i, _ in port_cols]
        layer.row_ids = keep
        layer.signs = [s for _, s in port_cols]           # type: ignore[attr-defined]
        layers.append(layer)
    return Netlist(layers, sorted(unattached), source="net")


def diff_netlists(got: Netlist, want: Netlist, max_diffs: int = 200) -> list[Diff]:
    """Differences of `got` (from a map) against `want` (from a net). Row and column numbers are `want`'s own."""
    out: list[Diff] = []

    def add(**kw):
        if len(out) < max_diffs:
            out.append(Diff(**kw))

    if len(got.layers) != len(want.layers):
        add(what="layers", layer=0, row=None, col=None, lane=None, got=len(got.layers), want=len(want.layers),
            text=f"layer count: map {len(got.layers)}, net {len(want.layers)}")
        return out
    for g, w in zip(got.layers, want.layers):
        l = w.index
        if len(g.ports) != len(w.ports):
            add(what="ports", layer=l, row=None, col=None, lane=None, got=len(g.ports), want=len(w.ports),
                text=f"layer {l}: map has {len(g.ports)} gate ports, net implies {len(w.ports)}")
            continue
        if g.n_rows != w.n_rows:
            add(what="rows", layer=l, row=None, col=None, lane=None, got=g.n_rows, want=w.n_rows,
                text=f"layer {l}: map has {g.n_rows} rows, net implies {w.n_rows}")
            continue
        rid = lambda r: w.row_ids[r] if w.row_ids is not None else r      # noqa: E731
        sg = getattr(w, "signs", None)

        def where(p):
            col = w.col_ids[p] if w.col_ids is not None else p
            port = "" if sg is None or not (l == 0 and len(set(sg)) > 1) else (" (port -x)" if sg[p] > 0 else " (port x)")
            return col, port

        for r in range(g.n_rows):
            if int(g.bias[r]) != int(w.bias[r]):
                add(what="bias", layer=l, row=rid(r), col=None, lane=None, got=int(g.bias[r]), want=int(w.bias[r]),
                    text=f"layer {l}, row {rid(r)}: bias map {int(g.bias[r])}, net {int(w.bias[r])}")
        for p in range(len(w.ports)):
            col, port = where(p)
            if len(g.limits[p]) != len(w.limits[p]):
                add(what="lanes", layer=l, row=None, col=col, lane=None, got=len(g.limits[p]), want=len(w.limits[p]),
                    text=f"layer {l}, column {col}{port}: map has {len(g.limits[p])} lanes, net {len(w.limits[p])}")
                continue
            for k, (a, b) in enumerate(zip(g.limits[p], w.limits[p])):
                if a != b:
                    add(what="threshold", layer=l, row=None, col=col, lane=k, got=-a, want=-b,
                        text=f"layer {l}, column {col}{port}, lane {k}: threshold map {-a}, net {-b} "
                             f"(gate limit map {a}, net {b})")
            for k in range(len(w.limits[p])):
                for r in range(g.n_rows):
                    a, b = int(g.fx[p][k, r]), int(w.fx[p][k, r])
                    if a != b:
                        add(what="weight", layer=l, row=rid(r), col=col, lane=k, got=a, want=b,
                            text=f"layer {l}, row {rid(r)}, column {col}{port}, lane {k}: weight map {a:+d}, net {b:+d}")
                    a = int(g.fxl[p][k, r]) if g.fxl is not None else b
                    if a != b:
                        add(what="twin weight", layer=l, row=rid(r), col=col, lane=k, got=a, want=b,
                            text=f"layer {l}, row {rid(r)}, column {col}{port}, lane {k}: weight on the unit's twin light "
                                 f"map {a:+d}, net {b:+d}")
        if sorted(g.dead) != sorted(w.dead):
            add(what="dead", layer=l, row=None, col=None, lane=None, got=len(g.dead), want=len(w.dead),
                text=f"layer {l}: units nothing reads differ (map {len(g.dead)}, net {len(w.dead)}: bias/weights)")
    if sorted(got.unattached) != sorted(want.unattached):
        add(what="unattributed", layer=0, row=None, col=None, lane=None, got=sorted(got.unattached),
            want=sorted(want.unattached),
            text=f"unattributed light start values: map {sorted(got.unattached)[:8]}, net {sorted(want.unattached)[:8]}")
    return out


# --- extraction -----------------------------------------------------------------------------------------------------
def _chain(tm: TextMap, barrier: int) -> tuple[list[int], int]:
    """The lane's sectors in x order and the index of the barrier in it. A lane is a path: every sector has at most
    one neighbour on each side."""
    nb = sorted(tm.nbr.get(barrier, ()), key=lambda s: tm.x0[s])
    if len(nb) != 2 or not tm.x0[nb[0]] < tm.x0[barrier] < tm.x0[nb[1]]:
        raise ExtractError(f"start barrier sector {barrier} (tag {tm.tag[barrier]}) does not sit between two sectors "
                           "along x")
    sides = []
    for first in nb:
        path, prev, cur = [], barrier, first
        while True:
            path.append(cur)
            nxt = [s for s in tm.nbr.get(cur, ()) if s != prev]
            if not nxt:
                break
            if len(nxt) > 1:
                raise ExtractError(f"sector {cur} of the lane at barrier {barrier} has {len(nxt) + 1} neighbours: "
                                   "a lane is a path of sectors")
            prev, cur = cur, nxt[0]
            if len(path) > 100000:
                raise ExtractError("lane chain does not end")
        sides.append(path)
    west, east = sides
    return west[::-1] + [barrier] + east, len(west)


def _effect(line: tuple) -> tuple[str, int, int] | None:
    """(kind, tag, signed amount) of an effect line, None for any other special."""
    sp, a0, a1, a2 = line[2], line[3], line[4], line[5]
    if sp not in EFFECTS:
        return None
    kind, sign = EFFECTS[sp]
    return kind, a0, sign * (a1 if kind == "light" else a2)


@dataclass
class _Lane:
    doll: Doll
    findings: list[str]


def _read_lane(tm: TextMap, barrier: int, layer: int, seen: dict) -> _Lane | None:
    chain, ib = _chain(tm, barrier)
    if ib != 2:
        raise ExtractError(f"lane at barrier {barrier}: {ib} sectors west of the start barrier, expected 2 (home, home2)")
    if len(chain) < ib + 4:
        return None
    pad, gate = chain[ib + 1], chain[ib + 2]
    if tm.tag[gate] in (0, CONV_TAG):
        return None                                   # the layer's clock lane: no gate
    findings: list[str] = []
    f = tm.floor[pad]
    walk = [chain[0], chain[1], pad]
    if any(tm.floor[s] != f for s in walk):
        findings.append(f"lane at barrier {barrier} (layer {layer}): walkway floors differ before the gate "
                        f"{[tm.floor[s] for s in walk]}")
    if tm.ceil[gate] != CEIL:
        raise ExtractError(f"gate sector {gate} (tag {tm.tag[gate]}) has ceiling {tm.ceil[gate]}, not {CEIL}: only floor "
                           "gates carry an MLP's ports")
    after = chain[ib + 3:ib + 5]
    if any(tm.floor[s] != f for s in after[1:]):
        findings.append(f"lane at barrier {barrier} (layer {layer}): walkway floors differ after the gate")
    lines: list[tuple[str, int, int]] = []
    wave = wave_i = None
    n_lines = 0
    for a, b in zip(chain[ib + 2:], chain[ib + 3:]):
        found = tm.lines_between(a, b)
        n_lines += sum(1 for ln in found if _effect(ln) is not None)
        if wave is None and b != chain[ib + 3] and tm.tag[b] not in (0, CONV_TAG):
            wave, wave_i = tm.tag[b], chain.index(b)
        eff = {e for e in map(_effect, found) if e is not None}
        others = [ln[2] for ln in found if ln[2] and _effect(ln) is None]
        if others:
            findings.append(f"lane at barrier {barrier} (layer {layer}): unexpected special(s) {others} inside the lane")
        for ln in found:
            if _effect(ln) is not None:
                if not ln[8] & PLAYERCROSS:
                    findings.append(f"lane at barrier {barrier}: effect line without playercross")
                if _effect(ln)[0] in ("plane", "cplane") and ln[4] < MOVE_SPEED_MIN:
                    findings.append(f"lane at barrier {barrier}: plane mover speed {ln[4]} < {MOVE_SPEED_MIN}")
        if len(eff) > 1:
            findings.append(f"lane at barrier {barrier}: two different effect lines between one pair of sectors")
        lines.extend(sorted(eff)[:1] if len(eff) == 1 else [])
    # presence checks of the lane's mechanics
    for a, b in zip(chain, chain[1:]):
        found = tm.lines_between(a, b)
        if not found or any(ln[8] & BLOCKING or ln[1] < 0 for ln in found):
            findings.append(f"lane at barrier {barrier} (layer {layer}): impassable link between sectors {a} and {b}")
            break
    if tm.things_in(chain[0], PLAYER1_START) < 1:
        findings.append(f"lane at barrier {barrier} (layer {layer}): no voodoo doll in its home sector")
    home = [ln for ln in tm.lines_between(chain[0], chain[1]) if ln[2] == EXEC_ALWAYS and ln[3] == 4]
    if not home:
        findings.append(f"lane at barrier {barrier} (layer {layer}): no self-tagging line between home and home2")
    doll = Doll(layer, tm.tag[gate], f + STEP_UP, lines, chain, wave, tm.tag[chain[ib + 3]],
                home[0][5] if home else None, wave_i, n_lines)
    seen.setdefault("gates", {}).setdefault(tm.tag[gate], set()).add(tm.floor[gate])
    return _Lane(doll, findings)


def extract(wad) -> Netlist:
    """Recover the netlist of a compiled one-map MLP from its WAD alone (module docstring)."""
    return extract_lumps(read_lumps(wad), str(wad))


def extract_lumps(lumps: dict[str, bytes], source: str = "") -> Netlist:
    """`extract` on the lumps of a map already read (TEXTMAP, BEHAVIOR)."""
    tm = TextMap(lumps["TEXTMAP"].decode())
    findings: list[str] = []
    released = {ln[3] for ln in tm.lines if ln[2] == LOWER_TO_LOWEST}
    frozen = {ln[3] for ln in tm.lines if ln[2] == FLOOR_MOVE_TO}
    barriers = [s for s in range(tm.n_sectors) if s in tm.nbr and (tm.tag[s] == START_TAG or STAGE_TAG0 < tm.tag[s] < STAGE_TAG0 + 4000)]
    if not barriers:
        raise ExtractError("no start barrier (sector tag 2 with neighbours): not a compiled one-map net")
    tags = sorted({tm.tag[s] for s in barriers})
    n_layers = len(tags)
    want = [START_TAG] + [STAGE_TAG0 + s for s in range(1, n_layers)]
    if tags != want:
        raise ExtractError(f"start barrier tags {tags}, expected {want} (layer 0 = {START_TAG}, layer s = {STAGE_TAG0} + s)")
    layer_of = {t: i for i, t in enumerate(tags)}
    seen: dict = {}
    dolls: list[list[Doll]] = [[] for _ in range(n_layers)]
    for b in barriers:
        lane = _read_lane(tm, b, layer_of[tm.tag[b]], seen)
        if lane is not None:
            dolls[lane.doll.layer].append(lane.doll)
            findings += lane.findings
    if any(not d for d in dolls):
        raise ExtractError(f"layer(s) {[i for i, d in enumerate(dolls) if not d]} have no doll lane")
    lights = {tm.tag[s]: tm.light[s] for s in range(tm.n_sectors) if tm.isolated[s] and tm.tag[s] > 0}
    if any(tm.special[s] for s in range(tm.n_sectors) if tm.isolated[s] and tm.tag[s] > 0):
        findings.append("an accumulator light sector has a sector special (it would change its light level)")
    nl = _assemble(tm, dolls, lights, seen, findings, released, frozen, lumps, source)
    plane_contract(nl)
    return nl


def plane_contract(nl: Netlist) -> None:
    """Run the plane contract (planes.plane_errors: start bounds and the plane-move bound over all inputs) on the recovered
    net and record the result in `checks`; a violation is a finding that starts with "plane contract"."""
    try:
        mlp = nl.to_mlp()
    except ValueError as e:                    # ExtractError and RangeError: the netlist is not a valid MLP
        nl.not_checked.append(f"plane contract (the netlist is not a valid MLP: {str(e)[:90]})")
        return
    errs = mlp.plane_errors()
    nl.checks["plane contract"] = "ok" if not errs else f"{len(errs)} violation(s)"
    nl.findings += [f"plane contract: {e}" for e in errs]


# --- ACS (BEHAVIOR) --------------------------------------------------------------------------------------------------
# The compiler's scripts are straight-line specials calls (reset, release, copy), run here by a tiny interpreter of the
# compact ("ACSe") bytecode acc writes. Opcode numbers are acc's pcode.h enumeration; any other opcode raises.
(OP_NOP, OP_TERMINATE, OP_PUSHNUMBER, OP_LSPEC1, OP_LSPEC5, OP_LT, OP_GT, OP_ASSIGNSCRIPTVAR, OP_PUSHSCRIPTVAR, OP_GOTO,
 OP_IFGOTO, OP_DROP, OP_UNARYMINUS, OP_IFNOTGOTO, OP_PUSHBYTE, OP_LSPEC1DIRECTB, OP_LSPEC5DIRECTB, OP_PUSHBYTES,
 OP_PUSH2BYTES, OP_PUSH5BYTES, OP_EQ, OP_NE, OP_LE, OP_GE) = (0, 1, 3, 4, 8, 21, 22, 25, 28, 52, 53, 54, 78, 79, 167, 168, 172, 175,
                                                            176, 179, 19, 20, 23, 24)
OP_GETSECTORLIGHTLEVEL = 281     # two-byte form: 240, 281 - 240 - 0
OP_GETSECTORFLOORZ, OP_RSHIFT, OP_ASSIGNGLOBALVAR = 261, 77, 181
THING_CHANGE_TID = 176


def acs_scripts(beh: bytes) -> dict[int, tuple[int, int]]:
    """Script number -> (code offset, argument count) from the ACSe script-pointer chunk."""
    if beh[:4] != b"ACS\0" or len(beh) < 16:
        raise ExtractError("BEHAVIOR is not the ACS format acc writes")
    dirofs = struct.unpack("<i", beh[4:8])[0]
    if beh[dirofs - 4:dirofs] != b"ACSe":
        raise ExtractError("BEHAVIOR is not the compact ACSe format acc writes (no hidden ACSe header)")
    pos, end = struct.unpack("<i", beh[dirofs - 8:dirofs - 4])[0], dirofs - 8
    while pos + 8 <= end:
        cid, size = beh[pos:pos + 4], struct.unpack("<i", beh[pos + 4:pos + 8])[0]
        if cid == b"SPTR":
            body = beh[pos + 8:pos + 8 + size]
            return {n: (off, argc) for n, _t, argc, off in (struct.unpack("<HBBi", body[i:i + 8])
                                                              for i in range(0, len(body) - 7, 8))}
        pos += 8 + size
    raise ExtractError("BEHAVIOR has no script pointer chunk")


def run_script(beh: bytes, offset: int, *args, **kw) -> list[tuple[int, tuple]]:
    """_run_script, with every malformed-code failure (stack underflow, a jump or read outside the lump) as ExtractError."""
    try:
        return _run_script(beh, offset, *args, **kw)
    except (IndexError, struct.error) as e:
        raise ExtractError(f"malformed ACS code from offset {offset}: {type(e).__name__}") from None


def _run_script(beh: bytes, offset: int, args=(), lights=None, max_steps: int = 2_000_000, floors=None,
                globals_out: dict | None = None) -> list[tuple[int, tuple]]:
    """Run one ACS script and return its line-special calls [(special, args)]. `lights(tag)` supplies
    GetSectorLightLevel, `floors(tag)` GetSectorFloorZ (16.16); global-variable writes go to `globals_out`."""
    locs: dict[int, int] = dict(enumerate(args))
    stack: list[int] = []
    calls: list[tuple[int, tuple]] = []
    pc = offset
    for _ in range(max_steps):
        op = beh[pc]
        pc += 1
        if op >= 240:
            op = 240 + ((op - 240) << 8) + beh[pc]
            pc += 1
        if op == OP_NOP:
            continue
        if op == OP_TERMINATE:
            if stack:
                raise ExtractError(f"ACS script leaves {len(stack)} value(s) on the stack at offset {pc - 1}")
            return calls
        if op == OP_PUSHNUMBER:
            stack.append(struct.unpack("<i", beh[pc:pc + 4])[0])
            pc += 4
        elif op == OP_PUSHBYTE:
            stack.append(beh[pc])
            pc += 1
        elif op == OP_PUSHBYTES:
            n = beh[pc]
            stack.extend(beh[pc + 1:pc + 1 + n])
            pc += 1 + n
        elif OP_PUSH2BYTES <= op <= OP_PUSH5BYTES:
            n = op - OP_PUSH2BYTES + 2
            stack.extend(beh[pc:pc + n])
            pc += n
        elif OP_LSPEC1 <= op <= OP_LSPEC5:
            k = op - OP_LSPEC1 + 1
            sp = beh[pc]
            pc += 1
            if len(stack) < k:
                raise ExtractError(f"ACS stack underflow at offset {pc - 2}")
            vals = tuple(stack[-k:])
            del stack[-k:]
            calls.append((sp, vals))
        elif OP_LSPEC1DIRECTB <= op <= OP_LSPEC5DIRECTB:
            k = op - OP_LSPEC1DIRECTB + 1
            calls.append((beh[pc], tuple(beh[pc + 1:pc + 1 + k])))
            pc += 1 + k
        elif op == OP_ASSIGNSCRIPTVAR:
            locs[beh[pc]] = stack.pop()
            pc += 1
        elif op == OP_PUSHSCRIPTVAR:
            stack.append(locs.get(beh[pc], 0))
            pc += 1
        elif op in (OP_LT, OP_GT, OP_EQ, OP_NE, OP_LE, OP_GE):
            if len(stack) < 2:
                raise ExtractError(f"ACS stack underflow at offset {pc - 1}")
            b, a = stack.pop(), stack.pop()
            stack.append(int({OP_LT: a < b, OP_GT: a > b, OP_EQ: a == b, OP_NE: a != b, OP_LE: a <= b, OP_GE: a >= b}[op]))
        elif op == OP_UNARYMINUS:
            stack.append(-stack.pop())
        elif op == OP_DROP:
            stack.pop()
        elif op == OP_GOTO:
            pc = struct.unpack("<i", beh[pc:pc + 4])[0]
        elif op in (OP_IFGOTO, OP_IFNOTGOTO):
            v = stack.pop()
            target = struct.unpack("<i", beh[pc:pc + 4])[0]
            pc = target if (v != 0) == (op == OP_IFGOTO) else pc + 4
        elif op == OP_RSHIFT:
            b, a = stack.pop(), stack.pop()
            stack.append(a >> b)
        elif op == OP_ASSIGNGLOBALVAR:
            if globals_out is None:
                raise ExtractError("script writes a global variable but no sink was supplied")
            globals_out[beh[pc]] = stack.pop()
            pc += 1
        elif op == OP_GETSECTORFLOORZ:
            if floors is None:
                raise ExtractError("script reads a floor height but no values were supplied")
            y, x = stack.pop(), stack.pop()
            if x or y:
                raise ExtractError("GetSectorFloorZ called with a nonzero position")
            stack.append(floors(stack.pop()))
        elif op == OP_GETSECTORLIGHTLEVEL:
            if lights is None:
                raise ExtractError("script reads a light level but no values were supplied")
            stack.append(lights(stack.pop()))
        else:
            raise ExtractError(f"ACS opcode {op} at offset {pc - 1} is outside the straight-line subset this reader "
                               "supports")
    raise ExtractError("ACS script did not end")


def copy_tables(beh: bytes, light_tags: set[int]) -> tuple[dict[int, int], dict[int, int]]:
    """Opt 3's copy scripts (script number 30 + layer): {light tag: plane tag} and {layer: script number}. Each script
    runs on two sets of test light levels; every plane write must be +-(one light), found from both runs."""
    scripts = acs_scripts(beh)
    table: dict[int, int] = {}
    layers: dict[int, int] = {}
    for num, (off, argc) in sorted(scripts.items()):
        if not 30 <= num < 100:
            continue
        runs = []
        for sign in (1, -1):
            read: dict[int, int] = {}
            runs.append((read, run_script(beh, off, lights=lambda t, read=read, sign=sign:
                                          read.setdefault(t, sign * (1000 + 7 * len(read) + (t % 5)))), sign))
        for ci, (sp, a) in enumerate(runs[0][1]):
            if sp not in (FLOOR_MOVE_TO, CEIL_MOVE_TO):
                raise ExtractError(f"copy script {num} calls special {sp}, expected Floor_MoveToValue")
            heights = []
            for read, calls, _ in runs:
                tag, _speed, h, neg = calls[ci][1]
                heights.append(-h if neg else h)
            cands = [(t, s) for t in runs[0][0] for s in (1, -1)
                     if all(s * r[t] == h for (r, _c, _g), h in zip(runs, heights))]
            if len(cands) != 1:
                raise ExtractError(f"copy script {num}: plane {a[0]} gets a value that is not one light level")
            t, s = cands[0]
            if t not in light_tags or s != -1:
                raise ExtractError(f"copy script {num}: plane {a[0]} is set to {'+' if s > 0 else '-'}light {t}; the MLP "
                                   "map copies minus a light level")
            table[t] = a[0]
        layers[num - 30] = num
    return table, layers


# --- mechanics: every line class, the scripts, the lane geometry ---------------------------------------------------------
SPEED_RANGE = (150000, 262144)   # mover speeds that reach any port height in one tic (engine facts: 150,000 and 262,144 do,
                                 # 300,000 and 500,000 pass through wrong heights)
CLOSED = 13000                   # freeze lines raise a walkway to this height (tile.CLOSED)
DOLL_H = 56                      # voodoo doll height (tile.DOLL_H)
CONVEYOR_ARGS = (1, 0, 1, 255, 128)   # Scroll_Floor(tag 1, 0, carry, xmove 255, ymove 128); four of them (tile.CONVEYOR)
CONVEYOR_STACK = 4
TELEPORT_DEST = 14               # thing type of a teleport destination
THING_FLAGS = {"single", "skill1", "skill2", "skill3", "skill4", "skill5"}
TRIGGER = {"twosided", "playercross", "repeatspecial"}
NOT_CHECKED = [
    "vertex coordinates, sidedefs and textures (lane lengths, wave spacing and clock positions: timing)",
    "which clock lines a clock doll crosses first, and the positions of clock-chunk teleport destinations (timing)",
    "floor, ceiling and light of sectors that are not gates, barriers, lane ground or accumulators",
    "thing angles and the coop and dm thing flags",
    "the names behind the readout slots (the map holds none)",
    "whether the engine executes the map as read",
]


def _speed_ok(v: int) -> bool:
    return SPEED_RANGE[0] <= v <= SPEED_RANGE[1]


class _Problems:
    """Findings grouped by message with a count and the first place."""

    def __init__(self):
        self.items: dict[str, list] = {}

    def add(self, msg: str, where=None):
        rec = self.items.setdefault(msg, [0, where])
        rec[0] += 1

    def lines(self) -> list[str]:
        return [f"{msg} ({n} time{'s' if n > 1 else ''}{'' if w is None else f', first at {w}'})"
                for msg, (n, w) in self.items.items()]


def _check_lines(tm: TextMap, lights, port_tags, waves, start_tags, freeze_tags, copy_nums, n_lane_lines) -> tuple[list[str], dict]:
    """Every linedef with a special: its class, every argument, its trigger flags. Plain two-sided lines: no flags."""
    pr = _Problems()
    count: Counter = Counter()
    dest_count = Counter(t[3] for t in tm.things if t[2] == TELEPORT_DEST)
    for i, (a, b, sp, a0, a1, a2, a3, a4, fl) in enumerate(tm.lines):
        names = tm.flag_names(fl)
        if sp == 0:
            if b >= 0 and names != {"twosided"}:
                pr.add(f"two-sided line with flags {sorted(names - {'twosided'})}", f"linedef {i}")
            continue
        prob = None
        if sp in (LIGHT_RAISE, LIGHT_LOWER):
            name = "weight (light)"
            prob = None if a0 in lights and a1 >= 0 and not (a2 or a3 or a4) else "light line: tag or arguments"
        elif sp in (FLOOR_RAISE, FLOOR_LOWER):
            name = "weight (plane)"
            prob = None if a0 in port_tags and _speed_ok(a1) and a2 >= 0 and not (a3 or a4) \
                else "plane line: tag, mover speed or arguments"
        elif sp in (CEIL_RAISE, CEIL_LOWER):
            name, prob = "ceiling effect", "ceiling effect line: not an MLP map"
        elif sp == LOWER_TO_LOWEST:
            name = "release"
            prob = None if (a0 in waves or a0 in start_tags) and _speed_ok(a1) and not (a2 or a3 or a4) \
                else "barrier release line: tag, mover speed or arguments"
        elif sp == FLOOR_MOVE_TO:
            name = "freeze"
            prob = None if a0 in freeze_tags and _speed_ok(a1) and a2 == CLOSED and not (a3 or a4) \
                else "freeze line: tag, mover speed, target height or arguments"
        elif sp == EXEC_ALWAYS:
            name = "script start"
            ok = (a0 == 4 and a1 == 0 and not (a3 or a4)) or (a0 in copy_nums and not (a1 or a2 or a3 or a4))
            prob = None if ok else f"ACS_ExecuteAlways line: script {a0} or arguments"
        elif sp == TELEPORT:
            name = "teleport"
            prob = None if dest_count.get(a0) == 1 and not (a1 or a2 or a3 or a4) \
                else "teleport line: no single destination thing with that id, or arguments"
        elif sp == SCROLL_FLOOR:
            name = "conveyor"
            prob = None if (a0, a1, a2, a3, a4) == CONVEYOR_ARGS else "Scroll_Floor arguments differ from the compiler's"
            if prob is None:
                count[name] += 1
            else:
                pr.add(prob, f"linedef {i}")
            continue
        else:
            name, prob = f"special {sp}", f"special {sp} is not read"
        if prob is None and (b < 0 or names != TRIGGER):
            prob = f"{name} line: trigger flags {sorted(names)} (expected playercross and repeatspecial on a two-sided line)"
        if prob is not None:
            pr.add(prob, f"linedef {i}")
        else:
            count[name] += 1
    out = pr.lines()
    if count["conveyor"] < CONVEYOR_STACK:
        out.append(f"{count['conveyor']} conveyor scroller line(s), the compiler writes {CONVEYOR_STACK}")
    n_w = count["weight (light)"] + count["weight (plane)"]
    if n_w != n_lane_lines:
        out.append(f"{n_w} weight lines in the map, {n_lane_lines} on doll lanes: the others are outside any lane")
    return out, dict(count)


def _ground_rules(tm: TextMap, ch: list[int], barriers: set[int], skip: set[int], pr: _Problems) -> None:
    """Steps and headroom between consecutive open sectors of a chain, and the barriers of the chain: closed they block,
    opened (Floor_LowerToLowest: the floor goes to the lower neighbour's) they leave a walkable step."""
    for i, sec in enumerate(ch):
        if i not in skip and tm.ceil[sec] - tm.floor[sec] < DOLL_H:
            pr.add(f"headroom below {DOLL_H} in a chain sector", f"sector {sec}")
    for i in range(len(ch) - 1):
        if i in skip or i + 1 in skip:
            continue
        a, b = ch[i], ch[i + 1]
        if tm.floor[b] - tm.floor[a] > STEP_UP:
            pr.add(f"step up of more than {STEP_UP} along a chain", f"sectors {a}, {b}")
        if min(tm.ceil[a], tm.ceil[b]) - max(tm.floor[a], tm.floor[b]) < DOLL_H:
            pr.add(f"headroom below {DOLL_H} along a chain", f"sectors {a}, {b}")
    for bi in barriers:
        if not 0 < bi < len(ch) - 1:
            pr.add("barrier at the end of a chain", f"sector {ch[bi]}")
            continue
        w, e, bs = tm.floor[ch[bi - 1]], tm.floor[ch[bi + 1]], ch[bi]
        low = min(w, e)
        if tm.floor[bs] - max(w, e) <= STEP_UP:
            pr.add("barrier that does not block when closed", f"sector {bs}")
        if e - low > STEP_UP or tm.ceil[bs] - low < DOLL_H:
            pr.add("barrier that does not open to a walkable step", f"sector {bs}")


def _check_lanes(tm: TextMap, all_dolls: list[Doll], start_tags: set[int]) -> list[str]:
    """Doll lanes and the clock's chains: no sector special, the conveyor tag on walkable ground, steps and headroom,
    barriers; every sector boundary closed; the player and doll things."""
    pr = _Problems()
    for d in all_dolls:
        ch, ib = d.sectors, 2
        for i, sec in enumerate(ch):
            if tm.special[sec]:
                pr.add("lane sector with a sector special", f"sector {sec}")
            if i not in (ib, ib + 2, ib + 3, d.wave_i) and tm.tag[sec] != CONV_TAG:
                pr.add(f"lane ground sector without the conveyor tag {CONV_TAG}", f"sector {sec}")
        if d.wave_i is None:
            pr.add("lane without a wave barrier", f"sector {ch[0]}")
            continue
        _ground_rules(tm, ch, {ib, d.wave_i}, {ib, ib + 2, d.wave_i}, pr)
    compute = {s for d in all_dolls for s in d.sectors}
    seen: set[int] = set()
    for s0 in range(tm.n_sectors):
        if s0 in seen or s0 in compute or s0 not in tm.nbr:
            continue
        comp, todo = [], [s0]
        seen.add(s0)
        while todo:
            x = todo.pop()
            comp.append(x)
            for y in tm.nbr[x]:
                if y not in seen and y not in compute:
                    seen.add(y)
                    todo.append(y)
        comp.sort(key=lambda t: tm.x0[t])
        bar = {i for i, sec in enumerate(comp) if tm.tag[sec] in start_tags}
        for a, b in zip(comp, comp[1:]):
            if b not in tm.nbr[a]:
                pr.add("clock chain that is not a path of adjacent sectors", f"sector {a}")
        for i, sec in enumerate(comp):
            if tm.special[sec]:
                pr.add("clock sector with a sector special", f"sector {sec}")
            if i not in bar and tm.tag[sec] != CONV_TAG:
                pr.add(f"clock ground sector without the conveyor tag {CONV_TAG}", f"sector {sec}")
        _ground_rules(tm, comp, bar, bar, pr)
    if tm.open_sectors:
        pr.add("sector whose boundary lines do not close", f"sector {tm.open_sectors[0]} ({len(tm.open_sectors)} sectors)")
    # things: the dolls, the player and the teleport destinations
    homes = {a2 for _, b, sp, a0, _, a2, *_ in tm.lines if sp == EXEC_ALWAYS and a0 == 4 and b >= 0}
    starts = [t for t in tm.things if t[2] == PLAYER1_START]
    if len(starts) != len(homes) + 1:
        pr.add(f"{len(starts)} player-1 start things, expected one per doll ({len(homes)}) and the player")
    inside = ~tm.isolated
    for t in tm.things:
        if t[2] in (PLAYER1_START, TELEPORT_DEST) and not THING_FLAGS <= t[4]:
            pr.add(f"thing of type {t[2]} that does not spawn in single player (flags {sorted(t[4])})")
        if t[2] == TELEPORT_DEST and not (inside & (tm.x0 <= t[0]) & (t[0] <= tm.x1) & (tm.y0 <= t[1]) & (t[1] <= tm.y1)).any():
            pr.add("teleport destination outside every lane sector", f"thing id {t[3]}")
    return pr.lines()


def _check_scripts(beh: bytes, tm: TextMap, internal: dict[int, int], barriers: dict[int, int],
                   copy_layers: dict[int, int], all_lights: dict[int, int]) -> tuple[list[str], list[str], dict]:
    """Findings, NOT CHECKED entries and check notes from the ACS: scripts 1, 2, 3, 4, 7, copy scripts and readout pages."""
    out: list[str] = []
    nc: list[str] = []
    chk: dict = {}
    scripts = acs_scripts(beh)
    for num in (1, 2, 3, 4, 7):
        if num not in scripts:
            out.append(f"ACS script {num} is missing")
    # helpers: the input writer, the tid script and the layer-0 release
    def run(num, args=()):
        return run_script(beh, scripts[num][0], args) if num in scripts else []
    for num, special in ((1, FLOOR_MOVE_TO), (2, CEIL_MOVE_TO)):
        calls = run(num, (11, 22, 1))
        if len(calls) != 1 or calls[0][0] != special or calls[0][1][0] != 11 or calls[0][1][2:] != (22, 1) \
                or not _speed_ok(calls[0][1][1]):
            out.append(f"ACS script {num} does not write its tag, height and sign argument with a plane mover")
    calls = run(4, (9,))
    if calls != [(THING_CHANGE_TID, (0, 9))]:
        out.append("ACS script 4 does not give the activator the TID it is called with")
    calls = run(7)
    if len(calls) != 1 or calls[0][0] != LOWER_TO_LOWEST or calls[0][1][0] != START_TAG or not _speed_ok(calls[0][1][1]) \
            or any(calls[0][1][2:]):
        out.append("ACS script 7 does not release layer 0's start barrier")
    # the reset script
    calls = run(3)
    tids: dict[int, int] = {}
    for a, b, sp, a0, a1, a2, *_ in tm.lines:
        if sp == EXEC_ALWAYS and a0 == 4 and b >= 0:
            tids.setdefault(a2, a if tm.x0[a] < tm.x0[b] else b)
    seen_l: Counter = Counter()
    seen_p: Counter = Counter()
    seen_t: Counter = Counter()
    dest_ids = Counter(t[3] for t in tm.things if t[2] == TELEPORT_DEST)
    for sp, a in calls:
        if sp == LIGHT_CHANGE_TO and len(a) == 2 and a[0] in all_lights:
            seen_l[a[0]] += 1
            if a[1] != all_lights[a[0]]:
                out.append(f"reset script restores light {a[0]} to {a[1]}, the map starts it at {all_lights[a[0]]}")
        elif sp == FLOOR_MOVE_TO and a[0] in internal or sp == FLOOR_MOVE_TO and a[0] in barriers:
            seen_p[a[0]] += 1
            val = -a[2] if a[3] else a[2]
            want = internal.get(a[0], barriers.get(a[0]))
            if val != want:
                out.append(f"reset script sets plane or barrier {a[0]} to {val}, the map has {want}")
            if not _speed_ok(a[1]) or a[3] not in (0, 1):
                out.append(f"reset script move of {a[0]}: mover speed {a[1]} or sign flag {a[3]}")
        elif sp == TELEPORT_OTHER and a[0] in tids:
            seen_t[a[0]] += 1
            home = tids[a[0]]
            ids = [t[3] for t in tm.things_at(home, TELEPORT_DEST)]
            if a[1] not in ids or dest_ids[a[1]] != 1 or a[2] != 0:
                out.append(f"reset script sends doll TID {a[0]} to {a[1]}: no single destination thing of that id in its "
                           "home sector")
        else:
            out.append(f"reset script call (special {sp}, arguments {a}) matches nothing the compiler writes")
    miss_l = [t for t in all_lights if seen_l[t] != 1]
    if miss_l:
        out.append(f"reset script does not restore {len(miss_l)} light(s) exactly once (first tag {miss_l[0]})")
    miss_p = [t for t in {**internal, **barriers} if seen_p[t] != 1]
    if miss_p:
        out.append(f"reset script does not reset {len(miss_p)} plane(s) or barrier(s) exactly once (first tag {miss_p[0]})")
    miss_t = [t for t in tids if seen_t[t] != 1]
    if miss_t:
        out.append(f"reset script does not send {len(miss_t)} doll(s) home exactly once (first TID {miss_t[0]})")
    chk["reset script (3)"] = (f"{len(calls)} calls: {len(seen_l)} lights, {len(seen_p)} planes and barriers, {len(seen_t)} dolls "
                               "sent to a destination in their home sector")
    # doll things
    bad_things = 0
    for tid, home in tids.items():
        dolls = tm.things_at(home, PLAYER1_START)
        if len(dolls) != 1 or not THING_FLAGS <= dolls[0][4]:
            bad_things += 1
    if bad_things:
        out.append(f"{bad_things} home sector(s) without exactly one voodoo doll thing that spawns in single player")
    # readout pages
    pages = sorted(n for n in scripts if 100 <= n < 200)
    try:
        lt, pl, order_ok = Counter(), Counter(), True
        for k, num in enumerate(pages):
            g: dict = {}
            run_script(beh, scripts[num][0], lights=lambda t: 1_000_000 + t, floors=lambda t: (2_000_000 + t) << 16,
                       globals_out=g)
            if sorted(g) != list(range(1, len(g) + 1)) or (k < len(pages) - 1 and len(g) != 60) or num != 100 + k:
                order_ok = False
            for v in g.values():
                if 1_000_000 <= v < 2_000_000:
                    lt[v - 1_000_000] += 1
                elif 2_000_000 <= v < 3_000_000:
                    pl[v - 2_000_000] += 1
                else:
                    order_ok = False
        if not order_ok or set(lt) != set(all_lights) or set(pl) != set(internal) or any(c != 1 for c in (*lt.values(), *pl.values())):
            out.append("readout pages do not read every light and every internal plane exactly once")
        chk["readout pages"] = f"{len(pages)} pages read {len(lt)} lights and {len(pl)} planes, each once"
    except ExtractError as e:
        out.append(f"readout pages (scripts 100 and up) cannot be read: {e}")
    other = sorted(n for n in scripts if n not in (1, 2, 3, 4, 7) and not 100 <= n < 200 and n not in copy_layers.values())
    if other:
        nc.append(f"ACS scripts {other} (motion script 8, readout guard 9, world scripts) are not read")
    return out, nc, chk


# --- assembling the netlist -----------------------------------------------------------------------------------------
def _assemble(tm: TextMap, dolls: list[list[Doll]], lights: dict[int, int], seen: dict, findings: list[str],
              released: set[int], frozen: set[int], lumps: dict[str, bytes], source: str) -> Netlist:
    n = len(dolls)
    gate_floor = {t: sorted(v) for t, v in seen["gates"].items()}
    beh = lumps.get("BEHAVIOR", b"")
    copy, copy_layers = copy_tables(beh, set(lights)) if beh else ({}, {})
    ports = [sorted({d.port for d in ds}) for ds in dolls]
    for t in range(n):
        for u in range(t + 1, n):
            both = set(ports[t]) & set(ports[u])
            if both:
                raise ExtractError(f"gate port tag {sorted(both)[0]} is read by layers {t} and {u}: an MLP layer reads "
                                   "only the previous layer's units")
    attached: set[int] = set()
    # final layer rows: lights written by the last layer, plus the lights above its first one that nothing writes
    final = n - 1
    written_final = sorted({tag for d in dolls[final] for k, tag, _ in d.lines if k == "light"})
    bad = [(k, tag) for d in dolls[final] for k, tag, _ in d.lines if k != "light" or tag not in lights]
    if bad:
        findings.append(f"output layer writes {bad[0][0]} {bad[0][1]}, which is not an output light")
    if not written_final:
        raise ExtractError("the last layer's dolls write no light (tile maps keep their weight lines in a shared zone and are not read)")
    final_tags = [t for t in sorted(lights) if t >= written_final[0]] if n > 1 else sorted(lights)
    layers: list[Layer] = []
    pair_votes: dict[int, Counter] = {}
    for s in range(n):
        last = s == final
        row_tags = final_tags if last else ports[s + 1]
        row_of = {t: r for r, t in enumerate(row_tags)}
        R = len(row_tags)
        by_port: dict[int, list[Doll]] = {}
        for d in dolls[s]:
            by_port.setdefault(d.port, []).append(d)
        lane_rows: dict[int, list[tuple[Doll, np.ndarray, np.ndarray]]] = {}
        dead: dict[tuple[int, int], dict] = {}
        for p, tag in enumerate(ports[s]):
            for d in by_port[tag]:
                w = np.zeros(R, dtype=np.int64)
                wlt = np.zeros(R, dtype=np.int64)
                i, seq = 0, d.lines
                while i < len(seq):
                    kind, t, delta = seq[i]
                    if last:
                        if kind == "light" and t in row_of:
                            w[row_of[t]] += delta
                            wlt[row_of[t]] += delta
                            attached.add(t)
                        i += 1
                        continue
                    if kind != "light":
                        findings.append(f"layer {s}: a lane writes {kind} {t} before any light of its unit")
                        i += 1
                        continue
                    attached.add(t)
                    if t in copy:
                        plane, wl, wp = copy[t], delta, delta
                        i += 1
                    elif i + 1 < len(seq) and seq[i + 1][0] == "plane":
                        plane, wl, wp = seq[i + 1][1], delta, -seq[i + 1][2]
                        i += 2
                    elif i + 1 < len(seq) and seq[i + 1][0] == "light":
                        t2, d2 = seq[i + 1][1], seq[i + 1][2]
                        attached.add(t2)
                        key = (t, t2)
                        rec = dead.setdefault(key, {})
                        rec[(tag, id(d))] = (delta, d.limit)
                        if d2 != -delta:
                            findings.append(f"layer {s}: unit lights {t} and {t2} get unequal amounts {delta}, {d2}")
                        i += 2
                        continue
                    else:
                        findings.append(f"layer {s}: a lane ends after light {t} without its twin")
                        i += 1
                        continue
                    pair_votes.setdefault(plane, Counter())[t] += 1
                    if plane not in row_of:
                        findings.append(f"layer {s}: a line targets plane {plane}, which no gate of layer {s + 1} reads")
                        continue
                    if wl != wp:
                        findings.append(f"layer {s}, unit with plane {plane}: light line {wl:+d} and plane line {wp:+d} "
                                        "of one doll disagree")
                    w[row_of[plane]] += wp
                    wlt[row_of[plane]] += wl
                lane_rows.setdefault(p, []).append((d, w, wlt))
        # lanes of each port in canonical order
        limits, fxs, fxls = [], [], []
        for p, tag in enumerate(ports[s]):
            ls = lane_rows[p]
            order = sorted(range(len(ls)), key=lambda k: (ls[k][0].limit, tuple(int(v) for v in ls[k][1])))
            limits.append([ls[k][0].limit for k in order])
            fxs.append(np.array([ls[k][1] for k in order], dtype=np.int64).reshape(len(order), R))
            fxls.append(np.array([ls[k][2] for k in order], dtype=np.int64).reshape(len(order), R))
        # biases
        bias = np.zeros(R, dtype=np.int64)
        for r, t in enumerate(row_tags):
            if last:
                bias[r] = lights.get(t, 0)
                attached.add(t)
            else:
                fl = gate_floor[t]
                if len(fl) > 1:
                    findings.append(f"layer {s + 1}: gate sectors of port {t} start at different heights {fl[:3]}")
                bias[r] = -fl[0]
        # twin lights of hidden units, from the line pairing
        if not last:
            for r, t in enumerate(row_tags):
                votes = pair_votes.get(t)
                if votes:
                    (light_tag, _), = votes.most_common(1)
                    if len(votes) > 1:
                        findings.append(f"layer {s}: plane {t} is paired with several lights {sorted(votes)}")
                    if light_tag in lights and lights[light_tag] != bias[r]:
                        findings.append(f"layer {s}, plane {t}: twin light {light_tag} starts at {lights[light_tag]}, the "
                                        f"plane at {-bias[r]} (bias {bias[r]})")
        # dead units
        dlist = []
        for (t1, t2), rec in sorted(dead.items()):
            b = lights.get(t1, 0)
            if lights.get(t2, 0) != -b:
                findings.append(f"layer {s}: unit lights {t1}, {t2} start at {b}, {lights.get(t2)} (not negated)")
            per_port = []
            for tag in ports[s]:
                lanes = sorted((d.limit, rec.get((tag, id(d)), (0, 0))[0]) for d in by_port[tag])
                per_port.append(tuple(w for _, w in lanes))
            dlist.append((int(b), tuple(per_port)))
        layers.append(Layer(s, ports[s], limits, fxs, bias, row_tags, sorted(dlist), last, fxl=None if (last or copy) else fxls))
    # targets of hidden planes must be the next layer's ports
    for s in range(n - 1):
        stray = {t for d in dolls[s] for k, t, _ in d.lines if k == "plane" and t not in set(ports[s + 1])}
        if stray:
            findings.append(f"layer {s} writes plane tag(s) {sorted(stray)[:4]} that no gate of layer {s + 1} reads")
    if any(k == "cplane" for ds in dolls for d in ds for k, _, _ in d.lines):
        findings.append("a ceiling effect line is present: not an MLP map")
    unattached = sorted(v for t, v in lights.items() if t not in attached)
    chk: dict = {}
    nc = list(NOT_CHECKED)
    all_dolls = [d for ds in dolls for d in ds]
    n_lanes = len(all_dolls)
    miss_wave = [d for d in all_dolls if d.wave is None or d.wave not in released]
    if miss_wave:
        findings.append(f"{len(miss_wave)} lane(s) have a wave barrier that no Floor_LowerToLowest line releases")
    miss_start = [s for s in range(1, n) if STAGE_TAG0 + s not in released]
    if miss_start:
        findings.append(f"start barrier of layer(s) {miss_start} is never released by a clock line")
    freeze_missing = [d for s, ds in enumerate(dolls) if s < n - 1 for d in ds if d.freeze not in frozen]
    if freeze_missing:
        findings.append(f"{len(freeze_missing)} lane(s) of layers before the last have a freeze barrier that no "
                        "Floor_MoveToValue line raises")
    for s, num in copy_layers.items():
        if not any(ln[2] == EXEC_ALWAYS and ln[3] == num for ln in tm.lines):
            findings.append(f"copy script {num} of layer {s} is never started by a line")
    start_tags = {START_TAG, *(STAGE_TAG0 + s for s in range(1, n))}
    waves = {d.wave for d in all_dolls if d.wave is not None}
    freeze_tags = {d.freeze for d in all_dolls}
    f_lines, counts = _check_lines(tm, lights, {d.port for d in all_dolls}, waves, start_tags, freeze_tags,
                                   set(copy_layers.values()), sum(d.n_lines for d in all_dolls))
    findings += f_lines
    findings += _check_lanes(tm, all_dolls, start_tags)
    chk["lines"] = ("every line special read (class, tag, every argument, mover speed, trigger flags): "
                    + ", ".join(f"{v} {k}" for k, v in sorted(counts.items())))
    if tm.extra_line_keys:
        nc.append(f"linedef keys {sorted(tm.extra_line_keys)}")
    if beh:
        barrier_floor: dict[int, list[int]] = {}
        tagset = start_tags | waves | freeze_tags
        for sec in range(tm.n_sectors):
            if tm.tag[sec] in tagset and not tm.isolated[sec]:
                barrier_floor.setdefault(tm.tag[sec], []).append(tm.floor[sec])
        barriers = {t: f[0] for t, f in barrier_floor.items() if len(set(f)) == 1}
        for t, f in barrier_floor.items():
            if len(set(f)) > 1:
                findings.append(f"sectors tagged {t} (barrier) have different floors {sorted(set(f))[:3]}")
        internal = {t: -int(layers[s - 1].bias[layers[s - 1].row_tags.index(t)]) for s in range(1, n) for t in ports[s]}
        f_s, nc_s, chk_s = _check_scripts(beh, tm, internal, barriers, copy_layers, lights)
        findings += f_s
        nc += nc_s
        chk.update(chk_s)
    else:
        findings.append("the map has no BEHAVIOR lump: no reset, release or readout script")
    chk["lanes"] = f"{n_lanes} doll lanes in {n} layers; wave, freeze and start barriers released: " \
                   + ("yes" if not (miss_wave or miss_start or freeze_missing) else "no")
    if copy:
        chk["copy bus"] = f"{len(copy)} lights copied into planes by ACS scripts {sorted(copy_layers.values())}"
    return Netlist(layers, unattached, findings, chk, source, nc)


# --- command line ---------------------------------------------------------------------------------------------------
def shape(nl: Netlist) -> list[int]:
    """Widths input, layer 0 rows, layer 1 rows, ... of the netlist in the net's own terms (a signed input layer
    counts one input per port pair)."""
    L0 = nl.layers[0]
    n_in = len(L0.ports) // 2 if nl.signed_reading() else len(L0.ports)
    return [n_in] + [L.n_rows for L in nl.layers]


def compared(want: Netlist) -> dict[str, int]:
    """How many numbers `diff` compares for a netlist: lane weights, biases, lane thresholds, nothing else."""
    return dict(weights=sum(int(f.size) for L in want.layers for f in L.fx),
                biases=sum(L.n_rows for L in want.layers),
                thresholds=sum(len(x) for L in want.layers for x in L.limits))


def summary(nl: Netlist) -> str:
    lanes = sum(len(x) for L in nl.layers for x in L.limits)
    return (f"{' -> '.join(map(str, shape(nl)))}, {'signed' if nl.signed_reading() else 'relu'} input, {len(nl.layers)} "
            f"layer(s), {lanes} doll lanes")


def add_extract_arguments(p) -> None:
    p.add_argument("wad", help="a compiled one-map MLP (.wad); only this file is read")
    p.add_argument("--dump", action="store_true", help="print the recovered netlist per layer (default: one summary line)")
    p.add_argument("--full", action="store_true", help="--dump: print every weight row and bias")
    p.add_argument("--net", help="also compare with this MLP (.npz): the same check as `verify --extract`")
    p.add_argument("--json", metavar="FILE", help="write the result record as JSON")


def run_compare(wad, net_path, *, dump: bool = False, full: bool = False, json_path=None) -> int:
    """Print the extraction of `wad` and, with `net_path`, the comparison; exit code 1 on any difference or finding."""
    import json
    import sys
    from .kit import MLP
    try:
        nl = extract(wad)
    except ExtractError as e:
        print(f"extract: {wad}: not readable as a compiled MLP: {e}", file=sys.stderr)
        return 1
    print(f"extract: {Path(wad).name}: {summary(nl)} (read from the WAD alone)")
    if dump:
        print(nl.dump(full=full))
    rec = dict(wad=str(wad), shape=shape(nl), signed_input=nl.signed_reading(), findings=nl.findings, checks=nl.checks,
               not_checked=nl.not_checked)
    code = 1 if nl.findings else 0
    if net_path is not None:
        net = MLP.load(net_path)
        diffs = nl.diff(net)
        want = expected_netlist(net)
        n = compared(want)
        rec.update(net=str(net_path), differences=[str(d) for d in diffs], compared=n)
        if diffs:
            print(f"DIFFERENT from {Path(net_path).name}: {len(diffs)} difference(s) (showing up to 10)")
            for d in diffs[:10]:
                print(f"  {d}")
            code = 1
        else:
            verdict = "EQUAL" if not nl.findings else "arithmetic EQUAL, but findings below"
            print(f"{verdict} to {Path(net_path).name}: {n['weights']:,} lane weights, {n['biases']:,} biases and "
                  f"{n['thresholds']:,} lane thresholds match, with {len(want.unattached)} unattributed light(s) matching "
                  f"as a multiset; plane contract: {nl.checks.get('plane contract', 'not checked')}")
        for f in nl.findings:
            print(f"  finding: {f}")
        if not diffs and nl.findings:
            print("NOT EQUAL: the map has findings (above)")
    elif nl.findings:
        for f in nl.findings:
            print(f"  finding: {f}")
    print("NOT CHECKED: " + "; ".join(nl.not_checked))
    if json_path:
        Path(json_path).write_text(json.dumps(rec, indent=1) + "\n")
    return code


def run_extract_cli(args) -> int:
    return run_compare(args.wad, args.net, dump=args.dump, full=args.full, json_path=args.json)
