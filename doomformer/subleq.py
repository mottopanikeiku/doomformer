"""A SUBLEQ machine in ONE map: instructions and data share one memory, so programs rewrite themselves.

    sm = SubleqMachine(slots=6)                 # 6 instruction slots = 18 memory cells
    wad = sm.compile("subleq.wad")              # one map; the bytes do not depend on the program
    sm.run(wad, MULTIPLY(7, 9))                 # program and data are initial state (plane heights)
    sm.run(wad, SELFMOD())                      # another program, the same sha256

One step (`interpret` is the Python reference, tolerance 0 against the map):

    s = pc;  a, b, c = mem[3s], mem[3s+1], mem[3s+2]      # the instruction is three memory cells
    mem[b] = mem[b] - mem[a]                              # a and b are ADDRESSES: indirect read, indirect write
    pc = c if mem[b] <= 0 else s + 1                      # c is a slot index; c < 0 or c >= slots, or s + 1 == slots: halt

Memory: N = 3 * slots cells (the default 18), each a signed integer in [-64, 63]. Addresses a and b must be 0..N-1;
the interpreter refuses a program that leaves either range, refuses a result outside [-64, 63], and refuses a step whose
write address b is one of the three cells of the instruction being executed (found by the fuzz: the fetch dolls of the
executing slot wait at ceiling gates on those cells, and a lowering that ends below a gate they have slipped into is
blocked; docs/machine.md). An instruction can rewrite any other instruction. The program
counter counts instruction triples (slot s is the cells 3s, 3s+1, 3s+2), so a jump target is a slot number. Any cell may
be read or written as data or executed as code, so `subleq x, loop.a, next` changes the operand of another instruction.

How the map does it. A cell is two plane heights, lo (0..7) and hi (-8..7), value = 8 hi + lo, each with a negated twin.
Every value below is a plane height; a doll reads values by crossing gates and writes them by crossing lines
(Floor_RaiseByValue / Floor_LowerByValue). Eight layers per step, one pass of the level's loop:
  F1 fetch    the doll of the active slot (lane gate pc_s >= 1) copies the digits of its three cells into the numbers
              al ah (a), bl bh (b), cl cp cn (c) by thermometer corridors of ceiling gates
  F2 decode   equality gates on those numbers raise one-hot ad_j, bd_j (the addresses) and ct_t (the jump target; ct_S is
              the halt latch of a target outside 0..slots-1)
  R  read     lane gate ad_j >= 1, equality gates on cell j's digits: one-hot xl_v, xh_v of the value of mem[a]
  N2 number   one crossbar doll per one-hot value adds v to the numbers xl and xh
  W  write    lane gate bd_j >= 1, thermometer corridors on xl and xh: cell j's digits go down by xl and 8 xh
  T  test     lane gate bd_j >= 1: le_j when 8 hi + lo <= 0 (hi < 0, or hi == 0 and lo <= 0); nb_j when lo < 0
  U  settle   le_j -> le; nb_j: lo += 8, hi -= 1 (the borrow)
  V  branch   pc_s -= 1; pc_{s+1} += 1; if le: pc_{s+1} -= 1 and pc_c += 1 (gate ct_c >= 1)
Within a layer no doll reads a value another doll of that layer writes, except the lane gates of the layer's own state
writes (V reads pc_s on the lane gate, before any release). pc is one-hot (pc_S is the halted pc). The ACS level script
is the clock and the halt check, as in machine.py; with clock="dolls" a loop doll and line specials are (dollclock.py,
docs/machine.md).
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from . import selfgen as sg
from .machine import Step, controller_acs, play
from .ops import fx, neg, output_pair
from .pure import compile_pure
from .tile import Block, Gate, Stage, TileSpec

OBSERVER = 11000
SUBLEQ_BOOT = 16
VMIN, VMAX = -64, 63
DEFAULT_SLOTS = 6


def split(v: int) -> tuple[int, int]:
    """(lo, hi) of a cell value: value = 8 * hi + lo, lo in 0..7."""
    return v & 7, v >> 3


def lo_name(j: int) -> str:
    return f"r{2 * j}"


def hi_name(j: int) -> str:
    return f"r{2 * j + 1}"


def xh_name(v: int) -> str:
    return f"xh{v}" if v > 0 else f"xhm{-v}"


XH_VALUES = [v for v in range(-8, 8) if v]
XL_VALUES = list(range(1, 8))


def _clamp(v, lo, hi):
    return max(lo, min(hi, v))


@dataclass(frozen=True)
class SubleqProgram:
    """mem: the initial memory (cell values); slots: the machine's capacity; symbols: name -> cell; lines: the source."""
    mem: tuple
    slots: int
    symbols: tuple = ()
    lines: tuple = ()
    n_code: int = 0

    @property
    def N(self) -> int:
        return 3 * self.slots

    @property
    def halted(self) -> int:
        return self.slots

    @property
    def regs(self) -> tuple:
        return tuple(f"r{i}" for i in range(2 * self.N))

    def digits(self, mem=None) -> tuple:
        out = []
        for v in (self.mem if mem is None else mem):
            out += list(split(v))
        return tuple(out)

    def logical(self, values) -> dict[str, int]:
        return {f"m{j}": 8 * int(values[2 * j + 1]) + int(values[2 * j]) for j in range(self.N)}

    def want(self, step: Step, outputs) -> dict[str, int]:
        base = {f"pc{k}": int(k == step.pc) for k in range(self.slots + 1)}
        base.update({f"r{i}": v for i, v in enumerate(step.regs)})
        base["pass"], base["done"] = step.idx, int(step.idx > 0)          # the doll clock's tick layer (dollclock.py)
        out = {}
        for n in outputs:
            b = n.lstrip("-")
            v = base[b] if b in base else step.fires.get(b, 0)
            out[n] = -v if n.startswith("-") else v
        return out


# --- assembler ----------------------------------------------------------------------------------------------------
def assemble_subleq(source, slots: int = DEFAULT_SLOTS) -> SubleqProgram:
    """Text -> initial memory. Lines (`;` or `#` start a comment, a `label:` may precede any line):

        label: subleq A B C     one instruction (three cells at the next free slot): A and B are addresses, C a slot
        .cell name value        one data cell; data cells follow the code (cell 3 * instructions + k, in order)
        .at cell value          set cell `cell` (a number or symbol) to `value`, wherever it is

    A, B, C, value: an integer, a data-cell name, `label` (a label as C is its slot; as A or B it is the first cell of
    that instruction), `label.a` / `label.b` / `label.c` (that instruction's cell), `halt` (= -1, only as C)."""
    lines = source.splitlines() if isinstance(source, str) else list(source)
    ins, data, ats, labels, listing = [], [], [], {}, []
    for raw in lines:
        text = re.split(r"[;#]", raw, 1)[0]
        while True:
            m = re.match(r"\s*([A-Za-z_][\w]*)\s*:\s*(.*)", text)
            if not m:
                break
            if m.group(1) in labels:
                raise ValueError(f"label {m.group(1)!r} defined twice")
            labels[m.group(1)] = len(ins)
            text = m.group(2)
        tok = text.replace(",", " ").split()
        if not tok:
            continue
        op = tok[0].lower()
        if op == "subleq" and len(tok) == 4:
            ins.append(tok[1:])
        elif op == ".cell" and len(tok) == 3:
            data.append((tok[1], tok[2]))
        elif op == ".at" and len(tok) == 3:
            ats.append((tok[1], tok[2]))
        else:
            raise ValueError(f"bad line {raw.strip()!r} (subleq A B C | .cell name value | .at cell value)")
        listing.append(" ".join(tok))
    n = 3 * slots
    if len(ins) > slots:
        raise ValueError(f"{len(ins)} instructions do not fit {slots} slots")
    if 3 * len(ins) + len(data) > n:
        raise ValueError(f"{len(ins)} instructions and {len(data)} data cells do not fit {n} cells")
    sym = {name: 3 * len(ins) + k for k, (name, _) in enumerate(data)}

    def number(tok: str, role: str) -> int:
        if re.fullmatch(r"-?\d+", tok):
            return int(tok)
        if tok == "halt":
            return -1
        if tok in sym:
            return sym[tok]
        m = re.fullmatch(r"([A-Za-z_]\w*)\.([abc])", tok)
        if m and m.group(1) in labels:
            return 3 * labels[m.group(1)] + "abc".index(m.group(2))
        if tok in labels:
            return labels[tok] if role == "c" else 3 * labels[tok]
        raise ValueError(f"unknown name {tok!r}")

    mem = [0] * n
    for i, (a, b, c) in enumerate(ins):
        mem[3 * i:3 * i + 3] = [number(a, "a"), number(b, "b"), number(c, "c")]
    for k, (_, v) in enumerate(data):
        mem[3 * len(ins) + k] = number(v, "v")
    for cell, v in ats:
        mem[number(cell, "a")] = number(v, "v")
    for v in mem:
        if not VMIN <= v <= VMAX:
            raise ValueError(f"value {v} outside {VMIN}..{VMAX}")
    return SubleqProgram(tuple(mem), slots, tuple(sorted(sym.items(), key=lambda kv: kv[1])), tuple(listing), len(ins))


# --- the reference: a Python SUBLEQ interpreter ----------------------------------------------------------------------
def interpret(p: SubleqProgram, max_steps: int = 100_000, strict: bool = True) -> list[Step]:
    """Step 0 is the initial state, step i the state after the i-th instruction with every value the map's layers
    produce in it (`fires`; names in the module docstring). Ends at the halted pc or after max_steps. A step with an
    address outside the memory or a result outside -64..63 raises ValueError; with strict=False the trace ends before it
    (the fuzz of models/subleq/fuzz.py runs the part of a random program the map can run)."""
    S, N = p.slots, p.N
    mem, pc = list(p.mem), 0
    trace = [Step(pc, p.digits(mem), {}, 0)]
    while pc < S and len(trace) <= max_steps:
        a, b, c = mem[3 * pc:3 * pc + 3]
        k = len(trace)
        if not (0 <= a < N and 0 <= b < N):
            if not strict:
                break
            raise ValueError(f"step {k}: slot {pc} addresses cell {a if not 0 <= a < N else b}, outside 0..{N - 1}")
        if 3 * pc <= b <= 3 * pc + 2:                  # the map cannot run it: module docstring, docs/machine.md
            if not strict:
                break
            raise ValueError(f"step {k}: slot {pc} writes its own instruction cell {b}; the map cannot run a step whose "
                             "executing instruction is rewritten (docs/machine.md, limits)")
        f: dict[str, int] = {}
        (al, ahh), (bl, bhh), (cl, chh) = (split(v) for v in mem[3 * pc:3 * pc + 3])
        f.update(al=al, ah=_clamp(ahh, 0, 2), bl=bl, bh=_clamp(bhh, 0, 2), cl=cl, cp=int(chh >= 1), cn=int(chh <= -1))
        f[f"ad{a}"], f[f"bd{b}"] = 1, 1
        target = c if 0 <= c < S else S
        f[f"ct{target}"] = 1
        xl, xh = split(mem[a])
        if xl:
            f[f"xl{xl}"] = 1
        if xh:
            f[xh_name(xh)] = 1
        f["xl"], f["xh"] = xl, xh
        y = mem[b] - mem[a]
        if not VMIN <= y <= VMAX:
            if not strict:
                break
            raise ValueError(f"step {k}: mem[{b}] - mem[{a}] = {mem[b]} - {mem[a]} = {y} outside {VMIN}..{VMAX}")
        bl_, bh_ = split(mem[b])
        lo_u, hi_u = bl_ - xl, bh_ - xh
        le = hi_u < 0 or (hi_u == 0 and lo_u <= 0)
        assert le == (y <= 0)
        if le:
            f[f"le{b}"], f["le"] = 1, 1
        if lo_u < 0:
            f[f"nb{b}"] = 1
        mem[b] = y
        pc = target if le else (pc + 1 if pc + 1 <= S else S)
        trace.append(Step(pc, p.digits(mem), f, len(trace)))
    return trace


# --- the map -----------------------------------------------------------------------------------------------------
class _Layers:
    """The eight layers of the machine plus the doll bookkeeping the tamper test needs."""

    NAMES = ("fetch", "decode", "read", "number", "write", "test", "settle", "branch")

    def __init__(self, S: int):
        self.S, self.N = S, 3 * S
        self.layers = [TileSpec(n) for n in self.NAMES]
        self.F1, self.F2, self.R, self.N2, self.W, self.T, self.U, self.V = self.layers
        self._pairs: dict = {}
        self.tag_of_block: dict = {}                 # (layer name, label) -> block index
        self._build()

    # gate constructors: lanes are floor gates, stage gates ceiling gates
    @staticmethod
    def lane_ge(spec, name, k=1):
        return Gate(spec.port(neg(name), "floor"), -k, "floor")

    @staticmethod
    def lane_le(spec, name, k):
        return Gate(spec.port(name, "floor"), k, "floor")

    @staticmethod
    def st_ge(spec, name, k):
        return Gate(spec.port(name, "ceil"), k, "ceil")

    @staticmethod
    def st_le(spec, name, k):
        return Gate(spec.port(neg(name), "ceil"), -k, "ceil")

    def P(self, spec, name):
        key = (spec.name, name)
        if key not in self._pairs:
            self._pairs[key] = output_pair(spec, name)
        return self._pairs[key]

    def add(self, spec, label, lane, gates, effects):
        """One doll: lane gate, stage gates in ascending walking height, the effects after the last gate."""
        gates = sorted(gates, key=lambda g: g.walk_floor)
        stages = [Stage(g, []) for g in gates[:-1]] + [Stage(gates[-1] if gates else None, effects)]
        self.tag_of_block[(spec.name, label)] = len(spec.blocks)
        spec.blocks.append(Block([lane], stages))

    def thermo(self, spec, label, lane, source, levels, target, w=1):
        """A corridor of ceiling gates source >= k (k in levels), each followed by an effect of w on `target`: adds
        w * (number of levels source reaches)."""
        stages = [Stage(self.st_ge(spec, source, k), fx(self.P(spec, target), w)) for k in levels]
        self.tag_of_block[(spec.name, label)] = len(spec.blocks)
        spec.blocks.append(Block([lane], stages))

    def conds(self, spec, name, value, lo, hi):
        """Conditions name == value for a value known to lie in lo..hi: [(op, name, k)] (only the needed ones)."""
        out = []
        if value > lo:
            out.append(("ge", name, value))
        if value < hi:
            out.append(("le", name, value))
        return out

    def doll(self, spec, label, lane_cond, conds, effects):
        """A doll whose first condition is the lane and the rest stage gates (conds: (op, name, k) triples)."""
        allc = list(lane_cond) + list(conds)
        first, rest = allc[0], allc[1:]
        lane = self.lane_ge(spec, first[1], first[2]) if first[0] == "ge" else self.lane_le(spec, first[1], first[2])
        gates = [self.st_ge(spec, n, k) if op == "ge" else self.st_le(spec, n, k) for op, n, k in rest]
        self.add(spec, label, lane, gates, effects)

    def _build(self):
        S, N = self.S, self.N
        F1, F2, R, N2, W, T, U, V = self.layers
        P = lambda spec, name: self.P(spec, name)
        pc = lambda s: f"pc{s}"
        # F1: copy the three cells of the active slot into numbers (thermometer corridors of ceiling gates)
        for s in range(S):
            lane = self.lane_ge(F1, pc(s))
            for k, (lname, hname) in enumerate((("al", "ah"), ("bl", "bh"))):
                self.thermo(F1, f"{lname}{s}", lane, lo_name(3 * s + k), range(1, 8), lname)
                self.thermo(F1, f"{hname}{s}", lane, hi_name(3 * s + k), range(1, 3), hname)
            self.thermo(F1, f"cl{s}", lane, lo_name(3 * s + 2), range(1, 8), "cl")
            self.thermo(F1, f"cp{s}", lane, hi_name(3 * s + 2), range(1, 2), "cp")
            self.thermo(F1, f"cn{s}", lane, neg(hi_name(3 * s + 2)), range(1, 2), "cn")
        # F2: decode the numbers into one-hot values
        for j in range(N):
            for pre, lname, hname in (("ad", "al", "ah"), ("bd", "bl", "bh")):
                c = self.conds(F2, lname, j % 8, 0, 7) + self.conds(F2, hname, j // 8, 0, 2)
                self.doll(F2, f"{pre}{j}", c[:1], c[1:], fx(P(F2, f"{pre}{j}"), 1))
        for t in range(S):
            c = self.conds(F2, "cl", t, 0, 7) + [("le", "cp", 0), ("le", "cn", 0)]
            self.doll(F2, f"ct{t}", c[:1], c[1:], fx(P(F2, f"ct{t}"), 1))
        halt = fx(P(F2, f"ct{S}"), 1)
        self.doll(F2, "cth_p", [("ge", "cp", 1)], [], halt)
        self.doll(F2, "cth_n", [("ge", "cn", 1)], [], halt)
        self.doll(F2, "cth_l", [("ge", "cl", S)], [("le", "cp", 0), ("le", "cn", 0)], halt)
        # R: one-hot digits of mem[a]
        for j in range(N):
            for v in XL_VALUES:
                self.doll(R, f"xl{j}_{v}", [("ge", f"ad{j}", 1)], self.conds(R, lo_name(j), v, 0, 7),
                          fx(P(R, f"xl{v}"), 1))
            for v in XH_VALUES:
                self.doll(R, f"xh{j}_{v}", [("ge", f"ad{j}", 1)], self.conds(R, hi_name(j), v, -8, 7),
                          fx(P(R, xh_name(v)), 1))
        # N2: one-hot -> numbers
        for v in XL_VALUES:
            self.add(N2, f"xl{v}", self.lane_ge(N2, f"xl{v}"), [], fx(P(N2, "xl"), v))
        for v in XH_VALUES:
            self.add(N2, f"xh{v}", self.lane_ge(N2, xh_name(v)), [], fx(P(N2, "xh"), v))
        # W: subtract the digits of x from the digits of cell b
        for j in range(N):
            lane = self.lane_ge(W, f"bd{j}")
            self.thermo(W, f"lo{j}", lane, "xl", range(1, 8), lo_name(j), -1)
            self.thermo(W, f"hp{j}", lane, "xh", range(1, 8), hi_name(j), -1)
            self.thermo(W, f"hn{j}", lane, neg("xh"), range(1, 9), hi_name(j), 1)
        # T: is the new value <= 0, did the low digit borrow
        for j in range(N):
            self.doll(T, f"le1_{j}", [("ge", f"bd{j}", 1)], [("ge", neg(hi_name(j)), 1)], fx(P(T, f"le{j}"), 1))
            self.doll(T, f"le2_{j}", [("ge", f"bd{j}", 1)],
                      [("ge", hi_name(j), 0), ("ge", neg(hi_name(j)), 0), ("ge", neg(lo_name(j)), 0)],
                      fx(P(T, f"le{j}"), 1))
            self.tag_of_block[(T.name, f"nb{j}")] = len(T.blocks)
            T.blocks.append(Block([self.lane_le(T, lo_name(j), -1)], [Stage(None, fx(P(T, f"nb{j}"), 1))]))
        # U: gather le, borrow
        for j in range(N):
            self.add(U, f"le{j}", self.lane_ge(U, f"le{j}"), [], fx(P(U, "le"), 1))
            self.add(U, f"nb{j}", self.lane_ge(U, f"nb{j}"), [],
                     fx(P(U, lo_name(j)), 8) + fx(P(U, hi_name(j)), -1))
        # V: the program counter
        for s in range(S):
            self.add(V, f"cl{s}", self.lane_ge(V, pc(s)), [], fx(P(V, pc(s)), -1))
            self.add(V, f"nx{s}", self.lane_ge(V, pc(s)), [], fx(P(V, pc(s + 1)), 1))
            self.add(V, f"un{s}", self.lane_ge(V, pc(s)), [self.st_ge(V, "le", 1)], fx(P(V, pc(s + 1)), -1))
        for t in range(S + 1):
            self.add(V, f"tk{t}", self.lane_ge(V, "le"), [self.st_ge(V, f"ct{t}", 1)], fx(P(V, pc(t)), 1))
        # observers: a floor port for every state value (both signs) the layers do not already read
        names = [pc(k) for k in range(S + 1)] + [f"r{i}" for i in range(2 * N)]
        have = {key.split(":", 1)[1] for spec in self.layers for key in spec.ports}
        for n in names:
            for m in (n, neg(n)):
                if m not in have:
                    N2.blocks.append(Block([Gate(N2.port(m, "floor"), OBSERVER, "floor")], [Stage(None, [])]))
                    have.add(m)
        self.names = names

    def state(self):
        st = {}
        for n in self.names:
            v = 1 if n == "pc0" else 0
            st[n], st[neg(n)] = v, -v
        return st

    def doll_index(self, layer: str, label: str) -> int:
        """Global index of the doll of (layer, block label): doll tags are DOLL_TID0 + index (one clock doll per layer
        precedes the next layer's dolls)."""
        k = self.NAMES.index(layer)
        base = sum(spec.n_dolls + 1 for spec in self.layers[:k])
        return base + self.tag_of_block[(layer, label)]


def compile_subleq(S: int, wad: Path, *, max_passes: int = 1_000_000, layout_only: bool = False, opt=0,
                   clock: str = "acs"):
    """Compile the machine with `S` instruction slots (3 S cells) into `wad`; the bytes do not depend on a program.
    clock="acs": the level's loop is an ACS OPEN script (machine.controller_acs). clock="dolls": no OPEN script, a loop
    doll and line specials restart the level (dollclock.py); the ACS lump keeps the readout pages, the motion script and
    the two mover scripts the host's initial-state writes use."""
    if clock not in ("acs", "dolls"):
        raise ValueError(f"clock must be 'acs' or 'dolls', not {clock!r}")
    L = _Layers(S)
    layers, state = L.layers, L.state()
    actions = None
    if clock == "dolls":
        from .dollclock import tick_layer
        tick, extra, actions = tick_layer(f"pc{S}")
        layers = layers + [tick]
        state = {**state, **extra}
    n_dolls = sum(s.n_dolls + 1 for s in layers)
    consumed = {key.split(":", 1)[1] for s in layers for key in s.ports}
    outs = [o for s in layers for o, _ in s.outputs]
    readout = list(state) + [o for o in dict.fromkeys(outs) if o in consumed and o not in state]
    lights = [o for o in outs if o not in consumed and o not in state]
    pages = -(-(len(lights) + len(readout)) // sg.PAGE)
    if 100 + pages > sg.LOOP:
        raise ValueError(f"subleq machine: {pages} readout pages would reach the level's script {sg.LOOP}")
    hold = sg.hold_tics(pages)
    tags = sg.port_tags(layers)
    halt_tag = tags[f"floor:pc{S}"]
    if clock == "dolls":
        from .dollclock import HOLD_TICS, loop_world
        hold = max(HOLD_TICS, hold + 24)
        world = loop_world(layers, state, f"pc{S}", n_dolls, hold)
    else:
        def world(mb):
            mb.rect(-768, -512, -512, -256, floor=0, ceil=128)
            mb.thing(-640, -384, 1)
            return controller_acs(n_dolls, max_passes, hold, SUBLEQ_BOOT, halt_tag)

    meta = compile_pure(layers, wad, "subleq", state=state, readout=readout, world=world, layout_only=layout_only,
                        opt=opt, actions=actions)
    if layout_only:
        return meta, L
    if meta.layout["tags"] != tags or meta.n_dolls != n_dolls:
        raise AssertionError("subleq machine: the controller's plan and the emitted map differ")
    meta.layout["machine"] = dict(halt_tag=halt_tag, hold=hold, boot=SUBLEQ_BOOT, max_passes=max_passes, slots=S,
                                  subleq=True, clock=clock)
    if clock == "dolls":
        from .dollclock import finish_wad
        meta.layout["machine"]["rewind_lines"], meta.layout["machine"]["rewind_rows"] = world.chain_lines, world.rows
        meta.layout["machine"]["dollclock"] = finish_wad(wad, meta, loaders=True)
    meta.save(Path(wad).with_suffix(".json"))
    return meta, L


def loader(p: SubleqProgram):
    """The host's initial-state writes: every nonzero digit of the program's memory (both signs of each value)."""
    def load(game, meta):
        tags = meta.layout["tags"]
        writes = {f"r{i}": v for i, v in enumerate(p.digits()) if v}
        for name, v in writes.items():
            for key, tag in tags.items():
                kind, val = key.split(":", 1)
                if val == name:
                    game.load(tag, v, kind)
                elif val == neg(name):
                    game.load(tag, -v, kind)
    return load


class SubleqMachine:
    """One compiled map, any SUBLEQ program that fits `slots` instruction slots (module docstring)."""

    def __init__(self, slots: int = DEFAULT_SLOTS):
        if not 2 <= slots <= 8:
            raise ValueError("slots must be 2..8 (the halt test on the jump target reads one digit)")
        self.slots = slots

    @property
    def cells(self) -> int:
        return 3 * self.slots

    def compile(self, wad, **kw):
        """Compile the one map into `wad` (and wad.json); returns the TileMeta."""
        return compile_subleq(self.slots, Path(wad), **kw)[0]

    def assemble(self, source) -> SubleqProgram:
        return source if isinstance(source, SubleqProgram) else assemble_subleq(source, self.slots)

    def reference(self, source, max_steps: int = 100_000) -> list[Step]:
        return interpret(self.assemble(source), max_steps)

    def run(self, wad, source, *, max_steps: int = 100_000, on_step=None, clock: str = "acs"):
        """Load `source` into the compiled map `wad` (compiled with the same `clock`), play, and check every step against
        `interpret`."""
        p = self.assemble(source)
        cls = None
        if clock == "dolls":
            from .dollclock import DollClock as cls
        return play(Path(wad), p, interpret(p, max_steps), on_step=on_step, loader=loader(p), clock_cls=cls)


# --- ready programs (initial memory is code + data; every one runs on the same map bytes) ------------------------------
def ADD(a: int = 3, b: int = 4) -> str:
    """c = a + b with a zero cell z: z = -a, z = -(a + b), c = c - z."""
    return f"""
        subleq a, z, next1        ; z = -a
next1:  subleq b, z, next2        ; z = -(a + b)
next2:  subleq z, c, next3        ; c = a + b
next3:  subleq z, z, halt         ; z = 0 <= 0: jump to -1, halt
.cell a {a}
.cell b {b}
.cell z 0
.cell c 0"""


def MULTIPLY(x: int = 7, y: int = 9) -> str:
    """p = x * y by repeated addition: p -= (-y), x -= 1, until x <= 0. Needs x >= 1 and x * y <= 63."""
    return f"""
loop:   subleq ny, p, next        ; p = p + y
next:   subleq one, x, halt       ; x = x - 1; x <= 0: halt
        subleq z, z, loop         ; z = 0 <= 0: jump back
.cell ny {-y}
.cell p 0
.cell one 1
.cell x {x}
.cell z 0"""


def SELFMOD(values=(5, 3, 7)) -> str:
    """z = -(v0 + v1 + v2). The first instruction's operand a is a pointer into the array; the second instruction adds 1
    to that operand cell (a store into the program), so the first instruction reads the next element each round."""
    v0, v1, v2 = values
    return f"""
rd:     subleq v0, z, bump        ; z = z - mem[rd.a]; rd.a starts at v0's cell
bump:   subleq m1, rd.a, count    ; rd.a = rd.a + 1  (the program rewrites its own operand)
count:  subleq m1, n, rd          ; n = n + 1; n <= 0: another round
        subleq n, n, halt         ; n > 0: clear the counter, halt
.cell v0 {v0}
.cell v1 {v1}
.cell v2 {v2}
.cell z 0
.cell m1 -1
.cell n -2"""


READY = {"add": (ADD, 2, "c", "a + b"), "multiply": (MULTIPLY, 2, "p", "x * y"),
         "selfmod": (SELFMOD, 0, "z", "-(5 + 3 + 7), operand of the first instruction rewritten each round")}


# --- negative control: change one decode line of the compiled map ---------------------------------------------------
def tamper(wad: Path, p: SubleqProgram, address: int) -> dict:
    """Retarget the decode line of address `address` (operand a): the doll of F2 that raises ad_<address> lowers the gate
    port of ad_<address + 1> instead, in the compiled WAD's TEXTMAP. The interpreter predicts the first difference at
    the first step whose operand a is `address`; ValueError if no step has it."""
    from .machine import _wad_bytes, _wad_lumps
    from .pure import FLOOR_LOWER
    from .tile import DOLL_TID0, TileMeta
    wad = Path(wad)
    meta = TileMeta.load(wad.with_suffix(".json"))
    N = p.N
    if not 0 <= address < N:
        raise ValueError(f"address {address} outside 0..{N - 1}")
    trace = interpret(p)
    first = next((i for i, s in enumerate(trace) if f"ad{address}" in s.fires), None)
    if first is None:
        raise ValueError(f"no step of this program reads cell {address} as operand a")
    L = _Layers(p.slots)
    doll = L.doll_index("decode", f"ad{address}")
    tags = meta.layout["tags"]
    old, new = tags[f"floor:-ad{address}"], tags[f"floor:-ad{(address + 1) % N}"]
    lumps = _wad_lumps(wad.read_bytes())
    tm = next(i for i, (n, _) in enumerate(lumps) if n == "TEXTMAP")
    text = lumps[tm][1].decode()
    blocks = list(re.finditer(r"linedef\s*\{[^}]*\}", text))
    mine = lambda d: re.compile(rf"special = (?:226; arg0 = 4; arg1 = 0; arg2 = {DOLL_TID0 + d}|"
                                rf"176; arg0 = 0; arg1 = {DOLL_TID0 + d});")
    lo = next(i for i, m in enumerate(blocks) if mine(doll).search(m.group(0)))
    hi = next((i for i, m in enumerate(blocks) if i > lo and mine(doll + 1).search(m.group(0))), len(blocks))
    line_re = re.compile(r"special = (\d+); arg0 = (\d+);")
    for i in range(lo, hi):
        line = blocks[i].group(0)
        m = line_re.search(line)
        if m and int(m.group(1)) == FLOOR_LOWER and int(m.group(2)) == old:
            text = text[:blocks[i].start()] + line.replace(m.group(0), f"special = {FLOOR_LOWER}; arg0 = {new};") \
                + text[blocks[i].end():]
            lumps[tm] = ("TEXTMAP", text.encode())
            wad.write_bytes(_wad_bytes(lumps))
            return dict(address=address, doll=doll, changed=f"decode line of cell {address}: Floor_LowerByValue on the "
                        f"gate port of ad{address} -> ad{(address + 1) % N}", predicted_step=first)
    raise ValueError(f"no decode line found in the doll of ad{address}")
