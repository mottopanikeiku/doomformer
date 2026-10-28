"""`df.Machine`: a counter machine compiled into ONE DOOM map. Registers and a one-hot program counter are state values
(plane heights, pure.compile_pure `state`), the program is geometry, and a halt latch ends the run.

    m = df.Machine('''
            top:  jzdec b done          ; b == 0 -> done, else b -= 1
                  inc a
                  jmp top
            done: halt''', registers={"a": 3, "b": 4})
    result = m.run()            # compiles, plays the map, checks EVERY step of the trace against a Python interpreter
    result.registers            # {"a": 7, "b": 0}

Instructions (one machine step each; `halt` and falling off the end stop the machine):

    inc r                 r += 1
    add r k               r += k (k a constant, may be negative)
    jzdec r L             if r == 0 goto L, else r -= 1
    jmp L
    jge r k L / jlt r k L if r >= k (r < k) goto L, else continue
    addr a b              a += b where 0 <= b <= base - 1 (b is copied by an ADC corridor of base - 1 gates; a == b is
                          allowed). It is how multi-limb numbers add in a few steps instead of one step per unit.
    halt

Macros (expanded to the instructions above before compiling): clear r; move s d (d += s, s = 0); copy s d t (d = s,
d and t scratch registers that start 0); bigadd A B (A += B for multi-limb registers, carry by jlt/add/add).
Multi-limb registers: `limbs={"a": 4}` stores a as 4 base-`base` limbs a.0 (least significant) ... a.3. Every limb must stay
in 0..base-1 when it is an `addr` source, so choose enough limbs: the interpreter raises on an addr source above
base-1. `result.registers` joins the limbs back into one integer.

Per step the map does (two layers, one pass of the level's loop):
  decide layer: the doll of the active instruction (the lane gate reads pc_k >= 1) reads the registers it needs, by
      gates, and raises a fire value (f/z/n/t/u + instruction number, b for addr).
  apply layer: one doll per fire; it crosses the lines that lower pc_k, raise pc_next and change the register by
      Floor_RaiseByValue / Floor_LowerByValue on the register's plane heights. Fire values are re-zeroed by the level's
      rewind (script 3); state values are not.
Python advances the clock and runs readout scripts through readout.LoadThenReadout (sealed before the first tic: no
host writes at all), reads pc and every register after every step, and compares the whole readout with `interpret`'s
trace at tolerance 0. What does the control: the level's loop (an ACS OPEN script, selfgen-style: rewind, drop the start
barriers, wait for the dolls to settle, hold the window for Python's read), which also leaves the loop when the
halt latch's plane height (pc_halt) is >= 1. That script is the machine's clock and its halting check; the dolls are
its data path and its program counter. docs/machine.md has the full account and the limits.
"""
from __future__ import annotations

import hashlib
import re
import struct
import time
from dataclasses import dataclass, field
from pathlib import Path

from . import selfgen as sg
from .kit import MismatchError
from .ops import fx, neg, output_pair
from .pure import FLOOR_RAISE, compile_pure
from .readout import LoadThenReadout, readout_commands
from .tile import DOLL_TID0, MOTION_SCRIPT, PORT_MAX, Block, Gate, Stage, TileSpec

DEFAULT_BASE = 16
MAX_PASSES = 1_000_000          # the level's loop bound; the halt latch ends every shipped run long before it
BOOT = 8                        # tics before the first pass (selfgen.BOOT)
OBSERVER = 11000                # gate limit of the effect-less observer dolls that give every state value a readout port


class MachineMismatch(MismatchError):
    """DOOM's readout of `step` differs from the interpreter's trace. `.step` counts from 1; 0 is the initial state."""

    def __init__(self, step: int, mismatches: list[tuple[str, int, int]], pc: int | None = None):
        self.step, self.pc = step, pc
        super().__init__(f"machine step {step}", mismatches)


# --- assembler ---------------------------------------------------------------------------------------------------
SIGNATURE = {"inc": "R", "add": "RK", "jzdec": "RL", "jmp": "L", "jge": "RKL", "jlt": "RKL", "addr": "RR", "halt": ""}
MACRO_ARITY = {"clear": 1, "move": 2, "copy": 3, "bigadd": 2}


@dataclass(frozen=True)
class Program:
    """An assembled program: `code` holds tuples (op, operands...) with registers as indices and labels as pcs; `halt`
    is `jmp len(code)`. `regs` are the physical register names (a limbed register `a` is a.0, a.1, ...)."""
    code: tuple
    regs: tuple
    init: tuple
    base: int
    limbs: dict = field(default_factory=dict)      # logical name -> tuple of register indices, least significant first
    lines: tuple = ()                              # the expanded source, one string per instruction

    @property
    def halted(self) -> int:
        return len(self.code)

    def want(self, step, outputs) -> dict[str, int]:
        return expected_values(self, step, outputs)

    def logical(self, values) -> dict[str, int]:
        """Register values as {name: int}; limbs are joined (limb i has weight base**i)."""
        out = {n: int(v) for n, v in zip(self.regs, values) if "." not in n}
        for name, idx in self.limbs.items():
            out[name] = sum(int(values[i]) * self.base ** j for j, i in enumerate(idx))
        return out


def _split_limbs(value: int, count: int, base: int) -> list[int]:
    if value < 0:
        raise ValueError("limbed registers start non-negative")
    out = []
    for _ in range(count - 1):
        out.append(value % base)
        value //= base
    out.append(value)
    return out


def _expand(lines: list[str], limbs: dict[str, int], base: int, uid: list | None = None) -> list[str]:
    """Macros -> primitive source lines (label-only lines kept). `uid` numbers the generated labels."""
    out, uid = [], uid if uid is not None else [0]

    def fresh():
        uid[0] += 1
        return f".m{uid[0]}"

    def tokens(text):
        text = re.split(r"[;#]", text, 1)[0]
        label = None
        m = re.match(r"\s*([A-Za-z_.][\w.]*)\s*:(.*)", text)
        if m:
            label, text = m.group(1), m.group(2)
        return label, text.replace(",", " ").split()

    for raw in lines:
        label, tok = tokens(raw)
        if label:
            out.append(f"{label}:")
        if not tok:
            continue
        op, args = tok[0].lower(), tok[1:]
        if op in MACRO_ARITY:
            if len(args) != MACRO_ARITY[op]:
                raise ValueError(f"{op} takes {MACRO_ARITY[op]} operands: {raw.strip()!r}")
            c, e = fresh(), fresh()
            if op == "clear":
                out += [f"{c}: jzdec {args[0]} {e}", f"jmp {c}", f"{e}:"]
            elif op == "move":
                out += [f"{c}: jzdec {args[0]} {e}", f"inc {args[1]}", f"jmp {c}", f"{e}:"]
            elif op == "copy":
                s, d, t = args
                out += [f"{c}: jzdec {s} {e}", f"inc {d}", f"inc {t}", f"jmp {c}", f"{e}:"]
                out += _expand([f"move {t} {s}"], limbs, base, uid)
            else:
                a, b = args
                if a not in limbs or b not in limbs or limbs[a] != limbs[b]:
                    raise ValueError(f"bigadd needs two limbed registers with the same limb count: {raw.strip()!r}")
                n = limbs[a]
                out += [f"addr {a}.{i} {b}.{i}" for i in range(n)]
                for i in range(n - 1):
                    skip = fresh()
                    out += [f"jlt {a}.{i} {base} {skip}", f"add {a}.{i} {-base}", f"add {a}.{i + 1} 1", f"{skip}:"]
        else:
            out.append(f"{op} " + " ".join(args))
    return out


def assemble(source, registers: dict | None = None, limbs: dict | None = None, base: int = DEFAULT_BASE) -> Program:
    """Assemble text (or a list of lines) into a Program. `registers`: name -> initial value; `limbs`: name -> number of
    limbs (the register then lives in a.0 ... a.{n-1})."""
    lines = source.splitlines() if isinstance(source, str) else list(source)
    registers, limbs = dict(registers or {}), dict(limbs or {})
    if base < 2 or base - 1 > 60:
        raise ValueError("base must be 2 .. 61 (addr corridors have base - 1 gates)")
    names, init, groups = [], [], {}
    for name, value in registers.items():
        if name in limbs:
            digits = _split_limbs(int(value), limbs[name], base)
            groups[name] = tuple(range(len(names), len(names) + len(digits)))
            names += [f"{name}.{i}" for i in range(len(digits))]
            init += digits
        else:
            names.append(name)
            init.append(int(value))
    for name in limbs:
        if name not in registers:
            raise ValueError(f"limbs for {name!r}, which is not a register")
    if any(abs(v) > PORT_MAX for v in init):
        raise ValueError(f"initial register value beyond +-{PORT_MAX}")
    index = {n: i for i, n in enumerate(names)}
    flat = _expand(lines, limbs, base)
    labels, items = {}, []
    for text in flat:
        while True:
            m = re.match(r"\s*([A-Za-z_.][\w.]*)\s*:\s*(.*)", text)
            if not m:
                break
            if m.group(1) in labels:
                raise ValueError(f"label {m.group(1)!r} defined twice")
            labels[m.group(1)] = len(items)
            text = m.group(2)
        if text.strip():
            items.append(text.split())
    code, listing = [], []
    for tok in items:
        op, args = tok[0], tok[1:]
        if op not in SIGNATURE:
            raise ValueError(f"unknown instruction {' '.join(tok)!r}")
        sig = SIGNATURE[op]
        if len(args) != len(sig):
            raise ValueError(f"{op} takes {len(sig)} operand(s): {' '.join(tok)!r}")
        out = [op]
        for kind, a in zip(sig, args):
            if kind == "R":
                if a not in index:
                    raise ValueError(f"unknown register {a!r} in {' '.join(tok)!r}")
                out.append(index[a])
            elif kind == "K":
                out.append(int(a))
            else:
                if a not in labels:
                    raise ValueError(f"unknown label {a!r} in {' '.join(tok)!r}")
                out.append(labels[a])
        if op == "halt":
            out = ["jmp", len(items)]
        code.append(tuple(out))
        listing.append(" ".join(tok))
    if not code:
        raise ValueError("empty program")
    return Program(tuple(code), tuple(names), tuple(init), base, groups, tuple(listing))


# --- the reference: a Python interpreter ---------------------------------------------------------------------------
def fire_names(code: tuple) -> dict[int, list[str]]:
    """Fire values of every instruction, in the order the compiler declares them."""
    out = {}
    for k, ins in enumerate(code):
        op = ins[0]
        out[k] = {"inc": [f"f{k}"], "add": [f"f{k}"], "jmp": [f"f{k}"], "jzdec": [f"z{k}", f"n{k}"],
                  "jge": [f"t{k}", f"u{k}"], "jlt": [f"t{k}", f"u{k}"], "addr": [f"f{k}", f"b{k}"]}[op]
    return out


@dataclass
class Step:
    pc: int
    regs: tuple
    fires: dict            # fire name -> value this step produced (empty at step 0)
    idx: int = 0           # the step number (0 = the initial state); the doll clock's `pass` counter equals it


def interpret(p: Program, max_steps: int = 100_000) -> list[Step]:
    """The trace: Step 0 is the initial state, step i the state after the i-th instruction, with that instruction's fire
    values. Ends at the halted pc or after max_steps (then the last step's pc is not p.halted)."""
    pc, regs = 0, list(p.init)
    trace = [Step(pc, tuple(regs), {}, 0)]
    W = p.base - 1
    while pc < p.halted and len(trace) <= max_steps:
        ins = p.code[pc]
        op, fires = ins[0], {}
        nxt = pc + 1
        if op == "inc":
            regs[ins[1]] += 1
            fires[f"f{pc}"] = 1
        elif op == "add":
            regs[ins[1]] += ins[2]
            fires[f"f{pc}"] = 1
        elif op == "jmp":
            nxt = ins[1]
            fires[f"f{pc}"] = 1
        elif op == "jzdec":
            if regs[ins[1]] == 0:
                nxt = ins[2]
                fires[f"z{pc}"] = 1
            else:
                regs[ins[1]] -= 1
                fires[f"n{pc}"] = 1
        elif op in ("jge", "jlt"):
            taken = regs[ins[1]] >= ins[2] if op == "jge" else regs[ins[1]] < ins[2]
            fires[f"t{pc}" if taken else f"u{pc}"] = 1
            if taken:
                nxt = ins[3]
        else:                                                    # addr
            b = regs[ins[2]]
            if not 0 <= b <= W:
                raise ValueError(f"step {len(trace)}: addr {p.regs[ins[1]]} {p.regs[ins[2]]} with "
                                 f"{p.regs[ins[2]]} = {b}, outside 0..{W}")
            regs[ins[1]] += b
            fires[f"f{pc}"] = 1
            fires[f"b{pc}"] = b
        if any(abs(v) > PORT_MAX for v in regs):
            raise ValueError(f"step {len(trace)}: a register left +-{PORT_MAX}: {regs}")
        pc = nxt
        trace.append(Step(pc, tuple(regs), fires, len(trace)))
    return trace


def expected_values(p: Program, step: Step, outputs) -> dict[str, int]:
    """Every name the map exposes (meta.outputs) with the value the interpreter says it has after `step`."""
    want = {}
    for k in range(p.halted + 1):
        want[f"pc{k}"] = int(k == step.pc)
    for i, v in enumerate(step.regs):
        want[f"r{i}"] = v
    for names in fire_names(p.code).values():
        for n in names:
            want[n] = step.fires.get(n, 0)
    want["pass"], want["done"] = step.idx, int(step.idx > 0)          # the doll clock's tick layer (dollclock.py)
    out = {}
    for name in outputs:
        base = name.lstrip("-")
        out[name] = -want[base] if name.startswith("-") else want[base]
    return out


# --- the map -----------------------------------------------------------------------------------------------------
def _specs(p: Program):
    N, W = p.halted, p.base - 1
    D, B = TileSpec("decide"), TileSpec("apply")
    pc = lambda k: f"pc{k}"
    reg = lambda i: f"r{i}"
    lane = lambda k: [Gate(D.port(neg(pc(k)), "floor"), -1, "floor")]            # pc_k >= 1
    ge = lambda r, k: Gate(D.port(neg(reg(r)), "floor"), -k, "floor")            # r >= k
    le = lambda r, k: Gate(D.port(reg(r), "floor"), k, "floor")                  # r <= k
    fire_pairs = {}
    effects = {}                                                                # fire -> {value name: weight}
    lanes_b = []                                                                # (fire name, lane limit, effects)

    def fire(name):
        fire_pairs[name] = output_pair(D, name)
        return fire_pairs[name]

    def eff(*pairs):
        out = {}
        for n, w in pairs:
            out[n] = out.get(n, 0) + w
        return {n: w for n, w in out.items() if w}

    for k, ins in enumerate(p.code):
        op = ins[0]
        if op in ("inc", "add", "jmp"):
            f = fire(f"f{k}")
            D.blocks.append(Block(lane(k), [Stage(None, fx(f, 1))]))
            target, extra = (ins[1], []) if op == "jmp" else (k + 1, [(reg(ins[1]), 1 if op == "inc" else ins[2])])
            effects[f"f{k}"] = eff((pc(k), -1), (pc(target), 1), *extra)
        elif op == "jzdec":
            _, r, t = ins
            z, n = fire(f"z{k}"), fire(f"n{k}")
            D.blocks.append(Block(lane(k), [Stage(le(r, 0), fx(z, 1))]))
            D.blocks.append(Block(lane(k), [Stage(ge(r, 1), fx(n, 1))]))
            effects[f"z{k}"] = eff((pc(k), -1), (pc(t), 1))
            effects[f"n{k}"] = eff((pc(k), -1), (pc(k + 1), 1), (reg(r), -1))
        elif op in ("jge", "jlt"):
            _, r, c, t = ins
            yes, no = fire(f"t{k}"), fire(f"u{k}")
            hi, lo = ge(r, c), le(r, c - 1)
            D.blocks.append(Block(lane(k), [Stage(hi if op == "jge" else lo, fx(yes, 1))]))
            D.blocks.append(Block(lane(k), [Stage(lo if op == "jge" else hi, fx(no, 1))]))
            effects[f"t{k}"] = eff((pc(k), -1), (pc(t), 1))
            effects[f"u{k}"] = eff((pc(k), -1), (pc(k + 1), 1))
        else:                                                                    # addr a b
            _, a, b = ins
            f, fb = fire(f"f{k}"), fire(f"b{k}")
            stages = [Stage(None, fx(f, 1))] + [Stage(Gate(D.port(reg(b), "ceil"), j, "ceil"), fx(fb, 1))
                                                for j in range(1, W + 1)]
            D.blocks.append(Block(lane(k), stages))
            effects[f"f{k}"] = eff((pc(k), -1), (pc(k + 1), 1))
            for j in range(1, W + 1):
                lanes_b.append((f"b{k}", -j, {reg(a): 1}))
    # apply layer: one doll per fire (and one per ADC step of addr)
    pairs = {}

    def P(n):
        if n not in pairs:
            pairs[n] = output_pair(B, n)
        return pairs[n]

    for name, eff_ in effects.items():
        B.blocks.append(Block([Gate(B.port(neg(name), "floor"), -1, "floor")],
                              [Stage(None, [e for n, w in eff_.items() for e in fx(P(n), w)])]))
    for name, limit, eff_ in lanes_b:
        B.blocks.append(Block([Gate(B.port(neg(name), "floor"), limit, "floor")],
                              [Stage(None, [e for n, w in eff_.items() for e in fx(P(n), w)])]))
    names = [pc(k) for k in range(N + 1)] + [reg(i) for i in range(len(p.regs))]
    # the halt latch and every state value need a floor port in the decide layer for the readout
    D.blocks.append(Block([Gate(D.port(pc(N), "floor"), OBSERVER, "floor")], [Stage(None, [])]))
    have = {key.split(":", 1)[1] for key in D.ports}
    for n in names:
        for m in (n, neg(n)):
            if m not in have:
                D.blocks.append(Block([Gate(D.port(m, "floor"), OBSERVER, "floor")], [Stage(None, [])]))
    state = {}
    start = dict.fromkeys(names, 0)
    start[pc(0)] = 1
    for i, v in enumerate(p.init):
        start[reg(i)] = v
    for n in names:
        state[n], state[neg(n)] = start[n], -start[n]
    return D, B, state, names, fire_pairs


def controller_acs(n_dolls: int, max_passes: int, hold: int, boot: int, halt_tag: int) -> str:
    """The level's clock and halting check: selfgen.level_acs plus one read per pass, the halt latch's plane height.
    A pass: rewind (from the second), drop the start barriers, wait until no doll moved over one window, HOLD for
    Python's read. The loop ends before the next pass when the latch is up."""
    return f"""
// --- machine controller (doomformer/machine.py): clock, rewind, halt check ---
int sg_x[{n_dolls}]; int sg_y[{n_dolls}]; int sg_z[{n_dolls}];
int sg_phase = {sg.PH_BOOT}; int sg_pass; int sg_start = -1; int sg_hold = -1;
function int sg_moved(void) {{
  int moved = 0;
  for (int i = 0; i < {n_dolls}; i++) {{
    int tid = {DOLL_TID0} + i;
    if (!IsTIDUsed(tid)) {{ moved++; continue; }}
    int x = GetActorX(tid); int y = GetActorY(tid); int z = GetActorZ(tid);
    if (x != sg_x[i] || y != sg_y[i] || z != sg_z[i]) {{ moved++; sg_x[i] = x; sg_y[i] = y; sg_z[i] = z; }}
  }}
  return moved;
}}
script {sg.LOOP} OPEN {{
  Delay({boot});
  for (int p = 0; p < {max_passes}; p++) {{
    if (GetSectorFloorZ({halt_tag}, 0, 0) >= 1) break;
    if (p > 0) {{ ACS_ExecuteAlways(3, 0); Delay({sg.REWIND}); }}
    sg_phase = {sg.PH_COMPUTE}; sg_start = Timer();
    ACS_ExecuteAlways(7, 0);
    Delay({sg.STILL});
    sg_moved();
    do {{ Delay({sg.STILL}); }} while (sg_moved());
    sg_pass = p + 1; sg_hold = Timer(); sg_phase = {sg.PH_HOLD};
    Delay({hold});
  }}
  sg_phase = {sg.PH_DONE};
}}
script {sg.STATUS} (void) {{ o1 = sg_phase; o2 = sg_pass; o3 = sg_start; o4 = sg_hold; o5 = Timer(); o6 = {max_passes}; }}
"""


def compile_machine(p: Program, wad: Path, *, max_passes: int = MAX_PASSES, boot: int = BOOT,
                    layout_only: bool = False, opt=0, clock: str = "acs"):
    """Compile `p` into `wad`. Returns (TileMeta, info) where info holds the names and counts the runner needs.
    clock="acs" (the default): the level's loop is an ACS OPEN script. clock="dolls": no OPEN script; a loop doll, a
    tick layer and line specials restart the level (doomformer/dollclock.py)."""
    if clock not in ("acs", "dolls"):
        raise ValueError(f"clock must be 'acs' or 'dolls', not {clock!r}")
    D, B, state, names, fires = _specs(p)
    layers = [D, B]
    actions = None
    if clock == "dolls":
        from .dollclock import tick_layer
        tick, extra, actions = tick_layer(f"pc{p.halted}")
        layers.append(tick)
        state = {**state, **extra}
    n_dolls = sum(s.n_dolls + 1 for s in layers)
    consumed = {key.split(":", 1)[1] for s in layers for key in s.ports}
    readout = list(state) + [m for m in (neg(f) for f in fires) if m in consumed]
    lights = [o for s in layers for o, _ in s.outputs if o not in consumed and o not in state]
    reads = len(lights) + len(readout)
    pages = -(-reads // sg.PAGE)
    hold = sg.hold_tics(pages)
    tags = sg.port_tags(layers)
    halt_tag = tags[f"floor:pc{p.halted}"]
    if clock == "dolls":
        from .dollclock import HOLD_TICS, loop_world
        hold = max(HOLD_TICS, hold + 24)
        world = loop_world(layers, state, f"pc{p.halted}", n_dolls, hold)
    else:
        def world(mb):
            mb.rect(-768, -512, -512, -256, floor=0, ceil=128)
            mb.thing(-640, -384, 1)
            return controller_acs(n_dolls, max_passes, hold, boot, halt_tag)

    meta = compile_pure(layers, wad, "machine", state=state, readout=readout, world=world, layout_only=layout_only,
                        opt=opt, actions=actions)
    if layout_only:
        return meta, None
    if meta.layout["tags"] != tags or meta.n_dolls != n_dolls:
        raise AssertionError("machine: the controller's plan and the emitted map differ")
    meta.layout["machine"] = dict(halt_tag=halt_tag, hold=hold, boot=boot, max_passes=max_passes,
                                  b_dolls=B.n_dolls, d_dolls=D.n_dolls, clock=clock)
    if clock == "dolls":
        from .dollclock import finish_wad
        meta.layout["machine"]["rewind_lines"] = world.chain_lines
        meta.layout["machine"]["dollclock"] = finish_wad(wad, meta)
    meta.save(Path(wad).with_suffix(".json"))
    return meta, dict(names=names, fires=fires)


def _apply_order(p: Program) -> list[str]:
    order, adc = [], []
    W = p.base - 1
    for k, ins in enumerate(p.code):
        op = ins[0]
        if op in ("inc", "add", "jmp", "addr"):
            order.append(f"f{k}")
            if op == "addr":
                adc += [f"b{k}"] * W
        elif op == "jzdec":
            order += [f"z{k}", f"n{k}"]
        elif op in ("jge", "jlt"):
            order += [f"t{k}", f"u{k}"]
    return order + adc


# --- running it --------------------------------------------------------------------------------------------------
@dataclass
class Result:
    program: Program
    steps: int
    registers: dict
    pc: int
    tics: int                  # game tics from the first tic to the end of the last HOLD
    seconds: float
    dolls: int
    sectors: int
    pass_tics: list
    checked: int               # values read back and compared (tolerance 0)
    audit: dict
    halted: bool
    wad_sha256: str
    trace: list = field(repr=False, default_factory=list)       # DOOM's (pc, registers) per step, from step 0

    def summary(self) -> str:
        a = self.audit
        return (f"{self.dolls:,} dolls, {self.sectors:,} sectors, {self.steps:,} steps, {self.tics:,} game tics "
                f"({min(self.pass_tics):,} to {max(self.pass_tics):,} per step), {self.seconds:.0f} s, "
                f"{self.checked:,} values checked, audit: {sum(a['commands'].values()):,} readout commands, "
                f"{a['non_readout_commands']} other, {a['loaded']} state loads")


class Clock:
    """The level's loop seen from Python: advance tics, read STATUS and the motion script, wait for HOLD windows."""

    commands = (f"puke -{sg.STATUS}", f"puke -{MOTION_SCRIPT}")

    def __init__(self, runner, game):
        from .runtime import USER
        self.runner, self.game, self._user = runner, game, USER

    def script(self, script: int, n: int) -> list[int]:
        self.game.send_game_command(f"puke -{script}")
        self.game.advance_action(1)
        return [int(self.game.get_game_variable(v)) for v in self._user[:n]]

    def status(self) -> dict[str, int]:
        return dict(zip(sg.STATUS_NAMES, self.script(sg.STATUS, len(sg.STATUS_NAMES))))

    def motion(self) -> tuple[int, int]:
        moved, untagged = self.script(MOTION_SCRIPT, 2)
        return moved, untagged

    def wait_hold(self, k: int, max_tics: int = 400_000) -> dict[str, int]:
        waited = 0
        while True:
            s = self.status()
            if s["phase"] == sg.PH_HOLD and s["pass"] == k:
                return s
            if s["pass"] >= k or s["phase"] == sg.PH_DONE:
                raise RuntimeError(f"missed the HOLD window of step {k}: {s}")
            if waited > max_tics:
                raise RuntimeError(f"step {k} did not finish within {max_tics} tics: {s}")
            self.game.advance_action(sg.POLL)
            waited += sg.POLL

    def finish(self, last_pass: int, hold: int) -> None:
        """After the halt: the level reached DONE with the last pass number and nothing moves."""
        self.game.advance_action(3 * hold)
        s = self.status()
        if s["phase"] != sg.PH_DONE or s["pass"] != last_pass:
            raise RuntimeError(f"the halt latch did not stop the level: {s}")
        self.motion()
        self.game.advance_action(sg.STILL)
        moved, untagged = self.motion()
        if moved or untagged:
            raise RuntimeError(f"dolls still moving after the halt: {moved} moved, {untagged} untagged")

    def read_still(self, k: int, s: dict[str, int]) -> dict[str, int]:
        """Every readout of the HOLD window `s`, proven read inside one still window."""
        self.motion()
        got = self.runner.read()
        moved, untagged = self.motion()
        after = self.status()
        if after["phase"] != sg.PH_HOLD or after["pass"] != k or after["hold"] != s["hold"] or moved or untagged:
            raise RuntimeError(f"step {k} was not read inside a still HOLD window: before {s}, after {after}, "
                               f"{moved} dolls moved, {untagged} untagged")
        return got


def sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def play(wad: Path, p: Program, expected: list[Step], *, on_step=None, loader=None, resolution=None,
         clock_cls=None) -> Result:
    """Run the compiled map `wad` of program `p` and compare its readout with `expected` (interpret's trace) at every
    step, tolerance 0. `loader(game, meta)` may write state values before the first pass (stored programs); the game is
    sealed (readout only) afterwards. `clock_cls` is `Clock` (the ACS level script, the default) or
    dollclock.DollClock. Raises MachineMismatch at the first differing step."""
    from .runtime import TileRunner
    kw = {} if resolution is None else dict(resolution=resolution)
    runner = TileRunner(wad, **kw)
    meta = runner.meta
    clock_cls = clock_cls or Clock
    game = runner.game = LoadThenReadout(runner.game, readout_commands(len(meta.pages), *clock_cls.commands))
    try:
        if loader is not None:
            loader(game, meta)
        game.seal()
        clock = clock_cls(runner, game)
        t0 = time.time()
        game.advance_action(2)
        checked, trace, pass_tics = 0, [], []
        halted = expected[-1].pc == p.halted

        def check(k, got, step, record=True):
            nonlocal checked
            if set(got) != set(meta.outputs):
                raise AssertionError("incomplete readout")
            want = p.want(step, meta.outputs)
            bad = [(n, v, want[n]) for n, v in got.items() if v != want[n]]
            pcs = [i for i in range(p.halted + 1) if got.get(f"pc{i}") == 1]
            now = (pcs[0] if len(pcs) == 1 else -1, tuple(got[f"r{i}"] for i in range(len(p.regs))))
            if record:
                trace.append(now)
                checked += len(got)
            if bad:
                raise MachineMismatch(k, bad, now[0])

        check(0, runner.read(), expected[0])
        for k in range(1, len(expected)):
            s = clock.wait_hold(k)
            got = clock.read_still(k, s)
            if s["hold"] > s["start"]:
                pass_tics.append(s["hold"] - s["start"])
            check(k, got, expected[k])
            if on_step is not None:
                on_step(k, expected[k])
        if halted:                                    # the latch stops the level: nothing moves afterwards
            clock.finish(len(expected) - 1, meta.layout["machine"]["hold"])
            check(len(expected) - 1, runner.read(), expected[-1], record=False)
        tics = int(game.get_episode_time())
        seconds = time.time() - t0
        last = expected[-1]
        return Result(p, len(expected) - 1, p.logical(last.regs), last.pc, tics, seconds, meta.n_dolls,
                      meta.n_sectors, pass_tics, checked, game.audit(), halted, sha256(wad), trace)
    finally:
        runner.close()


class Machine:
    """A counter-machine program that runs inside one DOOM map (module docstring)."""

    def __init__(self, program, registers: dict | None = None, *, limbs: dict | None = None,
                 base: int = DEFAULT_BASE):
        self.program = assemble(program, registers, limbs, base)

    @property
    def halted_pc(self) -> int:
        return self.program.halted

    def reference(self, max_steps: int = 100_000) -> list[Step]:
        return interpret(self.program, max_steps)

    def compile(self, wad: Path, **kw):
        """Compile into `wad` (and wad.json); returns the TileMeta."""
        return compile_machine(self.program, Path(wad), **kw)[0]

    def run(self, wad: Path | None = None, *, max_steps: int = 100_000, on_step=None, clock: str = "acs") -> Result:
        """Compile (into `wad`, or a temporary directory), play and check every step against `reference`.
        clock="dolls": the map has no OPEN script (doomformer/dollclock.py)."""
        import tempfile
        expected = self.reference(max_steps)
        cls = None
        if clock == "dolls":
            from .dollclock import DollClock as cls
        if wad is not None:
            self.compile(Path(wad), clock=clock)
            return play(Path(wad), self.program, expected, on_step=on_step, clock_cls=cls)
        with tempfile.TemporaryDirectory(prefix="doomformer-machine-") as tmp:
            path = Path(tmp) / "machine.wad"
            self.compile(path, clock=clock)
            return play(path, self.program, expected, on_step=on_step, clock_cls=cls)


# --- ROM tamper: change one instruction line of the compiled map ------------------------------------------------------
def _wad_lumps(data: bytes) -> list[tuple[str, bytes]]:
    n, directory = struct.unpack("<ii", data[4:12])
    out = []
    for i in range(n):
        offset, size, name = struct.unpack("<ii8s", data[directory + 16 * i:directory + 16 * (i + 1)])
        out.append((name.rstrip(b"\0").decode(), data[offset:offset + size]))
    return out


def _wad_bytes(lumps: list[tuple[str, bytes]]) -> bytes:
    body, directory = b"", b""
    for name, blob in lumps:
        directory += struct.pack("<ii8s", 12 + len(body), len(blob), name.encode())
        body += blob
    return struct.pack("<4sii", b"PWAD", len(lumps), 12 + len(body)) + body + directory


def tamper(wad: Path, p: Program, instruction: int, *, what: str = "weight") -> dict:
    """Rewrite one effect line of instruction `instruction`'s apply-layer doll in the compiled WAD (its TEXTMAP).
    what='weight': the amount of the line that changes the register the readout reads (inc, add, jzdec's decrement),
    |w| -> |w| + 1. what='target': the line that raises the next pc is retargeted to pc_halted. Returns the change and the
    step at which the interpreter predicts the first difference (the first step that fires that doll)."""
    from .tile import TileMeta
    wad = Path(wad)
    meta = TileMeta.load(wad.with_suffix(".json"))
    expr = meta.layout["read_expressions"]
    tag_of = lambda name: int(re.search(r"Get\w+\((\d+)", expr[name]).group(1))
    ins = p.code[instruction]
    names = fire_names(p.code)[instruction]
    if what == "weight":
        if ins[0] not in ("inc", "add", "jzdec") or (ins[0] == "add" and ins[2] == 0):
            raise ValueError("only inc, add and jzdec have a register weight line")
        name, reg = (names[1] if ins[0] == "jzdec" else names[0]), f"r{ins[1]}"
        target_tag = tag_of(reg)
        pc_tags = None
    else:
        name, target_tag = names[0], None
        pc_tags = {tag_of(f"pc{k}") for k in range(p.halted + 1)}
    doll_id = meta.layout["machine"]["d_dolls"] + 1 + _apply_order(p).index(name)
    lumps = _wad_lumps(wad.read_bytes())
    tm = next(i for i, (n, _) in enumerate(lumps) if n == "TEXTMAP")
    text = lumps[tm][1].decode()
    blocks = list(re.finditer(r"linedef\s*\{[^}]*\}", text))
    mine = lambda d: re.compile(rf"special = (?:226; arg0 = 4; arg1 = 0; arg2 = {DOLL_TID0 + d}|"
                                rf"176; arg0 = 0; arg1 = {DOLL_TID0 + d});")
    lo = next(i for i, m in enumerate(blocks) if mine(doll_id).search(m.group(0)))
    hi = next((i for i, m in enumerate(blocks) if i > lo and mine(doll_id + 1).search(m.group(0))), len(blocks))
    line_re = re.compile(r"special = (\d+); arg0 = (\d+); arg1 = (\d+); arg2 = (\d+);")
    changed = None
    for i in range(lo, hi):
        line = blocks[i].group(0)
        m = line_re.search(line)
        if not m or int(m.group(1)) not in (FLOOR_RAISE, 20, 41, 40):
            continue
        special, tag, amount = int(m.group(1)), int(m.group(2)), int(m.group(4))
        if what == "weight" and tag == target_tag:
            new = line.replace(m.group(0), m.group(0).replace(f"arg2 = {amount};", f"arg2 = {amount + 1};"))
            changed = (reg, amount, amount + 1)
        elif what == "target" and tag in pc_tags and special == FLOOR_RAISE:
            new = line.replace(f"arg0 = {tag};", f"arg0 = {tag_of(f'pc{p.halted}')};")
            changed = (f"pc -> pc{p.halted}",)
        else:
            continue
        text = text[:blocks[i].start()] + new + text[blocks[i].end():]
        break
    if changed is None:
        raise ValueError(f"no {what} line found in the doll of {name} (instruction {instruction})")
    lumps[tm] = ("TEXTMAP", text.encode())
    wad.write_bytes(_wad_bytes(lumps))
    trace = interpret(p)
    first = next((i for i, s in enumerate(trace) if name in s.fires), None)
    return dict(instruction=instruction, fire=name, what=what, changed=changed, predicted_step=first)
