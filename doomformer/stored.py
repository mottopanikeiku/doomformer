"""A stored-program register machine in ONE map: the compiled WAD is the same bytes for every program. The program is
state: each slot's instruction is a handful of one-hot flag values (plane heights) that the host writes before the first
tic through `readout.LoadThenReadout` (the same writes that load the registers). After the first tic the host sends only
readout scripts and clock advances, like `df.Machine`.

    sm = StoredMachine(slots=12, registers=("a", "b", "c", "d"))
    wad = sm.compile("stored.wad")                       # one map
    sm.run(wad, MULTIPLY, {"a": 3, "b": 4})              # program 1 and its data are loaded as initial state
    sm.run(wad, POWER, {"a": 3})                         # program 2 on the same bytes

Instruction set (slot i, fall-through to slot i + 1; target `halt` or any pc >= slots stops the machine):

    inc r L     r += 1, goto L          jmp L     goto L (an inc without a register)
    dec r L     if r >= 1: r -= 1, goto L; else fall through to the next slot
    halt        jmp to the halted pc

Per step three layers decode and apply it (every value below is a plane height):
  fetch    the doll of the active slot (lane gate pc_i >= 1) copies the slot's set flags into cur_* values: cur_inc,
           cur_dec, cur_reg_r, cur_target_j. One doll per (slot, flag); a flag the slot does not have stops its doll.
  decide   from cur_* and the registers: raise ir_r / dr_r (the register changes), gi_j / gd_j_r / fl_i_r (the next pc)
           and cl_i (clear pc_i). Gates only, no state is written here.
  apply    one doll per fire value; it crosses the lines that change registers and the one-hot pc.
`interpret` is the NumPy-free reference (pure Python ints); the runner compares every exposed value after every step,
tolerance 0. The ACS level script is the clock and the halt check, as in machine.py.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from . import selfgen as sg
from .machine import Step, controller_acs, play
from .ops import fx, neg, output_pair
from .pure import compile_pure
from .tile import PORT_MAX, Block, Gate, Stage, TileSpec

OBSERVER = 11000
STORED_BOOT = 16


@dataclass(frozen=True)
class StoredProgram:
    """slots: per slot (op, register or -1, target); `n_slots` is the machine's capacity (the halted pc)."""
    slots: tuple
    n_slots: int
    regs: tuple
    init: tuple
    lines: tuple = ()

    @property
    def halted(self) -> int:
        return self.n_slots

    def flags(self) -> dict[str, int]:
        """Every program flag value (0/1) of the memory."""
        R, N = len(self.regs), self.n_slots
        out = {}
        for i in range(N):
            op, r, t = self.slots[i]
            out[f"oi{i}"], out[f"od{i}"] = int(op == "inc"), int(op == "dec")
            for k in range(R):
                out[f"rg{i}_{k}"] = int(r == k)
            for j in range(N + 1):
                out[f"tg{i}_{j}"] = int(t == j)
        return out

    def want(self, step: Step, outputs) -> dict[str, int]:
        N, R = self.n_slots, len(self.regs)
        want = dict(self.flags())
        for k in range(N + 1):
            want[f"pc{k}"] = int(k == step.pc)
        for k, v in enumerate(step.regs):
            want[f"r{k}"] = v
        for n in fire_values(N, R):
            want[n] = step.fires.get(n, 0)
        return {name: -want[name[1:]] if name.startswith("-") else want[name] for name in outputs}

    def logical(self, values) -> dict[str, int]:
        return {n: int(v) for n, v in zip(self.regs, values)}


def fire_values(N: int, R: int) -> list[str]:
    """Names of the fetch and decide values (fires), in the order the compiler declares them."""
    return (["ci", "cd"] + [f"cr{k}" for k in range(R)] + [f"ct{j}" for j in range(N + 1)]
            + [f"ir{k}" for k in range(R)] + [f"dr{k}" for k in range(R)] + [f"gi{j}" for j in range(N + 1)]
            + [f"gd{j}_{k}" for j in range(N + 1) for k in range(R)]
            + [f"fl{i}_{k}" for i in range(N) for k in range(R)] + [f"cl{i}" for i in range(N)])


def assemble_stored(source, slots: int, registers: dict) -> StoredProgram:
    """Assemble `source` (text or lines) for a machine with `slots` slots and the named registers (name -> initial
    value). Unused slots are filled with `halt`."""
    lines = source.splitlines() if isinstance(source, str) else list(source)
    names = list(registers)
    items, labels = [], {}
    for raw in lines:
        text = re.split(r"[;#]", raw, 1)[0]
        while True:
            m = re.match(r"\s*([A-Za-z_.][\w.]*)\s*:\s*(.*)", text)
            if not m:
                break
            if m.group(1) in labels:
                raise ValueError(f"label {m.group(1)!r} defined twice")
            labels[m.group(1)] = len(items)
            text = m.group(2)
        if text.strip():
            items.append(text.replace(",", " ").split())
    if len(items) > slots:
        raise ValueError(f"{len(items)} instructions do not fit {slots} slots")
    code = []
    for tok in items:
        op, args = tok[0].lower(), tok[1:]
        need = {"inc": 2, "dec": 2, "jmp": 1, "halt": 0}.get(op)
        if need is None or len(args) != need:
            raise ValueError(f"bad instruction {' '.join(tok)!r} (inc r L | dec r L | jmp L | halt)")
        def target(a):
            if a == "halt":
                return slots
            if a in labels:
                return labels[a]
            if a.lstrip("-").isdigit():
                return int(a)
            raise ValueError(f"unknown label {a!r} in {' '.join(tok)!r}")

        def reg(a):
            if a not in names:
                raise ValueError(f"unknown register {a!r} in {' '.join(tok)!r}")
            return names.index(a)

        if op == "halt":
            code.append(("inc", -1, slots))
        elif op == "jmp":
            code.append(("inc", -1, target(args[0])))
        else:
            code.append((op, reg(args[0]), target(args[1])))
    listing = [" ".join(t) for t in items]
    while len(code) < slots:
        code.append(("inc", -1, slots))
        listing.append("halt")
    for _, _, t in code:
        if not 0 <= t <= slots:
            raise ValueError(f"jump target {t} outside 0..{slots}")
    return StoredProgram(tuple(code), slots, tuple(names), tuple(int(v) for v in registers.values()), tuple(listing))


def interpret(p: StoredProgram, max_steps: int = 100_000) -> list[Step]:
    pc, regs = 0, list(p.init)
    trace = [Step(pc, tuple(regs), {})]
    while pc < p.halted and len(trace) <= max_steps:
        op, r, t = p.slots[pc]
        f = {f"ct{t}": 1}
        f["ci" if op == "inc" else "cd"] = 1
        f[f"cl{pc}"] = 1
        if r >= 0:
            f[f"cr{r}"] = 1
        nxt = t
        if op == "inc":
            f[f"gi{t}"] = 1
            if r >= 0:
                regs[r] += 1
                f[f"ir{r}"] = 1
        elif regs[r] >= 1:
            regs[r] -= 1
            f[f"dr{r}"], f[f"gd{t}_{r}"] = 1, 1
        else:
            f[f"fl{pc}_{r}"] = 1
            nxt = pc + 1
        if any(abs(v) > PORT_MAX for v in regs):
            raise ValueError(f"step {len(trace)}: a register left +-{PORT_MAX}: {regs}")
        pc = nxt
        trace.append(Step(pc, tuple(regs), f))
    return trace


def _specs(N: int, R: int):
    F, X, Y = TileSpec("fetch"), TileSpec("decide"), TileSpec("apply")
    pc, reg = (lambda k: f"pc{k}"), (lambda k: f"r{k}")
    ge = lambda spec, name, k=1: Gate(spec.port(neg(name), "floor"), -k, "floor")          # name >= k
    le = lambda spec, name, k=0: Gate(spec.port(name, "floor"), k, "floor")                 # name <= k
    lane = lambda spec, name: [ge(spec, name)]
    cur = {n: output_pair(F, n) for n in ["ci", "cd"] + [f"cr{k}" for k in range(R)] + [f"ct{j}" for j in range(N + 1)]}
    flags = []
    for i in range(N):
        names = [f"oi{i}", f"od{i}"] + [f"rg{i}_{k}" for k in range(R)] + [f"tg{i}_{j}" for j in range(N + 1)]
        targets = ["ci", "cd"] + [f"cr{k}" for k in range(R)] + [f"ct{j}" for j in range(N + 1)]
        flags += names
        for flag, c in zip(names, targets):
            F.blocks.append(Block(lane(F, pc(i)), [Stage(ge(F, flag), fx(cur[c], 1))]))
    fires = {}

    def fire(name):
        fires[name] = output_pair(X, name)
        return fires[name]

    for i in range(N):
        X.blocks.append(Block(lane(X, pc(i)), [Stage(None, fx(fire(f"cl{i}"), 1))]))
    for k in range(R):
        X.blocks.append(Block(lane(X, "ci"), [Stage(ge(X, f"cr{k}"), fx(fire(f"ir{k}"), 1))]))
        X.blocks.append(Block(lane(X, "cd"), [Stage(ge(X, f"cr{k}"), []), Stage(ge(X, reg(k)), fx(fire(f"dr{k}"), 1))]))
    for j in range(N + 1):
        X.blocks.append(Block(lane(X, "ci"), [Stage(ge(X, f"ct{j}"), fx(fire(f"gi{j}"), 1))]))
        for k in range(R):
            X.blocks.append(Block(lane(X, "cd"), [Stage(ge(X, f"cr{k}"), []), Stage(ge(X, f"ct{j}"), []),
                                                  Stage(ge(X, reg(k)), fx(fire(f"gd{j}_{k}"), 1))]))
    for i in range(N):
        for k in range(R):
            X.blocks.append(Block(lane(X, pc(i)), [Stage(ge(X, "cd"), []), Stage(ge(X, f"cr{k}"), []),
                                                   Stage(le(X, reg(k), 0), fx(fire(f"fl{i}_{k}"), 1))]))
    pairs = {}

    def P(n):
        if n not in pairs:
            pairs[n] = output_pair(Y, n)
        return pairs[n]

    def apply(name, eff):
        Y.blocks.append(Block(lane(Y, name), [Stage(None, [e for n, w in eff for e in fx(P(n), w)])]))

    for i in range(N):
        apply(f"cl{i}", [(pc(i), -1)])
    for k in range(R):
        apply(f"ir{k}", [(reg(k), 1)])
        apply(f"dr{k}", [(reg(k), -1)])
    for j in range(N + 1):
        apply(f"gi{j}", [(pc(j), 1)])
        for k in range(R):
            apply(f"gd{j}_{k}", [(pc(j), 1)])
    for i in range(N):
        for k in range(R):
            apply(f"fl{i}_{k}", [(pc(i + 1), 1)])
    names = [pc(k) for k in range(N + 1)] + [reg(k) for k in range(R)]
    X.blocks.append(Block([Gate(X.port(pc(N), "floor"), OBSERVER, "floor")], [Stage(None, [])]))
    have = {key.split(":", 1)[1] for s in (F, X) for key in s.ports}
    for n in names:
        for m in (n, neg(n)):
            if m not in have:
                X.blocks.append(Block([Gate(X.port(m, "floor"), OBSERVER, "floor")], [Stage(None, [])]))
                have.add(m)
    state = {}
    for n in names + flags:
        v = 1 if n == pc(0) else 0
        state[n], state[neg(n)] = v, -v
    return [F, X, Y], state, names, flags, fires


def compile_stored(N: int, R: int, wad: Path, *, max_passes: int = 1_000_000):
    layers, state, names, flags, fires = _specs(N, R)
    n_dolls = sum(s.n_dolls + 1 for s in layers)
    consumed = {key.split(":", 1)[1] for s in layers for key in s.ports}
    fire_all = [n for n in fire_values(N, R)]
    flag_set = set(flags)
    readout = ([n for n in state if n.lstrip("-") not in flag_set] + [neg(f) for f in flags]
               + [neg(f) for f in fire_all if neg(f) in consumed])
    lights = [o for s in layers for o, _ in s.outputs if o not in consumed and o not in state]
    pages = -(-(len(lights) + len(readout)) // sg.PAGE)
    if 100 + pages > sg.LOOP:
        raise ValueError(f"stored machine: {pages} readout pages would reach the level's script {sg.LOOP}")
    hold = sg.hold_tics(pages)
    tags = sg.port_tags(layers)
    halt_tag = tags[f"floor:pc{N}"]

    def world(mb):
        mb.rect(-768, -512, -512, -256, floor=0, ceil=128)
        mb.thing(-640, -384, 1)
        return controller_acs(n_dolls, max_passes, hold, STORED_BOOT, halt_tag)

    meta = compile_pure(layers, wad, "stored", state=state, readout=readout, world=world)
    if meta.layout["tags"] != tags or meta.n_dolls != n_dolls:
        raise AssertionError("stored machine: the controller's plan and the emitted map differ")
    meta.layout["machine"] = dict(halt_tag=halt_tag, hold=hold, boot=STORED_BOOT, max_passes=max_passes,
                                  slots=N, registers=R, stored=True)
    meta.save(Path(wad).with_suffix(".json"))
    return meta


def loader(p: StoredProgram):
    """The host's initial-state writes for program `p`: its flags and its registers (both signs of every value)."""
    def load(game, meta):
        tags = meta.layout["tags"]
        writes = {name: v for name, v in p.flags().items() if v}
        writes.update({f"r{k}": v for k, v in enumerate(p.init) if v})
        for name, v in writes.items():
            for key, tag in tags.items():
                kind, val = key.split(":", 1)
                if val == name:
                    game.load(tag, v, kind)
                elif val == neg(name):
                    game.load(tag, -v, kind)
    return load


class StoredMachine:
    """One compiled map, any program that fits `slots` and `len(registers)` (module docstring)."""

    def __init__(self, slots: int = 12, registers=("a", "b", "c", "d")):
        self.slots, self.registers = slots, tuple(registers)

    def compile(self, wad):
        """Compile the one map into `wad` (and wad.json); returns the TileMeta."""
        return compile_stored(self.slots, len(self.registers), Path(wad))

    def assemble(self, source, registers: dict | None = None) -> StoredProgram:
        init = {n: 0 for n in self.registers}
        init.update(registers or {})
        unknown = set(init) - set(self.registers)
        if unknown:
            raise ValueError(f"unknown registers {sorted(unknown)}")
        return assemble_stored(source, self.slots, init)

    def run(self, wad, source, registers: dict | None = None, *, max_steps: int = 100_000, on_step=None):
        """Load `source` and `registers` into the compiled map `wad`, play, and check every step against `interpret`."""
        p = source if isinstance(source, StoredProgram) else self.assemble(source, registers)
        return play(Path(wad), p, interpret(p, max_steps), on_step=on_step, loader=loader(p))


# --- two programs for the same bytes --------------------------------------------------------------------------------
MULTIPLY = """
outer: dec a body             ; c = a * b, b kept in d
       jmp end
body:  dec b move
       jmp rest
move:  inc c next
next:  inc d body
rest:  dec d back
       jmp outer
back:  inc b rest
end:   halt"""

POWER = """
outer: dec a dbl              ; c = 2 ** a: double c a times (c starts 1)
       jmp end
dbl:   dec c two
       jmp back
two:   inc d again
again: inc d dbl
back:  dec d unload
       jmp outer
unload: inc c back
end:   halt"""
