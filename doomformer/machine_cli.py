"""`doomformer machine`: run a ready counter-machine program in ONE DOOM map and check every step (docs/machine.md)."""
from __future__ import annotations

import sys
import tempfile
import time
from pathlib import Path

from .machine_programs import PROGRAMS


def add_parser(sub) -> None:
    names = ", ".join(PROGRAMS) + ", stored"
    m = sub.add_parser("machine", help="a counter machine inside ONE DOOM map: registers are plane heights, the program "
                                       "is geometry; every step is checked against a Python interpreter (tolerance 0)",
                       description="Runs a program on a counter machine compiled into one DOOM map. Registers and the "
                                   "one-hot program counter are state values (plane heights) changed by voodoo dolls; "
                                   "the program is map geometry; a halt latch ends the run. Python advances the clock "
                                   "and reads, and compares the program counter and every register after every step "
                                   "with its own interpreter (tolerance 0). The level's ACS script is the clock and the "
                                   "halt check (docs/machine.md).")
    m.add_argument("program", nargs="?", default="list", help=f"one of: {names}, or list")
    m.add_argument("args", nargs="*", type=int, help="integer arguments (list shows them), e.g. multiply 7 9")
    m.add_argument("--base", type=int, default=16, help="limb base of the multi-limb programs fib and pow2 (default 16)")
    m.add_argument("--tamper", type=int, metavar="K", default=None,
                   help="change one line of instruction K in the compiled map first; the run must diverge at the step "
                        "where the interpreter first executes that line (negative control)")
    m.add_argument("--tamper-what", choices=["weight", "target"], default="weight",
                   help="weight: the register line gets one more unit; target: the next pc is pc_halt")
    m.add_argument("--max-steps", type=int, default=200_000, help="give up if the interpreter has not halted by then")
    m.add_argument("--wad", metavar="PATH", help="keep the compiled map here")


def _listing() -> int:
    print("programs (arguments in order; every step is one pass of the level's loop, about 200 to 600 game tics):")
    for s in PROGRAMS.values():
        print(f"  {s.name:9s} {' '.join(s.args):5s}  {s.blurb}")
    print("  stored              one compiled map, two programs (multiply, power) loaded as initial state")
    print("\nexamples: doomformer machine multiply 7 9 | fib 20 | gcd 48 18 | primes 12 | stored")
    return 0


def _stored(args) -> int:
    from . import machine
    from .stored import MULTIPLY, POWER, StoredMachine, interpret
    sm = StoredMachine(slots=12, registers=("a", "b", "c", "d"))
    cases = [("multiply", MULTIPLY, {"a": 2, "b": 3}, "c", 6, "2 * 3"), ("power", POWER, {"a": 3, "c": 1}, "c", 8, "2 ^ 3")]
    with tempfile.TemporaryDirectory(prefix="doomformer-stored-") as tmp:
        wad = Path(args.wad or Path(tmp) / "stored.wad")
        meta = sm.compile(wad)
        digest = machine.sha256(wad)
        print(f"one map, {meta.n_dolls:,} dolls, {meta.n_sectors:,} sectors, sha256 {digest[:16]}...\n"
              f"the program is the initial height of one-hot flag values; the host writes them before the first tic")
        for name, src, regs, reg, want, text in cases:
            p = sm.assemble(src, regs)
            steps = len(interpret(p)) - 1
            print(f"\nprogram {name} ({text}): {steps} steps by the interpreter ...", flush=True)
            r = sm.run(wad, p)
            same = machine.sha256(wad) == digest
            ok = r.registers[reg] == want and same
            print(f"  {reg} = {r.registers[reg]} (expected {want}) | {r.summary()}\n"
                  f"  map sha256 after the run equals before: {same}")
            if not ok:
                return 1
        print(f"\nEXACT: both programs ran on the same map bytes ({digest}); every step's pc, registers, flags and "
              f"fire values equal the interpreter's at tolerance 0")
    return 0


def run(args) -> int:
    name = args.program
    if name == "list":
        return _listing()
    if name == "stored":
        return _stored(args)
    if name not in PROGRAMS:
        print(f"doomformer machine: unknown program {name!r}; try `doomformer machine list`", file=sys.stderr)
        return 2
    spec = PROGRAMS[name]
    if len(args.args) != len(spec.args):
        print(f"doomformer machine {name}: takes {len(spec.args)} argument(s): {' '.join(spec.args)}", file=sys.stderr)
        return 2
    kw = {"base": args.base} if name in ("fib", "pow2") else {}
    machine, reg, expect, text = spec.build(*args.args, **kw)
    p = machine.program
    ref = machine.reference(args.max_steps)
    steps = len(ref) - 1
    if ref[-1].pc != p.halted:
        print(f"doomformer machine: the interpreter did not halt within {args.max_steps} steps", file=sys.stderr)
        return 1
    want = p.logical(ref[-1].regs)[reg]
    print(f"{name}: {text}\n  {len(p.code)} instructions, registers {' '.join(p.regs)}; the interpreter halts after "
          f"{steps:,} steps with {reg} = {want} (Python computes {expect})", flush=True)
    if want != expect:
        print("doomformer machine: the program is wrong for these arguments (interpreter != Python)", file=sys.stderr)
        return 1
    from .machine import MachineMismatch, play, tamper
    every = max(1, steps // 10)
    t0 = time.time()

    def progress(k, step):
        if k % every == 0:
            print(f"  step {k:,}/{steps:,}  pc {step.pc}  {p.logical(step.regs)}  ({time.time() - t0:.0f} s)", flush=True)

    with tempfile.TemporaryDirectory(prefix="doomformer-machine-") as tmp:
        wad = Path(args.wad or Path(tmp) / "machine.wad")
        meta = machine.compile(wad)
        print(f"  map: {meta.n_dolls:,} voodoo dolls, {meta.n_sectors:,} sectors, span {meta.layout['span'][0]:,} x "
              f"{meta.layout['span'][1]:,}", flush=True)
        change = None
        if args.tamper is not None:
            try:
                change = tamper(wad, p, args.tamper, what=args.tamper_what)
            except ValueError as e:
                print(f"doomformer machine: {e}", file=sys.stderr)
                return 2
            print(f"  TAMPER: instruction {args.tamper} ({p.lines[args.tamper]}): {change['changed']}; the interpreter "
                  f"first executes it at step {change['predicted_step']}", flush=True)
        try:
            r = play(wad, p, ref, on_step=progress)
        except MachineMismatch as e:
            if change is None:
                print(f"MISMATCH at step {e.step}: {e}")
                return 1
            ok = e.step == change["predicted_step"]
            print(f"  diverged at step {e.step}: {e}\nTAMPER {'DETECTED' if ok else 'MISSED THE PREDICTION'}: first "
                  f"differing step {e.step}, predicted {change['predicted_step']}")
            return 0 if ok else 1
        except RuntimeError as e:
            print(f"MISMATCH: {e}")
            return 1
    if change is not None:
        print("TAMPER NOT DETECTED: the tampered map ran the whole trace exactly")
        return 1
    a = r.audit
    print(f"\n{reg} = {r.registers[reg]} (expected {expect})\n"
          f"  {r.steps:,} steps | {r.dolls:,} dolls | {r.sectors:,} sectors | {r.tics:,} game tics "
          f"({min(r.pass_tics):,} to {max(r.pass_tics):,} per step) | {r.seconds:.0f} s\n"
          f"  audit: {sum(a['commands'].values()):,} readout commands, {a['non_readout_commands']} other commands or "
          f"game calls, {a['loaded']} state loads (the map starts with the program and the arguments baked in)\n"
          f"  clock: the level's ACS script (rewind, barrier release, settle check, halt-latch read); data path and "
          f"program counter: dolls\n"
          f"EXACT: the pc and every register of all {r.steps:,} steps equal the interpreter's ({r.checked:,} values, "
          f"tolerance 0); the halt latch stopped the level")
    return 0 if r.registers[reg] == expect else 1
