"""Random SUBLEQ programs on the one compiled map, every step against the interpreter at tolerance 0.

A program is 18 random cells: values from -64..63 biased to the edges (0, +-1, +-8, +-63, -64), instruction operands that
are valid addresses (so the program also writes into its own instructions) and jump targets in -1..7. The interpreter runs
the program until it halts, leaves the memory or the value range (the map is never asked to run that step), or reaches
STEPS steps. Seeds are printed; the same seed gives the same programs.

    python models/subleq/fuzz.py --programs 12 --steps 20 --seed 1 --workers 3 [--clock dolls] [--latch]
"""
from __future__ import annotations

import argparse
import random
import tempfile
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures.process import BrokenProcessPool
from pathlib import Path

from doomformer.machine import sha256
from doomformer.subleq import SubleqMachine, SubleqProgram, interpret

EDGE = [-64, -63, -9, -8, -7, -1, 0, 1, 7, 8, 9, 62, 63]


def random_program(rng: random.Random, slots: int = 6) -> SubleqProgram:
    n = 3 * slots
    code = 3 * (slots - 2)
    mem = []
    for j in range(n):
        if j < code and j % 3 != 2:                               # an address operand: mostly the data cells, sometimes code
            mem.append(rng.randrange(code, n) if rng.random() < 0.8 else rng.randrange(n))
        elif j < code:                                            # a jump target
            mem.append(rng.choice([-1, 0, 1, 2, 3, 4, 5, 6, 7]))
        else:                                                     # data: small values, sometimes an edge of -64..63
            mem.append(rng.choice(EDGE) if rng.random() < 0.3 else rng.randrange(-4, 5))
    return SubleqProgram(tuple(mem), slots)


def run_one(args):
    from doomformer.machine import play
    from doomformer.subleq import loader
    wad, seed, steps, clock, latch = args
    p = random_program(random.Random(seed))
    ref = interpret(p, steps, strict=False, latch=latch)
    if len(ref) < 2:
        return seed, 0, 0, "no runnable step"
    cls = None
    if clock == "dolls":
        from doomformer.dollclock import DollClock as cls
    from doomformer.machine import MachineMismatch
    try:
        r = play(Path(wad), p, ref, loader=loader(p), clock_cls=cls)
    except MachineMismatch as e:
        return seed, len(ref) - 1, 0, f"MISMATCH at step {e.step}: {e}"
    flagged = " (the map flagged the own-cell write at step %d)" % ref[-1].idx if ref[-1].fires.get("viol") else ""
    return seed, len(ref) - 1, r.checked, "EXACT" + flagged + (" (halted)" if ref[-1].pc == p.halted else "")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--programs", type=int, default=12)
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--clock", choices=["acs", "dolls"], default="acs")
    ap.add_argument("--violating", action="store_true", help="run only the programs that write their own executing instruction")
    ap.add_argument("--latch", action="store_true",
                    help="also run the programs that write their own executing instruction: the map's latch must flag them "
                         "at the interpreter's step")
    a = ap.parse_args()
    with tempfile.TemporaryDirectory() as tmp:
        wad = Path(tmp) / "subleq.wad"
        SubleqMachine(6).compile(wad, clock=a.clock)
        digest = sha256(wad)
        print(f"one map, clock {a.clock}, sha256 {digest}; {a.programs} random programs, seeds {a.seed}.."
              f"{a.seed + a.programs - 1}, at most {a.steps} steps each", flush=True)
        seeds = range(a.seed, a.seed + a.programs)
        if a.violating:                           # only the seeds whose program writes its own executing instruction
            seeds = [s for s in seeds if interpret(random_program(random.Random(s)), a.steps, strict=False,
                                                   latch=True)[-1].fires.get("viol")]
            print(f"  {len(seeds)} of {a.programs} programs write their own executing instruction: seeds {seeds}", flush=True)
        jobs = [(str(wad), s, a.steps, a.clock, True) for s in seeds] if a.violating else \
            [(str(wad), s, a.steps, a.clock, a.latch) for s in seeds]
        total_steps = total_values = bad = 0
        pending = list(jobs)
        for attempt in range(3):                  # a worker the system kills (memory pressure) is retried, never skipped
            if not pending:
                break
            done = []
            try:
                with ProcessPoolExecutor(a.workers) as ex:
                    for job, (seed, steps, values, verdict) in zip(pending, ex.map(run_one, pending)):
                        done.append(job)
                        total_steps += steps
                        total_values += values
                        bad += not verdict.startswith(("EXACT", "no runnable"))
                        print(f"  seed {seed}: {steps} steps, {values:,} values, {verdict}", flush=True)
            except BrokenProcessPool:
                print(f"  (a worker process was killed; retrying {len(pending) - len(done)} programs)", flush=True)
            pending = [j for j in pending if j not in done]
        if pending:
            raise SystemExit(f"{len(pending)} programs never ran: seeds {[j[1] for j in pending]}")
        assert sha256(wad) == digest
        print(f"{a.programs} programs, {total_steps} steps, {total_values:,} values compared at tolerance 0, "
              f"{bad} mismatches; map sha256 unchanged")
        if bad:
            raise SystemExit(1)


if __name__ == "__main__":
    main()
