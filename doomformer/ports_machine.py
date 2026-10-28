"""The doll-clock counter machine and the SUBLEQ machine, exported for other ZDoom-family ports.

`doomformer machine multiply 7 9 --clock dolls --export DIR` (and `... subleq add 3 4 --clock dolls --export DIR`) writes

    DIR/machine.wad    the map `--clock dolls` compiles, with one more ACS script, a LOGGER (script 40, OPEN), only in
                       this export, never in a default build. The map's own doll clock runs the machine (a loop doll and
                       line specials; no OPEN script in the default build). The logger does what the Python runner does and
                       sends no command: it waits until the counter `pass` equals k, waits until no doll moves for 8 tics
                       (the stillness test of selfgen.STILL), and then logs every value the Python runner reads, in one
                       tic:   FACT machine step=k pc0=0 pc1=1 ... r0=7 ... -f0=0 ...
                       Step 0 is the initial state (logged at tic 2). A step that does not come within STALL tics logs
                       `FACT machine step=k stalled pass=P` and stops; the last step ends with ConsoleCommand("quit").
                       For SUBLEQ the logger first writes the program's initial memory (the host's writes of
                       subleq.loader, as Floor_MoveToValue / Ceiling_MoveToValue calls).
    DIR/expected.txt   the same lines, computed by the machine's interpreter (`Program.want`): no engine produced them.

A port that prints identical lines runs the doll clock for the whole trace exactly. `python -m doomformer.ports_machine
--run-vizdoom WAD` runs the export in ViZDoom (the engine console on stdout) for comparison.
"""
from __future__ import annotations

import contextlib
import re
import sys
from pathlib import Path

from .machine import _wad_bytes, _wad_lumps
from .selfgen import STILL
from .tile import DOLL_TID0, MOVE_SPEED
from .udmf import compile_acs

LOG_TAG = "FACT"
CASE = "machine"
STALL = 6000                 # tics a step may wait for the counter before the logger reports `stalled`


@contextlib.contextmanager
def tags_below_2_15():
    """Build with every sector tag and TID below 32,768: the stage and freeze tag bases of pure.py (60,000 and 64,000), the
    doll clock's loop barrier tag (90,000) and its teleport TIDs (47,000 ...). Zandronum 3.2.1 does not find sector tags
    from 32,768 up (docs/engine_limits.md, "Other ports"); the default build is untouched."""
    from . import dollclock, pure
    names = [(pure, "STAGE_TAG0", 25000), (pure, "FREEZE_TAG0", 30000), (dollclock, "STAGE_TAG0", 25000),
             (dollclock, "LOOP_BARRIER", 31000), (dollclock, "LOOP_HOME_TID", 32000), (dollclock, "ROW_TID0", 32100),
             (dollclock, "LOOP_TID", 32500)]
    saved = [(m, n, getattr(m, n)) for m, n, _ in names]
    for m, n, v in names:
        setattr(m, n, v)
    try:
        yield
    finally:
        for m, n, v in saved:
            setattr(m, n, v)


def _check_low(wad: Path) -> None:
    ids = [int(v) for v in re.findall(rb"\bid = (\d+);", wad.read_bytes())]
    if not ids or max(ids) > 32767:
        raise AssertionError(f"an id of {max(ids)} remains")


def logger_acs(meta, steps: int, writes: list[tuple[str, int, int]] = (), stall: int = STALL) -> str:
    """ACS of the logger. `writes`: (kind, tag, value) initial-state writes (subleq); `steps`: the last step number."""
    n = meta.n_dolls
    names = list(meta.outputs)
    reads = meta.layout["read_expressions"]
    passv = reads["pass"]
    line = lambda k: (f's:"{LOG_TAG} {CASE} step=", d:{k}' + "".join(f', s:" {nm}=", d:{reads[nm]}' for nm in names))
    wr = "\n".join(f"  {'Floor' if kind == 'floor' else 'Ceiling'}_MoveToValue({tag}, {MOVE_SPEED}, {abs(v)}, {int(v < 0)});"
                   for kind, tag, v in writes)
    return f"""
int lg_x[{n}];
int lg_y[{n}];
int lg_z[{n}];
function int lg_moved(void) {{
  int moved = 0;
  for (int i = 0; i < {n}; i++) {{
    int tid = {DOLL_TID0} + i;
    if (!IsTIDUsed(tid)) {{ moved++; continue; }}
    int px = GetActorX(tid);
    int py = GetActorY(tid);
    int pz = GetActorZ(tid);
    if (px != lg_x[i] || py != lg_y[i] || pz != lg_z[i]) {{ moved++; lg_x[i] = px; lg_y[i] = py; lg_z[i] = pz; }}
  }}
  return moved;
}}
script 40 OPEN {{
  int k;
  int tic;
  int m;
{wr}
  Delay(2);
  Log({line(0)});
  for (k = 1; k <= {steps}; k++) {{
    tic = 0;
    while ({passv} < k) {{
      Delay(1);
      tic++;
      if (tic > {stall}) {{
        Log(s:"{LOG_TAG} {CASE} step=", d:k, s:" stalled pass=", d:{passv});
        Terminate;
      }}
    }}
    lg_moved();
    do {{
      Delay({STILL});
      m = lg_moved();
    }} while (m != 0 && {passv} == k);
    if ({passv} != k) {{
      Log(s:"{LOG_TAG} {CASE} step=", d:k, s:" missed pass=", d:{passv});
      Terminate;
    }}
    Log({line('k')});
  }}
  Delay(2);
  ConsoleCommand("quit");
}}
"""


def expected_lines(p, trace, outputs) -> list[str]:
    return [f"{LOG_TAG} {CASE} step={k}" + "".join(f" {n}={v}" for n, v in p.want(step, outputs).items())
            for k, step in enumerate(trace)]


def subleq_writes(p, meta) -> list[tuple[str, int, int]]:
    """The host's initial-state writes of subleq.loader as (kind, tag, value)."""
    from .ops import neg
    tags = meta.layout["tags"]
    out = []
    for i, v in enumerate(p.digits()):
        if not v:
            continue
        for key, tag in tags.items():
            kind, val = key.split(":", 1)
            if val == f"r{i}":
                out.append((kind, tag, v))
            elif val == neg(f"r{i}"):
                out.append((kind, tag, -v))
    return out


def _write_export(out: Path, wad: Path, meta, p, trace, writes) -> list[str]:
    dc = meta.layout["machine"]["dollclock"]
    lumps = _wad_lumps(wad.read_bytes())
    bh = next(i for i, (nm, _) in enumerate(lumps) if nm == "BEHAVIOR")
    lumps[bh] = ("BEHAVIOR", compile_acs(dc["acs"] + logger_acs(meta, len(trace) - 1, writes)))
    final = out / "machine.wad"
    final.write_bytes(_wad_bytes(lumps))
    lines = expected_lines(p, trace, list(meta.outputs))
    (out / "expected.txt").write_text("\n".join(lines) + "\n")
    return lines


def export(out: Path, program: str = "multiply", args: tuple = (7, 9), *, base: int = 16, max_steps: int = 200_000,
           low_tags: bool = False) -> dict:
    """Write DIR/machine.wad and DIR/expected.txt for a counter program or `subleq` (program "subleq": args = (name,
    *numbers)). low_tags: every sector tag and TID below 32,768 (tags_below_2_15). Returns what the CLI prints."""
    with tags_below_2_15() if low_tags else contextlib.nullcontext():
        return _export(Path(out), program, args, base, max_steps, low_tags)


def _export(out: Path, program: str, args: tuple, base: int, max_steps: int, low_tags: bool) -> dict:
    import tempfile
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="doomformer-ports-machine-") as tmp:
        wad = Path(tmp) / "machine.wad"
        if program == "subleq":
            from .subleq import READY, SubleqMachine, interpret
            which, nums = args[0], [int(a) for a in args[1:]]
            if which not in READY:
                raise ValueError(f"subleq program {which!r}: one of {', '.join(READY)}")
            make, need, *_ = READY[which]
            if len(nums) not in (0, need):
                raise ValueError(f"subleq {which} takes {need} integer argument(s)")
            sm = SubleqMachine()
            meta = sm.compile(wad, clock="dolls")
            p = sm.assemble(make(*nums) if nums else make())
            trace = interpret(p, max_steps)
            lines = _write_export(out, wad, meta, p, trace, subleq_writes(p, meta))
            title = f"subleq {which} {' '.join(map(str, nums))}".strip()
        else:
            from .machine_programs import PROGRAMS
            if program not in PROGRAMS:
                raise ValueError(f"unknown program {program!r}")
            kw = {"base": base} if program in ("fib", "pow2") else {}
            machine, *_ = PROGRAMS[program].build(*[int(a) for a in args], **kw)
            meta = machine.compile(wad, clock="dolls")
            p = machine.program
            trace = machine.reference(max_steps)
            lines = _write_export(out, wad, meta, p, trace, [])
            title = f"{program} {' '.join(map(str, args))}"
    if low_tags:
        _check_low(out / "machine.wad")
    steps = len(trace) - 1
    return dict(title=title, steps=steps, values=len(lines) * len(meta.outputs), per_step=len(meta.outputs),
                dolls=meta.n_dolls, sectors=meta.n_sectors, halted=trace[-1].pc == p.halted,
                estimate=meta.layout["estimated_settle_tics"], wad=str(out / "machine.wad"))


def export_cli(args) -> int:
    """`doomformer machine PROGRAM ARGS --clock dolls --export DIR`."""
    if args.clock != "dolls" or args.blind or args.tamper is not None:
        print("doomformer machine --export: needs --clock dolls and no --blind or --tamper", file=sys.stderr)
        return 2
    try:
        r = export(Path(args.export), args.program, tuple(args.args), base=args.base, max_steps=args.max_steps,
                   low_tags=args.low_tags)
    except (ValueError, KeyError) as e:
        print(f"doomformer machine --export: {e}", file=sys.stderr)
        return 2
    print(f"{r['title']}: {r['steps']} steps; {r['dolls']:,} voodoo dolls, {r['sectors']:,} sectors; "
          f"{r['per_step']} values per step logged by ACS script 40 (the logger, only in this export)\n"
          f"wrote {args.export}/machine.wad and expected.txt ({r['steps'] + 1} lines, {r['values']:,} values, from the "
          f"interpreter). Run the WAD in a port and diff its `FACT machine` lines with expected.txt "
          f"(docs/engine_limits.md, 'The machine')." + (" Every sector tag and TID is below 32,768." if args.low_tags else ""))
    return 0


# The same-tic probe (models/dollclock/probe_same_tic.py) as exported maps ---------------------------------------------------
SAME_TIC_EXPECTED = {4: 14, 16: 0}     # ViZDoom 1.3.2: dolls (of 24) not teleported home by tic 500, chain cells 4 and 16 wide
SAME_TIC_FAR = 60                      # units from the doll's position at tic 2 that count as "not home"


def export_same_tic(out: Path) -> list[str]:
    """Four WADs: cells 4 and 16, each with the default tags and with every tag below 32,768 (`_tags15`)."""
    lines = []
    for low in (False, True):
        with tags_below_2_15() if low else contextlib.nullcontext():
            lines += _export_same_tic(Path(out), "_tags15" if low else "")
    Path(out, "expected.txt").write_text("\n".join(lines) + "\n")
    return lines


def _export_same_tic(out: Path, suffix: str) -> list[str]:
    """`same_tic_cell4.wad` and `same_tic_cell16.wad` plus expected.txt in `out`: the counter machine `add 3 4` with the
    doll clock's rewind chain built from cells 4 and 16 units wide. A logger script records every doll's x at tic 2 and
    logs, at tic 500, how many dolls are more than 60 units away (those were not teleported home by their TeleportOther
    line, because two line specials crossed in the same tic do not both act in ZDoom 2.8.1). Expected values: ViZDoom's."""
    import tempfile

    from . import dollclock
    from .machine_programs import PROGRAMS
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    lines = []
    saved = dollclock.CELL
    try:
        for cell, want in SAME_TIC_EXPECTED.items():
            dollclock.CELL = cell
            machine, *_ = PROGRAMS["add"].build(3, 4)
            with tempfile.TemporaryDirectory(prefix="doomformer-same-tic-") as tmp:
                wad = Path(tmp) / "m.wad"
                meta = machine.compile(wad, clock="dolls")
                n = meta.n_dolls
                acs = meta.layout["machine"]["dollclock"]["acs"] + f"""
int st_x[{n}];
script 41 OPEN {{
  int far = 0;
  Delay(2);
  for (int i = 0; i < {n}; i++) {{ st_x[i] = GetActorX({DOLL_TID0} + i) >> 16; }}
  Delay(498);
  for (int j = 0; j < {n}; j++) {{
    int d = (GetActorX({DOLL_TID0} + j) >> 16) - st_x[j];
    if (d < 0) {{ d = -d; }}
    if (d > {SAME_TIC_FAR}) {{ far++; }}
  }}
  Log(s:"{LOG_TAG} same_tic_cell{cell}{suffix} far=", d:far);
  ConsoleCommand("quit");
}}
"""
                lumps = _wad_lumps(wad.read_bytes())
                bh = next(i for i, (nm, _) in enumerate(lumps) if nm == "BEHAVIOR")
                lumps[bh] = ("BEHAVIOR", compile_acs(acs))
                (out / f"same_tic_cell{cell}{suffix}.wad").write_bytes(_wad_bytes(lumps))
            lines.append(f"{LOG_TAG} same_tic_cell{cell}{suffix} far={want}")
    finally:
        dollclock.CELL = saved
    return lines


def run_in_vizdoom(wad: Path, max_tics: int = 1_000_000) -> int:
    """Run the export in ViZDoom with the console on stdout for `max_tics` tics (the logger's quit is refused there too)."""
    from .facts import _new_game
    g = _new_game(Path(wad), console=True)
    try:
        for _ in range(max_tics // 50):
            g.advance_action(50)
    except Exception:                      # an engine crash or exit ends the run
        pass
    return 0


if __name__ == "__main__":
    if len(sys.argv) >= 3 and sys.argv[1] == "--run-vizdoom":
        sys.exit(run_in_vizdoom(Path(sys.argv[2]), int(sys.argv[3]) if len(sys.argv) > 3 else 1_000_000))
    print(__doc__)
