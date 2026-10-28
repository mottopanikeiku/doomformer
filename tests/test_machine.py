"""df.Machine: a counter machine compiled into one DOOM map, every step of the trace checked against a Python
interpreter at tolerance 0. Covers the assembler and interpreter (no game), a ROM program, multi-limb carry, the ROM
tamper negative control (the trace diverges at the predicted step), the stored-program machine (same WAD bytes, two
programs, a mismatch when the loaded program is not the reference one) and the CLI. At most one game at a time."""
from __future__ import annotations

import hashlib

import pytest

from doomformer import cli
from doomformer.machine import MachineMismatch, assemble, interpret, play, tamper
from doomformer.machine_programs import PROGRAMS
from doomformer.stored import StoredMachine
from doomformer.stored import interpret as stored_interpret

ADD = """
top:  jzdec b done
      inc a
      jmp top
done: halt"""


def test_assembler_and_interpreter():
    p = assemble(ADD, {"a": 3, "b": 4})
    assert p.code == (("jzdec", 1, 3), ("inc", 0), ("jmp", 0), ("jmp", 4))
    tr = interpret(p)
    assert len(tr) - 1 == 14 and tr[-1].pc == p.halted and p.logical(tr[-1].regs) == {"a": 7, "b": 0}
    assert tr[1].fires == {"n0": 1} and tr[0].fires == {}
    for bad in ("foo a", "inc zz", "jmp nowhere", "inc a\ninc a b"):
        with pytest.raises(ValueError):
            assemble(bad, {"a": 0})
    with pytest.raises(ValueError, match="twice"):
        assemble("x: inc a\nx: halt", {"a": 0})


def test_ready_programs_interpret_correctly():
    """Every ready program equals Python's own answer (no game): the interpreter is what DOOM is compared with."""
    for name, args in [("add", (3, 4)), ("multiply", (7, 9)), ("power", (2, 6)), ("gcd", (48, 18)), ("gcd", (7, 0)),
                       ("fib", (20,)), ("fib", (0,)), ("pow2", (40,)), ("collatz", (7,)), ("primes", (12,))]:
        m, reg, want, _ = PROGRAMS[name].build(*args)
        tr = interpret(m.program, 1_000_000)
        assert tr[-1].pc == m.program.halted
        assert m.program.logical(tr[-1].regs)[reg] == want, (name, args)
    m, *_ = PROGRAMS["multiply"].build(7, 9)
    assert len(interpret(m.program)) - 1 == 464


def test_limits_are_refused():
    with pytest.raises(ValueError, match="outside 0"):                      # addr source above base - 1
        interpret(assemble("addr a b", {"a": 1, "b": 20}))
    with pytest.raises(ValueError, match="left"):
        interpret(assemble("loop: add a 5000\njmp loop", {"a": 0}), 10)
    with pytest.raises(ValueError):
        assemble("halt", {"a": 99999})


def test_rom_program_in_doom_is_exact(tmp_path):
    m, reg, want, _ = PROGRAMS["multiply"].build(2, 3)
    r = m.run(tmp_path / "mul.wad")
    assert r.registers[reg] == want == 6 and r.halted and r.steps == 50
    assert r.trace[0] == (0, (2, 3, 0, 0)) and r.trace[-1] == (m.program.halted, (0, 3, 6, 0))
    a = r.audit
    assert a["non_readout_commands"] == 0 and a["forbidden"] == {} and a["loaded"] == 0 and a["sealed"]
    assert set(a["commands"]) <= {"puke -100", "puke -901", "puke -8"}
    assert r.checked > 0


def test_multilimb_carry_and_branches_in_doom(tmp_path):
    """2^10 in 4 base-8 limbs: ripple carries across limbs; jge/jlt/add are exercised by the carry macro."""
    m, reg, want, _ = PROGRAMS["pow2"].build(10, base=8)
    assert want == 1024 and len(m.program.limbs["a"]) == 4
    r = m.run(tmp_path / "pow2.wad")
    assert r.registers[reg] == 1024 and r.halted
    assert r.trace[-1][1][:4] == (0, 0, 0, 2)                              # 1024 = 2 * 8^3: the carry reached limb 3
    assert r.audit["non_readout_commands"] == 0


def test_tamper_diverges_at_the_predicted_step(tmp_path):
    m, reg, want, _ = PROGRAMS["multiply"].build(2, 3)
    p, ref = m.program, interpret(m.program)
    for k, what in [(2, "weight"), (1, "target")]:
        wad = tmp_path / f"t{k}.wad"
        m.compile(wad)
        info = tamper(wad, p, k, what=what)
        with pytest.raises(MachineMismatch) as e:
            play(wad, p, ref)
        assert info["predicted_step"] is not None and e.value.step == info["predicted_step"]
    clean = tmp_path / "clean.wad"
    m.compile(clean)
    assert play(clean, p, ref).registers[reg] == want                       # the untampered twin passes


SIX = dict(slots=6, registers=("a", "b"))
ADD_STORED = """
loop: dec b body
      halt
body: inc a loop"""
DOUBLE_STORED = """
loop: dec a body
      halt
body: inc b mid
mid:  inc b loop"""


def test_stored_program_same_wad_two_programs(tmp_path):
    sm = StoredMachine(**SIX)
    wad = tmp_path / "stored.wad"
    sm.compile(wad)
    digest = hashlib.sha256(wad.read_bytes()).hexdigest()
    ra = sm.run(wad, ADD_STORED, {"a": 2, "b": 3})
    rb = sm.run(wad, DOUBLE_STORED, {"a": 3})
    assert hashlib.sha256(wad.read_bytes()).hexdigest() == digest
    assert ra.registers == {"a": 5, "b": 0} and rb.registers == {"a": 0, "b": 6}
    assert ra.steps != rb.steps and ra.wad_sha256 == rb.wad_sha256 == digest
    for r in (ra, rb):
        a = r.audit
        assert a["non_readout_commands"] == 0 and a["sealed"] and a["loaded"] > 0
        assert set(a["commands"]) <= {f"puke -{100 + i}" for i in range(20)} | {"puke -901", "puke -8"}
    # the loaded program is what runs: program B's memory checked against program A's trace fails at step 1
    pa, pb = sm.assemble(ADD_STORED, {"a": 3}), sm.assemble(DOUBLE_STORED, {"a": 3})
    from doomformer.stored import loader
    with pytest.raises(MachineMismatch) as e:
        play(wad, pa, stored_interpret(pa), loader=loader(pb))
    assert e.value.step == 0


def test_cli_machine(capsys):
    assert cli.main(["machine", "list"]) == 0
    assert cli.main(["machine", "add", "2", "1"]) == 0
    out = capsys.readouterr().out
    assert "a = 3 (expected 3)" in out and "EXACT:" in out and "0 other commands" in out
    assert cli.main(["machine", "add", "1", "1", "--tamper", "1"]) == 0
    assert "TAMPER DETECTED" in capsys.readouterr().out
