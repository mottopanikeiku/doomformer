"""SUBLEQ machine (doomformer/subleq.py) and the doll-driven clock (doomformer/dollclock.py).

No game: the assembler, the interpreter (the reference every DOOM step is compared with), the readout-only ACS lump.
One game at a time: SUBLEQ add on the one compiled map (every step exact, the map's sha256 unchanged), a decode-line
tamper that must diverge at the predicted step (step 1; the selfmod run's tamper at step 4 is logged in models/subleq), and the counter machine with no OPEN script (add, tamper). The long runs
(multiply 7 x 9, the self-modifying program: 20 and 10 steps, 2 to 3 minutes each) need DOOMFORMER_HEAVY=1."""
from __future__ import annotations

import os
import re

import pytest

from doomformer import machine as M
from doomformer.machine import MachineMismatch
from doomformer.machine_programs import PROGRAMS
from doomformer.subleq import (ADD, MULTIPLY, SELFMOD, SubleqMachine, SubleqProgram, assemble_subleq, interpret, split,
                               tamper)

HEAVY = os.environ.get("DOOMFORMER_HEAVY") == "1"


def test_split_and_assembler():
    assert [split(v) for v in (0, 7, 8, -1, -64, 63)] == [(0, 0), (7, 0), (0, 1), (7, -1), (0, -8), (7, 7)]
    p = assemble_subleq(ADD(3, 4))
    assert p.mem[:12] == (12, 14, 1, 13, 14, 2, 14, 15, 3, 14, 14, -1) and p.mem[12:16] == (3, 4, 0, 0)
    assert p.N == 18 and p.n_code == 4 and dict(p.symbols) == {"a": 12, "b": 13, "z": 14, "c": 15}
    q = assemble_subleq(SELFMOD())
    assert q.mem[3:6] == (16, 0, 2)                  # bump: m1, rd.a (cell 0), count (slot 2)
    for bad in ("subleq a b", ".cell x", "nonsense", "subleq x y z\n.cell q 99", "l: subleq 0 0 0\nl: subleq 0 0 0"):
        with pytest.raises(ValueError):
            assemble_subleq(bad)
    with pytest.raises(ValueError, match="do not fit"):
        assemble_subleq("\n".join(["subleq 0 0 0"] * 7))


def test_interpreter_programs():
    def final(src):
        tr = interpret(assemble_subleq(src))
        p = assemble_subleq(src)
        assert tr[-1].pc == p.halted
        return len(tr) - 1, p.logical(tr[-1].regs)
    steps, m = final(ADD(3, 4))
    assert steps == 4 and m["m15"] == 7 and m["m14"] == 0
    steps, m = final(MULTIPLY(7, 9))
    assert steps == 20 and m["m10"] == 63
    steps, m = final(SELFMOD())
    assert steps == 10 and m["m15"] == -15
    tr = interpret(assemble_subleq(SELFMOD()))
    cell0 = [p for p in ([assemble_subleq(SELFMOD()).logical(s.regs)["m0"] for s in tr])]
    assert cell0 == [12, 12, 13, 13, 13, 14, 14, 14, 15, 15, 15]            # the instruction's own operand is rewritten


def test_semantics_and_refusals():
    # mem[b] - mem[a] <= 0 jumps to slot c, otherwise falls through to the next slot
    src = "subleq a, b, skip\nsubleq a, a, halt\nskip: subleq z, z, halt\n.cell a 3\n.cell b {}\n.cell z 0"
    assert [s.pc for s in interpret(assemble_subleq(src.format(3)))] == [0, 2, 6]
    assert [s.pc for s in interpret(assemble_subleq(src.format(9)))] == [0, 1, 6]
    with pytest.raises(ValueError, match="outside 0..17"):
        interpret(assemble_subleq("subleq 18 0 halt"))
    with pytest.raises(ValueError, match="outside -64..63"):
        interpret(assemble_subleq("subleq a, b, halt\n.cell a -60\n.cell b 20"))
    with pytest.raises(ValueError, match="outside -64..63"):
        assemble_subleq(".cell a 64")
    # a step that writes one of the executing instruction's own cells is refused (fuzz seed 122 of models/subleq/fuzz.py:
    # DOOM kept a ceiling too high there); the same write into another slot's cell is run
    own = SubleqProgram((16, 2, 4, 15, 5, 0, 17, 17, 5, 17, 10, 3, 3, 63, 4, -3, 3, 1), 6)
    with pytest.raises(ValueError, match="writes its own instruction cell 2"):
        interpret(own)
    assert len(interpret(own, 5, strict=False)) == 1                     # the trace ends before the refused step
    other = assemble_subleq("subleq a, 5, next\nnext: subleq z, z, halt\n.cell a 3\n.cell z 0")
    assert interpret(other)[-1].pc == other.halted


def test_blind_run_ending_at_a_latch_step(tmp_path):
    """--blind --latch: no command during the run, one read: the latch is up and the level halted (halt latch >= 1)."""
    from doomformer.dollclock import play_blind
    from doomformer.subleq import loader
    own = SubleqMachine(6).assemble("subleq a, b, next\nnext: subleq k, next.c, halt\n.cell a 1\n.cell b 5\n.cell k 1")
    SubleqMachine(6).compile(tmp_path / "b.wad", clock="dolls")
    ref = interpret(own, latch=True)
    r = play_blind(tmp_path / "b.wad", own, ref, loader=loader(own))
    assert not r["bad"] and r["halted"] and r["sha_equal"] and r["audit"]["non_readout_commands"] == 0
    ref[-1].fires["viol"] = 0                                             # a wrong expectation must be caught
    assert play_blind(tmp_path / "b.wad", own, ref, loader=loader(own))["bad"]


def test_latch_trace_is_the_refused_step():
    own = assemble_subleq("subleq a, b, next\nnext: subleq k, next.c, halt\n.cell a 1\n.cell b 5\n.cell k 1")
    tr = interpret(own, latch=True)
    assert [s.pc for s in tr] == [0, 1, own.halted] and tr[-1].only == ("viol",) and tr[-1].idx == 2
    with pytest.raises(ValueError, match="writes its own instruction cell 5"):
        interpret(own)
    clean = interpret(assemble_subleq(ADD(3, 4)), latch=True)
    assert all("viol" not in s.fires for s in clean) and clean[-1].pc == 6


def test_halting_rules():
    p = assemble_subleq("subleq a, a, 7\n.cell a 1")                  # 0 <= 0, c = 7 >= 6 slots: halt
    assert interpret(p)[-1].pc == p.halted
    p = assemble_subleq("subleq a, a, -3\n.cell a 1")                  # c < 0: halt
    assert interpret(p)[-1].pc == p.halted
    p = assemble_subleq("subleq a, b, 0\n.cell a 1\n.cell b 5")        # 4 > 0: fall through to slot 1, which is data
    assert interpret(p, 1)[1].pc == 1


def test_step_values_are_complete():
    """Every value the map produces is in the trace: one-hot decodes, digits of x, the borrow, the branch."""
    f = interpret(assemble_subleq(MULTIPLY(7, 9)))[1].fires        # p = 0 - (-9): x = -9 = 8 * (-2) + 7, the low digit borrows
    assert (f["ad9"], f["bd10"], f["ct1"], f["xl7"], f["xhm2"], f["xl"], f["xh"], f["nb10"]) == (1, 1, 1, 1, 1, 7, -2, 1)
    assert "le" not in f and f["ah"] == 1 and f["al"] == 1 and f["bl"] == 2 and f["bh"] == 1


def test_counter_machine_acs_has_no_open_script(tmp_path):
    """clock="dolls": the ACS lump holds the readout pages and the motion script only; the dolls tag themselves with
    the line special Thing_ChangeTID, not ACS_ExecuteAlways."""
    m, *_ = PROGRAMS["add"].build(3, 4)
    meta = m.compile(tmp_path / "dc.wad", clock="dolls")
    info = meta.layout["machine"]["dollclock"]
    assert info["scripts"] == [8, 100] and info["tag_lines"] == meta.n_dolls
    text = dict(M._wad_lumps((tmp_path / "dc.wad").read_bytes()))["TEXTMAP"].decode()
    assert "special = 226" not in text and text.count("special = 176") == meta.n_dolls + 1    # + the loop doll's own tag


def test_doll_clock_counter_add_is_exact_and_tamper_fires(tmp_path):
    from doomformer.dollclock import DollClock
    m, reg, want, _ = PROGRAMS["add"].build(3, 4)
    r = m.run(tmp_path / "dc.wad", clock="dolls")
    assert r.registers[reg] == want and r.halted and r.audit["non_readout_commands"] == 0
    assert set(r.audit["commands"]) <= {f"puke -{100 + i}" for i in range(8)} | {"puke -8"}      # pages and motion only
    m.compile(tmp_path / "t.wad", clock="dolls")
    change = M.tamper(tmp_path / "t.wad", m.program, 1)
    with pytest.raises(MachineMismatch) as e:
        M.play(tmp_path / "t.wad", m.program, m.reference(), clock_cls=DollClock)
    assert e.value.step == change["predicted_step"] == 2


def test_doll_clock_rewind_chain_folds_into_rows(tmp_path, monkeypatch):
    """A chain longer than a row continues on the next row after a Teleport_NoFog line; the machine stays exact."""
    from doomformer import dollclock
    monkeypatch.setattr(dollclock, "ROW_END", dollclock.X_ORIGIN + 500)
    m, reg, want, _ = PROGRAMS["add"].build(3, 4)
    meta = m.compile(tmp_path / "fold.wad", clock="dolls")
    text = dict(M._wad_lumps((tmp_path / "fold.wad").read_bytes()))["TEXTMAP"].decode()
    assert len(set(re.findall(rf"special = 71; arg0 = ({dollclock.ROW_TID0 + 1}|{dollclock.ROW_TID0 + 2});", text))) == 2
    r = M.play(tmp_path / "fold.wad", m.program, m.reference(), clock_cls=dollclock.DollClock)
    assert r.registers[reg] == want and meta.layout["machine"]["rewind_lines"] > 40


def test_subleq_add_on_one_map_and_decode_tamper(tmp_path):
    sm = SubleqMachine(6)
    wad = tmp_path / "subleq.wad"
    sm.compile(wad)
    digest = M.sha256(wad)
    r = sm.run(wad, ADD(3, 4))
    assert r.registers["m15"] == 7 and r.steps == 4 and r.audit["non_readout_commands"] == 0 and r.audit["loaded"] > 0
    assert M.sha256(wad) == digest
    # the in-map latch: a program with an own-cell write is flagged by the map at the step the interpreter names
    own = "subleq a, b, next\nnext: subleq k, next.c, halt\n.cell a 1\n.cell b 5\n.cell k 1"
    v = sm.run(wad, own, latch=True)
    assert v.steps == 2 and v.halted and interpret(sm.assemble(own), latch=True)[-1].fires == {"viol": 1}
    assert M.sha256(wad) == digest
    p = sm.assemble(ADD(3, 4))
    change = tamper(wad, p, 12)                                  # the decode line of cell 12 (a): first operand a at step 1
    with pytest.raises(MachineMismatch) as e:
        sm.run(wad, p)
    assert e.value.step == change["predicted_step"] == 1


@pytest.mark.skipif(not HEAVY, reason="DOOMFORMER_HEAVY=1: about 80 s")
def test_subleq_add_with_doll_clock(tmp_path):
    sm = SubleqMachine(6)
    wad = tmp_path / "subleq_dolls.wad"
    meta = sm.compile(wad, clock="dolls")
    assert meta.layout["machine"]["dollclock"]["scripts"][:3] == [1, 2, 8]
    r = sm.run(wad, ADD(3, 4), clock="dolls")
    assert r.registers["m15"] == 7 and r.audit["non_readout_commands"] == 0


@pytest.mark.skipif(not HEAVY, reason="2 to 3 minutes per program; DOOMFORMER_HEAVY=1")
def test_subleq_multiply_and_selfmod_same_bytes(tmp_path):
    sm = SubleqMachine(6)
    wad = tmp_path / "subleq.wad"
    sm.compile(wad)
    digest = M.sha256(wad)
    assert sm.run(wad, MULTIPLY(7, 9)).registers["m10"] == 63
    assert sm.run(wad, SELFMOD()).registers["m15"] == -15
    assert M.sha256(wad) == digest
