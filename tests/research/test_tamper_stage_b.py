"""Negative controls for the Stage-B clock of his-brain-is-the-level (demos/brainlevel.py) and the fly loop: a tampered
clock or a tampered comparator line in the compiled map must fail its check for the intended reason; the level's ACS
compares no logits; the self-running maps never read the readout variables."""
from __future__ import annotations

import copy
import re
import struct

import numpy as np
import pytest
import vizdoom as vzd

from demos import brainlevel
from demos import flyloop as F
from demos.brainlevel import ClockRun, build_brainlevel
from doomformer import policy, pure
from doomformer.onemap import tiny_model, value
from doomformer.spiking import IntNet
from doomformer.tile import TileMeta


# --- Stage B: the level's ACS clock (demos/brainlevel.py) on a tiny policy-vocabulary transformer ------------------
@pytest.fixture(scope="module")
def policy_model():
    model = tiny_model()
    model.vocab = policy.VOCAB
    rng = np.random.default_rng(19)
    model.p["tok_emb"] = rng.integers(-2, 3, (len(model.vocab), model.d))
    model.p["Wun"] = rng.integers(-2, 3, (len(model.vocab), model.d))
    model.p["bun"] = rng.integers(-2, 3, len(model.vocab))
    return model


@pytest.fixture(scope="module")
def ordered_model(policy_model):
    """Biases fix the action logits' order for every input, R > F > L, so a tamper's effect is known in advance."""
    model = copy.deepcopy(policy_model)
    reach = int(np.abs(model.p["Wun"][policy.ACTION_IDS]).sum(1).max()) * len(model.thr_in(model.L))
    model.p["bun"][policy.ACTION_IDS] = [0, 4 * reach + 2, 2 * reach + 1]
    return model


def rewrite_textmap(wad, pattern: str, replacement: str) -> int:
    """Regex-substitute the WAD's TEXTMAP lump in place; returns the number of substitutions."""
    data = wad.read_bytes()
    n, directory = struct.unpack("<ii", data[4:12])
    lumps, count = [], 0
    for i in range(n):
        offset, size, name = struct.unpack("<ii8s", data[directory + 16 * i:directory + 16 * (i + 1)])
        blob = data[offset:offset + size]
        if name.rstrip(b"\0") == b"TEXTMAP":
            text, count = re.subn(pattern, replacement, blob.decode())
            blob = text.encode()
        lumps.append((name, blob))
    body, table = b"", b""
    for name, blob in lumps:
        table += struct.pack("<ii8s", 12 + len(body), len(blob), name)
        body += blob
    wad.write_bytes(struct.pack("<4sii", b"PWAD", len(lumps), 12 + len(body)) + body + table)
    return count


def clock_run(model, tmp_path, monkeypatch, *replacements, line=None):
    """Build the Stage-B map with the clock source edited by (old, new, count) and, if `line` = (pattern(meta),
    replacement) is given, exactly one TEXTMAP linedef rewritten; open its readout-only host. The compute period is
    the Stage-B layout's settle estimate; the positive control shows it suffices."""
    clock_acs = brainlevel.clock_acs

    def tampered(*args):
        source = clock_acs(*args)
        for old, new, count in replacements:
            assert source.count(old) == count
            source = source.replace(old, new)
        return source

    monkeypatch.setattr(brainlevel, "clock_acs", tampered)
    period = build_brainlevel(model, tmp_path / "plan.wad", layout_only=True, period=1).layout["estimated_settle_tics"]
    wad = tmp_path / "clock.wad"
    meta = build_brainlevel(model, wad, period=period)
    if line is not None:
        assert rewrite_textmap(wad, line[0](meta), line[1]) == 1
    return ClockRun(model, wad, seed=4, resolution=vzd.ScreenResolution.RES_160X120)


def test_stage_b_positive_control_and_guard(policy_model, tmp_path, monkeypatch):
    """Untampered clock: exact decision. Mid-run, set_action and a chained `puke -STATUS ; kill` (logged as the
    status read by a first-two-tokens counter) raise before reaching the engine; the marine stays alive."""
    run = clock_run(policy_model, tmp_path, monkeypatch)
    try:
        decision = run.decide()
        assert decision is not None and decision["tokens"] == policy.observe(run.g.get_state(), run.g)
        health = run.g.get_game_variable(vzd.GameVariable.HEALTH)
        with pytest.raises(RuntimeError, match="rejected: set_action"):
            run.g.set_action([0, 0])
        chained = f"puke -{brainlevel.STATUS} ; kill"
        with pytest.raises(RuntimeError, match="non-readout command rejected"):
            run.g.send_game_command(chained)
        run.play(lambda: None)
        assert run.g.get_game_variable(vzd.GameVariable.HEALTH) == health > 0
        audit = run.audit()
        assert audit["forbidden"] == {"set_action": 1, chained: 1} and audit["commands"][chained] == 1
    finally:
        run.close()


def test_c1_clock_senses_a_tampered_bearing(policy_model, tmp_path, monkeypatch):
    run = clock_run(policy_model, tmp_path, monkeypatch, ("bl_sense();", "bl_sense(); bl_bearing += 1;", 1))
    try:
        with pytest.raises(AssertionError, match="ACS tokens") as err:
            run.decide()
    finally:
        run.close()
    acs, observed = (list(map(int, t.split(", "))) for t in re.findall(r"\[([\d, ]+)\]", str(err.value)))
    assert acs == [observed[0], observed[1] + 1, observed[2]]


def test_c2_one_comparator_line_sign_flipped(ordered_model, tmp_path, monkeypatch):
    """The R-vs-F comparator doll (it passes: s_R >= s_F) lowers the -wins_R floor port; its line now raises it, so
    the port reads wins_R = 0 instead of 2: no winner-layer gate opens, the level receives no action, and the check
    stops at the winner's one-hot light."""
    tag = lambda meta: meta.layout["tags"][f"floor:-{value('argw', ordered_model.T - 1, 1)}"]
    run = clock_run(ordered_model, tmp_path, monkeypatch,
                    line=(lambda meta: rf"special = {pure.FLOOR_LOWER}; (arg0 = {tag(meta)}; arg1 = \d+; arg2 = 1;)",
                          rf"special = {pure.FLOOR_RAISE}; \1"))
    try:
        with pytest.raises(AssertionError) as err:
            run.decide()
        assert str(err.value) == f"{value('argz', ordered_model.T - 1, 1)}: DOOM 0, intref 1"
        assert run.status()["action"] == brainlevel.NO_ACTION
    finally:
        run.close()


def test_c3_winner_doll_hands_the_wrong_action(ordered_model, tmp_path, monkeypatch):
    """R's winner line runs ACS_ExecuteAlways(CHOOSE, 0, 2) instead of (.., 1): every doll value is exact, but the
    action the level received is not the dolls' winner."""
    run = clock_run(ordered_model, tmp_path, monkeypatch,
                    line=(lambda meta: rf"(special = {pure.EXEC_ALWAYS}; arg0 = {brainlevel.CHOOSE}; arg1 = 0; "
                                       rf"arg2 = )1;", r"\g<1>2;"))
    try:
        with pytest.raises(AssertionError, match=r"level action 2, dolls' winner 1, argmax of the L/R/F logits "
                                                 r"\[-?\d+, -?\d+, -?\d+\] is 1$"):
            run.decide()
    finally:
        run.close()


def test_the_level_compares_no_logits(policy_model, tmp_path, monkeypatch):
    """Outside the readout pages the Stage-B ACS reads no sector at all (so no logit), and it only tests the stored
    action for equality; three winner lines in the map hand the action over."""
    sources = []
    compile_acs = pure.compile_acs
    monkeypatch.setattr(pure, "compile_acs", lambda acs: sources.append(acs) or compile_acs(acs))
    wad = tmp_path / "clock.wad"
    meta = build_brainlevel(policy_model, wad, period=64)
    page = re.compile(r"script (\d+) \(void\) \{ o1 = ")
    lines = sources[0].splitlines()
    level = "\n".join(line for line in lines if not page.match(line))
    assert [int(page.match(line).group(1)) for line in lines if page.match(line)] == [
        100 + i for i in range(len(meta.pages))]
    assert "GetSector" not in level and "logit" not in level
    assert set(re.findall(r"bl_action\s*([<>!=]=?)", level)) == {"=", "=="}
    assert f"script {brainlevel.CHOOSE} (int action) {{ bl_action=action; }}" in level
    assert TileMeta.load(wad.with_suffix(".json")).layout["stage_stats"][-1]["name"] == "argmax.winner"
    assert [rewrite_textmap(wad, rf"special = {pure.EXEC_ALWAYS}; arg0 = {brainlevel.CHOOSE}; arg1 = 0; "
                                 rf"arg2 = {a};", r"\g<0>") for a in range(3)] == [1, 1, 1]


# --- the self-running maps never read the USER readout variables Python's readout scripts fill -------------------
READOUT_READ = re.compile(r"\bo\d+\b(?!\s*=[^=])")


def readout_reads(acs: str) -> list[str]:
    return READOUT_READ.findall(re.sub(r"global int \d+:o\d+;", "", acs))


def test_maps_never_read_the_readout_variables(policy_model, tmp_path, monkeypatch):
    sources = []
    compile_acs = pure.compile_acs
    monkeypatch.setattr(pure, "compile_acs", lambda acs: sources.append(acs) or compile_acs(acs))
    build_brainlevel(policy_model, tmp_path / "clock.wad", period=64)
    net = IntNet(["MN9_left"], ["s0"], [[0]], [[8]], theta=10, leak=1,
                 meta=dict(input_groups={"sugar": [0]}, input_p_per_step={"sugar": 0.5}))
    F.compile_flyloop(net, tmp_path / "loop.wad", stations=(F.Station("sugar", 30, 3),), seed=7, period=64)
    assert len(sources) == 3                                    # Stage-B map; fly twin and loop map
    assert all("script 100 (void) { o1 = " in acs for acs in sources)
    assert [readout_reads(acs) for acs in sources] == [[], [], []]
    assert readout_reads("bl_logits[0] = o50; o1 = 2; o2 += 1; if (o3 == 4) o4 = o5;") == ["o50", "o2", "o3", "o5"]
