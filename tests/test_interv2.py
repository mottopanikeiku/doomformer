"""Runtime lane doors in a one-map brain, direction interventions and probes, the planted circuit, tautology flags.

Light: about a minute with the game tests. The in-level lesion run (demos.brainlevel --lesion-at) is a logged command;
`DOOMFORMER_HEAVY=1` adds a four-decision run of it.
"""
from __future__ import annotations

import os

import numpy as np
import pytest

import doomformer as df
from doomformer import groundtruth as gt
from doomformer import intervene as iv
from doomformer import patching, steer
from doomformer.intref import forward
from doomformer.onemap import tiny_model
from doomformer.orchestrator import DoomTransformer

TOKS = [0, 2, 1]
DONOR = [0, 3, 4]


def test_direction_and_project_semantics():
    m = tiny_model()
    tr = {}
    forward(m, TOKS, tr)
    x = tr["L0.x1"][2].copy()
    u = np.array([1, 0, -1, 2])
    got = {}
    forward(m, TOKS, got, interventions=[iv.add_direction("L0.x1", u, 2, pos=[2])])
    assert np.array_equal(got["L0.x1"][2], np.clip(x + 2 * u, -m.K, m.K))       # clipped to the staircase range
    assert np.array_equal(got["L0.x1"][:2], tr["L0.x1"][:2])                    # other positions untouched
    forward(m, TOKS, got, interventions=[iv.project_out("L0.x1", u, pos=[2])])
    c = int(np.sign(x @ u) * (abs(x @ u) // (u @ u)))
    assert np.array_equal(got["L0.x1"][2], x - c * u) and abs(got["L0.x1"][2] @ u) < u @ u
    sub = iv.add_direction("L0.x1", [3, -3], 1, units=[0, 3], pos=[2])        # a direction over a subset of the units
    forward(m, TOKS, got, interventions=[sub])
    assert got["L0.x1"][2, 1] == x[1] and got["L0.x1"][2, 0] == np.clip(x[0] + 3, -m.K, m.K)
    for bad in (lambda: iv.add_direction("L0.x1", [0, 0, 0, 0]), lambda: iv.add_direction("L0.x1", [1.5, 0, 0, 0]),
                lambda: iv.compile_plan(m, None, [iv.add_direction("L0.x1", [1, 2])]),
                lambda: iv.fold(m, [iv.add_direction("L0.x1", u)])):
        with pytest.raises(ValueError):
            bad()


def test_probe_fits_a_direction_that_steers():
    from doomformer import policy, shipped
    from doomformer.intref import QModel
    m = QModel.load(shipped.resolve("doomplayer"))
    ctx = policy.all_contexts()
    bearing = [policy.VOCAB[c[1]] for c in ctx]
    fit = [i for i, b in enumerate(bearing) if b in "im"]
    pr = df.probe(m, "x0", [ctx[i] for i in fit], [bearing[i] == "i" for i in fit], pos=1)
    assert pr.accuracy == 1.0 and pr.direction.dtype == np.int64 and abs(pr.direction).max() == pr.scale
    m_ctx = [c for c, b in zip(ctx, bearing) if b == "m" and policy.act_index(forward(m, c)[-1]) != 2]
    assert m_ctx and all(policy.act_index(steer.steer(m, c, pr, 2)) == 2 for c in m_ctx)
    with pytest.raises(ValueError):
        df.probe(m, "x0", ctx[:3], [0, 1, 2])


def test_direction_exact_in_doom(tmp_path):
    net = tiny_model()
    with df.compile(net, tmp_path / "tf", mode="tiles", workers=2) as brain:
        eng = DoomTransformer(net, brain.wad, workers=2)
        try:
            for ivs in ([iv.add_direction("L0.x1", [1, 0, -1, 2], 1, pos=[1, 2])],
                        [iv.project_out("x0", [1, 1, 0, -1], pos=[2])],
                        [iv.add_direction("L0.m", [1, -1, 0, 2, 1], 1, pos=[2])]):
                eng.use(ivs)
                logits = eng.forward(TOKS)
                rep = eng.check_tokens(TOKS)
                assert rep["exact"] and np.array_equal(logits, forward(net, TOKS, interventions=ivs)[-1]), ivs
                assert rep["patches"][1] > 0
        finally:
            eng.close()


def test_planted_circuit_recovery_matches_the_construction():
    m = gt.planted_model()
    assert gt.top1(m) == 1.0
    res = gt.check_recovery(m, pairs_per_kind=6, out=lambda *_: None)
    for kind in ("prev", "cur"):
        assert res[kind]["tested"] > 3000 and res[kind]["planted"] == res[kind]["nonzero"] > 0 and not res[kind]["wrong"]
    assert res["decoy"]["decoy_active"] == res["decoy"]["pairs"] and not res["decoy"]["wrong"]
    rows = gt.circuit_check(m, out=lambda *_: None)
    assert rows[("H0", "MLP")]["faithfulness"] == 1.0 and rows[("H1", "MLP")]["faithfulness"] == 0.0


def test_backup_head_splits_single_cell_recovery_and_circuit_search_finds_two_circuits():
    m = gt.planted_model(backup=True)
    assert gt.top1(m) == 1.0
    res = gt.check_recovery(m, kinds=("prev", "cur"), pairs_per_kind=6, out=lambda *_: None, backup=True)
    for kind in ("prev", "cur"):
        assert res[kind]["planted"] == res[kind]["nonzero"] > 0 and not res[kind]["wrong"]
    clean, corrupt = gt.make_pairs("prev")[0]
    rows = patching.patch_recovery(m, clean, corrupt, [dict(site="L0.ao", pos=(3,), heads=h) for h in ((0,), (1,), (0, 1))],
                                   answer=(gt.answer(clean), gt.answer(corrupt))).rows
    assert [r["recovery"] for r in rows] == [0.5, 0.5, 1.0]
    for h in (0, 1):                                                    # neither head is necessary
        assert all(np.argmax(forward(m, c, interventions=[iv.zero("L0.ao", heads=[h])])[-1]) == gt.answer(c)
                   for c in gt.contexts())
    circuits = gt.circuit_check(m, out=lambda *_: None)
    assert circuits[("H0", "MLP")]["faithfulness"] == circuits[("H1", "MLP")]["faithfulness"] == 1.0
    assert circuits[("H0",)]["faithfulness"] == circuits[("MLP",)]["faithfulness"] == 0.0


def test_planted_cells_exact_in_doom():
    m = gt.planted_model()
    clean, corrupt = gt.make_pairs("prev")[5]
    specs = gt.cell_specs(m)
    cells = gt.planted_cells("prev", clean, corrupt)
    on = [i for i, s in enumerate(specs) if gt._expected_nonzero("prev", s, cells)]
    pick = on[:2] + [on[-1]] + [i for i in range(len(specs)) if i not in on][:1]
    with patching.DoomRuns(m, workers=2) as runs:
        tab = patching.patch_recovery(m, clean, corrupt, specs, answer=(gt.answer(clean), gt.answer(corrupt)),
                                      doom=pick, runs=runs)
    assert tab.exact and len(tab.doom_rows) == len(pick)


def test_whole_residual_rows_are_flagged():
    m = tiny_model()
    tab = patching.patch_recovery(m, TOKS, DONOR, patching.grid(m, per_head=True))
    flagged = {(r["site"], tuple(r["pos"])): r["flag"] for r in tab.rows if r["flag"]}
    assert flagged[("L0.x1", (2,))] == flagged[("L0.x2", (2,))] == "tautological"     # last layer, last position
    assert flagged[("x0", (1,))] == flagged[("L0.x1", (0,))] == "whole residual"
    assert flagged[("L0.m", (2,))] == flagged[("L0.ao", (2,))] == "whole site"            # every unit at the read position
    assert ("L0.m", (1,)) not in flagged and ("L0.v", (2,)) in flagged                      # earlier positions say which position
    assert all(r["flag"] == "" for r in tab.rows if r["site"].split(".")[-1] in ("q", "k", "v", "ao", "m")
               and r["pos"] != (2,))
    assert len(tab.headline) == len(tab.rows) - len(flagged) and "tautological" in tab.text()
    unit = patching.patch_recovery(m, TOKS, DONOR, [dict(site="L0.x2", pos=(2,), units=(0, 1))])
    assert unit.rows[0]["flag"] == ""                                                    # not every unit
    tau = [r for r in tab.rows if r["flag"] == "tautological"]
    assert tau and all(abs(r["recovery"] - 1) < 1e-9 for r in tau)


def test_lane_doors_close_between_decisions_and_match_the_masked_reference(tmp_path):
    from doomformer.onemap import build_onemap, compare_trace, token_ports
    from doomformer.pure import DOOR_OPEN
    from doomformer.runtime import TileRunner
    from doomformer.tile import MOVE_SPEED
    net = tiny_model()
    lesion = {"x0": [1], "L0.x1": [0, 2], "L0.m": [3], "L0.x2": [1], "L0.ao": [0]}
    doors, tags = iv.lane_doors(net, lesion)
    assert sorted(tags) == sorted((k, u) for k, us in lesion.items() for u in us)

    def world(m):
        m.rect(-768, -512, -512, -256, floor=0, ceil=128)
        m.thing(-640, -384, 1)
        return f"script 210 (int tag) {{ Ceiling_LowerByValue(tag, {MOVE_SPEED}, {DOOR_OPEN}); }}"

    wad = tmp_path / "doors.wad"
    meta = build_onemap(net, wad, world=world, doors=doors)
    assert set(meta.layout["doors"]) == {str(t) for t in tags.values()}
    plain = build_onemap(net, tmp_path / "plain.wad", world=lambda m: world(m))
    assert "doors" not in plain.layout
    runner = TileRunner(wad)
    try:
        def run(ablate):
            tr = {}
            forward(net, TOKS, tr, ablate=ablate)
            got = runner.run(token_ports(net, TOKS))[0]
            return compare_trace(got, tr, len(TOKS))

        assert run(None) > 0                                     # doors open: the healthy brain
        runner.game.send_game_command("puke -3")                  # the rewind sends every doll home first
        runner.game.advance_action(6)
        for tag in sorted(set(tags.values())):
            runner.game.send_game_command(f"puke -210 {tag}")
        runner.game.advance_action(4)
        assert run(lesion) > 0                                   # doors closed: the masked brain, every value
        with pytest.raises(AssertionError):
            run(None)                                            # and no longer the healthy one
    finally:
        runner.close()


def test_lesion_when_is_a_condition_on_the_sensed_tokens_in_the_clock_script():
    from types import SimpleNamespace

    from demos import brainlevel as bl
    from doomformer import policy
    cond, pred = bl.parse_when("near+ahead")
    assert "bl_distance" in cond and "bl_bearing" in cond and cond.count("&&") == 1
    v = policy.VOCAB
    assert pred([0, v.index("i"), v.index("1")]) and not pred([0, v.index("i"), v.index("2")])
    assert not pred([0, v.index("j"), v.index("0")]) and not pred([0, v.index("?"), v.index("4")])
    assert bl.parse_when("near")[1]([0, v.index("c"), v.index("0")]) and not bl.parse_when("ahead")[1]([0, v.index("c"), v.index("0")])
    for bad in ("", "far", "near+", "near ahead"):
        with pytest.raises(ValueError):
            bl.parse_when(bad)
    meta = SimpleNamespace(ports={})
    when = bl.clock_acs(meta, 100, 40, 1, [52000, 52001], None, "near")
    count = bl.clock_acs(meta, 100, 40, 1, [52000, 52001], 3, None)
    assert when.index("bl_sense();") < when.index("if (bl_doors == 0 && ((bl_distance") and "bl_step + 1 >=" not in when
    assert count.index("bl_step + 1 >= 3") < count.index("bl_sense();") and "bl_distance <=" not in count
    assert when.count("Ceiling_LowerByValue") == 2 == count.count("Ceiling_LowerByValue")
    with pytest.raises(ValueError):
        bl.build_brainlevel(None, None, period=1, doors={"z0:0:0": 1}, lesion_at=1, lesion_when="near")


@pytest.mark.skipif(not os.environ.get("DOOMFORMER_HEAVY"), reason="a full level run; set DOOMFORMER_HEAVY=1")
def test_brainlevel_lesion_at_flips_the_action_in_the_level(tmp_path):
    from demos.brainlevel import run_brainlevel
    stats = run_brainlevel(seed=2, decisions=4, opt=3, lesion={"x0": [0], "L0.ao": [7], "L0.x1": [7], "L0.m": [0],
                                                               "L0.x2": [0]}, lesion_at=3, workdir=tmp_path,
                           out=lambda *_: None)
    assert stats["exact"] == 4 and stats["doors"] == [False, False, True, True] and stats["changed_at"][0] == 3
    assert stats["audit"]["non_readout_commands"] == 0
    when = run_brainlevel(seed=2, decisions=4, opt=3, lesion={"x0": [0], "L0.ao": [7], "L0.x1": [7], "L0.m": [0],
                                                              "L0.x2": [0]}, lesion_when="ahead", workdir=tmp_path / "w",
                          out=lambda *_: None)
    assert when["exact"] == 4 and when["doors"] == [False, False, True, True] and when["changed_at"][0] == 3
    assert when["audit"]["non_readout_commands"] == 0
