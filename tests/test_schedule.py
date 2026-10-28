"""The static schedule checker (doomformer/schedule.py): its predicted tics against engine positions, clean verdicts on
the shipped schedules (opt 0, 1, 3), and planted failures whose witness input makes DOOM mismatch the reference while
the fixed or default map stays exact. At most one DOOM game at a time."""
from __future__ import annotations

import dataclasses
import json

import numpy as np
import pytest

import doomformer as df
from doomformer import pure
from doomformer import schedule as S
from doomformer.cli import main
from doomformer.kit import _build
from doomformer.onemap import tiny_model
from doomformer.pure import TUNING_FILE, Timing, compile_pure
from doomformer.runtime import TileRunner
from doomformer.tile import CLOSED


def hello_mlp() -> df.MLP:
    """The net of `doomformer hello` (seed 0): 4-6-5-3, three thresholds per layer."""
    rng = np.random.default_rng(0)
    sizes = [4, 6, 5, 3]
    return df.MLP([rng.integers(-20, 21, (sizes[l + 1], sizes[l])) for l in range(3)],
                  [rng.integers(-30, 31, sizes[l + 1]) for l in range(3)],
                  [np.cumsum(rng.integers(15, 40, 3)) for _ in range(3)])


def net591() -> df.MLP:
    """Verify's fuzzer case: 4 lane dolls whose walkways (-5772, -8082) need two-tic barrier drops."""
    return df.MLP([[[156]], [[3]]], [[1617], [0]], [[5748, 8058], [11999]])


def compile_mlp(net, wad, **kw):
    stages = net._stages()
    consumed = {key.split(":", 1)[1] for s in stages for key in s.ports}
    readout = [n for s in stages for n, _ in s.outputs if n in consumed]
    return compile_pure(stages, wad, "mlp", readout=readout, **kw)


def exact(net, wad, xs) -> list[bool]:
    """Run every x in one game; True where every exposed value equals the integer reference."""
    meta = json.loads(wad.with_suffix(".json").read_text())
    out = []
    with TileRunner(wad) as r:
        for x in xs:
            x = np.asarray(x)
            ports = {}
            for key in meta["ports"]:
                name = key.split(":", 1)[1]
                v = int(x[int(name.lstrip("-")[1:])])
                ports[name] = -v if name.startswith("-") else v
            got = r.run(ports)[0]
            ref = net.reference(x)
            out.append(all(v == net._expected(n, ref) for n, v in got.items()))
    return out


@pytest.mark.parametrize("which,opt", [("hello", 0), ("hello", 1), ("hello", 3), ("tiny", 0), ("tiny", 1)])
def test_shipped_schedules_are_clean(tmp_path, which, opt):
    if opt == 3 and not TUNING_FILE.is_file():
        pytest.skip(f"{TUNING_FILE.name} is missing (opt=3 builds with its constants)")
    net = hello_mlp() if which == "hello" else tiny_model()
    wad = tmp_path / "map.wad"
    _build(net, wad, opt=opt)
    rep = df.check_schedule(wad)
    assert rep.ok and rep.hazard_count == 0, str(rep)
    assert {m["name"] for m in rep.margins} >= {"sync_wait", "clock_margin"}
    if opt == 3:
        assert any(m["check"] == "copy" for m in rep.margins)


def test_predicted_tics_match_the_engine(tmp_path):
    """Every release line and every lane cell of a hello pass is crossed in the predicted tic (engine positions)."""
    net, x = hello_mlp(), np.array([100, -60, 30, 0])
    wad = tmp_path / "hello.wad"
    meta = _build(net, wad)
    ref = net.reference(x)
    values = {f"x{i}": int(v) for i, v in enumerate(x)}
    values.update({f"h{l}_{j}": int(v) for l, h in enumerate(ref["hidden"], 1) for j, v in enumerate(h)})
    lay = S.Layout(S.MapData.parse(S.textmap(wad)), json.loads(wad.with_suffix(".json").read_text()))
    tl = S.Timeline(lay, values=values)
    ports = {}
    for key in meta.ports:
        name = key.split(":", 1)[1]
        v = int(x[int(name.lstrip("-")[1:])])
        ports[name] = -v if name.startswith("-") else v
    trace = []
    with TileRunner(wad) as r:
        r.start(ports)
        for _ in range(460):
            r.game.advance_action(1)            # model tic t = state after t + 1 advances
            trace.append({o.id: (o.position_x, o.position_y) for o in r.game.get_state().objects
                          if o.name == "DoomPlayer"})
    md = lay.md

    def crossing(chain, y, x_line):
        x0, x1 = md.x0[chain[0]], md.x1[chain[-1]]
        oid = next(i for i, (px, py) in trace[0].items() if abs(py - y) < 1e-6 and x0 - 1 <= px <= x1)
        return next(t for t, f in enumerate(trace) if f[oid][0] > x_line)

    checked = 0
    for s, clock in enumerate(lay.layer_clock):
        chain = clock.segments[0][0]
        y = (md.y0[chain[0]] + md.y1[chain[0]]) / 2
        for ln, tic in zip(clock.lines, tl.clock[s]):
            if ln.role in ("release", "next"):
                assert crossing(chain, y, ln.x) == tic, (s, ln)
                checked += 1
    for i, lane in enumerate(lay.lanes):
        for cell, iv in zip(lane.cells, tl.lane[i].cells or []):
            if iv is not None:
                assert iv[0] <= crossing(lane.chain, lane.y, cell.x) <= iv[1], (i, cell)
                checked += 1
    assert checked > 100


def test_barrier_float_is_flagged_and_its_witness_fails(tmp_path, monkeypatch):
    """Planted: the old absolute barrier height (CLOSED) makes net591's t=5748 lane stop on a half-lowered barrier and
    float ~7 tics, so two dolls of consecutive waves write -h1_0 in one tic. The checker gives the witness; DOOM loses
    the move there. The fixed compiler is clean and exact on the same input."""
    net = net591()
    with monkeypatch.context() as m:
        m.setattr(pure, "_barrier_height", lambda floors: CLOSED)
        bad = tmp_path / "old.wad"
        compile_mlp(net, bad)
    rep = df.check_schedule(bad, net=net)
    assert not rep.ok and rep.hazard_count == 1
    h = rep.hazards[0]
    assert h["certain"] and h["value"] == "-h1_0" and h["witness"]["x"] == [-8058]
    good = tmp_path / "fixed.wad"
    compile_mlp(net, good)
    assert df.check_schedule(good, net=net).ok
    assert exact(net, bad, [h["witness"]["x"], [-9000]]) == [False, False]
    assert exact(net, good, [h["witness"]["x"], [-9000]]) == [True, True]


def test_tight_serial_is_flagged_and_its_witness_fails(tmp_path):
    """Planted: hello with wave spacing 8 units (half a tic) in layer 0. The first certain hazard's witness input
    mismatches in DOOM; the default and opt=1 maps are unflagged and exact on it."""
    net = hello_mlp()
    bad = tmp_path / "s8.wad"
    compile_mlp(net, bad, serial={0: 8})
    rep = df.check_schedule(bad, net=net)
    assert not rep.ok and rep.hazards[0]["certain"]
    x = rep.hazards[0]["witness"]["x"]
    assert exact(net, bad, [x]) == [False]
    for opt in (0, 1):
        wad = tmp_path / f"o{opt}.wad"
        _build(net, wad, opt=opt)
        assert df.check_schedule(wad, net=net).ok
        assert exact(net, wad, [x]) == [True]


def test_copy_line_before_the_last_light_hit_is_flagged(tmp_path):
    """Planted (opt 3): clock_margin 0 puts the copy line before the last light hits of layer 0; the copy writes stale
    planes and DOOM mismatches. (copy_settle 4 and 16 units were exact: no row needs it.)"""
    if not TUNING_FILE.is_file():
        pytest.skip(f"{TUNING_FILE.name} is missing (opt=3 builds with its constants)")
    net = hello_mlp()
    wad = tmp_path / "cm0.wad"
    _build(net, wad, opt=3, timing=dataclasses.replace(Timing.tuned(copy=True), clock_margin=0))
    rep = df.check_schedule(wad)
    assert not rep.ok and rep.hazard_count == 0
    assert any(m["check"] == "copy" and not m["ok"] for m in rep.margins)
    assert exact(net, wad, [[51, 92, -144, 93]]) == [False]


def test_lint_command(tmp_path, capsys):
    net = hello_mlp()
    good, bad = tmp_path / "good.wad", tmp_path / "bad.wad"
    _build(net, good)
    compile_mlp(net, bad, serial={0: 0})
    assert main(["lint", str(good)]) == 0
    assert "OK" in capsys.readouterr().out
    assert main(["lint", str(bad), "--json"]) == 1
    rep = json.loads(capsys.readouterr().out)
    assert rep["hazard_count"] > 0 and rep["minimums"]["separation"] == 1


def test_small_barrier_stop_family_is_flagged(tmp_path, monkeypatch):
    """Planted: with the old barrier height, thresholds [5728, 5736] (signed input) put the two lane walkways 2 and 10
    units under the barrier's first-tic stop. The engine steps onto the 10-unit stop (the doll floats) and not onto the
    2-unit one (8 units or fewer are absorbed, schedule.LOWERING_ABSORBED), so the second wave's doll of the first lane
    writes in the tic of the other lane's. The checker flags it and DOOM fails on the extreme input; the fixed
    compiler is clean and exact."""
    net = df.MLP([[[21, -4], [-24, -26]], [[-20, 22]]], [[26, 13], [-12]], [[5728, 5736], [84, 236]], signed_input=True)
    x = [5756, 5756]
    with monkeypatch.context() as m:
        m.setattr(pure, "_barrier_height", lambda floors: CLOSED)
        bad = tmp_path / "old.wad"
        compile_mlp(net, bad, opt=1)
    rep = df.check_schedule(bad, net=net)
    assert not rep.ok and rep.hazard_count > 0
    good = tmp_path / "fixed.wad"
    compile_mlp(net, good, opt=1)
    assert df.check_schedule(good, net=net).ok
    assert exact(net, bad, [x]) == [False] and exact(net, good, [x]) == [True]
