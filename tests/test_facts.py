"""`doomformer facts`: the engine facts of the docs as probe maps, run in ViZDoom (about 15 s for all of them)."""
from __future__ import annotations

import json

import pytest

from doomformer import facts


def test_every_fact_holds_in_this_engine(tmp_path):
    rows = facts.run(workdir=tmp_path)
    names = {r.probe for r in rows}
    assert len(names) >= 15 and names == {p.name for p in facts.PROBES}
    failed = [f"{r.case}: expected {r.expected}, observed {r.observed} {r.error}" for r in rows if not r.ok]
    assert not failed, "\n".join(failed)


def test_a_wrong_expectation_is_a_fail_row(tmp_path):
    case = facts._light_saturates()[0]
    case.expected = dict(case.expected, down=-32767)
    p = facts.Probe("light_wrong", "", "", lambda: [case])
    (row,) = facts.run_probe(p, tmp_path)
    assert not row.ok and row.observed["down"] == -32768 and row.expected["down"] == -32767


def test_export_and_check(tmp_path, capsys):
    names = facts.export(tmp_path, only=["light_saturates", "mover_same_tic", "voodoo_tid"])
    assert names == ["light_saturates", "mover_same_tic", "voodoo_tid"]
    expected = (tmp_path / "expected.txt").read_text().splitlines()
    assert expected[0] == "FACT light_saturates up1=30160 up2=32767 down=-32768"
    assert all((tmp_path / f"{n}.wad").is_file() for n in names)
    run_txt = (tmp_path / "RUN.txt").read_text()
    assert "gzdoom" in run_txt and "were not run" in run_txt
    for name, ok, got in facts.check_export(tmp_path):
        assert ok, f"{name}: {got}"
    assert facts.main(["--only", "voodoo_tid", "--json"]) == 0
    assert json.loads(capsys.readouterr().out)["rows"][0]["ok"] is True


def test_unknown_probe_is_refused():
    with pytest.raises(KeyError, match="unknown probe"):
        facts.run(["nope"])
