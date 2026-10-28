"""plan's calibrated settle estimate and its error band (doomformer/cost.py), against the committed measured settles
(models/schedule/plan_calibration.json); `plan` accepts a model directory or name. No game."""
import json
from pathlib import Path

import numpy as np
import pytest

import doomformer as df
from doomformer import cost
from doomformer.cli import _load

CAL = Path(__file__).resolve().parent.parent / "models" / "schedule" / "plan_calibration.json"


def test_refit_reproduces_constants_and_held_out_errors_stay_in_band():
    r = cost.refit(CAL)
    assert r["per_layer"] == pytest.approx(cost.SETTLE_PER_LAYER, abs=0.05)
    assert r["start"] == pytest.approx(cost.SETTLE_START, abs=0.5)
    for label in ("fit", "heldout"):
        assert r[label]
        for net, opt, measured, model, pct, diff in r[label]:
            band = max(cost.SETTLE_BAND_TICS, cost.SETTLE_BAND_FRACTION * measured)
            assert abs(diff) <= band, (label, net, opt, measured, model)


def test_plan_reports_calibrated_tics_and_band():
    p = df.plan(df.MLP([np.array([[3, -2], [1, 4]]), np.array([[1, 2]])], [np.array([5, 0]), np.array([0])],
                       [np.array([20, 40]), np.array([30, 60])]), opt=3)
    m = p["map"]
    assert m["estimated_settle_tics"] > 0 and m["calibrated_tics"] > 0
    assert m["tics_band"] >= cost.SETTLE_BAND_TICS and m["tics_error_pct"] == round(100 * m["tics_band"] / m["calibrated_tics"], 1)
    est = df.plan(df.MLP([np.array([[3, -2], [1, 4]]), np.array([[1, 2]])], [np.array([5, 0]), np.array([0])],
                         [np.array([20, 40]), np.array([30, 60])]), opt=3)["map"]["estimated_settle_tics"]
    assert est == m["estimated_settle_tics"]       # the geometry bound (the runner's timeout) is not calibrated


def test_every_calibration_row_has_a_measurement_and_terms():
    data = json.loads(CAL.read_text())
    assert len(data["fit"]) >= 40 and len(data["heldout"]) >= 24
    nets = {r["net"] for r in data["heldout"]}
    assert len(nets) >= 6 and not nets & {r["net"] for r in data["fit"]}      # held-out nets are not in the fit
    for r in data["fit"] + data["heldout"]:
        assert r["measured"] > 0 and r["terms"]["layers"] >= 2 and set(r["terms"]) == {
            "wave", "chunk", "sync", "corr", "margin", "copy", "layers"}


def test_load_accepts_model_directory_and_name():
    assert isinstance(_load(str(Path(__file__).resolve().parent.parent / "models" / "mnist")), df.MLP)
    assert isinstance(_load("mnist"), df.MLP)
    with pytest.raises(SystemExit) as e:
        _load("models/nosuchmodel")
    assert "mnist" in str(e.value) and "model directory" in str(e.value)
