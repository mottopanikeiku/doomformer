"""Reference-only checks of the FlyWire-derived circuits in models/fly (no DOOM)."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from doomformer.spiking import IntNet

ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize("seed", range(5))
def test_fly_feeding_logic(seed):
    """The committed FlyWire circuit: sugar drives MN9, adding bitter silences it, no input stays silent."""
    net = IntNet.load(ROOT / "models" / "fly" / "feeding.json")
    groups, p = net.meta["input_groups"], net.meta["input_p_per_step"]
    mn9 = [net.names.index("MN9_left"), net.names.index("MN9_right")]
    counts = {}
    for cond in ("none", "sugar", "sugar+bitter"):
        rng = np.random.default_rng(1000 + seed)
        X = np.zeros((120, len(net.inputs)), dtype=np.int64)
        for grp in ([] if cond == "none" else cond.split("+")):
            X[:, groups[grp]] = rng.random((120, len(groups[grp]))) < p
        counts[cond] = int(net.run(X)[1][:, mn9].sum())
    assert counts["none"] == 0
    assert counts["sugar"] > 0
    assert counts["sugar+bitter"] == 0


BRAIN_OUTPUTS = {"feeding": ["MN9_left", "MN9_right"], "escape": ["GF_1", "GF_2"], "grooming": ["aDN1_right"]}
BRAIN_STIMULI = {"none": None, "sugar": "feeding", "sugar+bitter": None, "lc4": "escape", "jo": "grooming"}


@pytest.mark.parametrize("seed", range(3))
def test_fly_brain_each_stimulus_drives_only_its_behaviour(seed):
    """The combined FlyWire circuit: sugar -> MN9, looming (LC4) -> Giant Fiber, antennal touch (JO) -> aDN1;
    every other output stays silent (no crosstalk); bitter silences feeding; no input stays silent."""
    net = IntNet.load(ROOT / "models" / "fly" / "brain.json")
    groups, p = net.meta["input_groups"], net.meta["input_p_per_step"]
    for stim, behaviour in BRAIN_STIMULI.items():
        rng = np.random.default_rng(2000 + seed)
        X = np.zeros((120, len(net.inputs)), dtype=np.int64)
        for grp in ([] if stim == "none" else stim.split("+")):
            X[:, groups[grp]] = rng.random((120, len(groups[grp]))) < p[grp]
        S = net.run(X)[1]
        for name, outs in BRAIN_OUTPUTS.items():
            n = int(S[:, [net.names.index(o) for o in outs]].sum())
            assert (n > 0) == (name == behaviour), f"{stim}: {name} fired {n} times"
