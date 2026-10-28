"""Coverage of the exactness check (doomformer/exposure.py, df.coverage): the check compares every value the reference
defines, for an MLP, a spiking net and a transformer, in one map and in tiles; a changed value is detected for every
name; and a lowering that stops exposing one value is reported."""
from __future__ import annotations

import numpy as np
import pytest

import doomformer as df
from doomformer.onemap import exposed_names, tiny_model
from doomformer.spiking import IntNet

NET = df.MLP([[[5, -3], [2, 7], [1, 1]], [[1, 2, 3]]], [[1, -2, 0], [5]], [[1, 3], [1, 2]])
SPIKING = IntNet(["a", "b", "c"], ["u", "w"], [[0, 2, 0], [1, 0, 0], [0, 3, -1]], [[2, 0], [0, 1], [1, 1]], 3, 1)


def test_mlp_one_map_compares_every_value(tmp_path):
    cov = df.coverage(NET, [3, -2])
    assert cov.complete, cov
    assert (cov.reference_total, cov.twins_total, cov.compared) == (4, 3, 7)      # 3 hidden + 1 output, 3 twins
    assert cov.detected == cov.perturbed == 7 and not cov.missing and not cov.undetected
    assert cov.counts == {"staircase counts": 5} and cov.comparisons == 14     # defined, not exposed
    assert "complete" in str(cov) and cov.as_dict()["compared"] == 7


def test_mlp_tiles_compare_every_value():
    cov = df.coverage(NET, [3, -2], mode="tiles")
    assert cov.complete and cov.mode == "tiles", cov
    assert (cov.reference_total + cov.twins_total, cov.compared, cov.detected) == (7, 7, 7)


def test_spiking_compares_every_defined_value_and_lists_the_twin_planes_it_does_not_read():
    cov = df.coverage(SPIKING, [1, 0])
    assert cov.complete, cov
    assert (cov.reference_total, cov.compared, cov.detected) == (6, 6, 6)        # 3 spikes + 3 potentials
    assert cov.twins_total == 0 and cov.counts == {"twin planes -s (read by gates only)": 3}


def test_transformer_one_map_and_tiles():
    m = tiny_model()
    cov = df.coverage(m)
    assert cov.complete, cov
    assert cov.compared == cov.reference_total + cov.twins_total == 225 and cov.padding == 0
    assert cov.detected == cov.perturbed == 225
    tiles = df.coverage(m, mode="tiles")                                        # the tile engine's own count
    assert tiles.complete and tiles.compared == 225 and tiles.detected is None


def test_negative_control_a_dropped_readout_name_is_reported(tmp_path):
    """The lowering silently stops exposing one name (the name-set guard of v0.5.0 is bypassed by editing both the
    readout page and the brain's expected set); the check compares one value less and coverage fails, naming it."""
    with df.compile(NET, tmp_path / "net.wad", mode="map", limits=False) as brain:
        runner_pages = brain.runner.meta.pages
        dropped = runner_pages[-1].pop()                                       # the last name: no slot shifts
        brain._exposed = NET.exposed_names() - {dropped}
        cov = df.coverage(brain, [3, -2])
        assert not cov.complete and cov.compared == 6
        assert dropped in cov.missing + cov.twins_unread, (dropped, cov)
        assert "INCOMPLETE" in str(cov)


def test_the_name_guard_still_catches_a_one_sided_drop(tmp_path):
    with df.compile(NET, tmp_path / "net.wad", mode="map", limits=False) as brain:
        brain.runner.meta.pages[-1].pop()
        with pytest.raises(df.ReadoutError):
            df.coverage(brain, [3, -2])


def test_reference_names_do_not_come_from_the_lowering():
    """The reference side of the count is built from net.reference: weakening exposed_names does not shrink it."""
    from doomformer.exposure import _mlp_reference
    sums, twins, _, _ = _mlp_reference(NET, np.array([3, -2]))
    assert sum(map(len, sums.values())) + sum(map(len, twins.values())) == len(NET.exposed_names()) == 7
    assert exposed_names(tiny_model())
