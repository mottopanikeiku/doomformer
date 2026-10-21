"""Settling on maps with more dolls than ViZDoom lists (runtime.VIZ_MAX_OBJECTS = 4096, docs/engine_limits.md).

4,201 dolls wait at their layer's sync barrier while the (unlisted) clock doll walks to the release line, then a
second layer reads the first one's result. A check that only looks at the listed objects sees 4,096 resting dolls
and stops before anything was computed; TileRunner.settle must wait for every doll."""
from __future__ import annotations

import pytest

from doomformer import runtime
from doomformer.ops import crossbar, output_pair, output_single
from doomformer.pure import compile_pure
from doomformer.runtime import VIZ_MAX_OBJECTS, TileRunner
from doomformer.tile import Block, Gate, Stage, TileSpec

FILLERS = 4200


@pytest.fixture(scope="module")
def big_wad(tmp_path_factory):
    first, second = TileSpec("crowd"), TileSpec("read")
    crossbar(first, "x", [1], [(output_pair(first, "h"), 1)], signed=False)        # h = [x >= 1]
    gate = Gate(first.port("-x", "floor"), -1, "floor")
    first.blocks += [Block([gate], [Stage(None, [])]) for _ in range(FILLERS)]     # dolls with nothing to add
    crossbar(second, "h", [1], [(output_single(second, "y"), 1)], signed=False)   # y = [h >= 1]
    wad = tmp_path_factory.mktemp("settle") / "crowd.wad"
    meta = compile_pure([first, second], wad, readout=["-h"])
    assert meta.n_dolls == FILLERS + 4 > VIZ_MAX_OBJECTS
    return wad


def test_more_dolls_than_vizdoom_lists_settle_exactly(big_wad):
    runner = TileRunner(big_wad)
    try:
        assert len(runner.game.get_state().objects) == VIZ_MAX_OBJECTS          # the engine fact
        for x in (1, 0, 1):
            got, tics, _, _ = runner.run({"-x": -x})
            assert (got["-h"], got["y"]) == (-x, x), (x, got, tics)
    finally:
        runner.close()


def test_listed_objects_alone_settle_too_early(big_wad, monkeypatch):
    """Negative control: believing the object list is complete reproduces the false settle."""
    monkeypatch.setattr(runtime, "VIZ_MAX_OBJECTS", 10 ** 9)
    runner = TileRunner(big_wad)
    try:
        got, tics, _, _ = runner.run({"-x": -1})
        assert (got["-h"], got["y"]) != (-1, 1), tics
    finally:
        runner.close()


def test_old_maps_without_the_motion_script_refuse(big_wad):
    runner = TileRunner(big_wad)
    try:
        del runner.meta.layout["motion_script"]
        runner.start({"-x": -1})
        with pytest.raises(RuntimeError, match="recompile"):
            runner.settle()
    finally:
        runner.close()
