"""The map-extent limit (docs/engine_limits.md): every one-map build and every tile map stays within MAX_SPAN per
axis, plan reports the span, plan/compile honour both modes' fits, the world hook refuses worlds beyond the limit, and
the player does not drift anywhere it accepts."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import vizdoom as vzd

import doomformer as df
from doomformer import pure, tile
from doomformer.intref import QModel
from doomformer.onemap import build_onemap, tiny_model
from doomformer.ops import crossbar, fx, output_single
from doomformer.tile import Block, Effect, Gate, Stage, TileSpec
from doomformer.tiled import tile_plan

MODELS = Path(__file__).resolve().parent.parent / "models"
NETS = [df.MLP([[[1, 1, 1]]], [[0]], [[1]]),                    # the nets that drifted beyond the limit
        df.MLP([[[-3, 2, 0]]], [[0]], [[10, 20]])]


def room(x0, y0, x1, y1):
    def world(m):
        m.rect(x0, y0, x1, y1, floor=0, ceil=256)
        m.thing((x0 + x1) // 2, (y0 + y1) // 2, 1)
        return ""
    return world


def test_layouts_stay_within_the_span_limit():
    cases = [(tiny_model(), None), (QModel.load(MODELS / "doomnames_onemap"), None)]
    for model, world in cases:
        layout = build_onemap(model, Path("unused.wad"), layout_only=True, world=world).layout
        x0, y0, x1, y1 = layout["bounds"]
        assert layout["fits"] and layout["span"] == [x1 - x0, y1 - y0]
        assert max(layout["span"]) <= pure.MAX_SPAN == 32000
        assert pure.X_ORIGIN <= x0 and x1 <= pure.X_END
        assert layout["world_area"] is not None


def test_plan_reports_span_and_world_area():
    net = df.MLP([np.ones((3, 4), int), np.ones((2, 3), int)], [np.zeros(3, int), np.zeros(2, int)],
                 [[5, 10], [5, 10]])
    p = df.plan(net)
    assert p["fits"] and p["span"] == p["map"]["span"] and max(p["span"]) <= pure.MAX_SPAN
    top = p["map"]["bounds"][3]
    assert p["map"]["world_area"] == [pure.X_ORIGIN, top - pure.MAX_SPAN, pure.X_END, pure.WORLD_TOP]


def test_too_large_layouts_fail_clearly(tmp_path):
    # 600 dolls that climb to a ceiling gate at 12,000 (18,321-unit corridors): one per row, 28,840 units tall.
    # The old 58,000-wide rows held three per row (9,600 tall).
    spec = TileSpec("tall")
    for i in range(600):
        gates = [Gate(spec.port("-x", "floor"), -1, "floor")]
        spec.blocks.append(Block(gates, [Stage(Gate(spec.port("k", "ceil"), 12000, "ceil"),
                                               fx(output_single(spec, f"y{i}"), 1))]))
    layout = pure.compile_pure([spec], tmp_path / "tall.wad", layout_only=True).layout
    assert not layout["fits"] and layout["span"][1] > pure.MAX_SPAN
    with pytest.raises(ValueError, match="folded layout exceeds"):
        pure.compile_pure([spec], tmp_path / "tall.wad")
    # one doll whose corridor alone is longer than a row (9,000 effect cells = 36,000 units)
    spec = TileSpec("long")
    crossbar(spec, "x", [1], [(output_single(spec, f"y{i}"), 1) for i in range(9000)], signed=False)
    with pytest.raises(ValueError, match="corridor width"):
        pure.compile_pure([spec], tmp_path / "long.wad", layout_only=True)


def test_tile_maps_obey_the_span_limit(tmp_path):
    # one doll whose zone holds 8,100 effect lines (32,400 units): the tile map is wider than the limit
    spec = TileSpec("wide")
    out = spec.output("y")
    spec.blocks.append(Block([Gate(spec.port("-x", "floor"), -1, "floor")], [Stage(None, [Effect(out, 1)] * 8100)]))
    layout = tile.compile_tile(spec, tmp_path / "wide.wad", layout_only=True).layout
    assert not layout["fits"] and layout["span"][0] > tile.MAX_SPAN == pure.MAX_SPAN
    with pytest.raises(ValueError, match="tile map spans"):
        tile.compile_tile(spec, tmp_path / "wide.wad")
    assert not (tmp_path / "wide.wad").exists()
    # MLP 1600x256: its layer-0 tile spans 32,488 units (tile_plan used to say fits=True)
    rng = np.random.default_rng(0)
    net = df.MLP([rng.integers(-3, 4, (256, 1600)), rng.integers(-3, 4, (4, 256))],
                 [np.zeros(256, int), np.zeros(4, int)], [[5, 10], [5, 10]])
    planned = tile_plan(net)
    assert not planned["fits"] and planned["span"] == [32488, 23952]
    assert "mlp_l0 spans 32488 x 23952" in planned["reason"]
    small = tile_plan(tiny_model())
    assert small["fits"] and max(small["span"]) <= tile.MAX_SPAN and "reason" not in small


def test_plan_and_compile_honour_both_modes_fits(tmp_path, monkeypatch):
    net = df.MLP([np.ones((3, 4), int), np.ones((2, 3), int)], [np.zeros(3, int), np.zeros(2, int)],
                 [[5, 10], [5, 10]])
    p = df.plan(net)
    assert p["mode"] == "map" and p["fits"] and p["tiles"]["fits"] and "reason" not in p
    monkeypatch.setattr(tile, "MAX_SPAN", 100)                     # every tile map is now too large
    p = df.plan(net)
    assert p["mode"] == "map" and p["fits"] and not p["tiles"]["fits"] and "reason" not in p
    with pytest.raises(ValueError, match="tiles do not fit"):
        df.compile(net, tmp_path / "t.wad", mode="tiles")
    assert df.plan(net, mode="tiles")["reason"] == p["tiles"]["reason"]
    monkeypatch.setattr(pure, "MAX_SPAN", 100)                     # and so is the one map
    p = df.plan(net)
    assert p["mode"] == "map" and not p["fits"] and p["reason"].startswith("fits neither one map")
    with pytest.raises(ValueError, match="fits neither one map"):
        df.compile(net, tmp_path / "n.wad")
    monkeypatch.setattr(tile, "MAX_SPAN", 32000)                   # tiles fit again: auto falls back to them
    p = df.plan(net)
    assert p["mode"] == "tiles" and p["tiles"]["fits"] and "reason" not in p


def test_world_beyond_the_limit_is_rejected(tmp_path):
    net = NETS[0]
    area = df.plan(net)["map"]["world_area"]
    x0, y0, x1, y1 = area
    with pytest.raises(ValueError, match=r"span 32001 x .*world inside \[-16000, "):
        df.compile(net, tmp_path / "east.wad", world=room(x1 - 2048, -7168, x1 + 1, -5120))
    with pytest.raises(ValueError, match=r"x 32001 units"):
        df.compile(net, tmp_path / "south.wad", world=room(-1024, y0 - 1, 1024, y0 + 2047))
    with pytest.raises(ValueError, match="spans"):                  # compile_pure itself, no kit hook
        pure.compile_pure(net._stages(), tmp_path / "raw.wad", world=room(x1, -7168, x1 + 2048, -5120))
    brain = df.compile(net, tmp_path / "corner.wad", world=room(x1 - 2048, y0, x1, y0 + 2048))
    assert brain.meta.layout["world_area"] == area and brain.meta.layout["span"] == [32000, 32000]


@pytest.mark.parametrize("corner", ["east", "south-east", "south-west"])
def test_no_drift_anywhere_the_hook_accepts(tmp_path, corner):
    """The rooms that drifted 422 units in 35 tics beyond the limit, at the extremes of the accepted world area."""
    for i, net in enumerate(NETS):
        x0, y0, x1, y1 = df.plan(net)["map"]["world_area"]
        rx = x1 - 2048 if corner != "south-west" else x0
        ry = -7168 if corner == "east" else y0
        with df.compile(net, tmp_path / f"{corner}{i}.wad", world=room(rx, ry, rx + 2048, ry + 2048)) as brain:
            assert max(brain.meta.layout["span"]) <= pure.MAX_SPAN
            g = brain.runner.game
            g.advance_action(35)
            assert (g.get_game_variable(vzd.GameVariable.POSITION_X),
                    g.get_game_variable(vzd.GameVariable.POSITION_Y)) == (rx + 1024, ry + 1024)
            x = np.array([40, -25, 10])
            np.testing.assert_array_equal(brain(x), net.reference(x)["y"])
            assert (g.get_game_variable(vzd.GameVariable.POSITION_X),
                    g.get_game_variable(vzd.GameVariable.POSITION_Y)) == (rx + 1024, ry + 1024)
