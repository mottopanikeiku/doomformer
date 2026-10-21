"""Small real-game checks of the multi-game Brain API and its zero-tolerance checks."""
from pathlib import Path

import numpy as np
import pytest

import doomformer as df
from doomformer import intref
from doomformer.onemap import tiny_model
from doomformer.tiled import TiledMLPBrain, TiledTransformerBrain


def test_transformer_tiles_verified(tmp_path):
    net = tiny_model()
    with df.compile(net, tmp_path / "tf", mode="tiles", workers=2) as brain:
        assert isinstance(brain, TiledTransformerBrain)
        assert isinstance(brain, df.Brain)
        for tokens in ([0], [0, 1], [0, 2, 1], [0]):
            np.testing.assert_array_equal(brain(tokens), intref.forward(net, tokens))
            assert brain.info["checked"] > 0
        assert brain.generate("A", max_new=1) == intref.generate(net, "A", max_new=1)
        assert brain.info["tile_runs"] == len(brain.pool.log) > brain.info["runs"]
        assert brain.engine.negation_errors() == 0
        # Corrupt an exposed intermediate (not the logits): cached context must be checked again.
        brain.engine.pos[0]["z0_0"] += 1
        with pytest.raises(df.MismatchError) as err:
            brain.logits(brain._tokens)
        assert any(name.startswith("z0:") for name, _, _ in err.value.mismatches)


def test_mlp_tiles_verified_and_mismatch(tmp_path):
    net = df.MLP([[[2, -1], [-1, 2]], [[1, -2]]], [[1, -1], [3]], [[1, 3], [1, 2]])
    x = np.array([[4, -2], [-2, 3]])
    with df.compile(net, tmp_path / "mlp.wad", mode="tiles", workers=1) as brain:
        assert isinstance(brain, TiledMLPBrain)
        np.testing.assert_array_equal(brain(x), net.reference(x)["y"])
        assert brain.info["tiles"] == 2 and brain.info["tile_runs"] == 4
        assert brain.info["checked"] == 5
        net.biases[-1][0] += 1
        with pytest.raises(df.MismatchError):
            brain(x[0])


def test_plan_both_modes_and_selection(tmp_path):
    net = tiny_model()
    report = df.plan(net, workers=2)
    assert report["map"]["tiles"] == 1
    assert report["tiles"]["tiles"] == 2 + net.L * (4 + 2 * net.T)
    assert report["map"]["dolls"] == report["dolls"]
    assert df.plan(net, mode="map")["mode"] == "map"
    assert df.plan(net, mode="tiles")["mode"] == "tiles"
    model = df.Transformer.load(Path(__file__).resolve().parents[1] / "models" / "doomnames")
    big = df.plan(model)
    assert big["mode"] == "tiles"
    assert not big["map"]["fits"] or big["map"]["estimated_settle_tics"] > big["auto_settle_tics"]
    small = df.MLP([[[1]]], [[0]], [[1]])
    assert df.plan(small)["mode"] == "map"
    with df.compile(small, tmp_path / "map.wad") as brain:
        assert brain.info["mode"] == "map"
        assert brain.info["dolls"] == df.plan(small)["dolls"]


def test_mode_errors(tmp_path):
    net = tiny_model()
    for kwargs in ({"mode": "bad"}, {"workers": 0}, {"workers": True}):
        with pytest.raises(ValueError):
            df.plan(net, **kwargs)
        with pytest.raises(ValueError):
            df.compile(net, tmp_path / "bad", **kwargs)
    spikes = df.SpikingNet(["n"], ["i"], [[0]], [[1]], theta=1, leak=0)
    with pytest.raises(ValueError, match="spiking"):
        df.compile(spikes, mode="tiles")
    with pytest.raises(ValueError, match="require mode='map'"):
        df.compile(net, mode="tiles", world=lambda m: "")
