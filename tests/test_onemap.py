"""Full random-weight transformer regression in a single real DOOM game."""
import numpy as np

from doomformer.intref import forward
from doomformer.onemap import (argmax_from_readout, argmax_reference, build_onemap, compare_trace,
                               logits_from_readout, tiny_model, token_ports)
from doomformer.runtime import TileRunner
from doomformer.ops import crossbar, output_pair, output_single
from doomformer.pure import compile_pure
from doomformer.tile import TileSpec


def test_tiny_transformer_in_one_map(tmp_path):
    model = tiny_model()
    wad = tmp_path / "transformer.wad"
    meta = build_onemap(model, wad)
    planned = build_onemap(model, tmp_path / "unused.wad", layout_only=True)
    assert planned.n_sectors == meta.n_sectors
    assert planned.layout["bounds"] == meta.layout["bounds"]
    assert set(meta.ports) == {f"floor:{sign}tok:{t}" for sign in ("", "-") for t in range(model.T)}
    runner = TileRunner(wad)
    try:
        for tokens in ([0, 1, 2], [3, 4, 0], [0], [2, 2], [4, 3, 1]):
            trace = {}
            expected = forward(model, list(tokens), trace)
            got = runner.run(token_ports(model, list(tokens)))[0]
            assert compare_trace(got, trace, len(tokens)) > 0
            np.testing.assert_array_equal(logits_from_readout(model, got, len(tokens)), expected)
    finally:
        runner.close()


def test_comparator_reference_is_numpy_argmax():
    """The dolls' pairwise-comparison argmax (wins_a >= k-1) equals np.argmax, lowest index on ties."""
    rng = np.random.default_rng(5)
    model = tiny_model()
    for _ in range(2000):
        k = int(rng.integers(2, 6))
        logits = np.zeros((model.T, len(model.vocab)), dtype=np.int64)
        logits[-1] = rng.integers(-3, 4, len(model.vocab))           # many ties
        choose = list(rng.permutation(len(model.vocab))[:k])
        assert argmax_reference(model, choose, logits, {}) == int(np.argmax(logits[-1][choose]))


def test_dolls_choose_the_argmax(tmp_path):
    """4-way doll argmax after the logits (tokens 1 and 2 have identical unembedding rows, so they always tie):
    every comparator value, win count and the one-hot winner are exact; ties go to the lower index."""
    model = tiny_model()
    model.p["Wun"][2], model.p["bun"][2] = model.p["Wun"][1], model.p["bun"][1]
    choose = [1, 2, 3, 4]
    wad = tmp_path / "choose.wad"
    meta = build_onemap(model, wad, choose=choose)
    assert [s["name"] for s in meta.layout["stage_stats"]][-2:] == ["argmax.compare", "argmax.winner"]
    runner = TileRunner(wad)
    try:
        winners = []
        for tokens in ([0, 0, 0], [0, 1, 0], [0, 0, 3]):
            trace = {}
            expected = forward(model, tokens, trace)
            best = argmax_reference(model, choose, expected, trace)
            got = runner.run(token_ports(model, tokens))[0]
            assert compare_trace(got, trace, len(tokens)) == len(got)
            np.testing.assert_array_equal(logits_from_readout(model, got, len(tokens)), expected)
            winners.append(argmax_from_readout(got, len(choose), model.T - 1))
            assert winners[-1] == best == int(np.argmax(expected[-1][choose]))
        assert winners == [0, 2, 0]                                    # [6, 6, 3, -7]: the tie goes to index 0
    finally:
        runner.close()


def test_folded_clock_random_layer(tmp_path):
    """More than 12,000 units of release events must survive a clock teleport."""
    rng = np.random.default_rng(123)
    weights = rng.integers(-3, 4, 205)
    weights[weights == 0] = 1
    first, second = TileSpec("random"), TileSpec("read")
    hidden = output_pair(first, "h")
    for w in weights:
        crossbar(first, "x", [1], [(hidden, int(w))], signed=False)
    result = output_single(second, "y")
    crossbar(second, "h", [1, 10, 20], [(result, 1)], signed=False)
    wad = tmp_path / "folded.wad"
    meta = compile_pure([first, second], wad, readout=["-h"])
    assert meta.layout["stage_stats"][0]["waves"] == 205
    runner = TileRunner(wad)
    try:
        for x in (1, 0, 1):
            got = runner.run({"-x": -x})[0]
            expected = int(weights.sum()) * x
            assert got["-h"] == -expected
            assert got["y"] == sum(expected >= t for t in (1, 10, 20))
    finally:
        runner.close()


def test_same_wave_collision_is_detected(tmp_path):
    """Negative control: two same-wave hits lose a plane move, not a light hit."""
    for simultaneous in (False, True):
        first, second = TileSpec("write"), TileSpec("read")
        hidden = output_pair(first, "h")
        crossbar(first, "x", [1, 1], [(hidden, 1)], signed=False)
        result = output_single(second, "y")
        crossbar(second, "h", [1, 2], [(result, 1)], signed=False)
        wad = tmp_path / f"collision{simultaneous}.wad"
        compile_pure([first, second], wad, readout=["-h"], serial={0: 0} if simultaneous else None)
        runner = TileRunner(wad)
        try:
            got = runner.run({"-x": -1})[0]
            exact = got["-h"] == -2 and got["y"] == 2
            assert exact != simultaneous
        finally:
            runner.close()
