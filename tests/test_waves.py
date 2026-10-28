"""Wave scheduling at opt >= 1 (doomformer/waves.py, docs/scheduling.md): the colourings, the exclusivity analysis, and
the engine facts the relaxations rest on (a light line loses no same-tic hit; a plane mover does)."""
import numpy as np
import pytest

from doomformer import waves
from doomformer.onemap import argmax_layers
from doomformer.ops import crossbar, equals, output_pair, output_single
from doomformer.pure import compile_pure, opt_passes
from doomformer.runtime import TileRunner
from doomformer.tile import TileSpec


def _stats(layers, opt, **kw):
    return compile_pure(layers, "unused.wad", layout_only=True, opt=opt, **kw).layout["stage_stats"]


def _waves(layers, opt, **kw):
    return [s["waves"] for s in _stats(layers, opt, **kw)]


def test_opt_levels_are_validated():
    assert opt_passes(0) == frozenset()
    assert "tuned" in opt_passes(2) and "tuned" not in opt_passes(1)
    assert opt_passes({"lights"}) == frozenset({"lights"})
    assert "copy" in opt_passes(3) and "copy" not in opt_passes(2)
    for bad in (4, -1, True, 1.0, {"nope"}, "1"):
        with pytest.raises(ValueError):
            opt_passes(bad)


@pytest.mark.parametrize("k", range(2, 13))
def test_circle_method_is_a_minimal_proper_edge_colouring(k):
    col = waves.circle_colours(k)
    assert len(col) == k * (k - 1) // 2
    for v in range(k):                       # at each vertex all colours differ
        around = [c for (a, b), c in col.items() if v in (a, b)]
        assert len(around) == len(set(around))
    assert len(set(col.values())) == (k - 1 if k % 2 == 0 else k)


def test_comparison_layer_is_coloured_optimally():
    """k scores compared pairwise (k (k - 1) / 2 dolls, two targets each): first fit needs up to twice the optimum
    (15 waves for k = 10); DSATUR plus the circle method give k - 1 (even k) or k (odd k) for every k."""
    assert _waves(argmax_layers(10, [5]), 0)[0] == 15
    info = _stats(argmax_layers(10, [5]), 1)[0]
    assert info["waves"] == 9 and info["lower_bound"] == 9
    assert _waves(argmax_layers(10, [5]), {"exclusive", "lights"})[0] == 15      # first fit: the colouring pass is off
    rr = 0
    for k in range(2, 21):
        st = _stats(argmax_layers(k, [0]), 1)[0]
        assert st["waves"] == max(1, k - 1 if k % 2 == 0 else k), k
        rr += st["round_robin"]
    assert rr > 0                                                                  # DSATUR alone is not always optimal


def test_dsatur_colours_cliques_cycles_and_bipartite_graphs_exactly():
    def adj(edges, n):
        out = {i: set() for i in range(n)}
        for a, b in edges:
            out[a].add(b)
            out[b].add(a)
        return out
    n = 9
    clique = adj([(a, b) for a in range(n) for b in range(a + 1, n)], n)
    even = adj([(i, (i + 1) % 8) for i in range(8)], 8)
    odd = adj([(i, (i + 1) % 9) for i in range(9)], 9)
    for g, colours in ((clique, n), (even, 2), (odd, 3)):
        col = waves.dsatur(g)
        assert max(col.values()) + 1 == colours
        assert all(col[a] != col[b] for a in g for b in g[a])


def test_exclusive_dolls_share_a_wave_and_others_do_not():
    """One-hot lookups (equals) and the plus / minus dolls of a signed read can never both pass their gates."""
    s = TileSpec("lookup")
    z = output_pair(s, "z")
    for token in range(6):
        equals(s, "tok", token, [(z, token + 1)])
    t = TileSpec("read")
    crossbar(t, "z", [1, 3], [(output_single(t, "y"), 1)], signed=True)
    w = TileSpec("signed")
    h = output_pair(w, "h")
    crossbar(w, "x", [2, 5], [(h, 3)], signed=True)      # lanes: x>=2, x>=5, -x>=2, -x>=5
    r = TileSpec("sink")
    crossbar(r, "h", [1], [(output_single(r, "o"), 1)], signed=True)
    assert _waves([s, t], 0) == [6, 4]                   # at opt 0 the four dolls of the light-only read conflict
    assert _waves([s, t], 1) == [1, 1]
    assert _waves([w, r], 0) == [4, 2]
    assert _waves([w, r], 1) == [2, 1]                   # x>=2 and x>=5 both pass for x=7: two waves, signs share
    assert _waves([w, r], {"lights"}) == [4, 1]
    assert _waves([w, r], {"exclusive"}) == [2, 1]


def test_values_written_by_the_layer_are_not_assumed_constant():
    """A gate on a value the same layer writes (a state value) gives no exclusivity; on another value it does."""
    own, other = TileSpec("self"), TileSpec("other")
    h, k = output_pair(own, "h"), output_pair(other, "k")
    for v in (1, 2):
        equals(own, "h", v, [(h, 1)])
        equals(other, "g", v, [(k, 1)])
    assert _waves([own], 0, state={"h": 0, "-h": 0}) == [2]
    assert _waves([own], 1, state={"h": 0, "-h": 0}) == [2]
    nxt = TileSpec("next")
    crossbar(nxt, "k", [1], [(output_single(nxt, "o"), 1)], signed=False)
    assert _waves([other, nxt], 0) == [2, 1]
    assert _waves([other, nxt], 1) == [1, 1]


def test_lights_only_layer_is_one_wave():
    s = TileSpec("sums")
    y = output_single(s, "y")
    for _ in range(40):
        crossbar(s, "x", [1], [(y, 2)], signed=False)
    assert _waves([s], 0) == [40]
    assert _waves([s], 1) == [1]


def _two_dolls(tmp_path, name, **kw):
    """Two dolls pass at x = 1 and both add to h (a light) and to -h (a plane the next layer reads)."""
    first, second = TileSpec("write"), TileSpec("read")
    hidden = output_pair(first, "h")
    crossbar(first, "x", [1, 1], [(hidden, 1)], signed=False)
    result = output_single(second, "y")
    crossbar(second, "h", [1, 2], [(result, 1)], signed=False)
    wad = tmp_path / f"{name}.wad"
    compile_pure([first, second], wad, readout=["-h"], **kw)
    return wad


def test_light_hits_in_one_tic_are_not_lost_but_plane_moves_are(tmp_path):
    """The relaxation 'lights' rests on this: two dolls released together add 2 to the light `h` but the second
    plane move on `-h` is dropped. (tests/test_onemap.py::test_same_wave_collision_is_detected checks the plane.)"""
    runner = TileRunner(_two_dolls(tmp_path, "together", serial={0: 0}))
    try:
        got = runner.run({"-x": -1})[0]
        assert got["h"] == 2, "a light line lost a same-tic hit"
        assert got["-h"] != -2, "the plane collision did not happen: the negative control is void"
    finally:
        runner.close()


def test_opt1_schedule_is_exact_on_the_small_cases(tmp_path):
    """Signed read of a two-sided value at opt 1 (plus and minus dolls in one wave) and a sink layer with lights."""
    rng = np.random.default_rng(5)
    first, second = TileSpec("signed"), TileSpec("sink")
    h = output_pair(first, "h")
    crossbar(first, "x", [2, 5, 9], [(h, 3)], signed=True)
    r = output_single(second, "y")
    crossbar(second, "h", [1, 4], [(r, 2)], signed=True)
    wads = {opt: tmp_path / f"o{opt}.wad" for opt in (0, 1)}
    for opt, wad in wads.items():
        compile_pure([first, second], wad, readout=["h", "-h"], opt=opt)
    runners = {opt: TileRunner(wad) for opt, wad in wads.items()}
    try:
        tics = {0: 0, 1: 0}
        for x in [int(v) for v in rng.integers(-12, 13, 8)] + [0, 12, -12, 5, -5]:
            outs = {}
            for opt, runner in runners.items():
                outs[opt], tics[opt], _, _ = runner.run({"x": x, "-x": -x})
            ctr = sum(x >= t for t in (2, 5, 9)) - sum(-x >= t for t in (2, 5, 9))
            assert outs[0] == outs[1]
            assert outs[1]["-h"] == -3 * ctr and outs[1]["h"] == 3 * ctr
            assert outs[1]["y"] == 2 * (sum(3 * ctr >= t for t in (1, 4)) - sum(-3 * ctr >= t for t in (1, 4)))
        assert tics[1] < tics[0]
    finally:
        for runner in runners.values():
            runner.close()


COPY = frozenset({"exclusive", "lights", "colour", "copy"})


def _mlp(seed=3, sizes=(5, 7, 4, 3), signed=True):
    import doomformer as df
    rng = np.random.default_rng(seed)
    return df.MLP([rng.integers(-9, 10, (sizes[l + 1], sizes[l])) for l in range(len(sizes) - 1)],
                  [rng.integers(-20, 21, sizes[l + 1]) for l in range(len(sizes) - 1)],
                  [np.cumsum(rng.integers(3, 12, 3)) for _ in range(len(sizes) - 1)], signed_input=signed)


def test_copy_bus_layout_one_wave_per_layer_and_actions_keep_planes():
    from doomformer.kit import _build
    net = _mlp()
    plain = _build(net, "unused.wad", layout_only=True, opt=1)
    copy = _build(net, "unused.wad", layout_only=True, opt=COPY)
    assert [s["waves"] for s in copy.layout["stage_stats"]] == [1, 1, 1]
    assert [bool(s.get("copy")) for s in copy.layout["stage_stats"]] == [True, True, False]
    assert copy.n_sectors < plain.n_sectors                         # the negated twins have no effect lines
    assert copy.layout["estimated_settle_tics"] < plain.layout["estimated_settle_tics"]
    # a value with an action keeps its plane accumulator (the action line needs the hit): layer 0 is not copied
    act = {f"h1_{j}": [dict(special=80, arg0=1)] for j in range(7)}
    kept = compile_pure(net._stages(), "unused.wad", layout_only=True, opt=COPY, actions=act)
    assert not kept.layout["stage_stats"][0].get("copy") and kept.layout["stage_stats"][0]["waves"] > 1


def test_copy_bus_is_exact_in_doom_and_faster(tmp_path):
    import doomformer as df
    rng = np.random.default_rng(11)
    net = _mlp()
    with df.compile(net, tmp_path / "a.wad", mode="map", opt=0) as a, \
            df.compile(net, tmp_path / "c.wad", mode="map", opt=COPY) as c:
        for x in [rng.integers(-40, 41, net.n_in) for _ in range(6)] + [np.zeros(net.n_in, int), np.full(net.n_in, 40)]:
            ya, yc = a(x), c(x)                                     # each call checks every exposed value
            assert np.array_equal(ya, yc)
        assert c.info["checked"] == a.info["checked"]
        assert c.info["tics"] < a.info["tics"]


def test_barrier_height_avoids_the_two_tic_float():
    """A barrier needing two mover tics stops its first tic 1..24 above a walkway in -5774..-5751 when it starts at
    13,000; the doll steps onto it and arrives about 7 tics late (found by `doomformer fuzz`). Layer barriers start lower."""
    from doomformer.pure import _barrier_height
    from doomformer.tile import CLOSED
    assert _barrier_height({0, 100, -2000}) == CLOSED
    for f in (-5774, -5764, -5751):
        h = _barrier_height({f, 0})
        assert h < CLOSED and h > f + 24 and not 1 <= h - 18750 - f <= 24


def test_late_barrier_float_regression(tmp_path):
    """Two nets from the fuzzer whose lane floors sit in the float window: one plane move was lost at every opt."""
    import doomformer as df
    for net, xs in ((df.MLP([[[156]], [[3]]], [[1617], [0]], [[5748, 8058], [11999]]), (-9000, 9000)),
                    (df.MLP([[[255]], [[3]]], [[0], [0]], [[5740, 11999], [11999]]), (12000,))):
        for opt in (0, 1):
            with df.compile(net, tmp_path / f"n{opt}.wad", mode="map", opt=opt) as brain:
                for x in xs:
                    brain.run([x], check=True)                  # raises MismatchError on any differing value
