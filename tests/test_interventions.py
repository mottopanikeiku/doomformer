"""Interventions (zero, set, mean, resample, scale) on the integer reference and on the dolls, and patch recovery.

The reference tests need no game. The DOOM tests run the tiny transformer in tile maps: every intervened run equals the
equally intervened reference at tolerance 0, and a wrong patch value fails by the amount the reference predicts.
"""
from __future__ import annotations

import numpy as np
import pytest

import doomformer as df
from doomformer import intervene as iv
from doomformer import patching
from doomformer.intref import QModel, forward, generate
from doomformer.onemap import tiny_model
from doomformer.orchestrator import DoomTransformer

TOKS = [0, 2, 1]
DONOR = [0, 3, 4]


def trace(m, tokens, **kw):
    tr = {}
    forward(m, tokens, tr, **kw)
    return tr


def test_zero_interventions_equal_ablate():
    m = tiny_model()
    ab = {"x0": [0, 3], "L0.m": [2], "L0.ao": [1], "L0.x1": [0], "L0.x2": [3]}
    want = forward(m, TOKS, ablate=ab)
    got = forward(m, TOKS, interventions=[iv.zero(k, units=u) for k, u in ab.items()])
    assert np.array_equal(want, got)
    assert np.array_equal(want, forward(m, TOKS, ablate=ab, interventions=[]))
    assert generate(m, "A", ablate=ab, max_new=1) == generate(m, "A", interventions=[iv.zero("L0.m", units=[2])] + [
        iv.zero(k, units=u) for k, u in ab.items() if k != "L0.m"], max_new=1)


def test_positions_units_and_heads():
    m = tiny_model()
    plain = trace(m, TOKS)
    tr = trace(m, TOKS, interventions=[iv.set_value("L0.x1", 2, units=[1, 2], pos=[1])])
    want = plain["L0.x1"].copy()
    want[1, [1, 2]] = 2
    assert np.array_equal(tr["L0.x1"], want)
    assert np.array_equal(tr["L0.z1"], plain["L0.z1"])                 # the pre-activation is not patched
    # heads: ao of head h = units h*dh .. (h+1)*dh - 1 (tiny_model has one head: all units)
    assert np.array_equal(trace(m, TOKS, interventions=[iv.zero("L0.ao", heads=[0])])["L0.ao"], np.zeros((3, 4), int))
    xl = QModel.load("doomplayer")                                      # H = 2, dh = 8
    ctx = [0, 12, 23]
    tr = trace(xl, ctx, interventions=[iv.zero("L0.v", heads=[1], pos=[2])])
    assert (tr["L0.v"][2, 8:] == 0).all() and np.array_equal(tr["L0.v"][:2], trace(xl, ctx)["L0.v"][:2])
    assert np.array_equal(tr["L0.v"][2, :8], trace(xl, ctx)["L0.v"][2, :8])
    assert iv.compile_plan(xl, None, [iv.zero("L0.k", heads=[1])]).ops[0].units == tuple(range(8, 16))
    # positions beyond the run are skipped: a prefix run of the same context agrees with the full run
    plan = iv.compile_plan(m, None, [iv.set_value("L0.x2", 1, pos=[2])])
    assert np.array_equal(forward(m, TOKS[:2], interventions=plan), forward(m, TOKS)[:2])


def test_resample_mean_scale_values():
    m = tiny_model()
    clean, own = trace(m, DONOR), trace(m, TOKS)
    assert np.array_equal(forward(m, TOKS, interventions=[iv.resample("L0.ao", TOKS)]), forward(m, TOKS))
    got = trace(m, TOKS, interventions=[iv.resample("L0.v", DONOR, pos=[1, 2], src_pos=[2, 1])])
    assert np.array_equal(got["L0.v"][1], clean["L0.v"][2]) and np.array_equal(got["L0.v"][2], clean["L0.v"][1])
    assert np.array_equal(got["L0.v"][0], own["L0.v"][0])
    data = [TOKS, DONOR, [0, 1, 1], [0, 4]]
    acts = [trace(m, s)["L0.x2"] for s in data]
    half_up = lambda rows: (2 * np.sum(rows, 0) + len(rows)) // (2 * len(rows))
    got = trace(m, TOKS, interventions=[iv.mean("L0.x2", data)])["L0.x2"]
    assert np.array_equal(got[1], half_up([a[1] for a in acts])) and np.array_equal(got[2], half_up([a[2] for a in acts[:3]]))
    pooled = trace(m, TOKS, interventions=[iv.mean("L0.x2", data, by="all")])["L0.x2"]
    assert (pooled == half_up(np.concatenate(acts))).all()
    got = trace(m, TOKS, interventions=[iv.scale("L0.x1", 3, 2)])["L0.x1"]
    x = own["L0.x1"]
    assert np.array_equal(got, np.clip(np.sign(x) * (np.abs(x) * 3 // 2), -2, 2))
    assert np.array_equal(trace(m, TOKS, interventions=[iv.scale("L0.x1", -1)])["L0.x1"], -own["L0.x1"])


def test_errors():
    m = tiny_model()
    with pytest.raises(ValueError, match="not reachable"):
        forward(m, TOKS, interventions=[iv.set_value("L0.x1", 3)])            # K = 2
    with pytest.raises(ValueError, match="not reachable"):
        forward(m, TOKS, interventions=[iv.set_value("L0.m", -1)])            # m is a relu count
    with pytest.raises(ValueError, match="unknown site"):
        iv.compile_plan(m, None, [iv.zero("L0.nope")])
    with pytest.raises(ValueError, match="1 layer"):
        iv.compile_plan(m, None, [iv.zero("L1.x1")])
    with pytest.raises(ValueError, match="units or heads"):
        iv.zero("L0.ao", units=[0], heads=[0])
    with pytest.raises(ValueError, match="units must be"):
        iv.compile_plan(m, None, [iv.zero("L0.x1", units=[9])])
    with pytest.raises(ValueError, match="heads apply"):
        iv.compile_plan(m, None, [iv.zero("L0.x1", heads=[0])])
    with pytest.raises(ValueError, match="src_pos outside"):
        iv.compile_plan(m, None, [iv.resample("L0.x1", [0, 1], pos=[2])])
    dup = tiny_model()
    dup.p["L0.thr_x1"] = np.array([2, 2])                                      # level 1 cannot be reached
    with pytest.raises(ValueError, match="not reachable"):
        forward(dup, TOKS, interventions=[iv.set_value("L0.x1", 1)])
    forward(dup, TOKS, interventions=[iv.set_value("L0.x1", 2)])
    gated = tiny_model()
    gated.mlp = "reglu"                                                        # m = g * u: zero only
    iv.compile_plan(gated, None, [iv.zero("L0.m")])
    with pytest.raises(ValueError, match="zero only"):
        iv.compile_plan(gated, None, [iv.set_value("L0.m", 1)])


def softmax_model():
    m = tiny_model()
    m.attn, m.R = "softmax", 2
    m.p = {k: v for k, v in m.p.items() if k != "L0.thr_o"}
    return m


@pytest.fixture(scope="module")
def tiles(tmp_path_factory):
    net = tiny_model()
    with df.compile(net, tmp_path_factory.mktemp("tiles") / "tf", mode="tiles", workers=2) as brain:
        yield net, brain


def test_softmax_attention_ao_patches_exact_in_doom(tmp_path):
    """ao of softmax attention is a count of P/N lanes: the host writes the lanes of the new count."""
    net = softmax_model()
    assert net.lv(0) == 3
    with df.compile(net, tmp_path / "soft", mode="tiles", workers=2) as brain:
        eng = DoomTransformer(net, brain.wad, workers=2)
        try:
            for ivs in ([iv.set_value("L0.ao", 2, units=[0, 3], pos=[1, 2])], [iv.set_value("L0.ao", -3, units=[1])],
                        [iv.resample("L0.ao", DONOR)], [iv.scale("L0.ao", 2)], [iv.zero("L0.ao", units=[2], pos=[1])]):
                rep = _run(eng, net, ivs)
                assert rep["exact"] and rep["patches"][0] == rep["patches"][1]
        finally:
            eng.close()


def _run(eng, net, ivs, tokens=TOKS):
    eng.use(ivs)
    logits = eng.forward(tokens)
    rep = eng.check_tokens(tokens)
    assert np.array_equal(logits, forward(net, tokens, interventions=ivs)[-1])
    return rep


def test_every_mode_exact_in_doom_and_a_wrong_patch_fails_by_the_predicted_amount(tiles):
    net, brain = tiles
    eng = DoomTransformer(net, brain.wad, workers=2)
    try:
        cases = [
            [iv.zero("x0", units=[1], pos=[1])],                                   # doors, one position
            [iv.zero("L0.m", units=[0, 3])],                                       # doors, every position
            [iv.zero("L0.q", units=[0], pos=[2])],                                 # bus patch (no doors at q)
            [iv.set_value("L0.x1", 1, units=[0, 2], pos=[1, 2])],
            [iv.set_value("L0.ao", 2, units=[3], pos=[2])],
            [iv.scale("L0.x1", 2, 1, pos=[1, 2])],
            [iv.resample("L0.k", DONOR, pos=[0, 1])],
            [iv.resample("x0", DONOR, pos=[1])],
            [iv.mean("L0.ao", [TOKS, DONOR, [0, 1, 1]])],
            [iv.set_value("x0", 1, units=[0], pos=[1]), iv.zero("x0", units=[0], pos=[2]), iv.scale("L0.x2", 1, 2)],
        ]
        for ivs in cases:
            rep = _run(eng, net, ivs)
            assert rep["exact"] and rep["patches"][0] == rep["patches"][1], ivs
        # doors close per position: the same unit at position 2 only leaves position 1's tensors untouched
        _run(eng, net, [iv.zero("x0", units=[1], pos=[2])])
        assert eng.patches == []

        # tamper: DOOM runs a patch value one level off from the one the reference is given
        right = wrong = None
        for site in ("L0.x1", "L0.x2"):
            for pos in (1, 2):
                for unit in range(net.d):
                    for a, b in ((1, 2), (0, 1), (-1, 0)):
                        r, w = iv.set_value(site, a, units=[unit], pos=[pos]), iv.set_value(site, b, units=[unit], pos=[pos])
                        if right is None and not np.array_equal(forward(net, TOKS, interventions=[r])[-1],
                                                                forward(net, TOKS, interventions=[w])[-1]):
                            right, wrong = r, w
        assert right is not None
        eng.use([wrong])
        logits = eng.forward(TOKS)
        rep = eng.check_tokens(TOKS, interventions=[right])
        assert not rep["exact"] and rep["patches"][0] < rep["patches"][1]
        predicted = forward(net, TOKS, interventions=[wrong])[-1] - forward(net, TOKS, interventions=[right])[-1]
        assert predicted.any() and np.array_equal(logits - forward(net, TOKS, interventions=[right])[-1], predicted)
        ref_right, ref_wrong = trace(net, TOKS, interventions=[right]), trace(net, TOKS, interventions=[wrong])
        causal = np.tril(np.ones((3, 3), dtype=np.int64))
        for key, (equal, total) in rep["tensors"].items():                    # a tensor differs exactly where predicted
            masked = (lambda a: a * causal) if key.endswith("score") else (lambda a: a)
            assert equal == int((eng.trace()[key] == masked(ref_right[key])).sum()), key
            assert (equal == total) == np.array_equal(masked(ref_right[key]), masked(ref_wrong[key])), key
        assert eng.check_tokens(TOKS)["exact"]                                 # exact against its own reference
    finally:
        eng.close()


def test_prefix_reuse_and_tile_brain(tiles):
    net, brain = tiles
    eng = DoomTransformer(net, brain.wad, workers=2)
    try:
        ivs = [iv.set_value("L0.x2", 1, units=[0], pos=[2])]
        eng.use(ivs)
        eng.forward(TOKS, reuse=True)
        runs = len(eng.pool.log)
        eng.use([iv.set_value("L0.x2", -1, units=[0], pos=[2])])              # same plan at positions 0 and 1
        got = eng.forward(TOKS, reuse=True)
        assert 0 < len(eng.pool.log) - runs < runs and eng.check_tokens(TOKS)["exact"]
        assert np.array_equal(got, forward(net, TOKS, interventions=[iv.set_value("L0.x2", -1, units=[0], pos=[2])])[-1])
    finally:
        eng.close()
    brain.intervene([iv.resample("L0.v", DONOR, pos=[1])])
    assert np.array_equal(brain(TOKS[:2]), forward(net, TOKS[:2], interventions=[iv.resample("L0.v", DONOR, pos=[1])]))
    assert brain.info["checked"] > 0
    brain.intervene()
    assert np.array_equal(brain(TOKS[:2]), forward(net, TOKS[:2]))


def test_patch_recovery_reference_and_doom(tiles):
    net, brain = tiles
    ld = lambda toks: int(forward(net, toks)[-1][1] - forward(net, toks)[-1][2])
    seqs = [[0, a, b] for a in range(1, 5) for b in range(1, 5)]
    clean, corrupt = max(((c, k) for c in seqs for k in seqs), key=lambda ck: ld(ck[0]) - ld(ck[1]))
    assert ld(clean) != ld(corrupt)
    table = df.patch_recovery(net, clean, corrupt, patching.grid(net, positions=[1, 2]), answer=(1, 2), doom=2,
                              tiles=brain.wad)
    assert len(table.rows) == 2 * 8 and len(table.doom_rows) == 2 and table.exact
    full = df.patch_recovery(net, clean, corrupt, [("x0", [0, 1, 2])], answer=(1, 2))           # every position of x0 = the clean run
    assert full.rows[0]["recovery"] == 1.0
    noise = df.patch_recovery(net, clean, corrupt, [("L0.x2", [0, 1, 2])], answer=(1, 2), direction="noising")
    assert noise.rows[0]["recovery"] == 1.0
    assert "recovery" in table.text() and "EXACT" in table.text()


def test_fold_puts_constant_interventions_into_a_one_map_brain(tmp_path):
    from doomformer.onemap import build_onemap, compare_trace, logits_from_readout, token_ports
    from doomformer.runtime import TileRunner
    net = tiny_model()
    ivs = [iv.zero("x0", units=[1]), iv.set_value("L0.x1", 1, units=[0, 2]), iv.set_value("L0.m", 2, units=[3]),
           iv.set_value("L0.x2", -1, units=[1]), iv.set_value("L0.ao", 1, units=[0]),
           iv.mean("L0.m", [TOKS, DONOR], units=[0], by="all")]
    folded = iv.fold(net, ivs)
    assert not np.array_equal(folded.p["L0.Wup"], net.p["L0.Wup"]) and not (folded.p["Wun"][:, 1] != 0).any()
    for tokens in ([0], [0, 1], TOKS, DONOR):
        want, got = trace(net, tokens, interventions=ivs), trace(folded, tokens)
        assert np.array_equal(want["logits"], got["logits"])
        for key in want:
            if key.split(".")[-1].startswith("z") or key.endswith("score") or key == "z0":
                assert np.array_equal(want[key], got[key]), key
    for bad, why in (([iv.scale("L0.x1", 2)], "scale"), ([iv.zero("L0.q")], "products"),
                     ([iv.zero("L0.x1", pos=[1])], "every position"), ([iv.resample("L0.x1", DONOR)], "same at every")):
        with pytest.raises(ValueError, match=why):
            iv.fold(net, bad)
    wad = tmp_path / "folded.wad"
    build_onemap(folded, wad)
    runner = TileRunner(wad)                       # one map, no host between the layers: the intervention is in the weights
    try:
        for tokens in (TOKS, DONOR):
            ref = {}
            expected = forward(net, tokens, ref, interventions=ivs)
            got = runner.run(token_ports(folded, tokens))[0]
            assert compare_trace(got, trace(folded, tokens), len(tokens)) > 0
            assert np.array_equal(logits_from_readout(folded, got, len(tokens)), expected)
    finally:
        runner.close()
