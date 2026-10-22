"""Softmax attention computed by the dolls: quantised exponential (thr_a staircase), runtime division (P/N comparison
lanes), checked with tolerance 0 in one map and in tiles, plus a tamper control and intref unit checks."""
import json
import re
import shutil
import struct

import numpy as np
import pytest

import doomformer as df
from doomformer.intref import QModel, adc_relu, adc_signed, check_limits, forward, param_names
from doomformer.onemap import build_onemap, compare_trace, logits_from_readout, token_ports
from doomformer.orchestrator import DoomTransformer
from doomformer.runtime import TileRunner

LIGHT_LOWER = 111                       # Light_LowerByValue(tag, amount)
SEQS = ([0, 1, 2], [3, 4, 0], [0, 2, 4])


def softmax_model(seed: int = 1, R: int = 2) -> QModel:
    """T=3, d=4, H=2 (dh=2), KA=3, K=2, R=2 -> LV=3. Seed 1 with SEQS covers positive, negative and zero numerators,
    nonzero numerators truncated to 0 (both signs), den = 1 (every e = 0) and saturated e = KA."""
    rng = np.random.default_rng(seed)
    V, T, d, H, dm, K, KA = 5, 3, 4, 2, 4, 2, 3
    w = lambda shape: rng.integers(-2, 3, shape, dtype=np.int64)
    p = dict(tok_emb=w((V, d)), pos_emb=w((T, d)), thr_x0=np.array([1, 2]), Wun=w((V, d)), bun=w(V))
    for k in "qkv":
        p["L0.W" + k], p["L0.b" + k] = w((d, d)), w(d)
        p["L0.thr_" + k] = np.array([1, 3])
    p["L0.thr_a"] = np.array([1, 2, 4])
    p.update({"L0.Wo": w((d, d)), "L0.r1": w(d), "L0.bo": w(d), "L0.thr_x1": np.array([1, 2]),
              "L0.Wup": w((dm, d)), "L0.bup": w(dm), "L0.thr_m": np.array([1, 2]), "L0.Wdn": w((d, dm)),
              "L0.r2": w(d), "L0.bdn": w(d), "L0.thr_x2": np.array([1, 2])})
    return QModel("$ABCD", T, d, H, dm, K, KA, K, 1, p, attn="softmax", R=R)


def mismatches(got: dict, trace: dict, n: int) -> list[tuple[str, int, int]]:
    """Every readback that differs from intref (compare_trace stops at the first)."""
    bad = []
    for name, actual in got.items():
        try:
            compare_trace({name: actual}, trace, n)
        except AssertionError as e:
            ref = int(str(e).rsplit("intref ", 1)[1])
            bad.append((name, actual, ref))
    return bad


def relu_forward_before(m: QModel, tokens: list[int]) -> np.ndarray:
    """intref.forward as it was before softmax attention existed (relu attention only)."""
    p, n = m.p, len(tokens)
    x = adc_signed(p["tok_emb"][tokens] + p["pos_emb"][np.arange(n)], p["thr_x0"])
    causal = np.tril(np.ones((n, n), dtype=np.int64))
    for l in range(m.L):
        q_ = {k: p[f"L{l}.{k}"] for k in ("Wq", "bq", "Wk", "bk", "Wv", "bv", "thr_q", "thr_k", "thr_v", "thr_a",
                                          "thr_o", "Wo", "r1", "bo", "thr_x1", "Wup", "bup", "thr_m", "Wdn", "r2",
                                          "bdn", "thr_x2")}
        zq, zk, zv = (x @ q_[W].T + q_[b] for W, b in (("Wq", "bq"), ("Wk", "bk"), ("Wv", "bv")))
        q, k, v = adc_signed(zq, q_["thr_q"]), adc_signed(zk, q_["thr_k"]), adc_signed(zv, q_["thr_v"])
        qh, kh, vh = (u.reshape(n, m.H, m.dh) for u in (q, k, v))
        a = adc_relu(np.einsum("thd,shd->hts", qh, kh), q_["thr_a"]) * causal
        zo = np.einsum("hts,she->the", a, vh).reshape(n, m.d)
        ao = np.stack([adc_signed(zo[t], q_["thr_o"][t]) for t in range(n)])
        x1 = adc_signed(ao @ q_["Wo"].T + q_["r1"] * x + q_["bo"], q_["thr_x1"])
        mh = adc_relu(x1 @ q_["Wup"].T + q_["bup"], q_["thr_m"])
        x = adc_signed(mh @ q_["Wdn"].T + q_["r2"] * x1 + q_["bdn"], q_["thr_x2"])
    return x @ p["Wun"].T + p["bun"]


def test_division_is_truncation():
    """ao == trunc(R*num/den) elementwise, den = 1 + sum e, P/N as in the contract, |ao| <= LV."""
    rng = np.random.default_rng(0)
    for trial in range(60):
        R = int(rng.integers(1, 4))
        m = softmax_model(seed=100 + trial, R=R)
        m.p["L0.thr_a"] = np.sort(rng.integers(1, 6, m.KA))
        tokens = list(rng.integers(0, len(m.vocab), int(rng.integers(1, m.T + 1))))
        tr = {}
        forward(m, tokens, tr)
        num, den, ao = tr["L0.zo"], tr["L0.den"], tr["L0.ao"]
        den_j = np.repeat(den, m.dh, axis=1)
        np.testing.assert_array_equal(ao, np.trunc(R * num / den_j).astype(np.int64))
        np.testing.assert_array_equal(den, 1 + tr["L0.e"].sum(-1).T)
        lv = np.arange(1, m.lv(0) + 1)
        np.testing.assert_array_equal(tr["L0.P"], R * num[..., None] - lv * den_j[..., None] + 1)
        np.testing.assert_array_equal(tr["L0.N"], -R * num[..., None] - lv * den_j[..., None] + 1)
        assert np.abs(ao).max() <= m.lv(0) == R * len(m.p["L0.thr_v"]) - 1
        assert "L0.a" not in tr and tr["L0.P"].shape == (len(tokens), m.d, m.lv(0))


@pytest.mark.parametrize("name", ["doomnames", "doombrain", "doomnames_onemap"])
def test_relu_models_unchanged(name):
    m = QModel.load(name)
    assert (m.attn, m.R) == ("relu", 1) and "L0.thr_o" in param_names(m.L, m.attn)
    rng = np.random.default_rng(3)
    for _ in range(8):
        tokens = [0] + list(rng.integers(1, len(m.vocab), int(rng.integers(0, m.T))))
        tr = {}
        np.testing.assert_array_equal(forward(m, tokens, tr), relu_forward_before(m, tokens))
        assert "L0.den" not in tr and "L0.a" in tr


def test_config_roundtrip(tmp_path):
    m = softmax_model()
    assert "L0.thr_o" not in param_names(1, "softmax") and "L0.thr_o" not in m.layer(0)
    m.save(tmp_path / "sm")
    cfg = json.loads((tmp_path / "sm" / "config.json").read_text())
    assert (cfg["attn"], cfg["R"]) == ("softmax", 2)
    back = QModel.load(tmp_path / "sm")
    assert (back.attn, back.R) == ("softmax", 2)
    for tokens in SEQS:
        np.testing.assert_array_equal(forward(back, tokens), forward(m, tokens))
    with pytest.raises(ValueError):
        QModel(m.vocab, m.T, m.d, m.H, m.dm, m.K, m.KA, m.KM, 1, m.p, attn="softmax", R=0)
    with pytest.raises(ValueError):
        QModel(m.vocab, m.T, m.d, m.H, m.dm, m.K, m.KA, m.KM, 1, m.p, attn="sparsemax")


def test_fixture_covers_the_cases():
    m = softmax_model()
    assert check_limits(m, list(SEQS)) == []
    traces = [{} for _ in SEQS]
    for s, tr in zip(SEQS, traces):
        forward(m, s, tr)
    num, ao, den, e = (np.concatenate([tr[k].reshape(-1) for tr in traces])
                       for k in ("L0.zo", "L0.ao", "L0.den", "L0.e"))
    assert (ao > 0).any() and (ao < 0).any() and (num == 0).any()
    assert ((num > 0) & (ao == 0)).any() and ((num < 0) & (ao == 0)).any()      # truncation towards zero
    assert (den == 1).any() and (den > 1).any() and (e == m.KA).any()


def test_softmax_one_map_exact_and_tamper(tmp_path):
    m = softmax_model()
    wad = tmp_path / "softmax.wad"
    meta = build_onemap(m, wad)
    assert [s["name"] for s in meta.layout["stage_stats"]][3:6] == ["L0.den", "L0.wsum", "L0.out"]
    target = "L0.P:2:1:2"                         # l = 3: its -l lines are the only 3-unit lowerings of its light
    tampered = tmp_path / "tampered.wad"
    shutil.copy(wad, tampered)
    shutil.copy(wad.with_suffix(".json"), tampered.with_suffix(".json"))
    tag = int(re.fullmatch(r"GetSectorLightLevel\((\d+)\)", meta.layout["read_expressions"][target]).group(1))
    assert retarget_lowerings(tampered, tag, 3, 4) == m.T * m.KA       # one per e lane and key position

    runner = TileRunner(wad)
    try:
        for tokens in SEQS:
            trace = {}
            expected = forward(m, tokens, trace)
            got = runner.run(token_ports(m, tokens))[0]
            checked = compare_trace(got, trace, len(tokens))
            kinds = {name.lstrip("-").split(":")[0] for name in got}
            assert {"L0.den", "L0.zo", "L0.P", "L0.N", "L0.score", "L0.z1", "logits"} <= kinds
            assert checked == sum(1 for name in got if _active(name, len(tokens)))
            np.testing.assert_array_equal(logits_from_readout(m, got, len(tokens)), expected)
    finally:
        runner.close()

    runner = TileRunner(tampered)
    crossings = []
    try:
        for tokens in SEQS[:2]:
            trace = {}
            forward(m, tokens, trace)
            got = runner.run(token_ports(m, tokens))[0]
            ref = int(trace["L0.P"][2, 1, 2])
            dolls = int(trace["L0.den"][2, 0]) - 1             # e dolls of head 0 at t=2: each lowers 1 too far
            crossings.append(dolls)
            assert mismatches(got, trace, len(tokens)) == ([(target, ref - dolls, ref)] if dolls else [])
    finally:
        runner.close()
    assert crossings == [0, 4]                                  # a silent control (den = 1) and a caught tamper


def _active(name: str, n: int) -> bool:
    key, *idx = name.lstrip("-").split(":")
    idx = list(map(int, idx))
    return idx[1] < n and idx[2] < n if key.endswith("score") else idx[0] < n


def retarget_lowerings(wad, tag: int, old: int, new: int) -> int:
    """Light_LowerByValue(tag, old) -> (tag, new) on every such linedef of the WAD's TEXTMAP; returns how many."""
    data = wad.read_bytes()
    n, directory = struct.unpack("<ii", data[4:12])
    lumps, count = [], 0
    for i in range(n):
        offset, size, name = struct.unpack("<ii8s", data[directory + 16 * i:directory + 16 * (i + 1)])
        blob = data[offset:offset + size]
        if name.rstrip(b"\0") == b"TEXTMAP":
            text, count = re.subn(rf"(special = {LIGHT_LOWER}; arg0 = {tag}; arg1 = ){old};", rf"\g<1>{new};",
                                  blob.decode())
            blob = text.encode()
        lumps.append((name, blob))
    body, table = b"", b""
    for name, blob in lumps:
        table += struct.pack("<ii8s", 12 + len(body), len(blob), name)
        body += blob
    wad.write_bytes(struct.pack("<4sii", b"PWAD", len(lumps), 12 + len(body)) + body + table)
    return count


def test_softmax_tiles_exact(tmp_path):
    m = softmax_model()
    report = df.plan(m, mode="tiles")
    assert report["tiles"]["tiles"] == 2 + m.L * (5 + 2 * m.T)
    with df.compile(m, tmp_path / "tiles", mode="tiles", workers=2) as brain:
        for tokens in SEQS:
            np.testing.assert_array_equal(brain(tokens), forward(m, tokens))
            trace = brain.engine.trace()
            assert trace["L0.P"].shape == (len(tokens), m.d, m.lv(0)) and trace["L0.den"].shape == (len(tokens), m.H)
            assert brain.engine.negation_errors() == 0
        brain.engine.pos[2]["L0.den_0"] += 1                   # a corrupted readback must be caught
        with pytest.raises(df.MismatchError) as err:
            brain.logits(brain._tokens)
        assert [name for name, _, _ in err.value.mismatches] == ["L0.den:2:0", "-L0.den_0:2"]
        tile_dir = brain.wad
    # ablation: the out tile closes every P/N lane door of the ablated unit, like intref zeroes ao
    plain, ablated = {}, {}
    forward(m, SEQS[1], plain)
    forward(m, SEQS[1], ablated, ablate={"L0.ao": [2]})
    assert (plain["L0.ao"][:, 2] != 0).any() and not np.array_equal(plain["L0.z1"], ablated["L0.z1"])
    engine = DoomTransformer(m, tile_dir, workers=2, ablate={"L0.ao": [2]})
    try:
        engine.forward(SEQS[1])
        assert engine.check_tokens(SEQS[1])["exact"]
    finally:
        engine.close()
