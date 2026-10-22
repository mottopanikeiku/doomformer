"""RMSNorm (norm="rms") and the gated MLP (mlp="reglu"): the reference identities, one map and tiles exact with
tolerance 0 in every exposed value (s, r, P/N, zg, zu, ...), a tamper control, and the shipped models' lowering
unchanged."""
import dataclasses
import hashlib
import itertools
import json
import math
import re
import shutil
import struct

import numpy as np
import pytest

import doomformer as df
from doomformer import shipped
from doomformer.intref import (PORT_MAX, QModel, adc_relu, adc_signed, check_limits, forward, layer_arrays,
                               param_names, rmsnorm)
from doomformer.model_tiles import tile_specs
from doomformer.onemap import (build_onemap, compare_trace, logits_from_readout, tiny_model, token_ports,
                               transformer_layers)
from doomformer.orchestrator import DoomTransformer
from doomformer.runtime import TileRunner

SEQS = ([0, 1, 4], [0, 4, 2], [0, 3, 1])
LIGHT_LOWER = 111                       # Light_LowerByValue(tag, amount)


def arch_model(seed: int = 1, norm: str = "rms", mlp: str = "reglu", attn: str = "relu", Rn: int = 2) -> QModel:
    """T=3, d=4, H=2, dm=3, K=KM=2, Rn=2 (LN=3). Seed 1 with SEQS covers, at both norm sites: s = 0 (BOS at position 0
    has z0 = 0, and bv = bo = 0 keep x1 = 0 there), r saturated at K (token 4: every |c| = K), n of both signs; and
    m = g*u of both signs, zero, and saturated g and u."""
    rng = np.random.default_rng(seed)
    V, T, d, H, dm, K = 5, 3, 4, 2, 3, 2
    w = lambda shape: rng.integers(-2, 3, shape, dtype=np.int64)
    p = dict(tok_emb=w((V, d)), pos_emb=w((T, d)), thr_x0=np.array([1, 2]), Wun=w((V, d)), bun=w(V))
    p["tok_emb"][0] = -p["pos_emb"][0]
    p["tok_emb"][4] = [5, -5, 5, -5]
    for k in "qkv":
        p["L0.W" + k], p["L0.b" + k] = w((d, d)), w(d)
        p["L0.thr_" + k] = np.array([1, 3])
    p["L0.thr_a"] = np.array([1, 2])
    if attn == "relu":
        p["L0.thr_o"] = np.tile([1, 3], (T, 1))
    p.update({"L0.Wo": w((d, d)), "L0.r1": 2 * w(d), "L0.bo": w(d), "L0.thr_x1": np.array([1, 2]),
              "L0.Wdn": w((d, dm)), "L0.r2": w(d), "L0.bdn": w(d), "L0.thr_x2": np.array([1, 2])})
    p["L0.bv"][:] = 0
    p["L0.bo"][:] = 0
    if mlp == "reglu":
        p.update({"L0.Wg": w((dm, d)), "L0.bg": w(dm), "L0.thr_g": np.array([1, 2]),
                  "L0.Wu": w((dm, d)), "L0.bu": w(dm), "L0.thr_u": np.array([1, 3])})
    else:
        p.update({"L0.Wup": w((dm, d)), "L0.bup": w(dm), "L0.thr_m": np.array([1, 2])})
    return QModel("$ABCD", T, d, H, dm, K, K, K, 1, p, attn=attn, R=2 if attn == "softmax" else 1, norm=norm,
                  Rn=Rn if norm == "rms" else 1, mlp=mlp)


def traces(m: QModel, seqs=SEQS) -> list[dict]:
    out = []
    for s in seqs:
        tr = {}
        forward(m, list(s), tr)
        out.append(tr)
    return out


def test_norm_definition_exhaustive():
    """Over every staircase vector c in [-K, K]^d: r = floor(sqrt(s/d)), n = trunc(Rn*c/(1+r)) and |n| <= LN, i.e. the
    LN comparison levels never saturate; P/N as in the contract."""
    for d, K in ((1, 3), (2, 3), (3, 2), (4, 3), (6, 2)):
        c = np.array(list(itertools.product(range(-K, K + 1), repeat=d)), dtype=np.int64)
        s = (c * c).sum(-1)
        for Rn in range(1, 6):
            m = QModel("$", 1, d, 1, 1, K, K, K, 1, {"thr_x0": np.arange(1, K + 1)}, norm="rms", Rn=Rn)
            tr = {}
            n = rmsnorm(m, 0, "att", c, tr)
            r = np.array([math.isqrt(int(v) // d) for v in s])
            LN = m.ln(0, "att")
            assert LN == min(Rn * K, math.isqrt(Rn * Rn * d - 1))
            np.testing.assert_array_equal(tr["s_att"], s)
            np.testing.assert_array_equal(tr["r_att"], r)
            np.testing.assert_array_equal(n, np.sign(c) * (Rn * np.abs(c) // (1 + r[:, None])))
            assert np.abs(n).max() <= LN
            lv = np.arange(1, LN + 1)
            np.testing.assert_array_equal(tr["P_att"], Rn * c[..., None] - lv * (1 + r)[:, None, None] + 1)
            np.testing.assert_array_equal(tr["N_att"], -Rn * c[..., None] - lv * (1 + r)[:, None, None] + 1)


def test_identities_random_models():
    """n == trunc(Rn*c/(1+r)) at both sites, m == g*u, and every pre-activation is the plain matrix product of what its
    consumers read, over random models, thresholds and inputs."""
    rng = np.random.default_rng(0)
    for trial in range(40):
        m = arch_model(seed=100 + trial, attn=("relu", "softmax")[trial % 2], Rn=int(rng.integers(1, 5)))
        for key in ("thr_x0", "L0.thr_x1", "L0.thr_g", "L0.thr_u"):
            m.p[key] = np.sort(rng.integers(1, 5, 2))
        tokens = list(rng.integers(0, len(m.vocab), int(rng.integers(1, m.T + 1))))
        (tr,) = traces(m, [tokens])
        q = m.layer(0)
        for site, x in (("att", tr["L0.x"]), ("mlp", tr["L0.x1"])):
            r = tr[f"L0.r_{site}"]
            np.testing.assert_array_equal(r, [math.isqrt(int(v) // m.d) for v in (x * x).sum(-1)])
            np.testing.assert_array_equal(tr[f"L0.n_{site}"], np.trunc(m.Rn * x / (1 + r[:, None])).astype(np.int64))
            assert np.abs(tr[f"L0.n_{site}"]).max() <= m.ln(0, site)
        np.testing.assert_array_equal(tr["L0.zq"], tr["L0.n_att"] @ q["Wq"].T + q["bq"])
        np.testing.assert_array_equal(tr["L0.zg"], tr["L0.n_mlp"] @ q["Wg"].T + q["bg"])
        np.testing.assert_array_equal(tr["L0.zu"], tr["L0.n_mlp"] @ q["Wu"].T + q["bu"])
        g, u = adc_relu(tr["L0.zg"], q["thr_g"]), adc_signed(tr["L0.zu"], q["thr_u"])
        np.testing.assert_array_equal(tr["L0.m"], g * u)
        np.testing.assert_array_equal(tr["L0.z2"], tr["L0.m"] @ q["Wdn"].T + q["r2"] * tr["L0.x1"] + q["bdn"])


def test_config_and_arrays(tmp_path):
    assert layer_arrays("relu", "reglu")[-10:] == ("Wg", "bg", "thr_g", "Wu", "bu", "thr_u", "Wdn", "r2", "bdn",
                                                   "thr_x2")
    assert "L0.Wup" not in param_names(1, "softmax", "reglu") and "L0.thr_o" not in param_names(1, "softmax", "reglu")
    m = arch_model(attn="softmax")
    m.save(tmp_path / "arch")
    cfg = json.loads((tmp_path / "arch" / "config.json").read_text())
    assert (cfg["norm"], cfg["Rn"], cfg["mlp"], cfg["attn"]) == ("rms", 2, "reglu", "softmax")
    back = QModel.load(tmp_path / "arch")
    assert (back.norm, back.Rn, back.mlp) == ("rms", 2, "reglu") and set(back.p) == set(m.p)
    for tokens in SEQS:
        np.testing.assert_array_equal(forward(back, list(tokens)), forward(m, list(tokens)))
    base = dict(vocab=m.vocab, T=m.T, d=m.d, H=m.H, dm=m.dm, K=m.K, KA=m.KA, KM=m.KM, L=1, p=m.p)
    for bad in (dict(norm="layer"), dict(mlp="swiglu"), dict(norm="rms", Rn=0), dict(Rn=2)):
        with pytest.raises(ValueError):
            QModel(**base, **bad)
    old = QModel.load("doomnames")
    assert (old.norm, old.Rn, old.mlp) == ("none", 1, "relu")


def test_fixture_covers_the_cases():
    m = arch_model()
    assert check_limits(m, list(SEQS)) == []
    trs = traces(m)
    cat = lambda key: np.concatenate([tr[key].reshape(-1) for tr in trs])      # noqa: E731
    for site in ("att", "mlp"):
        s, r, n = cat(f"L0.s_{site}"), cat(f"L0.r_{site}"), cat(f"L0.n_{site}")
        assert (s == 0).any() and (r == 0).any() and (r == m.K).any() and (s == m.d * m.K ** 2).any()
        assert (n > 0).any() and (n < 0).any() and (n == 0).any()
        P = cat(f"L0.P_{site}")
        assert (P >= 1).any() and (P < 1).any()
    g, u, mm = cat("L0.g"), cat("L0.u"), cat("L0.m")
    assert (mm > 0).any() and (mm < 0).any() and (mm == 0).any() and (g == m.KM).any() and (np.abs(u) == m.KM).any()
    for key in ("L0.zg", "L0.zu"):
        assert (cat(key) > 0).any() and (cat(key) < 0).any()
    big = arch_model()
    big.p["thr_x0"] = np.arange(1, 61)                         # d*Kc^2 = 14400: thr_r leaves the gate range
    assert any("thr_r" in e and str(PORT_MAX) in e for e in check_limits(big, [[0]]))
    big = arch_model()
    big.p["L0.Wu"] = big.p["L0.Wu"] * 200
    assert any(e.startswith("L0.Wu:") for e in check_limits(big, [[0]]))


def mismatches(got: dict, trace: dict, n: int) -> list[tuple[str, int, int]]:
    bad = []
    for name, actual in got.items():
        try:
            compare_trace({name: actual}, trace, n)
        except AssertionError as e:
            bad.append((name, actual, int(str(e).rsplit("intref ", 1)[1])))
    return bad


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


def _active(name: str, n: int) -> bool:
    key, *idx = name.lstrip("-").split(":")
    idx = list(map(int, idx))
    return idx[1] < n and idx[2] < n if key.endswith("score") else idx[0] < n


def test_one_map_exact_and_tamper(tmp_path):
    """RMSNorm and the gated MLP inside ONE map: every exposed value equals intref. Tamper: the r lanes of
    position 0 lower P_att:0:1:2 (l = 3) by 3 each; making them lower by 4 must shift exactly that value by -r."""
    m = arch_model()
    wad = tmp_path / "arch.wad"
    meta = build_onemap(m, wad)
    stages = [s["name"] for s in meta.layout["stage_stats"]]
    assert stages[1:4] == ["L0.sumsq_att", "L0.norm_att", "L0.qkv"] and stages[7:11] == [
        "L0.sumsq_mlp", "L0.norm_mlp", "L0.mlp_up", "L0.mlp_dn"]
    target = "L0.P_att:0:1:2"
    tampered = tmp_path / "tampered.wad"
    shutil.copy(wad, tampered)
    shutil.copy(wad.with_suffix(".json"), tampered.with_suffix(".json"))
    tag = int(re.fullmatch(r"GetSectorLightLevel\((\d+)\)", meta.layout["read_expressions"][target]).group(1))
    assert retarget_lowerings(tampered, tag, 3, 4) == m.K           # one line per r lane (Rn = 2: c lanes move by 2)

    runner = TileRunner(wad)
    try:
        for tokens in SEQS[:2]:
            trace = {}
            expected = forward(m, list(tokens), trace)
            got = runner.run(token_ports(m, list(tokens)))[0]
            kinds = {name.lstrip("-").split(":")[0] for name in got}
            assert {f"L0.{k}_{site}" for k in "srPN" for site in ("att", "mlp")} | {"L0.zg", "L0.zu", "L0.z2"} <= kinds
            assert "L0.zm" not in kinds
            assert compare_trace(got, trace, len(tokens)) == sum(1 for name in got if _active(name, len(tokens)))
            np.testing.assert_array_equal(logits_from_readout(m, got, len(tokens)), expected)
    finally:
        runner.close()

    runner = TileRunner(tampered)
    shifts = []
    try:
        for tokens in ([0, 1, 4], [4, 2]):
            trace = {}
            forward(m, tokens, trace)
            got = runner.run(token_ports(m, tokens))[0]
            r, ref = int(trace["L0.r_att"][0]), int(trace["L0.P_att"][0, 1, 2])
            shifts.append(r)
            assert mismatches(got, trace, len(tokens)) == ([(target, ref - r, ref)] if r else [])
    finally:
        runner.close()
    assert shifts == [0, m.K]                                         # a silent control (s = 0) and a caught tamper


@pytest.mark.parametrize("norm,mlp,attn", [("rms", "reglu", "softmax"), ("rms", "relu", "relu"),
                                           ("none", "reglu", "relu")])
def test_tiles_exact(tmp_path, norm, mlp, attn):
    m = arch_model(norm=norm, mlp=mlp, attn=attn)
    report = df.plan(m, mode="tiles")
    assert report["tiles"]["tiles"] == 2 + (5 if attn == "softmax" else 4) + 2 * m.T + (4 if norm == "rms" else 0)
    with df.compile(m, tmp_path / "tiles", mode="tiles", workers=2) as brain:
        for tokens in SEQS:
            np.testing.assert_array_equal(brain(list(tokens)), forward(m, list(tokens)))
            trace = brain.engine.trace()
            assert brain.engine.negation_errors() == 0
            if norm == "rms":
                for site in ("att", "mlp"):
                    assert trace[f"L0.P_{site}"].shape == (len(tokens), m.d, m.ln(0, site))
                    assert trace[f"L0.s_{site}"].shape == trace[f"L0.r_{site}"].shape == (len(tokens),)
            hidden = {"L0.zg", "L0.zu"} if mlp == "reglu" else {"L0.zm"}
            assert hidden <= set(trace) and not ({"L0.zg", "L0.zm"} - hidden) & set(trace)
        bad = ("L0.r_mlp", 2) if norm == "rms" else ("L0.zg_1", 2)
        brain.engine.pos[bad[1]][bad[0]] += 1                     # a corrupted readback must be caught
        with pytest.raises(df.MismatchError) as err:
            brain.logits(brain._tokens)
        names = [name for name, _, _ in err.value.mismatches]
        assert names == (["L0.r_mlp:2"] if norm == "rms" else ["L0.zg:2:1", "-L0.zg_1:2"])
        tile_dir = brain.wad
    # ablation: a unit of x0 (read by sumsq/norm or qkv) and a hidden unit (its gated corridors) closed by doors
    ablate = {"x0": [1], "L0.m": [0]}
    plain, ablated = {}, {}
    forward(m, list(SEQS[1]), plain)
    forward(m, list(SEQS[1]), ablated, ablate=ablate)
    assert not np.array_equal(plain["L0.z2"], ablated["L0.z2"])
    engine = DoomTransformer(m, tile_dir, workers=2, ablate=ablate)
    try:
        engine.forward(list(SEQS[1]))
        assert engine.check_tokens(list(SEQS[1]))["exact"]
    finally:
        engine.close()


def _digest(specs) -> str:
    h = hashlib.sha256()
    for s in specs:
        h.update(json.dumps(dataclasses.asdict(s), sort_keys=True, default=int).encode())
    return h.hexdigest()[:16]


# sha256 of every TileSpec (ports, outputs, blocks) as lowered by doomformer v0.3.0 (6edb8b0), before norm/reglu
LOWERED_V030 = {"tiny_model.map": "c5e07bfb240fa29e", "tiny_model.tiles": "1f194ac9285b0340",
                "doomnames.tiles": "ef93ea5178db6dff", "doomnames_softmax.tiles": "dcd29d6f09f4508d",
                "doomnames_onemap.map": "98a6a8f4f2f54fba", "doomnames_onemap_softmax.map": "954870cbd992f708",
                "doombrain.tiles": "c140a2d59d018219"}


def test_shipped_lowering_unchanged():
    """The relu and softmax models lower to exactly the tile specs (hence the same WADs) as before norm/reglu."""
    got = {}
    for name in ("tiny_model", "doomnames", "doomnames_softmax", "doomnames_onemap", "doomnames_onemap_softmax",
                 "doombrain"):
        m = tiny_model() if name == "tiny_model" else QModel.load(name)
        assert (m.norm, m.mlp) == ("none", "relu")
        if f"{name}.map" in LOWERED_V030:
            got[f"{name}.map"] = _digest(transformer_layers(m))
        if f"{name}.tiles" in LOWERED_V030:
            got[f"{name}.tiles"] = _digest(tile_specs(m))
    assert got == LOWERED_V030


def test_torch_mirrors_match_intref():
    """train_qat (tile shape) and train_brain (one-map shape): the hard torch forward and the int64 torch forward equal
    intref.forward for random-init models with norm="rms" and mlp="reglu"."""
    torch = pytest.importorskip("torch")
    dtype = torch.get_default_dtype()
    try:
        from training import train_brain, train_qat                 # train_qat sets float64 as the default dtype
        _check_mirrors(torch, train_brain, train_qat)
    finally:
        torch.set_default_dtype(dtype)


def _check_mirrors(torch, train_brain, train_qat):
    tok = torch.tensor([[0, 3, 7, 1, 9, 0], [0, 12, 5, 0, 0, 0]])
    valid = torch.tensor([[True] * 6, [True] * 3 + [False] * 3])
    for norm, mlp, attn in (("rms", "reglu", "softmax"), ("rms", "relu", "relu"), ("none", "reglu", "relu")):
        kw = dict(attn=attn, R=2 if attn == "softmax" else 1, norm=norm, Rn=3 if norm == "rms" else 1, mlp=mlp)
        for mod, net in ((train_qat, train_qat.Net(30, 6, 16, 2, 8, 7, 7, 7, seed=1, **kw)),
                         (train_brain, train_brain.Net(30, 6, 16, 2, 8, 7, 7, 7, 1, seed=1, **kw))):
            net.quant, net.hard, net.calibrating, net.valid = False, True, True, valid
            with torch.no_grad():
                net(tok, valid)
            net.calibrating, net.quant = False, True
            with torch.no_grad():
                hard = net(tok, valid)[0]
            m = net.qmodel() if mod is train_qat else net.qmodel(train_qat.VOCAB)
            args = (m.H,) if mod is train_qat else (m.H, m.L)
            ints = mod.int_forward(m.p, tok, *args, m.attn, m.R, m.norm, m.Rn, m.mlp)
            for i, n in enumerate(valid.sum(1).tolist()):
                ref = forward(m, tok[i, :n].tolist())
                np.testing.assert_array_equal(ints[i, :n].numpy(), ref)
                np.testing.assert_array_equal(hard[i, :n].detach().numpy(), ref)


def test_shipped_modern():
    """doomnames_modern: rms + reglu + softmax, limits clean on its training names."""
    m = QModel.load("doomnames_modern")
    assert (m.norm, m.mlp, m.attn) == ("rms", "reglu", "softmax")
    names = sorted(shipped.corpus("doomnames_modern"))
    assert check_limits(m, [[0] + [m.vocab.index(c) for c in n] for n in names]) == []
