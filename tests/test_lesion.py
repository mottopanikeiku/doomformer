"""`doomformer.lesion`: the batched zero-ablation evaluator equals `intref.forward(..., ablate=)` bit for bit; the
subset enumeration, the cascades, the KL helper and the shuffled copies behave as documented. No DOOM."""
from __future__ import annotations

import dataclasses
import itertools
import math

import numpy as np
import pytest

from doomformer import intref, lesion, policy
from doomformer.intref import QModel

IDS = policy.ACTION_IDS
FIRE = policy.ACTIONS.index("F")


def random_model(seed: int, L: int = 2) -> QModel:
    """Small random relu/none/relu model: both signs, duplicate thresholds allowed, thr_o per position."""
    rng = np.random.default_rng(seed)
    V, T, d, H, dm, K = 6, 4, 8, 2, 12, 3
    thr = lambda: np.sort(rng.integers(1, 80, K)).astype(np.int64)
    w = lambda *s: rng.integers(-40, 41, s).astype(np.int64)
    p = dict(tok_emb=w(V, d), pos_emb=w(T, d), thr_x0=thr(), Wun=w(V, d), bun=w(V))
    for l in range(L):
        p |= {f"L{l}.{k}": v for k, v in dict(
            Wq=w(d, d), bq=w(d), Wk=w(d, d), bk=w(d), Wv=w(d, d), bv=w(d), thr_q=thr(), thr_k=thr(), thr_v=thr(),
            thr_a=thr(), thr_o=np.sort(rng.integers(1, 40, (T, K)), axis=1).astype(np.int64), Wo=w(d, d),
            r1=rng.integers(0, 60, d).astype(np.int64), bo=w(d), thr_x1=thr(), Wup=w(dm, d), bup=w(dm), thr_m=thr(),
            Wdn=w(d, dm), r2=rng.integers(0, 60, d).astype(np.int64), bdn=w(d), thr_x2=thr()).items()}
    return QModel(vocab="$abcde", T=T, d=d, H=H, dm=dm, K=K, KA=K, KM=K, L=L, p=p)


def random_masks(m: QModel, S: int, rng) -> dict:
    """S lesions; every lesion draws a random subset of the five-site families, so single- and multi-site occur."""
    masks = {}
    for key, n in lesion.sites(m).items():
        on = rng.random(S) < 0.4
        density = rng.choice([0.05, 0.3, 0.7], size=S)
        mk = (rng.random((S, n)) < density[:, None]) & on[:, None]
        if mk.any() or rng.random() < 0.5:
            masks[key] = mk
    return masks


def ablate_dicts(masks: dict, S: int) -> list[dict]:
    return [{k: np.nonzero(mk[s])[0].tolist() for k, mk in masks.items()} for s in range(S)]


@pytest.mark.parametrize("name", ["doomplayer", "doomplayer_xl", "random2", "random1"])
def test_batch_equals_intref(name):
    m = random_model(3, 2) if name == "random2" else random_model(4, 1) if name == "random1" else QModel.load(name)
    rng = np.random.default_rng(7)
    if name.startswith("random"):
        tokens = rng.integers(0, len(m.vocab), (5, m.T))
    else:
        tokens = np.array(policy.all_contexts())
    S = 120
    masks = random_masks(m, S, rng)
    ablates = ablate_dicts(masks, S)
    last = lesion.forward_lesions(m, tokens, masks)
    full = lesion.forward_lesions(m, tokens, masks, all_positions=True)
    assert last.shape == (S, len(tokens), len(m.vocab)) and full.shape == last.shape[:2] + (m.T if name.startswith("random") else 3, last.shape[2])
    multi = sum(len(a) > 1 for a in ablates)
    assert multi >= 10                                              # several lesions span more than one site
    for s in range(S):
        for b in rng.choice(len(tokens), 2, replace=False):
            ref = intref.forward(m, tokens[b].tolist(), ablate=ablates[s])
            assert np.array_equal(ref[-1], last[s, b])
            assert np.array_equal(ref, full[s, b])


def test_two_hundred_lesions_on_doomplayer_every_context():
    m = QModel.load("doomplayer")
    tokens = np.array(policy.all_contexts())
    rng = np.random.default_rng(11)
    masks = random_masks(m, 210, rng)
    ablates = ablate_dicts(masks, 210)
    lg = lesion.forward_lesions(m, tokens, masks)
    for s in range(210):
        ref = np.array([intref.forward(m, t.tolist(), ablate=ablates[s])[-1] for t in tokens[::9]])
        assert np.array_equal(ref, lg[s, ::9])


def test_no_lesion_and_empty_mask_equal_healthy():
    m = QModel.load("doomplayer")
    tokens = np.array(policy.all_contexts())
    healthy = np.array([intref.forward(m, t.tolist())[-1] for t in tokens])
    assert np.array_equal(lesion.healthy_logits(m, tokens), healthy)
    zero = {"x0": np.zeros((3, m.d), dtype=bool)}
    assert all(np.array_equal(x, healthy) for x in lesion.forward_lesions(m, tokens, zero))


def test_rejects_other_architectures():
    m = QModel.load("doomplayer")
    for kw in (dict(attn="softmax"), dict(norm="rms"), dict(mlp="reglu")):
        bad = dataclasses.replace(m, **kw)
        with pytest.raises(ValueError):
            lesion.forward_lesions(bad, [[0, 1, 2]], {})


def test_exactness_bound_holds_for_shipped_models():
    for name in ("doomplayer", "doomplayer_xl", "doombrain"):
        assert lesion.check_exact(QModel.load(name)) < 2 ** 53
    big = QModel.load("doomplayer")
    big.p = dict(big.p, Wun=big.p["Wun"] * 2 ** 40)
    with pytest.raises(AssertionError):
        lesion.check_exact(big)


@pytest.mark.parametrize("n,k", [(6, 1), (7, 2), (9, 3), (10, 5), (8, 8), (13, 6)])
def test_subset_chunks_visit_every_subset_once(n, k):
    got = [tuple(r) for c in lesion.subset_chunks(n, k, chunk=11) for r in c.tolist()]
    assert len(got) == math.comb(n, k) == len(set(got))
    assert set(got) == set(itertools.combinations(range(n), k))


def test_masks_from_global_matches_masks_from_ablate():
    m = QModel.load("doomplayer")
    names = lesion.global_names(m)
    ids = np.array([[0, 5, 17], [20, 21, 95], [16, 40, 90]])
    a = lesion.masks_from_global(m, ids)
    b = lesion.masks_from_ablate(m, [lesion.ablate_of(names, row) for row in ids])
    assert a.keys() == b.keys() and all(np.array_equal(a[k], b[k]) for k in a)


def test_cascade_equals_full_evaluation_and_pacifists_on_x0():
    m = QModel.load("doomplayer")
    tokens = np.array(policy.all_contexts())
    base = lesion.action_tables(m, tokens, {}, IDS, S=1)[0]
    sub = next(lesion.subset_chunks(16, 7, 1 << 14))
    assert len(sub) == math.comb(16, 7)
    masks = {"x0": np.zeros((len(sub), 16), dtype=bool)}
    masks["x0"][np.arange(len(sub))[:, None], sub] = True
    tables = lesion.action_tables(m, tokens, masks, IDS)
    pac, ch = lesion.classify(m, tokens, base, FIRE, IDS, masks)
    assert np.array_equal(pac, lesion.pacifist(tables, FIRE))
    assert np.array_equal(ch, lesion.changes_action(tables, base))
    order = np.random.default_rng(0).permutation(len(tokens))
    pac2, ch2 = lesion.classify(m, tokens, base, FIRE, IDS, masks, (order, order[::-1]))
    assert np.array_equal(pac, pac2) and np.array_equal(ch, ch2)
    assert pac.sum() == 64 and ch.sum() == len(sub)                 # the numbers quoted in docs/DEMOS.md
    assert (7,) == tuple({int(s.sum()) for s in masks["x0"][pac]})
    assert np.all(masks["x0"][pac][:, [0, 11]])                     # units 0 and 11 are in every one of them


def test_no_small_pacifist_on_x0_and_five_is_minimal():
    m = QModel.load("doomplayer")
    tokens = np.array(policy.all_contexts())
    found = {}
    for k in range(1, 6):
        sub = next(lesion.subset_chunks(16, k, 1 << 14))
        mk = np.zeros((len(sub), 16), dtype=bool)
        mk[np.arange(len(sub))[:, None], sub] = True
        pac = lesion.pacifist(lesion.action_tables(m, tokens, {"x0": mk}, IDS), FIRE)
        found[k] = [tuple(s) for s in sub[pac].tolist()]
    assert [len(found[k]) for k in range(1, 6)] == [0, 0, 0, 0, 7]
    assert (0, 3, 11, 12, 14) in found[5]


def test_pooled_pairs_have_no_pacifist():
    m = QModel.load("doomplayer")
    tokens = np.array(policy.all_contexts())
    n = sum(lesion.sites(m).values())
    for k in (1, 2):
        for sub in lesion.subset_chunks(n, k):
            tables = lesion.action_tables(m, tokens, lesion.masks_from_global(m, sub), IDS)
            assert not lesion.pacifist(tables, FIRE).any()


def test_kl_semantics():
    rng = np.random.default_rng(0)
    h = rng.integers(-3000, 3000, (7, 10)).astype(np.int64)
    l = rng.integers(-3000, 3000, (4, 7, 10)).astype(np.int64)
    assert np.all(lesion.mean_kl(h, np.stack([h, h]), 150.0) == 0)
    kl = lesion.kl_per_context(h, l, 150.0)
    assert kl.shape == (4, 7) and (kl > 0).all()
    p = np.exp(h[2] / 150 - (h[2] / 150).max()); p /= p.sum()
    q = np.exp(l[1, 2] / 150 - (l[1, 2] / 150).max()); q /= q.sum()
    assert kl[1, 2] == pytest.approx((p * np.log(p / q)).sum(), rel=1e-12)
    idx = [1, 2, 3]
    ph, ql = np.exp(h[:, idx] / 50), np.exp(l[:, :, idx] / 50)
    ph /= ph.sum(-1, keepdims=True); ql /= ql.sum(-1, keepdims=True)
    want = (ph * np.log(ph / ql)).sum(-1).mean(-1)
    assert np.allclose(lesion.mean_kl(h, l, 50.0, idx), want, rtol=1e-12)


def test_teacher_forced_kl_equals_reference_loop():
    m = QModel.load("doombrain")
    names = ["DEMON CANYON", "HANGAR", "NUCLEAR PLANT"]
    groups = [np.array([[0] + [m.vocab.index(c) for c in n]]) for n in names]
    masks = lesion.masks_from_global(m, np.array([[3], [100], [300]]))
    full, noeos = lesion.mean_kl_batches(m, groups, masks, 150.0, drop=(0,))
    sites = lesion.global_names(m)
    for s, gid in enumerate((3, 100, 300)):
        ab = lesion.ablate_of(sites, [gid])
        tot = tot2 = n = 0
        for g in groups:
            h = intref.forward(m, g[0].tolist())
            x = intref.forward(m, g[0].tolist(), ablate=ab)
            tot += lesion.kl_per_context(h, x[None], 150.0).sum()
            tot2 += lesion.kl_per_context(h, x[None], 150.0, list(range(1, 30))).sum()
            n += len(g[0])
        assert full[s] == pytest.approx(tot / n, rel=1e-12) and noeos[s] == pytest.approx(tot2 / n, rel=1e-12)


def test_wilson_interval():
    lo, hi = lesion.wilson(0, 2000)
    assert lo == 0 and hi == pytest.approx(0.0019, abs=2e-4)
    lo, hi = lesion.wilson(50, 100)
    assert lo < 0.5 < hi and hi - lo < 0.21


def test_shuffled_copy_keeps_values_and_vectors():
    m = QModel.load("doomplayer")
    s = lesion.shuffled_copy(m, 3)
    assert s is not m and not np.array_equal(s.p["L0.Wq"], m.p["L0.Wq"])
    for k, v in m.p.items():
        if v.ndim == 2 and k.split(".")[-1] in ("Wq", "Wk", "Wv", "Wo", "Wup", "Wdn", "Wun"):
            assert np.array_equal(np.sort(s.p[k].ravel()), np.sort(v.ravel()))
        else:
            assert np.array_equal(s.p[k], v)
    assert np.array_equal(lesion.shuffled_copy(m, 3).p["Wun"], s.p["Wun"])
