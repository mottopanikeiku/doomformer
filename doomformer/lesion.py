"""Exact batched zero-ablation reference: many lesions of one relu/none/relu QModel in one vectorised pass.

`intref.forward(m, tokens, ablate={key: units})` zeroes activation units (in DOOM: closes their doors). This module
evaluates S such lesions on a batch of B token sequences at once and returns the same logits, bit for bit.

Lesion sites are the keys of `demos.shotgun.ablation_keys`: "x0" and, for every layer l, "L{l}.ao", "L{l}.x1",
"L{l}.m", "L{l}.x2". A lesion set is a dict `masks`: key -> bool array [S, units]; a missing key lesions nothing.

Exactness. All parameters and activations are small integers and every GEMM runs in float64. A float64 holds every
integer up to 2**53 and a sum of integers below that bound is exact in any order, so the float64 GEMMs equal the int64
sums of `intref`. `check_exact` proves the bound statically (sum of |w| times the largest input level, per output
row, for every GEMM) and `forward_lesions` calls it first. The staircases count `z >= t` exactly as
`intref.adc_signed` / `adc_relu` do (compare, then sum), with the same causal attention.

Cost. A lesion at a late site leaves earlier tensors untouched, so tensors keep a leading lesion axis of size 1 until
a lesion differs; nothing is recomputed per lesion before the first lesioned site. With `all_positions=False` the last
layer evaluates only the last query position (all that an action or a next-token sample needs).
"""
from __future__ import annotations

import itertools
import math

import numpy as np

from .intref import QModel

F64 = np.float64
EXACT_LIMIT = 2 ** 53


def sites(m: QModel) -> dict[str, int]:
    """Every lesion site -> number of units (same order and keys as `demos.shotgun.ablation_keys`)."""
    out = {"x0": m.d}
    for l in range(m.L):
        out |= {f"L{l}.ao": m.d, f"L{l}.x1": m.d, f"L{l}.m": m.dm, f"L{l}.x2": m.d}
    return out


def _check_model(m: QModel) -> None:
    if (m.attn, m.norm, m.mlp) != ("relu", "none", "relu"):
        raise ValueError(f"lesion.py supports relu attention, norm none, relu MLP; got attn={m.attn!r}, "
                         f"norm={m.norm!r}, mlp={m.mlp!r}")


def check_exact(m: QModel) -> int:
    """Largest possible |partial sum| of any GEMM of the model, bounded from the weights and the staircase heights;
    raises AssertionError when it reaches 2**53 (float64 sums would no longer be exact). Returns the bound."""
    _check_model(m)
    p = m.p
    lev = max(len(p["thr_x0"]), *(len(m.layer(l)[k]) for l in range(m.L) for k in ("thr_q", "thr_a", "thr_x1", "thr_m",
                                                                                   "thr_x2")),
              *(m.layer(l)["thr_o"].shape[-1] for l in range(m.L)))
    assert max(int(np.abs(a).max()) for a in p.values() if a.size) < 2 ** 31, "parameters too large for the bound"
    row = lambda w, bias, x: int((np.abs(w).sum(1) * x + bias).max())   # sum of |w| times the input bound + |bias|
    worst = row(p["Wun"], np.abs(p["bun"]), lev)
    for l in range(m.L):
        q = m.layer(l)
        worst = max(worst, *(row(q[w], np.abs(q[b]), lev) for w, b in (("Wq", "bq"), ("Wk", "bk"), ("Wv", "bv"))))
        worst = max(worst, m.dh * lev * lev, m.T * lev * lev)             # score (q.k), zo (sum of a * v)
        worst = max(worst, row(q["Wo"], np.abs(q["bo"]) + np.abs(q["r1"]) * lev, lev), row(q["Wup"], np.abs(q["bup"]), lev),
                    row(q["Wdn"], np.abs(q["bdn"]) + np.abs(q["r2"]) * lev, lev))
    z0 = int((np.abs(p["tok_emb"]).max(0) + np.abs(p["pos_emb"]).max(0)).max())
    worst = max(worst, z0)
    assert worst < EXACT_LIMIT, f"GEMM bound {worst} >= 2**53: float64 would not be exact"
    return worst


def _stair(z: np.ndarray, thr: np.ndarray, signed: bool) -> np.ndarray:
    """sum_k [z >= thr_k] (minus sum_k [-z >= thr_k] if signed); thr broadcasts against z on the last axis of `thr`
    removed (thr[..., k] against z), which covers 1-D thresholds and the per-position/per-unit thr_o."""
    out = np.zeros(np.broadcast_shapes(z.shape, thr.shape[:-1]), dtype=F64)
    for k in range(thr.shape[-1]):
        t = thr[..., k]
        out += z >= t
        if signed:
            out -= (-z) >= t
    return out


def _lin(x: np.ndarray, w: np.ndarray, b: np.ndarray) -> np.ndarray:
    """x [..., din] @ w.T + b as one GEMM over the flattened leading axes."""
    y = x.reshape(-1, x.shape[-1]) @ w.T
    y += b
    return y.reshape(*x.shape[:-1], w.shape[0])


def _keep(masks: dict, key: str, S: int) -> np.ndarray | None:
    """float64 [S, 1, 1, units] of 1 (keep) / 0 (lesioned), or None when the site has no lesion."""
    mk = masks.get(key)
    if mk is None:
        return None
    mk = np.asarray(mk, dtype=bool)
    assert mk.ndim == 2 and mk.shape[0] == S, f"{key}: mask must be [S={S}, units], got {mk.shape}"
    return (~mk).astype(F64)[:, None, None, :]


def _forward_chunk(m: QModel, P: dict, tokens: np.ndarray, keeps: dict, all_positions: bool) -> np.ndarray:
    """Logits for one lesion chunk. Tensors are [S or 1, B, n, ...]."""
    B, n = tokens.shape
    z0 = P["tok_emb"][tokens] + P["pos_emb"][:n]                       # [B, n, d]
    x = _stair(z0, P["thr_x0"], True)[None]
    if keeps.get("x0") is not None:
        x = x * keeps["x0"]
    for l in range(m.L):
        q_ = P[l]
        last = l == m.L - 1 and not all_positions
        nq = 1 if last else n                                          # query positions computed
        q = _stair(_lin(x, q_["Wq"], q_["bq"]), q_["thr_q"], True)
        k = _stair(_lin(x, q_["Wk"], q_["bk"]), q_["thr_k"], True)
        v = _stair(_lin(x, q_["Wv"], q_["bv"]), q_["thr_v"], True)
        S_ = x.shape[0]
        sh = lambda u, rows: u.reshape(S_, B, rows, m.H, m.dh)
        qh = sh(q[:, :, n - nq:], nq).transpose(0, 1, 3, 2, 4)         # [S,B,H,nq,dh]
        kh = sh(k, n).transpose(0, 1, 3, 4, 2)                         # [S,B,H,dh,n]
        vh = sh(v, n).transpose(0, 1, 3, 2, 4)                         # [S,B,H,n,dh]
        score = np.matmul(qh, kh)                                      # [S,B,H,nq,n]
        causal = np.tril(np.ones((n, n), dtype=F64))[n - nq:]
        a = _stair(score, q_["thr_a"], False) * causal
        zo = np.matmul(a, vh).transpose(0, 1, 3, 2, 4).reshape(S_, B, nq, m.d)
        ao = _stair(zo, q_["thr_o"][n - nq:n, None, :], True)
        if keeps.get(f"L{l}.ao") is not None:
            ao = ao * keeps[f"L{l}.ao"]
        xr = x[:, :, n - nq:]
        z1 = _lin(ao, q_["Wo"], q_["bo"]) + q_["r1"] * xr
        x1 = _stair(z1, q_["thr_x1"], True)
        if keeps.get(f"L{l}.x1") is not None:
            x1 = x1 * keeps[f"L{l}.x1"]
        mh = _stair(_lin(x1, q_["Wup"], q_["bup"]), q_["thr_m"], False)
        if keeps.get(f"L{l}.m") is not None:
            mh = mh * keeps[f"L{l}.m"]
        z2 = _lin(mh, q_["Wdn"], q_["bdn"]) + q_["r2"] * x1
        x = _stair(z2, q_["thr_x2"], True)
        if keeps.get(f"L{l}.x2") is not None:
            x = x * keeps[f"L{l}.x2"]
    logits = _lin(x if all_positions else x[:, :, -1:], P["Wun"], P["bun"])
    return logits if all_positions else logits[:, :, 0]


def forward_lesions(m: QModel, tokens, masks: dict, all_positions: bool = False, max_rows: int = 1 << 16,
                    S: int | None = None) -> np.ndarray:
    """Logits of S lesions on a token batch: float64 [S, B, V] (last position), or [S, B, n, V] with
    `all_positions`. `tokens` is [B, n] (one sequence [n] is accepted); `masks[key]` is bool [S, units]. Equal to
    stacking `intref.forward(m, tokens[b], ablate={key: where(masks[key][s])})` (last row, or all rows). Lesions run
    in chunks of about `max_rows` token rows to bound memory."""
    _check_model(m)
    check_exact(m)
    tokens = np.asarray(tokens, dtype=np.int64)
    if tokens.ndim == 1:
        tokens = tokens[None]
    assert 1 <= tokens.shape[1] <= m.T
    if S is None:
        S = next(iter(masks.values())).shape[0] if masks else 1
    P = {k: np.asarray(m.p[k], dtype=F64) for k in ("tok_emb", "pos_emb", "thr_x0", "Wun", "bun")}
    for l in range(m.L):
        P[l] = {k: np.asarray(v, dtype=F64) for k, v in m.layer(l).items()}
    step = max(1, max_rows // (tokens.shape[0] * tokens.shape[1]))
    out = []
    for s0 in range(0, S, step):
        s1 = min(S, s0 + step)
        keeps = {k: _keep({k: mk[s0:s1]}, k, s1 - s0) for k, mk in masks.items()}
        lg = _forward_chunk(m, P, tokens, keeps, all_positions)
        if lg.shape[0] != s1 - s0:                                      # no lesion in this chunk touched the batch
            lg = np.broadcast_to(lg, (s1 - s0,) + lg.shape[1:])
        out.append(lg)
    return np.concatenate(out) if out else np.zeros((0, tokens.shape[0], P["Wun"].shape[0]))


# ---------------------------------------------------------------------------------------------- lesion sets
def masks_from_ablate(m: QModel, ablates: list[dict]) -> dict:
    """[{key: [units]}, ...] -> masks {key: bool [S, units]} for every site used by at least one lesion."""
    keys = sites(m)
    masks = {}
    for s, ab in enumerate(ablates):
        for k, units in ab.items():
            if k not in masks:
                masks[k] = np.zeros((len(ablates), keys[k]), dtype=bool)
            masks[k][s, list(units)] = True
    return masks


def masks_from_global(m: QModel, ids: np.ndarray) -> dict:
    """ids [S, k] of global neuron numbers (sites in `sites(m)` order, units in order within a site) ->
    masks {key: bool [S, units]}; a site no neuron falls on is left out."""
    ids = np.asarray(ids)
    masks, lo = {}, 0
    for key, n in sites(m).items():
        sel = (ids >= lo) & (ids < lo + n)
        if sel.any():
            mk = np.zeros((ids.shape[0], n), dtype=bool)
            r, c = np.nonzero(sel)
            mk[r, ids[r, c] - lo] = True
            masks[key] = mk
        lo += n
    return masks


def global_names(m: QModel) -> list[tuple[str, int]]:
    """(site, unit) of every global neuron number."""
    return [(k, u) for k, n in sites(m).items() for u in range(n)]


def ablate_of(names: list[tuple[str, int]], ids) -> dict:
    """Global neuron numbers -> an `intref` ablate dict."""
    out: dict[str, list[int]] = {}
    for i in ids:
        k, u = names[int(i)]
        out.setdefault(k, []).append(u)
    return {k: sorted(v) for k, v in out.items()}


def _colex(n: int, r: int) -> np.ndarray:
    """All r-subsets of range(n) in colex order, int [C(n, r), r]; the first C(M, r) rows are the r-subsets of
    range(M) for every M <= n."""
    if r == 0:
        return np.zeros((1, 0), dtype=np.int32)
    parts = [np.concatenate([_colex(c, r - 1), np.full((math.comb(c, r - 1), 1), c, dtype=np.int32)], axis=1)
             for c in range(r - 1, n)]
    return np.concatenate(parts) if parts else np.zeros((0, r), dtype=np.int32)


def subset_chunks(n: int, k: int, chunk: int = 1 << 15):
    """All k-subsets of range(n), each exactly once, as int32 arrays [rows, k] of about `chunk` rows (the last
    chunk is smaller). Prefixes of k-3 elements come from itertools, the last three from a precomputed table."""
    tail = min(k, 3)
    base = _colex(n, tail)
    buf, rows = [], 0
    for prefix in itertools.combinations(range(n), k - tail):
        p = prefix[-1] if prefix else -1
        blk = base[:math.comb(n - p - 1, tail)] + (p + 1)
        if prefix:
            blk = np.concatenate([np.broadcast_to(np.array(prefix, dtype=np.int32), (len(blk), len(prefix))), blk], axis=1)
        buf.append(blk)
        rows += len(blk)
        if rows >= chunk:
            out = np.concatenate(buf)
            for i in range(0, len(out) - chunk + 1, chunk):
                yield out[i:i + chunk]
            rest = out[(len(out) // chunk) * chunk:]
            buf, rows = ([rest], len(rest)) if len(rest) else ([], 0)
    if rows:
        yield np.concatenate(buf)


def n_subsets(n: int, k: int) -> int:
    return math.comb(n, k)


def wilson(hits: int, n: int, z: float = 1.959964) -> tuple[float, float]:
    """Wilson score interval (default 95 %) of a rate hits / n."""
    if n == 0:
        return (0.0, 1.0)
    p = hits / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return (0.0 if hits == 0 else max(0.0, c - h)), (1.0 if hits == n else min(1.0, c + h))


# ---------------------------------------------------------------------------------------------- actions, predicates
def action_tables(m: QModel, tokens, masks: dict, action_ids, **kw) -> np.ndarray:
    """int [S, B]: index into `action_ids` of the largest action-token logit of every lesion on every context
    (lowest index on ties, as `policy.act_index`)."""
    lg = forward_lesions(m, tokens, masks, **kw)
    return np.argmax(lg[..., list(action_ids)], axis=-1)


def pacifist(tables: np.ndarray, fire: int) -> np.ndarray:
    """bool [S]: the lesion never chooses action index `fire` on any context."""
    return ~(tables == fire).any(axis=-1)


def changes_action(tables: np.ndarray, base: np.ndarray) -> np.ndarray:
    """bool [S]: the lesion changes the chosen action on at least one context."""
    return (tables != np.asarray(base)).any(axis=-1)


def survivors(m: QModel, tokens, masks: dict, bad, action_ids, stages=(1, 2, 4, 8, 16, 32, 64, 1 << 30)) -> np.ndarray:
    """bool [S]: the lesion has no context for which `bad(actions, rows)` holds, evaluated in stages of growing
    context count so that a lesion is dropped at the first context that condemns it. `tokens` [B, n] is in the order
    the contexts are tried; `bad(actions [s, b], rows)` gets the action indices of the surviving lesions on the
    contexts `rows` (an index range into `tokens`) and returns bool [s, b]. The result equals evaluating every
    lesion on every context; only the work shrinks."""
    S = next(iter(masks.values())).shape[0]
    alive = np.arange(S)
    lo = 0
    for st in stages:
        hi = min(len(tokens), lo + st)
        if lo >= hi or alive.size == 0:
            break
        sub = {k: mk[alive] for k, mk in masks.items()}
        rows = np.arange(lo, hi)
        acts = action_tables(m, tokens[lo:hi], sub, action_ids)
        alive = alive[~bad(acts, rows).any(axis=1)]
        lo = hi
    out = np.zeros(S, dtype=bool)
    out[alive] = True
    return out


def context_orders(m: QModel, tokens, base, fire: int, action_ids, masks: dict) -> tuple[np.ndarray, np.ndarray]:
    """Orders of the contexts for the two cascades, most condemning first, estimated on the sample lesions `masks`:
    (pacifist order: contexts where a lesion most often still chooses `fire`; changed order: contexts where the
    action most often changes). Only the speed of `classify` depends on them, never its result."""
    acts = action_tables(m, tokens, masks, action_ids)
    return (np.argsort(-(acts == fire).mean(0), kind="stable"),
            np.argsort(-(acts != np.asarray(base)).mean(0), kind="stable"))


def classify(m: QModel, tokens, base, fire: int, action_ids, masks: dict, orders=None) -> tuple[np.ndarray, np.ndarray]:
    """(pacifist, changed) bool [S]: the lesion never chooses action `fire` on any context / changes the action on
    at least one context. Exactly `pacifist(action_tables(...))` and `changes_action(action_tables(...), base)`,
    computed by staged cascades that stop at the first context that decides a lesion."""
    tokens, base = np.asarray(tokens), np.asarray(base)
    op, oc = orders if orders is not None else (np.arange(len(tokens)),) * 2
    pac = survivors(m, tokens[op], masks, lambda a, rows: a == fire, action_ids)
    bc = base[oc]
    unchanged = survivors(m, tokens[oc], masks, lambda a, rows: a != bc[rows], action_ids)
    return pac, ~unchanged


# ---------------------------------------------------------------------------------------------- logit KL
def _log_softmax(z: np.ndarray, tau: float) -> np.ndarray:
    z = np.asarray(z, dtype=F64) / tau
    z = z - z.max(-1, keepdims=True)
    return z - np.log(np.exp(z).sum(-1, keepdims=True))


def kl_per_context(healthy: np.ndarray, lesioned: np.ndarray, tau: float = 150.0, idx=None) -> np.ndarray:
    """KL(softmax(healthy / tau) || softmax(lesioned / tau)) per context. `healthy` [..., V] (one row per context),
    `lesioned` [S, ..., V]; `idx` restricts the softmax to those vocabulary entries (the three action tokens for
    the marine). Returns [S, ...]."""
    h, l = np.asarray(healthy, dtype=F64), np.asarray(lesioned, dtype=F64)
    if idx is not None:
        h, l = h[..., list(idx)], l[..., list(idx)]
    lp, lq = _log_softmax(h, tau), _log_softmax(l, tau)
    return (np.exp(lp) * (lp - lq)).sum(-1)


def mean_kl(healthy: np.ndarray, lesioned: np.ndarray, tau: float = 150.0, idx=None) -> np.ndarray:
    """Mean over contexts of `kl_per_context`: float64 [S]."""
    kl = kl_per_context(healthy, lesioned, tau, idx)
    return kl.reshape(kl.shape[0], -1).mean(1)


def healthy_logits(m: QModel, tokens, **kw) -> np.ndarray:
    """Logits of the unlesioned model on the batch (last position, or all positions with `all_positions`)."""
    return forward_lesions(m, tokens, {}, S=1, **kw)[0]


# ---------------------------------------------------------------------------------------------- teacher-forced scans
def mean_kl_batches(m: QModel, groups: list[np.ndarray], masks: dict, tau: float = 150.0, drop=(), block: int = 32,
                    **kw) -> tuple[np.ndarray, np.ndarray]:
    """Teacher-forced mean KL of S lesions over every position of every token sequence. `groups` is a list of token
    arrays [B_g, n_g] (sequences of equal length n_g, no padding); every position counts once. Returns
    (KL over the full vocabulary, KL over the vocabulary without the entries in `drop`), each float64 [S]."""
    S = next(iter(masks.values())).shape[0]
    keep = [i for i in range(m.p["Wun"].shape[0]) if i not in set(drop)]
    tot, tot_k, npos = np.zeros(S), np.zeros(S), 0
    health = [healthy_logits(m, g, all_positions=True, **kw) for g in groups]
    for s0 in range(0, S, block):
        sub = {k: mk[s0:s0 + block] for k, mk in masks.items()}
        for g, h in zip(groups, health):
            lg = forward_lesions(m, g, sub, all_positions=True, S=min(block, S - s0), **kw)     # [s, B, n, V]
            for acc, idx in ((tot, None), (tot_k, keep)):
                acc[s0:s0 + block] += kl_per_context(h, lg, tau, idx).reshape(lg.shape[0], -1).sum(1)
    npos = sum(g.size for g in groups)
    return tot / npos, tot_k / npos


def shuffled_copy(m: QModel, seed: int) -> QModel:
    """The model with the entries of every weight matrix (Wq, Wk, Wv, Wo, Wup, Wdn of every layer and Wun) permuted
    at random; embeddings, biases, residual scales r1 / r2 and thresholds are kept."""
    import dataclasses

    rng = np.random.default_rng(seed)
    p = dict(m.p)
    for name in ["Wun"] + [f"L{l}.{w}" for l in range(m.L) for w in ("Wq", "Wk", "Wv", "Wo", "Wup", "Wdn")]:
        p[name] = rng.permutation(p[name].ravel()).reshape(p[name].shape)
    return dataclasses.replace(m, p=p)
