"""A complete integer transformer in one pure DOOM map.

Only token IDs (and their negations) enter the map. Position embeddings are static
biases. All preactivations, including causal attention scores, are exposed; the
staircase activations are physical gates, not host-computed intermediate inputs.
Unused context positions compute independently and cannot influence earlier ones.
No runtime Python bus, model arithmetic in ACS, or per-layer game reloads.

`choose` (optional) adds the decision: the dolls take the exact argmax over the
chosen tokens' logits at the last context position (lowest index wins ties, like
np.argmax). The unembedding also writes d_ab = logit_a - logit_b + 1 (a < b, weights
Wun_a - Wun_b) into floor ports; layer "argmax.compare" gates [d_ab >= 1] and counts
wins_a = sum_{b>a} [s_a >= s_b] - sum_{b<a} [s_b >= s_a] + a; layer "argmax.winner"
gates [wins_a >= k-1], one-hot. `actions` on value("argz", T-1, a) hand the winner
to the level.
"""
from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import numpy as np

from .intref import PORT_MAX, W_MAX, QModel, adc_relu
from .ops import crossbar, equals, multiply, neg, output_pair, output_single
from .pure import compile_pure
from .tile import TileSpec
from .udmf import MapBuilder


def value(key: str, *indices: int) -> str:
    return key + ":" + ":".join(map(str, indices))


def transformer_layers(m: QModel, choose: list[int] | None = None) -> list[TileSpec]:
    """Lower the full context, retaining intref's exact thresholds and residuals.

    choose: vocabulary indices whose last-position logits the dolls compare (see the module docstring)."""
    p, d, T = m.p, m.d, m.T
    if choose is not None and (len(choose) < 2 or len(set(choose)) != len(choose)
                               or any(not 0 <= c < len(m.vocab) for c in choose)):
        raise ValueError("choose: at least two distinct in-vocabulary token indices")
    layers = []
    s = TileSpec("embed")
    for t in range(T):
        z = [output_pair(s, value("z0", t, j), p["pos_emb"][t, j]) for j in range(d)]
        for token in range(len(m.vocab)):
            equals(s, value("tok", t), token, [(z[j], p["tok_emb"][token, j]) for j in range(d)])
    layers.append(s)
    incoming = "z0"
    for l in range(m.L):
        q, pre, thr = m.layer(l), f"L{l}.", m.thr_in(l)
        s = TileSpec(pre + "qkv")
        for t in range(T):
            outs = {k: [output_pair(s, value(pre + "z" + k, t, j), q["b" + k][j])
                        for j in range(d)] for k in "qkv"}
            for i in range(d):
                crossbar(s, value(incoming, t, i), thr,
                         [(outs[k][j], q["W" + k][j, i]) for k in "qkv" for j in range(d)])
        layers.append(s)
        s = TileSpec(pre + "score")
        for h in range(m.H):
            for t in range(T):
                for u in range(t + 1):
                    out = output_pair(s, value(pre + "score", h, t, u))
                    for j in range(h * m.dh, (h + 1) * m.dh):
                        multiply(s, value(pre + "zq", t, j), q["thr_q"], True,
                                 value(pre + "zk", u, j), q["thr_k"], True, out)
        layers.append(s)
        s = TileSpec(pre + "wsum")
        for t in range(T):
            for j in range(d):
                out = output_pair(s, value(pre + "zo", t, j))
                for u in range(t + 1):
                    multiply(s, value(pre + "score", j // m.dh, t, u), q["thr_a"], False,
                             value(pre + "zv", u, j), q["thr_v"], True, out)
        layers.append(s)
        s = TileSpec(pre + "out")
        for t in range(T):
            outs = [output_pair(s, value(pre + "z1", t, j), q["bo"][j]) for j in range(d)]
            for i in range(d):
                crossbar(s, value(pre + "zo", t, i), q["thr_o"][t],
                         [(outs[j], q["Wo"][j, i]) for j in range(d)])
                crossbar(s, value(incoming, t, i), thr, [(outs[i], q["r1"][i])])
        layers.append(s)
        s = TileSpec(pre + "mlp_up")
        for t in range(T):
            outs = [output_pair(s, value(pre + "zm", t, j), q["bup"][j]) for j in range(m.dm)]
            for i in range(d):
                crossbar(s, value(pre + "z1", t, i), q["thr_x1"],
                         [(outs[j], q["Wup"][j, i]) for j in range(m.dm)])
        layers.append(s)
        s = TileSpec(pre + "mlp_dn")
        for t in range(T):
            outs = [output_pair(s, value(pre + "z2", t, j), q["bdn"][j]) for j in range(d)]
            for i in range(m.dm):
                crossbar(s, value(pre + "zm", t, i), q["thr_m"],
                         [(outs[j], q["Wdn"][j, i]) for j in range(d)], signed=False)
            for i in range(d):
                crossbar(s, value(pre + "z1", t, i), q["thr_x1"], [(outs[i], q["r2"][i])])
        layers.append(s)
        incoming = pre + "z2"
    s = TileSpec("unembed")
    thr = m.thr_in(m.L)
    for t in range(T):
        outs = [output_single(s, value("logits", t, j), p["bun"][j]) for j in range(len(m.vocab))]
        diffs = []
        if choose is not None and t == T - 1:
            for a, b in pairs(len(choose)):
                w = p["Wun"][choose[a]] - p["Wun"][choose[b]]
                bias = int(p["bun"][choose[a]] - p["bun"][choose[b]]) + 1
                bound = abs(bias) + int(np.abs(w).sum()) * len(thr)
                if np.abs(w).max(initial=0) > W_MAX or bound > PORT_MAX:
                    raise ValueError(f"choose: d_{a}{b} needs |W| <= {W_MAX} (has {np.abs(w).max()}) and a port "
                                     f"bound <= {PORT_MAX} (has {bound})")
                diffs.append((output_pair(s, value("argd", t, a, b), bias), w))
        for i in range(d):
            crossbar(s, value(incoming, t, i), thr,
                     [(outs[j], p["Wun"][j, i]) for j in range(len(m.vocab))]
                     + [(pair, w[i]) for pair, w in diffs])
    layers.append(s)
    if choose is not None:
        layers += argmax_layers(len(choose), T - 1)
    return layers


def pairs(k: int) -> list[tuple[int, int]]:
    return [(a, b) for a in range(k) for b in range(a + 1, k)]


def argmax_layers(k: int, t: int) -> list[TileSpec]:
    """One-hot argmax of k scores whose differences d_ab = s_a - s_b + 1 (a < b) are ports value("argd", t, a, b)."""
    compare = TileSpec("argmax.compare")
    wins = [output_pair(compare, value("argw", t, a), a) for a in range(k)]
    for a, b in pairs(k):
        crossbar(compare, value("argd", t, a, b), [1], [(wins[a], 1), (wins[b], -1)], signed=False)
    winner = TileSpec("argmax.winner")
    for a in range(k):
        crossbar(winner, value("argw", t, a), [k - 1], [(output_single(winner, value("argz", t, a)), 1)],
                 signed=False)
    return [compare, winner]


def argmax_reference(m: QModel, choose: list[int], logits: np.ndarray, trace: dict) -> int:
    """NumPy twin of the dolls' comparator on intref's logits [n, V]; fills trace "argd", "argw", "argz" (only a full
    context, n == T, is compared: the dolls compare position T-1) and returns the winner's index into `choose`."""
    n, k = len(logits), len(choose)
    s = np.asarray(logits[-1], dtype=np.int64)[list(choose)]
    diff = s[:, None] - s[None, :] + 1
    above = adc_relu(diff, [1]) * np.triu(np.ones((k, k), dtype=np.int64), 1)
    wins = above.sum(1) - above.sum(0) + np.arange(k)
    onehot = adc_relu(wins, [k - 1])
    if onehot.sum() != 1:
        raise AssertionError(f"comparator reference is not one-hot: {onehot.tolist()}")
    for key, v in (("argd", diff), ("argw", wins), ("argz", onehot)):
        trace[key] = np.zeros((n, *v.shape), dtype=np.int64)
        if n == m.T:
            trace[key][-1] = v
    return int(np.argmax(onehot))


def argmax_from_readout(got: dict[str, int], k: int, t: int) -> int:
    """Index of DOOM's one-hot winner (raises unless exactly one argz light is 1 and the rest 0)."""
    z = [got[value("argz", t, a)] for a in range(k)]
    if sorted(z) != [0] * (k - 1) + [1]:
        raise AssertionError(f"dolls' winner lights are not one-hot: {z}")
    return z.index(1)


def build_onemap(m: QModel, wad: Path, *, layout_only: bool = False,
                 world: Callable[[MapBuilder], str] | None = None,
                 actions: dict[str, list[dict]] | None = None,
                 extra_lumps: list[tuple[str, bytes]] | None = None,
                 choose: list[int] | None = None):
    layers = transformer_layers(m, choose)
    consumed = {key.split(":", 1)[1] for s in layers for key in s.ports}
    readout = [name for s in layers for name, _ in s.outputs if name in consumed]
    produced = {name for s in layers for name, _ in s.outputs}
    if set(actions or {}) - produced:
        raise ValueError(f"actions on values the map does not compute: {sorted(set(actions) - produced)}")
    return compile_pure(layers, wad, "transformer", readout=readout, world=world, actions=actions,
                        layout_only=layout_only, extra_lumps=extra_lumps)


def token_ports(m: QModel, tokens: list[int]) -> dict[str, int]:
    if not 1 <= len(tokens) <= m.T or any(t < 0 or t >= len(m.vocab) for t in tokens):
        raise ValueError("tokens must be a nonempty in-vocabulary sequence within context")
    return {key: v for t in range(m.T) for key, v in
            ((value("tok", t), tokens[t] if t < len(tokens) else 0),
             (neg(value("tok", t)), -tokens[t] if t < len(tokens) else 0))}


def compare_trace(got: dict[str, int], trace: dict, n: int) -> int:
    """Check every exposed scalar and negation for active positions; return count.

    Noncausal scores are intentionally absent: no consumer can read them. Future
    padded positions are excluded, exactly as in the tile orchestrator.
    """
    checked = 0
    for name, actual in got.items():
        negative = name.startswith("-")
        key, *raw = name.lstrip("-").split(":")
        idx = tuple(map(int, raw))
        if key.endswith("score"):
            if idx[1] >= n or idx[2] >= n:
                continue
        elif idx[0] >= n:
            continue
        expected = int(trace[key][idx]) * (-1 if negative else 1)
        if actual != expected:
            raise AssertionError(f"{name}: DOOM {actual}, intref {expected}")
        checked += 1
    return checked


def logits_from_readout(m: QModel, got: dict[str, int], n: int) -> np.ndarray:
    return np.array([[got[value("logits", t, j)] for j in range(len(m.vocab))] for t in range(n)], dtype=np.int64)


def tiny_model(seed: int = 17) -> QModel:
    """Small random-weight intref fixture; includes both signs and duplicate gates."""
    rng = np.random.default_rng(seed)
    d, T, dm, V, K = 4, 3, 5, 5, 2
    weights = lambda shape: rng.integers(-2, 3, shape, dtype=np.int64)
    p = dict(tok_emb=weights((V, d)), pos_emb=weights((T, d)), thr_x0=np.array([1, 2]),
             Wun=weights((V, d)), bun=weights(V))
    for k in "qkv":
        p["L0.W" + k], p["L0.b" + k] = weights((d, d)), weights(d)
        p["L0.thr_" + k] = np.array([1, 3])
    p.update({"L0.thr_a": np.array([1, 2]), "L0.thr_o": np.tile([1, 3], (T, 1)),
              "L0.Wo": weights((d, d)), "L0.r1": weights(d), "L0.bo": weights(d),
              "L0.thr_x1": np.array([1, 2]), "L0.Wup": weights((dm, d)), "L0.bup": weights(dm),
              "L0.thr_m": np.array([1, 2]), "L0.Wdn": weights((d, dm)), "L0.r2": weights(d),
              "L0.bdn": weights(d), "L0.thr_x2": np.array([1, 2])})
    return QModel("$ABCD", T, d, 1, dm, K, K, K, 1, p)
