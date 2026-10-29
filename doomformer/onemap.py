"""A complete integer transformer in one pure DOOM map.

Only token IDs (and their negations) enter the map. Position embeddings are static
biases. All preactivations, including causal attention scores, are exposed; the
staircase activations are physical gates, not host-computed intermediate inputs.
Unused context positions compute independently and cannot influence earlier ones.
No runtime Python bus, model arithmetic in ACS, or per-layer game reloads. Softmax-attention models
(attn="softmax") add den/P/N values and replace wsum/out (see softmax_layers); RMSNorm (norm="rms") adds
s/P/N/r values before qkv and mlp_up (norm_layers); the gated MLP (mlp="reglu") exposes zg/zu in place of zm.

`choose` (optional) adds the decision: the dolls take the exact argmax over the
chosen tokens' logits at the last context position, or at each of `positions`
(lowest index wins ties, like np.argmax). The unembedding also writes d_ab = logit_a - logit_b + 1 (a < b, weights
Wun_a - Wun_b) into floor ports; layer "argmax.compare" gates [d_ab >= 1] and counts
wins_a = sum_{b>a} [s_a >= s_b] - sum_{b<a} [s_b >= s_a] + a; layer "argmax.winner"
gates [wins_a >= k-1], one-hot. `actions` on value("argz", t, a) hand the winner
to the level (selfgen.py: the winner writes the next position's token).
"""
from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import numpy as np

from .intref import PORT_MAX, W_MAX, QModel, adc_relu
from .ops import attend, crossbar, equals, gated, levels, multiply, neg, output_pair, output_single, square
from .pure import compile_pure
from .tile import TileSpec
from .udmf import MapBuilder


def value(key: str, *indices: int) -> str:
    return key + ":" + ":".join(map(str, indices))


def transformer_layers(m: QModel, choose: list[int] | None = None,
                       positions: list[int] | None = None) -> list[TileSpec]:
    """Lower the full context, retaining intref's exact thresholds and residuals.

    choose: vocabulary indices whose logits the dolls compare (see the module docstring) at each context position in
    `positions` (default: the last, T-1)."""
    p, d, T = m.p, m.d, m.T
    if choose is not None and (len(choose) < 2 or len(set(choose)) != len(choose)
                               or any(not 0 <= c < len(m.vocab) for c in choose)):
        raise ValueError("choose: at least two distinct in-vocabulary token indices")
    positions = [T - 1] if positions is None else [int(t) for t in positions]
    if choose is not None and (not positions or len(set(positions)) != len(positions)
                               or any(not 0 <= t < T for t in positions)):
        raise ValueError(f"positions: distinct context positions in 0..{T - 1}")
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
        if m.norm == "rms":
            layers += norm_layers(m, l, "att", incoming)
        s = TileSpec(pre + "qkv")
        for t in range(T):
            outs = {k: [output_pair(s, value(pre + "z" + k, t, j), q["b" + k][j])
                        for j in range(d)] for k in "qkv"}
            for i in range(d):
                _read_unit(s, m, l, "att", t, i, value(incoming, t, i), thr,
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
        if m.attn == "softmax":
            layers += softmax_layers(m, l, incoming)
        else:
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
        if m.norm == "rms":
            layers += norm_layers(m, l, "mlp", pre + "z1")
        s = TileSpec(pre + "mlp_up")
        hidden = (("zg", "Wg", "bg"), ("zu", "Wu", "bu")) if m.mlp == "reglu" else (("zm", "Wup", "bup"),)
        for t in range(T):
            outs = [(output_pair(s, value(pre + z, t, j), q[b][j]), W, j) for z, W, b in hidden for j in range(m.dm)]
            for i in range(d):
                _read_unit(s, m, l, "mlp", t, i, value(pre + "z1", t, i), q["thr_x1"],
                           [(pair, q[W][j, i]) for pair, W, j in outs])
        layers.append(s)
        s = TileSpec(pre + "mlp_dn")
        for t in range(T):
            outs = [output_pair(s, value(pre + "z2", t, j), q["bdn"][j]) for j in range(d)]
            for i in range(m.dm):
                if m.mlp == "reglu":
                    gated(s, value(pre + "zg", t, i), q["thr_g"], value(pre + "zu", t, i), q["thr_u"],
                          [(outs[j], q["Wdn"][j, i]) for j in range(d)])
                else:
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
        if choose is not None and t in positions:
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
        layers += argmax_layers(len(choose), positions)
    return layers


def softmax_layers(m: QModel, l: int, incoming: str) -> list[TileSpec]:
    """Softmax attention of layer l after its scores (intref's softmax branch), three layers:
      den : den:t:h = 1 + sum_u e(h,t,u), crossbars of the e lanes (score staircase thr_a)
      wsum: zo:t:j = sum_u e*v, P:t:j:k and N:t:j:k (l = k+1) = +-R*zo - l*den + 1 (ops.attend)
      out : z1 = bo + Wo @ (sum_l [P_l >= 1] - sum_l [N_l >= 1]) + r1 * x, one indicator lane per P_l / N_l."""
    q, pre, d, T, LV = m.layer(l), f"L{l}.", m.d, m.T, m.lv(l)
    den = TileSpec(pre + "den")
    for h in range(m.H):
        for t in range(T):
            out = output_pair(den, value(pre + "den", t, h), 1)
            for u in range(t + 1):
                crossbar(den, value(pre + "score", h, t, u), q["thr_a"], [(out, 1)], signed=False)
    wsum = TileSpec(pre + "wsum")
    for t in range(T):
        for j in range(d):
            num = output_pair(wsum, value(pre + "zo", t, j))
            P = [output_pair(wsum, value(pre + "P", t, j, k), -k) for k in range(LV)]
            N = [output_pair(wsum, value(pre + "N", t, j, k), -k) for k in range(LV)]
            for u in range(t + 1):
                attend(wsum, value(pre + "score", j // m.dh, t, u), q["thr_a"], value(pre + "zv", u, j), q["thr_v"],
                       num, P, N, m.R)
    out = TileSpec(pre + "out")
    for t in range(T):
        outs = [output_pair(out, value(pre + "z1", t, j), q["bo"][j]) for j in range(d)]
        for i in range(d):
            for k in range(LV):
                for key, sign in (("P", 1), ("N", -1)):
                    crossbar(out, value(pre + key, t, i, k), [1], [(outs[j], sign * q["Wo"][j, i]) for j in range(d)],
                             signed=False)
            crossbar(out, value(incoming, t, i), m.thr_in(l), [(outs[i], q["r1"][i])])
    return [den, wsum, out]


def norm_layers(m: QModel, l: int, site: str, incoming: str) -> list[TileSpec]:
    """RMSNorm of layer l at `site` (intref.rmsnorm) on the staircase of value incoming:t:i, two layers:
      sumsq: s_<site>:t = sum_i c_i^2 (ops.square: lane k of either sign adds 2k-1)
      norm : P_<site>:t:i:k and N_<site>:t:i:k (l = k+1, bias -k) get +-Rn per c_i lane and -(k+1) per lane of the
             staircase of s (thr_r = d*k^2, r = floor(sqrt(s/d))); r_<site>:t counts those lanes.
    Consumers read n through indicator lanes (_read_unit)."""
    pre, d, T, thr, LN = f"L{l}.", m.d, m.T, m.thr_norm(l, site), m.ln(l, site)
    sumsq, norm = TileSpec(f"{pre}sumsq_{site}"), TileSpec(f"{pre}norm_{site}")
    for t in range(T):
        s = output_pair(sumsq, value(f"{pre}s_{site}", t))
        for i in range(d):
            square(sumsq, value(incoming, t, i), thr, s)
        P = [[output_pair(norm, value(f"{pre}P_{site}", t, i, k), -k) for k in range(LN)] for i in range(d)]
        N = [[output_pair(norm, value(f"{pre}N_{site}", t, i, k), -k) for k in range(LN)] for i in range(d)]
        r = output_single(norm, value(f"{pre}r_{site}", t))
        for i in range(d):
            crossbar(norm, value(incoming, t, i), thr, [(p, m.Rn) for p in P[i]] + [(q, -m.Rn) for q in N[i]])
        crossbar(norm, value(f"{pre}s_{site}", t), m.thr_r(l, site),
                 [(r, 1)] + [(row[k], -(k + 1)) for rows in (P, N) for row in rows for k in range(LN)], signed=False)
    return [sumsq, norm]


def _read_unit(spec: TileSpec, m: QModel, l: int, site: str, t: int, i: int, x: str, thr, targets):
    """A consumer's read of input unit i at position t: with norm="rms" the indicator lanes of P_<site>:t:i:k /
    N_<site>:t:i:k (the count n_i), otherwise the staircase of value x."""
    if m.norm == "rms":
        LN, pre = m.ln(l, site), f"L{l}."
        levels(spec, [value(f"{pre}P_{site}", t, i, k) for k in range(LN)],
               [value(f"{pre}N_{site}", t, i, k) for k in range(LN)], targets)
    else:
        crossbar(spec, x, thr, targets)


def pairs(k: int) -> list[tuple[int, int]]:
    return [(a, b) for a in range(k) for b in range(a + 1, k)]


def argmax_layers(k: int, positions: list[int]) -> list[TileSpec]:
    """One-hot argmax of k scores at each position t, whose differences d_ab = s_a - s_b + 1 (a < b) are ports
    value("argd", t, a, b). All positions share the two layers (their dolls never touch the same sector)."""
    compare = TileSpec("argmax.compare")
    for t in positions:
        wins = [output_pair(compare, value("argw", t, a), a) for a in range(k)]
        for a, b in pairs(k):
            crossbar(compare, value("argd", t, a, b), [1], [(wins[a], 1), (wins[b], -1)], signed=False)
    winner = TileSpec("argmax.winner")
    for t in positions:
        for a in range(k):
            crossbar(winner, value("argw", t, a), [k - 1], [(output_single(winner, value("argz", t, a)), 1)],
                     signed=False)
    return [compare, winner]


def _comparator(s: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(d_ab, wins, one-hot) of the dolls' comparator on the scores s [k] (argmax_layers)."""
    k = len(s)
    diff = s[:, None] - s[None, :] + 1
    above = adc_relu(diff, [1]) * np.triu(np.ones((k, k), dtype=np.int64), 1)
    wins = above.sum(1) - above.sum(0) + np.arange(k)
    onehot = adc_relu(wins, [k - 1])
    if onehot.sum() != 1:
        raise AssertionError(f"comparator reference is not one-hot: {onehot.tolist()}")
    return diff, wins, onehot


def argmax_reference(m: QModel, choose: list[int], logits: np.ndarray, trace: dict) -> int:
    """NumPy twin of the dolls' comparator on intref's logits [n, V]; fills trace "argd", "argw", "argz" (only a full
    context, n == T, is compared: the dolls compare position T-1) and returns the winner's index into `choose`."""
    n = len(logits)
    values = _comparator(np.asarray(logits[-1], dtype=np.int64)[list(choose)])
    for key, v in zip(("argd", "argw", "argz"), values):
        trace[key] = np.zeros((n, *v.shape), dtype=np.int64)
        if n == m.T:
            trace[key][-1] = v
    return int(np.argmax(values[2]))


def argmax_trace(m: QModel, choose: list[int], logits: np.ndarray, trace: dict, positions: list[int]) -> list[int]:
    """argmax_reference for comparators at several positions of a full context (logits [T, V]): fills trace "argd",
    "argw", "argz" at those rows (zeros elsewhere) and returns each position's winner (index into `choose`)."""
    if len(logits) != m.T:
        raise ValueError("argmax_trace compares a full context")
    k = len(choose)
    shapes = dict(argd=(k, k), argw=(k,), argz=(k,))
    for key, shape in shapes.items():
        trace[key] = np.zeros((m.T, *shape), dtype=np.int64)
    winners = []
    for t in positions:
        for key, v in zip(shapes, _comparator(np.asarray(logits[t], dtype=np.int64)[list(choose)])):
            trace[key][t] = v
        winners.append(int(np.argmax(trace["argz"][t])))
    return winners


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
                 choose: list[int] | None = None, guard: bool = True, opt=0, timing=None,
                 doors: dict[str, int] | None = None):
    layers = transformer_layers(m, choose)
    consumed = {key.split(":", 1)[1] for s in layers for key in s.ports}
    readout = [name for s in layers for name, _ in s.outputs if name in consumed]
    produced = {name for s in layers for name, _ in s.outputs}
    if set(actions or {}) - produced:
        raise ValueError(f"actions on values the map does not compute: {sorted(set(actions) - produced)}")
    return compile_pure(layers, wad, "transformer", readout=readout, world=world, actions=actions,
                        layout_only=layout_only, extra_lumps=extra_lumps, guard=guard, opt=opt, timing=timing,
                        doors=doors)


def exposed_names(m: QModel) -> set[str]:
    """Every value name a one-map brain of m reads back (all outputs of the lowered layers)."""
    return {name for s in transformer_layers(m) for name, _ in s.outputs}


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
