"""Circuit analysis of an induction model with the intervention API: patching recovery per head and role, and
circuit-only runs (every other head and MLP mean-ablated), on the integer reference and on voodoo dolls.

The task is `doomformer.induction`: after BOS and a prefix of distinct tokens, a run of the prefix repeats and the model
must give, at the last repeated token, the token that followed its first occurrence. A pair is two token lists that
differ in two swapped positions and in the answer; `clean` ends at the last queried position. Two kinds:
    kind="answer"   swap the answer token (position j + 1, right after the first occurrence j of the query token) with
                    another prefix token (position k). Every position still has the same previous token, so keys and
                    attention patterns are the same in both runs: the pair tests what is copied.
    kind="source"   swap the tokens at j and k (k not next to j). The query token now first occurs at k, so the
                    induction heads must find position k + 1: the pair tests how the source is found (previous-token
                    information at the position after each occurrence, the keys that carry it, the queries).
The metric is logit[answer] - logit[distractor], where the distractor is the token the corrupt run must give.

Roles name positions relative to the sequence, so rows aggregate over pairs of different lengths:
    query  the last position t (the token being asked about)       src   j, the first occurrence of the query token
    ans    j + 1, the token that followed it                       dist  k ("answer" kind: where the answer went)
    src2   k ("source" kind: the new first occurrence)             ans2  k + 1 ("source" kind)
    bos    position 0                                              rest  every other position (all at once)

`role_recovery` patches one cell at a time (a head's output, q, k or v of a head, or a whole x/MLP site) from the clean
run into the corrupt run at one role's position, for every pair, and reports the mean recovery fraction.
`faithfulness` keeps a chosen set of heads and MLP layers and replaces the output of every other head and every MLP
hidden unit, at every position, by its mean over a dataset (`doomformer.intervene.mean`), and reports the share of the
logit difference the circuit alone restores:
    (LD(circuit) - LD(all mean-ablated)) / (LD(full model) - LD(all mean-ablated)).
Both functions can run chosen rows in DOOM (`DoomRuns`), exact at tolerance 0.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import induction as I
from . import intervene as iv
from .intref import QModel, forward

ROLES = {"answer": ("query", "src", "ans", "dist", "bos", "rest"),
         "source": ("query", "src", "ans", "src2", "ans2", "bos", "rest")}


@dataclass
class Pair:
    kind: str
    clean: list[int]
    corrupt: list[int]
    answer: int
    distractor: int
    roles: dict[str, tuple[int, ...]]

    def ld(self, logits) -> int:
        return int(logits[self.answer] - logits[self.distractor])


def induction_pairs(n: int, seed: int = 0, split: str = "test", kind: str = "answer") -> list[Pair]:
    """n clean/corrupt pairs of `kind` from the held-out sequences of `doomformer.induction` (see the module
    docstring). The same seed gives the same clean sequences for both kinds."""
    if kind not in ROLES:
        raise ValueError(f"kind must be one of {tuple(ROLES)}")
    sets = I.test_sets()[split]
    rng = np.random.default_rng(seed)
    out = []
    for i in rng.permutation(len(sets["tok"])):
        row, tgt, length = sets["tok"][i], sets["tgt"][i], int(sets["length"][i])
        queried = [t for t in range(length) if tgt[t] != I.IGNORE]
        t = queried[-1]
        seq = row[: t + 1].tolist()
        j = seq.index(seq[t], 1)                                    # first occurrence of the query token
        prefix = t - len(queried)                                   # positions 1 .. prefix hold the distinct tokens
        banned = set(range(j - len(queried) + 1, j + 2))            # the run's sources and the answer
        free = [k for k in range(1, prefix + 1) if k not in banned]
        if kind == "source":                                        # k + 1 must be a prefix position as well
            free = [k for k in free if k < prefix]
        if not free:
            continue
        k = int(rng.choice(free))
        corrupt = list(seq)
        if kind == "answer":
            corrupt[j + 1], corrupt[k] = seq[k], seq[j + 1]
            answer, distractor = seq[j + 1], seq[k]
            named = {"query": t, "src": j, "ans": j + 1, "dist": k, "bos": 0}
        else:
            corrupt[j], corrupt[k] = seq[k], seq[j]
            answer, distractor = seq[j + 1], corrupt[k + 1]
            named = {"query": t, "src": j, "ans": j + 1, "src2": k, "ans2": k + 1, "bos": 0}
        roles = {r: (p,) for r, p in named.items()}
        roles["rest"] = tuple(p for p in range(t + 1) if p not in set(named.values()))
        out.append(Pair(kind, seq, corrupt, answer, distractor, roles))
        if len(out) == n:
            break
    return out


def mean_data(pairs: list[Pair]) -> list[list[int]]:
    """The clean sequences of the pairs: the dataset the mean ablation averages over."""
    return [p.clean for p in pairs]


def cells(m: QModel, kinds=("ao", "q", "k", "v"), whole=("x1", "m", "x2")) -> list[tuple]:
    """(site, head or None) of every cell: per head for kinds, the whole site for `whole`."""
    out = []
    for l in range(m.L):
        out += [(f"L{l}.{k}", h) for k in kinds for h in range(m.H)]
        out += [(f"L{l}.{k}", None) for k in whole]
    return out


def _resample(site, head, donor, pos):
    return iv.resample(site, donor, heads=None if head is None else [head], pos=pos)


def role_recovery(m: QModel, pairs: list[Pair], cell_list=None, roles=None, runs=None, doom_rows=()) -> dict:
    """Mean recovery per (cell, role) over the pairs, patching clean into corrupt.

    Returns {"cells": [...], "roles": [...], "recovery": array [cells, roles], "std": same, "doom": [row dicts]}.
    doom_rows: (cell index, role, pair index) triples to run in DOOM with `runs` (a DoomRuns); each row dict has
    exact, values, seconds and the reference recovery."""
    cell_list = cell_list or cells(m)
    roles = roles or ROLES[pairs[0].kind]
    rec = np.zeros((len(cell_list), len(roles), len(pairs)))
    clean_ld = np.array([p.ld(forward(m, p.clean)[-1]) for p in pairs])
    corrupt_ld = np.array([p.ld(forward(m, p.corrupt)[-1]) for p in pairs])
    plans = {}
    for ci, (site, head) in enumerate(cell_list):
        for ri, role in enumerate(roles):
            for pi, p in enumerate(pairs):
                pos = p.roles[role]
                if not pos:
                    continue
                plan = iv.compile_plan(m, None, [_resample(site, head, p.clean, pos)])
                plans[(ci, ri, pi)] = plan
                ld = p.ld(forward(m, p.corrupt, interventions=plan)[-1])
                denom = clean_ld[pi] - corrupt_ld[pi]
                rec[ci, ri, pi] = (ld - corrupt_ld[pi]) / denom if denom else 0.0
    doom = []
    for ci, role, pi in doom_rows:
        ri = list(roles).index(role)
        p, plan = pairs[pi], plans[(ci, ri, pi)]
        rep = runs.run(p.corrupt, plan)
        doom.append(dict(cell=cell_list[ci], role=role, pair=pi, exact=rep["exact"], values=rep["values"],
                         seconds=rep["seconds"], patched=rep["patches"], recovery=float(rec[ci, ri, pi])))
    return dict(cells=cell_list, roles=list(roles), recovery=rec.mean(-1), std=rec.std(-1), doom=doom,
                clean_ld=clean_ld, corrupt_ld=corrupt_ld)


def circuit_plan(m: QModel, keep_heads, keep_mlp, data: list[list[int]]) -> list:
    """Interventions that mean-ablate every head output not in keep_heads ((layer, head) pairs) and the hidden units of
    every MLP layer not in keep_mlp, at every position, over `data`."""
    ivs = []
    for l in range(m.L):
        drop = [h for h in range(m.H) if (l, h) not in keep_heads]
        if drop:
            ivs.append(iv.mean(f"L{l}.ao", data, heads=drop))
        if l not in keep_mlp:
            ivs.append(iv.mean(f"L{l}.m", data))
    return ivs


def faithfulness(m: QModel, pairs: list[Pair], keep_heads, keep_mlp=(), data=None, runs=None, doom_pairs=(),
                 cache: dict | None = None) -> dict:
    """Share of the logit difference the circuit restores (see the module docstring), over the clean runs of `pairs`.

    keep_heads: (layer, head) pairs; keep_mlp: layers whose MLP stays. data: mean-ablation dataset (default: the
    clean sequences of the pairs). doom_pairs: indices of pairs whose circuit-only and all-ablated runs also run in
    DOOM with `runs`; returns their exactness in "doom". cache: donor traces kept between calls."""
    data = data or mean_data(pairs)
    keep_heads = {tuple(h) for h in keep_heads}
    circuit = circuit_plan(m, keep_heads, set(keep_mlp), data)
    none = circuit_plan(m, set(), set(), data)
    cache = {} if cache is None else cache
    plans = {name: iv.compile_plan(m, None, ivs, cache) for name, ivs in (("circuit", circuit), ("none", none))}
    full = np.array([p.ld(forward(m, p.clean)[-1]) for p in pairs])
    got = {name: np.array([p.ld(forward(m, p.clean, interventions=plan)[-1]) for p in pairs])
           for name, plan in plans.items()}
    top1 = float(np.mean([int(np.argmax(forward(m, p.clean, interventions=plans["circuit"])[-1])) == p.answer
                          for p in pairs]))
    denom = (full - got["none"]).mean()
    doom = []
    for pi in doom_pairs:
        for name, plan in plans.items():
            rep = runs.run(pairs[pi].clean, plan)
            doom.append(dict(pair=pi, run=name, exact=rep["exact"], values=rep["values"], seconds=rep["seconds"],
                             patched=rep["patches"]))
    return dict(faithfulness=float((got["circuit"] - got["none"]).mean() / denom) if denom else float("nan"),
                ld_full=float(full.mean()), ld_circuit=float(got["circuit"].mean()), ld_none=float(got["none"].mean()),
                top1_circuit=top1, top1_full=float(np.mean([int(np.argmax(forward(m, p.clean)[-1])) == p.answer
                                                            for p in pairs])), doom=doom)
