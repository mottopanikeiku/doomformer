"""The induction task: copy the token that followed the first occurrence.

A sequence is BOS, then `m` distinct random tokens (the prefix), then `r` tokens that repeat a run of the prefix
(`r` >= 1, the run starts at a random prefix position). At every repeated token the target is the token that followed
that token's first occurrence, so the last token of the run has the prefix token after the run as target:

    $ K F C A P J D | C A P       tokens fed (BOS $, prefix of 7, repeat of 3)
                     A P J       targets at the three repeated tokens

The prefix tokens are drawn without replacement from V = 16 tokens, so every token occurs once in the prefix and the
answer is unique. No count table over tokens can solve it: given the current token, the next token is uniform over the
other prefix tokens. The only way to answer is to look back in the sequence.

Held-out data. `test_sets` returns fixed sequences. `train_batch` draws fresh random sequences and drops every one
that is in the test, unseen or dev sets, so no training sequence is a held-out one. The "unseen" set is stricter: 24
ordered token pairs (HELD_PAIRS) never occur next to each other in any training sequence, and every queried pair of an
unseen sequence is one of them. A model that had stored token transitions has never seen any of them.

Targets are IGNORE (-100) except at the repeated tokens. Everything here is NumPy (the trainer converts to torch).
"""
from __future__ import annotations

import numpy as np

V = 16                                  # content tokens 1..16 (letters A..P); token 0 is BOS
VOCAB = "$" + "ABCDEFGHIJKLMNOP"
T = 10                                  # context length of the shipped model
IGNORE = -100
R_CAP = 2                               # at most 2 repeated tokens
N_HELD = 24                             # held-out ordered token pairs
SEEDS = dict(test=20_260_101, unseen=20_260_102, dev=20_260_103, pairs=7)
N_TEST = 4000
N_UNSEEN = 2000
N_DEV = 2000

_POW = (np.random.default_rng(1).integers(1, 2 ** 62, 64, dtype=np.uint64) * np.uint64(2) + np.uint64(1))


def held_pairs(v: int = V) -> np.ndarray:
    """Boolean [v+1, v+1]: True at the N_HELD ordered pairs (a, b), a != b, that training sequences never contain."""
    allp = np.array([(a, b) for a in range(1, v + 1) for b in range(1, v + 1) if a != b])
    pick = allp[np.random.default_rng(SEEDS["pairs"]).permutation(len(allp))[:N_HELD]]
    held = np.zeros((v + 1, v + 1), dtype=bool)
    held[pick[:, 0], pick[:, 1]] = True
    return held


def row_hash(tok: np.ndarray) -> np.ndarray:
    """64-bit hash of every row of tok [n, T] (a row determines the whole sequence, targets included)."""
    return (tok.astype(np.uint64) * _POW[: tok.shape[1]]).sum(1)


def _draw(rng, n: int, t: int, v: int):
    """n sequences, unfiltered: tok [n, t], tgt [n, t], length [n], prefix length m [n], run start i [n], run length r."""
    perm = np.argsort(rng.random((n, v)), axis=1) + 1               # random order of tokens 1..v
    m = rng.integers(5, t - 1, size=n)                              # prefix length, 5 .. t-2
    rmax = np.minimum(np.minimum(R_CAP, m - 1), t - 1 - m)
    r = 1 + (rng.random(n) * rmax).astype(np.int64)                 # 1 .. rmax
    i = 1 + (rng.random(n) * (m - r)).astype(np.int64)              # the run starts at prefix position i, i + r <= m
    pos = np.arange(t + 1)[None]
    src = np.where(pos <= m[:, None], pos, i[:, None] + pos - m[:, None] - 1)   # prefix, then the run, one past it
    e = np.where(pos == 0, 0, np.take_along_axis(perm, np.clip(src - 1, 0, v - 1), 1))
    length = 1 + m + r
    e = np.where(pos <= length[:, None], e, 0)
    tok = np.where(pos[:, :t] < length[:, None], e[:, :t], 0)
    inrun = (pos[:, :t] > m[:, None]) & (pos[:, :t] <= (m + r)[:, None])
    tgt = np.where(inrun, e[:, 1:], IGNORE)
    return tok.astype(np.int64), tgt.astype(np.int64), length, m, i, r


def sample(rng, n: int, t: int = T, v: int = V, *, forbid: np.ndarray | None = None,
           queries: np.ndarray | None = None, exclude: np.ndarray | None = None, unique: bool = False):
    """n sequences (tok [n, t], tgt [n, t], length [n]). forbid: boolean pair matrix, no two adjacent tokens of the
    sequence may be a pair in it; queries: every queried (current, target) pair must be in it; exclude: row hashes
    that must not occur; unique: no sequence twice."""
    out, have = [], 0
    seen = np.zeros(0, dtype=np.uint64) if exclude is None else exclude
    while have < n:
        tok, tgt, length, m, i, r = _draw(rng, max(4 * (n - have), 256), t, v)
        keep = np.ones(len(tok), dtype=bool)
        if forbid is not None:
            pos = np.arange(t - 1)[None]
            adjacent = (pos >= 1) & (pos <= length[:, None] - 2)    # pair (pos, pos + 1) lies in the sequence
            keep &= ~(forbid[tok[:, :-1], tok[:, 1:]] & adjacent).any(1)
        if queries is not None:
            keep &= (queries[tok, np.where(tgt != IGNORE, tgt, 0)] | (tgt == IGNORE)).all(1)
        h = row_hash(tok)
        keep &= ~np.isin(h, seen)
        if unique:
            first = np.zeros(len(h), dtype=bool)
            first[np.unique(np.where(keep, h, 0), return_index=True)[1]] = True
            keep &= first
        out.append((tok[keep], tgt[keep], length[keep]))
        if exclude is not None or unique:
            seen = np.concatenate([seen, h[keep]])
        have += int(keep.sum())
    tok, tgt, length = (np.concatenate(x)[:n] for x in zip(*out))
    return tok, tgt, length


_CACHE: dict = {}


def test_sets(t: int = T, v: int = V) -> dict:
    """The fixed evaluation sets: "test" (random sequences), "unseen" (queried pairs in HELD_PAIRS), "dev" (model
    selection, drawn like training data). Each is a dict of tok, tgt, length; no sequence twice within or across sets.
    Same arrays on every call."""
    key = (t, v)
    if key not in _CACHE:
        held, sets, used = held_pairs(v), {}, np.zeros(0, dtype=np.uint64)
        for name, n, kw in (("test", N_TEST, {}), ("unseen", N_UNSEEN, dict(queries=held)),
                            ("dev", N_DEV, dict(forbid=held))):
            tok, tgt, length = sample(np.random.default_rng(SEEDS[name]), n, t, v, exclude=used, unique=True, **kw)
            sets[name] = dict(tok=tok, tgt=tgt, length=length)
            used = np.concatenate([used, row_hash(tok)])
        _CACHE[key] = sets
    return _CACHE[key]


def excluded(t: int = T, v: int = V) -> np.ndarray:
    """Row hashes of every held-out sequence (test, unseen, dev): the training stream never emits one."""
    return np.concatenate([row_hash(s["tok"]) for s in test_sets(t, v).values()])


def train_batch(rng, n: int, t: int = T, v: int = V):
    """n fresh training sequences: no held pair next to each other, none in a held-out set."""
    return sample(rng, n, t, v, forbid=held_pairs(v), exclude=excluded(t, v))


def rule(tokens) -> list[int | None]:
    """The task itself: for every position, the token that followed the first occurrence of the current token,
    or None where the current token is the first occurrence (or BOS)."""
    first: dict[int, int] = {}
    out: list[int | None] = []
    for t, x in enumerate(tokens):
        out.append(tokens[first[x] + 1] if x in first and first[x] + 1 < len(tokens) else None)
        if x != 0 and x not in first:
            first[x] = t
    return out


def answer(tokens) -> int | None:
    """The correct next token after `tokens` (the rule at the last position), or None when the last token is new."""
    t = list(tokens)
    first = {x: i for i, x in reversed(list(enumerate(t[:-1]))) if x != 0}
    return t[first[t[-1]] + 1] if t[-1] in first else None


def encode(text: str) -> list[int]:
    """Token ids of a string of letters A..P with BOS in front."""
    bad = sorted({c for c in text.upper() if c not in VOCAB[1:]})
    if bad:
        raise ValueError(f"tokens must be letters A..P, not {bad}")
    return [0] + [VOCAB.index(c) for c in text.upper()]


def decode(tokens) -> str:
    return "".join(VOCAB[t] for t in tokens if t)


# ----------------------------------------------------------------------------- trivial baselines
def _ctx_keys(tok: np.ndarray, n: int, v: int) -> np.ndarray:
    """Radix key of the last n tokens ending at every position of tok [N, T] (BOS before the sequence)."""
    key = np.zeros_like(tok)
    for j in range(n):
        back = np.zeros_like(tok)
        back[:, j:] = tok[:, : tok.shape[1] - j]
        key = key * (v + 1) + back
    return key


def baselines(train, test: dict, v: int = V) -> dict:
    """Accuracy of count tables and heuristics on `test` (dict tok, tgt). The tables are fit on the supervised
    positions of `train` (tok, tgt), the data most favourable to them, and back off to lower orders for unseen
    contexts. Returns name -> (accuracy over all queried tokens, accuracy at the first queried token of a sequence)."""
    ttok, ttgt = train
    stok, sgt = test["tok"], test["tgt"]
    sup, tsup = sgt != IGNORE, ttgt != IGNORE
    first = np.zeros_like(sup)
    first[np.arange(len(sup)), sup.argmax(1)] = True
    first &= sup
    res = {}

    def report(name, pred):
        res[name] = (float((pred[sup] == sgt[sup]).mean()), float((pred[first] == sgt[first]).mean()))

    pred = np.full(stok.shape, int(np.argmax(np.bincount(ttgt[tsup], minlength=v + 1))))
    names = {1: "unigram", 2: "bigram table", 3: "trigram table", 4: "4-gram table"}
    for n in range(1, 5):
        if n > 1:
            kt, yt = _ctx_keys(ttok, n - 1, v)[tsup], ttgt[tsup]
            up, cnt = np.unique(kt * (v + 1) + yt, return_counts=True)
            ks, ys = up // (v + 1), up % (v + 1)
            order = np.lexsort((-cnt, ks))
            keys, best = np.unique(ks[order], return_index=True)
            table = ys[order][best]
            ks_test = _ctx_keys(stok, n - 1, v)
            at = np.clip(np.searchsorted(keys, ks_test), 0, len(keys) - 1)
            pred = np.where(keys[at] == ks_test, table[at], pred)
        report(names[n], pred)
    best = (-1.0, 1)
    for lag in range(1, stok.shape[1]):                              # the token `lag` positions back
        shifted = np.zeros_like(stok)
        shifted[:, lag - 1:] = stok[:, : stok.shape[1] - lag + 1]
        acc = float((shifted[sup] == sgt[sup]).mean())
        best = max(best, (acc, lag))
    lag = best[1]
    shifted = np.zeros_like(stok)
    shifted[:, lag - 1:] = stok[:, : stok.shape[1] - lag + 1]
    report(f"copy the token {lag} back (best fixed lag)", shifted)
    seen = np.array([[len(set(row[1: t + 1].tolist())) for t in range(len(row))] for row in stok])
    uniform = 1.0 / np.maximum(seen - 1, 1)                           # a token seen before, not the current one
    res["guess a token already seen (expected)"] = (float(uniform[sup].mean()), float(uniform[first].mean()))
    report("the rule (look back, exact)", np.array([[x or 0 for x in rule(row.tolist())] for row in stok]))
    return res
