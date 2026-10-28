"""Induction in DOOM: a transformer that learned to look back, run on voodoo dolls, every value checked.

    python -m demos.induction                      # one held-out sequence
    python -m demos.induction KFCAPJDC             # your sequence (letters A..P, at most 9): what follows C?
    python -m demos.induction --heldout 25         # the first 25 held-out sequences
    python -m demos.induction --set unseen         # sequences whose queried token pairs never occur in training

The model (`models/induction`, 2 layers, d=32) was trained on random sequences: a prefix of distinct tokens, then a
repeat of a run of it ($ KFCAPJD CAP). At a repeated token it has to answer with the token that followed that token's
first occurrence. No table over tokens can do that, and the held-out sequences are ones it never saw (see
`doomformer/induction.py` and `models/induction/EVAL.md`).

Here DOOM computes the forward pass in operation tiles (`df.compile(net, mode="tiles")`, `--workers` game processes,
Python copies the integers between games). Every value DOOM exposes is compared with `intref.forward` at tolerance 0,
and the reference says what the rule says: the last printed column. `--mode map` puts a model into one map (only a
tiny model fits, see docs/DEMOS.md).
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np

from doomformer import induction as I
from doomformer import intref
from doomformer.orchestrator import DoomTransformer

from . import say


class SharedGames(DoomTransformer):
    """DoomTransformer whose tile runs share games. The stock engine gives every head, key slot and output unit its
    own logical game (`wsum_t3_h1_e5`), and with one live game per worker each run then starts a new game. Here the
    tasks of a stage that run the same tile are dealt over `workers` logical games (`L0.wsum_3#0`, `#1`, `#2`), so a
    worker runs its share of the stage on one game without restarting it. The same tiles run on the same inputs and
    every output is still checked against intref at tolerance 0."""

    def _stage(self, label, tasks):
        dealt, out = {}, []
        for _, tile, inputs in tasks:
            k = dealt.get(tile, 0)
            dealt[tile] = k + 1
            out.append((f"{tile}#{k % self.workers}", tile, inputs))
        return super()._stage(label, out)


def share_prefix(brain, tokens: list[int]) -> int:
    """Tile mode: keep the DOOM-computed positions that `tokens` shares with the previous sequence (causal attention:
    a position depends only on the tokens up to it), so only the new positions run. Returns how many are kept."""
    engine, old = getattr(brain, "_engine", None), getattr(brain, "_tokens", [])
    k = 0
    while k < min(len(old), len(tokens)) and old[k] == tokens[k]:
        k += 1
    if engine is not None and 0 < k < len(old):
        del engine.pos[k:]
        engine.score = {key: v for key, v in engine.score.items() if key[2] < k}
        brain._tokens = old[:k]
    return k


def queried(tokens: list[int]) -> list[int]:
    """Positions of the sequence at which the rule has an answer (the token occurred before)."""
    return [t for t, x in enumerate(I.rule(tokens)) if x is not None]


def line(tokens: list[int], logits: np.ndarray) -> tuple[str, int, int]:
    """One result line: the sequence, then DOOM's prediction at every queried position against the rule."""
    rule = I.rule(tokens)
    pos = queried(tokens)
    pred = {t: int(np.argmax(logits[t])) for t in pos}
    hit = sum(pred[t] == rule[t] for t in pos)
    shown = " ".join(I.VOCAB[x] if x else "$" for x in tokens)
    asks = "  ".join(f"{I.VOCAB[tokens[t]]} -> {I.VOCAB[pred[t]]}" + ("" if pred[t] == rule[t] else f" (rule {I.VOCAB[rule[t]]})")
                     for t in pos) or "no repeated token"
    return f"{shown:<22} DOOM answers  {asks}", hit, len(pos)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("sequence", nargs="?", help="letters A..P (no BOS); omit to use held-out sequences")
    ap.add_argument("--model", default="induction", help="a shipped model name or a model directory")
    ap.add_argument("--heldout", type=int, default=0, metavar="N", help="run the first N held-out sequences")
    ap.add_argument("--set", choices=("test", "unseen"), default="test", help="which held-out set")
    ap.add_argument("--mode", choices=("tiles", "map"), default="tiles")
    ap.add_argument("--workers", type=int, default=3, help="DOOM game processes in tile mode (default 3)")
    a = ap.parse_args(argv)
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    import doomformer as df

    net = df.Transformer.load(a.model)
    if a.sequence:
        try:
            seqs = [I.encode(a.sequence)]
        except ValueError as e:
            raise SystemExit(f"induction: {e}")
        if len(seqs[0]) > net.T:
            raise SystemExit(f"induction: this model reads at most {net.T - 1} tokens")
    else:
        s = I.test_sets(net.T)[a.set]
        seqs = [s["tok"][i, : s["length"][i]].tolist() for i in range(max(a.heldout, 1))]
    plan = df.plan(net, mode="tiles" if a.mode == "tiles" else "map")
    where = plan["tiles"] if a.mode == "tiles" else plan["map"]
    say(f"model {a.model}: {net.L} layers, d={net.d}, {net.H} heads, {net.attn} attention, K={net.K}, context {net.T}; "
        f"{len(seqs)} sequence{'s' if len(seqs) != 1 else ''}"
        + (f" (held-out set '{a.set}')" if not a.sequence else ""))
    say(f"DOOM: {where['dolls']:,} voodoo dolls, {where['sectors']:,} sectors, "
        + (f"{where['tiles']} tile maps, {a.workers} games at a time" if a.mode == "tiles" else "one map"))
    t0 = time.time()
    exact = right = total = values = 0
    with df.compile(net, mode=a.mode, **({"workers": a.workers} if a.mode == "tiles" else {})) as brain:
        say(f"compiled in {time.time() - t0:.0f} s")
        if a.mode == "tiles":
            brain._engine = SharedGames(net, brain.wad, workers=a.workers, max_live=1)
        for n, tokens in enumerate(seqs, 1):
            t1 = time.time()
            runs0 = brain.info.get("tile_runs", 0)
            if a.mode == "tiles":
                share_prefix(brain, tokens)
            try:
                logits = brain.logits(tokens)
            except df.MismatchError as e:
                say(f"MISMATCH on {I.decode(tokens)}: {e}")
                return 1
            text, hit, nq = line(tokens, logits)
            ref = intref.forward(net, tokens)
            same = np.array_equal(logits, ref)
            exact += same
            right += hit
            total += nq
            values += brain.info["checked"]
            last = I.answer(tokens)
            runs = f"{brain.info.get('tile_runs', 0) - runs0:,} tile runs, " if a.mode == "tiles" else ""
            say(f"[{n:>2}/{len(seqs)}] {text}   {'EXACT' if same else 'MISMATCH'} ({brain.info['checked']:,} values, "
                f"{runs}{time.time() - t1:.0f} s)")
            if a.sequence:
                say(f"next token after {I.decode(tokens)}: DOOM says {I.VOCAB[int(np.argmax(logits[-1]))]}"
                    + (f", the rule says {I.VOCAB[last]}" if last is not None else
                       " (the last token is new, the rule has no answer)"))
        info = dict(brain.info)
    say(f"{exact}/{len(seqs)} sequences exact; DOOM's answer equals the rule at {right}/{total} queried tokens; "
        f"{values:,} values checked at tolerance 0; "
        + (f"{info.get('tile_runs', 0):,} tile runs; " if a.mode == "tiles" else "")
        + f"{time.time() - t0:.0f} s wall (shared machine)")
    say("EXACT: every value DOOM exposed equals the NumPy integer reference" if exact == len(seqs) else "MISMATCH")
    return 0 if exact == len(seqs) else 1


if __name__ == "__main__":
    sys.exit(main())
