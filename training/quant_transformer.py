"""Char-level transformer through `Transformer.from_torch`: train with the staircase recipe, lower, compare the
perplexity float / staircase / integer with n-gram baselines on the same held-out names, run held-out names in DOOM.

Corpus: training/data/levels.json (`clean` names of the DOOM family's levels; training/train_brain.py's fixed split:
10 % `val` touched only for reporting, `dev` for checkpoint selection, the rest `train`). Every number is bits per
character over the characters of a name plus its end marker, teacher forced, so perplexity = 2 ** bits.

  python training/quant_transformer.py train --name relu_none --attn relu --norm none --mlp relu
  python training/quant_transformer.py train --name soft_rms_reglu --attn softmax --norm rms --Rn 4 --mlp reglu
  python training/quant_transformer.py report --name relu_none            # perplexities, n-grams, exactness check
  python training/quant_transformer.py doom --name relu_none --n 3 --max-chars 6 --workers 3

Writes models/quant/transformer/<name>/ (QModel dir + info.json). Needs the `train` extra.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))

import train_brain as tb  # noqa: E402

import doomformer as df  # noqa: E402
from doomformer import intref, torch_models as tm  # noqa: E402

OUT = HERE.parent / "models" / "quant" / "transformer"
LN2 = math.log(2.0)


def pack(corp, names):
    tok, tgt = tb.make_batch(corp["vocab"], names, corp["T"])
    if corp.get("no_end"):                      # windows of running text: no end marker to predict
        tgt[:, -1] = tm.IGNORE
    return tok, tgt, tgt != tm.IGNORE


PYDOC_VOCAB = "$ abcdefghijklmnopqrstuvwxyz.,()"


def load_pydoc(seed: int = 1234, window: int = 23, stride: int = 6, n_dev: int = 300, n_val: int = 1500):
    """Second corpus: the reference prose of Python's `pydoc_data.topics` (stdlib, offline), lower-cased, other characters
    become spaces. Topics are split 69 / 4 / 8 into train / dev / val; windows are `window` characters (the model reads
    BOS + window). dev and val windows are kept only if their characters occur NOWHERE in the train text, so no n-gram
    or network is scored on text it has seen (topics repeat each other)."""
    import random
    import re
    from pydoc_data.topics import topics
    keys = sorted(topics)
    random.Random(seed).shuffle(keys)
    groups = dict(val=keys[:8], dev=keys[8:12], train=keys[12:])
    clean = lambda t: re.sub(" +", " ", re.sub(r"[^a-z.,() ]", " ", t.lower())).strip()      # noqa: E731
    text = {k: [clean(topics[x]) for x in v] for k, v in groups.items()}
    train_text = " ".join(text["train"])
    seen = {train_text[i:i + window] for i in range(len(train_text) - window + 1)}
    wins = lambda docs, step: [w for t in docs for w in (t[i:i + window] for i in range(0, len(t) - window + 1, step))]  # noqa: E731
    novel = lambda ws: [w for w in ws if w not in seen]                                    # noqa: E731
    rnd = random.Random(seed)
    dev, val = novel(wins(text["dev"], window)), novel(wins(text["val"], window))
    rnd.shuffle(dev)
    rnd.shuffle(val)
    return dict(vocab=PYDOC_VOCAB, T=window + 1, train=wins(text["train"], stride), dev=dev[:n_dev], val=val[:n_val],
                names=[], no_end=True, source="pydoc_data.topics (novel windows)")


def get_corpus(a):
    return load_pydoc() if a.corpus == "pydoc" else tb.load_corpus(tb.LEVELS, a.dev)


# ----------------------------------------------------------------------------- n-gram baselines
class NGram:
    """Interpolated Witten-Bell n-gram over the symbols of `<BOS> name` predicting the next symbol (the last is the end
    marker 0): order n conditions on the previous n - 1 symbols, BOS included, and falls back to shorter contexts."""

    def __init__(self, order: int, vocab: int, names, encode, end: bool = True):
        self.n, self.V, self.end = order, vocab, end
        self.counts = [defaultdict(Counter) for _ in range(order)]      # counts[k][k previous symbols][next symbol]
        for name in names:
            seq = encode(name)
            tgt = seq[1:] + ([0] if end else [])
            for t in range(len(tgt)):
                for k in range(min(order, t + 2)):
                    self.counts[k][tuple(seq[t + 1 - k:t + 1])][tgt[t]] += 1

    def prob(self, ctx: tuple, y: int) -> float:
        p = 1.0 / self.V
        for k in range(min(self.n, len(ctx) + 1)):
            c = self.counts[k].get(ctx[len(ctx) - k:] if k else ())
            if not c:
                break
            tot, types = sum(c.values()), len(c)
            lam = tot / (tot + types)
            p = lam * c.get(y, 0) / tot + (1 - lam) * p
        return p

    def bits(self, names, encode) -> float:
        nll = cnt = 0
        for name in names:
            seq = encode(name)
            tgt = seq[1:] + ([0] if self.end else [])
            for t in range(len(tgt)):
                nll -= math.log2(self.prob(tuple(seq[max(0, t + 2 - self.n):t + 1]), tgt[t]))
                cnt += 1
        return nll / cnt


def ngram_table(corp, train_names, val_names, orders=(1, 2, 3, 4, 5, 6, 8)):
    enc = lambda n: tb.encode(corp["vocab"], n)       # noqa: E731
    out = {}
    for o in orders:
        g = NGram(o, len(corp["vocab"]), train_names, enc, end=not corp.get("no_end"))
        out[o] = g.bits(val_names, enc)
    return out


# ----------------------------------------------------------------------------- evaluation
@torch.no_grad()
def bits(net, data, quant: bool, hard: bool) -> float:
    """Cross-entropy in bits per character of a GPT in the given mode (float stage / act-QAT / integer)."""
    saved = (net.quant, net.hard, net.training)
    net.quant, net.hard = quant, hard
    net.eval()
    tok, tgt, valid = data
    nll = cnt = 0.0
    for i in range(0, len(tok), 512):
        t, g, v = tok[i:i + 512], tgt[i:i + 512], valid[i:i + 512]
        n = int(v.sum(1).max())
        ce, _, _, _ = tm.loss_fn(net, t[:, :n], g[:, :n], v[:, :n])
        nll += ce.item() * int(v.sum())
        cnt += int(v.sum())
    net.quant, net.hard = saved[0], saved[1]
    net.train(saved[2])
    return nll / cnt / LN2


def train_float_gpt(corp, train, dev, args, log):
    """Plain float GPT of the same shape (no staircases), Adam, best dev checkpoint."""
    torch.manual_seed(args.seed)
    m = tm.FloatGPT(len(corp["vocab"]), corp["T"], args.d, args.d // 8, args.dm, args.L)
    opt = torch.optim.AdamW(m.parameters(), 3e-3, weight_decay=0.01)
    steps = args.float_steps
    best, state = 1e9, None
    tok, tgt, valid = train
    for step in range(steps):
        for g in opt.param_groups:
            g["lr"] = 3e-3 * (0.05 + 0.95 * 0.5 * (1 + math.cos(math.pi * step / steps)))
        idx = torch.randint(0, len(tok), (args.bs,))
        v = valid[idx]
        n = int(v.sum(1).max())
        out = m(tok[idx][:, :n])
        loss = F.cross_entropy(out.reshape(-1, m.V), tgt[idx][:, :n].reshape(-1), ignore_index=tm.IGNORE)
        opt.zero_grad()
        loss.backward()
        opt.step()
        if step % 100 == 0 or step == steps - 1:
            with torch.no_grad():
                dv = float_bits(m, dev)
            if dv < best:
                best, state = dv, {k: v.clone() for k, v in m.state_dict().items()}
            if step % 500 == 0:
                log(f"[plain float] step {step} loss {loss.item() / LN2:.3f} bits, dev {dv:.3f} bits")
    m.load_state_dict(state)
    return m


@torch.no_grad()
def float_bits(m, data) -> float:
    tok, tgt, valid = data
    m.eval()
    n = int(valid.sum(1).max())
    out = m(tok[:, :n])
    ce = F.cross_entropy(out.reshape(-1, m.V), tgt[:, :n].reshape(-1), ignore_index=tm.IGNORE, reduction="sum")
    m.train()
    return ce.item() / int(valid.sum()) / LN2


# ----------------------------------------------------------------------------- commands
def make_net(corp, a):
    return tm.GPT(len(corp["vocab"]), corp["T"], a.d, a.d // 8, a.dm, a.K, a.KA, a.KM, a.L, a.S, a.seed, 0.0,
                  attn=a.attn, R=a.R, norm=a.norm, Rn=a.Rn, mlp=a.mlp)


def cmd_train(a):
    torch.set_num_threads(a.threads)
    torch.manual_seed(a.seed)
    corp = get_corpus(a)
    train, dev = pack(corp, corp["train"]), pack(corp, corp["dev"])
    out = OUT / a.name
    t0 = time.time()

    def log(s):
        print(f"{time.time() - t0:7.1f}s {s}", flush=True)

    net = make_net(corp, a)
    log(f"{corp['source']}: train {len(corp['train'])} dev {len(corp['dev'])} val {len(corp['val'])} names, T={corp['T']}, "
        f"vocab {len(corp['vocab'])}, params {sum(p.numel() for p in net.parameters())}")
    res = tm.fit(net, train, dev, steps=(a.steps1, a.steps2, a.steps3), lrs=(a.lr1, a.lr2, a.lr3), bs=a.bs, alpha=a.alpha,
                 log=log, seed=a.seed)
    qm = df.Transformer.from_torch(net, corp["vocab"])
    qm.save(out)
    plain = train_float_gpt(corp, train, dev, a, log) if a.float_steps else None
    if plain is not None:
        torch.save(plain.state_dict(), out / "plain_float.pt")
    info = dict(args=tb.record_args(a), fit=res, int_logit_scale=float(net.lg.exp()), wall_s=time.time() - t0,
                n_train=len(corp["train"]), n_dev=len(corp["dev"]), n_val=len(corp["val"]), split_seed=tb.SPLIT_SEED,
                params=sum(p.numel() for p in net.parameters()))
    (out / "info.json").write_text(json.dumps(info, indent=1, default=float))
    tm.save(net, out / "gpt.pt")
    log(f"saved {tb.repo_rel(out)}")


def load_net(corp, a):
    info = json.loads((OUT / a.name / "info.json").read_text())
    return tm.load(OUT / a.name / "gpt.pt"), argparse.Namespace(**info["args"]), info


def cmd_report(a):
    torch.set_num_threads(a.threads)
    a.corpus = json.loads((OUT / a.name / "info.json").read_text())["args"].get("corpus", "levels")
    corp = get_corpus(a)
    net, args, info = load_net(corp, a)
    val = pack(corp, corp["val"])
    res = dict(name=a.name, params=info["params"], n_val=len(corp["val"]), n_chars=int(val[2].sum()))
    res["float_stage_bits"] = bits(net, val, False, False)
    res["act_qat_bits"] = bits(net, val, False, True)
    res["integer_bits"] = bits(net, val, True, True)
    if (OUT / a.name / "plain_float.pt").exists():
        plain = tm.FloatGPT(len(corp["vocab"]), corp["T"], args.d, args.d // 8, args.dm, args.L)
        plain.load_state_dict(torch.load(OUT / a.name / "plain_float.pt"))
        res["plain_float_bits"] = float_bits(plain, val)
    res["ngram_bits"] = {str(o): b for o, b in ngram_table(corp, corp["train"], corp["val"]).items()}
    # exactness: the lowered QModel's integer forward equals the torch integer forward on held-out names
    qm = df.Transformer.load(OUT / a.name)
    same = tot = 0
    with torch.no_grad():
        for name in corp["val"][:100]:
            tok = torch.tensor([tb.encode(corp["vocab"], name)])
            lg, _, _ = net(tok)
            ref = intref.forward(qm, tok[0].tolist())
            same += int(np.array_equal(lg[0].numpy().astype(np.int64), ref))
            tot += 1
    res["qmodel_equals_torch_integer_forward"] = f"{same}/{tot} names"
    res["integer_acc"] = float(tm.evaluate(net, val)[0])
    for k in ("float_stage_bits", "act_qat_bits", "integer_bits", "plain_float_bits"):
        if k in res:
            res[k.replace("_bits", "_ppl")] = 2 ** res[k]
    res["gap_integer_vs_float_stage_ppl_pct"] = 100 * (res["integer_ppl"] / res["float_stage_ppl"] - 1)
    if "plain_float_ppl" in res:
        res["gap_integer_vs_plain_float_ppl_pct"] = 100 * (res["integer_ppl"] / res["plain_float_ppl"] - 1)
    best_ng = min(res["ngram_bits"].values())
    res["best_ngram_order"] = int(min(res["ngram_bits"], key=res["ngram_bits"].get))
    res["integer_beats_best_ngram"] = bool(res["integer_bits"] < best_ng)
    (OUT / a.name / "report.json").write_text(json.dumps(res, indent=1) + "\n")
    print(json.dumps(res, indent=1))


def cmd_doom(a):
    a.corpus = json.loads((OUT / a.name / "info.json").read_text())["args"].get("corpus", "levels")
    corp = get_corpus(a)
    qm = df.Transformer.load(OUT / a.name)
    names = [n for n in corp["val"] if len(n) <= a.max_chars][:a.n]
    plan = df.plan(qm, mode="tiles")["tiles"]
    print(f"{a.name}: {qm.L} layers d={qm.d} H={qm.H} {qm.attn} attention, norm {qm.norm}, mlp {qm.mlp}; "
          f"tiles: {plan['dolls']:,} dolls, {plan['sectors']:,} sectors, {plan['tiles']} tile maps", flush=True)
    t0 = time.time()
    rows = []
    with df.compile(qm, mode="tiles", workers=a.workers) as brain:
        print(f"compiled in {time.time() - t0:.0f} s", flush=True)
        for name in names:
            toks = tb.encode(corp["vocab"], name)
            t1 = time.time()
            logits = brain.logits(toks)
            ref = intref.forward(qm, toks)
            exact = bool(np.array_equal(logits, ref))
            rows.append(dict(name=name, tokens=len(toks), exact=exact, values=int(brain.info["checked"]),
                             tile_runs=int(brain.info.get("tile_runs", 0)), seconds=round(time.time() - t1, 1),
                             next_char_argmax="".join(corp["vocab"][int(i)] if i else "$" for i in logits.argmax(-1))))
            print(rows[-1], flush=True)
    out = dict(model=a.name, layers=qm.L, d=qm.d, dolls=plan["dolls"], sectors=plan["sectors"], tile_maps=plan["tiles"],
               workers=a.workers, sequences=rows, all_exact=all(r["exact"] for r in rows),
               wall_seconds=round(time.time() - t0, 1))
    (OUT / a.name / "doom.json").write_text(json.dumps(out, indent=1, default=str) + "\n")
    print("EXACT" if out["all_exact"] else "MISMATCH", f"{out['wall_seconds']} s wall (shared machine)")
    return 0 if out["all_exact"] else 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=("train", "report", "doom"))
    ap.add_argument("--name", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--L", type=int, default=2)
    ap.add_argument("--d", type=int, default=32)
    ap.add_argument("--dm", type=int, default=96)
    ap.add_argument("--K", type=int, default=7)
    ap.add_argument("--KA", type=int, default=7)
    ap.add_argument("--KM", type=int, default=7)
    ap.add_argument("--S", type=float, default=128.0)
    ap.add_argument("--bs", type=int, default=256)
    ap.add_argument("--steps1", type=int, default=1500)
    ap.add_argument("--steps2", type=int, default=2000)
    ap.add_argument("--steps3", type=int, default=1500)
    ap.add_argument("--lr1", type=float, default=0.01)
    ap.add_argument("--lr2", type=float, default=0.005)
    ap.add_argument("--lr3", type=float, default=0.002)
    ap.add_argument("--alpha", type=float, default=0.5)
    ap.add_argument("--float-steps", type=int, default=3000, help="plain float GPT baseline steps (0: skip)")
    ap.add_argument("--attn", choices=("relu", "softmax"), default="relu")
    ap.add_argument("--R", type=int, default=1)
    ap.add_argument("--norm", choices=("none", "rms"), default="none")
    ap.add_argument("--Rn", type=int, default=1)
    ap.add_argument("--mlp", choices=("relu", "reglu"), default="relu")
    ap.add_argument("--dev", type=int, default=300)
    ap.add_argument("--corpus", choices=("levels", "pydoc"), default="levels")
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--n", type=int, default=3, help="doom: held-out names to run")
    ap.add_argument("--max-chars", type=int, default=6, help="doom: only held-out names of at most this many characters")
    ap.add_argument("--workers", type=int, default=3)
    a = ap.parse_args()
    torch.set_default_dtype(torch.float32)
    if a.cmd == "train":
        cmd_train(a)
    elif a.cmd == "report":
        cmd_report(a)
    else:
        raise SystemExit(cmd_doom(a))


if __name__ == "__main__":
    main()
