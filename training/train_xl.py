"""Quantisation-aware training of the biggest practical DOOM brain ("doombrain_xl": L<=4, d<=64, dm<=256).

Same recipe, split (SPLIT_SEED, `load_corpus`) and torch model as `train_brain.py` (imported, not copied); this
script only lifts the size asserts and makes every stage resumable (checkpoint every `--ckpt-every` steps),
because a run takes ~2 h.  Usage:

  OMP_NUM_THREADS=4 nice -n 10 python training/train_xl.py --out models/doombrain_xl \
      --L 4 --d 64 --dm 256 --steps2 6000 --steps3 2000 --lr2 0.01 --lr3 0.003 --alpha 0.5 --S 64
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import sys
import time
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))

import train_brain as tb  # noqa: E402

STAGES = ("float", "act-qat", "full-qat")


def run_stage(net, name, steps, lr, quant, hard, alpha, train, dev, bs, log, ckpt, ckpt_every, start=0, opt_state=None,
              drop=0.0, wd=0.0, lr_end=0.05, eval_every=100, keep_best=False):
    tok, tgt, valid = train
    net.quant, net.hard, net.alpha, net.drop = quant, hard, alpha, drop
    net.train()
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=0.0)
    if opt_state is not None:
        opt.load_state_dict(opt_state)
    best, best_state = (1e9,), None
    N = len(tok)
    perm, pos = torch.randperm(N), 0
    for step in range(start, steps):
        cur = lr * (lr_end + (1 - lr_end) * 0.5 * (1 + math.cos(math.pi * step / steps)))
        for g in opt.param_groups:
            g["lr"] = cur
        if pos + bs > N:
            perm, pos = torch.randperm(N), 0
        idx = perm[pos:pos + bs]
        pos += bs
        t, g_, v = tok[idx], tgt[idx], valid[idx]
        n = int(v.sum(1).max())
        t, g_, v = t[:, :n], g_[:, :n], v[:, :n]
        ce, pen, _, _ = tb.loss_fn(net, t, g_, v)
        opt.zero_grad()
        (ce + pen).backward()
        opt.step()
        if wd:
            with torch.no_grad():
                for p in net.p.values():
                    p.mul_(1 - cur * wd)
        if step % eval_every == 0 or step == steps - 1:
            acc, seq, ice, worst = tb.evaluate(net, dev)
            if keep_best and ice < best[0]:
                best, best_state = (ice,), copy.deepcopy(net.state_dict()) | {"_units": copy.deepcopy(net.units)}
            if step % (eval_every * 4) == 0 or step == steps - 1:
                log(f"[{name}] step {step:5d} lr {cur:.4f} train-ce {ce.item():.4f} pen {float(pen):.3f} | "
                    f"dev int acc {acc:.4f} ce {ice:.4f} maxz {max(worst.values())}")
        if (step + 1) % ckpt_every == 0 and step + 1 < steps:
            torch.save(dict(stage=name, step=step + 1, state=net.state_dict(), units=net.units, opt=opt.state_dict()),
                       ckpt)
    if keep_best and best_state is not None:
        units = best_state.pop("_units")
        net.load_state_dict(best_state)
        net.units = units
        log(f"[{name}] restored best dev int ce {best[0]:.4f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(HERE.parent / "models" / "doombrain_xl"))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--L", type=int, default=4)
    ap.add_argument("--d", type=int, default=64)
    ap.add_argument("--dm", type=int, default=256)
    ap.add_argument("--K", type=int, default=7)
    ap.add_argument("--KA", type=int, default=7)
    ap.add_argument("--KM", type=int, default=7)
    ap.add_argument("--S", type=float, default=64.0)
    ap.add_argument("--bs", type=int, default=256)
    ap.add_argument("--steps2", type=int, default=6000)
    ap.add_argument("--steps3", type=int, default=2000)
    ap.add_argument("--lr2", type=float, default=0.01)
    ap.add_argument("--lr3", type=float, default=0.003)
    ap.add_argument("--wd", type=float, default=0.0)
    ap.add_argument("--drop2", type=float, default=0.0, help="activation dropout in act-QAT (stage 2) only")
    ap.add_argument("--drop3", type=float, default=0.0, help="activation dropout in full QAT (stage 3)")
    ap.add_argument("--alpha", type=float, default=0.5)
    ap.add_argument("--dev", type=int, default=300)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--ckpt-every", type=int, default=250)
    ap.add_argument("--corpus", default=str(tb.LEVELS))
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    torch.set_default_dtype(torch.float32)
    torch.manual_seed(a.seed)
    H = a.d // 8
    corp = tb.load_corpus(Path(a.corpus), a.dev)
    vocab, T = corp["vocab"], corp["T"]
    assert a.L <= 4 and a.d <= 64 and a.d % 8 == 0 and a.dm <= 256 and T <= 24
    assert max(a.K, a.KA, a.KM) <= 7 and len(vocab) <= 32
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    ckpt = out / "ckpt.pt"
    t0 = time.time()

    def log(s):
        print(f"{time.time() - t0:7.1f}s {s}", flush=True)

    def pack(names):
        tok, tgt = tb.make_batch(vocab, names, T)
        return tok, tgt, tgt != tb.IGNORE

    train, dev = pack(corp["train"]), pack(corp["dev"])
    net = tb.Net(len(vocab), T, a.d, H, a.dm, a.K, a.KA, a.KM, a.L, a.S, a.seed, 0.0)
    log(f"corpus {corp['source']}: {len(corp['names'])} names; train {len(corp['train'])} dev {len(corp['dev'])} "
        f"val {len(corp['val'])}; T={T} vocab {len(vocab)}; params {sum(p.numel() for p in net.parameters())}")

    stage_i, start, opt_state = 1, 0, None   # stage 1 (float) is skipped as in the best recipe (--steps1 0)
    if ckpt.exists():
        ck = torch.load(ckpt, weights_only=False)
        net.load_state_dict(ck["state"])
        net.units = ck["units"]
        stage_i, start, opt_state = STAGES.index(ck["stage"]), ck["step"], ck["opt"]
        log(f"resumed {ck['stage']} at step {start}")
    else:
        net.quant, net.hard, net.calibrating = False, True, True
        sub = torch.randperm(len(train[0]))[:600]
        tk, _, vd = (x[sub] for x in train)
        net.valid = vd
        net.eval()
        with torch.no_grad():
            net(tk, vd)
        net.calibrating = False
        log("calibrated thresholds")

    if stage_i <= 1:
        run_stage(net, "act-qat", a.steps2, a.lr2, False, True, a.alpha, train, dev, a.bs, log, ckpt, a.ckpt_every,
                  start if stage_i == 1 else 0, opt_state if stage_i == 1 else None, drop=a.drop2, wd=a.wd)
        stage_i, start, opt_state = 2, 0, None
        torch.save(dict(stage="full-qat", step=0, state=net.state_dict(), units=net.units, opt=None), ckpt)
        ck = None
    run_stage(net, "full-qat", a.steps3, a.lr3, True, True, a.alpha, train, dev, a.bs, log, ckpt, a.ckpt_every,
              start, opt_state, drop=a.drop3, wd=a.wd, keep_best=True)

    qm = net.qmodel(vocab)
    qm.save(out)
    acc, seq, ice, worst = tb.evaluate(net, dev)
    tacc, _, _, _ = tb.evaluate(net, train)
    (out / "train_info.json").write_text(json.dumps(
        dict(dev_int_acc=acc, train_int_acc=tacc, dev_int_ce=ice, seed=a.seed, wall_s=time.time() - t0,
             int_logit_scale=float(net.lg.exp()), args=tb.record_args(a), split_seed=tb.SPLIT_SEED,
             n_train=len(corp["train"]), n_dev=len(corp["dev"]), n_val=len(corp["val"])), indent=1))
    log(f"final torch-int train acc {tacc:.4f} dev acc {acc:.4f} ce {ice:.4f} worst partial sums {worst}")
    log(f"saved {tb.repo_rel(a.out)}")


if __name__ == "__main__":
    main()
