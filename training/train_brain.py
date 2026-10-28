"""Quantisation-aware training of the bigger DOOM-name transformer ("doombrain", L<=2, d<=32, H=d/8).

Adapts train_qat.py to the L-layer QModel format ("L{l}.<name>"), the real-level-name corpus
(training/data/levels.json, field `clean`) and mini-batch training.  The torch model mirrors
`doomformer.intref.forward` exactly; every tensor is an integer-valued float64 tensor when
`quant=True, hard=True`, so the same code is the integer model.

Stages (as in train_qat): 1 float (smooth staircases) -> 2 act-QAT (hard staircases, float weights)
-> 3 full QAT (weights, biases, thresholds rounded, straight-through).  Thresholds are cumulative sums
of positive increments (positive and nondecreasing by construction) calibrated from activation quantiles.

Split (fixed seed): 10 % of the (deduplicated) names are `val` (touched only for reporting), `dev`
names (taken from the rest) select checkpoints / hyper-parameters, the rest is `train`.

--attn softmax (intref attn="softmax") as in train_qat.py: e = adc_relu(q.k, thr_a) with thr_a = c + sigma*ln(k+1)
per layer (c, sigma > 0 learned, calibrated at the median and 97th percentile of the positive scores), den = 1 + sum e,
ao = trunc(R * sum e v / den), no thr_o; the backward pass (and the float stage) uses R * sum e v / den.

--norm rms --Rn N and --mlp reglu (intref norm="rms", mlp="reglu") as in train_qat.py: rmsnorm_ste before q/k/v and
before the MLP, m = relu-staircase(Wg x + bg) * signed-staircase(Wu x + bu) in place of the relu MLP.

Usage: python training/train_brain.py --out models/doombrain [--seed 0] [--L 2 --d 32 --dm 96] [--attn softmax --R 1]
       [--norm rms --Rn 4] [--mlp reglu]
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))

from doomformer import intref  # noqa: E402,F401

from doomformer.torch_models import (EXP_C_Q, FLIM, IGNORE, PLIM, PORT_MAX, W_MAX, Z_MAX, Net, evaluate,  # noqa: E402,F401
                                     int_forward, inv_softplus, loss_fn, rmsnorm_ste, round_ste, run_stage)
BASE_VOCAB = "$ ABCDEFGHIJKLMNOPQRSTUVWXYZ'!"
LEVELS = HERE / "data" / "levels.json"
SPLIT_SEED = 12345
ROOT = HERE.parent


def repo_rel(path) -> str:
    """`path` relative to the repository root when it lies inside it, so run records hold no local absolute paths."""
    try:
        return Path(path).resolve().relative_to(ROOT).as_posix()
    except ValueError:
        return str(path)


def record_args(a) -> dict:
    """`vars(a)` for train_info.json, with absolute paths inside the repository made repo-relative."""
    return {k: repo_rel(v) if isinstance(v, str) and Path(v).is_absolute() else v for k, v in vars(a).items()}


# ----------------------------------------------------------------------------- data
def load_corpus(path: Path = LEVELS, dev: int = 150, seed: int = SPLIT_SEED):
    """-> dict(names, vocab, train, dev, val, T).  Deterministic, deduplicated, 10 % val."""
    names = None
    if Path(path).exists():
        rows = json.loads(Path(path).read_text())
        names = sorted({r["clean"] for r in rows if r.get("clean")})
        source = repo_rel(path)
    if names is None:
        from data import NAMES
        names, source = sorted(set(NAMES)), "training/data.py NAMES"
    extra = sorted({c for n in names for c in n} - set(BASE_VOCAB))
    vocab = BASE_VOCAB + "".join(extra)
    assert len(vocab) <= 32, vocab
    rnd = random.Random(seed)
    order = names[:]
    rnd.shuffle(order)
    nval = round(0.1 * len(order))
    val, rest = order[:nval], order[nval:]
    dev = min(dev, len(rest) // 8)
    return dict(names=names, vocab=vocab, val=val, dev=rest[:dev], train=rest[dev:],
                T=max(len(n) for n in names) + 1, source=source)


def encode(vocab: str, name: str) -> list[int]:
    return [0] + [vocab.index(c) for c in name]


def encode_target(vocab: str, name: str) -> list[int]:
    return [vocab.index(c) for c in name] + [0]


def make_batch(vocab: str, names: list[str], T: int):
    tok = torch.zeros(len(names), T, dtype=torch.long)
    tgt = torch.full((len(names), T), IGNORE, dtype=torch.long)
    for i, n in enumerate(names):
        x, y = encode(vocab, n), encode_target(vocab, n)
        tok[i, : len(x)] = torch.tensor(x)
        tgt[i, : len(y)] = torch.tensor(y)
    return tok, tgt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(HERE.parent / "models" / "doombrain"))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--L", type=int, default=2)
    ap.add_argument("--d", type=int, default=32)
    ap.add_argument("--dm", type=int, default=96)
    ap.add_argument("--K", type=int, default=7)
    ap.add_argument("--KA", type=int, default=7)
    ap.add_argument("--KM", type=int, default=7)
    ap.add_argument("--S", type=float, default=128.0)
    ap.add_argument("--bs", type=int, default=256)
    ap.add_argument("--steps1", type=int, default=3000)
    ap.add_argument("--steps2", type=int, default=2000)
    ap.add_argument("--steps3", type=int, default=3000)
    ap.add_argument("--lr1", type=float, default=0.01)
    ap.add_argument("--lr2", type=float, default=0.005)
    ap.add_argument("--lr3", type=float, default=0.002)
    ap.add_argument("--wd", type=float, default=0.0, help="decoupled weight decay on float weights")
    ap.add_argument("--drop", type=float, default=0.0, help="activation dropout (train only)")
    ap.add_argument("--alpha", type=float, default=0.5, help="surrogate sigmoid width / threshold step")
    ap.add_argument("--dev", type=int, default=300)
    ap.add_argument("--recal", action="store_true", help="re-calibrate thresholds after the float stage")
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--dtype", default="float32", choices=["float32", "float64"],
                    help="exact for the integer stages: every integer-valued tensor stays far below 2^24")
    ap.add_argument("--corpus", default=str(LEVELS))
    ap.add_argument("--attn", choices=("relu", "softmax"), default="relu",
                    help="relu: ReLU staircase weights, fixed 1/(t+1) in thr_o; softmax: exp staircase, ao = trunc(R num/den)")
    ap.add_argument("--R", type=int, default=1, help="softmax only: output resolution, ao in [-(R*K-1), R*K-1]")
    ap.add_argument("--norm", choices=("none", "rms"), default="none",
                    help="rms: integer RMSNorm before q/k/v and the MLP, n = trunc(Rn c / (1 + floor(sqrt(s/d))))")
    ap.add_argument("--Rn", type=int, default=1, help="rms only: output resolution of the norm")
    ap.add_argument("--mlp", choices=("relu", "reglu"), default="relu",
                    help="reglu: m = relu-staircase(Wg x + bg) * signed-staircase(Wu x + bu)")
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    torch.set_default_dtype(getattr(torch, a.dtype))
    torch.manual_seed(a.seed)
    H = a.d // 8
    corp = load_corpus(Path(a.corpus), a.dev)
    vocab, T = corp["vocab"], corp["T"]
    assert a.L <= 2 and a.d <= 32 and a.d % 8 == 0 and a.dm <= 96 and T <= 24
    assert max(a.K, a.KA, a.KM) <= 7 and len(vocab) <= 32

    t0 = time.time()

    def log(s):
        print(f"{time.time() - t0:7.1f}s {s}", flush=True)

    def pack(names):
        tok, tgt = make_batch(vocab, names, T)
        return tok, tgt, tgt != IGNORE

    train, dev = pack(corp["train"]), pack(corp["dev"])
    net = Net(len(vocab), T, a.d, H, a.dm, a.K, a.KA, a.KM, a.L, a.S, a.seed, a.drop, attn=a.attn, R=a.R, norm=a.norm,
              Rn=a.Rn, mlp=a.mlp)
    log(f"corpus {corp['source']}: {len(corp['names'])} names; train {len(corp['train'])} dev {len(corp['dev'])} "
        f"val {len(corp['val'])}; T={T} vocab {len(vocab)}; params {sum(p.numel() for p in net.parameters())}")

    def calibrate():
        net.quant, net.hard, net.calibrating = False, True, True
        sub = torch.randperm(len(train[0]))[:600]
        tk, _, vd = (x[sub] for x in train)
        net.valid = vd
        net.eval()
        with torch.no_grad():
            net(tk, vd)
        net.calibrating = False

    calibrate()
    run_stage(net, "float", a.steps1, a.lr1, False, False, a.alpha, train, dev, a.bs, log, wd=a.wd)
    net.quant, net.hard = False, False
    net.eval()
    with torch.no_grad():
        lg, _, _ = net(dev[0], dev[2])
    facc = (lg.argmax(-1)[dev[2]] == dev[1][dev[2]]).double().mean().item()
    log(f"float model (smooth ADC, unrounded) dev teacher-forced acc {facc:.4f}")
    if a.recal:
        calibrate()
    run_stage(net, "act-qat", a.steps2, a.lr2, False, True, a.alpha, train, dev, a.bs, log, wd=a.wd)
    run_stage(net, "full-qat", a.steps3, a.lr3, True, True, a.alpha, train, dev, a.bs, log, wd=a.wd, keep_best=True)

    qm = net.qmodel(vocab)
    out = Path(a.out)
    qm.save(out)
    acc, seq, ice, worst = evaluate(net, dev)
    tacc, _, _, _ = evaluate(net, train)
    (out / "train_info.json").write_text(json.dumps(
        dict(float_dev_acc=facc, dev_int_acc=acc, train_int_acc=tacc, seed=a.seed, wall_s=time.time() - t0,
             int_logit_scale=float(net.lg.exp()), args=record_args(a), split_seed=SPLIT_SEED,
             n_train=len(corp["train"]), n_dev=len(corp["dev"]), n_val=len(corp["val"]), **net.arch_info(),
             **net.attn_info()), indent=1))
    log(f"final torch-int train acc {tacc:.4f} dev acc {acc:.4f} ce {ice:.4f} worst partial sums {worst}")
    log(f"saved {repo_rel(a.out)}")


if __name__ == "__main__":
    main()
