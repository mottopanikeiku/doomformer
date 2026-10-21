"""Train "doomplayer": an integer transformer (L<=2, d<=32, H=d/8, dm<=64, K<=7) that plays defend_the_center.

1. Data: scripted aimbot rollouts (doomformer.policy.expert_index, exact positions) in ViZDoom.  Every decision
   gives (context tokens = policy.observe, expert action).  Rollouts are executed by the expert with an
   epsilon of random actions (state diversity) and optionally by the learner itself (--dagger rounds: the
   integer model plays, the expert relabels the visited states).
2. Model: train_brain.Net (multi-layer QAT, same recipe: act-QAT then full QAT, thresholds as cumulative sums),
   loss = cross-entropy over the ACTION-token logits at the last position only (the host argmaxes over the
   action tokens, see policy.act).  The unembedding rows of non-action tokens are exported as zeros.
3. Gate: check_limits == [] on every dataset context and on every context observe() can produce; the standalone
   int64 torch forward must equal intref.forward on 100 % of them.

Usage: python training/train_policy.py --out models/doomplayer [--episodes 600] [--dagger 0]
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT))

from doomformer import intref, policy  # noqa: E402
from doomformer.intref import QModel  # noqa: E402
from train_brain import FLIM, PLIM, Net, int_forward, record_args, repo_rel  # noqa: E402

NOISE = (0.0, 0.0, 0.1, 0.25, 0.5)       # epsilon of random actions cycled over the rollout episodes
DATA = HERE / "data" / "doomplayer_dataset.npz"


def run_stage_generic(net, name, steps, lr, quant, hard, alpha, train, bs, log, loss_fn, evaluate_fn, lr_end=0.05,
                      eval_every=200, keep_best=False, dev=None):
    """train_brain.run_stage with a pluggable loss / evaluation (loss on the action logits of the last position)."""
    import copy
    import math
    tok, y, valid = train
    net.quant, net.hard, net.alpha = quant, hard, alpha
    net.train()
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=0.0)
    best, best_state = (1e9,), None
    N = len(tok)
    perm, pos = torch.randperm(N), 0
    for step in range(steps):
        cur = lr * (lr_end + (1 - lr_end) * 0.5 * (1 + math.cos(math.pi * step / steps)))
        for g in opt.param_groups:
            g["lr"] = cur
        if pos + bs > N:
            perm, pos = torch.randperm(N), 0
        idx = perm[pos:pos + bs]
        pos += bs
        ce, pen, _, _ = loss_fn(tok[idx], y[idx], valid[idx])
        opt.zero_grad()
        (ce + pen).backward()
        opt.step()
        if step % eval_every == 0 or step == steps - 1:
            acc, ice, worst = evaluate_fn(dev or train)
            if keep_best and ice < best[0]:
                best, best_state = (ice,), copy.deepcopy(net.state_dict()) | {"_units": copy.deepcopy(net.units)}
            if step % (eval_every * 5) == 0 or step == steps - 1:
                log(f"[{name}] step {step:5d} lr {cur:.4f} train-ce {ce.item():.4f} pen {pen.item():.3f} | "
                    f"{'dev' if dev else 'train'} int acc {acc:.4f} ce {ice:.4f} maxz {max(worst.values())}")
    if keep_best and best_state is not None:
        units = best_state.pop("_units")
        net.load_state_dict(best_state)
        net.units = units
        log(f"[{name}] restored best dev int ce {best[0]:.4f}")


# ----------------------------------------------------------------------------- data collection
def _rollout_worker(args):
    first, count, seed0, noise, model_dir, beta = args
    torch.set_num_threads(1)
    game = policy.make_game(render=False)
    model = QModel.load(model_dir) if model_dir else None
    rows = []
    for e in range(first, first + count):
        rng = np.random.default_rng(seed0 + e)
        eps = noise[e % len(noise)]
        # beta: probability that an episode is driven by the learner (DAgger) instead of the expert
        learner = model is not None and rng.random() < beta
        ep_rows = []

        def choose(state, g):
            ctx = policy.observe(state, g)
            exp = policy.expert_index(state, g)
            en = policy.enemies(state, g)
            ep_rows.append((ctx, exp, en[0] if en else (np.nan, np.nan), e))
            if learner:
                a = policy.act_index(intref.forward(model, ctx)[-1])
            else:
                a = exp
            return int(rng.integers(len(policy.ACTIONS))) if rng.random() < eps else a

        policy.play_episode(game, choose, seed=seed0 + e)
        rows += ep_rows
    game.close()
    return rows


def collect(episodes: int, seed0: int, workers: int, model_dir=None, beta=0.0, first=0) -> dict:
    chunk = max(1, episodes // (workers * 4))
    jobs = [(first + i, min(chunk, episodes - i), seed0, NOISE, model_dir, beta) for i in range(0, episodes, chunk)]
    with mp.get_context("spawn").Pool(workers) as pool:
        rows = [r for part in pool.map(_rollout_worker, jobs) for r in part]
    return dict(ctx=np.array([r[0] for r in rows], dtype=np.int8), y=np.array([r[1] for r in rows], dtype=np.int8),
                dist=np.array([r[2][0] for r in rows], dtype=np.float32),
                bearing=np.array([r[2][1] for r in rows], dtype=np.float32),
                ep=np.array([r[3] for r in rows], dtype=np.int32))


def merge(a: dict | None, b: dict) -> dict:
    return b if a is None else {k: np.concatenate([a[k], b[k]]) for k in b}


def split(ep: np.ndarray, test_frac=0.2, dev_frac=0.1):
    """Episode-level split: ep % 10 selects (deterministic, independent of the noise cycle length 5)."""
    h = (ep * 7919) % 100
    test = h < test_frac * 100
    dev = (h >= test_frac * 100) & (h < (test_frac + dev_frac) * 100)
    return ~test & ~dev, dev, test


# ----------------------------------------------------------------------------- training
def pack(d: dict, mask: np.ndarray, T: int):
    tok = torch.as_tensor(d["ctx"][mask].astype(np.int64))
    y = torch.as_tensor(d["y"][mask].astype(np.int64))
    return tok, y, torch.ones(tok.shape, dtype=torch.bool)


def make_loss(net):
    ids = torch.tensor(policy.ACTION_IDS)

    def loss_fn(tok, y, valid):
        logits, worst, fin = net(tok, valid)
        a = logits[:, -1][:, ids] * net.lg.exp()
        ce = F.cross_entropy(a, y)
        pen = (sum(F.relu(w - PLIM) for w in worst.values()) + sum(F.relu(w - FLIM) for w in fin.values())) / 1000.0
        return ce, pen, a, {**worst, **{"fin_" + k: v for k, v in fin.items()}}

    return loss_fn


@torch.no_grad()
def evaluate(net, data, loss_fn, bs=4096):
    saved = (net.quant, net.hard, net.training)
    net.quant, net.hard = True, True
    net.eval()
    tok, y, valid = data
    hit = ce = 0.0
    worst = {}
    for i in range(0, len(tok), bs):
        c, _, a, w = loss_fn(tok[i:i + bs], y[i:i + bs], valid[i:i + bs])
        hit += (a.argmax(-1) == y[i:i + bs]).sum().item()
        ce += c.item() * len(a)
        for k, x in w.items():
            worst[k] = max(worst.get(k, 0), int(x))
    net.quant, net.hard = saved[0], saved[1]
    net.train(saved[2])
    return hit / len(tok), ce / len(tok), worst


def train_model(args, d: dict, log):
    tr, dv, te = split(d["ep"])
    V, T = len(policy.VOCAB), policy.CONTEXT_LEN
    train, dev, test = pack(d, tr, T), pack(d, dv, T), pack(d, te, T)
    torch.manual_seed(args.seed)
    net = Net(V, T, args.d, args.d // 8, args.dm, args.K, args.K, args.K, args.L, args.S, args.seed)
    loss_fn = make_loss(net)
    ev = lambda data: evaluate(net, data, loss_fn)
    log(f"train {len(train[0])} dev {len(dev[0])} test {len(test[0])} decisions; params "
        f"{sum(p.numel() for p in net.parameters())}")

    net.quant, net.hard, net.calibrating = False, True, True
    sub = torch.randperm(len(train[0]))[:20000]
    net.valid = train[2][sub]
    net.eval()
    with torch.no_grad():
        net(train[0][sub], net.valid)
    net.calibrating = False

    rs = lambda *a, **k: run_stage_generic(net, *a, loss_fn=loss_fn, evaluate_fn=ev, **k)
    rs("float", args.steps1, args.lr1, False, False, args.alpha, train, args.bs, log, dev=dev)
    rs("act-qat", args.steps2, args.lr2, False, True, args.alpha, train, args.bs, log, dev=dev)
    rs("full-qat", args.steps3, args.lr3, True, True, args.alpha, train, args.bs, log, keep_best=True, dev=dev)
    return net, dict(train=train, dev=dev, test=test)


def export(net: Net) -> QModel:
    m = net.qmodel(policy.VOCAB)
    keep = np.zeros(len(policy.VOCAB), dtype=bool)
    keep[policy.ACTION_IDS] = True
    m.p["Wun"][~keep] = 0                      # the host only reads the action-token logits
    m.p["bun"][~keep] = 0
    return m


def verify(m: QModel, d: dict, log) -> dict:
    """check_limits on every context observe() can produce and every dataset context; torch-int == intref."""
    uniq = {tuple(c) for c in d["ctx"].tolist()} | {tuple(c) for c in policy.all_contexts()}
    seqs = [list(c) for c in sorted(uniq)]
    errs = intref.check_limits(m, seqs)
    tok = torch.as_tensor(np.array(seqs), dtype=torch.int64)
    ti = int_forward(m.p, tok, m.H, m.L).numpy()
    bad = sum(not np.array_equal(ti[i], intref.forward(m, s)) for i, s in enumerate(seqs))
    # every dataset decision (not only the unique contexts): torch-int logits per context looked up per row
    index = {s: i for i, s in enumerate(map(tuple, seqs))}
    rows = np.array([index[tuple(c)] for c in d["ctx"].tolist()])
    log(f"check_limits: {errs or 'OK'} | {len(seqs)} unique contexts covering {len(rows)} dataset decisions, "
        f"torch-int != intref in {bad}")
    assert not errs, errs
    assert bad == 0
    return dict(limits_errors=errs, unique_contexts=len(seqs), dataset_decisions=int(len(rows)), mismatches=bad)


def agreement(m: QModel, d: dict, mask: np.ndarray) -> float:
    cache, hit = {}, 0
    for c, y in zip(d["ctx"][mask].tolist(), d["y"][mask].tolist()):
        k = tuple(c)
        if k not in cache:
            cache[k] = policy.act_index(intref.forward(m, c)[-1])
        hit += cache[k] == y
    return hit / int(mask.sum())


def table_ceiling(d: dict, tr: np.ndarray, te: np.ndarray) -> float:
    """Best possible agreement of ANY function of the tokens (majority expert action per context)."""
    key = d["ctx"][:, 1].astype(np.int64) * 64 + d["ctx"][:, 2]
    maj = {}
    for k in np.unique(key[tr]):
        maj[k] = np.bincount(d["y"][tr][key[tr] == k], minlength=3).argmax()
    pred = np.array([maj.get(k, 0) for k in key[te]])
    return float((pred == d["y"][te]).mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "models" / "doomplayer"))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--L", type=int, default=2)
    ap.add_argument("--d", type=int, default=32)
    ap.add_argument("--dm", type=int, default=64)
    ap.add_argument("--K", type=int, default=7)
    ap.add_argument("--S", type=float, default=64.0)
    ap.add_argument("--bs", type=int, default=256)
    ap.add_argument("--steps1", type=int, default=0)
    ap.add_argument("--steps2", type=int, default=12000)
    ap.add_argument("--steps3", type=int, default=4000)
    ap.add_argument("--lr1", type=float, default=0.01)
    ap.add_argument("--lr2", type=float, default=0.01)
    ap.add_argument("--lr3", type=float, default=0.003)
    ap.add_argument("--alpha", type=float, default=0.5)
    ap.add_argument("--episodes", type=int, default=600, help="expert rollout episodes for the base dataset")
    ap.add_argument("--dagger", type=int, default=0, help="DAgger rounds (learner rollouts, expert labels)")
    ap.add_argument("--dagger-episodes", type=int, default=200)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--recollect", action="store_true", help="ignore the cached dataset")
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    torch.set_default_dtype(torch.float32)
    assert a.L <= 2 and a.d <= 32 and a.d % 8 == 0 and a.dm <= 64 and a.K <= 7
    assert policy.CONTEXT_LEN <= 8 and len(policy.VOCAB) <= 32
    t0 = time.time()

    def log(s):
        print(f"{time.time() - t0:7.1f}s {s}", flush=True)

    DATA.parent.mkdir(exist_ok=True)
    if DATA.exists() and not a.recollect:
        d = dict(np.load(DATA))
        log(f"loaded {repo_rel(DATA)}: {len(d['y'])} decisions, {len(np.unique(d['ep']))} episodes")
    else:
        d = collect(a.episodes, seed0=1000, workers=a.workers)
        np.savez_compressed(DATA, **d)
        log(f"collected {len(d['y'])} decisions in {a.episodes} expert episodes -> {repo_rel(DATA)}")
    nxt = int(d["ep"].max()) + 1
    info = {}
    out = Path(a.out)
    for rnd in range(a.dagger + 1):
        net, sets = train_model(a, d, log)
        m = export(net)
        m.save(out)
        if rnd < a.dagger:
            add = collect(a.dagger_episodes, seed0=1000, workers=a.workers, model_dir=str(out), beta=0.7, first=nxt)
            nxt += a.dagger_episodes
            d = merge(d, add)
            log(f"DAgger round {rnd + 1}: +{len(add['y'])} decisions (dataset {len(d['y'])})")
    tr, dv, te = split(d["ep"])
    m = export(net)
    m.save(out)
    info["verify"] = verify(m, d, log)
    info["agreement"] = dict(train=agreement(m, d, tr), dev=agreement(m, d, dv), test=agreement(m, d, te),
                             bin_ceiling_test=table_ceiling(d, tr, te))
    log(f"expert agreement (integer model, intref): {info['agreement']}")
    info |= dict(args=record_args(a), wall_s=time.time() - t0, n_decisions=int(len(d["y"])),
                 n_episodes=int(len(np.unique(d["ep"]))), int_logit_scale=net.lg.exp().item())
    (out / "train_info.json").write_text(json.dumps(info, indent=1))
    log(f"saved {repo_rel(out)}")


if __name__ == "__main__":
    main()
