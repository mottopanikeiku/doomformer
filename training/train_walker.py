"""Train "doomwalker": an integer transformer (L<=2, d<=32, H=d/8, dm<=64, K<=7) that walks levelgen levels.

1. Data: the scripted player (demos.walker.expert_index, exact info) plays levels generated from MANY DIFFERENT
   NAMES (training/data/levels.json; DEMON CANYON and the evaluation names are never used).  Every decision gives
   (context tokens = walker.observe, expert action).  Episodes are driven by the expert with action noise (random
   actions held for 1-3 decisions) and, in DAgger rounds, partly by the learner itself (the integer model plays, the
   expert relabels the visited states) so the student sees and learns to recover from its own mistakes.
2. Model: train_brain.Net (multi-layer QAT, same recipe as train_policy: act-QAT then full QAT), loss = cross-entropy
   over the ACTION-token logits at the last position (walker.act takes the argmax over the action tokens only).
3. Gate: check_limits == [] on every dataset context and on every context observe() can produce; the standalone
   int64 torch forward must equal intref.forward on 100 % of them.

Usage: python training/train_walker.py --out models/doomwalker [--levels 170 --per-level 12 --dagger 2]
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import random
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

from doomformer import intref  # noqa: E402

from demos import walker
from doomformer.intref import QModel  # noqa: E402
from train_brain import FLIM, PLIM, Net, int_forward, record_args, repo_rel  # noqa: E402
from train_policy import run_stage_generic  # noqa: E402

NOISE = (0.0, 0.0, 0.05, 0.1, 0.2)       # epsilon of random actions cycled over the rollout episodes
DATA = HERE / "data" / "doomwalker_dataset.npz"
LEVEL_DIR = ROOT / "build" / "walker_levels"
HELD_OUT_SEED = 7
HELD_OUT = 10                           # evaluation names besides DEMON CANYON (never trained on)
MAX_DECISIONS = 450


def level_names() -> tuple[list[str], list[str]]:
    """(training names, held-out evaluation names) from the level-name corpus; DEMON CANYON is in neither."""
    d = json.loads((HERE / "data" / "levels.json").read_text())
    clean = sorted({" ".join(x["clean"].upper().split()) for x in d if x.get("clean")})
    clean = [n for n in clean if 3 <= len(n) <= 30 and n != "DEMON CANYON"]
    random.Random(HELD_OUT_SEED).shuffle(clean)
    return clean[HELD_OUT:], clean[:HELD_OUT]


# ----------------------------------------------------------------------------- data collection
def _rollout_worker(args):
    jobs, seed0, noise, model_dir, beta, skill = args
    torch.set_num_threads(1)
    model = QModel.load(model_dir) if model_dir else None
    rows, eps = [], []
    for lvl, name, ep_ids in jobs:
        try:
            info, wad = walker.prepare_level(name, LEVEL_DIR)
        except RuntimeError:
            continue
        game = walker.make_game(False, wad=wad, skill=skill)
        for e in ep_ids:
            rng = np.random.default_rng(seed0 + e)
            eps_noise = noise[e % len(noise)]
            learner = model is not None and rng.random() < beta
            ep_rows = []
            burst = [0, 0]                                   # remaining decisions, action

            def choose(state, g):
                ctx = walker.observe(state, g)
                exp = walker.expert_index(state, g)
                ep_rows.append((ctx, exp, e, lvl))
                a = walker.act_index(intref.forward(model, ctx)[-1]) if learner else exp
                if burst[0] > 0:
                    burst[0] -= 1
                    return burst[1]
                if rng.random() < eps_noise:
                    burst[:] = [int(rng.integers(0, 3)), int(rng.integers(len(walker.ACTIONS)))]
                    return burst[1]
                return a

            r = walker.play_episode(game, choose, seed=seed0 + e, max_decisions=MAX_DECISIONS)
            rows += ep_rows
            eps.append((e, lvl, int(r["reached_exit"]), r["progress"], int(r["died"]), r["decisions"], r["kills"],
                        int(learner), eps_noise))
        game.close()
    return rows, eps


def collect(names: list[str], per_level: int, seed0: int, workers: int, skill: int, first_ep: int,
            model_dir=None, beta=0.0) -> tuple[dict, np.ndarray]:
    jobs = [(i, n, [first_ep + i * per_level + k for k in range(per_level)]) for i, n in enumerate(names)]
    chunks = [jobs[i::workers * 4] for i in range(workers * 4)]
    with mp.get_context("spawn").Pool(workers) as pool:
        out = pool.map(_rollout_worker, [(c, seed0, NOISE, model_dir, beta, skill) for c in chunks if c])
    rows = [r for part, _ in out for r in part]
    eps = np.array([e for _, part in out for e in part], dtype=np.float64).reshape(-1, 9)
    d = dict(ctx=np.array([r[0] for r in rows], dtype=np.int8), y=np.array([r[1] for r in rows], dtype=np.int8),
             ep=np.array([r[2] for r in rows], dtype=np.int32), lvl=np.array([r[3] for r in rows], dtype=np.int16))
    return d, eps


def merge(a: dict | None, b: dict) -> dict:
    return b if a is None else {k: np.concatenate([a[k], b[k]]) for k in b}


def split(ep: np.ndarray, test_frac=0.2, dev_frac=0.1):
    """Episode-level split (deterministic hash of the episode id)."""
    h = (ep.astype(np.int64) * 7919) % 100
    test = h < test_frac * 100
    dev = (h >= test_frac * 100) & (h < (test_frac + dev_frac) * 100)
    return ~test & ~dev, dev, test


# ----------------------------------------------------------------------------- training
def pack(d: dict, mask: np.ndarray):
    tok = torch.as_tensor(d["ctx"][mask].astype(np.int64))
    y = torch.as_tensor(d["y"][mask].astype(np.int64))
    return tok, y, torch.ones(tok.shape, dtype=torch.bool)


def make_loss(net, class_w: torch.Tensor):
    ids = torch.tensor(walker.ACTION_IDS)

    def loss_fn(tok, y, valid):
        logits, worst, fin = net(tok, valid)
        a = logits[:, -1][:, ids] * net.lg.exp()
        ce = F.cross_entropy(a, y, weight=class_w)
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
    V, T = len(walker.VOCAB), walker.CONTEXT_LEN
    train, dev, test = pack(d, tr), pack(d, dv), pack(d, te)
    torch.manual_seed(args.seed)
    net = Net(V, T, args.d, args.d // 8, args.dm, args.K, args.K, args.K, args.L, args.S, args.seed)
    freq = np.bincount(d["y"][tr], minlength=len(walker.ACTIONS)).astype(np.float64) + 1.0
    cw = torch.tensor((freq.sum() / freq) ** args.class_power, dtype=torch.float32)
    cw = cw / (cw * torch.tensor(freq / freq.sum(), dtype=torch.float32)).sum()     # mean weight 1 over the data
    log(f"action counts {dict(zip(walker.ACTIONS, freq.astype(int).tolist()))} class weights "
        f"{[round(x, 2) for x in cw.tolist()]}")
    loss_fn = make_loss(net, cw)
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
    return net


def export(net: Net) -> QModel:
    m = net.qmodel(walker.VOCAB)
    keep = np.zeros(len(walker.VOCAB), dtype=bool)
    keep[walker.ACTION_IDS] = True
    m.p["Wun"][~keep] = 0                      # the host only reads the action-token logits
    m.p["bun"][~keep] = 0
    return m


def verify(m: QModel, d: dict, log) -> dict:
    """check_limits on every context observe() can produce and every dataset context; torch-int == intref."""
    uniq = {tuple(c) for c in d["ctx"].tolist()} | {tuple(c) for c in walker.all_contexts()}
    seqs = [list(c) for c in sorted(uniq)]
    errs = intref.check_limits(m, seqs)
    tok = torch.as_tensor(np.array(seqs), dtype=torch.int64)
    ti = int_forward(m.p, tok, m.H, m.L).numpy()
    bad = sum(not np.array_equal(ti[i], intref.forward(m, s)) for i, s in enumerate(seqs))
    log(f"check_limits: {errs or 'OK'} | {len(seqs)} unique contexts (dataset + every observable context) covering "
        f"{len(d['y'])} dataset decisions, torch-int != intref in {bad}")
    assert not errs, errs
    assert bad == 0
    return dict(limits_errors=errs, unique_contexts=len(seqs), dataset_decisions=int(len(d["y"])), mismatches=bad)


def policy_table(m: QModel, ctxs) -> dict:
    cache = {}
    for c in ctxs:
        k = tuple(c)
        if k not in cache:
            cache[k] = walker.act_index(intref.forward(m, list(k))[-1])
    return cache


def agreement(m: QModel, d: dict, mask: np.ndarray) -> dict:
    ctx, y = d["ctx"][mask], d["y"][mask]
    table = policy_table(m, ctx.tolist())
    pred = np.array([table[tuple(c)] for c in ctx.tolist()])
    per = {a: float((pred[y == i] == i).mean()) if (y == i).any() else None for i, a in enumerate(walker.ACTIONS)}
    return dict(all=float((pred == y).mean()), per_action=per)


def table_ceiling(d: dict, tr: np.ndarray, te: np.ndarray) -> float:
    """Best possible agreement of ANY function of the context tokens (majority expert action per context)."""
    key = (d["ctx"].astype(np.int64) * np.array([1, 64, 4096, 262144, 16777216])).sum(1)
    maj = {}
    for k in np.unique(key[tr]):
        maj[k] = np.bincount(d["y"][tr][key[tr] == k], minlength=len(walker.ACTIONS)).argmax()
    pred = np.array([maj.get(k, 0) for k in key[te]])
    return float((pred == d["y"][te]).mean())


def ambiguity(d: dict) -> dict:
    """The expert is meant to be a function of the context tokens: count decisions whose label differs from the
    majority label of the same context (0 = the learner can match the expert exactly)."""
    key = (d["ctx"].astype(np.int64) * np.array([1, 64, 4096, 262144, 16777216])).sum(1)
    order = np.argsort(key, kind="stable")
    k, y = key[order], d["y"][order].astype(np.int64)
    starts = np.flatnonzero(np.r_[True, k[1:] != k[:-1]])
    counts = np.add.reduceat(np.eye(len(walker.ACTIONS), dtype=np.int64)[y], starts, axis=0)
    return dict(contexts=len(starts), decisions=len(y), ambiguous=int((counts.sum(1) - counts.max(1)).sum()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "models" / "doomwalker"))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--L", type=int, default=1)
    ap.add_argument("--d", type=int, default=16)
    ap.add_argument("--dm", type=int, default=32)
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
    ap.add_argument("--class-power", type=float, default=0.5, help="loss weight of action c ~ (1/freq_c)^power")
    ap.add_argument("--levels", type=int, default=170, help="training level names")
    ap.add_argument("--per-level", type=int, default=12, help="expert episodes per level for the base dataset")
    ap.add_argument("--dagger", type=int, default=2, help="DAgger rounds (learner rollouts, expert labels)")
    ap.add_argument("--dagger-per-level", type=int, default=6)
    ap.add_argument("--beta", type=float, default=0.7, help="share of DAgger episodes driven by the learner")
    ap.add_argument("--skill", type=int, default=walker.SKILL)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--recollect", action="store_true", help="ignore the cached dataset")
    ap.add_argument("--data", default=str(DATA))
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    torch.set_default_dtype(torch.float32)
    assert a.L <= 2 and a.d <= 32 and a.d % 8 == 0 and a.dm <= 64 and a.K <= 7
    assert walker.CONTEXT_LEN <= 5 and len(walker.VOCAB) <= 32 and len(walker.ACTIONS) <= 8
    t0 = time.time()

    def log(s):
        print(f"{time.time() - t0:7.1f}s {s}", flush=True)

    train_names, held = level_names()
    names = train_names[:a.levels]
    assert "DEMON CANYON" not in names and not set(names) & set(held)
    log(f"{len(names)} training level names (e.g. {names[:4]}), held-out eval names {held}")
    data_path = Path(a.data)
    data_path.parent.mkdir(exist_ok=True)
    ep_stats = []
    if data_path.exists() and not a.recollect:
        d = dict(np.load(data_path))
        ep_stats = list(d.pop("eps", np.zeros((0, 9))))
        log(f"loaded {repo_rel(data_path)}: {len(d['y'])} decisions, {len(np.unique(d['ep']))} episodes")
    else:
        d, eps = collect(names, a.per_level, 1000, a.workers, a.skill, 0)
        ep_stats = list(eps)
        np.savez_compressed(data_path, **d, eps=np.array(ep_stats))
        ex = eps[eps[:, 8] == 0]
        log(f"collected {len(d['y'])} decisions in {len(eps)} expert episodes -> {repo_rel(data_path)}; noise-free expert: "
            f"exit {ex[:, 2].mean():.3f}, progress {ex[:, 3].mean():.3f}, kills {ex[:, 6].mean():.2f}")
    nxt = int(d["ep"].max()) + 1
    out = Path(a.out)
    info = {}
    for rnd in range(a.dagger + 1):
        net = train_model(a, d, log)
        m = export(net)
        m.save(out)
        if rnd < a.dagger:
            add, eps = collect(names, a.dagger_per_level, 1000, a.workers, a.skill, nxt, model_dir=str(out),
                               beta=a.beta)
            nxt += len(names) * a.dagger_per_level
            d = merge(d, add)
            ep_stats += list(eps)
            ln = eps[eps[:, 7] == 1]
            log(f"DAgger round {rnd + 1}: +{len(add['y'])} decisions (dataset {len(d['y'])}); learner episodes: "
                f"exit {ln[:, 2].mean():.3f}, progress {ln[:, 3].mean():.3f}, kills {ln[:, 6].mean():.2f} "
                f"(with noise {ln[:, 8].mean():.2f} avg eps)")
            np.savez_compressed(data_path, **d, eps=np.array(ep_stats))
    tr, dv, te = split(d["ep"])
    m = export(net)
    m.save(out)
    info["ambiguity"] = ambiguity(d)
    log(f"expert label ambiguity over the dataset: {info['ambiguity']}")
    info["verify"] = verify(m, d, log)
    info["agreement"] = dict(train=agreement(m, d, tr), dev=agreement(m, d, dv), test=agreement(m, d, te),
                             table_ceiling_test=table_ceiling(d, tr, te))
    log(f"expert agreement (integer model, intref): {json.dumps(info['agreement'])}")
    info |= dict(args=record_args(a), wall_s=time.time() - t0, n_decisions=int(len(d["y"])),
                 n_episodes=int(len(np.unique(d["ep"]))), n_levels=len(names), held_out=held,
                 int_logit_scale=net.lg.exp().item())
    (out / "train_info.json").write_text(json.dumps(info, indent=1))
    log(f"saved {repo_rel(out)}")


if __name__ == "__main__":
    main()
