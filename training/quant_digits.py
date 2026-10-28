"""Quantisation study on the 8x8 optical digits (UCI optdigits): float vs PTQ-max vs PTQ-mse vs QAT, lowered with
`MLP.from_torch`, integer accuracy by the NumPy reference (needs the `train` extra; no scikit-learn needed).

Data: https://archive.ics.uci.edu/ml/machine-learning-databases/optdigits/ (optdigits.tra 3,823 rows, optdigits.tes
1,797 rows, 8x8 pixels 0..16, written by different people). The 1,797 rows of optdigits.tes are the rows of
sklearn.datasets.load_digits. Protocols:
  writer: train on optdigits.tra, test on optdigits.tes (different writers),
  mixed : the 1,797 rows of optdigits.tes split 70/30 at random (seeded), like the usual scikit-learn digits split.
Inputs are x / 16.

Models (float, Adam + one-cycle, 60 epochs, batch 64):
  cnn         Conv(1>8, 3x3, p1) ReLU Conv(8>8, 3x3, s2, p1) ReLU Flatten Linear(128>10)           (1,954 parameters)
  mlp6        Linear 64>48 ReLU, four times Linear 48>48 ReLU, Linear 48>10                          (6 Linear layers)
  cnn_bn_pool Conv(1>8) BatchNorm ReLU MaxPool(2) Conv(8>16) BatchNorm ReLU AvgPool(2) Flatten Linear(64>10)

For K in 4, 8, and two ways to set the staircase levels (`uniform`: levels=K for every staircase, the input's too, as
in the earlier probes; `in16`: [16, K, K, ...]: the input keeps all 17 pixel values):
  ptq_max  MLP.from_torch(model, calib, levels)                           (v0.5.0 behaviour)
  ptq_mse  MLP.from_torch(model, calib, levels, calibrate="mse")          (MSE step, equalise)
  qat      qat.fit(model, loader, levels, steps=15 epochs) then from_torch  (lr 5e-4: the library defaults; 15 epochs
           as in training/train_mnist.py, = 900 steps for 3,823 rows, 300 for 1,258)
Calibration rows: the first 1,000 training rows. Hyperparameters are the library defaults, not tuned on the test
rows. Writes <out>/run_<protocol>_<seed>.json; `--aggregate` merges them into <out>/results.json and EVAL.md.

usage: quant_digits.py --protocol writer|mixed --seed 0 [--out models/quant] [--qat-epochs 15]
       quant_digits.py --aggregate [--out models/quant]
       quant_digits.py --sweep --protocol writer --seed 0 --models cnn      (validation split; log in sweep_validation.log)
       quant_digits.py --sweep-equalise --protocol writer --seed 0 --models cnn   (log in qat_equalise_validation.log)
       quant_digits.py --doom 5 --models cnn mlp6 cnn_bn_pool --protocol writer --seed 0 [--workers 3] [--k 4]
         trains float + QAT as above (levels=K for every staircase), lowers the QAT model and runs the first N test
         digits through DOOM (df.run_many, one map per game if the net fits one map, else tiles); every value of every pass is compared with the
         NumPy integer reference at tolerance 0 (MismatchError otherwise). Writes <out>/doom_<model>.json.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
import urllib.request
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "build" / "optdigits"
URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/optdigits/"
SHA256 = {"optdigits.tra": "e1b683cc211604fe8fd8c4417e6a69f31380e0c61d4af22e93cc21e9257ffedd",
          "optdigits.tes": "6ebb3d2fee246a4e99363262ddf8a00a3c41bee6014c373ed9d9216ba7f651b8"}
MODELS = ("cnn", "mlp6", "cnn_bn_pool")
EXTRA = ("cnn_mp", "mlp_gelu")          # nets with more than one max pool / a GELU, run with --models cnn_mp mlp_gelu --ks 2 4 8
KS = (4, 8)
CONVENTIONS = ("uniform", "in16")
FLOAT_EPOCHS, FLOAT_LR, BATCH = 60, 3e-3, 64


def load_optdigits(name: str) -> tuple[np.ndarray, np.ndarray]:
    path = DATA / name
    if not path.is_file():
        DATA.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(URL + name, path)
    if hashlib.sha256(path.read_bytes()).hexdigest() != SHA256[name]:
        raise SystemExit(f"{path}: sha256 differs from the file this study used")
    a = np.loadtxt(path, delimiter=",", dtype=np.int64)
    return a[:, :64].astype(np.float32), a[:, 64]


def split(protocol: str, seed: int):
    if protocol == "writer":
        (Xtr, ytr), (Xte, yte) = load_optdigits("optdigits.tra"), load_optdigits("optdigits.tes")
    elif protocol == "mixed":
        X, y = load_optdigits("optdigits.tes")
        perm = np.random.default_rng(1000 + seed).permutation(len(X))
        cut = int(round(0.7 * len(X)))
        Xtr, ytr, Xte, yte = X[perm[:cut]], y[perm[:cut]], X[perm[cut:]], y[perm[cut:]]
    else:
        raise SystemExit("protocol must be writer or mixed")
    return Xtr.reshape(-1, 1, 8, 8) / 16, ytr, Xte.reshape(-1, 1, 8, 8) / 16, yte


def build(name: str, seed: int = 0):
    """A freshly initialised model (initial weights from `seed`)."""
    import torch
    import torch.nn as nn
    torch.manual_seed(seed)
    if name == "cnn":
        return nn.Sequential(nn.Conv2d(1, 8, 3, padding=1), nn.ReLU(), nn.Conv2d(8, 8, 3, stride=2, padding=1), nn.ReLU(),
                             nn.Flatten(), nn.Linear(128, 10))
    if name == "mlp6":
        layers = [nn.Flatten(), nn.Linear(64, 48), nn.ReLU()]
        for _ in range(4):
            layers += [nn.Linear(48, 48), nn.ReLU()]
        return nn.Sequential(*layers, nn.Linear(48, 10))
    if name == "cnn_bn_pool":
        return nn.Sequential(nn.Conv2d(1, 8, 3, padding=1), nn.BatchNorm2d(8), nn.ReLU(), nn.MaxPool2d(2),
                             nn.Conv2d(8, 16, 3, padding=1), nn.BatchNorm2d(16), nn.ReLU(), nn.AvgPool2d(2),
                             nn.Flatten(), nn.Linear(64, 10))
    if name == "cnn_mp":
        return nn.Sequential(nn.Conv2d(1, 8, 3, padding=1), nn.BatchNorm2d(8), nn.ReLU(), nn.MaxPool2d(2),
                             nn.Conv2d(8, 16, 3, padding=1), nn.BatchNorm2d(16), nn.ReLU(), nn.MaxPool2d(2),
                             nn.Flatten(), nn.Linear(64, 10))
    if name == "mlp_gelu":
        return nn.Sequential(nn.Flatten(), nn.Linear(64, 64), nn.GELU(), nn.Linear(64, 64), nn.GELU(), nn.Linear(64, 10))
    raise SystemExit(f"unknown model {name}")


def n_staircases(model) -> int:
    """Activations (hidden staircases) of the model: ReLU and GELU modules."""
    import torch.nn as nn
    return sum(isinstance(m, (nn.ReLU, nn.GELU)) for m in model.modules())


def train_float(model, Xtr, ytr, seed: int):
    import torch
    import torch.nn.functional as F
    torch.manual_seed(seed)
    X, y = torch.tensor(Xtr), torch.tensor(ytr)
    opt = torch.optim.Adam(model.parameters(), FLOAT_LR)
    steps = FLOAT_EPOCHS * ((len(X) + BATCH - 1) // BATCH)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, FLOAT_LR, total_steps=steps)
    model.train()
    for _ in range(FLOAT_EPOCHS):
        perm = torch.randperm(len(X))
        for i in range(0, len(X), BATCH):
            b = perm[i:i + BATCH]
            loss = F.cross_entropy(model(X[b]), y[b])
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
    return model.eval()


def make_loader(X, y, seed: int):
    import torch
    from torch.utils.data import DataLoader, TensorDataset
    return DataLoader(TensorDataset(torch.tensor(X), torch.tensor(y)), batch_size=BATCH, shuffle=True,
                      generator=torch.Generator().manual_seed(seed))


def torch_acc(model, X, y) -> float:
    import torch
    with torch.no_grad():
        return float((model(torch.tensor(X)).argmax(1).numpy() == y).mean())


def int_acc(net, X, y) -> float:
    return float((net.reference(net.encode(X.astype(np.float64)))["y"].argmax(1) == y).mean())


def run(protocol: str, seed: int, qat_epochs: int, models=MODELS, ks=KS, conventions=CONVENTIONS) -> dict:
    import torch

    import doomformer as df
    from doomformer import qat

    torch.set_num_threads(1)
    Xtr, ytr, Xte, yte = split(protocol, seed)
    calib = Xtr[:1000]
    qat_steps = qat_epochs * ((len(Xtr) + BATCH - 1) // BATCH)
    out = dict(protocol=protocol, seed=seed, n_train=len(Xtr), n_test=len(Xte), qat_epochs=qat_epochs, qat_steps=qat_steps,
               models={})
    for name in models:
        t0 = time.time()
        model = train_float(build(name, seed), Xtr, ytr, seed)
        res = dict(float=torch_acc(model, Xte, yte), params=sum(p.numel() for p in model.parameters()))
        for K in ks:
            for conv, levels in (("uniform", K), ("in16", [16] + [K] * n_staircases(model))):
                if conv not in conventions:
                    continue
                tag = f"K{K}_{conv}"
                r = {}
                r["ptq_max"] = int_acc(df.MLP.from_torch(model, calib, levels=levels), Xte, yte)
                mse = df.MLP.from_torch(model, calib, levels=levels, calibrate="mse")
                r["ptq_mse"] = int_acc(mse, Xte, yte)
                q = qat.fit(model, make_loader(Xtr, ytr, seed), levels=levels, steps=qat_steps, calib_x=torch.tensor(calib),
                            seed=seed)
                net = df.MLP.from_torch(q, calib)
                gap = qat.lowering_gap(q, net, torch.tensor(Xte), yte)
                r |= dict(qat=gap["accuracy_integer"], qat_torch=gap["accuracy_staircase_model"],
                          qat_agree=gap["argmax_agreement"])
                p = df.plan(net)
                size = p if p["fits"] else p["tiles"]
                r |= dict(dolls=size["dolls"], sectors=size["sectors"], layers=net.n_layers, one_map=bool(p["fits"]),
                          recommended=p["recommended"])
                res[tag] = r
                print(f"[{protocol} seed {seed}] {name} {tag}: " + " ".join(f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}"
                                                                           for k, v in r.items()), flush=True)
        res["seconds"] = round(time.time() - t0, 1)
        out["models"][name] = res
        print(f"[{protocol} seed {seed}] {name}: float {res['float']:.4f} ({res['seconds']} s)", flush=True)
    return out


def doom(protocol: str, seed: int, qat_epochs: int, name: str, n: int, workers: int, K: int, out_dir: Path) -> dict:
    """QAT model `name` at levels K, lowered, its first n test digits through DOOM: all values exact, DOOM class vs the
    integer reference's class vs the label."""
    import torch

    import doomformer as df
    from doomformer import qat

    torch.set_num_threads(1)
    Xtr, ytr, Xte, yte = split(protocol, seed)
    calib = Xtr[:1000]
    qat_steps = qat_epochs * ((len(Xtr) + BATCH - 1) // BATCH)
    model = train_float(build(name, seed), Xtr, ytr, seed)
    q = qat.fit(model, make_loader(Xtr, ytr, seed), levels=K, steps=qat_steps, calib_x=torch.tensor(calib), seed=seed)
    net = df.MLP.from_torch(q, calib)
    acc = int_acc(net, Xte, yte)
    xs = net.encode(Xte[:n].astype(np.float64))
    plan = df.plan(net)
    info = []
    t0 = time.time()
    mode = "map" if plan["fits"] else "tiles"                      # one game per worker; tiles hold one game per layer
    workers = workers if mode == "map" else 1
    ys = df.run_many(net, list(xs), workers=workers, mode=mode, progress=lambda i: info.append(i))
    wall = time.time() - t0
    ref = net.reference(xs)["y"]
    exact = all(np.array_equal(y, r) for y, r in zip(ys, ref))
    doom_cls, ref_cls = [int(np.argmax(y)) for y in ys], [int(np.argmax(r)) for r in ref]
    res = dict(model=name, protocol=protocol, seed=seed, levels=K, qat_epochs=qat_epochs, qat_steps=qat_steps, integer_test_accuracy=acc,
               digits=n, test_index=list(range(n)), labels=[int(v) for v in yte[:n]], doom_class=doom_cls,
               reference_class=ref_cls, float_class=[int(v) for v in torch_logits(model, Xte[:n]).argmax(1)],
               all_values_exact=bool(exact), values_checked=[int(i["checked"]) for i in sorted(info, key=lambda i: i["index"])],
               tics=[int(i["tics"]) for i in sorted(info, key=lambda i: i["index"])],
               pass_seconds=[round(float(i["seconds"]), 1) for i in sorted(info, key=lambda i: i["index"])],
               wall_seconds=round(wall, 1), workers=workers, mode=mode, recommended=plan["recommended"], layers=net.n_layers,
               dolls=(plan if plan["fits"] else plan["tiles"])["dolls"], sectors=(plan if plan["fits"] else plan["tiles"])["sectors"],
               one_map=bool(plan["fits"]))
    (out_dir / f"doom_{name}.json").write_text(json.dumps(res, indent=1) + "\n")
    print(f"[doom {name} K={K}] {n} digits: exact={exact} DOOM class {doom_cls} reference {ref_cls} labels {res['labels']} "
          f"integer test accuracy {acc:.4f}, {res['dolls']} dolls, {res['sectors']} sectors, mode {res['mode']}, "
          f"{wall:.0f} s wall (workers={workers}, loaded shared machine)", flush=True)
    return res


def sweep(protocol: str, seed: int, name: str) -> None:
    """QAT settings (steps, lr, distill) against PTQ mse at K = 4 on a validation split: the first 80 % of the training rows
    train the float model and the QAT, the last 20 % are scored (the test rows are not used). Prints one line each."""
    import itertools

    import torch

    import doomformer as df
    from doomformer import qat

    torch.set_num_threads(1)
    Xtr, ytr, _, _ = split(protocol, seed)
    n = int(len(Xtr) * 0.8)
    Xa, ya, Xv, yv = Xtr[:n], ytr[:n], Xtr[n:], ytr[n:]
    model = train_float(build(name, seed), Xa, ya, seed)
    calib = Xa[:1000]
    ptq = df.MLP.from_torch(model, calib, levels=4, calibrate="mse")
    print(f"{protocol} {name} {seed} VAL float {torch_acc(model, Xv, yv):.4f} ptq_mse {int_acc(ptq, Xv, yv):.4f}", flush=True)
    for steps, lr, distill in itertools.product((300, 1000, 3000), (1e-4, 3e-4, 1e-3), (0.0, 0.5)):
        q = qat.fit(model, make_loader(Xa, ya, seed), levels=4, steps=steps, calib_x=torch.tensor(calib), lr=lr,
                    distill=distill, seed=seed)
        print(f"{protocol} {name} {seed} steps {steps} lr {lr} distill {distill} qat_int "
              f"{int_acc(df.MLP.from_torch(q, calib), Xv, yv):.4f}", flush=True)


def sweep_equalise(protocol: str, seed: int, name: str) -> None:
    """qat.fit(equalise=False) against equalise=True and PTQ mse at K = 4 on a validation split (see `sweep`)."""
    import torch

    import doomformer as df
    from doomformer import qat

    torch.set_num_threads(1)
    Xtr, ytr, _, _ = split(protocol, seed)
    n = int(len(Xtr) * 0.8)
    Xa, ya, Xv, yv = Xtr[:n], ytr[:n], Xtr[n:], ytr[n:]
    model = train_float(build(name, seed), Xa, ya, seed)
    calib = Xa[:1000]
    steps = qat_steps_for(n)
    ptq = df.MLP.from_torch(model, calib, levels=4, calibrate="mse")
    out = [f"{protocol} {name} {seed} VAL float {torch_acc(model, Xv, yv):.4f} ptq_mse {int_acc(ptq, Xv, yv):.4f}"]
    for eq in (False, True):
        q = qat.fit(model, make_loader(Xa, ya, seed), levels=4, steps=steps, calib_x=torch.tensor(calib), seed=seed, equalise=eq)
        out.append(f"qat eq={eq}: {int_acc(df.MLP.from_torch(q, calib), Xv, yv):.4f}")
    print(" | ".join(out), flush=True)


def qat_steps_for(n_train: int, epochs: int = 15) -> int:
    return epochs * ((n_train + BATCH - 1) // BATCH)


def torch_logits(model, X):
    import torch
    with torch.no_grad():
        return model(torch.tensor(X)).numpy()


def aggregate(out_dir: Path) -> None:
    runs = [json.loads(p.read_text()) for p in sorted(out_dir.glob("run_*.json"))]
    if not runs:
        raise SystemExit("no run_*.json files")
    lines = []
    for protocol in ("writer", "mixed"):
        rs = [r for r in runs if r["protocol"] == protocol]
        if not rs:
            continue
        seeds = sorted({r["seed"] for r in rs})
        lines.append(f"### Protocol `{protocol}`: train {rs[0]['n_train']:,}, test {rs[0]['n_test']:,}, seeds "
                     f"{', '.join(map(str, seeds))} (mean, min-max in percent; integer accuracy by the NumPy reference)\n")
        lines.append("| model | float | K | levels | PTQ max | PTQ mse | QAT | QAT torch | dolls |")
        lines.append("|---|---|---|---|---|---|---|---|---|")
        for name in MODELS + EXTRA:
            ms = [r["models"][name] for r in rs if name in r["models"]]
            if not ms:
                continue
            tags = sorted({t for m in ms for t in m if t.startswith("K")}, key=lambda t: (int(t[1:].split("_")[0]), t.endswith("in16")))
            for tag in tags:
                have = [m for m in ms if tag in m]
                K, conv = int(tag[1:].split("_")[0]), tag.split("_")[1]
                cell = lambda key: (lambda v: f"{v.mean():.1f} ({v.min():.1f}-{v.max():.1f})")(  # noqa: E731
                    np.array([m[tag][key] for m in have]) * 100)
                lv = f"all {K}" if conv == "uniform" else f"input 16, hidden {K}"
                f1 = np.array([m["float"] for m in have]) * 100
                lines.append(f"| {name} | {f1.mean():.1f} ({f1.min():.1f}-{f1.max():.1f}) | {K} | {lv} | {cell('ptq_max')} | "
                             f"{cell('ptq_mse')} | {cell('qat')} | {cell('qat_torch')} | "
                             f"{have[0][tag]['dolls']:,} ({'map' if have[0][tag]['one_map'] else 'tiles'}) |")
        lines.append("")
    doom_runs = [json.loads(p.read_text()) for p in sorted(out_dir.glob("doom_*.json"))]
    if doom_runs:
        lines.append("### DOOM: the QAT nets at levels=K, first test digits, every value compared at tolerance 0\n")
        lines.append("| model | K | digits | all values exact | values per pass | tics per pass | dolls | sectors | mode | DOOM class | "
                     "integer class | label | float class | integer test accuracy | wall |")
        lines.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
        for d in doom_runs:
            lines.append(f"| {d['model']} | {d['levels']} | {d['digits']} | {'yes' if d['all_values_exact'] else 'NO'} | "
                         f"{d['values_checked'][0]} | {d['tics'][0]} | {d['dolls']:,} | {d['sectors']:,} | {d['mode']} | "
                         f"{''.join(map(str, d['doom_class']))} | {''.join(map(str, d['reference_class']))} | "
                         f"{''.join(map(str, d['labels']))} | {''.join(map(str, d['float_class']))} | "
                         f"{d['integer_test_accuracy'] * 100:.2f} % | {d['wall_seconds']:.0f} s, {d['workers']} games |")
        lines.append("")
    (out_dir / "results.json").write_text(json.dumps(runs, indent=1) + "\n")
    (out_dir / "results_table.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--protocol", default="writer")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--qat-epochs", type=int, default=15)
    ap.add_argument("--models", nargs="*", default=list(MODELS))
    ap.add_argument("--ks", nargs="*", type=int, default=list(KS), help="staircase levels K to study")
    ap.add_argument("--conventions", nargs="*", default=list(CONVENTIONS), help="uniform (levels=K) and/or in16")
    ap.add_argument("--out", type=Path, default=ROOT / "models" / "quant")
    ap.add_argument("--aggregate", action="store_true")
    ap.add_argument("--doom", type=int, default=0, help="run this many test digits of the K-level QAT nets in DOOM")
    ap.add_argument("--sweep-equalise", action="store_true", help="qat.fit equalise on/off on a validation split")
    ap.add_argument("--sweep", action="store_true", help="QAT steps/lr/distill on a validation split of the training rows")
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--k", type=int, default=4)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    if a.aggregate:
        aggregate(a.out)
        return
    if a.sweep_equalise:
        for name in a.models:
            sweep_equalise(a.protocol, a.seed, name)
        return
    if a.sweep:
        for name in a.models:
            sweep(a.protocol, a.seed, name)
        return
    if a.doom:
        for name in a.models:
            doom(a.protocol, a.seed, a.qat_epochs, name, a.doom, a.workers, a.k, a.out)
        return
    res = run(a.protocol, a.seed, a.qat_epochs, tuple(a.models), tuple(a.ks), tuple(a.conventions))
    suffix = "" if tuple(a.ks) == KS and tuple(a.conventions) == CONVENTIONS else "_K" + "-".join(map(str, a.ks))
    (a.out / f"run_{a.protocol}_{a.seed}{suffix}.json").write_text(json.dumps(res, indent=1) + "\n")


if __name__ == "__main__":
    main()
