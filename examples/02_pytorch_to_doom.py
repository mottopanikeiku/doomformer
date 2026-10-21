"""02 - Train a float network, run it in DOOM.

Toy task: "is the demon inside the circle?". A point (x, y) in [-1, 1]^2 is labelled 1 when it lies within 0.6 of the
point (0.1, -0.1). The script trains a small float MLP (PyTorch if installed, otherwise plain NumPy SGD), quantises it
into an integer MLP, and runs test points through the DOOM map.

Three different things get compared, and they mean different things:
  float model   : the model you trained.
  integer model : an APPROXIMATION of the float model (weights and activations are quantised, activations become
                  8-level staircases). It can disagree with the float model near the boundary; that is quantisation
                  error and the report below measures it.
  DOOM          : computes the integer model EXACTLY. brain(x) raises MismatchError on any difference with the
                  independent integer reference, so "EXACT" below is a bit-for-bit claim about the integer model only.

    python examples/02_pytorch_to_doom.py [--points 24] [--seed 0]

Writes examples/out/02_circle.png. Exits non-zero if DOOM differs from the integer reference.
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

import doomformer as df

CENTER, RADIUS = np.array([0.1, -0.1]), 0.6
HIDDEN = 12
OUT = Path(__file__).resolve().parent / "out"


def label(p: np.ndarray) -> np.ndarray:
    return (np.linalg.norm(p - CENTER, axis=-1) < RADIUS).astype(np.int64)


def train_torch(X: np.ndarray, Y: np.ndarray, seed: int):
    import torch
    from torch import nn
    torch.manual_seed(seed)
    net = nn.Sequential(nn.Linear(2, HIDDEN), nn.ReLU(), nn.Linear(HIDDEN, 2))
    opt = torch.optim.Adam(net.parameters(), lr=0.03)
    xt, yt = torch.tensor(X, dtype=torch.float32), torch.tensor(Y)
    for _ in range(600):
        opt.zero_grad()
        nn.functional.cross_entropy(net(xt), yt).backward()
        opt.step()
    net.eval()
    return net, lambda p: net(torch.tensor(p, dtype=torch.float32)).detach().numpy()


def train_numpy(X: np.ndarray, Y: np.ndarray, seed: int):
    rng = np.random.default_rng(seed)
    W1, b1 = rng.normal(0, 1.0, (HIDDEN, 2)), rng.normal(0, 0.3, HIDDEN)
    W2, b2 = rng.normal(0, 0.5, (2, HIDDEN)), np.zeros(2)
    onehot = np.eye(2)[Y]
    for step in range(4000):                                   # full-batch gradient descent with momentum-free SGD
        h = np.maximum(X @ W1.T + b1, 0)
        z = h @ W2.T + b2
        p = np.exp(z - z.max(1, keepdims=True))
        p /= p.sum(1, keepdims=True)
        dz = (p - onehot) / len(X)
        dh = (dz @ W2) * (h > 0)
        lr = 0.8
        W2 -= lr * dz.T @ h
        b2 -= lr * dz.sum(0)
        W1 -= lr * dh.T @ X
        b1 -= lr * dh.sum(0)

    def forward(pts):
        return np.maximum(pts @ W1.T + b1, 0) @ W2.T + b2
    return [(W1, b1), (W2, b2)], forward


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--points", type=int, default=24, help="test points run through DOOM")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    X = rng.uniform(-1, 1, (3000, 2))
    Y = label(X)
    try:
        import torch  # noqa: F401
        model, float_forward = train_torch(X, Y, args.seed)
        mlp = df.MLP.from_torch(model, X[:500], levels=8)
        how = "PyTorch"
    except ImportError:
        layers, float_forward = train_numpy(X, Y, args.seed)
        mlp = df.MLP.from_numpy(layers, X[:500], levels=8)
        how = "NumPy"
    Xt = rng.uniform(-1, 1, (4000, 2))
    float_acc = float((float_forward(Xt).argmax(1) == label(Xt)).mean())
    int_acc = float(np.mean([np.argmax(mlp.reference(mlp.encode(p))["y"]) == t for p, t in zip(Xt, label(Xt))]))
    print(f"trained with {how}: accuracy on 4000 fresh points: float model {float_acc:.3f}, integer model "
          f"{int_acc:.3f} (integer reference only; DOOM is run on the test points below)")
    print("quantisation report (integer model vs float model, calibration data):", mlp.quantization_report(X[:500]))

    # Test points: half uniform, half close to the circle edge where quantisation error matters most.
    n_edge = args.points // 2
    ang = rng.uniform(0, 2 * np.pi, n_edge)
    rad = RADIUS + rng.uniform(-0.1, 0.1, n_edge)
    edge = CENTER + np.stack([rad * np.cos(ang), rad * np.sin(ang)], 1)
    pts = np.clip(np.concatenate([rng.uniform(-1, 1, (args.points - n_edge, 2)), edge]), -1, 1)
    truth = label(pts)

    plan = df.plan(mlp)
    print(f"DOOM map: {plan['dolls']} voodoo dolls, {plan['sectors']} sectors")
    rows, ok = [], True
    t0 = time.time()
    with df.compile(mlp) as brain:
        for p, t in zip(pts, truth):
            xi = mlp.encode(p)
            try:
                y = brain(xi)                              # DOOM vs integer reference, raises on any difference
                exact = True
            except df.MismatchError as e:
                print("MISMATCH:", e)
                y, exact, ok = mlp.reference(xi)["y"], False, False
            f_cls = int(float_forward(p[None])[0].argmax())
            i_cls = int(np.argmax(mlp.reference(xi)["y"]))
            d_cls = int(np.argmax(y))
            rows.append((p, int(t), f_cls, i_cls, d_cls, exact))
    print(f"{len(rows)} passes in DOOM in {time.time() - t0:.0f}s\n")
    print("   point (x, y)     truth  float  integer  DOOM  DOOM==integer")
    for p, t, f, i, d, ex in rows:
        print(f"  ({p[0]:+.2f}, {p[1]:+.2f})      {t}      {f}      {i}       {d}    {'EXACT' if ex else 'MISMATCH'}"
              f"{'' if d == t else '   <- the integer model is wrong here (DOOM reproduces its error)'}")
    n = len(rows)
    print(f"\nagreement with the truth: float {sum(r[2] == r[1] for r in rows)}/{n}, integer "
          f"{sum(r[3] == r[1] for r in rows)}/{n}, DOOM {sum(r[4] == r[1] for r in rows)}/{n}")
    print(f"DOOM == integer reference on {sum(r[5] for r in rows)}/{n} points (bit-exact outputs and hidden values); "
          f"float vs integer class agreement {sum(r[2] == r[3] for r in rows)}/{n} (quantisation, not a DOOM error)")

    # Picture: the integer model's decision map (reference arithmetic), the true circle, and the DOOM test points.
    G, S = 64, 6
    grid = np.linspace(-1, 1, G)
    cls = np.array([[int(np.argmax(mlp.reference(mlp.encode(np.array([gx, gy])))["y"])) for gx in grid]
                    for gy in grid[::-1]])
    img = Image.fromarray(np.where(cls[..., None] == 1, [255, 200, 120], [40, 50, 70]).astype(np.uint8))
    img = img.resize((G * S, G * S), Image.NEAREST)
    canvas = Image.new("RGB", (G * S, G * S + 40), (0, 0, 0))
    canvas.paste(img, (0, 40))
    img = canvas
    d = ImageDraw.Draw(img)

    def px(p):
        return ((p[0] + 1) / 2 * (G * S - 1), 40 + (1 - (p[1] + 1) / 2) * (G * S - 1))
    cx, cy = px(CENTER)
    r = RADIUS / 2 * (G * S - 1)
    d.ellipse([cx - r, cy - r, cx + r, cy + r], outline=(255, 255, 255), width=2)
    for p, t, f, i, dm, ex in rows:
        x, y = px(p)
        col = (60, 220, 90) if dm == t else (230, 50, 50)
        d.ellipse([x - 6, y - 6, x + 6, y + 6], fill=col, outline=(0, 0, 0))
    d.text((8, 5), "orange: integer model says INSIDE; ring: truth", fill=(255, 255, 255))
    d.text((8, 20), "dots: points run in DOOM (green = right, red = wrong)", fill=(255, 255, 255))
    OUT.mkdir(exist_ok=True)
    img.save(OUT / "02_circle.png")
    print(f"wrote {OUT / '02_circle.png'}")
    print("ALL EXACT" if ok else "FAILED: DOOM differs from the integer reference")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
