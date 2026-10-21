"""11 - A convolutional network reads a glyph, DOOM does the arithmetic.

Toy task: which way does the arrow on a 7x7 pixel glyph point (up, down, left or right)? The glyphs are drawn at a
random shift of up to one pixel and every pixel is flipped with probability 0.03, so the network has to generalise.

    Conv2d(1 -> 3 channels, 3x3) -> Hardtanh(-1, 1) -> Flatten -> Linear(75 -> 4)

is trained with PyTorch (or, if torch is not installed or you pass --numpy, with plain NumPy), then
`df.MLP.from_torch` / `from_numpy` turns it into an integer MLP:

  * the convolution is LOWERED to an exact (mostly zero) linear layer: nothing is approximated there,
  * Hardtanh becomes a 3-level staircase (an APPROXIMATION of the float function; its offset -1 is folded into the
    next layer's bias),
  * weights are rounded to integers, the glyph pixels are staircase-quantised (one level: a pixel is 0 or 1).

Three things get compared, and they mean different things:
  float model   : the trained model.
  integer model : an approximation of the float model (the quantisation report says how good).
  DOOM          : computes the integer model EXACTLY. `net.predict(brain, x)` runs the integer glyph through the
                  DOOM map; brain(x) raises MismatchError if any output or hidden value differs from the independent
                  integer reference (tolerance 0). The argmax of DOOM's four output integers is taken by Python.

    python examples/11_conv_in_doom.py [--glyphs 12] [--seed 0] [--numpy]

Exits non-zero if DOOM differs from the integer reference on any glyph.
"""
import argparse
import sys
import time

import numpy as np

import doomformer as df
from doomformer.interop import conv2d_matrix

N = 7
NAMES = ["up", "down", "left", "right"]
HIDDEN_LEVELS = 3
CH = 3


def base_glyphs() -> np.ndarray:
    # one arrow (head and shaft in the middle of the 7x7 grid), the others are its mirror images / rotations
    g = np.zeros((4, N, N))
    for r, c in [(1, 3), (2, 2), (2, 4), (3, 1), (3, 5), (2, 3), (3, 3), (4, 3), (5, 3)]:
        g[0, r, c] = 1                                     # up:    head on top, shaft down
    g[1] = g[0][::-1]                                      # down:  mirrored vertically
    g[2] = g[0].T[::-1]                                    # left:  rotated (transpose + flip)
    g[3] = g[2][:, ::-1]                                   # right: mirrored horizontally
    return g


def make_data(n: int, rng: np.random.Generator, noise: float = 0.03):
    base = base_glyphs()
    y = rng.integers(0, 4, n)
    x = np.zeros((n, 1, N, N))
    for i, k in enumerate(y):
        dy, dx = rng.integers(-1, 2, 2)
        g = np.roll(base[k], (dy, dx), (0, 1))
        flip = rng.random((N, N)) < noise
        x[i, 0] = np.where(flip, 1 - g, g)
    return x, y


def show(glyph: np.ndarray) -> list[str]:
    return ["".join("#" if v > 0.5 else "." for v in row) for row in glyph[0]]


# --- training --------------------------------------------------------------------------------------------------------
def train_torch(X, Y, seed):
    import torch
    from torch import nn
    torch.manual_seed(seed)
    model = nn.Sequential(nn.Conv2d(1, CH, 3), nn.Hardtanh(-1.0, 1.0), nn.Flatten(), nn.Linear(CH * 25, 4))
    opt = torch.optim.Adam(model.parameters(), lr=0.02)
    xt, yt = torch.tensor(X, dtype=torch.float32), torch.tensor(Y)
    for _ in range(300):
        opt.zero_grad()
        nn.functional.cross_entropy(model(xt), yt).backward()
        opt.step()
    model.double().eval()
    return model


def patches(x: np.ndarray) -> np.ndarray:
    """(n, 1, 7, 7) -> (n, 5, 5, 3, 3): every 3x3 window."""
    w = np.lib.stride_tricks.sliding_window_view(x[:, 0], (3, 3), axis=(1, 2))      # (n, 5, 5, 3, 3)
    return w


def train_numpy(X, Y, seed):
    rng = np.random.default_rng(seed)
    K, bk = rng.normal(0, 0.5, (CH, 3, 3)), np.zeros(CH)
    W, b = rng.normal(0, 0.3, (4, CH * 25)), np.zeros(4)
    P, onehot = patches(X), np.eye(4)[Y]
    n, mom = len(X), [0 * K, 0 * bk, 0 * W, 0 * b]
    for _ in range(600):
        z = np.einsum("nyxkl,ckl->ncyx", P, K) + bk[None, :, None, None]
        h = np.clip(z, -1, 1)
        f = h.reshape(n, -1)
        o = f @ W.T + b
        p = np.exp(o - o.max(1, keepdims=True))
        p /= p.sum(1, keepdims=True)
        do = (p - onehot) / n
        dh = (do @ W).reshape(h.shape) * ((z > -1) & (z < 1))
        grads = [np.einsum("ncyx,nyxkl->ckl", dh, P), dh.sum((0, 2, 3)), do.T @ f, do.sum(0)]
        for i, (prm, g) in enumerate(zip((K, bk, W, b), grads)):
            mom[i] = 0.9 * mom[i] + g
            prm -= 0.2 * mom[i]
    M, bb, _ = conv2d_matrix(K[:, None], bk, (1, N, N))
    return [(M, bb, ("clamp", -1.0, 1.0)), (W, b)]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--glyphs", type=int, default=12, help="test glyphs run through DOOM")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--numpy", action="store_true", help="train with NumPy even if torch is installed")
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    X, Y = make_data(2000, rng)
    levels = [1, HIDDEN_LEVELS]                  # input staircase: 1 level (a pixel is 0 or 1); hidden: 3 levels
    try:
        if args.numpy:
            raise ImportError
        model = train_torch(X, Y, args.seed)
        net = df.MLP.from_torch(model, X[:500], levels=levels)
        float_forward, how = net.float_forward, "PyTorch"
        import torch
        with torch.no_grad():
            torch_out = model(torch.tensor(X[:200]))
        assert np.abs(torch_out.numpy() - net.float_forward(X[:200])).max() < 1e-9      # the lowering equals torch's conv
    except ImportError:
        layers = train_numpy(X, Y, args.seed)
        net = df.MLP.from_numpy(layers, X[:500], levels=levels)
        float_forward, how = net.float_forward, "NumPy"

    Xt, Yt = make_data(2000, rng)
    float_acc = float((float_forward(Xt).argmax(1) == Yt).mean())
    int_pred = np.argmax(net.reference(net.encode(Xt))["y"], axis=1)
    int_acc = float((int_pred == Yt).mean())
    print(f"trained with {how}: {net.n_layers} integer layers ({net.n_in} inputs, {net.weights[0].shape[0]} conv outputs, "
          f"{net.n_out} classes)")
    print(f"accuracy on 2000 fresh glyphs: float model {float_acc:.3f}, integer model {int_acc:.3f}, "
          f"float vs integer class agreement {float((int_pred == float_forward(Xt).argmax(1)).mean()):.3f} "
          "(integer reference only; DOOM is run on the test glyphs below)")
    rep = net.quantization_report(X[:500])
    print(f"quantisation (calibration glyphs): max |decoded integer output - float output| {rep['max_abs_error']:.3f} "
          f"(float outputs reach {rep['float_max_abs']:.2f}); layer peaks "
          + ", ".join(f"{d['layer']} {d.get('gate_usage', d['partial_sum_usage']):.0%}" for d in rep["layers"])
          + " of their exact range; warnings: " + (", ".join(rep["warnings"]) or "none"))
    plan = df.plan(net)
    print(f"DOOM map: {plan['dolls']} voodoo dolls, {plan['sectors']} sectors, fits={plan['fits']}")

    tx, ty = make_data(args.glyphs, np.random.default_rng(args.seed + 1000))
    rows, ok, t0 = [], True, time.time()
    with df.compile(net) as brain:
        for g, t in zip(tx, ty):
            xi = net.encode(g)
            ref_cls = int(np.argmax(net.reference(xi)["y"]))
            try:
                doom_cls = net.predict(brain, xi)            # DOOM computes y (checked); Python takes the argmax
                exact = True
            except df.MismatchError as e:
                print("MISMATCH:", e)
                doom_cls, exact, ok = -1, False, False
            rows.append((g, int(t), int(float_forward(g[None])[0].argmax()), ref_cls, doom_cls, exact))
    print(f"{len(rows)} glyph passes in DOOM in {time.time() - t0:.0f}s\n")
    print("  glyph        truth  float  integer  DOOM   DOOM == integer reference")
    for g, t, f, i, d, ex in rows:
        art = show(g)
        print(f"  {art[0]}     {NAMES[t]:>5} {NAMES[f]:>6} {NAMES[i]:>8} {NAMES[d] if d >= 0 else '?':>5}   "
              f"{'EXACT' if ex else 'MISMATCH'}")
        for r in art[1:]:
            print(f"  {r}")
        print()
    n = len(rows)
    print(f"agreement with the truth: float {sum(r[2] == r[1] for r in rows)}/{n}, integer "
          f"{sum(r[3] == r[1] for r in rows)}/{n}, DOOM {sum(r[4] == r[1] for r in rows)}/{n}")
    print(f"float vs integer class agreement {sum(r[2] == r[3] for r in rows)}/{n} (quantisation, not a DOOM error); "
          f"DOOM == integer reference (all four outputs and every hidden value) on {sum(r[5] for r in rows)}/{n} glyphs")
    print("ALL EXACT" if ok else "FAILED: DOOM differs from the integer reference")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
