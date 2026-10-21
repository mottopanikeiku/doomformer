"""Train the MNIST MLP that `doomformer mnist` and `doomformer draw` run in one DOOM map (needs the `train` extra).

Pixels are binarised (> 127 of 255), so every ink pixel is one input doll. 784 -> 64 -> 10 with ReLU. Every batch
holds 128 MNIST training digits (half of them randomly shifted (+-3 px), scaled (0.75-1.15), rotated (+-15 deg),
sheared and made thicker or thinner (3x3 max/min filter) before binarising) and 128 synthetic *walked* digits, drawn
from 450,000 painted with the DrawPad canvas brush: 150,000 `drawpad.walked_strokes` (the 3 DIGIT_STROKES variants per
digit that `doomformer draw --digit` walks) and 75,000 `varied_strokes` (those and 3 to 6 EXTRA_STROKES variants per
digit: narrow, open and overlapping 0s, curly 2s, open 4s, single-stroke 8s; every stroke moved and its ends cut or
extended a little; sizes, widths, rotations, slants, wobble and placement over wider ranges), each painted along the
stroke polylines (`rasterise`) and at every tic of a `SimWalker` walking it with the stand-in player's controller and
DOOM's measured player physics (`walked_pixels`).
So it reads digits drawn by walking on the DOOM canvas as well as MNIST's. 20 float epochs (Adam, lr 2e-3), then 15
quantisation-aware epochs (lr 5e-4, cosine decay to 1e-5) whose hidden activation is the 8-level staircase
clip(round(relu(h) / D), 0, 8) (straight-through gradient). Each QAT epoch starts with `equalise` (hidden units rescaled
to equal 99th percentiles) and `step_size` (the D with the least squared staircase error), both on 5,000 calibration
digits (2,500 MNIST + 2,500 walked). `MLP.from_numpy` turns that staircase into the integer net's thresholds; the
winner layers (`drawpad.with_argmax`: pairwise comparisons, win counts, one-hot) then let DOOM itself pick the digit.

Writes models/mnist/mlp.npz (MLP.save, the 4-matrix winner net) and models/mnist/digits.npz (the first 100 test
digits, bit-packed, with labels) and prints the integer model's accuracy (NumPy integer reference) on all 10,000
MNIST test digits and on 5,000 held-out walked digits (drawpad.walked_strokes with another random stream than
training; painted along the strokes and walked by a SimWalker), before and after the winner layers' rescaling of the
score layer. Every random draw is seeded: the same arguments write the same files byte for byte.

usage: train_mnist.py [--seed 0] [--hidden 64] [--float-epochs 20] [--qat-epochs 15] [--out models/mnist]
"""
from __future__ import annotations

import argparse
import gzip
import math
import urllib.request
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

import doomformer as df
from doomformer import drawpad

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "build" / "mnist"
URL = "https://ossci-datasets.s3.amazonaws.com/mnist/"
FILES = ("train-images-idx3-ubyte", "train-labels-idx1-ubyte", "t10k-images-idx3-ubyte", "t10k-labels-idx1-ubyte")
HIDDEN, LEVELS, FLOAT_EPOCHS, QAT_EPOCHS, LR, QAT_LR = 64, 8, 20, 15, 2e-3, 5e-4
N_PAD, N_VARIED, WALKED_BATCH, N_HELD_OUT, CHUNK = 150000, 75000, 128, 5000, 2500
EQUALISE_Q = 0.99
PAD_STREAM, VARIED_STREAM = 2, 3   # training digits use seeds [seed, stream, chunk]: never a `draw --digit` stream
HELD_OUT_SEED = 1_000_003          # walked-digit test set: its own random stream, never used in training


def load(name: str) -> np.ndarray:
    f = DATA / f"{name}.gz"
    if not f.is_file():
        DATA.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(URL + f.name, f)
    raw = gzip.open(f).read()
    return np.frombuffer(raw, np.uint8, offset=16).reshape(-1, 784) if "images" in name else \
        np.frombuffer(raw, np.uint8, offset=8)


# --- more ways to walk a digit ----------------------------------------------------------------------------------------
def _extra_strokes() -> dict[int, list[list[list[tuple[float, float]]]]]:
    """More stroke variants in DIGIT_STROKES' frame (20 units tall, about 12 wide, y down): narrow, open and
    overlapping 0s, curly and flat 2s, open and closed 4s, single-stroke 8s, clockwise 9s and so on."""
    A = drawpad._arc
    return {
        0: [[A(6, 10, 3.5, 10, 90, 450)], [A(6, 10, 5, 10, 70, 450)], [A(6, 10, 5.5, 10, 60, -310)],
            [A(6, 10, 5, 10, 100, 420)], [A(6, 10, 5, 10, -90, 270)], [A(6, 10, 4, 10, 120, 470)]],
        1: [[[(6, 0), (6.5, 20)]], [[(2, 5), (7, 0), (7, 20)]], [[(4, 2), (6, 0), (6, 20)]]],
        2: [[A(6, 5, 5, 5, 160, -30) + [(1, 19), (2.5, 16.5), (4.5, 17.5), (3, 20), (12, 20)]],
            [A(6, 5, 5.5, 5, 170, -30) + [(1, 20)], [(3, 19.5), (8, 19.5), (12, 19)]],
            [A(6, 6, 5.5, 6, 150, -10) + A(9, 20, 9, 8, 180, 120) + [(5, 20), (12, 20)]]],
        3: [[[(1, 0), (11, 0), (5, 8)] + A(6, 14, 5.5, 6, 90, -150)],
            [A(6, 4.5, 5, 4.5, 150, -90) + [(4, 9.5)] + A(6, 14.5, 5.5, 5.5, 90, -160)],
            [A(6, 5, 5, 5, 140, -90) + A(6, 15, 5.5, 5, 90, -180)]],
        4: [[[(4, 0), (0, 12), (12, 12)], [(10, 3), (9, 20)]],
            [[(12, 14), (0, 14), (8, 0), (8, 20)]],
            [[(1, 0), (1, 10), (11, 10)], [(10, 1), (10, 20)]]],
        5: [[[(11, 0), (2, 0), (1, 8), (5, 7)] + A(6, 13.5, 5.5, 6.5, 110, -150)],
            [[(2, 0), (1, 9)] + A(5.5, 14.5, 5.5, 5.5, 140, -170), [(2, 0), (11, 0.5)]],
            [[(11, 0), (3, 0), (2, 9)] + A(6, 14.5, 6, 5.5, 140, -130)]],
        6: [[[(10, 0), (4, 4), (1, 11), (1.5, 16), (6, 20), (10.5, 16), (9, 11), (4, 11), (1.5, 14)]],
            [[(7, 0), (2, 9)] + A(6, 14, 5, 6, 180, 530)],
            [A(10, 9, 9, 9, 110, 180) + A(6, 14.5, 5, 5.5, 180, 520)]],
        7: [[[(0, 3), (0, 0), (12, 0), (5, 20)]],
            [[(0, 0), (12, 0), (9, 6), (6, 13), (5, 20)]],
            [[(0, 0), (12, 0)], [(12, 0), (4, 20)]]],
        8: [[A(6, 4.5, 3.5, 4.5, 90, 450), A(6, 14.5, 6, 5.5, 90, 450)],
            [A(6, 5, 5, 5, 60, 270) + A(6, 15, 5.5, 5, 90, -270) + A(6, 5, 5, 5, -90, 70)],
            [A(6, 5, 5, 5, 90, -90) + A(6, 15, 5.5, 5, 90, 450) + A(6, 5, 5, 5, -90, -270)]],
        9: [[A(6, 5, 5, 5, 0, -360) + [(11, 5), (10, 20)]],
            [A(5.5, 5, 5, 5, 20, 380) + [(10, 7), (8, 20)]],
            [A(6, 6, 5.5, 6, 0, 360) + [(11.5, 6), (11, 20)]]],
    }


EXTRA_STROKES = _extra_strokes()


def _cut_or_extend(p: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Stroke p with each end cut short or extended along its direction by up to 1.2 units (at most 6 % of its
    length): the gaps and overlaps of a hand that does not quite meet its starting point."""
    q = drawpad._densify(p, 0.25)
    length = 0.25 * (len(q) - 1)
    for end in (0, 1):
        a = rng.uniform(-1.2, 1.2) * min(1.0, 0.06 * length / 1.2)
        if end:
            q = q[::-1]
        if a > 0:
            q = q[min(int(a / 0.25), len(q) - 2):]
        elif a < 0:
            d = q[0] - q[min(4, len(q) - 1)]
            q = np.concatenate([[q[0] - a * d / max(np.linalg.norm(d), 1e-9)], q])
        if end:
            q = q[::-1]
    return q


def varied_strokes(digit: int, rng: np.random.Generator) -> list[np.ndarray]:
    """Like drawpad.walked_strokes over wider ranges: a variant from DIGIT_STROKES + EXTRA_STROKES, every stroke
    moved by about 0.5 units and scaled by about 6 % on its own and its ends cut or extended, then the whole digit
    13-22 pixels tall, 0.7-1.3 times as wide, rotated +-16 deg, slanted (shear +-0.35), wobbled (0.3-1.0 px) and
    placed off-centre by up to 4 pixels."""
    pool = drawpad.DIGIT_STROKES[digit] + EXTRA_STROKES[digit]
    v = pool[rng.integers(len(pool))]
    h = rng.uniform(13, 22) / 20
    sx, th, shear = h * rng.uniform(0.7, 1.3), math.radians(rng.uniform(-16, 16)), rng.uniform(-0.35, 0.35)
    rot = np.array([[math.cos(th), -math.sin(th)], [math.sin(th), math.cos(th)]])
    wobble = rng.uniform(0.3, 1.0)
    out = []
    for stroke in v:
        p = (np.asarray(stroke, dtype=np.float64) - (6, 10)) * (1 + rng.normal(0, 0.06, 2)) + rng.normal(0, 0.5, 2)
        p = _cut_or_extend(p, rng)
        p = np.stack([p[:, 0] * sx + shear * p[:, 1] * h, p[:, 1] * h], 1) @ rot.T
        p = drawpad._densify(p, 0.5)
        knots = max(2, int(len(p) / 8) + 2)
        noise = rng.normal(0, wobble, (knots, 2))
        t = np.linspace(0, knots - 1, len(p))
        out.append(p + np.stack([np.interp(t, np.arange(knots), noise[:, k]) for k in (0, 1)], 1))
    allp = np.concatenate(out)
    lo, hi = allp.min(0), allp.max(0)
    centre = np.array([drawpad.SIZE / 2, drawpad.SIZE / 2]) + rng.uniform(-4, 4, 2)
    shift = np.clip(centre - (lo + hi) / 2, 1.5 - lo, drawpad.SIZE - 1.5 - hi)
    return [p + shift for p in out]


def _walked_chunk(job) -> tuple[np.ndarray, np.ndarray]:
    seed, k, n, kind = job
    rng = np.random.default_rng(list(seed) + [k])
    ideal, sim = [], []
    for i in range(n):
        strokes = (drawpad.walked_strokes if kind == "pad" else varied_strokes)(i % 10, rng)
        ideal.append(drawpad.rasterise(strokes))
        sim.append(drawpad.walked_pixels(strokes))
    return np.stack(ideal), np.stack(sim)


def walked(n: int, seed: list[int], kind: str = "pad") -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """n walked digits (labels cycle 0..9; n a multiple of CHUNK), kind "pad" (drawpad.walked_strokes, what
    `doomformer draw --digit` walks) or "varied" (varied_strokes), each painted with the DrawPad brush two ways: along
    the stroke polylines (`rasterise`) and at every tic of a SimWalker walking them the way the stand-in player does
    in DOOM (`walked_pixels`). Chunks of CHUNK digits with seeds seed + [chunk] (the same digits for any number of
    worker processes). Returns (n, 784) uint8 pixels painted along the strokes, the same by the SimWalker, labels."""
    import multiprocessing
    from concurrent.futures import ProcessPoolExecutor
    jobs = [(tuple(seed), k, CHUNK, kind) for k in range(n // CHUNK)]
    with ProcessPoolExecutor(min(8, len(jobs)), mp_context=multiprocessing.get_context("spawn")) as ex:
        parts = list(ex.map(_walked_chunk, jobs))
    return np.concatenate([p[0] for p in parts]), np.concatenate([p[1] for p in parts]), np.arange(n) % 10


def augment(gray: torch.Tensor, g: torch.Generator) -> torch.Tensor:
    """Random shift/scale/rotation/shear and stroke thickness on (B, 784) grayscale in [0, 1], then binarised."""
    B = len(gray)
    img = gray.view(B, 1, 28, 28)
    u = lambda lo, hi: lo + (hi - lo) * torch.rand(B, generator=g)
    s, th, sh = u(0.75, 1.15), u(-15, 15) * math.pi / 180, u(-0.2, 0.2)
    tx, ty = u(-3, 3) / 14, u(-3, 3) / 14
    c, si = torch.cos(th), torch.sin(th)
    theta = torch.stack([torch.stack([c / s, (-si + sh) / s, tx], 1), torch.stack([si / s, c / s, ty], 1)], 1)
    img = F.grid_sample(img, F.affine_grid(theta, img.shape, align_corners=False), align_corners=False)
    k = torch.rand(B, generator=g)
    thick = F.max_pool2d(img, 3, 1, 1)
    thin = -F.max_pool2d(-img, 2, 1, 0)
    thin = F.pad(thin, (0, 1, 0, 1))
    img = torch.where((k < 0.3).view(B, 1, 1, 1), thick, torch.where((k > 0.85).view(B, 1, 1, 1), thin, img))
    return (img.view(B, 784) > 0.5).float()


@torch.no_grad()
def equalise(l1: torch.nn.Linear, l2: torch.nn.Linear, calib: torch.Tensor) -> None:
    """Rescale every hidden unit j (incoming weights x c_j, outgoing / c_j; ReLU is positively homogeneous, so the
    float model is unchanged) towards the same 99th percentile of its positive activations on the calibration digits
    (factors kept within 1/4..4, so a nearly dead unit cannot inflate the weight scale). The hidden staircase has ONE
    step size per layer; equal ranges make it fit every unit."""
    h = l1(calib)
    m = torch.stack([torch.quantile(h[h[:, j] > 0, j], EQUALISE_Q) if (h[:, j] > 0).sum() > 10 else h.new_zeros(())
                     for j in range(h.shape[1])])
    live = m > 0
    c = torch.ones_like(m)
    c[live] = (m[live].mean() / m[live]).clamp(0.25, 4.0)
    l1.weight.mul_(c[:, None])
    l1.bias.mul_(c)
    l2.weight.div_(c[None, :])


@torch.no_grad()
def step_size(h: torch.Tensor) -> float:
    """The staircase step D (levels at D, 2D, .. LEVELS*D; higher activations stay at the top level) with the least
    squared error between relu(h) and clip(round(relu(h) / D), 0, LEVELS) * D over the calibration pre-activations h:
    D = a * max(h) / LEVELS for a in 0.25..1 (61 steps). The largest activations are rare; a step from the maximum
    alone (a = 1) leaves most activations on the lowest levels."""
    r, top = torch.relu(h), h.max().item() / LEVELS
    errs = [(((r - torch.clamp(torch.round(r / (a * top)), 0, LEVELS) * a * top) ** 2).mean().item(), a * top)
            for a in np.linspace(0.25, 1.0, 61)]
    return min(errs)[1]


class Staircase(torch.autograd.Function):
    """clip(round(relu(h) / D), 0, K) * D forward, identity backward."""

    @staticmethod
    def forward(ctx, h, D, K):
        return torch.clamp(torch.round(torch.relu(h) / D), 0, K) * D

    @staticmethod
    def backward(ctx, g):
        return g, None, None


def accuracy(net, bits: np.ndarray, labels: np.ndarray) -> float:
    """Share of digits whose one-hot (winner net) or first maximal score (score net) is the label."""
    y = net.reference(net.encode(bits.astype(np.float64)))["y"]
    return float((np.argmax(y, 1) == labels).mean())


def main() -> None:
    """Train, quantise, add the winner layers, save, report (see the module docstring)."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--hidden", type=int, default=HIDDEN)
    ap.add_argument("--float-epochs", type=int, default=FLOAT_EPOCHS)
    ap.add_argument("--qat-epochs", type=int, default=QAT_EPOCHS)
    ap.add_argument("--out", type=Path, default=ROOT / "models" / "mnist")
    args = ap.parse_args()
    seed = args.seed
    torch.set_num_threads(4)
    Xtr, ytr, Xte, yte = (load(n) for n in FILES)
    gray, Y = torch.tensor(Xtr / 255.0, dtype=torch.float32), torch.tensor(ytr, dtype=torch.long)
    pi, ps, py = walked(N_PAD, [seed, PAD_STREAM], "pad")
    vi, vs, vy = walked(N_VARIED, [seed, VARIED_STREAM], "varied")
    Wx = torch.tensor(np.concatenate([pi, ps, vi, vs]))                      # uint8, 450,000 x 784
    Wy = torch.tensor(np.concatenate([py, py, vy, vy]), dtype=torch.long)
    pick = torch.randperm(len(Wx), generator=torch.Generator().manual_seed(seed))[:2500]
    calib = torch.cat([(gray[:2500] > 127 / 255).float(), Wx[pick].float()])
    torch.manual_seed(seed)
    g = torch.Generator().manual_seed(seed)
    l1, l2 = torch.nn.Linear(784, args.hidden), torch.nn.Linear(args.hidden, 10)
    opt = torch.optim.Adam([*l1.parameters(), *l2.parameters()], LR)
    batches = math.ceil(len(gray) / 128)
    qat_steps = args.qat_epochs * batches
    for epoch in range(args.float_epochs + args.qat_epochs):
        D = None
        if epoch >= args.float_epochs:
            equalise(l1, l2, calib)
            D = step_size(l1(calib))
        perm, wperm = torch.randperm(len(gray), generator=g), torch.randperm(len(Wx), generator=g)
        for k, i in enumerate(range(0, len(gray), 128)):
            if D is not None:                                 # QAT: cosine decay from QAT_LR to QAT_LR / 50
                q = (epoch - args.float_epochs) * batches + k
                for grp in opt.param_groups:
                    grp["lr"] = QAT_LR * (0.02 + 0.98 * 0.5 * (1 + math.cos(math.pi * q / qat_steps)))
            b = perm[i:i + 128]
            x = (gray[b] > 127 / 255).float()
            half = len(b) // 2
            x = torch.cat([augment(gray[b[:half]], g), x[half:]])
            j = (k * WALKED_BATCH) % len(Wx)
            wb = wperm[j:j + WALKED_BATCH]
            x, y = torch.cat([x, Wx[wb].float()]), torch.cat([Y[b], Wy[wb]])
            h = l1(x)
            out = l2(torch.relu(h) if D is None else Staircase.apply(h, D, LEVELS))
            opt.zero_grad()
            F.cross_entropy(out, y).backward()
            opt.step()
    equalise(l1, l2, calib)
    D = step_size(l1(calib))
    bits = Xte > 127
    with torch.no_grad():
        f = lambda x: l2(torch.relu(l1(torch.tensor(x, dtype=torch.float32)))).argmax(1).numpy()
        print(f"float model: MNIST test {(f(bits) == yte).mean():.4f}")
    top = LEVELS * D
    stair = df.staircase(lambda z: np.clip(z, 0.0, top), (0.0, top), LEVELS)      # thresholds (k - 1/2) D
    layers = [(l1.weight.detach().double().numpy(), l1.bias.detach().double().numpy(), stair),
              (l2.weight.detach().double().numpy(), l2.bias.detach().double().numpy())]
    scores = df.MLP.from_numpy(layers, calib.double().numpy(), levels=[1, LEVELS])
    net = drawpad.with_argmax(scores)
    hx, sx, hy = walked(N_HELD_OUT, [HELD_OUT_SEED], "pad")
    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    net.save(out / "mlp.npz")
    np.savez_compressed(out / "digits.npz", bits=np.packbits(bits[:100], axis=1), labels=yte[:100],
                        test_index=np.arange(100))
    unused = int((np.abs(net.weights[0]).sum(0) == 0).sum())
    for name, m in (("integer score net (np.argmax of the scores)", scores),
                    (f"winner net (DOOM's one-hot; score layer x{net.score_scale:.4f})", net)):
        print(f"{name}: MNIST test {accuracy(m, bits, yte):.4f} on 10,000; held-out walked digits "
              f"{accuracy(m, hx, hy):.4f} (strokes) / {accuracy(m, sx, hy):.4f} (SimWalker) on {N_HELD_OUT:,} each")
    print(f"pixels without weights: {unused}")
    print(f"plan: {df.plan(net, mode='map')['map']}")


if __name__ == "__main__":
    main()
