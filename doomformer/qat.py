"""Quantisation-aware training for `MLP.from_torch` (needs torch).

    qmodel = df.qat.fit(model, loader, levels=4, steps=3000)        # model: a trained nn.Sequential
    net = df.MLP.from_torch(qmodel, calib_x)                         # the staircases and the input quantiser carry over
    df.qat.lowering_gap(qmodel, net, x, y)                           # torch staircase model vs the integer net

`fit` copies the model, puts a K-level `Stair` in place of every activation (ReLU, ReLU6, Hardtanh, Sigmoid, Tanh,
GELU, SiLU) and an `InputStair` in front, and fine-tunes with a straight-through gradient: the forward pass computes
the staircase the integer net will compute, the backward pass treats it as the identity (the recipe of
training/train_mnist.py; ste="clip" uses the gradient of the clipped activation instead). The step of every
staircase is set from calibration rows before training and again every `refresh` steps (ReLU: the squared-error
optimal `a * max / K`, see `quantize.relu_step`; other activations: `quantize.fit_staircase`), the same rules as
`MLP.from_numpy(calibrate="mse")`. BatchNorm layers train with frozen running statistics (eval mode) so that the
folded layer is the trained one; Dropout stays active while training.

`from_torch` reads the thresholds, the offset and the step of every `Stair` and the step of the `InputStair`, so the
integer net computes the same staircases the model was trained with. What differs is the rounding of the weights to
integers (|w| <= 255 per layer) and of the thresholds, and the integer input encoding (round(x / (D / 16)) before the
input staircase, which differs from the float input staircase only within 1/32 of a level boundary): `lowering_gap`
measures it.
"""
from __future__ import annotations

import copy
import math

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from .interop import Act, Staircase, resolve_act
from .quantize import EQUALISE_Q, fit_staircase, input_steps, relu_step


def _base(spec):
    """The torch function of an activation spec (name, or ("hardtanh", lo, hi))."""
    if isinstance(spec, (tuple, list)):
        lo, hi = float(spec[1]), float(spec[2])
        return lambda h: F.hardtanh(h, lo, hi)
    return {"relu": F.relu, "relu6": F.relu6, "sigmoid": torch.sigmoid, "tanh": torch.tanh,
            "gelu": F.gelu, "gelu_tanh": lambda h: F.gelu(h, approximate="tanh"), "silu": F.silu}[spec]


def _spec_of(m) -> object | None:
    """The activation spec of an activation module, None for any other module."""
    if isinstance(m, nn.ReLU):
        return "relu"
    if isinstance(m, nn.ReLU6):
        return "relu6"
    if isinstance(m, nn.Hardtanh):
        return ("hardtanh", float(m.min_val), float(m.max_val))
    if isinstance(m, nn.Sigmoid):
        return "sigmoid"
    if isinstance(m, nn.Tanh):
        return "tanh"
    if isinstance(m, nn.GELU):
        return "gelu_tanh" if m.approximate == "tanh" else "gelu"
    if isinstance(m, nn.SiLU):
        return "silu"
    return None


class InputStair(nn.Module):
    """Quantise the input to clip(round(x / step), -K or 0, K) * step per feature (half away from zero, as the integer
    net's input staircase does), with an identity gradient. `step` has the shape of one sample; `signed` selects
    levels -K..K (else 0..K)."""

    def __init__(self, step, levels: int, signed: bool):
        super().__init__()
        self.register_buffer("step", torch.as_tensor(step, dtype=torch.float32).clone())
        self.levels, self.signed = int(levels), bool(signed)

    def forward(self, x):
        step = self.step.to(x.dtype)
        n = torch.clamp(torch.floor(x.detach().abs() / step + 0.5), 0, self.levels)
        q = (torch.sign(x.detach()) * n if self.signed else torch.where(x.detach() > 0, n, torch.zeros_like(n))) * step
        return x + (q - x).detach()

    def extra_repr(self):
        return f"levels={self.levels}, signed={self.signed}, shape={tuple(self.step.shape)}"


class Stair(nn.Module):
    """A K-level staircase for one activation: value = lo + step * #{k : h >= thr[k]} (exactly what DOOM counts), with
    a straight-through gradient. Until `calibrate` has run it is the float activation."""

    def __init__(self, levels: int, act="relu", ste: str = "identity"):
        super().__init__()
        if ste not in ("identity", "clip"):
            raise ValueError("ste must be 'identity' or 'clip'")
        self.levels, self.act, self.ste = int(levels), act, ste
        self.register_buffer("thr", torch.zeros(self.levels))
        self.register_buffer("lo", torch.zeros(()))
        self.register_buffer("step", torch.zeros(()))
        self.register_buffer("ready", torch.zeros((), dtype=torch.bool))

    def calibrate(self, h, calibrate: str = "mse") -> None:
        """Set thresholds / offset / step from the pre-activations h (a tensor of any shape)."""
        v = h.detach().double().cpu().numpy().ravel()
        if self.act == "relu":
            D = relu_step(v, self.levels, calibrate, floor=1e-12)
            thr, lo, step = (np.arange(1, self.levels + 1) - 0.5) * D, 0.0, D
        else:
            st = fit_staircase(resolve_act(self.act), v, self.levels, calibrate, eps=1e-6)
            thr, lo, step = np.asarray(st), st.lo, st.step
        self.thr.copy_(torch.as_tensor(thr, dtype=torch.float32))
        self.lo.fill_(float(lo))
        self.step.fill_(float(step))
        self.ready.fill_(True)

    def as_act(self) -> Act:
        """The `interop.Act` a lowering uses: thresholds fixed, float function = the activation this replaced."""
        if not bool(self.ready):
            raise ValueError("this Stair has no thresholds yet: run qat.fit (or Stair.calibrate)")
        base = resolve_act(self.act)
        thr = self.thr.detach().double().cpu().numpy()
        lo, step = float(self.lo), float(self.step)
        st = Staircase(thr, lo, step, (float(thr[0]), float(thr[-1])), base)
        return Act("staircase", f"stair({base.name}, {self.levels})", fn=base, stair=st)

    def forward(self, h):
        fn = _base(self.act)
        if not bool(self.ready):
            return fn(h)
        counts = torch.searchsorted(self.thr.to(h.dtype), h.detach().contiguous(), right=True)
        q = self.lo.to(h.dtype) + self.step.to(h.dtype) * counts.to(h.dtype)
        soft = h if self.ste == "identity" else torch.clamp(fn(h), float(self.lo), float(self.lo + self.step * self.levels))
        return soft + (q - soft).detach()

    def extra_repr(self):
        return f"{self.act}, levels={self.levels}, ste={self.ste}"


def _convert(module: nn.Module, levels: list[int], ste: str, stairs: list[Stair]) -> nn.Module:
    """A copy of the Sequential with Stair modules in place of the activations (consuming `levels` in order)."""
    if isinstance(module, nn.Sequential):
        return nn.Sequential(*[_convert(m, levels, ste, stairs) for m in module])
    spec = _spec_of(module)
    if spec is None:
        return module
    st = Stair(levels[len(stairs) + 1], spec, ste)
    stairs.append(st)
    return st


def _count_acts(module: nn.Module) -> int:
    if isinstance(module, nn.Sequential):
        return sum(_count_acts(m) for m in module)
    return int(_spec_of(module) is not None)


def _calibrate(model: nn.Module, stairs: list[Stair], calib: torch.Tensor, calibrate: str) -> None:
    """Set every staircase from the pre-activations it sees on `calib`, front to back (earlier staircases quantise
    what the later ones see)."""
    for st in stairs:
        seen = []
        hook = st.register_forward_pre_hook(lambda m, args: seen.append(args[0].detach()))
        was_ready = st.ready.clone()
        st.ready.fill_(False)
        try:
            with torch.no_grad():
                model(calib)
        finally:
            hook.remove()
        st.ready.copy_(was_ready)
        st.calibrate(seen[0], calibrate)


def _leaves(module: nn.Module):
    if isinstance(module, nn.Sequential):
        for m in module:
            yield from _leaves(m)
    else:
        yield module


_PASS = (nn.Flatten, nn.Identity, nn.AvgPool1d, nn.AvgPool2d, nn.MaxPool1d, nn.MaxPool2d, nn.AdaptiveAvgPool1d,
         nn.AdaptiveAvgPool2d)
_WEIGHTED = (nn.Linear, nn.Conv1d, nn.Conv2d)


@torch.no_grad()
def _equalise(qmodel: nn.Module, calib: torch.Tensor, q: float = EQUALISE_Q) -> int:
    """Rescale the channels (units) behind every ReLU `Stair` towards the same q-quantile of their positive
    pre-activations: the channel's incoming weights and bias (or its BatchNorm gain and shift) times c, the next
    layer's columns divided by c, c within 1/4..4. The float function is unchanged; pooling and flattening between
    the two layers are fine (positive per-channel factors commute with them). A ReLU whose neighbours are not a
    Linear/Conv pair (a BatchNorm after it, a grouped conv next) is left alone. Returns the number of staircases
    rescaled. `quantize.equalise_layers` does the same on the lowered float layers."""
    mods = list(_leaves(qmodel))
    done = 0
    for i, m in enumerate(mods):
        if not (isinstance(m, Stair) and m.act == "relu"):
            continue
        j = i - 1
        bn = mods[j] if isinstance(mods[j], (nn.BatchNorm1d, nn.BatchNorm2d)) else None
        prev = mods[j - 1 if bn is not None else j] if j >= 1 else None
        k = i + 1
        while k < len(mods) and isinstance(mods[k], _PASS):
            k += 1
        nxt = mods[k] if k < len(mods) else None
        if not isinstance(prev, _WEIGHTED) or not isinstance(nxt, _WEIGHTED):
            continue
        if isinstance(nxt, (nn.Conv1d, nn.Conv2d)) and nxt.groups != 1 or (bn is not None and bn.weight is None):
            continue
        seen = []
        hook = m.register_forward_pre_hook(lambda mod, args: seen.append(args[0].detach()))
        try:
            qmodel(calib)
        finally:
            hook.remove()
        h = seen[0]
        C = h.shape[1]
        hv = h.transpose(0, 1).reshape(C, -1)
        mq = torch.zeros(C, dtype=h.dtype)
        for ch in range(C):
            pos = hv[ch][hv[ch] > 0]
            if len(pos) > 10:
                mq[ch] = torch.quantile(pos, q)
        live = mq > 0
        if not bool(live.any()):
            continue
        c = torch.ones_like(mq)
        c[live] = (mq[live].mean() / mq[live]).clamp(0.25, 4.0)
        scale_in = bn if bn is not None else prev
        w = scale_in.weight
        w.mul_(c.view(-1, *([1] * (w.dim() - 1))))
        if scale_in.bias is not None:
            scale_in.bias.mul_(c)
        wn = nxt.weight
        if isinstance(nxt, nn.Linear):
            if wn.shape[1] % C:
                raise ValueError("internal: the next Linear does not divide into the channels")
            wn.view(wn.shape[0], C, -1).div_(c.view(1, C, 1))
        else:
            wn.div_(c.view(1, C, *([1] * (wn.dim() - 2))))
        done += 1
    return done


def _frozen_bn_train(model: nn.Module) -> None:
    model.train()
    for m in model.modules():
        if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
            m.eval()


def _default_loss(out, y):
    if y.dtype in (torch.int64, torch.int32, torch.uint8):
        return F.cross_entropy(out, y.long())
    return F.mse_loss(out, y.to(out.dtype))


def fit(model: nn.Module, loader, levels=8, steps: int = 2000, *, lr: float = 5e-4, loss_fn=None, calib_x=None,
        refresh: int | None = None, ste: str = "identity", calibrate: str = "mse", signed_input: bool | None = None,
        distill: float = 0.0, temperature: float = 2.0, equalise: bool = False, seed: int = 0, log=None) -> nn.Sequential:
    """Fine-tune `model` (a trained nn.Sequential of Linear / Conv / pooling / BatchNorm / Flatten / activations, the
    modules `MLP.from_torch` lowers) with K-level staircases and return the quantisation-aware copy (eval mode).

    loader: an iterable of (x, y) batches that can be iterated again and again (a DataLoader, a list), cycled for
    `steps` optimiser steps. levels: one int for every staircase, or the input's followed by one per activation
    (as for `from_numpy`); the last layer has no activation. loss_fn(out, y): default cross-entropy for integer
    labels, mean squared error otherwise. calib_x: rows that set the staircase steps and the input steps (default: the
    first batches of the loader, up to 1,000 rows). refresh: steps between re-fits of the staircase steps (default
    steps / 15, the recipe's epoch). ste: "identity" or "clip", see the module doc. calibrate: "mse" (default) or
    "max" for the step rule. signed_input: levels -K..K for the input (default: iff calib_x has a negative value).
    distill: weight a in [0, 1] of a distillation term: the loss becomes (1 - a) * loss_fn + a * T^2 * KL(softmax(teacher
    / T) || softmax(out / T)) with the teacher = `model` itself (the float model before quantisation) and
    T = temperature; for classifiers (2-D outputs) only, 0 (default) switches it off. lr: Adam learning rate,
    cosine-decayed to lr / 50. equalise: rescale the channels behind every ReLU towards equal 99th percentiles before
    training and at every refresh (function-preserving, `_equalise`). log(step, loss): called every refresh if given."""
    if steps < 1:
        raise ValueError("steps must be >= 1")
    torch.manual_seed(seed)
    qm = copy.deepcopy(model)
    teacher = copy.deepcopy(model).eval() if distill else None
    n_act = _count_acts(qm)
    if isinstance(levels, (int, np.integer)):
        Ks = [int(levels)] * (n_act + 1)
    else:
        Ks = [int(k) for k in levels]
        if len(Ks) != n_act + 1:
            raise ValueError(f"levels: this model has {n_act} activations (plus the input): need {n_act + 1} levels, got {len(Ks)}")
    if min(Ks) < 1:
        raise ValueError("levels must be >= 1")
    if iter(loader) is loader:
        raise ValueError("loader is a one-shot iterator (a generator): fit cycles it for `steps` steps, so pass something "
                         "that can be iterated again (a DataLoader, a list of batches)")
    stairs: list[Stair] = []
    body = _convert(qm, Ks, ste, stairs)
    if calib_x is None:
        rows, n = [], 0
        for x, _ in loader:
            rows.append(x.detach())
            n += len(x)
            if n >= 1000:
                break
        if not rows:
            raise ValueError("loader yielded no batches (empty): nothing to calibrate or train on")
        calib_x = torch.cat(rows)[:1000]
    calib = torch.as_tensor(calib_x).detach()
    dtype = next((p.dtype for p in body.parameters()), calib.dtype)
    calib = calib.to(dtype)
    signed = bool((calib < 0).any()) if signed_input is None else bool(signed_input)
    D_in = input_steps(calib.double().numpy().reshape(len(calib), -1), Ks[0], signed, calibrate)
    qmodel = nn.Sequential(InputStair(D_in.reshape(calib.shape[1:]), Ks[0], signed), body)
    qmodel.eval()
    _calibrate(qmodel, stairs, calib, calibrate)
    if equalise and _equalise(qmodel, calib):
        _calibrate(qmodel, stairs, calib, calibrate)
    params = [p for p in qmodel.parameters() if p.requires_grad]
    opt = torch.optim.Adam(params, lr)
    refresh = max(1, steps // 15) if refresh is None else int(refresh)
    _frozen_bn_train(qmodel)
    step, loss_fn = 0, loss_fn or _default_loss
    while step < steps:
        seen_batch = False
        for x, y in loader:
            seen_batch = True
            if step >= steps:
                break
            if step and step % refresh == 0:
                qmodel.eval()
                if equalise:
                    _equalise(qmodel, calib)
                _calibrate(qmodel, stairs, calib, calibrate)
                _frozen_bn_train(qmodel)
            for g in opt.param_groups:
                g["lr"] = lr * (0.02 + 0.98 * 0.5 * (1 + math.cos(math.pi * step / steps)))
            out = qmodel(x.to(dtype))
            loss = loss_fn(out, y)
            if teacher is not None:
                with torch.no_grad():
                    t = teacher(x.to(dtype))
                kl = F.kl_div(F.log_softmax(out / temperature, -1), F.softmax(t / temperature, -1), reduction="batchmean")
                loss = (1 - distill) * loss + distill * temperature ** 2 * kl
            opt.zero_grad()
            loss.backward()
            opt.step()
            step += 1
            if log is not None and step % refresh == 0:
                log(step, float(loss))
        if not seen_batch:
            raise ValueError("loader yielded no batches (empty, or exhausted after the first pass)")
    qmodel.eval()
    _calibrate(qmodel, stairs, calib, calibrate)
    return qmodel


def lowering_gap(qmodel: nn.Module, net, x, y=None, float_model: nn.Module | None = None) -> dict:
    """How far the integer net `net` (= MLP.from_torch(qmodel, ...)) is from the torch model it was lowered from, on
    the rows x (N, *sample): max / mean absolute difference of the outputs (decoded integer outputs against the torch
    staircase model's), the share of rows with the same argmax, and with labels y the accuracy of the torch staircase
    model and of the integer net (and of `float_model`, the model before `fit`, if given). NumPy integer reference;
    DOOM is not involved."""
    x = torch.as_tensor(x)
    with torch.no_grad():
        dtype = next(qmodel.parameters()).dtype
        out_q = qmodel(x.to(dtype)).double().numpy()
        out_f = None if float_model is None else float_model(x.to(next(float_model.parameters()).dtype)).double().numpy()
    xn = x.double().numpy()
    out_i = net.decode(net.reference(net.encode(xn))["y"])
    rep = dict(n=len(out_q), max_abs_error=float(np.abs(out_i - out_q).max()), mean_abs_error=float(np.abs(out_i - out_q).mean()),
               float_max_abs=float(np.abs(out_q).max()))
    if out_q.shape[1] > 1:
        rep["argmax_agreement"] = float((out_i.argmax(1) == out_q.argmax(1)).mean())
        if y is not None:
            y = np.asarray(y)
            rep |= dict(accuracy_staircase_model=float((out_q.argmax(1) == y).mean()),
                        accuracy_integer=float((out_i.argmax(1) == y).mean()))
            if out_f is not None:
                rep["accuracy_float"] = float((out_f.argmax(1) == y).mean())
    return rep
