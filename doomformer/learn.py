"""DOOM learns: perceptrons whose weights are ceiling heights, trained by voodoo dolls inside one map.

One run of the map = one training example (x binary features, y in {-1, 0, +1} per head):

  layer "forward"  every active input's dolls climb a corridor of that weight's ceiling gates (thresholds
                   1..K): one +1 (or -1 for the negative twin) on the head's score floor per gate they fit
                   under, so  s_h = sum_i x_i * clamp(w_hi, -K, K).
  layer "learn"    perceptron rule: for every active input i, a doll passes the cap gate (w_hi < CAP to raise,
                   > -CAP to lower), the label gate (y_h = +1 or -1) and the mistake gate (y_h * s_h <= 0) and
                   crosses Ceiling_Raise/LowerByValue(w_hi, 1). y_h = 0 means "no lesson for this head".
Weights are `state`: they start at 0 and survive the per-example rewind, so the map accumulates what it learned;
the cap keeps every ceiling inside [-CAP, CAP] however long it trains.
Python only writes x and y and reads s and w; every multiply, add, comparison and weight update is dolls.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .ops import adc_lanes, fx, neg, output_pair
from .pure import compile_pure
from .tile import Block, Gate, Stage, TileMeta, TileSpec


CAP = 100          # weights live in [-CAP, CAP]: far from engine limits, and far from what the demos reach (<= 16)


def perceptron_specs(n_feat: int, heads: int, K: int, cap: int = CAP):
    """(forward spec, learn spec, state, readout) of the learner; callers may add blocks before compiling."""
    fwd = TileSpec("forward")
    score = [output_pair(fwd, f"s{h}") for h in range(heads)]
    for h in range(heads):
        for i in range(n_feat):
            for sb in (1, -1):                         # corridor under w (adds +1s) and under -w (adds -1s)
                port_w = fwd.port(f"w{h}_{i}" if sb > 0 else f"-w{h}_{i}", "ceil")
                stages = [Stage(Gate(port_w, k, "ceil"), fx(score[h], sb)) for k in range(1, K + 1)]
                fwd.blocks.append(Block(adc_lanes(fwd, f"x{i}", [1], 1), stages))
    lrn = TileSpec("learn")
    for h in range(heads):
        for i in range(n_feat):
            w = output_pair(lrn, f"w{h}_{i}")
            for ys in (1, -1):
                room = Gate(lrn.port(f"-w{h}_{i}" if ys > 0 else f"w{h}_{i}", "ceil"), 1 - cap, "ceil")  # ys*w < cap
                label = Gate(lrn.port(neg(f"y{h}") if ys > 0 else f"y{h}", "floor"), -1, "floor")   # ys*y >= 1
                mistake = Gate(lrn.port(f"s{h}" if ys > 0 else f"-s{h}", "floor"), 0, "floor")      # ys*s <= 0
                lrn.blocks.append(Block(adc_lanes(lrn, f"x{i}", [1], 1),
                                        [Stage(room, []), Stage(label, []), Stage(mistake, fx(w, ys))]))
    state = {f"{p}w{h}_{i}": 0 for h in range(heads) for i in range(n_feat) for p in ("", "-")}
    readout = [f"s{h}" for h in range(heads)] + [f"w{h}_{i}" for h in range(heads) for i in range(n_feat)]
    return fwd, lrn, state, readout


def build_perceptrons(n_feat: int, heads: int, K: int, wad: Path, name: str = "learner", cap: int = CAP) -> TileMeta:
    fwd, lrn, state, readout = perceptron_specs(n_feat, heads, K, cap)
    return compile_pure([fwd, lrn], wad, name, state=state, readout=readout)


@dataclass
class PerceptronRef:
    """Exact integer twin of the DOOM learner."""
    n_feat: int
    heads: int
    K: int
    cap: int = CAP
    w: np.ndarray = field(init=False)

    def __post_init__(self):
        self.w = np.zeros((self.heads, self.n_feat), dtype=np.int64)

    def scores(self, x: np.ndarray) -> np.ndarray:
        return np.clip(self.w, -self.K, self.K) @ x

    def mistake(self, y: int, s: int) -> bool:
        return bool(y) and y * s <= 0

    def step(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        s = self.scores(x)
        for h in range(self.heads):
            if self.mistake(y[h], s[h]):
                self.w[h] = np.clip(self.w[h] + y[h] * x, -self.cap, self.cap)
        return s


def ports_for(x: np.ndarray, y: np.ndarray) -> dict[str, int]:
    out = {}
    for i, v in enumerate(x):
        out[f"x{i}"], out[f"-x{i}"] = int(v), -int(v)
    for h, v in enumerate(y):
        out[f"y{h}"], out[f"-y{h}"] = int(v), -int(v)
    return out
