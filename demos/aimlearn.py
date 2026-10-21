"""DOOM learns to aim: two online perceptrons on arena features, taught by the aimbot every decision.

features (19, binary): one-hot bearing bin of the nearest demon (17), "no enemy" (1), bias (1)
head 0 "fire?":  label +1 if the aimbot fires, else -1
head 1 "turn":   label +1 if the aimbot turns left, -1 if right, 0 (no lesson) when it fires
action, chosen by dolls in the same map: FIRE if s_fire > 0, else TURN LEFT if s_turn >= 0, else TURN RIGHT
(a blank brain spins left and never shoots). Two act dolls read the score floors and add to the "act" light:
+2 if s_fire >= 1; +1 if s_turn <= -1 and s_fire <= 0. So act is the index into policy.ACTIONS ("L", "R", "F").
The scores and act come from the map BEFORE it learns from this decision's label: the marine acts on what it knew.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from doomformer import policy
from doomformer.learn import perceptron_specs, ports_for
from doomformer.ops import adc_lanes, fx, output_single
from doomformer.pure import compile_pure
from doomformer.tile import Block, Gate, Stage, TileMeta

N_FEAT = len(policy.BEARING_CHARS) + 2
HEADS = 2
K = 3


def build_learner(wad: Path, name: str = "aim_learner", world=None) -> TileMeta:
    """world: compile_pure world hook, e.g. arena.build to put his arena and his brain in one map."""
    fwd, lrn, state, readout = perceptron_specs(N_FEAT, HEADS, K)
    act = output_single(lrn, "act")
    fire = Gate(lrn.port("-s0", "floor"), -1, "floor")                       # s_fire >= 1
    lrn.blocks.append(Block(adc_lanes(lrn, "go", [1], 1), [Stage(fire, fx(act, 2))]))
    right = Gate(lrn.port("s1", "floor"), -1, "floor")                       # s_turn <= -1
    hold = Gate(lrn.port("s0", "floor"), 0, "floor")                         # s_fire <= 0
    lrn.blocks.append(Block(adc_lanes(lrn, "go", [1], 1), [Stage(right, []), Stage(hold, fx(act, 1))]))
    return compile_pure([fwd, lrn], wad, name, state=state, readout=readout, world=world)


def ports(x: np.ndarray, y: np.ndarray) -> dict[str, int]:
    return ports_for(x, y) | {"go": 1, "-go": -1}


def features(state, game) -> np.ndarray:
    x = np.zeros(N_FEAT, dtype=np.int64)
    en = policy.enemies(state, game)
    if en:
        x[policy.BEARING_CHARS.index(policy.VOCAB[policy.bearing_token(en[0][1])])] = 1
    else:
        x[N_FEAT - 2] = 1
    x[N_FEAT - 1] = 1
    return x


def labels(expert_idx: int) -> np.ndarray:
    a = policy.ACTIONS[expert_idx]
    return np.array([1 if a == "F" else -1, {"L": 1, "R": -1, "F": 0}[a]], dtype=np.int64)


def action(s: np.ndarray) -> int:
    """Reference for the act dolls."""
    if s[0] > 0:
        return policy.ACTIONS.index("F")
    return policy.ACTIONS.index("L" if s[1] >= 0 else "R")
