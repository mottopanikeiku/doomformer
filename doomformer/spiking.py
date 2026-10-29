"""DOOM spikes: integer spiking neurons, one DOOM map run per time step.

The neuron (integer leaky integrate-and-fire with linear leak, TrueNorth-style), exact contract:

    s_t     = [v_t >= theta]                                   spikes this step
    v_{t+1} = v_t - theta * s_t                                reset by subtraction
                  - leak * [v_t >= leak] + leak * [v_t <= -leak]   linear leak toward 0 (never past it)
                  + W @ s_t + U @ x_t                          synapses (one step delay) and external inputs

v: membrane potentials (int), W[i, j]: added to neuron i when neuron j spikes, U[i, m]: added to neuron i when
input line m is on (x_t in {0, 1}^M). Every comparison reads v_t (a snapshot), every update is a sum, so DOOM can
do it: one map, two layers per step (fire: gates on v, write spikes, reset and leak; synapses: every spiking
neuron's doll crosses one raise/lower line per synapse onto its targets' v), v persists as `state`.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .ops import adc_lanes, fx, output_pair
from .pure import compile_pure
from .tile import Block, Stage, TileMeta, TileSpec

V_MAX = 12000          # |v| must stay inside the gate range (PORT_MAX)


@dataclass
class IntNet:
    names: list[str]                 # neuron labels (e.g. "MN9_left" or a FlyWire id)
    inputs: list[str]                # external input lines
    W: np.ndarray                    # (N, N) int: W[post, pre]
    U: np.ndarray                    # (N, M) int: U[post, input]
    theta: int
    leak: int
    meta: dict = field(default_factory=dict)     # provenance (source data, scaling, dt, ...)

    def __post_init__(self):
        self.W = np.asarray(self.W, dtype=np.int64)
        self.U = np.asarray(self.U, dtype=np.int64).reshape(len(self.names), len(self.inputs))
        n = len(self.names)
        if self.W.shape != (n, n):
            raise ValueError(f"W shape {self.W.shape} != ({n}, {n})")
        if self.theta < 1 or self.leak < 0:
            raise ValueError("need theta >= 1 and leak >= 0")

    @property
    def n(self) -> int:
        return len(self.names)

    def step(self, v: np.ndarray, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """One exact time step: returns (v_next, s)."""
        v = np.asarray(v, dtype=np.int64)
        x = np.asarray(x, dtype=np.int64)
        s = (v >= self.theta).astype(np.int64)
        v_next = (v - self.theta * s - self.leak * (v >= self.leak) + self.leak * (v <= -self.leak)
                  + self.W @ s + self.U @ x)
        if np.abs(v_next).max(initial=0) > V_MAX:
            i = int(np.abs(v_next).argmax())
            raise OverflowError(f"v[{self.names[i]}] = {int(v_next[i])} outside +-{V_MAX}")
        return v_next, s

    def run(self, X: np.ndarray, v0: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
        """X: (T, M) input spikes. Returns (V, S): V (T+1, N) potentials (V[0] = v0), S (T, N) spikes."""
        v = np.zeros(self.n, dtype=np.int64) if v0 is None else np.asarray(v0, dtype=np.int64)
        V, S = [v], []
        for x in np.asarray(X, dtype=np.int64):
            v, s = self.step(v, x)
            V.append(v)
            S.append(s)
        return np.array(V), np.array(S).reshape(len(S), self.n)

    def save(self, path: Path):
        Path(path).write_text(json.dumps(dict(names=self.names, inputs=self.inputs, theta=int(self.theta),
                                              leak=int(self.leak), meta=self.meta,
                                              W=[[int(i), int(j), int(w)] for i, j in zip(*np.nonzero(self.W))
                                                 for w in [self.W[i, j]]],
                                              U=[[int(i), int(m), int(w)] for i, m in zip(*np.nonzero(self.U))
                                                 for w in [self.U[i, m]]])))

    @classmethod
    def load(cls, path: Path) -> "IntNet":
        d = json.loads(Path(path).read_text())
        n, m = len(d["names"]), len(d["inputs"])
        W, U = np.zeros((n, n), np.int64), np.zeros((n, m), np.int64)
        for i, j, w in d["W"]:
            W[i, j] = w
        for i, k, w in d["U"]:
            U[i, k] = w
        return cls(d["names"], d["inputs"], W, U, d["theta"], d["leak"], d.get("meta", {}))


def build_spiking(net: IntNet, wad: Path, name: str = "spiking", world=None,
                  actions: dict[str, list[dict]] | None = None, *, layout_only: bool = False,
                  guard: bool = True) -> TileMeta:
    """Compile `net` into one map; one run = one time step (state v_i persists between runs).

    Ports and values (neuron i, input m):  inputs "-x{m}" (write -x_t[m]);  state "v{i}"/"-v{i}";  spikes "s{i}"
    (lights) with twins "-s{i}" (the synapse layer's lane gates);  readout "-v{i}" (the potential after the step).
    layer "fire":     per neuron: doll [v >= theta] -> s += 1, v -= theta;  [v >= leak] -> v -= leak;
                      [v <= -leak] -> v += leak
    layer "synapses": per spiking neuron j / active input m: one doll crossing one raise/lower line per target i
    """
    fire, syn = TileSpec("fire"), TileSpec("synapses")
    v_f = [output_pair(fire, f"v{i}") for i in range(net.n)]
    v_s = [output_pair(syn, f"v{i}") for i in range(net.n)]
    spike = [output_pair(fire, f"s{i}") for i in range(net.n)]
    for i in range(net.n):
        fire.blocks.append(Block(adc_lanes(fire, f"v{i}", [net.theta], 1),
                                 [Stage(None, fx(spike[i], 1) + fx(v_f[i], -net.theta))]))
        if net.leak:
            fire.blocks.append(Block(adc_lanes(fire, f"v{i}", [net.leak], 1), [Stage(None, fx(v_f[i], -net.leak))]))
            fire.blocks.append(Block(adc_lanes(fire, f"v{i}", [net.leak], -1), [Stage(None, fx(v_f[i], net.leak))]))
    for col, src, mat in [(j, f"s{j}", net.W) for j in range(net.n)] + \
                         [(m, f"x{m}", net.U) for m in range(len(net.inputs))]:
        effects = [e for i in np.nonzero(mat[:, col])[0] for e in fx(v_s[i], int(mat[i, col]))]
        if effects:
            syn.blocks.append(Block(adc_lanes(syn, src, [1], 1), [Stage(None, effects)]))
    state = {f"{p}v{i}": 0 for i in range(net.n) for p in ("", "-")}
    readout = [f"-v{i}" for i in range(net.n)]
    acts = {f"s{net.names.index(k)}": v for k, v in (actions or {}).items()}   # actions: neuron name -> specials
    return compile_pure([fire, syn], wad, name, state=state, readout=readout, world=world, actions=acts,
                        layout_only=layout_only, guard=guard)


def step_ports(x: np.ndarray, meta: TileMeta) -> dict[str, int]:
    """Input port values for one step (only the input lines the map actually reads)."""
    want = {key.split(":", 1)[1] for key in meta.ports}
    return {f"-x{m}": -int(v) for m, v in enumerate(x) if f"-x{m}" in want}


def read_step(out: dict[str, int], n: int) -> tuple[np.ndarray, np.ndarray]:
    """(v_next, s) from one run's outputs."""
    v = np.array([-out[f"-v{i}"] for i in range(n)], dtype=np.int64)
    s = np.array([out[f"s{i}"] for i in range(n)], dtype=np.int64)
    return v, s
