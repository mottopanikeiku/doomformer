"""Model-level building blocks expressed as tile blocks.

Naming convention for the bus: a value called "X" travels together with its negation "-X".
Producers emit both accumulators; consumers' gates take whichever one makes the comparison work,
so the bus only ever copies integers verbatim.
"""
from __future__ import annotations

from .tile import Block, Effect, Gate, Stage, TileSpec


def neg(name: str) -> str:
    return name[1:] if name.startswith("-") else f"-{name}"


def output_pair(spec: TileSpec, name: str, bias: int = 0) -> tuple[int, int]:
    """Accumulator for `name` and its negation (negated bias, negated effects)."""
    return spec.output(name, bias), spec.output(neg(name), -int(bias))


def output_single(spec: TileSpec, name: str, bias: int = 0) -> tuple[int, None]:
    """Accumulator without a negated twin (for values no gate consumes, e.g. logits)."""
    return spec.output(name, bias), None


def fx(pair: tuple[int, int | None], w: int) -> list[Effect]:
    if not w:
        return []
    return [Effect(pair[0], int(w))] + ([Effect(pair[1], -int(w))] if pair[1] is not None else [])


def adc_lanes(spec: TileSpec, x: str, thr, sign: int) -> list[Gate]:
    """Floor gates. sign=+1: lane k passes iff x >= thr[k] (fed -x). sign=-1: iff -x >= thr[k] (fed x)."""
    port = spec.port(neg(x) if sign > 0 else x, "floor")
    return [Gate(port, -int(t), "floor") for t in thr]


def crossbar(spec: TileSpec, x: str, thr, targets: list[tuple[tuple[int, int], int]], signed: bool = True):
    """Adds  w * adc(x)  to every target pair; adc is adc_signed or adc_relu."""
    for sign in (1, -1) if signed else (1,):
        effects = [e for pair, w in targets for e in fx(pair, sign * w)]
        if effects:
            spec.blocks.append(Block(adc_lanes(spec, x, thr, sign), [Stage(None, effects)]))


def multiply(spec: TileSpec, a: str, thr_a, signed_a: bool, b: str, thr_b, signed_b: bool,
             pair: tuple[int, int], w: int = 1):
    """Adds  w * adc(a) * adc(b): a's dolls climb a corridor of b's ceiling gates, one effect per gate."""
    for sa in (1, -1) if signed_a else (1,):
        for sb in (1, -1) if signed_b else (1,):
            port_b = spec.port(b if sb > 0 else neg(b), "ceil")
            stages = [Stage(Gate(port_b, int(t), "ceil"), fx(pair, sa * sb * w)) for t in thr_b]
            spec.blocks.append(Block(adc_lanes(spec, a, thr_a, sa), stages))


def attend(spec: TileSpec, e: str, thr_e, v: str, thr_v, num: tuple[int, int], P: list[tuple[int, int]],
           N: list[tuple[int, int]], R: int):
    """Softmax attention, one key position: with e = adc_relu(e port, thr_e) and v = adc_signed(v port, thr_v) adds
    e*v to num, R*e*v - l*e to P[l-1] and -R*e*v - l*e to N[l-1] (l = 1..len(P)).
    Two multiplier blocks (+v and -v corridors) on the same e lanes; the +v block starts with an ungated stage,
    so every e doll adds its -l exactly once."""
    for sb in (1, -1):
        port_b = spec.port(v if sb > 0 else neg(v), "ceil")
        per_level = fx(num, sb) + [x for p, q in zip(P, N) for x in fx(p, sb * R) + fx(q, -sb * R)]
        stages = [Stage(Gate(port_b, int(t), "ceil"), per_level) for t in thr_v]
        if sb > 0:
            stages.insert(0, Stage(None, [x for l, (p, q) in enumerate(zip(P, N), 1) for x in fx(p, -l) + fx(q, -l)]))
        spec.blocks.append(Block(adc_lanes(spec, e, thr_e, 1), stages))


def square(spec: TileSpec, x: str, thr, pair: tuple[int, int]):
    """Adds  adc_signed(x, thr)^2  to pair: one block per lane, lane k (1-based, either sign) adds 2k - 1. The passing
    lanes of a sign are always its first c (thresholds nondecreasing), so they add 1 + 3 + ... + (2c - 1) = c^2."""
    for sign in (1, -1):
        for k, gate in enumerate(adc_lanes(spec, x, thr, sign), 1):
            spec.blocks.append(Block([gate], [Stage(None, fx(pair, 2 * k - 1))]))


def levels(spec: TileSpec, P: list[str], N: list[str], targets: list[tuple[tuple[int, int], int]]):
    """Adds  (#{k: P[k] >= 1} - #{k: N[k] >= 1}) * w  to every target pair: one block per sign whose lanes are the
    indicator gates [value >= 1] of the listed values (a count read through comparisons, see intref.rmsnorm)."""
    for names, sign in ((P, 1), (N, -1)):
        effects = [e for pair, w in targets for e in fx(pair, sign * w)]
        if effects:
            spec.blocks.append(Block([g for v in names for g in adc_lanes(spec, v, [1], 1)], [Stage(None, effects)]))


def gated(spec: TileSpec, g: str, thr_g, u: str, thr_u, targets: list[tuple[tuple[int, int], int]]):
    """Adds  adc_relu(g, thr_g) * adc_signed(u, thr_u) * w  to every target pair (the gated MLP's g*u times a column
    of Wdn): g's lanes climb a corridor of u's ceiling gates, one corridor per sign of u, and every passed stage adds
    +-w to all targets, so the product g*u never exists as a value."""
    for su in (1, -1):
        port = spec.port(u if su > 0 else neg(u), "ceil")
        effects = [e for pair, w in targets for e in fx(pair, su * w)]
        stages = [Stage(Gate(port, int(t), "ceil"), list(effects)) for t in thr_u]
        if any(st.effects for st in stages):
            spec.blocks.append(Block(adc_lanes(spec, g, thr_g, 1), stages))


def equals(spec: TileSpec, x: str, value: int, targets: list[tuple[tuple[int, int], int]]):
    """One doll passes iff x == value (x <= value, then -x <= -value); adds w to each target pair."""
    effects = [e for pair, w in targets for e in fx(pair, w)]
    if effects:
        spec.blocks.append(Block([Gate(spec.port(x, "floor"), int(value), "floor")],
                                 [Stage(Gate(spec.port(neg(x), "floor"), -int(value), "floor"), effects)]))
