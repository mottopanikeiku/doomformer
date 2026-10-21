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


def equals(spec: TileSpec, x: str, value: int, targets: list[tuple[tuple[int, int], int]]):
    """One doll passes iff x == value (x <= value, then -x <= -value); adds w to each target pair."""
    effects = [e for pair, w in targets for e in fx(pair, w)]
    if effects:
        spec.blocks.append(Block([Gate(spec.port(x, "floor"), int(value), "floor")],
                                 [Stage(Gate(spec.port(neg(x), "floor"), -int(value), "floor"), effects)]))
