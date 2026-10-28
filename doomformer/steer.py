"""Rank-1 probes and steering on the integer activations.

    import doomformer as df
    from doomformer import intervene as iv
    pr = df.probe(model, "x0", contexts, labels, pos=1)           # difference of class means at one site and position
    logits = forward(model, tokens, interventions=[pr.add(alpha=1)])   # x0 at position 1 moves by alpha * direction
    forward(model, tokens, interventions=[pr.project()])               # remove the component along the direction

`probe` fits one direction: the integer activations of the site (the counts the staircases output, `forward(trace=...)`)
at position `pos` of every context are averaged per class, and the direction is the difference of the class means, scaled
and rounded to integers with largest entry `scale` (so the direction is an integer vector and the intervention is
exact in integers). The fit is floating point, deterministic for a given dataset; what an intervention does with the
integer vector is integer arithmetic only (`doomformer.intervene.add_direction`, `project_out`: the sum or the rounded
projection is clipped to the site's count range). The same intervention runs in DOOM (bus patch of the pre-activations,
checked at tolerance 0 against the reference; docs/interventions.md); the fit never does.

`labels` has one entry per context with exactly two distinct values; `positive` names the one the direction points to
(default: the larger). The probe reports the share of the fitting contexts a threshold at the midpoint of the two class
means along the direction classifies correctly (fit accuracy, not held-out).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import intervene as iv
from .intref import QModel, forward


@dataclass
class Probe:
    site: str
    pos: int                       # position in the context; -1 = the last token of each context
    direction: np.ndarray          # int64 [units of the site]
    scale: int                     # largest |entry| of the direction
    mean0: np.ndarray              # float class means (negative class, positive class)
    mean1: np.ndarray
    n0: int
    n1: int
    accuracy: float                # fit accuracy of the midpoint threshold along the direction
    classes: tuple                 # (negative label, positive label)

    def _pos(self, tokens=None, pos=None):
        p = self.pos if pos is None else pos
        return [len(tokens) - 1 if p == -1 else p] if tokens is not None else [p]

    def add(self, alpha: int = 1, tokens=None, pos=None) -> iv.Intervention:
        """alpha * direction added at the probe's position (tokens: the context, to resolve pos = -1)."""
        return iv.add_direction(self.site, self.direction, alpha, pos=self._pos(tokens, pos))

    def project(self, tokens=None, pos=None) -> iv.Intervention:
        """The component along the direction removed at the probe's position (see `intervene.project_out`)."""
        return iv.project_out(self.site, self.direction, pos=self._pos(tokens, pos))

    def score(self, activation: np.ndarray) -> np.ndarray:
        """Position along the direction: activation . direction (integer)."""
        return np.asarray(activation, dtype=np.int64) @ self.direction


def _activations(m: QModel, contexts, site: str, pos: int) -> np.ndarray:
    rows = []
    for tokens in contexts:
        tr: dict = {}
        forward(m, list(tokens), tr)
        rows.append(tr[site][len(tokens) - 1 if pos == -1 else pos])
    return np.array(rows, dtype=np.int64)


def probe(m: QModel, site: str, contexts, labels, *, pos: int = -1, positive=None, scale: int = 4) -> Probe:
    """Difference-of-means direction of integer activations (see the module docstring)."""
    iv.parse_site(site, m)
    labels = list(labels)
    if len(labels) != len(contexts):
        raise ValueError("one label per context")
    classes = sorted(set(labels), key=str)
    if len(classes) != 2:
        raise ValueError(f"labels must take exactly two values, got {classes}")
    pos_label = classes[1] if positive is None else positive
    neg_label = classes[0] if classes[1] == pos_label else classes[1]
    if pos_label not in classes:
        raise ValueError(f"positive={positive!r} is not one of the labels {classes}")
    x = _activations(m, contexts, site, pos)
    is_pos = np.array([lab == pos_label for lab in labels])
    m1, m0 = x[is_pos].mean(0), x[~is_pos].mean(0)
    diff = m1 - m0
    if not diff.any():
        raise ValueError("the two classes have the same mean activation at this site: no direction")
    u = np.rint(scale * diff / np.abs(diff).max()).astype(np.int64)
    mid = (m1 + m0) / 2 @ u
    acc = float(np.mean((x @ u > mid) == is_pos))
    return Probe(site, pos, u, scale, m0, m1, int((~is_pos).sum()), int(is_pos.sum()), acc, (neg_label, pos_label))


def steer(m: QModel, tokens, pr: Probe, alpha: int):
    """Logits at the last position of `tokens` with alpha * direction added at the probe's site and position."""
    return forward(m, list(tokens), interventions=[pr.add(alpha, tokens)])[-1]
