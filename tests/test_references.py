"""Fast reference-only tests (no DOOM): the integer contracts DOOM is checked against."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from doomformer.spiking import IntNet

ROOT = Path(__file__).resolve().parent.parent


def test_leak_stops_at_zero():
    """A lone neuron below threshold decays to a fixed point and stays there (no oscillation around 0)."""
    net = IntNet(["a"], [], np.zeros((1, 1)), np.zeros((1, 0)), theta=100, leak=3)
    for v0 in (50, 2, -2, -50):
        v = np.array([v0])
        for _ in range(40):
            v, s = net.step(v, np.zeros(0))
            assert not s.any()
        assert abs(int(v[0])) < 3
        assert (net.step(v, np.zeros(0))[0] == v).all()


def test_overflow_is_refused():
    net = IntNet(["a", "b"], ["x"], [[0, 0], [5000, 0]], [[13000], [0]], theta=1, leak=0)
    with pytest.raises(OverflowError):
        net.step(np.zeros(2), np.ones(1))
