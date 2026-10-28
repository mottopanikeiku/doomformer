"""The schedule levels opt=1 to opt=3 (pure.compile_pure `opt`) in real headless DOOM games, one at a time: the hello
MLP at opt 1, 2 and 3 and tiny_model at opt 1 and 3, every exposed value equal to the integer reference (tolerance 0).
opt 2 and 3 build with doomformer/tuning.json and skip without it."""
from __future__ import annotations

import numpy as np
import pytest

import doomformer as df
from doomformer.kit import _build
from doomformer.onemap import tiny_model
from doomformer.pure import TUNING_FILE, opt_passes


def hello_mlp() -> df.MLP:
    """The net of `doomformer hello` (seed 0): 4-6-5-3, three thresholds per layer."""
    rng = np.random.default_rng(0)
    sizes = [4, 6, 5, 3]
    return df.MLP([rng.integers(-20, 21, (sizes[l + 1], sizes[l])) for l in range(3)],
                  [rng.integers(-30, 31, sizes[l + 1]) for l in range(3)],
                  [np.cumsum(rng.integers(15, 40, 3)) for _ in range(3)])


def needs_tuning(opt: int):
    if "tuned" in opt_passes(opt) and not TUNING_FILE.is_file():
        pytest.skip(f"{TUNING_FILE.name} is missing (opt={opt} builds with its constants)")


@pytest.mark.parametrize("opt", [1, 2, 3])
def test_hello_mlp_exact_at_opt(tmp_path, opt):
    needs_tuning(opt)
    net = hello_mlp()
    rng = np.random.default_rng(1)
    with df.compile(net, tmp_path / "mlp.wad", opt=opt) as brain:
        assert brain.meta.layout["opt"] == sorted(opt_passes(opt))
        for _ in range(3):
            x = rng.integers(-150, 151, net.n_in)
            ref = net.reference(x)
            np.testing.assert_array_equal(brain(x), ref["y"])       # brain(x) raises MismatchError on any value
            assert brain.info["checked"] > net.n_out


@pytest.mark.parametrize("opt", [1, 3])
def test_tiny_transformer_exact_at_opt(tmp_path, opt):
    needs_tuning(opt)
    m = tiny_model()
    rng = np.random.default_rng(3)
    before = _build(m, tmp_path / "opt0.wad", layout_only=True).layout["stage_stats"]
    with df.compile(m, tmp_path / "tiny.wad", opt=opt) as brain:
        after = brain.meta.layout["stage_stats"]
        assert sum(s["waves"] for s in after) < sum(s["waves"] for s in before)
        for n in (m.T, 1):
            tokens = [0] + [int(rng.integers(1, len(m.vocab))) for _ in range(n - 1)]
            brain.logits(tokens)                                   # every exposed value vs intref.forward
            assert brain.info["checked"] > 0
