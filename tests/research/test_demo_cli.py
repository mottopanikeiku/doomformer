"""`python -m demos.<name>` runs at tiny settings: real DOOM games, every decision or step checked exactly."""
from __future__ import annotations

import numpy as np
import pytest

from demos import aimlearn, brainlevel, flyloop, lobotomy
from doomformer import policy, shipped
from doomformer.intref import QModel
from doomformer.onemap import tiny_model
from doomformer.spiking import IntNet


def _gif_ok(path):
    from PIL import Image
    assert path.stat().st_size < 3_000_000
    with Image.open(path) as im:
        assert im.n_frames >= 2 and im.info.get("loop") == 0


@pytest.mark.parametrize("module", [aimlearn, brainlevel, flyloop, lobotomy])
def test_help_names_the_command(module, capsys):
    with pytest.raises(SystemExit) as stop:
        module.main(["--help"])
    assert stop.value.code == 0
    assert f"python -m demos.{module.__name__.split('.')[-1]}" in capsys.readouterr().out


def test_learner_checks_every_decision_and_writes_a_gif(tmp_path):
    lines = []
    gif = tmp_path / "learn.gif"
    stats = aimlearn.run_learner(seed=4, decisions=4, gif=str(gif), workdir=str(tmp_path), out=lines.append)
    assert stats["decisions"] == stats["exact"] == 4
    assert stats["values_per_decision"] == 41                      # 2 scores, the action, 38 weights
    assert sum("EXACT" in line and "MISMATCH" not in line for line in lines) >= 4
    assert any(line.startswith("4/4 decisions EXACT") for line in lines)
    _gif_ok(gif)


def test_brain_is_the_level_runs_from_the_command_layer(tmp_path):
    model = tiny_model()
    model.vocab = policy.VOCAB
    rng = np.random.default_rng(19)
    model.p["tok_emb"] = rng.integers(-2, 3, (len(model.vocab), model.d))
    model.p["Wun"] = rng.integers(-2, 3, (len(model.vocab), model.d))
    model.p["bun"] = rng.integers(-2, 3, len(model.vocab))
    lines = []
    gif = tmp_path / "brain.gif"
    stats = brainlevel.run_brainlevel(seed=4, decisions=2, lives=1, model=model, gif=str(gif),
                                      workdir=str(tmp_path), out=lines.append)
    assert stats["decisions"] == stats["exact"] == 2
    assert stats["audit"]["non_readout_commands"] == 0
    assert any(line.startswith("2/2 decisions EXACT") for line in lines)
    assert any("non-readout: 0" in line for line in lines)
    _gif_ok(gif)


def test_fly_loop_runs_from_the_command_layer(tmp_path):
    W = np.array([[0, -20, 0], [0, 0, 0], [5, 0, 0]])
    U = np.array([[8, 8, 0, 0], [0, 0, 6, 6], [0, 0, 0, 0]])
    net = IntNet(["MN9_left", "h0", "h1"], ["s0", "s1", "b0", "b1"], W, U, theta=10, leak=1,
                 meta=dict(input_groups={"sugar": [0, 1], "bitter": [2, 3]},
                           input_p_per_step={"sugar": 0.5, "bitter": 0.5}))
    lines = []
    gif = tmp_path / "fly.gif"
    r = flyloop.run_fly(seed=7, steps=4, net=net, stations=(flyloop.Station("sugar", 30, 3),), gif=str(gif),
                        workdir=str(tmp_path), every=2, out=lines.append)
    assert r["steps"] == r["exact"] == r["prng_ok"] == r["settled"] == 4
    assert r["audit"]["non_readout_commands"] == 0 and r["action_mismatch"] == 0
    assert any(line.startswith("4/4 steps EXACT") for line in lines)
    _gif_ok(gif)


def test_lobotomy_pacifist_is_one_of_the_scanned_blasts():
    m = QModel.load(shipped.resolve("doomplayer"))
    contexts = policy.all_contexts()
    base = lobotomy.action_table(m, contexts)
    table = lobotomy.action_table(m, contexts, lobotomy.DOCUMENTED)
    assert lobotomy.personality(table, base, contexts).startswith("PACIFIST")
    assert lobotomy.play(m, lobotomy.DOCUMENTED, 2, 9000)["kills"] == 0
    found = lobotomy.scan(m, contexts, base, blasts=40, pellets=7)
    assert found["singles"] == 96 and found["single_changed"] == 16     # the README's 16 of 96
    changed = [c for c, b, t in zip(contexts, base, table) if b != t]
    done = lobotomy.doom_check(m, lobotomy.DOCUMENTED, changed[:1], out=lambda line: None)
    assert done["decisions"] == done["exact"] == 1 and done["changed"] == 1
