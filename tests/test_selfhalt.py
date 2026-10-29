"""Generation that stops at EOS (selfhalt.py): the winner doll writes a per-position halt latch, the level reads the
frontier's latch and leaves its loop; one map (same sha256) for every one-letter prompt, the prompt loaded as initial
state; a miswired latch fails."""
import hashlib

import numpy as np
import pytest

from doomformer import intref, selfgen, selfhalt
from doomformer.kit import MismatchError
from tests.test_selfgen import random_model

# T=4, V=5, one-letter prompts (at most 2 passes). "C": EOS at the frontier in pass 1, writes "C". "B": writes "BAA" in
# both passes; in pass 1 position 2 (not yet final) picks EOS, so a single latch for all positions would stop at "BA".
SEED = 24


def test_stop_pass_is_greedy_generation():
    rng = np.random.default_rng(1)
    for trial in range(300):
        T, V = int(rng.integers(3, 8)), int(rng.integers(3, 9))
        m = random_model(trial, T, V, ("relu", "softmax")[trial % 2])
        prompt = "".join(m.vocab[i] for i in rng.integers(1, V, int(rng.integers(0, T - 1))))
        ref = selfhalt.reference(m, prompt)
        pos = selfgen.positions(m, prompt)
        for p in ref:
            assert [int(p.trace["halt"][t]) for t in pos] == [int(w == 0) for w in p.winners]
        k = selfhalt.stop_pass(ref)
        assert selfgen.text(m, ref[k - 1].written, prompt) == intref.generate(m, prompt, temperature=0)
    m = random_model(SEED)
    assert [p.winners for p in selfhalt.reference(m, "B")] == [[1, 0], [1, 1]]


def test_one_map_two_prompts(tmp_path):
    m = random_model(SEED)
    wad = tmp_path / "selfhalt.wad"
    meta = selfhalt.build_until(m, wad, 1)
    assert meta.ports == {}                                         # nothing for the host to write as input
    sha = hashlib.sha256(wad.read_bytes()).hexdigest()
    done = {}
    for prompt in ("C", "B"):
        ref = selfhalt.reference(m, prompt)
        assert not intref.check_limits(m, [p.tokens for p in ref])
        assert hashlib.sha256(wad.read_bytes()).hexdigest() == sha
        with selfhalt.UntilEosRun(m, wad, prompt) as run:
            boot = run.load()
            wrote = run.run()
            audit = run.audit()
        assert wrote == intref.generate(m, prompt, temperature=0)
        assert boot["loads"] == audit["loaded"] == 2 and audit["sealed"]
        assert audit["non_readout_commands"] == 0 and audit["forbidden"] == {}
        assert set(audit["commands"]) <= run.game.allowed
        assert all(p["checked"] == len(meta.outputs) for p in run.passes)
        done[prompt] = (wrote, [p["winners"] for p in run.passes], run.schedule["passes"])
    assert done["C"] == ("C", [[0, 0]], 2)                          # stopped after pass 1 of at most 2
    assert done["B"] == ("BAA", [[1, 0], [1, 1]], 2)                # ran the bound past a spurious EOS at position 2


def test_miswired_latch_runs_past_eos(tmp_path):
    m = random_model(SEED)
    wad = tmp_path / "miswired.wad"
    selfhalt.build_until(m, wad, 1, miswire={(1, 0): 0})           # position 1's EOS winner writes latch 0
    with selfhalt.UntilEosRun(m, wad, "C") as run:
        with pytest.raises(MismatchError) as err:
            run.run()
        assert err.value.mismatches == [("halt:1", 0, 1)]
        after = run.after_hold(1)
        assert after["phase"] == selfgen.PH_COMPUTE and after["latch"] == 0     # the level starts pass 2 after EOS


class Dropping:
    """The engine, except that one exact console command is dropped (a readout script or guard that never runs)."""

    def __init__(self, game, command: str):
        self._game, self.command = game, command

    def send_game_command(self, cmd, *a):
        return None if cmd == self.command else self._game.send_game_command(cmd, *a)

    def __getattr__(self, name):
        return getattr(self._game, name)


@pytest.mark.parametrize("drop", ["page", "guard"])
def test_boot_state_reads_are_guarded(tmp_path, drop):
    """The boot-time check of the state values reads every state page through the guard. With an empty prompt the
    state is all zero, so before this a dropped page was accepted (checked=11). A dropped page or guard must raise."""
    from doomformer.readout import GUARD_COMMAND
    m = random_model(24)
    wad = tmp_path / "boot.wad"
    selfhalt.build_until(m, wad, 0)
    with selfhalt.UntilEosRun(m, wad, "") as run:
        page = run.schedule["state_pages"][-1]
        run.game._game = Dropping(run.game._game, f"puke -{100 + page}" if drop == "page" else GUARD_COMMAND)
        with pytest.raises(RuntimeError, match="readout guard for page" if drop == "guard" else "readout page"):
            run.load()
