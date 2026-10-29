"""Readout-only access to the live game of a map that runs itself.

ReadoutOnly wraps a vizdoom.DoomGame. A console command must equal one allowlisted string exactly (no extra
arguments, no `;` chains); any other command, and every game method outside a small read-only set (set_action,
make_action, new_episode, ...), raises RuntimeError at the call, before it reaches the engine. Attempts are counted
before rejection, so audit() cannot hide them. df.Agent and the self-running demo maps (demos/brainlevel.ClockRun,
demos/flyloop.LoopRun) wrap their live game with it; calibration twins, whose state the host does write, are not.
"""
from __future__ import annotations

from collections import Counter

from .tile import GUARD_NONCE, GUARD_SCRIPT

READ_ONLY = frozenset({"advance_action", "get_game_variable", "get_episode_time", "get_state", "get_total_reward",
                       "is_episode_finished", "close"})
GUARD_COMMAND = f"puke -{GUARD_SCRIPT} {GUARD_NONCE}"    # TileRunner.read's poison before each page of a guarded map


def readout_commands(n_pages: int, *extra: str) -> set[str]:
    """Generated readout pages and the exact USER-slot poison command, plus `extra` exact commands."""
    return {f"puke -{100 + i}" for i in range(n_pages)} | {GUARD_COMMAND} | set(extra)


class ReadoutOnly:
    """Reject console commands outside the exact readout allowlist, and all host writes."""

    def __init__(self, game, allowed):
        self._game = game
        self.allowed = frozenset(allowed)
        self.commands = Counter()
        self.rejected = Counter()

    def send_game_command(self, command):
        self.commands[command] += 1
        if command not in self.allowed:
            self.rejected[command] += 1
            raise RuntimeError(f"non-readout command rejected: {command!r}")
        return self._game.send_game_command(command)

    def __getattr__(self, name):
        if name in READ_ONLY:
            return getattr(self._game, name)
        self.rejected[name] += 1
        raise RuntimeError(f"host game mutation/access rejected: {name}")

    def audit(self):
        return dict(commands=dict(self.commands), forbidden=dict(self.rejected),
                    non_readout_commands=sum(self.rejected.values()))


class LoadThenReadout(ReadoutOnly):
    """ReadoutOnly plus a load phase: until `seal()` the host may set state values (plane heights) through the map's
    own mover scripts 1 (floor) and 2 (ceiling), one `load()` call per port, before the first tic that matters. After
    `seal()` it is plain ReadoutOnly. This is how a stored program enters a map: the map file is the same bytes for
    every program, only these initial heights differ. audit() reports the loads next to the rejected commands."""

    def __init__(self, game, allowed):
        super().__init__(game, allowed)
        self.loaded = 0
        self.sealed = False

    def load(self, tag: int, value: int, kind: str = "floor"):
        if self.sealed:
            self.rejected["load after seal"] += 1
            raise RuntimeError("state load after the run started")
        self.loaded += 1
        self._game.send_game_command(f"puke -{1 if kind == 'floor' else 2} {tag} {abs(int(value))} {int(value < 0)}")

    def seal(self):
        self.sealed = True

    def audit(self):
        return dict(super().audit(), loaded=self.loaded, sealed=self.sealed)
