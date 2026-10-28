"""A doll-driven step clock for compile_pure machines: no OPEN script, the dolls restart the level themselves.

`compile_machine(p, wad, clock="dolls")` (machine.py) builds the same layers plus one `tick` layer and this module's loop:

  * tick layer   one doll, last in the pass: it adds 1 to the state value `pass` and 1 to the value `done`; crossing the
                 line that raises `done` also runs Floor_LowerToLowest on the loop barrier (compile_pure `actions`).
  * loop doll    one extra voodoo doll in the world area, on a corridor of lines and sectors, never tagged:
                   pad -> boot delay -> [Floor_LowerToLowest START barrier: layer 0 starts] -> loop barrier (closed;
                   the tick doll opens it) -> hold delay (Python reads here) -> halt gate (floor gate on pc_halt: a
                   raised latch blocks it for good) -> rewind lines -> Teleport_NoFog back to the pad.
  * rewind       one line special per action of the ACS rewind script: Floor_MoveToValue / Ceiling_MoveToValue (barriers
                 up, non-state ports to their biases, freeze barriers open), TeleportOther (every doll home),
                 Light_ChangeToValue (light outputs). They are crossed in an order that keeps the dolls behind closed
                 barriers: barriers first, then dolls, then ports and lights.
  * the dolls tag themselves with the line special Thing_ChangeTID instead of ACS script 4.
  * the ACS lump is rebuilt after compile_pure: only the readout pages and the stillness (motion) script remain.

Python reads the counter `pass` (one readout page) to find the window of pass k, reads every value there between two
motion checks, and checks the counter again. It sends no command that is not a readout page or the motion script.
"""
from __future__ import annotations

import re
from pathlib import Path

from . import selfgen as sg
from .machine import _wad_bytes, _wad_lumps
from .ops import fx, neg, output_pair
from .pure import CLOSED, LOWER_TO_LOWEST, X_ORIGIN, map_bounds, world_area
from .tile import (CONV_TAG, DOLL_TID0, HOME_TID0, MOTION_SCRIPT, MOVE_SPEED, OUT_TAG0, PAGE, START_TAG, Block, Gate, Stage,
                   TileSpec, motion_acs)

LOOP_BARRIER = 90000            # tag of the loop barrier
LOOP_HOME_TID = 47000           # teleport destination of the loop doll (the pad)
ROW_TID0 = 47100                # teleport destinations of the chain's later rows
ROW_END = X_ORIGIN + 27000      # a row of the rewind chain ends here and the doll teleports to the start of the next one
LOOP_TID = 47500                # the loop doll tags itself (a probe reads its position); the motion script ignores it
STAGE_TAG0 = 60000              # pure.STAGE_TAG0: barrier and freeze tags are >= this
OBSERVER = 11000
FLOOR_MOVE_TO, CEIL_MOVE_TO, TELEPORT_OTHER, LIGHT_TO, TELEPORT_NOFOG, CHANGE_TID = 37, 47, 76, 112, 71, 176
UNITS_PER_TIC = 16              # the clock's walking speed on the conveyor (compile_pure's own estimate)
BOOT_TICS = 24                  # pad to the first lowering: every doll has tagged itself, every plane has settled
HOLD_TICS = 120                 # a window long enough for one poll interval and every readout page
CELL = 16                       # width of a rewind cell = one tic of walking: two of these lines crossed in one tic do not both act
GAP = 64                        # clearance before a teleport line (same-tic teleport truncation, docs/engine_limits.md)


def tick_layer(halt: str) -> tuple[TileSpec, dict, dict]:
    """The extra last layer and what compile_pure needs for it: (spec, state additions, actions)."""
    spec = TileSpec("tick")
    pair_done, pair_pass = output_pair(spec, "done"), output_pair(spec, "pass")
    always = Gate(spec.port(halt, "floor"), OBSERVER, "floor")          # passes for every value of the latch
    spec.blocks.append(Block([always], [Stage(None, fx(pair_done, 1) + fx(pair_pass, 1))]))
    for m in ("pass", neg("pass")):
        spec.blocks.append(Block([Gate(spec.port(m, "floor"), OBSERVER, "floor")], [Stage(None, [])]))
    state = {"pass": 0, "-pass": 0}
    actions = {"done": [dict(special=LOWER_TO_LOWEST, arg0=LOOP_BARRIER, arg1=MOVE_SPEED)]}
    return spec, state, actions


def loop_world(layers: list[TileSpec], state: dict, halt: str, total_dolls: int, hold_tics: int = HOLD_TICS):
    """The `world` callback of compile_pure: the real player's closet, the loop doll and its corridor. Returns ACS ''."""
    tags = sg.port_tags(layers)
    bias = {o: b for spec in layers for o, b in spec.outputs}
    home0 = max(HOME_TID0, DOLL_TID0 + total_dolls + 1)

    def world(mb):
        mb.rect(-768, -512, -512, -256, floor=0, ceil=128)
        # the rewind, derived from what is built: every sector of a barrier or freeze tag goes back to its height,
        # every non-state port to its bias, every light output to its level
        seen, bars, lights = set(), [], []
        for r in mb.rects:
            if (r.tag == START_TAG or r.tag >= STAGE_TAG0) and r.tag not in seen:
                seen.add(r.tag)
                bars.append(dict(special=FLOOR_MOVE_TO, arg0=r.tag, arg1=MOVE_SPEED, arg2=abs(r.floor),
                                 arg3=int(r.floor < 0)))
            elif r.tag >= OUT_TAG0 and r.tag < STAGE_TAG0 and r.ftex == "FLAT14" and r.tag not in seen:
                seen.add(r.tag)
                lights.append(dict(special=LIGHT_TO, arg0=r.tag, arg1=r.light))
        bars.append(dict(special=FLOOR_MOVE_TO, arg0=LOOP_BARRIER, arg1=MOVE_SPEED, arg2=CLOSED, arg3=0))
        teleports = [dict(special=TELEPORT_OTHER, arg0=DOLL_TID0 + i, arg1=home0 + i, arg2=0) for i in range(total_dolls)]
        ports = []
        for key, tag in tags.items():
            kind, val = key.split(":", 1)
            if val in state or val not in bias:
                continue
            h = bias[val]
            ports.append(dict(special=FLOOR_MOVE_TO if kind == "floor" else CEIL_MOVE_TO, arg0=tag, arg1=MOVE_SPEED,
                              arg2=abs(h), arg3=int(h < 0)))
        chain = bars + teleports + ports + lights
        # the corridor: one row in the world area, every sector on the walking floor F (a gate step is exactly 24)
        F, W = -24, 40
        area = world_area(map_bounds(mb))
        if area is None:
            raise ValueError("no room for the loop doll's corridor in the map's world area")
        x, y = X_ORIGIN, min(area[3], -2200) - 400
        group = 10 ** 6
        kw = dict(ceil=16000, group=group, tag=CONV_TAG)
        pieces = []

        def add(width, **o):
            nonlocal x
            r = mb.rect(x, y, x + width, y + W, floor=o.pop("floor", F), **{**kw, **o})
            x += width
            pieces.append(r)
            return r

        pad = add(48, light=192)
        mb.thing(X_ORIGIN + 16, y + W // 2, 1)                       # the loop doll (not tagged, not in the motion script)
        mb.thing(X_ORIGIN + 16, y + W // 2, 14, id=LOOP_HOME_TID)
        add(BOOT_TICS * UNITS_PER_TIC)
        mb.link(pad, pieces[-1], special=CHANGE_TID, arg0=0, arg1=LOOP_TID, playercross=True, repeatspecial=True)
        r_lower = add(4)
        mb.link(pieces[-2], r_lower, special=LOWER_TO_LOWEST, arg0=START_TAG, arg1=MOVE_SPEED, playercross=True,
                repeatspecial=True)
        add(64, light=200)
        add(8, floor=CLOSED, tag=LOOP_BARRIER, light=224)            # the loop barrier: the tick doll lowers it
        add(hold_tics * UNITS_PER_TIC, light=192)                    # Python's window
        init = state.get(halt, 0)
        add(8, floor=init, tag=tags[f"floor:{halt}"], light=224)     # the halt gate: passes while the latch is <= 0
        add(128, light=176)
        prev = add(4)
        rows = 0

        def fold():
            """End the row: clearance, a Teleport_NoFog line to the start of a new row one lane lower."""
            nonlocal x, y, rows, prev
            add(GAP, light=160)
            cur = add(4)
            rows += 1
            mb.link(pieces[-2], cur, special=TELEPORT_NOFOG, arg0=ROW_TID0 + rows, playercross=True, repeatspecial=True)
            add(64, light=160)
            x, y = X_ORIGIN, y - 2 * W
            if y < area[1]:
                raise ValueError("the rewind chain needs more rows than the world area holds")
            mb.thing(X_ORIGIN + 16, y + W // 2, 14, id=ROW_TID0 + rows)
            add(64, light=160)
            prev = add(4)

        for spec_ in chain:
            if x + CELL + GAP + 4 + 64 + 64 > ROW_END:
                fold()
            cur = add(CELL, light=200)
            mb.link(prev, cur, **spec_, playercross=True, repeatspecial=True)
            prev = cur
        add(GAP, light=160)
        cur = add(4)
        mb.link(pieces[-2], cur, special=TELEPORT_NOFOG, arg0=LOOP_HOME_TID, playercross=True, repeatspecial=True)
        add(64, light=160)
        world.chain_lines, world.rows = len(chain), rows + 1
        world.corridor = (X_ORIGIN, x, y)
        mb.thing(-640, -384, 1)                                      # last player-1 start = the real player
        return ""
    return world


def finish_wad(wad: Path, meta, loaders: bool = False) -> dict:
    """Rewrite the compiled map: dolls tag themselves with the line special Thing_ChangeTID, and the ACS lump keeps the
    readout pages and the motion script only (plus, with loaders=True, the two mover scripts 1 and 2 that the host's
    initial-state writes of a stored program use). Returns what was done, for the report."""
    wad = Path(wad)
    lumps = _wad_lumps(wad.read_bytes())
    tm = next(i for i, (n, _) in enumerate(lumps) if n == "TEXTMAP")
    bh = next(i for i, (n, _) in enumerate(lumps) if n == "BEHAVIOR")
    text = lumps[tm][1].decode()
    text, n_tag = re.subn(r"special = 226; arg0 = 4; arg1 = 0; arg2 = (\d+);",
                          rf"special = {CHANGE_TID}; arg0 = 0; arg1 = \1; arg2 = 0;", text)
    reads = meta.layout["read_expressions"]
    decl = "\n".join(f"global int {i}:o{i};" for i in range(1, PAGE + 1))
    pages = []
    for pi in range(len(meta.pages)):
        body = " ".join(f"o{i + 1} = {reads[n]};" for i, n in enumerate(meta.pages[pi]))
        pages.append(f"script {100 + pi} (void) {{ {body} }}")
    from .udmf import compile_acs
    movers = (f"script 1 (int tag, int h, int neg) {{ Floor_MoveToValue(tag, {MOVE_SPEED}, h, neg); }}\n"
              f"script 2 (int tag, int h, int neg) {{ Ceiling_MoveToValue(tag, {MOVE_SPEED}, h, neg); }}\n") if loaders else ""
    acs = f'#include "zcommon.acs"\n{decl}\n{movers}' + "\n".join(pages) + "\n" + motion_acs(meta.n_dolls) + "\n"
    lumps[tm] = ("TEXTMAP", text.encode())
    lumps[bh] = ("BEHAVIOR", compile_acs(acs))
    wad.write_bytes(_wad_bytes(lumps))
    scripts = sorted(int(s) for s in re.findall(r"^script (\d+)", acs, re.M))
    return dict(tag_lines=n_tag, scripts=scripts, acs=acs)


# --- the runner's side ---------------------------------------------------------------------------------------------
class DollClock:
    """Python's view of the doll-driven loop: poll the counter `pass`, read inside the window, never write."""

    commands = (f"puke -{MOTION_SCRIPT}",)
    POLL = 8

    def __init__(self, runner, game):
        from .runtime import USER
        self.runner, self.game, self._user = runner, game, USER
        meta = runner.meta
        for pi, page in enumerate(meta.pages):
            if "pass" in page:
                self.page, self.slot = pi, page.index("pass")
                break
        else:
            raise ValueError("the map has no `pass` readout")
        self.last_seen = 0

    def passes(self) -> int:
        self.game.send_game_command(f"puke -{100 + self.page}")
        self.game.advance_action(1)
        return int(self.game.get_game_variable(self._user[self.slot]))

    def motion(self) -> tuple[int, int]:
        self.game.send_game_command(f"puke -{MOTION_SCRIPT}")
        self.game.advance_action(1)
        return tuple(int(self.game.get_game_variable(v)) for v in self._user[:2])

    def wait_hold(self, k: int, max_tics: int = 400_000) -> dict[str, int]:
        waited = 0
        while True:
            p = self.passes()
            if p == k:
                now = int(self.game.get_episode_time())
                for _ in range(4):                    # the tick doll still walks off its corridor: wait until none moves
                    self.motion()
                    self.game.advance_action(sg.STILL)
                    if not self.motion()[0]:
                        break
                s = dict(hold=now, start=self.last_seen or now, pass_=p)
                self.last_seen = now
                return s
            if p > k:
                raise RuntimeError(f"missed the window of step {k}: the counter is already {p}")
            if waited > max_tics:
                raise RuntimeError(f"step {k} did not finish within {max_tics} tics: pass = {p}")
            self.game.advance_action(self.POLL)
            waited += self.POLL

    def read_still(self, k: int, s: dict[str, int]) -> dict[str, int]:
        self.motion()
        got = self.runner.read()
        moved, untagged = self.motion()
        after = self.passes()
        if after != k or moved or untagged:
            raise RuntimeError(f"step {k} was not read inside a still window: pass {after}, {moved} dolls moved, "
                               f"{untagged} untagged")
        return got

    def finish(self, last_pass: int, hold: int) -> None:
        """After the halt: the latch blocks the loop doll for good, so nothing moves and the counter stays."""
        self.game.advance_action(4 * hold)
        self.motion()
        self.game.advance_action(sg.STILL)
        moved, untagged = self.motion()
        p = self.passes()
        if p != last_pass or moved or untagged:
            raise RuntimeError(f"the machine did not stay halted: pass {p} (expected {last_pass}), {moved} dolls moved")
