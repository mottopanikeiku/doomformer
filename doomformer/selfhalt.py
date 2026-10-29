"""Generation that stops at EOS: the winner doll raises a halt latch, the level reads it and leaves its loop.

selfgen.py runs a fixed number of passes, T-1-len(prompt), and bakes the prompt into the map. Here the network also
decides when the loop ends, and one map serves every prompt of one length.

  * the latch: a state value halt:t (initial 0) per writing position t. The winner lines of position t
    (selfgen.write_actions: tok:t+1 := a, -tok:t+1 := -a) get one more Floor_MoveToValue line on port halt:t:
    height 1 for the EOS winner argz:t:0, height 0 for every other winner argz:t:a. Every writing position has
    exactly one winner per pass, so after every pass halt:t = [winner at t is EOS] (tolerance 0). The port is the
    floor gate of one extra lane per position in layer "embed": the gate never closes (limit 1, the latch is 0 or 1)
    and its doll crosses no effect line. The lane exists so that the port has a sector.
  * one latch per position: in pass p (from 1) only the winner at the frontier len(prompt)+p-1 is final; later
    positions read placeholder tokens and may pick EOS. A single latch written by every position would stop the loop
    on such a spurious EOS.
  * the level (level_acs_until): selfgen's loop with an upper bound of T-1-len(prompt) passes. After a pass settles
    and before its HOLD window, the ACS reads one plane height, the frontier's latch (GetSectorFloorZ), and after the
    HOLD it leaves the loop if that height is >= 1. This read is the only value the ACS reads: it picks no token
    and computes nothing. The latch itself is written mechanically, by the line the winner's doll crosses.
  * one map, many prompts: the map is built for a prompt length. Every token port starts at 0. Before the first pass
    the host writes the prompt's tokens into ports tok:t and -tok:t (t = 1..len(prompt)) with
    readout.LoadThenReadout.load (the map's own mover script 1) and seals; after that it only advances the clock and
    runs readout scripts. The level waits `boot` tics before the first pass: enough for the loads and a read-back
    of the state values.

UntilEosRun checks the loaded state values before pass 1, every value of every pass against selfgen.iterate plus the
latches, the latch the level read, and that the level left its loop after exactly the first pass whose frontier
winner is EOS (or after the whole bound). The text must equal intref.generate(m, prompt, temperature=0).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from .intref import QModel
from .kit import MismatchError
from .onemap import transformer_layers, value
from .ops import neg
from .pure import FLOOR_MOVE_TO, compile_pure
from .readout import LoadThenReadout, readout_commands
from .selfgen import (BOOT, LOOP, PH_BOOT, PH_COMPUTE, PH_DONE, PH_HOLD, POLL, REWIND, STATUS, STATUS_NAMES, STILL,
                      LoopRun, Pass, hold_tics, initial_tokens, iterate, port_tags, positions, text, write_actions)
from .tile import DOLL_TID0, MOTION_SCRIPT, MOVE_SPEED, PAGE, Block, Gate, Stage, TileMeta

LATCH_LIMIT = 1             # the latch lane's gate: passes iff halt:t <= 1, i.e. always
LOAD = 4                    # tics from the first host tic to settled loads (one tic, the load scripts, <= 2 mover tics)
STATUS_UNTIL = STATUS_NAMES + ("latch",)     # "passes" is the bound; "latch" the frontier latch the level read last


# --- the reference ----------------------------------------------------------------------------------------------------
def reference(m: QModel, prompt: str = "") -> list[Pass]:
    """selfgen.iterate plus trace["halt"][t] = [winner at t is EOS] at every writing position t (0 elsewhere)."""
    passes = iterate(m, prompt)
    pos = positions(m, prompt)
    for p in passes:
        latch = np.zeros(m.T, dtype=np.int64)
        latch[pos] = [int(a == 0) for a in p.winners]
        p.trace["halt"] = latch
    return passes


def stop_pass(passes: list[Pass]) -> int:
    """The pass (from 1) after which the level leaves its loop: the first whose frontier winner is EOS, else the
    last."""
    for k, p in enumerate(passes, 1):
        if p.winners[k - 1] == 0:
            return k
    return len(passes)


# --- the map ----------------------------------------------------------------------------------------------------------
def writing_positions(m: QModel, prompt_length: int) -> list[int]:
    """selfgen.positions for any prompt of this length."""
    if not 0 <= prompt_length <= m.T - 2:
        raise ValueError(f"prompt length: 0 to {m.T - 2}, so that one position is generated")
    return list(range(prompt_length, m.T - 1))


def layers_until(m: QModel, pos: list[int]) -> list:
    """selfgen's layers plus one latch lane per writing position in layer "embed" (see the module docstring)."""
    layers = transformer_layers(m, list(range(len(m.vocab))), pos)
    embed = layers[0]
    for t in pos:
        embed.blocks.append(Block([Gate(embed.port(value("halt", t), "floor"), LATCH_LIMIT, "floor")],
                                  [Stage(None, [])]))
    return layers


def latch_actions(m: QModel, layers, pos: list[int], miswire: dict | None = None) -> dict[str, list[dict]]:
    """selfgen.write_actions plus the latch line: the winner of argz:t:a moves port halt:t to [a == 0]. `miswire`
    {(t, a): h} makes that line write h instead (negative control only)."""
    tags = port_tags(layers)
    actions = write_actions(m, layers, pos)
    for t in pos:
        for a in range(len(m.vocab)):
            h = (miswire or {}).get((t, a), int(a == 0))
            actions[value("argz", t, a)].append(dict(special=FLOOR_MOVE_TO, arg0=tags["floor:" + value("halt", t)],
                                                     arg1=MOVE_SPEED, arg2=abs(h), arg3=int(h < 0)))
    return actions


def level_acs_until(n_dolls: int, bound: int, hold: int, boot: int, latch_tags: list[int]) -> str:
    """selfgen.level_acs with an upper bound of passes and an exit: after pass p settles, read the latch of writing
    position p (one GetSectorFloorZ), hold, and leave the loop if it was >= 1. STATUS adds o7 = the last latch read."""
    if len(latch_tags) != bound:
        raise ValueError("one latch per pass")
    return f"""
// --- generation loop that stops at EOS (doomformer/selfhalt.py) ---
int sg_x[{n_dolls}]; int sg_y[{n_dolls}]; int sg_z[{n_dolls}];
int sg_latch_tag[{bound}] = {{ {", ".join(map(str, latch_tags))} }};
int sg_phase = {PH_BOOT}; int sg_pass; int sg_start = -1; int sg_hold = -1; int sg_latch = -1;
function int sg_moved(void) {{
  int moved = 0;
  for (int i = 0; i < {n_dolls}; i++) {{
    int tid = {DOLL_TID0} + i;
    if (!IsTIDUsed(tid)) {{ moved++; continue; }}
    int x = GetActorX(tid); int y = GetActorY(tid); int z = GetActorZ(tid);
    if (x != sg_x[i] || y != sg_y[i] || z != sg_z[i]) {{ moved++; sg_x[i] = x; sg_y[i] = y; sg_z[i] = z; }}
  }}
  return moved;
}}
script {LOOP} OPEN {{
  Delay({boot});
  for (int p = 0; p < {bound}; p++) {{
    if (p > 0) {{ ACS_ExecuteAlways(3, 0); Delay({REWIND}); }}
    sg_phase = {PH_COMPUTE}; sg_start = Timer();
    ACS_ExecuteAlways(7, 0);
    Delay({STILL});
    sg_moved();
    do {{ Delay({STILL}); }} while (sg_moved());
    sg_latch = GetSectorFloorZ(sg_latch_tag[p], 0, 0) >> 16;
    sg_pass = p + 1; sg_hold = Timer(); sg_phase = {PH_HOLD};
    Delay({hold});
    if (sg_latch >= 1) break;
  }}
  sg_phase = {PH_DONE};
}}
script {STATUS} (void) {{
  o1 = sg_phase; o2 = sg_pass; o3 = sg_start; o4 = sg_hold; o5 = Timer(); o6 = {bound}; o7 = sg_latch;
}}
"""


def boot_tics(state_pages: int) -> int:
    """Boot wait: the loads, two status reads, every page holding a state value read with its guard (two tics a page;
    these maps are guarded) and a margin: 32 tics less the guard tics, at least 8. Up to 24 state pages the wait is as
    long as it was before the guard."""
    return BOOT + LOAD + 2 + 2 * state_pages + max(32 - state_pages, 8)


def build_until(m: QModel, wad: Path, prompt_length: int, *, layout_only: bool = False,
                miswire: dict | None = None) -> TileMeta:
    """Compile the map for every prompt of `prompt_length` characters (the prompt is not in the map: every token port
    starts at 0, UntilEosRun loads the prompt). meta.layout["selfhalt"] holds the schedule UntilEosRun needs."""
    pos = writing_positions(m, prompt_length)
    layers = layers_until(m, pos)
    state = {k: 0 for t in range(m.T) for k in (value("tok", t), neg(value("tok", t)))}
    state.update({value("halt", t): 0 for t in pos})
    consumed = {key.split(":", 1)[1] for s in layers for key in s.ports}
    readout = [name for s in layers for name, _ in s.outputs if name in consumed] + list(state)
    n_dolls = sum(spec.n_dolls + 1 for spec in layers)
    reads = sum(len(s.outputs) for s in layers) + len(state)     # every output is a light or a readout plane
    pages = -(-reads // PAGE)
    if 100 + pages > LOOP:
        raise ValueError(f"selfhalt: {pages} readout pages would reach the level's script {LOOP}")
    state_pages = list(range((reads - len(state)) // PAGE, pages))   # the state values are the last reads
    hold, boot = hold_tics(pages), boot_tics(len(state_pages))
    tags = port_tags(layers)
    latch_tags = [tags["floor:" + value("halt", t)] for t in pos]

    def world(mb) -> str:
        mb.rect(-768, -512, -512, -256, floor=0, ceil=128)
        mb.thing(-640, -384, 1)                                    # last player-1 start = the player
        return level_acs_until(n_dolls, len(pos), hold, boot, latch_tags)

    meta = compile_pure(layers, wad, "selfhalt", state=state, readout=readout, world=world,
                        actions=latch_actions(m, layers, pos, miswire), layout_only=layout_only)
    names = [n for page in meta.pages for n in page]
    if (meta.n_dolls != n_dolls or len(meta.outputs) != reads or meta.layout["tags"] != tags
            or names[-len(state):] != list(state)):
        raise AssertionError("selfhalt: the loop's plan and the emitted map differ")
    meta.layout["selfhalt"] = dict(prompt_length=prompt_length, positions=pos, passes=len(pos), hold=hold, boot=boot,
                                   state_pages=state_pages, latch_tags=latch_tags)
    if not layout_only:
        meta.save(Path(wad).with_suffix(".json"))
    return meta


def prompt_loads(m: QModel, meta: TileMeta, prompt: str) -> list[tuple[int, int, str]]:
    """(tag, height, kind) of every port carrying a prompt token: tok:t and -tok:t for t = 1..len(prompt)."""
    toks = initial_tokens(m, prompt)
    by_value: dict[str, list[str]] = {}
    for key in meta.layout["tags"]:
        by_value.setdefault(key.split(":", 1)[1], []).append(key)
    loads = []
    for t in range(1, 1 + len(prompt)):
        for name, h in ((value("tok", t), toks[t]), (neg(value("tok", t)), -toks[t])):
            if name not in by_value:
                raise ValueError(f"no port carries {name}")
            loads += [(meta.layout["tags"][key], h, key.split(":", 1)[0]) for key in by_value[name]]
    return loads


# --- running it -------------------------------------------------------------------------------------------------------
class UntilEosRun(LoopRun):
    """The live game of a `build_until` map for `prompt`. `load()` writes the prompt's tokens (the only host writes),
    seals and checks the state values; then readout only. `next_pass()` reads and checks a pass like LoopRun, plus
    the latches and the latch the level read; `run()` also checks that the level stops after `stop` passes."""

    def __init__(self, m: QModel, wad: Path, prompt: str = "", resolution=None):
        from .runtime import USER, TileRunner
        self.m, self._user, self.prompt = m, USER, prompt
        kw = {} if resolution is None else dict(resolution=resolution)
        self.runner = TileRunner(wad, **kw)
        self.meta = self.runner.meta
        self.schedule = self.meta.layout["selfhalt"]
        if self.schedule["positions"] != positions(m, prompt):
            raise ValueError(f"this map is for prompts of {self.schedule['prompt_length']} characters, not "
                             f"{len(prompt)}")
        self.game = self.runner.game = LoadThenReadout(self.runner.game, readout_commands(
            len(self.meta.pages), f"puke -{STATUS}", f"puke -{MOTION_SCRIPT}"))
        self.reference = reference(m, prompt)
        self.stop = stop_pass(self.reference)
        self.loads = prompt_loads(m, self.meta, prompt)
        self.boot: dict = {}
        self.passes: list[dict] = []

    def status(self) -> dict[str, int]:
        return dict(zip(STATUS_UNTIL, self._script(STATUS, len(STATUS_UNTIL))))

    def load(self) -> dict:
        """Write the prompt into the token ports, seal, then read every state value back inside the boot wait:
        tok:t and -tok:t must hold the prompt's tokens and the placeholders, every latch 0."""
        if self.game.sealed:
            raise RuntimeError("the prompt is already loaded")
        self.game.advance_action(1)
        for tag, h, kind in self.loads:
            self.game.load(tag, h, kind)
        self.game.advance_action(LOAD - 1)
        self.game.seal()
        before = self.status()
        got = self.runner.read(self.schedule["state_pages"])     # poisoned and checked page by page, as every read
        after = self.status()
        if before["phase"] != PH_BOOT or after["phase"] != PH_BOOT:
            raise RuntimeError(f"the loads were not checked inside the level's boot wait: before {before}, "
                               f"after {after}")
        toks = initial_tokens(self.m, self.prompt)
        want = {value("tok", t): toks[t] for t in range(self.m.T)}
        want.update({neg(value("tok", t)): -toks[t] for t in range(self.m.T)})
        want.update({value("halt", t): 0 for t in self.schedule["positions"]})
        bad = [(name, got.get(name), v) for name, v in want.items() if got.get(name) != v]
        if bad:
            raise MismatchError(f"state values before pass 1 (prompt {self.prompt!r})", bad)
        self.boot = dict(loads=len(self.loads), checked=len(want), timer=after["timer"], boot=self.schedule["boot"])
        return self.boot

    def next_pass(self, max_tics: int = 200_000) -> dict:
        if not self.game.sealed:
            raise RuntimeError("load() the prompt before the first pass")
        s = self.status()
        if s["phase"] == PH_DONE:
            raise RuntimeError(f"the level left its loop after pass {s['pass']}; pass {len(self.passes) + 1} never "
                               "ran")
        out = super().next_pass(max_tics)
        k = out["index"]
        t = self.schedule["positions"][k - 1]
        want = int(self.reference[k - 1].trace["halt"][t])
        if out["status"]["latch"] != want:
            raise AssertionError(f"pass {k}: the level read latch halt:{t} = {out['status']['latch']}, "
                                 f"reference {want}")
        out["latch"] = want
        return out

    def after_hold(self, k: int, max_tics: int = 20_000) -> dict[str, int]:
        """Advance the clock until the HOLD window of pass k ends; the status then (the next pass computes, or done)."""
        waited = 0
        while True:
            s = self.status()
            if s["phase"] != PH_HOLD or s["pass"] != k:
                return s
            if waited > max_tics:
                raise RuntimeError(f"the HOLD window of pass {k} did not end within {max_tics} tics: {s}")
            self.game.advance_action(POLL)
            waited += POLL

    def run(self, on_pass=None) -> str:
        """Load the prompt, then every pass, checked, until the level leaves its loop. It must leave it after exactly
        `stop` passes; returns the text DOOM wrote (it equals intref.generate or this raises)."""
        from .intref import generate
        if not self.game.sealed:
            self.load()
        while True:
            p = self.next_pass()
            if on_pass is not None:
                on_pass(p)
            k = p["index"]
            s = self.after_hold(k)
            if s["phase"] == PH_DONE:
                if k != self.stop or s["pass"] != k:
                    raise AssertionError(f"the level left its loop after pass {s['pass']}; the reference stops "
                                         f"after pass {self.stop}")
                break
            if k == self.stop:
                raise AssertionError(f"pass {k}: the frontier wrote EOS (or the bound is reached) but the level "
                                     f"ran on: {s}")
        wrote = text(self.m, self.passes[-1]["tokens"], self.prompt)
        ref = generate(self.m, self.prompt, temperature=0)
        if wrote != ref:
            raise AssertionError(f"DOOM wrote {wrote!r}, intref.generate {ref!r}")
        return wrote
