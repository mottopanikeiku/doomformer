"""Generation without Python: one map writes a whole text, the level repeats its own passes.

Greedy decoding as a fixed-point (Jacobi) iteration. The context starts as tokens = [BOS] + prompt + placeholders
(token 0) up to T. One pass computes the logits at every position; then each writing position t (len(prompt) <=
t <= T-2) writes the argmax of its logits (lowest index on ties) into tokens[t+1]. Attention is causal, so after pass
p (counting from 1) positions up to len(prompt) + p hold their final token, and T-1-len(prompt) passes give exactly
intref.generate(m, prompt, temperature=0) (up to its first EOS; positions after it may hold anything). `iterate` is
the NumPy reference of every pass.

The map (`build`):
  * the brain is onemap.transformer_layers with comparators over the whole vocabulary at every writing position
    (choose = all tokens, positions = writing positions): unembed writes d_ab = logit_a - logit_b + 1 per pair,
    layer "argmax.compare" counts wins, layer "argmax.winner" lights the one-hot argz:t:a.
  * the token ports tok:t and -tok:t are state values (compile_pure `state`): they start at the prompt's tokens and
    0 (placeholders), and the rewind (script 3) leaves them alone.
  * token writes are mechanical: the winner doll of position t (its argz:t:a lane is the only one whose gate opens)
    crosses two lines with Floor_MoveToValue specials that move port tok:t+1 to height a and -tok:t+1 to -a.
    No ACS chooses or computes a token.
  * the level's loop (OPEN script LOOP) runs exactly T-1-len(prompt) passes: rewind (script 3, from the second pass
    on), drop the start barriers (script 7), then sample every doll's position every STILL tics until no doll moved
    during one window (the test runtime.TileRunner.settle runs from Python), then HOLD the finished pass for `hold`
    tics while Python reads it, then the next pass. The ACS reads positions and counts tics; it reads no value.

Python (`LoopRun`) only advances the game clock and runs readout scripts (pages 100.., STATUS, MOTION) through
readout.ReadoutOnly. In every HOLD it reads every exposed value (all positions, comparators, the token ports after the
writes), confirms with the motion script and STATUS that nothing moved and the window did not end while it read, and
compares every value with the pass of `iterate` (tolerance 0).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .intref import QModel, forward
from .kit import MismatchError
from .onemap import argmax_from_readout, argmax_trace, transformer_layers, value
from .ops import neg
from .pure import FLOOR_MOVE_TO, compile_pure
from .readout import ReadoutOnly, readout_commands
from .tile import DOLL_TID0, MOTION_SCRIPT, MOVE_SPEED, PAGE, PORT_TAG0, TileMeta

LOOP, STATUS = 900, 901     # level scripts (above the readout pages 100.., which pass 200 in the shipped maps)
BOOT = 8                    # tics before the first pass (every doll has stepped onto its home line and tagged itself)
REWIND = 3                  # tics after the rewind script (its plane movers need <= 2)
STILL = 8                   # tics between the level's motion samples (TileRunner.settle's default window)
POLL = 16                   # tics between Python's status reads while a pass computes
PH_BOOT, PH_COMPUTE, PH_HOLD, PH_DONE = 0, 1, 2, 3
STATUS_NAMES = ("phase", "pass", "start", "hold", "timer", "passes")


# --- the reference ----------------------------------------------------------------------------------------------------
def positions(m: QModel, prompt: str) -> list[int]:
    """Context positions whose argmax the dolls write into the next position: len(prompt) .. T-2."""
    if len(prompt) > m.T - 2:
        raise ValueError(f"prompt: at most {m.T - 2} characters, so that one position is generated")
    bad = [c for c in prompt if c not in m.vocab]
    if bad:
        raise ValueError(f"prompt characters {bad} are not in the vocabulary {m.vocab!r}")
    return list(range(len(prompt), m.T - 1))


def initial_tokens(m: QModel, prompt: str) -> list[int]:
    positions(m, prompt)
    toks = [0] + [m.vocab.index(c) for c in prompt]
    return toks + [0] * (m.T - len(toks))


def text(m: QModel, tokens: list[int], prompt: str) -> str:
    """The prompt and the generated tokens up to (without) the first EOS (token 0), as intref.generate returns it."""
    gen = list(tokens[1 + len(prompt):])
    if 0 in gen:
        gen = gen[:gen.index(0)]
    return prompt + "".join(m.vocab[t] for t in gen)


@dataclass
class Pass:
    tokens: list[int]           # the context the pass reads
    written: list[int]          # the context after the pass's writes
    winners: list[int]          # argmax at each writing position
    logits: np.ndarray          # [T, V]
    trace: dict = field(repr=False)


def iterate(m: QModel, prompt: str = "") -> list[Pass]:
    """Every pass of the fixed-point iteration (T-1-len(prompt) of them); trace holds intref.forward's values and the
    comparators (onemap.argmax_trace) at the writing positions."""
    pos, toks = positions(m, prompt), initial_tokens(m, prompt)
    out = []
    for _ in range(len(pos)):
        trace = {}
        logits = forward(m, toks, trace)
        winners = argmax_trace(m, list(range(len(m.vocab))), logits, trace, pos)
        written = list(toks)
        for t, a in zip(pos, winners):
            written[t + 1] = a
        out.append(Pass(toks, written, winners, logits, trace))
        toks = written
    return out


# --- the map -----------------------------------------------------------------------------------------------------------
def port_tags(layers) -> dict[str, int]:
    """compile_pure's tag of every port key (PORT_TAG0 + order of first appearance)."""
    tags: dict[str, int] = {}
    for spec in layers:
        for key in spec.ports:
            tags.setdefault(key, PORT_TAG0 + len(tags))
    return tags


def write_actions(m: QModel, layers, pos: list[int], miswrite: dict | None = None) -> dict[str, list[dict]]:
    """Winner lines: the doll of argz:t:a moves port tok:t+1 to a and -tok:t+1 to -a. `miswrite` {(t, a): b} makes
    that line write b instead (negative control only)."""
    tags = port_tags(layers)
    actions = {}
    for t in pos:
        for a in range(len(m.vocab)):
            b = (miswrite or {}).get((t, a), a)
            tok = value("tok", t + 1)
            actions[value("argz", t, a)] = [
                dict(special=FLOOR_MOVE_TO, arg0=tags["floor:" + key], arg1=MOVE_SPEED, arg2=b, arg3=int(sign < 0))
                for key, sign in ((tok, 1), (neg(tok), -1))]
    return actions


def level_acs(n_dolls: int, passes: int, hold: int) -> str:
    """The level's loop and STATUS readout (see the module docstring). sg_moved compares every doll with its position
    at the previous sample; an untagged doll counts as moving."""
    return f"""
// --- generation loop (doomformer/selfgen.py) ---
int sg_x[{n_dolls}]; int sg_y[{n_dolls}]; int sg_z[{n_dolls}];
int sg_phase = {PH_BOOT}; int sg_pass; int sg_start = -1; int sg_hold = -1;
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
  Delay({BOOT});
  for (int p = 0; p < {passes}; p++) {{
    if (p > 0) {{ ACS_ExecuteAlways(3, 0); Delay({REWIND}); }}
    sg_phase = {PH_COMPUTE}; sg_start = Timer();
    ACS_ExecuteAlways(7, 0);
    Delay({STILL});
    sg_moved();
    do {{ Delay({STILL}); }} while (sg_moved());
    sg_pass = p + 1; sg_hold = Timer(); sg_phase = {PH_HOLD};
    Delay({hold});
  }}
  sg_phase = {PH_DONE};
}}
script {STATUS} (void) {{ o1 = sg_phase; o2 = sg_pass; o3 = sg_start; o4 = sg_hold; o5 = Timer(); o6 = {passes}; }}
"""


def hold_tics(meta_pages: int) -> int:
    """HOLD window: one poll interval to notice it, every readout page, four status/motion reads, a margin."""
    return POLL + meta_pages + 4 + 32


def build(m: QModel, wad: Path, prompt: str = "", *, layout_only: bool = False,
          miswrite: dict | None = None, opt: int = 0) -> TileMeta:
    """Compile the self-generating map for `prompt` (baked in: its tokens are the token ports' initial heights).
    meta.layout["selfgen"] holds the schedule LoopRun needs. opt: the brain's schedule level, as df.compile."""
    pos = positions(m, prompt)
    init = initial_tokens(m, prompt)
    layers = transformer_layers(m, list(range(len(m.vocab))), pos)
    state = {k: v for t in range(m.T) for k, v in ((value("tok", t), init[t]), (neg(value("tok", t)), -init[t]))}
    consumed = {key.split(":", 1)[1] for s in layers for key in s.ports}
    readout = [name for s in layers for name, _ in s.outputs if name in consumed] + list(state)
    n_dolls = sum(spec.n_dolls + 1 for spec in layers)
    reads = sum(len(s.outputs) for s in layers) + len(state)     # every output is a light or a readout plane
    pages = -(-reads // PAGE)
    if 100 + pages > LOOP:
        raise ValueError(f"selfgen: {pages} readout pages would reach the level's script {LOOP}")
    hold = hold_tics(pages)

    def world(mb) -> str:
        mb.rect(-768, -512, -512, -256, floor=0, ceil=128)
        mb.thing(-640, -384, 1)                                    # last player-1 start = the player
        return level_acs(n_dolls, len(pos), hold)

    meta = compile_pure(layers, wad, "selfgen", state=state, readout=readout, world=world,
                        actions=write_actions(m, layers, pos, miswrite), layout_only=layout_only, opt=opt)
    if meta.n_dolls != n_dolls or len(meta.outputs) != reads or meta.layout["tags"] != port_tags(layers):
        raise AssertionError("selfgen: the loop's plan and the emitted map differ")
    meta.layout["selfgen"] = dict(prompt=prompt, positions=pos, passes=len(pos), hold=hold, tokens=init)
    if not layout_only:
        meta.save(Path(wad).with_suffix(".json"))
    return meta


# --- running it ---------------------------------------------------------------------------------------------------------
def _expected(name: str, p: Pass) -> int:
    negative = name.startswith("-")
    key, *raw = name.lstrip("-").split(":")
    idx = tuple(map(int, raw))
    v = p.written[idx[0]] if key == "tok" else int(p.trace[key][idx])
    return -v if negative else v


class LoopRun:
    """The live game of a `build` map, readout only. `next_pass()` waits for the level's next HOLD window, reads every
    exposed value and checks it against `iterate` (MismatchError on any difference)."""

    def __init__(self, m: QModel, wad: Path, resolution=None):
        from .runtime import USER, TileRunner
        self.m, self._user = m, USER
        kw = {} if resolution is None else dict(resolution=resolution)
        self.runner = TileRunner(wad, **kw)
        self.meta = self.runner.meta
        self.schedule = self.meta.layout["selfgen"]
        self.game = self.runner.game = ReadoutOnly(self.runner.game, readout_commands(
            len(self.meta.pages), f"puke -{STATUS}", f"puke -{MOTION_SCRIPT}"))
        self.reference = iterate(m, self.schedule["prompt"])
        if self.schedule["tokens"] != self.reference[0].tokens or self.schedule["positions"] != positions(
                m, self.schedule["prompt"]):
            raise ValueError("this map was built for another model or prompt")
        self.passes: list[dict] = []

    def _script(self, script: int, n: int) -> list[int]:
        self.game.send_game_command(f"puke -{script}")
        self.game.advance_action(1)
        return [int(self.game.get_game_variable(v)) for v in self._user[:n]]

    def status(self) -> dict[str, int]:
        return dict(zip(STATUS_NAMES, self._script(STATUS, len(STATUS_NAMES))))

    def motion(self) -> tuple[int, int]:
        """(dolls that moved since the previous motion read, dolls without their TID)."""
        moved, untagged = self._script(MOTION_SCRIPT, 2)
        return moved, untagged

    def next_pass(self, max_tics: int = 200_000) -> dict:
        k = len(self.passes) + 1
        if k > self.schedule["passes"]:
            raise RuntimeError("every pass has run")
        waited = 0
        while True:
            s = self.status()
            if s["phase"] == PH_HOLD and s["pass"] == k:
                break
            if s["pass"] >= k or s["phase"] == PH_DONE:
                raise RuntimeError(f"missed the HOLD window of pass {k}: {s}")
            if waited > max_tics:
                raise RuntimeError(f"pass {k} did not finish within {max_tics} tics: {s}")
            self.game.advance_action(POLL)
            waited += POLL
        self.motion()
        got = self.runner.read()
        moved, untagged = self.motion()
        after = self.status()
        if after["phase"] != PH_HOLD or after["pass"] != k or after["hold"] != s["hold"] or moved or untagged:
            raise RuntimeError(f"pass {k} was not read inside a still HOLD window: before {s}, after {after}, "
                               f"{moved} dolls moved, {untagged} untagged")
        return self.check(k, got, s)

    def check(self, k: int, got: dict[str, int], status: dict[str, int]) -> dict:
        ref = self.reference[k - 1]
        if set(got) != set(self.meta.outputs):
            raise AssertionError("incomplete readout")
        bad = [(name, v, _expected(name, ref)) for name, v in got.items() if v != _expected(name, ref)]
        if bad:
            raise MismatchError(f"self-generation pass {k} (context {ref.tokens})", bad)
        tokens = [got[value("tok", t)] for t in range(self.m.T)]
        winners = [argmax_from_readout(got, len(self.m.vocab), t) for t in self.schedule["positions"]]
        if tokens != ref.written or winners != ref.winners:
            raise AssertionError(f"pass {k}: DOOM wrote {tokens} (winners {winners}), reference {ref.written}")
        out = dict(index=k, read=ref.tokens, tokens=tokens, winners=winners, checked=len(got),
                   tics=status["hold"] - status["start"], status=status)
        self.passes.append(out)
        return out

    def run(self, on_pass=None) -> str:
        """Every pass, checked; returns the text DOOM wrote (it equals intref.generate or this raises)."""
        from .intref import generate
        while len(self.passes) < self.schedule["passes"]:
            p = self.next_pass()
            if on_pass is not None:
                on_pass(p)
        prompt = self.schedule["prompt"]
        wrote = text(self.m, self.passes[-1]["tokens"], prompt)
        ref = generate(self.m, prompt, temperature=0)
        if wrote != ref:
            raise AssertionError(f"DOOM wrote {wrote!r}, intref.generate {ref!r}")
        return wrote

    def audit(self) -> dict:
        return self.game.audit()

    def close(self):
        self.runner.close()

    def __enter__(self) -> "LoopRun":
        return self

    def __exit__(self, *exc):
        self.close()
