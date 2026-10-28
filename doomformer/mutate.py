"""Mutation harness: how much of a compiled map does the check actually test?

`doomformer fuzz --mutate` compiles a small MLP, then mutates the compiled map one site at a time and runs the net's
inputs through each mutant against the integer reference. A mutant is KILLED when any pass differs from the
reference (or the game fails to settle, crashes or hits the wall-clock limit: a broken map is not a passing map) and
SURVIVES when every pass is exact. Mutation sites:

  TEXTMAP  every linedef special argument (weights, targets, mover speeds, teleports, TIDs, scrollers), special
           numbers (raise <-> lower), sector floor / ceiling heights and light levels, thing ids;
  ACS      every integer literal of the compiled script source (the reset script, the readout pages, the port
           scripts, the motion script), changed by +-1, and the `>> 16` of the plane readouts.

The ACS source is captured from the compiler (the lump is bytecode) and recompiled with acc after the edit.
Survivors get a one-line reason from a rule table (`REASONS`): the rules say which survivors are expected to be
equivalent mutants (a value the reset script overwrites, a speed no pass depends on, ...). A survivor no rule explains
is listed as unexplained; those are the ones to read.

    from doomformer.mutate import run_mutation, demo_net
    report = run_mutation(demo_net(), per_class=20, workers=3)
"""
from __future__ import annotations

import json
import queue
import random
import re
import shutil
import struct
import tempfile
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from . import _mp

LIGHT_RAISE, LIGHT_LOWER, FLOOR_LOWER, FLOOR_RAISE, FLOOR_MOVE_TO, CEIL_LOWER, CEIL_RAISE = 110, 111, 20, 23, 37, 40, 41
LOWER_TO_LOWEST, EXEC_ALWAYS, TELEPORT, SCROLL = 21, 226, 71, 223
PAIRS = {LIGHT_RAISE: LIGHT_LOWER, LIGHT_LOWER: LIGHT_RAISE, FLOOR_RAISE: FLOOR_LOWER, FLOOR_LOWER: FLOOR_RAISE,
         CEIL_RAISE: CEIL_LOWER, CEIL_LOWER: CEIL_RAISE}


# --- WAD lumps ------------------------------------------------------------------------------------------------------
def read_lumps(path) -> list[tuple[str, bytes]]:
    data = Path(path).read_bytes()
    n, directory = struct.unpack("<ii", data[4:12])
    out = []
    for i in range(n):
        offset, size, name = struct.unpack("<ii8s", data[directory + 16 * i:directory + 16 * (i + 1)])
        out.append((name.rstrip(b"\0").decode(), data[offset:offset + size]))
    return out


def write_lumps(path, lumps: list[tuple[str, bytes]]) -> None:
    data, directory = b"", b""
    for name, blob in lumps:
        directory += struct.pack("<ii8s", 12 + len(data), len(blob), name.encode())
        data += blob
    Path(path).write_bytes(struct.pack("<4sii", b"PWAD", len(lumps), 12 + len(data)) + data + directory)


def edit_textmap(src, dst, fn) -> None:
    """Copy the WAD `src` (and its .json) to `dst` with TEXTMAP replaced by fn(text)."""
    lumps = [(n, fn(b.decode()).encode() if n == "TEXTMAP" else b) for n, b in read_lumps(src)]
    write_lumps(dst, lumps)
    shutil.copyfile(Path(src).with_suffix(".json"), Path(dst).with_suffix(".json"))


def tamper_weight(src, dst, delta: int = 1) -> str:
    """dst := src with the first light weight line (Light_RaiseByValue / Light_LowerByValue amount) changed by
    `delta`; returns a description. In compile order the first such line belongs to layer 0, input 0, lowest lane."""
    done = []
    pattern = re.compile(r"(special = (110|111); arg0 = (\d+); arg1 = )(\d+);")

    def repl(m):
        if done:
            return m.group(0)
        old = int(m.group(4))
        name = "Light_RaiseByValue" if m.group(2) == "110" else "Light_LowerByValue"
        done.append(f"{name} tag {m.group(3)}: {old} -> {old + delta}")
        return f"{m.group(1)}{old + delta};"

    edit_textmap(src, dst, lambda text: pattern.sub(repl, text))
    if not done:
        raise ValueError(f"{src}: no light weight line to tamper with")
    return done[0]


# --- the net and its inputs -----------------------------------------------------------------------------------------
def demo_net():
    """The default mutation target: a 3-4-3 two-layer MLP with signed inputs (light AND plane weights)."""
    from .intref import adc_signed
    from .kit import MLP
    rng = np.random.default_rng(7)
    W = [rng.integers(-9, 10, (4, 3)), rng.integers(-9, 10, (3, 4))]
    b = [rng.integers(-5, 6, 4), rng.integers(-5, 6, 3)]
    t0 = np.array([2, 4, 6])
    xs = rng.integers(-12, 13, (64, 3))
    h = adc_signed(xs, t0) @ W[0].T + b[0]
    t1 = np.sort(np.unique(np.clip(h[h >= 1], 1, None)))[::3][:3]
    return MLP(W, b, [t0, t1], signed_input=True)


# --- sites ----------------------------------------------------------------------------------------------------------
@dataclass
class Mutant:
    id: int
    cls: str
    where: str
    kind: str            # "textmap" | "acs"
    line: int            # textmap line index, or ACS source offset of the literal
    old: str
    new: str
    info: dict


_KV = re.compile(r"(\w+) = ([^;]+);")


def _kv(line: str) -> dict[str, str]:
    return {k: v.strip() for k, v in _KV.findall(line)}


def _set(line: str, key: str, value) -> str:
    return re.sub(rf"\b{key} = [^;]+;", f"{key} = {value};", line, count=1)


def textmap_mutants(text: str, meta, start: int = 0) -> list[Mutant]:
    """All single-site TEXTMAP mutants of one compiled map (sectors: +-1 on floor, ceiling, light; linedefs: +-1 on each
    argument, raise <-> lower, effect retargeting)."""
    out: list[Mutant] = []
    lines = text.split("\n")
    port_tags = set(meta.ports.values()) if meta.ports else set()
    gate_tags = port_tags | {int(t) for t in meta.layout.get("tags", {}).values()}
    out_tags = {int(t) for t in re.findall(r"\((\d+)\)", " ".join(meta.layout.get("read_expressions", {}).values()))
                if int(t) >= 20000}
    effect_tags: dict[int, list[int]] = {}
    for ln in lines:
        if ln.startswith("linedef"):
            kv = _kv(ln)
            s = int(kv.get("special", 0))
            if s in PAIRS:
                effect_tags.setdefault(s, []).append(int(kv["arg0"]))
    sector_i = -1

    def add(cls, where, i, new, **info):
        out.append(Mutant(start + len(out), cls, where, "textmap", i, lines[i], new, info))

    for i, ln in enumerate(lines):
        if ln.startswith("sector"):
            sector_i += 1
            kv = _kv(ln)
            tag = int(kv["id"])
            role = ("gate" if tag in gate_tags else "output" if tag in out_tags
                    else f"walk:{kv['texturefloor'].strip(chr(34))}/{kv['lightlevel']}" if tag == 1
                    else "barrier" if tag == 2 or tag >= 60000 else "other")
            for key, cls in (("heightfloor", f"sector-floor-{role}"), ("heightceiling", f"sector-ceiling-{role}"),
                             ("lightlevel", f"sector-light-{role}")):
                for d in (-1, 1):
                    v = int(kv[key]) + d
                    add(cls, f"sector {sector_i} (tag {tag}) {key} {kv[key]} -> {v}", i, _set(ln, key, v),
                        role=role, key=key, tag=tag)
        elif ln.startswith("linedef") and "special" in ln:
            kv = _kv(ln)
            s = int(kv["special"])
            for k in range(5):
                key = f"arg{k}"
                if key not in kv:
                    continue
                for d in (-1, 1):
                    v = int(kv[key]) + d
                    add(_line_class(s, k), f"linedef {i} special {s} {key} {kv[key]} -> {v}", i, _set(ln, key, v),
                        special=s, arg=k)
            if s in PAIRS:
                add("effect-sign", f"linedef {i} special {s} -> {PAIRS[s]}", i, _set(ln, "special", PAIRS[s]),
                    special=s, arg=-1)
                tags = sorted(set(effect_tags[s]))
                other = tags[(tags.index(int(kv["arg0"])) + 1) % len(tags)]
                if other != int(kv["arg0"]):
                    add("effect-target", f"linedef {i} special {s} arg0 {kv['arg0']} -> {other} (another target)", i,
                        _set(ln, "arg0", other), special=s, arg=0)
    return out


def _line_class(s: int, k: int) -> str:
    if s in (LIGHT_RAISE, LIGHT_LOWER):
        return {0: "light-weight-tag", 1: "light-weight"}.get(k, "line-other")
    if s in (FLOOR_RAISE, FLOOR_LOWER, CEIL_RAISE, CEIL_LOWER):
        return {0: "plane-weight-tag", 1: "plane-mover-speed", 2: "plane-weight"}.get(k, "line-other")
    if s == FLOOR_MOVE_TO:
        return {0: "move-to-tag", 1: "move-to-speed", 2: "move-to-height"}.get(k, "line-other")
    if s == LOWER_TO_LOWEST:
        return {0: "barrier-tag", 1: "barrier-speed"}.get(k, "line-other")
    if s == EXEC_ALWAYS:
        return {0: "exec-script", 1: "exec-map", 2: "doll-tid"}.get(k, "line-other")
    if s == TELEPORT:
        return "teleport-tid"
    if s == SCROLL:
        return "scroller"
    return "line-other"


_ACS_SCRIPT = re.compile(r"\bscript\s+(\d+)")
_ACS_STMT = re.compile(r"[^;{}]+;")
_INT = re.compile(r"(?<![\w.])(\d+)(?![\w.])")


def _call_context(stmt: str, pos: int) -> tuple[str, int | None]:
    """(innermost call name, argument index) at offset `pos` of an ACS statement."""
    stack: list[list] = []
    for i, ch in enumerate(stmt[:pos]):
        if ch == "(":
            name = re.search(r"(\w+)\s*$", stmt[:i])
            stack.append([name.group(1) if name else "", 0])
        elif ch == ")" and stack:
            stack.pop()
        elif ch == "," and stack:
            stack[-1][1] += 1
    return (stack[-1][0], stack[-1][1]) if stack else ("", None)


def acs_mutants(source: str, start: int = 0) -> list[Mutant]:
    """Every integer literal of a compiled script, +-1, plus the `>> 16` of plane readouts as `>> 15`."""
    out: list[Mutant] = []
    headers = [(m.start(), int(m.group(1))) for m in _ACS_SCRIPT.finditer(source)]

    def script_at(pos: int) -> int | None:
        cur = None
        for p, n in headers:
            if p <= pos:
                cur = n
        return cur

    for sm in _ACS_STMT.finditer(source):
        stmt = sm.group(0)
        if stmt.lstrip().startswith(("#", "global")) or "script" in stmt.split("(")[0] and "{" in stmt:
            continue
        head = re.sub(r"^\s*(script\s+\d+\s*\([^)]*\)\s*\{\s*)", "", stmt).strip()
        sc = script_at(sm.start())
        for im in _INT.finditer(stmt):
            n = int(im.group(1))
            before = stmt[:im.start(1)]
            if head.startswith("int mo_") or ("[" in before and "]" in stmt[im.start(1):][:3]) \
                    or re.search(r">>\s*$", before):
                continue
            call, arg = _call_context(stmt, im.start(1))
            cls = _acs_class(sc)
            for d in (-1, 1):
                out.append(Mutant(start + len(out), cls, f"script {sc} {call or 'expr'}"
                                  f"{'' if arg is None else f' arg{arg}'}: literal {n} -> {n + d} ({head[:44]})",
                                  "acs", sm.start() + im.start(1), str(n), str(n + d),
                                  dict(script=sc, call=call, arg=arg)))
        if ">> 16" in stmt:
            out.append(Mutant(start + len(out), "acs-readout-shift", f"script {sc}: `>> 16` -> `>> 15` ({head[:40]})",
                              "acs", sm.start() + stmt.index(">> 16") + 3, "16", "15",
                              dict(script=sc, call="shift", arg=None)))
    return out


def _acs_class(sc) -> str:
    if sc is None:
        return "acs-other"
    if 100 <= sc < 200:
        return "acs-readout"
    return {3: "acs-reset", 1: "acs-port-write", 2: "acs-port-write", 4: "acs-tid", 7: "acs-start-barrier",
            8: "acs-motion", 9: "acs-guard"}.get(sc, "acs-other")


# --- reasons for survivors ------------------------------------------------------------------------------------------
def reason(m: Mutant, ctx: dict | None = None) -> str | None:
    """A rule-based one-line reason why a surviving mutant is expected to be equivalent, or None (unexplained).
    The rules read the compiler's design (docs/HOW_IT_WORKS.md); they are expectations, not proofs."""
    c, ctx = m.cls, ctx or {}
    family = c.split(":")[0]
    if family in ctx.get("demoted", ()):             # a site of this family died on the deep input set: no rule left
        if family == "sector-floor-walk":
            return ("input gap: not equivalent. A walking floor +-1 moves the gate threshold of the lanes that use it by one, "
                    "and an input exactly on that threshold shows it (tests/test_mutation_rules.py pins one); "
                    "the input set did not land on it")
        return None
    if c.startswith("sector-light-"):
        return "light level of a sector no script reads" if not c.endswith("output") else None
    if c.startswith("sector-floor-walk"):
        return "floor of a walking sector: +-1 unit stays inside the 24-unit step-up slack"
    if c in ("sector-ceiling-output", "sector-floor-output"):
        return "height of an output display sector: only its light level is read"
    if c in ("sector-floor-barrier", "sector-ceiling-barrier"):
        return "a barrier is 13,000 units above the walking floor when closed: 1 unit changes nothing"
    if c.startswith("sector-ceiling-walk"):
        return "ceiling 16,000 units above the 56-unit dolls"
    if c in ("sector-floor-other", "sector-ceiling-other"):
        return "a room or closet no doll's path crosses"
    if c == "sector-ceiling-gate":
        return "ceiling far above the dolls (a floor gate) or the value the reset script restores (a ceiling gate)"
    if c == "sector-floor-gate":
        return "initial height: the reset script and the port writes set it before every pass but the first"
    if c in ("plane-mover-speed", "move-to-speed", "barrier-speed") or (
            m.info.get("call") in ("Floor_MoveToValue", "Ceiling_MoveToValue", "Floor_LowerToLowest",
                                   "Ceiling_LowerToFloor") and m.info.get("arg") == 1):
        return "mover speed 150,000 is about 18,750 units per tic: +-1 changes no move's finish tic"
    closed = c == "move-to-height" or (c == "acs-reset" and m.info.get("call") == "Floor_MoveToValue"
                                       and m.info.get("arg") == 2)
    if closed and re.search(r"\b13000 -> (13001|12999)", m.where):
        return "a closed gate or barrier sits at 13,000: one unit more or less changes no pass"
    if c == "scroller" and m.info.get("arg") in (3, 4):
        return "conveyor speed byte: +-1 leaves the carry direction and the dolls' arrival unchanged"
    if c == "exec-map":
        return "unused map argument of ACS_ExecuteAlways"
    if m.info.get("call", "").startswith("GetSector") and m.info.get("arg") in (1, 2):
        return "x / y point argument of GetSector*Z: a sector's plane is flat"
    if m.info.get("call") == "TeleportOther" and m.info.get("arg") == 2:
        return "fog flag of TeleportOther: only a spawn effect"
    if c == "acs-motion" and ctx.get("n_dolls", 1 << 30) + 1 < 4096:
        return "the motion script runs only on maps with >= 4,096 objects (runtime.settle); this one has fewer"
    return _redundancy_reason(m, ctx)


def _redundancy_reason(m: Mutant, ctx: dict) -> str | None:
    """Rules for single-site changes to mechanisms the compiler builds more than once, or builds for maps that learn.
    A reason that starts with "redundancy:" or "inert in forward nets:" is backed by an aggregate control that fails
    or passes as stated (tests/test_mutation_rules.py, models/verify/mutation_survivor_study.json); `run_mutation` still gives
    such a survivor the deep input set."""
    c, info = m.cls, m.info
    if c == "scroller" and ctx.get("scrollers", 0) >= 2:
        return ("redundancy: the carry belt is a stack of scrollers; changing one leaves the others carrying "
                "(removing 2 of 4 stays exact, 3 of 4 fails)")
    if info.get("special") == FLOOR_MOVE_TO and not ctx.get("learning"):
        return ("inert in forward nets: a freeze line stops a late doll of a finished layer and a forward pass has none "
                "(all freeze lines removed stays exact); a learning map needs them (test_freeze_barriers_are_necessary)")
    st = re.search(r"Floor_MoveToValue\((\d+), \d+, (\d+), (-?\d+)\);\)", m.where) if c == "acs-reset" else None
    if st and info.get("call") == "Floor_MoveToValue":
        tag, height, arg = int(st.group(1)), int(st.group(2)), info.get("arg")
        if height == 13000 and tag >= 64000 and arg in (0, 3):
            return ("redundancy: one wave barrier is reset wrongly (not closed, or set below the walkway); the other waves "
                    "still serialise the moves (all wave resets removed, or every other one, fails)")
        if height == 13000 and tag == 2 and arg in (0, 3):
            return "redundancy: script 7 lowers the layer-0 start barrier on every run; a wrong reset changes only when it opens"
        if height != 13000 and tag >= 64000:
            if arg == 3 and m.old != "0" and m.new != "0":
                return "the negate flag of Floor_MoveToValue is a boolean: any nonzero value negates the height (identical)"
            if arg == 2 and abs(int(m.new) - int(m.old)) == 1:
                return "freeze walkway height +-1: inside the 24-unit step-up slack"
            if arg == 0 and tag not in ctx.get("raised", {tag}):
                return "inert in forward nets: no line ever raises this freeze barrier (the last layer is not frozen)"
    return None


# --- running mutants ------------------------------------------------------------------------------------------------
def capture_acs(net, wad: Path, guard: bool = True) -> str:
    """Compile `net` to `wad` and return the ACS source the compiler produced."""
    from . import kit, pure
    seen = []
    orig = pure.compile_acs
    pure.compile_acs = lambda src: (seen.append(src), orig(src))[1]
    try:
        kit.compile(net, wad, mode="map", guard=guard).close()
    finally:
        pure.compile_acs = orig
    return seen[-1]


def capture_tile_acs(spec, wad: Path) -> str:
    """Compile one operation tile (tile.compile_tile) to `wad` and return the ACS source the compiler produced."""
    from . import tile
    seen = []
    orig = tile.compile_acs
    tile.compile_acs = lambda src: (seen.append(src), orig(src))[1]
    try:
        tile.compile_tile(spec, wad)
    finally:
        tile.compile_acs = orig
    return seen[-1]


def tile_case(net, layer: int, x) -> tuple[dict[str, int], dict[str, int]]:
    """(ports, expected outputs) of MLP tile `layer` for input x: the previous layer's reference values are what the
    bus writes (an input and its negation, or a hidden value and its twin)."""
    ref = net.reference(x)
    if layer == 0:
        xs = [int(v) for v in np.asarray(x).reshape(-1)]
        ports = {f"{sign}x{i}": v * (-1 if sign else 1) for i, v in enumerate(xs) for sign in ("", "-")}
    else:
        names = [f"{sign}h{layer}_{j}" for j in range(len(ref["hidden"][layer - 1])) for sign in ("", "-")]
        ports = {n: net._expected(n, ref) for n in names}
    last = layer == net.n_layers - 1
    outs = [f"y{j}" for j in range(net.n_out)] if last else [
        f"{sign}h{layer + 1}_{j}" for j in range(len(ref["hidden"][layer])) for sign in ("", "-")]
    return ports, {n: net._expected(n, ref) for n in outs}


def apply_acs(source: str, m: Mutant) -> str:
    assert source[m.line:m.line + len(m.old)] == m.old, "ACS source changed"
    return source[:m.line] + m.new + source[m.line + len(m.old):]


def _build_mutant(base: Path, source: str, m: Mutant, dst: Path) -> None:
    lumps = read_lumps(base)
    if m.kind == "textmap":
        def fn(text):
            lines = text.split("\n")
            assert lines[m.line] == m.old
            lines[m.line] = m.new
            return "\n".join(lines)
        lumps = [(n, fn(b.decode()).encode() if n == "TEXTMAP" else b) for n, b in lumps]
    else:
        from .udmf import compile_acs
        code = compile_acs(apply_acs(source, m))
        lumps = [(n, code if n == "BEHAVIOR" else b) for n, b in lumps]
    write_lumps(dst, lumps)
    shutil.copyfile(base.with_suffix(".json"), dst.with_suffix(".json"))


def _run_mutant(job: dict) -> dict:
    """Build and run one mutant; returns {id, killed, how}."""
    from .fleet import open_brain
    from .kit import RangeError
    from .fuzz import net_of
    m = Mutant(**job["mutant"])
    net = net_of("mlp", job["spec"]) if job.get("spec") else None
    t0 = time.perf_counter()
    with tempfile.TemporaryDirectory(prefix="dfm-") as d:
        wad = Path(d) / "m.wad"
        try:
            _build_mutant(Path(job["base"]), job["source"], m, wad)
        except Exception as e:                                      # e.g. acc rejects the mutated source
            return dict(id=m.id, killed=True, how=f"does not build: {type(e).__name__}: {str(e)[:80]}")
        if job.get("learn") is not None:
            return _run_learning_mutant(job, m, wad, t0)
        if job.get("tile") is not None:
            return _run_tile_mutant(job, net, m, wad, t0)
        try:
            brain = open_brain(net, wad)
        except Exception as e:
            return dict(id=m.id, killed=True, how=f"does not start: {type(e).__name__}: {str(e)[:80]}")
        try:
            brain.runner.max_tics = job["max_tics"]
            for x in job["inputs"]:
                try:
                    brain(np.asarray(x))
                except RangeError:
                    continue
                except Exception as e:
                    how = "mismatch" if type(e).__name__ == "MismatchError" else f"{type(e).__name__}"
                    return dict(id=m.id, killed=True, how=how, detail=str(e)[:120],
                                seconds=round(time.perf_counter() - t0, 2))
        finally:
            brain.close()
    return dict(id=m.id, killed=False, how="survived", seconds=round(time.perf_counter() - t0, 2))


def _run_tile_mutant(job: dict, net, m: Mutant, wad: Path, t0: float) -> dict:
    """Run one mutant of an operation tile: every case's ports in, every output compared with the reference."""
    from .kit import RangeError
    from .runtime import TileRunner
    try:
        runner = TileRunner(wad)
    except Exception as e:                                          # noqa: BLE001
        return dict(id=m.id, killed=True, how=f"does not start: {type(e).__name__}: {str(e)[:80]}")
    try:
        runner.max_tics = job["max_tics"]
        for x in job["inputs"]:
            try:
                net.check(np.asarray(x), planes=False)
            except RangeError:
                continue
            ports, want = tile_case(net, job["tile"], x)
            try:
                got = runner.run(ports)[0]
            except Exception as e:                                  # noqa: BLE001
                return dict(id=m.id, killed=True, how=type(e).__name__, detail=str(e)[:120])
            bad = {n: (got.get(n), v) for n, v in want.items() if got.get(n) != v}
            if bad:
                return dict(id=m.id, killed=True, how="mismatch", detail=str(list(bad.items())[:3])[:120],
                            seconds=round(time.perf_counter() - t0, 2))
    finally:
        runner.close()
    return dict(id=m.id, killed=False, how="survived", seconds=round(time.perf_counter() - t0, 2))


def capture_learner(wad: Path, n_feat: int = 6, K: int = 3, cap: int = 100) -> str:
    """Compile the online perceptron (doomformer.learn: weights are ceiling heights the dolls raise and lower) to `wad`
    and return its ACS source."""
    from . import learn, pure
    seen = []
    orig = pure.compile_acs
    pure.compile_acs = lambda src: (seen.append(src), orig(src))[1]
    try:
        learn.build_perceptrons(n_feat, 1, K, wad, cap=cap)
    finally:
        pure.compile_acs = orig
    return seen[-1]


def learning_examples(n_feat: int, n: int, seed: int = 0) -> list[tuple[list[int], int]]:
    """n (x, y) examples of a random teacher, x in {0, 1}^F with x[0] = 1, y = +-1 (the examples of the learning tests)."""
    rng = np.random.default_rng(seed)
    teacher = rng.integers(-3, 4, n_feat)
    out = []
    for _ in range(n):
        x = (rng.random(n_feat) < 0.5).astype(np.int64)
        x[0] = 1
        out.append(([int(v) for v in x], 1 if teacher @ x > 0 else -1))
    return out


def _run_learning_mutant(job: dict, m: Mutant, wad: Path, t0: float) -> dict:
    """Train the mutated perceptron map on the examples: every spike and every weight must match PerceptronRef after
    every example (weights persist between examples, so a late doll that is not frozen corrupts the next ones)."""
    from .learn import PerceptronRef, ports_for
    from .runtime import TileRunner
    F, K, cap = job["learn"]
    try:
        runner = TileRunner(wad, recycle=True)
    except Exception as e:                                          # noqa: BLE001
        return dict(id=m.id, killed=True, how=f"does not start: {type(e).__name__}: {str(e)[:80]}")
    ref = PerceptronRef(F, 1, K, cap)
    try:
        runner.max_tics = job["max_tics"]
        for x, y in job["inputs"]:
            xs, ys = np.asarray(x, dtype=np.int64), np.asarray([y], dtype=np.int64)
            s = ref.step(xs, ys)
            try:
                got = runner.run(ports_for(xs, ys))[0]
            except Exception as e:                                  # noqa: BLE001
                return dict(id=m.id, killed=True, how=type(e).__name__, detail=str(e)[:120])
            if got["s0"] != s[0] or [got[f"w0_{i}"] for i in range(F)] != ref.w[0].tolist():
                return dict(id=m.id, killed=True, how="mismatch", seconds=round(time.perf_counter() - t0, 2))
    finally:
        runner.close()
    return dict(id=m.id, killed=False, how="survived", seconds=round(time.perf_counter() - t0, 2))


def run_learning_mutation(*, per_class: int = 20, seed: int = 0, workers: int = 1, n_examples: int = 40,
                          n_feat: int = 6, K: int = 3, cap: int = 100, timeout: float = 120.0) -> dict:
    """run_mutation for the online perceptron map (doomformer.learn): the same sites and classes, the examples as
    inputs. The freeze barriers matter here (test_freeze_barriers_are_necessary), so special 37 mutants are killed."""
    from .learn import PerceptronRef, ports_for
    from .runtime import TileRunner
    rng = random.Random(seed)
    t0 = time.perf_counter()
    with tempfile.TemporaryDirectory(prefix="dflm-") as d:
        base = Path(d) / "base.wad"
        source = capture_learner(base, n_feat, K, cap)
        from .tile import TileMeta
        meta = TileMeta.load(base.with_suffix(".json"))
        text = dict(read_lumps(base))["TEXTMAP"].decode()
        sites = textmap_mutants(text, meta)
        sites += acs_mutants(source, start=len(sites))
        ctx = dict(n_dolls=meta.n_dolls, scrollers=text.count("special = 223;"), learning=True,
                   raised={int(t) for t in re.findall(r"special = 37; arg0 = (\d+);", text)})
        examples = learning_examples(n_feat, n_examples, seed)
        runner = TileRunner(base, recycle=True)                     # the clean map must learn exactly first
        ref, tics = PerceptronRef(n_feat, 1, K, cap), 0
        try:
            for x, y in examples:
                xs, ys = np.asarray(x, dtype=np.int64), np.asarray([y], dtype=np.int64)
                s = ref.step(xs, ys)
                got, t, _, _ = runner.run(ports_for(xs, ys))
                assert got["s0"] == s[0] and [got[f"w0_{i}"] for i in range(n_feat)] == ref.w[0].tolist()
                tics = max(tics, t)
        finally:
            runner.close()
        by_class: dict[str, list[Mutant]] = {}
        for m in sites:
            by_class.setdefault(m.cls, []).append(m)
        picked = []
        for cls in sorted(by_class):
            picked += rng.sample(by_class[cls], min(per_class, len(by_class[cls])))
        jobs = [dict(mutant=asdict(m), spec=None, base=str(base), source=source, inputs=examples, learn=[n_feat, K, cap],
                     max_tics=max(400, 4 * tics)) for m in picked]
        res = run_jobs(jobs, workers, timeout)
    table = {}
    for cls, pool in sorted(by_class.items()):
        got = [res[m.id] for m in pool if m.id in res]
        table[cls] = dict(sites=len(pool), sampled=len(got), killed=sum(r["killed"] for r in got),
                          survived=sum(not r["killed"] for r in got))
    surv = [dict(cls=m.cls, where=m.where, reason=reason(m, ctx) or "unexplained")
            for m in picked if not res[m.id]["killed"]]
    killed = sum(res[m.id]["killed"] for m in picked)
    return dict(net=f"perceptron F={n_feat} K={K} cap={cap}", seed=seed, per_class=per_class, inputs=len(examples),
                sites=len(sites), sampled=len(picked), killed=killed, survived=len(picked) - killed,
                kill_rate=round(killed / len(picked), 4), unexplained=sum(s["reason"] == "unexplained" for s in surv),
                classes=table, survivors=surv, seconds=round(time.perf_counter() - t0, 1))


def _worker(wid: int, tasks, results):
    _mp.exit_with_parent()
    while True:
        job = tasks.get()
        if job is None:
            return
        results.put(("start", wid, job["mutant"]["id"]))
        try:
            res = _run_mutant(job)
        except BaseException as e:                                  # noqa: BLE001
            res = dict(id=job["mutant"]["id"], killed=True, how=f"harness error {type(e).__name__}: {e}"[:100])
        results.put(("done", wid, res))


def run_jobs(jobs: list[dict], workers: int, timeout: float) -> dict[int, dict]:
    """Run mutant jobs on worker processes; a worker that dies or runs past `timeout` s on one mutant kills it
    (that mutant counts as killed: crash / timeout) and is replaced."""
    ctx = _mp.context()
    tasks, results = ctx.Queue(), ctx.Queue()
    for j in jobs:
        tasks.put(j)
    out: dict[int, dict] = {}
    procs: dict[int, object] = {}
    current: dict[int, tuple[int, float]] = {}
    nxt = 0

    def spawn():
        nonlocal nxt
        p = ctx.Process(target=_worker, args=(nxt, tasks, results), daemon=True)
        with _mp.detached_main():
            p.start()
        procs[nxt] = p
        nxt += 1

    for _ in range(workers):
        spawn()
    try:
        while len(out) < len(jobs):
            try:
                msg = results.get(timeout=1.0)
            except queue.Empty:
                msg = None
            if msg:
                if msg[0] == "start":
                    current[msg[1]] = (msg[2], time.time())
                else:
                    out[msg[2]["id"]] = msg[2]
                    current.pop(msg[1], None)
            for wid, p in list(procs.items()):
                over = wid in current and time.time() - current[wid][1] > timeout
                if (not p.is_alive() or over) and wid in current:
                    mid = current.pop(wid)[0]
                    out.setdefault(mid, dict(id=mid, killed=True, how="timeout" if over else "engine crash"))
                    if p.is_alive():
                        p.terminate()
                    del procs[wid]
                    spawn()
                elif not p.is_alive():
                    del procs[wid]
                    spawn()
    finally:
        for p in procs.values():
            if p.is_alive():
                p.terminate()
    return out


def _deep_inputs(rng, net, n: int, planes: bool = True) -> list[list[int]]:
    """n random inputs (scales 3, 12, 100, 12,000 per input) that the net's range check accepts."""
    from .kit import RangeError
    out = []
    while len(out) < n:
        s = int(rng.choice([3, 12, 100, 12000]))
        x = rng.integers(-s, s + 1, net.n_in)
        try:
            net.check(x, planes=planes)
        except RangeError:
            continue
        out.append([int(v) for v in x])
    return out


def _edge_inputs(rng, net, planes: bool = True) -> list[list[int]]:
    """Threshold neighbours of the input staircase: for every input i and first-layer threshold t, x_i = +-(t-1, t, t+1)
    with the other inputs at 0 and at random values within +-3 t; the inputs the range check accepts."""
    from .kit import RangeError
    out, seen = [], set()
    for i in range(net.n_in):
        for t in map(int, net.thresholds[0]):
            for sign in ((1, -1) if net.signed_input else (1,)):
                for d in (-1, 0, 1):
                    for rand in (False, True):
                        x = rng.integers(-3 * t, 3 * t + 1, net.n_in) if rand else np.zeros(net.n_in, dtype=np.int64)
                        x[i] = sign * max(0, t + d)
                        key = tuple(int(v) for v in x)
                        if key in seen:
                            continue
                        seen.add(key)
                        try:
                            net.check(x, planes=planes)
                        except RangeError:
                            continue
                        out.append(list(key))
    return out


def run_mutation(net, *, per_class: int = 20, seed: int = 0, workers: int = 1, n_inputs: int = 24,
                 timeout: float = 120.0, progress=None, tile: int | None = None, n_deep: int = 150,
                 deep_per_class: int = 6) -> dict:
    """Mutate the compiled map of `net` (an MLP) and report the kill rate by class. Returns a JSON-ready dict:
    classes: {class: {sites, sampled, killed, survived}}, survivors: [{class, where, reason}].
    tile=k mutates operation tile k of the tile-mode lowering instead (one tile map: no waves, lights accumulate; its
    ports are written with the reference's previous-layer values and every output is compared)."""
    from .fuzz import _mlp_inputs, rng_for, spec_of
    from .kit import MLP
    if not isinstance(net, MLP):
        raise TypeError("run_mutation takes a df.MLP")
    rng = random.Random(seed)
    nrng = rng_for(seed, 0, "mlp")
    t0 = time.perf_counter()
    with tempfile.TemporaryDirectory(prefix="dfmut-") as d:
        base = Path(d) / "base.wad"
        source = capture_acs(net, base) if tile is None else capture_tile_acs(net._stages()[tile], base)
        from .tile import TileMeta
        meta = TileMeta.load(base.with_suffix(".json"))
        text = dict(read_lumps(base))["TEXTMAP"].decode()
        sites = textmap_mutants(text, meta)
        sites += acs_mutants(source, start=len(sites))
        ctx = dict(n_dolls=meta.n_dolls, scrollers=text.count("special = 223;"),
                   raised={int(t) for t in re.findall(r"special = 37; arg0 = (\d+);", text)})
        inputs, _ = _mlp_inputs(nrng, net, [], n_inputs, planes=tile is None)
        tics = 0
        if tile is None:
            from .fleet import open_brain
            brain = open_brain(net, base)                          # the clean map must pass first
            try:
                for x in inputs:
                    brain(np.asarray(x))
                    tics = max(tics, brain.info["tics"])
            finally:
                brain.close()
        else:
            from .runtime import TileRunner
            runner = TileRunner(base)
            try:
                for x in inputs:
                    ports, want = tile_case(net, tile, x)
                    got, t, _, _ = runner.run(ports)
                    if any(got.get(n) != v for n, v in want.items()):
                        raise RuntimeError(f"the clean tile {tile} differs from the reference at x={list(x)}")
                    tics = max(tics, t)
            finally:
                runner.close()
        by_class: dict[str, list[Mutant]] = {}
        for m in sites:
            by_class.setdefault(m.cls, []).append(m)
        picked = []
        for cls in sorted(by_class):
            pool = by_class[cls]
            picked += rng.sample(pool, min(per_class, len(pool)))
        spec = spec_of(net)
        jobs = [dict(mutant=asdict(m), spec=spec, base=str(base), source=source, inputs=inputs, tile=tile,
                     max_tics=max(400, 4 * tics)) for m in picked]
        res = run_jobs(jobs, workers, timeout)
        survivors = [m for m in picked if not res[m.id]["killed"]]
        n_deep_inputs = 0
        # a survivor gets a second, larger input set before it is called one
        if survivors:
            more, _ = _mlp_inputs(nrng, net, [np.asarray(x) + 1 for x in inputs], 3 * n_inputs, tile is None)
            extra = [x for x in more if x not in inputs]
            jobs2 = [dict(j, inputs=extra) for j in jobs if j["mutant"]["id"] in {m.id for m in survivors}]
            res2 = run_jobs(jobs2, workers, timeout)
            for mid, r in res2.items():
                if r["killed"]:
                    res[mid] = dict(r, how=r["how"] + " (on the second input set)")
        # EVERY class with a survivor gets the deep set, whatever its rule says: `deep_per_class` survivors of each class
        # (all of them if fewer) run on `n_deep` random inputs at four scales plus the threshold neighbours of every
        # input (t-1, t, t+1 with the other inputs at 0 and at random). A class where any of them dies loses its rule.
        left = [m for m in survivors if not res[m.id]["killed"]]
        deep_stats: dict[str, dict] = {}
        if left:
            deep = _deep_inputs(nrng, net, n_deep, tile is None) + _edge_inputs(nrng, net, tile is None)
            by_surv: dict[str, list[Mutant]] = {}
            for m in left:
                by_surv.setdefault(m.cls, []).append(m)
            sel = [m for cls in sorted(by_surv) for m in rng.sample(by_surv[cls], min(deep_per_class, len(by_surv[cls])))]
            jobs3 = [dict(j, inputs=deep) for j in jobs if j["mutant"]["id"] in {m.id for m in sel}]
            for mid, r in run_jobs(jobs3, workers, timeout).items():
                if r["killed"]:
                    res[mid] = dict(r, how=r["how"] + " (on the deep input set)")
            for m in sel:
                st = deep_stats.setdefault(m.cls, dict(survivors=len(by_surv[m.cls]), tested=0, killed=0))
                st["tested"] += 1
                st["killed"] += int(res[m.id]["killed"])
            ctx["demoted"] = {cls.split(":")[0] for cls, st in deep_stats.items() if st["killed"]}
            n_deep_inputs = len(deep)
    table = {}
    for cls, pool in sorted(by_class.items()):
        got = [res[m.id] for m in pool if m.id in res]
        table[cls] = dict(sites=len(pool), sampled=len(got), killed=sum(r["killed"] for r in got),
                          survived=sum(not r["killed"] for r in got), **{f"deep_{k}": v for k, v in deep_stats.get(cls, {}).items()})
    surv = []
    for m in picked:
        if not res[m.id]["killed"]:
            surv.append(dict(cls=m.cls, where=m.where, reason=reason(m, ctx) or "unexplained"))
    how = {}
    for m in picked:
        if res[m.id]["killed"]:
            k = re.sub(r"\(on the .*", "", res[m.id]["how"]).split(":")[0].strip()
            how[k] = how.get(k, 0) + 1
    tot = len(picked)
    killed = sum(res[m.id]["killed"] for m in picked)
    return dict(net=spec, tile=tile, seed=seed, per_class=per_class, inputs=len(inputs), deep_inputs=n_deep_inputs,
                deep_per_class=deep_per_class, sites=len(sites), sampled=tot,
                killed=killed, survived=tot - killed, kill_rate=round(killed / tot, 4),
                unexplained=sum(s["reason"] == "unexplained" for s in surv),
                input_gap=sum(s["reason"].startswith("input gap") for s in surv), how_killed=how, classes=table,
                survivors=surv, seconds=round(time.perf_counter() - t0, 1))


# --- aggregate controls for the survivors ----------------------------------------------------------------------------
def control_variants(source: str, text: str) -> dict[str, tuple[str, str]]:
    """name -> (TEXTMAP, ACS source) of maps that change a whole mechanism at once: k of the stacked scrollers removed,
    every freeze line removed, every wave-barrier reset removed / half of them removed / set below the walkway, and the
    first one only. The single-site survivors of those mechanisms are explained by what these maps show."""
    lines = text.split("\n")
    out = {"clean": (text, source)}
    scroll = [i for i, ln in enumerate(lines) if "special = 223;" in ln]
    for k in range(1, len(scroll) + 1):
        t = list(lines)
        for i in scroll[:k]:
            t[i] = t[i].replace("special = 223;", "special = 0;")
        out[f"scrollers removed: {k} of {len(scroll)}"] = ("\n".join(t), source)
    t = [ln.replace("special = 37;", "special = 0;") for ln in lines]
    out[f"freeze lines removed: all {sum('special = 37;' in ln for ln in lines)}"] = ("\n".join(t), source)
    pat = re.compile(r"Floor_MoveToValue\((\d+), 150000, 13000, 0\);")
    wave = [int(m.group(1)) for m in pat.finditer(source) if int(m.group(1)) >= 64000]

    def sub(keep):
        return pat.sub(lambda m: m.group(0) if keep(int(m.group(1))) else "", source)

    def pit(which):
        return pat.sub(lambda m: f"Floor_MoveToValue({m.group(1)}, 150000, 13000, 1);" if which(int(m.group(1)))
                       else m.group(0), source)

    if wave:
        half = set(wave[::2])
        out["wave barrier resets removed: the first"] = (text, sub(lambda t: t != wave[0]))
        out["wave barrier resets removed: every other"] = (text, sub(lambda t: t not in half))
        out["wave barrier resets removed: all"] = (text, sub(lambda t: t < 64000))
        out["wave barrier reset below the walkway: the first"] = (text, pit(lambda t: t == wave[0]))
        out["wave barrier resets below the walkway: all"] = (text, pit(lambda t: t >= 64000))
    return out


def survivor_controls(net, inputs: list[list[int]] | None = None, n_inputs: int = 24, workers: int = 1,
                      only: set[str] | None = None) -> list[dict]:
    """Run `control_variants` of `net`'s map against the reference: [{name, exact, inputs, detail}]. `only` selects
    names (a prefix match). The clean map must be exact."""
    from .fleet import open_brain
    from .fuzz import _mlp_inputs, rng_for
    from .kit import RangeError
    nrng = rng_for(0, 0, "mlp")
    if inputs is None:
        inputs, _ = _mlp_inputs(nrng, net, [], n_inputs)
    rows = []
    with tempfile.TemporaryDirectory(prefix="dfctl-") as d:
        base = Path(d) / "base.wad"
        source = capture_acs(net, base)
        text = dict(read_lumps(base))["TEXTMAP"].decode()
        tics = 0
        brain = open_brain(net, base)
        try:
            for x in inputs:
                brain(np.asarray(x))
                tics = max(tics, brain.info["tics"])
        finally:
            brain.close()
        from .udmf import compile_acs
        for name, (t, src) in control_variants(source, text).items():
            if only and not any(name.startswith(o) for o in only):
                continue
            wad = Path(d) / "c.wad"
            code = compile_acs(src)
            write_lumps(wad, [(n, t.encode() if n == "TEXTMAP" else code if n == "BEHAVIOR" else b)
                              for n, b in read_lumps(base)])
            shutil.copyfile(base.with_suffix(".json"), wad.with_suffix(".json"))
            row = dict(name=name, exact=True, inputs=len(inputs), detail="")
            try:
                brain = open_brain(net, wad)
            except Exception as e:                                          # noqa: BLE001
                rows.append(dict(row, exact=False, detail=f"does not start: {type(e).__name__}"))
                continue
            try:
                brain.runner.max_tics = max(400, 4 * tics)
                for k, x in enumerate(inputs):
                    try:
                        brain(np.asarray(x))
                    except RangeError:
                        continue
                    except Exception as e:                                  # noqa: BLE001
                        row.update(exact=False, detail=f"input {k}: {type(e).__name__}: {str(e)[:100]}")
                        break
            finally:
                brain.close()
            rows.append(row)
    return rows


def format_report(rep: dict) -> str:
    lines = [f"mutation: {rep['sampled']} of {rep['sites']} sites sampled (<= {rep['per_class']} per class, seed "
             f"{rep['seed']}), {rep['inputs']} inputs each; killed {rep['killed']}, survived {rep['survived']} "
             f"({rep['unexplained']} unexplained, {rep.get('input_gap', 0)} input gaps); kill rate {100 * rep['kill_rate']:.1f}%",
             f"{'class':26} {'sites':>6} {'sampled':>8} {'killed':>7} {'survived':>9}"]
    for cls, r in rep["classes"].items():
        lines.append(f"{cls:26} {r['sites']:6} {r['sampled']:8} {r['killed']:7} {r['survived']:9}")
    seen = {}
    for s in rep["survivors"]:
        seen.setdefault((s["cls"], s["reason"]), []).append(s["where"])
    if seen:
        lines.append("survivors (class: count, reason, first site):")
        for (cls, why), wh in seen.items():
            lines.append(f"  {cls}: {len(wh)}, {why}; e.g. {wh[0]}")
    return "\n".join(lines)


# --- CLI ------------------------------------------------------------------------------------------------------------
def run_cli(args) -> int:
    from .kit import MLP
    net = MLP.load(args.net) if getattr(args, "net", None) else demo_net()
    if getattr(args, "learning", False):
        rep = run_learning_mutation(per_class=args.per_class, seed=args.seed, workers=args.workers)
        rep.setdefault("per_class", args.per_class)
    else:
        rep = run_mutation(net, per_class=args.per_class, seed=args.seed, workers=args.workers,
                           tile=getattr(args, "tile", None))
    print(format_report(rep))
    if getattr(args, "controls", False):
        rep["controls"] = survivor_controls(net, workers=args.workers)
        print("aggregate controls (a whole mechanism changed at once; exact = every input matched the reference):")
        for r in rep["controls"]:
            print(f"  {r['name']:<52} {'exact' if r['exact'] else 'FAILS'} ({r['inputs']} inputs) {r['detail']}")
    if args.json:
        Path(args.json).write_text(json.dumps(rep, indent=1) + "\n")
    weight_bearing = ("light-weight", "plane-weight", "effect-sign", "effect-target")
    return 1 if any(rep["classes"].get(c, {}).get("survived") for c in weight_bearing) else 0
