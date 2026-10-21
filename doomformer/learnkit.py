"""`df.Perceptron`: an online perceptron whose weights are ceiling heights in a DOOM map and whose learning rule is
executed by voodoo dolls (the map is built by doomformer/learn.py; manual: docs/learning.md).

    p = df.Perceptron(n_features=3, heads=1)
    with p.compile() as learner:                     # builds the map; DOOM starts on the first step
        scores, action = learner.step(x, y)          # one training example = one run of the map
        learner.weights                              # read back from the game

What DOOM does in one ``step(x, y)`` (x binary features, y in {-1, 0, +1} per head):

  * forward layer: for every active feature i and head h, dolls climb a corridor of the weight's ceiling gates and add
    +1 to head h's score floor once per gate they fit under: ``s_h = sum_i x_i * clamp(w_hi, -clip, clip)``.
  * learn layer, reading the scores of THIS step (before any update): dolls choose the action ("1 + the first head with
    s_h >= 1, else 0"), light the ``right``/``wrong`` verdict outputs, and apply the perceptron rule: for every active
    feature i, a doll that passes the cap gate, the label gate (y_h != 0) and the mistake gate (y_h * s_h <= 0) crosses
    Ceiling_RaiseByValue / Ceiling_LowerByValue (weight, 1). The weight is the ceiling height; it persists in the map from
    step to step (the map is never reloaded) and stays within [-cap, cap].

What Python does: it builds the map, writes x and y into gate heights before a run, waits for the dolls to settle,
reads scores, action, verdicts and ALL weights (plus their negated twins) back, and compares each with the integer twin
``PerceptronRef`` (tolerance 0). Python adds, multiplies, compares and updates nothing that is presented as DOOM's
result; the twin is only what DOOM is checked against. Any difference raises ``MismatchError``.
"""
from __future__ import annotations

import hashlib
import os
import uuid
from pathlib import Path

import numpy as np

from .acc import cache_dir
from .kit import Brain, MismatchError, _ints, _world_hook
from .learn import CAP, PerceptronRef, perceptron_specs, ports_for
from .ops import adc_lanes, fx, output_single
from .pure import compile_pure
from .tile import Block, Gate, Stage, TileMeta

CLIP = 3                # default clamp of a weight inside the score
CAP_MAX = 1000          # largest accepted cap (conservative: the map's exactness range is tile.PORT_MAX = 12000)


def _output_names(heads: int) -> list[str]:
    """Light outputs the learn layer adds to the readout (besides the scores and weights)."""
    return (["act"] + [f"pick{k}" for k in range(heads + 1)]
            + [f"{v}{h}" for v in ("right", "wrong") for h in range(heads)])


def _build_specs(n: int, heads: int, clip: int, cap: int):
    """The perceptron layers of learn.py plus the action and verdict dolls of the learn layer."""
    fwd, lrn, state, readout = perceptron_specs(n, heads, clip, cap)
    readout = readout + [f"-w{h}_{i}" for h in range(heads) for i in range(n)]      # the twins the forward layer reads
    act = output_single(lrn, "act")
    pick = [output_single(lrn, f"pick{k}") for k in range(heads + 1)]
    right = [output_single(lrn, f"right{h}") for h in range(heads)]
    wrong = [output_single(lrn, f"wrong{h}") for h in range(heads)]
    lanes = adc_lanes(lrn, "go", [1], 1)                                          # one doll per block, "go" = 1

    def pos(h):                                                                   # s_h >= 1
        return Gate(lrn.port(f"-s{h}", "floor"), -1, "floor")

    def nonpos(h):                                                                # s_h <= 0
        return Gate(lrn.port(f"s{h}", "floor"), 0, "floor")

    def label(h, sign):                                                           # y_h >= 1 / y_h <= -1
        return Gate(lrn.port(f"-y{h}" if sign > 0 else f"y{h}", "floor"), -1, "floor")

    # Corridors may only climb: the s >= 1 gate (walk floor -25) always precedes the s <= 0 gates (walk floor -24).
    stages = [Stage(nonpos(j), []) for j in range(heads)]
    stages[-1].effects = fx(pick[0], 1)                                           # action 0: no head positive
    lrn.blocks.append(Block(lanes, stages))
    for h in range(heads):
        stages = [Stage(pos(h), [])] + [Stage(nonpos(j), []) for j in range(h)]
        stages[-1].effects = fx(act, h + 1) + fx(pick[h + 1], 1)                  # action h+1: head h is first positive
        lrn.blocks.append(Block(lanes, stages))
        lrn.blocks.append(Block(lanes, [Stage(label(h, 1), []), Stage(pos(h), fx(right[h], 1))]))
        lrn.blocks.append(Block(lanes, [Stage(label(h, -1), []), Stage(nonpos(h), fx(right[h], 1))]))
        lrn.blocks.append(Block(lanes, [Stage(label(h, 1), []), Stage(nonpos(h), fx(wrong[h], 1))]))
        lrn.blocks.append(Block(lanes, [Stage(label(h, -1), []), Stage(pos(h), fx(wrong[h], 1))]))
    return fwd, lrn, state, readout


class Perceptron:
    """Specification of a DOOM-trained perceptron: `heads` independent heads over `n_features` binary features.

    cap:  weights live in [-cap, cap]; a lesson that would leave the range is stopped by a gate in the map.
    clip: a weight enters the score clamped to [-clip, clip] (the forward corridor has `clip` gates per weight).
    The weights start at 0. `compile()` builds the map and returns the `Learner`."""

    def __init__(self, n_features: int, heads: int = 1, cap: int = CAP, clip: int = CLIP):
        for name, v in (("n_features", n_features), ("heads", heads), ("cap", cap), ("clip", clip)):
            if isinstance(v, bool) or not isinstance(v, (int, np.integer)):
                raise TypeError(f"Perceptron: {name} must be an int, got {type(v).__name__}")
        if n_features < 1 or heads < 1:
            raise ValueError("Perceptron: n_features and heads must be >= 1")
        if not 1 <= clip <= cap <= CAP_MAX:
            raise ValueError(f"Perceptron: need 1 <= clip <= cap <= {CAP_MAX}, got clip={clip}, cap={cap}")
        self.n_features, self.heads, self.cap, self.clip = int(n_features), int(heads), int(cap), int(clip)

    def reference(self) -> PerceptronRef:
        """A fresh integer twin (all weights 0): the independent reference every DOOM step is checked against."""
        return PerceptronRef(self.n_features, self.heads, self.clip, self.cap)

    def _specs(self):
        return _build_specs(self.n_features, self.heads, self.clip, self.cap)

    def plan(self) -> dict:
        """Count the map without building it: dict(dolls, sectors, fits)."""
        fwd, lrn, state, readout = self._specs()
        meta = compile_pure([fwd, lrn], Path("layout_only.wad"), "perceptron", state=state, readout=readout,
                            layout_only=True)
        return dict(dolls=meta.n_dolls, sectors=meta.n_sectors, fits=bool(meta.layout["fits"]))

    def _build(self, wad: Path, world=None, actions=None) -> TileMeta:
        fwd, lrn, state, readout = self._specs()
        return compile_pure([fwd, lrn], wad, "perceptron", state=state, readout=readout, world=world,
                            actions=actions)

    def action_names(self) -> list[str]:
        """Valid keys of `compile(actions=...)`: pick0 (no head positive), pick1.. (head k-1 is the first positive),
        right{h} / wrong{h} (verdict of head h against its label)."""
        return [n for n in _output_names(self.heads) if n != "act"]

    def compile(self, wad=None, *, world=None, actions: dict[str, list[dict]] | None = None, resolution=None,
                configure=None) -> "Learner":
        """Build the map and return its Learner (like `df.compile`).

        wad: output path (its .json metadata is written next to it); default: a content-addressed file in
        ~/.cache/doomformer/brains. world: `fn(MapBuilder) -> acs_source`, see docs/BUILD_YOUR_OWN.md. actions:
        {output name: [df.act specials]} with the names of `action_names()`; each output receives +1 from exactly one
        doll per step when its condition holds, so its specials fire once per such step (they are crossed by the doll
        that adds the +1, i.e. DOOM's decision, not by Python)."""
        if actions:
            valid = set(self.action_names())
            for name, specials in actions.items():
                if name not in valid:
                    raise KeyError(f"actions: {name!r} is not an output of this perceptron (valid: {sorted(valid)})")
                if not isinstance(specials, (list, tuple)) or not all(isinstance(s, dict) and "special" in s
                                                                      for s in specials):
                    raise TypeError(f"actions[{name!r}] must be a list of dicts with 'special' (see df.act)")
        hook = _world_hook(world) if world is not None else None
        if wad is not None:
            path = Path(wad)
            path.parent.mkdir(parents=True, exist_ok=True)
            meta = self._build(path, hook, actions)
        else:
            d = cache_dir() / "brains"
            d.mkdir(parents=True, exist_ok=True)
            tmp = d / f".tmp-{os.getpid()}-{uuid.uuid4().hex[:8]}.wad"
            try:
                meta = self._build(tmp, hook, actions)
                digest = hashlib.sha256(tmp.read_bytes() + tmp.with_suffix(".json").read_bytes()).hexdigest()[:16]
                path = d / f"perceptron-{digest}.wad"
                os.replace(tmp.with_suffix(".json"), path.with_suffix(".json"))
                os.replace(tmp, path)
            finally:
                for p in (tmp, tmp.with_suffix(".json")):
                    p.unlink(missing_ok=True)
        return Learner(self, path, meta, resolution=resolution, configure=configure)

    def __repr__(self):
        return f"Perceptron(n_features={self.n_features}, heads={self.heads}, cap={self.cap}, clip={self.clip})"


class Learner(Brain):
    """A compiled Perceptron plus its running game. `step(x, y)` -> (scores, action); `weights` is read from DOOM.

    twin: the PerceptronRef every step is compared with (it advances with each step, also on a mismatch).
    info: the Brain info plus steps (examples seen), lessons (steps that changed a weight) and checked (values
    compared in the last step)."""

    kind = "perceptron"

    def __init__(self, net: Perceptron, wad: Path, meta: TileMeta, resolution=None, configure=None):
        super().__init__(net, wad, meta, resolution=resolution, configure=configure)
        self.twin = net.reference()
        self.info.update(steps=0, lessons=0)

    # -- validation ---------------------------------------------------------------------------------------------
    def _xy(self, x, y) -> tuple[np.ndarray, np.ndarray]:
        x, y = _ints(x, "x", 1), _ints(y, "y", 1)
        n, H = self.net.n_features, self.net.heads
        if x.shape != (n,) or ((x != 0) & (x != 1)).any():
            raise ValueError(f"x must be {n} features, each 0 or 1")
        if y.shape != (H,) or (np.abs(y) > 1).any():
            raise ValueError(f"y must be {H} labels, each -1, 0 (no lesson for that head) or +1")
        return x, y

    def _ports(self, x: np.ndarray, y: np.ndarray) -> dict[str, int]:
        return ports_for(x, y) | {"go": 1, "-go": -1}

    # -- reading DOOM ---------------------------------------------------------------------------------------------
    def _read_weights(self, got: dict[str, int]) -> tuple[np.ndarray, np.ndarray]:
        n, H = self.net.n_features, self.net.heads
        w = np.array([[got[f"w{h}_{i}"] for i in range(n)] for h in range(H)], dtype=np.int64)
        wneg = np.array([[got[f"-w{h}_{i}"] for i in range(n)] for h in range(H)], dtype=np.int64)
        return w, wneg

    @property
    def weights(self) -> np.ndarray:
        """The weights as DOOM holds them right now, shape (heads, n_features): the ceiling heights of the weight
        sectors, read through the game (not the twin). Not compared with anything; `step` does that."""
        return self._read_weights(self.runner.read())[0]

    # -- one training example ---------------------------------------------------------------------------------------
    def step(self, x, y) -> tuple[np.ndarray, int]:
        """One example in DOOM: returns (scores, action), both as DOOM computed them, after checking against the twin.

        x: n_features values in {0, 1}; y: one label per head in {-1, 0, +1} (0 = no lesson for that head).
        scores are computed from the weights BEFORE this step's lesson, and DOOM's action dolls choose from them: action
        = 1 + the index of the first head with score >= 1, else 0. Then the dolls apply the perceptron update. Checked,
        with tolerance 0: scores, action (act and every pick), verdict outputs, every weight and its negated twin.
        Raises MismatchError (`.mismatches`: (name, DOOM, reference)) on any difference."""
        x, y = self._xy(x, y)
        n, H, twin = self.net.n_features, self.net.heads, self.twin
        before = twin.w.copy()
        s = twin.step(x, y)
        action = int(np.argmax(s >= 1)) + 1 if (s >= 1).any() else 0
        right = [int((y[h] == 1 and s[h] >= 1) or (y[h] == -1 and s[h] <= 0)) for h in range(H)]
        wrong = [int(bool(y[h]) and not right[h]) for h in range(H)]
        expect = {f"s{h}": int(s[h]) for h in range(H)}
        expect |= {"act": action} | {f"pick{k}": int(k == action) for k in range(H + 1)}
        expect |= {f"right{h}": right[h] for h in range(H)} | {f"wrong{h}": wrong[h] for h in range(H)}
        for h in range(H):
            for i in range(n):
                expect[f"w{h}_{i}"] = int(twin.w[h, i])
                expect[f"-w{h}_{i}"] = -int(twin.w[h, i])
        got = self._run(self._ports(x, y))
        self.info["steps"] += 1
        self.info["lessons"] += int((twin.w != before).any())
        self.info["checked"] = len(expect)
        bad = [(name, got[name], want) for name, want in expect.items() if got[name] != want]
        if bad:
            raise MismatchError(f"perceptron step {self.info['steps'] - 1} x={x.tolist()} y={y.tolist()}", bad)
        return np.array([got[f"s{h}"] for h in range(H)], dtype=np.int64), int(got["act"])

    def predict(self, x) -> tuple[np.ndarray, int]:
        """`step` with no label: DOOM computes the scores and chooses the action, no weight changes (all labels 0).
        Still a full checked run of the map."""
        return self.step(x, np.zeros(self.net.heads, dtype=np.int64))

    def __repr__(self):
        return (f"<Learner {self.net} {self.info['dolls']} dolls, {self.info['sectors']} sectors, "
                f"{self.info['steps']} steps, {self.wad}>")
