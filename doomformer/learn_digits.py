"""DOOM learns to read digits from a blank brain: a multiclass perceptron whose weights are ceiling heights.

One run of the map is one training example (x in {0,1}^n features, the true class t). Four layers of voodoo dolls
in ONE map, no Python between them:

  forward   weight (c, i) is the ceiling height w_ci of one sector. For every active feature i a doll walks a corridor of
            2K ceiling gates (thresholds 1-K .. K) under w_ci; each gate it fits under adds 1 to the score of class c
            and, for every other class b, +-1 to the pair difference d_cb. So  s_c = sum_i x_i (K + clip(w_ci, -K, K)).
  compare   one doll per class pair a < b passes iff s_a >= s_b (d_ab = s_a - s_b + 1 >= 1) and adds +1 to wins_a and
            -1 to wins_b;  W_c = #{j > c: s_c >= s_j} - #{j < c: s_j >= s_c} + c  (the comparator of df.with_argmax).
  winner    one doll per class passes iff W_c >= C - 1: that is the first maximal score (np.argmax). It lights z_c and
            lowers q_c by one; a second doll per class, behind the label's gate, raises q_c. q_c = 1 + y_c - z_c.
  learn     weight (c, i), feature i active: the doll lowers w_ci by 1, then crosses a gate under q_c >= 1 and raises
            w_ci by 1, then a gate under q_c >= 2 and raises it by 1:  w_ci += q_c - 1 = y_c - z_c. That is the perceptron
            rule: +x to the true class and -x to the predicted class when they differ, nothing otherwise.

The weights are `state`: they start at 0, survive the per-example rewind and are never written by the host. Python
writes the input ports x_i and y_c (the example and its label as a one-hot) and reads the scores, the pair differences,
the win counts, the winner lights, q and every weight; each is compared with `DigitRef`, a NumPy integer twin, at
tolerance 0. The update rule, the loss and the labels are Python's design; the dolls execute them.
"""
from __future__ import annotations

import hashlib
import os
import re
import time
import uuid
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from . import shipped
from .acc import cache_dir
from .kit import Brain, MismatchError, ReadoutError, _ints
from .ops import adc_lanes
from .pure import compile_pure
from .readout import GUARD_COMMAND
from .tile import Block, Effect, Gate, Stage, TileMeta, TileSpec

CLIP = 3                  # a weight enters its score clamped to [-CLIP, CLIP]
TAMPER_LINES = ("lower", "keep", "raise")
STEP_MAX = 9999           # |w| <= steps seen, and every plane value must stay within the exactness range (12,000)


def _names(C: int, n: int) -> dict:
    return dict(
        pairs=[(a, b) for a in range(C) for b in range(a + 1, C)],
        score=[f"s{c}" for c in range(C)],
        diff={(a, b): f"-d{a}_{b}" for a in range(C) for b in range(a + 1, C)},     # the port the comparator reads: -d_ab
        wins={c: f"-wins{c}" for c in range(C)},                                       # -W_c
        z=[f"z{c}" for c in range(C)], q=[f"q{c}" for c in range(C)],
        w=[[f"w{c}_{i}" for i in range(n)] for c in range(C)])


def digit_specs(n: int, C: int, clip: int = CLIP, tamper: tuple[int, int, str] | None = None):
    """(layers, state, readout) of the learner. tamper=(c, i, line): one of the three update lines of weight (c, i) moves
    its ceiling by 2 instead of 1: "lower" (crossed whenever feature i is on), "keep" (crossed when q_c >= 1) or "raise"
    (crossed when q_c >= 2, the true class that was not predicted). The negative control of the tests and of
    `doomformer learn --tamper`; the twin is not tampered, so DOOM must diverge at the first step that crosses the line."""
    nm = _names(C, n)
    pairs = nm["pairs"]
    fwd = TileSpec("forward")
    score = [fwd.output(nm["score"][c], 0) for c in range(C)]
    diff = {p: fwd.output(nm["diff"][p], -1) for p in pairs}                         # -d_ab starts at -1
    for c in range(C):
        for i in range(n):
            port = fwd.port(nm["w"][c][i], "ceil")
            per_gate = [Effect(score[c], 1)]
            per_gate += [Effect(diff[(c, b)], -1) for b in range(c + 1, C)]            # d_cb grows with s_c
            per_gate += [Effect(diff[(a, c)], +1) for a in range(c)]                   # d_ac falls with s_c
            stages = [Stage(Gate(port, t, "ceil"), list(per_gate)) for t in range(1 - clip, clip + 1)]
            fwd.blocks.append(Block(adc_lanes(fwd, f"x{i}", [1], 1), stages))
    cmp_ = TileSpec("compare")
    wins = {c: cmp_.output(nm["wins"][c], -c) for c in range(C)}                      # -W_c starts at -c
    for a, b in pairs:
        cmp_.blocks.append(Block(adc_lanes(cmp_, f"d{a}_{b}", [1], 1),
                                 [Stage(None, [Effect(wins[a], -1), Effect(wins[b], +1)])]))
    win = TileSpec("winner")
    z = [win.output(nm["z"][c], 0) for c in range(C)]
    q = [win.output(nm["q"][c], 1) for c in range(C)]
    for c in range(C):
        win.blocks.append(Block(adc_lanes(win, f"wins{c}", [C - 1], 1),
                                [Stage(None, [Effect(z[c], 1), Effect(q[c], -1)])]))   # the predicted class
        win.blocks.append(Block(adc_lanes(win, f"y{c}", [1], 1), [Stage(None, [Effect(q[c], 1)])]))   # the label
    lrn = TileSpec("learn")
    for c in range(C):
        for i in range(n):
            w = lrn.output(nm["w"][c][i], 0)
            line = lambda name: 2 if tamper == (c, i, name) else 1          # a tampered line moves its ceiling by 2
            lrn.blocks.append(Block(adc_lanes(lrn, f"x{i}", [1], 1), [
                Stage(None, [Effect(w, -line("lower"))]),
                Stage(Gate(lrn.port(nm["q"][c], "ceil"), 1, "ceil"), [Effect(w, line("keep"))]),
                Stage(Gate(lrn.port(nm["q"][c], "ceil"), 2, "ceil"), [Effect(w, line("raise"))])]))
    state = {nm["w"][c][i]: 0 for c in range(C) for i in range(n)}
    # planes the next layer's dolls read are read back as planes (at opt 3 that is the copied height, not the accumulator)
    readout = list(nm["diff"].values()) + list(nm["wins"].values()) + nm["q"] + [nm["w"][c][i] for c in range(C)
                                                                                for i in range(n)]
    return [fwd, cmp_, win, lrn], state, readout


# --- the integer twin -------------------------------------------------------------------------------------------------
@dataclass
class DigitRef:
    """NumPy integer twin of the DOOM learner: the multiclass perceptron with weights clamped to [-clip, clip] inside the
    score. `step(x, t)` returns every value the map exposes (name -> int) and advances the weights."""
    n: int
    C: int
    clip: int = CLIP
    w: np.ndarray = field(init=False)

    def __post_init__(self):
        self.w = np.zeros((self.C, self.n), dtype=np.int64)

    def scores(self, x: np.ndarray) -> np.ndarray:
        return (np.clip(self.w, -self.clip, self.clip) + self.clip) @ x

    def comparator(self, s: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """(d_ab for every pair, W_c, z_c) of the dolls' comparator on the scores s."""
        C = len(s)
        ge = s[:, None] >= s[None, :]                                  # [s_a >= s_b]
        up = np.triu(np.ones((C, C), dtype=bool), 1)                   # pairs a < b
        wins = (ge & up).sum(1) - (ge.T & up.T).sum(1) + np.arange(C)  # #{j>c: s_c>=s_j} - #{j<c: s_j>=s_c} + c
        z = (wins >= C - 1).astype(np.int64)
        d = s[:, None] - s[None, :] + 1
        return d, wins.astype(np.int64), z

    def step(self, x: np.ndarray, t: int) -> dict[str, int]:
        s = self.scores(x)
        d, wins, z = self.comparator(s)
        if z.sum() != 1 or int(np.argmax(z)) != int(np.argmax(s)):
            raise AssertionError("comparator reference is not the one-hot argmax")
        y = (np.arange(self.C) == t).astype(np.int64)
        out = {f"s{c}": int(s[c]) for c in range(self.C)}
        out |= {f"-d{a}_{b}": -int(d[a, b]) for a in range(self.C) for b in range(a + 1, self.C)}
        out |= {f"-wins{c}": -int(wins[c]) for c in range(self.C)}
        out |= {f"z{c}": int(z[c]) for c in range(self.C)} | {f"q{c}": int(1 + y[c] - z[c]) for c in range(self.C)}
        self.w = self.w + (y - z)[:, None] * x[None, :]
        out |= {f"w{c}_{i}": int(self.w[c, i]) for c in range(self.C) for i in range(self.n)}
        return out


def first_crossing(examples, n: int, C: int, c: int, i: int, line: str, clip: int = CLIP) -> int | None:
    """Index of the first example at which the twin says a doll crosses update line `line` of weight (c, i) (None if
    none does): "lower" when feature i is on, "keep" when it is on and q_c >= 1, "raise" when it is on and q_c >= 2."""
    ref = DigitRef(n, C, clip)
    need = {"lower": 0, "keep": 1, "raise": 2}[line]
    for k, (x, t) in enumerate(examples):
        got = ref.step(np.asarray(x), int(t))
        if x[i] == 1 and got[f"q{c}"] >= need:
            return k
    return None


# --- auditing what Python sends --------------------------------------------------------------------------------------
class StepAudit:
    """The live game as the training loop sees it: every console command is classified and counted, and a command
    outside the six classes below raises before it reaches the engine. In particular a write to a weight sector (any
    tag but the input ports') is rejected, so the weights change only by what the dolls do.

      input write   `puke -1|-2 <tag> <value> <neg>` with the tag of an input port (x_i, y_c): the example and label
      rewind        `puke -3`: ACS script 3 sends the dolls home and restores the non-state values (never a weight)
      start         `puke -7`: drops the start barriers (the host starts the clock)
      motion check  `puke -8`: counts the dolls that moved since the previous check (settle detection)
      readout guard `puke -9 100003` (guarded maps): fills the USER variables with a poison value before each page
      readout       `puke -100..`: copies light levels and plane heights into USER variables
    """
    _WRITE = re.compile(r"puke -([12]) (\d+) (\d+) ([01])")

    def __init__(self, game, input_tags: set[int], n_pages: int, guard: bool = True):
        self._game = game
        self.input_tags = frozenset(input_tags)
        self.readouts = frozenset(f"puke -{100 + i}" for i in range(n_pages))
        self.guard = GUARD_COMMAND if guard else None
        self.counts, self.rejected = Counter(), Counter()

    def _kind(self, command: str) -> str:
        m = self._WRITE.fullmatch(command)
        if m and int(m.group(2)) in self.input_tags:
            return "input write"
        if command == self.guard:
            return "readout guard"
        return {"puke -3": "rewind", "puke -7": "start", "puke -8": "motion check"}.get(
            command, "readout" if command in self.readouts else "other")

    def send_game_command(self, command):
        kind = self._kind(command)
        if kind == "other":
            self.rejected[command] += 1
            raise RuntimeError(f"command rejected (not an input write, rewind, start, motion check, readout guard or "
                               f"readout): {command!r}")
        self.counts[kind] += 1
        return self._game.send_game_command(command)

    def __getattr__(self, name):
        from .readout import READ_ONLY
        if name in READ_ONLY:
            return getattr(self._game, name)
        self.rejected[name] += 1
        raise RuntimeError(f"host game mutation/access rejected: {name}")

    def audit(self) -> dict:
        return dict(input_writes=self.counts["input write"], rewinds=self.counts["rewind"],
                    starts=self.counts["start"], motion_checks=self.counts["motion check"],
                    guards=self.counts["readout guard"], readouts=self.counts["readout"],
                    other=sum(self.rejected.values()))


# --- the learner ----------------------------------------------------------------------------------------------------
@dataclass
class Step:
    """One training example as DOOM ran it (every field read from the game; all compared with the twin)."""
    index: int
    label: int
    predicted: int                 # DOOM's winner light
    scores: np.ndarray             # s_c before the update
    margin: int                    # best score minus the runner-up (0 = a tie, lowest index wins)
    correct: bool
    changed: int                   # weights DOOM changed in this step
    checked: int                   # values compared with the twin (all must be equal)
    tics: int
    weights: np.ndarray            # (C, n) as DOOM holds them after the update
    seconds: float = 0.0


class DigitPerceptron:
    """Specification: `classes` heads over `n_features` binary features (include a constant 1 feature for a bias).
    clip: a weight enters its score clamped to [-clip, clip]. The weights start at 0; `compile()` builds the map."""

    def __init__(self, n_features: int, classes: int = 10, clip: int = CLIP):
        for name, v in (("n_features", n_features), ("classes", classes), ("clip", clip)):
            if isinstance(v, bool) or not isinstance(v, (int, np.integer)):
                raise TypeError(f"DigitPerceptron: {name} must be an int, got {type(v).__name__}")
        if n_features < 1 or classes < 2 or clip < 1:
            raise ValueError("DigitPerceptron: need n_features >= 1, classes >= 2, clip >= 1")
        self.n_features, self.classes, self.clip = int(n_features), int(classes), int(clip)

    def reference(self) -> DigitRef:
        return DigitRef(self.n_features, self.classes, self.clip)

    def _build(self, wad: Path, opt, tamper=None, layout_only: bool = False) -> TileMeta:
        if tamper is not None and not (len(tamper) == 3 and 0 <= tamper[0] < self.classes
                                       and 0 <= tamper[1] < self.n_features and tamper[2] in TAMPER_LINES):
            raise ValueError(f"tamper must be (class, feature, line) with line in {TAMPER_LINES}")
        layers, state, readout = digit_specs(self.n_features, self.classes, self.clip, tamper)
        return compile_pure(layers, wad, "digits", state=state, readout=readout, opt=opt, layout_only=layout_only)

    def plan(self, opt=3) -> dict:
        """Count the map without building it: dolls, sectors, estimated settle tics, whether it fits."""
        meta = self._build(Path("layout_only.wad"), opt, layout_only=True)
        return dict(dolls=meta.n_dolls, sectors=meta.n_sectors, fits=bool(meta.layout["fits"]),
                    tics=meta.layout["estimated_settle_tics"], values=len(meta.outputs))

    def compile(self, wad=None, *, opt=3, tamper=None, resolution=None, configure=None) -> "DigitLearner":
        """Build the map and return its learner. opt: the schedule level of docs/scheduling.md (default 3: one wave per
        layer, an ACS copy bus between the layers; 0 to 2 need one wave per weight in the forward layer and take
        tens of thousands of tics). tamper=(c, i, "lower"|"keep"|"raise"): negative control (see digit_specs)."""
        if wad is not None:
            path = Path(wad)
            path.parent.mkdir(parents=True, exist_ok=True)
            meta = self._build(path, opt, tamper)
        else:
            d = cache_dir() / "brains"
            d.mkdir(parents=True, exist_ok=True)
            tmp = d / f".tmp-{os.getpid()}-{uuid.uuid4().hex[:8]}.wad"
            try:
                meta = self._build(tmp, opt, tamper)
                digest = hashlib.sha256(tmp.read_bytes() + tmp.with_suffix(".json").read_bytes()).hexdigest()[:16]
                path = d / f"digits-{digest}.wad"
                os.replace(tmp.with_suffix(".json"), path.with_suffix(".json"))
                os.replace(tmp, path)
            finally:
                for p in (tmp, tmp.with_suffix(".json")):
                    p.unlink(missing_ok=True)
        return DigitLearner(self, path, meta, resolution=resolution, configure=configure)

    def __repr__(self):
        return f"DigitPerceptron(n_features={self.n_features}, classes={self.classes}, clip={self.clip})"


class DigitLearner(Brain):
    """A compiled DigitPerceptron plus its running game. `step(x, label)` runs one example in DOOM and returns a Step
    after comparing every exposed value with the twin; `weights` is read from the game, not from the twin."""

    kind = "digits"

    def __init__(self, net: DigitPerceptron, wad: Path, meta: TileMeta, resolution=None, configure=None):
        super().__init__(net, wad, meta, resolution=resolution, configure=configure)
        self.twin = net.reference()
        self.info.update(steps=0, lessons=0, total_tics=0)
        self._weights = np.zeros((net.classes, net.n_features), dtype=np.int64)
        self._audit = None

    @property
    def runner(self):
        r = super().runner
        if self._audit is None:
            weight_tags = {t for k, t in self.meta.layout["tags"].items() if re.fullmatch(r"w\d+_\d+", k.split(":", 1)[1])}
            if weight_tags & set(self.meta.ports.values()):
                raise AssertionError("a weight port is also an input port")
            self._audit = StepAudit(r.game, set(self.meta.ports.values()), len(self.meta.pages),
                                    "guard_script" in self.meta.layout)
            r.game = self._audit
        return r

    def audit(self) -> dict:
        """What Python sent to the game so far (StepAudit): counts per class and `other` (rejected; 0 on a clean run)."""
        self.runner
        return self._audit.audit()

    def _xt(self, x, label) -> tuple[np.ndarray, int]:
        x = _ints(x, "x", 1)
        n, C = self.net.n_features, self.net.classes
        if x.shape != (n,) or ((x != 0) & (x != 1)).any():
            raise ValueError(f"x must be {n} features, each 0 or 1")
        if isinstance(label, bool) or not isinstance(label, (int, np.integer)) or not 0 <= label < C:
            raise ValueError(f"label must be an int in 0..{C - 1}")
        return x, int(label)

    @property
    def weights(self) -> np.ndarray:
        """The weights as DOOM holds them now, shape (classes, n_features): ceiling heights read through the game."""
        got = self.runner.read()
        return self._read_weights(got)

    def _read_weights(self, got: dict[str, int]) -> np.ndarray:
        C, n = self.net.classes, self.net.n_features
        return np.array([[got[f"w{c}_{i}"] for i in range(n)] for c in range(C)], dtype=np.int64)

    def step(self, x, label) -> Step:
        """One example in DOOM: Python writes x and the one-hot label, waits, reads everything, compares everything with
        the twin (tolerance 0) and raises df.MismatchError on any difference. The twin has advanced by then."""
        x, label = self._xt(x, label)
        if self.info["steps"] >= STEP_MAX:
            raise ValueError(f"weights are ceiling heights: at most {STEP_MAX} steps keep every |w| inside the "
                             "exactness range of the map")
        C, n, twin = self.net.classes, self.net.n_features, self.twin
        want = twin.step(x, label)
        ports = {f"-x{i}": -int(x[i]) for i in range(n)} | {f"-y{c}": -int(c == label) for c in range(C)}
        t0 = time.time()
        got = self._run(ports)
        seconds = time.time() - t0
        self.info["steps"] += 1
        self.info["checked"] = len(want)
        self.info["total_tics"] += self.info["tics"]
        index = self.info["steps"] - 1
        missing = set(want) ^ set(got)
        if missing:
            raise ReadoutError(f"digits step {index}: readout and reference differ in names: {sorted(missing)[:6]}")
        bad = [(name, got[name], v) for name, v in want.items() if got[name] != v]
        if bad:
            raise MismatchError(f"digits step {index} label {label}", bad)
        s = np.array([got[f"s{c}"] for c in range(C)], dtype=np.int64)
        z = [got[f"z{c}"] for c in range(C)]
        predicted = z.index(1)
        weights = self._read_weights(got)
        changed = int((weights != self._weights).sum())
        self._weights = weights
        self.info["lessons"] += int(changed > 0)
        top = np.sort(s)[::-1]
        return Step(index, label, predicted, s, int(top[0] - top[1]), predicted == label, changed, len(want),
                    int(self.info["tics"]), weights, seconds)

    def __repr__(self):
        return (f"<DigitLearner {self.net} {self.info['dolls']} dolls, {self.info['sectors']} sectors, "
                f"{self.info['steps']} steps, {self.wad}>")


# --- the data and the training loop --------------------------------------------------------------------------------
THRESHOLD = 8             # a feature is 1 where the 0..16 grey level of a pixel is at least this
CROP = (1, 7)             # rows and columns kept of the 8x8 image: the centre 6x6 (36 pixels, plus the bias feature)


@dataclass
class DigitsData:
    x_train: np.ndarray         # (N, n_features) 0/1 ints, the last column is the constant 1
    y_train: np.ndarray
    x_test: np.ndarray          # the held-out digits
    y_test: np.ndarray

    @property
    def n_features(self) -> int:
        return self.x_train.shape[1]


def load_digits(threshold: int = THRESHOLD) -> DigitsData:
    """The shipped 8x8 digits (models/digits, CC BY 4.0, UCI via scikit-learn) as 37 binary features with the stored
    held-out split: 1,297 training digits, 500 held-out."""
    d = np.load(shipped.data("digits/digits8x8.npz"))
    lo, hi = CROP
    f = (d["images"][:, lo:hi, lo:hi] >= threshold).reshape(len(d["labels"]), -1).astype(np.int64)
    f = np.concatenate([f, np.ones((len(f), 1), dtype=np.int64)], axis=1)
    y, held = d["labels"].astype(np.int64), d["heldout"]
    return DigitsData(f[~held], y[~held], f[held], y[held])


def stream(n_train: int, steps: int, seed: int = 0) -> np.ndarray:
    """The order of the training examples: a fresh seeded permutation of the pool per pass, `steps` indices long."""
    rng = np.random.default_rng(seed)
    out = []
    while sum(map(len, out)) < steps:
        out.append(rng.permutation(n_train))
    return np.concatenate(out)[:steps]


def accuracy(weights: np.ndarray, x: np.ndarray, y: np.ndarray, clip: int | None = CLIP) -> float:
    """Held-out accuracy of a weight matrix in NumPy: argmax of the scores (clip=None: the plain, unclamped scores)."""
    w = weights if clip is None else np.clip(weights, -clip, clip) + clip
    return float(((x @ w.T).argmax(1) == y).mean())


def plain_perceptron(data: DigitsData, order: np.ndarray, marks, clip: int | None = None, classes: int = 10) -> dict:
    """Not DOOM: the textbook multiclass integer perceptron (no clamp unless clip is given) on the same stream; returns
    {steps seen: held-out accuracy} at `marks`, and the number of mistakes (updates)."""
    w = np.zeros((classes, data.n_features), dtype=np.int64)
    out, mistakes = {}, 0
    for t, i in enumerate(order, 1):
        x, y = data.x_train[i], int(data.y_train[i])
        s = (w if clip is None else np.clip(w, -clip, clip) + clip) @ x
        p = int(np.argmax(s))
        if p != y:
            w[y] += x
            w[p] -= x
            mistakes += 1
        if t in marks:
            out[t] = accuracy(w, data.x_test, data.y_test, clip)
    return dict(curve=out, mistakes=mistakes)


def doom_forward(weights: np.ndarray, x: np.ndarray, clip: int = CLIP, opt: int = 0) -> list[int]:
    """The weights DOOM learned, run through the repository's ordinary forward path: a one-layer df.MLP (scores
    s = (clip(w) + clip) x) with df.with_argmax winner layers, compiled into one map; every value of every pass is
    checked against the integer reference (df.MismatchError otherwise). Returns DOOM's winner (one-hot light) per row."""
    from .drawpad import with_argmax
    from .kit import MLP, compile as compile_net
    W = np.clip(weights, -clip, clip) + clip
    net = with_argmax(MLP([W], [np.zeros(len(W), dtype=np.int64)], [np.array([1])], signed_input=False))
    out = []
    with compile_net(net, opt=opt) as brain:
        for row in x:
            y = brain(row)
            if sorted(y.tolist()) != [0] * (len(W) - 1) + [1]:
                raise AssertionError(f"the dolls' winner lights are not one-hot: {y.tolist()}")
            out.append(int(np.argmax(y)))
    return out


def train(learner: DigitLearner, data: DigitsData, steps: int, seed: int = 0, every: int = 100, on_step=None,
          on_checkpoint=None) -> dict:
    """Train online in DOOM on the seeded stream: one DigitLearner.step per example (every value compared with the twin,
    tolerance 0). Every `every` steps (and at step 0) the weights DOOM holds are scored on the held-out digits in
    NumPy. on_step(step, running_accuracy) and on_checkpoint(steps, accuracy) are called as they happen. Returns the
    learning curve and totals."""
    order = stream(len(data.x_train), steps, seed)
    curve = {0: accuracy(learner._weights, data.x_test, data.y_test, learner.net.clip)}
    if on_checkpoint:
        on_checkpoint(0, curve[0])
    right = lessons = 0
    t0 = time.time()
    for k, i in enumerate(order, 1):
        r = learner.step(data.x_train[i], int(data.y_train[i]))
        right += r.correct
        lessons += r.changed > 0
        if on_step:
            on_step(r, right / k)
        if k % every == 0 or k == steps:
            curve[k] = accuracy(r.weights, data.x_test, data.y_test, learner.net.clip)
            if on_checkpoint:
                on_checkpoint(k, curve[k])
    return dict(steps=steps, curve=curve, online_accuracy=right / max(1, steps), mistakes=steps - right,
                seconds=time.time() - t0, tics=learner.info["total_tics"], weights=learner._weights.copy(),
                order=order)
