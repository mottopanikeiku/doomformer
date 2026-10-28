"""Coverage of the exactness check: which of the values the integer reference defines does a pass compare?

    cov = df.coverage(brain)             # runs one pass on `x` (default: zeros / the first token sequence the range check accepts)
    print(cov)                           # N of N values, by group
    assert cov.complete

`reference` is what the integer reference defines for that pass, taken from the reference's own arrays (not from the
lowering): every hidden pre-activation and its negated twin, every output (MLP); every spike, its twin and every
potential (spiking); every pre-activation, score, softmax and RMSNorm intermediate, logit and their twins at the
active positions (transformer). `exposed` is what the map reads back. `compared` is what the check compares: for a
one-map brain and for MLP tiles it is measured, not assumed: the readback of the real pass is replayed once per
exposed name with that value changed by 1, and the name counts as compared when the brain's own check raises
MismatchError naming it (`detected`). `brain.info["checked"]` is the check's own count and must equal `reference`.

The staircase counts (input counts, hidden activations, attention and MLP activations) are values of the reference
that no map exposes: the gates compute them and only their effect on the sums is read back. They are listed
(`counts`), not claimed.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

# staircase outputs of the transformer reference (intref): defined, computed by gates, not exposed
COUNT_KEYS = ("x0", "x", "x1", "x2", "q", "k", "v", "a", "e", "ao", "m", "g", "u", "n_att", "n_mlp")
PERTURB_CAP = 3000            # at most this many names are perturbed per pass (a seeded sample above it)


@dataclass
class Coverage:
    kind: str                                   # mlp | spiking | transformer
    mode: str                                   # map | tiles
    reference: dict[str, int]                   # group -> values the reference defines and a map can expose
    twins: dict[str, int]                       # group -> negated twins the lowering creates for them (derived values)
    exposed: dict[str, int] | None              # group -> names (values and twins) read back; None: tile transformer
    compared: int                               # the check's own count (brain.info["checked"])
    detected: int | None                        # names whose +1 perturbation the check detects (None: not measured)
    perturbed: int | None                       # names perturbed (all, or a seeded sample above PERTURB_CAP)
    missing: list[str] = field(default_factory=list)       # reference values the map does not read back
    twins_unread: list[str] = field(default_factory=list)  # twins that exist but are not read back (planes gates read)
    padding: int = 0                            # names read back at padding positions (not compared by design)
    undetected: list[str] = field(default_factory=list)    # names perturbed and not detected
    counts: dict[str, int] = field(default_factory=dict)   # staircase counts the reference defines (not exposed)
    comparisons: int | None = None              # gate comparisons the reference makes (MLP): not exposed either
    pass_input: object = None

    @property
    def reference_total(self) -> int:
        return sum(self.reference.values())

    @property
    def twins_total(self) -> int:
        return sum(self.twins.values())

    @property
    def exposed_total(self) -> int | None:
        return None if self.exposed is None else sum(self.exposed.values())

    @property
    def complete(self) -> bool:
        """Every value the reference defines is read back and compared by the check; every name read back (value or
        twin) is compared; a changed value is detected."""
        if (self.missing or self.twins_unread or self.undetected
                or self.compared != self.reference_total + self.twins_total):
            return False
        return self.detected is None or self.detected == self.perturbed

    def __str__(self) -> str:
        rows = [f"coverage ({self.kind}, {self.mode}): the check compared {self.compared} values: "
                f"{self.reference_total - len(self.missing)} of the {self.reference_total} the reference defines"
                f" and {self.twins_total - len(self.twins_unread)} of {self.twins_total} negated twins"
                + (" (complete)" if self.complete else " (INCOMPLETE)")]
        for g, n in {**self.reference, **self.twins}.items():
            shown = "" if self.exposed is None else f"  read back {self.exposed.get(g, 0):>8,}"
            rows.append(f"  {g:<28s} defined {n:>8,}{shown}")
        if self.detected is not None:
            rows.append(f"  a value changed by 1 is detected for {self.detected} of {self.perturbed} perturbed names")
        if self.counts:
            rows.append("  not exposed (computed by gates, checked through the sums they feed): "
                        + ", ".join(f"{g} {n:,}" for g, n in self.counts.items())
                        + (f"; {self.comparisons:,} gate comparisons" if self.comparisons else ""))
        if self.padding:
            rows.append(f"  read back at padding positions, not compared by design: {self.padding}")
        for what, names in (("values missing from the readout", self.missing),
                            ("twin planes not read back", self.twins_unread),
                            ("not detected when changed", self.undetected)):
            if names:
                rows.append(f"  {what}: {names[:6]}" + (f" (+{len(names) - 6} more)" if len(names) > 6 else ""))
        return "\n".join(rows)

    def as_dict(self) -> dict:
        return dict(kind=self.kind, mode=self.mode, reference=self.reference, reference_total=self.reference_total,
                    twins=self.twins, twins_total=self.twins_total, exposed=self.exposed,
                    exposed_total=self.exposed_total, compared=self.compared, detected=self.detected,
                    perturbed=self.perturbed, complete=self.complete, counts=self.counts,
                    comparisons=self.comparisons, missing=self.missing, twins_unread=self.twins_unread,
                    padding=self.padding, undetected=self.undetected)


# --- what the reference defines --------------------------------------------------------------------------------------
def _mlp_reference(net, x):
    ref = net.reference(x)
    sums: dict[str, list[str]] = {"hidden pre-activations h": [], "outputs y": []}
    twins: dict[str, list[str]] = {"twins -h": []}
    for l, h in enumerate(ref["hidden"], 1):
        for j in range(len(h)):
            sums["hidden pre-activations h"].append(f"h{l}_{j}")
            twins["twins -h"].append(f"-h{l}_{j}")
    sums["outputs y"] = [f"y{o}" for o in range(len(ref["y"]))]
    counts = {"staircase counts": int(sum(a.size for a in ref["activations"]))}
    gates = sum(len(t) * a.size * (2 if l == 0 and net.signed_input else 1)
                for l, (t, a) in enumerate(zip(net.thresholds, ref["activations"])))
    return sums, twins, counts, int(gates)


def _spiking_reference(net):
    """The spikes and the potentials (a potential is read as the plane -v). The twin planes -s are read by the synapse
    gates only and never read back: listed as not exposed, not as twins."""
    n = net.n
    return ({"spikes s": [f"s{i}" for i in range(n)], "potentials v (read as -v)": [f"-v{i}" for i in range(n)]}, {},
            {"twin planes -s (read by gates only)": n})


def _transformer_reference(net, tokens):
    from .intref import forward
    trace: dict = {}
    forward(net, tokens, trace)
    n = len(tokens)
    sums: dict[str, list[str]] = {}
    twins: dict[str, list[str]] = {}
    counts: dict[str, int] = {}
    for key, arr in trace.items():
        arr = np.asarray(arr)
        base = key.split(".")[-1]
        if base in COUNT_KEYS:
            counts[base] = counts.get(base, 0) + int(_causal(arr) if base in ("a", "e") else arr.size)
            continue
        if base == "score":
            idx = [(h, t, s) for h in range(arr.shape[0]) for t in range(n) for s in range(t + 1)]
        else:
            idx = list(np.ndindex(*arr.shape))
        names = [key + ":" + ":".join(map(str, i)) for i in idx]
        sums.setdefault(base, []).extend(names)
        if base != "logits":                                    # the unembedding has no twin: nothing reads it
            twins.setdefault(base, []).extend("-" + nm for nm in names)
    return sums, {f"-{b}": v for b, v in twins.items()}, dict(sorted(counts.items()))


def _causal(arr) -> int:
    n = arr.shape[-1]
    return int(arr.shape[0] * n * (n + 1) // 2)


# --- measuring what the check compares -------------------------------------------------------------------------------
def _sample(names: list[str], cap: int, seed: int = 0) -> list[str]:
    if len(names) <= cap:
        return names
    rng = np.random.default_rng(seed)
    return [names[i] for i in sorted(rng.choice(len(names), cap, replace=False).tolist())]


def _detect(call, readback: dict[str, int], names: list[str]) -> list[str]:
    """For each name: `call(perturbed readback)` must raise MismatchError naming it. Returns the undetected names."""
    from .kit import MismatchError
    missed = []
    for name in names:
        bad = dict(readback)
        bad[name] = int(bad[name]) + 1
        try:
            call(bad)
        except MismatchError as e:
            if any(m[0] == name for m in e.mismatches):
                continue
        except Exception:                                       # another kind of failure is not a detection
            pass
        missed.append(name)
    return missed


class _ReplayPool:
    """A stand-in for orchestrator.TilePool that answers `run` from recorded outputs (for the perturbation replay)."""

    def __init__(self, nxt, real):
        self._next, self.log = nxt, real.log

    def run(self, jobs):
        return [self._next()]


def _record_tiles(brain) -> None:
    """Make an MLP tile brain keep the tile outputs of its last pass (`_tile_outputs`) for the perturbation replay."""
    if getattr(brain, "_recording", False):
        return
    pool = brain.pool
    real = pool.run
    brain._tile_outputs = []

    def run(jobs):
        out = real(jobs)
        brain._tile_outputs.extend(out)
        return out
    pool.run = run
    inner = brain.run

    def run_pass(x, check=False):
        brain._tile_outputs.clear()
        return inner(x, check)
    brain.run = run_pass
    brain._recording = True


def coverage(obj, x=None, *, mode: str = "map", perturb: bool = True, cap: int = PERTURB_CAP, opt: int = 0) -> Coverage:
    """Run one pass of a compiled brain (or of `obj` compiled into a temporary map / tiles) and report what the check
    compares. x: an MLP input vector, a spiking input vector (0/1), or a token list (default: zeros, no spikes, or the
    first full-length token sequence the range check accepts). obj: a Brain, an MLP, a SpikingNet or a Transformer;
    mode and opt are used only when obj is a net. perturb=False skips the per-name replay (only `compared` is then
    measured)."""
    from .kit import Brain, MLPBrain, SpikingBrain, TransformerBrain, compile
    own = None
    brain = obj
    if not isinstance(obj, Brain):
        import tempfile
        from pathlib import Path
        tmp = tempfile.TemporaryDirectory(prefix="dfcov-")
        wad = Path(tmp.name) / ("net.tiles" if mode == "tiles" else "net.wad")
        brain = compile(obj, wad, mode=mode, limits=False, **(dict(opt=opt) if mode == "map" else {}))
        own = (brain, tmp)
    try:
        if isinstance(brain, MLPBrain):
            return _mlp(brain, x, perturb, cap)
        if isinstance(brain, SpikingBrain):
            return _spiking(brain, x, perturb, cap)
        if isinstance(brain, TransformerBrain):
            return _transformer(brain, x, perturb, cap)
        raise TypeError(f"coverage takes a Brain, an MLP, a SpikingNet or a Transformer, not {type(obj).__name__}")
    finally:
        if own is not None:
            own[0].close()
            own[1].cleanup()


def _mlp(brain, x, perturb, cap) -> Coverage:
    net = brain.net
    x = np.zeros(net.n_in, dtype=np.int64) if x is None else net._x(x)
    tiles = brain.info.get("mode") == "tiles"
    if tiles:
        _record_tiles(brain)
    sums, twins, counts, gates = _mlp_reference(net, x)
    got = brain.run(x, check=True)                              # the real pass (MismatchError if DOOM differs)
    compared = int(brain.info["checked"])
    names = _sample(sorted(_defined(sums, twins) & set(got)), cap) if perturb else []
    missed: list[str] = []
    if perturb:
        if tiles:
            tile_outs = [dict(o) for o in brain._tile_outputs]

            def call(bad):
                outs = iter([{k: bad.get(k, v) for k, v in o.items()} for o in tile_outs])
                pool = brain._pool
                brain._pool = _ReplayPool(lambda: next(outs), pool)
                try:
                    brain.run(x, check=True)
                finally:
                    brain._pool = pool
        else:
            def call(bad):
                real = brain._run
                brain._run = lambda ports: dict(bad)
                try:
                    brain.run(x, check=True)
                finally:
                    brain._run = real
        missed = _detect(call, got, names)
    return _build("mlp", brain, sums, twins, got, compared, names if perturb else None, missed, counts, gates, x.tolist())


def _spiking(brain, x, perturb, cap) -> Coverage:
    net = brain.net
    x = np.zeros(len(net.inputs), dtype=np.int64) if x is None else np.asarray(x, dtype=np.int64)
    v0 = brain.v.copy()
    sums, twins, unread_by_design = _spiking_reference(net)
    readback: dict[str, int] = {}
    real = brain._run

    def spy(ports):
        readback.update(real(ports))
        return readback
    brain._run = spy
    try:
        brain.step(x)                                           # the real pass
    finally:
        brain._run = real
    compared, v1, missed = int(brain.info["checked"]), brain.v.copy(), []
    names = _sample(sorted(_defined(sums, twins) & set(readback)), cap) if perturb else []
    if perturb:
        def call(bad):
            brain.v = v0.copy()
            brain._run = lambda ports: dict(bad)
            try:
                brain.step(x)
            finally:
                brain._run = real
                brain.v = v1.copy()
        missed = _detect(call, readback, names)
    return _build("spiking", brain, sums, twins, readback, compared, names if perturb else None, missed,
                  unread_by_design, None, x.tolist())


def _transformer(brain, x, perturb, cap) -> Coverage:
    from .intref import check_limits
    from .kit import limit_sequences
    net = brain.net
    if x is None:
        seqs = limit_sequences(net)
        full = [s for s in seqs if len(s) == net.T and not check_limits(net, [s])]
        x = full[0] if full else next((s for s in seqs if not check_limits(net, [s])), [0])
    tokens = [int(t) for t in x]
    sums, twins, counts = _transformer_reference(net, tokens)
    tiles = brain.info.get("mode") == "tiles"
    readback: dict[str, int] = {}
    real = None if tiles else brain._run
    if not tiles:
        def spy(ports):
            readback.update(real(ports))
            return readback
        brain._run = spy
    try:
        brain.logits(tokens)                                    # the real pass
    finally:
        if not tiles:
            brain._run = real
    compared, missed = int(brain.info["checked"]), []
    names = _sample(sorted(_defined(sums, twins) & set(readback)), cap) if perturb and not tiles else []
    if names:
        def call(bad):
            brain._run = lambda ports: dict(bad)
            try:
                brain.logits(tokens)
            finally:
                brain._run = real
        missed = _detect(call, readback, names)
    return _build("transformer", brain, sums, twins, None if tiles else readback, compared,
                  names if names else None, missed, counts, None, tokens)


def _defined(sums, twins) -> set[str]:
    return {n for g in (sums, twins) for v in g.values() for n in v}


def _build(kind, brain, sums, twins, readback, compared, perturbed_names, undetected, counts, comparisons, x) -> Coverage:
    mode = brain.info.get("mode", "map")
    reference = {g: len(v) for g, v in sums.items()}
    twin_n = {g: len(v) for g, v in twins.items()}
    padding = 0
    if readback is None:                                        # tile transformer: the check's count only, no names
        exposed, missing, unread = None, [], []
    else:
        read = set(readback)
        exposed = {g: sum(1 for n in v if n in read) for g, v in {**sums, **twins}.items()}
        missing = sorted(n for v in sums.values() for n in v if n not in read)
        unread = sorted(n for v in twins.values() for n in v if n not in read)
        padding = len(read - _defined(sums, twins))
    perturbed = None if perturbed_names is None else len(perturbed_names)
    detected = None if perturbed is None else perturbed - len(undetected)
    return Coverage(kind, mode, reference, twin_n, exposed, compared, detected, perturbed, missing, unread, padding,
                    sorted(undetected), counts, comparisons, x)
