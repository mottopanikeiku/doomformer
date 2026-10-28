"""Verified multi-game brains, reusing model_tiles and orchestrator's integer bus.

DOOM dolls perform each tile's integer arithmetic. ACS resets gates and copies
sector lights to USER variables; it does not evaluate the model. Python copies
read-back integers (including DOOM-computed negation twins) into subsequent games,
schedules tiles, samples tokens, and independently verifies every exposed value.
There is at most one live DOOM game per worker; tile changes may reload a game.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np

from .acc import cache_dir
from .intref import QModel, forward
from .kit import Brain, MLPBrain, MismatchError, TransformerBrain
from .model_tiles import _compiler_version, _compile, build, spec_key, tile_specs
from .orchestrator import DoomTransformer, TilePool
from .tile import MAX_SPAN, TileMeta, compile_tile

AUTO_SETTLE_TICS = 2000


def specs_for(net):
    return tile_specs(net) if isinstance(net, QModel) else net._stages()


def tile_plan(net) -> dict:
    """Exact emitted sector/doll counts summed over distinct tile WADs (not runs). span: the largest tile extent per
    axis; fits: every tile spans <= MAX_SPAN per axis (docs/engine_limits.md) and has < 20,000 dolls (doll and
    reset TIDs); otherwise reason names the tiles that do not fit."""
    metas = [compile_tile(s, Path("layout_only.wad"), layout_only=True) for s in specs_for(net)]
    bad = [f"{m.name} spans {m.layout['span'][0]} x {m.layout['span'][1]} units" for m in metas
           if not m.layout["fits"]]
    bad += [f"{m.name} has {m.n_dolls} dolls" for m in metas if m.n_dolls >= 20000]
    report = dict(dolls=sum(m.n_dolls for m in metas), sectors=sum(m.n_sectors for m in metas),
                  fits=not bad, span=[max(m.layout["span"][i] for m in metas) for i in (0, 1)], tiles=len(metas),
                  max_dolls=max(m.n_dolls for m in metas), max_sectors=max(m.n_sectors for m in metas),
                  per_tile=[dict(name=m.name, sectors=m.n_sectors, dolls=m.n_dolls, ports=len(m.ports),
                                 outputs=len(m.outputs),
                                 zone=max((b["zone"][2] - b["zone"][0] for b in m.layout["blocks"]), default=0))
                            for m in metas])
    if bad:
        shown = "; ".join(bad[:3]) + (f" (+{len(bad) - 3} more)" if len(bad) > 3 else "")
        report["reason"] = (f"{shown}; a tile map must span <= {MAX_SPAN} units per axis (docs/engine_limits.md) "
                            "and hold < 20,000 dolls")
    return report


def compile_tiles(net, wad=None, *, workers=1, backend="local", address=None):
    """Build every tile (cached by content) and return the multi-game Brain; ValueError if a tile does not fit.
    backend/address: where the tile games run (orchestrator.TilePool: "local" processes or "ray" actors)."""
    planned = tile_plan(net)
    if not planned["fits"]:
        raise ValueError(f"tiles do not fit: {planned['reason']}")
    specs = specs_for(net)
    version = _compiler_version()
    if wad is None:
        digest = hashlib.sha256("".join(spec_key(s, version) for s in specs).encode()).hexdigest()[:16]
        directory = cache_dir() / "brains" / f"tiles-{digest}"
    else:
        directory = Path(wad)
        if directory.suffix == ".wad":
            directory = directory.with_suffix(".tiles")
    directory.mkdir(parents=True, exist_ok=True)
    if isinstance(net, QModel):
        manifest = build(net, directory, jobs=workers)
    else:
        manifest = {}
        for spec in specs:
            path, key = directory / f"{spec.name}.wad", spec_key(spec, version)
            stamp = path.with_suffix(".key")
            if path.exists() and stamp.exists() and stamp.read_text() == key:
                meta = TileMeta.load(path.with_suffix(".json"))
            else:
                meta = _compile(spec, path, key)
            manifest[spec.name] = dict(dolls=meta.n_dolls, sectors=meta.n_sectors)
    cls = TiledTransformerBrain if isinstance(net, QModel) else TiledMLPBrain
    return cls(net, directory, manifest, workers, backend=backend, address=address)


class TiledBrain(Brain):
    """Brain lifecycle and counters for a directory of WADs, not one game.

    `wad` is the tile directory; use `engine.pool` (transformer) or `pool` (MLP)
    for execution logs. Single-map runner/run_script/world/actions are unavailable.
    `runs` counts forward requests; `tile_runs` counts individual tile executions.
    """

    def __init__(self, net, directory, manifest, workers, backend="local", address=None):
        self.net, self.workers = net, workers
        self.backend, self.address = backend, address
        self._pool = None
        self.info = dict(kind=self.kind, mode="tiles", wad=str(directory), workers=workers, backend=backend,
                         tiles=len(manifest), dolls=sum(v["dolls"] for v in manifest.values()),
                         sectors=sum(v["sectors"] for v in manifest.values()), runs=0,
                         tile_runs=0, tics=None, checked=None)

    @property
    def runner(self):
        raise TypeError("tile mode has multiple games, not brain.runner; use mode='map' for world/recording hooks")

    @property
    def pool(self):
        if self._pool is None:
            self._pool = TilePool(self.wad, self.workers, max_live=1, backend=self.backend, address=self.address)
        return self._pool

    def _account(self, pool, start):
        self.info["runs"] += 1
        self.info["tile_runs"] += len(pool.log) - start
        self.info["tics"] = sum(entry["tics"] for entry in pool.log[start:])

    def close(self):
        if self._pool is not None:
            self._pool.close()
            self._pool = None


class TiledMLPBrain(TiledBrain, MLPBrain):
    """One tile per MLP layer. Python copies hidden integers, never activations."""
    kind = "mlp"

    def run(self, x, check=False):
        x = self.net._x(x)
        if x.ndim != 1:
            raise ValueError("run takes one input vector")
        self.net.check(x, planes=False)                 # tile mode moves no planes (planes.py)
        values = {f"{sign}x{i}": int(v) * (-1 if sign else 1)
                  for i, v in enumerate(x) for sign in ("", "-")}
        got = {}
        pool = self.pool
        start = len(pool.log)
        for l in range(self.net.n_layers):
            tile = f"mlp_l{l}"
            (out,) = pool.run([(tile, tile, values)])
            got.update(out)
            values = out
        self._account(pool, start)
        if check:
            ref = self.net.reference(x)
            bad = [(n, v, self.net._expected(n, ref)) for n, v in got.items()
                   if v != self.net._expected(n, ref)]
            self.info["checked"] = len(got)
            if bad:
                raise MismatchError(f"tiled MLP pass x={x.tolist()}", bad)
        return got


class TiledTransformerBrain(TiledBrain, TransformerBrain):
    """Incremental causal transformer with every exposed tensor/twin checked.

    Prefix reuse only reuses integers previously read from DOOM. Each logits call
    rechecks the entire requested context against an independent intref.forward.
    TransformerBrain.generate supplies the same Python sampling as map mode.
    """
    kind = "transformer"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._engine = None
        self._tokens = []

    @property
    def engine(self):
        if self._engine is None:
            self._engine = DoomTransformer(self.net, self.wad, workers=self.workers, max_live=1,
                                           backend=self.backend, address=self.address)
        return self._engine

    @property
    def pool(self):
        return self.engine.pool

    def intervene(self, interventions=None, ablate: dict | None = None):
        """Run the following contexts with these interventions / this ablation (doomformer.intervene); every call of
        the brain is then checked against the equally intervened reference. No arguments: back to the plain model."""
        self.engine.use(interventions, ablate)
        self.engine.reset()
        self._tokens = []

    def logits(self, tokens):
        tokens = [int(t) for t in tokens]
        if not 1 <= len(tokens) <= self.net.T or any(t < 0 or t >= len(self.net.vocab) for t in tokens):
            raise ValueError("tokens must be a nonempty in-vocabulary sequence within context")
        ref = {}
        forward(self.net, tokens, ref, **self.engine._ref_kwargs())
        engine = self.engine
        if tokens[:len(self._tokens)] != self._tokens:
            engine.reset()
            self._tokens = []
        start = len(engine.pool.log)
        for tok in tokens[len(self._tokens):]:
            engine.step(tok)
            self._tokens.append(tok)
        self._account(engine.pool, start)
        trace = engine.trace()
        bad, checked = [], 0
        for key, val in trace.items():
            want = ref[key]
            mask = np.tril(np.ones(val.shape, dtype=bool)) if key.endswith("score") else np.ones(val.shape, bool)
            checked += int(mask.sum())
            for idx in zip(*np.nonzero((val != want) & mask)):
                bad.append((key + ":" + ":".join(map(str, idx)), int(val[idx]), int(want[idx])))
        # Negations are independent light readbacks, not Python-created values (patched cells: as DOOM read them).
        for t in range(len(engine.pos)):
            values = engine.view(t)
            for key, val in values.items():
                if key.startswith("-"):
                    checked += 1
                    want = -values[key[1:]]
                    if val != want:
                        bad.append((f"{key}:{t}", val, want))
        for c in engine.patches:                     # the host's patch value must realise the reference's activation
            checked += 1
            want = int(ref[c["site"]][c["pos"], c["unit"]])
            if c["count"] != want:
                bad.append((f"patch:{c['site']}:{c['pos']}:{c['unit']}", c["count"], want))
        for key, values in engine.score.items():
            checked += 1
            if values["-score"] != -values["score"]:
                bad.append((f"-score:{key}", values["-score"], -values["score"]))
        self.info["checked"] = checked
        if bad:
            raise MismatchError(f"tiled transformer pass {tokens}", bad)
        return trace["logits"]

    __call__ = logits

    def close(self):
        if self._engine is not None:
            self._engine.close()
            self._engine = None
            self._tokens = []
