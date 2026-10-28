"""Many independent passes of one map, spread over many DOOM games: df.run_many.

    ys = df.run_many(net, xs, workers=8)                        # 8 game processes on this machine
    ys = df.run_many(net, xs, backend="ray", address="auto")    # Ray actors, on this machine or a whole cluster

The map is compiled once, in the calling process. Every worker holds one live DOOM game with that map and runs whole
passes; every pass is checked against the integer reference exactly as `brain(x)` does (MLP: every hidden value and
output; Transformer: every exposed preactivation, score and logit, as `brain.logits(tokens)`). Python sends inputs to
idle games and collects checked outputs; no value crosses between games.

backend="local" starts worker processes (multiprocessing spawn, doomformer._mp). backend="ray" starts one Ray actor
per game (num_cpus=1) and sends it the WAD as bytes through Ray's object store, so the nodes need no shared
filesystem; each node needs the same doomformer and vizdoom installed (not the ACS compiler). A Ray actor that dies
(node lost, process killed) costs only its current input, which runs again once on another game.
"""
from __future__ import annotations

import logging
import os
import queue
import shutil
import signal
import socket
import tempfile
import time
import uuid
from collections import Counter, deque
from collections.abc import Callable, Sequence
from pathlib import Path

import numpy as np

from . import _mp
from .kit import MismatchError

RAY_HINT = 'backend="ray" needs Ray: pip install "doomformer[ray]"'
NAMESPACE = "doomformer"           # Ray namespace of the named game actors (doomformer-<run>-<game>)


# --- Ray plumbing, shared with orchestrator.TilePool ----------------------------------------------------------------
def ray_module():
    """The `ray` package, imported on first use; ImportError with the install command when it is missing."""
    try:
        import ray
    except ImportError:
        raise ImportError(RAY_HINT) from None
    return ray


def connect(ray, address: str | None) -> bool:
    """Join a Ray cluster unless this process already did. address: a cluster address ("auto", "host:6379",
    "ray://host:10001"); None tries a running cluster ("auto") and else starts a local Ray on this machine.
    Returns True when this call initialised Ray (the caller then shuts it down)."""
    if ray.is_initialized():
        return False
    kw = dict(logging_level=logging.WARNING, configure_logging=True)
    if address:
        ray.init(address=address, **kw)
        return True
    try:
        ray.init(address="auto", **kw)
    except ConnectionError:                    # no running cluster: a local Ray on this machine
        ray.init(include_dashboard=False, **kw)
    return True


def pack(paths: Sequence[Path], names: Sequence[str] | None = None) -> dict[str, bytes]:
    """File contents by name, to ship compiled maps to workers that may run on other machines."""
    return {name: Path(p).read_bytes() for p, name in zip(paths, names or [Path(p).name for p in paths])}


def unpack(files: dict[str, bytes], prefix: str) -> str:
    """Write shipped files into a fresh temporary directory on this machine; returns the directory."""
    d = tempfile.mkdtemp(prefix=prefix)
    for name, data in files.items():
        Path(d, name).write_bytes(data)
    return d


def stop_actors(ray, actors: list, grace: float):
    """Ask the actors to close their games (up to `grace` s), then kill them."""
    if grace and actors:
        refs = []
        for a in actors:
            try:
                refs.append(a.close.remote())
            except Exception:                  # already dead
                pass
        if refs:
            ray.wait(refs, num_returns=len(refs), timeout=grace)
    for a in actors:
        try:
            ray.kill(a, no_restart=True)
        except Exception:
            pass


# --- one game ---------------------------------------------------------------------------------------------------------
def open_brain(net, wad: Path):
    """A map Brain on an already compiled WAD (its .json next to it), with its game started."""
    from .kit import MLPBrain, TransformerBrain, _kind
    from .tile import TileMeta
    wad = Path(wad)
    brain = {"mlp": MLPBrain, "transformer": TransformerBrain}[_kind(net)](net, wad, TileMeta.load(wad.with_suffix(
        ".json")))
    brain.runner                               # start the game now, so pass times measure passes only
    return brain


def _pass(brain, x) -> tuple:
    """One checked pass: ("ok", output, checked, tics, seconds), ("mismatch", mismatches) or ("error", text)."""
    t0 = time.perf_counter()
    try:
        out = brain.logits(x) if brain.kind == "transformer" else brain(x)
    except MismatchError as e:
        return ("mismatch", e.mismatches)
    except Exception as e:
        return ("error", f"{type(e).__name__}: {e}")
    return ("ok", out, brain.info["checked"], brain.info["tics"], time.perf_counter() - t0)


def _terminated(signum, frame):
    raise SystemExit(0)                        # unwinds the worker, so `finally` closes its game


def _local_worker(net, wad: str, tasks, results, game: int):
    if hasattr(os, "setpgrp"):
        os.setpgrp()                           # the engine process shares this group (orchestrator.stop_processes)
    signal.signal(signal.SIGTERM, _terminated)
    _mp.exit_with_parent()
    brain = None
    try:
        try:
            brain = open_brain(net, Path(wad))
        except Exception as e:
            results.put((None, game, "error", f"starting the game: {type(e).__name__}: {e}"))
            return
        while True:
            job = tasks.get()
            if job is None:
                break
            i, x = job
            results.put((i, game) + _pass(brain, x))
    finally:
        if brain is not None:
            brain.close()


class _Game:
    """Ray actor body: one DOOM game holding the shipped map."""

    def __init__(self, net, files: dict[str, bytes]):
        self.dir = unpack(files, "doomformer-map-")
        self.brain = open_brain(net, Path(self.dir) / "brain.wad")

    def ready(self) -> tuple[str, int]:
        return socket.gethostname(), os.getpid()

    def run(self, x) -> tuple:
        return _pass(self.brain, x)

    def close(self):
        self.brain.close()
        shutil.rmtree(self.dir, ignore_errors=True)


# --- run_many -------------------------------------------------------------------------------------------------------
class _Collector:
    """Outputs in input order; raises on the first mismatch or failed pass; reports progress."""

    def __init__(self, n: int, progress: Callable[[dict], None] | None):
        self.out: list = [None] * n
        self.done = 0
        self.progress = progress

    @property
    def pending(self) -> bool:
        return self.done < len(self.out)

    def take(self, i: int | None, game: int, status: str, *rest, retried: bool = False):
        if status == "mismatch":
            err = MismatchError(f"input {i}", rest[0])
            err.index = i
            raise err
        if status == "error":
            raise RuntimeError(f"input {i}: {rest[0]}" if i is not None else f"DOOM game {game}: {rest[0]}")
        out, checked, tics, seconds = rest
        self.out[i] = out
        self.done += 1
        if self.progress is not None:
            self.progress(dict(index=i, done=self.done, total=len(self.out), game=game, output=out,
                               checked=checked, tics=tics, seconds=seconds, retried=retried))


def _validated(net, kind: str, inputs, planes: bool = True) -> list:
    xs = []
    for i, x in enumerate(inputs):
        try:
            if kind == "mlp":
                x = net._x(x)
                if x.ndim != 1:
                    raise ValueError("expected one input vector")
                net.check(x, planes=planes)
            else:
                x = [int(t) for t in x]
                if not 1 <= len(x) <= net.T or any(t < 0 or t >= len(net.vocab) for t in x):
                    raise ValueError(f"tokens must be 1 to {net.T} ids in range(0, {len(net.vocab)})")
        except (TypeError, ValueError) as e:
            raise type(e)(f"input {i}: {e}") from None
        xs.append(x)
    return xs


def run_many(net, inputs, *, workers: int | None = None, backend: str = "local", address: str | None = None,
             world=None, actions=None, progress: Callable[[dict], None] | None = None,
             mode: str = "map", opt: int = 0) -> list[np.ndarray]:
    """Run every input through `net` in its own pass on one map spread over `workers` DOOM games; returns the outputs
    in input order (MLP: y as brain(x); Transformer: inputs are token lists, outputs the logits as
    brain.logits(tokens)).

    Every pass is checked exactly; the first difference raises df.MismatchError whose `.index` is the input's
    index, a failed pass RuntimeError naming the input. Inputs are range-checked before any game starts.
    workers: games at once (default: local min(CPU count, inputs); Ray: the cluster's CPUs, at most one per input).
    backend: "local" (processes on this machine) or "ray" (Ray actors, num_cpus=1 each; `address` as for
    ray.init, default: a running cluster, else a local Ray). An actor that dies costs only its current input, which
    runs once more on another game; a second death of the same input raises RuntimeError.
    world/actions: as df.compile (the map is compiled once here; workers get the WAD).
    progress(info): called after each checked pass with index, done, total, game, output, checked (values
    compared), tics, seconds (the pass's wall time) and retried.
    mode: "map" (default: every game holds the whole net as one map), "tiles" (MLP, backend "local": every worker
    holds one game per layer and Python copies the integer layer values between them: the multi-map design, much
    faster for wide nets; doomformer/tilefleet.py; tics are summed over the layers) or "auto" (the one plan()
    recommends; `world`/`actions` need "map"). The same values are checked in both modes.
    opt: the one-map schedule level, as df.compile (0 to 3; tiles have no waves and ignore it)."""
    from .kit import _kind, compile, plan
    kind = _kind(net)
    if mode not in ("map", "tiles", "auto"):
        raise ValueError("mode must be 'map', 'tiles' or 'auto'")
    if mode == "auto":
        mode = "map" if world is not None or actions is not None or kind != "mlp" or backend != "local" \
            else plan(net)["recommended"]
    if mode == "tiles":
        if kind != "mlp" or backend != "local" or world is not None or actions is not None:
            raise ValueError("run_many(mode='tiles') runs an MLP on this machine (backend='local', no world/actions); "
                             "use mode='map' for Transformers, Ray and worlds")
    if kind == "spiking":
        raise TypeError("run_many runs independent passes of an MLP or Transformer; a spiking net keeps its state "
                        "in one game between steps")
    if backend not in ("local", "ray"):
        raise ValueError("backend must be 'local' or 'ray'")
    if workers is not None and (isinstance(workers, bool) or not isinstance(workers, int) or workers < 1):
        raise ValueError("workers must be a positive integer")
    xs = _validated(net, kind, inputs, planes=mode != "tiles")      # tile mode moves no planes (planes.py)
    if not xs:
        return []
    ray = ray_module() if backend == "ray" else None     # a missing Ray fails before the compile
    with compile(net, mode=mode, world=world, actions=actions, opt=opt) as brain:
        wad = brain.wad                                  # tiles: the directory of the layer WADs
    col = _Collector(len(xs), progress)
    if backend == "ray":
        _run_ray(ray, net, wad, xs, workers, address, col)
    else:
        from .tilefleet import tile_worker
        _run_local(net, wad, xs, min(workers or os.cpu_count() or 1, len(xs)), col,
                   tile_worker if mode == "tiles" else _local_worker)
    return col.out


def _run_local(net, wad: Path, xs: list, workers: int, col: _Collector, target=_local_worker):
    from .orchestrator import stop_processes
    ctx = _mp.context()
    tasks, results = ctx.Queue(), ctx.Queue()
    procs = [ctx.Process(target=target, args=(net, str(wad), tasks, results, k), daemon=True)
             for k in range(workers)]
    with _mp.detached_main():
        for p in procs:
            p.start()
    sent = 0
    for _ in range(min(len(xs), 2 * workers)):   # a short queue: idle games pick the next input
        tasks.put((sent, xs[sent]))
        sent += 1
    finished = False
    try:
        while col.pending:
            try:
                i, game, *msg = results.get(timeout=1.0)
            except queue.Empty:
                dead = [k for k, p in enumerate(procs) if not p.is_alive()]
                if dead:
                    raise RuntimeError(f"DOOM game process(es) {dead} died") from None
                continue
            col.take(i, game, *msg)
            if sent < len(xs):
                tasks.put((sent, xs[sent]))
                sent += 1
        finished = True
    finally:
        if finished:
            for _ in procs:
                tasks.put(None)
        stop_processes(procs, 30.0 if finished else 0.0, (tasks, results))


def _run_ray(ray, net, wad: Path, xs: list, workers: int | None, address: str | None, col: _Collector):
    started = connect(ray, address)
    actors: dict[int, object] = {}
    finished = False
    try:
        if workers is None:
            workers = max(1, int(ray.cluster_resources().get("CPU", 1)))
        workers = min(workers, len(xs))
        files = ray.put(pack([wad, wad.with_suffix(".json")], ["brain.wad", "brain.json"]))
        net_ref = ray.put(net)
        game = ray.remote(num_cpus=1, max_restarts=0)(_Game)
        run_id = uuid.uuid4().hex[:8]
        for k in range(workers):
            actors[k] = game.options(name=f"doomformer-{run_id}-{k}", namespace=NAMESPACE).remote(net_ref, files)
        pending = {actors[k].ready.remote(): (k, None) for k in actors}     # task -> (game, input or None)
        todo, idle, attempts, loaded = deque(range(len(xs))), [], Counter(), set()
        while col.pending:
            while idle and todo:
                k, i = idle.pop(), todo.popleft()
                pending[actors[k].run.remote(xs[i])] = (k, i)
            if not pending:
                raise RuntimeError("no DOOM game left to run the remaining inputs")
            (ref,), _ = ray.wait(list(pending), num_returns=1)
            k, i = pending.pop(ref)
            try:
                msg = ray.get(ref)
            except ray.exceptions.RayActorError as e:
                actors.pop(k, None)
                if i is not None:
                    attempts[i] += 1
                    if attempts[i] > 1:
                        raise RuntimeError(f"input {i}: its DOOM game died on two actors; last: {e}") from None
                    todo.appendleft(i)                # once more, on another game
                if not actors:
                    raise RuntimeError(f"every DOOM game actor died; last: {e}") from None
                continue
            if i is None:
                loaded.add(k)
            else:
                col.take(i, k, *msg, retried=attempts[i] > 0)
            idle.append(k)
        finished = True
    finally:                                          # games still waiting for a CPU are killed without a close
        stop_actors(ray, [a for k, a in actors.items() if k in loaded], 30.0 if finished else 0.0)
        stop_actors(ray, [a for k, a in actors.items() if k not in loaded], 0.0)
        if started:
            ray.shutdown()
