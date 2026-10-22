"""Run the integer transformer across many DOOM instances.

The bus only copies integers: every tile output (and its DOOM-computed negation) is written verbatim
into the ports of the tiles that consume it. Each tile execution runs in a persistent ViZDoom instance
owned by a worker; instance ids decide how many distinct DOOM games exist:
  granularity="slot": one game per tile kind and head/slot (score_h{h}_s{s}, wsum_t{t}_h{h}_e{e}, ...)
  granularity="step": additionally one game per generated position for score tiles -> hundreds of games
Workers are local processes (backend="local") or Ray actors (backend="ray", see doomformer.fleet), which receive the
tile WADs as bytes and need no shared filesystem.
"""
from __future__ import annotations

import multiprocessing as mp
import os
import queue
import signal
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from . import _mp, intref
from .intref import QModel


def _terminated(signum, frame):
    raise SystemExit(0)                            # unwinds the worker, so the `finally` below closes every game


def _execute(runners: OrderedDict, tile_dir, job: tuple, record: dict | None, max_live: int) -> tuple:
    """Run one tile task in this worker's games; returns the result tuple TilePool.run collects."""
    from .runtime import TileRunner
    task_id, instance, tile, inputs, hero = job
    try:
        if instance not in runners:
            while len(runners) >= max_live:        # bound memory: close the least recently used game
                runners.popitem(last=False)[1].close()
            kw = dict(automap=True, resolution=record["resolution"]) if record else {}
            runners[instance] = TileRunner(Path(tile_dir) / f"{tile}.wad", **kw)
        runners.move_to_end(instance)
        runner = runners[instance]
        closed_ports = set(inputs.pop("__closed__", []))
        closed = {b["index"] for b in runner.meta.layout["blocks"]
                  if b["lane_port"].split(":", 1)[1].lstrip("-") in closed_ports}
        if closed != getattr(runner, "closed_blocks", set()):
            runner.set_doors(closed)               # ablation: shut the zone doors of those blocks
            runner.closed_blocks = closed
        t0 = time.perf_counter()
        out, tics, frames, full = runner.run(
            inputs, record_every=record["every"] if record else 0, cell=record["cell"] if record else None,
            keep_full=hero)
        return (task_id, out, tics, time.perf_counter() - t0, frames, full, None)
    except Exception as e:                         # surfaced to the caller, never swallowed
        broken = runners.pop(instance, None)       # a game that failed mid-run is not trusted again
        if broken is not None:
            try:
                broken.close()
            except Exception:
                pass
        return (task_id, None, 0, 0.0, [], [], f"{instance}: {e!r}")


def _worker(tile_dir: str, tasks: mp.Queue, results: mp.Queue, record: dict | None, max_live: int):
    if hasattr(os, "setpgrp"):
        os.setpgrp()                               # its engine processes share this group (stop_processes)
    signal.signal(signal.SIGTERM, _terminated)
    _mp.exit_with_parent()
    runners: OrderedDict = OrderedDict()
    try:
        while True:
            job = tasks.get()
            if job is None:
                break
            results.put(_execute(runners, tile_dir, job, record, max_live))
    finally:
        for r in runners.values():
            r.close()


class _TileActor:
    """Ray actor body (doomformer.fleet makes it remote): the tile WADs arrive as bytes, the games run here."""

    def __init__(self, files: dict[str, bytes], record: dict | None, max_live: int):
        from .fleet import unpack
        self.dir = unpack(files, "doomformer-tiles-")
        self.record, self.max_live = record, max_live
        self.runners: OrderedDict = OrderedDict()

    def ready(self) -> bool:
        return True

    def run(self, job: tuple) -> tuple:
        return _execute(self.runners, self.dir, job, self.record, self.max_live)

    def close(self):
        import shutil
        while self.runners:
            self.runners.popitem()[1].close()
        shutil.rmtree(self.dir, ignore_errors=True)


def stop_processes(procs: list, grace: float, queues: tuple = ()):
    """Wait up to `grace` s for worker processes that were asked to finish, then terminate, then kill them with the
    engine processes they spawned. queues: their queues, released when a worker had to be terminated."""
    end = time.monotonic() + grace
    for p in procs:
        p.join(timeout=max(0.0, end - time.monotonic()))
    stragglers = [p for p in procs if p.is_alive()]
    for p in stragglers:
        p.terminate()                              # SIGTERM: the worker closes its games and exits
    end = time.monotonic() + 10
    for p in stragglers:
        p.join(timeout=max(0.0, end - time.monotonic()))
        if p.is_alive():                           # hung inside the engine: kill it and the engine processes it spawned
            try:
                os.killpg(p.pid, signal.SIGKILL)   # the worker's own group (it called setpgrp), never ours
            except (AttributeError, ProcessLookupError, PermissionError):
                pass
            p.kill()
            p.join()
    if stragglers:                                 # unread tasks of a terminated worker must not block exit
        for q in queues:
            q.cancel_join_thread()


class TilePool:
    def __init__(self, tile_dir: Path, workers: int, record: dict | None = None, max_live: int = 64,
                 timeout: float | None = 900.0, backend: str = "local", address: str | None = None):
        """record: None, or {"every": tics between automap frames, "cell": (w, h), "resolution": vzd res}.
        max_live: live ViZDoom instances per worker before the least recently used one is closed.
        timeout: default deadline in seconds of one run() call (None: wait forever). A run() that exceeds it, loses a
        worker or is interrupted terminates the pool, which then refuses further runs (close() it).
        backend: "local" (worker processes on this machine) or "ray" (one Ray actor per worker, num_cpus=1, on the
        cluster at `address`, default: a running cluster, else a local Ray; the tile WADs are shipped as bytes).
        A pool is also a context manager."""
        if backend not in ("local", "ray"):
            raise ValueError("backend must be 'local' or 'ray'")
        self.record = record
        self.timeout = timeout
        self.workers = workers
        self.backend = backend
        self.procs: list = []
        if backend == "ray":
            from . import fleet
            self._ray = fleet.ray_module()
            self._ray_started = fleet.connect(self._ray, address)
            files = self._ray.put(fleet.pack(sorted(p for p in Path(tile_dir).iterdir()
                                                    if p.suffix in (".wad", ".json"))))   # once, for every actor
            actor = self._ray.remote(num_cpus=1, max_restarts=0)(_TileActor)
            self.actors = [actor.remote(files, record, max_live) for _ in range(workers)]
            self._refs: dict = {}                  # in-flight Ray task -> worker
        else:
            ctx = _mp.context()
            self.results = ctx.Queue()
            self.queues = [ctx.Queue() for _ in range(workers)]
            self.procs = [ctx.Process(target=_worker, args=(str(tile_dir), q, self.results, record, max_live),
                                      daemon=True) for q in self.queues]
            with _mp.detached_main():
                for p in self.procs:
                    p.start()
        self.home: dict[str, int] = {}             # instance id -> worker (sticky: the game persists)
        self.next_task = 0
        self.log: list[dict] = []
        self._unusable: str | None = None          # why the pool cannot run tasks any more
        self._stopped = False

    def _send(self, worker: int, job: tuple):
        if self.backend == "ray":
            self._refs[self.actors[worker].run.remote(job)] = worker
        else:
            self.queues[worker].put(job)

    def _receive(self, wait: float) -> tuple | None:
        """The next finished task's result tuple, or None after `wait` s; RuntimeError when a worker died."""
        if self.backend == "ray":
            done, _ = self._ray.wait(list(self._refs), num_returns=1, timeout=wait)
            if not done:
                return None
            worker = self._refs.pop(done[0])
            try:
                return self._ray.get(done[0])
            except self._ray.exceptions.RayActorError as e:
                raise RuntimeError(f"tile worker {worker} (Ray actor) died: {e}") from None
        try:
            return self.results.get(timeout=wait)
        except queue.Empty:
            dead = [i for i, p in enumerate(self.procs) if not p.is_alive()]
            if dead:
                raise RuntimeError(f"tile workers {dead} died")
            return None

    def run(self, tasks: list[tuple[str, str, dict]], timeout: float | None = None) -> list[dict]:
        """tasks: (instance id, tile name, inputs). Runs them concurrently; returns outputs in order.
        When recording, the first task of every call also returns full-resolution frames (the "hero").
        A task that raises does not disturb the pool: every task of the call is collected first, then the first
        error (in task order) is raised as RuntimeError and the pool stays usable (the failed game is dropped).
        timeout: deadline in seconds for the whole call, default self.timeout; TimeoutError when exceeded."""
        if self._unusable:
            raise RuntimeError(f"tile pool is unusable: {self._unusable}")
        limit = self.timeout if timeout is None else timeout
        deadline = None if limit is None else time.monotonic() + limit
        ids = []
        for i, (instance, tile, inputs) in enumerate(tasks):
            if instance not in self.home:
                self.home[instance] = len(self.home) % self.workers
            tid = self.next_task
            self.next_task += 1
            ids.append(tid)
            hero = self.record is not None and self.record.get("hero", True) and i == 0
            self._send(self.home[instance], (tid, instance, tile, inputs, hero))
        meta = {tid: (instance, tile, inputs) for tid, (instance, tile, inputs) in zip(ids, tasks)}
        got, errors = {}, {}
        try:
            while len(got) + len(errors) < len(ids):
                wait = 1.0 if deadline is None else min(1.0, deadline - time.monotonic())
                if wait <= 0:
                    raise TimeoutError(f"tile run exceeded its {limit:g} s deadline "
                                       f"({len(got) + len(errors)} of {len(ids)} tasks done)")
                result = self._receive(wait)
                if result is None:
                    continue
                tid, out, tics, secs, frames, full, err = result
                if err:
                    errors[tid] = err
                    continue
                got[tid] = out
                self.log.append(dict(task=tid, instance=meta[tid][0], tile=meta[tid][1], inputs=meta[tid][2],
                                     outputs=out, tics=tics, secs=secs, frames=frames, full=full))
        except BaseException as e:                 # tasks are still in flight: their late results must not leak
            self._unusable = f"{type(e).__name__}: {e}"
            self._stop(grace=0)
            raise
        if errors:
            first = errors[min(errors)]
            raise RuntimeError(first if len(errors) == 1 else f"{first} (and {len(errors) - 1} more failed tasks)")
        return [got[t] for t in ids]

    @property
    def instances(self) -> int:
        return len(self.home)

    def _stop(self, grace: float):
        """Ask the workers to finish (they close their games), wait up to `grace` s, then terminate, then kill."""
        self._stopped = True
        if self.backend == "ray":
            from .fleet import stop_actors
            stop_actors(self._ray, self.actors, grace)
            self._refs.clear()
            if self._ray_started:
                self._ray.shutdown()
            return
        if grace:
            for q in self.queues:
                q.put(None)
        stop_processes(self.procs, grace, (*self.queues, self.results))

    def close(self, grace: float = 30.0):
        """Stop all workers; they get `grace` seconds to close their games, stragglers are terminated. Idempotent."""
        if not self._stopped:
            self._stop(grace)
        self._unusable = self._unusable or "closed"

    def __enter__(self) -> "TilePool":
        return self

    def __exit__(self, *exc):
        self.close()


def _pairs(prefix: str, values: dict, local: str, idx: list[int]) -> dict:
    """Bind global outputs prefix_i / -prefix_i to tile-local ports local{k} / -local{k}."""
    out = {}
    for k, i in enumerate(idx):
        out[f"{local}{k}"] = values[f"{prefix}_{i}"]
        out[f"-{local}{k}"] = values[f"-{prefix}_{i}"]
    return out


def _bind(values: dict, gname: str, local: str, n: int) -> dict:
    """Bind global gname_i (and its negation) to tile-local local_i for i < n."""
    return _pairs(gname, values, f"{local}_", list(range(n)))


def _store(vals: dict, out: dict, pre: str):
    """Store a tile's local outputs under global names (pre + name; negations keep the leading '-')."""
    for k, v in out.items():
        vals["-" + pre + k[1:] if k.startswith("-") else pre + k] = v


@dataclass
class DoomTransformer:
    model: QModel
    tile_dir: Path
    workers: int = 8
    granularity: str = "slot"
    pos: list[dict] = field(default_factory=list)      # per position: every bus value (name -> int)
    score: dict = field(default_factory=dict)          # (l, h, t, s) -> {"score": v, "-score": -v}
    record: dict | None = None                         # see TilePool; enables automap frame capture
    max_live: int = 64                                 # live DOOM instances per worker (LRU beyond that)
    ablate: dict | None = None                         # intref ablation keys -> units; DOOM closes doors
    backend: str = "local"                             # see TilePool: "local" processes or "ray" actors
    address: str | None = None                         # Ray cluster address (backend="ray")

    def __post_init__(self):
        self.pool = TilePool(self.tile_dir, self.workers, self.record, self.max_live, backend=self.backend,
                             address=self.address)
        self.stage_times: list[tuple[str, float]] = []
        self.timeline: list[dict] = []                 # per stage: label, position, first/last log index

    def _stage(self, label, tasks):
        t0 = time.perf_counter()
        start = len(self.pool.log)
        out = self.pool.run(tasks)
        self.stage_times.append((label, time.perf_counter() - t0))
        self.timeline.append(dict(label=label, t=len(self.pos) - 1, log=(start, len(self.pool.log))))
        return out

    def _closed(self, consumed: dict) -> list[str]:
        """Local port names whose units are ablated. consumed: intref ablation key -> local prefix (port
        "<prefix>_<unit>") or a function unit -> local port names."""
        if not self.ablate:
            return []
        return [name for key, local in consumed.items() for u in self.ablate.get(key, [])
                for name in (local(u) if callable(local) else [f"{local}_{u}"])]

    def _task(self, inst: str, tile: str, inputs: dict, consumed: dict | None = None):
        closed = self._closed(consumed or {})
        return (inst, tile, inputs | ({"__closed__": closed} if closed else {}))

    def step(self, tok: int) -> np.ndarray:
        """Append token `tok` at the next position; returns that position's logits."""
        m, t = self.model, len(self.pos)
        if t >= m.T:
            raise ValueError("context full")
        d, dh, H, g = m.d, m.dh, m.H, self.granularity
        vals: dict[str, int] = {}
        self.pos.append(vals)
        (o,) = self._stage("embed", [self._task("embed", "embed", {"tok": tok, "-tok": -tok, "pos": t, "-pos": -t})])
        _store(vals, o, "")
        x_name, x_key = "z0", "x0"                     # bus name of the current layer input, its ablation key
        for l in range(m.L):
            pre = f"L{l}."
            if m.norm == "rms":
                inputs, consumed = self._norm(l, "att", x_name, x_key), {}
            else:
                inputs, consumed = _bind(vals, x_name, "in", d), {x_key: "in"}
            (o,) = self._stage(pre + "qkv", [self._task(pre + "qkv", pre + "qkv", inputs, consumed)])
            _store(vals, o, pre)
            tasks, keys = [], []
            for h in range(H):
                hidx = list(range(h * dh, (h + 1) * dh))
                for s in range(t + 1):
                    inst = f"{pre}score_h{h}_s{s}" if g == "slot" else f"{pre}score_t{t}_h{h}_s{s}"
                    tasks.append(self._task(inst, pre + "score", _pairs(pre + "zq", vals, "q", hidx)
                                            | _pairs(pre + "zk", self.pos[s], "k", hidx)))
                    keys.append((l, h, t, s))
            for key, out in zip(keys, self._stage(pre + "score", tasks)):
                self.score[key] = out
            if m.attn == "softmax":
                self._softmax(l, x_name, x_key)
            else:
                tasks = []
                for h in range(H):
                    for e in range(dh):
                        j = h * dh + e
                        tasks.append(self._task(f"{pre}wsum_t{t}_h{h}_e{e}", f"{pre}wsum_{t}", self._av(l, h, j)))
                for j, out in enumerate(self._stage(pre + "wsum", tasks)):
                    vals[f"{pre}zo_{j}"], vals[f"-{pre}zo_{j}"] = out["o"], out["-o"]
                (o,) = self._stage(pre + "out", [self._task(
                    f"{pre}out_{t}", f"{pre}out_{t}", _bind(vals, pre + "zo", "zo", d) | _bind(vals, x_name, "in", d),
                    {pre + "ao": "zo", x_key: "in"})])
                _store(vals, o, pre)
            if m.norm == "rms":
                inputs, consumed = self._norm(l, "mlp", pre + "z1", pre + "x1"), {}
            else:
                inputs, consumed = _bind(vals, pre + "z1", "z1", d), {pre + "x1": "z1"}
            (o,) = self._stage(pre + "mlp_up", [self._task(pre + "mlp_up", pre + "mlp_up", inputs, consumed)])
            _store(vals, o, pre)
            hidden = ("zg", "zu") if m.mlp == "reglu" else ("zm",)
            inputs = {k: v for h in hidden for k, v in _bind(vals, pre + h, h, m.dm).items()}
            (o,) = self._stage(pre + "mlp_dn", [self._task(
                pre + "mlp_dn", pre + "mlp_dn", inputs | _bind(vals, pre + "z1", "z1", d),
                {pre + "m": hidden[0], pre + "x1": "z1"})])
            _store(vals, o, pre)
            x_name, x_key = pre + "z2", pre + "x2"
        (o,) = self._stage("unembed", [self._task("unembed", "unembed", _bind(vals, x_name, "in", d),
                                                  {x_key: "in"})])
        _store(vals, o, "")
        return np.array([vals[f"logit_{v}"] for v in range(len(m.vocab))])

    def _norm(self, l: int, site: str, x_name: str, x_key: str) -> dict:
        """RMSNorm of layer l at `site` on the newest position: the sumsq tile (s), then the norm tile (P/N and r) on
        s's DOOM-computed lights. Returns the consumer tile's inputs P_i_k / N_i_k (and negations) as read back."""
        m, pre, vals = self.model, f"L{l}.", self.pos[-1]
        inputs = _bind(vals, x_name, "in", m.d)
        tile = f"{pre}sumsq_{site}"
        (o,) = self._stage(tile, [self._task(tile, tile, inputs, {x_key: "in"})])
        vals[f"{pre}s_{site}"], vals[f"-{pre}s_{site}"] = o["s"], o["-s"]
        tile = f"{pre}norm_{site}"
        (o,) = self._stage(tile, [self._task(tile, tile, inputs | {"s": o["s"], "-s": o["-s"]}, {x_key: "in"})])
        vals[f"{pre}r_{site}"] = o["r"]
        levels = {name: v for name, v in o.items() if name != "r"}
        for name, v in levels.items():                 # P_i_k / -P_i_k -> L{l}.P_<site>_i_k / -L{l}.P_<site>_i_k
            sign, base = ("-", name[1:]) if name.startswith("-") else ("", name)
            key, rest = base.split("_", 1)
            vals[f"{sign}{pre}{key}_{site}_{rest}"] = v
        return levels

    def _av(self, l: int, h: int, j: int | None) -> dict:
        """Inputs of a wsum/den tile at the newest position t: a{s} = score (l, h, t, s), and for unit j (None: no v)
        v{s} = zv_j at position s, for every s <= t."""
        pre, t = f"L{l}.", len(self.pos) - 1
        inputs = {}
        for s in range(t + 1):
            sc = self.score[(l, h, t, s)]
            inputs |= {f"a{s}": sc["score"], f"-a{s}": sc["-score"]}
            if j is not None:
                inputs |= {f"v{s}": self.pos[s][f"{pre}zv_{j}"], f"-v{s}": self.pos[s][f"-{pre}zv_{j}"]}
        return inputs

    def _softmax(self, l: int, x_name: str, x_key: str):
        """Softmax attention of layer l at the newest position: den (one tile run per head) and wsum (one per unit) in
        one stage, then the out tile on P/N's DOOM-computed lights."""
        m, pre, t = self.model, f"L{l}.", len(self.pos) - 1
        vals, dh, LV = self.pos[t], m.dh, m.lv(l)
        tasks = [self._task(f"{pre}den_t{t}_h{h}", f"{pre}den_{t}", self._av(l, h, None)) for h in range(m.H)]
        tasks += [self._task(f"{pre}wsum_t{t}_h{h}_e{e}", f"{pre}wsum_{t}", self._av(l, h, h * dh + e))
                  for h in range(m.H) for e in range(dh)]
        outs = self._stage(pre + "wsum", tasks)
        for h, out in enumerate(outs[:m.H]):
            vals[f"{pre}den_{h}"], vals[f"-{pre}den_{h}"] = out["den"], out["-den"]
        inputs = _bind(vals, x_name, "in", m.d)
        for j, out in enumerate(outs[m.H:]):
            vals[f"{pre}zo_{j}"], vals[f"-{pre}zo_{j}"] = out["o"], out["-o"]
            for key in ("P", "N"):
                for k in range(LV):
                    for sign in ("", "-"):
                        vals[f"{sign}{pre}{key}_{j}_{k}"] = inputs[f"{sign}{key}_{j}_{k}"] = out[f"{sign}{key}_{k}"]
        ablated = {pre + "ao": lambda u: [f"{key}_{u}_{k}" for key in ("P", "N") for k in range(LV)], x_key: "in"}
        (o,) = self._stage(pre + "out", [self._task(pre + "out", pre + "out", inputs, ablated)])
        _store(vals, o, pre)

    def reset(self):
        """Forget all positions (keeps the DOOM games alive) so the brain can serve a new request."""
        self.pos.clear()
        self.score.clear()

    def forward(self, tokens: list[int]) -> np.ndarray:
        """Fresh context: run `tokens` through DOOM, return the last position's logits."""
        self.reset()
        logits = None
        for tok in tokens:
            logits = self.step(tok)
        return logits

    def check_tokens(self, tokens: list[int]) -> dict:
        """Bit-exact check of every DOOM-computed tensor of the current context against intref."""
        ref = {}
        intref.forward(self.model, tokens[: len(self.pos)], ref, ablate=self.ablate)
        causal = np.tril(np.ones((len(self.pos), len(self.pos)), dtype=np.int64))
        tensors = {}
        for key, val in self.trace().items():
            want = ref[key] * causal if key.endswith("score") else ref[key]
            tensors[key] = (int((val == want).sum()), int(val.size))
        exact = all(a == b for a, b in tensors.values()) and self.negation_errors() == 0
        return dict(exact=exact, tensors=tensors)

    def generate(self, prompt: str = "", max_new: int | None = None, temperature: float = 0.0,
                 seed: int = 0) -> str:
        """Mirrors intref.generate from a fresh context; the host only picks a token from DOOM's logits."""
        m = self.model
        self.reset()
        toks = [0] + [m.vocab.index(c) for c in prompt]
        rng = np.random.default_rng(seed)
        logits = None
        for tok in toks:
            logits = self.step(tok)
        budget = max_new if max_new is not None else m.T - len(toks)
        for i in range(budget):
            if len(toks) >= m.T:
                break
            if i:
                logits = self.step(toks[-1])
            nxt = intref.choose(logits, temperature, rng)
            if nxt == 0:
                break
            toks.append(nxt)
        return "".join(m.vocab[i] for i in toks[1:])

    def trace(self) -> dict:
        """DOOM-computed tensors laid out like intref.forward(trace=...)."""
        m, n = self.model, len(self.pos)
        arr = lambda pre, w: np.array([[p[f"{pre}_{i}"] for i in range(w)] for p in self.pos])
        out = dict(z0=arr("z0", m.d), logits=arr("logit", len(m.vocab)))
        for l in range(m.L):
            pre = f"L{l}."
            score = np.zeros((m.H, n, n), dtype=np.int64)
            for (ll, h, t, s), v in self.score.items():
                if ll == l:
                    score[h, t, s] = v["score"]
            out[pre + "score"] = score
            hidden = (("zg", m.dm), ("zu", m.dm)) if m.mlp == "reglu" else (("zm", m.dm),)
            for key, w in (("zq", m.d), ("zk", m.d), ("zv", m.d), ("zo", m.d), ("z1", m.d), *hidden, ("z2", m.d)):
                out[pre + key] = arr(pre + key, w)
            if m.attn == "softmax":
                out[pre + "den"] = arr(pre + "den", m.H)
                for key in ("P", "N"):
                    out[pre + key] = np.array([[[p[f"{pre}{key}_{j}_{k}"] for k in range(m.lv(l))] for j in range(m.d)]
                                               for p in self.pos])
            for site in ("att", "mlp") if m.norm == "rms" else ():
                LN = m.ln(l, site)
                for key in ("s", "r"):
                    out[f"{pre}{key}_{site}"] = np.array([p[f"{pre}{key}_{site}"] for p in self.pos])
                for key in ("P", "N"):
                    out[f"{pre}{key}_{site}"] = np.array([[[p[f"{pre}{key}_{site}_{i}_{k}"] for k in range(LN)]
                                                           for i in range(m.d)] for p in self.pos])
        return out

    def negation_errors(self) -> int:
        """Count bus values whose DOOM-computed twin is not exactly their negation."""
        bad = sum(1 for p in self.pos for k, v in p.items()
                  if not k.startswith("-") and f"-{k}" in p and p[f"-{k}"] != -v)
        return bad + sum(1 for v in self.score.values() if v["-score"] != -v["score"])

    def compare_with_reference(self, prompt: str, text: str, temperature: float = 0.0, seed: int = 0) -> dict:
        """Check generated text and every DOOM-computed intermediate against intref, bit for bit."""
        m = self.model
        rep = self.check_tokens([0] + [m.vocab.index(c) for c in text])
        ref_text = intref.generate(m, prompt, temperature=temperature, seed=seed, ablate=self.ablate)
        return dict(exact=rep["exact"] and text == ref_text, ref_text=ref_text, tensors=rep["tensors"],
                    negation_errors=self.negation_errors())

    def close(self):
        self.pool.close()
