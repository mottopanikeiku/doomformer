"""`doomformer bench`: the engine-time measurements behind docs/scaling.md, as commands.

    doomformer bench idle                         the idle conveyor tax: CPU per tic at conveyor stack 4, 2, 1, 0
    doomformer bench pass --net mnist --opt 0 3   engine CPU of one pass of a net in one map, against NumPy
    doomformer bench tiles --workers 1,2,4,8,16   MNIST in tile mode: digits per minute, CPU per digit per worker count

Engine CPU is the user+sys time of the ViZDoom processes (read from /proc, Linux only), not wall time. Every command
prints the load average at its start and end: on a shared machine the wall times and, a little, the CPU times
are not the quiet-machine figures (a log is worth keeping when the 1-minute load average stays under 4).
"""
from __future__ import annotations

import argparse
import os
import time

import numpy as np

TICK = os.sysconf("SC_CLK_TCK") if hasattr(os, "sysconf") else 100
IDLE_NETS = (([12, 16, 6], 3), ([32, 32, 10], 3), ([64, 64, 10], 3))      # costcheck.make(shape, K): weights +-3, seed 0
IDLE_TICS, IDLE_REPEATS = 300, 5


def _procs() -> dict[int, tuple[int, str, float]]:
    """pid -> (parent pid, command name, user+sys CPU seconds) of every process."""
    out = {}
    for name in os.listdir("/proc"):
        if name.isdigit():
            try:
                with open(f"/proc/{name}/stat") as f:
                    head, tail = f.read().rsplit(")", 1)
            except OSError:
                continue
            st = tail.split()
            out[int(name)] = (int(st[1]), head.split("(", 1)[1], (int(st[11]) + int(st[12])) / TICK)
    return out


def engine_cpu(descendants: bool = False) -> float:
    """CPU seconds of the ViZDoom engines that are children of this process (descendants=True: any depth, as the
    engines of tile workers are) so far."""
    procs, me = _procs(), os.getpid()
    total = 0.0
    for pid, (ppid, comm, cpu) in procs.items():
        if "vizdoom" not in comm:
            continue
        anc, hops = ppid, 0
        while anc not in (0, 1) and hops < 8:
            if anc == me:
                total += cpu
                break
            if not descendants and hops == 0:
                break
            anc, hops = procs.get(anc, (0,))[0], hops + 1
    return total


def load() -> float:
    return os.getloadavg()[0]


def _need_linux() -> None:
    if not os.path.isdir("/proc/self"):
        raise SystemExit("bench reads the engine's CPU time from /proc (Linux only)")


# --- bench idle -------------------------------------------------------------------------------------------------------
def make(shape, K, seed=0):
    """The random MLP of demos/costcheck.py: weights in [-3, 3], biases in [-10, 10], K thresholds per layer."""
    import doomformer as df
    rng = np.random.default_rng(seed)
    W = [rng.integers(-3, 4, (b, a)) for a, b in zip(shape, shape[1:])]
    B = [rng.integers(-10, 11, b) for b in shape[1:]]
    T = [np.arange(1, K + 1) * (20 if l == 0 else 12 * max(1, shape[l] // 8)) for l in range(len(W))]
    return df.MLP(W, B, T)


def idle(args) -> int:
    import doomformer as df
    from doomformer import tile
    from doomformer.cost import MAP_NS_PER_SECTOR_TIC
    stacks = [int(v) for v in args.stacks.split(",")]
    print(f"idle conveyor tax: one-map builds of random +-3 weight MLPs (`make` below = demos.costcheck.make, K=3), {IDLE_TICS} idle tics x "
          f"{IDLE_REPEATS} per stack; load average {load():.1f} at start", flush=True)
    print(f"{'net':<12}{'sectors':>8}{'stack':>6}{'us per tic':>12}{'ns per sector-tic':>19}  note")
    saved = tile.CONVEYOR["stack"]
    try:
        builds = [("-".join(map(str, shape)), make(shape, K), 0) for shape, K in IDLE_NETS] if not args.mnist_only else []
        if args.mnist or args.mnist_only:
            from doomformer import shipped
            builds.append(("mnist opt 3", df.MLP.load(shipped.data("mnist/mlp.npz")), 3))
        for label, net, opt in builds:
            base = None
            for stack in stacks:
                tile.CONVEYOR["stack"] = stack
                with df.compile(net, mode="map", opt=opt) as brain:
                    game = brain.runner.game
                    sectors = brain.meta.n_sectors
                    game.advance_action(20)
                    c0, t0 = engine_cpu(), time.perf_counter()
                    for _ in range(IDLE_REPEATS):
                        game.advance_action(IDLE_TICS)
                    cpu, wall = engine_cpu() - c0, time.perf_counter() - t0
                n = IDLE_TICS * IDLE_REPEATS
                # stack 0: the engine's 10 ms CPU timer is too coarse, so the wall time is shown (these maps do not run)
                per = (wall if stack == 0 else cpu) / n * 1e6
                base = base or per
                print(f"{label:<12}{sectors:>8,}{stack:>6}{per:>12.0f}{per * 1e3 / sectors:>19.1f}  "
                      f"{'wall time' if stack == 0 else 'engine CPU'}, {base / per:.1f}x below stack {stacks[0]}", flush=True)
    finally:
        tile.CONVEYOR["stack"] = saved
    print(f"the cost model's constant is {MAP_NS_PER_SECTOR_TIC:.0f} ns per sector-tic (busy passes); load average "
          f"{load():.1f} at end")
    return 0


# --- bench pass -------------------------------------------------------------------------------------------------------
def _load_net(name: str):
    import doomformer as df
    from doomformer import shipped, tune
    if name == "mnist":
        net = df.MLP.load(shipped.data("mnist/mlp.npz"))
        return net, np.asarray(tune.mnist_digits(1)[0])
    net = tune.make_net(name)
    return net, np.asarray(tune.make_inputs(net, 1, 7)[0][0])


def one_pass(args) -> int:
    import timeit

    import doomformer as df
    net, x = _load_net(args.net)
    ref = timeit.timeit(lambda: net.reference(x), number=200) / 200
    print(f"{args.net}: NumPy reference {ref * 1e6:.1f} us per pass; load average {load():.1f} at start", flush=True)
    for opt in args.opt:
        with df.compile(net, mode="map", opt=opt) as brain:
            brain(x)                                              # warm-up pass
            c0, t0 = engine_cpu(), time.perf_counter()
            for _ in range(args.passes):
                brain(x)
            cpu, wall = (engine_cpu() - c0) / args.passes, (time.perf_counter() - t0) / args.passes
            tics, sectors = brain.info["tics"], brain.meta.n_sectors
        print(f"opt {opt}: engine CPU {cpu:.2f} s per pass, wall {wall:.2f} s, {cpu / ref:,.0f}x NumPy | {tics:,} game tics, "
              f"{sectors:,} sectors, {cpu / tics / sectors * 1e9:.0f} ns per sector-tic | {args.passes} timed pass"
              f"{'es' if args.passes > 1 else ''}, load average {load():.1f}", flush=True)
    return 0


# --- bench tiles ------------------------------------------------------------------------------------------------------
def tiles(args) -> int:
    import doomformer as df
    from doomformer import shipped, tune
    net = df.MLP.load(shipped.data("mnist/mlp.npz"))
    base = tune.mnist_digits(100)
    n = args.digits
    digits = np.asarray([base[i % len(base)] for i in range(n)])
    xs = net.encode(digits.astype(np.float64))
    print(f"MNIST in tile mode, {n} digits (the {len(base)} shipped test digits, repeated {-(-n // len(base))}x when "
          f"N > {len(base)}), every pass checked exactly; {os.cpu_count()} CPUs; load average {load():.1f} at start", flush=True)
    print(f"{'workers':>7}{'engines':>8}{'wall s':>8}{'digits/min':>11}{'steady/min':>11}{'mean pass s':>12}"
          f"{'engine CPU/digit':>17}{'all CPU/digit':>14}  load")
    L = net.n_layers
    for w in (int(v) for v in args.workers.split(",")):
        w = min(w, n)
        marks = {}
        passes = []

        def progress(info, w=w, marks=marks, passes=passes):
            passes.append(info["seconds"])
            if info["done"] == info["total"]:                          # sampled while the workers and engines are alive
                marks["last"] = (time.perf_counter(), engine_cpu(True), _all_cpu())
            if info["done"] == w and "first" not in marks:             # the first wave finished: games are loaded
                marks["first"] = (time.perf_counter(), engine_cpu(True), _all_cpu())

        t0, load0 = time.perf_counter(), load()
        cpu0, all0 = engine_cpu(True), _all_cpu()
        df.run_many(net, xs, workers=w, mode="tiles", progress=progress)
        wall = time.perf_counter() - t0
        t2, cpu1, all1 = marks.get("last", (time.perf_counter(), cpu0, all0))
        t1, c1, a1 = marks.get("first", (t0, cpu0, all0))
        steady = (n - w) / (t2 - t1) * 60 if n > w else float("nan")
        cpu_d = (cpu1 - c1) / (n - w) if n > w else float("nan")
        all_d = (all1 - a1) / (n - w) if n > w else float("nan")
        print(f"{w:>7}{L * w:>8}{wall:>8.1f}{n / wall * 60:>11.1f}{steady:>11.1f}{float(np.mean(passes)):>12.2f}"
              f"{cpu_d:>17.3f}{all_d:>14.3f}  {load0:.1f} -> {load():.1f}", flush=True)
    print("steady/min and the CPU columns start when the first wave of passes is done (the games are loaded): "
          "(N - workers) digits; engine CPU = the ViZDoom processes, all CPU adds the Python workers")
    return 0


def _all_cpu() -> float:
    """CPU seconds of every descendant process of this one (engines and Python workers)."""
    procs, me = _procs(), os.getpid()
    total = 0.0
    for pid, (ppid, comm, cpu) in procs.items():
        anc, hops = ppid, 0
        while anc not in (0, 1) and hops < 8:
            if anc == me:
                total += cpu
                break
            anc, hops = procs.get(anc, (0,))[0], hops + 1
    return total


# --- command line -----------------------------------------------------------------------------------------------------
def add_arguments(p: argparse.ArgumentParser) -> None:
    sub = p.add_subparsers(dest="bench_cmd", required=True, metavar="what")
    i = sub.add_parser("idle", help="the idle conveyor tax: engine CPU per tic of three nets at conveyor stack 4, 2, 1, 0")
    i.add_argument("--mnist", action="store_true", help="also the shipped MNIST net at --opt 3")
    i.add_argument("--mnist-only", action="store_true", help="only the shipped MNIST net at opt 3")
    i.add_argument("--stacks", default="4,2,1,0", help="conveyor scrollers per sector to build with (default 4,2,1,0)")
    o = sub.add_parser("pass", help="engine CPU of one pass in one map, against the NumPy reference")
    o.add_argument("--net", default="mnist", help="mnist or a tune reference net name (default mnist)")
    o.add_argument("--opt", type=int, nargs="+", choices=range(4), default=[0, 3], help="schedule levels (default 0 3)")
    o.add_argument("--passes", type=int, default=1, help="timed passes per level (default 1; opt 0 MNIST takes about 75 s)")
    t = sub.add_parser("tiles", help="MNIST in tile mode: digits per minute and engine CPU per digit per worker count")
    t.add_argument("--workers", default="1,2,4,8,16", help="comma list of worker counts (default 1,2,4,8,16)")
    t.add_argument("--digits", type=int, default=200, help="digits per worker count (default 200)")


def run(args) -> int:
    _need_linux()
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    return {"idle": idle, "pass": one_pass, "tiles": tiles}[args.bench_cmd](args)
