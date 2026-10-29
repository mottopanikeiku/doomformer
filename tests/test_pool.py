"""TilePool and TileRunner lifecycle: failures, deadlines, shutdown, and the working directory ViZDoom leaves alone."""
import multiprocessing
import os
import signal
import time
from types import SimpleNamespace

import pytest

from doomformer.model_tiles import build
from doomformer.onemap import tiny_model
from doomformer.orchestrator import TilePool
from doomformer.runtime import TileRunner


@pytest.fixture(scope="module")
def tile_dir(tmp_path_factory):
    d = tmp_path_factory.mktemp("pool_tiles")
    build(tiny_model(), d, jobs=2)
    return d


def embed(tok, pos=0):
    return {"tok": tok, "-tok": -tok, "pos": pos, "-pos": -pos}


def test_failed_task_does_not_poison_the_pool(tile_dir):
    with TilePool(tile_dir, 2, timeout=120) as pool:
        good = lambda tok: (f"g{tok}", "embed", embed(tok))      # noqa: E731
        bad = lambda name: (name, "no_such_tile", embed(0))      # noqa: E731
        want = [pool.run([good(t)])[0] for t in (0, 1)]
        assert want[0] != want[1]
        with pytest.raises(RuntimeError, match=r"a: .*FileNotFoundError.*\(and 1 more failed tasks\)"):
            pool.run([bad("a"), good(0), bad("c")])
        # every earlier task was collected: the next call must not see a stale error or a stale result
        assert pool.run([good(1)]) == [want[1]]
        with pytest.raises(RuntimeError, match=r"^c: .*FileNotFoundError") as err:
            pool.run([bad("c")])
        assert "more failed" not in str(err.value)
        assert pool.run([good(0), good(1)]) == want
        assert [e["instance"] for e in pool.log].count("c") == 0   # failed tasks are not logged as results


def test_run_deadline_stops_the_pool(tile_dir):
    pool = TilePool(tile_dir, 1, timeout=120)
    procs = list(pool.procs)
    try:
        with pytest.raises(TimeoutError, match="deadline"):
            pool.run([("g", "embed", embed(0))], timeout=0.001)
        with pytest.raises(RuntimeError, match="unusable"):
            pool.run([("g", "embed", embed(0))])
    finally:
        pool.close()
    assert not any(p.is_alive() for p in procs)


def test_close_leaves_no_children(tile_dir):
    pool = TilePool(tile_dir, 2)
    procs = list(pool.procs)
    pool.run([("a", "embed", embed(0)), ("b", "embed", embed(1))])
    assert all(p.is_alive() for p in procs)
    pool.close()
    pool.close()                                                  # idempotent
    assert not any(p.is_alive() for p in procs)
    assert multiprocessing.active_children() == []
    with pytest.raises(RuntimeError, match="unusable"):
        pool.run([("a", "embed", embed(0))])


def test_runner_context_manager_and_clean_cwd(tile_dir, tmp_path, monkeypatch):
    """ViZDoom's _vizdoom.ini and _vizdoom/ go to the cache, not to the caller's directory."""
    wad = tile_dir / "embed.wad"
    work = tmp_path / "work"
    work.mkdir()
    monkeypatch.chdir(work)
    with TileRunner(wad) as runner:
        out = runner.run({"tok": 1, "-tok": -1, "pos": 0, "-pos": 0})[0]
        assert out
        game = runner.game
        assert game.is_running()
    assert not game.is_running()
    assert list(work.iterdir()) == []


@pytest.mark.skipif(not hasattr(signal, "SIGSTOP"), reason="needs POSIX job-control signals")
def test_hung_worker_is_killed_at_the_deadline(tile_dir, monkeypatch):
    pool = TilePool(tile_dir, 2, timeout=120)
    procs = list(pool.procs)
    try:
        pool.run([("a", "embed", embed(0)), ("b", "embed", embed(1))])
        # the workers send their results through one pipe under one shared lock. A worker stopped after writing its
        # result but before releasing the lock would block the other worker's results too. A later result from the
        # other worker shows the victim has released it.
        pool.run([("b", "embed", embed(1))])
        victim = procs[0].pid
        # the only signals this test (and TilePool._stop) can send go to the pool's own worker, which leads its own
        # process group: never to this process, its group, or anything else on the machine
        assert victim and victim > 1 and victim != os.getpid()
        assert os.getpgid(victim) == victim != os.getpgrp()
        os.kill(victim, signal.SIGSTOP)                           # alive, but never answers
        # the 1 s deadline should time the hung worker, not how fast a loaded runner finishes the healthy task. So
        # run()'s clock stands still until the healthy result is in pool.log (or 60 s have passed), then runs at real
        # speed.
        logged, frozen_at, started = len(pool.log), time.monotonic(), []

        def clock():
            if not started and (len(pool.log) > logged or time.monotonic() - frozen_at > 60):
                started.append(time.monotonic())
            return time.monotonic() - started[0] if started else 0.0

        with monkeypatch.context() as m:
            m.setattr("doomformer.orchestrator.time", SimpleNamespace(monotonic=clock))
            with pytest.raises(TimeoutError, match="1 of 2 tasks done"):
                pool.run([("a", "embed", embed(0)), ("b", "embed", embed(1))], timeout=1)
        assert 1 <= time.monotonic() - started[0] < 60            # waited out the deadline, then stopped the pool
    finally:
        pool.close()
    assert not any(p.is_alive() for p in procs)


ORPHAN = """
import os, sys, time
from doomformer.orchestrator import TilePool
pool = TilePool(sys.argv[1], 1, timeout=120)
pool.run([("a", "embed", {"tok": 1, "-tok": -1, "pos": 0, "-pos": 0})])          # the worker now holds a live game
print(pool.procs[0].pid, flush=True)
time.sleep(600)
"""


def _children(pid: int) -> set[int]:
    """Every process whose parent is `pid` (Linux /proc)."""
    out = set()
    for entry in os.listdir("/proc"):
        if entry.isdigit():
            try:
                with open(f"/proc/{entry}/stat") as f:
                    if int(f.read().rsplit(")", 1)[1].split()[1]) == pid:
                        out.add(int(entry))
            except (OSError, ValueError, IndexError):
                pass
    return out


@pytest.mark.skipif(not os.path.isdir("/proc"), reason="reads /proc")
def test_workers_and_engines_exit_when_the_parent_is_killed(tile_dir):
    """SIGKILL the process that owns a pool: its worker closes its DOOM games and exits on its own."""
    import subprocess
    import sys
    parent = subprocess.Popen([sys.executable, "-c", ORPHAN, str(tile_dir)], stdout=subprocess.PIPE, text=True)
    worker = int(parent.stdout.readline())
    engines = _children(worker)
    assert engines                                                # the worker's ViZDoom engine is running
    parent.kill()
    parent.wait()
    deadline = time.monotonic() + 30
    alive = lambda: {p for p in {worker} | engines if os.path.exists(f"/proc/{p}")    # noqa: E731
                     and open(f"/proc/{p}/stat").read().rsplit(")", 1)[1].split()[0] != "Z"}
    while alive() and time.monotonic() < deadline:
        time.sleep(0.5)
    assert not alive(), f"left running after the parent died: {sorted(alive())}"
