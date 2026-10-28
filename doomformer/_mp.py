"""Worker processes that are safe in a multi-threaded parent and in scripts without a __main__ guard.

fork() in a process that already runs threads (torch, ViZDoom, a ProcessPoolExecutor's own manager thread) can
deadlock the child, and Python 3.12+ warns about it. Workers therefore always start with "spawn": the same
semantics on Linux and macOS, about 50 ms per worker (about 0.2 s for four, measured).
"""
from __future__ import annotations

import contextlib
import multiprocessing as mp
import os
import signal
import sys
import threading
import time


def context():
    return mp.get_context("spawn")


def exit_with_parent(poll: float = 1.0) -> None:
    """Call at the start of a worker. If the parent dies without closing the pool (killed, crashed, `timeout`), the
    worker sends itself SIGTERM, so its `finally` closes the DOOM games instead of leaving engines running."""
    parent = os.getppid()

    def watch():
        while os.getppid() == parent:
            time.sleep(poll)
        os.kill(os.getpid(), signal.SIGTERM)

    threading.Thread(target=watch, name="exit-with-parent", daemon=True).start()


@contextlib.contextmanager
def detached_main():
    """Keep spawned workers from re-importing the parent's __main__ script.

    A spawned child imports the parent's main script again; a script without an `if __name__ == "__main__"` guard
    would then run itself inside every worker. Our worker functions live in the package and need nothing from the
    script, so its origin is hidden while the workers start (wrap a ProcessPoolExecutor's whole `with` block)."""
    main = sys.modules.get("__main__")
    saved = {name: getattr(main, name) for name in ("__file__", "__spec__") if hasattr(main, name)}
    if main is not None:
        main.__spec__ = None
        saved.setdefault("__spec__", None)
        if "__file__" in saved:
            del main.__file__
    try:
        yield
    finally:
        if main is not None:
            for name, value in saved.items():
                setattr(main, name, value)


_WATCHDOG = """P=$1; E=$2
while [ "$(ps -o ppid= -p $E 2>/dev/null | tr -d ' ')" = "$P" ]; do sleep 1; done
case "$(ps -o stat= -p $P 2>/dev/null)" in
  ""|Z*) case "$(ps -o comm= -p $E 2>/dev/null)" in *vizdoom*) kill -9 $E;; esac;;
esac"""
_watchdogs: list = []


def children() -> set[int]:
    """Pids of this process's child processes: /proc on Linux, `pgrep -P` elsewhere; empty when neither works."""
    me = os.getpid()
    if os.path.isdir("/proc/self"):
        found = set()
        for name in os.listdir("/proc"):
            if name.isdigit():
                try:
                    with open(f"/proc/{name}/stat") as f:
                        if int(f.read().rsplit(")", 1)[1].split()[1]) == me:
                            found.add(int(name))
                except (OSError, ValueError, IndexError):
                    pass
        return found
    import subprocess
    try:
        out = subprocess.run(["pgrep", "-P", str(me)], capture_output=True, text=True, timeout=5).stdout
    except (OSError, subprocess.SubprocessError):
        return set()
    return {int(p) for p in out.split()}


def guard_engine(pid: int) -> None:
    """Kill the ViZDoom engine process `pid` (a child of this process) if this process dies without closing its game.

    ViZDoom's engine is a separate process that keeps running (about 0.5 to 1.2 GB for a big map) after `kill`, a crash
    or `kill -9` of its Python parent. A tiny detached `sh` loop, one per engine, polls once a second and sends the
    engine SIGKILL when the parent is gone or a zombie; it exits by itself when the engine is no longer the parent's
    child (an ordinary `close()`). It needs `sh`, `ps` and `tr` (Linux, macOS) and does nothing on Windows."""
    if os.name == "nt":
        return
    import subprocess
    _watchdogs[:] = [p for p in _watchdogs if p.poll() is None]        # reap the ones that finished
    try:
        _watchdogs.append(subprocess.Popen(["sh", "-c", _WATCHDOG, "doomformer-watchdog", str(os.getpid()), str(pid)],
                                           stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                                           stderr=subprocess.DEVNULL, start_new_session=True))
    except OSError:
        pass
