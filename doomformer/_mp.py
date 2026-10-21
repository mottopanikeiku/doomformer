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
