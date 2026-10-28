"""doomformer bench: the /proc readers and the command line (the measurements themselves run games and are logged commands,
models/bench/)."""
import os
import subprocess
import sys

import pytest

from doomformer import bench
from doomformer.cli import main


@pytest.mark.skipif(not os.path.isdir("/proc/self"), reason="Linux only")
def test_engine_cpu_counts_only_vizdoom_descendants():
    assert bench.engine_cpu() == 0.0 and bench.engine_cpu(descendants=True) == 0.0
    child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(5)"])
    try:
        procs = bench._procs()
        assert procs[child.pid][0] == os.getpid()
        assert bench._all_cpu() >= 0.0 and bench.engine_cpu(True) == 0.0       # a Python child is not an engine
    finally:
        child.kill()


def test_bench_command_line(capsys):
    with pytest.raises(SystemExit):
        main(["bench", "--help"])
    out = capsys.readouterr().out
    assert all(w in out for w in ("idle", "pass", "tiles"))
    with pytest.raises(SystemExit):
        main(["bench"])                                  # a measurement is required
