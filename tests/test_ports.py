"""The Python-free hello brain and the runner of the other-ports workflow (docs/engine_limits.md, "Other ports")."""
import os
import subprocess
import sys
from pathlib import Path

import pytest

from doomformer import ports

ROOT = Path(__file__).resolve().parent.parent
RUNNER = ROOT / ".github" / "scripts" / "ports_run.py"


def _fact_lines(wad: Path) -> list[str]:
    r = subprocess.run([sys.executable, "-m", "doomformer.facts", "--run-wad", str(wad)], capture_output=True, text=True,
                       timeout=120, env=dict(os.environ, OMP_NUM_THREADS="1"))
    return [ln.strip() for ln in (r.stdout + r.stderr).splitlines() if ln.startswith(ports.LOG_TAG + " ")]


@pytest.mark.parametrize("low_tags", [False, True])
def test_hello_driver_log_equals_numpy_reference(tmp_path, low_tags):
    """ViZDoom runs the driver script and logs every readout; the lines equal the NumPy reference byte for byte."""
    expected = ports.build_hello(tmp_path, low_tags=low_tags)
    assert len(expected) == ports.RUNS and all(len(x.split()) == 28 for x in expected)
    assert _fact_lines(tmp_path / "hello.wad") == expected
    assert (tmp_path / "expected.txt").read_text().splitlines() == expected


def test_hello_expected_is_the_hello_command_inputs():
    net, inputs = ports.hello_net()
    assert (net.n_in, net.n_out) == (4, 3) and len(inputs) == 5
    assert [int(v) for v in inputs[0]] == [-71, -82, 64, 37]       # `doomformer hello` input 1


@pytest.mark.skipif(not RUNNER.exists(), reason="the repository's .github/scripts is not part of an installed package")
def test_runner_diffs_fact_lines(tmp_path):
    """The runner passes a case whose engine prints the expected line, flags a different one, and ends the engine after
    the line (the fake engine would sleep for 60 s)."""
    wads = tmp_path / "wads"
    wads.mkdir()
    (wads / "a.wad").write_bytes(b"")
    (wads / "b.wad").write_bytes(b"")
    (wads / "expected.txt").write_text("FACT a x=1\nFACT b x=2\n")
    engine = tmp_path / "engine.sh"
    engine.write_text('#!/bin/bash\nc=$(basename "$1" .wad)\n'
                      'if [ "$c" = a ]; then echo "FACT a x=1"; else echo "FACT b x=3"; fi\nsleep 60\n')
    engine.chmod(0o755)
    r = subprocess.run([sys.executable, str(RUNNER), "--name", "fake", "--wads", str(wads), "--iwad", "none",
                        "--out", str(tmp_path / "out"), "--cmd", f"{engine} {{wad}}", "--no-xvfb", "--timeout", "30"],
                       capture_output=True, text=True, timeout=60)
    assert r.returncode == 0, r.stderr
    assert "1 PASS, 1 DIFF, 0 ERROR" in r.stdout
    assert (tmp_path / "out" / "observed.txt").read_text() == "FACT a x=1\nFACT b x=3\n"
