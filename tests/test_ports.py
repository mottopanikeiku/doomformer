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


def _run_logger(wad: Path, tics: int) -> list[str]:
    r = subprocess.run([sys.executable, "-m", "doomformer.ports_machine", "--run-vizdoom", str(wad), str(tics)],
                       capture_output=True, text=True, timeout=300, env=dict(os.environ, OMP_NUM_THREADS="1"))
    return [ln.strip() for ln in (r.stdout + r.stderr).splitlines() if ln.startswith("FACT ")]


def test_machine_export_log_equals_interpreter(tmp_path):
    """`machine add 1 1 --clock dolls --export`: ViZDoom runs the doll-clock map with the logger script and prints the
    interpreter's 6 lines (pc and 28 values per step) byte for byte."""
    from doomformer import ports_machine
    r = ports_machine.export(tmp_path, "add", (1, 1))
    assert r["steps"] == 5 and r["per_step"] == 28 and r["halted"]
    expected = (tmp_path / "expected.txt").read_text().splitlines()
    assert len(expected) == 6 and expected[0].startswith("FACT machine step=0 ")
    assert _run_logger(tmp_path / "machine.wad", 6000) == expected


def test_same_tic_export_matches_vizdoom(tmp_path):
    from doomformer import dollclock, ports_machine
    lines = ports_machine.export_same_tic(tmp_path)
    assert lines == ["FACT same_tic_cell4 far=14", "FACT same_tic_cell16 far=0", "FACT same_tic_cell4_tags15 far=14",
                     "FACT same_tic_cell16_tags15 far=0"] and dollclock.CELL == 16
    for line in lines:
        assert _run_logger(tmp_path / f"{line.split()[1]}.wad", 700) == [line]


def test_low_tags_export_is_the_same_trace(tmp_path):
    """The tags-below-32,768 build of a machine logs the same lines in ViZDoom and leaves the default build alone."""
    from doomformer import dollclock, ports_machine, pure
    ports_machine.export(tmp_path / "a", "add", (1, 1), low_tags=True)
    assert (pure.STAGE_TAG0, dollclock.LOOP_BARRIER) == (60000, 90000)
    expected = (tmp_path / "a" / "expected.txt").read_text().splitlines()
    assert _run_logger(tmp_path / "a" / "machine.wad", 6000) == expected
    ports_machine.export(tmp_path / "b", "add", (1, 1))
    assert (tmp_path / "b" / "expected.txt").read_text().splitlines() == expected


@pytest.mark.skipif(not os.environ.get("DOOMFORMER_HEAVY"), reason="the SUBLEQ map takes about 90 s of ViZDoom time")
def test_subleq_export_log_equals_interpreter(tmp_path):
    from doomformer import ports_machine
    ports_machine.export(tmp_path, "subleq", ("add", 3, 4))
    expected = (tmp_path / "expected.txt").read_text().splitlines()
    assert _run_logger(tmp_path / "machine.wad", 20000) == expected
