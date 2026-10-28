"""Static extraction (doomformer/extract.py): the net is read back from the compiled .wad alone. No DOOM game starts.

Positive controls: extract(compile(net)).equals(net) on random nets at every schedule level, the recovered MLP, a map
copied without its .json. Negative controls: one changed weight line, sign, retargeted line, gate height, start
value (models/extract/controls.py) differ from the net in exactly the changed place; a broken encoding convention
raises ExtractError; the CLI exits 1 on a difference and prints the first differing weight.
"""
from __future__ import annotations

import importlib.util
import io
import re
import shutil
import struct
from contextlib import redirect_stdout
from pathlib import Path

import numpy as np
import pytest

import doomformer as df
from doomformer import cli, extract as ex, fuzz
from doomformer.kit import MLP
from doomformer.mutate import read_lumps, write_lumps

ROOT = Path(__file__).resolve().parent.parent


def fuzz_net(index: int, seed: int = 0) -> MLP:
    return fuzz.net_of("mlp", fuzz.gen_case("mlp", seed, index)["spec"])


def build(net: MLP, path: Path, opt: int = 0) -> Path:
    df.compile(net, path, mode="map", limits=False, opt=opt).close()
    return path


def load_controls():
    spec = importlib.util.spec_from_file_location("extract_controls", ROOT / "models" / "extract" / "controls.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.mark.parametrize("opt", [0, 1, 2, 3])
def test_random_nets_are_recovered_exactly(tmp_path, opt):
    n = 0
    for i in range(12):
        case = fuzz.gen_case("mlp", 0, i)
        if case["planned"] == "unsupported":
            continue
        net = fuzz.net_of("mlp", case["spec"])
        nl = ex.extract(build(net, tmp_path / f"n{i}.wad", opt))
        assert nl.diff(net) == [] and nl.findings == [], (i, nl.diff(net)[:2], nl.findings[:2])
        n += 1
    assert n >= 10


def test_reads_only_the_wad(tmp_path):
    """The sidecar .json of `compile` is removed and the .wad copied elsewhere: the result is the same."""
    net = fuzz_net(10)
    wad = build(net, tmp_path / "a" / "net.wad") if (tmp_path / "a").mkdir() is None else None
    lone = tmp_path / "b" / "lone.wad"
    lone.parent.mkdir()
    shutil.copyfile(wad, lone)
    assert not lone.with_suffix(".json").exists()
    nl = ex.extract(lone)
    assert nl.equals(net)


def test_recovered_mlp_has_the_nets_parameters(tmp_path):
    """to_mlp(): weights, biases, thresholds and the signed or relu mode of a net without parts a map cannot show."""
    for index in (14, 16, 17, 19):                                  # unsigned and signed, 1 to 3 layers
        net = fuzz_net(index)
        if any(not W[:, i].any() for W in net.weights[:1] for i in range(W.shape[1])) or not all(
                W.any(axis=0).all() for W in net.weights[1:]):
            continue
        rec = ex.extract(build(net, tmp_path / f"n{index}.wad")).to_mlp()
        assert rec.signed_input == net.signed_input
        for a, b in zip(rec.weights + rec.biases + rec.thresholds, net.weights + net.biases + net.thresholds):
            assert np.array_equal(a, b)


def test_parts_the_map_cannot_show(tmp_path):
    """An input with an all-zero column has no lane; an unread unit keeps its lights. `equals` still holds."""
    net = MLP([np.array([[2, 0, -1], [3, 0, 4]]), np.array([[1, 0], [0, 0]])], [np.array([5, -6]), np.array([7, 1])],
              [np.array([1, 4]), np.array([2])], signed_input=True)
    nl = ex.extract(build(net, tmp_path / "z.wad"))
    assert nl.equals(net)
    assert ex.shape(nl) == [2, 1, 2]                                # input column 1 and the unread unit 1 are not rows
    assert len(nl.layers[0].dead) == 1                              # unit 1 has a nonzero row and no reader
    assert nl.unattached == []


@pytest.fixture(scope="module")
def controls_run(tmp_path_factory):
    work = tmp_path_factory.mktemp("controls")
    mod = load_controls()
    net = fuzz_net(10)
    wad = build(net, work / "clean.wad")
    return net, mod.controls(net, wad, work), work


def test_negative_controls_differ_exactly_where_edited(controls_run):
    net, recs, _ = controls_run
    names = {r["name"] for r in recs}
    assert {"weight_plus1", "sign", "retarget", "gate_height", "pad_height", "output_bias", "hidden_plane_line",
            "hidden_twin_line", "hidden_plane_start"} <= names
    for r in recs:
        assert r["ok"], r
    by = {r["name"]: r for r in recs}
    assert by["weight_plus1"]["n"] == 1 and by["sign"]["n"] == 1 and by["retarget"]["n"] == 2
    m = re.search(r"layer 3, row \d+, column \d+, lane \d+: weight map \+(\d+), net \+(\d+)$", by["weight_plus1"]["differences"][0])
    assert m and int(m.group(1)) == int(m.group(2)) + 1
    assert "threshold" in by["gate_height"]["differences"][0]
    assert by["gate_height"]["findings"] == []                     # a consistent gate move only the comparison sees


def test_cli_verify_extract(controls_run, tmp_path):
    net, recs, work = controls_run
    npz = tmp_path / "net.npz"
    net.save(npz)
    out = io.StringIO()
    with redirect_stdout(out):
        assert cli.main(["verify", "--extract", str(work / "clean.wad"), "--net", str(npz)]) == 0
    assert "EQUAL" in out.getvalue()
    out = io.StringIO()
    with redirect_stdout(out):
        assert cli.main(["verify", "--extract", str(work / "weight_plus1.wad"), str(npz)]) == 1
    text = out.getvalue()
    assert "DIFFERENT" in text and re.search(r"layer 3, row \d+, column \d+, lane \d+: weight map \+\d+, net \+\d+", text)
    out = io.StringIO()
    with redirect_stdout(out):
        assert cli.main(["extract", "--dump", str(work / "clean.wad")]) == 0
    dump = out.getvalue()
    assert all(f"layer {l} (" in dump for l in range(4)) and "output layer" in dump


def retag_start_barrier(src: Path, dst: Path):
    lumps = [(n, b.decode().replace("id = 2;", "id = 3;").encode() if n == "TEXTMAP" else b) for n, b in read_lumps(src)]
    write_lumps(dst, lumps)


def test_broken_conventions_fail_loudly(tmp_path):
    net = fuzz_net(6)
    wad = build(net, tmp_path / "net.wad")
    bad = tmp_path / "retag.wad"
    retag_start_barrier(wad, bad)
    with pytest.raises(ex.ExtractError, match="start barrier"):
        ex.extract(bad)
    junk = tmp_path / "junk.wad"
    junk.write_bytes(b"not a wad at all, only text")
    with pytest.raises(ex.ExtractError):
        ex.extract(junk)
    # a lane whose gate sector is not a floor gate (ceiling lowered)
    text = dict(read_lumps(wad))["TEXTMAP"].decode()
    broken = re.sub(r"(heightceiling = )16000(; texturefloor = \"FLAT23\"; textureceiling = \"F_SKY1\"; lightlevel = 224; id = 1000;)",
                    r"\g<1>15000\g<2>", text, count=1)
    assert broken != text
    ceil = tmp_path / "ceil.wad"
    write_lumps(ceil, [(n, broken.encode() if n == "TEXTMAP" else b) for n, b in read_lumps(wad)])
    with pytest.raises(ex.ExtractError, match="floor gates"):
        ex.extract(ceil)


def test_reset_script_is_read(tmp_path):
    """The reset script restores every written light to its start level: a changed literal in the BEHAVIOR lump is a
    finding even though the TEXTMAP is untouched."""
    net = MLP([np.array([[2, -1], [0, 3]]), np.array([[1, -2]])], [np.array([3, 4]), np.array([5])],
              [np.array([1, 4]), np.array([2])], signed_input=True)
    wad = build(net, tmp_path / "net.wad")
    assert ex.extract(wad).equals(net)
    lumps = read_lumps(wad)
    beh = dict(lumps)["BEHAVIOR"]
    offset = ex.acs_scripts(beh)[3][0]
    calls = ex.run_script(beh, offset)
    tag, level = next(a for sp, a in calls if sp == ex.LIGHT_CHANGE_TO and a[1] == 5)
    pattern = b"\x03" + struct.pack("<i", tag) + b"\xa7\x05\x05\x70"
    assert beh.count(pattern) == 1
    patched = beh.replace(pattern, pattern.replace(b"\xa7\x05", b"\xa7\x06"))
    bad = tmp_path / "bad.wad"
    write_lumps(bad, [(n, patched if n == "BEHAVIOR" else b) for n, b in lumps])
    nl = ex.extract(bad)
    assert nl.diff(net) == []
    assert any("reset script restores light" in f for f in nl.findings)
    assert not nl.equals(net)


def test_opt3_copy_bus_is_decoded(tmp_path):
    net = fuzz_net(10)
    nl = ex.extract(build(net, tmp_path / "n3.wad", opt=3))
    assert nl.equals(net)
    assert "copy bus" in nl.checks
    assert "copy bus" not in ex.extract(build(net, tmp_path / "n0.wad", opt=0)).checks
