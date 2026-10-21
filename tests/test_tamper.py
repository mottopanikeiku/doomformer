"""Negative controls: a tampered map or readout must fail its check, for the intended reason.

Each tamper is applied to a real compiled map after the integer reference is fixed: a linedef weight in the TEXTMAP
(MLP and one-map transformer) and a readout page that returns a Python-written input port instead of the output.
Untampered twins are the positive controls.
"""
from __future__ import annotations

import re
import struct
from pathlib import Path

import numpy as np
import pytest

import doomformer as df
from doomformer import pure
from doomformer.intref import forward
from doomformer.onemap import tiny_model

LIGHT_RAISE = 110        # Light_RaiseByValue(tag, amount): a positive weight on a light accumulator
MLP_NET = df.MLP([[[5, -3], [2, 7]]], [[1, -2]], [[1]])     # signed input, one threshold: one doll per sign


def lumps(wad: Path) -> list[tuple[str, bytes]]:
    data = wad.read_bytes()
    n, directory = struct.unpack("<ii", data[4:12])
    out = []
    for i in range(n):
        offset, size, name = struct.unpack("<ii8s", data[directory + 16 * i:directory + 16 * (i + 1)])
        out.append((name.rstrip(b"\0").decode(), data[offset:offset + size]))
    return out


def retarget_lines(wad: Path, tag: int, old: int, new: int) -> int:
    """Light_RaiseByValue(tag, old) -> (tag, new) on every such linedef of the WAD's TEXTMAP; returns how many."""
    pattern = re.compile(rf"(special = {LIGHT_RAISE}; arg0 = {tag}; arg1 = ){old};")
    rewritten, data, directory = [], b"", b""
    count = 0
    for name, blob in lumps(wad):
        if name == "TEXTMAP":
            text, count = pattern.subn(rf"\g<1>{new};", blob.decode())
            blob = text.encode()
        rewritten.append((name, blob))
    for name, blob in rewritten:
        directory += struct.pack("<ii8s", 12 + len(data), len(blob), name.encode())
        data += blob
    wad.write_bytes(struct.pack("<4sii", b"PWAD", len(rewritten), 12 + len(data)) + data + directory)
    return count


def light_tag(meta, name: str) -> int:
    return int(re.fullmatch(r"GetSectorLightLevel\((\d+)\)", meta.layout["read_expressions"][name]).group(1))


def test_e1_one_mlp_linedef_weight(tmp_path):
    """y0's +5 line from the x0 >= 1 doll becomes +6: exactly the inputs whose doll crosses it are off by one."""
    crossing, silent = ([1, 0], [2, -3], [1, 1]), ([0, 0], [-1, 3], [-2, -1])
    with df.compile(MLP_NET, tmp_path / "clean.wad") as brain:
        for x in crossing + silent:
            np.testing.assert_array_equal(brain(x), MLP_NET.reference(x)["y"])
    wad = tmp_path / "tampered.wad"
    with df.compile(MLP_NET, wad) as brain:
        assert retarget_lines(wad, light_tag(brain.meta, "y0"), 5, 6) == 1
        for x in silent:
            np.testing.assert_array_equal(brain(x), MLP_NET.reference(x)["y"])
        for x in crossing:
            y0 = int(MLP_NET.reference(x)["y"][0])
            with pytest.raises(df.MismatchError) as err:
                brain(x)
            assert err.value.mismatches == [("y0", y0 + 1, y0)]


def test_e3_readout_page_returns_the_input_port(tmp_path, monkeypatch):
    """A readout page that reports the Python-written x0 gate floor instead of y0's light is caught."""
    x = [4, -1]
    y0 = int(MLP_NET.reference(x)["y"][0])
    with df.compile(MLP_NET, tmp_path / "clean.wad") as brain:
        np.testing.assert_array_equal(brain(x), MLP_NET.reference(x)["y"])
        read = f"= {brain.meta.layout['read_expressions']['y0']};"
        echo = f"= (GetSectorFloorZ({brain.meta.ports['floor:x0']}, 0, 0) >> 16);"
    compile_acs = pure.compile_acs

    def tampered(acs):
        assert acs.count(read) == 1
        return compile_acs(acs.replace(read, echo))

    monkeypatch.setattr(pure, "compile_acs", tampered)
    with df.compile(MLP_NET, tmp_path / "echo.wad") as brain:
        with pytest.raises(df.MismatchError) as err:
            brain(x)
    assert x[0] != y0 and err.value.mismatches == [("y0", x[0], y0)]


def test_one_map_transformer_linedef_weight(tmp_path):
    """Unembedding weight Wun[1, 3] = 5 (unique in its row) raised to 6 on both of its positive-doll lines into
    logits:2:1: only token sequences whose final residual x2[2, 3] is positive (a doll crosses) mismatch, by it."""
    model = tiny_model()
    model.p["Wun"][1, 3] = 5
    crossing, silent = [0, 1, 0], [0, 0, 0]
    activation = {}
    for tokens in (crossing, silent):
        trace = {}
        forward(model, tokens, trace)
        activation[tuple(tokens)] = int(trace["L0.x2"][2, 3])
    assert activation == {tuple(crossing): 1, tuple(silent): -2}
    wad = tmp_path / "transformer.wad"
    with df.compile(model, wad) as brain:
        assert retarget_lines(wad, light_tag(brain.meta, "logits:2:1"), 5, 6) == 2      # thresholds 1 and 2
        np.testing.assert_array_equal(brain.logits(silent), forward(model, silent))
        reference = int(forward(model, crossing)[2, 1])
        with pytest.raises(df.MismatchError) as err:
            brain.logits(crossing)
    assert err.value.mismatches == [("logits:2:1", reference + 1, reference)]
