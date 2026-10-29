"""The museum (doomformer/museum.py): the hand-written network reads every digit, the rewiring switch is exactly one
weight change, the build reproduces the committed hashes, and in a headless game the 7 -> MUTATE -> RESTORE tour reads
7, 1, 7 with every value equal to the NumPy reference. At most one game at a time."""
from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np

from doomformer import museum as mu
from doomformer.extract import diff_netlists, expected_netlist, extract

RECORD = Path(__file__).resolve().parents[1] / "models" / "museum"


def test_the_network_reads_every_digit_and_the_flip_turns_7_into_1():
    for d, bits in enumerate(mu.SEGMENT_BITS):
        r = mu.reference(bits)
        assert r["digit"] == d and r["scores"][d] == 7 and max(r["scores"][:d] + r["scores"][d + 1:]) < 7
    seven, flipped = mu.reference(mu.SEGMENT_BITS[7]), mu.reference(mu.SEGMENT_BITS[7], flip=True)
    assert flipped["digit"] == 1 and flipped["scores"][7] == seven["scores"][7] - 2 == 5 and flipped["scores"][1] == 6
    w, w2 = mu.score_net().weights[0], mu.score_net(flip=True).weights[0]
    assert (w != w2).sum() == 1 and w[7, 0] == 1 and w2[7, 0] == -1      # one weight, segment a -> digit 7


def test_reference_record_matches_the_code():
    table = mu.reference_table()
    assert len(table["rows"]) == 256
    assert (RECORD / "reference.json").read_text() == json.dumps(table, indent=1) + "\n"


def test_selftest_expected_lines_match_the_code():
    """models/museum/selftest_expected.txt (what `museum export` writes, and what ViZDoom logged) is the NumPy reference."""
    want = (RECORD / "selftest_expected.txt").read_text().splitlines()
    assert mu.expected_lines(sorted(mu.reader().exposed_names())) == want and len(want) == 12


def test_build_reproduces_the_committed_hashes_and_the_flip_is_one_weight(tmp_path):
    """Built from scratch, the four files equal models/museum/SHA256SUMS. Statically (no game): the map's netlist is
    the network's, and with the two id'd lines rewired as MUTATE does it is the netlist of the network with w_7a = -1."""
    r = mu.build(tmp_path)
    want = dict(line.split()[::-1] for line in (RECORD / "SHA256SUMS").read_text().splitlines())
    assert r["sha256"] == want
    wad = tmp_path / "museum.wad"
    assert diff_netlists(extract(wad), expected_netlist(mu.reader())) == []
    data = wad.read_bytes()
    spec = mu.flip_specials(mu.plan())
    lines = sorted(re.findall(rb"linedef \{ id = [12]; [^\n]*", data))           # id 1: the light line, 2: the floor
    assert len(lines) == 2
    for line, old, new in zip(lines, spec["original"], spec["flipped"]):
        assert f"special = {old[0]}; arg0 = {old[1]};".encode() in line
        # what SetLineSpecial does in the game; "110"->"111" and "20"->"23" keep every lump offset
        data = data.replace(line, line.replace(f"special = {old[0]};".encode(), f"special = {new[0]};".encode()))
    (tmp_path / "flipped.wad").write_bytes(data)
    got = extract(tmp_path / "flipped.wad")
    assert diff_netlists(got, expected_netlist(mu.reader(flip=True))) == []
    assert diff_netlists(got, expected_netlist(mu.reader())) != []


def test_tour_reads_7_then_1_then_7_exactly(tmp_path):
    """Python presses only the marine's keys (walking onto pads a, b, c; USE on READ, MUTATE, READ, RESTORE, READ);
    each READ is watched through ReadoutOnly and every value compared with the network as wired at that moment."""
    with mu.Museum(workdir=tmp_path) as m:
        reads = m.tour()
        status = m.status()
    assert [r.digit for r in reads] == [7, 1, 7] and [r.flip for r in reads] == [False, True, False]
    assert all(r.bits == mu.SEGMENT_BITS[7] and r.checked == 148 for r in reads)
    assert reads[0].scores == reads[2].scores and np.subtract(reads[0].scores, reads[1].scores).tolist() == [
        0, 0, 0, 0, 0, 0, 0, 2, 0, 0]
    assert all(r.audit["non_readout_commands"] == 0 for r in reads)
    assert status["mutations"] == 2 and status["flip"] == 0 and status["reads"] == 3
