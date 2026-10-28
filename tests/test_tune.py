"""doomformer tune: the committed timing table (doomformer/tuning.json), the Timing it ships, and one re-measured row."""
from dataclasses import asdict

from doomformer import tune
from doomformer.pure import Timing


def _passes(row: dict) -> bool:
    return all(cell["fail"] == 0 for cell in row["nets"].values())


def test_table_schema_and_edges():
    data = tune.load()
    assert data["format"] == "doomformer-tuning-1" and data["provisional"] is False
    assert data["commands"] and data["date"] and data["vizdoom"] and data["machine_load_note"]
    assert set(data["shipped"]) == set(tune.SHIPPED_KEYS)
    plane, copy = tune.shipped_dicts(data)
    assert data["shipped"] == plane and data["shipped_copy"] == copy
    for mode in tune.MODES.values():
        measured = set(data.get(mode.table, {}))
        assert measured | set(data["not_measured"][mode.shipped]) == set(mode.constants)
        shipped = Timing(**{**plane, **(copy if mode.name == "copy" else {})})
        for const in data["not_measured"][mode.shipped]:          # unmeasured constants ship their default
            assert getattr(shipped, const) == getattr(Timing(), const)
        for const, e in data.get(mode.table, {}).items():
            assert e["default"] == getattr(Timing(), const)
            is_copy = mode.name == "copy"
            # the rule's value, or a larger one with its reason (clock_margin: the static checker's bound)
            assert e.get("shipped_by_rule", e["shipped"]) == tune.shipped_value(const, e["edge"], is_copy)
            assert e["shipped"] == e.get("shipped_by_rule", e["shipped"]) or (
                e["shipped"] > e["shipped_by_rule"] and e["shipped_reason"])
            assert e["multiple"] == (tune.COPY_RULE[const][0] if is_copy and const in tune.COPY_RULE
                                     else tune.MULTIPLE[const])
            assert e["shipped"] % 8 == 0 and e["edge_tics"] == e["edge"] / tune.UNITS_PER_TIC
            rows = {r["value"]: r for r in e["table"]}
            assert e.get("scan_top", e["default"]) in rows and e["edge"] in rows
            assert e["shipped"] in rows or "shipped_reason" in e          # an override is checked by the joint run
            # the edge: every tested value at or above it passes on every net, the largest failing value fails
            assert all(_passes(r) for v, r in rows.items() if v >= e["edge"])
            if e["largest_failing"] is None:
                assert e["edge"] == e["lowest_tested"] == min(rows)
            else:
                assert e["largest_failing"] < e["edge"] and not _passes(rows[e["largest_failing"]])
                assert max(v for v, r in rows.items() if not _passes(r)) == e["largest_failing"]
            for r in rows.values():
                assert set(r["nets"]) == set(e["nets"])
                for net, cell in r["nets"].items():
                    assert cell["n"] == e["inputs"][net]["inputs"] and 0 <= cell["fail"] <= cell["n"]
            for net in e["nets"]:
                info = e["inputs"][net]
                if net in tune.NETS_ALL:                     # generated nets; mnist and the trained models are fixed
                    assert info["seed"] == tune.SCAN_SEED + tune.NETS_ALL.index(net) and info["random"] >= 6
    joint = data["joint"]
    final = asdict(tune.mode_timing(data, tune.MODES["plane"]))
    if "joint_note" in data:                                       # the joint run predates the clock_margin override
        final["clock_margin"] = joint["timing"]["clock_margin"]
    assert joint["failures"] == 0 and joint["timing"] == final
    assert all(cell["fail"] == 0 for cell in joint["nets"].values())


def test_tuned_timing_loads():
    data = tune.load()
    assert Timing.tuned() == Timing(**data["shipped"])
    assert Timing.tuned(copy=True) == Timing(**{**data["shipped"], **data["shipped_copy"]})


def test_remeasure_hello_serial(tmp_path):
    """hello at the shipped serial again with its first 6 scan inputs, in this process (one game at a time): every
    exposed value exact; at a value where the table records hello failing on one of those inputs, the same inputs fail
    again (runs are deterministic)."""
    entry = tune.load()["constants"]["serial"]
    cells = {r["value"]: r["nets"]["hello"] for r in entry["table"]}
    failing = [v for v, c in cells.items() if any(i < 6 for i, *_ in c["bad"])]
    values = [entry["shipped"]] + ([max(failing)] if failing else [])
    refs = tune.reference(["hello"], 6, tune.SCAN_SEED)
    refs["hello"].inputs = refs["hello"].inputs[:6]
    with tune.Bench(refs, 0, tmp_path, tune.MODES["plane"].passes) as bench:
        got = bench.run([("hello", Timing(serial=v)) for v in values])
    assert got[0]["fail"] == 0
    for v, r in zip(values[1:], got[1:]):
        assert sorted(i for i, *_ in r["bad"]) == sorted(i for i, *_ in cells[v]["bad"] if i < 6), v


def test_opt0_hashes_file_and_tiny_build():
    """doomformer/opt0_hashes.json holds the sha256 of the opt 0 .wad and .json of the six models; the tiny model's
    (no game, under a second) is rebuilt and compared. `doomformer tune --hashes` checks all six."""
    data = tune.json.loads(tune.HASH_FILE.read_text())
    assert set(data["models"]) == set(tune.HASH_MODELS)
    for h in data["models"].values():
        assert len(h["wad"]) == len(h["json"]) == 64
    got = tune.opt0_hashes(("tiny",), echo=lambda s: None)["tiny"]
    assert got["wad"] == data["models"]["tiny"]["wad"] and got["json"] == data["models"]["tiny"]["json"]
