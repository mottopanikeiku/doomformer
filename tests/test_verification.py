"""The verification tools: fuzzing finds planted compiler bugs, `verify` proves a small net over its whole domain and
fails a tampered map, and the three holes in the exactness check (dead readout page, dropped readout name,
unchecked transformer ranges) are closed."""
from __future__ import annotations

import numpy as np
import pytest

import doomformer as df
from doomformer import fuzz, verify
from doomformer.intref import forward
from doomformer.kit import limit_sequences
from doomformer.mutate import _run_mutant, acs_mutants, capture_acs, read_lumps, tamper_weight, textmap_mutants
from doomformer.onemap import exposed_names, tiny_model
from doomformer.tile import TileMeta

NET = df.MLP([[[5, -3], [2, 7], [1, 1]], [[1, 2, 3]]], [[1, -2, 0], [5]], [[1, 3], [1, 2]])    # 2 inputs, 2 layers
NET1 = df.MLP([[[5], [2], [-1]], [[1, 2, 3]]], [[1, -2, 0], [5]], [[1, 3], [1, 2]])             # 1 input, 2 layers


# --- fuzz -----------------------------------------------------------------------------------------------------------
def test_fuzz_passes_clean_and_finds_planted_bugs():
    """6 nets per bug (in this process, so the monkeypatched compiler is the one that runs): clean passes, every
    planted lowering bug is a mismatch of exact values."""
    clean = fuzz.fuzz(nets=6, seed=0)
    assert clean["failures"] == [] and clean["passes"] >= 24 and clean["values"] > 200
    for name, edge in (("gate-gt", False), ("w255", True), ("dup-thr", False), ("bias-twin", False)):
        out = fuzz.fuzz(nets=12, seed=0, edge=edge, plant_name=name)
        assert out["mismatches"] >= 1, f"planted bug {name} not found in 12 nets"
        assert all(f["mismatches"] for f in out["failures"] if f["status"] == "mismatch")
    assert fuzz.fuzz(nets=3, seed=0)["failures"] == []             # the plant is undone after the run


def test_fuzz_replay_file_reproduces_a_failure(tmp_path):
    out = fuzz.fuzz(nets=6, seed=0, plant_name="gate-gt")
    assert out["cases"]
    path = tmp_path / "failures.json"
    fuzz.save_replay(path, out["cases"])
    cases = fuzz.load_replay(path)
    assert [c["id"] for c in cases] == [c["id"] for c in out["cases"]]
    assert all(r["status"] == "ok" for r in fuzz.run_cases(cases))                    # the real compiler is exact
    assert all(r["status"] == "mismatch" for r in fuzz.run_cases(cases, plant_name="gate-gt"))
    path.write_text("{}")
    with pytest.raises(ValueError, match="replay"):
        fuzz.load_replay(path)


def test_fuzz_cases_depend_on_seed_and_index_only():
    a = fuzz.gen_case("mlp", 3, 7)
    assert a == fuzz.gen_case("mlp", 3, 7) and a != fuzz.gen_case("mlp", 3, 8) and a != fuzz.gen_case("mlp", 4, 7)
    net = fuzz.net_of("mlp", a["spec"])
    assert fuzz.spec_of(net) == a["spec"]


def test_all_zero_layer_is_unsupported_with_a_clear_message(tmp_path):
    net = df.MLP([[[1, 2]], [[0]]], [[0], [3]], [[1], [1]])
    with pytest.raises(ValueError, match="all zero"):
        df.compile(net, tmp_path / "zero.wad")
    case = dict(id=0, kind="mlp", spec=fuzz.spec_of(net), inputs=[[1, 1]], planned="unsupported")
    res = fuzz.run_case(case, dict(guard=True))
    assert res["status"] == "unsupported" and "all zero" in res["detail"]


# --- the three holes ------------------------------------------------------------------------------------------------
class DeadPage:
    """The engine, except that readout page 0's script command is dropped (a page that never runs)."""

    def __init__(self, game):
        self._game = game

    def send_game_command(self, cmd, *a):
        return None if cmd == "puke -100" else self._game.send_game_command(cmd, *a)

    def __getattr__(self, name):
        return getattr(self._game, name)


def test_dead_readout_page_fails_loudly_on_a_repeated_input(tmp_path):
    """Before: USER1.. kept the last pass's values, so a page that never ran passed when the input repeated."""
    x = [4, -1]
    with df.compile(NET, tmp_path / "guarded.wad", guard=True) as brain:
        np.testing.assert_array_equal(brain(x), NET.reference(x)["y"])
        np.testing.assert_array_equal(brain(x), NET.reference(x)["y"])
        brain.runner.game = DeadPage(brain.runner.game)
        with pytest.raises(RuntimeError, match="did not run"):
            brain(x)
    with df.compile(NET, tmp_path / "plain.wad") as brain:             # the default map is byte-identical to before
        assert "guard_script" not in brain.meta.layout
        assert b"nonce" not in (tmp_path / "plain.wad").read_bytes()


def test_guard_needs_one_map(tmp_path):
    with pytest.raises(ValueError, match="mode='map'"):
        df.compile(NET, tmp_path / "t", mode="tiles", guard=True)


def test_dropping_a_readout_name_fails_instead_of_shrinking_the_check(tmp_path):
    """Before: deleting a name from meta.pages made `checked` one smaller and nothing complained."""
    x = [4, -1]
    with df.compile(NET, tmp_path / "pages.wad") as brain:
        brain(x)
        full = brain.info["checked"]
        assert full == len(NET.exposed_names())
        page = brain.runner.meta.pages[-1]
        dropped = page.pop()                                           # the last name: the others keep their slots
        with pytest.raises(df.ReadoutError, match=dropped.replace("-", r"\-")):
            brain(x)


def _fake_readback(net, tokens) -> dict[str, int]:
    """What a correct map reads back, from intref's trace (no game)."""
    tr = {}
    forward(net, tokens, tr)
    got = {}
    for name in exposed_names(net):
        key, *raw = name.lstrip("-").split(":")
        try:
            v = int(tr[key][tuple(map(int, raw))])
        except (KeyError, IndexError):                                 # a non-causal score, never compared
            v = 0
        got[name] = -v if name.startswith("-") else v
    return got


def test_transformer_readback_set_is_checked(tmp_path):
    m = tiny_model()
    tokens = [0, 1, 2]
    brain = df.compile(m, tmp_path / "t.wad")
    full = _fake_readback(m, tokens)
    brain._run = lambda ports: dict(full)
    np.testing.assert_array_equal(brain.logits(tokens), forward(m, tokens))
    short = dict(full)
    short.pop(next(n for n in short if n.startswith("L0.zq:")))
    brain._run = lambda ports: short
    with pytest.raises(df.ReadoutError, match="missing"):
        brain.logits(tokens)


def test_transformer_ranges_are_checked_at_compile(tmp_path):
    """Before: bun[1] = 40,000 compiled and failed at run time as `DOOM 32767 != reference 40004`."""
    m = tiny_model()
    m.p["bun"] = m.p["bun"].copy()
    m.p["bun"][1] = 40000
    with pytest.raises(df.RangeError, match=r"limits=False"):
        df.compile(m, tmp_path / "bad.wad")
    df.compile(m, tmp_path / "bad.wad", limits=False).close()          # the opt-out still builds
    assert limit_sequences(m) and all(s[0] == 0 or len(s) == m.T for s in limit_sequences(m))


# --- verify ---------------------------------------------------------------------------------------------------------
def test_verify_proves_the_whole_domain_and_fails_a_tampered_map(tmp_path):
    full = verify.verify(NET1, "classes")                              # the whole +-12,000 domain
    assert full["verdict"] == "VERIFIED" and full["classes"] == 5 and full["covered"] == 24001
    assert full["inputs_run"] < 20 and full["values"] > 0
    assert verify.verify(NET1, "all", domain=(-40, 40), workers=1)["verdict"] == "VERIFIED"
    from doomformer.kit import compile
    compile(NET1, tmp_path / "ok.wad").close()
    what = tamper_weight(tmp_path / "ok.wad", tmp_path / "bad.wad", 1)
    assert "5 -> 6" in what
    for mode, dom in (("classes", (-12000, 12000)), ("all", (-40, 40))):
        out = verify.verify(NET1, mode, domain=dom, wad=tmp_path / "bad.wad")
        assert out["verdict"] == "FAILED" and out["first_mismatch"]["values"]
    with pytest.raises(ValueError, match="narrow --domain"):
        verify.verify(NET1, "all", max_inputs=1000)                    # 24,001 inputs: over the budget
    assert verify.verify(NET1, "classes", domain=(-40, 40))["classes"] == 5


def test_input_classes_are_constant_and_maximal():
    from doomformer.intref import adc_relu, adc_signed
    for signed, thr in ((True, [1, 3, 3]), (False, [2, 2, 7]), (True, [12000])):
        cls = verify.input_classes(thr, signed, -12000, 12000)
        f = adc_signed if signed else adc_relu
        xs = np.arange(-12000, 12001)
        a = f(xs, np.array(thr))
        assert cls[0][0] == -12000 and cls[-1][1] == 12000
        for (lo, hi), nxt in zip(cls, cls[1:] + [None]):
            assert (a[lo + 12000:hi + 12001] == a[lo + 12000]).all()          # constant on the class
            if nxt:
                assert a[hi + 12000] != a[nxt[0] + 12000]                     # and it changes at the next one


# --- mutation harness -----------------------------------------------------------------------------------------------
def test_mutants_of_weights_are_killed_and_cosmetic_ones_survive(tmp_path):
    base = tmp_path / "base.wad"
    source = capture_acs(NET, base)
    meta = TileMeta.load(base.with_suffix(".json"))
    text = dict(read_lumps(base))["TEXTMAP"].decode()
    sites = textmap_mutants(text, meta) + acs_mutants(source, start=10 ** 6)
    classes = {m.cls for m in sites}
    assert {"light-weight", "plane-weight", "effect-sign", "effect-target", "sector-light-other", "acs-reset",
            "acs-readout", "acs-readout-shift"} <= classes
    pick = {c: next(m for m in sites if m.cls == c) for c in ("light-weight", "effect-sign", "sector-light-other")}
    spec = fuzz.spec_of(NET)
    inputs = [[1, 1], [4, -2], [0, 0], [3, 3], [-3, 1]]
    verdict = {}
    for c, m in pick.items():
        job = dict(mutant=m.__dict__, spec=spec, base=str(base), source=source, inputs=inputs, max_tics=3000)
        verdict[c] = _run_mutant(job)["killed"]
    assert verdict == {"light-weight": True, "effect-sign": True, "sector-light-other": False}


# --- bugs the fuzzer found --------------------------------------------------------------------------------------
def test_gated_unit_with_a_zero_wdn_column_has_no_stray_port():
    """Found by `fuzz --kind transformer` (seed 0): the gated MLP allocated the u port before it knew a gate would use
    it, the readout then read a sector that does not exist (0 instead of -4)."""
    from doomformer.ops import gated, output_pair
    from doomformer.tile import TileSpec
    spec = TileSpec("t")
    pair = output_pair(spec, "z")
    gated(spec, "g", [1, 2], "u", [1, 3], [(pair, 0)])
    assert spec.ports == [] and spec.blocks == []
    gated(spec, "g", [1, 2], "u", [1, 3], [(pair, 2)])
    assert sorted(spec.ports) == ["ceil:-u", "ceil:u", "floor:-g"]
    assert len(spec.blocks) == 2


@pytest.mark.parametrize("thr, w, b, x", [([5748, 8058], 156, 1617, -12000), ([5740, 11999], 255, 0, 12000)])
def test_mlp_fuzz_scheduling_regressions(tmp_path, thr, w, b, x):
    """Found by `fuzz` (seed 0 net 591, `--edge` seed 1 net 246): a barrier that needs two mover tics left a doll one
    tic late and one plane move was lost (-h1_0 DOOM -255 != reference -510)."""
    net = df.MLP([[[w]], [[3]]], [[b], [0]], [thr, [11999]])
    with df.compile(net, tmp_path / "r.wad") as brain:
        np.testing.assert_array_equal(brain([x]), net.reference([x])["y"])
