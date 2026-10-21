"""The public toolkit (`import doomformer as df`): real headless DOOM games for every brain kind and every df.act
helper, plus the pure-Python contract (validation, quantisation, persistence, plan, play's error path)."""
from __future__ import annotations

import subprocess
from pathlib import Path

import numpy as np
import pytest

import doomformer as df
from doomformer import acc, cli, intref, shipped
from doomformer.onemap import tiny_model
from doomformer.pure import X_ORIGIN

CHECKOUT_MODELS = Path(__file__).resolve().parent.parent / "models"


def random_mlp(seed: int = 0, sizes=(4, 6, 5, 3), signed_input: bool = True) -> df.MLP:
    rng = np.random.default_rng(seed)
    n = len(sizes) - 1
    return df.MLP([rng.integers(-20, 21, (sizes[l + 1], sizes[l])) for l in range(n)],
                  [rng.integers(-30, 31, sizes[l + 1]) for l in range(n)],
                  [np.cumsum(rng.integers(15, 40, 3)) for _ in range(n)], signed_input=signed_input)


def test_three_layer_mlp_exact_including_hidden(tmp_path):
    net = random_mlp()
    rng = np.random.default_rng(1)
    planned = df.plan(net)
    with df.compile(net, tmp_path / "mlp.wad") as brain:
        assert (brain.info["dolls"], brain.info["sectors"]) == (planned["dolls"], planned["sectors"])
        for _ in range(4):
            x = rng.integers(-150, 151, net.n_in)
            ref = net.reference(x)
            np.testing.assert_array_equal(brain(x), ref["y"])
            # every hidden unit twice: the light h and the gate floor -h the next layer reads
            assert brain.info["checked"] == net.n_out + 2 * sum(len(h) for h in ref["hidden"])
            raw = brain.run(x)
            for l, h in enumerate(ref["hidden"], start=1):
                assert [raw[f"h{l}_{j}"] for j in range(len(h))] == h.tolist()
                assert [-raw[f"-h{l}_{j}"] for j in range(len(h))] == h.tolist()
        assert brain.info["tics"] > 0 and brain.info["runs"] == 8


def test_mismatch_is_raised_when_the_reference_is_altered(tmp_path):
    net = random_mlp(seed=3, sizes=(3, 4, 2))
    x = np.array([90, -60, 40])
    with df.compile(net, tmp_path / "mlp.wad") as brain:
        brain(x)
        net.biases[0][1] += 1                       # the map still adds the old bias to h1_1
        with pytest.raises(df.MismatchError) as err:
            brain(x)
        assert ("h1_1", int(net.reference(x)["hidden"][0][1]) - 1, int(net.reference(x)["hidden"][0][1])) \
            in err.value.mismatches


def float_model(seed: int = 5):
    rng = np.random.default_rng(seed)
    layers = [(rng.normal(0, 0.6, (6, 3)), rng.normal(0, 0.2, 6)), (rng.normal(0, 0.6, (3, 6)), rng.normal(0, 0.2, 3))]
    return layers, rng.normal(0, 1.0, (200, 3))


def test_from_numpy_round_trip(tmp_path):
    layers, calib = float_model()
    net = df.MLP.from_numpy(layers, calib, levels=8)
    assert net.signed_input and net.n_layers == 2
    rep = net.quantization_report(calib)
    assert rep["in_range"] == len(calib) and rep["argmax_agreement"] >= 0.8
    assert rep["max_abs_error"] < 0.5 * rep["float_max_abs"]
    net.save(tmp_path / "q.npz")
    back = df.MLP.load(tmp_path / "q.npz")
    for a, b in zip(net.weights + net.biases + net.thresholds, back.weights + back.biases + back.thresholds):
        np.testing.assert_array_equal(a, b)
    np.testing.assert_array_equal(back.in_scale, net.in_scale)
    assert back.out_scale == net.out_scale and back.quantization_report(calib) == rep
    x = calib[:3]
    with df.compile(back, tmp_path / "q.wad") as brain:
        y = brain(back.encode(x))                   # three passes, each checked against the integer reference
    np.testing.assert_array_equal(y, back.reference(back.encode(x))["y"])
    assert np.abs(back.decode(y) - back.float_forward(x)).max() <= rep["max_abs_error"] + 1e-12


def test_from_torch_matches_from_numpy():
    torch = pytest.importorskip("torch")
    layers, calib = float_model()
    model = torch.nn.Sequential(torch.nn.Linear(3, 6), torch.nn.ReLU(), torch.nn.Linear(6, 3))
    with torch.no_grad():
        for lin, (W, b) in zip((model[0], model[2]), layers):
            lin.weight.copy_(torch.tensor(W))
            lin.bias.copy_(torch.tensor(b))
    a = df.MLP.from_torch(model, torch.tensor(calib, dtype=torch.float64))
    b = df.MLP.from_numpy([(m.weight.detach().numpy().astype(np.float64), m.bias.detach().numpy().astype(np.float64))
                           for m in (model[0], model[2])], calib)
    for u, v in zip(a.weights + a.biases + a.thresholds, b.weights + b.biases + b.thresholds):
        np.testing.assert_array_equal(u, v)
    with pytest.raises(ValueError):
        df.MLP.from_torch(torch.nn.Sequential(torch.nn.Linear(3, 2), torch.nn.ReLU()), calib)


PLAYER_TID, DOOR_TAG, LAMP_TAG = 900, 10, 11


def act_world(m) -> str:
    """A room with the player, a closed door in its east wall (another room behind it) and a lamp closet."""
    m.rect(-1024, -3072, 0, -2048, floor=0, ceil=128, light=160)
    m.rect(0, -2600, 16, -2500, floor=0, ceil=0, tag=DOOR_TAG, wall="DOOR3")
    m.rect(16, -3072, 512, -2048, floor=0, ceil=128, light=160)
    m.rect(-2048, -3072, -1984, -3008, floor=0, ceil=64, light=100, tag=LAMP_TAG)
    m.thing(-768, -2560, 1)
    return f"""
int kit_count;
script 200 ENTER {{ Thing_ChangeTID(0, {PLAYER_TID}); }}
script 201 (int k) {{ kit_count += k; }}
script 202 (void) {{ o1 = GetSectorCeilingZ({DOOR_TAG}, 0, 0) >> 16; o2 = GetSectorLightLevel({LAMP_TAG});
  o3 = kit_count; o4 = GetActorX({PLAYER_TID}) >> 16; o5 = GetActorY({PLAYER_TID}) >> 16; }}
"""


def test_act_helpers_fire_in_doom(tmp_path):
    net = df.MLP([[[2, -3, 1]]], [[5]], [[10, 20, 30]])          # y0 = 5 + [2, -3, 1] @ adc_signed(x)
    acts = {"y0": [df.act.door_open(DOOR_TAG, 64), df.act.light(LAMP_TAG, 200),
                   df.act.thrust_player(0, 8, PLAYER_TID), df.act.script(201, 1)]}
    with df.compile(net, tmp_path / "act.wad", world=act_world, actions=acts) as brain:
        door0, lamp0, count0, px0, py0 = brain.run_script(202, read=5)
        assert (door0, lamp0, count0, px0, py0) == (0, 100, 0, -768, -2560)
        x = np.array([25, -15, 5])                                 # a0 = [2, -1, 0]: 2 + 1 positive units
        assert brain(x).tolist() == [12]
        door, lamp, count, px, py = brain.run_script(202, read=5)
        assert count == 3                                          # once per positive contribution unit
        assert door == 124 and lamp == 200                         # door opened to neighbour ceiling - 4, lamp set
        assert px > px0 + 64 and py == py0                         # three east thrusts moved the player east
        assert brain(np.array([-25, 15, -5])).tolist() == [-2]     # no positive unit: nothing fires
        assert brain.run_script(202, read=5)[2] == 3


def test_world_hook_rejects_broken_worlds(tmp_path):
    net = random_mlp(sizes=(2, 2))

    def no_player(m):
        m.rect(-1024, -3072, 0, -2048, floor=0, ceil=128)
        return ""

    def on_the_brain(m):
        m.rect(X_ORIGIN - 100, -100, X_ORIGIN + 1000, 400, floor=0, ceil=128)   # brain lanes start at (X_ORIGIN, 0)
        m.thing(X_ORIGIN + 500, 0, 1)
        return ""

    with pytest.raises(ValueError, match="player start"):
        df.compile(net, tmp_path / "a.wad", world=no_player)
    with pytest.raises(ValueError, match="overlaps"):
        df.compile(net, tmp_path / "b.wad", world=on_the_brain)
    with pytest.raises(KeyError):
        df.compile(net, tmp_path / "c.wad", actions={"y7": [df.act.light(5, 1)]})
    with pytest.raises(ValueError):
        df.act.door_open(1000)                                     # a brain port tag
    with pytest.raises(ValueError):
        df.act.thrust_player(0, 5, 0)                              # tid 0 would push the voodoo doll


def test_spiking_step_exact(tmp_path):
    rng = np.random.default_rng(2)
    n, m = 8, 3
    W = np.where(rng.random((n, n)) < 0.3, rng.integers(-12, 16, (n, n)), 0)
    np.fill_diagonal(W, 0)
    U = np.where(rng.random((n, m)) < 0.5, rng.integers(4, 18, (n, m)), 0)
    net = df.SpikingNet([f"n{i}" for i in range(n)], [f"in{k}" for k in range(m)], W, U, theta=12, leak=1)
    X = (rng.random((8, m)) < 0.5).astype(np.int64)
    V, S = net.run(X)
    with df.compile(net, tmp_path / "spikes.wad") as brain:
        for t in range(len(X)):
            v, s = brain.step(X[t])
            np.testing.assert_array_equal(s, S[t])
            np.testing.assert_array_equal(v, V[t + 1])
    assert S.sum() > 0


def test_tiny_transformer_exact(tmp_path):
    model = tiny_model()
    with df.compile(model, tmp_path / "tf.wad") as brain:
        for tokens in ([0, 1, 2], [0, 3]):
            np.testing.assert_array_equal(brain.logits(tokens), intref.forward(model, tokens))
            assert brain.info["checked"] > 0
        assert brain.generate("A") == intref.generate(model, "A")


def test_transformer_load_and_plan():
    m = df.Transformer.load("doomnames_onemap")
    p = df.plan(m)
    assert p["fits"] and p["dolls"] > 1000 and p["sectors"] > 10000


def test_shipped_models_load_by_name_or_path(tmp_path, monkeypatch):
    assert "doomnames_onemap" in df.available_models() and "doomnames" in df.available_models()
    by_name = df.Transformer.load("doomnames_onemap")
    by_path = df.Transformer.load(CHECKOUT_MODELS / "doomnames_onemap")
    assert by_name.vocab == by_path.vocab and by_name.p.keys() == by_path.p.keys()
    assert all(np.array_equal(by_name.p[k], by_path.p[k]) for k in by_name.p)
    monkeypatch.chdir(tmp_path)                                         # outside the checkout
    assert df.Transformer.load("models/doomnames_onemap").T == by_name.T
    with pytest.raises(FileNotFoundError, match="doomnames_onemap"):    # the message lists the shipped names
        df.Transformer.load("no_such_model")
    with pytest.raises(FileNotFoundError):
        df.Transformer.load(tmp_path / "models" / "x" / "doomnames_onemap")


def test_installed_package_data_is_searched_first(tmp_path, monkeypatch):
    pkg_models, other = tmp_path / "doomformer" / "models", tmp_path / "models"
    for root, name in ((pkg_models, "mine"), (other, "mine"), (other, "only_here")):
        (root / name).mkdir(parents=True)
        for f in ("config.json", "params.npz"):
            (root / name / f).write_text(str(root))
    monkeypatch.setattr(shipped, "_ROOTS", (pkg_models, other))
    assert shipped.resolve("mine") == pkg_models / "mine"
    assert shipped.resolve("only_here") == other / "only_here"
    assert shipped.names() == ["mine", "only_here"]


def test_shipped_corpora():
    names = shipped.corpus("doomnames")
    assert len(names) == 136 and "FEAR" in names and all(len(n) < df.Transformer.load("doomnames").T for n in names)
    assert "FEAR" in shipped.corpus("doomnames_onemap")          # rows with a "clean" field
    assert shipped.corpus("doombrain") is None


def test_topdown_picture_of_a_compiled_wad(tmp_path):
    with df.compile(random_mlp(), tmp_path / "mlp.wad") as brain:
        img = df.record.topdown(brain.wad, size=(500, 260), title="t")
    px = np.asarray(img).reshape(-1, 3)
    assert img.size == (500, 260)
    assert (px == (255, 160, 40)).all(-1).any() and (px == (70, 90, 140)).all(-1).any()   # weight lines and walls


def test_cli_reports_setup_errors_on_one_line(monkeypatch, capsys):
    def no_acc():
        raise acc.AccError("ACC is not available: missing cmake")
    monkeypatch.setattr(acc, "ensure", no_acc)
    assert cli.main(["hello"]) == 1
    out = capsys.readouterr()
    assert out.err.strip() == "doomformer hello: ACC is not available: missing cmake" and "Traceback" not in out.err
    with pytest.raises(SystemExit, match="shipped name"):
        cli.main(["plan", "no_such_model"])
    assert cli.main(["plan", "doomnames_onemap"]) == 0


def test_cli_demo_fast_is_exact(tmp_path, capsys):
    png = tmp_path / "demo.png"
    assert cli.main(["demo", "--fast", "--png", str(png)]) == 0
    out = capsys.readouterr().out
    assert "EXACT" in out and "values read back" in out and png.stat().st_size > 1000


def test_cli_demo_default_writes_a_whole_name_in_tiles_mode(tmp_path, capsys):
    png = tmp_path / "tile.png"
    assert cli.main(["demo", "--workers", "2", "--letters", "2", "--png", str(png)]) == 0
    out = capsys.readouterr().out
    assert "EXACT" in out and "tile runs" in out and "training" in out and png.stat().st_size > 1000


def test_acc_build_steps_time_out_and_fail_cleanly(monkeypatch):
    def hang(cmd, **kw):
        assert kw["timeout"] > 0
        raise subprocess.TimeoutExpired(cmd, kw["timeout"])
    monkeypatch.setattr(subprocess, "run", hang)
    with pytest.raises(acc.AccError, match="timed out"):
        acc._run("git clone", ["git", "clone", "x"], timeout=1, hint="Is github.com reachable?")
    monkeypatch.setattr(subprocess, "run", lambda cmd, **kw: subprocess.CompletedProcess(cmd, 128, "", "no route"))
    with pytest.raises(acc.AccError, match="reachable"):
        acc._run("git clone", ["git", "clone", "x"], timeout=1, hint="Is github.com reachable?")


def test_mlp_contract_without_doom(tmp_path):
    net = random_mlp()
    x = np.array([[10, -20, 30, -40], [0, 0, 0, 0]])
    r = net.reference(x)
    assert r["y"].shape == (2, 3) and [h.shape for h in r["hidden"]] == [(2, 6), (2, 5)]
    a0 = intref.adc_signed(x, net.thresholds[0])
    np.testing.assert_array_equal(r["hidden"][0], a0 @ net.weights[0].T + net.biases[0])
    net.check(x)
    with pytest.raises(df.RangeError):
        net.check([12001, 0, 0, 0])
    big = df.MLP([np.full((1, 200), 255)], [np.zeros(1, int)], [[1, 2, 3]])
    with pytest.raises(df.RangeError, match="partial sums"):
        big.check(np.full(200, 5))
    with pytest.raises(df.RangeError):
        df.MLP([[[256]]], [[0]], [[1]])
    with pytest.raises(df.RangeError):
        df.MLP([[[1]]], [[0]], [[3, 2]])
    with pytest.raises(TypeError):
        df.MLP([[[0.5]]], [[0]], [[1]])
    net.save(tmp_path / "net.npz")
    back = df.MLP.load(tmp_path / "net.npz")
    np.testing.assert_array_equal(back.reference(x)["y"], r["y"])
    assert df.plan(net) == df.plan(back)
    assert df.plan(df.SpikingNet(["a", "b"], ["i"], [[0, 3], [2, 0]], [[5], [0]], theta=4, leak=1))["dolls"] > 0


def test_play_argument_handling(tmp_path, monkeypatch):
    with pytest.raises(TypeError):
        df.play(42)
    with pytest.raises(FileNotFoundError):
        df.play(tmp_path / "missing.wad")
    wad = tmp_path / "x.wad"
    wad.write_bytes(b"PWAD")
    monkeypatch.delenv("DISPLAY", raising=False)
    monkeypatch.delenv("WAYLAND_DISPLAY", raising=False)
    monkeypatch.setattr("sys.platform", "linux")
    with pytest.raises(RuntimeError, match="display"):
        df.play(wad)


def test_acc_lookup_order(tmp_path, monkeypatch):
    found = acc.ensure()
    assert found.is_file() and (acc.include_dir(found) / "zcommon.acs").is_file()
    monkeypatch.setenv("DOOMFORMER_ACC", str(found))
    assert acc.ensure() == found
    monkeypatch.setenv("DOOMFORMER_ACC", str(tmp_path / "nope"))
    with pytest.raises(acc.AccError):
        acc.ensure()
    monkeypatch.delenv("DOOMFORMER_ACC")
    monkeypatch.setenv("DOOMFORMER_ACC_IGNORE_REPO", "1")
    monkeypatch.setenv("DOOMFORMER_ACC_CACHE", str(tmp_path / "cache"))
    assert acc.find() == (None, "")                                # would build from scratch on ensure()
