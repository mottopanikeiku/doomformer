"""df.run_many over local game processes and Ray actors, and tile mode on Ray: order, exactness, mismatch index,
retry of a killed actor; `doomformer mnist`'s report on a stand-in fleet. The Ray tests skip when Ray is not installed
(pip install "doomformer[ray]")."""
import json
import sys

import numpy as np
import pytest

import doomformer as df
from doomformer import cli, intref, kit
from doomformer.onemap import tiny_model

NET = dict(weights=[[[2, -1], [-1, 2]], [[1, -2]]], biases=[[1, -1], [3]], thresholds=[[1, 3], [1, 2]])


def small():
    return df.MLP(**NET)


def inputs(n, seed=0):
    return np.random.default_rng(seed).integers(-4, 5, size=(n, 2))


@pytest.fixture(scope="module")
def ray():
    ray = pytest.importorskip("ray")
    started = not ray.is_initialized()
    if started:
        ray.init(num_cpus=2, include_dashboard=False)
    yield ray
    if started:
        ray.shutdown()


def test_local_outputs_in_input_order_and_exact():
    net, xs, seen = small(), inputs(10), []
    ys = df.run_many(net, xs, workers=3, progress=seen.append)
    assert len(ys) == len(xs)
    for x, y in zip(xs, ys):
        np.testing.assert_array_equal(y, net.reference(x)["y"])
    assert sorted(e["index"] for e in seen) == list(range(10)) and [e["done"] for e in seen] == list(range(1, 11))
    assert {e["checked"] for e in seen} == {5} and {e["game"] for e in seen} <= {0, 1, 2}
    assert df.run_many(net, []) == []


def test_local_mismatch_names_the_input(monkeypatch):
    """DOOM runs the map of net A while the check uses net B, which differs only where x[1] is active."""
    a = df.MLP([[[1, 2]]], [[0]], [[1]])
    b = df.MLP([[[1, 3]]], [[0]], [[1]])
    real = kit.compile
    monkeypatch.setattr(kit, "compile", lambda net, **kw: real(a, **kw))
    xs = [[k % 3, 0] for k in range(6)]
    xs[4] = [0, 1]
    with pytest.raises(df.MismatchError) as err:
        df.run_many(b, xs, workers=2)
    assert err.value.index == 4 and str(err.value).startswith("input 4: ")
    assert ("y0", 2, 3) in err.value.mismatches


def test_inputs_are_checked_before_any_game():
    net = small()
    with pytest.raises(df.RangeError, match="^input 1: "):
        df.run_many(net, [[1, 1], [20000, 0]])
    with pytest.raises(ValueError, match="^input 0: "):
        df.run_many(net, [[1, 1, 1]])
    with pytest.raises(ValueError, match="backend"):
        df.run_many(net, [[1, 1]], backend="slurm")
    with pytest.raises(TypeError, match="spiking"):
        df.run_many(df.SpikingNet(["n"], ["i"], [[0]], [[1]], theta=1, leak=0), [[1]])


def test_missing_ray_says_how_to_install(monkeypatch):
    monkeypatch.setitem(sys.modules, "ray", None)
    with pytest.raises(ImportError, match=r'pip install "doomformer\[ray\]"'):
        df.run_many(small(), [[1, 1]], backend="ray")
    with pytest.raises(ImportError, match=r'pip install "doomformer\[ray\]"'):
        df.compile(small(), mode="tiles", backend="ray")
    with pytest.raises(ValueError, match="mode='tiles'"):
        df.compile(small(), backend="ray")


def fake_fleet(fail_at=None):
    """Stands in for df.run_many: the reference's outputs, reported the way the games report them."""
    def run_many(net, xs, *, workers=None, progress=None, **kw):
        out = []
        for i, x in enumerate(xs):
            if i == fail_at:
                err = df.MismatchError(f"input {i}", [("y3", 1, 0)])
                err.index = i
                raise err
            out.append(net.reference(x)["y"])
            progress(dict(index=i, done=i + 1, total=len(xs), game=i % workers, output=out[-1], checked=216,
                          tics=5000, seconds=60.0, retried=False))
        return out
    return run_many


def test_cli_mnist_large_run_report_and_record(monkeypatch, capsys, tmp_path):
    monkeypatch.setattr(df, "run_many", fake_fleet())
    record = tmp_path / "run.json"
    assert cli.main(["mnist", "--digits", "12", "--workers", "3", "--seed", "0", "--record", str(record)]) == 0
    out = capsys.readouterr().out
    assert "[12/12]" in out and "12/12 EXACT" in out and "digits per minute in 3 DOOM games" in out
    assert out.count("DOOM reads:") == cli.MNIST_ART                      # art for the first few digits only
    rec = json.loads(record.read_text())
    assert rec["digits"] == rec["done"] == rec["exact"] == len(rec["results"]) == 12 and rec["error"] is None
    assert rec["correct"] == sum(r["doom"] == r["label"] for r in rec["results"]) and rec["games"] == 3


def test_cli_mnist_mismatch_is_reported_with_the_digit(monkeypatch, capsys, tmp_path):
    monkeypatch.setattr(df, "run_many", fake_fleet(fail_at=5))
    record = tmp_path / "run.json"
    assert cli.main(["mnist", "--digits", "12", "--workers", "2", "--seed", "0", "--record", str(record)]) == 1
    out = capsys.readouterr().out
    rec = json.loads(record.read_text())
    assert rec["done"] == 5 and rec["error"].startswith("test #") and ": input 5: 1 value(s) differ" in rec["error"]
    assert "MISMATCH: test #" in out and "5/12 EXACT" in out


def test_ray_run_many_exact(ray):
    net, xs = small(), inputs(8, seed=1)
    ys = df.run_many(net, xs, workers=2, backend="ray")
    for x, y in zip(xs, ys):
        np.testing.assert_array_equal(y, net.reference(x)["y"])


def test_ray_killed_actor_input_runs_again(ray):
    net, xs, seen, killed = small(), inputs(30, seed=2), [], []
    names = lambda: {a["name"] for a in ray.util.list_named_actors(all_namespaces=True)        # noqa: E731
                     if a["namespace"] == "doomformer"}
    earlier = names()                                # games of earlier runs may still be listed

    def kill_the_other_game(e):
        seen.append(e)
        if e["done"] == 3 and not killed:            # the other game is busy or about to get its next input
            other = sorted(n for n in names() - earlier if not n.endswith(f"-{e['game']}"))[0]
            ray.kill(ray.get_actor(other, namespace="doomformer"))
            killed.append(other)

    ys = df.run_many(net, xs, workers=2, backend="ray", progress=kill_the_other_game)
    assert len(killed) == 1
    for x, y in zip(xs, ys):
        np.testing.assert_array_equal(y, net.reference(x)["y"])
    retried = [e for e in seen if e["retried"]]
    assert len(retried) == 1 and len(seen) == len(xs)
    assert not killed[0].endswith(f"-{retried[0]['game']}")   # it ran again on the surviving game


def test_ray_tile_mode_transformer_and_mlp(ray, tmp_path):
    net = tiny_model()
    with df.compile(net, tmp_path / "tf", mode="tiles", workers=2, backend="ray") as brain:
        assert brain.info["backend"] == "ray"
        for tokens in ([0], [0, 1]):
            np.testing.assert_array_equal(brain(tokens), intref.forward(net, tokens))
        assert brain.generate("A", max_new=1) == intref.generate(net, "A", max_new=1)
        assert brain.pool.backend == "ray" and brain.pool.procs == []
    mlp, xs = small(), inputs(3, seed=3)
    with df.compile(mlp, tmp_path / "mlp.wad", mode="tiles", workers=2, backend="ray") as brain:
        np.testing.assert_array_equal(brain(xs), mlp.reference(xs)["y"])
        assert brain.info["checked"] == 5
        mlp.biases[-1][0] += 1
        with pytest.raises(df.MismatchError):
            brain(xs[0])
