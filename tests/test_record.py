"""brain.record: real headless DOOM games filmed to GIF/MP4. Frames come only from state that is read (the film must not
change a result or the length of a pass), and the EXACT label exists only for passes whose comparison passed."""
from __future__ import annotations

import numpy as np
import pytest
from PIL import Image

import doomformer as df
from doomformer import intref
from doomformer.onemap import tiny_model
from doomformer.record import Recording


def xor_net() -> df.MLP:
    return df.MLP([np.array([[1, 1], [1, 1]]), np.array([[1, -1]])], [np.array([0, -1]), np.array([0])],
                  [np.array([1]), np.array([1])], signed_input=False)


def gif_frames(path) -> list[np.ndarray]:
    im = Image.open(path)
    out = []
    for i in range(im.n_frames):
        im.seek(i)
        out.append(np.asarray(im.convert("RGB")).copy())
    return out


def green_text(frame: np.ndarray) -> int:
    """Pixels of the EXACT line's green (90, 220, 100), loosely, in the bottom quarter of the frame."""
    f = frame[-frame.shape[0] // 4:].astype(int)
    return int(((f[..., 1] > 170) & (f[..., 0] < 150) & (f[..., 2] < 150)).sum())


def red_text(frame: np.ndarray) -> int:
    f = frame[-frame.shape[0] // 4:].astype(int)
    return int(((f[..., 0] > 200) & (f[..., 1] < 120) & (f[..., 2] < 120)).sum())


def test_xor_gif_is_verified_and_changes_nothing(tmp_path):
    net = xor_net()
    inputs = [[0, 0], [0, 1], [1, 0], [1, 1]]
    with df.compile(net, tmp_path / "a.wad") as brain:
        rec = brain.record(inputs, tmp_path / "xor.gif", width=480)
    assert isinstance(rec, Recording) and rec.verified and rec.passes == 4
    assert [int(y[0]) for y in rec.results] == [0, 1, 1, 0]
    for x, y in zip(inputs, rec.results):
        np.testing.assert_array_equal(y, net.reference(np.array(x))["y"])
    assert rec.checked == 4 * 5 and rec.bytes == (tmp_path / "xor.gif").stat().st_size < 5_000_000
    frames = gif_frames(tmp_path / "xor.gif")
    assert len(frames) == rec.frames > 4 * 10                      # a run of 192 tics sampled every 8 tics
    assert green_text(frames[0]) == 0 and green_text(frames[-1]) > 50      # EXACT only once the pass is verified
    assert all(green_text(f) == 0 for f in frames[:rec.frames // 4 - 1])    # none while the first pass is running
    # nothing changed: an unrecorded twin brain runs the same passes with the same outputs and game tics
    tics = []
    with df.compile(net, tmp_path / "b.wad") as plain:
        for x, y in zip(inputs, rec.results):
            np.testing.assert_array_equal(plain(np.array(x)), y)
            tics.append(plain.info["tics"])
    assert sum(tics) == rec.tics


def test_mp4_and_view_both_with_a_world_camera(tmp_path):
    net = xor_net()

    def world(m):
        m.rect(-768, -2816, -512, -2560, floor=0, ceil=128, light=200, ftex="FLOOR4_8", wall="STARTAN3")
        m.rect(-512, -2816, -256, -2560, floor=0, ceil=128, light=200, ftex="FLAT14", wall="COMPBLUE")
        m.thing(-700, -2688, 1)                                           # the player looks east
        m.thing(-300, -2688, 9025, angle=180, id=77)                      # a security camera looking back at him
        return "script 202 (int cam) { if (cam) { ChangeCamera(77, 0, 0); } else { ChangeCamera(0, 0, 0); } }"

    with df.compile(net, tmp_path / "w.wad", world=world, mode="map") as brain:
        plain = brain.record([1, 0], tmp_path / "fp.gif", view="firstperson", every=32, width=320, hold=0)
        cam = brain.record([1, 0], tmp_path / "cam.mp4", view="both", every=32, width=480, camera_script=202)
        assert plain.verified and cam.verified and cam.bytes > 0 and (tmp_path / "cam.mp4").stat().st_size > 1000
        np.testing.assert_array_equal(cam.results[0], plain.results[0])
        shot_camera = brain.record([1, 0], tmp_path / "cam.gif", view="firstperson", every=32, width=320,
                                   camera_script=202, hold=0)
    a, b = gif_frames(tmp_path / "fp.gif")[0], gif_frames(tmp_path / "cam.gif")[0]
    assert a.shape == b.shape and np.abs(a[:-60].astype(int) - b[:-60].astype(int)).mean() > 5   # a different view
    assert shot_camera.verified


def test_mismatch_still_writes_the_file_but_never_says_exact(tmp_path):
    net = xor_net()
    with df.compile(net, tmp_path / "m.wad") as brain:
        brain([1, 0])
        net.biases[0][1] += 1                                            # the map still adds the old bias
        with pytest.raises(df.MismatchError):
            brain.record([[0, 0], [1, 1]], tmp_path / "bad.gif", width=480)
    frames = gif_frames(tmp_path / "bad.gif")
    assert len(frames) > 10 and all(green_text(f) == 0 for f in frames) and red_text(frames[-1]) > 50


def test_spiking_steps_exact(tmp_path):
    net = df.SpikingNet(["a", "b"], ["kick"], np.array([[0, 0], [12, 0]]), np.array([[12], [0]]), theta=12, leak=0)
    X = np.array([[1], [0], [0], [1]])
    _, S = net.run(X)
    with df.compile(net, tmp_path / "s.wad") as brain:
        rec = brain.record(X, tmp_path / "s.gif", view="both", every=16, width=400)
    assert rec.verified and rec.passes == 4
    np.testing.assert_array_equal(np.array([r[1] for r in rec.results]), S)
    assert len(gif_frames(tmp_path / "s.gif")) == rec.frames


def test_transformer_generate_every_step_recorded(tmp_path):
    model = tiny_model()
    with df.compile(model, tmp_path / "t.wad") as brain:
        rec = brain.record("A", tmp_path / "t.gif", every=64, width=480, hold=0.2, max_new=2)
    assert rec.verified and rec.passes >= 1 and rec.checked > 0
    assert rec.results == [intref.generate(model, "A", max_new=2)]


def test_arguments_are_validated(tmp_path):
    with df.compile(xor_net(), tmp_path / "v.wad") as brain:
        for kw, exc in [(dict(every=4), ValueError), (dict(every=12), ValueError), (dict(view="sideways"), ValueError),
                        (dict(fps=0), ValueError), (dict(camera_script=201), ValueError)]:
            with pytest.raises(exc):
                brain.record([1, 0], tmp_path / "x.gif", **kw)
        with pytest.raises(ValueError):
            brain.record([1, 0], tmp_path / "x.avi")
        with pytest.raises(df.RangeError):
            brain.record([1, 10 ** 6], tmp_path / "x.gif")               # the exact checks run before DOOM does
    assert not list(tmp_path.glob("x.*"))


def test_big_maps_use_the_density_drawing(tmp_path, monkeypatch):
    from doomformer import record as rec_mod
    with df.compile(xor_net(), tmp_path / "d.wad") as brain:                 # compiled, no game started
        lines = rec_mod._Map(brain.wad, (320, 160))
        monkeypatch.setattr(rec_mod, "MAX_DRAWN_LINES", 10)
        dense = rec_mod._Map(brain.wad, (320, 160))
    assert lines.base.shape == dense.base.shape == (160, 320, 3) and lines.n_lines == dense.n_lines > 10
    assert (dense.base != np.array(rec_mod.BG)).any(axis=2).sum() > 50
