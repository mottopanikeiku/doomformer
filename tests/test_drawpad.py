"""DrawPad: winner layers (DOOM's own argmax) and the level where the marine walks a digit and DOOM reads it."""
import os
import shutil
import signal
import subprocess
import sys

import numpy as np
import pytest

import doomformer as df
from doomformer import cli, drawpad
from doomformer.runtime import USER


def score_net(n: int, K: int) -> df.MLP:
    """One layer whose n scores are the signed staircase of the n inputs: s = clip(x, -K, K)."""
    return df.MLP([np.eye(n, dtype=np.int64)], [np.zeros(n, dtype=np.int64)], [np.arange(1, K + 1)], signed_input=True)


def test_winner_layers_equal_numpy_argmax_including_ties():
    rng = np.random.default_rng(0)
    for n in (2, 3, 10):
        net = df.with_argmax(score_net(n, 3))
        assert net.score_scale == 1.0 and [W.shape[0] for W in net.weights] == [n * (n - 1) // 2, n, n]
        x = rng.integers(-4, 5, (3000, n))                 # many ties, and values beyond the staircase (clipped)
        y = net.reference(x)["y"]
        np.testing.assert_array_equal(y, np.eye(n, dtype=np.int64)[np.argmax(np.clip(x, -3, 3), 1)])
        for row in x[:50]:
            net.check(row)


def test_winner_layers_rescale_a_score_layer_that_does_not_fit():
    rng = np.random.default_rng(1)
    scores = df.MLP([rng.integers(-60, 61, (16, 12)), rng.integers(-255, 256, (10, 16))],
                    [rng.integers(-20, 21, 16), rng.integers(-500, 501, 10)],
                    [np.array([1, 2, 3]), np.arange(1, 9) * 40])
    net = drawpad.with_argmax(scores)
    f = net.score_scale
    assert 0 < f < 1
    W, b = np.rint(scores.weights[-1] * f).astype(np.int64), np.rint(scores.biases[-1] * f).astype(np.int64)
    x = rng.integers(-150, 151, (2000, 12))
    s = scores.reference(x)["activations"][-1] @ W.T + b            # the rescaled integer scores
    np.testing.assert_array_equal(net.reference(x)["y"], np.eye(10, dtype=np.int64)[np.argmax(s, 1)])
    d = net.weights[-3]                                             # every pair row fits the gate range for ANY input
    assert np.abs(d).max() <= 255 and (np.abs(net.biases[-3]) + np.abs(d).sum(1) * 8).max() <= 12000


def count_world(m):
    m.rect(-768, -2816, -512, -2560, floor=0, ceil=128)
    m.thing(-640, -2688, 1)
    return """
int fired[4];
script 201 (int d) { fired[d]++; }
script 202 (void) { o1 = fired[0]; o2 = fired[1]; o3 = fired[2]; o4 = fired[3]; }
"""


def test_doom_one_hot_and_its_action_fire_for_the_winner_only(tmp_path):
    net = drawpad.with_argmax(score_net(4, 3))
    actions = {f"y{d}": [df.act.script(201, d)] for d in range(4)}
    fired = np.zeros(4, dtype=np.int64)
    with df.compile(net, tmp_path / "argmax.wad", world=count_world, actions=actions) as brain:
        for x in ([2, 2, -1, 0], [-1, 3, 3, 3], [0, 0, 0, 0], [-3, -2, -2, 1], [-5, 4, 9, 7]):
            y = brain(np.array(x))                                   # every value checked against the reference
            winner = int(np.argmax(np.clip(x, -3, 3)))
            assert y.tolist() == np.eye(4, dtype=int)[winner].tolist()
            fired[winner] += 1
            assert brain.run_script(202, read=4) == fired.tolist()  # exactly one firing, the winner's


class RealTimeGame:
    """ViZDoom in spectator mode as measured (docs/drawpad.md, "Readouts"): the engine keeps real time, so one
    advance_action(n) runs n or more tics, and the commands sent between two advances run together on one tic that can
    lie beyond the tics the next advance runs. Script s writes USER1 = s, USER2 = how often s ran; S_ACK counts in
    ACS global 0, which ViZDoom reports as the reward (16.16 fixed point)."""

    def __init__(self, seed: int):
        self.rng = np.random.default_rng(seed)
        self.tic, self.users, self.acks, self.runs, self.pending, self.lands = 0, [0] * 60, 0, {}, [], None

    def send_game_command(self, cmd: str):
        if self.lands is None:                           # the first command since the last advance picks the tic
            self.lands = self.tic + 1 + int(self.rng.integers(0, 3))
        self.pending.append((self.lands, int(cmd.split()[1].lstrip("-"))))

    def advance_action(self, n: int = 1):
        self.lands = None
        for _ in range(n + int(self.rng.integers(0, 2))):
            self.tic += 1
            for _, s in [p for p in self.pending if p[0] <= self.tic]:
                if s == drawpad.S_ACK:
                    self.acks += 1
                else:
                    self.runs[s] = self.runs.get(s, 0) + 1
                    self.users[:2] = [s, self.runs[s]]
            self.pending = [p for p in self.pending if p[0] > self.tic]

    def get_game_variable(self, v):
        return self.users[USER.index(v)]

    def get_total_reward(self) -> float:
        return self.acks / 65536


def test_a_readout_waits_for_its_ack_when_commands_land_late():
    scripts = (drawpad.S_STATUS, drawpad.S_CANVAS)
    g, runs = RealTimeGame(0), {}
    pad = object.__new__(drawpad.DrawPad)                # only the readout protocol, no map
    pad.spectator, pad._acks, pad.readouts, pad.readout_waits = True, 0, 0, 0
    for k in range(400):
        s = scripts[k % 2]
        runs[s] = runs.get(s, 0) + 1
        assert pad._users(g, s, 2) == [s, runs[s]]       # always the values of the script just sent
    assert pad.readouts == 400 and 0 < pad.readout_waits < 400
    stale = 0
    for k in range(400):                                 # the old way on the same engine: puke, one advance, read
        g.send_game_command(f"puke -{scripts[k % 2]}")
        g.advance_action(1)
        stale += g.users[0] != scripts[k % 2]
    assert stale > 0


@pytest.fixture(scope="module")
def pad():
    with drawpad.DrawPad() as p:
        yield p


@pytest.mark.timeout(1500)                  # MNIST map (~160k sectors): about 90 s here, 5+ min on a 3-core CI runner
def test_walked_digit_is_read_exactly_and_the_host_is_read_only(pad):
    pad.draw(drawpad.walked_strokes(1, np.random.default_rng(0)))
    reads = pad.status()["reads"] + 1
    pad.press("read")
    ink = pad.status()["ink"]
    pad.press("clear")                                     # refused while DOOM reads
    blocked = []

    def try_to_steer(p, tic):                              # during the pass every host write must be refused
        try:
            p.game.set_action([10.0, 0, 0, 0, 0])
        except RuntimeError as e:
            blocked.append(str(e))

    r = pad.verify(reads, on_tic=try_to_steer)
    assert blocked and all("set_action" in e for e in blocked)
    assert r.audit["forbidden"] == {"set_action": len(blocked)}
    assert r.digit == 1 and r.y == np.eye(10, dtype=int)[1].tolist() and r.rasterised and r.canvas
    assert r.checked == len(pad.brain.meta.outputs) and r.status["fires"] == 1
    assert (r.status["clears"], r.status["ink"]) == (0, ink) and r.pixels.sum() == ink
    assert np.array_equal(r.pixels, pad.canvas())         # ports = the level's canvas
    assert r.audit["commands"][f"puke -{drawpad.S_ACK}"] == r.readouts > 0
    assert r.readout_waits == 0                            # PLAYER mode: every ack one tic after its readout


@pytest.mark.timeout(1500)                  # MNIST map (~160k sectors): about 90 s here, 5+ min on a 3-core CI runner
def test_clear_wipes_canvas_ports_and_display(pad):
    pad.press("clear")
    pad.tic()
    pad.tic()
    st = pad.status()
    assert (st["ink"], st["digit"], st["fires"], st["clears"]) == (0, -1, 0, 1)
    assert not pad.ports().any()


class Keyboard:
    """A player's hand on the keys of a headless game: taps USE on the `when`-th advance, nothing otherwise."""

    def __init__(self, game, when: int):
        self.game, self.when, self.n = game, when, 0

    def advance_action(self, tics: int = 1):
        self.n += 1
        self.game.set_action([0.0, 0.0, 0.0, 0.0, float(self.n == self.when)])
        return self.game.advance_action(tics)

    def __getattr__(self, name):
        return getattr(self.game, name)


@pytest.mark.timeout(1500)                  # MNIST map (~160k sectors): about 90 s here, 5+ min on a 3-core CI runner
def test_watch_checks_the_read_a_player_presses(pad):
    """The interactive loop, headless: the marine stands at the READ switch, a key tap presses USE while watch() runs,
    and watch() sees the READ in the level's status, verifies the pass and hands it over."""
    pad.path = []
    pad.draw(drawpad.walked_strokes(7, np.random.default_rng([0, 7])))
    a, b = drawpad.READ_X
    pad.walk(drawpad.route(pad.where()[:2], ((a + b) / 2, drawpad.PODIUM_Y[0] - 28)))
    pad.turn_to(90)
    raw = pad.game
    got = []

    class Done(Exception):
        pass

    def on_read(r):
        got.append(r)
        raise Done

    pad._game = pad.brain.runner.game = Keyboard(raw, when=3)
    try:
        with pytest.raises(Done):
            pad.watch(on_read, on_start=got.append)
    finally:
        pad._game = pad.brain.runner.game = raw
    st, r = got
    assert isinstance(r, drawpad.Read) and st["reads"] == r.status["reads"]
    assert r.digit == 7 and r.canvas and r.rasterised and r.audit["non_readout_commands"] == 0


@pytest.mark.timeout(1500)                  # MNIST map (~160k sectors): about 90 s here, 5+ min on a 3-core CI runner
def test_cli_draw_digit_headless(capsys):
    assert cli.main(["draw", "--digit", "7"]) == 0
    out = capsys.readouterr().out
    assert "DOOM reads: 7  (right)" in out and "EXACT" in out and "ports equal the 3x3 brush" in out
    assert "ports equal the level's canvas" in out and "Python sent" in out and "(0 rejected)" in out


def test_cli_draw_without_a_display_points_to_headless(monkeypatch):
    monkeypatch.delenv("DISPLAY", raising=False)
    monkeypatch.delenv("WAYLAND_DISPLAY", raising=False)
    monkeypatch.setattr("sys.platform", "linux")
    with pytest.raises(SystemExit, match="draw --digit 7"):
        cli.main(["draw"])


class StopHere(Exception):
    pass


@pytest.mark.parametrize("argv, opt", [(["draw"], 3), (["draw", "--opt", "0"], 0), (["draw", "--digit", "7"], 3)])
def test_cli_draw_defaults_to_opt_3_and_opt_0_is_still_there(monkeypatch, argv, opt):
    """The CLI command defaults to --opt 3 (the Python API default stays 0); the pad is built at the asked level."""
    seen = {}

    def fake_pad(*a, **kw):
        seen.update(kw)
        raise StopHere

    monkeypatch.setenv("DISPLAY", ":99")
    monkeypatch.setattr(drawpad, "DrawPad", fake_pad)
    with pytest.raises(StopHere):
        cli.main(argv)
    assert seen["opt"] == opt


def test_the_python_api_default_stays_opt_0():
    import inspect
    assert inspect.signature(drawpad.DrawPad.__init__).parameters["opt"].default == 0


def test_hud_says_how_long_the_read_takes_and_the_banner_covers_every_opt():
    tags, ink = list(range(784)), 1
    assert "of about 8 s" in drawpad.level_acs(tags, ink, 8)
    assert "of about" not in drawpad.level_acs(tags, ink)
    assert set(cli.DRAW_OPT_NOTE) == {0, 1, 2, 3} and "light levels" in cli.DRAW_OPT_NOTE[3]
    text = cli.DRAW_HELP.format(seconds=8, opt=3, opt_note=cli.DRAW_OPT_NOTE[3])
    read_line = next(i for i, line in enumerate(text.splitlines()) if line.startswith("  READ"))
    read_part = "\n".join(text.splitlines()[read_line:read_line + 3])
    assert "about 8 s of game time" in read_part and "163" not in read_part      # the comparison is in the opt line
    assert "  --opt 3  An ACS script copies" in text and "(the original --opt 0 needs about 163 s)" in text
    assert round(drawpad.MEASURED_READ_TICS[3] / 35) == 8 and round(drawpad.MEASURED_READ_TICS[0] / 35) == 163


@pytest.fixture(scope="module")
def xvfb():
    """A private X display (Xvfb) for the spectator-mode window; skipped where Xvfb is not installed."""
    exe = shutil.which("Xvfb")
    if exe is None:
        pytest.skip("needs Xvfb for a game window")
    r, w = os.pipe()
    proc = subprocess.Popen([exe, "-displayfd", str(w), "-screen", "0", "640x480x24", "-nolisten", "tcp"],
                            pass_fds=(w,), stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    os.close(w)
    display = ":" + os.read(r, 16).decode().strip()
    os.close(r)
    old = os.environ.get("DISPLAY")
    os.environ["DISPLAY"] = display
    yield display
    proc.terminate()
    proc.wait()
    if old is None:
        os.environ.pop("DISPLAY", None)
    else:
        os.environ["DISPLAY"] = old


SPECTATOR = """
import vizdoom as vzd
from doomformer import drawpad
with drawpad.DrawPad(spectator=True, resolution=vzd.ScreenResolution.RES_320X240) as sp:
    g = sp.game
    g.send_game_command(f"puke -{drawpad.S_READ}")    # the test stands in for the USE press
    g.advance_action(12)
    timer = 0
    for _ in range(150):
        st = sp.status()
        assert st["reads"] == 1 and st["phase"] >= drawpad.THINKING and st["timer"] > timer, st
        timer = st["timer"]
        assert not sp.canvas().any()
    assert sp.readouts == 300
"""


@pytest.mark.timeout(1800)
def test_spectator_readouts_are_fresh_while_the_brain_runs(xvfb):
    """Spectator mode keeps real time; while the dolls run, every readout must still return the values of the script
    just sent: STATUS with a later Timer() each time (a stale one held the CANVAS rows: all zero), CANVAS all zero.
    It runs in its own process: ViZDoom's window start can segfault when the X server is starved (seen at load
    average 80), which must not take the whole test run down. Only that signal gets a second try."""
    for attempt in range(2):
        r = subprocess.run([sys.executable, "-c", SPECTATOR], env=dict(os.environ, DISPLAY=xvfb), text=True,
                           capture_output=True, timeout=850)
        if r.returncode != -signal.SIGSEGV:
            break
    assert r.returncode == 0, (r.returncode, r.stdout[-2000:], r.stderr[-4000:])
