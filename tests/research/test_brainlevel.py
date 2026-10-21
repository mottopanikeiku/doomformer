"""The transformer and arena share one real map; recycling retains exact values."""
import numpy as np
import vizdoom as vzd

from demos import arena
from demos.brainlevel import build_brainlevel
from doomformer.intref import forward
from doomformer.onemap import compare_trace, logits_from_readout, tiny_model, token_ports
from doomformer.runtime import TileRunner


def test_transformer_and_arena_in_one_game(tmp_path):
    model = tiny_model()
    wad = tmp_path / "brainlevel.wad"
    meta = build_brainlevel(model, wad)
    planned = build_brainlevel(model, tmp_path / "plan.wad", layout_only=True)
    assert planned.n_sectors == meta.n_sectors
    runner = TileRunner(wad, configure=arena.configure)
    game = runner.game
    try:
        game.send_game_command(f"puke -{arena.NEW_LIFE}")
        game.advance_action(1)
        for tokens in ([0, 1, 2], [0, 4, 3]):
            game.set_action(arena.HANDS_OFF)
            game.send_game_command(f"puke -{arena.FREEZE}")
            game.advance_action(1)
            trace = {}
            expected = forward(model, list(tokens), trace)
            got = runner.run(token_ports(model, list(tokens)))[0]
            assert set(got) == set(meta.outputs)
            assert compare_trace(got, trace, len(tokens)) == len(got)
            np.testing.assert_array_equal(logits_from_readout(model, got, len(tokens)), expected)
            assert game.get_game_variable(vzd.GameVariable.POSITION_X) == arena.CX
            assert game.get_game_variable(vzd.GameVariable.POSITION_Y) == arena.CY
            assert any(o.name == "Demon" for o in game.get_state().objects)
            game.send_game_command(f"puke -{arena.UNFREEZE}")
            game.make_action(arena.buttons("L"), 4)
    finally:
        runner.close()

def test_autonomous_transformer_readout_only(tmp_path):
    from doomformer import policy
    from demos.brainlevel import ClockRun, build_twin

    model = tiny_model()
    model.vocab = policy.VOCAB
    rng = np.random.default_rng(19)
    model.p["tok_emb"] = rng.integers(-2, 3, (len(model.vocab), model.d))
    model.p["Wun"] = rng.integers(-2, 3, (len(model.vocab), model.d))
    model.p["bun"] = rng.integers(-2, 3, len(model.vocab))
    calibration = tmp_path / "cal.wad"
    build_twin(model, calibration)
    runner = TileRunner(calibration, configure=arena.configure)
    try:
        settled = runner.run(token_ports(model, [0, 4, 23]))[1]
    finally:
        runner.close()
    wad = tmp_path / "clock.wad"
    build_brainlevel(model, wad, period=settled + max(64, settled // 8))
    run = ClockRun(model, wad, seed=4, resolution=vzd.ScreenResolution.RES_160X120)
    try:
        for _ in range(2):
            samples = []
            decision = run.decide(
                on_think=lambda tokens, tic, budget: samples.append((list(tokens), tic, budget)),
                think_every=128)
            assert decision is not None
            assert decision["tokens"] == policy.observe(run.g.get_state(), run.g)
            assert decision["comparator"]["onehot"] == [int(a == decision["action"]) for a in range(3)]
            assert len(samples) > 2
            assert all(tokens == decision["tokens"] for tokens, _, _ in samples)
            assert all(budget == run.schedule["period"] for _, _, budget in samples)
            assert samples[0][1] <= 3
            assert all(tic % 128 == 0 for _, tic, _ in samples[1:])
            assert all(a[1] < b[1] for a, b in zip(samples, samples[1:]))
            run.play(lambda: None)
        assert run.decisions == 2
        assert not run.audit()["forbidden"]
    finally:
        run.close()
