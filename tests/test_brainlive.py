"""Live demons (demos/brainlive.py): the life rules and the control arenas, without the dolls (a few seconds)."""
import pytest

pytest.importorskip("vizdoom")

from demos import brainlive as bl  # noqa: E402


def test_life_rows_end_a_life_at_the_first_of_death_empty_pistol_or_cap():
    rec = bl.Record()
    # (health, ammo, kills, awake) per tic: life 1 starts at t=10 and dies at t=14; life 2 starts at t=20 and runs dry at 23
    seq = {10: (100, 26, 0, 0), 11: (100, 26, 1, 1), 12: (60, 25, 1, 2), 13: (30, 25, 2, 3), 14: (1, 25, 3, 4),
           15: (1, 25, 9, 5), 20: (100, 26, 9, 0), 21: (100, 1, 9, 1), 22: (80, 1, 10, 2), 23: (80, 0, 11, 3),
           24: (80, 0, 12, 4)}
    for t, v in seq.items():
        rec.hist[t] = v
    rows = bl.life_rows(rec, [10, 20], end_time=30)
    assert rows == [dict(life=1, kills=3, alive=4, reason="dead"), dict(life=2, kills=2, alive=3, reason="ammo")]
    assert bl.summarise(rows)["kills_total"] == 5


def test_frozen_aimbot_reproduces_the_documented_first_lives_of_seed_4(tmp_path):
    """The aimlearn demo's aimbot on seed 4 makes 9, 26, 26 kills in its first three lives (demos/README.md)."""
    r = bl.run_control("aimbot", seed=4, lives=3, frozen=True, workdir=tmp_path)
    assert [x["kills"] for x in r["rows"]] == [9, 26, 26]
    assert [x["reason"] for x in r["rows"]] == ["dead", "ammo", "ammo"]


def test_latency_costs_kills_and_the_live_clock_ends_every_life(tmp_path):
    frozen = bl.run_control("model", seed=4, lives=3, frozen=True, workdir=tmp_path)["summary"]
    slow = bl.run_control("model", seed=4, lives=3, delay=773, workdir=tmp_path)["summary"]
    assert slow["lives"] == 3 and slow["ended"]["dead"] == 3 and slow["kills_total"] == 0
    assert frozen["kills_per_life"] > 10 and slow["alive_median"] < 400
