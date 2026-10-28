"""The lint soundness campaign (doomformer/lintfuzz.py): jobs are reproducible, a flagged map that DOOM runs wrong is a
true positive, a clean exact map a true negative, and no job of a small fixed batch is a false negative. One game at a
time; the full campaign (1,000 nets x 3 opt levels) is a logged command, models/schedule/lintfuzz-seed0/."""
import json

from doomformer import lintfuzz as lf


def test_draws_are_reproducible():
    a = lf.draw_timing(5, 3, 2)
    assert a == lf.draw_timing(5, 3, 2) and a != lf.draw_timing(5, 4, 2)
    case = lf.draw_case(5, 3)
    assert json.dumps(case) == json.dumps(lf.draw_case(5, 3)) and len(case["inputs"]) <= lf.N_INPUTS
    for opt in lf.OPTS:
        timing, over = lf.draw_timing(5, 1, opt)
        assert over and all(getattr(timing, k) == v for k, v in over.items())
        assert (timing.sync_margin is None) == (opt == 1)


def test_small_batch_has_no_false_negative():
    """Net 1 of seed 9: opt 1 with serial 11 is flagged and DOOM is wrong; at opt 2 serial 49 it is clean and exact."""
    flagged = lf.run_job(9, 1, 1)[0]
    assert flagged["verdict"] == "TP" and flagged["doom_fail"] > 0 and "serial" in flagged["margins_failed"]
    clean = lf.run_job(9, 1, 2)[0]
    assert clean["verdict"] == "TN" and clean["doom_fail"] == 0
    assert lf.summarize([flagged, clean])["false_negatives"] == 0
