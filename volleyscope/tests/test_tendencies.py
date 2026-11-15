import numpy as np
import pytest

from test_stats import ev, rally
from vball.ball.events import build_rallies
from vball.stats.core import MatchSet, compute_stats
from vball.synth.evaluate import truth_flights, truth_tracks
from vball.synth.world import Params, World
from vball.tendencies.core import (Tendency, attack_zone, coaching_notes, defence_tendencies,
                                   hitter_tendencies, local, report_lines, setter_distribution)


def sim(seed=4, n=40, **params):
    w = World(seed=seed, n_rallies=n, params=Params(**params) if params else None)
    times = np.arange(0, w.duration, 1 / 30)
    tracks = truth_tracks(w, times)
    ms = MatchSet(build_rallies(truth_flights(w), tracks, 30), "Home", "Away")
    return w, ms, tracks


def test_tendency_reports_insufficient_data_with_its_sample():
    t = Tendency("x", {"line": 3}, 3, min_n=8)
    assert not t.sufficient and t.text() == "x: insufficient data (n=3, need 8)"
    t2 = Tendency("x", {"line": 6, "cross": 4}, 10, min_n=8)
    assert t2.sufficient and "line 60% (6)" in t2.text() and "(n=10)" in t2.text()
    assert Tendency("x", {}, 0).text().endswith("(n=0, need 8)")


def test_local_frame_and_attack_zone_mirror_for_far_team():
    assert local("near", (1, 8)) == (1, 1) and local("far", (8, 10)) == (1, 1)
    assert attack_zone("near", (1.5, 8.3)) == "left front"
    assert attack_zone("far", (7.5, 9.7)) == "left front"        # far team's left is high x
    assert attack_zone("near", (7.5, 5.0)) == "right back row"
    assert attack_zone("near", (4.5, 8.5)) == "middle front"


def test_setter_distribution_counts_match_truth():
    w, ms, _ = sim()
    d = setter_distribution([ms], "Home")
    truth_sets = [c for c in w.truth_contacts() if c["type"] == "set" and c["team"] == "near"]
    assert d["overall"].n == len(truth_sets)
    assert sum(d[f"rotation {k}"].n for k in range(1, 7)) == d["overall"].n
    assert d["row"].n == d["overall"].n
    # the simulator's hitters attack from their lanes: all three appear
    assert {k.split()[0] for k in d["overall"].counts} == {"left", "middle", "right"}
    assert any(not d[f"rotation {k}"].sufficient for k in range(1, 7)) or d["overall"].n >= 48


def test_hitter_shot_tendency_follows_the_simulated_bias():
    w, ms, _ = sim(n=40, shot_p=(0.8, 0.1, 0.05, 0.05))
    h = hitter_tendencies([ms], "Away")
    assert h["overall"].sufficient
    assert max(h["overall"].shares().items(), key=lambda kv: kv[1])[0] == "line"
    assert h["in system"].n + h["out of system"].n == h["overall"].n
    one = hitter_tendencies([ms], "Away", number=99)
    assert one["overall"].n == 0 and "insufficient data" in one["overall"].text()


def test_single_hitter_filter():
    w, ms, _ = sim()
    num = w.players[3].number
    mine = [c for c in w.truth_contacts() if c["type"] == "attack" and c["pid"] == 3]
    h = hitter_tendencies([ms], "Home", num)
    assert h["result"].n == h["overall"].n == len(mine)
    assert h["result"].counts.get("kill", 0) == sum(c["outcome"] == "kill" for c in mine)


def test_defence_tendencies_describe_formation():
    w, ms, tracks = sim()
    d = defence_tendencies([ms], [tracks], "Away", 30)
    assert d["blockers"].sufficient and d["open zones"].sufficient
    assert set(d["blockers"].counts) <= {"0 blockers", "1 blocker", "2 blockers", "3 blockers"}
    # simulated back row stands 6.5-7 m off the net
    assert 4.0 < d["depth"]["mean_back_court_depth_m"] < 8.0
    assert "n=" in d["open zones"].text()
    none = defence_tendencies([ms], [None], "Away", 30)
    assert not none["blockers"].sufficient and none["depth"]["mean_back_court_depth_m"] is None


def test_coaching_notes_need_evidence():
    w, ms, _ = sim(n=6)
    stats = compute_stats([ms])
    num = w.players[0].number
    out = coaching_notes(stats, [ms], "Home", num)
    assert out["skipped"] and all("need 8" in s or "only" in s for s in out["skipped"])
    assert coaching_notes(stats, [ms], "Home", 99) == {"notes": [], "skipped": ["no contacts were attributed to this player"]}


def test_coaching_notes_flag_real_problems():
    evs = [ev(0, 0, "near", 7, "serve", "in_play")]
    for i in range(10):
        evs.append(ev(0, i + 1, "near", 7, "attack", "error" if i < 5 else "kill", shot="line", in_system=False))
    for i in range(10):
        evs.append(ev(0, 20 + i, "near", 7, "serve", "error" if i < 3 else "in_play"))
        evs.append(ev(0, 40 + i, "near", 7, "reception", "ok", grade=1))
    ms = MatchSet([rally(0, "near", "near", evs)], "Home", "Away")
    notes = " ".join(coaching_notes(compute_stats([ms]), [ms], "Home", 7)["notes"])
    assert "efficiency is +0.00 on 10 swings" in notes
    assert "100% of your 10 attacks were line" in notes
    assert "Out of system you gave away 5 of 10" in notes
    assert "missed 3 of 11 serves" in notes
    assert "Serve receive averages 1.00 on 10 passes" in notes


def test_report_prints_sample_sizes_and_insufficient_data():
    w, ms, tracks = sim(n=12)
    lines = report_lines([ms], [tracks], "Home", "Away", 30)
    text = "\n".join(lines)
    assert "TENDENCY REPORT - Home" in text
    assert "insufficient data (n=" in text            # per-rotation splits can't all be filled from 12 rallies
    assert "(n=" in text
    body = [l for l in lines[3:] if l.strip() and not l.strip().endswith(("distribution", "(team)", "defence", "attack"))]
    assert all("n=" in l for l in body)               # every tendency line carries its sample


@pytest.mark.slow
def test_end_to_end_tendency_report(match):
    ms = MatchSet(match["rallies"], "Home", "Away")
    lines = report_lines([ms], [match["labelled"]], "Home", "Away", match["src"].fps)
    print("\n" + "\n".join(lines[:14]))
    assert any("insufficient data" in l for l in lines)   # 8 rallies cannot support most tendencies
    assert all("n=" in l for l in lines[3:] if l.startswith("  "))
