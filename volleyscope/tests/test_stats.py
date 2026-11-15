import json

import numpy as np
import pytest

from vball.ball.events import Event, Rally, build_rallies
from vball.stats.core import (MatchSet, Stat, annotate_rotations, classify_lost_point, compute_stats,
                              player_stats, stats_to_json)
from vball.synth.evaluate import truth_flights, truth_tracks
from vball.synth.world import World


def ev(rally, idx, team, number, typ, outcome=None, conf=1.0, pos=(4, 4, 1), **attrs):
    return Event(rally, idx, float(idx), idx, list(pos), team, None, number, typ, outcome, attrs, conf)


def rally(rid, serve_team, winner, events, reason="kill", conf=1.0, landing=None):
    return Rally(rid, serve_team, events, 0.0, 5.0, landing, winner, reason, conf)


def simple_match():
    """Three hand-written rallies with known totals."""
    r0 = rally(0, "near", "far", [
        ev(0, 0, "near", 7, "serve", "in_play"),
        ev(0, 1, "far", 5, "reception", "ok", grade=3),
        ev(0, 2, "far", 8, "set", "assist", in_system=True),
        ev(0, 3, "far", 12, "attack", "kill", shot="tip", in_system=True, target=[4, 7.5])], landing=[4, 7.5, 0.1])
    r1 = rally(1, "far", "near", [
        ev(1, 0, "far", 5, "serve", "in_play"),
        ev(1, 1, "near", 7, "reception", "ok", grade=1),
        ev(1, 2, "near", 9, "set", "set", in_system=False),
        ev(1, 3, "near", 7, "attack", "kill", shot="cross", in_system=False, target=[8, 17])], landing=[8, 17, 0.1])
    r2 = rally(2, "near", "far", [
        ev(2, 0, "near", 7, "serve", "error")], reason="serve_error", landing=[4, 19.5, 0.1])
    return MatchSet([r0, r1, r2], "Home", "Away")


def test_stat_states():
    assert Stat("x").value is None and Stat("x").status == "no data" and Stat("x").text() == "no data"
    assert Stat("x", 1, 2).status == "low sample"
    assert Stat("x", 6, 10).status == "ok" and Stat("x", 6, 10).text() == "60% (6/10)"
    assert Stat("x", 6, 10, uncertain=4).status == "uncertain"
    assert Stat("x", 3, kind="count").value == 3
    assert Stat("x", 2, 10, kind="ratio").text().startswith("+0.200")


def test_player_stats_from_hand_written_match():
    s = compute_stats([simple_match()])
    me = player_stats(s, "Home", 7)["stats"]
    assert me["attack_attempts"].value == 1 and me["kills"].value == 1
    assert me["kill_pct"].value == 1.0 and me["kill_pct"].status == "low sample"
    assert me["attack_eff"].value == 1.0
    assert me["serve_in_pct"].num == 1 and me["serve_in_pct"].den == 2
    assert me["serve_errors"].value == 1
    assert me["reception_avg"].value == 1.0 and me["reception_1"].value == 1.0
    assert me["error_rate"].num == 1 and me["error_rate"].den == 4
    setter = player_stats(s, "Away", 8)["stats"]
    assert setter["assist_pct"].value == 1.0 and setter["sets"].value == 1
    assert setter["kill_pct"].value is None and setter["kill_pct"].status == "no data"   # never fabricated
    assert setter["reception_avg"].value is None


def test_efficiency_counts_errors_and_blocks():
    evs = [ev(0, 0, "near", 4, "serve", "in_play")]
    for i, oc in enumerate(["kill", "kill", "error", "blocked", "dug", "dug"]):
        evs.append(ev(0, i + 1, "near", 9, "attack", oc))
    s = compute_stats([MatchSet([rally(0, "near", "near", evs)], "A", "B")])
    st = player_stats(s, "A", 9)["stats"]
    assert st["kill_pct"].value == pytest.approx(2 / 6)
    assert st["attack_eff"].value == pytest.approx((2 - 1 - 1) / 6)
    assert st["attack_errors"].value == 2


def test_rotation_and_score_tracking():
    ms = simple_match()
    rots, score = annotate_rotations(ms)
    assert score == {"near": 1, "far": 2}
    assert [r["rotation"] for r in rots] == [{"near": 1, "far": 1}, {"near": 1, "far": 2}, {"near": 2, "far": 2}]
    assert rots[1]["score_before"] == {"near": 0, "far": 1}
    assert all(not r["flags"] for r in rots)


def test_rotation_flagged_when_serve_or_winner_unknown():
    ms = simple_match()
    ms.rallies[1].serve_team = None
    rots, _ = annotate_rotations(ms)
    assert "server_assumed" in rots[1]["flags"] and "rotation_uncertain" in rots[2]["flags"]
    assert rots[1]["server"] == "far"                # previous winner
    ms2 = simple_match()
    ms2.rallies[0].winner = None
    rots2, score2 = annotate_rotations(ms2)
    assert "rotation_uncertain" in rots2[1]["flags"] and score2 == {"near": 1, "far": 1}


def test_team_side_out_and_lost_point_breakdown():
    s = compute_stats([simple_match()])
    home, away = s["teams"]["Home"], s["teams"]["Away"]
    assert home["side_out"].num == 1 and home["side_out"].den == 1          # received once, won it
    assert away["side_out"].num == 2 and away["side_out"].den == 2
    assert home["points_lost"] == 2
    assert home["situations"]["tip"].num == 1 and home["situations"]["tip"].den == 2
    assert home["by_rotation"][1]["lost"] == 1 and home["by_rotation"][2]["lost"] == 1
    lp = home["lost_points"]
    assert lp[0]["cause"] == "kill" and lp[0]["by_team"] == "Away" and lp[0]["by_number"] == 12 and lp[0]["tags"] == ["tip"]
    assert lp[1]["cause"] == "serve_error" and lp[1]["by_team"] == "Home" and lp[1]["by_number"] == 7
    away_lost, = away["lost_points"]
    assert set(away_lost["tags"]) == {"deep_corner"}                        # (8, 17): deep right corner
    assert s["scores"] == [{"Home": 1, "Away": 2}]


def test_broken_play_and_out_of_system_tags():
    r = rally(0, "far", "far", [
        ev(0, 0, "far", 5, "serve", "in_play"),
        ev(0, 1, "near", 7, "reception", "ok", grade=1),
        ev(0, 2, "near", 9, "set", "set", in_system=False),
        ev(0, 3, "near", 4, "attack", "error", in_system=False)], reason="attack_error")
    cause, resp, tags = classify_lost_point(r, "near")
    assert cause == "attack_error" and resp.number == 4
    assert set(tags) == {"broken_play", "out_of_system_attack"}


def test_unidentified_contacts_are_kept_apart():
    evs = [ev(0, 0, "near", None, "serve", "ace")]
    s = compute_stats([MatchSet([rally(0, "near", "near", evs, "ace")], "A", "B")])
    assert list(s["players"]) == [("A", None)]
    assert s["players"][("A", None)]["label"] == "unidentified"
    assert s["players"][("A", None)]["stats"]["aces"].value == 1


def test_low_confidence_events_mark_stat_uncertain():
    evs = [ev(0, 0, "near", 4, "serve", "in_play")]
    for i in range(8):
        evs.append(ev(0, i + 1, "near", 9, "attack", "kill", conf=0.5 if i < 4 else 1.0))
    s = compute_stats([MatchSet([rally(0, "near", "near", evs)], "A", "B")])
    k = player_stats(s, "A", 9)["stats"]["kill_pct"]
    assert k.uncertain == 4 and k.status == "uncertain" and "[uncertain]" in k.text()


def test_unknown_winner_rally_not_counted_in_side_out():
    ms = simple_match()
    ms.rallies[1].winner = None
    s = compute_stats([ms])
    assert s["teams"]["Home"]["side_out"].den == 0 and s["teams"]["Home"]["side_out"].value is None
    assert s["teams"]["Home"]["unknown_winner"] == 1


def test_per_rotation_split_and_two_sets_with_side_switch():
    a = simple_match()
    b = simple_match()
    b.near, b.far = "Away", "Home"                   # teams changed ends
    s = compute_stats([a, b])
    assert s["teams"]["Home"]["points_lost"] == 2 + 1
    me = s["players"][("Home", 7)]
    assert me["stats"]["serve_in_pct"].den == 2      # set 1 only; in set 2 number 7 on that side is Away's
    assert set(me["by_rotation"]) == {1, 2}
    assert me["by_rotation"][2]["serve_errors"].value == 1


def test_stats_json_serialisable():
    out = json.dumps(stats_to_json(compute_stats([simple_match()])))
    assert '"no data"' in out and "Home" in out


def test_stats_on_simulated_match_are_consistent():
    w = World(seed=4, n_rallies=25)
    times = np.arange(0, w.duration, 1 / 30)
    rallies = build_rallies(truth_flights(w), truth_tracks(w, times), 30)
    s = compute_stats([MatchSet(rallies, "Home", "Away")])
    truth = w.truth_contacts()
    for p in w.players:
        team = "Home" if p.team == "near" else "Away"
        mine = [c for c in truth if c["pid"] == p.pid]
        st = s["players"].get((team, p.number))
        if not mine:
            assert st is None
            continue
        att = [c for c in mine if c["type"] == "attack"]
        assert st["stats"]["attack_attempts"].value == len(att)
        assert st["stats"]["kills"].value == sum(c["outcome"] == "kill" for c in att)
        rec = [c for c in mine if c["type"] == "reception"]
        if rec:
            assert st["stats"]["reception_avg"].value == pytest.approx(np.mean([c["grade"] for c in rec]))
        else:
            assert st["stats"]["reception_avg"].value is None
    wins = sum(r["winner"] == "near" for r in w.rallies)
    assert s["scores"][0] == {"Home": wins, "Away": 25 - wins}
    assert s["teams"]["Home"]["points_lost"] == 25 - wins
    # rotations follow the simulator's own bookkeeping
    rots, _ = annotate_rotations(MatchSet(rallies))
    for info, tr in zip(rots, w.rallies):
        assert info["rotation"]["near"] - 1 == tr["rotation"]["near"] % 6
        assert info["rotation"]["far"] - 1 == tr["rotation"]["far"] % 6


@pytest.mark.slow
def test_end_to_end_stats_close_to_truth(match):
    w, times = match["world"], match["times"]
    ref = compute_stats([MatchSet(build_rallies(truth_flights(w), truth_tracks(w, times), 30), "Home", "Away")])
    got = compute_stats([MatchSet(match["rallies"], "Home", "Away")])
    print("\nSCORE truth", ref["scores"], "pipeline", got["scores"])
    assert got["scores"] == ref["scores"]
    for team in ("Home", "Away"):
        assert got["teams"][team]["side_out"].den == ref["teams"][team]["side_out"].den
        assert got["teams"][team]["side_out"].num == ref["teams"][team]["side_out"].num
    ka = sum(p["stats"]["kills"].value for p in got["players"].values())
    kr = sum(p["stats"]["kills"].value for p in ref["players"].values())
    aa = sum(p["stats"]["attack_attempts"].value for p in got["players"].values())
    ar = sum(p["stats"]["attack_attempts"].value for p in ref["players"].values())
    print("kills", ka, "of true", kr, "| attack attempts", aa, "of true", ar)
    assert abs(ka - kr) <= 1 and abs(aa - ar) <= 2
