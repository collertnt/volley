import numpy as np
import pytest

from test_stats import ev, rally
from vball.ball.events import build_rallies
from vball.players.track import Identity, PlayerTracks
from vball.pov.core import (LUNGE, REACTION, SPEED, Scenario, analyse, cover_radius, reconstruct, render_pov,
                            render_topdown, score_placement, shots_from_opponent, shots_from_scenario,
                            suggest_placement, validate_placement)
from vball.stats.core import MatchSet
from vball.synth.evaluate import truth_flights, truth_tracks
from vball.synth.world import Params, World

BASE = [(1.5, 2.5), (4.5, 2.0), (7.5, 2.5), (1.5, 6.8), (4.5, 7.3), (7.5, 6.8)]


def scenario(landing, T=0.6, defenders=BASE, shot="tip"):
    return Scenario("near", [2.0, 9.7, 2.9], list(landing), T,
                    [{"label": f"#{k + 1}", "number": k + 1, "xy": list(p)} for k, p in enumerate(defenders)], shot,
                    {"number": 12, "type": "attack"}, [{"label": "#12", "number": 12, "xy": [2.0, 9.8]}])


def test_cover_radius_model():
    assert cover_radius(0.2) == LUNGE
    assert cover_radius(1.3) == pytest.approx(LUNGE + SPEED * (1.3 - REACTION))


def test_reachable_ball_is_called_execution_not_positioning():
    a = analyse(scenario((4.6, 2.6), T=0.6))
    assert a["reachable"] and a["kind"] == "execution" and "reachable" in a["diagnosis"]
    assert a["defenders"][0]["label"] == "#2" and a["defenders"][0]["shortfall"] == 0


def test_tip_behind_the_block_with_everyone_deep():
    deep = [(1.5, 1.5), (4.5, 1.0), (7.5, 1.5), (1.5, 8.6), (4.5, 8.6), (7.5, 8.6)]     # three at the net, three deep
    a = analyse(scenario((4.5, 6.2), T=0.5, defenders=deep))
    assert not a["reachable"] and a["kind"] == "too_deep"
    assert "Everyone was deep" in a["diagnosis"]
    n = a["defenders"][0]
    assert n["shortfall"] == pytest.approx(n["dist"] - cover_radius(0.5))
    # standing where the analysis says would have been just enough
    moved = [list(p) for p in deep]
    moved[n["number"] - 1] = n["needed_xy"]
    assert analyse(scenario((4.5, 6.2), T=0.5, defenders=moved))["reachable"]


def test_ball_into_a_gap_names_the_two_defenders():
    spread = [(0.5, 3.0), (8.5, 3.0), (0.5, 7.5), (8.5, 7.5), (0.5, 0.5), (8.5, 0.5)]
    a = analyse(scenario((4.5, 3.0), T=0.45, defenders=spread, shot="cross"))
    assert a["kind"] == "too_spread" and "gap between #1 and #2" in a["diagnosis"] and "8.0 m apart" in a["diagnosis"]


def test_deep_ball_over_a_shallow_defence():
    shallow = [(1.5, 6.0), (4.5, 5.5), (7.5, 6.0), (1.5, 8.0), (4.5, 8.2), (7.5, 8.0)]
    a = analyse(scenario((4.5, 0.6), T=0.5, defenders=shallow, shot="line"))
    assert a["kind"] == "too_shallow"


def test_no_defenders_tracked_says_so():
    a = analyse(scenario((4.5, 3.0), defenders=[]))
    assert a["kind"] == "unknown" and "cannot be judged" in a["diagnosis"]


def lost_rally():
    return rally(0, "near", "far", [
        ev(0, 0, "near", 7, "serve", "in_play"),
        ev(0, 1, "far", 5, "reception", "ok", grade=3),
        ev(0, 2, "far", 8, "set", "assist"),
        ev(0, 3, "far", 12, "attack", "kill", pos=(2.0, 9.7, 2.9), shot="tip", target=[4.5, 6.2])], landing=[4.5, 6.2, 0.1])


def tracks_at(frame, positions, team="near"):
    ids = [Identity(k, team, k + 1, 1.0, np.array([frame]), np.array([p], float), np.array([True]), np.array([False]))
           for k, p in enumerate(positions)]
    return PlayerTracks(ids, 30)


def test_reconstruct_from_a_rally():
    r = lost_rally()
    r.t_end = 3.6
    scn, why = reconstruct(r, tracks_at(3, BASE[:4]), "near", 30)
    assert why is None and scn.shot == "tip" and scn.landing == [4.5, 6.2] and scn.T == pytest.approx(0.6)
    assert len(scn.defenders) == 4 and scn.missing == 2
    assert any("only 4 of 6" in n for n in scn.notes)                 # says what it could not see
    assert "2 defender(s) were not tracked" in analyse(scn)["diagnosis"]
    assert scn.to_json()["defenders"][0]["label"] == "#1"


def test_reconstruct_drops_duplicate_tracks_of_one_player():
    r = lost_rally()
    tr = tracks_at(3, BASE + [(4.6, 2.1)])
    tr.identities[6].number = 2                       # same number as an existing track
    scn, _ = reconstruct(r, tr, "near", 30)
    assert len(scn.defenders) == 6 and any("duplicate" in n for n in scn.notes)
    assert suggest_placement(shots_from_scenario(scn), "near", 9, iters=20)[1] is not None


def test_reconstruct_refuses_when_there_is_nothing_to_show():
    r = lost_rally()
    assert reconstruct(r, None, "far", 30)[1] == "this rally was not lost by that side"
    own = rally(0, "near", "far", [ev(0, 0, "near", 7, "serve", "error")], reason="serve_error")
    scn, why = reconstruct(own, None, "near", 30)
    assert scn is None and "own error" in why


def test_pov_render_shows_landing_and_differs_per_player():
    scn = scenario((4.5, 6.2), T=0.5, defenders=[(1.5, 1.5), (4.5, 1.0), (7.5, 1.5), (1.5, 8.6), (4.5, 8.6), (7.5, 8.6)])
    a = analyse(scn)
    views = [render_pov(scn, k, analysis=a) for k in range(6)]
    assert all(v.shape[1] == 960 and v.shape[0] > 540 for v in views)
    assert not np.array_equal(views[0], views[1])
    red = lambda im: ((im[..., 2] > 200) & (im[..., 1] < 80) & (im[..., 0] < 80)).sum()
    assert red(views[1]) > 20                 # back-row player sees the landing ring in front of him
    yellow = lambda im: ((im[..., 2] > 200) & (im[..., 1] > 180) & (im[..., 0] < 110)).sum()
    assert yellow(views[1]) > 30              # and the ball path


def test_topdown_render():
    scn = scenario((4.5, 6.2))
    img = render_topdown(scn)
    assert img.shape[0] > img.shape[1] and img.std() > 5
    assert render_topdown(scn, placement=[[4.5, 6.0]]).shape == img.shape


def test_simulator_scores_and_validates():
    shots = [{"landing": [4.5, 6.2], "T": 0.5, "weight": 1.0}, {"landing": [8.0, 1.0], "T": 0.6, "weight": 1.0}]
    full = score_placement([[4.5, 6.0], [7.8, 1.2]], shots, "near")
    assert full["score"] == 100.0 and full["reached"] == 2
    half = score_placement([[4.5, 6.0]], shots, "near")
    assert half["score"] == 50.0 and half["per_shot"][1]["shortfall"] > 2
    partial = score_placement([[4.5, 6.0], [8.0, 1.0 + cover_radius(0.6) + 1.0]], shots, "near")
    assert partial["score"] == pytest.approx(75.0)
    bad = score_placement([[4.5, 12.0]], shots, "near")
    assert bad["score"] is None and "wrong side of the net" in bad["problems"][0]
    assert validate_placement([[4.5, 3]] * 7, "near") == ["place between 1 and 6 players"]
    assert validate_placement([[12, 3]], "near") == ["player 1 is outside the sidelines"]
    assert score_placement([[4.5, 12.0]], shots, "far")["score"] is not None
    assert score_placement([[4.5, 3]], [], "near")["score"] is None


def test_suggested_placement_beats_base_formation():
    rng = np.random.default_rng(1)
    shots = [{"landing": [float(rng.uniform(1, 8)), float(rng.uniform(4.5, 7.5))], "T": 0.55, "weight": 1.0} for _ in range(12)]
    base = score_placement([list(p) for p in BASE], shots, "near")["score"]
    best, sc = suggest_placement(shots, "near", seed=0, iters=150)
    assert sc >= base and len(best) == 6 and not validate_placement(best, "near")


def test_scenario_feeds_simulator_and_actual_placement_scores_zero_for_unreached_ball():
    scn = scenario((4.5, 6.2), T=0.5, defenders=[(1.5, 1.5), (4.5, 1.0), (7.5, 1.5), (1.5, 8.6), (4.5, 8.6), (7.5, 8.6)])
    shots = shots_from_scenario(scn)
    actual = score_placement([d["xy"] for d in scn.defenders], shots, "near")
    fixed = score_placement([analyse(scn)["defenders"][0]["needed_xy"]], shots, "near")
    assert actual["score"] < 100 and fixed["score"] == 100.0


def test_opponent_shot_chart_from_simulated_match():
    w = World(seed=4, n_rallies=20)
    times = np.arange(0, w.duration, 1 / 30)
    tracks = truth_tracks(w, times)
    ms = MatchSet(build_rallies(truth_flights(w), tracks, 30), "Home", "Away")
    shots = shots_from_opponent([ms], "Away", "near")
    true = [c for c in w.truth_contacts() if c["type"] == "attack" and c["team"] == "far" and c["outcome"] in ("kill", "dug")]
    assert len(shots) == len(true) and all(s["landing"][1] < 9 for s in shots)
    # every lost point either reconstructs or explains why not
    for r in ms.rallies:
        if r.winner == "far":
            scn, why = reconstruct(r, tracks, "near", 30)
            assert (scn is None) != (why is None)
            if scn is not None:
                assert len(scn.defenders) == 6 and analyse(scn)["kind"] != "unknown"


@pytest.mark.slow
def test_end_to_end_pov_on_rendered_match(match, tmp_path):
    import cv2
    done = 0
    for r in match["rallies"]:
        for side in ("near", "far"):
            scn, why = reconstruct(r, match["labelled"], side, match["src"].fps)
            if scn is None:
                continue
            a = analyse(scn)
            img = render_pov(scn, 0, analysis=a)
            assert cv2.imwrite(str(tmp_path / f"pov_{r.id}_{side}.png"), img)
            done += 1
    print("\nPOV scenarios reconstructed:", done)
    assert done >= 3
