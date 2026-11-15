import json

import numpy as np
import pytest

from vball.ball.events import (Event, EventConfig, Rally, build_rallies, in_court, rallies_from_json,
                               rallies_to_json)
from vball.ball.track import BallConfig, Flight, cameras, fit_flights, junction, link_tracklets, track_ball
from vball.court.track import CourtState
from vball.geometry import BALL_R, GVEC
from vball.synth.evaluate import evaluate_events, truth_flights, truth_tracks
from vball.synth.render import end_camera
from vball.synth.world import Params, World, make_flight

FPS = 30


def project_flights(flights, cam_fn, n_frames, noise=0.0, drop=(), seed=0):
    """Per-frame ball candidates from true flights."""
    rng = np.random.default_rng(seed)
    cands = [[] for _ in range(n_frames)]
    for f in range(n_frames):
        t = f / FPS
        for fl in flights:
            if fl.t0 <= t < fl.t1 and f not in drop:
                uv = cam_fn(t).project([fl.pos(t)])[0][0] + rng.normal(0, noise, 2)
                cands[f].append((float(uv[0]), float(uv[1]), 4.0))
    return cands


def states_for(cam_fn, n):
    return [CourtState(cam_fn(f / FPS).H, 1.0, "lines") for f in range(n)]


CAM = lambda t: end_camera(pan=1.5 * np.sin(0.6 * t))


def test_cameras_recover_projection():
    st = states_for(CAM, 60)
    Ps = cameras(st, (960, 540))
    X = np.array([4.0, 12.0, 3.0, 1.0])
    for f in (0, 30, 59):
        q = Ps[f] @ X
        assert np.allclose(q[:2] / q[2], CAM(f / FPS).project([X[:3]])[0][0], atol=0.5)
    assert cameras([CourtState(None, 0, "lost")], (960, 540)) == [None]


def test_tracklets_follow_ball_and_ignore_static_and_slow_things():
    fl = make_flight(0.0, (4, 1, 2.5), (5, 15, 0.7), 1.3)
    cands = project_flights([fl], CAM, 39)
    for f in range(39):
        cands[f].append((100.0, 100.0, 3.0))                 # a light on the wall
        cands[f].append((300.0 + 0.5 * f, 400.0, 3.0))       # a yellow shoe walking
    tr = link_tracklets(cands)
    pts = [p for t in tr for p in t]
    assert len(pts) >= 28                                    # (slow frames at the apex may be cut out)
    for f, u, v in pts:                                      # every kept point is the real ball
        assert np.allclose((u, v), CAM(f / FPS).project([fl.pos(f / FPS)])[0][0], atol=0.01)


def test_ball_track_survives_meeting_a_slow_distractor():
    fl = make_flight(0.0, (4, 1, 2.5), (5, 15, 0.7), 1.3)
    cands = project_flights([fl], CAM, 39)
    end = cands[38][0]
    for f in range(39, 120):                                 # a shoe sits where the ball ended
        cands.append([(end[0] + 0.3 * (f - 39), end[1], 3.0)])
    tr = link_tracklets(cands)
    assert tr and max(t[-1][0] for t in tr) <= 48            # the shoe's 80 frames are not ball


def test_flights_split_at_a_contact_and_meet_there():
    a = make_flight(0.0, (4, -0.9, 2.5), (6, 14, 0.7), 1.25)
    b = make_flight(a.t1, a.pos(a.t1), (5.5, 10, 2.1), 1.4)
    n = int(b.t1 * FPS)
    flights, _ = track_ball(project_flights([a, b], CAM, n, noise=0.3), states_for(CAM, n), (960, 540), FPS)
    assert len(flights) == 2
    j = junction(*flights)
    assert j.kind == "contact"
    assert abs(j.t - a.t1) < 0.08
    assert np.linalg.norm(j.pos - a.pos(a.t1)) < 0.5


def test_flight_stitched_across_missing_detections():
    a = make_flight(0.0, (4, -0.9, 2.5), (6, 14, 0.7), 1.25)
    n = int(a.t1 * FPS)
    flights, _ = track_ball(project_flights([a], CAM, n, noise=0.3, drop=set(range(12, 22))),
                            states_for(CAM, n), (960, 540), FPS)
    assert len(flights) == 1 and flights[0].t1 - flights[0].t0 > 1.0


def test_landing_is_a_bounce_and_is_located_on_the_floor():
    a = make_flight(0.0, (2, 8.3, 2.9), (7.5, 15.0, BALL_R), 0.6)
    v = a.v0 + GVEC * 0.6
    b = Flight(a.t1, a.t1 + 0.45, a.pos(a.t1), np.array([v[0] * 0.55, v[1] * 0.55, -v[2] * 0.55]), 0, 10)
    pre = make_flight(-1.2, (5.5, 8.0, 2.1), (2, 8.3, 2.9), 1.2)
    n0 = int(1.2 * FPS)
    shift = lambda fl: Flight(fl.t0 + 1.2, fl.t1 + 1.2, fl.p0, fl.v0, 0, 10)
    fls = [shift(pre), shift(a), shift(b)]
    n = int(fls[-1].t1 * FPS)
    flights, _ = track_ball(project_flights(fls, CAM, n, noise=0.3), states_for(CAM, n), (960, 540), FPS)
    assert len(flights) == 3
    j = junction(flights[1], flights[2])
    assert j.kind == "bounce"
    assert np.linalg.norm(j.pos[:2] - np.array([7.5, 15.0])) < 0.6      # short fast flight, pinned at both ends


def test_no_candidates_no_flights():
    flights, tr = track_ball([[] for _ in range(50)], states_for(CAM, 50), (960, 540), FPS)
    assert flights == [] and tr == []


def test_flight_landing_and_net_crossing():
    fl = make_flight(0.0, (4, 5, 2.0), (4, 13, BALL_R), 1.0)
    t, p = Flight(0, 1, fl.p0, fl.v0, 0, 30).landing()
    assert t == pytest.approx(1.0, abs=1e-6) and np.allclose(p[:2], (4, 13), atol=1e-6)
    assert Flight(0, 1, fl.p0, fl.v0, 0, 30).crosses_net(0, 1)


# ------------------------------------------------------------- event rules
def rules_on_truth(seed, n=10, **params):
    w = World(seed=seed, n_rallies=n, params=Params(**params) if params else None)
    times = np.arange(0, w.duration, 1 / FPS)
    return w, build_rallies(truth_flights(w), truth_tracks(w, times), FPS)


def test_rules_reproduce_truth_given_perfect_tracking():
    """Classification logic in isolation: perfect ball and player data in,
    the simulator's own labels must come out."""
    tot = {}
    for seed in range(6):
        w, rallies = rules_on_truth(seed)
        for k, v in evaluate_events(w, rallies).items():
            if isinstance(v, int):
                tot[k] = tot.get(k, 0) + v
    print("\nRULES ON PERFECT INPUT", tot)
    assert tot["matched"] == tot["true_contacts"] and tot["spurious"] == 0
    assert tot["type_ok"] == tot["true_contacts"]
    assert tot["player_ok"] == tot["true_contacts"]
    assert tot["attack_outcome_ok"] == tot["attacks"]
    assert tot["grade_ok"] == tot["passes"]
    assert tot["winner_ok"] == tot["true_rallies"]
    assert tot["shot_ok"] >= 0.85 * tot["attacks"]       # line vs roll overlap near the speed threshold


@pytest.mark.parametrize("params,reason,last_type,outcome", [
    (dict(serve_err=1.0), "serve_error", "serve", "error"),
    (dict(serve_err=0, ace=1.0), "ace", "serve", "ace"),
    (dict(serve_err=0, ace=0, outcome_p=(0, 0, 0, 1), shot_p=(1, 0, 0, 0), grade_p=(1, 0, 0)), "blocked", "block", "stuff"),
    (dict(serve_err=0, ace=0, outcome_p=(0, 0, 1, 0), shot_p=(1, 0, 0, 0), grade_p=(1, 0, 0)), "attack_error", "attack", "error"),
    (dict(serve_err=0, ace=0, outcome_p=(1, 0, 0, 0), shot_p=(0, 0, 1, 0), grade_p=(1, 0, 0)), "kill", "attack", "kill"),
])
def test_each_rally_ending(params, reason, last_type, outcome):
    w, rallies = rules_on_truth(3, n=5, **params)
    assert len(rallies) == 5
    for r, tr in zip(rallies, w.rallies):
        assert r.winner == tr["winner"] and r.reason == reason
        assert r.events[-1].type == last_type and r.events[-1].outcome == outcome
    if reason == "blocked":
        att = rallies[0].events[-2]
        assert att.type == "attack" and att.outcome == "blocked"
    if reason == "kill":
        assert all(r.events[-1].attrs["shot"] == "tip" for r in rallies)


def test_free_ball_and_out_of_system():
    w, rallies = rules_on_truth(2, n=6, serve_err=0, ace=0, grade_p=(0, 0, 1), freeball_when_oos=1.0, max_exchanges=1)
    ev = [e for r in rallies for e in r.events]
    assert any(e.type == "free_ball" for e in ev)
    for e in ev:
        if e.type == "free_ball":
            assert e.attrs["in_system"] is False
        if e.type in ("reception",):
            assert e.attrs["grade"] == 1


def test_contact_without_tracked_player_is_flagged_not_invented():
    w = World(seed=1, n_rallies=2)
    rallies = build_rallies(truth_flights(w), None, FPS)
    ev = [e for r in rallies for e in r.events]
    assert ev and all(e.number is None and "no_player_tracked" in e.flags and e.conf <= 0.5 for e in ev)


def test_unknown_winner_filled_from_next_serve_and_flagged():
    w = World(seed=1, n_rallies=3)
    fl = truth_flights(w)
    end0 = w.rallies[0]["t_end"]
    # cut the final flight of rally 0 short and remove its bounce: the ball is "lost"
    keep = []
    for f in fl:
        if abs(f.t1 - end0) < 1e-6:
            keep.append(Flight(f.t0, f.t0 + 0.12, f.p0, f.v0, 0, 5))
        elif abs(f.t0 - end0) < 1e-6:
            continue
        else:
            keep.append(f)
    times = np.arange(0, w.duration, 1 / FPS)
    rallies = build_rallies(keep, truth_tracks(w, times), FPS)
    r0 = rallies[0]
    if "ball_lost_before_landing" in r0.flags:
        assert "winner_inferred_from_next_serve" in r0.flags
        assert r0.winner == rallies[1].serve_team
    else:
        assert "landing_extrapolated" in r0.flags and r0.conf < 1.0


def test_close_line_call_is_marked():
    cfg = EventConfig()
    a = make_flight(0.0, (4, -0.9, 2.5), (4, 17.9, BALL_R), 1.3)
    fl = [Flight(a.t0, a.t1, a.p0, a.v0, 0, 30)]
    r, = build_rallies(fl, None, FPS, cfg)
    assert "close_line_call" in r.flags and r.conf < 1.0
    assert in_court(4, 18.05) and not in_court(4, 18.3)


def test_rallies_json_roundtrip():
    w, rallies = rules_on_truth(1, n=3)
    back = rallies_from_json(json.loads(json.dumps(rallies_to_json(rallies))))
    assert [e.type for r in back for e in r.events] == [e.type for r in rallies for e in r.events]
    assert back[0].winner == rallies[0].winner and isinstance(back[0].events[0], Event)


# ------------------------------------------------------------- end to end
@pytest.mark.slow
def test_end_to_end_events_on_rendered_match(match):
    ev = evaluate_events(match["world"], match["rallies"])
    print("\nEVENTS", ev)
    assert ev["matched"] >= 0.9 * ev["true_contacts"]
    assert ev["spurious"] <= 3
    assert ev["type_ok"] >= 0.85 * ev["matched"]
    assert ev["winner_ok"] >= ev["true_rallies"] - 1
    assert ev["attack_outcome_ok"] >= 0.85 * ev["attacks"]
    assert ev["grade_ok"] >= 0.85 * ev["passes"]
    assert ev["player_wrong"] <= 0.2 * ev["matched"]


@pytest.mark.slow
def test_without_corrections_unread_players_are_unknown_not_guessed(match):
    ev = [e for r in match["rallies_raw"] for e in r.events]
    known = {p.number for p in match["world"].players}
    assert all(e.number is None or e.number in known for e in ev)
    assert any(e.number is None and "player_number_unknown" in e.flags for e in ev)
