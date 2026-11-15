import numpy as np

from vball.geometry import NET_H, NET_Y
from vball.synth.render import end_camera, make_clip, render
from vball.synth.world import Params, World


def test_world_is_deterministic():
    a, b = World(seed=3, n_rallies=3), World(seed=3, n_rallies=3)
    assert a.truth_contacts() == b.truth_contacts()


def test_ball_is_continuous_at_contacts():
    w = World(seed=5, n_rallies=6)
    for a, b in zip(w.flights, w.flights[1:]):
        if abs(a.t1 - b.t0) < 1e-9:
            assert np.allclose(a.pos(a.t1), b.p0, atol=1e-6)


def test_every_net_crossing_clears_the_net():
    w = World(seed=7, n_rallies=10)
    for fl in w.flights:
        a, b = fl.p0, fl.pos(fl.t1)
        if (a[1] - NET_Y) * (b[1] - NET_Y) < 0 and fl.t1 - fl.t0 > 0.3:
            tc = fl.t0 + (NET_Y - a[1]) / fl.v0[1]
            assert fl.pos(tc)[2] > NET_H


def test_rallies_are_well_formed():
    w = World(seed=9, n_rallies=12)
    for r in w.rallies:
        cs = r["contacts"]
        assert cs[0]["type"] == "serve" and cs[0]["team"] == r["serve_team"]
        assert r["winner"] in ("near", "far")
        ts = [c["t"] for c in cs]
        assert ts == sorted(ts) and r["t_end"] > ts[-1]
        for c in cs:
            if c["type"] in ("reception", "dig"):
                assert c["grade"] in (1, 2, 3)


def test_players_stay_on_their_own_side():
    w = World(seed=2, n_rallies=4)
    for t in np.arange(0, w.duration, 0.2):
        for p in w.players:
            y = w.player_pos(p.pid, t)[1]
            assert (y < NET_Y) == (p.team == "near")


def test_params_force_outcomes():
    w = World(seed=1, n_rallies=5, params=Params(serve_err=1.0))
    assert all(r["reason"] == "serve_error" for r in w.rallies)
    w = World(seed=1, n_rallies=5, params=Params(serve_err=0, ace=0, outcome_p=(0, 0, 0, 1),
                                                 shot_p=(1, 0, 0, 0), grade_p=(1, 0, 0)))
    assert all(r["reason"] == "blocked" for r in w.rallies)


def test_render_and_clip(tmp_path):
    w = World(seed=1, n_rallies=1)
    img = render(w, end_camera(), 2.0)
    assert img.shape == (540, 960, 3) and img.std() > 10
    path, cams, times = make_clip(w, t1=0.3, path=tmp_path / "c.mp4")
    import cv2
    cap = cv2.VideoCapture(str(path))
    n = 0
    while cap.read()[0]:
        n += 1
    assert n == len(times) == len(cams)
