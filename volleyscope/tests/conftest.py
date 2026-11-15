import numpy as np
import pytest

from vball.synth.render import end_camera
from vball.synth.world import World

GRID = np.array([(x, y) for x in np.linspace(0, 9, 7) for y in np.linspace(0, 18, 13)])


def court_error(H, cam, size=(960, 540), visible_only=True):
    """Court-space error (m) of an estimated court->image homography against
    the true camera, over a grid of floor points."""
    from vball.geometry import apply_h
    uv = apply_h(cam.H, GRID)
    est = apply_h(np.linalg.inv(H), uv)
    e = np.linalg.norm(est - GRID, axis=1)
    if visible_only:
        vis = (uv[:, 0] > 0) & (uv[:, 0] < size[0]) & (uv[:, 1] > 0) & (uv[:, 1] < size[1])
        e = e[vis]
    return e


@pytest.fixture(scope="session")
def world():
    return World(seed=1, n_rallies=3)


@pytest.fixture(scope="session")
def pan_cam():
    return lambda t: end_camera(pan=3.0 * np.sin(0.8 * t), zoom=1.0 + 0.35 * np.sin(0.5 * t))


@pytest.fixture(scope="session")
def match(tmp_path_factory):
    """One rendered synthetic clip (through a real video file) analysed once and
    shared by the end-to-end tests of every phase."""
    from vball.pipeline import Config, run_ball, run_court, run_players
    from vball.players.track import apply_corrections
    from vball.synth.evaluate import truth_corrections
    import copy
    from vball.synth.render import make_clip
    from vball.video import FrameSource
    w = World(seed=11, n_rallies=8)        # includes an ace, a free ball, digs and a block
    cam_fn = lambda t: end_camera(pan=2.0 * np.sin(0.5 * t), zoom=1.0 + 0.15 * np.sin(0.3 * t))
    path = tmp_path_factory.mktemp("clip") / "match.mp4"
    _, cams, times = make_clip(w, cam_fn, path=path)
    src = FrameSource(str(path))
    cfg = Config()
    states = run_court(src, cfg)
    tracks, balls, info = run_players(src, states, cfg)
    # what a reviewer would do with the corrections file: label every track
    labelled = apply_corrections(copy.deepcopy(tracks), truth_corrections(w, times, tracks))
    flights, rallies = run_ball(balls, states, labelled, src.size, src.fps, cfg)
    _, rallies_raw = run_ball(balls, states, tracks, src.size, src.fps, cfg)
    return {"world": w, "cams": cams, "times": times, "src": src, "cfg": cfg, "states": states,
            "tracks": tracks, "labelled": labelled, "balls": balls, "info": info, "path": path,
            "flights": flights, "rallies": rallies, "rallies_raw": rallies_raw}
