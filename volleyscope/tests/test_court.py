import json

import cv2
import numpy as np
import pytest

from conftest import court_error
from vball.court.detect import CourtConfig, canonical, detect_court, line_mask, refine, support
from vball.court.track import CourtTracker, homography_from_points, load_calibration, track_video
from vball.geometry import apply_h
from vball.synth.render import Camera, end_camera, render

FLIP = np.array([[-1, 0, 9.0], [0, -1, 18.0], [0, 0, 1.0]])


def sym_error(H, cam):
    """Error allowing the court's 180-degree symmetry (end labels are a convention)."""
    return min(court_error(H, cam).max(), court_error(H @ FLIP, cam).max())


def test_line_mask_finds_lines_not_floor(world):
    cam = end_camera()
    img = render(world, cam, 0.5, players=False, ball=False)
    m = line_mask(img, CourtConfig())
    on = apply_h(cam.H, [(4.5, 0), (0, 9), (9, 5), (4.5, 6)]).astype(int)
    off = apply_h(cam.H, [(4.5, 3), (2, 15), (7, 10.5)]).astype(int)
    assert all(m[v - 1:v + 2, u - 1:u + 2].any() for u, v in on)
    assert not any(m[v, u] for u, v in off)


@pytest.mark.parametrize("name,cam,tol", [
    ("end", end_camera(), 0.15),
    ("end_pan_zoom", end_camera(pan=2.5, zoom=1.2), 0.15),
    ("corner", Camera((-6, -8, 7), (4.5, 9, 0), 640), 0.2),
    ("side", Camera((-12, 9, 7), (4.5, 9, 0), 560), 0.2),
])
def test_global_detection_with_players(world, name, cam, tol):
    img = render(world, cam, 4.0)
    H, info = detect_court(img)
    assert H is not None, info
    assert sym_error(H, cam) < tol
    assert info["determined"]


def test_end_camera_orientation_is_canonical(world):
    cam = end_camera()
    H, _ = detect_court(render(world, cam, 4.0))
    assert court_error(H, cam).max() < 0.15        # no flip: y=0 is the near end


def test_detection_returns_none_without_a_court():
    rng = np.random.default_rng(0)
    img = rng.integers(90, 110, (540, 960, 3), dtype=np.uint8)
    H, info = detect_court(img)
    assert H is None and "reason" in info


def test_canonical_is_idempotent():
    cam = end_camera()
    H = canonical(cam.H)
    assert np.allclose(canonical(H), H)
    assert np.allclose(canonical(cam.H @ FLIP), H)


def test_refine_recovers_from_offset(world):
    cam = end_camera()
    img = render(world, cam, 4.0)
    shift = np.array([[1, 0, 7.0], [0, 1, -5.0], [0, 0, 1.0]])
    H = refine(shift @ cam.H, line_mask(img, CourtConfig()), CourtConfig())
    assert court_error(H, cam).max() < 0.15


def test_tracking_through_pan_and_zoom(world, pan_cam):
    tr = CourtTracker()
    mean, worst, status = [], [], []
    for t in np.arange(0, 10, 1 / 30):
        cam = pan_cam(t)
        s = tr.update(render(world, cam, t))
        assert s.H is not None
        e = court_error(s.H, cam)
        mean.append(e.mean()); worst.append(e.max()); status.append(s.status)
    assert np.mean(mean) < 0.06
    assert np.max(worst) < 0.35
    assert status.count("lines") / len(status) > 0.95


def test_tracking_when_zoomed_past_the_near_end_line(world):
    # zoom in until the near end line and both near corners leave the frame
    def cam_fn(t):
        return Camera((4.5, -10, 6.5), (4.5, 12.0, 0.5), 680 * (1.0 + 0.25 * min(t, 4.0)))
    tr = CourtTracker()
    worst = []
    for t in np.arange(0, 6, 1 / 30):
        cam = cam_fn(t)
        s = tr.update(render(world, cam, t))
        worst.append(court_error(s.H, cam).max())
    near_end = apply_h(cam.H, [(4.5, 0)])[0]
    assert near_end[1] > 540                       # really out of frame at the end
    assert np.max(worst) < 0.35


def test_occluded_lines_fall_back_to_motion_propagation(world, pan_cam):
    """Something (a person walking past the lens) slides in front of the court,
    hides nearly all of it for a second, then leaves. The camera keeps panning."""
    tr = CourtTracker()
    statuses, confs, worst = [], [], []
    for i, t in enumerate(np.arange(0, 4, 1 / 30)):
        cam = pan_cam(t)
        img = render(world, cam, t)
        cover = np.clip(min(i - 30, 85 - i) / 12.0, 0, 1)      # 0 -> 1 -> 0
        if cover > 0:
            x1 = -1 + 11 * cover
            uv = apply_h(cam.H, [(-1, -1), (x1, -1), (x1, 17), (-1, 17)]).astype(np.int32)
            cv2.fillPoly(img, [uv], (60, 90, 60))
        s = tr.update(img)
        assert s.H is not None
        statuses.append(s.status); confs.append(s.conf)
        worst.append(court_error(s.H, cam).max())
    assert set(statuses[46:70]) <= {"propagated", "weak"}      # fully covered
    assert max(confs[46:70]) < 0.5 < min(confs[:25])           # and it says so
    assert max(worst[46:70]) < 0.6                             # drifts, but stays usable
    assert statuses[-1] == "lines" and worst[-1] < 0.2         # recovers once lines are back


def test_camera_cut_triggers_redetection(world):
    tr = CourtTracker()
    cam_a, cam_b = end_camera(), Camera((2.0, -11, 8), (5.5, 9, 0), 720)
    for t in np.arange(0, 1, 1 / 30):
        tr.update(render(world, cam_a, t))
    s = tr.update(render(world, cam_b, 1.0))
    assert s.cut
    assert s.H is not None and court_error(s.H, cam_b).max() < 0.2


def test_manual_calibration(tmp_path, world):
    cam = end_camera(pan=1.0)
    court = [(0, 0), (9, 0), (9, 18), (0, 18), (0, 9), (9, 9)]
    image = apply_h(cam.H, court)
    H = homography_from_points(image, court)
    assert court_error(H, cam).max() < 1e-6
    p = tmp_path / "calib.json"
    p.write_text(json.dumps({"frame": 3, "points": [{"image": list(map(float, i)), "court": list(c)}
                                                     for i, c in zip(image, court)]}))
    frame, H2 = load_calibration(p)
    assert frame == 3 and court_error(H2, cam).max() < 1e-6
    with pytest.raises(ValueError):
        homography_from_points(image[:3], court[:3])


def test_track_video_with_manual_init_only(world):
    # global detection disabled by an impossible threshold: manual init must carry it
    cfg = CourtConfig()
    cam = end_camera()
    frames = [render(world, cam, t) for t in np.arange(0, 0.5, 1 / 30)]
    states = track_video(frames, cfg, init_H=cam.H)
    assert all(s.status == "lines" for s in states)
    assert court_error(states[-1].H, cam).max() < 0.15


def test_support_reports_missing_lines(world):
    cam = end_camera()
    img = render(world, cam, 1.0, players=False, ball=False)
    m = line_mask(img, CourtConfig())
    full = support(cam.H, m)
    assert len(full["lines"]) == 7 and full["determined"]
    m[:, :480] = 0                                 # left half hidden
    half = support(cam.H, m)
    assert 5 not in half["lines"]                  # left sideline gone
    assert not half["determined"]


def test_impossible_fit_is_rejected():
    from vball.court.detect import plausible_view
    cam = end_camera()
    assert plausible_view(cam.H, cam.size)
    squashed = np.array([[90.0, 0, 50], [0, 1.5, 330], [0, 0, 1.0]])     # court 18 m deep drawn 27 px tall
    assert not plausible_view(squashed, (960, 540))


def test_static_camera_uses_calibration_for_every_frame(tmp_path, world):
    from vball.pipeline import Config, run_court
    from vball.video import FrameSource
    cam = end_camera()
    court = [(0, 0), (9, 0), (9, 18), (0, 18)]
    p = tmp_path / "c.json"
    p.write_text(json.dumps({"frame": 0, "points": [{"image": list(map(float, i)), "court": list(c)}
                                                     for i, c in zip(apply_h(cam.H, court), court)]}))
    frames = [np.zeros((540, 960, 3), np.uint8)] * 5                    # nothing to see: still calibrated
    states = run_court(FrameSource(frames), Config(calibration=str(p), static_camera=True))
    assert len(states) == 5 and all(s.status == "manual" and court_error(s.H, cam).max() < 1e-3 for s in states)
    with pytest.raises(ValueError):
        run_court(FrameSource(frames), Config(static_camera=True))
