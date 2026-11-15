import numpy as np
import pytest

from vball.geometry import (GVEC, apply_h, calib_from_h, fit_ballistic, focal_from_h,
                            model_points, side_of)
from vball.synth.render import Camera, end_camera


def test_apply_h_roundtrip():
    cam = end_camera()
    pts = np.array([(0, 0), (9, 18), (4.5, 9), (2, 13)], float)
    back = apply_h(np.linalg.inv(cam.H), apply_h(cam.H, pts))
    assert np.allclose(back, pts, atol=1e-9)


def test_homography_matches_projection_on_floor():
    cam = end_camera(pan=1.0)
    pts = np.array([(1, 2), (8, 16)], float)
    uv, _ = cam.project(np.c_[pts, np.zeros(2)])
    assert np.allclose(apply_h(cam.H, pts), uv, atol=1e-9)


def test_model_points_cover_all_lines():
    pts, ids, dirs = model_points(0.5)
    assert set(ids) == set(range(7))
    assert pts[:, 0].min() == 0 and pts[:, 0].max() == 9 and pts[:, 1].max() == 18
    assert np.allclose(np.linalg.norm(dirs, axis=1), 1)


def test_side_of():
    assert side_of(3) == "near" and side_of(12) == "far"


@pytest.mark.parametrize("cam", [
    end_camera(), end_camera(pan=2.0, zoom=1.3),
    Camera((-6, -8, 7), (4.5, 9, 0), 640), Camera((-12, 9, 7), (4.5, 9, 0), 560)])
def test_camera_recovered_from_homography(cam):
    f = focal_from_h(cam.H, cam.size)
    assert f == pytest.approx(cam.f, rel=1e-6)
    cal = calib_from_h(cam.H, cam.size)
    assert np.allclose(cal.C, cam.pos, atol=1e-6)
    X = np.array([(4.5, 9, 2.43), (1, 3, 3.0), (8, 15, 0.5)])
    assert np.allclose(cal.project(X), cam.project(X)[0], atol=1e-6)


def test_calib_rejects_mirrored_homography():
    cam = end_camera()
    mirror = cam.H @ np.array([[-1, 0, 9.0], [0, 1, 0], [0, 0, 1.0]])
    assert calib_from_h(mirror, cam.size) is None


def test_fit_ballistic_exact():
    cam = end_camera()
    p0, v0 = np.array([3.0, 2.0, 2.5]), np.array([1.0, 9.0, 4.0])
    ts = np.arange(0, 1.0, 1 / 30)
    X = p0 + v0 * ts[:, None] + 0.5 * GVEC * ts[:, None] ** 2
    uv, _ = cam.project(X)
    p, v, rms, _ = fit_ballistic(ts, uv, [cam.P] * len(ts))
    assert np.allclose(p, p0, atol=1e-6) and np.allclose(v, v0, atol=1e-6) and rms < 1e-6


def test_fit_ballistic_with_pixel_noise_and_moving_camera():
    rng = np.random.default_rng(0)
    p0, v0 = np.array([6.0, 14.0, 0.8]), np.array([-0.5, -4.0, 6.0])
    ts = np.arange(0, 1.2, 1 / 30)
    cams = [end_camera(pan=2 * t) for t in ts]
    X = p0 + v0 * ts[:, None] + 0.5 * GVEC * ts[:, None] ** 2
    uv = np.array([c.project([x])[0][0] for c, x in zip(cams, X)]) + rng.normal(0, 0.5, (len(ts), 2))
    p, v, rms, _ = fit_ballistic(ts, uv, [c.P for c in cams])
    assert np.linalg.norm(p - p0) < 0.6          # depth is the weak direction
    assert rms < 1.5


def test_fit_ballistic_needs_four_points():
    cam = end_camera()
    assert fit_ballistic([0, 0.1, 0.2], np.zeros((3, 2)), [cam.P] * 3) is None
