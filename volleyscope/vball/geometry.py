"""Court geometry, homography helpers and camera recovery.

Court coordinate system (metres), used by everything downstream:
    x: 0..9   across the court
    y: 0..18  along the court; the net is at y = 9
    z: up from the floor
The "near" team plays on y < 9, the "far" team on y > 9.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

COURT_W = 9.0
COURT_L = 18.0
NET_Y = 9.0
NET_H = 2.43          # men's height; only used for plausibility checks
ATTACK_DIST = 3.0
BALL_R = 0.105
G = 9.81
GVEC = np.array([0.0, 0.0, -G])

# name, (x0, y0), (x1, y1)
COURT_LINES = [
    ("end_near", (0, 0), (9, 0)),
    ("attack_near", (0, 6), (9, 6)),
    ("centre", (0, 9), (9, 9)),
    ("attack_far", (0, 12), (9, 12)),
    ("end_far", (0, 18), (9, 18)),
    ("side_left", (0, 0), (0, 18)),
    ("side_right", (9, 0), (9, 18)),
]
Y_LINES = [0.0, 6.0, 9.0, 12.0, 18.0]   # lines of constant y
X_LINES = [0.0, 9.0]                    # lines of constant x


def apply_h(H: np.ndarray, pts) -> np.ndarray:
    """Apply a 3x3 homography to (N,2) points."""
    pts = np.asarray(pts, dtype=float).reshape(-1, 2)
    q = np.c_[pts, np.ones(len(pts))] @ np.asarray(H, float).T
    with np.errstate(divide="ignore", invalid="ignore"):
        return q[:, :2] / q[:, 2:3]


def model_points(step: float = 0.25):
    """Sample points along every court line.

    Returns (pts (N,2), line_id (N,), direction (N,2))."""
    pts, ids, dirs = [], [], []
    for i, (_, a, b) in enumerate(COURT_LINES):
        a, b = np.array(a, float), np.array(b, float)
        n = int(round(np.linalg.norm(b - a) / step)) + 1
        t = np.linspace(0, 1, n)[:, None]
        pts.append(a + t * (b - a))
        ids.append(np.full(n, i))
        d = (b - a) / np.linalg.norm(b - a)
        dirs.append(np.tile(d, (n, 1)))
    return np.vstack(pts), np.concatenate(ids), np.vstack(dirs)


def side_of(y: float) -> str:
    return "near" if y < NET_Y else "far"


def other(team: str) -> str:
    return "far" if team == "near" else "near"


@dataclass
class Calib:
    """A full pinhole camera recovered from a court homography."""
    K: np.ndarray
    R: np.ndarray
    t: np.ndarray

    @property
    def P(self) -> np.ndarray:
        return self.K @ np.c_[self.R, self.t]

    @property
    def C(self) -> np.ndarray:
        return -self.R.T @ self.t

    @property
    def f(self) -> float:
        return float(self.K[0, 0])

    def project(self, X) -> np.ndarray:
        X = np.asarray(X, float).reshape(-1, 3)
        q = np.c_[X, np.ones(len(X))] @ self.P.T
        return q[:, :2] / q[:, 2:3]

    def depth(self, X) -> np.ndarray:
        X = np.asarray(X, float).reshape(-1, 3)
        return (X @ self.R.T + self.t)[:, 2]


def focal_from_h(H: np.ndarray, size) -> float | None:
    """Focal length (pixels) from a court->image homography.

    Assumes square pixels and the principal point at the image centre.
    Returns None when the view is degenerate for this estimate."""
    cx, cy = size[0] / 2.0, size[1] / 2.0
    T = np.array([[1, 0, -cx], [0, 1, -cy], [0, 0, 1.0]])
    Hc = T @ H
    Hc = Hc / np.linalg.norm(Hc)
    h1, h2 = Hc[:, 0], Hc[:, 1]
    a = np.array([h1[2] * h2[2], h1[2] ** 2 - h2[2] ** 2])
    b = np.array([-(h1[0] * h2[0] + h1[1] * h2[1]),
                  (h2[0] ** 2 + h2[1] ** 2) - (h1[0] ** 2 + h1[1] ** 2)])
    den = float(a @ a)
    if den < 1e-18:
        return None
    f2 = float(a @ b) / den
    if not np.isfinite(f2) or f2 <= 1.0:
        return None
    return float(np.sqrt(f2))


def calib_from_h(H: np.ndarray, size, f: float | None = None) -> Calib | None:
    """Recover K, R, t from a court->image homography."""
    if H is None:
        return None
    if f is None:
        f = focal_from_h(H, size)
    if f is None:
        return None
    K = np.array([[f, 0, size[0] / 2.0], [0, f, size[1] / 2.0], [0, 0, 1.0]])
    A = np.linalg.inv(K) @ H
    lam = 2.0 / (np.linalg.norm(A[:, 0]) + np.linalg.norm(A[:, 1]))
    A = A * lam
    if (A @ np.array([COURT_W / 2, COURT_L / 2, 1.0]))[2] < 0:
        A = -A
    r1, r2, t = A[:, 0], A[:, 1], A[:, 2]
    R = np.c_[r1, r2, np.cross(r1, r2)]
    U, _, Vt = np.linalg.svd(R)
    R = U @ Vt
    if np.linalg.det(R) < 0:
        return None
    cal = Calib(K, R, t)
    if cal.C[2] <= 0.2:      # camera must be above the floor
        return None
    return cal


def fit_ballistic_anchored(times, uv, Ps, t_anchor, p_anchor, g: float = G):
    """Like fit_ballistic, but the path is known to pass through `p_anchor` at
    `t_anchor` (e.g. the point where the previous flight ended), so only the
    velocity is solved for. Far better conditioned for short flights.

    Returns (v at t_anchor, rms_px) or None."""
    times, uv = np.asarray(times, float), np.asarray(uv, float)
    if len(times) < 3:
        return None
    p_anchor = np.asarray(p_anchor, float)
    rows, rhs = [], []
    for ti, (u, v), P in zip(times, uv, Ps):
        dt = ti - t_anchor
        base = np.r_[p_anchor + 0.5 * GVEC * (g / G) * dt * dt, 1.0]
        for coord, val in ((0, u), (1, v)):
            r = val * P[2] - P[coord]
            rows.append(r[:3] * dt)
            rhs.append(-(r @ base))
    A, b = np.array(rows), np.array(rhs)
    scale = np.linalg.norm(val * 0 + np.array([np.linalg.norm(P[2]) for P in Ps]).repeat(2)[:, None], axis=1, keepdims=True) + 1e-12
    sol, _, rank, _ = np.linalg.lstsq(A / scale, b / scale[:, 0], rcond=None)
    if rank < 3:
        return None
    err = []
    for ti, obs, P in zip(times, uv, Ps):
        dt = ti - t_anchor
        q = P @ np.r_[p_anchor + sol * dt + 0.5 * GVEC * (g / G) * dt * dt, 1.0]
        err.append(np.hypot(*(q[:2] / q[2] - obs)))
    return sol, float(np.sqrt(np.mean(np.square(err))))


def fit_ballistic(times, uv, Ps, g: float = G):
    """Fit a drag-free 3D trajectory p(t) = p0 + v0 t + 0.5 g t^2 to image
    observations of the ball, given a 3x4 projection matrix per observation.

    Linear least squares in (p0, v0). Returns (p0, v0, rms_px, cond) with
    t measured from times[0], or None when the system is rank deficient."""
    times = np.asarray(times, float)
    uv = np.asarray(uv, float)
    if len(times) < 4:
        return None
    t0 = times[0]
    rows, rhs = [], []
    for ti, (u, v), P in zip(times, uv, Ps):
        dt = ti - t0
        gz = -0.5 * g * dt * dt
        for coord, val in ((0, u), (1, v)):
            r = val * P[2] - P[coord]            # r . [X;1] = 0
            rows.append(np.r_[r[:3], r[:3] * dt])
            rhs.append(-(r[2] * gz + r[3]))
    A = np.array(rows)
    b = np.array(rhs)
    scale = np.linalg.norm(A, axis=1, keepdims=True) + 1e-12
    A, b = A / scale, b / scale[:, 0]
    sol, _, rank, sv = np.linalg.lstsq(A, b, rcond=None)
    if rank < 6:
        return None
    p0, v0 = sol[:3], sol[3:]
    err = []
    for ti, obs, P in zip(times, uv, Ps):
        dt = ti - t0
        X = p0 + v0 * dt + 0.5 * GVEC * (g / G) * dt * dt
        q = P @ np.r_[X, 1.0]
        err.append(np.hypot(*(q[:2] / q[2] - obs)))
    return p0, v0, float(np.sqrt(np.mean(np.square(err)))), float(sv[0] / max(sv[-1], 1e-12))
