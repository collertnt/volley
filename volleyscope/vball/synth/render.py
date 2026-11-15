"""Renderer and camera for the synthetic world."""
from __future__ import annotations

import cv2
import numpy as np

from ..geometry import BALL_R, COURT_LINES, NET_H

WALL_Y = 27.0
TEAM_COL = {"near": (140, 50, 20), "far": (30, 30, 170)}   # BGR
FLOOR_OUT, FLOOR_IN, WALL = (120, 135, 95), (70, 120, 180), (105, 100, 98)
LINE_COL, BALL_COL = (245, 245, 245), (60, 215, 240)


class Camera:
    def __init__(self, pos, target, f, size=(960, 540)):
        self.pos = np.asarray(pos, float)
        self.size = size
        self.f = float(f)
        fwd = np.asarray(target, float) - self.pos
        fwd /= np.linalg.norm(fwd)
        right = np.cross(fwd, [0, 0, 1.0])
        right /= np.linalg.norm(right)
        down = np.cross(fwd, right)
        self.R = np.vstack([right, down, fwd])
        self.t = -self.R @ self.pos
        self.K = np.array([[f, 0, size[0] / 2], [0, f, size[1] / 2], [0, 0, 1.0]])
        self.P = self.K @ np.c_[self.R, self.t]

    @property
    def H(self):
        """court (x, y, 1) -> image homography for the floor plane."""
        return self.P[:, [0, 1, 3]]

    def cam(self, X):
        return np.asarray(X, float).reshape(-1, 3) @ self.R.T + self.t

    def project(self, X):
        c = self.cam(X)
        return np.c_[self.f * c[:, 0] / c[:, 2] + self.size[0] / 2,
                     self.f * c[:, 1] / c[:, 2] + self.size[1] / 2], c[:, 2]


def end_camera(t=0.0, pan=0.0, zoom=1.0, size=(960, 540), pos=(4.5, -10.0, 6.5), f=680.0):
    """Default view: tripod behind the near end line, elevated."""
    return Camera(pos, (4.5 + pan, 8.0, 0.5), f * zoom, size)


def _poly(img, cam, pts3, col, aa=True):
    c = cam.cam(pts3)
    near = 0.3
    out = []                         # Sutherland-Hodgman against the near plane
    n = len(c)
    for i in range(n):
        a, b = c[i], c[(i + 1) % n]
        if a[2] >= near:
            out.append(a)
        if (a[2] >= near) != (b[2] >= near):
            s = (near - a[2]) / (b[2] - a[2])
            out.append(a + s * (b - a))
    if len(out) < 3:
        return
    o = np.array(out)
    uv = np.c_[cam.f * o[:, 0] / o[:, 2] + cam.size[0] / 2, cam.f * o[:, 1] / o[:, 2] + cam.size[1] / 2]
    uv = np.clip(uv, -2e4, 2e4)
    cv2.fillPoly(img, [np.round(uv * 8).astype(np.int32)], col, cv2.LINE_AA if aa else cv2.LINE_8, shift=3)


def _rect(img, u0, v0, u1, v1, col):
    cv2.rectangle(img, (int(round(min(u0, u1))), int(round(min(v0, v1)))),
                  (int(round(max(u0, u1))), int(round(max(v0, v1)))), col, -1)


def draw_player(img, cam, spec, pos):
    (uv, depth) = cam.project([pos])
    d = float(depth[0])
    if d < 0.5:
        return
    s = cam.f / d
    u, v = uv[0]
    Hh = spec.height * s
    col = TEAM_COL[spec.team]
    for sx in (-1, 1):                               # legs + socks
        _rect(img, u + sx * 0.05 * s, v, u + sx * 0.19 * s, v - 0.47 * Hh, spec.skin)
        _rect(img, u + sx * 0.05 * s, v, u + sx * 0.19 * s, v - 0.14 * Hh, spec.socks)
        _rect(img, u + sx * 0.23 * s, v - 0.52 * Hh, u + sx * 0.32 * s, v - 0.80 * Hh, spec.skin)  # arms
    _rect(img, u - 0.21 * s, v - 0.40 * Hh, u + 0.21 * s, v - 0.53 * Hh, (25, 25, 25))             # shorts
    _rect(img, u - 0.23 * s, v - 0.50 * Hh, u + 0.23 * s, v - 0.82 * Hh, col)                      # jersey
    r = max(1, int(round(0.105 * s)))
    hc = (int(round(u)), int(round(v - 0.90 * Hh)))
    cv2.circle(img, hc, r, spec.skin, -1, cv2.LINE_AA)
    cv2.ellipse(img, hc, (r, r), 0, 180, 360, spec.hair, -1, cv2.LINE_AA)
    th = 0.20 * Hh                                   # digit height in px
    if th >= 6:
        txt = str(spec.number)
        scale = th / 22.0
        thick = max(1, int(round(th / 7)))
        (tw, tH), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, scale, thick)
        if tw > 0.36 * s:                            # two digits must still fit on the jersey
            scale *= 0.36 * s / tw
            thick = max(1, int(round(22 * scale / 7)))
            (tw, tH), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, scale, thick)
        org = (int(round(u - tw / 2)), int(round(v - 0.66 * Hh + tH / 2)))
        cv2.putText(img, txt, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (245, 245, 245), thick, cv2.LINE_AA)


def render(world, cam, t, noise=3.0, rng=None, players=True, ball=True):
    w, h = cam.size
    img = np.empty((h, w, 3), np.uint8)
    img[:] = WALL
    for x0, x1, z0, z1, col in world.wall_rects:
        _poly(img, cam, [(x0, WALL_Y, z0), (x1, WALL_Y, z0), (x1, WALL_Y, z1), (x0, WALL_Y, z1)], col)
    _poly(img, cam, [(-12, -9.5, 0), (21, -9.5, 0), (21, WALL_Y, 0), (-12, WALL_Y, 0)], FLOOR_OUT)
    _poly(img, cam, [(0, 0, 0), (9, 0, 0), (9, 18, 0), (0, 18, 0)], FLOOR_IN)
    hw = 0.025
    for _, a, b in COURT_LINES:
        if a[0] == b[0]:
            q = [(a[0] - hw, a[1] - hw, 0), (a[0] + hw, a[1] - hw, 0), (b[0] + hw, b[1] + hw, 0), (b[0] - hw, b[1] + hw, 0)]
        else:
            q = [(a[0] - hw, a[1] - hw, 0), (b[0] + hw, b[1] - hw, 0), (b[0] + hw, b[1] + hw, 0), (a[0] - hw, a[1] + hw, 0)]
        _poly(img, cam, q, LINE_COL)

    items = []
    if players:
        for p in world.players:
            pos = world.player_pos(p.pid, t)
            items.append((float(cam.cam([pos])[0, 2]), "p", p, pos))
    bp = world.ball(t) if ball else None
    if bp is not None:
        items.append((float(cam.cam([bp])[0, 2]), "b", None, bp))
    items.append((float(cam.cam([(4.5, 9, 1.5)])[0, 2]), "n", None, None))
    for d, kind, spec, pos in sorted(items, key=lambda it: -it[0]):
        if kind == "p":
            draw_player(img, cam, spec, pos)
        elif kind == "b":
            if d > 0.5:
                uv, _ = cam.project([pos])
                r = max(1.5, BALL_R * cam.f / d)
                cv2.circle(img, (int(round(uv[0, 0] * 8)), int(round(uv[0, 1] * 8))), int(round(r * 8)),
                           BALL_COL, -1, cv2.LINE_AA, shift=3)
        else:
            net = img.copy()
            _poly(net, cam, [(-0.5, 9, NET_H - 1.0), (9.5, 9, NET_H - 1.0), (9.5, 9, NET_H), (-0.5, 9, NET_H)], (40, 40, 40))
            cv2.addWeighted(net, 0.25, img, 0.75, 0, img)
            _poly(img, cam, [(-0.5, 9, NET_H - 0.06), (9.5, 9, NET_H - 0.06), (9.5, 9, NET_H), (-0.5, 9, NET_H)], (225, 225, 225))
            for px in (-0.9, 9.9):
                _poly(img, cam, [(px - 0.05, 9, 0), (px + 0.05, 9, 0), (px + 0.05, 9, 2.55), (px - 0.05, 9, 2.55)], (50, 50, 50))
    if noise:
        rng = rng or np.random.default_rng(int(t * 1000) % (2 ** 31))
        img = np.clip(img.astype(np.int16) + rng.normal(0, noise, img.shape).astype(np.int16), 0, 255).astype(np.uint8)
    return img


def make_clip(world, cam_fn=None, fps=30, t0=0.0, t1=None, path=None, times=None, **kw):
    """Render a clip. Returns (frames or path, list of cameras, times)."""
    cam_fn = cam_fn or (lambda t: end_camera(t))
    if times is None:
        t1 = world.duration if t1 is None else t1
        times = np.arange(t0, t1, 1.0 / fps)
    frames, cams, writer = [], [], None
    for t in times:
        cam = cam_fn(float(t))
        img = render(world, cam, float(t), **kw)
        cams.append(cam)
        if path:
            if writer is None:
                writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, cam.size)
            writer.write(img)
        else:
            frames.append(img)
    if writer is not None:
        writer.release()
    return (path if path else frames), cams, np.asarray(times)
