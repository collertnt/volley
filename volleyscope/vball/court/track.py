"""Per-frame court homography tracking.

Strategy per frame:
  1. predict: estimate global image motion from the previous frame (sparse
     optical flow + RANSAC homography) -> handles pan and zoom;
  2. correct: snap the predicted court onto the visible line pixels;
  3. if the lines can't be used (occluded / out of frame) keep the prediction
     and say so ("propagated"); if tracking is lost or the shot cuts, re-detect.
"""
from __future__ import annotations

import json
from dataclasses import dataclass

import cv2
import numpy as np

from ..geometry import apply_h
from .detect import CourtConfig, canonical, detect_court, line_mask, refine, support

_CORNERS = np.array([(0, 0), (9, 0), (9, 18), (0, 18), (4.5, 9)], float)


@dataclass
class CourtState:
    H: np.ndarray | None            # court -> image
    conf: float                     # 0..1
    status: str                     # lines | weak | propagated | lost
    cut: bool = False

    @property
    def Hinv(self):
        return None if self.H is None else np.linalg.inv(self.H)

    def to_court(self, pts):
        return apply_h(self.Hinv, pts)

    def to_image(self, pts):
        return apply_h(self.H, pts)


def homography_from_points(image_pts, court_pts):
    """Manual calibration: >=4 clicked image points with known court positions."""
    image_pts, court_pts = np.asarray(image_pts, np.float32), np.asarray(court_pts, np.float32)
    if len(image_pts) < 4:
        raise ValueError("need at least 4 point pairs")
    H, _ = cv2.findHomography(court_pts, image_pts, 0)
    if H is None:
        raise ValueError("degenerate calibration points")
    return H / H[2, 2]


def load_calibration(path, scale=1.0):
    """Read {"frame": n, "points": [{"image": [u, v], "court": [x, y]}, ...]}."""
    d = json.load(open(path))
    img = np.array([p["image"] for p in d["points"]], float) * scale
    crt = [p["court"] for p in d["points"]]
    return int(d.get("frame", 0)), homography_from_points(img, crt)


class CourtTracker:
    def __init__(self, cfg: CourtConfig | None = None, init_H=None):
        self.cfg = cfg or CourtConfig()
        self.H = None if init_H is None else init_H / init_H[2, 2]
        self.prev_gray = None
        self.frame_idx = -1
        self.fail = 0
        self.reinit_attempts = 0
        self.next_reinit = 0

    def _motion(self, gray):
        """Homography prev image -> current image, or None."""
        pg = self.prev_gray
        pts = cv2.goodFeaturesToTrack(pg, 300, 0.01, 7)
        if pts is None or len(pts) < 12:
            return None, 0.0
        nxt, st, _ = cv2.calcOpticalFlowPyrLK(pg, gray, pts, None, winSize=(21, 21), maxLevel=3)
        good = st[:, 0] == 1
        if good.sum() < 12:
            return None, 0.0
        M, inl = cv2.findHomography(pts[good], nxt[good], cv2.RANSAC, 2.0)
        if M is None:
            return None, 0.0
        ratio = float(inl.mean())
        if inl.sum() < 12 or ratio < 0.3:
            return None, ratio
        # a real camera move is small between frames and explains the new image;
        # anything else is a spurious fit across a shot change
        h, w = gray.shape
        box = np.array([(0, 0), (w, 0), (w, h), (0, h)], float)
        if np.abs(apply_h(M, box) - box).max() > 0.3 * w:
            return None, ratio
        warped = cv2.warpPerspective(pg, M, (w, h))
        valid = cv2.warpPerspective(np.full_like(pg, 255), M, (w, h)) > 0
        if valid.mean() < 0.3 or float(np.mean(cv2.absdiff(warped, gray)[valid])) > 12:
            return None, ratio
        S = np.diag([2.0, 2.0, 1.0])                     # flow runs at half resolution
        return S @ M @ np.linalg.inv(S), ratio

    def _try_detect(self, frame, mask):
        if self.frame_idx < self.next_reinit:
            return None
        H, _ = detect_court(frame, self.cfg, mask)
        if H is None:
            self.reinit_attempts += 1
            self.next_reinit = self.frame_idx + self.cfg.reinit_every * min(2 ** self.reinit_attempts, 32)
        else:
            self.reinit_attempts = 0
        return H

    def update(self, frame) -> CourtState:
        self.frame_idx += 1
        cfg = self.cfg
        gray = cv2.cvtColor(cv2.resize(frame, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2GRAY)
        mask = line_mask(frame, cfg)
        cut = False
        M, diff, have_prev = None, 0.0, False
        if self.prev_gray is not None and self.prev_gray.shape == gray.shape:
            have_prev = True
            M, _ = self._motion(gray)
            if M is None:
                diff = float(np.mean(cv2.absdiff(gray, self.prev_gray)))
        self.prev_gray = gray

        if self.H is None:
            H = self._try_detect(frame, mask)
            if H is None:
                return CourtState(None, 0.0, "lost", cut)
            self.H, self.fail = H, 0
            sup = support(H, mask, cfg=cfg)
            return CourtState(H, float(min(1.0, sup["frac"] / 0.8)), "lines", cut)

        pred = self.H if M is None else M @ self.H
        pred = pred / pred[2, 2]
        H = refine(pred, mask, cfg)
        sup = support(H, mask, cfg=cfg)
        h, w = mask.shape
        shift = np.nanmax(np.linalg.norm(apply_h(H, _CORNERS) - apply_h(pred, _CORNERS), axis=1))
        lines_ok = np.isfinite(shift) and shift <= 0.25 * w and sup["frac"] >= 0.3 and len(sup["lines"]) >= 2
        if have_prev and M is None and not (lines_ok and sup["determined"] and sup["frac"] >= 0.6):
            # image motion could not be explained: shot change, or a big occlusion
            self.next_reinit = min(self.next_reinit, self.frame_idx) if diff > 18 else self.next_reinit
            Hn = self._try_detect(frame, mask)
            if Hn is not None:
                moved = np.linalg.norm(apply_h(Hn, _CORNERS) - apply_h(pred, _CORNERS), axis=1).max()
                self.H, self.fail = Hn, 0
                sup = support(Hn, mask, cfg=cfg)
                return CourtState(Hn, float(min(1.0, sup["frac"] / 0.8)), "lines", bool(moved > 0.05 * w))
            if diff > 18:
                self.H = None
                return CourtState(None, 0.0, "lost", True)
        if not lines_ok:
            H, status, conf = pred, "propagated", 0.3 * (0.9 ** self.fail)
            self.fail += 1
        elif not sup["determined"]:
            status, conf = "weak", 0.5 * min(1.0, sup["frac"] / 0.8)
            self.fail = max(0, self.fail - 1)
        else:
            status, conf = "lines", min(1.0, sup["frac"] / 0.8)
            self.fail = 0
        self.H = H
        if self.fail >= cfg.reinit_every:
            Hn = self._try_detect(frame, mask)
            if Hn is not None:
                # keep the same court orientation as before the loss when ambiguous
                self.H, self.fail, status, conf = Hn, 0, "lines", 0.8
        return CourtState(self.H, float(conf), status, cut)


def track_video(frames, cfg: CourtConfig | None = None, init_H=None, init_frame=0):
    """Run the tracker over an iterable of frames. Returns a list of CourtState."""
    tr = CourtTracker(cfg)
    out = []
    for i, f in enumerate(frames):
        if init_H is not None and i == init_frame:
            tr.H = init_H / init_H[2, 2]
        out.append(tr.update(f))
    return out
