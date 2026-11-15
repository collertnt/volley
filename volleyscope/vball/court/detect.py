"""Court line extraction, global court detection and line-based refinement.

All homographies named H map court metres -> image pixels. Invert for
image -> court.
"""
from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations

import cv2
import numpy as np

from ..geometry import COURT_LINES, X_LINES, Y_LINES, apply_h, model_points


@dataclass
class CourtConfig:
    tophat_thr: int = 22        # how much brighter than its surroundings a line pixel is
    max_sat: int = 90           # lines are assumed white-ish: low saturation ...
    min_val: int = 130          # ... and bright
    search_radius: int = 14     # px searched either side of the predicted line
    min_line_support: float = 0.35
    max_lines: int = 10
    reinit_every: int = 15


def line_mask(frame: np.ndarray, cfg: CourtConfig) -> np.ndarray:
    """Binary mask of thin bright low-saturation structures (candidate court lines)."""
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    v, s = hsv[..., 2], hsv[..., 1]
    k = max(7, (frame.shape[1] // 70) | 1)
    th = cv2.morphologyEx(v, cv2.MORPH_TOPHAT, cv2.getStructuringElement(cv2.MORPH_RECT, (k, k)))
    m = (th > cfg.tophat_thr) & (s < cfg.max_sat) & (v > cfg.min_val)
    return m.astype(np.uint8) * 255


# --------------------------------------------------------------------------
# refinement: point-to-line Gauss-Newton on the 8 homography parameters
# --------------------------------------------------------------------------
_MP, _MID, _MDIR = model_points(0.25)
_IS_X = np.array([COURT_LINES[i][1][0] == COURT_LINES[i][2][0] for i in range(len(COURT_LINES))])


def _match(H, mask, radius):
    """For each model point, find the line pixel nearest along the image normal.

    Returns (visible idx, signed offset px (nan = no match), normals, projected pts)."""
    h, w = mask.shape
    q = apply_h(H, _MP)
    q2 = apply_h(H, _MP + 0.1 * _MDIR)
    tang = q2 - q
    norm = np.linalg.norm(tang, axis=1)
    ok = np.isfinite(q).all(1) & (norm > 1e-6)
    ok &= (q[:, 0] >= 2) & (q[:, 0] < w - 2) & (q[:, 1] >= 2) & (q[:, 1] < h - 2)
    idx = np.where(ok)[0]
    if len(idx) == 0:
        return idx, np.array([]), np.zeros((0, 2)), q
    tg = tang[idx] / norm[idx, None]
    nrm = np.c_[-tg[:, 1], tg[:, 0]]
    offs = np.arange(-radius, radius + 1)
    px = np.round(q[idx, 0, None] + offs[None] * nrm[:, 0, None]).astype(int)
    py = np.round(q[idx, 1, None] + offs[None] * nrm[:, 1, None]).astype(int)
    inside = (px >= 0) & (px < w) & (py >= 0) & (py < h)
    hits = np.zeros(px.shape, bool)
    hits[inside] = mask[py[inside], px[inside]] > 0
    cost = np.where(hits, np.abs(offs)[None], 10 ** 6)
    best = offs[np.argmin(cost, axis=1)]
    has = hits.any(1)
    near = hits & (np.abs(offs[None] - best[:, None]) <= 3)      # centre of the stripe
    cnt = np.maximum(near.sum(1), 1)
    off = (near * offs[None]).sum(1) / cnt
    off = np.where(has, off, np.nan)
    return idx, off, nrm, q


def support(H, mask, radius=3, cfg: CourtConfig | None = None):
    """Per-line support of a homography. Returns dict with the fraction of
    in-frame model points lying on line pixels, and which lines are supported."""
    cfg = cfg or CourtConfig()
    idx, off, _, _ = _match(H, mask, radius)
    good = np.isfinite(off) & (np.abs(off) <= 2.5) if len(idx) else np.array([], bool)
    lines = []
    for li in range(len(COURT_LINES)):
        sel = _MID[idx] == li
        if sel.sum() >= 8 and good[sel].mean() >= cfg.min_line_support:
            lines.append(li)
    nx = sum(_IS_X[li] for li in lines)
    ny = len(lines) - nx
    return {"n_visible": int(len(idx)), "frac": float(good.mean()) if len(idx) else 0.0,
            "lines": lines, "n_x": int(nx), "n_y": int(ny), "determined": bool(nx >= 2 and ny >= 2)}


def refine(H0, mask, cfg: CourtConfig, radius=None, iters=5):
    """Refine a court->image homography against the line mask.

    Under-constrained views (too few visible lines) are handled by damping, so
    the unobservable part of the motion stays at the prediction H0."""
    radius = radius or cfg.search_radius
    H = H0 / H0[2, 2]
    for it in range(iters):
        r = radius if it < 2 else max(4, radius // 2)
        idx, off, nrm, q = _match(H, mask, r)
        m = np.isfinite(off) if len(idx) else np.array([], bool)
        if m.sum() < 12:
            break
        # drop whole lines that are barely matched (likely captured by clutter)
        keep = m.copy()
        for li in np.unique(_MID[idx]):
            sel = _MID[idx] == li
            if m[sel].mean() < 0.2:
                keep[sel] = False
        sel = np.where(keep)[0]
        if len(sel) < 12:
            break
        X, Y = _MP[idx[sel], 0], _MP[idx[sel], 1]
        den = H[2, 0] * X + H[2, 1] * Y + 1.0
        u, v = q[idx[sel], 0], q[idx[sel], 1]
        z = np.zeros_like(X)
        du = np.c_[X, Y, np.ones_like(X), z, z, z, -u * X, -u * Y] / den[:, None]
        dv = np.c_[z, z, z, X, Y, np.ones_like(X), -v * X, -v * Y] / den[:, None]
        J = nrm[sel, 0, None] * du + nrm[sel, 1, None] * dv
        res = off[sel]
        wgt = np.where(np.abs(res) < 2.0, 1.0, 2.0 / np.maximum(np.abs(res), 1e-9))
        JW = J * wgt[:, None]
        A = J.T @ JW
        A += 1e-3 * np.diag(np.diag(A)) + 1e-9 * np.eye(8)
        try:
            delta = np.linalg.solve(A, JW.T @ res)
        except np.linalg.LinAlgError:
            break
        H = H.copy()
        H.flat[:8] += delta
        if not np.isfinite(H).all():
            return H0 / H0[2, 2]
    return H


# --------------------------------------------------------------------------
# global detection from a single frame
# --------------------------------------------------------------------------
def _merge_segments(segs, max_lines):
    """Cluster Hough segments into distinct lines. Returns [(point, dir, length)]."""
    segs = sorted(segs, key=lambda s: -np.hypot(s[2] - s[0], s[3] - s[1]))
    clusters = []
    for x0, y0, x1, y1 in segs:
        a, b = np.array([x0, y0], float), np.array([x1, y1], float)
        d = (b - a) / max(np.linalg.norm(b - a), 1e-9)
        placed = False
        for c in clusters:
            if abs(d @ c["d"]) < np.cos(np.deg2rad(2.5)):
                continue
            n = np.array([-c["d"][1], c["d"][0]])
            if abs((a - c["p"]) @ n) < 4 and abs((b - c["p"]) @ n) < 4:
                c["pts"] += [a, b]
                placed = True
                break
        if not placed:
            clusters.append({"p": a, "d": d, "pts": [a, b]})
    out = []
    for c in clusters:
        pts = np.array(c["pts"])
        mean = pts.mean(0)
        _, _, vt = np.linalg.svd(pts - mean)
        d = vt[0]
        proj = (pts - mean) @ d
        out.append((mean, d, float(proj.max() - proj.min())))
    out.sort(key=lambda l: -l[2])
    return out[:max_lines]


def _score(H, dt, pts):
    h, w = dt.shape
    q = apply_h(H, pts)
    ok = np.isfinite(q).all(1) & (q[:, 0] >= 0) & (q[:, 0] < w - 1) & (q[:, 1] >= 0) & (q[:, 1] < h - 1)
    if ok.sum() < 30:
        return -1e9
    qi = q[ok].astype(int)
    hit = dt[qi[:, 1], qi[:, 0]] < 2.5
    return float(hit.sum() - 0.6 * (~hit).sum())


def canonical(H):
    """Resolve the 180-degree court symmetry: the y=0 end is the one nearer the
    bottom of the image (or the left, for a side-on camera)."""
    a, b = apply_h(H, [(4.5, 0.0), (4.5, 18.0)])
    dv, du = a[1] - b[1], a[0] - b[0]
    flip = dv < 0 if abs(dv) >= 0.25 * abs(du) else du > 0
    if flip:
        S = np.array([[-1, 0, 9.0], [0, -1, 18.0], [0, 0, 1.0]])
        H = H @ S
    return H / H[2, 2]


def plausible_view(H, size):
    """Could a real camera see a 9 x 18 m court like this? Rejects squashed or
    inside-out fits that happen to line up with floor seams or banners."""
    from ..geometry import calib_from_h
    w, h = size
    cal = calib_from_h(H, size)
    if cal is None or not (0.3 * w <= cal.f <= 6 * w) or not (0.5 <= cal.C[2] <= 40):
        return False
    q = apply_h(H, [(0, 0), (9, 0), (9, 18), (0, 18)])
    if not np.isfinite(q).all():
        return False
    area = abs(cv2.contourArea(q.astype(np.float32)))
    return area >= 0.03 * w * h


def detect_court(frame, cfg: CourtConfig | None = None, mask=None):
    """Find the court in a single frame with no prior.

    Returns (H, info) or (None, info). Works when at least two sidelines and
    two cross-court lines are visible as white-ish lines; see README limits."""
    cfg = cfg or CourtConfig()
    mask = line_mask(frame, cfg) if mask is None else mask
    h, w = mask.shape
    segs = cv2.HoughLinesP(mask, 1, np.pi / 360, threshold=max(30, w // 25),
                           minLineLength=int(w * 0.07), maxLineGap=int(w * 0.02))
    if segs is None:
        return None, {"reason": "no line segments"}
    lines = _merge_segments(np.asarray(segs).reshape(-1, 4), cfg.max_lines)
    n = len(lines)
    if n < 4:
        return None, {"reason": f"only {n} lines"}
    hom = [np.cross(np.r_[p, 1.0], np.r_[p + d, 1.0]) for p, d, _ in lines]
    dt = cv2.distanceTransform(255 - mask, cv2.DIST_L2, 3)
    pts = model_points(0.5)[0]
    ypairs = [(a, b) for a in Y_LINES for b in Y_LINES if a != b]

    def inter(i, j):
        x = np.cross(hom[i], hom[j])
        if abs(x[2]) < 1e-9:
            return None
        p = x[:2] / x[2]
        if not (-1.5 * w < p[0] < 2.5 * w and -1.5 * h < p[1] < 2.5 * h):
            return None
        return p

    cands = []
    for i, j in combinations(range(n), 2):               # candidate sidelines
        for k, l in combinations([m for m in range(n) if m not in (i, j)], 2):   # cross lines
            quad = [inter(i, k), inter(j, k), inter(j, l), inter(i, l)]
            if any(p is None for p in quad):
                continue
            quad = np.array(quad, np.float32)
            if not cv2.isContourConvex(quad.reshape(-1, 1, 2)):
                continue
            area = cv2.contourArea(quad, oriented=True)
            if abs(area) < 0.01 * w * h:
                continue
            for xa, xb in ((X_LINES[0], X_LINES[1]), (X_LINES[1], X_LINES[0])):
                for ya, yb in ypairs:
                    # a camera above the floor sees the court mirrored in image
                    # (v-down) coordinates: opposite winding to court coordinates
                    c_area = (xb - xa) * (yb - ya)
                    if c_area * area > 0:
                        continue
                    court = np.array([(xa, ya), (xb, ya), (xb, yb), (xa, yb)], np.float32)
                    H = cv2.getPerspectiveTransform(court, quad).astype(float)
                    cands.append((_score(H, dt, pts), H))
    if not cands:
        return None, {"reason": "no consistent line quadruple"}
    cands.sort(key=lambda c: -c[0])
    best, best_s, best_sup = None, -1e9, None
    fine = model_points(0.25)[0]
    for _, H in cands[:6]:
        Hr = refine(H, mask, cfg, radius=6)
        s = _score(Hr, dt, fine)
        if s > best_s:
            best, best_s, best_sup = Hr, s, support(Hr, mask, cfg=cfg)
    info = {"score": best_s, "n_lines": n, **(best_sup or {})}
    if best is not None and not plausible_view(best, (w, h)):
        info["reason"] = "best hypothesis is not a physically possible view of a court"
        return None, info
    if best is None or not best_sup["determined"] or best_sup["frac"] < 0.55:
        info["reason"] = "best hypothesis not well supported"
        return None, info
    return canonical(best), info
