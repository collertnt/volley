"""Foreground extraction and player / ball-candidate detection.

Default (tested) path, CPU only:
  * a background image is built in a camera-stabilised reference view, using
    the court homographies to undo pan and zoom (exact for a camera rotating on
    a tripod; an approximation if the camera itself is carried around);
  * anything that differs from it is foreground; blobs standing on the floor
    with a plausible human size are players, small round blobs are ball
    candidates.
Optional paths: OpenCV's HOG person detector, and a torchvision detector if
PyTorch and a local weights file are available (see README: untested here).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

from ..court.detect import CourtConfig, line_mask
from ..geometry import BALL_R, apply_h


@dataclass
class DetectConfig:
    fg_thresh: int = 30
    bg_samples: int = 30
    floor_colour_share: float = 0.06    # a colour present over at least this share of the floor area is 'floor'
    canvas_scale: float = 1.6
    person_h: float = 1.85
    min_h_frac: float = 0.45
    max_h_frac: float = 1.5
    margin_side: float = 3.0      # metres outside the sidelines still considered
    margin_end: float = 5.0       # metres behind the end lines (servers)
    detector: str = "background"  # background | hog | torch
    torch_weights: str | None = None


@dataclass
class Detection:
    bbox: tuple                   # x, y, w, h (pixels)
    foot: tuple                   # image point taken as the feet
    xy: np.ndarray                # court metres
    scale: float                  # pixels per metre at the feet
    hist: np.ndarray              # appearance descriptor
    Jinv: np.ndarray = None       # d(court)/d(pixel) at the feet
    team: str | None = None
    occluded: bool = False        # feet not seen; position inferred from head height


def local_jacobian(H, xy, eps=0.05):
    """2x2 Jacobian d(image)/d(court) of a court->image homography at xy."""
    p = np.array([xy, xy + np.array([eps, 0]), xy + np.array([0, eps])])
    q = apply_h(H, p)
    return np.c_[(q[1] - q[0]) / eps, (q[2] - q[0]) / eps]


def metric_scale(H, xy):
    """Pixels per metre for an upright object standing at court point xy."""
    return float(np.linalg.svd(local_jacobian(H, np.asarray(xy, float)), compute_uv=False)[0])


N_BINS = 19


def pixel_bins(hsv):
    """Quantise HSV pixels: 16 colour bins (8 hues x 2 saturations) + 3 grey levels."""
    hsv = hsv.astype(np.int32)
    col = (hsv[..., 1] > 50) & (hsv[..., 2] > 40)
    cb = (hsv[..., 0] * 8 // 180).clip(0, 7) * 2 + (hsv[..., 1] > 150)
    gb = 16 + hsv[..., 2] * 3 // 256
    return np.where(col, cb, gb)


def appearance(frame, mask, bbox):
    """Colour descriptor in three vertical bands (head / torso / legs)."""
    x, y, w, h = bbox
    bins = pixel_bins(cv2.cvtColor(frame[y:y + h, x:x + w], cv2.COLOR_BGR2HSV))
    m = mask[y:y + h, x:x + w] > 0 if mask is not None else np.ones((h, w), bool)
    out = []
    for a, b in ((0.0, 0.18), (0.18, 0.52), (0.52, 1.0)):
        r0, r1 = int(a * h), max(int(a * h) + 1, int(b * h))
        v = np.bincount(bins[r0:r1][m[r0:r1]], minlength=N_BINS).astype(float)
        out.append(v / v.sum() if v.sum() else v)
    return np.concatenate(out)


def torso_hist(det):
    return det.hist[N_BINS:2 * N_BINS]


class TeamColours:
    """Jersey colour bins per side, learned from clearly separated players."""

    def __init__(self):
        self.acc = {"near": np.zeros(N_BINS), "far": np.zeros(N_BINS)}
        self.bins = {"near": [], "far": []}

    def add(self, dets):
        for d in dets:
            if d.occluded or not (0 <= d.xy[0] <= 9):
                continue
            if 0 <= d.xy[1] < 8:
                self.acc["near"] += torso_hist(d)
            elif 10 < d.xy[1] <= 18:
                self.acc["far"] += torso_hist(d)

    def finish(self):
        """A jersey colour is one that is common on this team's torsos and rare
        on the other team's. Colours both teams show (skin on bare arms, white
        numbers, dark shorts) are ignored."""
        share = {t: v / v.sum() if v.sum() >= 5 else np.zeros(N_BINS) for t, v in self.acc.items()}
        for t, o in (("near", "far"), ("far", "near")):
            self.bins[t] = [int(b) for b in np.argsort(-share[t])[:3]
                            if share[t][b] >= 0.15 and share[o][b] < 0.5 * share[t][b]]
        self.usable = self.distinct = bool(self.bins["near"] and self.bins["far"])
        return self

    def team_of(self, det):
        """Team by jersey colour, or None if the colours don't decide it."""
        if not getattr(self, "distinct", False):
            return None
        th = torso_hist(det)
        sn, sf = th[self.bins["near"]].sum(), th[self.bins["far"]].sum()
        if max(sn, sf) < 0.25 or abs(sn - sf) < 0.15:
            return None
        return "near" if sn > sf else "far"


def hist_sim(a, b):
    """Bhattacharyya coefficient averaged over the three bands (1 = identical)."""
    return float(np.sqrt(a * b).sum() / 3.0)


class BallColour:
    """Colours that show up on small round moving blobs but not on players.

    Lets the ball be found while it is in front of a player, where it is part
    of the player's foreground blob and shape alone cannot isolate it."""

    def __init__(self):
        self.ball, self.body = np.zeros(N_BINS), np.zeros(N_BINS)
        self.bins = []

    def add(self, frame, mask, players, balls):
        hsv_bins = None
        for u, v, r in balls:
            x0, y0, x1, y1 = int(u - r - 1), int(v - r - 1), int(u + r + 2), int(v + r + 2)
            if x0 < 0 or y0 < 0:
                continue
            patch, m = frame[y0:y1, x0:x1], mask[y0:y1, x0:x1] > 0
            if patch.size and m.any():
                self.ball += np.bincount(pixel_bins(cv2.cvtColor(patch, cv2.COLOR_BGR2HSV))[m], minlength=N_BINS)
        for d in players:
            x, y, w, h = d.bbox
            m = mask[y:y + h, x:x + w] > 0
            if m.any():
                self.body += np.bincount(pixel_bins(cv2.cvtColor(frame[y:y + h, x:x + w], cv2.COLOR_BGR2HSV))[m], minlength=N_BINS)

    def finish(self):
        if self.ball.sum() < 50 or self.body.sum() < 50:
            return self
        b, o = self.ball / self.ball.sum(), self.body / self.body.sum()
        # common on ball-like blobs, at least 8x rarer on bodies (the blobs
        # include some hands and heads, so the share itself can be modest)
        self.bins = [int(k) for k in np.argsort(-b)[:3] if b[k] >= 0.12 and b[k] >= 8 * o[k]]
        return self


class BackgroundModel:
    """Median background in a stabilised reference view."""

    def __init__(self, cfg: DetectConfig | None = None):
        self.cfg = cfg or DetectConfig()
        self.bg = None

    def build(self, frames, states):
        """frames: iterable of frames; states: CourtState per frame."""
        cfg = self.cfg
        good = [i for i, s in enumerate(states) if s.H is not None and s.status in ("lines", "manual")]
        if len(good) < 5:
            good = [i for i, s in enumerate(states) if s.H is not None]
        if len(good) < 3:
            raise RuntimeError("no usable court homographies to stabilise the background")
        pick = set(np.unique(np.array(good)[np.linspace(0, len(good) - 1, min(cfg.bg_samples, len(good))).astype(int)]))
        ref = good[len(good) // 2]
        self.H_ref = states[ref].H
        stack, valid, bins, lines = [], [], [], []
        for i, f in enumerate(frames):
            if i not in pick:
                continue
            if not stack:
                h, w = f.shape[:2]
                cs = cfg.canvas_scale
                self.size = (int(w * cs), int(h * cs))
                self.T = np.array([[1, 0, (cs - 1) / 2 * w], [0, 1, (cs - 1) / 2 * h], [0, 0, 1.0]])
            A = self.T @ self.H_ref @ np.linalg.inv(states[i].H)
            wf = cv2.warpPerspective(f, A, self.size)
            stack.append(wf)
            valid.append(cv2.warpPerspective(np.full(f.shape[:2], 255, np.uint8), A, self.size) > 250)
            hsv = cv2.cvtColor(wf, cv2.COLOR_BGR2HSV).astype(np.int32)
            bins.append(((hsv[..., 0] // 12) * 16 + (hsv[..., 1] // 64) * 4 + hsv[..., 2] // 64).astype(np.uint8))
            lines.append(cv2.warpPerspective(line_mask(f, CourtConfig()), A, self.size) > 0)
        stack, valid, bins, lines = np.array(stack), np.array(valid), np.array(bins), np.array(lines)
        # Players stand in the same spots for much of a match, so a plain median
        # keeps "ghosts" of them. On the floor we therefore only trust samples
        # showing one of the floor's dominant colours (or a painted line).
        # Floor that is covered in every sample (the net, a post, or a player
        # who never moves) stays as it is: those cannot be told apart.
        m = cfg.margin_side
        poly = apply_h(self.T @ self.H_ref, [(-m, -m), (9 + m, -m), (9 + m, 18 + m), (-m, 18 + m)])
        floor = np.zeros(self.size[::-1], np.uint8)
        cv2.fillPoly(floor, [np.clip(poly, -3e4, 3e4).astype(np.int32)], 255)
        floor = floor > 0
        from ..geometry import COURT_LINES
        on_line = np.zeros(self.size[::-1], np.uint8)       # only painted court lines count as lines
        for _, a, b in COURT_LINES:
            q = apply_h(self.T @ self.H_ref, [a, b])
            if np.isfinite(q).all():
                cv2.line(on_line, tuple(np.clip(q[0], -3e4, 3e4).astype(int)), tuple(np.clip(q[1], -3e4, 3e4).astype(int)), 255, 7)
        lines &= (on_line > 0)[None]
        # A floor colour is one that is spread over a large part of the floor;
        # a jersey colour is packed into a few small patches, however many
        # pixels it adds up to.
        cs_ = 16
        hh, ww = floor.shape
        gh, gw = hh // cs_, ww // cs_
        cell = (np.arange(gh * cs_)[:, None] // cs_) * gw + (np.arange(gw * cs_)[None] // cs_)
        fl = floor[:gh * cs_, :gw * cs_]
        floor_cells = np.bincount(cell[fl], minlength=gh * gw) >= 0.5 * cs_ * cs_
        cover = np.zeros(256)
        for k in range(len(bins)):
            ok_px = fl & valid[k, :gh * cs_, :gw * cs_]
            hist = np.bincount(cell[ok_px] * 256 + bins[k, :gh * cs_, :gw * cs_][ok_px],
                               minlength=gh * gw * 256).reshape(gh * gw, 256)
            cover += (hist[floor_cells] >= 0.25 * cs_ * cs_).sum(0)
        cover /= max(1, len(bins) * floor_cells.sum())
        dominant = cover >= cfg.floor_colour_share
        trusted = valid & (~floor[None] | dominant[bins] | lines)
        bg = np.zeros(stack.shape[1:], np.uint8)
        ok = np.zeros(stack.shape[1:3], bool)
        import warnings
        for r in range(0, stack.shape[1], 64):
            sl = slice(r, r + 64)
            use = np.where((trusted[:, sl].sum(0) >= 3)[None], trusted[:, sl], valid[:, sl])
            s = stack[:, sl].astype(np.float32)
            s[~use] = np.nan
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                med = np.nanmedian(s, axis=0)
            ok[sl] = valid[:, sl].sum(0) >= 3
            bg[sl] = np.nan_to_num(med).astype(np.uint8)
        self.dominant = dominant
        self.bg, self.ok = cv2.GaussianBlur(bg, (3, 3), 0), ok.astype(np.uint8) * 255
        return self

    def foreground(self, frame, H):
        """Foreground mask of `frame` given its court->image homography."""
        h, w = frame.shape[:2]
        A = H @ np.linalg.inv(self.H_ref) @ np.linalg.inv(self.T)       # canvas -> this frame
        bg = cv2.warpPerspective(self.bg, A, (w, h))
        ok = cv2.warpPerspective(self.ok, A, (w, h)) > 250
        diff = cv2.absdiff(cv2.GaussianBlur(frame, (3, 3), 0), bg).max(axis=2)
        m = (diff > self.cfg.fg_thresh) & ok
        # Bare floor is never a player. Where the background itself is wrong
        # (a base position that always had someone standing on it never showed
        # its floor), an empty spot would otherwise light up as a phantom.
        mg = self.cfg.margin_side
        poly = apply_h(H, [(-mg, -mg), (9 + mg, -mg), (9 + mg, 18 + mg), (-mg, 18 + mg)])
        if np.isfinite(poly).all():
            region = np.zeros((h, w), np.uint8)
            cv2.fillPoly(region, [np.clip(poly, -3e4, 3e4).astype(np.int32)], 1)
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV).astype(np.int32)
            fb = (hsv[..., 0] // 12) * 16 + (hsv[..., 1] // 64) * 4 + hsv[..., 2] // 64
            m &= ~((region > 0) & self.dominant[fb])
        m = m.astype(np.uint8) * 255
        m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
        return m


def _in_play_area(xy, cfg):
    return (-cfg.margin_side <= xy[0] <= 9 + cfg.margin_side) and (-cfg.margin_end <= xy[1] <= 18 + cfg.margin_end)


def detect_from_mask(frame, mask, H, cfg: DetectConfig | None = None, colours: TeamColours | None = None,
                     ball_colour: BallColour | None = None):
    """Split a foreground mask into player detections and ball candidates.

    Returns (players: list[Detection], balls: list[(u, v, radius_px)])."""
    cfg = cfg or DetectConfig()
    Hinv = np.linalg.inv(H)
    n, lab, stats, cent = cv2.connectedComponentsWithStats(mask, connectivity=8)
    corners = [(0, 0), (9, 0), (0, 18), (9, 18)]
    scales = [metric_scale(H, c) for c in corners]
    d_lo, d_hi = 2 * BALL_R * min(scales) * 0.55, 2 * BALL_R * max(scales) * 2.2
    players, balls = [], []
    for i in range(1, n):
        x, y, w, h, area = stats[i]
        if area < 6:
            continue
        if d_lo <= w <= d_hi and d_lo <= h <= d_hi and 0.6 <= w / h <= 1.7 and area >= 0.6 * w * h:
            balls.append((float(cent[i][0]), float(cent[i][1]), float((w + h) / 4)))
            continue
        comp = (lab[y:y + h, x:x + w] == i)
        foot = _foot(comp, x, y)
        xy = apply_h(Hinv, [foot])[0]
        if not np.isfinite(xy).all() or not _in_play_area(xy, cfg):
            continue
        s = metric_scale(H, xy)
        ph = cfg.person_h * s
        if h < cfg.min_h_frac * ph or w < 0.18 * s:
            continue
        if h <= 1.3 * ph and w <= 1.15 * s:
            J = np.linalg.inv(local_jacobian(H, xy))
            players.append(Detection((int(x), int(y), int(w), int(h)), foot, xy, s,
                                     appearance(frame, mask, (x, y, w, h)), J))
            continue
        cands = None
        if colours is not None and getattr(colours, "usable", False):
            cands = _split_by_jersey(frame, comp, x, y, H, Hinv, cfg, colours)
        if not cands:
            cands = [(ft, occ, None) for ft, occ in _split(comp, x, y, H, Hinv, cfg)]
        for ft, occluded, team in cands:                            # several players in one blob
            pxy = apply_h(Hinv, [ft])[0]
            if not np.isfinite(pxy).all() or not _in_play_area(pxy, cfg):
                continue
            sj = metric_scale(H, pxy)
            hw, pj = int(0.3 * sj) + 1, cfg.person_h * sj
            x0, x1 = max(x, int(ft[0]) - hw), min(x + w, int(ft[0]) + hw + 1)
            y0, y1 = max(y, int(ft[1] - pj)), min(y + h, int(ft[1]))
            if x1 - x0 < 2 or y1 - y0 < 4:
                continue
            bb = (x0, y0, x1 - x0, y1 - y0)
            players.append(Detection(bb, ft, pxy, sj, appearance(frame, mask, bb),
                                     np.linalg.inv(local_jacobian(H, pxy)), occluded=occluded, team=team))
    if colours is not None:
        for d in players:
            if d.team is None:
                d.team = colours.team_of(d)
    if ball_colour is not None and ball_colour.bins:
        # ball-coloured patches inside the foreground, including on top of players
        bm = ((mask > 0) & np.isin(pixel_bins(cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)), ball_colour.bins)).astype(np.uint8)
        nb, _, bst, bcen = cv2.connectedComponentsWithStats(bm, connectivity=8)
        for i in range(1, nb):
            bx, by, bw, bh, ba = bst[i]
            if 0.5 * d_lo <= bw <= d_hi and 0.5 * d_lo <= bh <= d_hi and 0.5 <= bw / bh <= 2.0 and ba >= 0.5 * bw * bh \
                    and not any(np.hypot(bcen[i][0] - u, bcen[i][1] - v) < 3 for u, v, _ in balls):
                balls.append((float(bcen[i][0]), float(bcen[i][1]), float((bw + bh) / 4)))
    # the same player reported twice (e.g. jersey cut in two): keep the better one
    players.sort(key=lambda d: (d.occluded, -d.bbox[2] * d.bbox[3]))
    kept = []
    for d in players:
        if not any(np.linalg.norm(d.xy - k.xy) < 0.55 and abs(d.foot[0] - k.foot[0]) < 0.35 * d.scale
                   and (d.team is None or k.team is None or d.team == k.team) for k in kept):
            kept.append(d)
    return kept, balls


SHOULDER, HEM = 0.82, 0.50      # jersey top / bottom as a fraction of standing height


def _split_by_jersey(frame, comp, x0, y0, H, Hinv, cfg, colours):
    """Find each player's jersey inside a merged blob.

    Jerseys stay visible when legs are hidden behind a nearer player, so every
    jersey-coloured patch becomes one player, placed from its shoulder line and
    an assumed standing height."""
    h, w = comp.shape
    bins = pixel_bins(cv2.cvtColor(frame[y0:y0 + h, x0:x0 + w], cv2.COLOR_BGR2HSV))
    out = []

    def feet_from_shoulder(cx, top):
        """Iterate: pixel height depends on where the feet are."""
        xy = apply_h(Hinv, [(cx, top + 1)])[0]
        sj = metric_scale(H, xy) if np.isfinite(xy).all() else 0.0
        fv = top
        for _ in range(4):
            fv = top + SHOULDER * cfg.person_h * sj
            xy = apply_h(Hinv, [(cx, fv)])[0]
            if not np.isfinite(xy).all():
                return None, None
            sj = metric_scale(H, xy)
        return fv, sj

    for team in ("near", "far"):
        m = (comp & np.isin(bins, colours.bins[team])).astype(np.uint8)
        n, lab, st, cen = cv2.connectedComponentsWithStats(m, connectivity=8)
        frags = sorted([(st[i][1], i) for i in range(1, n) if st[i][4] >= 4])
        used = set()
        for top_rel, i in frags:                         # top-most first: the rear players
            if i in used:
                continue
            top, cx = y0 + top_rel, x0 + cen[i][0]
            fv, sj = feet_from_shoulder(cx, top)
            if fv is None:
                continue
            # everything inside this player's jersey rectangle is the same jersey
            # (the printed number cuts a jersey into several patches)
            grp = [i] + [j for _, j in frags if j not in used and j != i
                   and abs(x0 + cen[j][0] - cx) < 0.45 * sj and -1 <= y0 + cen[j][1] - top <= 0.68 * sj]
            used.add(i)
            used.update(grp)
            gx0 = min(st[j][0] for j in grp); gx1 = max(st[j][0] + st[j][2] for j in grp)
            gy1 = max(st[j][1] + st[j][3] for j in grp)
            bw, bh, area = gx1 - gx0, gy1 - top_rel, sum(st[j][4] for j in grp)
            if bw < 0.22 * sj or bh < 0.3 * sj or area < 0.08 * sj * sj:
                continue
            k2 = 2 if bw > 0.85 * sj else 1               # two jerseys side by side
            for j in range(k2):
                cxj = x0 + gx0 + (j + 0.5) * bw / k2
                c0, c1 = int(max(0, cxj - x0 - 0.2 * sj)), int(min(w, cxj - x0 + 0.2 * sj + 1))
                rows = np.where(comp[:, c0:c1].any(1))[0]
                low = y0 + rows[-1] + 1 if len(rows) else fv
                ph = cfg.person_h * sj
                if fv > low + 0.2 * ph:                   # would stand below the blob: not a jersey (e.g. socks)
                    continue
                if abs(low - fv) < 0.2 * ph:              # this player's own feet are visible
                    out.append(((float(cxj), float(low)), False, team))
                else:
                    out.append(((float(cxj), float(fv)), True, team))
    return out


def _split(comp, x0, y0, H, Hinv, cfg):
    """Colour-free fallback for a merged blob.

    Feet are read from the lower outline where visible; players whose legs are
    hidden behind someone nearer are located from the top of their head and an
    assumed standing height (less accurate: flagged `occluded`)."""
    from scipy.ndimage import maximum_filter1d, minimum_filter1d, uniform_filter1d
    cols = np.where(comp.any(0))[0]
    h = comp.shape[0]
    top = np.full(comp.shape[1], float(h))
    bot = np.full(comp.shape[1], -1.0)
    top[cols] = comp[:, cols].argmax(0)
    bot[cols] = h - 1 - comp[::-1][:, cols].argmax(0)
    foot0 = apply_h(Hinv, [(x0 + cols.mean(), y0 + h)])[0]
    s0 = metric_scale(H, foot0)
    win = max(3, int(0.6 * s0) | 1)
    sm = max(1, int(0.12 * s0))
    tops, bots = uniform_filter1d(top, sm, mode="nearest"), uniform_filter1d(bot, sm, mode="nearest")
    out = []
    for c in cols[(bots[cols] == maximum_filter1d(bots, win, mode="nearest")[cols])]:
        ft = (float(x0 + c), float(y0 + bot[c] + 1))
        if not any(abs(ft[0] - o[0][0]) < win / 2 for o in out):
            out.append((ft, False))
    heads = []
    for c in cols[(tops[cols] == minimum_filter1d(tops, win, mode="nearest")[cols])]:
        if any(abs(c - hc) < win / 2 for hc in heads):
            continue
        heads.append(c)
        fv = y0 + top[c] + cfg.person_h * s0
        for _ in range(3):
            xy = apply_h(Hinv, [(x0 + c, fv)])[0]
            if not np.isfinite(xy).all():
                break
            fv = y0 + top[c] + cfg.person_h * metric_scale(H, xy)
        ft = (float(x0 + c), float(fv))
        ph = fv - (y0 + top[c])
        if any(abs(ft[0] - o[0][0]) < 0.45 * s0 and abs(ft[1] - o[0][1]) < 0.3 * ph for o in out):
            continue
        out.append((ft, True))
    return out


def _foot(comp, x0, y0):
    rows = np.where(comp.any(1))[0]
    bottom = rows[-1]
    band = comp[max(rows[0], bottom - max(2, len(rows) // 8)):bottom + 1]
    cols = np.where(band.any(0))[0]
    return (float(x0 + cols.mean()), float(y0 + bottom + 1))


def detect_hog(frame, H, cfg: DetectConfig | None = None, _hog=[]):
    """OpenCV's built-in HOG pedestrian detector (no background model needed).
    Slower and much less sensitive on small/far players; smoke-tested only."""
    cfg = cfg or DetectConfig()
    if not _hog:
        hog = cv2.HOGDescriptor()
        hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
        _hog.append(hog)
    up = 2.0
    big = cv2.resize(frame, None, fx=up, fy=up)
    rects, _ = _hog[0].detectMultiScale(big, winStride=(8, 8), padding=(8, 8), scale=1.08)
    return _boxes_to_detections(frame, [tuple(int(v / up) for v in r) for r in rects], H, cfg)


def detect_torch(frame, H, cfg: DetectConfig, _model=[]):
    """torchvision Faster R-CNN (MobileNetV3) person detector, CPU.

    UNTESTED in the build environment (PyTorch was not installed there).
    Needs `torch_weights` pointing at a locally stored state dict; nothing is
    downloaded at runtime."""
    import torch
    import torchvision
    if not _model:
        m = torchvision.models.detection.fasterrcnn_mobilenet_v3_large_320_fpn(weights=None, weights_backbone=None)
        m.load_state_dict(torch.load(cfg.torch_weights, map_location="cpu"))
        _model.append(m.eval())
    t = torch.from_numpy(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)).permute(2, 0, 1).float() / 255
    with torch.no_grad():
        out = _model[0]([t])[0]
    boxes = [(int(b[0]), int(b[1]), int(b[2] - b[0]), int(b[3] - b[1]))
             for b, l, s in zip(out["boxes"].tolist(), out["labels"].tolist(), out["scores"].tolist())
             if l == 1 and s > 0.5]
    return _boxes_to_detections(frame, boxes, H, cfg)


def _boxes_to_detections(frame, boxes, H, cfg):
    Hinv = np.linalg.inv(H)
    out = []
    fh, fw = frame.shape[:2]
    for x, y, w, h in boxes:
        x, y = max(0, x), max(0, y)
        w, h = min(w, fw - x), min(h, fh - y)
        if w < 4 or h < 8:
            continue
        ft = (x + w / 2.0, float(y + h))
        xy = apply_h(Hinv, [ft])[0]
        if not np.isfinite(xy).all() or not _in_play_area(xy, cfg):
            continue
        s = metric_scale(H, xy)
        if not (cfg.min_h_frac * cfg.person_h * s <= h <= cfg.max_h_frac * cfg.person_h * s * 1.2):
            continue
        out.append(Detection((x, y, w, h), ft, xy, s, appearance(frame, None, (x, y, w, h)),
                             np.linalg.inv(local_jacobian(H, xy))))
    return out
