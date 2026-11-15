"""Ball tracking: image tracklets -> 3D ballistic flights -> junctions.

A single camera cannot see depth, but a ball in free flight follows a parabola
with known gravity, and that is enough to solve for its 3D path (ignoring air
drag and spin - see README). The path between two touches is one "flight";
where one flight ends and the next begins is a touch or a bounce.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.ndimage import median_filter

from ..geometry import (BALL_R, GVEC, NET_Y, calib_from_h, fit_ballistic, fit_ballistic_anchored,
                        focal_from_h)


@dataclass
class BallConfig:
    max_gap: int = 8              # frames a tracklet may miss
    min_len: int = 5
    fit_rms: float = 2.5          # px
    link_gap_s: float = 1.2       # flights further apart than this are separate rallies
    min_travel_px: float = 15.0
    min_speed_px: float = 1.2     # median image speed (px/frame) a real ball tracklet exceeds
    short_rms: float = 1.2        # stricter fit required of short flights


@dataclass
class Flight:
    t0: float
    t1: float
    p0: np.ndarray
    v0: np.ndarray
    rms: float
    n: int
    obs: list = field(default_factory=list)     # (frame, u, v) used in the fit
    end_kind: str | None = None                 # 'floor' when the flight was pinned to a seen landing
    pinned: bool = False                        # start/end fixed from neighbouring flights
    land_gap_px: float | None = None            # px between where the previous flight would land and where this one is first seen

    def pos(self, t):
        dt = t - self.t0
        return self.p0 + self.v0 * dt + 0.5 * GVEC * dt * dt

    def vel(self, t):
        return self.v0 + GVEC * (t - self.t0)

    def landing(self):
        """(time, position) where the ball centre reaches floor level, or None."""
        a, b, c = -0.5 * 9.81, self.v0[2], self.p0[2] - BALL_R
        disc = b * b - 4 * a * c
        if disc < 0:
            return None
        t = (-b - np.sqrt(disc)) / (2 * a)
        return self.t0 + t, self.pos(self.t0 + t)

    def crosses_net(self, t_from, t_to):
        a, b = self.pos(t_from)[1], self.pos(t_to)[1]
        return (a - NET_Y) * (b - NET_Y) < 0


def cameras(states, size):
    """A 3x4 projection matrix per frame (None where the court is unknown).

    Focal length is estimated per frame from the homography and median
    filtered over time, since the single-frame estimate is noisy."""
    f = np.array([focal_from_h(s.H, size) if s.H is not None else np.nan for s in states], float)
    if np.isfinite(f).sum() == 0:
        return [None] * len(states)
    filled = np.where(np.isfinite(f), f, np.nanmedian(f))
    smooth = median_filter(filled, size=min(len(f) | 1, 31), mode="nearest")
    out = []
    for s, fi in zip(states, smooth):
        cal = calib_from_h(s.H, size, fi) if s.H is not None else None
        out.append(None if cal is None else cal.P)
    return out


def link_tracklets(cands, cfg: BallConfig | None = None):
    """Greedy nearest-neighbour linking of per-frame ball candidates.

    Returns a list of tracklets, each a list of (frame, u, v)."""
    cfg = cfg or BallConfig()
    live, done = [], []
    for f, cs in enumerate(cands):
        cs = list(cs)
        used = set()
        pairs = []
        for ti, tr in enumerate(live):
            lf, lu, lv = tr[-1]
            gap = f - lf
            if len(tr) >= 2:
                pf, pu, pv = tr[-2]
                vel = np.array([lu - pu, lv - pv]) / max(1, lf - pf)
            else:
                vel = np.zeros(2)
            pred = np.array([lu, lv]) + vel * gap
            gate = (14.0 + 0.7 * np.linalg.norm(vel)) * (1 + 0.5 * (gap - 1)) + (30 if len(tr) < 2 else 0)
            for k, c in enumerate(cs):
                d = np.hypot(c[0] - pred[0], c[1] - pred[1])
                if d < gate:
                    pairs.append((d, ti, k))
        taken = set()
        for d, ti, k in sorted(pairs):                # closest prediction first, whichever tracklet it is
            if ti in taken or k in used:
                continue
            taken.add(ti); used.add(k)
            live[ti].append((f, cs[k][0], cs[k][1]))
        keep = []
        for tr in live:
            if f - tr[-1][0] > cfg.max_gap:
                done.append(tr)
            else:
                keep.append(tr)
        live = keep
        for k, c in enumerate(cs):
            if k not in used:
                live.append([(f, c[0], c[1])])
    done += live
    out = []
    for tr in done:
        for part in _fast_parts(tr, cfg):
            if len(part) < cfg.min_len:
                continue
            a = np.array(part)[:, 1:]
            if np.abs(np.diff(a, axis=0)).sum() < cfg.min_travel_px:
                continue                            # something small that does not move
            out.append(part)
    return sorted(out, key=lambda t: t[0][0])


def _fast_parts(tr, cfg):
    """Cut out stretches where the 'ball' moves at walking pace.

    A ball-coloured shoe or sleeve produces a slow track, and a real ball
    track can run into one when a rally ends near that player. Only the fast
    stretches can be a ball in flight."""
    if len(tr) < 3:
        return [tr]
    a = np.array(tr, float)
    step = np.linalg.norm(np.diff(a[:, 1:], axis=0), axis=1) / np.maximum(np.diff(a[:, 0]), 1)
    slow = median_filter(step, size=9, mode="nearest") < cfg.min_speed_px
    parts, cur = [], [tr[0]]
    for i in range(len(step)):
        if slow[i]:
            if len(cur) > 1:
                parts.append(cur)
            cur = [tr[i + 1]]
        else:
            cur.append(tr[i + 1])
    if len(cur) > 1:
        parts.append(cur)
    return parts


def _plausible(fl: Flight):
    for t in (fl.t0, 0.5 * (fl.t0 + fl.t1), fl.t1):
        p = fl.pos(t)
        if not (-8 < p[0] < 17 and -10 < p[1] < 28 and -0.6 < p[2] < 14):
            return False
    return np.linalg.norm(fl.v0) < 45


def _fit(obs, Ps, fps):
    r = fit_ballistic([o[0] / fps for o in obs], [(o[1], o[2]) for o in obs], [Ps[o[0]] for o in obs])
    if r is None:
        return None
    p0, v0, rms, _ = r
    return Flight(obs[0][0] / fps, obs[-1][0] / fps, p0, v0, rms, len(obs), list(obs))


def _resid(fl, o, Ps, fps):
    q = Ps[o[0]] @ np.r_[fl.pos(o[0] / fps), 1.0]
    return float(np.hypot(q[0] / q[2] - o[1], q[1] / q[2] - o[2]))


def fit_flights(tracklets, Ps, fps, cfg: BallConfig | None = None):
    """Cut each tracklet into ballistic flights, then stitch pieces of the
    same flight that were separated by a gap in the detections."""
    cfg = cfg or BallConfig()
    flights = []
    for tr in tracklets:
        pts = [o for o in tr if o[0] < len(Ps) and Ps[o[0]] is not None]
        n, i = len(pts), 0
        while i <= n - cfg.min_len:
            # a handful of noisy points far from the camera says little about
            # depth, so let the seed grow before deciding there is no flight here
            fl = None
            for seed in (cfg.min_len, 8, 12, 16):
                if i + seed > n:
                    break
                cand = _fit(pts[i:i + seed], Ps, fps)
                if cand is not None and cand.rms <= cfg.fit_rms and _plausible(cand):
                    fl, j = cand, i + seed
                    break
            if fl is None:
                i += 1
                continue
            while j < n:
                if _resid(fl, pts[j], Ps, fps) > 2.5 * max(1.0, fl.rms) + 2.0:
                    # one bad detection, or a real change of path?
                    if j + 2 < n and _resid(fl, pts[j + 1], Ps, fps) < 4.0 and _resid(fl, pts[j + 2], Ps, fps) < 4.0:
                        pts.pop(j); n -= 1
                        continue
                    break
                f2 = _fit(pts[i:j + 1], Ps, fps)
                if f2 is None or f2.rms > cfg.fit_rms or not _plausible(f2):
                    break
                fl, j = f2, j + 1
            flights.append(fl)
            i = j
    flights.sort(key=lambda f: f.t0)
    # two tracklets can cover the same moment (a false candidate): keep the longer flight
    out = []
    for fl in flights:
        if out and fl.t0 < out[-1].t1 - 1e-6:
            if fl.n > out[-1].n:
                out[-1] = fl
            continue
        out.append(fl)
    # stitch: if two neighbouring pieces fit one parabola, they are one flight
    merged = True
    while merged:
        merged = False
        for k in range(len(out) - 1):
            a, b = out[k], out[k + 1]
            if b.t0 - a.t1 > 0.9:
                continue
            j = _fit(a.obs + b.obs, Ps, fps)
            if j is not None and j.rms < min(cfg.fit_rms, 1.2 + max(a.rms, b.rms)) and _plausible(j):
                out[k:k + 2] = [j]
                merged = True
                break
    # robust refit, then drop short pieces that connect to nothing
    final = []
    for fl in out:
        good = [o for o in fl.obs if _resid(fl, o, Ps, fps) <= 2.0 * fl.rms + 1.0]
        if cfg.min_len <= len(good) < fl.n:
            r = _fit(good, Ps, fps)
            if r is not None and _plausible(r):
                fl = r
        final.append(fl)
    final = [fl for fl in final if fl.rms < (cfg.short_rms if fl.n < 15 else cfg.fit_rms)]
    keep = []
    for k, fl in enumerate(final):
        if fl.n >= 10:
            keep.append(fl)
            continue
        # a short piece is believed only if it joins up with a solid neighbour
        near = [o for o in (final[k - 1] if k else None, final[k + 1] if k + 1 < len(final) else None) if o is not None]
        tied = False
        for o in near:
            a, b = (o, fl) if o.t0 < fl.t0 else (fl, o)
            if b.t0 - a.t1 < 0.3 and o.n >= 10:
                tm = 0.5 * (a.t1 + b.t0)
                tied |= np.linalg.norm(a.pos(tm) - b.pos(tm)) < 1.0 and np.linalg.norm(fl.v0) < 30
        if tied:
            keep.append(fl)
    # Depth along the camera axis is weakly observed for short flights (two
    # different speeds away from the camera look almost the same). So pin
    # such flights to what is known at their ends: the previous flight's end
    # point, and either the next flight's start or the spot on the floor where
    # the ball is seen to land.
    def good(f):
        return f is not None and f.n >= 15 and f.rms < 1.0 and not f.pinned

    def rms_of(pA, tA, pB, tB, obs):
        T = tB - tA
        v = (pB - pA - 0.5 * GVEC * T * T) / T
        e = []
        for o in obs:
            dt = o[0] / fps - tA
            q = Ps[o[0]] @ np.r_[pA + v * dt + 0.5 * GVEC * dt * dt, 1.0]
            e.append(np.hypot(q[0] / q[2] - o[1], q[1] / q[2] - o[2]))
        return float(np.sqrt(np.mean(np.square(e)))), v

    for k in range(len(keep)):
        b = keep[k]
        a = keep[k - 1] if k and 0 <= b.t0 - keep[k - 1].t1 < 0.5 else None
        c = keep[k + 1] if k + 1 < len(keep) and 0 <= keep[k + 1].t0 - b.t1 < 0.9 else None
        if b.n >= 25 or not good(a):
            continue
        starts = [(tc, a.pos(tc)) for tc in np.linspace(a.t1, b.t0, max(2, int(round((b.t0 - a.t1) * fps)) * 2 + 1))
                  if a.pos(tc)[2] > 0.3]
        ends = []
        if c is not None:
            if good(c):
                ends += [(tc, c.pos(tc), "contact") for tc in
                         np.linspace(b.t1, c.t0, max(2, int(round((c.t0 - b.t1) * fps)) * 2 + 1))]
            f0, u0, v0_ = c.obs[0]
            Hz = Ps[f0][:, [0, 1, 3]] + np.outer(Ps[f0][:, 2], [0, 0, BALL_R])      # plane z = ball radius
            g = np.linalg.solve(Hz, np.array([u0, v0_, 1.0]))
            floor_pt = np.array([g[0] / g[2], g[1] / g[2], BALL_R])
            ends += [(f0 / fps - d / fps, floor_pt, "floor") for d in (0.0, 0.5, 1.0, 1.5, 2.0)]
        best = None
        for tA, pA in starts:
            for tB, pB, kind in ends:
                if tB - tA < 0.1:
                    continue
                r, v = rms_of(pA, tA, pB, tB, b.obs)
                if kind == "floor" and (v + GVEC * (tB - tA))[2] > 0:
                    continue
                if best is None or r < best[0]:
                    best = (r, v, tA, pA, kind)
        if best is not None and best[0] < 1.5:
            r, v, tA, pA, kind = best
            dt = b.t0 - tA
            nb = Flight(b.t0, b.t1, pA + v * dt + 0.5 * GVEC * dt * dt, v + GVEC * dt, r, b.n, b.obs,
                        "floor" if kind == "floor" else None, True)
            if _plausible(nb):
                keep[k] = nb
                continue
        # no usable end: at least pin the start
        best = None
        for tA, pA in starts:
            r = fit_ballistic_anchored([o[0] / fps for o in b.obs], [(o[1], o[2]) for o in b.obs],
                                       [Ps[o[0]] for o in b.obs], tA, pA)
            if r is not None and (best is None or r[1] < best[1]):
                best = (r[0], r[1], tA, pA)
        if best is not None and best[1] < max(1.5, 2.0 * b.rms):
            v, r, tA, pA = best
            dt = b.t0 - tA
            nb = Flight(b.t0, b.t1, pA + v * dt + 0.5 * GVEC * dt * dt, v + GVEC * dt, r, b.n, b.obs, None, True)
            if _plausible(nb):
                keep[k] = nb
    return keep


@dataclass
class Junction:
    t: float
    pos: np.ndarray
    kind: str                     # contact | bounce
    gap: float                    # seconds without observations around it
    mismatch: float               # metres between the two flights at the junction
    floor: tuple | None = None    # (t, pos) where the first flight would reach the floor, if within the gap


def junction(a: Flight, b: Flight) -> Junction:
    """Where and when flight a turned into flight b."""
    ts = np.linspace(a.t1 - 0.05, b.t0 + 0.05, max(3, int((b.t0 - a.t1 + 0.1) * 240)))
    d = [np.linalg.norm(a.pos(t) - b.pos(t)) for t in ts]
    k = int(np.argmin(d))
    t = float(ts[k])
    pos = 0.5 * (a.pos(t) + b.pos(t))
    land = a.landing()
    reaches_floor = land is not None and a.t1 - 0.1 <= land[0] <= b.t0 + 0.15
    if a.end_kind == "floor" and land is not None:
        return Junction(float(land[0]), land[1], "bounce", float(b.t0 - a.t1), float(d[k]))
    if reaches_floor and pos[2] >= 0.4 and d[k] > 1.5 and b.land_gap_px is not None and b.land_gap_px > 40:
        # The first path would hit the floor, but the ball next shows up
        # somewhere else: it was touched (out of sight) before it got there.
        tc = a.t1 + 0.03
        return Junction(float(tc), a.pos(tc), "contact", float(b.t0 - a.t1), float(d[k]))
    if reaches_floor and (pos[2] < 0.4 or d[k] > 1.5):
        # the two paths meet at floor level, or do not meet at all while the
        # first one runs into the floor: the ball landed
        return Junction(float(land[0]), land[1], "bounce", float(b.t0 - a.t1), float(d[k]))
    return Junction(t, pos, "contact", float(b.t0 - a.t1), float(d[k]), (float(land[0]), land[1]) if reaches_floor else None)


def track_ball(cands, states, size, fps, cfg: BallConfig | None = None):
    """Full ball pass. Returns (flights, tracklets)."""
    cfg = cfg or BallConfig()
    Ps = cameras(states, size)
    tracklets = link_tracklets(cands, cfg)
    flights = fit_flights(tracklets, Ps, fps, cfg)
    for a, b in zip(flights, flights[1:]):
        land = a.landing()
        if land is not None and b.obs and Ps[b.obs[0][0]] is not None:
            q = Ps[b.obs[0][0]] @ np.r_[land[1], 1.0]
            b.land_gap_px = float(np.hypot(q[0] / q[2] - b.obs[0][1], q[1] / q[2] - b.obs[0][2]))
    return flights, tracklets
