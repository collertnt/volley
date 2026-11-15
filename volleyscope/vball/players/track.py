"""Multi-player tracking in court coordinates, re-identification, identities.

Tracking happens on the floor plane (metres), not in the image: two players
crossing in the image are usually still a metre apart on the court.

Identity is built in two layers:
  * segments  - unbroken stretches where a player was followed frame to frame;
  * identities - segments stitched together after occlusions, camera cuts or
    time off-frame, using (in order of trust) the jersey number, position
    continuity, and appearance.
"""
from __future__ import annotations

import json
from collections import Counter, defaultdict
from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import linear_sum_assignment

from ..geometry import NET_Y
from .detect import Detection, hist_sim
from .jersey import NumberVotes, read_number


@dataclass
class TrackConfig:
    gate: float = 1.6             # metres, base association radius
    max_coast: float = 0.7        # seconds a track survives without a detection
    confirm_hits: int = 5
    max_per_team: int = 6
    reid_speed: float = 6.0       # m/s a lost player may have moved
    jersey_every: int = 2         # read numbers every n-th frame
    jersey_min_h: int = 80        # px (in the frame used for reading)
    jersey_backend: str = "template"
    interp_gap: float = 0.5       # seconds of missing data filled by interpolation


class Segment:
    _next = 0

    def __init__(self, det: Detection, frame: int, dt: float):
        self.sid = Segment._next
        Segment._next += 1
        self.x = np.array([det.xy[0], det.xy[1], 0.0, 0.0])
        self.P = np.diag([0.3, 0.3, 4.0, 4.0])
        self.app = det.hist.copy()
        self.team_votes = Counter()
        self.frames, self.xy, self.bbox, self.rays, self.occl = [], [], [], [], []
        self.hits, self.miss = 0, 0
        self.status = "tentative"
        self.identity = None
        self.votes = NumberVotes()
        self._record(det, frame)

    # --- Kalman filter, constant velocity --------------------------------
    def predict(self, dt):
        F = np.eye(4)
        F[0, 2] = F[1, 3] = dt
        a = 6.0
        q = np.diag([(0.5 * a * dt * dt) ** 2] * 2 + [(a * dt) ** 2] * 2)
        self.x = F @ self.x
        self.x[2:] *= 0.95
        self.P = F @ self.P @ F.T + q

    def update(self, det: Detection, frame: int):
        J = det.Jinv if det.Jinv is not None else np.eye(2) * 0.02
        sv = 4.0 * (3.0 if det.occluded else 1.0)       # feet row is the uncertain one
        R = J @ np.diag([2.0 ** 2, sv ** 2]) @ J.T + np.eye(2) * 0.05 ** 2
        Hm = np.zeros((2, 4))
        Hm[0, 0] = Hm[1, 1] = 1
        S = Hm @ self.P @ Hm.T + R
        K = self.P @ Hm.T @ np.linalg.inv(S)
        self.x = self.x + K @ (det.xy - self.x[:2])
        self.P = (np.eye(4) - K @ Hm) @ self.P
        if not det.occluded:
            self.app = 0.93 * self.app + 0.07 * det.hist
        self._record(det, frame)

    def _record(self, det, frame):
        self.hits += 1
        self.miss = 0
        if det.team:
            self.team_votes[det.team] += 1.0
        elif abs(det.xy[1] - NET_Y) > 1.0:
            self.team_votes["near" if det.xy[1] < NET_Y else "far"] += 0.3
        self.frames.append(frame)
        self.xy.append(self.x[:2].copy())
        self.bbox.append(tuple(int(v) for v in det.bbox))
        ray = (det.Jinv @ np.array([0.0, -1.0])) if det.Jinv is not None else np.array([0.0, 1.0])
        self.rays.append(ray / (np.linalg.norm(ray) + 1e-9))
        self.occl.append(bool(det.occluded))

    @property
    def team(self):
        return self.team_votes.most_common(1)[0][0] if self.team_votes else None

    @property
    def pos(self):
        return self.x[:2]


class PlayerTracker:
    def __init__(self, fps: float = 30.0, cfg: TrackConfig | None = None):
        self.cfg = cfg or TrackConfig()
        self.fps = fps
        self.dt = 1.0 / fps
        self.live: list[Segment] = []
        self.done: list[Segment] = []          # finished segments (lost or ended)
        self.lost: list[Segment] = []          # last segment of each identity not currently live
        self.n_ident = 0
        self.frame = -1

    # ------------------------------------------------------------------
    def _active_count(self, team):
        return sum(1 for s in self.live if s.status == "active" and s.team == team)

    def update(self, frame_idx: int, dets: list[Detection], frame=None, full_frame=None, scale=1.0):
        """Advance one frame. `frame` (and optionally the full-resolution frame)
        is only used to read jersey numbers."""
        cfg = self.cfg
        self.frame = frame_idx
        for s in self.live:
            s.predict(self.dt)
        # --- associate detections with live segments -------------------
        matched_d, matched_s = set(), set()
        if self.live and dets:
            C = np.full((len(self.live), len(dets)), 1e6)
            for i, s in enumerate(self.live):
                gate = min(4.0, cfg.gate + 4.0 * s.miss * self.dt)
                for j, d in enumerate(dets):
                    dteam = d.team or (None if abs(d.xy[1] - NET_Y) < 1.5 else ("near" if d.xy[1] < NET_Y else "far"))
                    if dteam and s.team and dteam != s.team:
                        continue                          # players never change sides mid-rally
                    dist = np.linalg.norm(d.xy - s.pos)
                    if dist < gate * (1.5 if d.occluded else 1.0):
                        C[i, j] = dist / gate + 0.4 * (1 - hist_sim(s.app, d.hist)) + (0.2 if s.status == "tentative" else 0)
            for i, j in zip(*linear_sum_assignment(C)):
                if C[i, j] < 1e5:
                    self.live[i].update(dets[j], frame_idx)
                    matched_s.add(i); matched_d.add(j)
                    self._read_number(self.live[i], dets[j], frame, full_frame, scale)
        # --- unmatched segments -----------------------------------------
        keep = []
        for i, s in enumerate(self.live):
            if i not in matched_s:
                s.miss += 1
                limit = cfg.max_coast * self.fps if s.status == "active" else 3
                if s.miss > limit:
                    if s.status == "active":
                        self.done.append(s)
                        self.lost.append(s)
                    continue
            keep.append(s)
        self.live = keep
        # --- unmatched detections start tentative segments ----------------
        for j, d in enumerate(dets):
            if j not in matched_d:
                if any(np.linalg.norm(d.xy - s.pos) < 0.5 for s in self.live):
                    continue                              # duplicate of something already tracked
                self.live.append(Segment(d, frame_idx, self.dt))
        # --- confirmation: new identity, or a lost player coming back -----
        for s in self.live:
            if s.status == "tentative" and s.hits >= cfg.confirm_hits:
                self._confirm(s)

    def _read_number(self, seg, det, frame, full_frame, scale):
        cfg = self.cfg
        if frame is None or det.occluded or self.frame % cfg.jersey_every:
            return
        img, bb = frame, det.bbox
        if full_frame is not None and scale != 1.0:
            img, bb = full_frame, tuple(int(round(v / scale)) for v in det.bbox)
        if bb[3] < cfg.jersey_min_h:
            return
        num, conf = read_number(img, bb, backend=cfg.jersey_backend)
        seg.votes.add(num, conf)

    def _confirm(self, seg: Segment):
        cfg = self.cfg
        team = seg.team
        t0, p0 = seg.frames[0], np.array(seg.xy[0])
        best, best_cost = None, 1e9
        for old in self.lost:
            if old.team and team and old.team != team:
                continue
            gap = (t0 - old.frames[-1]) * self.dt
            if gap <= 0:
                continue
            dist = np.linalg.norm(p0 - np.array(old.xy[-1]))
            reach = 2.0 + cfg.reid_speed * gap
            if dist > reach:
                continue
            cost = (1 - hist_sim(old.app, seg.app)) * 3.0 + min(1.0, dist / reach)
            if cost < best_cost:
                best, best_cost = old, cost
        full = team is not None and self._active_count(team) >= cfg.max_per_team
        if best is not None and (best_cost < 1.2 or full):
            seg.identity = best.identity
            self.lost.remove(best)
        elif full:
            return                                         # a 7th player on one side: stay tentative
        else:
            seg.identity = self.n_ident
            self.n_ident += 1
        seg.status = "active"

    def finish(self):
        """Close all live segments and build identities."""
        for s in self.live:
            if s.status == "active":
                self.done.append(s)
        self.live = []
        return build_identities(self.done, self.fps, self.cfg)


# ----------------------------------------------------------------------
@dataclass
class Identity:
    id: int
    team: str | None
    number: int | None
    number_conf: float
    frames: np.ndarray
    xy: np.ndarray
    observed: np.ndarray            # False where interpolated
    airborne: np.ndarray            # True during detected jumps
    bbox: dict = field(default_factory=dict)      # frame -> bbox, observed frames only
    number_source: str = "none"     # read | manual | none
    flags: list = field(default_factory=list)

    @property
    def label(self):
        return f"#{self.number}" if self.number is not None else f"T{self.id}"

    def at(self, frame):
        i = np.searchsorted(self.frames, frame)
        if i < len(self.frames) and self.frames[i] == frame:
            return self.xy[i]
        return None


class PlayerTracks:
    def __init__(self, identities: list[Identity], fps: float):
        self.identities = {i.id: i for i in identities}
        self.fps = fps

    def at(self, frame, max_gap=0):
        """[(identity, xy)] for every identity with a position at `frame`."""
        out = []
        for ident in self.identities.values():
            p = ident.at(frame)
            if p is not None:
                out.append((ident, p))
        return out

    def by_number(self, team, number):
        return [i for i in self.identities.values() if i.team == team and i.number == number]

    def to_json(self):
        return {"fps": self.fps, "identities": [
            {"id": i.id, "team": i.team, "number": i.number, "number_conf": round(i.number_conf, 3),
             "number_source": i.number_source, "flags": i.flags,
             "frames": i.frames.tolist(), "xy": np.round(i.xy, 3).tolist(),
             "observed": i.observed.astype(int).tolist(), "airborne": i.airborne.astype(int).tolist(),
             "bbox": {str(k): v for k, v in i.bbox.items()}} for i in self.identities.values()]}

    @classmethod
    def from_json(cls, d):
        ids = [Identity(i["id"], i["team"], i["number"], i["number_conf"], np.array(i["frames"], int),
                        np.array(i["xy"], float).reshape(-1, 2), np.array(i["observed"], bool),
                        np.array(i["airborne"], bool), {int(k): tuple(v) for k, v in i.get("bbox", {}).items()},
                        i.get("number_source", "none"), i.get("flags", [])) for i in d["identities"]]
        return cls(ids, d["fps"])


def suppress_jumps(xy, rays, fps, win_s=1.5, thresh=0.45):
    """Remove the apparent 'run away from the camera' a jump produces.

    Feet leaving the floor project further from the camera along the viewing
    ray. A short excursion along that ray that returns to where it started is
    treated as a jump: the along-ray coordinate is replaced by its running
    median and the frames are flagged airborne."""
    from scipy.ndimage import median_filter
    n = len(xy)
    if n < 5:
        return xy, np.zeros(n, bool)
    ray = np.median(rays, axis=0)
    ray /= np.linalg.norm(ray) + 1e-9
    along = xy @ ray
    k = max(3, int(win_s * fps) | 1)
    base = median_filter(along, size=k, mode="nearest")
    air = (along - base) > thresh
    out = xy.copy()
    out[air] += ((base - along)[air])[:, None] * ray[None]
    return out, air


def build_identities(segments: list[Segment], fps: float, cfg: TrackConfig | None = None) -> PlayerTracks:
    cfg = cfg or TrackConfig()
    # 1. online identities, then let confident jersey numbers overrule them
    groups = defaultdict(list)
    for s in segments:
        groups[s.identity].append(s)
    next_id = max([g for g in groups if g is not None], default=-1) + 1
    final = []                                   # list of (segments, team, number, conf)
    for gid, segs in groups.items():
        segs.sort(key=lambda s: s.frames[0])
        nums = [s.votes.result() for s in segs]
        confident = {n for n, c in nums if n is not None}
        if len(confident) <= 1:
            final.append(segs)
            continue
        # the chain mixes two numbers: split it, unnumbered pieces follow the
        # numbered piece before them
        cur_num, parts = None, defaultdict(list)
        first = next(n for n, _ in nums if n is not None)
        for s, (n, _) in zip(segs, nums):
            if n is not None:
                cur_num = n
            parts[cur_num if cur_num is not None else first].append(s)
        final.extend(parts.values())

    def summary(segs):
        votes = NumberVotes()
        team = Counter()
        for s in segs:
            votes.votes.update(s.votes.votes)
            votes.n += s.votes.n
            team.update(s.team_votes)
        num, conf = votes.result()
        return (team.most_common(1)[0][0] if team else None), num, conf

    # 2. merge chains that carry the same team + number and never overlap in time
    merged = []
    for segs in sorted(final, key=lambda g: g[0].frames[0]):
        team, num, conf = summary(segs)
        target = None
        if num is not None:
            for m in merged:
                mt, mn, _ = summary(m)
                if mn == num and mt == team:
                    fa = set(f for s in m for f in s.frames)
                    if not any(f in fa for s in segs for f in s.frames):
                        target = m
                        break
        if target is not None:
            target.extend(segs)
        else:
            merged.append(list(segs))

    idents = []
    for k, segs in enumerate(merged):
        segs.sort(key=lambda s: s.frames[0])
        team, num, conf = summary(segs)
        frames = np.concatenate([s.frames for s in segs]).astype(int)
        xy = np.concatenate([np.array(s.xy) for s in segs])
        rays = np.concatenate([np.array(s.rays) for s in segs])
        bbs = [b for s in segs for b in s.bbox]
        order = np.argsort(frames, kind="stable")
        frames, xy, rays = frames[order], xy[order], rays[order]
        bbs = [bbs[i] for i in order]
        _, uniq = np.unique(frames, return_index=True)
        frames, xy, rays = frames[uniq], xy[uniq], rays[uniq]
        bbs = [bbs[i] for i in uniq]
        xy, air = suppress_jumps(xy, rays, fps)
        # fill short gaps by interpolation
        gap = int(cfg.interp_gap * fps)
        F, X, O, A = [frames[0]], [xy[0]], [True], [air[0]]
        for i in range(1, len(frames)):
            d = frames[i] - frames[i - 1]
            if 1 < d <= gap:
                for f in range(frames[i - 1] + 1, frames[i]):
                    a = (f - frames[i - 1]) / d
                    F.append(f); X.append((1 - a) * xy[i - 1] + a * xy[i]); O.append(False); A.append(False)
            F.append(frames[i]); X.append(xy[i]); O.append(True); A.append(air[i])
        idents.append(Identity(k, team, num, conf, np.array(F), np.array(X), np.array(O), np.array(A),
                               dict(zip(frames.tolist(), bbs)), "read" if num is not None else "none"))
    # 3. two live players can't share a number: keep the better-supported one
    seen = {}
    for ident in sorted(idents, key=lambda i: -i.number_conf):
        if ident.number is None:
            continue
        key = (ident.team, ident.number)
        if key in seen:
            ident.flags.append(f"number {ident.number} also read on T{seen[key]}; dropped here")
            ident.number, ident.number_conf, ident.number_source = None, 0.0, "none"
        else:
            seen[key] = ident.id
    return PlayerTracks(idents, fps)


# ----------------------------------------------------------------------
def apply_corrections(tracks: PlayerTracks, corr: dict) -> PlayerTracks:
    """Apply a manual corrections file (see README):

        numbers: {track_id: jersey_number}      # set / fix a number
        teams:   {track_id: near|far}
        merge:   [[keep_id, other_id, ...]]     # same player, tracked as several
        ignore:  [track_id, ...]                # not a player (referee, coach)
    """
    ids = tracks.identities
    for tid in corr.get("ignore", []) or []:
        ids.pop(int(tid), None)
    for group in corr.get("merge", []) or []:
        group = [int(g) for g in group if int(g) in ids]
        if len(group) < 2:
            continue
        keep = ids[group[0]]
        for g in group[1:]:
            o = ids.pop(g)
            fr = np.concatenate([keep.frames, o.frames])
            order = np.argsort(fr, kind="stable")
            _, uniq = np.unique(fr[order], return_index=True)
            sel = order[uniq]
            keep.frames = fr[sel]
            keep.xy = np.concatenate([keep.xy, o.xy])[sel]
            keep.observed = np.concatenate([keep.observed, o.observed])[sel]
            keep.airborne = np.concatenate([keep.airborne, o.airborne])[sel]
            keep.bbox.update(o.bbox)
            if keep.number is None and o.number is not None:
                keep.number, keep.number_conf, keep.number_source = o.number, o.number_conf, o.number_source
    for tid, team in (corr.get("teams", {}) or {}).items():
        if int(tid) in ids:
            ids[int(tid)].team = team
    for tid, num in (corr.get("numbers", {}) or {}).items():
        if int(tid) in ids:
            ids[int(tid)].number = None if num is None else int(num)
            ids[int(tid)].number_conf = 1.0
            ids[int(tid)].number_source = "manual"
    return tracks


def load_corrections(path) -> dict:
    import yaml
    with open(path) as fh:
        return yaml.safe_load(fh) or {}
