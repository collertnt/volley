"""Lost-point reconstruction, first-person replay and the placement simulator.

Given a point lost to an opponent's ball, this module freezes the moment the
ball was hit: where each tracked defender stood, where the ball came from,
where it landed and how long it was in the air. A simple movement model then
says who could have reached it, and by how much they were out of position.

The movement model is a rule of thumb (reaction time, running speed, a
lunge), not a measurement of any player. See README.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field

import cv2
import numpy as np

from ..geometry import BALL_R, COURT_LINES, GVEC, NET_H, NET_Y, other
from ..synth.render import Camera, _poly
from ..tendencies.core import local

REACTION = 0.30     # s before a defender starts moving
SPEED = 4.5         # m/s once moving
LUNGE = 1.0         # m covered by the final reach / dive
EYE = 1.65          # m, eye height for the first-person view


def cover_radius(T):
    """Metres a defender can cover in a ball flight of T seconds."""
    return LUNGE + SPEED * max(0.0, T - REACTION)


@dataclass
class Scenario:
    side: str                       # the defending side (near / far)
    origin: list                    # where the ball was hit (x, y, z)
    landing: list                   # where it landed (x, y)
    T: float                        # flight time, s
    defenders: list                 # [{"label", "number", "xy"}]
    shot: str | None = None
    attacker: dict | None = None
    opponents: list = field(default_factory=list)
    missing: int = 0                # defenders that were not tracked at that moment
    rally: int | None = None
    notes: list = field(default_factory=list)

    def to_json(self):
        return json.loads(json.dumps(self.__dict__, default=lambda o: np.asarray(o).tolist()))


def reconstruct(rally, tracks, side, fps):
    """Scenario for a point `side` lost to an opponent's attack or serve.

    Returns (Scenario, None) or (None, reason) when there is nothing to show."""
    if rally.winner != other(side):
        return None, "this rally was not lost by that side"
    win = next((e for e in reversed(rally.events) if e.team == other(side) and e.outcome in ("kill", "ace")), None)
    if win is None:
        return None, f"point lost on own error ({rally.reason}): there was no opponent ball to defend"
    if win.pos is None:
        return None, "positions were not recorded for this rally (hand-tagged without court clicks)"
    last_touch = next((e for e in rally.events if e.idx > win.idx and e.team == side and e.type in ("dig", "reception")), None)
    landing = (last_touch.pos if last_touch is not None else None) or win.attrs.get("target") or rally.landing
    if landing is None:
        return None, "the landing spot was not seen"
    t_end = last_touch.t if last_touch is not None else rally.t_end
    T = float(max(0.15, t_end - win.t))
    frame = win.frame
    defenders, opponents = [], []
    for ident, xy in (tracks.at(frame) if tracks is not None else []):
        row = {"label": ident.label, "number": ident.number, "xy": [float(xy[0]), float(xy[1])]}
        (defenders if ident.team == side else opponents).append(row)
    notes = []
    # one player tracked as two identities (same number), or a stray extra track
    seen, uniq = set(), []
    for d in defenders:
        if d["number"] is not None and d["number"] in seen:
            continue
        seen.add(d["number"])
        uniq.append(d)
    if len(uniq) > 6:
        uniq = uniq[:6]
    if len(uniq) < len(defenders):
        notes.append(f"{len(defenders) - len(uniq)} duplicate or extra track(s) on the defending side were left out")
    defenders = uniq
    if len(defenders) < 6:
        notes.append(f"only {len(defenders)} of 6 defenders were tracked at this moment; the others are not shown")
    if "close_line_call" in rally.flags or rally.conf < 0.7:
        notes.append("this rally was detected with low confidence")
    if last_touch is not None:
        notes.append("the ball was touched but not controlled; the touch point is used as the landing spot")
    return Scenario(side, [float(v) for v in win.pos], [float(landing[0]), float(landing[1])], T, defenders,
                    win.attrs.get("shot"), {"number": win.number, "type": win.type}, opponents,
                    max(0, 6 - len(defenders)), rally.id, notes), None


def analyse(scn: Scenario):
    """Who could have reached the ball, and what the positioning error was."""
    L = np.array(scn.landing)
    R = cover_radius(scn.T)
    rows = []
    for d in scn.defenders:
        dist = float(np.linalg.norm(np.array(d["xy"]) - L))
        short = max(0.0, dist - R)
        vec = (L - np.array(d["xy"])) / max(dist, 1e-6)
        rows.append({**d, "dist": dist, "shortfall": short,
                     "needed_xy": (np.array(d["xy"]) + vec * short).tolist()})
    rows.sort(key=lambda r: r["dist"])
    out = {"cover_radius": R, "defenders": rows, "reachable": bool(rows and rows[0]["shortfall"] == 0)}
    if not rows:
        out["diagnosis"] = "No defenders were tracked at this moment, so positioning cannot be judged."
        out["kind"] = "unknown"
        return out
    n0 = rows[0]
    ld = local(scn.side, L)[1]                               # landing distance from the net
    depths = [local(scn.side, r["xy"])[1] for r in rows]
    back = [d for d in depths if d > 1.5]
    if n0["shortfall"] == 0:
        out["kind"] = "execution"
        out["diagnosis"] = (f"{n0['label']} was {n0['dist']:.1f} m from the ball with {scn.T:.2f} s to react, inside the "
                            f"{R:.1f} m a defender can cover in that time. Positioning was fine; the ball was reachable.")
    elif ld < 3.5 and back and min(back) - ld > 1.5:
        out["kind"] = "too_deep"
        out["diagnosis"] = (f"The ball dropped {ld:.1f} m from the net and the nearest back-court defender stood "
                            f"{min(back):.1f} m from the net. Everyone was deep: {n0['label']} was {n0['dist']:.1f} m away and "
                            f"could cover {R:.1f} m, {n0['shortfall']:.1f} m short.")
    elif ld > 6.0 and max(depths) < ld - 1.5:
        out["kind"] = "too_shallow"
        out["diagnosis"] = (f"The ball landed deep, {ld:.1f} m from the net, behind the deepest defender ({max(depths):.1f} m). "
                            f"{n0['label']} was {n0['shortfall']:.1f} m short.")
    elif len(rows) >= 2:
        a, b = rows[0], rows[1]
        gap = float(np.linalg.norm(np.array(a["xy"]) - np.array(b["xy"])))
        out["kind"] = "too_spread" if gap > 2 * R * 0.8 else "out_of_position"
        out["diagnosis"] = (f"The ball landed in the gap between {a['label']} and {b['label']}, who stood {gap:.1f} m apart. "
                            f"{a['label']} was {a['dist']:.1f} m from it and could cover {R:.1f} m in {scn.T:.2f} s: "
                            f"{a['shortfall']:.1f} m short.")
    else:
        out["kind"] = "out_of_position"
        out["diagnosis"] = f"{n0['label']} was {n0['dist']:.1f} m from the ball and {n0['shortfall']:.1f} m short."
    if scn.missing:
        out["diagnosis"] += f" ({scn.missing} defender(s) were not tracked and could change this picture.)"
    return out


# ---------------------------------------------------------------------------
# rendering
# ---------------------------------------------------------------------------
def _floor_ring(img, cam, centre, radius, col, thick=2):
    ang = np.linspace(0, 2 * np.pi, 73)
    pts = np.c_[centre[0] + radius * np.cos(ang), centre[1] + radius * np.sin(ang), np.zeros(73)]
    _polyline(img, cam, pts, col, thick)


def _polyline(img, cam, pts3, col, thick=2):
    c = cam.cam(pts3)
    for a, b in zip(c[:-1], c[1:]):
        if a[2] < 0.3 and b[2] < 0.3:
            continue
        if a[2] < 0.3:
            a = a + (0.3 - a[2]) / (b[2] - a[2]) * (b - a)
        if b[2] < 0.3:
            b = b + (0.3 - b[2]) / (a[2] - b[2]) * (a - b)
        p = (int(cam.f * a[0] / a[2] + cam.size[0] / 2), int(cam.f * a[1] / a[2] + cam.size[1] / 2))
        q = (int(cam.f * b[0] / b[2] + cam.size[0] / 2), int(cam.f * b[1] / b[2] + cam.size[1] / 2))
        if max(abs(p[0]), abs(p[1]), abs(q[0]), abs(q[1])) < 20000:
            cv2.line(img, p, q, col, thick, cv2.LINE_AA)


def _figure(img, cam, xy, col, label=None):
    uv, d = cam.project([(xy[0], xy[1], 0.0)])
    if d[0] < 0.5:
        return
    s = cam.f / d[0]
    u, v = uv[0]
    cv2.rectangle(img, (int(u - 0.22 * s), int(v - 1.5 * s)), (int(u + 0.22 * s), int(v)), col, -1)
    cv2.circle(img, (int(u), int(v - 1.68 * s)), max(2, int(0.12 * s)), (190, 200, 225), -1, cv2.LINE_AA)
    if label:
        cv2.putText(img, label, (int(u - 0.3 * s), int(v - 1.95 * s)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)


def _wrap(text, width):
    words, lines, cur = text.split(), [], ""
    for w in words:
        if len(cur) + len(w) + 1 > width:
            lines.append(cur)
            cur = w
        else:
            cur = (cur + " " + w).strip()
    return lines + ([cur] if cur else [])


def render_pov(scn: Scenario, index: int, size=(960, 540), analysis=None):
    """First-person view from defender `index` (in scn.defenders order) at the
    moment the ball was hit, looking at the hitter."""
    analysis = analysis or analyse(scn)
    me = scn.defenders[index]
    row = next(r for r in analysis["defenders"] if r["label"] == me["label"] and r["xy"] == me["xy"])
    eye = np.array([me["xy"][0], me["xy"][1], EYE])
    # eyes on the hitter, tilted down enough to keep the floor in view
    look = np.array([scn.origin[0], scn.origin[1], 0.3])
    if np.linalg.norm(look[:2] - eye[:2]) < 0.5:
        look = np.array([eye[0], NET_Y, 0.3])
    cam = Camera(eye, look, 0.5 * size[0], size)
    img = np.zeros((size[1], size[0], 3), np.uint8)
    img[:] = (120, 112, 108)
    _poly(img, cam, [(-12, -9, 0), (21, -9, 0), (21, 27, 0), (-12, 27, 0)], (110, 125, 90))
    _poly(img, cam, [(0, 0, 0), (9, 0, 0), (9, 18, 0), (0, 18, 0)], (70, 115, 170))
    for _, a, b in COURT_LINES:
        _polyline(img, cam, np.array([(a[0], a[1], 0), (b[0], b[1], 0)], float), (240, 240, 240), 2)
    # things on the far side of the net first, then the net, then my own side
    for o in scn.opponents:
        _figure(img, cam, o["xy"], (60, 60, 170))
    net = img.copy()
    _poly(net, cam, [(0, 9, NET_H - 1.0), (9, 9, NET_H - 1.0), (9, 9, NET_H), (0, 9, NET_H)], (40, 40, 40))
    cv2.addWeighted(net, 0.35, img, 0.65, 0, img)
    _polyline(img, cam, np.array([(0, 9, NET_H), (9, 9, NET_H)], float), (235, 235, 235), 2)
    for k, d in enumerate(scn.defenders):
        if k != index:
            _figure(img, cam, d["xy"], (170, 90, 40), d["label"])
    # ball path, landing spot, how far I could have got
    A, L = np.array(scn.origin, float), np.array([scn.landing[0], scn.landing[1], BALL_R])
    v0 = (L - A - 0.5 * GVEC * scn.T ** 2) / scn.T
    ts = np.linspace(0, scn.T, 30)
    _polyline(img, cam, A + v0 * ts[:, None] + 0.5 * GVEC * ts[:, None] ** 2, (60, 215, 240), 2)
    _floor_ring(img, cam, scn.landing, 0.35, (40, 40, 230), 3)
    _floor_ring(img, cam, me["xy"], analysis["cover_radius"], (90, 220, 90), 2)
    if row["shortfall"] > 0:
        _floor_ring(img, cam, row["needed_xy"], 0.3, (255, 255, 255), 2)
    fwd = (look - eye)[:2]
    fwd /= np.linalg.norm(fwd)
    to_l = np.array(scn.landing) - eye[:2]
    ang = np.degrees(np.arctan2(fwd[0] * to_l[1] - fwd[1] * to_l[0], fwd @ to_l))
    where = "ahead" if abs(ang) < 35 else ("to your left" if ang > 0 else "to your right") if abs(ang) < 125 else "behind you"
    head = f"View of {me['label']}  |  ball landed {row['dist']:.1f} m away, {where}  |  flight {scn.T:.2f} s"
    verdict = ("You could have reached it." if row["shortfall"] == 0 else
               f"You could cover {analysis['cover_radius']:.1f} m: {row['shortfall']:.1f} m short. White ring = where you needed to be.")
    diag = _wrap(analysis["diagnosis"], 105)
    top = 78 + 20 * len(diag)
    out = np.full((size[1] + top, size[0], 3), 25, np.uint8)
    out[top:] = img
    cv2.putText(out, head, (12, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(out, verdict, (12, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (140, 230, 140) if row["shortfall"] == 0 else (130, 170, 255), 1, cv2.LINE_AA)
    for k, line in enumerate(diag):
        cv2.putText(out, line, (12, 70 + 20 * k), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(out, "yellow = ball path   red ring = landing   green ring = your reach in that time",
                (12, 70 + 20 * len(diag)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (190, 190, 190), 1, cv2.LINE_AA)
    return out


def render_topdown(scn: Scenario, analysis=None, scale=36, placement=None):
    """Bird's-eye diagram of the same moment (or of a simulator placement)."""
    analysis = analysis or analyse(scn)
    m = 2.0
    W, H = int((9 + 2 * m) * scale), int((18 + 2 * m) * scale)
    img = np.full((H, W, 3), (110, 125, 90), np.uint8)

    def px(p):
        return int((p[0] + m) * scale), int(H - (p[1] + m) * scale)
    cv2.rectangle(img, px((0, 18)), px((9, 0)), (70, 115, 170), -1)
    for _, a, b in COURT_LINES:
        cv2.line(img, px(a), px(b), (240, 240, 240), 2, cv2.LINE_AA)
    cv2.line(img, px((-0.5, 9)), px((9.5, 9)), (30, 30, 30), 3, cv2.LINE_AA)
    for o in scn.opponents:
        cv2.circle(img, px(o["xy"]), 7, (60, 60, 170), -1, cv2.LINE_AA)
    R = analysis["cover_radius"]
    people = placement if placement is not None else [d["xy"] for d in scn.defenders]
    labels = [f"P{k + 1}" for k in range(len(people))] if placement is not None else [d["label"] for d in scn.defenders]
    for xy, lab in zip(people, labels):
        cv2.circle(img, px(xy), int(R * scale), (90, 220, 90), 1, cv2.LINE_AA)
        cv2.circle(img, px(xy), 8, (170, 90, 40), -1, cv2.LINE_AA)
        cv2.putText(img, lab, (px(xy)[0] + 10, px(xy)[1] - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.arrowedLine(img, px(scn.origin[:2]), px(scn.landing), (60, 215, 240), 2, cv2.LINE_AA, tipLength=0.06)
    cv2.circle(img, px(scn.landing), 9, (40, 40, 230), 2, cv2.LINE_AA)
    return img


# ---------------------------------------------------------------------------
# placement simulator
# ---------------------------------------------------------------------------
def shots_from_scenario(scn: Scenario):
    return [{"landing": list(scn.landing), "T": scn.T, "weight": 1.0, "shot": scn.shot}]


def shots_from_opponent(sets, opponent, side_of_defence):
    """Every attack `opponent` hit over the net in the analysed footage, as the
    set of balls a defence has to be ready for."""
    shots = []
    for ms in sets:
        for r in ms.rallies:
            seq = r.events
            for k, e in enumerate(seq):
                if e.type != "attack" or e.team is None or ms.name(e.team) != opponent or "target" not in e.attrs:
                    continue
                if e.outcome not in ("kill", "dug", "in_play"):
                    continue
                nxt = seq[k + 1] if k + 1 < len(seq) else None
                T = (nxt.t if nxt is not None else r.t_end) - e.t
                tgt = e.attrs["target"]
                if e.team == side_of_defence:                       # sets played from the other end: mirror
                    tgt = [9 - tgt[0], 18 - tgt[1]]
                shots.append({"landing": [float(tgt[0]), float(tgt[1])], "T": float(max(0.15, T)), "weight": 1.0,
                              "shot": e.attrs.get("shot")})
    return shots


def validate_placement(positions, side):
    problems = []
    if not 1 <= len(positions) <= 6:
        problems.append("place between 1 and 6 players")
    for k, p in enumerate(positions):
        x, d = local(side, p)
        if not (-0.5 <= p[0] <= 9.5):
            problems.append(f"player {k + 1} is outside the sidelines")
        if d < 0:
            problems.append(f"player {k + 1} is on the wrong side of the net")
        elif d > 10.5:
            problems.append(f"player {k + 1} is more than 1.5 m behind the end line")
    return problems


def score_placement(positions, shots, side):
    """Score 0-100: the weighted share of the given balls this placement gets to.

    A ball inside someone's reach scores 1; one just out of reach gets partial
    credit falling to 0 at 2 m short."""
    problems = validate_placement(positions, side)
    if problems:
        return {"score": None, "problems": problems, "per_shot": []}
    if not shots:
        return {"score": None, "problems": ["no balls to defend in this scenario"], "per_shot": []}
    P = np.array(positions, float)
    tot = wsum = 0.0
    per = []
    for s in shots:
        d = np.linalg.norm(P - np.array(s["landing"]), axis=1)
        k = int(np.argmin(d))
        short = max(0.0, float(d[k]) - cover_radius(s["T"]))
        credit = max(0.0, 1.0 - short / 2.0)
        per.append({"landing": s["landing"], "shot": s.get("shot"), "nearest_player": k + 1, "dist": float(d[k]),
                    "shortfall": short, "credit": credit})
        tot += credit * s.get("weight", 1.0)
        wsum += s.get("weight", 1.0)
    return {"score": round(100 * tot / wsum, 1), "problems": [], "per_shot": per,
            "reached": sum(p["shortfall"] == 0 for p in per), "n_shots": len(per)}


def suggest_placement(shots, side, n_players=6, seed=0, iters=400):
    """A good (not provably best) placement found by random-restart hill climbing,
    to compare a user's placement against."""
    n_players = int(min(6, max(1, n_players)))
    rng = np.random.default_rng(seed)

    def rand():
        pts = np.c_[rng.uniform(0.5, 8.5, n_players), rng.uniform(0.5, 8.5, n_players)]
        return [[p[0], NET_Y - p[1]] if side == "near" else [p[0], NET_Y + p[1]] for p in pts]

    best, best_s = None, -1.0
    for _ in range(6):
        cur = rand()
        cur_s = score_placement(cur, shots, side)["score"] or 0.0
        for _ in range(iters):
            cand = [list(p) for p in cur]
            k = rng.integers(n_players)
            cand[k][0] = float(np.clip(cand[k][0] + rng.normal(0, 0.7), 0.3, 8.7))
            lo, hi = (0.3, 8.7) if side == "near" else (9.3, 17.7)
            cand[k][1] = float(np.clip(cand[k][1] + rng.normal(0, 0.7), lo, hi))
            sc = score_placement(cand, shots, side)["score"]
            if sc is not None and sc >= cur_s:
                cur, cur_s = cand, sc
        if cur_s > best_s:
            best, best_s = cur, cur_s
    return best, best_s
