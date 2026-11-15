"""From ball flights to rallies, contacts, contact types and outcomes.

Everything here is rule based and geometric. The rules are written down in
README ("How contacts are classified") because they are approximations of
what a human scorer does, and every event carries a confidence and flags.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field

import numpy as np

from ..geometry import NET_Y, other, side_of
from .track import Flight, junction


@dataclass
class EventConfig:
    serve_zone: float = 1.2        # a rally starts with a contact within this distance of / behind an end line
    attack_min_z: float = 2.0      # ball height that separates an attack from a free ball
    block_window: float = 0.6      # s after an attack in which a net contact is a block
    max_player_dist: float = 2.5   # m, nearest player further than this -> contact not attributed
    grade3: float = 1.6            # set taken within this distance of the net -> pass grade 3
    grade2: float = 3.7
    slow_shot: float = 7.7         # m/s horizontal: below this an attack is a tip or roll shot
    tip_depth: float = 3.5         # m past the net: slow shots landing shorter are tips
    line_ratio: float = 0.17       # |dx|/|dy| below this is a line shot
    close_call: float = 0.35       # m from a boundary line: in/out is marked uncertain
    min_touch_dv: float = 4.0      # m/s change of ball velocity below which a break in the track is not a touch


@dataclass
class Event:
    rally: int
    idx: int
    t: float
    frame: int
    pos: list
    team: str | None
    track: int | None = None
    number: int | None = None
    type: str = "unknown"
    outcome: str | None = None
    attrs: dict = field(default_factory=dict)
    conf: float = 1.0
    flags: list = field(default_factory=list)


@dataclass
class Rally:
    id: int
    serve_team: str | None
    events: list
    t_start: float
    t_end: float
    landing: list | None = None
    winner: str | None = None
    reason: str | None = None
    conf: float = 1.0
    flags: list = field(default_factory=list)

    def to_json(self):
        d = asdict(self)
        return d


def in_court(x, y, margin=0.1):
    return -margin <= x <= 9 + margin and -margin <= y <= 18 + margin


def _near_line(x, y, tol):
    return min(abs(x), abs(x - 9), abs(y), abs(y - 18)) < tol and in_court(x, y, tol)


def _players_at(tracks, frame, team=None):
    out = []
    if tracks is None:
        return out
    for ident, xy in tracks.at(frame):
        if team is None or ident.team == team:
            out.append((ident, xy))
    return out


def _attribute(ev: Event, tracks, cfg, team=None):
    cands = _players_at(tracks, ev.frame, team or ev.team)
    if not cands:
        ev.flags.append("no_player_tracked")
        ev.conf *= 0.5
        return
    d = sorted((float(np.linalg.norm(xy - np.array(ev.pos[:2]))), ident) for ident, xy in cands)
    if d[0][0] > cfg.max_player_dist:
        ev.flags.append("no_player_near_contact")
        ev.conf *= 0.5
        return
    ev.track, ev.number = d[0][1].id, d[0][1].number
    ev.attrs["player_dist"] = round(d[0][0], 2)
    if len(d) > 1 and d[1][0] - d[0][0] < 0.5:
        ev.flags.append("two_players_equally_close")
        ev.conf *= 0.6
    if d[0][1].number is None:
        ev.flags.append("player_number_unknown")


def _serve_start(fl: Flight, tracks, fps, cfg):
    """The first flight of a rally is usually picked up a few frames after the
    serve. Follow the parabola back to where it passes closest to a player."""
    best = (1e9, fl.t0)
    for t in np.arange(fl.t0, fl.t0 - 0.6, -1.0 / fps):
        p = fl.pos(t)
        if not 0.3 < p[2] < 3.6:
            continue
        for _, xy in _players_at(tracks, int(round(t * fps))):
            d = float(np.linalg.norm(xy - p[:2]))
            if d < best[0]:
                best = (d, t)
    if best[0] < 1.5:
        return best[1], fl.pos(best[1]), True
    return fl.t0, fl.pos(fl.t0), False


def build_rallies(flights: list[Flight], tracks, fps: float, cfg: EventConfig | None = None) -> list[Rally]:
    cfg = cfg or EventConfig()
    rallies: list[Rally] = []
    i, n = 0, len(flights)
    last_end = -1e9
    while i < n:
        fl = flights[i]
        if fl.t0 - last_end < 0.8:              # still the dead ball bouncing away
            last_end = fl.t1
            i += 1
            continue
        chain, juncs, bounce = [fl], [], None
        while i + 1 < n and flights[i + 1].t0 - chain[-1].t1 < 1.2:
            j = junction(chain[-1], flights[i + 1])
            if j.kind == "contact" and j.floor is not None and j.mismatch > 0.8:
                # the two paths don't meet cleanly and the first one reaches the
                # floor: a touch needs a player there, otherwise the ball landed
                near = [np.linalg.norm(xy - j.floor[1][:2]) for _, xy in _players_at(tracks, int(round(j.floor[0] * fps)))]
                if not near or min(near) > 1.2:
                    j.kind, j.t, j.pos = "bounce", j.floor[0], j.floor[1]
            if j.kind == "bounce":
                bounce = j
                break
            dv = float(np.linalg.norm(flights[i + 1].vel(j.t) - chain[-1].vel(j.t)))
            if dv >= cfg.min_touch_dv or j.pos[2] < 0.5:
                juncs.append(j)                  # a touch changes the ball's velocity
            # else: one flight that was merely tracked in two pieces
            chain.append(flights[i + 1])
            i += 1
        i += 1
        last_end = chain[-1].t1
        r = _make_rally(len(rallies), chain, juncs, bounce, tracks, fps, cfg)
        if r is not None:
            rallies.append(r)
    _cross_check_winners(rallies)
    return rallies


def _make_rally(rid, chain, juncs, bounce, tracks, fps, cfg) -> Rally | None:
    flags = []
    t0, p0, found = _serve_start(chain[0], tracks, fps, cfg)
    contacts = [(t0, np.asarray(p0, float))] + [(j.t, j.pos) for j in juncs]
    # terminal
    landing, t_end = None, chain[-1].t1
    if bounce is not None:
        landing, t_end = bounce.pos, bounce.t
    else:
        land = chain[-1].landing()
        if land is not None and -0.1 <= land[0] - chain[-1].t1 < 0.7:
            landing, t_end = land[1], land[0]
            if land[0] - chain[-1].t1 > 0.2:
                flags.append("landing_extrapolated")
        else:
            flags.append("ball_lost_before_landing")
    serve_like = p0[1] < cfg.serve_zone or p0[1] > 18 - cfg.serve_zone
    if not serve_like:
        flags.append("serve_not_seen")
        if len(contacts) < 2 and landing is None:
            return None                              # an isolated arc with nothing around it
    events = []
    for k, (t, p) in enumerate(contacts):
        team = side_of(p[1])
        events.append(Event(rid, k, float(t), int(round(t * fps)), [round(float(v), 2) for v in p], team))
        if k and juncs[k - 1].gap > 0.35:
            events[-1].flags.append("contact_not_seen_directly")
            events[-1].conf *= 0.8
        if k and juncs[k - 1].mismatch > 1.0:
            events[-1].flags.append("contact_position_uncertain")
            events[-1].conf *= 0.7

    def end_of(k):
        """Where the flight after contact k ends (next contact or the floor)."""
        if k + 1 < len(contacts):
            return contacts[k + 1][1]
        return landing

    def crosses(k):
        e = end_of(k)
        return e is not None and side_of(e[1]) != events[k].team

    # contacts right at the net: the side is decided by where the ball came from / goes
    for k, ev in enumerate(events):
        if abs(ev.pos[1] - NET_Y) < 0.5 and k > 0:
            ev.attrs["at_net"] = True

    # ---- types ------------------------------------------------------------
    touch, poss = 0, None
    extra = []
    for k, ev in enumerate(events):
        z = ev.pos[2]
        prev = events[k - 1] if k else None
        if k == 0:
            ev.type = "serve" if serve_like else "unknown"
            touch, poss = 1, ev.team
            if not serve_like:
                ev.conf *= 0.5
        else:
            new_side = ev.team != poss
            if new_side:
                is_block = (abs(ev.pos[1] - NET_Y) < 1.0 and z > cfg.attack_min_z
                            and ev.t - prev.t < cfg.block_window and prev.type in ("attack", "free_ball"))
                if is_block:
                    ev.type, touch = "block", 0
                elif prev.type == "serve":
                    ev.type, touch = "reception", 1
                else:
                    ev.type, touch = "dig", 1
                    ev.attrs["versus"] = prev.type
            else:
                touch += 1
                if prev.type == "block":
                    ev.type, touch = "dig", 1
                    ev.attrs["versus"] = "block"
                elif touch == 2:
                    ev.type = "set"
                else:
                    ev.type = "attack" if z >= cfg.attack_min_z else "free_ball"
            poss = ev.team
            if crosses(k) and ev.type != "block":
                if ev.type in ("reception", "dig"):
                    if z >= cfg.attack_min_z + 0.3:
                        ev.type = "attack"
                        ev.flags.append("attack_on_first_touch")
                    else:
                        ev.attrs["overpass"] = True
                else:
                    if ev.type == "set":
                        ev.flags.append("attack_on_second_touch")
                    ev.type = "attack" if z >= cfg.attack_min_z else "free_ball"
            elif ev.type == "set" and k == len(events) - 1 and z >= cfg.attack_min_z and abs(ev.pos[1] - NET_Y) < 1.6 \
                    and touch >= 2 and landing is not None:
                pass
        _attribute(ev, tracks, cfg)

    # ---- an attack that comes straight back: blocked (or hit into the net) ----
    last = events[-1]
    if last.type in ("attack", "set", "free_ball") and landing is not None and side_of(landing[1]) == last.team \
            and abs(last.pos[1] - NET_Y) < 1.8 and last.pos[2] >= cfg.attack_min_z - 0.2 and len(events) >= 2:
        opp = other(last.team)
        blockers = [(abs(xy[0] - last.pos[0]), ident) for ident, xy in _players_at(tracks, last.frame, opp)
                    if abs(xy[1] - NET_Y) < 1.6 and abs(xy[0] - last.pos[0]) < 1.6]
        last.type = "attack"
        last.flags.append("ball_came_back_from_net")
        if blockers:
            b = min(blockers, key=lambda x: x[0])[1]
            last.outcome = "blocked"
            blk = Event(rid, len(events), last.t + 0.05, last.frame + 1,
                        [last.pos[0], NET_Y + (0.25 if opp == "far" else -0.25), last.pos[2]], opp,
                        b.id, b.number, "block", "stuff", {"inferred": True}, 0.6,
                        ["block_inferred_not_seen"] + ([] if b.number is not None else ["player_number_unknown"]))
            extra.append(blk)
        else:
            last.outcome = "error"
            last.attrs["error_kind"] = "net"
    events += extra

    rally = Rally(rid, events[0].team if events[0].type == "serve" else None, events, float(t0), float(t_end),
                  None if landing is None else [round(float(v), 2) for v in landing], flags=flags)
    _outcomes(rally, cfg)
    return rally


def _shot(ev: Event, target, dt, cfg):
    """Describe an attack by where and how fast it went."""
    d = np.array(target[:2]) - np.array(ev.pos[:2])
    dist = float(np.linalg.norm(d))
    speed = dist / max(dt, 1e-3)
    depth = abs(target[1] - NET_Y)
    ev.attrs.update(target=[round(float(target[0]), 2), round(float(target[1]), 2)], speed=round(speed, 1))
    if speed < cfg.slow_shot:
        ev.attrs["shot"] = "tip" if depth < cfg.tip_depth else "roll"
    else:
        ev.attrs["shot"] = "line" if abs(d[0]) < cfg.line_ratio * max(abs(d[1]), 1e-6) else "cross"


def _outcomes(r: Rally, cfg: EventConfig):
    ev = r.events
    seq = [e for e in ev if not e.attrs.get("inferred")]
    land = r.landing
    n = len(seq)

    def nxt(k):
        return seq[k + 1] if k + 1 < n else None

    # first-touch quality from where the second touch is played
    for k, e in enumerate(seq):
        if e.type in ("reception", "dig"):
            f = nxt(k)
            if e.attrs.get("overpass"):
                e.attrs["grade"] = 1
            elif f is not None and f.team == e.team:
                d = abs(f.pos[1] - NET_Y)
                e.attrs["grade"] = 3 if d <= cfg.grade3 else 2 if d <= cfg.grade2 else 1
            else:
                e.attrs["grade"] = 0
            e.outcome = "error" if e.attrs["grade"] == 0 else "ok"
    # in / out of system follows the pass
    for k, e in enumerate(seq):
        if e.type in ("set", "attack", "free_ball"):
            first = next((p for p in reversed(seq[:k]) if p.type in ("reception", "dig") and p.team == e.team), None)
            stop = next((p for p in reversed(seq[:k]) if p.team != e.team), None)
            if first is not None and (stop is None or stop.idx < first.idx) and "grade" in first.attrs:
                e.attrs["in_system"] = first.attrs["grade"] >= 2
    # attacks, serves, free balls: what happened next
    for k, e in enumerate(seq):
        if e.type not in ("attack", "serve", "free_ball") or e.outcome is not None:
            continue
        f = nxt(k)
        target = f.pos if f is not None else land
        if target is not None and e.type == "attack":
            _shot(e, target, (f.t if f is not None else r.t_end) - e.t, cfg)
        if f is None:
            if land is None:
                e.outcome = "unknown"
                e.conf *= 0.5
                e.flags.append("result_not_seen")
            else:
                inb = in_court(land[0], land[1])
                if _near_line(land[0], land[1], cfg.close_call):
                    e.flags.append("close_line_call")
                    e.conf *= 0.6
                if side_of(land[1]) == e.team:
                    e.outcome = "error"
                    e.attrs["error_kind"] = "net_or_short"
                elif inb:
                    e.outcome = "ace" if e.type == "serve" else "kill"
                else:
                    e.outcome = "error"
                    e.attrs["error_kind"] = "out"
        elif f.type == "block":
            g = nxt(k + 1)
            if g is None and land is not None and side_of(land[1]) == e.team:
                e.outcome, f.outcome = "blocked", "stuff"
            elif g is None and land is not None:
                e.outcome, f.outcome = ("kill" if in_court(land[0], land[1]) else "error"), "touch"
                if e.outcome == "error":
                    e.flags.append("out_after_block_touch_scored_as_error")
            else:
                e.outcome, f.outcome = "in_play", "touch"
        else:
            g = nxt(k + 1)
            if f.team != e.team and (g is None or g.team == e.team and False):
                # touched once by the other side, then dead
                if f.attrs.get("grade") == 0:
                    e.outcome = "ace" if e.type == "serve" else "kill"
                else:
                    e.outcome = "in_play"
            else:
                e.outcome = "in_play" if e.type != "attack" else "dug"
    for k, e in enumerate(seq):
        if e.type == "set":
            f = nxt(k)
            e.outcome = "assist" if (f is not None and f.type == "attack" and f.team == e.team and f.outcome == "kill") else "set"
            if f is not None and f.team == e.team:
                e.attrs["target_track"], e.attrs["target_number"] = f.track, f.number
                e.attrs["target_pos"] = f.pos[:2]
    # ---- who won the rally ---------------------------------------------------
    last = seq[-1]
    blocked = next((e for e in ev if e.outcome == "blocked"), None)
    if blocked is not None:
        r.winner, r.reason = other(blocked.team), "blocked"
    elif land is None:
        r.winner, r.reason, r.conf = None, "unknown", 0.3
    else:
        if _near_line(land[0], land[1], cfg.close_call):
            r.flags.append("close_line_call")
            r.conf *= 0.6
        if in_court(land[0], land[1]):
            loser = side_of(land[1])
            r.winner = other(loser)
            r.reason = {"serve": "ace" if last.team == r.winner else "serve_error",
                        "attack": "kill" if last.team == r.winner else "attack_error",
                        "free_ball": "free_ball_drop" if last.team == r.winner else "free_ball_error",
                        "reception": "reception_error", "dig": "dig_error", "set": "set_error",
                        "block": "block"}.get(last.type, "unknown")
        else:
            r.winner = other(last.team)
            r.reason = {"serve": "serve_error", "attack": "attack_error", "free_ball": "free_ball_error",
                        "reception": "reception_error", "dig": "dig_error", "set": "set_error",
                        "block": "block_out"}.get(last.type, "unknown")
        if last.type in ("reception", "dig") and last.attrs.get("grade") == 0:
            # the point belongs to whoever hit the ball they could not handle
            src = next((e for e in reversed(seq[:-1]) if e.team != last.team), None)
            if src is not None and src.type in ("attack", "serve"):
                src.outcome = "ace" if src.type == "serve" else "kill"
                r.reason = src.outcome
    if "ball_lost_before_landing" in r.flags or "landing_extrapolated" in r.flags:
        r.conf *= 0.6
    if "serve_not_seen" in r.flags:
        r.conf *= 0.7


def _cross_check_winners(rallies):
    """Whoever serves next won the previous rally. Use it to fill unknown
    winners and to flag disagreements."""
    for a, b in zip(rallies, rallies[1:]):
        if b.serve_team is None:
            continue
        if a.winner is None:
            a.winner, a.reason = b.serve_team, "inferred_from_next_serve"
            a.flags.append("winner_inferred_from_next_serve")
            a.conf = max(a.conf, 0.6)
        elif a.winner != b.serve_team:
            a.flags.append("winner_disagrees_with_next_server")
            a.conf *= 0.4


def rallies_to_json(rallies):
    return [r.to_json() for r in rallies]


def rallies_from_json(data):
    out = []
    for d in data:
        evs = [Event(**e) for e in d["events"]]
        out.append(Rally(**{**d, "events": evs}))
    return out
