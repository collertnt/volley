"""Synthetic volleyball match with exact ground truth.

A scripted rally generator: every ball flight is a drag-free parabola between
two contacts, every contact has a known player, type and outcome. The pipeline
is tested against this truth. This is deliberately *easier* than real footage
(see README): it checks the logic, not real-world accuracy.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..geometry import BALL_R, GVEC, NET_H, NET_Y, other

ZONES = {1: (7.5, 2.5), 6: (4.5, 2.0), 5: (1.5, 2.5), 4: (1.5, 6.8), 3: (4.5, 7.3), 2: (7.5, 6.8)}
ZONE_ORDER = [1, 6, 5, 4, 3, 2]
FRONT, BACK = (2, 3, 4), (1, 6, 5)


def L(team: str, p):
    """Team-local coordinates (own side is y<9) -> world."""
    p = np.asarray(p, float)
    if team == "near":
        return p.copy()
    q = p.copy()
    q[0], q[1] = 9.0 - p[0], 18.0 - p[1]
    return q


@dataclass
class Flight:
    t0: float
    t1: float
    p0: np.ndarray
    v0: np.ndarray

    def pos(self, t):
        dt = t - self.t0
        return self.p0 + self.v0 * dt + 0.5 * GVEC * dt * dt


def make_flight(t0, A, B, T):
    A, B = np.asarray(A, float), np.asarray(B, float)
    return Flight(t0, t0 + T, A, (B - A - 0.5 * GVEC * T * T) / T)


def fly(t0, A, B, T, check_net=True):
    """Flight from A to B, lengthened until it clears the net."""
    for _ in range(40):
        fl = make_flight(t0, A, B, T)
        if not check_net or (A[1] - NET_Y) * (B[1] - NET_Y) >= 0:
            return fl
        tc = (NET_Y - A[1]) / fl.v0[1]
        if fl.pos(t0 + tc)[2] >= NET_H + 0.25:
            return fl
        T += 0.05
    return fl


@dataclass
class PlayerSpec:
    pid: int
    team: str
    number: int
    height: float
    skin: tuple
    hair: tuple
    socks: tuple


@dataclass
class Params:
    serve_err: float = 0.08
    ace: float = 0.06
    grade_p: tuple = (0.5, 0.3, 0.2)            # P(grade 3, 2, 1)
    shot_p: tuple = (0.3, 0.4, 0.15, 0.15)      # line, cross, tip, roll
    outcome_p: tuple = (0.42, 0.33, 0.10, 0.15) # kill, dug, error, blocked
    freeball_when_oos: float = 0.5
    max_exchanges: int = 3
    dead_time: float = 2.5


class World:
    def __init__(self, seed=0, n_rallies=4, params: Params | None = None, numbers=None):
        self.rng = np.random.default_rng(seed)
        self.p = params or Params()
        rng = self.rng
        nums = numbers or {"near": [4, 7, 9, 11, 15, 23], "far": [2, 5, 8, 12, 17, 31]}
        self.players: list[PlayerSpec] = []
        for ti, team in enumerate(("near", "far")):
            for i in range(6):
                skin = tuple(int(c) for c in rng.integers((90, 130, 180), (150, 190, 235)))
                hair = tuple(int(c) for c in rng.integers(10, 140, 3))
                socks = tuple(int(c) for c in rng.integers(40, 255, 3))
                self.players.append(PlayerSpec(ti * 6 + i, team, nums[team][i],
                                               float(rng.uniform(1.75, 1.98)), skin, hair, socks))
        self.rot = {"near": 0, "far": 0}
        self.flights: list[Flight] = []
        self.rallies: list[dict] = []
        self.keys: dict[int, list] = {p.pid: [] for p in self.players}   # (t, x, y)
        self.jumps: dict[int, list] = {p.pid: [] for p in self.players}  # (t, height)
        self.phase = {p.pid: rng.uniform(0, 6.28, 2) for p in self.players}
        self.wall_rects = [(float(x), float(x + rng.uniform(1, 4)), float(z), float(z + rng.uniform(0.5, 2)),
                            tuple(int(c) for c in rng.integers(40, 210, 3)))
                           for x, z in zip(rng.uniform(-9, 16, 14), rng.uniform(0.3, 6, 14))]
        t, serve_team = 1.0, "near"
        for rid in range(n_rallies):
            end, winner = self._rally(rid, t, serve_team)
            if winner != serve_team:
                self.rot[winner] = (self.rot[winner] + 1) % 6
                serve_team = winner
            t = end + self.p.dead_time
        self.duration = t - self.p.dead_time + 1.0
        self._finish_keys()

    # ----- roster helpers -------------------------------------------------
    def team_players(self, team):
        return [p for p in self.players if p.team == team]

    def zone_of(self, pid):
        pl = self.players[pid]
        return ZONE_ORDER[(pid % 6 + self.rot[pl.team]) % 6]

    def in_zones(self, team, zones):
        return [p.pid for p in self.team_players(team) if self.zone_of(p.pid) in zones]

    def base(self, pid):
        return L(self.players[pid].team, np.array(ZONES[self.zone_of(pid)]))

    def _nearest(self, pids, world_xy, exclude=()):
        pids = [q for q in pids if q not in exclude]
        return min(pids, key=lambda q: np.linalg.norm(self.base(q) - world_xy[:2]))

    def _key(self, pid, t, xy, jump=0.0):
        team = self.players[pid].team
        loc = L(team, np.array([xy[0], xy[1]]))
        loc[1] = min(loc[1], 8.65)                  # feet stay on own side
        w = L(team, loc)
        self.keys[pid].append((t, float(w[0]), float(w[1])))
        if jump:
            self.jumps[pid].append((t, jump))

    # ----- rally script ---------------------------------------------------
    def _rally(self, rid, t_start, S):
        rng, p = self.rng, self.p
        R = other(S)
        rally = {"id": rid, "serve_team": S, "rotation": dict(self.rot), "contacts": [],
                 "t_start": t_start}
        self.rallies.append(rally)
        for pl in self.players:                      # everyone at base when the rally starts
            self._key(pl.pid, t_start, self.base(pl.pid))
        server = self.in_zones(S, (1,))[0]
        t = t_start + 1.2
        xs = rng.uniform(2, 7)
        A = L(S, (xs, -0.9, 2.5))
        self._key(server, t - 0.6, A)
        self._key(server, t, A)
        self._key(server, t + 1.0, self.base(server))
        u = rng.random()
        serve = self._contact(rally, t, A, server, "serve")
        if u < p.serve_err:
            B = L(S, (rng.uniform(1, 8), 18 + rng.uniform(0.8, 1.8), BALL_R))
            fl = fly(t, A, B, 1.3)
            serve["outcome"] = "error"
            return self._land(rally, fl, R, "serve_error"), R
        if u < p.serve_err + p.ace:
            B = L(S, (rng.uniform(1, 8), rng.uniform(11.5, 16.5), BALL_R))
            fl = fly(t, A, B, 1.25)
            serve["outcome"] = "ace"
            return self._land(rally, fl, S, "ace"), S
        serve["outcome"] = "in_play"
        recv = int(rng.choice(self.in_zones(R, BACK)))
        loc = np.array(ZONES[self.zone_of(recv)]) + rng.uniform(-0.8, 0.8, 2)
        loc = np.clip(loc, (1, 1), (8, 7.5))
        B = L(R, (loc[0], loc[1], 0.7))
        fl = fly(t, A, B, 1.25)
        self.flights.append(fl)
        return self._possession(rally, R, "reception", recv, B, fl.t1, 0)

    def _contact(self, rally, t, pos, pid, typ, **kw):
        c = {"t": float(t), "pos": [float(v) for v in pos], "pid": int(pid),
             "number": self.players[pid].number, "team": self.players[pid].team,
             "type": typ, "rally": rally["id"], **kw}
        rally["contacts"].append(c)
        return c

    def _land(self, rally, fl, winner, reason):
        self.flights.append(fl)
        end = fl.pos(fl.t1)
        v = fl.v0 + GVEC * (fl.t1 - fl.t0)
        vb = np.array([v[0] * 0.55, v[1] * 0.55, -v[2] * 0.55])
        bounce = Flight(fl.t1, fl.t1 + 0.45, end, vb)
        self.flights.append(bounce)
        rally.update(landing=[float(x) for x in end], t_end=float(fl.t1), winner=winner, reason=reason)
        return fl.t1

    def _possession(self, rally, T, first_type, p1, c1, t1, depth):
        rng, p = self.rng, self.p
        O = other(T)
        g = int(rng.choice([3, 2, 1], p=p.grade_p))
        dn = {3: rng.uniform(0.7, 1.3), 2: rng.uniform(2.3, 3.2), 1: rng.uniform(4.6, 6.2)}[g]
        sx = rng.uniform(4.5, 6.5) if g > 1 else rng.uniform(2.5, 6.5)
        Bs = L(T, (sx, 9 - dn, 2.1 if g > 1 else 1.1))
        pool = self.in_zones(T, FRONT) if g > 1 else [q.pid for q in self.team_players(T)]
        setter = self._nearest(pool, Bs, exclude=(p1,))
        self._key(p1, t1, c1)
        first = self._contact(rally, t1, c1, p1, first_type, grade=g)
        fl = fly(t1, c1, Bs, rng.uniform(1.3, 1.5))
        self.flights.append(fl)
        t2 = fl.t1
        self._key(setter, t2, Bs)
        in_sys = g >= 2
        hitter = int(rng.choice([q for q in self.in_zones(T, FRONT) if q != setter]))
        hx = float(np.clip(ZONES[self.zone_of(hitter)][0] + rng.uniform(-0.4, 0.4), 0.8, 8.2))
        setc = self._contact(rally, t2, Bs, setter, "set", in_system=in_sys, outcome="set")

        if not in_sys and depth < p.max_exchanges and rng.random() < p.freeball_when_oos:
            Af = L(T, (hx, 9 - rng.uniform(4.0, 5.5), 1.0))
            fl = fly(t2, Bs, Af, 1.2)
            self.flights.append(fl)
            t3 = fl.t1
            self._key(hitter, t3, Af)
            fb = self._contact(rally, t3, Af, hitter, "free_ball", in_system=False, outcome="in_play")
            dig = int(rng.choice(self.in_zones(O, BACK)))
            loc = np.clip(np.array(ZONES[self.zone_of(dig)]) + rng.uniform(-0.5, 0.5, 2), (1, 1), (8, 7.5))
            B = L(O, (loc[0], loc[1], 0.8))
            fl = fly(t3, Af, B, 1.7)
            self.flights.append(fl)
            first["next"] = "free_ball"
            return self._possession(rally, O, "dig", dig, B, fl.t1, depth + 1)

        off = rng.uniform(0.5, 0.9) if in_sys else rng.uniform(2.6, 3.4)
        Ah = L(T, (hx, 9 - off, 2.9 if in_sys else 2.75))
        fl = fly(t2, Bs, Ah, rng.uniform(1.0, 1.3))
        self.flights.append(fl)
        t3 = fl.t1
        self._key(hitter, t3, Ah, jump=0.55)
        shot = str(rng.choice(["line", "cross", "tip", "roll"], p=p.shot_p))
        outcome = str(rng.choice(["kill", "dug", "error", "blocked"], p=p.outcome_p))
        if depth >= p.max_exchanges and outcome == "dug":
            outcome = "kill"
        if shot in ("tip", "roll") and outcome in ("error", "blocked"):
            outcome = "kill"
        if not in_sys and outcome == "blocked":
            outcome = "dug" if depth < p.max_exchanges else "kill"
        if shot == "line":
            tx, ty, T_f = np.clip(hx + rng.uniform(-0.5, 0.5), 0.6, 8.4), rng.uniform(13.5, 17.2), rng.uniform(0.5, 0.65)
        elif shot == "cross":
            tx = rng.uniform(6.2, 8.4) if hx < 4.5 else rng.uniform(0.6, 2.8)
            ty, T_f = rng.uniform(12.5, 16.8), rng.uniform(0.55, 0.7)
        elif shot == "tip":
            tx, ty, T_f = np.clip(hx + rng.uniform(-1.5, 1.5), 0.8, 8.2), rng.uniform(10.0, 11.6), rng.uniform(0.8, 1.0)
        else:
            tx, ty, T_f = rng.uniform(1.5, 7.5), rng.uniform(12.5, 15.5), rng.uniform(1.0, 1.2)
        att = self._contact(rally, t3, Ah, hitter, "attack", shot=shot, in_system=in_sys, outcome=outcome)
        setc["target_pid"] = hitter

        if outcome == "blocked":
            blockers = self.in_zones(O, FRONT)
            bp_local = np.array([hx + rng.uniform(-0.25, 0.25), 9.25, 2.75])
            Bp = L(T, bp_local)
            blocker = min(blockers, key=lambda q: abs(self.base(q)[0] - Bp[0]))
            fl = make_flight(t3, Ah, Bp, 0.09 if in_sys else 0.2)
            self.flights.append(fl)
            tb = fl.t1
            self._key(blocker, tb, L(T, (bp_local[0], 9.45)), jump=0.45)
            self._contact(rally, tb, Bp, blocker, "block", outcome="stuff")
            land = L(T, (np.clip(hx + rng.uniform(-1, 1), 0.6, 8.4), 9 - rng.uniform(1.0, 2.5), BALL_R))
            fl = make_flight(tb, Bp, land, rng.uniform(0.45, 0.6))
            return self._land(rally, fl, O, "blocked"), O
        if outcome == "error":
            if shot == "line":
                ty = 18 + rng.uniform(0.7, 1.5)
            else:
                tx = -rng.uniform(0.7, 1.5) if tx < 4.5 else 9 + rng.uniform(0.7, 1.5)
            fl = fly(t3, Ah, L(T, (tx, ty, BALL_R)), T_f)
            return self._land(rally, fl, O, "attack_error"), O
        if outcome == "kill":
            fl = fly(t3, Ah, L(T, (tx, ty, BALL_R)), T_f)
            setc["outcome"] = "assist"
            return self._land(rally, fl, T, "kill"), T
        # dug
        B = L(T, (tx, ty, 0.6))
        pool = self.in_zones(O, BACK) if shot not in ("tip",) else [q.pid for q in self.team_players(O)]
        digger = self._nearest(pool, B)
        fl = fly(t3, Ah, B, T_f)
        self.flights.append(fl)
        return self._possession(rally, O, "dig", digger, B, fl.t1, depth + 1)

    # ----- motion ---------------------------------------------------------
    def _finish_keys(self):
        for pid, ks in self.keys.items():
            ks.sort()
            out = []
            for i, k in enumerate(ks):
                out.append(k)
                nxt = ks[i + 1][0] if i + 1 < len(ks) else 1e9
                if nxt - k[0] > 2.2:               # go back to base, then wait there
                    b = self._base_at(pid, k[0])
                    out.append((k[0] + 1.0, b[0], b[1]))
                    if nxt < 1e9:
                        out.append((nxt - 1.2, b[0], b[1]))
            self.keys[pid] = sorted(out)
            self._karr = None

    def _base_at(self, pid, t):
        # formation in force at time t (rotation changes between rallies)
        rot = self.rallies[0]["rotation"]
        for r in self.rallies:
            if r["t_start"] <= t + 1e-6:
                rot = r["rotation"]
        pl = self.players[pid]
        z = ZONE_ORDER[(pid % 6 + rot[pl.team]) % 6]
        return L(pl.team, np.array(ZONES[z]))

    def player_pos(self, pid, t):
        ks = self.keys[pid]
        ts = [k[0] for k in ks]
        x = np.interp(t, ts, [k[1] for k in ks])
        y = np.interp(t, ts, [k[2] for k in ks])
        ph = self.phase[pid]
        x += 0.12 * np.sin(0.9 * t + ph[0])
        y += 0.12 * np.sin(0.7 * t + ph[1])
        z = 0.0
        for tj, h in self.jumps[pid]:
            d = (t - tj) / 0.35
            if abs(d) < 1:
                z = max(z, h * (1 - d * d))
        return np.array([x, y, z])

    def ball(self, t):
        for fl in self.flights:
            if fl.t0 <= t < fl.t1:
                return fl.pos(t)
        return None

    def truth_contacts(self):
        return [c for r in self.rallies for c in r["contacts"]]
