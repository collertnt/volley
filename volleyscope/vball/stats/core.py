"""Player and team statistics from classified rallies.

Every number carries the sample it was computed from. A stat with no events
is None ("no data"), never 0. A stat built mostly on low-confidence events, or
on very few events, is marked uncertain. Contacts whose player could not be
identified are counted in an "unidentified" bucket and never assigned.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field

from ..geometry import NET_Y, other

LOW_CONF = 0.7         # events / rallies below this confidence count as uncertain
MIN_N = 5              # fewer events than this: "low sample"
UNCERTAIN_SHARE = 0.25


@dataclass
class Stat:
    name: str
    num: float = 0.0
    den: int = 0
    uncertain: int = 0          # how many of `den` came from low-confidence events
    kind: str = "pct"           # pct | ratio | avg | count

    @property
    def value(self):
        if self.kind == "count":
            return self.num
        if self.den == 0:
            return None
        return self.num / self.den

    @property
    def status(self):
        if self.den == 0 and self.kind != "count":
            return "no data"
        n = self.den if self.kind != "count" else int(self.num)
        if n and self.uncertain / max(n, 1) > UNCERTAIN_SHARE:
            return "uncertain"
        if self.kind != "count" and self.den < MIN_N:
            return "low sample"
        return "ok"

    def text(self):
        v = self.value
        if v is None:
            return "no data"
        if self.kind == "pct":
            s = f"{100 * v:.0f}% ({self.num:g}/{self.den})"
        elif self.kind == "ratio":
            s = f"{v:+.3f} (n={self.den})"
        elif self.kind == "avg":
            s = f"{v:.2f} (n={self.den})"
        else:
            s = f"{int(v)}"
        st = self.status
        return s if st == "ok" else f"{s} [{st}]"

    def to_json(self):
        return {"value": self.value, "num": self.num, "den": self.den, "uncertain": self.uncertain,
                "status": self.status, "text": self.text()}


def _low(e):
    return e.conf < LOW_CONF


class Tally:
    """Raw counts for one player (or one player in one rotation)."""

    def __init__(self):
        self.c = defaultdict(float)
        self.u = defaultdict(int)

    def add(self, key, e=None, w=1.0):
        self.c[key] += w
        if e is not None and _low(e):
            self.u[key] += 1

    def stats(self) -> dict:
        c, u = self.c, self.u
        att, rec, srv, dig, st = int(c["attack"]), int(c["reception"]), int(c["serve"]), int(c["dig"]), int(c["set"])
        contacts = int(c["contacts"])
        errors = c["attack_error"] + c["attack_blocked"] + c["serve_error"] + c["reception_error"] + c["dig_error"] \
            + c["free_ball_error"] + c["set_error"]
        return {
            "attack_attempts": Stat("Attack attempts", att, kind="count", uncertain=u["attack"]),
            "kills": Stat("Kills", c["kill"], kind="count", uncertain=u["kill"]),
            "kill_pct": Stat("Kill %", c["kill"], att, u["attack"]),
            "attack_eff": Stat("Attack efficiency", c["kill"] - c["attack_error"] - c["attack_blocked"], att, u["attack"], "ratio"),
            "attack_errors": Stat("Attack errors (incl. blocked)", c["attack_error"] + c["attack_blocked"], kind="count"),
            "reception_avg": Stat("Reception grade (0-3)", c["reception_grade_sum"], rec, u["reception"], "avg"),
            "reception_3": Stat("Perfect passes (3)", c["reception_g3"], rec, u["reception"]),
            "reception_2": Stat("Good passes (2)", c["reception_g2"], rec, u["reception"]),
            "reception_1": Stat("Poor passes (1)", c["reception_g1"], rec, u["reception"]),
            "reception_0": Stat("Reception errors (0)", c["reception_g0"], rec, u["reception"]),
            "serve_in_pct": Stat("Serve in %", srv - c["serve_error"], srv, u["serve"]),
            "aces": Stat("Aces", c["ace"], kind="count"),
            "serve_errors": Stat("Serve errors", c["serve_error"], kind="count"),
            "digs": Stat("Digs", c["dig_ok"], kind="count", uncertain=u["dig"]),
            "dig_pct": Stat("Dig success %", c["dig_ok"], dig, u["dig"]),
            "blocks": Stat("Blocks (stuff)", c["block_stuff"], kind="count", uncertain=u["block"]),
            "block_touches": Stat("Block touches", c["block_touch"], kind="count", uncertain=u["block"]),
            "sets": Stat("Sets", st, kind="count", uncertain=u["set"]),
            "assist_pct": Stat("Sets ending in a kill", c["assist"], st, u["set"]),
            "error_rate": Stat("Error rate (errors / contacts)", errors, contacts, u["contacts"]),
            "contacts": Stat("Contacts", contacts, kind="count", uncertain=u["contacts"]),
        }


def _tally_event(t: Tally, e):
    t.add("contacts", e)
    if e.type == "attack":
        t.add("attack", e)
        if e.outcome == "kill":
            t.add("kill", e)
        elif e.outcome == "error":
            t.add("attack_error", e)
        elif e.outcome == "blocked":
            t.add("attack_blocked", e)
    elif e.type == "reception":
        g = e.attrs.get("grade")
        if g is not None:
            t.add("reception", e)
            t.add("reception_grade_sum", w=g)
            t.add(f"reception_g{g}")
            if g == 0:
                t.add("reception_error")
    elif e.type == "serve":
        if e.outcome in ("ace", "error", "in_play"):
            t.add("serve", e)
            if e.outcome == "ace":
                t.add("ace")
            if e.outcome == "error":
                t.add("serve_error")
    elif e.type == "dig":
        t.add("dig", e)
        t.add("dig_ok" if e.outcome == "ok" else "dig_error")
    elif e.type == "block":
        t.add("block", e)
        t.add("block_stuff" if e.outcome == "stuff" else "block_touch")
    elif e.type == "set":
        t.add("set", e)
        if e.outcome == "assist":
            t.add("assist")
    elif e.type == "free_ball":
        if e.outcome == "error":
            t.add("free_ball_error")


# ---------------------------------------------------------------------------
@dataclass
class MatchSet:
    """One set (or one video): its rallies and which team is on which side."""
    rallies: list
    near: str = "Near team"
    far: str = "Far team"
    start_rotation: dict = field(default_factory=lambda: {"near": 1, "far": 1})

    def name(self, side):
        return self.near if side == "near" else self.far


def annotate_rotations(ms: MatchSet):
    """Give each rally the rotation each side is in (1-6, counted from the
    rotation the set started in) and the running score.

    A team rotates when it wins the serve back. The serve order is read from
    who actually serves each rally; where the serve was not seen, the previous
    rally's winner is used and the rally is flagged."""
    rot = dict(ms.start_rotation)
    score = {"near": 0, "far": 0}
    prev_server = None
    out = []
    uncertain = False
    for i, r in enumerate(ms.rallies):
        server = r.serve_team
        flags = []
        if server is None:
            prev = ms.rallies[i - 1] if i else None
            server = prev.winner if prev is not None and prev.winner else prev_server
            flags.append("server_assumed")
            uncertain = True
        if prev_server is not None and server is not None and server != prev_server:
            rot[server] = rot[server] % 6 + 1
        if uncertain:
            flags.append("rotation_uncertain")
        out.append({"rally": r.id, "server": server, "rotation": dict(rot), "score_before": dict(score), "flags": flags})
        if r.winner in score:
            score[r.winner] += 1
        elif r.winner is None:
            uncertain = True            # an unknown winner may hide a side-out
        prev_server = server if server is not None else prev_server
    return out, score


def _player_key(ms, e):
    team = ms.name(e.team) if e.team else "?"
    if e.number is not None:
        return (team, e.number)
    return (team, None)


DEEP, CORNER = 2.5, 2.5      # metres from the end line / sideline that count as "deep corner"


def classify_lost_point(r, side):
    """Why `side` lost rally r: (cause, responsible event or None, situation tags)."""
    seq = [e for e in r.events]
    tags = []
    ours = [e for e in seq if e.team == side]
    theirs = [e for e in seq if e.team != side]
    cause, resp = r.reason or "unknown", None
    term = next((e for e in reversed(seq) if e.outcome in ("kill", "ace", "error", "blocked")), None)
    if term is not None:
        resp = term
    # what kind of ball beat us
    win = next((e for e in reversed(theirs) if e.outcome in ("kill", "ace")), None)
    if win is not None and win.type == "attack":
        if win.attrs.get("shot") == "tip":
            tags.append("tip")
        tgt = win.attrs.get("target") or (r.landing[:2] if r.landing else None)
        if tgt is not None:
            depth = tgt[1] if side == "near" else 18 - tgt[1]
            if depth <= DEEP and min(tgt[0], 9 - tgt[0]) <= CORNER:
                tags.append("deep_corner")
    # broken play: our last possession started with a poor first touch
    last_first = next((e for e in reversed(ours) if e.type in ("reception", "dig")), None)
    if last_first is not None and last_first.attrs.get("grade") is not None and last_first.attrs["grade"] <= 1:
        tags.append("broken_play")
    # out-of-system attack: our last attack / free ball was out of system and we lost the point
    last_att = next((e for e in reversed(ours) if e.type in ("attack", "free_ball")), None)
    if last_att is not None and last_att.attrs.get("in_system") is False:
        tags.append("out_of_system_attack")
    return cause, resp, tags


def compute_stats(sets: list[MatchSet]) -> dict:
    """All player and team stats for a match (one or more sets)."""
    players = defaultdict(Tally)
    by_rot = defaultdict(lambda: defaultdict(Tally))
    teams = {}
    lost = defaultdict(list)
    names = set()
    final_scores = []
    for si, ms in enumerate(sets):
        names.update([ms.near, ms.far])
        rots, score = annotate_rotations(ms)
        final_scores.append({ms.near: score["near"], ms.far: score["far"]})
        for r, info in zip(ms.rallies, rots):
            for e in r.events:
                if e.team is None or e.type == "unknown":
                    continue
                key = _player_key(ms, e)
                _tally_event(players[key], e)
                _tally_event(by_rot[key][info["rotation"][e.team]], e)
            for side in ("near", "far"):
                T = teams.setdefault(ms.name(side), {"so": Stat("Side-out %"), "bp": Stat("Point-scoring % on own serve"),
                                                     "rot": defaultdict(lambda: {"played": 0, "lost": 0, "uncertain": 0}),
                                                     "sit": defaultdict(lambda: [0, 0]), "lost_total": 0,
                                                     "lost_uncertain": 0, "unknown_winner": 0})
                if r.winner is None:
                    T["unknown_winner"] += 1
                    continue
                unc = r.conf < LOW_CONF or bool(info["flags"])
                won = r.winner == side
                st = T["so"] if info["server"] == other(side) else T["bp"] if info["server"] == side else None
                if st is not None:
                    st.den += 1
                    st.num += won
                    st.uncertain += unc
                rr = T["rot"][info["rotation"][side]]
                rr["played"] += 1
                rr["uncertain"] += unc
                if not won:
                    rr["lost"] += 1
                    T["lost_total"] += 1
                    T["lost_uncertain"] += unc
                    cause, resp, tags = classify_lost_point(r, side)
                    for tg in tags:
                        T["sit"][tg][0] += 1
                        T["sit"][tg][1] += unc
                    lost[ms.name(side)].append({
                        "set": si + 1, "rally": r.id, "rotation": info["rotation"][side],
                        "rotation_uncertain": "rotation_uncertain" in info["flags"], "cause": cause, "tags": tags,
                        "by_team": ms.name(resp.team) if resp is not None and resp.team else None,
                        "by_number": resp.number if resp is not None else None,
                        "by_type": resp.type if resp is not None else None,
                        "conf": round(r.conf, 2), "flags": list(r.flags), "t": round(r.t_start, 1)})
    out_players = {}
    for key, t in players.items():
        team, num = key
        label = f"#{num}" if num is not None else "unidentified"
        out_players[key] = {"team": team, "number": num, "label": label, "stats": t.stats(),
                            "by_rotation": {k: v.stats() for k, v in sorted(by_rot[key].items())}}
    out_teams = {}
    for name, T in teams.items():
        sit = {}
        for tag, title in (("tip", "Lost to tips"), ("deep_corner", "Lost to the deep corner"),
                           ("broken_play", "Lost after a broken play (first touch graded 0-1)"),
                           ("out_of_system_attack", "Lost after an out-of-system attack")):
            n, u = T["sit"][tag]
            sit[tag] = Stat(title, n, T["lost_total"], u)
        out_teams[name] = {
            "side_out": T["so"], "break_point": T["bp"], "points_lost": T["lost_total"],
            "points_lost_uncertain": T["lost_uncertain"], "unknown_winner": T["unknown_winner"],
            "by_rotation": {k: dict(v) for k, v in sorted(T["rot"].items())}, "situations": sit,
            "lost_points": lost[name]}
    return {"players": out_players, "teams": out_teams, "scores": final_scores}


def player_stats(stats, team, number):
    return stats["players"].get((team, number))


def stats_to_json(stats):
    def conv(o):
        if isinstance(o, Stat):
            return o.to_json()
        if isinstance(o, dict):
            return {str(k): conv(v) for k, v in o.items()}
        if isinstance(o, (list, tuple)):
            return [conv(v) for v in o]
        return o
    return {"players": [conv(v) for v in stats["players"].values()], "teams": conv(stats["teams"]),
            "scores": stats["scores"]}
