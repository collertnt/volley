"""Tendency mining: setter distribution, hitter shots, opponent block and
defence positioning, and plain-language coaching notes.

Every tendency is a table of counts. Nothing is reported as a tendency
unless it has at least `min_n` observations; below that the entry says
"insufficient data" and shows how many it had.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass, field

import numpy as np

from ..geometry import NET_Y, other
from ..stats.core import LOW_CONF, MatchSet, annotate_rotations

MIN_N = 8


@dataclass
class Tendency:
    title: str
    counts: dict                       # label -> count
    n: int
    min_n: int = MIN_N
    uncertain: int = 0
    note: str = ""

    @property
    def sufficient(self):
        return self.n >= self.min_n

    def shares(self):
        return {k: v / self.n for k, v in self.counts.items()} if self.n else {}

    def text(self):
        if not self.sufficient:
            return f"{self.title}: insufficient data (n={self.n}, need {self.min_n})"
        parts = ", ".join(f"{k} {100 * v / self.n:.0f}% ({v})" for k, v in sorted(self.counts.items(), key=lambda kv: -kv[1]))
        tail = f" [{self.uncertain} of these from low-confidence events]" if self.uncertain else ""
        return f"{self.title}: {parts} (n={self.n}){tail}" + (f" - {self.note}" if self.note else "")

    def to_json(self):
        return {"title": self.title, "n": self.n, "sufficient": self.sufficient, "counts": self.counts,
                "uncertain": self.uncertain, "text": self.text()}


def local(side, pos):
    """Court position in the team's own frame: x left-to-right as the team
    faces the net, d = distance from the net."""
    x, y = pos[0], pos[1]
    if side == "near":
        return x, NET_Y - y
    return 9 - x, y - NET_Y


def attack_zone(side, pos):
    """Where along the net an attack was hit from, in the hitter's own terms."""
    x, d = local(side, pos)
    lane = "left" if x < 3 else "right" if x > 6 else "middle"
    return f"{lane} {'back row' if d > 3.0 else 'front'}"


def _tend(title, labels, events, min_n):
    c = Counter(labels)
    return Tendency(title, dict(c), sum(c.values()), min_n, sum(e.conf < LOW_CONF for e in events))


def _iter(sets):
    for ms in sets:
        rots, _ = annotate_rotations(ms)
        for r, info in zip(ms.rallies, rots):
            yield ms, r, info


def setter_distribution(sets, team, min_n=MIN_N):
    """Where `team`'s sets go: overall, by rotation, by front/back row and by
    the quality of the pass that came before."""
    rows = []
    for ms, r, info in _iter(sets):
        seq = r.events
        for k, e in enumerate(seq):
            if e.type != "set" or e.team is None or ms.name(e.team) != team:
                continue
            tgt = e.attrs.get("target_pos")
            if tgt is None:
                continue
            first = next((p for p in reversed(seq[:k]) if p.type in ("reception", "dig") and p.team == e.team), None)
            rows.append({"zone": attack_zone(e.team, tgt), "rot": info["rotation"][e.team],
                         "grade": first.attrs.get("grade") if first is not None else None,
                         "setter": e.number, "ev": e})
    out = {"overall": _tend(f"{team} set distribution", [r["zone"] for r in rows], [r["ev"] for r in rows], min_n)}
    for rot in range(1, 7):
        sub = [r for r in rows if r["rot"] == rot]
        out[f"rotation {rot}"] = _tend(f"{team} sets in rotation {rot}", [r["zone"] for r in sub], [r["ev"] for r in sub], min_n)
    for g, name in ((3, "perfect pass (3)"), (2, "good pass (2)"), (1, "poor pass (1)")):
        sub = [r for r in rows if r["grade"] == g]
        out[name] = _tend(f"{team} sets after a {name}", [r["zone"] for r in sub], [r["ev"] for r in sub], min_n)
    out["row"] = _tend(f"{team} front vs back row sets",
                       ["back row" if "back" in r["zone"] else "front row" for r in rows], [r["ev"] for r in rows], min_n)
    return out


def hitter_tendencies(sets, team, number=None, min_n=MIN_N):
    """Shot selection (line / cross / tip / roll) of one hitter or the whole
    team, overall and by situation."""
    rows = []
    for ms, r, info in _iter(sets):
        for e in r.events:
            if e.type != "attack" or e.team is None or ms.name(e.team) != team:
                continue
            if number is not None and e.number != number:
                continue
            rows.append(e)
    who = f"{team} #{number}" if number is not None else team
    allrows = rows
    rows = [e for e in allrows if "shot" in e.attrs]          # shot type is optional when tagging by hand
    out = {"overall": _tend(f"{who} shot selection", [e.attrs["shot"] for e in rows], rows, min_n)}
    for name, sel in (("in system", lambda e: e.attrs.get("in_system") is True),
                      ("out of system", lambda e: e.attrs.get("in_system") is False)):
        sub = [e for e in rows if sel(e)]
        out[name] = _tend(f"{who} shots {name}", [e.attrs["shot"] for e in sub], sub, min_n)
    for lane in ("left", "middle", "right"):
        sub = [e for e in rows if e.pos is not None and attack_zone(e.team, e.pos).startswith(lane)]
        out[f"from the {lane}"] = _tend(f"{who} shots from the {lane}", [e.attrs["shot"] for e in sub], sub, min_n)
    kills = [e for e in rows if e.outcome == "kill"]
    out["kills by shot"] = _tend(f"{who} kills by shot type", [e.attrs["shot"] for e in kills], kills, min_n)
    out["result"] = _tend(f"{who} attack results", [e.outcome or "unknown" for e in allrows], allrows, min_n)
    return out


def _grid(side, xy):
    """3x3 zone of a defender on their own half: (left/centre/right, short/mid/deep)."""
    x, d = local(side, xy)
    return ("left" if x < 3 else "right" if x > 6 else "centre") + "-" + ("short" if d < 3 else "deep" if d > 6 else "mid")


def defence_tendencies(sets, tracks_per_set, team, fps, min_n=MIN_N):
    """How `team` lines up when the other side attacks: how many blockers are
    at the net in front of the hitter, where the back-court defenders stand,
    and which areas are left open.

    "Blockers" means players standing within 1.2 m of the net and 1.5 m either
    side of the hitter at contact - a positional proxy; whether they jumped is
    only known if the tracker flagged them airborne."""
    blockers, open_zone, depth_rows, events = [], [], [], []
    for (ms, r, info) in _iter(sets):
        tracks = tracks_per_set[sets.index(ms)]
        if tracks is None:
            continue
        for e in r.events:
            if e.type != "attack" or e.team is None or ms.name(other(e.team)) != team:
                continue
            side = other(e.team)
            frame = e.frame
            defenders = [(i, xy) for i, xy in tracks.at(frame) if i.team == side]
            if len(defenders) < 4:
                continue                         # too few tracked to describe a formation
            nb = sum(1 for _, xy in defenders if abs(xy[1] - NET_Y) < 1.2 and abs(xy[0] - e.pos[0]) < 1.5)
            blockers.append(f"{min(nb, 3)} blocker{'s' if nb != 1 else ''}")
            back = [xy for _, xy in defenders if abs(xy[1] - NET_Y) >= 1.2]
            occupied = {_grid(side, xy) for xy in back}
            for z in ("left-short", "centre-short", "right-short", "left-mid", "centre-mid", "right-mid",
                      "left-deep", "centre-deep", "right-deep"):
                if z not in occupied:
                    open_zone.append(z)
            if back:
                depth_rows.append(float(np.mean([local(side, xy)[1] for xy in back])))
            events.append(e)
    n = len(events)
    out = {"blockers": _tend(f"{team} blockers in front of the hitter", blockers, events, min_n)}
    oz = Counter(open_zone)
    t = Tendency(f"{team} zones left empty when defending an attack", dict(oz), n, min_n,
                 sum(e.conf < LOW_CONF for e in events),
                 "share of attacks on which nobody stood in that zone")
    t.text = lambda t=t: (f"{t.title}: insufficient data (n={t.n}, need {t.min_n})" if not t.sufficient else
                          f"{t.title}: " + ", ".join(f"{k} {100 * v / t.n:.0f}%" for k, v in sorted(t.counts.items(), key=lambda kv: -kv[1])[:4])
                          + f" (n={t.n} attacks) - {t.note}")
    out["open zones"] = t
    out["depth"] = {"mean_back_court_depth_m": float(np.mean(depth_rows)) if n >= min_n else None, "n": n,
                    "sufficient": n >= min_n}
    return out


def coaching_notes(stats, sets, team, number, min_n=MIN_N):
    """Plain-language observations for one player, each with its evidence.

    Rules of thumb applied to the player's own numbers. A note is only made
    when the sample behind it reaches `min_n`; otherwise the area is listed as
    "not enough data to say"."""
    notes, skipped = [], []
    p = stats["players"].get((team, number))
    if p is None:
        return {"notes": [], "skipped": ["no contacts were attributed to this player"]}
    s = p["stats"]

    def enough(st, label):
        if st.den < min_n:
            skipped.append(f"{label}: only {st.den} on record (need {min_n})")
            return False
        return True

    k = s["kill_pct"]
    if enough(k, "attacking"):
        eff = s["attack_eff"].value
        unc = " (many of these attacks were low-confidence detections)" if k.status == "uncertain" else ""
        if eff < 0.1:
            notes.append(f"Attack efficiency is {eff:+.2f} on {k.den} swings: errors and blocks are cancelling out your kills. "
                         f"{int(s['attack_errors'].value)} of {k.den} attacks ended in an error or a block{unc}.")
        elif eff >= 0.3:
            notes.append(f"Attacking is a strength: {100 * k.value:.0f}% kills, efficiency {eff:+.2f} on {k.den} swings{unc}.")
        else:
            notes.append(f"Attacking is average: {100 * k.value:.0f}% kills, efficiency {eff:+.2f} on {k.den} swings{unc}.")
        h = hitter_tendencies(sets, team, number, min_n)
        t = h["overall"]
        if t.sufficient:
            top, share = max(t.shares().items(), key=lambda kv: kv[1])
            if share >= 0.65:
                notes.append(f"Predictable shot selection: {100 * share:.0f}% of your {t.n} attacks were {top} shots. "
                             f"Defences adjust to that; mix in another shot.")
        res_in = [e for ms, r, _ in _iter(sets) for e in r.events
                  if e.type == "attack" and e.team and ms.name(e.team) == team and e.number == number]
        oos = [e for e in res_in if e.attrs.get("in_system") is False]
        if len(oos) >= min_n:
            bad = sum(e.outcome in ("error", "blocked") for e in oos)
            if bad / len(oos) >= 0.3:
                notes.append(f"Out of system you gave away {bad} of {len(oos)} attacks (error or blocked). "
                             f"On a bad pass, a controlled shot deep keeps the rally alive.")
        elif oos:
            skipped.append(f"out-of-system attacking: only {len(oos)} on record (need {min_n})")
    r = s["reception_avg"]
    if enough(r, "serve receive"):
        if r.value < 1.8:
            notes.append(f"Serve receive averages {r.value:.2f} on {r.den} passes (3 is perfect): "
                         f"{int(s['reception_0'].num + s['reception_1'].num)} of them left your setter out of system.")
        elif r.value >= 2.3:
            notes.append(f"Serve receive is a strength: {r.value:.2f} average on {r.den} passes.")
        else:
            notes.append(f"Serve receive is serviceable: {r.value:.2f} average on {r.den} passes.")
    sv = s["serve_in_pct"]
    if enough(sv, "serving"):
        if sv.value < 0.85:
            notes.append(f"You missed {int(s['serve_errors'].value)} of {sv.den} serves ({100 * sv.value:.0f}% in). "
                         f"Each miss is a free point for the other side.")
        elif s["aces"].value == 0 and sv.value >= 0.95:
            notes.append(f"{100 * sv.value:.0f}% of {sv.den} serves in but no aces: there is room to serve tougher.")
        else:
            notes.append(f"Serving: {100 * sv.value:.0f}% in on {sv.den} serves, {int(s['aces'].value)} aces.")
    a = s["assist_pct"]
    if enough(a, "setting"):
        notes.append(f"{100 * a.value:.0f}% of your {a.den} sets ended in a kill.")
        d = setter_distribution(sets, team, min_n)["overall"]
        if d.sufficient:
            top, share = max(d.shares().items(), key=lambda kv: kv[1])
            if share >= 0.6:
                notes.append(f"Set distribution leans heavily one way: {100 * share:.0f}% of {d.n} sets went {top}.")
    dg = s["dig_pct"]
    if enough(dg, "digging"):
        notes.append(f"Digs: kept {int(dg.num)} of {dg.den} balls in play ({100 * dg.value:.0f}%).")
    er = s["error_rate"]
    if er.den >= min_n and er.value >= 0.2:
        notes.append(f"{100 * er.value:.0f}% of your {er.den} contacts ended the rally against your team.")
    return {"notes": notes, "skipped": skipped}


def report_lines(sets, tracks_per_set, team, opponent, fps, min_n=MIN_N):
    """The printed tendency report: a list of text lines."""
    lines = [f"TENDENCY REPORT - {team} (opponent: {opponent})",
             f"A tendency needs at least {min_n} observations; fewer is reported as insufficient data.", ""]
    lines.append("Setter distribution")
    for t in setter_distribution(sets, team, min_n).values():
        lines.append("  " + t.text())
    lines += ["", "Hitter shot tendencies (team)"]
    for t in hitter_tendencies(sets, team, None, min_n).values():
        lines.append("  " + t.text())
    numbers = sorted({e.number for ms, r, _ in _iter(sets) for e in r.events
                      if e.type == "attack" and e.number is not None and e.team and ms.name(e.team) == team})
    for num in numbers:
        lines.append(f"  #{num}: " + hitter_tendencies(sets, team, num, min_n)["overall"].text())
    lines += ["", f"Opponent ({opponent}) blocking and defence"]
    d = defence_tendencies(sets, tracks_per_set, opponent, fps, min_n)
    lines.append("  " + d["blockers"].text())
    lines.append("  " + d["open zones"].text())
    dep = d["depth"]
    lines.append("  " + (f"{opponent} back-court defenders stand on average {dep['mean_back_court_depth_m']:.1f} m from the net (n={dep['n']})"
                         if dep["sufficient"] else f"{opponent} defensive depth: insufficient data (n={dep['n']}, need {min_n})"))
    lines += ["", f"Opponent ({opponent}) attack"]
    for key in ("overall", "kills by shot"):
        lines.append("  " + hitter_tendencies(sets, opponent, None, min_n)[key].text())
    return lines
