"""Hand-tagged matches.

Reads the file saved by tagger/tagger.html (a person marks each touch while
watching the video) and turns it into the same rallies the statistics,
tendency and report code already use. Nothing here looks at the video.

In a tags file "near" is always the tagger's own team, drawn at the bottom of
the court diagram (y < 9), whatever end they actually played from.
"""
from __future__ import annotations

import json
import os

from .ball.events import Event, Rally, rallies_to_json
from .geometry import other

TYPES = {"serve", "reception", "set", "attack", "block", "dig", "free_ball"}
RESULTS = {"serve": {"ace", "error", "in_play"}, "attack": {"kill", "error", "blocked", "dug"},
           "block": {"stuff", "touch"}, "dig": {"ok", "error"}, "free_ball": {"in_play", "error"}}


def load_tags(path):
    with open(path) as fh:
        return json.load(fh)


def rallies_from_tags(tags: dict):
    """Returns (rallies, warnings). Warnings are things a person should look at;
    nothing is guessed to fill them in."""
    rallies, warnings = [], []
    for ri, r in enumerate(tags.get("rallies", [])):
        label = f"rally {ri + 1}"
        events = []
        for k, t in enumerate(r.get("touches", [])):
            typ, team = t.get("type"), t.get("team")
            if typ not in TYPES or team not in ("near", "far"):
                warnings.append(f"{label}: touch {k + 1} has an unknown type or team and was skipped")
                continue
            res = t.get("result")
            if typ in RESULTS and res not in RESULTS[typ]:
                warnings.append(f"{label}: {typ} at {t.get('t', 0):.1f}s has no valid result and was skipped")
                continue
            attrs = {}
            grade = t.get("grade")
            if typ == "reception":
                if grade not in (0, 1, 2, 3):
                    warnings.append(f"{label}: reception at {t.get('t', 0):.1f}s has no grade and was skipped")
                    continue
                attrs["grade"] = grade
                res = "error" if grade == 0 else "ok"
            elif typ == "dig" and grade in (0, 1, 2, 3):
                attrs["grade"] = grade
            if typ == "set":
                res = "set"
            if typ == "attack":
                if t.get("shot"):
                    attrs["shot"] = t["shot"]
                if t.get("target"):
                    attrs["target"] = [float(v) for v in t["target"]]
            num = t.get("number")
            flags = [] if num is not None else ["player_number_unknown"]
            pos = [float(t["pos"][0]), float(t["pos"][1]), 0.0] if t.get("pos") else None
            events.append(Event(ri, len(events), float(t.get("t", 0.0)), 0, pos, team, None,
                                None if num is None else int(num), typ, res, attrs, 1.0, flags))
        # in / out of system follows the first touch of the same possession
        for k, e in enumerate(events):
            if e.type in ("set", "attack", "free_ball"):
                for p in reversed(events[:k]):
                    if p.team != e.team:
                        break
                    if p.type in ("reception", "dig") and "grade" in p.attrs:
                        e.attrs["in_system"] = p.attrs["grade"] >= 2
                        break
            if e.type == "set":
                nxt = events[k + 1] if k + 1 < len(events) else None
                if nxt is not None and nxt.team == e.team and nxt.type == "attack":
                    e.attrs["target_number"] = nxt.number
                    if nxt.pos is not None:
                        e.attrs["target_pos"] = nxt.pos[:2]
                    if nxt.outcome == "kill":
                        e.outcome = "assist"
        winner = r.get("winner")
        serve_team = r.get("serve_team")
        if serve_team not in ("near", "far"):
            serve_team = None
            warnings.append(f"{label}: no server recorded; rotation numbers after this are uncertain")
        if winner not in ("near", "far"):
            winner = None
            if events:
                warnings.append(f"{label}: no point was awarded; it is left out of win/loss figures")
        reason = "tagged"
        last = events[-1] if events else None
        if last is not None and winner is not None:
            if last.outcome in ("kill", "ace") and last.team == winner:
                reason = last.outcome
            elif last.type == "block" and last.outcome == "stuff" and last.team == winner:
                reason = "blocked"
                prev = events[-2] if len(events) > 1 else None
                if prev is not None and prev.type == "attack":
                    prev.outcome = "blocked"
            elif last.outcome == "blocked" and last.team != winner:
                reason = "blocked"
            elif last.outcome == "error" and last.team != winner:
                reason = f"{last.type}_error"
            elif last.outcome in ("kill", "ace", "stuff") and last.team != winner:
                warnings.append(f"{label}: last touch is a {last.outcome} by one team but the point went to the other")
        if not events and winner is None:
            continue
        t0 = events[0].t if events else 0.0
        landing = None
        if last is not None and last.type == "attack" and "target" in last.attrs:
            landing = [last.attrs["target"][0], last.attrs["target"][1], 0.1]
        rallies.append(Rally(ri, serve_team, events, t0, events[-1].t if events else t0, landing, winner, reason, 1.0, []))
    return rallies, warnings


def import_tags(path, out_dir):
    """Write a tags file out as an analysis folder that the `player`, `team`
    and `tendencies` commands accept. Returns (n_rallies, n_touches, warnings)."""
    tags = load_tags(path)
    rallies, warnings = rallies_from_tags(tags)
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "rallies.json"), "w") as fh:
        json.dump(rallies_to_json(rallies), fh, indent=1)
    with open(os.path.join(out_dir, "tracks.json"), "w") as fh:
        json.dump({"fps": 30.0, "identities": []}, fh)
    info = {"manual": True, "fps": 30.0, "near": tags.get("near", "My Team"), "far": tags.get("far", "Opponent"),
            "video": tags.get("video", ""), "n_rallies": len(rallies),
            "n_touches": sum(len(r.events) for r in rallies), "warnings": warnings}
    with open(os.path.join(out_dir, "info.json"), "w") as fh:
        json.dump(info, fh, indent=1)
    return len(rallies), info["n_touches"], warnings
