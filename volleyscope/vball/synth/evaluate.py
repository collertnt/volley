"""Scoring pipeline output against the synthetic ground truth."""
from __future__ import annotations

from collections import Counter, defaultdict

import numpy as np
from scipy.optimize import linear_sum_assignment


def evaluate_tracking(world, times, tracks, radius=1.0):
    """Compare identities with true players.

    coverage    share of (player, frame) pairs with an identity within `radius` m
    mean_err    mean position error (m) on covered pairs
    id_switches times the identity following a true player changed
    purity      share of identity-frames spent on that identity's main player
    """
    covered = total = 0
    errs = []
    last = {}
    switches = 0
    per_ident = defaultdict(Counter)
    for fi, t in enumerate(times):
        truth = np.array([world.player_pos(p.pid, t)[:2] for p in world.players])
        cur = tracks.at(fi)
        total += len(truth)
        if not cur:
            continue
        pos = np.array([p for _, p in cur])
        D = np.linalg.norm(truth[:, None] - pos[None], axis=2)
        r, c = linear_sum_assignment(np.where(D < radius, D, 1e3))
        for i, j in zip(r, c):
            if D[i, j] >= radius:
                continue
            covered += 1
            errs.append(D[i, j])
            ident = cur[j][0].id
            per_ident[ident][i] += 1
            if i in last and last[i] != ident:
                switches += 1
            last[i] = ident
    main = {k: v.most_common(1)[0] for k, v in per_ident.items()}
    purity = sum(n for _, n in main.values()) / max(1, sum(sum(v.values()) for v in per_ident.values()))
    numbers = {"right": 0, "wrong": 0, "unread": 0}
    for k, (pid, _) in main.items():
        ident = tracks.identities[k]
        if ident.number is None:
            numbers["unread"] += 1
        elif ident.number == world.players[pid].number:
            numbers["right"] += 1
        else:
            numbers["wrong"] += 1
    return {"coverage": covered / max(1, total), "mean_err": float(np.mean(errs)) if errs else None,
            "p95_err": float(np.percentile(errs, 95)) if errs else None, "id_switches": switches,
            "purity": purity, "n_identities": len(tracks.identities), "numbers": numbers,
            "main_player": {k: v[0] for k, v in main.items()}}


def truth_corrections(world, times, tracks):
    """Build the manual-corrections dict a human reviewer would write, from truth."""
    ev = evaluate_tracking(world, times, tracks)
    return {"numbers": {k: world.players[pid].number for k, pid in ev["main_player"].items()}}


def evaluate_events(world, rallies, tracks=None, tol=0.3):
    """Match detected events to true contacts by time and compare labels."""
    truth = world.truth_contacts()
    det = [e for r in rallies for e in r.events]
    used, pairs = set(), []
    for c in truth:
        best = None
        for k, e in enumerate(det):
            if k in used or abs(e.t - c["t"]) > tol:
                continue
            if best is None or abs(e.t - c["t"]) < abs(det[best].t - c["t"]):
                best = k
        if best is not None:
            used.add(best)
            pairs.append((c, det[best]))
    res = {"true_contacts": len(truth), "detected": len(det), "matched": len(pairs),
           "spurious": len(det) - len(pairs)}
    res["type_ok"] = sum(c["type"] == e.type for c, e in pairs)
    res["team_ok"] = sum(c["team"] == e.team for c, e in pairs)
    res["pos_err"] = float(np.mean([np.linalg.norm(np.array(c["pos"]) - np.array(e.pos)) for c, e in pairs])) if pairs else None
    res["player_ok"] = sum(e.number == c["number"] for c, e in pairs)
    res["player_wrong"] = sum(e.number is not None and e.number != c["number"] for c, e in pairs)
    oc = [(c, e) for c, e in pairs if c["type"] == "attack" and "outcome" in c]
    res["attacks"] = len(oc)
    res["attack_outcome_ok"] = sum(c["outcome"] == e.outcome for c, e in oc)
    res["shot_ok"] = sum(c["shot"] == e.attrs.get("shot") for c, e in oc)
    gr = [(c, e) for c, e in pairs if c["type"] in ("reception", "dig")]
    res["passes"] = len(gr)
    res["grade_ok"] = sum(c["grade"] == e.attrs.get("grade") for c, e in gr)
    # rallies, matched by overlap in time
    rw = 0
    matched_r = 0
    for tr in world.rallies:
        cand = [r for r in rallies if r.t_start < tr["t_end"] and r.t_end > tr["contacts"][0]["t"]]
        if cand:
            matched_r += 1
            rw += cand[0].winner == tr["winner"]
    res.update(true_rallies=len(world.rallies), rallies_found=len(rallies), rallies_matched=matched_r, winner_ok=rw)
    return res


def truth_tracks(world, times, fps=30.0):
    """Perfect player tracks straight from the simulator (for testing the
    event logic without any vision error)."""
    from ..players.track import Identity, PlayerTracks
    frames = np.arange(len(times))
    ids = []
    for p in world.players:
        pos = np.array([world.player_pos(p.pid, t) for t in times])
        ids.append(Identity(p.pid, p.team, p.number, 1.0, frames, pos[:, :2], np.ones(len(times), bool),
                            pos[:, 2] > 0.05, {}, "manual"))
    return PlayerTracks(ids, fps)


def truth_flights(world):
    """Perfect ball flights straight from the simulator."""
    from ..ball.track import Flight
    return [Flight(f.t0, f.t1, f.p0.copy(), f.v0.copy(), 0.0, max(2, int((f.t1 - f.t0) * 30))) for f in world.flights]
