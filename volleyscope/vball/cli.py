"""Command line interface. Run `python -m vball.cli --help`."""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np


def _progress(stage, i, n):
    print(f"\r  {stage}: frame {i}/{n}", end="", file=sys.stderr, flush=True)


def _load_sets(dirs):
    from .pipeline import load_analysis
    from .stats.core import MatchSet
    sets, tracks, infos, fps = [], [], [], 30.0
    for d in dirs:
        a = load_analysis(d)
        inf = a["info"]
        sets.append(MatchSet(a["rallies"], inf.get("near", "Near team"), inf.get("far", "Far team")))
        tracks.append(a["tracks"])
        inf = dict(inf)
        if inf.get("manual"):
            infos.append(inf)
            fps = a["fps"]
            continue
        inf["numbers_read"] = f"{sum(i.number is not None for i in a['tracks'].identities.values())} of {len(a['tracks'].identities)} tracks"
        inf["note"] = _quality_note(inf, a["rallies"])
        infos.append(inf)
        fps = a["fps"]
    return sets, tracks, infos, fps


def _quality_note(info, rallies):
    notes = []
    if info.get("court_usable_share", 1) < 0.9:
        notes.append("court lost for part of the video")
    low = sum(r.conf < 0.7 for r in rallies)
    if low:
        notes.append(f"{low} of {len(rallies)} rallies low-confidence")
    if not info.get("team_colours_distinct", True):
        notes.append("team jersey colours could not be told apart")
    return "; ".join(notes) or "no problems detected"


def cmd_analyze(a):
    import cv2
    from .pipeline import Config, analyze, save_analysis
    from .players.track import load_corrections
    from .review import review_sheet
    cfg = Config()
    cfg.calibration = a.calibration
    cfg.static_camera = getattr(a, 'static_camera', False)
    cfg.proc_width = a.width
    cfg.detect.detector = a.detector
    cfg.detect.torch_weights = a.torch_weights
    cfg.track.jersey_backend = a.jersey_backend
    corr = load_corrections(a.corrections) if a.corrections else None
    print(f"Analysing {a.video} (three passes over the video; expect roughly real time or slower on a laptop CPU) ...")
    res = analyze(a.video, cfg, corrections=corr, progress=_progress)
    print(file=sys.stderr)
    res["info"].update(near=a.near, far=a.far, video=os.path.abspath(a.video))
    save_analysis(res, a.out)
    sheet = review_sheet(a.video, res["tracks"], cfg.proc_width)
    if sheet is not None:
        cv2.imwrite(os.path.join(a.out, "tracks_review.png"), sheet)
    _summary(res, a.out)


def _summary(res, out):
    inf = res["info"]
    ids = res["tracks"].identities.values()
    print(f"\nSaved to {out}/")
    print(f"  court found in {100 * inf['court_usable_share']:.0f}% of {inf['frames']} frames {inf['court_status']}")
    print(f"  {len(ids)} player tracks; jersey number read on {sum(i.number is not None for i in ids)} of them")
    print(f"  {inf['n_rallies']} rallies, {sum(len(r.events) for r in res['rallies'])} contacts; "
          f"{sum(r.conf < 0.7 for r in res['rallies'])} rallies low-confidence")
    unread = [i for i in ids if i.number is None]
    if unread:
        print(f"  {len(unread)} tracks have no number. Open {out}/tracks_review.png, then write the numbers into a "
              f"corrections file and run:  vball relabel {out} --corrections corrections.yaml")


def cmd_relabel(a):
    from .ball.events import rallies_to_json
    from .court.track import CourtState
    from .pipeline import Config, run_ball
    from .players.track import PlayerTracks, apply_corrections, load_corrections
    info = json.load(open(os.path.join(a.dir, "info.json")))
    raw = os.path.join(a.dir, "tracks_raw.json")
    tracks = PlayerTracks.from_json(json.load(open(raw if os.path.exists(raw) else os.path.join(a.dir, "tracks.json"))))
    apply_corrections(tracks, load_corrections(a.corrections))
    court = json.load(open(os.path.join(a.dir, "court.json")))
    states = [CourtState(None if c["H"] is None else np.array(c["H"]), c["conf"], c["status"]) for c in court]
    balls = json.load(open(os.path.join(a.dir, "balls.json")))
    flights, rallies = run_ball(balls, states, tracks, tuple(info["size"]), info["fps"], Config())
    json.dump(tracks.to_json(), open(os.path.join(a.dir, "tracks.json"), "w"))
    json.dump(rallies_to_json(rallies), open(os.path.join(a.dir, "rallies.json"), "w"), indent=1)
    ids = tracks.identities.values()
    print(f"Relabelled: {sum(i.number is not None for i in ids)} of {len(ids)} tracks now have a number; "
          f"{len(rallies)} rallies re-scored.")


def cmd_import_tags(a):
    from .tagging import import_tags
    n, touches, warnings = import_tags(a.tags, a.out)
    print(f"Imported {touches} tagged touches in {n} rallies into {a.out}/")
    for w in warnings:
        print("  check:", w)
    print(f"Next:  vball player {a.out} --team \"<your team name>\" --number <your number> --pdf me.pdf")


def cmd_player(a):
    from .report.pdf import player_pdf
    from .stats.core import compute_stats
    from .tendencies.core import coaching_notes, hitter_tendencies
    sets, tracks, infos, fps = _load_sets(a.dirs)
    stats = compute_stats(sets)
    p = stats["players"].get((a.team, a.number))
    print(f"{a.team} #{a.number}")
    if p is None:
        print("  No contacts attributed to this player. Known players:",
              sorted(f"{t} #{n}" for (t, n) in stats["players"] if n is not None) or "none")
    else:
        for st in p["stats"].values():
            print(f"  {st.name:34s} {st.text()}")
        print("\nTendencies")
        for t in hitter_tendencies(sets, a.team, a.number).values():
            print("  " + t.text())
        notes = coaching_notes(stats, sets, a.team, a.number)
        print("\nWhat the numbers suggest")
        for n in notes["notes"] or ["  (nothing yet: no area has enough recorded events)"]:
            print("  - " + n.strip())
        for s in notes["skipped"]:
            print("  not enough data: " + s)
    if a.pdf:
        player_pdf(a.pdf, sets, a.team, a.number, infos, stats)
        print(f"\nPDF written to {a.pdf}")


def cmd_team(a):
    from .report.pdf import team_pdf
    from .stats.core import compute_stats
    sets, tracks, infos, fps = _load_sets(a.dirs)
    stats = compute_stats(sets)
    opp = a.opponent or next((n for n in stats["teams"] if n != a.team), "Opponent")
    T = stats["teams"].get(a.team)
    if T is None:
        print(f"No team called {a.team!r}. Teams in this analysis: {sorted(stats['teams'])}")
        return
    print(f"{a.team} vs {opp}: scores {stats['scores']}")
    print("  " + T["side_out"].name + ": " + T["side_out"].text())
    print(f"  points lost: {T['points_lost']} ({T['points_lost_uncertain']} from low-confidence rallies)")
    for st in T["situations"].values():
        print(f"  {st.name}: {st.text()}")
    if a.pdf:
        team_pdf(a.pdf, sets, a.team, opp, tracks, fps, infos, stats)
        print(f"PDF written to {a.pdf}")


def cmd_tendencies(a):
    from .tendencies.core import report_lines
    sets, tracks, infos, fps = _load_sets(a.dirs)
    names = {n for ms in sets for n in (ms.near, ms.far)}
    opp = a.opponent or next((n for n in names if n != a.team), "Opponent")
    print("\n".join(report_lines(sets, tracks, a.team, opp, fps, a.min_n)))


def _side_of(ms, team):
    return "near" if ms.near == team else "far" if ms.far == team else None


def cmd_pov(a):
    import cv2
    from .pov.core import analyse, reconstruct, render_pov, render_topdown
    sets, tracks, infos, fps = _load_sets([a.dir])
    ms, tr = sets[0], tracks[0]
    side = _side_of(ms, a.team)
    if side is None:
        print(f"No team called {a.team!r} (teams: {ms.near}, {ms.far})")
        return
    lost = [r for r in ms.rallies if r.winner is not None and r.winner != side]
    if a.rally is None:
        print(f"Points lost by {a.team}:")
        for r in lost:
            scn, why = reconstruct(r, tr, side, fps)
            print(f"  rally {r.id} at {r.t_start:.0f}s: {r.reason}" + ("" if scn else f"  (no replay: {why})"))
        return
    r = next((r for r in ms.rallies if r.id == a.rally), None)
    if r is None:
        print(f"No rally {a.rally}")
        return
    scn, why = reconstruct(r, tr, side, fps)
    if scn is None:
        print("No replay for this rally:", why)
        return
    an = analyse(scn)
    os.makedirs(a.out, exist_ok=True)
    for k, d in enumerate(scn.defenders):
        cv2.imwrite(os.path.join(a.out, f"rally{r.id}_{d['label'].strip('#')}.png"), render_pov(scn, k, analysis=an))
    cv2.imwrite(os.path.join(a.out, f"rally{r.id}_topdown.png"), render_topdown(scn, an))
    json.dump({"scenario": scn.to_json(), "analysis": json.loads(json.dumps(an, default=float))},
              open(os.path.join(a.out, f"rally{r.id}_scenario.json"), "w"), indent=1)
    print(an["diagnosis"])
    for n in scn.notes:
        print("note:", n)
    print(f"{len(scn.defenders)} first-person views + top-down diagram + scenario file written to {a.out}/")


def cmd_simulate(a):
    import cv2
    from .pov.core import (Scenario, analyse, render_topdown, score_placement, shots_from_scenario,
                           suggest_placement)
    d = json.load(open(a.scenario))["scenario"]
    scn = Scenario(**d)
    shots = shots_from_scenario(scn)
    actual = score_placement([p["xy"] for p in scn.defenders], shots, scn.side)
    print(f"Scenario: rally {scn.rally}, {scn.shot or 'ball'} landing at ({scn.landing[0]:.1f}, {scn.landing[1]:.1f}), "
          f"flight {scn.T:.2f} s. Court is 9 x 18 m, net at y = 9; you defend the {'y < 9' if scn.side == 'near' else 'y > 9'} half.")
    if actual["score"] is not None:
        print(f"What the team actually did scores {actual['score']}.")
    if a.place:
        pos = [[float(v) for v in p.split(",")] for p in a.place.split(";") if p.strip()]
        res = score_placement(pos, shots, scn.side)
        if res["score"] is None:
            print("Placement not valid:", "; ".join(res["problems"]))
            return
        print(f"Your placement scores {res['score']} (reached {res['reached']} of {res['n_shots']} balls).")
        for s in res["per_shot"]:
            print(f"  ball at ({s['landing'][0]:.1f}, {s['landing'][1]:.1f}): player {s['nearest_player']} is {s['dist']:.1f} m away, "
                  + ("reachable" if s["shortfall"] == 0 else f"{s['shortfall']:.1f} m short"))
        if a.out:
            cv2.imwrite(a.out, render_topdown(scn, analyse(scn), placement=pos))
            print("Diagram written to", a.out)
    else:
        best, sc = suggest_placement(shots, scn.side, len(scn.defenders) or 6)
        print(f"A placement that scores {sc}: " + "; ".join(f"{p[0]:.1f},{p[1]:.1f}" for p in best))
        print('Try your own with  --place "x,y;x,y;x,y;x,y;x,y;x,y"')


def cmd_calibrate(a):
    """Click court points in one frame. Interactive; NOT tested in the build
    environment (it had no display)."""
    import cv2
    cap = cv2.VideoCapture(a.video)
    cap.set(cv2.CAP_PROP_POS_FRAMES, a.frame)
    ok, frame = cap.read()
    if not ok:
        print("cannot read that frame")
        return
    if a.export:
        cv2.imwrite(a.export, frame)
        print(f"Frame saved to {a.export}. Read pixel coordinates off it in any image viewer and fill in a file like:")
        print(json.dumps({"frame": a.frame, "points": [{"image": [0, 0], "court": [0, 0]}, {"image": [0, 0], "court": [9, 0]},
                                                         {"image": [0, 0], "court": [9, 18]}, {"image": [0, 0], "court": [0, 18]}]}, indent=1))
        return
    names = [("near-left corner", (0, 0)), ("near-right corner", (9, 0)), ("far-right corner", (9, 18)), ("far-left corner", (0, 18)),
             ("left end of centre line", (0, 9)), ("right end of centre line", (9, 9))]
    pts = []

    def click(evt, x, y, *_):
        if evt == cv2.EVENT_LBUTTONDOWN and len(pts) < len(names):
            pts.append((x, y))
    cv2.namedWindow("calibrate")
    cv2.setMouseCallback("calibrate", click)
    while True:
        vis = frame.copy()
        for p in pts:
            cv2.circle(vis, p, 5, (0, 0, 255), -1)
        msg = f"click: {names[len(pts)][0]}  (s = skip, q = done)" if len(pts) < len(names) else "q = done"
        cv2.putText(vis, msg, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
        cv2.imshow("calibrate", vis)
        k = cv2.waitKey(30) & 0xFF
        if k == ord("q"):
            break
        if k == ord("s") and len(pts) < len(names):
            pts.append(None)
    out = {"frame": a.frame, "points": [{"image": list(p), "court": list(c)} for p, (_, c) in zip(pts, names) if p is not None]}
    if len(out["points"]) < 4:
        print("need at least 4 points")
        return
    json.dump(out, open(a.out, "w"), indent=1)
    print("written", a.out)


def cmd_demo(a):
    """Generate a synthetic clip and run the whole pipeline on it."""
    from .synth.render import end_camera, make_clip
    from .synth.world import World
    os.makedirs(a.out, exist_ok=True)
    w = World(seed=a.seed, n_rallies=a.rallies)
    video = os.path.join(a.out, "demo.mp4")
    print(f"Rendering a {w.duration:.0f}-second synthetic clip ...")
    make_clip(w, lambda t: end_camera(pan=2.0 * np.sin(0.5 * t), zoom=1.0 + 0.15 * np.sin(0.3 * t)), path=video)
    ns = argparse.Namespace(video=video, out=os.path.join(a.out, "analysis"), near="Home", far="Away", calibration=None,
                            corrections=None, width=960, detector="background", torch_weights=None, jersey_backend="template")
    cmd_analyze(ns)
    print("\nThis is generated footage: it shows the program runs, not how accurate it is on a real match.")


def main(argv=None):
    p = argparse.ArgumentParser(prog="vball", description="Offline volleyball match analysis from one camera.")
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("analyze", help="analyse one video (one set per video is simplest)")
    s.add_argument("video"); s.add_argument("--out", required=True)
    s.add_argument("--near", default="Home", help="name of the team on the camera's side of the net")
    s.add_argument("--far", default="Away", help="name of the team on the far side")
    s.add_argument("--calibration", help="manual court calibration json (see `calibrate`)")
    s.add_argument("--static-camera", action="store_true",
                   help="the camera never moves: trust the manual calibration for every frame")
    s.add_argument("--corrections", help="corrections yaml (numbers / merge / ignore)")
    s.add_argument("--width", type=int, default=960, help="processing width in pixels")
    s.add_argument("--detector", default="background", choices=["background", "hog", "torch"])
    s.add_argument("--torch-weights"); s.add_argument("--jersey-backend", default="template", choices=["template", "tesseract"])
    s.set_defaults(fn=cmd_analyze)
    s = sub.add_parser("relabel", help="apply a corrections file and re-score, without re-reading the video")
    s.add_argument("dir"); s.add_argument("--corrections", required=True); s.set_defaults(fn=cmd_relabel)
    s = sub.add_parser("import-tags", help="turn a hand-tagged file from tagger/tagger.html into an analysis folder")
    s.add_argument("tags"); s.add_argument("--out", required=True); s.set_defaults(fn=cmd_import_tags)
    s = sub.add_parser("player", help="stats, tendencies and notes for one jersey number")
    s.add_argument("dirs", nargs="+"); s.add_argument("--team", required=True); s.add_argument("--number", type=int, required=True)
    s.add_argument("--pdf"); s.set_defaults(fn=cmd_player)
    s = sub.add_parser("team", help="team report")
    s.add_argument("dirs", nargs="+"); s.add_argument("--team", required=True); s.add_argument("--opponent"); s.add_argument("--pdf")
    s.set_defaults(fn=cmd_team)
    s = sub.add_parser("tendencies", help="print the tendency report")
    s.add_argument("dirs", nargs="+"); s.add_argument("--team", required=True); s.add_argument("--opponent")
    s.add_argument("--min-n", type=int, default=8); s.set_defaults(fn=cmd_tendencies)
    s = sub.add_parser("pov", help="first-person replay of a lost point (omit --rally to list them)")
    s.add_argument("dir"); s.add_argument("--team", required=True); s.add_argument("--rally", type=int); s.add_argument("--out", default="pov")
    s.set_defaults(fn=cmd_pov)
    s = sub.add_parser("simulate", help="score a defensive placement against a saved scenario")
    s.add_argument("scenario"); s.add_argument("--place"); s.add_argument("--out"); s.set_defaults(fn=cmd_simulate)
    s = sub.add_parser("calibrate", help="mark the court by hand in one frame")
    s.add_argument("video"); s.add_argument("--frame", type=int, default=0); s.add_argument("--out", default="calibration.json")
    s.add_argument("--export", help="just save the frame as an image instead of opening a window"); s.set_defaults(fn=cmd_calibrate)
    s = sub.add_parser("demo", help="render a synthetic clip and analyse it (checks the install)")
    s.add_argument("--out", default="demo_out"); s.add_argument("--seed", type=int, default=1); s.add_argument("--rallies", type=int, default=4)
    s.set_defaults(fn=cmd_demo)
    a = p.parse_args(argv)
    a.fn(a)


if __name__ == "__main__":
    main()
