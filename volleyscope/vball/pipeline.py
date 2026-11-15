"""Analysis passes over a video. Each pass re-reads the source; nothing but
per-frame numbers is kept in memory."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .ball.events import EventConfig
from .ball.track import BallConfig
from .court.detect import CourtConfig
from .court.track import CourtTracker, load_calibration
from .players.detect import (BackgroundModel, BallColour, DetectConfig, TeamColours, detect_from_mask, detect_hog,
                             detect_torch)
from .players.track import PlayerTracker, TrackConfig
from .video import FrameSource


@dataclass
class Config:
    court: CourtConfig = field(default_factory=CourtConfig)
    detect: DetectConfig = field(default_factory=DetectConfig)
    track: TrackConfig = field(default_factory=TrackConfig)
    ball: BallConfig = field(default_factory=BallConfig)
    events: EventConfig = field(default_factory=EventConfig)
    proc_width: int = 960
    calibration: str | None = None       # manual court calibration json
    static_camera: bool = False          # camera never moves: use the calibration for every frame


def run_court(source: FrameSource, cfg: Config, progress=None):
    """Pass 1: a CourtState per frame."""
    tr = CourtTracker(cfg.court)
    init_frame, init_H = -1, None
    if cfg.calibration:
        init_frame, init_H = load_calibration(cfg.calibration, scale=source.scale)
    if cfg.static_camera:
        if init_H is None:
            raise ValueError("static_camera needs a manual calibration file")
        from .court.track import CourtState
        return [CourtState(init_H, 0.9, "manual") for _ in range(len(source))] if len(source) else \
            [CourtState(init_H, 0.9, "manual") for _ in source]
    states = []
    for i, f in enumerate(source):
        if i == init_frame:
            tr.H = init_H
        states.append(tr.update(f))
        if progress and i % 200 == 0:
            progress("court", i, len(source))
    return states


def run_players(source: FrameSource, states, cfg: Config, progress=None):
    """Pass 2: player tracks and per-frame ball candidates.

    Returns (PlayerTracks, ball_candidates[list per frame of (u, v, r)], info)."""
    info = {"detector": cfg.detect.detector}
    bg = colours = ball_colour = None
    if cfg.detect.detector == "background":
        bg = BackgroundModel(cfg.detect).build(source, states)
        colours, ball_colour = TeamColours(), BallColour()
        usable = [i for i, s in enumerate(states) if s.H is not None]
        pick = set(usable[::max(1, len(usable) // 40)])
        for i, f in enumerate(source):
            if i in pick:
                m = bg.foreground(f, states[i].H)
                pl, bl = detect_from_mask(f, m, states[i].H, cfg.detect)
                colours.add(pl)
                ball_colour.add(f, m, pl, bl)
        colours.finish()
        ball_colour.finish()
        info["ball_colour_learned"] = bool(ball_colour.bins)
        info["team_colours_distinct"] = bool(colours.distinct)
    tracker = PlayerTracker(source.fps, cfg.track)
    balls = []
    for i, f in enumerate(source):
        st = states[i] if i < len(states) else None
        if st is None or st.H is None:
            tracker.update(i, [], f)
            balls.append([])
            continue
        if bg is not None:
            dets, bc = detect_from_mask(f, bg.foreground(f, st.H), st.H, cfg.detect, colours, ball_colour)
        elif cfg.detect.detector == "hog":
            dets, bc = detect_hog(f, st.H, cfg.detect), []
        else:
            dets, bc = detect_torch(f, st.H, cfg.detect), []
        tracker.update(i, dets, f)
        balls.append(bc)
        if progress and i % 200 == 0:
            progress("players", i, len(source))
    return tracker.finish(), balls, info


def run_ball(balls, states, tracks, size, fps, cfg: "Config"):
    """Pass 3 (no video needed): flights, then rallies with classified contacts."""
    from .ball.events import build_rallies
    from .ball.track import track_ball
    flights, _ = track_ball(balls, states, size, fps, cfg.ball)
    return flights, build_rallies(flights, tracks, fps, cfg.events)


def analyze(src, cfg: "Config | None" = None, corrections: dict | None = None, fps=None, progress=None):
    """Run every pass on a video path (or a list of frames).

    Returns a dict with states, tracks, flights, rallies and a summary of how
    much of the footage could actually be used."""
    from .players.track import apply_corrections
    cfg = cfg or Config()
    source = src if isinstance(src, FrameSource) else FrameSource(src, fps=fps, proc_width=cfg.proc_width)
    states = run_court(source, cfg, progress)
    usable = sum(s.H is not None for s in states)
    if usable < 5:
        raise RuntimeError(
            "The court could not be found in this video. Provide a manual calibration "
            "(see README: 'When the court is not found').")
    tracks, balls, info = run_players(source, states, cfg, progress)
    raw = tracks.to_json()                      # before any manual corrections, so they can be redone
    if corrections:
        apply_corrections(tracks, corrections)
    flights, rallies = run_ball(balls, states, tracks, source.size, source.fps, cfg)
    status = {}
    for s in states:
        status[s.status] = status.get(s.status, 0) + 1
    info.update(frames=len(states), fps=source.fps, size=list(source.size), court_status=status,
                court_usable_share=usable / max(1, len(states)),
                mean_court_conf=float(np.mean([s.conf for s in states])),
                n_flights=len(flights), n_rallies=len(rallies))
    return {"states": states, "tracks": tracks, "balls": balls, "flights": flights, "rallies": rallies,
            "info": info, "fps": source.fps, "size": source.size, "tracks_raw": raw}


def save_analysis(res, out_dir):
    """Write tracks.json, rallies.json, court.json and info.json."""
    import json
    import os
    from .ball.events import rallies_to_json
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "tracks.json"), "w") as fh:
        json.dump(res["tracks"].to_json(), fh)
    with open(os.path.join(out_dir, "rallies.json"), "w") as fh:
        json.dump(rallies_to_json(res["rallies"]), fh, indent=1)
    with open(os.path.join(out_dir, "court.json"), "w") as fh:
        json.dump([{"status": s.status, "conf": round(s.conf, 3),
                    "H": None if s.H is None else np.round(s.H, 6).tolist()} for s in res["states"]], fh)
    with open(os.path.join(out_dir, "balls.json"), "w") as fh:
        json.dump([[[round(float(v), 1) for v in c] for c in b] for b in res["balls"]], fh)
    if res.get("tracks_raw") is not None:
        with open(os.path.join(out_dir, "tracks_raw.json"), "w") as fh:
            json.dump(res["tracks_raw"], fh)
    with open(os.path.join(out_dir, "info.json"), "w") as fh:
        json.dump(res["info"], fh, indent=1)


def load_analysis(out_dir):
    import json
    import os
    from .ball.events import rallies_from_json
    from .players.track import PlayerTracks
    tracks = PlayerTracks.from_json(json.load(open(os.path.join(out_dir, "tracks.json"))))
    rallies = rallies_from_json(json.load(open(os.path.join(out_dir, "rallies.json"))))
    info = json.load(open(os.path.join(out_dir, "info.json")))
    return {"tracks": tracks, "rallies": rallies, "info": info, "fps": info["fps"]}
