import cv2
import numpy as np
import pytest

from vball.court.track import CourtState
from vball.players.detect import (BackgroundModel, Detection, TeamColours, detect_from_mask, detect_hog,
                                  hist_sim, local_jacobian, metric_scale)
from vball.players.jersey import (ALL_FONTS, NumberVotes, classify_digit, read_number, render_digit)
from vball.players.track import (PlayerTracker, PlayerTracks, apply_corrections, build_identities,
                                 load_corrections, suppress_jumps)
from vball.synth.evaluate import evaluate_tracking, truth_corrections
from vball.synth.render import end_camera, render
from vball.synth.world import World


# ---------------------------------------------------------------- detection
@pytest.fixture(scope="module")
def static_clip():
    w = World(seed=1, n_rallies=5)                 # long enough that everyone moves and rotates
    cam = end_camera()
    times = np.arange(0, w.duration, 1 / 4)
    frames = [render(w, cam, t) for t in times]
    states = [CourtState(cam.H, 1.0, "lines") for _ in frames]
    bg = BackgroundModel().build(frames, states)
    return w, cam, times, frames, states, bg


def test_metric_scale_matches_camera():
    cam = end_camera()
    for xy in [(4.5, 2.0), (2.0, 16.0)]:
        d = cam.cam([(xy[0], xy[1], 0)])[0, 2]
        assert metric_scale(cam.H, np.array(xy)) == pytest.approx(cam.f / d, rel=0.08)
    assert local_jacobian(cam.H, np.array([4.5, 9.0])).shape == (2, 2)


def test_background_has_no_player_ghosts(static_clip):
    """Players stand at their base spots much of the time. If they leaked into
    the background, an empty court would light up as foreground there."""
    w, cam, times, frames, states, bg = static_clip
    empty = render(w, cam, 0.0, players=False, ball=False)
    fg = bg.foreground(empty, cam.H)
    dets, _ = detect_from_mask(empty, fg, cam.H)
    print("empty-court foreground fraction", (fg > 0).mean(), "false players", len(dets))
    assert (fg > 0).mean() < 0.004
    assert len(dets) <= 2


def test_detection_recall_and_position(static_clip):
    w, cam, times, frames, states, bg = static_clip
    col = TeamColours()
    for f in frames[::6]:
        col.add(detect_from_mask(f, bg.foreground(f, cam.H), cam.H)[0])
    col.finish()
    assert col.distinct
    found = total = 0
    errs = []
    for f, t in zip(frames, times):
        dets, _ = detect_from_mask(f, bg.foreground(f, cam.H), cam.H, colours=col)
        truth = np.array([w.player_pos(p.pid, t)[:2] for p in w.players])
        D = np.linalg.norm(truth[:, None] - np.array([d.xy for d in dets])[None], axis=2)
        found += (D.min(1) < 1.0).sum(); total += 12
        errs += list(D.min(1)[D.min(1) < 1.0])
    assert found / total > 0.9
    assert np.mean(errs) < 0.3


def test_team_from_jersey_colour(static_clip):
    w, cam, times, frames, states, bg = static_clip
    col = TeamColours()
    for f in frames[::6]:
        col.add(detect_from_mask(f, bg.foreground(f, cam.H), cam.H)[0])
    col.finish()
    dets, _ = detect_from_mask(frames[3], bg.foreground(frames[3], cam.H), cam.H, colours=col)
    labelled = [d for d in dets if d.team]
    assert len(labelled) >= 8
    t = times[3]
    right = 0
    for d in labelled:
        true = min(w.players, key=lambda p: np.linalg.norm(w.player_pos(p.pid, t)[:2] - d.xy))
        right += d.team == true.team
    assert right / len(labelled) >= 0.9


def test_ball_candidate_found(static_clip):
    w, cam, times, frames, states, bg = static_clip
    hits = n = 0
    for f, t in zip(frames, times):
        b = w.ball(t)
        if b is None:
            continue
        uv = cam.project([b])[0][0]
        _, balls = detect_from_mask(f, bg.foreground(f, cam.H), cam.H)
        n += 1
        hits += any(np.hypot(u - uv[0], v - uv[1]) < 4 for u, v, _ in balls)
    assert n > 10 and hits / n > 0.7


def test_empty_mask_gives_nothing():
    cam = end_camera()
    dets, balls = detect_from_mask(np.zeros((540, 960, 3), np.uint8), np.zeros((540, 960), np.uint8), cam.H)
    assert dets == [] and balls == []


def test_hog_detector_smoke(static_clip):
    w, cam, times, frames, states, bg = static_clip
    assert isinstance(detect_hog(frames[0], cam.H), list)      # runs; accuracy not asserted


def test_hist_sim_bounds():
    a = np.r_[np.eye(19)[0], np.eye(19)[1], np.eye(19)[2]]
    b = np.r_[np.eye(19)[5], np.eye(19)[6], np.eye(19)[7]]
    assert hist_sim(a, a) == pytest.approx(1.0) and hist_sim(a, b) == 0.0


# ---------------------------------------------------------------- jersey
def _jersey(number, digit_px, colour=(140, 50, 20), font=cv2.FONT_HERSHEY_SIMPLEX, noise=4, seed=0):
    """A player-shaped patch: bbox (0, 0, w, h) with the number on the torso."""
    h = int(digit_px / 0.2)
    w = int(0.35 * h)
    img = np.full((h, w, 3), (70, 120, 180), np.uint8)
    cv2.rectangle(img, (int(0.14 * w), int(0.18 * h)), (int(0.86 * w), int(0.5 * h)), colour, -1)
    scale = digit_px / 22
    th = max(1, round(digit_px / 7))
    (tw, tH), _ = cv2.getTextSize(str(number), font, scale, th)
    if tw > 0.6 * w:
        scale *= 0.6 * w / tw
        th = max(1, round(22 * scale / 7))
        (tw, tH), _ = cv2.getTextSize(str(number), font, scale, th)
    cv2.putText(img, str(number), ((w - tw) // 2, int(0.34 * h + tH / 2)), font, scale, (245, 245, 245), th, cv2.LINE_AA)
    rng = np.random.default_rng(seed)
    return np.clip(img + rng.normal(0, noise, img.shape), 0, 255).astype(np.uint8), (0, 0, w, h)


def test_digit_classifier_on_training_fonts():
    for font in ALL_FONTS[:4]:
        for d in range(10):
            assert classify_digit(render_digit(d, font, 5))[0] == d


def test_digit_classifier_on_unseen_font():
    """Each font is held out of the templates in turn."""
    right = total = 0
    for font in ALL_FONTS:
        for d in range(10):
            right += classify_digit(render_digit(d, font, 5), exclude=(font,))[0] == d
            total += 1
    assert right / total >= 0.9


@pytest.mark.parametrize("px", [18, 26])
def test_read_number_when_legible(px):
    right = wrong = 0
    numbers = [1, 4, 7, 9, 3, 5, 8, 2, 6, 0] if px < 20 else [1, 4, 7, 9, 10, 12, 23, 35, 68, 99]
    for k, n in enumerate(numbers):
        img, bb = _jersey(n, px, seed=k)
        got, conf = read_number(img, bb)
        right += got == n
        wrong += got is not None and got != n
    assert right >= (6 if px < 20 else 8) and wrong == 0      # declining is allowed, guessing is not


def test_read_number_declines_when_too_small():
    for n in (7, 23):
        img, bb = _jersey(n, 7)
        assert read_number(img, bb) == (None, 0.0)


def test_read_number_plain_jersey_is_none():
    img, bb = _jersey(8, 20)
    cv2.rectangle(img, (0, 0), (bb[2], bb[3]), (140, 50, 20), -1)
    assert read_number(img, bb)[0] is None


def test_number_votes():
    v = NumberVotes()
    assert v.result() == (None, 0.0)
    for _ in range(7):
        v.add(12, 0.8)
    v.add(2, 0.5); v.add(None, 0)
    assert v.result()[0] == 12
    split = NumberVotes()
    for n in (3, 8, 3, 8, 3, 8):
        split.add(n, 0.7)
    assert split.result()[0] is None            # no majority: no number
    half = NumberVotes()
    for n in (2, 2, 2, 2, 23, 23, 23):
        half.add(n, 0.7)
    assert half.result()[0] == 23               # half-hidden two-digit number


def test_tesseract_backend_if_installed():
    pytest.importorskip("pytesseract")
    import shutil
    if not shutil.which("tesseract"):
        pytest.skip("tesseract binary not installed")
    img, bb = _jersey(7, 30)
    got, _ = read_number(img, bb, backend="tesseract")
    assert got in (7, None)


# ---------------------------------------------------------------- tracker
def det(x, y, hue=0, occluded=False, team=None):
    h = np.zeros(57)
    h[hue] = h[19 + hue] = h[38 + hue] = 1
    return Detection((0, 0, 20, 60), (0, 0), np.array([x, y], float), 30.0, h, np.eye(2) * 0.03, team, occluded)


def run_tracker(paths, n, fps=30, drop=lambda k, f: False, **kw):
    """paths: list of functions frame -> (x, y, hue)."""
    tr = PlayerTracker(fps)
    for f in range(n):
        dets = [det(*p(f), **kw) for k, p in enumerate(paths) if not drop(k, f)]
        tr.update(f, dets)
    return tr.finish()


def test_two_players_crossing_keep_identity():
    a = lambda f: (2 + 0.05 * f, 3.0, 2)
    b = lambda f: (7 - 0.05 * f, 3.6, 2)
    tracks = run_tracker([a, b], 100)
    assert len(tracks.identities) == 2
    for ident in tracks.identities.values():
        going_right = ident.xy[-1][0] > ident.xy[0][0]
        assert np.all(np.diff(ident.xy[:, 0]) * (1 if going_right else -1) > -0.05)


def test_reidentified_after_leaving_frame():
    a = lambda f: (3.0 + 0.01 * f, 4.0, 2)
    b = lambda f: (7.0, 5.0, 9)
    tracks = run_tracker([a, b], 200, drop=lambda k, f: k == 0 and 60 <= f < 150)
    assert len(tracks.identities) == 2
    longest = max(tracks.identities.values(), key=lambda i: i.frames[-1] - i.frames[0])
    assert longest.frames[0] < 10 and longest.frames[-1] > 190


def test_short_gap_is_interpolated_and_marked():
    a = lambda f: (3.0 + 0.02 * f, 4.0, 2)
    tracks = run_tracker([a], 80, drop=lambda k, f: 40 <= f < 48)
    ident, = tracks.identities.values()
    i = list(ident.frames).index(44)
    assert not ident.observed[i] and ident.xy[i][0] == pytest.approx(3.0 + 0.02 * 44, abs=0.15)


def test_player_never_switches_sides():
    a = lambda f: (4.0, 7.6, 2)
    b = lambda f: (4.0, 10.4, 2)
    tracks = run_tracker([a, b], 60)
    teams = sorted(i.team for i in tracks.identities.values())
    assert teams == ["far", "near"]


def test_seventh_player_on_a_side_is_not_confirmed():
    paths = [(lambda f, k=k: (1 + 1.2 * k, 3.0, 2)) for k in range(7)]
    tracks = run_tracker(paths, 60, drop=lambda k, f: k == 6 and f < 20, team="near")
    assert len(tracks.identities) == 6


def test_false_positive_flicker_makes_no_identity():
    a = lambda f: (3.0, 4.0, 2)
    b = lambda f: (6.0, 12.0, 2)
    tracks = run_tracker([a, b], 60, drop=lambda k, f: k == 1 and f % 10 > 1)
    assert len(tracks.identities) == 1


def test_suppress_jumps():
    fps = 30
    n = 90
    xy = np.tile([4.0, 7.0], (n, 1)).astype(float)
    rays = np.tile([0.0, 1.0], (n, 1))
    t = np.arange(n)
    hop = np.clip(1 - ((t - 45) / 9.0) ** 2, 0, None) * 1.4      # 0.6 s excursion along the ray
    xy[:, 1] += hop
    out, air = suppress_jumps(xy, rays, fps)
    assert np.abs(out[:, 1] - 7.0).max() < 0.5 and air[45] and not air[10]
    walk = np.c_[np.full(n, 4.0), 2 + 0.05 * t]                   # a real run is kept
    out2, air2 = suppress_jumps(walk, rays, fps)
    assert np.allclose(out2, walk, atol=0.5) and air2.sum() == 0


def test_same_number_merges_segments_and_conflicts_are_flagged():
    from vball.players.track import Segment
    def seg(f0, f1, x, number, ident, team="near"):
        s = Segment(det(x, 3.0, team=team), f0, 1 / 30)
        for f in range(f0 + 1, f1):
            s.update(det(x, 3.0, team=team), f)
        for _ in range(8):
            s.votes.add(number, 0.9)
        s.identity, s.status = ident, "active"
        return s
    tracks = build_identities([seg(0, 50, 2, 7, 0), seg(300, 350, 6, 7, 1)], 30)
    assert len(tracks.identities) == 1 and tracks.by_number("near", 7)
    clash = build_identities([seg(0, 50, 2, 7, 0), seg(10, 60, 6, 7, 1)], 30)
    nums = [i.number for i in clash.identities.values()]
    assert sorted(nums, key=str) == [7, None]
    assert any(i.flags for i in clash.identities.values())
    split = build_identities([seg(0, 50, 2, 7, 0), seg(60, 100, 2, 9, 0)], 30)     # one chain, two numbers
    assert sorted(i.number for i in split.identities.values()) == [7, 9]


def test_corrections_file(tmp_path):
    a = lambda f: (3.0, 4.0, 2)
    b = lambda f: (7.0, 5.0, 9)
    c = lambda f: (-2.0, 9.0, 12)
    tracks = run_tracker([a, b, c], 120, drop=lambda k, f: k == 0 and 40 <= f < 110 and False)
    ids = sorted(tracks.identities)
    p = tmp_path / "corr.yaml"
    p.write_text(f"numbers: {{{ids[0]}: 7}}\nteams: {{{ids[0]}: far}}\nmerge: [[{ids[0]}, {ids[1]}]]\nignore: [{ids[2]}]\n")
    out = apply_corrections(tracks, load_corrections(p))
    assert list(out.identities) == [ids[0]]
    me = out.identities[ids[0]]
    assert me.number == 7 and me.number_source == "manual" and me.team == "far"
    assert apply_corrections(out, {}).identities.keys() == out.identities.keys()


def test_tracks_json_roundtrip():
    tracks = run_tracker([lambda f: (3.0 + 0.01 * f, 4.0, 2)], 40)
    back = PlayerTracks.from_json(tracks.to_json())
    a, b = list(tracks.identities.values())[0], list(back.identities.values())[0]
    assert np.allclose(a.xy, b.xy, atol=1e-3) and (a.frames == b.frames).all() and a.team == b.team


# ---------------------------------------------------------------- end to end
@pytest.mark.slow
def test_end_to_end_tracking_on_rendered_match(match):
    ev = evaluate_tracking(match["world"], match["times"], match["tracks"])
    print("\nTRACKING", {k: v for k, v in ev.items() if k != "main_player"})
    assert ev["coverage"] > 0.93
    assert ev["mean_err"] < 0.3
    assert ev["purity"] > 0.85
    assert ev["numbers"]["wrong"] == 0            # may be unread, must not be wrong
    assert ev["numbers"]["right"] >= 2
    assert 12 <= ev["n_identities"] <= 18


@pytest.mark.slow
def test_corrections_complete_the_numbers(match):
    import copy
    tracks = copy.deepcopy(match["tracks"])
    corr = truth_corrections(match["world"], match["times"], tracks)
    apply_corrections(tracks, corr)
    ev = evaluate_tracking(match["world"], match["times"], tracks)
    assert ev["numbers"]["unread"] == 0 and ev["numbers"]["wrong"] == 0
