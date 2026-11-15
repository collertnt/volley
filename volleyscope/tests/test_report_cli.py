import json
import os

import numpy as np
import pytest

from test_stats import simple_match
from vball import cli
from vball.ball.events import build_rallies
from vball.pipeline import load_analysis, save_analysis
from vball.report.pdf import player_pdf, shot_chart_png, team_pdf
from vball.stats.core import MatchSet, compute_stats
from vball.synth.evaluate import truth_flights, truth_tracks
from vball.synth.world import World


def pdf_text(path):
    from pypdf import PdfReader
    return "\n".join(p.extract_text() for p in PdfReader(str(path)).pages)


@pytest.fixture(scope="module")
def sim():
    w = World(seed=4, n_rallies=30)
    times = np.arange(0, w.duration, 1 / 30)
    tracks = truth_tracks(w, times)
    return w, MatchSet(build_rallies(truth_flights(w), tracks, 30), "Home", "Away"), tracks


def test_player_pdf_contains_the_numbers(tmp_path, sim):
    pytest.importorskip("pypdf")
    w, ms, tracks = sim
    num = w.players[3].number
    stats = compute_stats([ms])
    p = player_pdf(tmp_path / "p.pdf", [ms], "Home", num, stats=stats)
    txt = pdf_text(tmp_path / "p.pdf")
    assert f"Home #{num}" in txt and "Kill %" in txt and "By rotation" in txt
    k = p["stats"]["kill_pct"]
    assert f"{k.num:g} / {k.den}" in txt
    assert "What the numbers suggest" in txt and "How to read this sheet" in txt


def test_player_pdf_for_unknown_number_says_so(tmp_path, sim):
    pytest.importorskip("pypdf")
    w, ms, _ = sim
    assert player_pdf(tmp_path / "x.pdf", [ms], "Home", 99) is None
    assert "No contacts were attributed to #99" in pdf_text(tmp_path / "x.pdf")


def test_small_sample_pdf_shows_no_data_not_zero(tmp_path):
    pytest.importorskip("pypdf")
    ms = simple_match()
    player_pdf(tmp_path / "s.pdf", [ms], "Away", 8)
    txt = pdf_text(tmp_path / "s.pdf")
    assert "no data" in txt and "low sample" in txt
    assert "insufficient data" in txt


def test_team_pdf(tmp_path, sim):
    pytest.importorskip("pypdf")
    w, ms, tracks = sim
    T = team_pdf(tmp_path / "t.pdf", [ms], "Home", "Away", [tracks], 30)
    txt = pdf_text(tmp_path / "t.pdf")
    assert "Home vs Away" in txt and "Points by rotation" in txt and "Every point lost" in txt
    assert f"How the {T['points_lost']} lost points were lost" in txt
    assert "Lost to tips" in txt and "Side-out %" in txt
    assert os.path.getsize(tmp_path / "t.pdf") > 3000


def test_shot_chart(sim):
    w, ms, _ = sim
    buf, n = shot_chart_png([ms], "Home")
    assert n > 5 and buf.getvalue()[:4] == b"\x89PNG"
    assert shot_chart_png([ms], "Home", 99) == (None, 0)


def _saved(tmp_path, sim, name="set1"):
    w, ms, tracks = sim
    res = {"tracks": tracks, "rallies": ms.rallies, "balls": [[] for _ in range(10)], "states": [],
           "info": {"fps": 30.0, "size": [960, 540], "frames": 10, "court_usable_share": 1.0, "n_rallies": len(ms.rallies),
                    "court_status": {"lines": 10}, "near": "Home", "far": "Away"}}
    d = tmp_path / name
    save_analysis(res, d)
    return d


def test_save_and_load_roundtrip(tmp_path, sim):
    d = _saved(tmp_path, sim)
    back = load_analysis(d)
    assert len(back["rallies"]) == len(sim[1].rallies) and back["info"]["near"] == "Home"
    assert len(back["tracks"].identities) == 12


def test_cli_player_team_tendencies(tmp_path, sim, capsys):
    pytest.importorskip("pypdf")
    d = _saved(tmp_path, sim)
    num = sim[0].players[3].number
    cli.main(["player", str(d), "--team", "Home", "--number", str(num), "--pdf", str(tmp_path / "me.pdf")])
    out = capsys.readouterr().out
    assert f"Home #{num}" in out and "Kill %" in out and "What the numbers suggest" in out
    assert (tmp_path / "me.pdf").exists()
    cli.main(["player", str(d), "--team", "Home", "--number", "99"])
    assert "No contacts attributed" in capsys.readouterr().out
    cli.main(["team", str(d), "--team", "Home", "--pdf", str(tmp_path / "team.pdf")])
    out = capsys.readouterr().out
    assert "Side-out %" in out and (tmp_path / "team.pdf").exists()
    cli.main(["team", str(d), "--team", "Nobody"])
    assert "No team called" in capsys.readouterr().out
    cli.main(["tendencies", str(d), "--team", "Home"])
    assert "TENDENCY REPORT - Home" in capsys.readouterr().out


def test_cli_two_sets_are_added_up(tmp_path, sim, capsys):
    a, b = _saved(tmp_path, sim, "s1"), _saved(tmp_path, sim, "s2")
    cli.main(["team", str(a), str(b), "--team", "Home"])
    out = capsys.readouterr().out
    one = compute_stats([sim[1]])["teams"]["Home"]["points_lost"]
    assert f"points lost: {2 * one}" in out


def test_cli_pov_and_simulate(tmp_path, sim, capsys):
    d = _saved(tmp_path, sim)
    cli.main(["pov", str(d), "--team", "Home"])
    listing = capsys.readouterr().out
    assert "Points lost by Home" in listing
    rid = next(int(l.split()[1]) for l in listing.splitlines() if l.strip().startswith("rally") and "no replay" not in l)
    out_dir = tmp_path / "pov"
    cli.main(["pov", str(d), "--team", "Home", "--rally", str(rid), "--out", str(out_dir)])
    out = capsys.readouterr().out
    assert "first-person views" in out
    files = os.listdir(out_dir)
    assert f"rally{rid}_topdown.png" in files and sum(f.endswith(".png") for f in files) == 7
    scen = out_dir / f"rally{rid}_scenario.json"
    cli.main(["simulate", str(scen)])
    assert "A placement that scores" in capsys.readouterr().out
    landing = json.load(open(scen))["scenario"]["landing"]
    cli.main(["simulate", str(scen), "--place", f"{landing[0]},{landing[1]}", "--out", str(tmp_path / "sim.png")])
    out = capsys.readouterr().out
    assert "Your placement scores 100.0" in out and (tmp_path / "sim.png").exists()
    cli.main(["simulate", str(scen), "--place", "4.5,15"])
    assert "not valid" in capsys.readouterr().out


@pytest.mark.slow
def test_cli_analyze_relabel_and_reports_on_video(match, tmp_path, capsys):
    """The real command-line path on a rendered video file, start to finish."""
    pytest.importorskip("pypdf")
    from vball.pipeline import save_analysis
    from vball.review import review_sheet
    from vball.synth.evaluate import truth_corrections
    import yaml
    res = {"tracks": match["tracks"], "rallies": match["rallies_raw"], "balls": match["balls"], "states": match["states"],
           "tracks_raw": match["tracks"].to_json(),
           "info": {"fps": match["src"].fps, "size": list(match["src"].size), "frames": len(match["states"]),
                    "court_usable_share": 1.0, "n_rallies": len(match["rallies_raw"]), "court_status": {}, "near": "Home", "far": "Away"}}
    d = tmp_path / "set1"
    save_analysis(res, d)
    sheet = review_sheet(str(match["path"]), match["tracks"])
    assert sheet is not None and sheet.shape[0] >= 190
    corr = tmp_path / "corr.yaml"
    corr.write_text(yaml.safe_dump(truth_corrections(match["world"], match["times"], match["tracks"])))
    cli.main(["relabel", str(d), "--corrections", str(corr)])
    assert "re-scored" in capsys.readouterr().out
    num = match["world"].players[0].number
    cli.main(["player", str(d), "--team", "Home", "--number", str(num), "--pdf", str(tmp_path / "me.pdf")])
    out = capsys.readouterr().out
    print(out)
    assert "Serve in %" in out
    cli.main(["team", str(d), "--team", "Home", "--pdf", str(tmp_path / "team.pdf")])
    assert "Home vs Away" in pdf_text(tmp_path / "team.pdf")
