import json
import os
import shutil
import subprocess

import pytest

from vball import cli
from vball.stats.core import MatchSet, compute_stats
from vball.tagging import import_tags, rallies_from_tags
from vball.tendencies.core import hitter_tendencies, setter_distribution

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PAGE = os.path.join(ROOT, "tagger", "tagger.html")


def T(t, team, number, typ, **kw):
    return {"t": t, "team": team, "number": number, "type": typ, **kw}


def tags():
    return {"version": 1, "near": "Pakmen", "far": "Flames", "video": "set1.mov", "rallies": [
        {"id": 0, "serve_team": "far", "winner": "near", "touches": [
            T(1, "far", 5, "serve", result="in_play"),
            T(2, "near", 7, "reception", grade=3),
            T(3, "near", 9, "set"),
            T(4, "near", 7, "attack", result="kill", shot="cross", pos=[1.5, 8.2], target=[8.0, 16.5])]},
        {"id": 1, "serve_team": "near", "winner": "far", "touches": [
            T(10, "near", 7, "serve", result="in_play"),
            T(11, "far", 5, "reception", grade=1),
            T(12, "far", 8, "set"),
            T(13, "far", 12, "attack", result="dug", shot="tip", pos=[7.0, 9.8], target=[4.0, 7.0]),
            T(14, "near", 9, "dig", result="ok", grade=2),
            T(15, "near", 4, "set"),
            T(16, "near", 7, "attack", result="blocked", pos=[1.5, 8.2]),
            T(17, "far", 12, "block", result="stuff")]},
        {"id": 2, "serve_team": "far", "winner": "near", "touches": [
            T(20, "far", 12, "serve", result="error")]},
        {"id": 3, "serve_team": "near", "winner": "near", "touches": [
            T(30, "near", 9, "serve", result="ace")]},
        {"id": 4, "serve_team": "near", "winner": "far", "touches": [
            T(40, "near", None, "attack", result="error")]},
    ]}


def test_tags_become_rallies_with_derived_fields():
    rallies, warnings = rallies_from_tags(tags())
    assert warnings == [] and len(rallies) == 5
    r0, r1 = rallies[0], rallies[1]
    assert [e.type for e in r0.events] == ["serve", "reception", "set", "attack"]
    assert r0.reason == "kill" and r0.winner == "near" and r0.serve_team == "far"
    assert r0.events[2].outcome == "assist" and r0.events[2].attrs["target_number"] == 7
    assert r0.events[2].attrs["target_pos"] == [1.5, 8.2]
    assert r0.events[3].attrs["in_system"] is True and r0.events[3].conf == 1.0
    assert r1.reason == "blocked" and r1.events[-2].outcome == "blocked" and r1.events[-1].outcome == "stuff"
    assert r1.events[3].attrs["in_system"] is False          # their pass was graded 1
    assert r1.events[6].attrs["in_system"] is True           # our dig was graded 2
    assert rallies[2].reason == "serve_error" and rallies[3].reason == "ace"
    assert rallies[4].events[0].number is None and "player_number_unknown" in rallies[4].events[0].flags


def test_stats_from_tags_match_a_hand_count():
    rallies, _ = rallies_from_tags(tags())
    s = compute_stats([MatchSet(rallies, "Pakmen", "Flames")])
    me = s["players"][("Pakmen", 7)]["stats"]
    assert me["attack_attempts"].value == 2 and me["kills"].value == 1
    assert me["attack_eff"].value == 0.0                     # 1 kill - 1 blocked over 2
    assert me["reception_avg"].value == 3.0 and me["serve_in_pct"].value == 1.0
    assert s["players"][("Pakmen", 9)]["stats"]["assist_pct"].num == 1
    assert s["players"][("Pakmen", 9)]["stats"]["aces"].value == 1
    assert s["players"][("Flames", 12)]["stats"]["blocks"].value == 1
    assert s["players"][("Pakmen", None)]["stats"]["attack_errors"].value == 1     # unknown number kept apart
    assert s["scores"] == [{"Pakmen": 3, "Flames": 2}]
    team = s["teams"]["Pakmen"]
    assert team["side_out"].num == 2 and team["side_out"].den == 2
    assert team["points_lost"] == 2
    assert [x["cause"] for x in team["lost_points"]] == ["blocked", "attack_error"]
    assert not any(st.uncertain for st in me.values())       # a person's tags are not "low confidence"


def test_tendencies_use_optional_court_clicks():
    rallies, _ = rallies_from_tags(tags())
    ms = MatchSet(rallies, "Pakmen", "Flames")
    h = hitter_tendencies([ms], "Pakmen", 7, min_n=1)
    assert h["result"].n == 2 and h["overall"].n == 1        # one attack had no shot type: not guessed
    assert h["overall"].counts == {"cross": 1} and h["from the left"].n == 1
    d = setter_distribution([ms], "Pakmen", min_n=1)
    assert d["overall"].counts == {"left front": 2}


def test_bad_tags_are_reported_not_guessed():
    bad = {"near": "A", "far": "B", "rallies": [
        {"serve_team": None, "winner": None, "touches": [
            T(1, "near", 3, "reception"),                    # no grade
            T(2, "near", 3, "attack"),                       # no result
            T(3, "near", 3, "spike", result="kill"),         # unknown type
            T(4, "near", 3, "set")]},
        {"serve_team": "near", "winner": "far", "touches": [T(9, "near", 3, "attack", result="kill")]},
        {"serve_team": "near", "winner": None, "touches": []}]}
    rallies, w = rallies_from_tags(bad)
    assert len(rallies) == 2 and [e.type for e in rallies[0].events] == ["set"]
    text = " | ".join(w)
    assert "no grade" in text and "no valid result" in text and "unknown type" in text
    assert "no server recorded" in text and "no point was awarded" in text
    assert "point went to the other" in text
    s = compute_stats([MatchSet(rallies, "A", "B")])
    assert s["teams"]["A"]["unknown_winner"] == 1


def test_import_and_cli_reports(tmp_path, capsys):
    pytest.importorskip("pypdf")
    src = tmp_path / "t.json"
    src.write_text(json.dumps(tags()))
    out = tmp_path / "set1"
    cli.main(["import-tags", str(src), "--out", str(out)])
    assert "Imported 15 tagged touches in 5 rallies" in capsys.readouterr().out
    cli.main(["player", str(out), "--team", "Pakmen", "--number", "7", "--pdf", str(tmp_path / "me.pdf")])
    txt = capsys.readouterr().out
    assert "Pakmen #7" in txt and "Kill %" in txt and "50% (1/2)" in txt
    cli.main(["team", str(out), "--team", "Pakmen", "--pdf", str(tmp_path / "team.pdf")])
    cli.main(["tendencies", str(out), "--team", "Pakmen"])
    assert "TENDENCY REPORT - Pakmen" in capsys.readouterr().out
    cli.main(["pov", str(out), "--team", "Pakmen"])
    assert "Points lost by Pakmen" in capsys.readouterr().out
    from pypdf import PdfReader
    page = "\n".join(p.extract_text() for p in PdfReader(str(tmp_path / "me.pdf")).pages)
    assert "tagged by hand" in page and "Court found" not in page


# ---- the page's own logic, run under node ----------------------------------
def run_core(script):
    node = shutil.which("node")
    if not node:
        pytest.skip("node not installed")
    html = open(PAGE).read()
    core = html.split('<script id="core">')[1].split("</script>")[0]
    res = subprocess.run([node, "-e", core + "\n" + script], capture_output=True, text=True, timeout=30)
    assert res.returncode == 0, res.stderr
    return json.loads(res.stdout)


def test_page_logic_point_endings_and_next_team():
    out = run_core("""
      var M = module.exports, r = {};
      r.kill = M.pointWinner({type:'attack', team:'A', result:'kill'});
      r.err = M.pointWinner({type:'serve', team:'A', result:'error'});
      r.ace = M.pointWinner({type:'serve', team:'B', result:'ace'});
      r.shank = M.pointWinner({type:'reception', team:'A', grade:0});
      r.pass = M.pointWinner({type:'reception', team:'A', grade:2});
      r.blocked = M.pointWinner({type:'attack', team:'A', result:'blocked'});
      r.stuff = M.pointWinner({type:'block', team:'B', result:'stuff'});
      r.next = [M.nextTeam({type:'serve', team:'A', result:'in_play'}), M.nextTeam({type:'reception', team:'B', grade:3}),
                M.nextTeam({type:'attack', team:'B', result:'dug'}), M.nextTeam({type:'set', team:'A'})];
      r.keys = [M.resultForKey('attack','k'), M.resultForKey('serve','k'), M.resultForKey('attack','o'), M.resultForKey('attack','l'),
                M.resultForKey('set','k'), M.resultForKey('dig','x')];
      r.invalid = [M.validateTouch({type:null}), M.validateTouch({type:'attack'}), M.validateTouch({type:'reception'}),
                   M.validateTouch({type:'set'}), M.validateTouch({type:'reception', grade:0})];
      console.log(JSON.stringify(r));""")
    assert (out["kill"], out["err"], out["ace"], out["shank"], out["stuff"]) == ("A", "B", "B", "B", "B")
    assert out["pass"] is None and out["blocked"] is None
    assert out["next"] == ["B", "B", "A", "A"]
    assert out["keys"] == ["kill", "ace", "dug", "blocked", None, "error"]
    assert out["invalid"][:3] == ["choose the kind of touch", "choose the result", "give the pass a grade 0-3"]
    assert out["invalid"][3] is None and out["invalid"][4] is None


def test_page_session_exports_a_file_python_accepts(tmp_path):
    exported = run_core("""
      var M = module.exports, s = M.newState();
      s.nameA = 'Pakmen'; s.nameB = 'Flames';
      M.current(s).server = 'B';
      M.addTouch(s, {t:1, team:'B', number:5, type:'serve', result:'in_play', grade:null, shot:null, pos:null, target:null});
      M.addTouch(s, {t:2, team:'A', number:7, type:'reception', result:null, grade:3, shot:null, pos:null, target:null});
      M.addTouch(s, {t:3, team:'A', number:9, type:'set', result:null, grade:null, shot:null, pos:null, target:null});
      M.addTouch(s, {t:4, team:'A', number:7, type:'attack', result:'kill', grade:null, shot:'line', pos:[1.5,8.2], target:[1.8,16]});
      var afterKill = {rallies:s.rallies.length, server:M.current(s).server, score:M.score(s)};
      M.addTouch(s, {t:9, team:'A', number:7, type:'serve', result:'in_play', grade:null, shot:null, pos:null, target:null});
      M.endRally(s, 'B');                                   // point awarded by hand
      M.addTouch(s, {t:15, team:'B', number:5, type:'serve', result:'error', grade:null, shot:null, pos:null, target:null});
      M.addTouch(s, {t:20, team:'A', number:7, type:'serve', result:'ace', grade:null, shot:null, pos:null, target:null});
      var undone = M.undo(s);                               // reopens the rally and removes the ace
      var out = M.exportTags(s), back = M.exportTags(M.importTags(out));
      console.log(JSON.stringify({afterKill:afterKill, undone:undone.result, out:out, same:JSON.stringify(out)===JSON.stringify(back),
                                  score:M.score(s)}));""")
    assert exported["afterKill"] == {"rallies": 2, "server": "A", "score": {"A": 1, "B": 0}}
    assert exported["undone"] == "ace" and exported["same"] is True
    assert exported["score"] == {"A": 2, "B": 1}
    p = tmp_path / "from_page.json"
    p.write_text(json.dumps(exported["out"]))
    n, touches, warnings = import_tags(p, tmp_path / "set")
    assert (n, touches) == (3, 6) and warnings == []
    rallies, _ = rallies_from_tags(exported["out"])
    s = compute_stats([MatchSet(rallies, "Pakmen", "Flames")])
    me = s["players"][("Pakmen", 7)]["stats"]
    assert me["kills"].value == 1 and me["reception_avg"].value == 3.0 and me["serve_in_pct"].den == 1
    assert s["scores"] == [{"Pakmen": 2, "Flames": 1}]


def test_page_is_self_contained():
    html = open(PAGE).read()
    assert "http://" not in html and "https://" not in html      # nothing fetched: works offline
    assert html.count("<script") == 2


FAKE_PAGE = r'''
var els={};
function El(id){ this.id=id; this.innerHTML=''; this.textContent=''; this.className=''; this.value=''; this.children=[]; this.files=[]; }
El.prototype.appendChild=function(c){ this.children.push(c); return c; };
El.prototype.insertAdjacentHTML=function(){}; El.prototype.getAttribute=function(){return null;};
El.prototype.getBoundingClientRect=function(){ return {left:0,top:0,width:150,height:300}; };
var listeners={};
var document={ getElementById:function(id){ return els[id]||(els[id]=new El(id)); }, createElement:function(t){ return new El(t); },
  querySelectorAll:function(){ return []; }, addEventListener:function(n,f){ listeners[n]=f; } };
var store={}; var localStorage={ getItem:function(k){ return store[k]||null; }, setItem:function(k,v){ store[k]=v; } };
var URL={createObjectURL:function(){return 'blob:x';}}; function Blob(){}; function FileReader(){}
'''
DRIVE = r'''
var vid=els['vid']; vid.currentTime=0; vid.paused=true; vid.play=function(){this.paused=false;}; vid.pause=function(){this.paused=true;};
function key(k){ listeners.keydown({key:k, target:{tagName:'BODY'}, preventDefault:function(){}}); }
function seq(t, ks){ vid.currentTime=t; ks.forEach(key); }
els['srvB'].onclick();                               // opponent serves first: entry switches to them
seq(1.0, ['5','s','o','Enter']);                     // opp #5 serve, in play  -> entry flips to my team
seq(2.0, ['7','r','3','Enter']);                     // my #7 reception, grade 3
seq(3.0, ['9','e','Enter']);                         // my #9 set
vid.currentTime=4.0; ['7','a','k','2'].forEach(key); // my #7 attack, kill, cross
els['court'].onclick({clientX:25, clientY:163});     // own half: where it was hit
els['court'].onclick({clientX:133, clientY:25});     // other half: where it landed
key('Enter');                                        // kill ends the rally, my team serves
seq(9.0, ['1','2','s','x','Enter']);                 // my #12 serve error -> point to opponent
seq(12.0, ['a','Enter']);                            // incomplete touch: must not be saved
var hint = els['hint'].textContent;
key('Escape');
els['ptA'].onclick();                                // point awarded by hand
key(' ');
console.log(JSON.stringify({hint:hint, score:els['score'].textContent, playing:!vid.paused, saved:JSON.parse(store['volleyscope-tags'])}));
'''


def test_page_keyboard_flow_in_a_stand_in_page():
    """Drives the page's real script with key presses and court clicks, using a
    minimal stand-in for the browser. (Not a substitute for opening it.)"""
    node = shutil.which("node")
    if not node:
        pytest.skip("node not installed")
    import re
    scripts = re.findall(r"<script[^>]*>(.*?)</script>", open(PAGE).read(), re.S)
    res = subprocess.run([node, "-e", FAKE_PAGE + scripts[0] + "\n" + scripts[1] + "\n" + DRIVE], capture_output=True, text=True, timeout=30)
    assert res.returncode == 0, res.stderr
    out = json.loads(res.stdout)
    assert out["hint"] == "Not saved: choose the result."
    assert out["score"] == "My Team 2 - 1 Opponent" and out["playing"] is True
    r0, r1, r2 = out["saved"]["rallies"]
    assert r0["serve_team"] == "far" and r0["winner"] == "near"
    assert [(t["team"], t["number"], t["type"]) for t in r0["touches"]] == [
        ("far", 5, "serve"), ("near", 7, "reception"), ("near", 9, "set"), ("near", 7, "attack")]
    assert r0["touches"][1]["grade"] == 3
    assert r0["touches"][3] == {"t": 4, "team": "near", "number": 7, "type": "attack", "result": "kill", "shot": "cross",
                                "pos": [1.5, 8.2], "target": [8, 16.5]}
    assert r1["touches"][0]["number"] == 12 and r1["winner"] == "far" and r2["winner"] == "near"
    rallies, warnings = rallies_from_tags(out["saved"])
    assert warnings == [] and rallies[0].events[3].attrs["shot"] == "cross"
