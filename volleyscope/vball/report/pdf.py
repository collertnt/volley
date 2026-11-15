"""PDF stat sheets: one per player, one per team/match."""
from __future__ import annotations

import io

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import Image, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

from ..stats.core import MatchSet, compute_stats
from ..tendencies.core import MIN_N, coaching_notes, hitter_tendencies, report_lines, setter_distribution

_SS = getSampleStyleSheet()
H1 = ParagraphStyle("h1", parent=_SS["Title"], fontSize=18, spaceAfter=4, alignment=0)
H2 = ParagraphStyle("h2", parent=_SS["Heading2"], fontSize=12, spaceBefore=10, spaceAfter=4)
BODY = ParagraphStyle("body", parent=_SS["BodyText"], fontSize=9, leading=12)
SMALL = ParagraphStyle("small", parent=BODY, fontSize=7.5, leading=9.5, textColor=colors.HexColor("#555555"))
GREY = colors.HexColor("#f0f0f0")
WARN = colors.HexColor("#b35c00")

LIMITS = ("How to read this sheet: every figure shows the sample behind it. 'no data' means nothing was observed, "
          "not zero. 'low sample' means fewer than 5 events. 'uncertain' means more than a quarter of the events "
          "behind the figure were low-confidence detections (ball or player partly hidden, close line call, "
          "player number not read). Figures come from automatic video analysis with a single camera; see the "
          "README for what that can and cannot see.")


CELL = ParagraphStyle("cell", parent=BODY, fontSize=8, leading=9.5)
CELLH = ParagraphStyle("cellh", parent=CELL, fontName="Helvetica-Bold")
_FLAG_WORDS = ("uncertain", "low sample", "no data", "insufficient data")


def _cell(c, head):
    """Wrap text so it stays inside its column; colour the caveat words."""
    if not isinstance(c, str):
        return c
    txt = c.replace("&", "&amp;").replace("<", "&lt;")
    if not head:
        for w in _FLAG_WORDS:
            txt = txt.replace(w, f'<font color="#b35c00">{w}</font>')
    return Paragraph(txt, CELLH if head else CELL)


def _table(rows, widths=None, header=True):
    rows = [[_cell(c, header and i == 0) for c in r] for i, r in enumerate(rows)]
    t = Table(rows, colWidths=widths, repeatRows=1 if header else 0)
    style = [("GRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#bbbbbb")),
             ("VALIGN", (0, 0), (-1, -1), "TOP"), ("TOPPADDING", (0, 0), (-1, -1), 2), ("BOTTOMPADDING", (0, 0), (-1, -1), 2)]
    if header:
        style.append(("BACKGROUND", (0, 0), (-1, 0), GREY))
    t.setStyle(TableStyle(style))
    return t


def _stat_row(st):
    v = st.value
    if v is None:
        val = "-"
    elif st.kind == "pct":
        val = f"{100 * v:.0f}%"
    elif st.kind == "ratio":
        val = f"{v:+.3f}"
    elif st.kind == "avg":
        val = f"{v:.2f}"
    else:
        val = f"{int(v)}"
    sample = f"{st.num:g} / {st.den}" if st.kind in ("pct",) else (f"n = {st.den}" if st.kind != "count" else "")
    return [st.name, val, sample, st.status]


def _footage_box(infos):
    if not infos:
        return []
    if all(inf.get("manual") for inf in infos):
        rows = [["Set", "Source", "Rallies", "Touches tagged", "Notes"]]
        for k, inf in enumerate(infos):
            w = inf.get("warnings", [])
            rows.append([str(k + 1), "tagged by hand", str(inf.get("n_rallies", "?")), str(inf.get("n_touches", "?")),
                         Paragraph(f"{len(w)} thing(s) to check: " + "; ".join(w[:3]) if w else "no problems found", SMALL)])
        return [Paragraph("What was tagged", H2), _table(rows, [10 * mm, 28 * mm, 18 * mm, 28 * mm, 90 * mm]),
                Paragraph("Only tagged touches are counted. If only some touches were tagged, percentages describe those touches.", SMALL)]
    rows = [["Set", "Frames", "Court found", "Rallies", "Tracks with a number", "Notes"]]
    for k, inf in enumerate(infos):
        rows.append([str(k + 1), str(inf.get("frames", "?")), f"{100 * inf.get('court_usable_share', 0):.0f}% of frames",
                     str(inf.get("n_rallies", "?")), str(inf.get("numbers_read", "?")),
                     Paragraph(inf.get("note", ""), SMALL)])
    return [Paragraph("Footage analysed", H2), _table(rows, [10 * mm, 18 * mm, 28 * mm, 16 * mm, 40 * mm, 60 * mm])]


def shot_chart_png(sets, team, number=None):
    """Top-down chart of attack directions (green kill, red error/blocked, grey other)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    rows = []
    for ms in sets:
        for r in ms.rallies:
            for e in r.events:
                if e.type == "attack" and e.team and ms.name(e.team) == team and "target" in e.attrs and e.pos is not None \
                        and (number is None or e.number == number):
                    a, b = e.pos[:2], e.attrs["target"]
                    if e.team == "far":                      # draw every attack going "up" the page
                        a, b = [9 - a[0], 18 - a[1]], [9 - b[0], 18 - b[1]]
                    rows.append((a, b, e.outcome))
    if not rows:
        return None, 0
    fig, ax = plt.subplots(figsize=(2.6, 4.6), dpi=150)
    ax.add_patch(plt.Rectangle((0, 0), 9, 18, fill=False, lw=1))
    for y in (6, 9, 12):
        ax.plot([0, 9], [y, y], color="k", lw=2 if y == 9 else 0.6)
    for a, b, oc in rows:
        col = {"kill": "#1a9641", "error": "#d7191c", "blocked": "#d7191c"}.get(oc, "#888888")
        ax.annotate("", xy=b, xytext=a, arrowprops=dict(arrowstyle="->", color=col, lw=1))
    ax.set_xlim(-1.5, 10.5); ax.set_ylim(-1.5, 19.5); ax.set_aspect("equal"); ax.axis("off")
    ax.set_title(f"Attacks (n={len(rows)}): green kill, red error/blocked", fontsize=6)
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return buf, len(rows)


def player_pdf(path, sets: list[MatchSet], team: str, number: int, infos=None, stats=None):
    """Write one player's stat sheet. Returns the stats dict entry (or None)."""
    stats = stats or compute_stats(sets)
    p = stats["players"].get((team, number))
    doc = SimpleDocTemplate(str(path), pagesize=A4, leftMargin=15 * mm, rightMargin=15 * mm, topMargin=14 * mm, bottomMargin=14 * mm,
                            title=f"{team} #{number}")
    el = [Paragraph(f"{team} #{number} - player stat sheet", H1),
          Paragraph(f"{len(sets)} set(s), {sum(len(m.rallies) for m in sets)} rallies analysed.", BODY)]
    el += _footage_box(infos)
    if p is None:
        el += [Spacer(1, 8), Paragraph(
            f"No contacts were attributed to #{number} of {team}. Either the player did not touch the ball in this "
            f"footage, or the jersey number was never read. Check tracks_review.png and the corrections file.", BODY)]
        un = stats["players"].get((team, None))
        if un is not None:
            el.append(Paragraph(f"{int(un['stats']['contacts'].value)} contacts by {team} could not be tied to a number.", BODY))
        doc.build(el)
        return None
    s = p["stats"]
    el.append(Paragraph("Totals", H2))
    order = ["attack_attempts", "kills", "kill_pct", "attack_eff", "attack_errors", "reception_avg", "reception_3",
             "reception_2", "reception_1", "reception_0", "serve_in_pct", "aces", "serve_errors", "sets", "assist_pct",
             "digs", "dig_pct", "blocks", "block_touches", "error_rate", "contacts"]
    rows = [["Stat", "Value", "Sample", "Status"]] + [_stat_row(s[k]) for k in order]
    chart, n = shot_chart_png(sets, team, number)
    tbl = _table(rows, [62 * mm, 20 * mm, 24 * mm, 22 * mm])
    if chart is not None:
        el.append(Table([[tbl, Image(chart, width=45 * mm, height=80 * mm)]], colWidths=[130 * mm, 48 * mm],
                        style=[("VALIGN", (0, 0), (-1, -1), "TOP")]))
    else:
        el.append(tbl)
    el.append(Paragraph("By rotation (counted from the rotation each set started in)", H2))
    rr = [["Rotation", "Attacks", "Kill %", "Efficiency", "Reception avg", "Serve in %", "Errors / contacts"]]
    for rot in range(1, 7):
        b = p["by_rotation"].get(rot)
        if b is None:
            rr.append([str(rot)] + ["no data"] * 6)
            continue
        rr.append([str(rot), str(int(b["attack_attempts"].value)), b["kill_pct"].text(), b["attack_eff"].text(),
                   b["reception_avg"].text(), b["serve_in_pct"].text(), b["error_rate"].text()])
    el.append(_table(rr, [16 * mm, 16 * mm, 30 * mm, 32 * mm, 30 * mm, 30 * mm, 30 * mm]))
    el.append(Paragraph("Tendencies", H2))
    for t in hitter_tendencies(sets, team, number).values():
        el.append(Paragraph(t.text(), BODY))
    notes = coaching_notes(stats, sets, team, number)
    el.append(Paragraph("What the numbers suggest", H2))
    if notes["notes"]:
        for nline in notes["notes"]:
            el.append(Paragraph("- " + nline, BODY))
    else:
        el.append(Paragraph("Nothing can be said yet: no area has enough recorded events.", BODY))
    if notes["skipped"]:
        el.append(Paragraph("Not enough data to comment on: " + "; ".join(notes["skipped"]) + ".", SMALL))
    un = stats["players"].get((team, None))
    if un is not None:
        el.append(Paragraph(f"Note: {int(un['stats']['contacts'].value)} {team} contacts could not be tied to a jersey number "
                            f"and are not in these figures. Some of them may be this player's.", SMALL))
    el += [Spacer(1, 6), Paragraph(LIMITS, SMALL)]
    doc.build(el)
    return p


def team_pdf(path, sets: list[MatchSet], team: str, opponent: str, tracks_per_set=None, fps=30.0, infos=None, stats=None):
    """Write the per-match team report."""
    stats = stats or compute_stats(sets)
    T = stats["teams"].get(team)
    doc = SimpleDocTemplate(str(path), pagesize=A4, leftMargin=15 * mm, rightMargin=15 * mm, topMargin=14 * mm, bottomMargin=14 * mm,
                            title=f"{team} match report")
    el = [Paragraph(f"{team} vs {opponent} - match report", H1)]
    sc = "  |  ".join(f"Set {k + 1}: {team} {s.get(team, '?')} - {s.get(opponent, '?')} {opponent}" for k, s in enumerate(stats["scores"]))
    el.append(Paragraph("Points counted from the footage: " + sc, BODY))
    el += _footage_box(infos)
    if T is None:
        el.append(Paragraph(f"No rallies were found for {team}.", BODY))
        doc.build(el)
        return None
    if T["unknown_winner"]:
        el.append(Paragraph(f"{T['unknown_winner']} rally(ies) had no visible ending and are left out of the win/loss figures.", BODY))
    el.append(Paragraph("Serve and side-out", H2))
    el.append(_table([["Stat", "Value", "Sample", "Status"], _stat_row(T["side_out"]), _stat_row(T["break_point"])],
                     [70 * mm, 20 * mm, 24 * mm, 22 * mm]))
    el.append(Paragraph("Points by rotation", H2))
    rr = [["Rotation", "Rallies played", "Points lost", "Lost %", "Uncertain rallies"]]
    for rot in range(1, 7):
        b = T["by_rotation"].get(rot)
        if not b or not b["played"]:
            rr.append([str(rot), "0", "-", "no data", ""])
        else:
            rr.append([str(rot), str(b["played"]), str(b["lost"]), f"{100 * b['lost'] / b['played']:.0f}%", str(b["uncertain"])])
    el.append(_table(rr, [22 * mm, 30 * mm, 26 * mm, 22 * mm, 34 * mm]))
    el.append(Paragraph(f"How the {T['points_lost']} lost points were lost", H2))
    el.append(_table([["Situation", "Share of lost points", "Sample", "Status"]] + [_stat_row(v) for v in T["situations"].values()],
                     [86 * mm, 34 * mm, 24 * mm, 22 * mm]))
    el.append(Paragraph("A point can fall into more than one situation, or none.", SMALL))
    el.append(Paragraph("Every point lost", H2))
    lp = [["Set", "Time", "Rot.", "Cause", "Attributed to", "Situation", "Conf."]]
    for x in T["lost_points"]:
        who = "-" if x["by_team"] is None else f"{x['by_team']} " + (f"#{x['by_number']}" if x["by_number"] is not None else "(number unknown)")
        lp.append([str(x["set"]), f"{x['t']:.0f}s", f"{x['rotation']}{'?' if x['rotation_uncertain'] else ''}",
                   x["cause"].replace("_", " "), who, ", ".join(t.replace("_", " ") for t in x["tags"]) or "-", f"{x['conf']:.2f}"])
    el.append(_table(lp, [10 * mm, 14 * mm, 12 * mm, 34 * mm, 44 * mm, 52 * mm, 12 * mm]))
    el.append(Paragraph("Players", H2))
    pr = [["Player", "Attacks", "Kill %", "Efficiency", "Reception avg", "Serve in %", "Digs", "Blocks"]]
    for (tm, num), p in sorted(stats["players"].items(), key=lambda kv: (kv[0][1] is None, kv[0][1] or 0)):
        if tm != team:
            continue
        s = p["stats"]
        pr.append([p["label"], str(int(s["attack_attempts"].value)), s["kill_pct"].text(), s["attack_eff"].text(),
                   s["reception_avg"].text(), s["serve_in_pct"].text(), str(int(s["digs"].value)), str(int(s["blocks"].value))])
    el.append(_table(pr, [20 * mm, 17 * mm, 28 * mm, 30 * mm, 28 * mm, 28 * mm, 13 * mm, 16 * mm]))
    el.append(Paragraph("Tendencies", H2))
    for line in report_lines(sets, tracks_per_set or [None] * len(sets), team, opponent, fps)[2:]:
        if line.strip():
            el.append(Paragraph(line.strip(), BODY if line.startswith("  ") else ParagraphStyle("b", parent=BODY, fontName="Helvetica-Bold", spaceBefore=4)))
    el += [Spacer(1, 6), Paragraph(LIMITS, SMALL)]
    doc.build(el)
    return T
