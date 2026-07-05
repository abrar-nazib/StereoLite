"""Emit ALL remaining thesis architecture diagrams as editable .drawio
files (grammar contract: WRITING_PLAN section 9). One function per figure.
Writes an HTML QA page embedding every diagram's viewer URL in stacked
iframes for the visual-review loop.

Structure sources: FINAL_MODEL_ARCHITECTURE.md sections 4-10 (per-block),
Scharstein taxonomy (classical pipeline), method_data.py (timeline),
review-paper compression taxonomy (family tree).
"""
from __future__ import annotations

import sys
from pathlib import Path

SRC = Path(__file__).resolve().parent
sys.path.insert(0, str(SRC))
from drawio_lib import (  # noqa: E402
    BLUE_F, BLUE_S, D, EDGE, GREEN_F, GREEN_S, GREY_F, GREY_S, LAV_F,
    LAV_S, ORAN_F, ORAN_S, RED_F, RED_S, SUP, YEL_F, YEL_S)

ROOT = Path("/home/abrar/Research/stero_research_claude")
OUT = ROOT / "thesis/book/figures"
sys.path.insert(0, str(ROOT / "review_paper/figures/_data"))
from method_data import METHODS  # noqa: E402

URLS: dict[str, str] = {}


def fig_1_1_geometry():
    d = D("geometry", 940, 460)
    VERM, BLUE, GREY = "#D55E00", "#0072B2", "#888888"
    oLx, oRx, ybox = 190, 600, 320
    Px, Py = 360, 65
    for x, name in ((oLx, "O<sub>L</sub>"), (oRx, "O<sub>R</sub>")):
        d.box(x - 40, ybox, 80, 48, "", fill="#e9ecef", stroke=EDGE)
        d.text(x - 20, ybox + 52, 40, 20, name, fs=12)
    # image planes
    for x in (oLx, oRx):
        d.line(x - 70, 250, x + 70, 250, width=2.6)
    # rays
    for x in (oLx, oRx):
        d.line(x, ybox, Px, Py, color=BLUE, width=1.4)
    # scene point
    d.node(Px - 7, Py - 7, 14, 14, "",
           f"ellipse;html=1;fillColor={VERM};strokeColor=#8a3000;")
    d.text(Px + 8, Py - 12, 30, 20, "P", fs=13, bold=True, color=VERM)
    # projections (ray/plane intersections)
    t = (250 - ybox) / (Py - ybox)
    for x, lbl, dx in ((oLx, "x<sub>L</sub>", 10), (oRx, "x<sub>R</sub>", 10)):
        xi = x + t * (Px - x)
        d.node(xi - 5, 245, 10, 10, "",
               f"ellipse;html=1;fillColor={BLUE};strokeColor=#1a4a7a;")
        d.text(xi + dx, 222, 34, 18, lbl, fs=11, color=BLUE)
    # f, B, Z annotations
    d.line(105, 250, 105, ybox, color=GREY, arrow="classic", width=1.0)
    d.line(105, ybox, 105, 250, color=GREY, arrow="classic", width=1.0)
    d.text(66, 272, 30, 20, "f", fs=12, color=GREY)
    d.line(oLx, 402, oRx, 402, color=GREY, arrow="classic", width=1.0)
    d.line(oRx, 402, oLx, 402, color=GREY, arrow="classic", width=1.0)
    d.text(345, 408, 110, 20, "baseline B", fs=11, color=GREY)
    d.line(Px, Py + 10, Px, ybox, color=VERM, dashed=True,
           arrow="classic", width=1.0)
    d.text(Px - 40, 190, 30, 20, "Z", fs=13, color=VERM)
    # relation panel
    d.text(720, 120, 190, 26, "d = x<sub>L</sub> − x<sub>R</sub>", fs=14)
    d.text(720, 165, 190, 34, "Z = f·B / d", fs=16, bold=True)
    d.text(715, 215, 200, 40,
           "near object → large d<br>far object → small d", fs=9,
           color=GREY)
    URLS["fig_1_1_stereo_geometry"] = d.save(
        OUT / "fig_1_1_stereo_geometry.drawio")


def fig_2_1_classical():
    d = D("classical", 1150, 240)
    steps = [
        ("1. Matching cost\ncomputation", "SAD · NCC · census"),
        ("2. Cost\naggregation", "local window ·\ncross-scale"),
        ("3. Disparity\noptimization", "WTA · SGM ·\ngraph cuts"),
        ("4. Disparity\nrefinement", "sub-pixel · LR check ·\nhole filling"),
    ]
    inp = d.box(30, 70, 130, 70, "<b>Rectified</b><br><b>stereo pair</b>",
                fill=GREY_F, fs=11)
    prev = inp
    x = 210
    for title, sub in steps:
        t = title.replace("\n", "<br>")
        s = sub.replace("\n", "<br>")
        b = d.box(x, 60, 165, 90,
                  f"<b>{t}</b><br><font style='font-size:9px'>{s}</font>",
                  fill=BLUE_F, stroke=BLUE_S, fs=11)
        d.edge(prev, b, exit_=(1, 0.5), entry=(0, 0.5))
        prev = b
        x += 215
    outp = d.box(x, 70, 130, 70, "<b>Disparity</b><br><b>map</b>",
                 fill=GREY_F, fs=11)
    d.edge(prev, outp, exit_=(1, 0.5), entry=(0, 0.5))
    URLS["fig_2_1_classical_pipeline"] = d.save(
        OUT / "fig_2_1_classical_pipeline.drawio")


def fig_2_2_timeline():
    d = D("timeline", 1500, 640)
    lanes = [
        ("Early end-to-end", "#e9ecef", "#666666"),
        ("3D cost volume", "#d5e8d4", "#82b366"),
        ("Iterative refinement", "#dae8fc", "#6c8ebf"),
        ("Foundation models", "#e1d5e7", "#9673a6"),
        ("Efficient (pre 2024)", "#ffe6cc", "#d79b00"),
        ("Efficient / edge (2024+)", "#fff2cc", "#d6b656"),
    ]
    items = {  # lane -> [(year, name)]
        0: [(2016, "MC-CNN"), (2016.6, "DispNetC"), (2017, "GC-Net")],
        1: [(2018, "PSMNet"), (2019, "GwcNet"), (2019.6, "GA-Net"),
            (2020, "AANet"), (2021, "CFNet"), (2022, "ACVNet")],
        2: [(2021, "RAFT-Stereo"), (2022, "CREStereo"), (2023, "IGEV"),
            (2024, "Selective"), (2025, "IGEV++")],
        3: [(2024.7, "DEFOM"), (2025.6, "FoundationStereo"),
            (2026.1, "MonSter"), (2025.0, "StereoAnywhere")],
        4: [(2018, "StereoNet"), (2019, "AnyNet"), (2019.9, "DeepPruner"),
            (2019.2, "MADNet"), (2021, "HITNet"), (2021.5, "CoEx"),
            (2021.95, "BGNet"), (2022.4, "MobileStereoNet")],
        5: [(2024, "DTP"), (2024.5, "LightStereo"), (2025, "BANet"),
            (2025.4, "LiteAnyStereo"), (2026, "Fast-FStereo")],
    }
    x0, xs = 150, 118  # x = x0 + (year-2016)*xs
    lane_y0, lane_h = 40, 88
    for li, (name, fill, stroke) in enumerate(lanes):
        y = lane_y0 + li * lane_h
        d.node(x0 - 20, y, (2026.8 - 2016) * xs + 40, lane_h - 8, "",
               f"rounded=0;html=1;fillColor={fill};strokeColor=none;"
               f"opacity=28;")
        d.text(2, y + lane_h / 2 - 22, 140, 36, f"<b>{name}</b>", fs=9.5,
               align="right")
        row = 0
        for year, m in sorted(items[li]):
            bx = x0 + (year - 2016) * xs
            by = y + 8 + (row % 2) * 38
            d.box(bx, by, 100, 28, m, fill=fill, stroke=stroke, fs=9)
            row += 1
    # ours
    y = lane_y0 + 5 * lane_h
    d.node(x0 + (2026.3 - 2016) * xs, y + 46, 118, 30,
           "<b>StereoLite (ours)</b>",
           f"rounded=1;html=1;fillColor=#ffffff;strokeColor={SUP};"
           f"strokeWidth=2;fontSize=9;")
    # axis
    ax_y = lane_y0 + 6 * lane_h + 6
    d.line(x0 - 20, ax_y, x0 + (2026.8 - 2016) * xs + 20, ax_y, width=1.4)
    for yr in range(2016, 2027):
        x = x0 + (yr - 2016) * xs
        d.line(x, ax_y, x, ax_y + 6, width=1.2)
        d.text(x - 22, ax_y + 8, 44, 16, str(yr), fs=9)
    URLS["fig_2_2_timeline"] = d.save(OUT / "fig_2_2_timeline.drawio")


def fig_2_5_taxonomy():
    d = D("taxonomy", 1420, 470)
    root = d.box(560, 20, 300, 52,
                 "<b>Compression techniques for</b><br>"
                 "<b>deep stereo matching</b>", fill=GREY_F, fs=12)
    fams = [
        ("Backbone\nsubstitution", "MobileStereoNet ·\nGhost/YOLO encoders",
         BLUE_F, BLUE_S),
        ("Cost-volume\ncompression", "cascade · BGNet grid ·\nDeepPruner · tiles (HITNet)",
         ORAN_F, ORAN_S),
        ("Iterative-loop\ncompression", "LightStereo · DTP ·\niteration pruning",
         GREEN_F, GREEN_S),
        ("Knowledge\ndistillation", "LiteAnyStereo ·\nDistill-then-Prune",
         LAV_F, LAV_S),
        ("Architectural\ncompression", "CoEx GCE ·\nCGI-Stereo", RED_F, RED_S),
        ("Adaptive\ncompute", "AnyNet anytime ·\nMADNet online", YEL_F, YEL_S),
        ("Neural architecture\nsearch", "LEAStereo ·\nEASNet", GREY_F, GREY_S),
    ]
    n = len(fams)
    w, gap = 172, 26
    total = n * w + (n - 1) * gap
    x = (1420 - total) / 2
    for title, ex, fill, stroke in fams:
        t = title.replace("\n", "<br>")
        e = ex.replace("\n", "<br>")
        fb = d.box(x, 150, w, 56, f"<b>{t}</b>", fill=fill, stroke=stroke,
                   fs=10.5)
        eb = d.box(x + 10, 240, w - 20, 56,
                   f"<font style='font-size:9px'>{e}</font>", fill="#ffffff",
                   stroke="#bbbbbb", fs=9)
        d.edge(root, fb, exit_=(0.5, 1), entry=(0.5, 0))
        d.edge(fb, eb, exit_=(0.5, 1), entry=(0.5, 0), width=1.0)
        x += w + gap
    URLS["fig_2_5_taxonomy"] = d.save(OUT / "fig_2_5_taxonomy.drawio")


def fig_3_2_encoders():
    d = D("encoders", 1150, 470)
    inp = d.box(30, 90, 150, 64, "<b>Left · Right</b><br>(B, 3, H, W)",
                fill=GREY_F, fs=11)
    bars = d.bars(260, 130, [150, 122, 96, 74], GREY_F, GREY_S, w=16, gap=10)
    d.text(210, 220, 220, 36,
           "<b>Shared encoder</b><br>YOLO26s layers 0-6 (pretrained)",
           fs=10.5)
    d.edge(inp, bars[0], exit_=(1, 0.5), entry=(0, 0.5))
    outs = [("f4 · H/4 · 128 ch", 82), ("f8 · H/8 · 256 ch", 128),
            ("f16 · H/16 · 256 ch", 174)]
    for lbl, y in outs:
        b = d.box(510, y - 16, 170, 34, lbl, fill=BLUE_F, stroke=BLUE_S,
                  fs=10)
        d.edge(bars[3], b, points=[(455, y)], exit_=(1, 0.5),
               entry=(0, 0.5), width=1.2)
    d.text(700, 112, 260, 60,
           "to cost volume + warp (1/16)<br>to refinement (1/8, 1/4)<br>"
           "to GEV + upsample masks (1/4)", fs=9, align="left")

    inp2 = d.box(30, 330, 150, 56, "<b>Left only</b><br>(B, 3, H, W)",
                 fill=GREY_F, fs=11)
    cb = d.bars(260, 358, [96, 74, 56], LAV_F, LAV_S, w=14, gap=9)
    d.text(210, 412, 220, 34,
           "<b>Context encoder</b><br>GhostConv + SE, GroupNorm/SiLU",
           fs=10.5)
    d.edge(inp2, cb[0], exit_=(1, 0.5), entry=(0, 0.5))
    c4 = d.box(430, 340, 150, 36, "ctx4 · H/4 · 32 ch", fill=LAV_F,
               stroke=LAV_S, fs=10)
    d.edge(cb[2], c4, exit_=(1, 0.5), entry=(0, 0.5))
    c8 = d.box(640, 300, 170, 34, "ctx8 (avg-pool of ctx4)", fill=LAV_F,
               stroke=LAV_S, fs=9.5)
    c16 = d.box(640, 382, 170, 34, "ctx16 (avg-pool of ctx4)", fill=LAV_F,
                stroke=LAV_S, fs=9.5)
    d.edge(c4, c8, exit_=(1, 0.5), entry=(0, 0.5), width=1.1)
    d.edge(c4, c16, exit_=(1, 0.5), entry=(0, 0.5), width=1.1)
    d.text(830, 330, 290, 54,
           "feeds every ConvGRU update;<br>ctx16 also initializes the"
           " hidden state h", fs=9.5, align="left")
    URLS["fig_3_2_encoders"] = d.save(OUT / "fig_3_2_encoders.drawio")


def fig_3_3_tile_init():
    d = D("tileinit", 1200, 360)
    fl = d.box(30, 60, 130, 44, "f<sub>L16</sub> (256 ch)", fill=BLUE_F,
               stroke=BLUE_S, fs=10.5)
    fr = d.box(30, 150, 130, 44, "f<sub>R16</sub> (256 ch)", fill=BLUE_F,
               stroke=BLUE_S, fs=10.5)
    cube = d.cube(240, 75, 110, 85,
                  "<b>Correlation volume</b><br>8 groups × 24 disparities")
    d.edge(fl, cube, exit_=(1, 0.5), entry=(0, 0.3))
    d.edge(fr, cube, exit_=(1, 0.5), entry=(0, 0.7))
    bars = d.bars(430, 118, [72, 56, 42], RED_F, RED_S)
    d.text(390, 172, 150, 34, "<b>3-D CNN</b><br>8-16-16-1", fs=10)
    d.edge(cube, bars[0], exit_=(1, 0.5), entry=(0, 0.5))
    sm = d.box(570, 96, 120, 44, "softmax<br>p(d)", fill=YEL_F,
               stroke=YEL_S, fs=10.5)
    d.edge(bars[2], sm, exit_=(1, 0.5), entry=(0, 0.5))
    da = d.box(760, 50, 190, 40, "d<sub>0</sub> = Σ p(d)·d  (soft-argmax)",
               fill=GREY_F, fs=10)
    ca = d.box(760, 145, 190, 40, "c<sub>0</sub> = max p(d)", fill=GREY_F,
               fs=10)
    d.edge(sm, da, exit_=(1, 0.3), entry=(0, 0.5))
    d.edge(sm, ca, exit_=(1, 0.7), entry=(0, 0.5))
    pill = d.box(1010, 88, 170, 64,
                 "<b>Tile state</b><br>T = (d<sub>0</sub>, 0, 0, "
                 "ctx16, c<sub>0</sub>)", fill=GREY_F, fs=10.5)
    d.edge(da, pill, exit_=(1, 0.5), entry=(0, 0.3))
    d.edge(ca, pill, exit_=(1, 0.5), entry=(0, 0.7))
    d.text(240, 250, 400, 30,
           "24 hypotheses at 1/16 scale cover roughly 0 to 368"
           " full-resolution pixels", fs=9.5, align="left")
    URLS["fig_3_3_tile_init"] = d.save(OUT / "fig_3_3_tile_init.drawio")


def fig_3_4_refinement():
    d = D("refine", 1330, 430)
    st = d.box(20, 150, 120, 66, "<b>state</b><br>(d, s<sub>x</sub>, "
               "s<sub>y</sub>, h, c)", fill=GREY_F, fs=10)
    warp = d.box(200, 60, 140, 46, "warp f<sub>R</sub> by d", fill=BLUE_F,
                 stroke=BLUE_S, fs=10.5)
    corr = d.box(200, 150, 140, 52, "local correlation<br>±2 (5 offsets)",
                 fill=ORAN_F, stroke=ORAN_S, fs=10)
    cat = d.box(410, 96, 170, 92,
                "<b>assemble x</b><br>[f<sub>L</sub>, warp(f<sub>R</sub>), "
                "d, s<sub>x</sub>, s<sub>y</sub>, c,<br>corr, ctx]",
                fill=GREY_F, fs=9.5)
    gru = d.box(650, 96, 160, 92,
                "<b>ConvGRU</b><br>z, r = σ(Conv[h,x])<br>"
                "q = tanh(Conv[r·h, x])<br>h' = (1−z)h + zq",
                fill=GREEN_F, stroke=GREEN_S, fs=8.5)
    head = d.box(880, 60, 150, 50, "<b>update head</b><br>2 layers · 48 ch",
                 fill=GREEN_F, stroke=GREEN_S, fs=10)
    gate = d.box(880, 150, 150, 56, "<b>4 sigmoid gates</b><br>"
                 "Δd, Δs<sub>x</sub>, Δs<sub>y</sub>, Δc", fill=GREY_F,
                 fs=10)
    newst = d.box(1110, 128, 150, 70,
                  "<b>updated state</b><br>d' = softplus(d + g·Δd)",
                  fill=GREY_F, fs=9.5)
    d.edge(st, warp, points=[(170, 183), (170, 83)], exit_=(1, 0.5),
           entry=(0, 0.5))
    d.edge(st, corr, exit_=(1, 0.5), entry=(0, 0.5))
    d.edge(warp, cat, exit_=(1, 0.5), entry=(0, 0.25))
    d.edge(corr, cat, exit_=(1, 0.5), entry=(0, 0.75))
    d.edge(cat, gru, exit_=(1, 0.5), entry=(0, 0.5))
    d.edge(gru, head, exit_=(1, 0.3), entry=(0, 0.5))
    d.edge(gru, gate, exit_=(1, 0.7), entry=(0, 0.5))
    d.edge(head, newst, exit_=(1, 0.5), entry=(0, 0.3))
    d.edge(gate, newst, exit_=(1, 0.5), entry=(0, 0.7))
    ctx = d.box(650, 290, 160, 40, "context (scale-matched)", fill=LAV_F,
                stroke=LAV_S, fs=9.5)
    d.edge(ctx, gru, dashed=True, color=LAV_S, exit_=(0.5, 0),
           entry=(0.5, 1))
    # recurrence
    d.edge(newst, st, points=[(1185, 380), (80, 380)], dashed=True,
           value="repeat ×2 (1/16) · ×3 (1/8) · ×3 (1/4)",
           exit_=(0.5, 1), entry=(0.5, 1))
    URLS["fig_3_4_refinement"] = d.save(OUT / "fig_3_4_refinement.drawio")


def fig_3_5_plane_prop():
    # squarer layout (equation row below) so text stays legible at print
    d = D("planeprop", 940, 620)
    d.box(70, 110, 200, 200, "", fill=GREY_F, stroke=EDGE)
    d.text(70, 190, 200, 44, "<b>parent tile</b><br>d, s<sub>x</sub>, "
           "s<sub>y</sub>", fs=17)
    d.line(120, 275, 215, 240, color="#0072B2", arrow="classic", width=1.8)
    d.text(200, 262, 48, 24, "s<sub>x</sub>", fs=15, color="#0072B2")
    d.text(300, 150, 190, 32, "<b>&#8593;2 upsample</b>", fs=17)
    offs = [("&#8722;¼, &#8722;¼", 0, 0),
            ("+¼, &#8722;¼", 1, 0),
            ("&#8722;¼, +¼", 0, 1),
            ("+¼, +¼", 1, 1)]
    for lbl, cx, cy in offs:
        x, y = 540 + cx * 140, 95 + cy * 140
        d.box(x, y, 130, 130,
              f"<font style='font-size:15px'>&#916; = ({lbl})</font>",
              fill="#eef4fb", stroke=BLUE_S)
    d.line(280, 210, 530, 210, arrow="classic", width=2.0)
    d.text(70, 420, 800, 60,
           "<font style='font-size:19px'>d<sub>child</sub> = "
           "2&#183;bilinear(d) + 2 s<sub>x</sub> &#916;x + "
           "2 s<sub>y</sub> &#916;y</font>", fs=19)
    d.text(70, 500, 810, 44,
           "<font style='font-size:15px'>the slopes shift each child off "
           "the parent plane; the factor 2 rescales the disparity unit to "
           "the finer grid</font>", fs=15, align="left")
    URLS["fig_3_5_plane_prop"] = d.save(OUT / "fig_3_5_plane_prop.drawio")


def fig_3_6_gev():
    d = D("gev", 1240, 430)
    dtap = d.box(30, 60, 130, 44, "tile d (1/4)", fill=GREY_F, fs=10.5)
    band = d.box(220, 55, 170, 54, "<b>band select</b><br>d ± 16 (33 bins)",
                 fill=YEL_F, stroke=YEL_S, fs=10)
    d.edge(dtap, band, exit_=(1, 0.5), entry=(0, 0.5))
    f4 = d.box(220, 160, 170, 40, "f<sub>L4</sub>, f<sub>R4</sub> (8 grp)",
               fill=BLUE_F, stroke=BLUE_S, fs=10)
    cube = d.cube(450, 80, 110, 85, "<b>narrow GEV</b><br>8 grp × 33 bins")
    d.edge(band, cube, exit_=(1, 0.5), entry=(0, 0.3))
    d.edge(f4, cube, exit_=(1, 0.5), entry=(0, 0.75))
    bars = d.bars(630, 122, [66, 52, 40], RED_F, RED_S)
    d.text(590, 172, 160, 32, "3 × (3×3×3) conv · 16 ch", fs=9)
    d.edge(cube, bars[0], exit_=(1, 0.5), entry=(0, 0.5))
    outs = d.box(770, 76, 190, 92,
                 "d<sub>gev</sub> = Σ p(d)·d<br>c<sub>gev</sub> = max p(d)"
                 "<br>g<sub>gev</sub> = φ(Σ G(d) p(d))",
                 fill=GREY_F, fs=9.5)
    d.edge(bars[2], outs, exit_=(1, 0.5), entry=(0, 0.5))
    gate = d.box(480, 270, 300, 74,
                 "<b>fail-soft gate</b>  w = σ(F(·))<br>"
                 "<font style='font-size:9px'>inputs: ctx4 · g<sub>gev"
                 "</sub> · c · c<sub>gev</sub> · |d<sub>gev</sub> − d| · d"
                 " · bias init −4 (w<sub>0</sub> ≈ 0.02)</font>",
                 fill=GREY_F, fs=10.5)
    d.edge(outs, gate, points=[(865, 307)], exit_=(0.5, 1), entry=(1, 0.5),
           dashed=True)
    d.edge(dtap, gate, points=[(95, 307)], exit_=(0.5, 1), entry=(0, 0.5),
           dashed=True)
    blend = d.box(1010, 250, 190, 60,
                  "d ← softplus(d + w·(d<sub>gev</sub> − d))<br>"
                  "<font style='font-size:9px'>slopes ×(1−w) · "
                  "c ← max(c, c<sub>gev</sub>)</font>", fill=GREEN_F,
                  stroke=GREEN_S, fs=9.5)
    d.edge(gate, blend, points=[(880, 340), (880, 280)], exit_=(1, 0.7),
           entry=(0, 0.5))
    URLS["fig_3_6_gev_fusion"] = d.save(OUT / "fig_3_6_gev_fusion.drawio")


def fig_3_7_upsample():
    d = D("upsample", 1060, 360)
    # coarse 3x3 neighborhood
    for i in range(3):
        for j in range(3):
            fill = "#eef4fb" if (i, j) != (1, 1) else BLUE_F
            d.box(60 + j * 52, 60 + i * 52, 48, 48, "", fill=fill,
                  stroke=BLUE_S, rounded=0)
    d.text(50, 224, 180, 30, "3×3 coarse neighborhood", fs=9.5)
    mask = d.box(300, 92, 190, 66,
                 "<b>mask head</b><br>9 weights / subpixel<br>softmax"
                 " (from f<sub>L</sub>)", fill=GREY_F, fs=9.5)
    d.line(222, 138, 298, 125, arrow="classic", width=1.3)
    mix = d.ellipse(540, 105, 40, 40, "⊗")
    d.edge(mask, mix, exit_=(1, 0.5), entry=(0, 0.5))
    fine = d.box(640, 96, 120, 58, "<b>fine pixel</b><br>(convex comb.)",
                 fill=GREEN_F, stroke=GREEN_S, fs=10)
    d.edge(mix, fine, exit_=(1, 0.5), entry=(0, 0.5))
    d.text(300, 250, 520, 56,
           "two 2× stages: 1/4 → 1/2 (masks from f<sub>L4</sub>) → full"
           " (masks from f<sub>L2</sub>);<br>disparity multiplied by 2 at"
           " each stage; weights non-negative and sum to one", fs=9.5,
           align="left")
    URLS["fig_3_7_upsample"] = d.save(OUT / "fig_3_7_upsample.drawio")


def fig_3_8_supervision():
    d = D("supervision", 1360, 330)
    gt = d.box(540, 20, 220, 40, "<b>ground-truth disparity</b><br>"
               "<font style='font-size:8.5px'>valid: finite, 0 &lt; d &lt;"
               " 320 px</font>", fill=GREY_F, fs=10)
    outs = [("d<sub>1/16</sub>", "0.10"), ("d<sub>1/8</sub>", "0.20"),
            ("d<sub>gev</sub>", "0.15"), ("d<sub>1/4</sub>", "0.30"),
            ("d<sub>1/2</sub>", "0.50"), ("d<sub>full</sub>", "1.00")]
    x = 60
    for name, wgt in outs:
        b = d.box(x, 140, 130, 44, name, fill=GREEN_F, stroke=GREEN_S,
                  fs=11)
        d.dot(x + 65, 196)
        d.text(x + 20, 206, 90, 18, f"L1 × {wgt}", fs=9, color=SUP)
        d.edge(gt, b, dashed=True, color=SUP, exit_=(0.5, 1),
               entry=(0.5, 0), width=0.9)
        x += 175
    d.box(1090, 240, 240, 80,
          "<b>full-res terms</b><br><font style='font-size:9px'>"
          "gradient 0.50 · threshold 0.20 ·<br>D1 0.20 · smooth 0.02 ·"
          " slant (gated) 0.30</font>", fill="#ffffff", stroke=SUP, fs=10)
    d.text(60, 250, 700, 30,
           "lower-scale predictions are resized to full resolution and"
           " rescaled before the L1 comparison", fs=9.5, align="left")
    URLS["fig_3_8_supervision"] = d.save(OUT / "fig_3_8_supervision.drawio")


def main():
    for f in (fig_1_1_geometry, fig_2_1_classical, fig_2_2_timeline,
              fig_2_5_taxonomy, fig_3_2_encoders, fig_3_3_tile_init,
              fig_3_4_refinement, fig_3_5_plane_prop, fig_3_6_gev,
              fig_3_7_upsample, fig_3_8_supervision):
        f()
    # QA page: all diagrams stacked in iframes
    html = ["<style>iframe{width:1500px;height:520px;border:1px solid #ccc;"
            "display:block;margin:6px 0}</style>"]
    for name, url in URLS.items():
        html.append(f"<div style='font:13px monospace'>{name}</div>")
        html.append(f'<iframe src="{url}"></iframe>')
    qa = Path("/tmp/claude-1000/-home-abrar-Research-stero-research-claude/"
              "b394c81f-1f73-4cd5-8f24-76f90855219a/scratchpad/"
              "drawio_qa.html")
    qa.write_text("\n".join(html))
    print(f"{len(URLS)} diagrams; QA page: {qa}")


if __name__ == "__main__":
    main()
