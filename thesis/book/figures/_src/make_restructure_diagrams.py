"""Restructure-round diagrams (2026-07-17): the four TODO-FABLE-DIAGRAM
markers left by the V2 restructure agents. Emits editable .drawio files +
crisp PNG previews via the diagrams.net viewer (same pipeline as
render_drawio_hires.py). One function per figure.

Figures:
  fig_2_7_reqmap        ch2  requirements -> technique-family map
  fig_3_12_methodflow   ch3  evidence-gated methodology flow
  fig_4_5_export        ch4  PyTorch -> INT8 TensorRT export pipeline
  fig_5_2_rig           ch5  stereo rig split + rectification schematic
All numbers shown are frozen values already present in the chapter prose.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from PIL import Image

SRC = Path(__file__).resolve().parent
sys.path.insert(0, str(SRC))
from drawio_lib import (  # noqa: E402
    BLUE_F, BLUE_S, D, EDGE, GREEN_F, GREEN_S, GREY_F, GREY_S, LAV_F,
    LAV_S, ORAN_F, ORAN_S, RED_F, RED_S, YEL_F, YEL_S)

OUT = SRC.parent  # thesis/book/figures
URLS: dict[str, str] = {}


def fig_2_7_reqmap():
    # Bipartite map. Family boxes are vertically aligned so every PRIMARY
    # edge is a dead-straight horizontal line (exit right-center, enter
    # left-center); the three non-straight edges each get a dedicated
    # vertical channel (x = 520 / 560 / 600) so nothing crosses a box and
    # secondary crossings are clean perpendiculars.
    d = D("reqmap", 920, 620)
    d.text(60, 12, 380, 26, "Edge-deployment requirements", fs=16, bold=True)
    d.text(540, 12, 380, 26, "Compression technique families", fs=16,
           bold=True)
    RW, RH, FX, FW, FH = 270, 66, 560, 340, 58
    reqs = {  # key: center y
        "pb": ("Parameter budget<br>(&lt; 3 M weights)", 98),
        "rt": ("Real-time latency<br>(&ge; 25 FPS)", 248),
        "sws": ("Small working set<br>(fits accelerator memory)", 378),
        "qf": ("Quantization-friendly<br>operators (INT8)", 508),
    }
    fams = {
        "bb": ("Backbone substitution", 98),
        "kd": ("Knowledge distillation", 170),
        "it": ("Iterative-loop compression", 248),
        "cv": ("Cost-volume compression", 378),
        "ar": ("Architectural compression<br>and adaptive compute", 508),
    }
    rid = {k: d.box(60, c - RH // 2, RW, RH, label, fill=ORAN_F,
                    stroke=ORAN_S, fs=15) for k, (label, c) in reqs.items()}
    fid = {k: d.box(FX, c - FH // 2, FW, FH, label, fill=GREEN_F,
                    stroke=GREEN_S, fs=15) for k, (label, c) in fams.items()}
    # primary answers: straight horizontals
    for r, f in (("pb", "bb"), ("rt", "it"), ("sws", "cv"), ("qf", "ar")):
        d.edge(rid[r], fid[f], exit_=(1, 0.5), entry=(0, 0.5))
    # pb -> kd (solid, second primary of the parameter budget)
    d.edge(rid["pb"], fid["kd"], exit_=(1, 0.8), entry=(0, 0.5),
           points=((460, 118), (460, 170)))
    # secondary contributions (dashed); channels far apart (x=500 vs 680)
    # so the two dashed paths never run adjacent
    SEC = "#8a8a8a"
    d.edge(rid["rt"], fid["ar"], exit_=(1, 0.75), entry=(0, 0.25),
           points=((420, 264), (420, 493)), dashed=True, color=SEC)
    d.edge(rid["sws"], fid["it"], exit_=(1, 0.25), entry=(0, 0.75),
           points=((500, 361), (500, 262)), dashed=True, color=SEC)
    d.line(60, 585, 104, 585, width=2.2)
    d.text(112, 573, 220, 24, "primary answer", fs=13)
    d.line(360, 585, 404, 585, width=2.2, dashed=True, color=SEC)
    d.text(412, 573, 260, 24, "secondary contribution", fs=13)
    URLS["fig_2_7_reqmap"] = d.save(OUT / "fig_2_7_reqmap.drawio")


def fig_3_12_methodflow():
    # Serpentine 2 x 3 grid (was one 6-wide row on a 1320 px canvas that
    # printed ~5 pt): row 1 flows left-to-right, row 2 right-to-left, so
    # the canvas is only 780 px wide and fonts print ~12 pt.
    d = D("methodflow", 780, 560)
    W, H = 220, 104
    xs = (30, 280, 530)
    y1, y2 = 60, 330
    stages = [
        ("Literature study<br>(surveyed families,<br>reference designs)",
         GREY_F, GREY_S, xs[0], y1),
        ("Requirements envelope<br>(params, latency,<br>memory, INT8 ops)",
         ORAN_F, ORAN_S, xs[1], y1),
        ("Single-knob ablations<br>on the 100-pair<br>overfit harness",
         BLUE_F, BLUE_S, xs[2], y1),
        ("Full-dataset<br>cloud training", LAV_F, LAV_S, xs[2], y2),
        ("Three-way evaluation<br>(in-domain, zero-shot,<br>rectification)",
         GREEN_F, GREEN_S, xs[1], y2),
        ("Edge optimization +<br>on-device latency", YEL_F, YEL_S,
         xs[0], y2),
    ]
    ids = [d.box(x, y, W, H, label, fill=f, stroke=s, fs=15)
           for label, f, s, x, y in stages]
    d.edge(ids[0], ids[1], exit_=(1, 0.5), entry=(0, 0.5))
    d.edge(ids[1], ids[2], exit_=(1, 0.5), entry=(0, 0.5))
    d.edge(ids[2], ids[3], exit_=(0.5, 1), entry=(0.5, 0))  # down
    d.edge(ids[3], ids[4], exit_=(0, 0.5), entry=(1, 0.5))  # right-to-left
    d.edge(ids[4], ids[5], exit_=(0, 0.5), entry=(1, 0.5))
    d.text(508, 12, 270, 24, "evidence gate: winners only advance",
           fs=13, color="#b85450")
    # reject/retune self-loop on the ablation stage (above the box)
    d.edge(ids[2], ids[2], exit_=(0.75, 0), entry=(0.25, 0), dashed=True,
           points=((695, 40), (585, 40)), value="reject / retune")
    # failed evaluation reopens the design: eval (row 2) up to ablations;
    # label drawn as free text under the loop, NOT as an edge value
    # (edge values center on the path and land on top of the eval box)
    d.edge(ids[4], ids[2], exit_=(0.5, 1), entry=(0, 0.75), dashed=True,
           points=((390, 500), (505, 500), (505, 138)))
    d.text(120, 505, 260, 24, "failed evaluation reopens the design",
           fs=13, color="#666666")
    URLS["fig_3_12_methodflow"] = d.save(OUT / "fig_3_12_methodflow.drawio")


def fig_4_5_export():
    d = D("export", 1280, 430)
    ck = d.box(40, 60, 190, 70, "Trained PyTorch<br>checkpoint (fp32)",
               fill=GREY_F, stroke=GREY_S, fs=12)
    go = d.box(300, 40, 270, 110,
               "Graph optimization<br>(pre-export, equivalence-proven):<br>"
               "context precomputed per scale,<br>fused gates and heads,<br>"
               "dead code removed, GEV band narrowed",
               fill=BLUE_F, stroke=BLUE_S, fs=11)
    sw = d.box(300, 250, 270, 110,
               "Operator swaps for INT8:<br>unfold in convex upsampling,<br>"
               "group normalization,<br>3D convolution",
               fill=RED_F, stroke=RED_S, fs=11)
    ex = d.box(650, 60, 190, 70, "ONNX export", fill=LAV_F, stroke=LAV_S, fs=12)
    ca = d.box(650, 250, 190, 70, "INT8 calibration", fill=YEL_F,
               stroke=YEL_S, fs=12)
    en = d.box(920, 150, 200, 80, "TensorRT engine<br>(eight-bit)",
               fill=GREEN_F, stroke=GREEN_S, fs=12)
    d.edge(ck, go)
    d.edge(go, sw, points=((435, 200),))
    d.edge(go, ex)
    d.edge(sw, ca)
    d.edge(ex, ca, points=((745, 180),))
    d.edge(ca, en)
    d.edge(ex, en, dashed=True)
    jd = d.box(920, 300, 200, 60, "Jetson Orin Nano<br>measured on device",
               fill=ORAN_F, stroke=ORAN_S, fs=12)
    d.edge(en, jd)
    d.text(300, 12, 420, 22,
           "1.74x latency reduction before quantization", fs=11, bold=True)
    URLS["fig_4_5_export"] = d.save(OUT / "fig_4_5_export.drawio")


def fig_5_2_rig():
    d = D("rig", 1120, 430)
    # sensor + SBS frame
    rig = d.box(40, 60, 170, 90, "Binocular rig<br>(single sensor<br>stream)",
                fill=GREY_F, stroke=GREY_S, fs=12)
    sbs = d.box(290, 50, 340, 110, "", fill="#ffffff", stroke=EDGE, rounded=0)
    d.edge(rig, sbs)
    d.line(460, 50, 460, 160, width=1.3, dashed=True)
    d.text(300, 56, 150, 20, "left half", fs=11)
    d.text(470, 56, 150, 20, "right half", fs=11)
    d.text(290, 165, 340, 20, "2560 x 720 side-by-side frame", fs=11)
    L = d.box(760, 40, 250, 62, "Left image<br>1280 x 720 (rectified)",
              fill=BLUE_F, stroke=BLUE_S, fs=12)
    R = d.box(760, 130, 250, 62, "Right image<br>1280 x 720 (rectified)",
              fill=GREEN_F, stroke=GREEN_S, fs=12)
    d.edge(sbs, L, value="split + rectify")
    d.edge(sbs, R)
    # camera geometry with baseline / focal annotation
    cLx, cRx, cy = 340, 560, 300
    for x, name in ((cLx, "camera L"), (cRx, "camera R")):
        d.box(x - 35, cy, 70, 44, "", fill="#e9ecef", stroke=EDGE)
        d.text(x - 45, cy + 48, 90, 18, name, fs=11)
    d.line(cLx, cy + 22, cRx, cy + 22, width=2)
    d.text(cLx + 60, cy - 2, 160, 20, "baseline B = 52 mm", fs=12, bold=True)
    d.line(cLx, cy, cLx, cy - 60, width=1.3, dashed=True)
    d.text(cLx - 240, cy - 52, 230, 20,
           "focal length f &#8776; 1005 px<br>at 1280-px width", fs=12)
    d.text(40, 385, 700, 24,
           "Epipolar lines become image rows after rectification; disparity "
           "d maps to depth Z = f B / d.", fs=11)
    URLS["fig_5_2_rig"] = d.save(OUT / "fig_5_2_rig.drawio")


def trim_white(png, pad=18):
    im = np.array(Image.open(png).convert("RGB")).astype(int)
    bright = im.mean(2)
    rng = im.max(2) - im.min(2)
    ink = (bright < 200) | (rng > 12)
    ys, xs = np.where(ink)
    if len(ys) == 0:
        return
    y0, y1 = max(0, ys.min() - pad), min(im.shape[0], ys.max() + pad)
    x0, x1 = max(0, xs.min() - pad), min(im.shape[1], xs.max() + pad)
    Image.fromarray(im[y0:y1, x0:x1].astype("uint8")).save(png)


def main():
    fig_2_7_reqmap()
    fig_3_12_methodflow()
    fig_4_5_export()
    fig_5_2_rig()
    from playwright.sync_api import sync_playwright
    with sync_playwright() as p:
        br = p.chromium.launch()
        for name, url in URLS.items():
            pg = br.new_page(device_scale_factor=3,
                             viewport={"width": 1700, "height": 900})
            pg.goto(url, wait_until="networkidle", timeout=90000)
            pg.wait_for_timeout(4000)
            out = OUT / f"{name}_preview.png"
            el = pg.query_selector("svg")
            (el or pg).screenshot(path=str(out))
            pg.close()
            trim_white(out)
            print("rendered", out, Image.open(out).size)
        br.close()


if __name__ == "__main__":
    main()
