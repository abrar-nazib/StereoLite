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
    # Serpentine 2 x 3 grid on a narrow canvas so the box font prints large
    # at \textwidth. Row 1 flows left to right, row 2 right to left. Both
    # feedback paths are dashed red and carry SHORT labels on the edge
    # itself; there is no free-floating caption anywhere in the figure.
    # The evaluation feedback rides the empty horizontal band between the
    # two rows (y = 248) and a dedicated vertical channel at x = 668, well
    # clear of the forward descent channel at x = 782.
    d = D("methodflow", 880, 470)
    W, H = 230, 110
    xs = (40, 325, 610)
    y1, y2 = 66, 320
    FB = "#b85450"

    def lab(t, color=FB):
        # edge() hardcodes fontSize=10; inline HTML lifts it to 14 px and
        # paints an opaque backdrop so the label never sits on the stroke.
        return ('<span style="font-size:14px;color:' + color +
                ';background-color:#ffffff;">&nbsp;' + t + '&nbsp;</span>')

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
    # forward serpentine
    d.edge(ids[0], ids[1], exit_=(1, 0.5), entry=(0, 0.5))
    d.edge(ids[1], ids[2], exit_=(1, 0.5), entry=(0, 0.5))
    d.edge(ids[2], ids[3], exit_=(0.75, 1), entry=(0.75, 0))  # descend
    d.edge(ids[3], ids[4], exit_=(0, 0.5), entry=(1, 0.5))    # right to left
    d.edge(ids[4], ids[5], exit_=(0, 0.5), entry=(1, 0.5))
    # a rejected ablation arm returns to the harness (compact loop above)
    d.edge(ids[2], ids[2], exit_=(0.88, 0), entry=(0.12, 0), dashed=True,
           color=FB, points=((812, 22), (638, 22)),
           value=lab("rejected arm"))
    # a failed evaluation reopens the design, routed through the clear band
    # between the rows instead of looping outside the figure
    d.edge(ids[4], ids[2], exit_=(0.75, 0), entry=(0.25, 1), dashed=True,
           color=FB, points=((497, 248), (668, 248)),
           value=lab("failed evaluation"))
    URLS["fig_3_12_methodflow"] = d.save(OUT / "fig_3_12_methodflow.drawio")


def fig_4_5_export():
    # Was 1280 x 430 (labels printed ~5 pt, boxes stuffed with paragraphs);
    # then a 6-row vertical spine, which printed legibly but ran taller than
    # wide and forced a full-page float with a blank lower-right quadrant.
    # Now a 3 x 3 serpentine grid: row 1 left-to-right, row 3 right-to-left,
    # the INT8 operator-swap branch fills the middle-right cell and the
    # 1.74x callout the middle-left cell, tied to the graph-optimization box
    # by a dashed leader. Every grid cell is occupied, so the bounding box is
    # roughly 2:1 and the float sits in a fraction of a page. Enumerated
    # optimizations stay in the prose and Table 4.2, not in the boxes.
    C1, C2, C3, W = 40, 390, 740, 310
    R1, R2, R3, H, H2 = 50, 210, 355, 95, 85
    d = D("export", 1130, 500)
    ck = d.box(C1, R1, W, H, "Trained PyTorch checkpoint<br>(fp32)",
               fill=GREY_F, stroke=GREY_S, fs=15)
    go = d.box(C2, R1, W, H,
               "Graph optimization<br>(pre-export)<br>"
               "four equivalence-proven changes",
               fill=BLUE_F, stroke=BLUE_S, fs=15)
    ex = d.box(C3, R1, W, H, "ONNX export", fill=LAV_F, stroke=LAV_S, fs=15)
    sw = d.box(C3, R2, W, H2,
               "Operator swaps for INT8<br>three operators swapped",
               fill=RED_F, stroke=RED_S, fs=15)
    ca = d.box(C3, R3, W, H, "INT8 calibration", fill=YEL_F, stroke=YEL_S,
               fs=15)
    en = d.box(C2, R3, W, H, "TensorRT engine<br>(eight-bit)",
               fill=GREEN_F, stroke=GREEN_S, fs=15)
    jd = d.box(C1, R3, W, H, "Jetson Orin Nano<br>measured on device",
               fill=ORAN_F, stroke=ORAN_S, fs=15)
    # callout anchored to the graph-optimization stage it describes
    nt = d.box(C2, R2, W, H2, "", fill="#ffffff", stroke="#b85450",
               dashed=True)
    d.text(C2, R2 + 17, W, 50,
           "1.74x latency reduction<br>before quantization", fs=14,
           bold=True, color="#b85450")
    d.edge(go, nt, exit_=(0.25, 1), entry=(0.25, 0), dashed=True,
           arrow=False, color="#b85450")
    d.edge(ck, go, exit_=(1, 0.5), entry=(0, 0.5))
    d.edge(go, ex, exit_=(1, 0.5), entry=(0, 0.5))
    # branch turns below the ONNX box, then drops the right column
    d.edge(go, sw, exit_=(0.9, 1), entry=(0.5, 0),
           points=((669, 177), (895, 177)))
    d.edge(sw, ca, exit_=(0.5, 1), entry=(0.5, 0))
    # full-precision graph reaches calibration on an outer rail so it does
    # not cut through the operator-swap box
    d.edge(ex, ca, exit_=(1, 0.5), entry=(1, 0.5),
           points=((1080, 97), (1080, 402)))
    d.edge(ca, en, exit_=(0, 0.5), entry=(1, 0.5))
    d.edge(en, jd, exit_=(0, 0.5), entry=(1, 0.5))
    URLS["fig_4_5_export"] = d.save(OUT / "fig_4_5_export.drawio")


def fig_5_2_rig():
    # Two stacked bands on a narrow canvas (was 1120 x 430, which printed
    # every label at roughly 5 pt at \textwidth). Band 1 is the capture and
    # split path, band 2 the rig geometry. No trailing caption sentence: the
    # chapter prose already states the epipolar and Z = f B / d relations.
    d = D("rig", 860, 500)
    # band 1: capture and split
    d.text(30, 12, 320, 24, "Capture and split", fs=14, bold=True,
           align="left")
    rig = d.box(30, 80, 150, 96, "Binocular rig<br>(single sensor<br>stream)",
                fill=GREY_F, stroke=GREY_S, fs=14)
    sbs = d.box(225, 84, 270, 88, "", fill="#ffffff", stroke=EDGE, rounded=0)
    d.edge(rig, sbs)
    d.line(360, 84, 360, 172, width=1.3, dashed=True)
    d.text(228, 90, 128, 22, "left half", fs=13)
    d.text(364, 90, 128, 22, "right half", fs=13)
    d.text(225, 178, 270, 22, "2560 x 720 side-by-side frame", fs=13)
    L = d.box(600, 60, 230, 66, "Left image<br>1280 x 720 (rectified)",
              fill=BLUE_F, stroke=BLUE_S, fs=14)
    R = d.box(600, 148, 230, 66, "Right image<br>1280 x 720 (rectified)",
              fill=GREEN_F, stroke=GREEN_S, fs=14)
    d.edge(sbs, L)
    d.edge(sbs, R)
    d.text(495, 192, 112, 40, "split and<br>rectify", fs=13)
    # band 2: rig geometry, centred under band 1; the focal-length note is
    # tied to the optical axis by a short leader so it cannot read as a
    # free-floating label
    d.text(30, 268, 320, 24, "Rig geometry", fs=14, bold=True, align="left")
    cLx, cRx, cy = 300, 620, 396
    for x, name in ((cLx, "camera L"), (cRx, "camera R")):
        d.box(x - 38, cy, 76, 48, "", fill="#e9ecef", stroke=EDGE)
        d.text(x - 60, cy + 52, 120, 22, name, fs=13)
    d.line(cLx, cy + 24, cRx, cy + 24, width=2)
    d.text(360, cy - 34, 200, 24, "baseline B = 52 mm", fs=14, bold=True)
    # image planes give the focal length a referent: f is the measured
    # gap between each optical centre and its rectified image plane
    d.line(cLx - 50, 330, cRx + 50, 330, width=1.3, dashed=True)
    d.text(cLx - 50, 300, (cRx - cLx) + 100, 22, "rectified image planes",
           fs=13)
    d.line(cLx, 330, cLx, cy, width=1.2, dashed=True)
    d.line(cRx, 330, cRx, cy, width=1.2, dashed=True)
    d.line(232, 363, 232, 330, width=1.4, arrow="classic")
    d.line(232, 363, 232, cy, width=1.4, arrow="classic")
    d.text(34, 340, 188, 48,
           "focal length f &#8776; 1005 px<br>at 1280 px width", fs=13)
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
