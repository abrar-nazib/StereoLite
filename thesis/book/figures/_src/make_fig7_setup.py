"""Figure 7.x: Simulation and experimental setup dataflow.
Datasets -> model + training -> evaluation/deployment platforms -> results.
Built with the shared drawio_lib grammar; rendered to a crisp PNG preview
via the diagrams.net viewer (headless chromium), matching the pipeline of
render_drawio_hires.py. Research-figure conventions: dashed grouping
containers with lane titles, one accent colour per functional role, a
single highlighted deployment target, orthogonal flow arrows.
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
from PIL import Image

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from drawio_lib import (  # noqa: E402
    D, EDGE, BLUE_F, BLUE_S, GREEN_F, GREEN_S, ORAN_F, ORAN_S,
    LAV_F, LAV_S, GREY_F, GREY_S, YEL_F, YEL_S, RED_F, RED_S)

OUT = HERE.parent  # thesis/book/figures


def build():
    d = D("fig_7_setup", 1180, 560)
    HDR = "#333333"

    # ---- lane title helper ----
    def title(x, w, t):
        d.text(x, 18, w, 22, t, fs=13, bold=True, color=HDR)

    # ======== Lane 1: Data sources ========
    title(40, 250, "Data sources")
    d.box(38, 50, 254, 392, "", fill="#fbfbfb", stroke="#bbbbbb", dashed=True)
    ds = []
    ds.append(d.box(60, 74, 210, 92,
        "SceneFlow FinalPass<br>(synthetic, dense GT)<br>"
        "<font color='#555'>35,454 train · 4,370 test<br>400 val / perturbation</font>",
        fill=BLUE_F, stroke=BLUE_S, fs=11))
    ds.append(d.box(60, 196, 210, 82,
        "Middlebury 2014<br>(real, zero-shot)<br>"
        "<font color='#555'>23 unseen indoor scenes</font>",
        fill=GREEN_F, stroke=GREEN_S, fs=11))
    ds.append(d.box(60, 308, 210, 92,
        "Physical stereo rig<br>2560 &#215; 720 side-by-side<br>"
        "<font color='#555'>real rectified pairs, no GT</font>",
        fill=ORAN_F, stroke=ORAN_S, fs=11))

    # ======== Lane 2: Model + training ========
    title(360, 200, "Model and training")
    model = d.box(372, 96, 190, 92,
        "<b>StereoLite</b><br>2.96 M parameters<br>"
        "<font color='#555'>compact recurrent<br>tile-plane network</font>",
        fill=LAV_F, stroke=LAV_S, fs=12)
    a100 = d.box(372, 250, 190, 78,
        "Cloud A100 80 GB<br>"
        "<font color='#555'>full-dataset training<br>(supervised, SceneFlow)</font>",
        fill=GREY_F, stroke=GREY_S, fs=11)

    # ======== Lane 3: Evaluation and deployment ========
    title(628, 268, "Evaluation and deployment")
    d.box(626, 50, 272, 440, "", fill="#fbfbfb", stroke="#bbbbbb", dashed=True)
    t4 = d.box(648, 74, 228, 78,
        "Cloud T4<br>"
        "<font color='#555'>zero-shot and ablation<br>accuracy evaluation</font>",
        fill=GREY_F, stroke=GREY_S, fs=11)
    rtx = d.box(648, 176, 228, 78,
        "RTX 3050 laptop<br>"
        "<font color='#555'>latency benchmark (fp16)</font>",
        fill=GREY_F, stroke=GREY_S, fs=11)
    jet = d.box(648, 278, 228, 88,
        "<b>Jetson Orin Nano</b><br>"
        "<font color='#555'>INT8 deployment target<br>on-device latency</font>",
        fill=RED_F, stroke=RED_S, fs=12)
    drv = d.box(648, 390, 228, 78,
        "Scripted evaluation drivers<br>"
        "<font color='#555'>full-test · MB14 · rectification sweep</font>",
        fill=YEL_F, stroke=YEL_S, fs=10)

    # ======== Lane 4: Results ========
    title(940, 210, "Results")
    res = d.box(952, 150, 196, 96,
        "Metrics and metadata<br>"
        "<font color='#555'>EPE, bad-<i>t</i>, D1-all<br>logged per run</font>",
        fill="#eef6ff", stroke=BLUE_S, fs=11)
    d.text(952, 268, 196, 24, "&#8594;&nbsp; Chapter 8", fs=12, bold=True,
           color=HDR)

    # ---- edges ----
    for s in ds:
        d.edge(s, model, color=EDGE, width=1.4,
               exit_=(1, 0.5), entry=(0, 0.5))
    d.edge(model, a100, value="train", color=EDGE, width=1.6,
           exit_=(0.5, 1), entry=(0.5, 0))
    d.edge(a100, t4, value="checkpoint", color=EDGE, width=1.6,
           exit_=(1, 0.4), entry=(0, 0.5))
    d.edge(a100, jet, color=EDGE, width=1.4,
           exit_=(1, 0.7), entry=(0, 0.5))
    d.edge(t4, res, color=EDGE, width=1.4, exit_=(1, 0.5), entry=(0, 0.35))
    d.edge(drv, res, color=EDGE, width=1.2, dashed=True,
           exit_=(1, 0.5), entry=(0, 0.75))
    return d


def trim(png, pad=18):
    im = np.array(Image.open(png).convert("RGB")).astype(int)
    ink = (im.mean(2) < 200) | (im.max(2) - im.min(2) > 12)
    ys, xs = np.where(ink)
    if len(ys) == 0:
        return
    y0, y1 = max(0, ys.min() - pad), min(im.shape[0], ys.max() + pad)
    x0, x1 = max(0, xs.min() - pad), min(im.shape[1], xs.max() + pad)
    Image.fromarray(im[y0:y1, x0:x1].astype("uint8")).save(png)


def main():
    d = build()
    url = d.save(OUT / "fig_7_setup.drawio")
    from playwright.sync_api import sync_playwright
    with sync_playwright() as p:
        br = p.chromium.launch()
        pg = br.new_page(device_scale_factor=3,
                         viewport={"width": 1700, "height": 900})
        pg.goto(url, wait_until="networkidle", timeout=90000)
        pg.wait_for_timeout(4000)
        out = OUT / "fig_7_setup_preview.png"
        el = pg.query_selector("svg")
        (el or pg).screenshot(path=str(out))
        pg.close()
        br.close()
    trim(out)
    print("rendered", out, Image.open(out).size)


if __name__ == "__main__":
    main()
