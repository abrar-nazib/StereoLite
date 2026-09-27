"""Rebuild the defense-deck methodology diagram in drawio (replaces the
matplotlib version whose arrows rendered badly).

Emits presentation/figs/methodology_pipeline.drawio and renders it to
presentation/figs/methodology_pipeline.png via the diagrams.net viewer +
headless chromium (device_scale_factor=3) with a white-border trim.

Throwaway builder; imports the thesis drawio_lib without modifying it.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from PIL import Image

HERE = Path(__file__).resolve().parent          # presentation/figs
ROOT = HERE.parents[1]                          # repo root
sys.path.insert(0, str(ROOT / "thesis/book/figures/_src"))
from drawio_lib import D  # noqa: E402

# panel palette (matches the matplotlib original)
NAVY_F, NAVY_S = "#E7EBF4", "#1F2C4E"
PURP_F, PURP_S = "#ECE8F2", "#4F4472"
CRM_F, CRM_S = "#FBF1E5", "#C24A1C"
GRN_F, GRN_S = "#EAF2EB", "#3F7A48"
DETAIL = "#555555"

PW, PH = 490, 210        # panel size
INSET, BH, GAP = 14, 30, 12   # sub-box inset / height / vertical gap
FULL_BAND = 4 * BH + 3 * GAP  # content band height for a 4-row panel


def add_panel(d, x, y, title, fill, stroke, rows, hilite_last=False):
    """Rounded group panel with title + stacked sub-step boxes.

    rows: list of (bold_label, detail_or_empty). Returns (panel_id, box_ids).
    """
    panel = d.node(
        x, y, PW, PH, "",
        f"rounded=1;arcSize=6;html=1;fillColor={fill};"
        f"strokeColor={stroke};strokeWidth=2;")
    d.text(x, y + 6, PW, 28, f"<b>{title}</b>", fs=18, color=stroke)
    n = len(rows)
    content_h = n * BH + (n - 1) * GAP
    y0 = y + 42 + (FULL_BAND - content_h) // 2
    bw = PW - 2 * INSET
    ids = []
    for i, (lab, det) in enumerate(rows):
        by = y0 + i * (BH + GAP)
        bstroke, bwid = stroke, 1
        if hilite_last and i == n - 1:
            bstroke, bwid = CRM_S, 2
        bid = d.node(
            x + INSET, by, bw, BH, "",
            f"rounded=1;arcSize=12;html=1;fillColor=#FFFFFF;"
            f"strokeColor={bstroke};strokeWidth={bwid};")
        if det:
            d.text(x + INSET + 12, by, 220, BH, f"<b>{lab}</b>",
                   fs=14, align="left")
            d.text(x + PW - INSET - 252, by, 240, BH, det,
                   fs=12, color=DETAIL, align="right")
        else:
            d.text(x + INSET, by, bw, BH, f"<b>{lab}</b>", fs=14)
        ids.append(bid)
    for a, b in zip(ids, ids[1:]):
        d.edge(a, b, exit_=(0.5, 1), entry=(0.5, 0), color=stroke, width=1.5)
    return panel, ids


def build():
    d = D("methodology", 1100, 510)
    x_l, x_r, y_t, y_b = 20, 590, 20, 280

    p1, _ = add_panel(d, x_l, y_t, "Training", NAVY_F, NAVY_S, [
        ("Scene Flow", "35,454 pairs, full finalpass"),
        ("Crop + augment", "native 384x640"),
        ("Train", "60k steps, A100, OneCycle"),
        ("Checkpoint", "2.96 M params, 12 MB fp32"),
    ])
    p2, ids2 = add_panel(d, x_r, y_t, "Optimization + Export", PURP_F, PURP_S, [
        ("Graph surgery", "1.74x faster fp32"),
        ("ONNX export", ""),
        ("INT8 operator swaps", ""),
        ("TensorRT engine", "calibrated"),
    ], hilite_last=True)
    p3, _ = add_panel(d, x_l, y_b, "On-Device Inference", CRM_F, CRM_S, [
        ("Stereo camera", "AR0144, rectified pair"),
        ("Pre-process", "BGR to tensor, crop / pad"),
        ("StereoLite", "tile init, refine x 8, upsample"),
        ("Disparity map", "px resolution"),
    ])
    p4, _ = add_panel(d, x_r, y_b, "3D Reconstruction", GRN_F, GRN_S, [
        ("Triangulate", "Z = f B / d"),
        ("Outlier filter", "+ downsample"),
        ("Open3D point cloud", ""),
    ])

    # panel-to-panel arrows
    d.edge(p1, p2, exit_=(1, 0.5), entry=(0, 0.5), color=NAVY_S, width=2)
    d.edge(p3, p4, exit_=(1, 0.5), entry=(0, 0.5), color=CRM_S, width=2)
    # deploy elbow: TensorRT engine box bottom -> panel 3 top edge
    trt_cx = x_r + INSET + (PW - 2 * INSET) / 2   # 835
    p3_cx = x_l + PW / 2                          # 265
    y_mid = 255
    d.edge(ids2[-1], p3, exit_=(0.5, 1), entry=(0.5, 0), color=CRM_S,
           width=2, points=[(trt_cx, y_mid), (p3_cx, y_mid)])
    # edge label as free text (drawio_lib.edge hardcodes 10 px labels)
    d.text(470, 228, 160, 22, "<b><i>deploy engine</i></b>",
           fs=14, color=CRM_S)

    url = d.save(HERE / "methodology_pipeline.drawio")
    return url


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


def render(url):
    from playwright.sync_api import sync_playwright
    out = HERE / "methodology_pipeline.png"
    with sync_playwright() as p:
        br = p.chromium.launch()
        pg = br.new_page(device_scale_factor=3,
                         viewport={"width": 2600, "height": 1300})
        pg.goto(url, wait_until="networkidle", timeout=120000)
        pg.wait_for_selector(".geDiagramContainer svg, svg g g",
                             timeout=120000)
        pg.wait_for_timeout(5000)
        el = pg.query_selector(".geDiagramContainer svg") or \
            pg.query_selector("svg")
        if el is not None:
            el.screenshot(path=str(out), scale="device")
        else:
            pg.screenshot(path=str(out), scale="device")
        pg.close()
        br.close()
    trim_white(out)
    print("rendered", out, Image.open(out).size)


if __name__ == "__main__":
    render(build())
