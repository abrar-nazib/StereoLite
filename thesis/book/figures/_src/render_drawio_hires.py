"""Render selected drawio diagrams to crisp high-res PNG previews via the
diagrams.net viewer (headless chromium, deviceScaleFactor=3). Crops the
white border. Rebuilds fig_2_2_timeline and fig_3_5_plane_prop only."""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
from PIL import Image

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import make_drawio_batch as B  # noqa: E402

OUT = HERE.parent  # thesis/book/figures
TARGETS = ["fig_2_2_timeline", "fig_3_5_plane_prop"]


def trim_white(png, pad=18):
    im = np.array(Image.open(png).convert("RGB")).astype(int)
    bright = im.mean(2)
    rng = im.max(2) - im.min(2)
    ink = (bright < 200) | (rng > 12)  # dark or colored; excludes bg
    ys, xs = np.where(ink)
    if len(ys) == 0:
        return
    y0, y1 = max(0, ys.min() - pad), min(im.shape[0], ys.max() + pad)
    x0, x1 = max(0, xs.min() - pad), min(im.shape[1], xs.max() + pad)
    Image.fromarray(im[y0:y1, x0:x1].astype("uint8")).save(png)


def main():
    B.fig_2_2_timeline()
    B.fig_3_5_plane_prop()
    from playwright.sync_api import sync_playwright
    with sync_playwright() as p:
        br = p.chromium.launch()
        for name in TARGETS:
            url = B.URLS[name]
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
