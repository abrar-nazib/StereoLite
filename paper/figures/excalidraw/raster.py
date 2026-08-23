"""Rasterize NAME.svg (exported from the browser) to NAME.png at 2x with resvg
(system-font fallback per glyph), plus optional zoomed crops for inspection.

    python3 raster.py architecture [x0,y0,x1,y1 ...]   (crop boxes in diagram units)
"""
import sys, pathlib
import resvg_py
from PIL import Image

HERE = pathlib.Path(__file__).parent
name = sys.argv[1]
svg = HERE / f"{name}.svg"
png = HERE / f"{name}.png"
svg_txt = svg.read_text().replace("Helvetica, Segoe UI Emoji", "Nimbus Sans, DejaVu Sans")
data = resvg_py.svg_to_bytes(svg_string=svg_txt, zoom=2, background="white",
                             font_family="Nimbus Sans", sans_serif_family="Nimbus Sans")
png.write_bytes(bytes(data))
im = Image.open(png)
print(png, im.size)
# figure width in diagram units is the svg width attribute
import re
w_units = float(re.search(r'width="([\d.]+)"', svg.read_text()[:600]).group(1))
s = im.width / w_units
for k, box in enumerate(sys.argv[2:]):
    x0, y0, x1, y1 = [float(v) for v in box.split(",")]
    out = HERE / f"_crop_{name}_{k}.png"
    im.crop((int(x0 * s), int(y0 * s), int(x1 * s), int(y1 * s))).save(out)
    print(out)
