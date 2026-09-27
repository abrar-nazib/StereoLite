"""Fig 3.1 (drawio edition): full StereoLite architecture overview.

Emits an EDITABLE draw.io file (thesis/book/figures/fig_3_1_architecture.drawio)
following the visual grammar of published stereo architecture figures
(IGEV Fig 3, BANet Fig 2, CREStereo Fig 2, DEFOM Fig 3, StereoAnywhere Fig 2):

- strictly orthogonal (Manhattan) edge routing, no diagonals;
- encoders drawn as layer-stack bar glyphs, module names OUTSIDE below;
- correlation/geometry volumes as 3-D cuboids, labels below;
- 3-D regularization as a bar stack (IGEV-style);
- numbered red stage markers (1)..(5) tying blocks to Ch 3 subsections;
- real image thumbnails (genuine L/R pair + the model's own prediction);
- operator circle, dashed lines for guidance/gating, legend row.

Also emits a companion print of the viewer URL so the diagram can be
rendered headlessly (diagrams.net viewer) for the visual QA loop.
"""
from __future__ import annotations

import base64
import io
import re
import zlib
from pathlib import Path
from urllib.parse import quote
from xml.sax.saxutils import escape

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[4]
ASSETS = ROOT / "model/benchmarks/thesis_assets"
RUN = ROOT / "model/benchmarks/20260704_fullsf_gev4onp_nc"
OUT = ROOT / "thesis/book/figures"
SCENE = "TEST_A_0000_t09"

# drawio default pastel palette
GREY_F, GREY_S = "#f5f5f5", "#666666"
ORAN_F, ORAN_S = "#ffe6cc", "#d79b00"
RED_F, RED_S = "#f8cecc", "#b85450"
GREEN_F, GREEN_S = "#d5e8d4", "#82b366"
LAV_F, LAV_S = "#e1d5e7", "#9673a6"
SUP = "#c1121f"
EDGE = "#333333"
FONT = "Times New Roman"
FONT_SCALE = 1.25


def _jpeg_b64(img: Image.Image, width: int = 320) -> str:
    img = img.convert("RGB")
    h = round(img.height * width / img.width)
    img = img.resize((width, h))
    buf = io.BytesIO()
    img.save(buf, "JPEG", quality=82)
    return base64.b64encode(buf.getvalue()).decode()


def _disp_jpeg_b64(d: np.ndarray) -> str:
    valid = d > 0
    vmax = np.percentile(d[valid], 98) if valid.any() else 1.0
    rgb = plt.get_cmap("turbo")(np.clip(d / max(vmax, 1e-6), 0, 1))[..., :3]
    rgb[~valid] = 0.12
    return _jpeg_b64(Image.fromarray((rgb * 255).astype(np.uint8)))


class D:
    """Tiny drawio XML builder."""

    def __init__(self):
        self.cells: list[str] = []
        self.n = 1

    def _id(self) -> str:
        self.n += 1
        return f"c{self.n}"

    def node(self, x, y, w, h, value="", style="") -> str:
        i = self._id()
        if "fontFamily=" not in style:
            style = style + f"fontFamily={FONT};"
        self.cells.append(
            f'<mxCell id="{i}" value="{escape(value, {chr(34): "&quot;"})}" '
            f'style="{style}" vertex="1" parent="1">'
            f'<mxGeometry x="{x}" y="{y}" width="{w}" height="{h}" '
            f'as="geometry"/></mxCell>')
        return i

    def edge(self, src, dst, points=(), value="", dashed=False,
             color=EDGE, exit_=None, entry=None, width=1.5):
        i = self._id()
        style = ("edgeStyle=orthogonalEdgeStyle;rounded=0;html=1;"
                 f"jettySize=auto;strokeColor={color};strokeWidth={width};"
                 f"fontSize=11;fontFamily={FONT};")
        if dashed:
            style += "dashed=1;"
        if exit_:
            style += f"exitX={exit_[0]};exitY={exit_[1]};exitDx=0;exitDy=0;"
        if entry:
            style += f"entryX={entry[0]};entryY={entry[1]};entryDx=0;entryDy=0;"
        pts = "".join(f'<mxPoint x="{px}" y="{py}"/>' for px, py in points)
        arr = f'<Array as="points">{pts}</Array>' if pts else ""
        self.cells.append(
            f'<mxCell id="{i}" value="{escape(value, {chr(34): "&quot;"})}" '
            f'style="{style}" edge="1" parent="1" source="{src}" '
            f'target="{dst}"><mxGeometry relative="1" as="geometry">{arr}'
            f'</mxGeometry></mxCell>')
        return i

    def xml(self) -> str:
        body = "".join(self.cells)
        body = re.sub(
            r"fontSize=(\d+)",
            lambda match: f"fontSize={round(int(match.group(1)) * FONT_SCALE)}",
            body,
        )
        return ('<mxfile host="app.diagrams.net"><diagram id="fig31" '
                'name="Fig 3.1"><mxGraphModel dx="1200" dy="800" grid="0" '
                'gridSize="10" guides="1" tooltips="1" connect="1" '
                'arrows="1" fold="1" page="1" pageScale="1" '
                'pageWidth="1650" pageHeight="780" math="0" shadow="0">'
                '<root><mxCell id="0"/><mxCell id="1" parent="0"/>'
                f'{body}</root></mxGraphModel></diagram></mxfile>')


def _bars(d: D, x, y_base, heights, fill, stroke, w=16, gap=10):
    """IGEV-style layer stack: bars vertically centered on y_base."""
    ids = []
    for i, h in enumerate(heights):
        ids.append(d.node(x + i * (w + gap), y_base - h / 2, w, h, "",
                          f"rounded=1;arcSize=30;html=1;fillColor={fill};"
                          f"strokeColor={stroke};fontFamily={FONT};"))
    return ids


def _text(d: D, x, y, w, h, value, size=11, bold=False, color="#000000",
          align="center"):
    fs = f"fontStyle=1;" if bold else ""
    return d.node(x, y, w, h, value,
                  f"text;html=1;align={align};verticalAlign=middle;"
                  f"fontSize={size};{fs}fontColor={color};"
                  f"fontFamily={FONT};")


def _stage(d: D, x, y, n):
    _text(d, x, y, 36, 20, f"({n})", size=13, bold=True, color="#cc0000")


def _sup_dot(d: D, x, y, label, lx, ly, lw=70):
    d.node(x, y, 10, 10, "",
           f"ellipse;html=1;fillColor={SUP};strokeColor=#8a0000;")
    _text(d, lx, ly, lw, 16,
          f'<i><font color="{SUP}">L</font></i>'
          f'<sub><font color="{SUP}">{label}</font></sub>', size=10)


def main():
    left_b64 = _jpeg_b64(Image.open(ASSETS / f"{SCENE}_left.png"))
    right_b64 = _jpeg_b64(Image.open(ASSETS / f"{SCENE}_right.png"))
    pred = np.array(Image.open(RUN / "images/val_00/step_053000.png")
                    ).astype(np.float32) / 256.0
    out_b64 = _disp_jpeg_b64(pred)

    d = D()
    img_style = ("shape=image;html=1;imageAspect=0;"
                 "verticalLabelPosition=bottom;verticalAlign=top;"
                 "fontSize=12;image=data:image/jpeg,{};")

    # ------- title -------
    _text(d, 1150, 18, 460, 26, "StereoLite · 2.96 M parameters",
          size=15, bold=True, align="right")

    # ------- inputs -------
    img_l = d.node(40, 70, 160, 90, "Left image", img_style.format(left_b64))
    img_r = d.node(40, 240, 160, 90, "Right image",
                   img_style.format(right_b64))

    # ------- shared encoder (bar stack) -------
    enc = _bars(d, 280, 200, [160, 130, 100, 76], GREY_F, GREY_S)
    _text(d, 230, 300, 220, 36,
          "<b>Shared Feature Encoder</b><br>YOLO26s layers 0-6 · pretrained",
          size=12)
    _stage(d, 300, 88, 1)
    d.edge(img_l, enc[0], points=[(240, 115), (240, 160)],
           exit_=(1, 0.5), entry=(0, 0.25))
    d.edge(img_r, enc[0], points=[(240, 285), (240, 240)],
           exit_=(1, 0.5), entry=(0, 0.75))

    # ------- cost volume 1/16 + 3-D CNN + init -------
    cube16 = d.node(490, 155, 110, 85,
                    "<b>Cost Volume 1/16</b><br>8 groups × 24 disparities",
                    f"shape=cube;whiteSpace=wrap;html=1;boundedLbl=1;"
                    f"backgroundOutline=1;darkOpacity=0.05;darkOpacity2=0.1;"
                    f"fillColor={ORAN_F};strokeColor={ORAN_S};"
                    f"verticalLabelPosition=bottom;verticalAlign=top;"
                    f"fontSize=11;")
    _stage(d, 520, 120, 2)
    d.edge(enc[3], cube16, value="f16 (L, R)", exit_=(1, 0.5),
           entry=(0, 0.5))

    reg = _bars(d, 660, 197, [70, 54, 40], RED_F, RED_S, w=14, gap=8)
    _text(d, 620, 250, 140, 34, "<b>3-D CNN</b><br>8-16-16-1 + softmax",
          size=11)
    d.edge(cube16, reg[0], exit_=(1, 0.5), entry=(0, 0.5))

    init = d.node(770, 160, 170, 80,
                  "<b>Tile-State Init</b><br>T = (d, sx, sy, h, c)<br>"
                  "d0 soft-argmax · h ← ctx16",
                  f"rounded=1;whiteSpace=wrap;html=1;fillColor={GREY_F};"
                  f"strokeColor={GREY_S};fontSize=11;")
    d.edge(reg[2], init, exit_=(1, 0.5), entry=(0, 0.5))

    # ------- recurrent ladder -------
    gru_style = (f"rounded=1;whiteSpace=wrap;html=1;fillColor={GREEN_F};"
                 f"strokeColor={GREEN_S};fontSize=11;")
    gru16 = d.node(490, 440, 140, 70,
                   "<b>ConvGRU × 2</b><br>1/16 · corr ±2", gru_style)
    gru8 = d.node(700, 440, 140, 70,
                  "<b>ConvGRU × 3</b><br>1/8 · corr ±2", gru_style)
    plus = d.node(905, 458, 34, 34, "⊕",
                  "ellipse;html=1;fontSize=17;fillColor=#ffffff;"
                  "strokeColor=#333333;")
    gru4 = d.node(1000, 440, 140, 70,
                  "<b>ConvGRU × 3</b><br>1/4 · corr ±2", gru_style)
    ups = d.node(1210, 440, 175, 70,
                 "<b>Plane Render +</b><br><b>Convex Upsample</b><br>"
                 "1/4 → 1/2 → full",
                 f"rounded=1;whiteSpace=wrap;html=1;fillColor={GREY_F};"
                 f"strokeColor={GREY_S};fontSize=11;")
    img_out = d.node(1440, 430, 160, 90, "Final disparity",
                     img_style.format(out_b64))
    _stage(d, 495, 412, 3)
    _stage(d, 1240, 412, 5)

    # T0 drop: init bottom -> around the left -> ladder start
    # (lane y=360 stays clear of the encoder caption at y=300..336)
    d.edge(init, gru16, points=[(842, 360), (420, 360), (420, 475)],
           exit_=(0.42, 1), entry=(0, 0.5))
    _text(d, 428, 400, 30, 18, "T<sub>0</sub>", size=11)

    d.edge(gru16, gru8, exit_=(1, 0.5), entry=(0, 0.5))
    _text(d, 628, 444, 76, 16, "plane ↑2", size=9)
    d.edge(gru8, plus, exit_=(1, 0.5), entry=(0, 0.5))
    _text(d, 838, 444, 70, 16, "plane ↑2", size=9)
    d.edge(plus, gru4, exit_=(1, 0.5), entry=(0, 0.5))
    d.edge(gru4, ups, exit_=(1, 0.5), entry=(0, 0.5))
    d.edge(ups, img_out, exit_=(1, 0.5), entry=(0, 0.5))

    # ------- GEV branch (over-the-top rail, StereoAnywhere-style) -------
    gev = d.node(900, 280, 105, 80,
                 "<b>GEV 1/4</b> · 8 groups<br>±16 band around d",
                 f"shape=cube;whiteSpace=wrap;html=1;boundedLbl=1;"
                 f"backgroundOutline=1;darkOpacity=0.05;darkOpacity2=0.1;"
                 f"fillColor={ORAN_F};strokeColor={ORAN_S};"
                 f"verticalLabelPosition=bottom;verticalAlign=top;"
                 f"fontSize=11;")
    _stage(d, 950, 252, 4)
    d.edge(enc[3], gev, points=[(366, 95), (952, 95)], value="f4 (L, R)",
           dashed=True, exit_=(0.5, 0), entry=(0.5, 0))
    gate = d.node(1060, 285, 150, 70,
                  "<b>Fail-Soft Fusion</b><br>w = σ(F(·))",
                  f"rounded=1;whiteSpace=wrap;html=1;fillColor={GREY_F};"
                  f"strokeColor={GREY_S};fontSize=11;")
    d.edge(gev, gate, exit_=(1, 0.5), entry=(0, 0.5))
    # gate output into the fusion node
    d.edge(gate, plus, points=[(1135, 418), (922, 418)],
           value="gated correction",
           dashed=True, exit_=(0.5, 1), entry=(0.5, 0))
    # narrow-band center tap: current tile d into the GEV (left face,
    # so the arrow does not pierce the cube caption below it)
    d.edge(gru8, gev, points=[(770, 400), (870, 400), (870, 336)],
           dashed=True, exit_=(0.5, 0), entry=(0, 0.7))

    # ------- context stream -------
    ctx = _bars(d, 60, 592, [64, 48, 36], LAV_F, LAV_S, w=14, gap=8)
    _text(d, 10, 636, 200, 34,
          "<b>Context Encoder</b><br>left image only · 32 ch", size=12)
    d.edge(img_l, ctx[0], points=[(20, 115), (20, 592)], dashed=True,
           color=LAV_S, exit_=(0, 0.5), entry=(0, 0.5))
    for gid, gx in ((gru16, 560), (gru8, 770), (gru4, 1070)):
        d.edge(ctx[2], gid, points=[(gx, 592)], dashed=True, color=LAV_S,
               exit_=(1, 0.5), entry=(0.5, 1))
    _text(d, 380, 598, 200, 18, "ctx16 · ctx8 · ctx4", size=10,
          color=LAV_S)

    # ------- supervision dots -------
    _sup_dot(d, 655, 470, "1/16", 630, 486)
    _sup_dot(d, 862, 470, "1/8", 837, 486)
    _sup_dot(d, 1168, 470, "1/4", 1143, 486)
    _sup_dot(d, 1025, 315, "gev", 1000, 331)
    _sup_dot(d, 1405, 470, "1/2, full", 1370, 486, lw=90)

    # ------- legend -------
    ly = 690
    def swatch(x, fill, stroke, label, w=200):
        d.node(x, ly, 16, 16, "",
               f"rounded=1;html=1;fillColor={fill};strokeColor={stroke};")
        _text(d, x + 22, ly - 2, w, 20, label, size=10, align="left")
    swatch(40, GREY_F, GREY_S, "feature encoder (2-D conv)")
    swatch(250, LAV_F, LAV_S, "context encoder")
    swatch(420, ORAN_F, ORAN_S, "correlation volume")
    swatch(600, RED_F, RED_S, "3-D convolution")
    swatch(770, GREEN_F, GREEN_S, "recurrent update (ConvGRU)")
    d.node(1000, ly + 3, 10, 10, "",
           f"ellipse;html=1;fillColor={SUP};strokeColor=#8a0000;")
    _text(d, 1018, ly - 2, 230, 20, "supervised output (multi-scale loss)",
          size=10, align="left")

    xml = d.xml()
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "fig_3_1_architecture.drawio").write_text(xml)
    print("saved", OUT / "fig_3_1_architecture.drawio",
          f"({len(xml)//1024} KB)")

    # viewer URL for the headless visual QA loop
    co = zlib.compressobj(9, zlib.DEFLATED, -15)
    raw = co.compress(xml.encode()) + co.flush()
    data = quote(base64.b64encode(raw).decode(), safe="")
    url = f"https://viewer.diagrams.net/?lightbox=1&nav=0&title=fig31#R{data}"
    (OUT / "_src/fig31_viewer_url.txt").write_text(url)
    print("viewer URL written to _src/fig31_viewer_url.txt",
          f"({len(url)//1024} KB)")


if __name__ == "__main__":
    main()
