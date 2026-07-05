"""Shared draw.io XML emitter for thesis architecture figures.
Extracted from make_fig31_drawio.py; every diagram follows the grammar
contract in WRITING_PLAN section 9."""
from __future__ import annotations

import base64
import zlib
from urllib.parse import quote
from xml.sax.saxutils import escape

# drawio pastel palette
GREY_F, GREY_S = "#f5f5f5", "#666666"
ORAN_F, ORAN_S = "#ffe6cc", "#d79b00"
RED_F, RED_S = "#f8cecc", "#b85450"
GREEN_F, GREEN_S = "#d5e8d4", "#82b366"
LAV_F, LAV_S = "#e1d5e7", "#9673a6"
BLUE_F, BLUE_S = "#dae8fc", "#6c8ebf"
YEL_F, YEL_S = "#fff2cc", "#d6b656"
SUP = "#c1121f"
EDGE = "#333333"


class D:
    """Tiny drawio XML builder (one page)."""

    def __init__(self, name: str = "fig", w: int = 1200, h: int = 700):
        self.cells: list[str] = []
        self.n = 1
        self.name = name
        self.w, self.h = w, h

    def _id(self) -> str:
        self.n += 1
        return f"c{self.n}"

    def node(self, x, y, w, h, value="", style="") -> str:
        i = self._id()
        self.cells.append(
            f'<mxCell id="{i}" value="{escape(value, {chr(34): "&quot;"})}" '
            f'style="{style}" vertex="1" parent="1">'
            f'<mxGeometry x="{x}" y="{y}" width="{w}" height="{h}" '
            f'as="geometry"/></mxCell>')
        return i

    def box(self, x, y, w, h, value="", fill=GREY_F, stroke=GREY_S,
            fs=11, rounded=1, dashed=False, align="center"):
        st = (f"rounded={rounded};whiteSpace=wrap;html=1;fillColor={fill};"
              f"strokeColor={stroke};fontSize={fs};align={align};")
        if dashed:
            st += "dashed=1;"
        return self.node(x, y, w, h, value, st)

    def cube(self, x, y, w, h, value="", fill=ORAN_F, stroke=ORAN_S, fs=11):
        return self.node(
            x, y, w, h, value,
            f"shape=cube;whiteSpace=wrap;html=1;boundedLbl=1;"
            f"backgroundOutline=1;darkOpacity=0.05;darkOpacity2=0.1;"
            f"fillColor={fill};strokeColor={stroke};"
            f"verticalLabelPosition=bottom;verticalAlign=top;fontSize={fs};")

    def bars(self, x, y_mid, heights, fill, stroke, w=14, gap=9):
        ids = []
        for i, h in enumerate(heights):
            ids.append(self.node(
                x + i * (w + gap), y_mid - h / 2, w, h, "",
                f"rounded=1;arcSize=30;html=1;fillColor={fill};"
                f"strokeColor={stroke};"))
        return ids

    def text(self, x, y, w, h, value, fs=11, bold=False, color="#000000",
             align="center"):
        b = "fontStyle=1;" if bold else ""
        return self.node(x, y, w, h, value,
                         f"text;html=1;align={align};verticalAlign=middle;"
                         f"fontSize={fs};{b}fontColor={color};")

    def ellipse(self, x, y, w, h, value="", fill="#ffffff", fs=14):
        return self.node(x, y, w, h, value,
                         f"ellipse;html=1;fontSize={fs};fillColor={fill};"
                         f"strokeColor={EDGE};")

    def dot(self, x, y, r=5):
        return self.node(x - r, y - r, 2 * r, 2 * r, "",
                         f"ellipse;html=1;fillColor={SUP};"
                         f"strokeColor=#8a0000;")

    def edge(self, src, dst, points=(), value="", dashed=False,
             color=EDGE, exit_=None, entry=None, width=1.5,
             orthogonal=True, arrow=True):
        i = self._id()
        st = (f"html=1;jettySize=auto;strokeColor={color};"
              f"strokeWidth={width};fontSize=10;")
        if orthogonal:
            st = "edgeStyle=orthogonalEdgeStyle;rounded=0;" + st
        if not arrow:
            st += "endArrow=none;"
        if dashed:
            st += "dashed=1;"
        if exit_:
            st += f"exitX={exit_[0]};exitY={exit_[1]};exitDx=0;exitDy=0;"
        if entry:
            st += f"entryX={entry[0]};entryY={entry[1]};entryDx=0;entryDy=0;"
        pts = "".join(f'<mxPoint x="{px}" y="{py}"/>' for px, py in points)
        arr = f'<Array as="points">{pts}</Array>' if pts else ""
        self.cells.append(
            f'<mxCell id="{i}" value="{escape(value, {chr(34): "&quot;"})}" '
            f'style="{st}" edge="1" parent="1" source="{src}" '
            f'target="{dst}"><mxGeometry relative="1" as="geometry">{arr}'
            f'</mxGeometry></mxCell>')
        return i

    def line(self, x0, y0, x1, y1, color=EDGE, width=1.3, dashed=False,
             arrow="none"):
        """Free line between coordinates (for geometry schematics)."""
        i = self._id()
        st = (f"html=1;strokeColor={color};strokeWidth={width};"
              f"endArrow={arrow};")
        if dashed:
            st += "dashed=1;"
        self.cells.append(
            f'<mxCell id="{i}" style="{st}" edge="1" parent="1">'
            f'<mxGeometry relative="1" as="geometry">'
            f'<mxPoint x="{x0}" y="{y0}" as="sourcePoint"/>'
            f'<mxPoint x="{x1}" y="{y1}" as="targetPoint"/>'
            f'</mxGeometry></mxCell>')
        return i

    def xml(self) -> str:
        body = "".join(self.cells)
        return ('<mxfile host="app.diagrams.net"><diagram id="d1" '
                f'name="{self.name}"><mxGraphModel dx="1000" dy="600" '
                'grid="0" gridSize="10" guides="1" tooltips="1" connect="1" '
                f'arrows="1" fold="1" page="1" pageScale="1" '
                f'pageWidth="{self.w}" pageHeight="{self.h}" math="0" '
                'shadow="0"><root><mxCell id="0"/>'
                '<mxCell id="1" parent="0"/>'
                f'{body}</root></mxGraphModel></diagram></mxfile>')

    def save(self, path):
        xml = self.xml()
        path.write_text(xml)
        co = zlib.compressobj(9, zlib.DEFLATED, -15)
        raw = co.compress(xml.encode()) + co.flush()
        data = quote(base64.b64encode(raw).decode(), safe="")
        return (f"https://viewer.diagrams.net/?lightbox=1&nav=0&"
                f"title={path.stem}#R{data}")
