"""Expand a compact, hand-written element spec into a full .excalidraw file.

The design (positions, sizes, colors, text, arrows) is written by hand in a
spec module; this helper only fills in Excalidraw boilerplate, centres bound
text inside shapes, wires arrow bindings, and embeds the image assets from
_assets.json. Usage:

    python3 build.py spec_architecture.py   -> architecture.excalidraw

Spec modules define ELEMENTS (list of dicts) and OUT (file stem). Supported
compact keys per element:
  kind: rect | ellipse | diamond | text | line | poly | arrow | image
  id, x, y, w, h, text, size (font px), bold, color (text/stroke), fill,
  stroke, sw (stroke width), dash (bool), round (bool), align, font,
  points (for line/poly/arrow, relative), start/end (arrow binding ids),
  head (arrow end head: arrow|triangle|dot|null), tail (start head),
  asset (image asset name), group (group id string), label (arrow label text)
"""
from __future__ import annotations
import importlib.util, json, sys, pathlib, hashlib

HERE = pathlib.Path(__file__).parent
ASSETS = json.load(open(HERE / "_assets.json"))
FONT_DEFAULT = 2  # Helvetica: clean paper-figure look


def seed(s: str) -> int:
    return int(hashlib.md5(s.encode()).hexdigest()[:7], 16)


def base(e, kind, sid):
    return {
        "type": kind, "id": sid, "x": e["x"], "y": e["y"],
        "width": e.get("w", 0), "height": e.get("h", 0),
        "strokeColor": e.get("stroke", e.get("color", "#1e293b")),
        "backgroundColor": e.get("fill", "transparent"),
        "fillStyle": "solid", "strokeWidth": e.get("sw", 1.5),
        "strokeStyle": "dashed" if e.get("dash") else "solid",
        "roughness": 0, "opacity": e.get("opacity", 100), "angle": e.get("angle", 0),
        "seed": seed(sid), "version": 1, "versionNonce": seed(sid + "n"),
        "isDeleted": False, "groupIds": [e["group"]] if e.get("group") else [],
        "boundElements": None, "link": None, "locked": False,
    }


def text_el(e, sid, container=None, x=None, y=None, w=None, h=None):
    size = e.get("size", 14)
    lines = e.get("text", "").split("\n")
    lh = 1.25
    tw = w if w is not None else e.get("w", max(len(l) for l in lines) * size * 0.58 + 4)
    th = h if h is not None else len(lines) * size * lh
    t = base(e, "text", sid)
    t.update({
        "x": x if x is not None else e["x"], "y": y if y is not None else e["y"],
        "width": tw, "height": th,
        "text": e.get("text", ""), "originalText": e.get("text", ""),
        "fontSize": size, "fontFamily": e.get("font", FONT_DEFAULT),
        "textAlign": e.get("align", "center" if container else "left"),
        "verticalAlign": "middle" if container else "top",
        "strokeColor": e.get("color", "#1e293b"), "backgroundColor": "transparent",
        "strokeWidth": 1, "containerId": container, "lineHeight": lh,
        "autoResize": True,
    })
    if e.get("bold"):
        t["fontFamily"] = e.get("font", FONT_DEFAULT)
        t["text"] = t["originalText"] = e["text"]
    return t


def build(elements):
    out, by_id = [], {}
    for e in elements:
        kind, sid = e["kind"], e["id"]
        if kind in ("rect", "ellipse", "diamond"):
            el = base(e, {"rect": "rectangle", "ellipse": "ellipse", "diamond": "diamond"}[kind], sid)
            el["strokeWidth"] = e.get("sw", 1.5)
            if kind == "rect":
                el["roundness"] = {"type": 3} if e.get("round", True) else None
            else:
                el["roundness"] = {"type": 2}
            out.append(el); by_id[sid] = el
            if e.get("text"):
                size = e.get("size", 14)
                lines = e["text"].split("\n")
                th = len(lines) * size * 1.25
                tw = max(len(l) for l in lines) * size * 0.58 + 4
                tw = min(tw, e["w"] - 8)
                t = text_el(e, sid + "_t", container=sid,
                            x=e["x"] + (e["w"] - tw) / 2, y=e["y"] + (e["h"] - th) / 2, w=tw, h=th)
                t["strokeColor"] = e.get("tcolor", e.get("color", "#1e293b"))
                el["boundElements"] = [{"id": t["id"], "type": "text"}]
                out.append(t)
        elif kind == "text":
            t = text_el(e, sid)
            out.append(t); by_id[sid] = t
        elif kind in ("line", "poly"):
            el = base(e, "line", sid)
            pts = e["points"]
            if kind == "poly" and pts[0] != pts[-1]:
                pts = pts + [pts[0]]
            el["points"] = pts
            el["width"] = max(p[0] for p in pts) - min(p[0] for p in pts)
            el["height"] = max(p[1] for p in pts) - min(p[1] for p in pts)
            el["roundness"] = None
            out.append(el); by_id[sid] = el
        elif kind == "arrow":
            el = base(e, "arrow", sid)
            pts = e["points"]
            el["points"] = pts
            el["width"] = max(p[0] for p in pts) - min(p[0] for p in pts)
            el["height"] = max(p[1] for p in pts) - min(p[1] for p in pts)
            el["startArrowhead"] = e.get("tail")
            el["endArrowhead"] = e.get("head", "arrow")
            el["roundness"] = {"type": 2} if e.get("curve") else None
            el["elbowed"] = False
            el["startBinding"] = {"elementId": e["start"], "focus": 0, "gap": 2} if e.get("start") else None
            el["endBinding"] = {"elementId": e["end"], "focus": 0, "gap": 2} if e.get("end") else None
            for key in ("start", "end"):
                tid = e.get(key)
                if tid and tid in by_id:
                    tgt = by_id[tid]
                    tgt["boundElements"] = (tgt.get("boundElements") or []) + [{"id": sid, "type": "arrow"}]
            out.append(el); by_id[sid] = el
            if e.get("label"):
                size = e.get("size", 11)
                lines = e["label"].split("\n")
                tw = max(len(l) for l in lines) * size * 0.58 + 4
                th = len(lines) * size * 1.25
                mx = e["x"] + sum(p[0] for p in pts) / len(pts)
                my = e["y"] + sum(p[1] for p in pts) / len(pts)
                # free-floating label (not bound): no white box over the arrow
                lab = text_el({"text": e["label"], "size": size, "color": e.get("lcolor", e.get("color", "#1e293b")),
                               "x": mx - tw / 2 + e.get("ldx", 0), "y": my - th / 2 + e.get("ldy", 0), "align": "center"},
                              sid + "_l")
                lab["textAlign"] = "center"
                out.append(lab)
        elif kind == "image":
            a = ASSETS[e["asset"]]
            el = base(e, "image", sid)
            el.update({"fileId": a["id"], "status": "saved", "scale": [1, 1],
                       "strokeColor": "transparent", "roundness": None})
            out.append(el); by_id[sid] = el
        else:
            raise ValueError(kind)
    return out


def main():
    spec_path = pathlib.Path(sys.argv[1])
    spec = importlib.util.spec_from_file_location("spec", spec_path)
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    elements = build(m.ELEMENTS)
    used = {ASSETS[e["asset"]]["id"]: ASSETS[e["asset"]] for e in m.ELEMENTS if e.get("kind") == "image"}
    doc = {"type": "excalidraw", "version": 2, "source": "https://excalidraw.com",
           "elements": elements,
           "appState": {"viewBackgroundColor": "#ffffff", "gridSize": 20},
           "files": {k: {kk: v[kk] for kk in ("id", "dataURL", "mimeType", "created")} for k, v in used.items()}}
    out = HERE / f"{m.OUT}.excalidraw"
    json.dump(doc, open(out, "w"))
    print(out, len(elements), "elements")


if __name__ == "__main__":
    main()
