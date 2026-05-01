"""High-resolution 3-way zoom comparison for the top loss candidates.

The standard mosaic shrinks each prediction to fit 9 tiles. This script
isolates 3 chosen variants and renders them larger, with a zoomed crop
on a region of interest so fine-detail differences are actually visible.

Default candidates: `L1_grad`, `cocktail_b05`, `stack_d1` (the three
contenders identified from the loss sweep). Override with --variants.

Run:
    python3 model/scripts/build_loss_zoom.py \
        --root model/benchmarks/loss_ablation_<TS> \
        --crop_region 80,170,360,500   # x0,y0,x1,y1 in pixels
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import cv2
import numpy as np


def latest_viz_step(viz_dir: Path) -> Path | None:
    rx = re.compile(r"step_(\d+)\.png")
    best, best_step = None, -1
    for f in viz_dir.glob("step_*.png"):
        m = rx.match(f.name)
        if m:
            s = int(m.group(1))
            if s > best_step:
                best_step = s
                best = f
    return best


def split_panel(panel: np.ndarray):
    H = panel.shape[0] // 2
    W = panel.shape[1] // 2
    return (panel[:H, :W], panel[:H, W:], panel[H:, :W])


def annotate(img, lines, scale=0.7, thick=2, color=(255, 255, 255), bg=True):
    out = img.copy()
    line_h = int(round(36 * scale / 0.7))
    bar_h = line_h * len(lines) + 8
    if bg:
        bg_layer = out.copy()
        cv2.rectangle(bg_layer, (0, 0), (out.shape[1], bar_h),
                       (0, 0, 0), -1)
        out = cv2.addWeighted(bg_layer, 0.6, out, 0.4, 0)
    for i, line in enumerate(lines):
        cv2.putText(out, line, (10, line_h * (i + 1) - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, scale, color, thick,
                    cv2.LINE_AA)
    return out


def upscale(img: np.ndarray, factor: int) -> np.ndarray:
    """Pixel-perfect nearest-neighbour upscaling for sharp inspection."""
    return cv2.resize(img, None, fx=factor, fy=factor,
                       interpolation=cv2.INTER_NEAREST)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True,
                    help="Path to loss_ablation_<TS>/ directory")
    ap.add_argument("--variants", default="L1_grad,cocktail_b05,stack_d1",
                    help="Comma-separated list of variant directory names")
    ap.add_argument("--crop_region",
                    default="120,160,360,360",
                    help="Crop coords x0,y0,x1,y1 (in pixels of original "
                         "640x384 frame). The default samples the centre-left "
                         "where the person/pole region typically lives.")
    ap.add_argument("--zoom", type=int, default=2,
                    help="Nearest-neighbour upscale factor for the crop")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    root = Path(args.root).resolve()
    out_path = Path(args.out) if args.out else (root / "comparison_zoom.png")
    variant_names = [v.strip() for v in args.variants.split(",")]
    x0, y0, x1, y1 = map(int, args.crop_region.split(","))

    # Collect.
    items = []
    for name in variant_names:
        vdir = root / name
        viz = vdir / "viz"
        if not viz.exists():
            print(f"  skip {name}: no viz dir at {viz}")
            continue
        panel_path = latest_viz_step(viz)
        if panel_path is None:
            print(f"  skip {name}: no step_*.png in {viz}")
            continue
        panel = cv2.imread(str(panel_path))
        if panel is None:
            print(f"  skip {name}: cannot read {panel_path}")
            continue
        meta_path = vdir / "meta.json"
        meta = {}
        if meta_path.exists():
            try:
                meta = json.loads(meta_path.read_text())
            except Exception:
                pass
        left, gt, pred = split_panel(panel)
        items.append((name, left, gt, pred, meta))
        print(f"  loaded {name:<14}  panel={panel_path.name}  shape={panel.shape}")

    if not items:
        raise SystemExit("no variants loaded")

    H, W = items[0][1].shape[:2]
    # Sanity-check the crop region.
    x0, x1 = max(0, x0), min(W, x1)
    y0, y1 = max(0, y0), min(H, y1)
    if x1 <= x0 or y1 <= y0:
        raise SystemExit(f"invalid crop {x0},{y0},{x1},{y1} for {W}x{H} frame")

    cw, ch = x1 - x0, y1 - y0
    print(f"frame {W}x{H}, crop {cw}x{ch} → upscale {args.zoom}× "
          f"to {cw*args.zoom}x{ch*args.zoom}")

    def boxed(img):
        out = img.copy()
        cv2.rectangle(out, (x0, y0), (x1, y1), (0, 255, 255), 2)
        return out

    # Header strip: full GT + full input with the crop region boxed.
    ref_left = annotate(boxed(items[0][1]),
                          ["INPUT (left, crop region in yellow)",
                           f"frame {W}x{H}"], scale=0.6)
    ref_gt = annotate(boxed(items[0][2]),
                        ["GROUND TRUTH (TURBO)", "crop region in yellow"],
                        scale=0.6)

    header_row = np.hstack([ref_left, ref_gt])

    # Per-variant: full pred (annotated) + zoomed crop.
    rows_full = []
    rows_zoom = []
    for name, _, _, pred, meta in items:
        fm = meta.get("final_metrics_all", {})
        epe = fm.get("epe", float("nan"))
        b05 = fm.get("bad_0.5", float("nan"))
        b1 = fm.get("bad_1.0", float("nan"))
        d1 = fm.get("d1_all", float("nan"))
        full_anno = annotate(boxed(pred), [
            f"{name}",
            f"EPE {epe:.3f}  bad-0.5 {b05:.1f}%  bad-1 {b1:.1f}%  D1 {d1:.1f}%",
        ], scale=0.6)
        rows_full.append(full_anno)

        # Crop & zoom.
        zoom = upscale(pred[y0:y1, x0:x1], args.zoom)
        gt_zoom = upscale(items[0][2][y0:y1, x0:x1], args.zoom)
        zoom_anno = annotate(zoom, [f"{name}  (zoom {args.zoom}×)"],
                               scale=0.65)
        rows_zoom.append(zoom_anno)

    # Layout: header [left | gt],
    #         full row of (pred for each variant) — concat horizontally,
    #         zoom row of (zoomed pred for each variant).
    full_strip = np.hstack(rows_full)
    zoom_strip = np.hstack(rows_zoom)

    # Add GT zoom at the start of the zoom strip.
    gt_zoom_anno = annotate(upscale(items[0][2][y0:y1, x0:x1], args.zoom),
                              [f"GT  (zoom {args.zoom}×)"], scale=0.65)
    zoom_strip = np.hstack([gt_zoom_anno, zoom_strip])

    # Pad/match widths.
    target_w = max(header_row.shape[1], full_strip.shape[1], zoom_strip.shape[1])

    def pad_to_width(img, w):
        if img.shape[1] >= w:
            return img[:, :w]
        pad = np.zeros((img.shape[0], w - img.shape[1], 3), dtype=img.dtype)
        return np.hstack([img, pad])

    header_row = pad_to_width(header_row, target_w)
    full_strip = pad_to_width(full_strip, target_w)
    zoom_strip = pad_to_width(zoom_strip, target_w)

    title_h = 40
    title = np.full((title_h, target_w, 3), (20, 20, 20), dtype=np.uint8)
    cv2.putText(title,
                f"Loss zoom comparison — {' / '.join(variant_names)}",
                (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.75,
                (220, 220, 220), 2, cv2.LINE_AA)

    out = np.vstack([title, header_row, full_strip, zoom_strip])
    cv2.imwrite(str(out_path), out)
    print(f"\nwrote {out_path}  ({out.shape[1]}x{out.shape[0]})")


if __name__ == "__main__":
    main()
