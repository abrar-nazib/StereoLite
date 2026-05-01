"""Build a side-by-side visual mosaic of loss-variant predictions.

Reads the final viz panel (highest-step `step_*.png`) from each
variant's viz directory, extracts the predicted-disparity quadrant,
stacks them in a grid with the left image and GT on top for reference,
and annotates each variant tile with its name and headline metrics
(EPE, bad-0.5, bad-1.0).

Run after a loss sweep completes:
    python3 model/scripts/build_loss_mosaic.py \
        --root model/benchmarks/loss_ablation_<TS>

Saves the mosaic to <root>/comparison_mosaic.png.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import cv2
import numpy as np


# Order matters: define the column ordering for the mosaic. Variants not
# present in a run are silently skipped.
PREFERRED_ORDER = [
    "L1", "L1_seq", "L1_grad", "L1_bad1",
    "cocktail", "cocktail_b05",
    "stack", "stack_d1", "charbonnier",
]


def latest_viz_step(viz_dir: Path) -> Path | None:
    """Return the viz panel with the highest step number."""
    best = None
    best_step = -1
    rx = re.compile(r"step_(\d+)\.png")
    for f in viz_dir.glob("step_*.png"):
        m = rx.match(f.name)
        if m:
            s = int(m.group(1))
            if s > best_step:
                best_step = s
                best = f
    return best


def split_panel(panel: np.ndarray):
    """The render_panel layout is a 2×2 grid:
        +------+--------+
        | left | GT     |
        +------+--------+
        | pred | stats  |
        +------+--------+
    Returns (left, gt, pred) as numpy arrays."""
    H = panel.shape[0] // 2
    W = panel.shape[1] // 2
    left = panel[:H, :W]
    gt = panel[:H, W:]
    pred = panel[H:, :W]
    return left, gt, pred


def annotate(img: np.ndarray, lines: list[str],
             scale: float = 0.6, thick: int = 2, color=(255, 255, 255),
             bg=True) -> np.ndarray:
    """Stamp `lines` on top-left of `img` with optional dark backdrop."""
    out = img.copy()
    H = len(lines)
    line_h = int(round(28 * scale / 0.65))
    bar_h = line_h * H + 8
    if bg:
        bg_layer = out.copy()
        cv2.rectangle(bg_layer, (0, 0), (out.shape[1], bar_h),
                       (0, 0, 0), -1)
        out = cv2.addWeighted(bg_layer, 0.55, out, 0.45, 0)
    for i, line in enumerate(lines):
        cv2.putText(out, line, (10, line_h * (i + 1) - 2),
                    cv2.FONT_HERSHEY_SIMPLEX, scale, color, thick,
                    cv2.LINE_AA)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True,
                    help="Path to loss_ablation_<TS>/ directory")
    ap.add_argument("--out", default=None,
                    help="Output PNG path (default: <root>/comparison_mosaic.png)")
    ap.add_argument("--cols", type=int, default=3,
                    help="Number of columns in the variant grid")
    args = ap.parse_args()

    root = Path(args.root).resolve()
    if not root.is_dir():
        raise SystemExit(f"missing: {root}")
    out_path = Path(args.out) if args.out else (root / "comparison_mosaic.png")

    # Find all variant directories with a viz subdir.
    candidates = []
    for var in sorted(root.iterdir()):
        if not var.is_dir():
            continue
        viz = var / "viz"
        if not viz.exists():
            continue
        viz_panel = latest_viz_step(viz)
        if viz_panel is None:
            continue
        meta_path = var / "meta.json"
        meta = {}
        if meta_path.exists():
            try:
                meta = json.loads(meta_path.read_text())
            except Exception:
                pass
        candidates.append((var.name, viz_panel, meta))

    if not candidates:
        raise SystemExit(f"no variants with viz panels found under {root}")

    # Sort by preferred order, fall back to alphabetical.
    def sort_key(item):
        name = item[0]
        if name in PREFERRED_ORDER:
            return (0, PREFERRED_ORDER.index(name))
        return (1, name)
    candidates.sort(key=sort_key)

    # Extract panels.
    print(f"found {len(candidates)} variants:")
    extracted: list[tuple[str, np.ndarray, np.ndarray, np.ndarray, dict]] = []
    for name, viz_panel, meta in candidates:
        panel = cv2.imread(str(viz_panel))
        if panel is None:
            print(f"  skip {name}: cannot read {viz_panel}")
            continue
        left, gt, pred = split_panel(panel)
        fm = meta.get("final_metrics_all", {})
        print(f"  {name:<14}  panel={viz_panel.name}  "
              f"EPE={fm.get('epe', float('nan')):.3f}  "
              f"bad-0.5={fm.get('bad_0.5', float('nan')):.1f}%  "
              f"bad-1={fm.get('bad_1.0', float('nan')):.1f}%")
        extracted.append((name, left, gt, pred, fm))

    # Use the first variant's left + GT for the reference header.
    ref_name, ref_left, ref_gt, _, _ = extracted[0]
    H, W = ref_left.shape[:2]

    # Annotate header tiles.
    header_left = annotate(ref_left, ["INPUT (left)", f"{W}x{H}"], scale=0.7,
                             thick=2)
    header_gt = annotate(ref_gt, ["GROUND TRUTH disparity", "(TURBO)"],
                          scale=0.7, thick=2)
    # Build header row: [left | GT]; if cols > 2, pad with empties.
    pad = np.zeros((H, W, 3), dtype=np.uint8)
    header_tiles = [header_left, header_gt]
    while len(header_tiles) < args.cols:
        header_tiles.append(pad.copy())
    header_row = np.hstack(header_tiles)

    # Build variant rows.
    rows = [header_row]
    cur: list[np.ndarray] = []
    for name, _, _, pred, fm in extracted:
        epe = fm.get("epe", float("nan"))
        b05 = fm.get("bad_0.5", float("nan"))
        b1 = fm.get("bad_1.0", float("nan"))
        d1 = fm.get("d1_all", float("nan"))
        labelled = annotate(pred, [
            f"{name}",
            f"EPE {epe:.3f}  bad-0.5 {b05:.1f}%",
            f"bad-1 {b1:.1f}%  D1 {d1:.1f}%",
        ], scale=0.55, thick=2)
        cur.append(labelled)
        if len(cur) == args.cols:
            rows.append(np.hstack(cur))
            cur = []
    if cur:
        # Pad the last row if it's incomplete.
        while len(cur) < args.cols:
            cur.append(pad.copy())
        rows.append(np.hstack(cur))

    mosaic = np.vstack(rows)

    # Add a thin separator between header and variants.
    sep_y = H
    cv2.line(mosaic, (0, sep_y), (mosaic.shape[1], sep_y), (40, 40, 40), 2)
    for i in range(1, args.cols):
        x = i * W
        cv2.line(mosaic, (x, 0), (x, mosaic.shape[0]), (40, 40, 40), 1)
    for i in range(2, len(rows)):
        y = i * H
        cv2.line(mosaic, (0, y), (mosaic.shape[1], y), (40, 40, 40), 1)

    # Add a title bar at the top.
    title_h = 36
    title = np.full((title_h, mosaic.shape[1], 3), (20, 20, 20),
                     dtype=np.uint8)
    cv2.putText(
        title,
        f"Loss-formulation visual comparison  ({root.name})",
        (12, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (220, 220, 220), 2,
        cv2.LINE_AA)
    mosaic = np.vstack([title, mosaic])

    cv2.imwrite(str(out_path), mosaic)
    print(f"\nwrote {out_path}  ({mosaic.shape[1]}x{mosaic.shape[0]})")


if __name__ == "__main__":
    main()
