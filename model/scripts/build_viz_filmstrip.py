"""Build a GIF + a static grid mosaic from a flat viz/ folder.

The overfit_ablation harness writes per-step 4-panel PNGs
(step_<N>.png) to a `<run>/viz/` directory. This script reads them
flat (no sub-folders needed), builds:

  - training_progression.gif : animated GIF, one frame per viz PNG,
    downscaled to a sensible width, with a step banner overlaid.
  - filmstrip.png            : a static NxM mosaic picking evenly-spaced
    frames across the run, useful for a single-figure summary.

Usage:
    python3 model/scripts/build_viz_filmstrip.py \\
        --viz_dir model/benchmarks/<run>/<variant>/viz \\
        --out_dir model/benchmarks/<run>/<variant> \\
        --max_w 1200 --frame_ms 400 --hold_last_ms 3000 \\
        --grid_rows 4 --grid_cols 4
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont


STEP_RE = re.compile(r"step_(\d+)\.png$")


def _load_font(size: int):
    candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVu-Sans-Bold.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/TTF/DejaVuSans-Bold.ttf",
    ]
    for p in candidates:
        try:
            return ImageFont.truetype(p, size)
        except OSError:
            continue
    return ImageFont.load_default()


def _list_steps(viz_dir: Path) -> list[tuple[int, Path]]:
    out: list[tuple[int, Path]] = []
    for p in viz_dir.glob("step_*.png"):
        m = STEP_RE.search(p.name)
        if m:
            out.append((int(m.group(1)), p))
    out.sort()
    return out


def _annotate(img: Image.Image, step: int, total: int,
              font: ImageFont.FreeTypeFont) -> Image.Image:
    out = img.copy()
    d = ImageDraw.Draw(out)
    pct = step / max(total, 1) * 100
    text = f"step {step}   ({pct:.0f}% of training)"
    band_h = 44
    d.rectangle([(0, 0), (out.width, band_h)], fill=(0, 0, 0))
    d.text((10, 6), text, fill=(255, 255, 0), font=font)
    return out


def build_gif(viz_dir: Path, out_path: Path, max_w: int = 1200,
              frame_ms: int = 400, hold_last_ms: int = 3000):
    steps = _list_steps(viz_dir)
    if not steps:
        print(f"  no step_*.png in {viz_dir}")
        return None
    total = steps[-1][0]
    font = _load_font(38)

    frames: list[Image.Image] = []
    for step, p in steps:
        img = Image.open(p).convert("RGB")
        if img.width > max_w:
            new_h = int(img.height * max_w / img.width)
            img = img.resize((max_w, new_h), Image.LANCZOS)
        frames.append(_annotate(img, step, total, font))

    durations = [frame_ms] * len(frames)
    if hold_last_ms > 0 and frames:
        n_extra = max(1, hold_last_ms // frame_ms)
        for _ in range(n_extra - 1):
            frames.append(frames[-1])
            durations.append(frame_ms)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    frames[0].save(
        str(out_path), save_all=True, append_images=frames[1:],
        duration=durations, loop=0, optimize=True,
    )
    sz_mb = out_path.stat().st_size / 1e6
    print(f"  -> {out_path}  ({sz_mb:.1f} MB, {len(frames)} frames)")
    return out_path


def build_mosaic(viz_dir: Path, out_path: Path, rows: int, cols: int,
                 max_w: int = 2400):
    """Static NxM grid of evenly-spaced frames with a step banner per cell."""
    steps = _list_steps(viz_dir)
    if not steps:
        print(f"  no step_*.png in {viz_dir}")
        return None
    n_needed = rows * cols
    if len(steps) < n_needed:
        idxs = list(range(len(steps)))
    else:
        idxs = [int(round(i * (len(steps) - 1) / (n_needed - 1)))
                for i in range(n_needed)]
    picked = [steps[i] for i in idxs]
    total = steps[-1][0]
    font = _load_font(28)

    cells = []
    for step, p in picked:
        img = Image.open(p).convert("RGB")
        if img.width > max_w // cols:
            new_w = max_w // cols
            new_h = int(img.height * new_w / img.width)
            img = img.resize((new_w, new_h), Image.LANCZOS)
        d = ImageDraw.Draw(img)
        band_h = 36
        d.rectangle([(0, 0), (img.width, band_h)], fill=(0, 0, 0))
        d.text((8, 4), f"step {step} ({step/total*100:.0f}%)",
               fill=(255, 255, 0), font=font)
        cells.append(np.array(img))

    # Pad to consistent cell size
    H = max(c.shape[0] for c in cells)
    W = max(c.shape[1] for c in cells)
    pad = []
    for c in cells:
        h, w = c.shape[:2]
        bottom = np.zeros((H - h, w, 3), dtype=c.dtype)
        right = np.zeros((H, W - w, 3), dtype=c.dtype)
        pad.append(np.hstack([np.vstack([c, bottom]), right]))
    grid = np.vstack([np.hstack(pad[r * cols:(r + 1) * cols])
                       for r in range(rows)])
    Image.fromarray(grid).save(str(out_path))
    sz_mb = out_path.stat().st_size / 1e6
    print(f"  -> {out_path}  ({sz_mb:.1f} MB, {rows}x{cols} = "
          f"{rows*cols} frames)")
    return out_path


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--viz_dir", required=True)
    p.add_argument("--out_dir", default="",
                   help="defaults to parent of viz_dir")
    p.add_argument("--max_w", type=int, default=1200)
    p.add_argument("--frame_ms", type=int, default=400)
    p.add_argument("--hold_last_ms", type=int, default=3000)
    p.add_argument("--grid_rows", type=int, default=4)
    p.add_argument("--grid_cols", type=int, default=4)
    p.add_argument("--skip_gif", action="store_true",
                   help="only build the static mosaic")
    args = p.parse_args()

    viz_dir = Path(args.viz_dir)
    out_dir = Path(args.out_dir) if args.out_dir else viz_dir.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    if not args.skip_gif:
        print("[1/2] training progression GIF")
        build_gif(viz_dir, out_dir / "training_progression.gif",
                  max_w=args.max_w, frame_ms=args.frame_ms,
                  hold_last_ms=args.hold_last_ms)

    print(f"\n[2/2] {args.grid_rows}x{args.grid_cols} static mosaic")
    build_mosaic(viz_dir, out_dir / "viz_filmstrip.png",
                  rows=args.grid_rows, cols=args.grid_cols,
                  max_w=args.max_w * 2)


if __name__ == "__main__":
    main()