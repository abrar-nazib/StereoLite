"""Build animated GIFs from a montage/ training folder.

Reads all step_*.png frames, downscales to a sensible width, overlays a
big step counter, and writes:

  <panel_dir>/training_progression.gif          (3-pair watch view)
  <panel_dir>/training_progression_full.gif     (all tracked pairs)

Pass --hold_last to repeat the final frame for a few seconds so the
viewer can study the converged result.
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont


STEP_RE = re.compile(r"step_(\d+)\.png$")


def _natural_sort(paths: list[Path]) -> list[Path]:
    return sorted(paths, key=lambda p: int(STEP_RE.search(p.name).group(1)))


def _load_font(size: int = 38):
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


def _annotate(img: Image.Image, step: int, font: ImageFont.FreeTypeFont,
               total_steps: int) -> Image.Image:
    out = img.copy()
    d = ImageDraw.Draw(out)
    pct = step / max(total_steps, 1) * 100
    text = f"step {step}   ({pct:.0f}% of training)"
    # Banner across the top
    banner_h = 56
    d.rectangle([(0, 0), (out.width, banner_h)],
                fill=(0, 0, 0))
    d.text((16, 8), text, font=font, fill=(255, 255, 255))
    return out


def build_gif(src_dir: Path, out_path: Path, max_w: int = 1200,
              frame_ms: int = 250, hold_last_ms: int = 2000):
    pngs = _natural_sort(list(src_dir.glob("step_*.png")))
    if not pngs:
        print(f"  no frames in {src_dir}")
        return None
    total_steps = int(STEP_RE.search(pngs[-1].name).group(1))
    print(f"  {len(pngs)} frames, total_steps={total_steps}")

    font = _load_font(size=32)
    frames: list[Image.Image] = []
    for p in pngs:
        # Read with cv2 (handles BGR), then convert to PIL RGB
        bgr = cv2.imread(str(p))
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(rgb)
        # Downscale if needed
        if img.width > max_w:
            new_h = int(img.height * max_w / img.width)
            img = img.resize((max_w, new_h), Image.LANCZOS)
        step = int(STEP_RE.search(p.name).group(1))
        img = _annotate(img, step, font, total_steps)
        frames.append(img)

    # Hold last frame longer
    durations = [frame_ms] * len(frames)
    if hold_last_ms > 0 and frames:
        n_extra = max(1, hold_last_ms // frame_ms)
        for _ in range(n_extra - 1):
            frames.append(frames[-1])
            durations.append(frame_ms)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    frames[0].save(
        str(out_path),
        save_all=True,
        append_images=frames[1:],
        duration=durations,
        loop=0,
        optimize=True,
    )
    sz_mb = out_path.stat().st_size / 1e6
    print(f"  -> {out_path}  ({sz_mb:.1f} MB, {len(frames)} frames)")
    return out_path


def build_grid_gif(panel_dir: Path, out_path: Path, rows: int, cols: int,
                    max_w: int = 1200, frame_ms: int = 250,
                    hold_last_ms: int = 2500):
    """Re-stack per-pair sequences into a (rows × cols) grid GIF.

    Each cell holds one full (L | GT | pred) triple from
    `panel_dir/pairs/pair_<i>/step_<S>.png`. With (rows, cols) = (4, 2)
    you get a portrait-leaning montage that fits on a slide.
    """
    pairs_dir = panel_dir / "pairs"
    pair_dirs = sorted([d for d in pairs_dir.iterdir() if d.is_dir()],
                        key=lambda d: int(d.name.split("_")[-1]))
    if not pair_dirs:
        print(f"  no per-pair dirs in {pairs_dir}")
        return None
    n_needed = rows * cols
    pair_dirs = pair_dirs[:n_needed]
    if len(pair_dirs) < n_needed:
        print(f"  WARN: only {len(pair_dirs)} pairs, padding with copies")

    # Get common step list (intersection of what exists in each pair)
    all_steps = None
    for pd in pair_dirs:
        steps = {int(STEP_RE.search(p.name).group(1))
                 for p in pd.glob("step_*.png")}
        all_steps = steps if all_steps is None else (all_steps & steps)
    common_steps = sorted(all_steps or [])
    print(f"  {len(common_steps)} common steps across {len(pair_dirs)} pairs, "
          f"layout {rows}x{cols}")

    font = _load_font(size=32)
    frames: list[Image.Image] = []
    for step in common_steps:
        # Load all per-pair tiles for this step
        tiles = []
        for pd in pair_dirs:
            p = pd / f"step_{step:05d}.png"
            bgr = cv2.imread(str(p))
            tiles.append(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
        # Pad to n_needed if short
        while len(tiles) < n_needed:
            tiles.append(np.zeros_like(tiles[0]))
        # Build rows × cols grid
        row_imgs = []
        for r in range(rows):
            row_tiles = tiles[r * cols:(r + 1) * cols]
            row_imgs.append(np.hstack(row_tiles))
        grid = np.vstack(row_imgs)
        img = Image.fromarray(grid)
        if img.width > max_w:
            new_h = int(img.height * max_w / img.width)
            img = img.resize((max_w, new_h), Image.LANCZOS)
        total = common_steps[-1] if common_steps else step
        img = _annotate(img, step, font, total)
        frames.append(img)

    durations = [frame_ms] * len(frames)
    if hold_last_ms > 0 and frames:
        n_extra = max(1, hold_last_ms // frame_ms)
        for _ in range(n_extra - 1):
            frames.append(frames[-1])
            durations.append(frame_ms)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    frames[0].save(
        str(out_path),
        save_all=True,
        append_images=frames[1:],
        duration=durations,
        loop=0,
        optimize=True,
    )
    sz_mb = out_path.stat().st_size / 1e6
    print(f"  -> {out_path}  ({sz_mb:.1f} MB, {len(frames)} frames)")
    return out_path


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--panel_dir", required=True)
    p.add_argument("--max_w", type=int, default=1200,
                   help="downscale frames wider than this")
    p.add_argument("--frame_ms", type=int, default=250)
    p.add_argument("--hold_last_ms", type=int, default=2500,
                   help="how long to hold the last frame")
    p.add_argument("--full_layout", default="4x2",
                   help="rows x cols layout for the full grid GIF "
                        "(e.g. 4x2, 2x4, 8x1)")
    args = p.parse_args()
    panel_dir = Path(args.panel_dir)

    print("[1/2] watch view (3-pair montage/)")
    build_gif(panel_dir / "montage",
              panel_dir / "training_progression.gif",
              max_w=args.max_w, frame_ms=args.frame_ms,
              hold_last_ms=args.hold_last_ms)

    rows, cols = (int(x) for x in args.full_layout.split("x"))
    print(f"\n[2/2] full grid (re-stacked {rows} rows x {cols} cols)")
    build_grid_gif(panel_dir,
                    panel_dir / f"training_progression_full_{rows}x{cols}.gif",
                    rows=rows, cols=cols,
                    max_w=args.max_w, frame_ms=args.frame_ms,
                    hold_last_ms=args.hold_last_ms)


if __name__ == "__main__":
    main()
