"""Build training-progression GIFs for results slides 16 and 17.

Layout per frame: (rows, cols) = (3, 1), i.e. three pair panels stacked
vertically. Each pair panel is the existing 2496×512 step montage
(left image | GT/teacher disparity | predicted disparity) saved by the
training script.

We pick the three pairs that achieve the lowest final-step EPE in each
run, so the slide showcases the best-case generalisation.

Outputs (saved next to this file):
    training_v8_top3.gif         — Scene Flow synthetic (slide 16)
    training_finetune_top3.gif   — indoor pseudo-GT finetune (slide 17)
"""
from __future__ import annotations

import csv
import io
import os
from pathlib import Path

from PIL import Image

ROOT = Path("/home/abrar/Research/stero_research_claude")
BENCH = ROOT / "model/benchmarks"
OUT = Path(__file__).resolve().parent

# Per-pair step PNGs are 2496×512. Stacking 3 vertically → 2496×1536.
# We downscale to TARGET_W to keep the GIF small (saved as 8-bit indexed,
# adaptive palette, optimized) and skip every other step so the deck
# does not balloon by 20 MB.
TARGET_W = 900
DURATION_MS = 280
STEP_STRIDE = 2          # 1 = every step, 2 = every other step
PALETTE_COLORS = 96      # adaptive palette size


def _topk_pairs(progress_csv: Path, k: int = 3) -> list[str]:
    """Return the k pair-IDs (e.g. 'pair_18') with the lowest final EPE."""
    with progress_csv.open() as f:
        rows = list(csv.DictReader(f))
    last = rows[-1]
    pair_cols = [c for c in last if c.startswith("pair_") and c.endswith("_epe")]
    finals: list[tuple[str, float]] = []
    for c in pair_cols:
        try:
            finals.append((c.replace("_epe", ""), float(last[c])))
        except (ValueError, TypeError):
            continue
    finals.sort(key=lambda x: x[1])
    return [pid for pid, _ in finals[:k]]


def _step_files(pairs_dir: Path, pair_id: str) -> list[Path]:
    """Sorted list of step_*.png paths for a given pair."""
    d = pairs_dir / pair_id
    return sorted(d.glob("step_*.png"))


def _build_gif(run_dir: Path, out_name: str, *, k: int = 3) -> Path:
    """Build a (k, 1)-stacked animated GIF for a training run."""
    progress = run_dir / "progress.csv"
    pairs_dir = run_dir / "pairs"
    if not progress.exists() or not pairs_dir.exists():
        raise FileNotFoundError(f"missing progress.csv or pairs/ in {run_dir}")

    pair_ids = _topk_pairs(progress, k=k)
    print(f"  {run_dir.name}: top-{k} pairs by final EPE -> {pair_ids}")

    # Build per-pair step lists; align by intersection of step filenames
    # (different pairs may have slightly different step coverage).
    step_lists = {pid: _step_files(pairs_dir, pid) for pid in pair_ids}
    name_sets = [set(p.name for p in lst) for lst in step_lists.values()]
    common = sorted(set.intersection(*name_sets))
    if not common:
        raise RuntimeError("no common step filenames across selected pairs")
    # Always keep the first and last steps so the GIF brackets the full
    # training arc. Stride-sample the middle.
    if STEP_STRIDE > 1:
        body = common[1:-1][::STEP_STRIDE]
        common = [common[0], *body, common[-1]]
    print(f"    {len(common)} sampled steps across pairs")

    # Build frames
    frames: list[Image.Image] = []
    for step_name in common:
        rows = []
        for pid in pair_ids:
            p = pairs_dir / pid / step_name
            im = Image.open(p).convert("RGB")
            rows.append(im)
        # vstack
        w_max = max(r.width for r in rows)
        h_total = sum(r.height for r in rows)
        canvas = Image.new("RGB", (w_max, h_total), (244, 239, 230))
        y = 0
        for r in rows:
            x = (w_max - r.width) // 2
            canvas.paste(r, (x, y))
            y += r.height
        # downscale to TARGET_W
        if canvas.width > TARGET_W:
            ratio = TARGET_W / canvas.width
            canvas = canvas.resize(
                (TARGET_W, int(canvas.height * ratio)),
                Image.LANCZOS)
        # Convert to a fixed adaptive palette so frames quantise consistently
        frames.append(canvas.convert(
            "P", palette=Image.ADAPTIVE, colors=PALETTE_COLORS))
        for r in rows:
            r.close()

    out_path = OUT / out_name
    frames[0].save(
        out_path,
        save_all=True,
        append_images=frames[1:],
        duration=DURATION_MS,
        loop=0,
        optimize=True,
        disposal=2,
    )
    print(f"    saved {out_path.name}  ({out_path.stat().st_size/1e6:.1f} MB,"
          f" {len(frames)} frames)")
    return out_path


def build_v8_gif() -> Path:
    """Slide 16 — Scene Flow synthetic (v8 run)."""
    return _build_gif(
        BENCH / "stereolite_v8_20260419-132735",
        "training_v8_top3.gif")


def build_finetune_gif() -> Path:
    """Slide 17 — indoor pseudo-GT finetune."""
    return _build_gif(
        BENCH / "stereolite_finetune_indoor_20260426-171158",
        "training_finetune_top3.gif")


if __name__ == "__main__":
    build_v8_gif()
    build_finetune_gif()
