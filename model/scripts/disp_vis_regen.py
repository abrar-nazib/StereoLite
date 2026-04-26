"""Regenerate disp_vis/*.png from disp_pseudo/*.npy for all CLEAN pairs.

Reads <pairs_dir>/clean_pairs.txt (produced by inspect_pseudo_dataset.py)
and writes a TURBO-coloured visualisation per pair into
<pairs_dir>/disp_vis/<basename>.png. Skips pairs not in the clean list.

Each visualisation is normalised per-pair to its own [p2, p98] disparity
range so far/close objects are easy to read regardless of scene depth.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--pairs_dir", required=True)
    args = p.parse_args()
    pairs = Path(args.pairs_dir)

    clean_path = pairs / "clean_pairs.txt"
    if not clean_path.exists():
        sys.exit(f"missing {clean_path} — run inspect_pseudo_dataset.py first")

    cleans = [l.strip() for l in clean_path.read_text().splitlines() if l.strip()]
    out_dir = pairs / "disp_vis"
    out_dir.mkdir(exist_ok=True)
    n_done = 0
    n_skip = 0
    for base in cleans:
        npy_path = pairs / "disp_pseudo" / f"{base}.npy"
        out_path = out_dir / f"{base}.png"
        try:
            d = np.load(npy_path)
        except Exception as e:
            print(f"  skip {base}: {type(e).__name__}: {e}")
            n_skip += 1
            continue
        valid = (d > 0.5) & np.isfinite(d)
        if valid.sum() < 16:
            lo, hi = 0.0, 60.0
        else:
            lo = float(np.percentile(d[valid], 2))
            hi = float(np.percentile(d[valid], 98))
        v = np.clip((d - lo) / max(hi - lo, 1e-6), 0, 1) * 255
        col = cv2.applyColorMap(v.astype(np.uint8), cv2.COLORMAP_TURBO)
        cv2.imwrite(str(out_path), col)
        n_done += 1
        if n_done % 200 == 0:
            print(f"  rendered {n_done}/{len(cleans)}", file=sys.stderr)
    print(f"\nrendered {n_done} pairs, skipped {n_skip}")


if __name__ == "__main__":
    main()
