"""Dataset loader for FoundationStereo pseudo-GT pairs.

Mirrors the (L, R, D) interface of `SceneFlowDriving` so it drops into
the existing trainer without code changes downstream. Reads the clean
filter list produced by `inspect_pseudo_dataset.py` and ignores
everything else.

Disparity-on-resize math (important):
  Disparity is a horizontal pixel shift between left and right views.
  When we resize an image from width W_native to W_train, pixel
  coordinates compress by a factor sx = W_train / W_native, so
  disparity values must be multiplied by the same sx. Vertical resize
  has no effect on disparity values themselves (only on per-pixel
  positions). We use INTER_LINEAR for the disparity resize because
  PFM/.npy disparities are continuous, and INTER_AREA for image
  downsampling because that gives cleaner photometric features.
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import torch


def list_pairs(pairs_dir: str | Path,
               clean_list: str | Path | None = None) -> list[tuple[str, str, str]]:
    """Return [(left_png, right_png, disp_npy), ...] for every clean pair."""
    pairs_dir = Path(pairs_dir)
    if clean_list is None:
        clean_list = pairs_dir / "clean_pairs.txt"
    clean_list = Path(clean_list)
    if not clean_list.exists():
        raise FileNotFoundError(
            f"{clean_list} missing — run inspect_pseudo_dataset.py first")
    bases = [l.strip() for l in clean_list.read_text().splitlines() if l.strip()]
    out = []
    for b in bases:
        L = pairs_dir / "left" / f"{b}.png"
        R = pairs_dir / "right" / f"{b}.png"
        D = pairs_dir / "disp_pseudo" / f"{b}.npy"
        if L.exists() and R.exists() and D.exists():
            out.append((str(L), str(R), str(D)))
    return out


def split_pairs(pairs: list, n_val: int, seed: int = 0):
    rng = np.random.default_rng(seed)
    idx = np.arange(len(pairs))
    rng.shuffle(idx)
    val = [pairs[i] for i in idx[:n_val]]
    train = [pairs[i] for i in idx[n_val:]]
    return train, val


class PseudoPairs(torch.utils.data.Dataset):
    """Returns (L, R, D) at fixed (h, w) with disparity correctly rescaled.

    L, R: float tensors (3, h, w) in [0, 255], BGR→RGB converted.
    D:    float tensor (1, h, w), disparity in pixels at the (h, w) scale.
    """

    def __init__(self, items: list[tuple[str, str, str]], h: int, w: int):
        self.items = items
        self.h = h
        self.w = w

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int):
        lp, rp, dp = self.items[idx]
        L = cv2.imread(lp)
        R = cv2.imread(rp)
        D_full = np.load(dp).astype(np.float32)        # (H_native, W_native)

        H_native, W_native = D_full.shape
        # Resize images with INTER_AREA (good for downsampling photos)
        L = cv2.resize(L, (self.w, self.h), interpolation=cv2.INTER_AREA)
        R = cv2.resize(R, (self.w, self.h), interpolation=cv2.INTER_AREA)
        # Resize disparity with INTER_LINEAR + horizontal scale factor
        sx = self.w / W_native
        D = cv2.resize(D_full, (self.w, self.h),
                       interpolation=cv2.INTER_LINEAR) * sx
        # Sanitise — kill NaN/Inf and negatives
        D[~np.isfinite(D) | (D < 0)] = 0.0

        Lt = torch.from_numpy(cv2.cvtColor(L, cv2.COLOR_BGR2RGB)).float() \
                  .permute(2, 0, 1)
        Rt = torch.from_numpy(cv2.cvtColor(R, cv2.COLOR_BGR2RGB)).float() \
                  .permute(2, 0, 1)
        Dt = torch.from_numpy(D).unsqueeze(0)
        return Lt, Rt, Dt


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--pairs_dir", required=True)
    p.add_argument("--n", type=int, default=3)
    args = p.parse_args()
    pairs = list_pairs(args.pairs_dir)
    print(f"found {len(pairs)} clean pairs under {args.pairs_dir}")
    ds = PseudoPairs(pairs[: args.n], h=512, w=832)
    for i in range(len(ds)):
        L, R, D = ds[i]
        valid = (D > 0.5) & torch.isfinite(D)
        print(f"  pair {i}: L={tuple(L.shape)}  D=[{D[valid].min():.2f}..{D[valid].max():.2f}px]"
              f"  mean={D[valid].mean():.2f}  shape={tuple(D.shape)}")
