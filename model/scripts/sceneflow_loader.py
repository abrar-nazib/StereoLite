"""Scene Flow Driving subset dataset loader.

After extracting:
    data/sceneflow_driving/frames_finalpass/{15,35}mm_focallength/scene_*/{left,right}/*.png
    data/sceneflow_driving/disparity/{15,35}mm_focallength/scene_*/left/*.pfm

This module exposes:
    enumerate_pairs(root)           -> list of (left_png, right_png, left_pfm)
    train_val_split(items, n_val)   -> (train, val) deterministic
    SceneFlowDataset                -> torch.utils.data.Dataset

PFM read code is borrowed from the standard Middlebury format; values are
positive disparities, sentinel `inf` marks invalid pixels.
"""
from __future__ import annotations

import os
import re
import struct
from pathlib import Path

import numpy as np
import torch
import cv2


# ---------------------------- PFM reader ---------------------------------- #
def read_pfm(path: str) -> np.ndarray:
    with open(path, "rb") as fp:
        header = fp.readline().decode("latin-1").rstrip()
        if header == "PF":
            color = True
        elif header == "Pf":
            color = False
        else:
            raise ValueError(f"Not a PFM file: {path}")
        line = fp.readline().decode("latin-1")
        m = re.match(r"^(\d+)\s+(\d+)\s*$", line)
        if not m:
            raise ValueError(f"Bad PFM header in {path}")
        w, h = int(m.group(1)), int(m.group(2))
        scale = float(fp.readline().decode("latin-1").rstrip())
        endian = "<" if scale < 0 else ">"
        data = np.frombuffer(fp.read(), dtype=endian + "f")
        if color:
            data = data.reshape(h, w, 3)
        else:
            data = data.reshape(h, w)
        data = np.flipud(data)        # PFM is bottom-to-top
        return data.copy()


# ---------------------------- pair enumeration ---------------------------- #
def enumerate_pairs(root: str) -> list[tuple[str, str, str]]:
    """Walk frames_finalpass and find matching disparity PFMs.

    Returns a list of (left_png, right_png, left_pfm).
    """
    rgb_root = os.path.join(root, "frames_finalpass")
    disp_root = os.path.join(root, "disparity")
    out: list[tuple[str, str, str]] = []
    for fl_dir in sorted(os.listdir(rgb_root)):
        # 15mm_focallength / 35mm_focallength
        for direction in sorted(os.listdir(os.path.join(rgb_root, fl_dir))):
            # scene_backwards / scene_forwards
            for speed in sorted(os.listdir(os.path.join(rgb_root, fl_dir, direction))):
                # fast / slow
                left_dir = os.path.join(rgb_root, fl_dir, direction, speed, "left")
                right_dir = os.path.join(rgb_root, fl_dir, direction, speed, "right")
                disp_dir = os.path.join(disp_root, fl_dir, direction, speed, "left")
                if not os.path.isdir(left_dir):
                    continue
                for fname in sorted(os.listdir(left_dir)):
                    if not fname.endswith(".png"):
                        continue
                    lp = os.path.join(left_dir, fname)
                    rp = os.path.join(right_dir, fname)
                    pp = os.path.join(disp_dir, fname.replace(".png", ".pfm"))
                    if os.path.exists(rp) and os.path.exists(pp):
                        out.append((lp, rp, pp))
    return out


def train_val_split(items: list, n_val: int, seed: int = 0):
    rng = np.random.default_rng(seed)
    idx = np.arange(len(items))
    rng.shuffle(idx)
    val_idx = idx[:n_val]
    train_idx = idx[n_val:]
    return [items[i] for i in train_idx], [items[i] for i in val_idx]


# ---------------------------- torch Dataset ------------------------------- #
class SceneFlowDriving(torch.utils.data.Dataset):
    """Returns (left_t, right_t, disp_t) at fixed (H, W). disp_t is in pixels
    at the saved (H, W); invalid pixels are 0."""

    def __init__(self, items: list, h: int, w: int):
        self.items = items
        self.h = h
        self.w = w

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        lp, rp, pp = self.items[idx]
        L = cv2.imread(lp)
        R = cv2.imread(rp)
        D_full = read_pfm(pp)        # (H, W) in pixels at native resolution
        H_native, W_native = D_full.shape
        L = cv2.resize(L, (self.w, self.h), interpolation=cv2.INTER_AREA)
        R = cv2.resize(R, (self.w, self.h), interpolation=cv2.INTER_AREA)
        # Resize disparity proportionally: a disparity of d_native at (H_native, W_native)
        # corresponds to d_native * (W/W_native) at (H, W).
        sx = self.w / W_native
        D = cv2.resize(D_full, (self.w, self.h), interpolation=cv2.INTER_LINEAR) * sx
        D[~np.isfinite(D) | (D < 0)] = 0
        Lt = torch.from_numpy(cv2.cvtColor(L, cv2.COLOR_BGR2RGB)).float().permute(2, 0, 1)
        Rt = torch.from_numpy(cv2.cvtColor(R, cv2.COLOR_BGR2RGB)).float().permute(2, 0, 1)
        Dt = torch.from_numpy(D.astype(np.float32)).unsqueeze(0)
        return Lt, Rt, Dt


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--root", default="/home/abrar/Research/stero_research_claude/data/sceneflow_driving")
    p.add_argument("--n", type=int, default=5)
    args = p.parse_args()
    items = enumerate_pairs(args.root)
    print(f"found {len(items)} stereo pairs under {args.root}")
    for lp, rp, pp in items[:args.n]:
        d = read_pfm(pp)
        print(f"  {os.path.basename(lp)}  disp range [{np.nanmin(d):.1f}, {np.nanmax(d):.1f}], shape={d.shape}")
