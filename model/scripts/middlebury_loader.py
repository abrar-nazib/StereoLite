"""Middlebury stereo dataset loader.

Expected after extracting Middlebury 2014/2021 zips somewhere under `root`:
scene folders containing at least:

    im0.png
    im1.png
    disp0.pfm

Optional masks such as `mask0nocc.png` / `mask0.png` are used when present.
The loader exposes the same `(left, right, disp)` tensor interface as the
SceneFlow loader so it can be used by `train_arch_sceneflow.py`.
"""
from __future__ import annotations

import os
from pathlib import Path

import cv2
import numpy as np
import torch

from sceneflow_loader import read_pfm


MASK_NAMES = ("mask0nocc.png", "mask0.png", "mask0nocc.bmp", "mask0.bmp")


def enumerate_middlebury(root: str) -> list[tuple[str, str, str]]:
    """Find Middlebury stereo triples under an extracted dataset root."""
    out: list[tuple[str, str, str]] = []
    for dirpath, _, filenames in os.walk(root):
        names = set(filenames)
        if {"im0.png", "im1.png", "disp0.pfm"}.issubset(names):
            base = Path(dirpath)
            out.append((
                str(base / "im0.png"),
                str(base / "im1.png"),
                str(base / "disp0.pfm"),
            ))
    return sorted(out)


def _mask_for_disp(disp_path: str) -> str | None:
    base = Path(disp_path).parent
    for name in MASK_NAMES:
        path = base / name
        if path.exists():
            return str(path)
    return None


def _read_disp(path: str) -> np.ndarray:
    disp = read_pfm(path).astype(np.float32)
    mask_path = _mask_for_disp(path)
    if mask_path:
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is not None and mask.shape == disp.shape:
            disp = disp.copy()
            disp[mask == 0] = 0
    disp[~np.isfinite(disp) | (disp < 0)] = 0
    return disp


def _to_tensors(left_bgr: np.ndarray, right_bgr: np.ndarray,
                disp: np.ndarray):
    lt = torch.from_numpy(cv2.cvtColor(left_bgr, cv2.COLOR_BGR2RGB)).float()
    rt = torch.from_numpy(cv2.cvtColor(right_bgr, cv2.COLOR_BGR2RGB)).float()
    dt = torch.from_numpy(disp.astype(np.float32)).unsqueeze(0)
    return lt.permute(2, 0, 1), rt.permute(2, 0, 1), dt


class MiddleburyResize(torch.utils.data.Dataset):
    """Resize each full scene to a fixed size and scale disparity by width."""

    def __init__(self, items: list[tuple[str, str, str]], h: int, w: int):
        self.items = items
        self.h = h
        self.w = w

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        lp, rp, pp = self.items[idx]
        left = cv2.imread(lp)
        right = cv2.imread(rp)
        disp_full = _read_disp(pp)
        hn, wn = disp_full.shape
        left = cv2.resize(left, (self.w, self.h), interpolation=cv2.INTER_AREA)
        right = cv2.resize(right, (self.w, self.h), interpolation=cv2.INTER_AREA)
        disp = cv2.resize(disp_full, (self.w, self.h),
                          interpolation=cv2.INTER_LINEAR) * (self.w / wn)
        disp[~np.isfinite(disp) | (disp < 0)] = 0
        return _to_tensors(left, right, disp)


class MiddleburyCrop(torch.utils.data.Dataset):
    """Native-resolution crop loader for boundary/detail diagnostics."""

    def __init__(self, items: list[tuple[str, str, str]], h: int, w: int,
                 train: bool = True, scale_min: float = 0.85,
                 scale_max: float = 1.10, color_aug: float = 0.08):
        self.items = items
        self.h = h
        self.w = w
        self.train = train
        self.scale_min = scale_min
        self.scale_max = scale_max
        self.color_aug = color_aug

    def __len__(self):
        return len(self.items)

    def _resize_if_needed(self, left, right, disp):
        hn, wn = disp.shape
        scale = 1.0
        if self.train and self.scale_max > 0:
            scale = float(np.random.uniform(self.scale_min, self.scale_max))
        scale = max(scale, self.h / hn, self.w / wn)
        if abs(scale - 1.0) < 1e-3:
            return left, right, disp
        nh = int(round(hn * scale))
        nw = int(round(wn * scale))
        left = cv2.resize(left, (nw, nh), interpolation=cv2.INTER_LINEAR)
        right = cv2.resize(right, (nw, nh), interpolation=cv2.INTER_LINEAR)
        disp = cv2.resize(disp, (nw, nh), interpolation=cv2.INTER_LINEAR) * scale
        return left, right, disp

    def _crop(self, left, right, disp):
        hn, wn = disp.shape
        if self.train:
            y0 = np.random.randint(0, max(hn - self.h + 1, 1))
            x0 = np.random.randint(0, max(wn - self.w + 1, 1))
        else:
            y0 = max((hn - self.h) // 2, 0)
            x0 = max((wn - self.w) // 2, 0)
        return (
            left[y0:y0 + self.h, x0:x0 + self.w],
            right[y0:y0 + self.h, x0:x0 + self.w],
            disp[y0:y0 + self.h, x0:x0 + self.w],
        )

    def _color_aug(self, left, right):
        if not self.train or self.color_aug <= 0:
            return left, right
        strength = self.color_aug
        gain = 1.0 + np.random.uniform(-strength, strength)
        bias = np.random.uniform(-12.0 * strength, 12.0 * strength)
        left = np.clip(left.astype(np.float32) * gain + bias, 0, 255)
        right = np.clip(right.astype(np.float32) * gain + bias, 0, 255)
        return left.astype(np.uint8), right.astype(np.uint8)

    def __getitem__(self, idx):
        lp, rp, pp = self.items[idx]
        left = cv2.imread(lp)
        right = cv2.imread(rp)
        disp = _read_disp(pp)
        left, right, disp = self._resize_if_needed(left, right, disp)
        left, right, disp = self._crop(left, right, disp)
        left, right = self._color_aug(left, right)
        disp[~np.isfinite(disp) | (disp < 0)] = 0
        return _to_tensors(left, right, disp)


class MiddleburyMixed(torch.utils.data.Dataset):
    """Mostly resized full scenes, with some native crops for sharpness."""

    def __init__(self, items: list[tuple[str, str, str]], h: int, w: int,
                 crop_prob: float = 0.25, scale_min: float = 0.9,
                 scale_max: float = 1.05, color_aug: float = 0.06):
        self.resize_ds = MiddleburyResize(items, h, w)
        self.crop_ds = MiddleburyCrop(
            items, h, w, train=True, scale_min=scale_min,
            scale_max=scale_max, color_aug=color_aug)
        self.crop_prob = crop_prob

    def __len__(self):
        return len(self.resize_ds)

    def __getitem__(self, idx):
        if np.random.rand() < self.crop_prob:
            return self.crop_ds[idx]
        return self.resize_ds[idx]
