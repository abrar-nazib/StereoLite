"""Full Scene Flow Driving trainer for StereoLite architecture variants.

This is the non-overfit counterpart to `overfit_arch_ablation.py`: it trains
on the full enumerated Scene Flow Driving split, validates on held-out pairs,
saves best/latest checkpoints, and writes visual sample panels.

Example:
    python model/scripts/train_arch_sceneflow.py \
      --arch yolo_ctx_gev4 --backbone yolo26s \
      --steps 100000 --batch 4 --height 384 --width 640
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import platform
import random
import sys
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "model" / "designs"))
sys.path.insert(0, str(ROOT / "model" / "scripts"))

from sceneflow_loader import (  # noqa: E402
    SceneFlowDriving, enumerate_pairs, read_pfm, train_val_split,
)
from middlebury_loader import (  # noqa: E402
    MiddleburyCrop, MiddleburyMixed, MiddleburyResize, enumerate_middlebury,
)
from overfit_arch_ablation import build_model, _NEW_ARCHES, _arch_flags  # noqa: E402


def dist_info() -> tuple[bool, int, int, int]:
    world = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    return world > 1, rank, local_rank, world


def setup_dist() -> tuple[bool, int, int, int]:
    distributed, rank, local_rank, world = dist_info()
    if distributed and not dist.is_initialized():
        if torch.cuda.is_available():
            torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl" if torch.cuda.is_available() else "gloo")
    return distributed, rank, local_rank, world


def cleanup_dist() -> None:
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


def is_rank0() -> bool:
    return int(os.environ.get("RANK", "0")) == 0


def unwrap_model(model: torch.nn.Module) -> torch.nn.Module:
    return model.module if isinstance(model, DDP) else model


class SceneFlowDrivingCrop(torch.utils.data.Dataset):
    """SceneFlow Driving loader that preserves native pixel detail.

    The older `SceneFlowDriving` path globally resizes a 960x540 frame to
    640x384. That is fast, but it compresses tree leaves, poles, and object
    boundaries before the model sees them. This dataset samples a native
    stereo-consistent crop, optionally after mild random scaling.
    """

    def __init__(self, items: list[tuple[str, str, str]], h: int, w: int,
                 train: bool = True, scale_min: float = 0.9,
                 scale_max: float = 1.15, color_aug: float = 0.2,
                 erase_prob: float = 0.25):
        self.items = items
        self.h = h
        self.w = w
        self.train = train
        self.scale_min = scale_min
        self.scale_max = scale_max
        self.color_aug = color_aug
        self.erase_prob = erase_prob

    def __len__(self):
        return len(self.items)

    def _resize_if_needed(self, left, right, disp):
        H, W = disp.shape
        scale = 1.0
        if self.train and self.scale_max > 0:
            scale = np.random.uniform(self.scale_min, self.scale_max)
        scale = max(scale, self.h / H, self.w / W)
        if abs(scale - 1.0) < 1e-3:
            return left, right, disp
        nh = int(round(H * scale))
        nw = int(round(W * scale))
        left = cv2.resize(left, (nw, nh), interpolation=cv2.INTER_LINEAR)
        right = cv2.resize(right, (nw, nh), interpolation=cv2.INTER_LINEAR)
        disp = cv2.resize(disp, (nw, nh), interpolation=cv2.INTER_LINEAR) * scale
        return left, right, disp

    def _crop(self, left, right, disp):
        H, W = disp.shape
        if self.train:
            y0 = np.random.randint(0, max(H - self.h + 1, 1))
            x0 = np.random.randint(0, max(W - self.w + 1, 1))
        else:
            y0 = max((H - self.h) // 2, 0)
            x0 = max((W - self.w) // 2, 0)
        y1, x1 = y0 + self.h, x0 + self.w
        return left[y0:y1, x0:x1], right[y0:y1, x0:x1], disp[y0:y1, x0:x1]

    def _color_aug(self, left, right):
        if not self.train or self.color_aug <= 0:
            return left, right
        strength = self.color_aug
        gain = 1.0 + np.random.uniform(-strength, strength)
        bias = np.random.uniform(-16.0 * strength, 16.0 * strength)
        gamma = 1.0 + np.random.uniform(-0.35 * strength, 0.35 * strength)
        left = np.clip((left.astype(np.float32) * gain + bias), 0, 255)
        right = np.clip((right.astype(np.float32) * gain + bias), 0, 255)
        left = np.clip(255.0 * ((left / 255.0) ** gamma), 0, 255)
        right = np.clip(255.0 * ((right / 255.0) ** gamma), 0, 255)
        # Tiny asymmetric exposure jitter improves matching robustness without
        # making the stereo correspondence physically inconsistent.
        rgain = 1.0 + np.random.uniform(-0.05 * strength, 0.05 * strength)
        right = np.clip(right * rgain, 0, 255)
        return left.astype(np.uint8), right.astype(np.uint8)

    def _erase_right(self, right):
        if (not self.train) or self.erase_prob <= 0:
            return right
        if np.random.rand() > self.erase_prob:
            return right
        H, W = right.shape[:2]
        out = right.copy()
        for _ in range(np.random.randint(1, 3)):
            ph = np.random.randint(max(8, H // 32), max(9, H // 8))
            pw = np.random.randint(max(8, W // 32), max(9, W // 8))
            y0 = np.random.randint(0, max(H - ph + 1, 1))
            x0 = np.random.randint(0, max(W - pw + 1, 1))
            fill = out[y0:y0 + ph, x0:x0 + pw].mean(axis=(0, 1), keepdims=True)
            out[y0:y0 + ph, x0:x0 + pw] = fill
        return out

    def __getitem__(self, idx):
        lp, rp, pp = self.items[idx]
        left = cv2.imread(lp)
        right = cv2.imread(rp)
        disp = read_pfm(pp)
        left, right, disp = self._resize_if_needed(left, right, disp)
        left, right, disp = self._crop(left, right, disp)
        left, right = self._color_aug(left, right)
        right = self._erase_right(right)
        disp = disp.astype(np.float32)
        disp[~np.isfinite(disp) | (disp < 0)] = 0
        lt = torch.from_numpy(cv2.cvtColor(left, cv2.COLOR_BGR2RGB)).float()
        rt = torch.from_numpy(cv2.cvtColor(right, cv2.COLOR_BGR2RGB)).float()
        dt = torch.from_numpy(disp).unsqueeze(0)
        return lt.permute(2, 0, 1), rt.permute(2, 0, 1), dt


class SceneFlowDrivingMixed(torch.utils.data.Dataset):
    """Mix full-frame resized samples and native crops.

    Resized samples teach global scene layout and match the earlier validation
    metric. Native crops preserve high-frequency details. Training only on
    crops made the current model underfit; mixing gives a gentler curriculum.
    """

    def __init__(self, items: list[tuple[str, str, str]], h: int, w: int,
                 crop_prob: float = 0.35, scale_min: float = 0.95,
                 scale_max: float = 1.05, color_aug: float = 0.1,
                 erase_prob: float = 0.1):
        self.resize_ds = SceneFlowDriving(items, h, w)
        self.crop_ds = SceneFlowDrivingCrop(
            items, h, w, train=True, scale_min=scale_min,
            scale_max=scale_max, color_aug=color_aug,
            erase_prob=erase_prob)
        self.crop_prob = crop_prob

    def __len__(self):
        return len(self.resize_ds)

    def __getitem__(self, idx):
        if np.random.rand() < self.crop_prob:
            return self.crop_ds[idx]
        return self.resize_ds[idx]


def valid_mask(gt: torch.Tensor, max_disp: float) -> torch.Tensor:
    return ((gt > 0) & (gt < max_disp) & torch.isfinite(gt)).float()


def ms_l1(pred: torch.Tensor, gt: torch.Tensor, val: torch.Tensor,
          scale: float) -> torch.Tensor:
    if pred.shape[-2:] != gt.shape[-2:]:
        pred = F.interpolate(pred, size=gt.shape[-2:],
                             mode="bilinear", align_corners=False) * scale
    return ((pred - gt).abs() * val).sum() / val.sum().clamp(min=1)


def grad_consistency(pred: torch.Tensor, gt: torch.Tensor,
                     val: torch.Tensor) -> torch.Tensor:
    gx_p = pred[..., :, 1:] - pred[..., :, :-1]
    gx_g = gt[..., :, 1:] - gt[..., :, :-1]
    gy_p = pred[..., 1:, :] - pred[..., :-1, :]
    gy_g = gt[..., 1:, :] - gt[..., :-1, :]
    vx = val[..., :, 1:] * val[..., :, :-1]
    vy = val[..., 1:, :] * val[..., :-1, :]
    lx = ((gx_p - gx_g).abs() * vx).sum() / vx.sum().clamp(min=1)
    ly = ((gy_p - gy_g).abs() * vy).sum() / vy.sum().clamp(min=1)
    return lx + ly


def threshold_hinge(pred: torch.Tensor, gt: torch.Tensor,
                    val: torch.Tensor) -> torch.Tensor:
    err = (pred - gt).abs()
    total = (
        (err - 0.5).clamp(min=0) ** 2
        + (err - 1.0).clamp(min=0) ** 2
        + (err - 2.0).clamp(min=0) ** 2
        + (err - 3.0).clamp(min=0) ** 2
    )
    return (total * val).sum() / val.sum().clamp(min=1)


def d1_hinge(pred: torch.Tensor, gt: torch.Tensor,
             val: torch.Tensor) -> torch.Tensor:
    err = (pred - gt).abs()
    rel = err / gt.clamp(min=1e-6)
    is_d1 = ((err > 3.0) & (rel > 0.05)).float()
    return (((err - 3.0).clamp(min=0) ** 2) * is_d1 * val).sum() / (
        val.sum().clamp(min=1))


def to_unit_image(x: torch.Tensor) -> torch.Tensor:
    return x / 255.0 if x.detach().amax() > 2.0 else x


def edge_smooth(pred: torch.Tensor, left: torch.Tensor,
                val: torch.Tensor, alpha: float = 10.0) -> torch.Tensor:
    grey = to_unit_image(left).mean(dim=1, keepdim=True)
    gx_i = (grey[..., :, 1:] - grey[..., :, :-1]).abs()
    gy_i = (grey[..., 1:, :] - grey[..., :-1, :]).abs()
    gx_d = (pred[..., :, 1:] - pred[..., :, :-1]).abs()
    gy_d = (pred[..., 1:, :] - pred[..., :-1, :]).abs()
    vx = val[..., :, 1:] * val[..., :, :-1]
    vy = val[..., 1:, :] * val[..., :-1, :]
    lx = (gx_d * torch.exp(-alpha * gx_i) * vx).sum() / vx.sum().clamp(min=1)
    ly = (gy_d * torch.exp(-alpha * gy_i) * vy).sum() / vy.sum().clamp(min=1)
    return lx + ly


def guided_guard_loss(out: dict[str, torch.Tensor], gt: torch.Tensor,
                      val: torch.Tensor, left: torch.Tensor) -> torch.Tensor:
    if "d_pre_guided" not in out or "guided_gate" not in out:
        return gt.new_zeros(())
    d0 = out["d_pre_guided"]
    d1 = out["d_final"]
    gate = out["guided_gate"]
    if d0.shape[-2:] != gt.shape[-2:]:
        d0 = F.interpolate(d0, size=gt.shape[-2:],
                           mode="bilinear", align_corners=False)
    if gate.shape[-2:] != gt.shape[-2:]:
        gate = F.interpolate(gate, size=gt.shape[-2:],
                             mode="bilinear", align_corners=False)

    gx_g = F.pad((gt[..., :, 1:] - gt[..., :, :-1]).abs(), (0, 1, 0, 0))
    gy_g = F.pad((gt[..., 1:, :] - gt[..., :-1, :]).abs(), (0, 0, 0, 1))
    grey = to_unit_image(left).mean(dim=1, keepdim=True)
    gx_l = F.pad((grey[..., :, 1:] - grey[..., :, :-1]).abs(), (0, 1, 0, 0))
    gy_l = F.pad((grey[..., 1:, :] - grey[..., :-1, :]).abs(), (0, 0, 0, 1))
    boundary = (2.5 * (gx_g + gy_g).clamp(max=1.0)
                + 5.0 * (gx_l + gy_l).clamp(max=0.2)).clamp(0.0, 1.0)
    smooth = (1.0 - boundary) * val
    denom = smooth.sum().clamp(min=1)
    delta_pen = ((d1 - d0).abs() * smooth).sum() / denom
    gate_pen = (gate * smooth).sum() / denom
    return delta_pen + 0.25 * gate_pen


def train_loss(out: dict[str, torch.Tensor], gt: torch.Tensor,
               left: torch.Tensor, max_disp: float,
               smooth_w: float = 0.02) -> tuple[torch.Tensor, dict[str, float]]:
    val = valid_mask(gt, max_disp)
    d_full = out["d_final"]
    if d_full.shape[-2:] != gt.shape[-2:]:
        d_full = F.interpolate(d_full, size=gt.shape[-2:],
                               mode="bilinear", align_corners=True)
        out = dict(out)
        out["d_final"] = d_full

    loss = (
        1.0 * ms_l1(d_full, gt, val, 1.0)
        + 0.5 * ms_l1(out["d_half"], gt, val, 2.0)
        + 0.3 * ms_l1(out["d4"], gt, val, 4.0)
        + 0.2 * ms_l1(out["d8"], gt, val, 8.0)
        + 0.1 * ms_l1(out["d16"], gt, val, 16.0)
        + 0.5 * grad_consistency(d_full, gt, val)
        + 0.2 * threshold_hinge(d_full, gt, val)
        + 0.2 * d1_hinge(d_full, gt, val)
    )
    if "d4_gev" in out:
        loss = loss + 0.15 * ms_l1(out["d4_gev"], gt, val, 4.0)
    guided_active = bool(float(out.get(
        "guided_active", gt.new_tensor(1.0)).detach().item()) > 0.5)
    if "d_pre_guided" in out and guided_active:
        loss = loss + 0.40 * ms_l1(out["d_pre_guided"], gt, val, 1.0)
        loss = loss + 0.20 * guided_guard_loss(out, gt, val, left)
    if smooth_w > 0:
        loss = loss + smooth_w * edge_smooth(d_full, left, val)

    with torch.no_grad():
        err = (d_full - gt).abs()
        denom = val.sum().clamp(min=1)
        diag = {
            "epe": float((err * val).sum() / denom),
            "bad1": float((((err > 1.0).float() * val).sum() / denom) * 100),
        }
    return loss, diag


@torch.no_grad()
def evaluate(model: torch.nn.Module, loader: DataLoader, device: torch.device,
             max_disp: float, amp: bool) -> dict[str, float]:
    was_training = model.training
    model.eval()
    sums = {"epe": 0.0, "bad1": 0.0, "bad2": 0.0, "d1": 0.0}
    count = 0
    for left, right, gt in loader:
        left = left.to(device, non_blocking=True)
        right = right.to(device, non_blocking=True)
        gt = gt.to(device, non_blocking=True)
        with torch.amp.autocast("cuda", dtype=torch.float16,
                                enabled=amp and device.type == "cuda"):
            pred = model(left, right)
        if pred.shape[-2:] != gt.shape[-2:]:
            pred = F.interpolate(pred, size=gt.shape[-2:],
                                 mode="bilinear", align_corners=True)
        val = valid_mask(gt, max_disp)
        err = (pred - gt).abs()
        denom = val.sum().clamp(min=1)
        rel = err / gt.clamp(min=1e-6)
        d1 = ((err > 3.0) & (rel > 0.05)).float()
        sums["epe"] += float((err * val).sum() / denom)
        sums["bad1"] += float((((err > 1.0).float() * val).sum() / denom) * 100)
        sums["bad2"] += float((((err > 2.0).float() * val).sum() / denom) * 100)
        sums["d1"] += float((d1 * val).sum() / denom * 100)
        count += 1
    if was_training:
        model.train()
    return {k: v / max(count, 1) for k, v in sums.items()}


def colourise_disp(d: np.ndarray, lo: float, hi: float) -> np.ndarray:
    x = np.clip((d.astype(np.float32) - lo) / max(hi - lo, 1e-6), 0, 1)
    return cv2.applyColorMap((x * 255).astype(np.uint8), cv2.COLORMAP_TURBO)


def annotate(img: np.ndarray, text: str) -> np.ndarray:
    out = img.copy()
    cv2.rectangle(out, (0, 0), (out.shape[1], 24), (0, 0, 0), -1)
    cv2.putText(out, text, (6, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.48,
                (255, 255, 255), 1, cv2.LINE_AA)
    return out


@torch.no_grad()
def save_samples(model: torch.nn.Module, items: list[tuple[str, str, str]],
                 device: torch.device, out_dir: Path, tag: str,
                 height: int, width: int, max_disp: float,
                 amp: bool, max_rows: int = 4,
                 crop_samples: bool = True,
                 suffix: str = "") -> None:
    was_training = model.training
    model.eval()
    rows = []
    for row_idx, (lp, rp, pp) in enumerate(items[:max_rows]):
        left_bgr = cv2.imread(lp)
        right_bgr = cv2.imread(rp)
        gt_native = read_pfm(pp)
        hn, wn = gt_native.shape
        if crop_samples:
            y0 = max((hn - height) // 2, 0)
            x0 = max((wn - width) // 2, 0)
            left = left_bgr[y0:y0 + height, x0:x0 + width]
            right = right_bgr[y0:y0 + height, x0:x0 + width]
            gt = gt_native[y0:y0 + height, x0:x0 + width]
            if left.shape[:2] != (height, width):
                left = cv2.resize(left, (width, height),
                                  interpolation=cv2.INTER_AREA)
                right = cv2.resize(right, (width, height),
                                   interpolation=cv2.INTER_AREA)
                gt = cv2.resize(gt, (width, height),
                                interpolation=cv2.INTER_LINEAR)
        else:
            left = cv2.resize(left_bgr, (width, height),
                              interpolation=cv2.INTER_AREA)
            right = cv2.resize(right_bgr, (width, height),
                               interpolation=cv2.INTER_AREA)
            gt = cv2.resize(gt_native, (width, height),
                            interpolation=cv2.INTER_LINEAR) * (width / wn)
        gt[~np.isfinite(gt) | (gt < 0)] = 0
        lt = torch.from_numpy(cv2.cvtColor(left, cv2.COLOR_BGR2RGB)).float()
        rt = torch.from_numpy(cv2.cvtColor(right, cv2.COLOR_BGR2RGB)).float()
        lt = lt.permute(2, 0, 1).unsqueeze(0).to(device)
        rt = rt.permute(2, 0, 1).unsqueeze(0).to(device)
        with torch.amp.autocast("cuda", dtype=torch.float16,
                                enabled=amp and device.type == "cuda"):
            pred = model(lt, rt)
        pred_np = pred.squeeze().float().cpu().numpy()
        valid = (gt > 0) & (gt < max_disp) & np.isfinite(gt)
        lo = float(np.percentile(gt[valid], 2)) if valid.any() else 0.0
        hi = float(np.percentile(gt[valid], 98)) if valid.any() else max_disp
        epe = float(np.abs(pred_np - gt)[valid].mean()) if valid.any() else 0.0
        rows.append(np.hstack([
            annotate(left, f"L {row_idx}"),
            annotate(colourise_disp(gt, lo, hi), "GT"),
            annotate(colourise_disp(pred_np, lo, hi),
                     f"pred EPE={epe:.2f} step {tag}"),
        ]))
    if rows:
        sample_dir = out_dir / "samples"
        sample_dir.mkdir(exist_ok=True)
        cv2.imwrite(str(sample_dir / f"step_{tag}{suffix}.png"),
                    np.vstack(rows))
    if was_training:
        model.train()


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--arch", default="yolo_ctx_gev4",
                   choices=["yolo_ctx", "yolo_ctx_gate", "yolo_ctx_gev4",
                            "yolo_ctx_hstereo", "yolo_ctx_guided",
                            "yolo_ctx_sharp",
                            "yolo_ctx_hrrefine"])
    p.add_argument("--backbone", default="yolo26s",
                   choices=["ghost", "yolo26n", "yolo26s"])
    p.add_argument("--dataset", default="sceneflow",
                   choices=["sceneflow", "middlebury"],
                   help="dataset layout under --data_root.")
    p.add_argument("--data_root", default=str(ROOT / "data" / "sceneflow_driving"))
    p.add_argument("--out_root", default=str(ROOT / "model" / "checkpoints"))
    p.add_argument("--run_name", default="")
    p.add_argument("--height", type=int, default=384)
    p.add_argument("--width", type=int, default=640)
    p.add_argument("--steps", type=int, default=100000)
    p.add_argument("--batch", type=int, default=4)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--weight_decay", type=float, default=1e-5)
    p.add_argument("--n_val", type=int, default=200)
    p.add_argument("--max_train", type=int, default=0,
                   help="0 = use all training pairs")
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max_disp", type=float, default=320.0)
    p.add_argument("--crop_train", type=int, default=0,
                   help="legacy flag; prefer --train_mode.")
    p.add_argument("--train_mode", default="mixed",
                   choices=["resize", "crop", "mixed"],
                   help="resize = old full-frame resized training; crop = "
                        "native random crops; mixed = mostly resize plus "
                        "some native crops.")
    p.add_argument("--crop_prob", type=float, default=0.35,
                   help="mixed mode: probability of sampling a native crop.")
    p.add_argument("--early_stop_metric", default="resize",
                   choices=["resize", "crop", "mean"],
                   help="which validation EPE controls best/early-stop.")
    p.add_argument("--scale_min", type=float, default=0.95)
    p.add_argument("--scale_max", type=float, default=1.05)
    p.add_argument("--color_aug", type=float, default=0.1)
    p.add_argument("--erase_prob", type=float, default=0.1)
    p.add_argument("--log_every", type=int, default=50)
    p.add_argument("--val_every", type=int, default=1000)
    p.add_argument("--save_every", type=int, default=5000)
    p.add_argument("--sample_every", type=int, default=1000)
    p.add_argument("--early_stop_patience", type=int, default=10,
                   help="stop after this many validation rounds without "
                        "val EPE improvement. 0 disables early stop.")
    p.add_argument("--early_stop_min_delta", type=float, default=0.002,
                   help="minimum val EPE decrease counted as improvement.")
    p.add_argument("--min_steps", type=int, default=5000,
                   help="do not early-stop before this many steps.")
    p.add_argument("--n_samples", type=int, default=4)
    p.add_argument("--ckpt_in", default="",
                   help="model-only warm start; optimizer/scheduler reset.")
    p.add_argument("--resume", default="")
    p.add_argument("--amp", type=int, default=1)
    p.add_argument("--smooth_w", type=float, default=0.02)
    p.add_argument("--guided_start_step", type=int, default=10000,
                   help="yolo_ctx_guided only: bypass guided refinement before "
                        "this step so the GEV4/base path learns first.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    distributed, rank, local_rank, world = setup_dist()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.benchmark = True

    device = torch.device(
        f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    amp = bool(args.amp) and device.type == "cuda"

    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_name = args.run_name or f"{args.arch}_{args.backbone}_{ts}"
    out_dir = Path(args.out_root) / run_name
    if is_rank0():
        out_dir.mkdir(parents=True, exist_ok=True)
    if distributed:
        dist.barrier()

    if is_rank0():
        print(f"enumerating {args.dataset} pairs under {args.data_root}...")
    if args.dataset == "middlebury":
        items = enumerate_middlebury(args.data_root)
    else:
        items = enumerate_pairs(args.data_root)
    if args.dataset == "middlebury" and len(items) <= args.n_val:
        old_n_val = args.n_val
        args.n_val = max(1, min(20, len(items) // 5))
        if is_rank0():
            print(f"middlebury has {len(items)} pairs; adjusted n_val "
                  f"{old_n_val} -> {args.n_val}")
    if len(items) <= args.n_val:
        raise RuntimeError(f"not enough pairs: found {len(items)}, n_val={args.n_val}")
    train_items, val_items = train_val_split(items, args.n_val, seed=args.seed)
    if args.max_train > 0:
        train_items = train_items[:args.max_train]
    if is_rank0():
        print(f"train={len(train_items)}  val={len(val_items)}")

    if args.dataset == "middlebury":
        if args.train_mode == "mixed":
            train_ds = MiddleburyMixed(
                train_items, args.height, args.width,
                crop_prob=args.crop_prob, scale_min=args.scale_min,
                scale_max=args.scale_max, color_aug=args.color_aug)
        elif args.train_mode == "crop":
            train_ds = MiddleburyCrop(
                train_items, args.height, args.width, train=True,
                scale_min=args.scale_min, scale_max=args.scale_max,
                color_aug=args.color_aug)
        else:
            train_ds = MiddleburyResize(train_items, args.height, args.width)
        val_resize_ds = MiddleburyResize(val_items, args.height, args.width)
        val_crop_ds = MiddleburyCrop(
            val_items, args.height, args.width, train=False,
            scale_min=1.0, scale_max=1.0, color_aug=0.0)
    else:
        if args.train_mode == "mixed":
            train_ds = SceneFlowDrivingMixed(
                train_items, args.height, args.width, crop_prob=args.crop_prob,
                scale_min=args.scale_min, scale_max=args.scale_max,
                color_aug=args.color_aug, erase_prob=args.erase_prob)
        elif args.train_mode == "crop":
            train_ds = SceneFlowDrivingCrop(
                train_items, args.height, args.width, train=True,
                scale_min=args.scale_min, scale_max=args.scale_max,
                color_aug=args.color_aug, erase_prob=args.erase_prob)
        else:
            train_ds = SceneFlowDriving(train_items, args.height, args.width)
        val_resize_ds = SceneFlowDriving(val_items, args.height, args.width)
        val_crop_ds = SceneFlowDrivingCrop(
            val_items, args.height, args.width, train=False,
            scale_min=1.0, scale_max=1.0, color_aug=0.0, erase_prob=0.0)
    loader_kwargs = {
        "batch_size": args.batch,
        "shuffle": not distributed,
        "num_workers": args.num_workers,
        "pin_memory": device.type == "cuda",
        "drop_last": True,
    }
    train_sampler = DistributedSampler(
        train_ds, num_replicas=world, rank=rank, shuffle=True,
        drop_last=True) if distributed else None
    if args.num_workers > 0:
        loader_kwargs["persistent_workers"] = True
        loader_kwargs["prefetch_factor"] = 2
    train_loader = DataLoader(train_ds, sampler=train_sampler, **loader_kwargs)
    val_workers = 0 if args.num_workers == 0 else max(1, args.num_workers // 2)
    val_resize_loader = DataLoader(
        val_resize_ds, batch_size=1, shuffle=False, num_workers=val_workers,
        pin_memory=device.type == "cuda")
    val_crop_loader = DataLoader(
        val_crop_ds, batch_size=1, shuffle=False, num_workers=val_workers,
        pin_memory=device.type == "cuda")

    _arch_flags["slant_patch_radius"] = 1
    model, cfg = build_model(args.arch, backbone=args.backbone)
    model = model.to(device)
    params = sum(p.numel() for p in model.parameters())
    if is_rank0():
        print(f"model={args.arch}/{args.backbone} params={params/1e6:.3f}M "
              f"world={world} per_gpu_batch={args.batch}")

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr,
                            weight_decay=args.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=args.steps, eta_min=args.lr * 0.05)
    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    start_step = 0
    best_epe = float("inf")
    bad_val_rounds = 0

    if args.ckpt_in:
        ckpt = torch.load(args.ckpt_in, map_location=device, weights_only=False)
        state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
        model.load_state_dict(state, strict=True)
        if is_rank0():
            print(f"loaded model weights from {args.ckpt_in}")

    if args.resume:
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model"], strict=True)
        if "opt" in ckpt:
            opt.load_state_dict(ckpt["opt"])
        if "sched" in ckpt:
            sched.load_state_dict(ckpt["sched"])
        if "scaler" in ckpt and amp:
            scaler.load_state_dict(ckpt["scaler"])
        start_step = int(ckpt.get("step", 0))
        best_epe = float(ckpt.get("best_epe", best_epe))
        if is_rank0():
            print(f"resumed {args.resume} at step {start_step}, best_epe={best_epe:.4f}")

    if distributed:
        model = DDP(model, device_ids=[local_rank] if device.type == "cuda" else None)

    meta = {
        "arch": args.arch,
        "backbone": args.backbone,
        "dataset": args.dataset,
        "height": args.height,
        "width": args.width,
        "steps": args.steps,
        "batch": args.batch,
        "global_batch": args.batch * world,
        "lr": args.lr,
        "n_train": len(train_items),
        "n_val": len(val_items),
        "seed": args.seed,
        "max_disp": args.max_disp,
        "train_mode": args.train_mode,
        "crop_train": bool(args.crop_train),
        "crop_prob": args.crop_prob,
        "early_stop_metric": args.early_stop_metric,
        "scale_min": args.scale_min,
        "scale_max": args.scale_max,
        "color_aug": args.color_aug,
        "erase_prob": args.erase_prob,
        "guided_start_step": args.guided_start_step,
        "params_total_M": round(params / 1e6, 4),
        "device": str(device),
        "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
        "distributed": distributed,
        "world_size": world,
        "pytorch": torch.__version__,
        "platform": platform.platform(),
        "started_at": datetime.now().isoformat(timespec="seconds"),
        "cfg": getattr(cfg, "__dict__", str(cfg)),
    }
    if is_rank0():
        (out_dir / "meta.json").write_text(json.dumps(meta, indent=2))

    csv_path = out_dir / "train.csv"
    write_header = not csv_path.exists() or start_step == 0
    cf = open(csv_path, "a", newline="") if is_rank0() else None
    writer = csv.writer(cf) if cf is not None else None
    if is_rank0() and write_header:
        writer.writerow(["step", "loss", "train_epe", "train_bad1",
                         "val_epe", "val_bad1", "val_bad2", "val_d1",
                         "val_resize_epe", "val_resize_bad1",
                         "val_crop_epe", "val_crop_bad1",
                         "lr", "elapsed_s", "peak_gb"])

    rng = np.random.default_rng(args.seed + 17)
    sample_items = [val_items[i] for i in rng.choice(
        len(val_items), size=min(args.n_samples, len(val_items)), replace=False)]

    model.train()
    epoch = 0
    if train_sampler is not None:
        train_sampler.set_epoch(epoch)
    it = iter(train_loader)
    t0 = time.time()
    running_loss, running_epe, running_bad1 = [], [], []

    def save_ckpt(name: str, step: int, val_metrics: dict[str, float] | None):
        payload = {
            "model": unwrap_model(model).state_dict(),
            "opt": opt.state_dict(),
            "sched": sched.state_dict(),
            "scaler": scaler.state_dict() if amp else None,
            "step": step,
            "best_epe": best_epe,
            "args": vars(args),
            "val": val_metrics,
        }
        torch.save(payload, out_dir / name)

    for step in range(start_step + 1, args.steps + 1):
        try:
            left, right, gt = next(it)
        except StopIteration:
            epoch += 1
            if train_sampler is not None:
                train_sampler.set_epoch(epoch)
            it = iter(train_loader)
            left, right, gt = next(it)
        left = left.to(device, non_blocking=True)
        right = right.to(device, non_blocking=True)
        gt = gt.to(device, non_blocking=True)

        opt.zero_grad(set_to_none=True)
        guided_active = not (
            args.arch == "yolo_ctx_guided"
            and step < max(1, args.guided_start_step)
        )
        raw_model = unwrap_model(model)
        if hasattr(raw_model, "use_guided_default"):
            raw_model.use_guided_default = guided_active
        with torch.amp.autocast("cuda", dtype=torch.float16, enabled=amp):
            out = model(left, right, aux=True)
            loss, diag = train_loss(out, gt, left, args.max_disp, args.smooth_w)
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(opt)
        scaler.update()
        sched.step()

        running_loss.append(float(loss.detach()))
        running_epe.append(diag["epe"])
        running_bad1.append(diag["bad1"])

        val_metrics = None
        val_resize = None
        val_crop = None
        stop_now = False
        do_val = step == 1 or step % args.val_every == 0 or step == args.steps
        if do_val and is_rank0():
            val_resize = evaluate(
                unwrap_model(model), val_resize_loader, device, args.max_disp, amp)
            val_crop = evaluate(
                unwrap_model(model), val_crop_loader, device, args.max_disp, amp)
            if args.early_stop_metric == "crop":
                val_metrics = val_crop
            elif args.early_stop_metric == "mean":
                val_metrics = {
                    k: 0.5 * (val_resize[k] + val_crop[k])
                    for k in val_resize
                }
            else:
                val_metrics = val_resize
            improved = val_metrics["epe"] < (best_epe - args.early_stop_min_delta)
            if improved:
                best_epe = val_metrics["epe"]
                bad_val_rounds = 0
                save_ckpt("best.pth", step, val_metrics)
            else:
                bad_val_rounds += 1
            save_ckpt("latest.pth", step, val_metrics)
            stop_now = (
                args.early_stop_patience > 0
                and step >= args.min_steps
                and bad_val_rounds >= args.early_stop_patience
            )
        if do_val and distributed:
            stop_tensor = torch.tensor(
                [1 if stop_now else 0], device=device, dtype=torch.int32)
            dist.broadcast(stop_tensor, src=0)
            stop_now = bool(stop_tensor.item())
            dist.barrier()

        if is_rank0() and (step == 1 or step % args.log_every == 0 or do_val):
            peak = (torch.cuda.max_memory_allocated() / 1e9
                    if device.type == "cuda" else 0.0)
            row = [
                step,
                float(np.mean(running_loss[-args.log_every:])),
                float(np.mean(running_epe[-args.log_every:])),
                float(np.mean(running_bad1[-args.log_every:])),
                "" if val_metrics is None else val_metrics["epe"],
                "" if val_metrics is None else val_metrics["bad1"],
                "" if val_metrics is None else val_metrics["bad2"],
                "" if val_metrics is None else val_metrics["d1"],
                "" if val_resize is None else val_resize["epe"],
                "" if val_resize is None else val_resize["bad1"],
                "" if val_crop is None else val_crop["epe"],
                "" if val_crop is None else val_crop["bad1"],
                opt.param_groups[0]["lr"],
                time.time() - t0,
                peak,
            ]
            writer.writerow(row)
            cf.flush()
            msg = (f"step {step:>6d} loss={row[1]:.4f} "
                   f"trainEPE={row[2]:.3f} bad1={row[3]:.1f}% "
                   f"lr={opt.param_groups[0]['lr']:.2e} peak={peak:.2f}GB")
            if val_metrics is not None:
                msg += (f" | valEPE={val_metrics['epe']:.3f} "
                        f"valBad1={val_metrics['bad1']:.1f}% "
                        f"resize={val_resize['epe']:.3f} "
                        f"crop={val_crop['epe']:.3f} "
                        f"best={best_epe:.3f} "
                        f"stall={bad_val_rounds}/{args.early_stop_patience}")
            print(msg, flush=True)

        if is_rank0() and (step == 1 or step % args.sample_every == 0 or step == args.steps):
            save_samples(unwrap_model(model), sample_items, device, out_dir, f"{step:06d}",
                         args.height, args.width, args.max_disp, amp,
                         crop_samples=False, suffix="_resize")
            save_samples(unwrap_model(model), sample_items, device, out_dir, f"{step:06d}",
                         args.height, args.width, args.max_disp, amp,
                         crop_samples=True, suffix="_crop")

        if is_rank0() and step % args.save_every == 0:
            save_ckpt(f"step_{step:06d}.pth", step, val_metrics)

        if do_val and stop_now:
            if is_rank0():
                print(
                    f"early stopping at step {step}: val EPE did not improve by "
                    f"{args.early_stop_min_delta:g} for {bad_val_rounds} "
                    "validation rounds.",
                    flush=True,
                )
            break

    if cf is not None:
        cf.close()
    if is_rank0():
        print(f"done. outputs -> {out_dir}")
        print(f"best val EPE: {best_epe:.4f}")
    cleanup_dist()


if __name__ == "__main__":
    main()
