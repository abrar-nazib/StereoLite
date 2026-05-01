"""Train StereoLite on Scene Flow Driving with REAL disparity GT.

This is the apples-to-apples training: HITNet's pretrained weights came from
the same dataset under the same supervision quality, so a fair comparison is
"both models on Scene Flow Driving test split with real GT."

Usage:
    python3 model/scripts/train_sceneflow.py --steps 5000 --batch 2 --lr 2e-4
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import os
import sys
import time

import cv2
import numpy as np
import torch

PROJ = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(PROJ, "model"))
sys.path.insert(0, os.path.join(PROJ, "model", "designs"))
sys.path.insert(0, os.path.join(PROJ, "model", "scripts"))

from sceneflow_loader import enumerate_pairs, train_val_split, SceneFlowDriving


def loss_l1_masked(pred, target, max_disp=192.0):
    """Single-scale L1 (legacy; kept for back-compat)."""
    valid = (target > 0) & (target < max_disp) & torch.isfinite(target)
    err = (pred - target).abs() * valid
    n = valid.sum().clamp(min=1.0)
    return err.sum() / n


def _valid_mask(target, max_disp=192.0):
    return ((target > 0) & (target < max_disp)
            & torch.isfinite(target)).float()


def _ms_l1(pred, gt, val, scale):
    """Per-scale L1; downsamples GT to pred's spatial size and rescales."""
    if pred.shape[-2:] != gt.shape[-2:]:
        from torch.nn.functional import interpolate
        pred = interpolate(pred, size=gt.shape[-2:],
                           mode="bilinear", align_corners=False) * scale
    return ((pred - gt).abs() * val).sum() / val.sum().clamp(min=1)


def _grad_consistency(pred, gt, val):
    gx_p = pred[..., :, 1:] - pred[..., :, :-1]
    gx_g = gt[..., :, 1:] - gt[..., :, :-1]
    gy_p = pred[..., 1:, :] - pred[..., :-1, :]
    gy_g = gt[..., 1:, :] - gt[..., :-1, :]
    vx = val[..., :, 1:] * val[..., :, :-1]
    vy = val[..., 1:, :] * val[..., :-1, :]
    lx = ((gx_p - gx_g).abs() * vx).sum() / vx.sum().clamp(min=1)
    ly = ((gy_p - gy_g).abs() * vy).sum() / vy.sum().clamp(min=1)
    return lx + ly


def _threshold_hinge_stack(pred, gt, val,
                           thresholds=(0.5, 1.0, 2.0, 3.0)):
    err = (pred - gt).abs()
    total = sum((err - t).clamp(min=0) ** 2 for t in thresholds)
    return (total * val).sum() / val.sum().clamp(min=1)


def _d1_relative_hinge(pred, gt, val):
    """KITTI D1-all: penalise pixels with abs err > 3 px AND rel err > 5%."""
    err = (pred - gt).abs()
    rel = err / gt.clamp(min=1e-6)
    is_d1 = ((err > 3.0) & (rel > 0.05)).float()
    h = (err - 3.0).clamp(min=0) ** 2 * is_d1
    return (h * val).sum() / val.sum().clamp(min=1)


def loss_stack_d1(preds: dict, gt: torch.Tensor, max_disp: float = 192.0):
    """Production loss for StereoLite — `stack_d1` from the loss sweep.

    On the 9-variant overfit ablation
    (model/benchmarks/loss_ablation_20260501-132948), this formulation
    led on EPE (0.591), bad-2 (5.64%), bad-3 (3.35%), and D1-all
    (3.35%). It targets a balance of sub-pixel accuracy AND outlier
    suppression, which matches the edge-deployment use-case claim.

    Components:
      - Multi-scale L1 over (full, /2, /4, /8, /16) with weights
        (1.0, 0.5, 0.3, 0.2, 0.1).
      - Sobel gradient consistency on full-res prediction (×0.5).
      - Threshold-stack squared hinge at {0.5, 1, 2, 3} px (×0.3).
      - KITTI D1-relative squared hinge (>3 px AND >5%) (×0.2).

    Expects `preds` to be the aux dict from StereoLite.forward(L, R, aux=True),
    with keys d_final, d_half, d4, d8, d16. `gt` is full-resolution
    ground-truth disparity.
    """
    val = _valid_mask(gt, max_disp)
    d_full = preds["d_final"]
    d_half = preds["d_half"]
    d4 = preds["d4"]
    d8 = preds["d8"]
    d16 = preds["d16"]
    base = (1.0 * _ms_l1(d_full, gt, val, 1.0)
            + 0.5 * _ms_l1(d_half, gt, val, 2.0)
            + 0.3 * _ms_l1(d4, gt, val, 4.0)
            + 0.2 * _ms_l1(d8, gt, val, 8.0)
            + 0.1 * _ms_l1(d16, gt, val, 16.0))
    return (base
            + 0.5 * _grad_consistency(d_full, gt, val)
            + 0.3 * _threshold_hinge_stack(d_full, gt, val)
            + 0.2 * _d1_relative_hinge(d_full, gt, val))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data_root", default=os.path.join(PROJ, "data", "sceneflow_driving"))
    p.add_argument("--ckpt_in", default=None)
    p.add_argument("--ckpt_out", default=os.path.join(PROJ, "model", "checkpoints", "stereolite_sf.pth"))
    p.add_argument("--steps", type=int, default=5000)
    p.add_argument("--batch", type=int, default=2)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--inf_h", type=int, default=384)
    p.add_argument("--inf_w", type=int, default=640)
    p.add_argument("--n_val", type=int, default=200)
    p.add_argument("--num_workers", type=int, default=2)
    p.add_argument("--max_train", type=int, default=None,
                   help="cap on training set size (None = use all)")
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"enumerating pairs under {args.data_root}...")
    items = enumerate_pairs(args.data_root)
    print(f"found {len(items)} pairs total")
    train, val = train_val_split(items, args.n_val)
    if args.max_train is not None:
        train = train[: args.max_train]
    print(f"train={len(train)}  val={len(val)}")

    train_ds = SceneFlowDriving(train, args.inf_h, args.inf_w)
    loader = torch.utils.data.DataLoader(train_ds, batch_size=args.batch,
                                          shuffle=True, num_workers=args.num_workers,
                                          pin_memory=True, persistent_workers=True)

    from StereoLite import StereoLite
    model = StereoLite().to(device)
    if args.ckpt_in is not None and os.path.exists(args.ckpt_in):
        sd = torch.load(args.ckpt_in, map_location=device, weights_only=False)
        if "model" in sd:
            sd = sd["model"]
        model.load_state_dict(sd, strict=True)
        print(f"resumed from {args.ckpt_in}")
    n_params = sum(p.numel() for p in model.parameters())
    print(f"StereoLite params: {n_params/1e6:.3f} M")

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.steps,
                                                        eta_min=args.lr / 10)

    log_path = args.ckpt_out.replace(".pth", "_train.csv")
    fp_log = open(log_path, "w", newline="")
    w_csv = csv.writer(fp_log)
    w_csv.writerow(["step", "loss", "lr", "ms_per_step"])

    model.train()
    t0 = time.time()
    running = []
    it = iter(loader)
    for step in range(1, args.steps + 1):
        try:
            L, R, D = next(it)
        except StopIteration:
            it = iter(loader)
            L, R, D = next(it)
        L, R, D = L.to(device), R.to(device), D.to(device)
        # Multi-scale outputs needed for the stack_d1 loss.
        preds = model(L, R, aux=True)
        # Resize d_final to match GT if needed (encoder strides may not
        # divide the input shape evenly).
        if preds["d_final"].shape[-2:] != D.shape[-2:]:
            preds["d_final"] = torch.nn.functional.interpolate(
                preds["d_final"], size=D.shape[-2:],
                mode="bilinear", align_corners=True)
        loss = loss_stack_d1(preds, D)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        running.append(loss.item())
        if step == 1 or step % 50 == 0 or step == args.steps:
            ms = (time.time() - t0) * 1000 / step
            mean = float(np.mean(running[-50:]))
            cur_lr = opt.param_groups[0]["lr"]
            w_csv.writerow([step, mean, cur_lr, ms])
            fp_log.flush()
            print(f"  step {step:5d}  loss={mean:6.3f}  lr={cur_lr:.2e}  {ms:6.1f} ms/step")

    fp_log.close()
    torch.save({"model": model.state_dict(),
                "params_M": n_params / 1e6,
                "args": vars(args)}, args.ckpt_out)
    print(f"saved -> {args.ckpt_out}")
    print(f"train log -> {log_path}")

    # Quick val EPE
    model.train()  # GroupNorm everywhere; train mode is fine
    val_ds = SceneFlowDriving(val, args.inf_h, args.inf_w)
    val_loader = torch.utils.data.DataLoader(val_ds, batch_size=1, shuffle=False)
    epes, bad1s = [], []
    with torch.no_grad():
        for L, R, D in val_loader:
            L, R, D = L.to(device), R.to(device), D.to(device)
            pred = model(L, R)
            if pred.shape != D.shape:
                pred = torch.nn.functional.interpolate(pred, size=D.shape[-2:],
                                                        mode="bilinear", align_corners=True)
            valid = (D > 0) & (D < 192) & torch.isfinite(D)
            n = valid.sum().clamp(min=1.0)
            epe = ((pred - D).abs() * valid).sum() / n
            bad1 = (((pred - D).abs() > 1.0).float() * valid).sum() / n
            epes.append(float(epe.item()))
            bad1s.append(float(bad1.item()))
    print(f"\nVAL on Scene Flow Driving ({len(epes)} pairs):")
    print(f"  EPE   = {float(np.mean(epes)):.3f} px")
    print(f"  bad1% = {float(np.mean(bad1s))*100:.1f}%")


if __name__ == "__main__":
    main()
