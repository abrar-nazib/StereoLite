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
    valid = (target > 0) & (target < max_disp) & torch.isfinite(target)
    err = (pred - target).abs() * valid
    n = valid.sum().clamp(min=1.0)
    return err.sum() / n


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
        pred = model(L, R)
        if pred.shape != D.shape:
            pred = torch.nn.functional.interpolate(pred, size=D.shape[-2:],
                                                    mode="bilinear", align_corners=True)
        loss = loss_l1_masked(pred, D)
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
