"""Sharpness-focused trainer for TileFMStereo with DAv2.

Trains on a small Scene Flow Driving subset (300-500 pairs) with:
  - L1 loss on disparity
  - Gradient-preserving loss (L1 on Sobel-x/y of disparity), weighted
  - Periodic sample-panel dumps so we can inspect sharpness as it evolves

Usage:
    python3 model/scripts/train_sharpness.py \
        --steps 3000 --max_train 400 --grad_loss_w 0.5
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
os.environ.setdefault("XFORMERS_DISABLED", "1")

from sceneflow_loader import enumerate_pairs, train_val_split, SceneFlowDriving


def l1_masked(pred, target, max_disp=192.0):
    valid = (target > 0) & (target < max_disp) & torch.isfinite(target)
    err = (pred - target).abs() * valid
    n = valid.sum().clamp(min=1.0)
    return err.sum() / n, valid


def sobel_xy(d: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    kx = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
                      device=d.device, dtype=d.dtype).view(1, 1, 3, 3) / 8
    ky = kx.transpose(-1, -2).contiguous()
    d_pad = torch.nn.functional.pad(d, (1, 1, 1, 1), mode="replicate")
    gx = torch.nn.functional.conv2d(d_pad, kx)
    gy = torch.nn.functional.conv2d(d_pad, ky)
    return gx, gy


def grad_loss(pred, target, valid):
    gx_p, gy_p = sobel_xy(pred)
    gx_t, gy_t = sobel_xy(target)
    err = ((gx_p - gx_t).abs() + (gy_p - gy_t).abs()) * valid
    n = valid.sum().clamp(min=1.0)
    return err.sum() / n


# Scale weights per intermediate disparity prediction. Stronger direct
# supervision at 1/8 (d8_cv) is the key change — it stops the network
# from collapsing to a monocular-only prediction by forcing the cost
# volume to learn matching.
SCALE_WEIGHTS = {
    "d_final": 1.0,
    "d_full_raw": 0.8,    # pre-refinement full-res (supervise before residual)
    "d2": 0.7,
    "d4": 0.5,
    "d4_cas": 0.8,        # narrow-range cascade at 1/4 — strong signal
    "d8": 0.3,
    "d8_cv": 1.0,
    "d16": 0.3,
    "d32": 0.2,
}


def multiscale_loss(preds: dict, D_full: torch.Tensor,
                    grad_w: float = 0.5, hinge_w: float = 0.3,
                    max_disp: float = 192.0):
    """Compute L1 + gradient + bad-1-hinge loss at every scale, weighted.

    D_full: GT disparity at full input resolution.
    Each intermediate disparity prediction is at scale 1/k and in 1/k-px
    units. We downsample GT to match, and scale by (W_pred / W_full).
    """
    total = torch.zeros((), device=D_full.device, dtype=D_full.dtype)
    diag = {}
    for key, pred in preds.items():
        w = SCALE_WEIGHTS.get(key)
        if w is None:
            continue
        target_hw = pred.shape[-2:]
        full_hw = D_full.shape[-2:]
        scale = target_hw[1] / full_hw[1]
        D_s = torch.nn.functional.interpolate(
            D_full, size=target_hw, mode="bilinear", align_corners=True) * scale
        D_s[~torch.isfinite(D_s)] = 0
        valid = (D_s > 0) & (D_s < max_disp * scale) & torch.isfinite(D_s)
        n = valid.sum().clamp(min=1.0)
        diff = (pred - D_s).abs() * valid
        l1 = diff.sum() / n
        g = grad_loss(pred, D_s, valid) if w > 0.1 else torch.zeros_like(l1)
        # Bad-1 hinge: penalize pixels >1 px off harder. Threshold scaled.
        thresh = 1.0 * scale
        hinge = (torch.relu(diff - thresh)).sum() / n
        part = l1 + grad_w * g + hinge_w * hinge
        total = total + w * part
        diag[key] = (float(l1.item()), float(hinge.item()))
    return total, diag


def colourise(d, lo, hi):
    d = d.astype(np.float32)
    v = np.clip((d - lo) / max(hi - lo, 1e-6), 0, 1) * 255
    return cv2.applyColorMap(v.astype(np.uint8), cv2.COLORMAP_TURBO)


def annotate(img, text, h=22):
    out = img.copy()
    cv2.rectangle(out, (0, 0), (out.shape[1], h), (0, 0, 0), -1)
    cv2.putText(out, text, (6, h - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.42,
                (255, 255, 255), 1, cv2.LINE_AA)
    return out


def save_sample_panels(model, track_pairs, device, out_dir, step, inf_h, inf_w,
                       history: list[dict], n_watch: int = 3):
    """Dump a per-step row for each tracked pair (one dir per pair) plus
    two montages:
      <out>/montage/step_XXXXX.png       -> first n_watch pairs (live window)
      <out>/montage_full/step_XXXXX.png  -> all track_pairs (archival)
    `history` accumulates per-pair EPE across steps for progress.csv."""
    from sceneflow_loader import read_pfm
    model_was_training = model.training
    model.train()
    pairs_dir = os.path.join(out_dir, "pairs")
    montage_dir = os.path.join(out_dir, "montage")
    montage_full_dir = os.path.join(out_dir, "montage_full")
    os.makedirs(pairs_dir, exist_ok=True)
    os.makedirs(montage_dir, exist_ok=True)
    os.makedirs(montage_full_dir, exist_ok=True)

    epes = []
    rows = []
    ms_list = []
    cuda_sync = device.type == "cuda"
    # Warmup once so the first-panel timing isn't distorted
    if track_pairs:
        lp0, rp0, _ = track_pairs[0]
        Lw = cv2.imread(lp0); Rw = cv2.imread(rp0)
        Lw = cv2.resize(Lw, (inf_w, inf_h)); Rw = cv2.resize(Rw, (inf_w, inf_h))
        Lt = torch.from_numpy(cv2.cvtColor(Lw, cv2.COLOR_BGR2RGB)).float().permute(2, 0, 1).unsqueeze(0).to(device)
        Rt = torch.from_numpy(cv2.cvtColor(Rw, cv2.COLOR_BGR2RGB)).float().permute(2, 0, 1).unsqueeze(0).to(device)
        with torch.no_grad():
            _ = model(Lt, Rt)
        if cuda_sync:
            torch.cuda.synchronize()

    with torch.no_grad():
        for i, (lp, rp, pp) in enumerate(track_pairs):
            L = cv2.imread(lp); R = cv2.imread(rp)
            D_n = read_pfm(pp)
            H_n, W_n = D_n.shape
            L_in = cv2.resize(L, (inf_w, inf_h))
            R_in = cv2.resize(R, (inf_w, inf_h))
            sx = inf_w / W_n
            D = cv2.resize(D_n, (inf_w, inf_h)) * sx
            D[~np.isfinite(D) | (D < 0)] = 0

            Lt = torch.from_numpy(cv2.cvtColor(L_in, cv2.COLOR_BGR2RGB)).float().permute(2, 0, 1).unsqueeze(0).to(device)
            Rt = torch.from_numpy(cv2.cvtColor(R_in, cv2.COLOR_BGR2RGB)).float().permute(2, 0, 1).unsqueeze(0).to(device)
            if cuda_sync:
                torch.cuda.synchronize()
            t0 = time.time()
            P_t = model(Lt, Rt)
            if cuda_sync:
                torch.cuda.synchronize()
            ms = (time.time() - t0) * 1000
            ms_list.append(ms)
            P = P_t.squeeze().cpu().numpy()

            valid = (D > 0) & (D < 192) & np.isfinite(D)
            if valid.sum() > 16:
                lo = float(np.percentile(D[valid], 2))
                hi = float(np.percentile(D[valid], 98))
            else:
                lo, hi = 0.0, 60.0
            epe = float((np.abs(P - D) * valid).sum() / max(valid.sum(), 1))
            epes.append(epe)

            tiles = [
                annotate(L_in, f"L pair{i:02d}  {inf_w}x{inf_h}"),
                annotate(colourise(D, lo, hi), f"GT  {D[valid].min():.1f}-{D[valid].max():.1f}"),
                annotate(colourise(P, lo, hi),
                         f"pred  EPE={epe:.2f}  {ms:.0f}ms  step {step}"),
            ]
            row = np.hstack(tiles)
            pdir = os.path.join(pairs_dir, f"pair_{i:02d}")
            os.makedirs(pdir, exist_ok=True)
            cv2.imwrite(os.path.join(pdir, f"step_{step:05d}.png"), row)
            rows.append(row)

    # Live-window montage: just the first n_watch pairs stacked vertically
    watch = rows[:n_watch]
    live_montage = np.vstack(watch)
    cv2.imwrite(os.path.join(montage_dir, f"step_{step:05d}.png"), live_montage)

    # Archival montage: all tracked pairs, 4 per row
    per_row = 4
    full_rows = []
    for r0 in range(0, len(rows), per_row):
        group = rows[r0:r0 + per_row]
        if len(group) < per_row:
            pad = np.zeros_like(group[0])
            while len(group) < per_row:
                group.append(pad)
        full_rows.append(np.hstack(group))
    if full_rows:
        full_montage = np.vstack(full_rows)
        cv2.imwrite(os.path.join(montage_full_dir, f"step_{step:05d}.png"),
                    full_montage)

    mean_epe = float(np.mean(epes))
    med_ms = float(np.median(ms_list)) if ms_list else 0.0
    history.append({"step": step, "mean_epe": mean_epe, "epes": epes,
                    "median_ms": med_ms})
    print(f"  -> step {step}: {len(epes)}-pair mean EPE = {mean_epe:.3f} px  "
          f"(range {min(epes):.2f} .. {max(epes):.2f})  "
          f"median inference {med_ms:.1f} ms")
    if not model_was_training:
        model.eval()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data_root", default=os.path.join(PROJ, "data", "sceneflow_driving"))
    p.add_argument("--ckpt_out", default=os.path.join(PROJ, "model", "checkpoints", "tilefm_fm_sharp.pth"))
    p.add_argument("--steps", type=int, default=3000)
    p.add_argument("--batch", type=int, default=2)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--inf_h", type=int, default=384)
    p.add_argument("--inf_w", type=int, default=640)
    p.add_argument("--n_val", type=int, default=50)
    p.add_argument("--max_train", type=int, default=400)
    p.add_argument("--num_workers", type=int, default=2)
    p.add_argument("--grad_loss_w", type=float, default=0.5,
                   help="weight of gradient loss term")
    p.add_argument("--hinge_loss_w", type=float, default=0.3,
                   help="weight of bad-1 hinge term")
    p.add_argument("--amp", action="store_true",
                   help="enable FP16 autocast for forward+backward")
    p.add_argument("--ckpt_in", default=None,
                   help="resume from this checkpoint (trainable params only)")
    p.add_argument("--panel_every", type=int, default=200,
                   help="save tracking panels every N steps")
    p.add_argument("--n_track_pairs", type=int, default=20,
                   help="number of fixed pairs to track across steps")
    p.add_argument("--n_watch_pairs", type=int, default=3,
                   help="first N tracked pairs shown in the live watcher")
    p.add_argument("--overfit", action="store_true",
                   help="overfit on the first --n_track_pairs of training set; "
                        "no held-out val, track on training data")
    p.add_argument("--cv_max_disp", type=int, default=24,
                   help="cost-volume max disparities at 1/8 scale")
    p.add_argument("--log_every", type=int, default=50)
    p.add_argument("--panel_dir", default=None)
    args = p.parse_args()

    if args.panel_dir is None:
        ts = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
        args.panel_dir = os.path.join(PROJ, "model", "benchmarks", f"sharp_{ts}")
    os.makedirs(args.panel_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device.type}  panel_dir={args.panel_dir}")

    items = enumerate_pairs(args.data_root)
    if args.overfit:
        # Use the first n_track_pairs as BOTH train and track set.
        track_pairs = items[: args.n_track_pairs]
        train = track_pairs
        val = track_pairs
    else:
        train, val = train_val_split(items, args.n_val)
        if args.max_train is not None:
            train = train[: args.max_train]
        if len(val) < args.n_track_pairs:
            track_pairs = val
        else:
            step_stride = max(1, len(val) // args.n_track_pairs)
            track_pairs = [val[i * step_stride] for i in range(args.n_track_pairs)]
    print(f"train={len(train)}  val={len(val)}  track={len(track_pairs)} pairs  "
          f"overfit={args.overfit}  watch={args.n_watch_pairs}")

    train_ds = SceneFlowDriving(train, args.inf_h, args.inf_w)
    loader = torch.utils.data.DataLoader(train_ds, batch_size=args.batch,
                                         shuffle=True, num_workers=args.num_workers,
                                         pin_memory=True, persistent_workers=True)

    from d1_tile import LiteFMStereo, LiteConfig
    model = LiteFMStereo(LiteConfig(use_dav2=True,
                                    cv_max_disp=args.cv_max_disp)).to(device)
    if args.ckpt_in is not None and os.path.exists(args.ckpt_in):
        ck = torch.load(args.ckpt_in, map_location=device, weights_only=False)
        sd = ck["model"] if "model" in ck else ck
        missing, unexpected = model.load_state_dict(sd, strict=False)
        print(f"resumed from {args.ckpt_in} (missing={len(missing)} unexpected={len(unexpected)})")
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    print(f"trainable={trainable/1e6:.3f} M  frozen DAv2={frozen/1e6:.3f} M")

    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                            lr=args.lr, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.steps,
                                                       eta_min=args.lr / 10)
    use_amp = args.amp and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    print(f"AMP (FP16 autocast): {'on' if use_amp else 'off'}")

    log_path = args.ckpt_out.replace(".pth", "_train.csv")
    fp_log = open(log_path, "w", newline="")
    w_csv = csv.writer(fp_log)
    w_csv.writerow(["step", "loss", "l1_final", "l1_cv", "lr", "ms_per_step"])

    history: list[dict] = []
    progress_path = os.path.join(args.panel_dir, "progress.csv")

    model.train()
    t0 = time.time()
    running_loss, running_l1_final, running_l1_cv = [], [], []
    it = iter(loader)
    for step in range(1, args.steps + 1):
        try:
            L, R, D = next(it)
        except StopIteration:
            it = iter(loader)
            L, R, D = next(it)
        L, R, D = L.to(device), R.to(device), D.to(device)
        with torch.amp.autocast("cuda", enabled=use_amp, dtype=torch.float16):
            preds = model(L, R, aux=True)
            if preds["d_final"].shape[-2:] != D.shape[-2:]:
                preds["d_final"] = torch.nn.functional.interpolate(
                    preds["d_final"], size=D.shape[-2:],
                    mode="bilinear", align_corners=True)
            # Loss computed in fp32 for stability.
            preds_fp32 = {k: v.float() for k, v in preds.items()}
            loss, diag = multiscale_loss(preds_fp32, D.float(),
                                         grad_w=args.grad_loss_w,
                                         hinge_w=args.hinge_loss_w)

        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(
            [p for p in model.parameters() if p.requires_grad], 1.0)
        scaler.step(opt)
        scaler.update()
        sched.step()

        running_loss.append(loss.item())
        running_l1_final.append(diag["d_final"][0])
        running_l1_cv.append(diag["d8_cv"][0])

        if step == 1 or step % args.log_every == 0 or step == args.steps:
            ms = (time.time() - t0) * 1000 / step
            cur_lr = opt.param_groups[0]["lr"]
            ml = float(np.mean(running_loss[-args.log_every:]))
            m_l1 = float(np.mean(running_l1_final[-args.log_every:]))
            m_cv = float(np.mean(running_l1_cv[-args.log_every:]))
            w_csv.writerow([step, ml, m_l1, m_cv, cur_lr, ms])
            fp_log.flush()
            print(f"  step {step:5d}  loss={ml:6.3f}  "
                  f"l1_final={m_l1:6.3f}  l1_cv={m_cv:6.3f}  "
                  f"lr={cur_lr:.2e}  {ms:6.1f} ms/step")

        if step % args.panel_every == 0 or step == args.steps:
            save_sample_panels(model, track_pairs, device, args.panel_dir,
                               step, args.inf_h, args.inf_w, history,
                               n_watch=args.n_watch_pairs)
            # Rewrite progress.csv after each dump
            with open(progress_path, "w", newline="") as fp:
                wp = csv.writer(fp)
                wp.writerow(["step", "mean_epe", "median_ms"] +
                            [f"pair_{j:02d}_epe" for j in range(len(track_pairs))])
                for h in history:
                    wp.writerow([h["step"], f"{h['mean_epe']:.4f}",
                                 f"{h.get('median_ms', 0.0):.1f}"] +
                                [f"{e:.4f}" for e in h["epes"]])

    fp_log.close()
    torch.save({"model": model.state_dict(),
                "trainable_M": trainable / 1e6,
                "args": vars(args)}, args.ckpt_out)
    print(f"saved -> {args.ckpt_out}")
    print(f"train log -> {log_path}")

    # Final quick val
    model.train()
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
            e = ((pred - D).abs() * valid).sum() / n
            b = (((pred - D).abs() > 1.0).float() * valid).sum() / n
            epes.append(float(e.item()))
            bad1s.append(float(b.item()))
    print(f"\nVAL ({len(epes)} pairs):  EPE={float(np.mean(epes)):.3f} px  "
          f"bad1={float(np.mean(bad1s))*100:.1f}%")


if __name__ == "__main__":
    main()
