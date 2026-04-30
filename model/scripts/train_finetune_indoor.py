"""Fine-tune StereoLite on FoundationStereo pseudo-GT indoor pairs.

Resumes from `model/checkpoints/stereolite_kaggle_baseline.pth` (the
30-epoch Scene Flow Driving checkpoint), then trains on the cleaned
indoor pseudo-GT set with a low peak LR so we don't blow away the
existing weights.

Identical loss / panel-saving / watcher-friendly behaviour as
`train_sharpness.py`, but uses `pseudo_pairs_loader.PseudoPairs`
and skips augmentation that would hurt at this scale.
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

from pseudo_pairs_loader import PseudoPairs, list_pairs, split_pairs

from train_sharpness import (
    multiscale_loss, random_erase_right, save_sample_panels)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--pairs_dir", default="/mnt/abrarssd/Datasets/"
                   "stereo_samples_20260425_104147")
    p.add_argument("--ckpt_in", default=os.path.join(
        PROJ, "model", "checkpoints", "stereolite_kaggle_baseline.pth"))
    p.add_argument("--ckpt_out", default=os.path.join(
        PROJ, "model", "checkpoints", "stereolite_finetune_indoor.pth"))
    p.add_argument("--steps", type=int, default=9000)
    p.add_argument("--batch", type=int, default=3)
    p.add_argument("--num_workers", type=int, default=2)
    p.add_argument("--lr", type=float, default=1e-4,
                   help="OneCycle peak LR (low for finetune)")
    p.add_argument("--inf_h", type=int, default=512)
    p.add_argument("--inf_w", type=int, default=832)
    p.add_argument("--n_val", type=int, default=50)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--random_erase_p", type=float, default=0.2)
    p.add_argument("--grad_loss_w", type=float, default=0.5)
    p.add_argument("--hinge_loss_w", type=float, default=0.3)
    p.add_argument("--smooth_loss_w", type=float, default=0.02)
    p.add_argument("--panel_every", type=int, default=200)
    p.add_argument("--n_track_pairs", type=int, default=8)
    p.add_argument("--n_watch_pairs", type=int, default=3)
    p.add_argument("--log_every", type=int, default=50)
    p.add_argument("--panel_dir", default=None)
    args = p.parse_args()

    if args.panel_dir is None:
        ts = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
        args.panel_dir = os.path.join(
            PROJ, "model", "benchmarks", f"stereolite_finetune_indoor_{ts}")
    os.makedirs(args.panel_dir, exist_ok=True)
    os.makedirs(os.path.dirname(args.ckpt_out), exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device.type}  panel_dir={args.panel_dir}")

    # ---- data ----
    all_pairs = list_pairs(args.pairs_dir)
    train, val = split_pairs(all_pairs, n_val=args.n_val, seed=args.seed)
    print(f"clean pairs: {len(all_pairs)}  -> train={len(train)} val={len(val)}")
    train_ds = PseudoPairs(train, h=args.inf_h, w=args.inf_w)
    loader = torch.utils.data.DataLoader(
        train_ds, batch_size=args.batch, shuffle=True,
        num_workers=args.num_workers, pin_memory=True,
        persistent_workers=args.num_workers > 0, drop_last=True)
    val_ds = PseudoPairs(val, h=args.inf_h, w=args.inf_w)
    val_loader = torch.utils.data.DataLoader(
        val_ds, batch_size=1, shuffle=False, num_workers=1, pin_memory=True)

    # ---- panel-tracking pairs (subset of val so we see real scenes) ----
    track_items = val[: args.n_track_pairs]
    # save_sample_panels expects (lp, rp, pp_or_npy) — convert .npy entries
    # into a wrapper format expected by training_panel rendering.
    # The existing save_sample_panels reads PFM via read_pfm; we need to
    # bypass that and feed disparity from .npy. So we do panel rendering
    # locally for indoor finetune.
    history: list[dict] = []

    # ---- model ----
    from StereoLite import StereoLite, StereoLiteConfig
    model = StereoLite(StereoLiteConfig(backbone="mobilenet")).to(device)
    if os.path.exists(args.ckpt_in):
        ck = torch.load(args.ckpt_in, map_location=device, weights_only=False)
        sd = ck["model"] if "model" in ck else ck
        missing, unexpected = model.load_state_dict(sd, strict=True)
        print(f"resumed from {args.ckpt_in}  "
              f"(missing={len(missing)} unexpected={len(unexpected)})")
        if "epe" in ck:
            print(f"  baseline (Kaggle) val EPE: {ck['epe']:.3f} px")
    else:
        print(f"!! ckpt_in not found, training from scratch: {args.ckpt_in}")
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"trainable={trainable/1e6:.3f} M")

    # ---- optim ----
    opt = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=args.lr, total_steps=args.steps,
        pct_start=0.05, anneal_strategy="cos",
        div_factor=25.0, final_div_factor=1e4)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    print(f"OneCycle LR  peak={args.lr:.1e}  steps={args.steps}")

    # ---- log ----
    log_path = args.ckpt_out.replace(".pth", "_train.csv")
    fp_log = open(log_path, "w", newline="")
    w_csv = csv.writer(fp_log)
    w_csv.writerow(["step", "loss", "l1_final", "l1_cv", "lr",
                     "ms_per_step"])

    progress_path = os.path.join(args.panel_dir, "progress.csv")

    # ---- training loop ----
    model.train()
    t0 = time.time()
    running_loss, running_l1f, running_l1cv = [], [], []
    it = iter(loader)
    best_epe = float("inf")
    for step in range(1, args.steps + 1):
        try:
            L, R, D = next(it)
        except StopIteration:
            it = iter(loader)
            L, R, D = next(it)
        L = L.to(device, non_blocking=True)
        R = R.to(device, non_blocking=True)
        D = D.to(device, non_blocking=True)
        if args.random_erase_p > 0:
            R = random_erase_right(R, prob=args.random_erase_p)

        with torch.amp.autocast("cuda", enabled=device.type == "cuda",
                                  dtype=torch.float16):
            preds = model(L, R, aux=True)
            if preds["d_final"].shape[-2:] != D.shape[-2:]:
                preds["d_final"] = torch.nn.functional.interpolate(
                    preds["d_final"], size=D.shape[-2:],
                    mode="bilinear", align_corners=True)
            preds_fp32 = {k: v.float() for k, v in preds.items()}
            loss, diag = multiscale_loss(
                preds_fp32, D.float(),
                grad_w=args.grad_loss_w, hinge_w=args.hinge_loss_w,
                smooth_w=args.smooth_loss_w, left_img=L.float())

        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(
            [p for p in model.parameters() if p.requires_grad], 1.0)
        scaler.step(opt)
        scaler.update()
        sched.step()

        running_loss.append(loss.item())
        running_l1f.append(diag["d_final"][0])
        running_l1cv.append(diag["d8_cv"][0])

        if step == 1 or step % args.log_every == 0 or step == args.steps:
            ms = (time.time() - t0) * 1000 / step
            cur_lr = opt.param_groups[0]["lr"]
            ml = float(np.mean(running_loss[-args.log_every:]))
            m_l1 = float(np.mean(running_l1f[-args.log_every:]))
            m_cv = float(np.mean(running_l1cv[-args.log_every:]))
            w_csv.writerow([step, ml, m_l1, m_cv, cur_lr, ms])
            fp_log.flush()
            print(f"  step {step:5d}  loss={ml:6.3f}  "
                  f"l1_final={m_l1:6.3f}  l1_cv={m_cv:6.3f}  "
                  f"lr={cur_lr:.2e}  {ms:6.1f} ms/step", flush=True)

        if step % args.panel_every == 0 or step == args.steps:
            _save_indoor_panels(model, track_items, device, args.panel_dir,
                                step, args.inf_h, args.inf_w, history,
                                n_watch=args.n_watch_pairs)
            with open(progress_path, "w", newline="") as fp:
                wp = csv.writer(fp)
                wp.writerow(["step", "mean_epe", "median_ms"] +
                            [f"pair_{j:02d}_epe"
                             for j in range(len(track_items))])
                for h in history:
                    wp.writerow([h["step"], f"{h['mean_epe']:.4f}",
                                  f"{h.get('median_ms', 0.0):.1f}"] +
                                [f"{e:.4f}" for e in h["epes"]])
            # Quick val pass for best-tracking
            if step % (args.panel_every * 5) == 0 or step == args.steps:
                epe, bad1 = _quick_val(model, val_loader, device)
                print(f"  ==> val EPE={epe:.3f}  bad1={bad1*100:.1f}%",
                      flush=True)
                if epe < best_epe:
                    best_epe = epe
                    _save_ckpt(model, args.ckpt_out.replace(".pth",
                                                             "_best.pth"),
                               step, epe, bad1, args)
                    print(f"  ==> saved new best (EPE {epe:.3f})", flush=True)

    fp_log.close()
    final_epe, final_bad1 = _quick_val(model, val_loader, device)
    _save_ckpt(model, args.ckpt_out, args.steps, final_epe, final_bad1, args)
    print(f"\nDone.  final val EPE={final_epe:.3f} px  bad1={final_bad1*100:.1f}%")
    print(f"checkpoints:")
    print(f"  best : {args.ckpt_out.replace('.pth', '_best.pth')}  "
          f"(EPE {best_epe:.3f})")
    print(f"  last : {args.ckpt_out}  (EPE {final_epe:.3f})")
    print(f"train log -> {log_path}")
    print(f"panels    -> {args.panel_dir}")


def _save_ckpt(model, path, step, epe, bad1, args):
    torch.save({"model": model.state_dict(), "step": step,
                 "epe": epe, "bad1": bad1, "args": vars(args)}, path)


@torch.no_grad()
def _quick_val(model, val_loader, device, max_disp=192.0):
    model.eval()
    epes, bad1s = [], []
    for L, R, D in val_loader:
        L, R, D = L.to(device), R.to(device), D.to(device)
        pred = model(L, R)
        if pred.shape != D.shape:
            pred = torch.nn.functional.interpolate(
                pred, size=D.shape[-2:], mode="bilinear", align_corners=True)
        valid = (D > 0.5) & (D < max_disp) & torch.isfinite(D)
        n = valid.sum().clamp(min=1.0)
        epes.append(float(((pred - D).abs() * valid).sum() / n))
        bad1s.append(float((((pred - D).abs() > 1.0).float() * valid).sum() / n))
    model.train()
    return float(np.mean(epes)), float(np.mean(bad1s))


def _save_indoor_panels(model, track_items, device, out_dir, step, inf_h,
                         inf_w, history, n_watch=3):
    """Render L | GT (pseudo) | pred for each tracked indoor pair."""
    pairs_dir = os.path.join(out_dir, "pairs")
    montage_dir = os.path.join(out_dir, "montage")
    montage_full_dir = os.path.join(out_dir, "montage_full")
    os.makedirs(pairs_dir, exist_ok=True)
    os.makedirs(montage_dir, exist_ok=True)
    os.makedirs(montage_full_dir, exist_ok=True)
    was_training = model.training
    model.eval()

    rows: list[np.ndarray] = []
    epes: list[float] = []
    ms_list: list[float] = []
    cuda_sync = device.type == "cuda"
    # Warmup once
    if track_items:
        lp, rp, dp = track_items[0]
        Lw = cv2.imread(lp); Rw = cv2.imread(rp)
        Lw = cv2.resize(Lw, (inf_w, inf_h))
        Rw = cv2.resize(Rw, (inf_w, inf_h))
        Lt = torch.from_numpy(cv2.cvtColor(Lw, cv2.COLOR_BGR2RGB)) \
                  .float().permute(2, 0, 1).unsqueeze(0).to(device)
        Rt = torch.from_numpy(cv2.cvtColor(Rw, cv2.COLOR_BGR2RGB)) \
                  .float().permute(2, 0, 1).unsqueeze(0).to(device)
        with torch.no_grad():
            _ = model(Lt, Rt)
        if cuda_sync:
            torch.cuda.synchronize()

    with torch.no_grad():
        for i, (lp, rp, dp) in enumerate(track_items):
            L = cv2.imread(lp); R = cv2.imread(rp)
            D_full = np.load(dp).astype(np.float32)
            H_n, W_n = D_full.shape
            L_in = cv2.resize(L, (inf_w, inf_h))
            R_in = cv2.resize(R, (inf_w, inf_h))
            sx = inf_w / W_n
            D = cv2.resize(D_full, (inf_w, inf_h),
                            interpolation=cv2.INTER_LINEAR) * sx
            D[~np.isfinite(D) | (D < 0)] = 0.0

            Lt = torch.from_numpy(cv2.cvtColor(L_in, cv2.COLOR_BGR2RGB)) \
                      .float().permute(2, 0, 1).unsqueeze(0).to(device)
            Rt = torch.from_numpy(cv2.cvtColor(R_in, cv2.COLOR_BGR2RGB)) \
                      .float().permute(2, 0, 1).unsqueeze(0).to(device)
            if cuda_sync:
                torch.cuda.synchronize()
            t0 = time.time()
            P = model(Lt, Rt).squeeze().float().cpu().numpy()
            if cuda_sync:
                torch.cuda.synchronize()
            ms_list.append((time.time() - t0) * 1000)

            valid = (D > 0.5) & (D < 192) & np.isfinite(D)
            if valid.sum() > 16:
                lo = float(np.percentile(D[valid], 2))
                hi = float(np.percentile(D[valid], 98))
            else:
                lo, hi = 0.0, 60.0
            epe = float((np.abs(P - D) * valid).sum()
                         / max(valid.sum(), 1))
            epes.append(epe)

            tile_l = _annot(L_in, f"L pair{i:02d}  {inf_w}x{inf_h}")
            tile_g = _annot(_colourise(D, lo, hi),
                            f"GT (pseudo)  {D[valid].min():.1f}-{D[valid].max():.1f}")
            tile_p = _annot(_colourise(P, lo, hi),
                            f"pred  EPE={epe:.2f}  {ms_list[-1]:.0f}ms  step {step}")
            row = np.hstack([tile_l, tile_g, tile_p])
            pdir = os.path.join(pairs_dir, f"pair_{i:02d}")
            os.makedirs(pdir, exist_ok=True)
            cv2.imwrite(os.path.join(pdir, f"step_{step:05d}.png"), row)
            rows.append(row)

    watch = rows[:n_watch]
    cv2.imwrite(os.path.join(montage_dir, f"step_{step:05d}.png"),
                np.vstack(watch))
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
        cv2.imwrite(os.path.join(montage_full_dir, f"step_{step:05d}.png"),
                    np.vstack(full_rows))

    mean_epe = float(np.mean(epes))
    med_ms = float(np.median(ms_list)) if ms_list else 0.0
    history.append({"step": step, "mean_epe": mean_epe, "epes": epes,
                     "median_ms": med_ms})
    print(f"  -> step {step}: {len(epes)}-pair mean EPE = {mean_epe:.3f} px  "
          f"(range {min(epes):.2f} .. {max(epes):.2f})  "
          f"median inference {med_ms:.1f} ms", flush=True)
    if was_training:
        model.train()


def _colourise(d, lo, hi):
    d = d.astype(np.float32)
    v = np.clip((d - lo) / max(hi - lo, 1e-6), 0, 1) * 255
    return cv2.applyColorMap(v.astype(np.uint8), cv2.COLORMAP_TURBO)


def _annot(img, text, h=22):
    out = img.copy()
    cv2.rectangle(out, (0, 0), (out.shape[1], h), (0, 0, 0), -1)
    cv2.putText(out, text, (6, h - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.42,
                (255, 255, 255), 1, cv2.LINE_AA)
    return out


if __name__ == "__main__":
    main()
