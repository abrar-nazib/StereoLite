"""Loss-formulation A/B harness for the StereoLite chassis.

Architecture is **fixed** to the current chassis (StereoLite_yolo with
backbone=ghost). The variable being studied is the **training loss** —
which loss formulation drives the lowest EPE / bad-T / D1-all on the
overfit harness?

Loss variants:
    L1            : multi-scale L1 only (RAFT-style baseline, γ=1)
    L1_seq        : multi-scale L1 with γ-weighted decay (γ=0.7)
                    matching RAFT's sequence-loss spirit at scale level
    L1_grad       : multi-scale L1 + Sobel gradient consistency
    L1_bad1       : multi-scale L1 + bad-1 squared hinge
    cocktail      : L1 + grad + bad1 (CURRENT default; the baseline)
    cocktail_b05  : cocktail + bad-0.5 hinge (target sub-pixel quality)
    stack         : L1 + grad + threshold-stack hinge (0.5, 1, 2, 3)
    stack_d1      : L1 + grad + threshold-stack + D1-relative hinge
    charbonnier   : Charbonnier (sqrt(err²+ε²)) + grad + bad1 hinge

Each variant trains 3000 steps on 20 fixed Scene Flow pairs, batch 4,
seed 42, lr 2e-4. Determinism patch is on. Inference latency is the
same across variants (architecture unchanged) so we don't bother
re-measuring.

Run one variant:
    python3 model/scripts/overfit_loss_ablation.py --loss cocktail

Run all back-to-back (use the orchestrator script for the full sweep).

Outputs land at:
    model/benchmarks/loss_ablation_<TS>/<loss>/{train.csv,curve.png,...}
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
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "model" / "designs"))
sys.path.insert(0, str(ROOT / "model" / "scripts"))

from StereoLite_yolo.model import StereoLite, StereoLiteConfig  # noqa: E402
from overfit_yolo_ablation import (  # noqa: E402
    BENCH_ROOT, load_or_cache_pairs, stereo_metrics, render_panel,
)


LOSS_CHOICES = [
    "L1", "L1_seq", "L1_grad", "L1_bad1",
    "cocktail", "cocktail_b05",
    "stack", "stack_d1",
    "charbonnier",
]


def ms_l1(pred, gt, val, scale):
    if pred.shape[-2:] != gt.shape[-2:]:
        pred = F.interpolate(pred, size=gt.shape[-2:],
                              mode="bilinear", align_corners=False) * scale
    return ((pred - gt).abs() * val).sum() / val.sum().clamp(min=1)


def ms_charbonnier(pred, gt, val, scale, eps=1e-3):
    if pred.shape[-2:] != gt.shape[-2:]:
        pred = F.interpolate(pred, size=gt.shape[-2:],
                              mode="bilinear", align_corners=False) * scale
    diff = pred - gt
    err = (diff * diff + eps * eps).sqrt()
    return (err * val).sum() / val.sum().clamp(min=1)


def grad_consistency(pred, gt, val):
    gx_p = pred[..., :, 1:] - pred[..., :, :-1]
    gx_g = gt[..., :, 1:] - gt[..., :, :-1]
    gy_p = pred[..., 1:, :] - pred[..., :-1, :]
    gy_g = gt[..., 1:, :] - gt[..., :-1, :]
    vx = val[..., :, 1:] * val[..., :, :-1]
    vy = val[..., 1:, :] * val[..., :-1, :]
    lx = ((gx_p - gx_g).abs() * vx).sum() / vx.sum().clamp(min=1)
    ly = ((gy_p - gy_g).abs() * vy).sum() / vy.sum().clamp(min=1)
    return lx + ly


def threshold_hinge(pred, gt, val, threshold):
    err = (pred - gt).abs()
    h = (err - threshold).clamp(min=0) ** 2
    return (h * val).sum() / val.sum().clamp(min=1)


def threshold_hinge_stack(pred, gt, val, thresholds=(0.5, 1.0, 2.0, 3.0)):
    err = (pred - gt).abs()
    total = sum((err - t).clamp(min=0) ** 2 for t in thresholds)
    return (total * val).sum() / val.sum().clamp(min=1)


def d1_relative_hinge(pred, gt, val):
    """KITTI D1-all: penalise pixels with abs err > 3 px AND rel err > 5%."""
    err = (pred - gt).abs()
    rel = err / gt.clamp(min=1e-6)
    is_d1 = ((err > 3.0) & (rel > 0.05)).float()
    h = (err - 3.0).clamp(min=0) ** 2 * is_d1
    return (h * val).sum() / val.sum().clamp(min=1)


def compute_loss(name: str, preds: dict, D: torch.Tensor, V: torch.Tensor):
    """Return scalar loss for the chosen variant."""
    d_full = preds["d_final"]
    d_half = preds["d_half"]
    d4 = preds["d4"]
    d8 = preds["d8"]
    d16 = preds["d16"]

    # Plain multi-scale L1 (uniform weights).
    if name == "L1":
        return (
            1.0 * ms_l1(d_full, D, V, 1.0)
            + 0.5 * ms_l1(d_half, D, V, 2.0)
            + 0.3 * ms_l1(d4,    D, V, 4.0)
            + 0.2 * ms_l1(d8,    D, V, 8.0)
            + 0.1 * ms_l1(d16,   D, V, 16.0)
        )

    # γ-decayed multi-scale L1 (RAFT-style; finer scales weighted more).
    # Five scales total. γ=0.7 → weights for (full, /2, /4, /8, /16) =
    # (1.0, 0.7, 0.49, 0.343, 0.2401).
    if name == "L1_seq":
        gamma = 0.7
        return (
            (gamma ** 0) * ms_l1(d_full, D, V, 1.0)
            + (gamma ** 1) * ms_l1(d_half, D, V, 2.0)
            + (gamma ** 2) * ms_l1(d4,    D, V, 4.0)
            + (gamma ** 3) * ms_l1(d8,    D, V, 8.0)
            + (gamma ** 4) * ms_l1(d16,   D, V, 16.0)
        )

    # L1 + gradient consistency (no hinge).
    if name == "L1_grad":
        base = (
            1.0 * ms_l1(d_full, D, V, 1.0)
            + 0.5 * ms_l1(d_half, D, V, 2.0)
            + 0.3 * ms_l1(d4,    D, V, 4.0)
            + 0.2 * ms_l1(d8,    D, V, 8.0)
            + 0.1 * ms_l1(d16,   D, V, 16.0)
        )
        return base + 0.5 * grad_consistency(d_full, D, V)

    # L1 + bad-1 hinge (no gradient).
    if name == "L1_bad1":
        base = (
            1.0 * ms_l1(d_full, D, V, 1.0)
            + 0.5 * ms_l1(d_half, D, V, 2.0)
            + 0.3 * ms_l1(d4,    D, V, 4.0)
            + 0.2 * ms_l1(d8,    D, V, 8.0)
            + 0.1 * ms_l1(d16,   D, V, 16.0)
        )
        return base + 0.2 * threshold_hinge(d_full, D, V, 1.0)

    # Current default: L1 + grad + bad-1.
    if name == "cocktail":
        base = (
            1.0 * ms_l1(d_full, D, V, 1.0)
            + 0.5 * ms_l1(d_half, D, V, 2.0)
            + 0.3 * ms_l1(d4,    D, V, 4.0)
            + 0.2 * ms_l1(d8,    D, V, 8.0)
            + 0.1 * ms_l1(d16,   D, V, 16.0)
        )
        return (base + 0.5 * grad_consistency(d_full, D, V)
                + 0.2 * threshold_hinge(d_full, D, V, 1.0))

    # Cocktail + bad-0.5 hinge (sub-pixel quality push).
    if name == "cocktail_b05":
        base = (
            1.0 * ms_l1(d_full, D, V, 1.0)
            + 0.5 * ms_l1(d_half, D, V, 2.0)
            + 0.3 * ms_l1(d4,    D, V, 4.0)
            + 0.2 * ms_l1(d8,    D, V, 8.0)
            + 0.1 * ms_l1(d16,   D, V, 16.0)
        )
        return (base + 0.5 * grad_consistency(d_full, D, V)
                + 0.2 * threshold_hinge(d_full, D, V, 1.0)
                + 0.3 * threshold_hinge(d_full, D, V, 0.5))

    # Threshold stack: hinges at 0.5, 1, 2, 3 simultaneously.
    if name == "stack":
        base = (
            1.0 * ms_l1(d_full, D, V, 1.0)
            + 0.5 * ms_l1(d_half, D, V, 2.0)
            + 0.3 * ms_l1(d4,    D, V, 4.0)
            + 0.2 * ms_l1(d8,    D, V, 8.0)
            + 0.1 * ms_l1(d16,   D, V, 16.0)
        )
        return (base + 0.5 * grad_consistency(d_full, D, V)
                + 0.3 * threshold_hinge_stack(d_full, D, V))

    # Threshold stack + D1-relative hinge.
    if name == "stack_d1":
        base = (
            1.0 * ms_l1(d_full, D, V, 1.0)
            + 0.5 * ms_l1(d_half, D, V, 2.0)
            + 0.3 * ms_l1(d4,    D, V, 4.0)
            + 0.2 * ms_l1(d8,    D, V, 8.0)
            + 0.1 * ms_l1(d16,   D, V, 16.0)
        )
        return (base + 0.5 * grad_consistency(d_full, D, V)
                + 0.3 * threshold_hinge_stack(d_full, D, V)
                + 0.2 * d1_relative_hinge(d_full, D, V))

    # Charbonnier base + grad + bad-1 hinge.
    if name == "charbonnier":
        base = (
            1.0 * ms_charbonnier(d_full, D, V, 1.0)
            + 0.5 * ms_charbonnier(d_half, D, V, 2.0)
            + 0.3 * ms_charbonnier(d4,    D, V, 4.0)
            + 0.2 * ms_charbonnier(d8,    D, V, 8.0)
            + 0.1 * ms_charbonnier(d16,   D, V, 16.0)
        )
        return (base + 0.5 * grad_consistency(d_full, D, V)
                + 0.2 * threshold_hinge(d_full, D, V, 1.0))

    raise ValueError(f"unknown loss name: {name}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--loss", choices=LOSS_CHOICES, required=True)
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--height", type=int, default=384)
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--n_pairs", type=int, default=20)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out_root", type=str, default="")
    ap.add_argument("--show", type=int, default=1,
                    help="1 = open OpenCV window with live panel; 0 = headless")
    ap.add_argument("--viz_interval_s", type=float, default=15.0)
    ap.add_argument("--viz_pair_idx", type=int, default=0)
    args = ap.parse_args()

    # Reproducibility patch.
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if args.out_root:
        out_root = Path(args.out_root)
    else:
        ts = datetime.now().strftime("%Y%m%d-%H%M%S")
        out_root = BENCH_ROOT / f"loss_ablation_{ts}"
    out = out_root / args.loss
    out.mkdir(parents=True, exist_ok=True)

    Ls, Rs, Ds, valid, pair_paths = load_or_cache_pairs(
        args.n_pairs, (args.height, args.width))
    Ls = Ls.to(device); Rs = Rs.to(device)
    Ds = Ds.to(device); valid = valid.to(device)

    cfg = StereoLiteConfig(backbone="ghost")
    model = StereoLite(cfg).to(device)
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"loss={args.loss}  arch=ghost  trainable={n_train/1e6:.3f} M")

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)

    meta = {
        "loss": args.loss,
        "arch": "current (StereoLite_yolo, backbone=ghost)",
        "steps": args.steps, "lr": args.lr, "batch": args.batch,
        "height": args.height, "width": args.width,
        "n_pairs": args.n_pairs, "seed": args.seed,
        "device": device,
        "gpu": torch.cuda.get_device_name(0) if device == "cuda" else None,
        "started_at": datetime.now().isoformat(timespec="seconds"),
        "params_train_M": round(n_train / 1e6, 4),
        "pair_paths": pair_paths,
    }
    (out / "meta.json").write_text(json.dumps(meta, indent=2))

    csv_path = out / "train.csv"
    cf = open(csv_path, "w", newline="")
    w = csv.writer(cf)
    w.writerow(["step", "loss", "epe", "rmse", "bad_0.5", "bad_1.0",
                "bad_2.0", "bad_3.0", "d1_all", "lr", "elapsed_s"])

    t0 = time.time()
    N = Ls.shape[0]
    rng = np.random.default_rng(args.seed)
    peak_mem = 0
    model.train()
    viz_dir = out / "viz"
    viz_dir.mkdir(exist_ok=True)
    last_viz_t = 0.0
    win_name = f"loss {args.loss}"
    viz_pair = max(0, min(args.viz_pair_idx, N - 1))
    Lviz = Ls[viz_pair:viz_pair + 1]
    Rviz = Rs[viz_pair:viz_pair + 1]
    Dviz = Ds[viz_pair:viz_pair + 1]
    Vviz = valid[viz_pair:viz_pair + 1]
    show_ok = bool(args.show) and bool(os.environ.get("DISPLAY"))

    for step in range(1, args.steps + 1):
        idx = rng.choice(N, size=args.batch, replace=False)
        L = Ls[idx]; R = Rs[idx]; D = Ds[idx]; V = valid[idx]

        preds = model(L, R, aux=True)
        loss = compute_loss(args.loss, preds, D, V)

        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        if device == "cuda":
            peak_mem = max(peak_mem, torch.cuda.max_memory_allocated())

        if step == 1 or step % 50 == 0 or step == args.steps:
            with torch.no_grad():
                mtr = stereo_metrics(preds["d_final"], D, V)
            w.writerow([
                step, f"{loss.item():.5f}",
                f"{mtr['epe']:.5f}", f"{mtr['rmse']:.5f}",
                f"{mtr['bad_0.5']:.3f}", f"{mtr['bad_1.0']:.3f}",
                f"{mtr['bad_2.0']:.3f}", f"{mtr['bad_3.0']:.3f}",
                f"{mtr['d1_all']:.3f}",
                f"{opt.param_groups[0]['lr']:.6f}",
                f"{time.time() - t0:.2f}",
            ])
            cf.flush()
            print(f"  step {step:>4d}  loss={loss.item():.4f}  "
                  f"EPE={mtr['epe']:.3f}  bad05={mtr['bad_0.5']:.1f}%  "
                  f"bad1={mtr['bad_1.0']:.1f}%  bad2={mtr['bad_2.0']:.1f}%  "
                  f"({time.time()-t0:.1f}s)", flush=True)

        now = time.time()
        if (now - last_viz_t) >= args.viz_interval_s or step == args.steps:
            last_viz_t = now
            with torch.no_grad():
                pred_v = model(Lviz, Rviz, aux=False)
                mtr_v = stereo_metrics(pred_v, Dviz, Vviz)
            stats = {
                "step": f"{step}/{args.steps}",
                "loss": f"{loss.item():.4f}",
                "EPE": f"{mtr_v['epe']:.3f} px",
                "RMSE": f"{mtr_v['rmse']:.3f} px",
                "median": f"{mtr_v['median_ae']:.3f} px",
                "bad-0.5": f"{mtr_v['bad_0.5']:.1f}%",
                "bad-1.0": f"{mtr_v['bad_1.0']:.1f}%",
                "bad-2.0": f"{mtr_v['bad_2.0']:.1f}%",
                "bad-3.0": f"{mtr_v['bad_3.0']:.1f}%",
                "D1-all": f"{mtr_v['d1_all']:.1f}%",
                "lr": f"{opt.param_groups[0]['lr']:.4g}",
                "elapsed": f"{(now - t0):.0f} s",
                "peak": f"{peak_mem/1e9:.2f} GB",
                "loss_fn": args.loss,
            }
            panel = render_panel(Lviz[0], Dviz[0, 0], pred_v[0, 0], stats)
            cv2.imwrite(str(viz_dir / f"step_{step:05d}.png"), panel)
            if show_ok:
                cv2.imshow(win_name, panel)
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    print("[viz] q pressed: aborting early.")
                    break

    elapsed = time.time() - t0
    cf.close()

    # Final eval on all pairs.
    model.eval()
    with torch.no_grad():
        all_pred = model(Ls, Rs, aux=False)
        final_metrics = stereo_metrics(all_pred, Ds, valid)
    print(f"\n=== {args.loss}: final EPE={final_metrics['epe']:.4f}  "
          f"bad-0.5={final_metrics['bad_0.5']:.2f}%  "
          f"bad-1.0={final_metrics['bad_1.0']:.2f}%  "
          f"D1-all={final_metrics['d1_all']:.2f}% ===")

    meta["finished_at"] = datetime.now().isoformat(timespec="seconds")
    meta["elapsed_s"] = round(elapsed, 2)
    meta["peak_gpu_mem_gb"] = round(peak_mem / 1e9, 3) if peak_mem else None
    meta["final_metrics_all"] = {k: round(v, 4) for k, v in final_metrics.items()}
    (out / "meta.json").write_text(json.dumps(meta, indent=2))

    # Curve.
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        steps, losses, epes, b05, b1, b2 = [], [], [], [], [], []
        with open(csv_path) as cfr:
            r = csv.DictReader(cfr)
            for row in r:
                steps.append(int(row["step"]))
                losses.append(float(row["loss"]))
                epes.append(float(row["epe"]))
                b05.append(float(row["bad_0.5"]))
                b1.append(float(row["bad_1.0"]))
                b2.append(float(row["bad_2.0"]))
        fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
        axes[0].plot(steps, losses, "b-", label="loss", alpha=0.7)
        axes[0].set_xlabel("step"); axes[0].set_ylabel("loss", color="b")
        axes[0].set_yscale("log"); axes[0].grid(alpha=0.3)
        ax1b = axes[0].twinx()
        ax1b.plot(steps, epes, "r-", label="EPE", alpha=0.7)
        ax1b.set_ylabel("EPE (px)", color="r")
        axes[1].plot(steps, b05, label="bad-0.5", alpha=0.7)
        axes[1].plot(steps, b1, label="bad-1.0", alpha=0.7)
        axes[1].plot(steps, b2, label="bad-2.0", alpha=0.7)
        axes[1].set_xlabel("step"); axes[1].set_ylabel("bad-T (%)")
        axes[1].set_yscale("log"); axes[1].grid(alpha=0.3); axes[1].legend()
        fig.suptitle(f"loss={args.loss}  EPE={final_metrics['epe']:.3f}  "
                     f"bad-0.5={final_metrics['bad_0.5']:.1f}%  "
                     f"bad-1.0={final_metrics['bad_1.0']:.1f}%")
        fig.tight_layout()
        fig.savefig(out / "curve.png", dpi=120)
        plt.close(fig)
    except Exception as e:
        print(f"plot failed: {e}")

    readme = f"""# Loss ablation: {args.loss}

**Run:** {meta['started_at']} → {meta['finished_at']}  ({meta['elapsed_s']:.1f} s)
**Arch:** current chassis (StereoLite_yolo, backbone=ghost), {meta['params_train_M']} M trainable

## Result (all {args.n_pairs} pairs, eval mode)

| Metric | Value |
|---|---|
| EPE (px) | **{final_metrics['epe']:.4f}** |
| RMSE (px) | {final_metrics['rmse']:.4f} |
| Median AE (px) | {final_metrics['median_ae']:.4f} |
| bad-0.5 (%) | **{final_metrics['bad_0.5']:.2f}** |
| bad-1.0 (%) | **{final_metrics['bad_1.0']:.2f}** |
| bad-2.0 (%) | {final_metrics['bad_2.0']:.2f} |
| bad-3.0 (%) | {final_metrics['bad_3.0']:.2f} |
| D1-all (%) | **{final_metrics['d1_all']:.2f}** |
"""
    (out / "README.md").write_text(readme)
    if show_ok:
        try:
            cv2.destroyWindow(win_name)
        except cv2.error:
            pass

    # Auto-update the master experiments summary so it stays current.
    try:
        import subprocess
        subprocess.run(
            [sys.executable, str(ROOT / "model" / "scripts"
                                  / "build_experiments_summary.py")],
            check=False, capture_output=True)
    except Exception as e:
        print(f"experiments summary update failed: {e}")


if __name__ == "__main__":
    main()
