"""3-way architecture A/B/C overfit comparison.

Architectures:
    current  : StereoLite_yolo with backbone=ghost.
               TileRefine iterated at 1/16 (×2), 1/8 (×3), 1/4 (×3),
               then ConvexUpsample 1/4 → 1/2 → full.

    v1_iter  : StereoLite_v1_iter.
               Same TileRefine iterations as current, PLUS additional
               iterations at 1/2 (×2). No ConvexUpsample. Plane equation
               upsample 1/2 → full.

    v2_hitnet: StereoLite_v2_hitnet.
               Replaces all TileRefine with HITNet's exact propagation
               block (residual blocks + dilated convs + local cost-volume
               augmentation, no BN, leaky ReLU). Single pass per scale at
               1/16, 1/8, 1/4, 1/2. Plane equation upsample throughout.
               No ConvexUpsample.

All three use the GhostConv encoder so the only delta is the
refinement+upsample design.

Run:
    python3 model/scripts/overfit_arch_ablation.py --arch current
    python3 model/scripts/overfit_arch_ablation.py --arch v1_iter
    python3 model/scripts/overfit_arch_ablation.py --arch v2_hitnet

Or run all three back-to-back into one timestamped directory:
    TS=$(date +%Y%m%d-%H%M%S)
    OUT=model/benchmarks/arch_ablation_$TS
    for a in current v1_iter v2_hitnet; do
        python3 model/scripts/overfit_arch_ablation.py --arch $a --out_root $OUT
    done

Outputs:
    model/benchmarks/arch_ablation_<TS>/<arch>/{train.csv,curve.png,README.md,viz/}
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
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "model" / "designs"))

# Reuse helpers from overfit_yolo_ablation.py
sys.path.insert(0, str(ROOT / "model" / "scripts"))
from overfit_yolo_ablation import (  # noqa: E402
    SF_ROOT_CANDIDATES, BENCH_ROOT, CACHE_PATH,
    load_or_cache_pairs, stereo_metrics, render_panel,
)


def build_model(arch: str):
    """Return (StereoLite_class, StereoLiteConfig_class) for the chosen arch.

    All three accept backbone="ghost" as the only valid backbone.
    """
    if arch == "current":
        from StereoLite_yolo.model import StereoLite, StereoLiteConfig
        cfg = StereoLiteConfig(backbone="ghost")
    elif arch == "v1_iter":
        from StereoLite_v1_iter.model import StereoLite, StereoLiteConfig
        cfg = StereoLiteConfig()
    elif arch == "v2_hitnet":
        from StereoLite_v2_hitnet.model import StereoLite, StereoLiteConfig
        cfg = StereoLiteConfig()
    else:
        raise ValueError(f"unknown arch: {arch}")
    return StereoLite(cfg), cfg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arch",
                    choices=["current", "v1_iter", "v2_hitnet"],
                    required=True)
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
    if device != "cuda":
        print("[WARN] no GPU detected; running on CPU will be very slow.")

    # Output dir.
    if args.out_root:
        out_root = Path(args.out_root)
    else:
        ts = datetime.now().strftime("%Y%m%d-%H%M%S")
        out_root = BENCH_ROOT / f"arch_ablation_{ts}"
    out = out_root / args.arch
    out.mkdir(parents=True, exist_ok=True)

    # Load pairs.
    Ls, Rs, Ds, valid, pair_paths = load_or_cache_pairs(
        args.n_pairs, (args.height, args.width))
    Ls = Ls.to(device); Rs = Rs.to(device)
    Ds = Ds.to(device); valid = valid.to(device)
    print(f"  L shape={tuple(Ls.shape)}  D range=[{Ds[valid > 0].min():.1f}, "
          f"{Ds[valid > 0].max():.1f}]")

    # Build model.
    model, cfg = build_model(args.arch)
    model = model.to(device)
    n_total = sum(p.numel() for p in model.parameters())
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"model: arch={args.arch}, total={n_total/1e6:.3f} M, "
          f"train={n_train/1e6:.3f} M")

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)

    meta = {
        "arch": args.arch,
        "steps": args.steps,
        "lr": args.lr,
        "batch": args.batch,
        "height": args.height,
        "width": args.width,
        "n_pairs": args.n_pairs,
        "seed": args.seed,
        "device": device,
        "gpu": torch.cuda.get_device_name(0) if device == "cuda" else None,
        "pytorch": torch.__version__,
        "platform": platform.platform(),
        "started_at": datetime.now().isoformat(timespec="seconds"),
        "params_total_M": round(n_total / 1e6, 4),
        "params_train_M": round(n_train / 1e6, 4),
        "encoder_out_channels": list(model.fnet.out_channels),
        "pair_paths": pair_paths,
    }
    (out / "meta.json").write_text(json.dumps(meta, indent=2))

    # Training loop.
    csv_path = out / "train.csv"
    cf = open(csv_path, "w", newline="")
    w = csv.writer(cf)
    w.writerow(["step", "loss", "epe", "rmse", "bad_1.0", "bad_2.0",
                "d1_all", "lr", "elapsed_s"])

    t0 = time.time()
    N = Ls.shape[0]
    rng = np.random.default_rng(args.seed)
    peak_mem = 0
    model.train()
    viz_dir = out / "viz"
    viz_dir.mkdir(exist_ok=True)
    last_viz_t = 0.0
    win_name = f"overfit {args.arch}"
    viz_pair = max(0, min(args.viz_pair_idx, N - 1))
    Lviz = Ls[viz_pair:viz_pair + 1]
    Rviz = Rs[viz_pair:viz_pair + 1]
    Dviz = Ds[viz_pair:viz_pair + 1]
    Vviz = valid[viz_pair:viz_pair + 1]
    show_ok = bool(args.show) and bool(os.environ.get("DISPLAY"))

    def ms_l1(pred, gt, val, scale):
        if pred.shape[-2:] != gt.shape[-2:]:
            pred = F.interpolate(pred, size=gt.shape[-2:],
                                  mode="bilinear", align_corners=False) * scale
        return ((pred - gt).abs() * val).sum() / val.sum().clamp(min=1)

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

    def bad1_hinge(pred, gt, val):
        err = (pred - gt).abs()
        hinge = (err - 1.0).clamp(min=0) ** 2
        return (hinge * val).sum() / val.sum().clamp(min=1)

    for step in range(1, args.steps + 1):
        idx = rng.choice(N, size=args.batch, replace=False)
        L = Ls[idx]; R = Rs[idx]; D = Ds[idx]; V = valid[idx]

        out_dict = model(L, R, aux=True)
        d_full = out_dict["d_final"]
        d_half = out_dict["d_half"]
        d4 = out_dict["d4"]
        d8 = out_dict["d8"]
        d16 = out_dict["d16"]

        loss = (
            1.0 * ms_l1(d_full, D, V, 1.0)
            + 0.5 * ms_l1(d_half, D, V, 2.0)
            + 0.3 * ms_l1(d4, D, V, 4.0)
            + 0.2 * ms_l1(d8, D, V, 8.0)
            + 0.1 * ms_l1(d16, D, V, 16.0)
            + 0.5 * grad_consistency(d_full, D, V)
            + 0.2 * bad1_hinge(d_full, D, V)
        )

        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        if device == "cuda":
            peak_mem = max(peak_mem, torch.cuda.max_memory_allocated())

        if step == 1 or step % 50 == 0 or step == args.steps:
            with torch.no_grad():
                mtr = stereo_metrics(d_full, D, V)
            w.writerow([
                step, f"{loss.item():.5f}",
                f"{mtr['epe']:.5f}", f"{mtr['rmse']:.5f}",
                f"{mtr['bad_1.0']:.3f}", f"{mtr['bad_2.0']:.3f}",
                f"{mtr['d1_all']:.3f}",
                f"{opt.param_groups[0]['lr']:.6f}",
                f"{time.time() - t0:.2f}",
            ])
            cf.flush()
            print(f"  step {step:>4d}  loss={loss.item():.4f}  "
                  f"EPE={mtr['epe']:.3f}  bad1={mtr['bad_1.0']:.1f}%  "
                  f"bad2={mtr['bad_2.0']:.1f}%  "
                  f"({time.time()-t0:.1f}s, peak={peak_mem/1e9:.2f} GB)",
                  flush=True)

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
                "arch": args.arch,
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

    # Final eval.
    model.eval()
    with torch.no_grad():
        all_pred = model(Ls, Rs, aux=False)
        final_metrics = stereo_metrics(all_pred, Ds, valid)
        final_epe = final_metrics["epe"]

    # Inference latency benchmark.
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True
    torch.use_deterministic_algorithms(False)
    inf_bench = {}
    if device == "cuda":
        torch.cuda.synchronize()
    with torch.no_grad():
        L1 = Ls[:1]; R1 = Rs[:1]
        for _ in range(10):
            _ = model(L1, R1, aux=False)
        if device == "cuda":
            torch.cuda.synchronize()
        per_call = []
        for _ in range(100):
            t = time.time()
            _ = model(L1, R1, aux=False)
            if device == "cuda":
                torch.cuda.synchronize()
            per_call.append((time.time() - t) * 1000.0)
    arr = np.asarray(per_call)
    inf_bench = {
        "n_warmup": 10, "n_timed": 100,
        "input_size_HW": [args.height, args.width],
        "batch_size": 1,
        "ms_mean": round(float(arr.mean()), 3),
        "ms_std": round(float(arr.std()), 3),
        "ms_median": round(float(np.median(arr)), 3),
        "ms_p95": round(float(np.percentile(arr, 95)), 3),
        "fps_mean": round(1000.0 / float(arr.mean()), 2),
    }
    print(f"[inference] mean={inf_bench['ms_mean']} ms  "
          f"median={inf_bench['ms_median']} ms  "
          f"p95={inf_bench['ms_p95']} ms  fps={inf_bench['fps_mean']}")

    meta["finished_at"] = datetime.now().isoformat(timespec="seconds")
    meta["elapsed_s"] = round(elapsed, 2)
    meta["peak_gpu_mem_gb"] = round(peak_mem / 1e9, 3) if peak_mem else None
    meta["final_epe_all"] = round(final_epe, 4)
    meta["final_metrics_all"] = {k: round(v, 4) for k, v in final_metrics.items()}
    meta["inference_bench"] = inf_bench
    (out / "meta.json").write_text(json.dumps(meta, indent=2))

    # Curves plot.
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        steps, losses, epes, bad1s = [], [], [], []
        with open(csv_path) as cfr:
            r = csv.DictReader(cfr)
            for row in r:
                steps.append(int(row["step"]))
                losses.append(float(row["loss"]))
                epes.append(float(row["epe"]))
                bad1s.append(float(row["bad_1.0"]))
        fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
        axes[0].plot(steps, losses, "b-", label="loss", alpha=0.7)
        axes[0].set_xlabel("step"); axes[0].set_ylabel("loss", color="b")
        axes[0].set_yscale("log"); axes[0].grid(alpha=0.3)
        ax1b = axes[0].twinx()
        ax1b.plot(steps, epes, "r-", label="EPE (full)", alpha=0.7)
        ax1b.set_ylabel("EPE (px)", color="r")
        axes[1].plot(steps, bad1s, "g-", label="bad-1.0 %", alpha=0.7)
        axes[1].set_xlabel("step"); axes[1].set_ylabel("bad-1.0 (%)")
        axes[1].set_yscale("log"); axes[1].grid(alpha=0.3); axes[1].legend()
        fig.suptitle(f"{args.arch}: overfit on {args.n_pairs} SF pairs "
                     f"(final EPE = {final_epe:.3f}, "
                     f"bad-1.0 = {final_metrics['bad_1.0']:.1f}%)")
        fig.tight_layout()
        fig.savefig(out / "curve.png", dpi=120)
        plt.close(fig)
    except Exception as e:
        print(f"plot failed: {e}")

    # Per-variant README.
    readme = f"""# Overfit ablation: {args.arch}

**Run:** {meta['started_at']} → {meta['finished_at']}  ({meta['elapsed_s']:.1f} s)
**GPU:** {meta['gpu']}

## Configuration

- arch: `{args.arch}`
- steps: {args.steps}, lr: {args.lr}, batch: {args.batch}
- input: {args.height}×{args.width}, {args.n_pairs} fixed pairs (seed {args.seed})
- model: {meta['params_total_M']} M total, {meta['params_train_M']} M trainable
- encoder out_channels (1/2, 1/4, 1/8, 1/16): {meta['encoder_out_channels']}
- peak GPU memory: {meta['peak_gpu_mem_gb']} GB

## Result (full-res, all {args.n_pairs} pairs, eval mode)

| Metric | Value |
|---|---|
| EPE (px) | **{final_metrics['epe']:.4f}** |
| RMSE (px) | {final_metrics['rmse']:.4f} |
| Median AE (px) | {final_metrics['median_ae']:.4f} |
| bad-0.5 (%) | {final_metrics['bad_0.5']:.2f} |
| bad-1.0 (%) | **{final_metrics['bad_1.0']:.2f}** |
| bad-2.0 (%) | {final_metrics['bad_2.0']:.2f} |
| bad-3.0 (%) | {final_metrics['bad_3.0']:.2f} |
| D1-all (%) | {final_metrics['d1_all']:.2f} |

## Inference latency

- mean: **{inf_bench['ms_mean']} ms** ({inf_bench['fps_mean']} FPS)
- median: {inf_bench['ms_median']} ms
- p95: {inf_bench['ms_p95']} ms
"""
    (out / "README.md").write_text(readme)
    print(f"\n=== {args.arch}: final EPE = {final_epe:.4f} ===")
    print(f"results saved to {out}")
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
