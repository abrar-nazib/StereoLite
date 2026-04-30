"""Overfit StereoLite_yolo on 10 fixed Scene Flow Driving pairs.

Sanity test: can the YOLO26-truncated backbone, plugged into StereoLite,
drive training EPE down to near-zero on a tiny memorized set? If yes,
the architecture wires up cleanly. If it stalls, something's wrong with
how the encoder feeds into the rest of the model.

Two runs back-to-back:
    --backbone yolo26n
    --backbone yolo26s

Outputs land at:
    model/benchmarks/yolo_ablation_<TS>/<backbone>/{train.csv,curve.png,README.md}

Run:
    python3 model/scripts/overfit_yolo_ablation.py --backbone yolo26n
    python3 model/scripts/overfit_yolo_ablation.py --backbone yolo26s
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
from StereoLite_yolo.model import StereoLite, StereoLiteConfig  # noqa: E402

# SF data candidates (the SSD's mount point keeps changing). Pick whichever
# is currently accessible. If both fail, fall back to local cache.
SF_ROOT_CANDIDATES = [
    Path("/media/abrar/AbrarSSD/Datasets/sceneflow_driving"),
    Path("/mnt/abrarssd/Datasets/sceneflow_driving"),
]
BENCH_ROOT = ROOT / "model" / "benchmarks"
CACHE_PATH = ROOT / ".cache" / "sf_overfit_pairs_v1.pt"


def find_sf_root() -> Path | None:
    for p in SF_ROOT_CANDIDATES:
        try:
            if (p / "frames_finalpass").exists():
                return p
        except OSError:
            continue
    return None


def read_pfm(path: Path) -> np.ndarray:
    """Standard PFM reader. Returns float32 HxW."""
    with open(path, "rb") as f:
        header = f.readline().rstrip()
        if header == b"PF":
            color = True
        elif header == b"Pf":
            color = False
        else:
            raise ValueError(f"not a PFM: {path}")
        dim_line = f.readline().decode().strip()
        while dim_line.startswith("#"):
            dim_line = f.readline().decode().strip()
        w, h = map(int, dim_line.split())
        scale = float(f.readline().rstrip())
        endian = "<" if scale < 0 else ">"
        data = np.fromfile(f, endian + "f")
    shape = (h, w, 3) if color else (h, w)
    img = np.reshape(data, shape)
    img = np.flipud(img)
    return np.ascontiguousarray(img, dtype=np.float32)


def pick_pairs(sf_root: Path, n: int = 10):
    """Pick n stable stereo triplets (left PNG, right PNG, disparity PFM).

    All from `15mm_focallength/scene_forwards/slow/` for consistency.
    Returns list of (left_path, right_path, disp_path).
    """
    base = sf_root / "frames_finalpass" / "15mm_focallength" / "scene_forwards" / "slow"
    disp = sf_root / "disparity" / "15mm_focallength" / "scene_forwards" / "slow"
    lefts = sorted((base / "left").glob("*.png"))[:n]
    pairs = []
    for L in lefts:
        R = base / "right" / L.name
        D = disp / "left" / (L.stem + ".pfm")
        assert R.exists(), R
        assert D.exists(), D
        pairs.append((L, R, D))
    return pairs


def load_or_cache_pairs(n: int, size: tuple[int, int],
                        cache_path: Path = CACHE_PATH):
    """Return (Ls, Rs, Ds, valid, pair_paths). Reads from cache if present
    and matches; otherwise loads from SSD once and caches to NVMe."""
    H, W = size
    if cache_path.exists():
        try:
            blob = torch.load(cache_path, map_location="cpu", weights_only=False)
            if (blob["n"] == n and blob["size"] == (H, W)):
                print(f"[cache] loaded {n} pairs from {cache_path}")
                return (blob["L"], blob["R"], blob["D"], blob["valid"],
                        blob["paths"])
        except Exception as e:
            print(f"[cache] {cache_path} unreadable ({e}); rebuilding")
    sf_root = find_sf_root()
    if sf_root is None:
        raise FileNotFoundError(
            f"No accessible SF root in {[str(p) for p in SF_ROOT_CANDIDATES]} "
            f"and no cache at {cache_path}. Mount the SSD or rebuild cache.")
    print(f"[cache] miss; loading from {sf_root}")
    pairs = pick_pairs(sf_root, n)
    Ls, Rs, Ds = [], [], []
    for L, R, D in pairs:
        l, r, d = load_pair(L, R, D, (H, W))
        Ls.append(l); Rs.append(r); Ds.append(d)
    Ls = torch.stack(Ls); Rs = torch.stack(Rs); Ds = torch.stack(Ds)
    valid = ((Ds > 0) & (Ds < 192)).float()
    paths = [{"left": str(L), "right": str(R), "disp": str(D)}
             for (L, R, D) in pairs]
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"n": n, "size": (H, W), "L": Ls, "R": Rs, "D": Ds,
                "valid": valid, "paths": paths}, cache_path)
    print(f"[cache] saved {n} pairs to {cache_path}")
    return Ls, Rs, Ds, valid, paths


def load_pair(L: Path, R: Path, D: Path,
              size: tuple[int, int]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Load one triplet, center-crop to size (H, W). Returns (L, R, disp)
    as torch tensors on CPU.
    """
    H, W = size
    img_l = np.asarray(Image.open(L).convert("RGB"), dtype=np.float32) / 255.0
    img_r = np.asarray(Image.open(R).convert("RGB"), dtype=np.float32) / 255.0
    disp  = read_pfm(D)
    h, w = disp.shape
    # Center crop
    top = (h - H) // 2
    left = (w - W) // 2
    img_l = img_l[top:top + H, left:left + W]
    img_r = img_r[top:top + H, left:left + W]
    disp  = disp[top:top + H, left:left + W]
    img_l = torch.from_numpy(img_l).permute(2, 0, 1).contiguous()
    img_r = torch.from_numpy(img_r).permute(2, 0, 1).contiguous()
    disp  = torch.from_numpy(disp).unsqueeze(0).contiguous()
    return img_l, img_r, disp


def epe(pred: torch.Tensor, gt: torch.Tensor, valid: torch.Tensor) -> float:
    n = valid.sum().clamp(min=1)
    return float(((pred - gt).abs() * valid).sum() / n)


def stereo_metrics(pred: torch.Tensor, gt: torch.Tensor,
                   valid: torch.Tensor) -> dict:
    """Standard stereo benchmark metrics computed on valid pixels only.

    Returns a dict with:
        epe         mean L1 (= average end-point error in px)
        rmse        sqrt(mean squared error)
        median_ae   median absolute error (robust to outliers)
        bad_0.5     % pixels with abs error > 0.5 px
        bad_1.0     % pixels with abs error > 1.0 px (KITTI common)
        bad_2.0     % pixels with abs error > 2.0 px (Middlebury default)
        bad_3.0     % pixels with abs error > 3.0 px
        d1_all      % pixels with abs error > 3.0 px AND > 5% of GT
                    (KITTI 2015 D1-all definition)

    All values are floats (percentages for bad_* and d1_all).
    """
    valid_b = valid > 0.5
    n = valid_b.sum().clamp(min=1).item()
    err = (pred - gt).abs()
    err_v = err[valid_b]
    sq = (pred - gt) ** 2
    sq_v = sq[valid_b]

    out = {
        "epe": float(err_v.mean()),
        "rmse": float(sq_v.mean().sqrt()),
        "median_ae": float(err_v.median()),
    }
    for thr in (0.5, 1.0, 2.0, 3.0):
        out[f"bad_{thr}"] = float((err_v > thr).float().mean()) * 100.0
    # D1-all: KITTI 2015 official definition
    rel_err = err / gt.clamp(min=1e-6)
    d1 = ((err > 3.0) & (rel_err > 0.05) & valid_b).float().sum() / n * 100.0
    out["d1_all"] = float(d1)
    return out


def _colorize_disp(d: np.ndarray, vmax: float = 192.0) -> np.ndarray:
    """Disp HxW -> BGR uint8 with TURBO colormap."""
    d = np.nan_to_num(d, nan=0.0, posinf=vmax, neginf=0.0)
    d8 = (np.clip(d, 0, vmax) / vmax * 255).astype(np.uint8)
    return cv2.applyColorMap(d8, cv2.COLORMAP_TURBO)


def render_panel(left_chw: torch.Tensor, gt_hw: torch.Tensor,
                 pred_hw: torch.Tensor, stats: dict,
                 max_disp: float = 192.0) -> np.ndarray:
    """Build a 2×2 panel: left | GT disp | pred disp | stats text."""
    L = (left_chw.detach().cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
    L = L.transpose(1, 2, 0)                           # CHW->HWC, RGB
    L_bgr = cv2.cvtColor(L, cv2.COLOR_RGB2BGR)
    GT = _colorize_disp(gt_hw.detach().cpu().numpy(), max_disp)
    PR = _colorize_disp(pred_hw.detach().cpu().numpy(), max_disp)

    H, W = L_bgr.shape[:2]
    pad = np.zeros((H, W, 3), dtype=np.uint8)
    # First line starts well below the y=22 quadrant label so they don't
    # overlap; tighter line spacing fits more metric rows in one panel.
    y = 60
    line_h = 28
    for k, v in stats.items():
        cv2.putText(pad, f"{k}: {v}", (10, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2,
                    cv2.LINE_AA)
        y += line_h

    top = np.hstack([L_bgr, GT])
    bot = np.hstack([PR, pad])
    panel = np.vstack([top, bot])

    # Labels above each quadrant (the "stats" panel doesn't get one
    # since it would collide with the metric rows).
    labels = [(L_bgr, "left"), (GT, "GT disp"), (PR, "pred disp")]
    for i, (_, txt) in enumerate(labels):
        x = (i % 2) * W + 10
        yy = (i // 2) * H + 22
        cv2.putText(panel, txt, (x, yy), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (255, 255, 0), 2, cv2.LINE_AA)
    return panel


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backbone",
                    choices=["yolo26n", "yolo26s", "ghost", "mobilenet"],
                    required=True)
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--height", type=int, default=384)
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--n_pairs", type=int, default=20)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out_root", type=str, default="")
    ap.add_argument("--show", type=int, default=1,
                    help="1 = open OpenCV window with live panel; 0 = headless")
    ap.add_argument("--viz_interval_s", type=float, default=15.0,
                    help="seconds between viz panel updates")
    ap.add_argument("--viz_pair_idx", type=int, default=0,
                    help="index of fixed pair (0..n_pairs-1) to visualize")
    args = ap.parse_args()

    # Reproducibility patch: set every RNG and force deterministic CUDA ops.
    # Required for cuBLAS determinism on Ampere+ GPUs.
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # warn_only because a few ops (e.g. some interpolate modes) lack a
    # deterministic GPU implementation; we want a warning, not a crash.
    torch.use_deterministic_algorithms(True, warn_only=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device != "cuda":
        print("[WARN] no GPU detected; running on CPU will be very slow.")

    # Output dir
    if args.out_root:
        out_root = Path(args.out_root)
    else:
        ts = datetime.now().strftime("%Y%m%d-%H%M%S")
        out_root = BENCH_ROOT / f"yolo_ablation_{ts}"
    out = out_root / args.backbone
    out.mkdir(parents=True, exist_ok=True)

    # Load 10 pairs (cached on local NVMe to insulate against SSD blips)
    Ls, Rs, Ds, valid, pair_paths = load_or_cache_pairs(
        args.n_pairs, (args.height, args.width))
    Ls = Ls.to(device); Rs = Rs.to(device)
    Ds = Ds.to(device); valid = valid.to(device)
    print(f"  L shape={tuple(Ls.shape)}  D range=[{Ds[valid > 0].min():.1f}, {Ds[valid > 0].max():.1f}]")

    # Build model
    cfg = StereoLiteConfig(backbone=args.backbone)
    model = StereoLite(cfg).to(device)
    n_total = sum(p.numel() for p in model.parameters())
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"model: backbone={args.backbone}, total={n_total/1e6:.3f} M, train={n_train/1e6:.3f} M")

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)

    # Save metadata
    meta = {
        "backbone": args.backbone,
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
        "sf_root_candidates": [str(p) for p in SF_ROOT_CANDIDATES],
        "cache_path": str(CACHE_PATH),
    }
    (out / "meta.json").write_text(json.dumps(meta, indent=2))

    # Training loop
    csv_path = out / "train.csv"
    with open(csv_path, "w", newline="") as cf:
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
        win_name = f"overfit {args.backbone}"
        viz_pair = max(0, min(args.viz_pair_idx, N - 1))
        Lviz = Ls[viz_pair:viz_pair + 1]
        Rviz = Rs[viz_pair:viz_pair + 1]
        Dviz = Ds[viz_pair:viz_pair + 1]
        Vviz = valid[viz_pair:viz_pair + 1]
        show_ok = bool(args.show) and bool(os.environ.get("DISPLAY"))

        for step in range(1, args.steps + 1):
            idx = rng.choice(N, size=args.batch, replace=False)
            L = Ls[idx]; R = Rs[idx]; D = Ds[idx]; V = valid[idx]

            out_dict = model(L, R, aux=True)
            d_full = out_dict["d_final"]
            d_half = out_dict["d_half"]
            d4     = out_dict["d4"]
            d8     = out_dict["d8"]
            d16    = out_dict["d16"]

            # Multi-scale L1 across (full, /2, /4, /8, /16) outputs.
            def ms_l1(pred, gt, val, scale):
                if pred.shape[-2:] != gt.shape[-2:]:
                    pred = F.interpolate(pred, size=gt.shape[-2:],
                                          mode="bilinear", align_corners=False) * scale
                return ((pred - gt).abs() * val).sum() / val.sum().clamp(min=1)

            # Gradient (Sobel-like) consistency on full-res. Kills smearing:
            # a model that blurs fine detail has wrong disparity GRADIENTS
            # even if its mean error is OK. Inspired by HITNet & many
            # depth/optical-flow networks.
            def grad_consistency(pred, gt, val):
                gx_p = pred[..., :, 1:] - pred[..., :, :-1]
                gx_g =   gt[..., :, 1:] -   gt[..., :, :-1]
                gy_p = pred[..., 1:, :] - pred[..., :-1, :]
                gy_g =   gt[..., 1:, :] -   gt[..., :-1, :]
                vx = val[..., :, 1:] * val[..., :, :-1]
                vy = val[..., 1:, :] * val[..., :-1, :]
                lx = ((gx_p - gx_g).abs() * vx).sum() / vx.sum().clamp(min=1)
                ly = ((gy_p - gy_g).abs() * vy).sum() / vy.sum().clamp(min=1)
                return lx + ly

            # Bad-1 hinge: extra pressure on pixels with > 1 px error so the
            # model can't "average smearing" its way to good EPE while
            # leaving every pixel slightly wrong. Squared hinge is gentle
            # near 0 and grows fast as error increases.
            def bad1_hinge(pred, gt, val):
                err = (pred - gt).abs()
                hinge = (err - 1.0).clamp(min=0) ** 2
                return (hinge * val).sum() / val.sum().clamp(min=1)

            loss = (
                1.0 * ms_l1(d_full, D, V, 1.0)
                + 0.5 * ms_l1(d_half, D, V, 2.0)
                + 0.3 * ms_l1(d4,    D, V, 4.0)
                + 0.2 * ms_l1(d8,    D, V, 8.0)
                + 0.1 * ms_l1(d16,   D, V, 16.0)
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
                    "step":     f"{step}/{args.steps}",
                    "loss":     f"{loss.item():.4f}",
                    "EPE":      f"{mtr_v['epe']:.3f} px",
                    "RMSE":     f"{mtr_v['rmse']:.3f} px",
                    "median":   f"{mtr_v['median_ae']:.3f} px",
                    "bad-1.0":  f"{mtr_v['bad_1.0']:.1f}%",
                    "bad-2.0":  f"{mtr_v['bad_2.0']:.1f}%",
                    "D1-all":   f"{mtr_v['d1_all']:.1f}%",
                    "lr":       f"{opt.param_groups[0]['lr']:.4g}",
                    "elapsed":  f"{(now - t0):.0f} s",
                    "peak":     f"{peak_mem/1e9:.2f} GB",
                    "backbone": args.backbone,
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

    # Final eval: full metric set on all 10 pairs (eval mode)
    model.eval()
    with torch.no_grad():
        all_pred = model(Ls, Rs, aux=False)
        final_metrics = stereo_metrics(all_pred, Ds, valid)
        final_epe = final_metrics["epe"]

    # Inference latency benchmark: batch=1, 10 warmup + 100 timed.
    # Re-enable fast cuDNN (turn off deterministic mode) so the latency
    # reflects production-speed inference, not the slower deterministic
    # variants used during training. Determinism is irrelevant here
    # because we're not backpropagating.
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
          f"p95={inf_bench['ms_p95']} ms  "
          f"fps={inf_bench['fps_mean']}")

    meta["finished_at"] = datetime.now().isoformat(timespec="seconds")
    meta["elapsed_s"] = round(elapsed, 2)
    meta["peak_gpu_mem_gb"] = round(peak_mem / 1e9, 3) if peak_mem else None
    meta["final_epe_all10"] = round(final_epe, 4)
    meta["final_metrics_all10"] = {k: round(v, 4) for k, v in final_metrics.items()}
    meta["inference_bench"] = inf_bench
    (out / "meta.json").write_text(json.dumps(meta, indent=2))

    # Plot curve
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        steps, losses, epes, bad1s = [], [], [], []
        with open(csv_path) as cf:
            r = csv.DictReader(cf)
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
        fig.suptitle(f"{args.backbone}: overfit on 10 SF pairs "
                     f"(final EPE = {final_epe:.3f}, "
                     f"bad-1.0 = {final_metrics['bad_1.0']:.1f}%)")
        fig.tight_layout()
        fig.savefig(out / "curve.png", dpi=120)
        plt.close(fig)
    except Exception as e:
        print(f"plot failed: {e}")

    # README
    readme = f"""# Overfit ablation: {args.backbone}

**Run:** {meta['started_at']} → {meta['finished_at']}  ({meta['elapsed_s']:.1f} s)
**GPU:** {meta['gpu']}
**PyTorch:** {meta['pytorch']}

## Configuration

- backbone: `{args.backbone}`
- steps: {args.steps}, lr: {args.lr}, batch: {args.batch}
- input: {args.height}×{args.width}, {args.n_pairs} fixed pairs (seed {args.seed})
- model: {meta['params_total_M']} M total, {meta['params_train_M']} M trainable
- encoder out_channels (1/2, 1/4, 1/8, 1/16): {meta['encoder_out_channels']}
- peak GPU memory: {meta['peak_gpu_mem_gb']} GB

## Result (full-res, all 10 pairs, eval mode)

| Metric | Value |
|---|---|
| EPE (mean L1, px) | **{final_metrics['epe']:.4f}** |
| RMSE (px) | {final_metrics['rmse']:.4f} |
| Median AE (px) | {final_metrics['median_ae']:.4f} |
| bad-0.5 (% > 0.5 px error) | {final_metrics['bad_0.5']:.2f} |
| bad-1.0 (% > 1.0 px error) | **{final_metrics['bad_1.0']:.2f}** |
| bad-2.0 (% > 2.0 px error) | {final_metrics['bad_2.0']:.2f} |
| bad-3.0 (% > 3.0 px error) | {final_metrics['bad_3.0']:.2f} |
| D1-all (KITTI definition, %) | {final_metrics['d1_all']:.2f} |

## Inference latency (batch=1, {args.height}×{args.width}, eval mode)

- mean: **{inf_bench['ms_mean']} ms** ({inf_bench['fps_mean']} FPS)
- median: {inf_bench['ms_median']} ms
- p95: {inf_bench['ms_p95']} ms
- std: {inf_bench['ms_std']} ms
- 10 warmup + 100 timed runs, `torch.cuda.synchronize()` around each call

## Loss + EPE curve

![curve](curve.png)

## Pair list

{chr(10).join(f"- {p['left']}" for p in meta['pair_paths'])}

## Files

- `train.csv`: per-step loss + EPE log
- `meta.json`: full configuration + result
- `curve.png`: loss + EPE plot
- `README.md`: this file
"""
    (out / "README.md").write_text(readme)
    print(f"\n=== {args.backbone}: final EPE = {final_epe:.4f} ===")
    print(f"results saved to {out}")
    if show_ok:
        try:
            cv2.destroyWindow(win_name)
        except cv2.error:
            pass


if __name__ == "__main__":
    main()
