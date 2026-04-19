"""Ablation: does DAv2 actually contribute at inference time?

Loads the v5 Scene Flow checkpoint and runs inference on held-out val pairs
in three modes:
  (A) normal: DAv2 features flow through as trained
  (B) zero:   replace DAv2 outputs with zeros — tests if the projection
              layers' biases alone can substitute
  (C) noise:  replace DAv2 outputs with Gaussian noise of the same stats
              — tests if DAv2 carries content or just occupies capacity

If mode A >> B,C, DAv2 is pulling weight. If they're close, DAv2 is dead
weight we could drop and retrain lighter without it.
"""
from __future__ import annotations

import argparse
import csv
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


class DAv2Interceptor:
    """Monkey-patches DAv2SmallFrozen.forward to replace outputs."""

    def __init__(self, dav2_module, mode: str):
        self.dav2 = dav2_module
        self.mode = mode
        self._orig_forward = dav2_module.forward

        # Capture first-call stats for 'noise' mode calibration
        self._stats = None

    def install(self):
        mode = self.mode
        orig = self._orig_forward
        obj = self

        @torch.no_grad()
        def new_forward(x: torch.Tensor):
            real = orig(x)
            if mode == "normal":
                return real
            if mode == "zero":
                return {k: torch.zeros_like(v) for k, v in real.items()}
            if mode == "noise":
                # Match mean/std of real per tensor, but shuffle entirely
                out = {}
                for k, v in real.items():
                    m = v.mean()
                    s = v.std()
                    out[k] = torch.randn_like(v) * s + m
                return out
            raise ValueError(f"unknown mode {mode}")

        self.dav2.forward = new_forward

    def restore(self):
        self.dav2.forward = self._orig_forward


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default=os.path.join(
        PROJ, "model", "checkpoints", "stereolite_sf_v5.pth"))
    p.add_argument("--data_root", default=os.path.join(
        PROJ, "data", "sceneflow_driving"))
    p.add_argument("--n_val", type=int, default=200)
    p.add_argument("--n_pairs", type=int, default=8,
                   help="number of val pairs to compare per mode")
    p.add_argument("--n_eval", type=int, default=50,
                   help="number of val pairs to compute mean EPE over")
    p.add_argument("--inf_h", type=int, default=512)
    p.add_argument("--inf_w", type=int, default=832)
    p.add_argument("--out_dir", default=os.path.join(
        PROJ, "model", "benchmarks", "ablation_dav2"))
    args = p.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    from sceneflow_loader import enumerate_pairs, train_val_split, read_pfm
    items = enumerate_pairs(args.data_root)
    _, val = train_val_split(items, args.n_val)

    # Evenly-spaced subset for visualisation
    step = max(1, len(val) // args.n_pairs)
    viz_pairs = [val[i * step] for i in range(args.n_pairs)]
    eval_pairs = val[: args.n_eval]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    from d1_tile import StereoLite, StereoLiteConfig
    model = StereoLite(StereoLiteConfig(cv_max_disp=48)).to(device)
    ck = torch.load(args.ckpt, map_location=device, weights_only=False)
    sd = ck["model"] if "model" in ck else ck
    missing, unexpected = model.load_state_dict(sd, strict=True)
    print(f"loaded {args.ckpt}  (missing={len(missing)} unexpected={len(unexpected)})")
    model.train()   # GroupNorm is batch-independent; train mode is fine at eval

    interceptor = DAv2Interceptor(model.dav2, "normal")

    def run_mode(mode: str, pairs, compute_viz: bool):
        interceptor.mode = mode
        interceptor.install()
        try:
            epes, ms_list = [], []
            tiles_per_pair = {} if compute_viz else None
            with torch.no_grad():
                for i, (lp, rp, pp) in enumerate(pairs):
                    L = cv2.imread(lp); R = cv2.imread(rp)
                    D_n = read_pfm(pp)
                    H_n, W_n = D_n.shape
                    L_in = cv2.resize(L, (args.inf_w, args.inf_h),
                                       interpolation=cv2.INTER_AREA)
                    R_in = cv2.resize(R, (args.inf_w, args.inf_h),
                                       interpolation=cv2.INTER_AREA)
                    sx = args.inf_w / W_n
                    D = cv2.resize(D_n, (args.inf_w, args.inf_h)) * sx
                    D[~np.isfinite(D) | (D < 0)] = 0

                    Lt = torch.from_numpy(cv2.cvtColor(L_in, cv2.COLOR_BGR2RGB)).float().permute(2, 0, 1).unsqueeze(0).to(device)
                    Rt = torch.from_numpy(cv2.cvtColor(R_in, cv2.COLOR_BGR2RGB)).float().permute(2, 0, 1).unsqueeze(0).to(device)
                    if device.type == "cuda":
                        torch.cuda.synchronize()
                    t0 = time.time()
                    P = model(Lt, Rt).squeeze().cpu().numpy()
                    if device.type == "cuda":
                        torch.cuda.synchronize()
                    ms = (time.time() - t0) * 1000
                    ms_list.append(ms)

                    valid = (D > 0) & (D < 192) & np.isfinite(D)
                    n = max(valid.sum(), 1)
                    epe = float((np.abs(P - D) * valid).sum() / n)
                    epes.append(epe)

                    if compute_viz:
                        lo = float(np.percentile(D[valid], 2)) if valid.sum() > 16 else 0.0
                        hi = float(np.percentile(D[valid], 98)) if valid.sum() > 16 else 60.0
                        pred_img = annotate(colourise(P, lo, hi),
                                            f"{mode}  EPE={epe:.2f}  {ms:.0f}ms")
                        tiles_per_pair[i] = pred_img
            return epes, ms_list, tiles_per_pair
        finally:
            interceptor.restore()

    # Eval on n_eval pairs for mean EPE
    print(f"\nevaluating {args.n_eval} pairs per mode ...")
    results = {}
    for mode in ("normal", "zero", "noise"):
        epes, ms_list, _ = run_mode(mode, eval_pairs, compute_viz=False)
        results[mode] = {
            "mean_epe": float(np.mean(epes)),
            "std_epe": float(np.std(epes)),
            "median_ms": float(np.median(ms_list)),
            "min_epe": float(np.min(epes)),
            "max_epe": float(np.max(epes)),
        }
        print(f"  {mode:7s}  EPE={results[mode]['mean_epe']:.3f} ± "
              f"{results[mode]['std_epe']:.3f}  min={results[mode]['min_epe']:.2f}  "
              f"max={results[mode]['max_epe']:.2f}  med ms={results[mode]['median_ms']:.0f}")

    # Build side-by-side panels for n_pairs visualisation
    print(f"\ngenerating {args.n_pairs} side-by-side comparison panels...")
    normal_tiles = run_mode("normal", viz_pairs, compute_viz=True)[2]
    zero_tiles = run_mode("zero", viz_pairs, compute_viz=True)[2]
    noise_tiles = run_mode("noise", viz_pairs, compute_viz=True)[2]

    for i, (lp, rp, pp) in enumerate(viz_pairs):
        L = cv2.imread(lp)
        D_n = read_pfm(pp)
        H_n, W_n = D_n.shape
        L_in = cv2.resize(L, (args.inf_w, args.inf_h))
        sx = args.inf_w / W_n
        D = cv2.resize(D_n, (args.inf_w, args.inf_h)) * sx
        D[~np.isfinite(D) | (D < 0)] = 0
        valid = (D > 0) & (D < 192) & np.isfinite(D)
        lo = float(np.percentile(D[valid], 2)) if valid.sum() > 16 else 0.0
        hi = float(np.percentile(D[valid], 98)) if valid.sum() > 16 else 60.0

        row = np.hstack([
            annotate(L_in, f"left pair{i:02d}"),
            annotate(colourise(D, lo, hi), f"GT"),
            normal_tiles[i],
            zero_tiles[i],
            noise_tiles[i],
        ])
        cv2.imwrite(os.path.join(args.out_dir, f"ablate_{i:02d}.png"), row)

    # Summary CSV
    with open(os.path.join(args.out_dir, "summary.csv"), "w", newline="") as fp:
        w = csv.writer(fp)
        w.writerow(["mode", "mean_epe", "std_epe", "min_epe", "max_epe", "median_ms"])
        for mode, r in results.items():
            w.writerow([mode, f"{r['mean_epe']:.4f}", f"{r['std_epe']:.4f}",
                        f"{r['min_epe']:.4f}", f"{r['max_epe']:.4f}",
                        f"{r['median_ms']:.1f}"])

    print(f"\npanels and summary at {args.out_dir}")
    print("\n--- verdict ---")
    dn = results["normal"]["mean_epe"]
    dz = results["zero"]["mean_epe"]
    dns = results["noise"]["mean_epe"]
    delta_zero = dz - dn
    delta_noise = dns - dn
    if delta_zero < 0.1 and delta_noise < 0.1:
        print(f"DAv2 contributes LITTLE: zero +{delta_zero:.2f}px, noise +{delta_noise:.2f}px")
    elif delta_zero > 1.0 or delta_noise > 1.0:
        print(f"DAv2 is LOAD-BEARING: zero +{delta_zero:.2f}px, noise +{delta_noise:.2f}px")
    else:
        print(f"DAv2 helps modestly: zero +{delta_zero:.2f}px, noise +{delta_noise:.2f}px")


if __name__ == "__main__":
    main()
