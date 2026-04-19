"""Apples-to-apples evaluation on Scene Flow Driving held-out:
StereoLite vs pretrained HITNet (Scene Flow finalpass).

Reports per-model EPE, bad-1, bad-3, latency. Saves comparison panels
[left | GT | HITNet | StereoLite] to model/benchmarks/sf_eval_<TS>/.
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
from hitnet_baseline import HitnetBaseline


def colourise(d, lo, hi):
    d = d.astype(np.float32)
    v = np.clip((d - lo) / max(hi - lo, 1e-6), 0, 1) * 255
    return cv2.applyColorMap(v.astype(np.uint8), cv2.COLORMAP_TURBO)


def annotate(img, text):
    out = img.copy()
    cv2.rectangle(out, (0, 0), (out.shape[1], 22), (0, 0, 0), -1)
    cv2.putText(out, text, (6, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.42,
                (255, 255, 255), 1, cv2.LINE_AA)
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data_root", default=os.path.join(PROJ, "data", "sceneflow_driving"))
    p.add_argument("--stereolite_ckpt", default=os.path.join(PROJ, "model", "checkpoints", "stereolite_sf.pth"))
    p.add_argument("--n_val", type=int, default=200)
    p.add_argument("--n_visualise", type=int, default=10)
    p.add_argument("--inf_h", type=int, default=384)
    p.add_argument("--inf_w", type=int, default=640)
    p.add_argument("--out_dir", default=None)
    args = p.parse_args()

    if args.out_dir is None:
        ts = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
        args.out_dir = os.path.join(PROJ, "model", "benchmarks", f"sf_eval_{ts}")
    os.makedirs(args.out_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    items = enumerate_pairs(args.data_root)
    _, val = train_val_split(items, args.n_val)
    print(f"evaluating on {len(val)} held-out pairs")

    # Load StereoLite
    from d1_tile import StereoLite
    stereolite = StereoLite().to(device)
    if os.path.exists(args.stereolite_ckpt):
        sd = torch.load(args.stereolite_ckpt, map_location=device, weights_only=False)
        if "model" in sd:
            sd = sd["model"]
        stereolite.load_state_dict(sd, strict=True)
        print(f"loaded StereoLite from {args.stereolite_ckpt}")
    else:
        print(f"WARNING: no StereoLite checkpoint at {args.stereolite_ckpt} — using random init")
    stereolite.train()        # GroupNorm; train mode is safe at any batch

    # Load HITNet
    print("loading HITNet pretrained...")
    hitnet = HitnetBaseline(device=device)

    epes_t, bad1_t, bad3_t, ms_t = [], [], [], []
    epes_h, bad1_h, bad3_h, ms_h = [], [], [], []

    visualised = 0
    for vi, (lp, rp, pp) in enumerate(val):
        from sceneflow_loader import read_pfm
        L = cv2.imread(lp); R = cv2.imread(rp)
        D_native = read_pfm(pp)
        H_n, W_n = D_native.shape

        L_in = cv2.resize(L, (args.inf_w, args.inf_h))
        R_in = cv2.resize(R, (args.inf_w, args.inf_h))
        sx = args.inf_w / W_n
        D = cv2.resize(D_native, (args.inf_w, args.inf_h)) * sx
        D[~np.isfinite(D) | (D < 0)] = 0
        Dt = torch.from_numpy(D.astype(np.float32)).unsqueeze(0).unsqueeze(0).to(device)

        Lt = torch.from_numpy(cv2.cvtColor(L_in, cv2.COLOR_BGR2RGB)).float().permute(2, 0, 1).unsqueeze(0).to(device)
        Rt = torch.from_numpy(cv2.cvtColor(R_in, cv2.COLOR_BGR2RGB)).float().permute(2, 0, 1).unsqueeze(0).to(device)

        # StereoLite
        t0 = time.time()
        with torch.no_grad():
            pt = stereolite(Lt, Rt)
        if device.type == "cuda":
            torch.cuda.synchronize()
        ms_tf = (time.time() - t0) * 1000

        # HITNet
        t0 = time.time()
        ph = hitnet(Lt, Rt)
        if device.type == "cuda":
            torch.cuda.synchronize()
        ms_hi = (time.time() - t0) * 1000

        valid = (Dt > 0) & (Dt < 192) & torch.isfinite(Dt)
        n_v = valid.sum().clamp(min=1.0)
        e_t = ((pt - Dt).abs() * valid).sum() / n_v
        e_h = ((ph - Dt).abs() * valid).sum() / n_v
        b1_t = (((pt - Dt).abs() > 1.0).float() * valid).sum() / n_v
        b1_h = (((ph - Dt).abs() > 1.0).float() * valid).sum() / n_v
        b3_t = (((pt - Dt).abs() > 3.0).float() * valid).sum() / n_v
        b3_h = (((ph - Dt).abs() > 3.0).float() * valid).sum() / n_v

        epes_t.append(float(e_t.item())); bad1_t.append(float(b1_t.item())); bad3_t.append(float(b3_t.item())); ms_t.append(ms_tf)
        epes_h.append(float(e_h.item())); bad1_h.append(float(b1_h.item())); bad3_h.append(float(b3_h.item())); ms_h.append(ms_hi)

        if visualised < args.n_visualise:
            valid_np = (D > 0) & (D < 192) & np.isfinite(D)
            if valid_np.sum() > 16:
                lo = float(np.percentile(D[valid_np], 2))
                hi = float(np.percentile(D[valid_np], 98))
            else:
                lo, hi = 0.0, 60.0
            tiles = [
                annotate(L_in, f"left  {os.path.basename(lp)}"),
                annotate(colourise(D, lo, hi), f"GT  {D[valid_np].min():.1f}..{D[valid_np].max():.1f}px"),
                annotate(colourise(ph.squeeze().cpu().numpy(), lo, hi),
                         f"HITNet  EPE={float(e_h.item()):.2f}  {ms_hi:.0f}ms"),
                annotate(colourise(pt.squeeze().cpu().numpy(), lo, hi),
                         f"TileFM  EPE={float(e_t.item()):.2f}  {ms_tf:.0f}ms"),
            ]
            row = np.hstack(tiles)
            cv2.imwrite(os.path.join(args.out_dir, f"sf_{vi:04d}.png"), row)
            visualised += 1

    print("\nSCENE FLOW DRIVING — apples-to-apples")
    print(f"  {'model':<14} {'params':>8} {'EPE px':>8} {'bad-1%':>8} {'bad-3%':>8} {'med ms':>8}")
    print(f"  {'StereoLite':<14} {sum(p.numel() for p in stereolite.parameters())/1e6:>7.2f}M "
          f"{float(np.mean(epes_t)):>8.3f} {float(np.mean(bad1_t))*100:>7.1f} "
          f"{float(np.mean(bad3_t))*100:>7.1f} {float(np.median(ms_t)):>7.1f}")
    print(f"  {'HITNet (SF)':<14} {hitnet.n_params/1e6:>7.2f}M "
          f"{float(np.mean(epes_h)):>8.3f} {float(np.mean(bad1_h))*100:>7.1f} "
          f"{float(np.mean(bad3_h))*100:>7.1f} {float(np.median(ms_h)):>7.1f}")

    with open(os.path.join(args.out_dir, "summary.csv"), "w", newline="") as fp:
        w = csv.writer(fp)
        w.writerow(["model", "params_M", "epe_px", "bad1_pct", "bad3_pct", "median_ms"])
        w.writerow(["StereoLite",
                    round(sum(p.numel() for p in stereolite.parameters())/1e6, 3),
                    round(float(np.mean(epes_t)), 3),
                    round(float(np.mean(bad1_t))*100, 2),
                    round(float(np.mean(bad3_t))*100, 2),
                    round(float(np.median(ms_t)), 1)])
        w.writerow(["HITNet_SF",
                    round(hitnet.n_params/1e6, 3),
                    round(float(np.mean(epes_h)), 3),
                    round(float(np.mean(bad1_h))*100, 2),
                    round(float(np.mean(bad3_h))*100, 2),
                    round(float(np.median(ms_h)), 1)])
    print(f"\nresults at {args.out_dir}")


if __name__ == "__main__":
    main()
