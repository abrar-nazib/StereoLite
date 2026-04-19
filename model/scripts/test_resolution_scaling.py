"""Resolution-scaling sanity test.

Takes a trained StereoLite checkpoint (trained at 384x640) and runs it
at multiple input resolutions against the same Scene Flow Driving val
pairs. Saves side-by-side panels so we can judge whether sharpness is
input-resolution-limited or architecture-limited.

Resolutions tested (all divisible by 32):
    384 x  640   (training resolution)
    512 x  832   (medium uplift)
    544 x  960   (~native Scene Flow, height rounded 540->544)
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


RESOLUTIONS = [
    (384, 640, "train_384x640"),
    (512, 832, "mid_512x832"),
    (544, 960, "native_544x960"),
]


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


def run_at_resolution(model, Lb, Rb, H, W, device):
    L_in = cv2.resize(Lb, (W, H), interpolation=cv2.INTER_AREA)
    R_in = cv2.resize(Rb, (W, H), interpolation=cv2.INTER_AREA)
    Lt = torch.from_numpy(cv2.cvtColor(L_in, cv2.COLOR_BGR2RGB)).float().permute(2, 0, 1).unsqueeze(0).to(device)
    Rt = torch.from_numpy(cv2.cvtColor(R_in, cv2.COLOR_BGR2RGB)).float().permute(2, 0, 1).unsqueeze(0).to(device)
    t0 = time.time()
    with torch.no_grad():
        d = model(Lt, Rt)
    if device.type == "cuda":
        torch.cuda.synchronize()
    ms = (time.time() - t0) * 1000
    return d.squeeze().cpu().numpy(), ms, L_in


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default=os.path.join(PROJ, "model", "checkpoints", "stereolite_sharp.pth"))
    p.add_argument("--data_root", default=os.path.join(PROJ, "data", "sceneflow_driving"))
    p.add_argument("--out_dir", default=os.path.join(PROJ, "model", "benchmarks", "res_scaling"))
    p.add_argument("--n_pairs", type=int, default=10)
    p.add_argument("--n_val", type=int, default=50)
    args = p.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    from sceneflow_loader import enumerate_pairs, train_val_split, read_pfm
    items = enumerate_pairs(args.data_root)
    _, val = train_val_split(items, args.n_val)
    # Pick the same pairs as the 20-pair tracker for consistency
    step = max(1, len(val) // 20)
    track = [val[i * step] for i in range(20)][: args.n_pairs]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    from d1_tile import StereoLite, StereoLiteConfig
    model = StereoLite(StereoLiteConfig(use_dav2=True)).to(device)
    ck = torch.load(args.ckpt, map_location=device, weights_only=False)
    sd = ck["model"] if "model" in ck else ck
    model.load_state_dict(sd, strict=True)
    model.train()  # GroupNorm batch-independent
    print(f"loaded {args.ckpt}")

    csv_path = os.path.join(args.out_dir, "summary.csv")
    fp = open(csv_path, "w", newline="")
    w = csv.writer(fp)
    w.writerow(["pair_id"] +
               [f"{tag}_epe" for _, _, tag in RESOLUTIONS] +
               [f"{tag}_ms" for _, _, tag in RESOLUTIONS])

    per_res_epes = {tag: [] for _, _, tag in RESOLUTIONS}
    per_res_ms = {tag: [] for _, _, tag in RESOLUTIONS}

    for i, (lp, rp, pp) in enumerate(track):
        Lb = cv2.imread(lp); Rb = cv2.imread(rp)
        D_native = read_pfm(pp)                 # native 540x960
        H_n, W_n = D_native.shape

        row_tiles = []
        # Native GT column for reference, rendered at training resolution for
        # compactness.
        D_tr = cv2.resize(D_native, (640, 384), interpolation=cv2.INTER_LINEAR) * (640 / W_n)
        D_tr[~np.isfinite(D_tr) | (D_tr < 0)] = 0
        valid_tr = (D_tr > 0) & (D_tr < 192) & np.isfinite(D_tr)
        if valid_tr.sum() > 16:
            lo = float(np.percentile(D_tr[valid_tr], 2))
            hi = float(np.percentile(D_tr[valid_tr], 98))
        else:
            lo, hi = 0.0, 60.0
        L_tr = cv2.resize(Lb, (640, 384))
        row_tiles.append(annotate(L_tr, f"L pair{i:02d}  native {W_n}x{H_n}"))
        row_tiles.append(annotate(colourise(D_tr, lo, hi),
                                  f"GT @ 384x640  lo={lo:.1f}px hi={hi:.1f}px"))

        epe_row = [f"pair_{i:02d}"]
        for H, W, tag in RESOLUTIONS:
            pred, ms, L_at = run_at_resolution(model, Lb, Rb, H, W, device)
            # Scale pred disparity from W pixels back to training ref (640w)
            scale_to_tr = 640 / W
            pred_tr = cv2.resize(pred, (640, 384), interpolation=cv2.INTER_LINEAR) * scale_to_tr
            epe = float((np.abs(pred_tr - D_tr) * valid_tr).sum() / max(valid_tr.sum(), 1))
            per_res_epes[tag].append(epe)
            per_res_ms[tag].append(ms)
            # Rescale pred back to its own resolution for visualisation
            D_at_native = cv2.resize(D_native, (W, H), interpolation=cv2.INTER_LINEAR) * (W / W_n)
            D_at_native[~np.isfinite(D_at_native) | (D_at_native < 0)] = 0
            v_at = (D_at_native > 0) & (D_at_native < 192) & np.isfinite(D_at_native)
            if v_at.sum() > 16:
                lo_at = float(np.percentile(D_at_native[v_at], 2))
                hi_at = float(np.percentile(D_at_native[v_at], 98))
            else:
                lo_at, hi_at = lo, hi
            pred_col = colourise(pred, lo_at, hi_at)
            # Resize pred to training display size for side-by-side
            pred_col_tr = cv2.resize(pred_col, (640, 384), interpolation=cv2.INTER_LINEAR)
            row_tiles.append(annotate(pred_col_tr,
                                      f"{tag}  EPE={epe:.2f}px  {ms:.0f}ms"))
            epe_row.append(f"{epe:.3f}")

        w.writerow(epe_row + [f"{per_res_ms[t][-1]:.1f}" for _, _, t in RESOLUTIONS])

        row = np.hstack(row_tiles)
        cv2.imwrite(os.path.join(args.out_dir, f"pair_{i:02d}.png"), row)
        print(f"  pair {i:02d}:  " +
              "  ".join(f"{t}={per_res_epes[t][-1]:.2f}px/{per_res_ms[t][-1]:.0f}ms"
                        for _, _, t in RESOLUTIONS))

    fp.close()

    print("\nRESOLUTION SCALING — summary")
    print(f"  {'resolution':<18} {'mean EPE':>10} {'median ms':>12}")
    for _, _, tag in RESOLUTIONS:
        e = float(np.mean(per_res_epes[tag]))
        m = float(np.median(per_res_ms[tag]))
        print(f"  {tag:<18} {e:>9.3f}  {m:>11.1f}")
    print(f"\npanels:  {args.out_dir}")


if __name__ == "__main__":
    main()
