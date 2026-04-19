"""Offline comparison on data/user_cam_1/:
    [left | FoundationStereo pseudo-GT | StereoLite student | abs-error]

Uses the same colour range per-pair for an honest visual comparison. Writes
per-pair panels to model/benchmarks/compare_student_vs_teacher/ plus a
summary row listing EPE (student vs teacher) and bad-1 / bad-3 %.
"""
from __future__ import annotations

import argparse
import csv
import os
import sys

import cv2
import numpy as np
import torch

PROJ = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(PROJ, "model"))
sys.path.insert(0, os.path.join(PROJ, "model", "designs"))


INF_H, INF_W = 384, 640


def to_tensor(bgr, device):
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    rgb = cv2.resize(rgb, (INF_W, INF_H), interpolation=cv2.INTER_AREA)
    return torch.from_numpy(rgb).float().permute(2, 0, 1).unsqueeze(0).to(device)


def colourise(d, lo, hi, cmap=cv2.COLORMAP_TURBO):
    d = d.astype(np.float32)
    v = np.clip((d - lo) / max(hi - lo, 1e-6), 0, 1) * 255
    return cv2.applyColorMap(v.astype(np.uint8), cmap)


def annotate(img, text, h=24):
    out = img.copy()
    cv2.rectangle(out, (0, 0), (out.shape[1], h), (0, 0, 0), -1)
    cv2.putText(out, text, (8, h - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.48,
                (255, 255, 255), 1, cv2.LINE_AA)
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data_root", default=os.path.join(PROJ, "data", "user_cam_1"))
    p.add_argument("--ckpt", default=os.path.join(
        PROJ, "model", "checkpoints", "user_cam_1", "student_d1.pth"))
    p.add_argument("--out_dir", default=os.path.join(
        PROJ, "model", "benchmarks", "compare_student_vs_teacher"))
    p.add_argument("--n", type=int, default=10,
                   help="number of pairs to visualise")
    args = p.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    from d1_tile import StereoLite
    model = StereoLite().to(device)
    ck = torch.load(args.ckpt, map_location=device, weights_only=False)
    sd = ck["model"] if "model" in ck else ck
    model.load_state_dict(sd, strict=True)
    model.train()
    n_params = sum(q.numel() for q in model.parameters())
    print(f"loaded student {args.ckpt}  {n_params/1e6:.3f} M")

    left_dir = os.path.join(args.data_root, "left")
    right_dir = os.path.join(args.data_root, "right")
    disp_dir = os.path.join(args.data_root, "disp_pseudo")
    all_ids = sorted([f.replace(".png", "") for f in os.listdir(left_dir)
                      if f.endswith(".png")])
    # evenly spaced selection
    if args.n < len(all_ids):
        step = len(all_ids) / args.n
        sel = [all_ids[int(i * step)] for i in range(args.n)]
    else:
        sel = all_ids

    csv_path = os.path.join(args.out_dir, "summary.csv")
    fp = open(csv_path, "w", newline="")
    w = csv.writer(fp)
    w.writerow(["id", "epe_px", "bad1_pct", "bad3_pct",
                "teacher_min", "teacher_max", "student_min", "student_max"])

    epes = []
    for i, pid in enumerate(sel):
        lp = os.path.join(left_dir, f"{pid}.png")
        rp = os.path.join(right_dir, f"{pid}.png")
        dp = os.path.join(disp_dir, f"{pid}.npy")

        Lb = cv2.imread(lp)
        Rb = cv2.imread(rp)
        D_teacher = np.load(dp)                       # (H, W) at save resolution

        # Ensure D_teacher matches inference resolution
        if D_teacher.shape != (INF_H, INF_W):
            sx = INF_W / D_teacher.shape[1]
            D_teacher = cv2.resize(D_teacher, (INF_W, INF_H),
                                   interpolation=cv2.INTER_LINEAR) * sx

        Lt = to_tensor(Lb, device)
        Rt = to_tensor(Rb, device)
        if i == 0:
            _ = model(Lt, Rt)                         # warmup
        with torch.no_grad():
            D_student = model(Lt, Rt).squeeze().cpu().numpy()
        D_student = np.clip(D_student, 0, 192)

        valid = (D_teacher > 0.5) & (D_teacher < 192) & np.isfinite(D_teacher)
        err = np.abs(D_student - D_teacher) * valid
        n_v = max(valid.sum(), 1)
        epe = err.sum() / n_v
        bad1 = ((np.abs(D_student - D_teacher) > 1.0) & valid).sum() / n_v * 100
        bad3 = ((np.abs(D_student - D_teacher) > 3.0) & valid).sum() / n_v * 100
        epes.append(float(epe))
        w.writerow([pid, f"{epe:.3f}", f"{bad1:.1f}", f"{bad3:.1f}",
                    f"{D_teacher.min():.1f}", f"{D_teacher.max():.1f}",
                    f"{D_student.min():.1f}", f"{D_student.max():.1f}"])

        # Shared colour range for fairness
        if valid.sum() > 16:
            lo = float(np.percentile(D_teacher[valid], 2))
            hi = float(np.percentile(D_teacher[valid], 98))
        else:
            lo, hi = 0.0, 80.0

        L_small = cv2.resize(Lb, (INF_W, INF_H))
        err_vis = np.clip(err / 8.0, 0, 1) * 255
        err_col = cv2.applyColorMap(err_vis.astype(np.uint8), cv2.COLORMAP_HOT)

        tiles = [
            annotate(L_small, f"left  {pid}"),
            annotate(colourise(D_teacher, lo, hi),
                     f"FoundationStereo (teacher)  range {lo:.1f}-{hi:.1f} px"),
            annotate(colourise(D_student, lo, hi),
                     f"StereoLite distilled (0.62M)  EPE={epe:.2f}  "
                     f"bad1={bad1:.0f}%  bad3={bad3:.0f}%"),
            annotate(err_col, "abs(student - teacher)  clipped 0-8 px  HOT"),
        ]
        row = np.hstack(tiles)
        cv2.imwrite(os.path.join(args.out_dir, f"cmp_{pid}.png"), row)
        print(f"  [{i+1}/{len(sel)}] {pid}  EPE={epe:.2f}  bad1={bad1:.0f}%  bad3={bad3:.0f}%")

    fp.close()
    print(f"\nMean EPE over {len(epes)} panels: {float(np.mean(epes)):.3f} px")
    print(f"Panels in {args.out_dir}")
    print(f"Summary  {csv_path}")


if __name__ == "__main__":
    main()
