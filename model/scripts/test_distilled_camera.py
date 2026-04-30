"""Quick live test of the StereoLite student checkpoint trained on
data/user_cam_1/ pseudo-GT.

Captures one fresh stereo pair, runs the new student, and saves a 3-window
panel (stereo feed | disparity | depth) to model/benchmarks/camera_distilled/.

Usage:
    python3 model/scripts/test_distilled_camera.py \
        --ckpt model/checkpoints/user_cam_1/student_d1.pth
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import cv2
import numpy as np
import torch

PROJ = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(PROJ, "model"))
sys.path.insert(0, os.path.join(PROJ, "model", "designs"))

INF_H, INF_W = 384, 640
BASELINE_M = 0.052
FOCAL_INF_PX = 960.0 * INF_W / 1280
DEPTH_NUM = FOCAL_INF_PX * BASELINE_M
DEPTH_MIN, DEPTH_MAX = 0.25, 5.0


def open_cam():
    cap = cv2.VideoCapture(2, cv2.CAP_V4L2)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 2560)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_FPS, 60)
    return cap


def to_tensor(bgr, device):
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    rgb = cv2.resize(rgb, (INF_W, INF_H), interpolation=cv2.INTER_AREA)
    return torch.from_numpy(rgb).float().permute(2, 0, 1).unsqueeze(0).to(device)


def colour_disp(d):
    d = d.astype(np.float32)
    valid = d > 0
    if valid.sum() < 16:
        return np.zeros((*d.shape, 3), dtype=np.uint8)
    lo, hi = np.percentile(d[valid], [2, 98])
    v = np.clip((d - lo) / max(hi - lo, 1e-6), 0, 1) * 255
    return cv2.applyColorMap(v.astype(np.uint8), cv2.COLORMAP_TURBO)


def colour_depth(d):
    z = np.where(d > 0.1, DEPTH_NUM / d, DEPTH_MAX)
    z = np.clip(z, DEPTH_MIN, DEPTH_MAX)
    norm = 1.0 - (z - DEPTH_MIN) / (DEPTH_MAX - DEPTH_MIN)
    return cv2.applyColorMap((norm * 255).astype(np.uint8), cv2.COLORMAP_INFERNO)


def annotate(img, text, h=24):
    out = img.copy()
    cv2.rectangle(out, (0, 0), (out.shape[1], h), (0, 0, 0), -1)
    cv2.putText(out, text, (8, h - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.48,
                (255, 255, 255), 1, cv2.LINE_AA)
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default=os.path.join(
        PROJ, "model", "checkpoints", "user_cam_1", "student_d1.pth"))
    p.add_argument("--out_dir", default=os.path.join(
        PROJ, "model", "benchmarks", "camera_distilled"))
    p.add_argument("--frames", type=int, default=1,
                   help="capture N fresh frames, save a panel per frame")
    args = p.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    from StereoLite import StereoLite
    model = StereoLite().to(device)
    ck = torch.load(args.ckpt, map_location=device, weights_only=False)
    sd = ck["model"] if "model" in ck else ck
    model.load_state_dict(sd, strict=True)
    model.train()   # GroupNorm batch-independent
    n_params = sum(q.numel() for q in model.parameters())
    print(f"loaded {args.ckpt}   {n_params/1e6:.3f} M params")

    print("opening camera and capturing...")
    cap = open_cam()
    for _ in range(15):
        cap.read()

    for i in range(args.frames):
        ok, frame = cap.read()
        if not ok:
            print(f"frame {i}: grab failed, retrying")
            continue
        mid = frame.shape[1] // 2
        Lb = frame[:, :mid].copy()
        Rb = frame[:, mid:].copy()

        Lt = to_tensor(Lb, device)
        Rt = to_tensor(Rb, device)

        # warmup once
        if i == 0:
            _ = model(Lt, Rt)
            if device.type == "cuda":
                torch.cuda.synchronize()

        t0 = time.time()
        with torch.no_grad():
            disp = model(Lt, Rt)
        if device.type == "cuda":
            torch.cuda.synchronize()
        ms = (time.time() - t0) * 1000
        d_np = disp.squeeze().cpu().numpy()
        d_np = np.clip(d_np, 0, 192)
        print(f"  frame {i}: {ms:.0f} ms, disp [{d_np.min():.1f}, {d_np.max():.1f}]")

        L_small = cv2.resize(Lb, (INF_W, INF_H))
        R_small = cv2.resize(Rb, (INF_W, INF_H))
        div = np.full((INF_H, 4, 3), (0, 255, 0), dtype=np.uint8)
        feed = annotate(np.hstack([L_small, div, R_small]),
                        f"Stereo Feed (L | R)   {INF_W}x{INF_H} per eye")
        disp_vis = annotate(colour_disp(d_np),
                            f"StereoLite distilled on user_cam_1 ({n_params/1e6:.2f}M)   {ms:.0f} ms")
        depth_vis = annotate(colour_depth(d_np),
                             f"Depth  range {DEPTH_MIN}-{DEPTH_MAX}m   INFERNO")
        bottom = np.hstack([disp_vis, depth_vis])
        composite = np.vstack([cv2.resize(feed, (bottom.shape[1], feed.shape[0])), bottom])
        out_path = os.path.join(args.out_dir, f"panel_distilled_{i:02d}.png")
        cv2.imwrite(out_path, composite)
        print(f"  saved {out_path}")

    cap.release()


if __name__ == "__main__":
    main()
