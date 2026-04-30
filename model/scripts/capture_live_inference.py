"""Interactive stereo capture + live inference from the StereoLite student.

Opens /dev/video2, runs the student at ~30-50 FPS, and displays:
    Left feed | Right feed
    Live disparity colourmap | Live depth colourmap

SPACE saves the raw stereo pair (full 1280x720 per eye) to
data/<name>/{left,right}/00000.png for later pseudo-GT + distillation.
U undoes the last save. Q / ESC quits.

Usage:
    python3 model/scripts/capture_live_inference.py --name user_cam_2 \
        --ckpt model/checkpoints/user_cam_1/student_d1.pth --target 100
"""
from __future__ import annotations

import argparse
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


CAM_DEVICE = 2
FRAME_W = 2560
FRAME_H = 720
FPS = 60

INF_H, INF_W = 384, 640
BASELINE_M = 0.052
FOCAL_INF_PX = 960.0 * INF_W / 1280
DEPTH_NUM = FOCAL_INF_PX * BASELINE_M
DEPTH_MIN, DEPTH_MAX = 0.25, 5.0


def open_cam(device: int) -> cv2.VideoCapture:
    cap = cv2.VideoCapture(device, cv2.CAP_V4L2)
    if not cap.isOpened():
        raise RuntimeError(f"cannot open /dev/video{device}")
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_H)
    cap.set(cv2.CAP_PROP_FPS, FPS)
    aw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    ah = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"opened /dev/video{device}: {aw}x{ah}")
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


def annotate_top(img, text, h=26):
    out = img.copy()
    cv2.rectangle(out, (0, 0), (out.shape[1], h), (0, 0, 0), -1)
    cv2.putText(out, text, (10, h - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.52,
                (255, 255, 255), 1, cv2.LINE_AA)
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--name", default=None,
                   help="subdir under data/ (default: user_cam_<ts>)")
    p.add_argument("--ckpt", default=os.path.join(
        PROJ, "model", "checkpoints", "user_cam_1", "student_d1.pth"))
    p.add_argument("--target", type=int, default=100)
    p.add_argument("--device", type=int, default=CAM_DEVICE)
    p.add_argument("--warmup", type=int, default=15)
    args = p.parse_args()

    if args.name is None:
        args.name = f"user_cam_{dt.datetime.now().strftime('%Y%m%d_%H%M%S')}"
    out_dir = os.path.join(PROJ, "data", args.name)
    left_dir = os.path.join(out_dir, "left")
    right_dir = os.path.join(out_dir, "right")
    os.makedirs(left_dir, exist_ok=True)
    os.makedirs(right_dir, exist_ok=True)

    existing = sorted([f for f in os.listdir(left_dir) if f.endswith(".png")])
    saved = len(existing)
    if saved:
        print(f"resuming: {saved} existing pairs under {out_dir}/")

    tdev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={tdev.type}; loading student from {args.ckpt}")
    from StereoLite import StereoLite
    model = StereoLite().to(tdev)
    ck = torch.load(args.ckpt, map_location=tdev, weights_only=False)
    sd = ck["model"] if "model" in ck else ck
    model.load_state_dict(sd, strict=True)
    model.train()     # GroupNorm batch-independent; keep train mode
    n_params = sum(q.numel() for q in model.parameters())
    print(f"student params={n_params/1e6:.3f} M")

    cap = open_cam(args.device)
    print(f"warming up camera ({args.warmup} frames)...")
    for _ in range(args.warmup):
        cap.read()

    # warmup model
    ok, frame = cap.read()
    if ok:
        mid = frame.shape[1] // 2
        _ = model(to_tensor(frame[:, :mid], tdev),
                  to_tensor(frame[:, mid:], tdev))
        if tdev.type == "cuda":
            torch.cuda.synchronize()

    win = f"Capture + live infer -> {args.name}  (SPACE=save, U=undo, Q=quit)"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win, 1600, 900)

    last_saved_name = "(none)"
    last_save_ms = 0.0
    ema_infer_ms = 0.0

    print(f"\ncapturing into {out_dir}/")
    print("  SPACE = save  U = undo  Q / ESC = quit\n")

    frames_since_save = 0
    min_gap = 3
    while True:
        ok, frame = cap.read()
        if not ok:
            continue
        frames_since_save += 1

        mid = frame.shape[1] // 2
        Lb = frame[:, :mid]
        Rb = frame[:, mid:]

        Lt = to_tensor(Lb, tdev)
        Rt = to_tensor(Rb, tdev)
        t0 = time.time()
        with torch.no_grad():
            disp = model(Lt, Rt)
        if tdev.type == "cuda":
            torch.cuda.synchronize()
        ms = (time.time() - t0) * 1000
        ema_infer_ms = 0.85 * ema_infer_ms + 0.15 * ms if ema_infer_ms else ms
        d_np = disp.squeeze().cpu().numpy()
        d_np = np.clip(d_np, 0, 192)

        L_small = cv2.resize(Lb, (INF_W, INF_H))
        R_small = cv2.resize(Rb, (INF_W, INF_H))
        div = np.full((INF_H, 4, 3), (0, 255, 0), dtype=np.uint8)
        feed_row = annotate_top(np.hstack([L_small, div, R_small]),
                                f"Stereo Feed (L | R)   saved {saved}/{args.target}   "
                                f"last={last_saved_name}   t_save={last_save_ms:.1f} ms")
        disp_vis = annotate_top(colour_disp(d_np),
                                f"Live disparity ({n_params/1e6:.2f}M)   "
                                f"{ms:5.1f} ms  EMA {ema_infer_ms:5.1f} ms  "
                                f"range [{d_np.min():.1f}, {d_np.max():.1f}] px")
        depth_vis = annotate_top(colour_depth(d_np),
                                 f"Depth  range {DEPTH_MIN}-{DEPTH_MAX} m   INFERNO")
        bottom_row = np.hstack([disp_vis, depth_vis])
        if feed_row.shape[1] != bottom_row.shape[1]:
            feed_row = cv2.resize(feed_row, (bottom_row.shape[1], feed_row.shape[0]))
        canvas = np.vstack([feed_row, bottom_row])
        cv2.imshow(win, canvas)

        k = cv2.waitKey(1) & 0xFF
        if k in (ord("q"), ord("Q"), 27):
            break
        if k == ord(" ") and frames_since_save >= min_gap:
            t0 = time.time()
            fname = f"{saved:05d}.png"
            cv2.imwrite(os.path.join(left_dir, fname), Lb)
            cv2.imwrite(os.path.join(right_dir, fname), Rb)
            last_save_ms = (time.time() - t0) * 1000
            last_saved_name = fname
            saved += 1
            frames_since_save = 0
            print(f"  [{saved:3d}/{args.target}] saved {fname}  ({last_save_ms:.1f} ms)")
            if saved >= args.target:
                print(f"reached target {args.target}")
                break
        if k in (ord("u"), ord("U")) and saved > 0:
            saved -= 1
            fname = f"{saved:05d}.png"
            for d in (left_dir, right_dir):
                path = os.path.join(d, fname)
                if os.path.exists(path):
                    os.remove(path)
            last_saved_name = f"(undone {fname})"
            print(f"  UNDO -> removed {fname}; now at {saved}")

    cap.release()
    cv2.destroyAllWindows()
    print(f"\nDone. {saved} pairs saved at {out_dir}/")
    print(f"\nNext:")
    print(f"  python3 model/scripts/run_teacher.py --pairs_dir data/{args.name}")
    print(f"  python3 model/scripts/distill_train.py --student d1 "
          f"--data_root data/{args.name} --out_dir model/checkpoints/{args.name}")


if __name__ == "__main__":
    main()
