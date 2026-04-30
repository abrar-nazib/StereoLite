"""Live stereo-camera inference for StereoLite v8.

Opens the CCB stereo camera (/dev/video2 by default, 2560x720 side-by-side),
splits L/R, runs the trained model, and shows a three-panel cv2 window:
[left | predicted disparity (TURBO) | right]

Controls:
    q or ESC  quit
    s         save current L / R / disparity PNGs to /tmp/
    f         freeze / resume the feed
    +  / -    cycle through inference resolutions

Usage:
    python3 model/scripts/live_stereolite.py
    python3 model/scripts/live_stereolite.py --ckpt model/checkpoints/stereolite_v8.pth \
        --device 2 --inf_h 384 --inf_w 768
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
os.environ.setdefault("XFORMERS_DISABLED", "1")


def open_stereo_camera(device: int, w: int = 2560, h: int = 720,
                        fps: int = 60) -> cv2.VideoCapture:
    cap = cv2.VideoCapture(device, cv2.CAP_V4L2)
    if not cap.isOpened():
        raise RuntimeError(f"cannot open /dev/video{device}")
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
    cap.set(cv2.CAP_PROP_FPS, fps)
    aw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    ah = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    afps = cap.get(cv2.CAP_PROP_FPS)
    print(f"opened /dev/video{device}: {aw}x{ah} @ {afps:.0f} fps")
    if aw != w:
        print(f"warning: expected {w}px wide, got {aw}px — may be a single "
              f"camera crop instead of stereo")
    return cap


def split_stereo(frame: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mid = frame.shape[1] // 2
    return frame[:, :mid], frame[:, mid:]


def colourise(d: np.ndarray, lo: float, hi: float) -> np.ndarray:
    v = np.clip((d - lo) / max(hi - lo, 1e-6), 0, 1) * 255
    return cv2.applyColorMap(v.astype(np.uint8), cv2.COLORMAP_TURBO)


def annotate(img: np.ndarray, lines: list[str], h: int = 24) -> np.ndarray:
    out = img.copy()
    bar_h = h * max(len(lines), 1)
    cv2.rectangle(out, (0, 0), (out.shape[1], bar_h), (0, 0, 0), -1)
    for i, line in enumerate(lines):
        cv2.putText(out, line, (8, (i + 1) * h - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.50, (255, 255, 255), 1,
                    cv2.LINE_AA)
    return out


@torch.no_grad()
def infer(model, L_bgr: np.ndarray, R_bgr: np.ndarray, inf_h: int, inf_w: int,
          device: torch.device) -> tuple[np.ndarray, float]:
    L = cv2.resize(L_bgr, (inf_w, inf_h), interpolation=cv2.INTER_AREA)
    R = cv2.resize(R_bgr, (inf_w, inf_h), interpolation=cv2.INTER_AREA)
    Lt = torch.from_numpy(cv2.cvtColor(L, cv2.COLOR_BGR2RGB)).float() \
            .permute(2, 0, 1).unsqueeze(0).to(device)
    Rt = torch.from_numpy(cv2.cvtColor(R, cv2.COLOR_BGR2RGB)).float() \
            .permute(2, 0, 1).unsqueeze(0).to(device)
    if device.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.time()
    disp = model(Lt, Rt).squeeze().float().cpu().numpy()
    if device.type == "cuda":
        torch.cuda.synchronize()
    return disp, (time.time() - t0) * 1000


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default=os.path.join(
        PROJ, "model", "checkpoints", "stereolite_v8.pth"))
    p.add_argument("--device", type=int, default=2,
                   help="/dev/video<N> for the stereo camera")
    p.add_argument("--inf_h", type=int, default=384)
    p.add_argument("--inf_w", type=int, default=768)
    p.add_argument("--cam_w", type=int, default=2560)
    p.add_argument("--cam_h", type=int, default=720)
    p.add_argument("--disp_min", type=float, default=0.0)
    p.add_argument("--disp_max", type=float, default=0.0,
                   help="colourmap upper bound; 0 = auto (percentile)")
    p.add_argument("--display_w", type=int, default=1600,
                   help="max width of the composed display window")
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device}  ckpt={args.ckpt}")
    print(f"inference resolution: {args.inf_h}x{args.inf_w}")

    from StereoLite import StereoLite, StereoLiteConfig
    model = StereoLite(StereoLiteConfig(backbone="mobilenet")).to(device)
    ck = torch.load(args.ckpt, map_location=device, weights_only=False)
    sd = ck["model"] if "model" in ck else ck
    missing, unexpected = model.load_state_dict(sd, strict=True)
    print(f"loaded (missing={len(missing)} unexpected={len(unexpected)})")
    # Eval mode: BN running stats in MobileNetV2 are frozen, consistent with
    # training stats. GroupNorm elsewhere is train/eval-identical.
    model.eval()

    # Warmup
    dummy = np.zeros((args.cam_h, args.cam_w // 2, 3), dtype=np.uint8)
    for _ in range(3):
        infer(model, dummy, dummy, args.inf_h, args.inf_w, device)
    print("warmup done")

    cap = open_stereo_camera(args.device, args.cam_w, args.cam_h)

    win = "StereoLite v8 — live  (q quit, s save, f freeze, +/- resolution)"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win, args.display_w, args.display_w // 3)

    # Cycle-able inference resolutions
    res_choices = [(256, 512), (320, 640), (384, 768), (448, 832), (512, 832)]
    try:
        res_idx = res_choices.index((args.inf_h, args.inf_w))
    except ValueError:
        res_choices.insert(0, (args.inf_h, args.inf_w))
        res_idx = 0

    frozen_frame = None
    save_count = 0
    fps_hist: list[float] = []
    ms_hist: list[float] = []
    last_wall = time.time()

    print("running. Controls: q/ESC quit  s save  f freeze  +/- cycle resolution")
    while True:
        if frozen_frame is None:
            ret, frame = cap.read()
            if not ret:
                print("frame grab failed")
                break
        else:
            frame = frozen_frame

        L_full, R_full = split_stereo(frame)
        inf_h, inf_w = res_choices[res_idx]
        disp, ms_inf = infer(model, L_full, R_full, inf_h, inf_w, device)

        if args.disp_max > 0:
            lo, hi = args.disp_min, args.disp_max
        else:
            valid = disp > 0.5
            if valid.sum() > 64:
                lo = float(np.percentile(disp[valid], 5))
                hi = float(np.percentile(disp[valid], 95))
            else:
                lo, hi = 0.0, 60.0

        disp_col = colourise(disp, lo, hi)
        disp_col = cv2.resize(disp_col, (L_full.shape[1], L_full.shape[0]))

        now = time.time()
        dt = now - last_wall
        last_wall = now
        if dt > 0:
            fps_hist.append(1.0 / dt)
        ms_hist.append(ms_inf)
        fps_hist = fps_hist[-30:]
        ms_hist = ms_hist[-30:]
        med_fps = float(np.median(fps_hist)) if fps_hist else 0.0
        med_ms = float(np.median(ms_hist))

        L_ann = annotate(L_full, [
            "left",
            f"cam {args.cam_w // 2}x{args.cam_h}  loop {med_fps:5.1f} fps",
        ])
        D_ann = annotate(disp_col, [
            "disparity (TURBO)",
            f"inf {inf_w}x{inf_h}  {med_ms:5.1f} ms  "
            f"range {lo:.1f}..{hi:.1f}{'  FROZEN' if frozen_frame is not None else ''}",
        ])
        R_ann = annotate(R_full, ["right", "StereoLite v8  |  2.14 M params"])

        composed = np.hstack([L_ann, D_ann, R_ann])
        if composed.shape[1] > args.display_w:
            scale = args.display_w / composed.shape[1]
            composed = cv2.resize(composed,
                                    (args.display_w,
                                     int(composed.shape[0] * scale)))
        cv2.imshow(win, composed)

        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), ord("Q"), 27):
            break
        elif key == ord("s"):
            cv2.imwrite(f"/tmp/stereolite_left_{save_count:04d}.png", L_full)
            cv2.imwrite(f"/tmp/stereolite_right_{save_count:04d}.png", R_full)
            cv2.imwrite(f"/tmp/stereolite_disp_{save_count:04d}.png", disp_col)
            np.save(f"/tmp/stereolite_disp_{save_count:04d}.npy", disp)
            print(f"saved pair {save_count:04d} to /tmp/stereolite_*")
            save_count += 1
        elif key == ord("f"):
            frozen_frame = frame.copy() if frozen_frame is None else None
            print("frozen" if frozen_frame is not None else "resumed")
        elif key in (ord("+"), ord("=")):
            res_idx = (res_idx + 1) % len(res_choices)
            print(f"resolution -> {res_choices[res_idx]}")
        elif key in (ord("-"), ord("_")):
            res_idx = (res_idx - 1) % len(res_choices)
            print(f"resolution -> {res_choices[res_idx]}")

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
