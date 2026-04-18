"""Interactive stereo-pair capture from the AR0144 camera.

Opens /dev/video2 (MJPG, 2560x720), shows a live preview window, and lets
the user press SPACE to save the current pair. Press Q / ESC to quit.

Saves raw stereo pairs to:
    data/<name>/left/00000.png
    data/<name>/right/00000.png

Where <name> defaults to user_camera_<timestamp>. The layout matches what
run_teacher.py and distill_train.py expect, so the downstream flow is:

    python3 model/scripts/capture_interactive.py --name user_cam_1
    python3 model/scripts/run_teacher.py --pairs_dir data/user_cam_1
    python3 model/scripts/distill_train.py --student d1 --pairs_dir data/user_cam_1

Preview window shows:
  - Left | Right feed
  - frame counter / target
  - last-capture thumbnail (so you can confirm a good pair was written)
"""
from __future__ import annotations

import argparse
import datetime as dt
import os
import time

import cv2
import numpy as np


CAM_DEVICE = 2
FRAME_W = 2560
FRAME_H = 720
FPS = 60


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
    if aw != FRAME_W:
        print(f"  WARNING: got {aw}px wide, expected {FRAME_W} for stereo")
    return cap


def annotate(img, lines, scale=0.6, thick=1):
    out = img.copy()
    y = 28
    for line in lines:
        cv2.putText(out, line, (12, y), cv2.FONT_HERSHEY_SIMPLEX, scale,
                    (0, 0, 0), thick + 2, cv2.LINE_AA)
        cv2.putText(out, line, (12, y), cv2.FONT_HERSHEY_SIMPLEX, scale,
                    (255, 255, 255), thick, cv2.LINE_AA)
        y += int(26 * scale / 0.6)
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--name", default=None,
                   help="subdir under data/ (default: user_camera_<ts>)")
    p.add_argument("--target", type=int, default=100,
                   help="target number of pairs to capture")
    p.add_argument("--device", type=int, default=CAM_DEVICE)
    p.add_argument("--preview_w", type=int, default=1600,
                   help="width of on-screen preview window")
    p.add_argument("--min_gap_frames", type=int, default=3,
                   help="min camera reads between saves to flush the buffer")
    p.add_argument("--warmup", type=int, default=15)
    args = p.parse_args()

    if args.name is None:
        args.name = f"user_camera_{dt.datetime.now().strftime('%Y%m%d_%H%M%S')}"
    proj = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    out_dir = os.path.join(proj, "data", args.name)
    left_dir = os.path.join(out_dir, "left")
    right_dir = os.path.join(out_dir, "right")
    os.makedirs(left_dir, exist_ok=True)
    os.makedirs(right_dir, exist_ok=True)

    # Resume if already populated
    existing = sorted([f for f in os.listdir(left_dir) if f.endswith(".png")])
    saved = len(existing)
    if saved:
        print(f"found {saved} existing pairs under {out_dir}/; appending")

    cap = open_cam(args.device)
    print(f"warming up ({args.warmup} frames)...")
    for _ in range(args.warmup):
        cap.read()

    win = f"Stereo capture -> {args.name}  (SPACE=save, Q=quit, U=undo last)"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win, args.preview_w, int(args.preview_w * 0.28) + 40)

    thumb = np.zeros((180, 320, 3), dtype=np.uint8)
    last_save_ms = 0.0
    last_saved_name = "(none)"
    frames_since_save = 0

    print(f"\ncapturing into {out_dir}/")
    print("  SPACE = save pair   Q / ESC = quit   U = undo last\n")

    while True:
        ok, frame = cap.read()
        if not ok:
            continue
        frames_since_save += 1

        mid = frame.shape[1] // 2
        left_full = frame[:, :mid]
        right_full = frame[:, mid:]

        # build preview: L | R side by side, downsampled
        pv_h = 360
        pv_w_each = int(left_full.shape[1] * pv_h / left_full.shape[0])
        lp = cv2.resize(left_full, (pv_w_each, pv_h))
        rp = cv2.resize(right_full, (pv_w_each, pv_h))
        div = np.full((pv_h, 6, 3), (0, 255, 0), dtype=np.uint8)
        stereo_pv = np.hstack([lp, div, rp])

        # side thumb panel with last saved crop
        thumb_resized = cv2.resize(thumb, (stereo_pv.shape[1] // 4, pv_h // 2))
        panel = np.zeros((pv_h, thumb_resized.shape[1], 3), dtype=np.uint8)
        panel[:thumb_resized.shape[0], :, :] = thumb_resized
        panel = annotate(panel,
                         [f"saved: {saved}/{args.target}",
                          f"last: {last_saved_name}",
                          f"t_save: {last_save_ms:5.1f} ms"],
                         scale=0.55)

        canvas = np.hstack([stereo_pv, panel])
        canvas = annotate(canvas,
                          [f"{args.name}   {saved}/{args.target} pairs  "
                           f"(SPACE=save, U=undo, Q=quit)"])

        cv2.imshow(win, canvas)
        k = cv2.waitKey(1) & 0xFF
        if k in (ord("q"), ord("Q"), 27):
            break
        if k == ord(" ") and frames_since_save >= args.min_gap_frames:
            t0 = time.time()
            fname = f"{saved:05d}.png"
            cv2.imwrite(os.path.join(left_dir, fname), left_full)
            cv2.imwrite(os.path.join(right_dir, fname), right_full)
            last_save_ms = (time.time() - t0) * 1000
            thumb = cv2.resize(np.hstack([left_full, right_full]),
                               (640, 180))
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
    print(f"\nNext steps:")
    print(f"  1) python3 model/scripts/run_teacher.py --pairs_dir {os.path.relpath(out_dir, proj)}")
    print(f"  2) python3 model/scripts/distill_train.py --student d1 --pairs_dir {os.path.relpath(out_dir, proj)}")


if __name__ == "__main__":
    main()
