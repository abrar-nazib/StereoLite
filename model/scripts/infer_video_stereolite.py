"""Run StereoLite on a stereo video pair and write a side-by-side
[Left | colorized disparity] MP4 at the source frame rate.

Pacing: by default the script processes every Nth frame (stride=2) and
writes at src_fps / stride, so playback timing is identical to the
original — just half the encoded frames. Set --stride 1 for full
density (all frames).

A warm-up pass over `--warmup_frames` evenly-spaced frames computes a
fixed disparity colormap range (p2 – p98), so the heatmap stays
visually stable across the whole clip instead of pulsing with each
frame's local min/max.
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import cv2
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "designs"))
from d1_tile import StereoLite, StereoLiteConfig  # noqa: E402


def colorize_disp(disp: np.ndarray, lo: float, hi: float) -> np.ndarray:
    """(H, W) float32 disparity → (H, W, 3) BGR uint8 with TURBO colormap."""
    norm = np.clip((disp - lo) / max(hi - lo, 1e-6), 0.0, 1.0)
    return cv2.applyColorMap((norm * 255).astype(np.uint8), cv2.COLORMAP_TURBO)


def predict_one(model, L_bgr: np.ndarray, R_bgr: np.ndarray,
                 device, inf_h: int, inf_w: int) -> np.ndarray:
    """One stereo pair → native-resolution disparity (H_n, W_n) float32."""
    H_n, W_n = L_bgr.shape[:2]
    L = cv2.resize(L_bgr, (inf_w, inf_h), interpolation=cv2.INTER_AREA)
    R = cv2.resize(R_bgr, (inf_w, inf_h), interpolation=cv2.INTER_AREA)
    Lt = torch.from_numpy(cv2.cvtColor(L, cv2.COLOR_BGR2RGB)).float() \
              .permute(2, 0, 1).unsqueeze(0).to(device, non_blocking=True)
    Rt = torch.from_numpy(cv2.cvtColor(R, cv2.COLOR_BGR2RGB)).float() \
              .permute(2, 0, 1).unsqueeze(0).to(device, non_blocking=True)
    with torch.no_grad():
        d = model(Lt, Rt).squeeze().cpu().numpy().astype(np.float32)
    sx = W_n / inf_w
    return cv2.resize(d, (W_n, H_n), interpolation=cv2.INTER_LINEAR) * sx


def warmup_range(model, capL, capR, n_frames: int, device,
                  inf_h: int, inf_w: int,
                  lo_pct: float = 5.0, hi_pct: float = 98.0,
                  edge_skip: float = 0.05) -> tuple[float, float]:
    """Sample n_frames across the middle (1-edge_skip) of the clip and
    return per-frame-median-aggregated p_lo / p_hi disparity bounds.

    Why per-frame medians? A single noisy frame (lens cap on, glare,
    motion blur) can pull pooled percentiles wildly. Taking each frame's
    median first makes the aggregate robust to those outliers.
    """
    total = int(capL.get(cv2.CAP_PROP_FRAME_COUNT))
    a, b = int(total * edge_skip), int(total * (1 - edge_skip))
    idxs = np.linspace(a, max(b - 1, a), n_frames).astype(int)
    per_frame_lo, per_frame_hi = [], []
    for i, fi in enumerate(idxs):
        capL.set(cv2.CAP_PROP_POS_FRAMES, fi)
        capR.set(cv2.CAP_PROP_POS_FRAMES, fi)
        okL, L = capL.read()
        okR, R = capR.read()
        if not (okL and okR):
            continue
        d = predict_one(model, L, R, device, inf_h, inf_w)
        valid = d[(d > 0.5) & np.isfinite(d)]
        if valid.size < 100:
            continue
        f_lo = float(np.percentile(valid, lo_pct))
        f_hi = float(np.percentile(valid, hi_pct))
        per_frame_lo.append(f_lo)
        per_frame_hi.append(f_hi)
        print(f"  warmup {i+1}/{len(idxs)}: frame {fi}  "
              f"p{lo_pct:.0f}={f_lo:.1f}  p{hi_pct:.0f}={f_hi:.1f}")
    if not per_frame_lo:
        capL.set(cv2.CAP_PROP_POS_FRAMES, 0)
        capR.set(cv2.CAP_PROP_POS_FRAMES, 0)
        return 1.0, 100.0
    # Median across frames is robust to a few weird ones
    lo = float(np.median(per_frame_lo))
    hi = float(np.median(per_frame_hi))
    capL.set(cv2.CAP_PROP_POS_FRAMES, 0)
    capR.set(cv2.CAP_PROP_POS_FRAMES, 0)
    return lo, hi


def label_panel(W: int, H: int, panels: list[np.ndarray],
                 titles: list[str]) -> np.ndarray:
    """Build a row of (each panel + caption strip on top)."""
    bar_h = 36
    out_W = sum(p.shape[1] for p in panels)
    out_H = H + bar_h
    canvas = np.full((out_H, out_W, 3), (12, 13, 17), dtype=np.uint8)
    x = 0
    for p, t in zip(panels, titles):
        canvas[bar_h:bar_h + p.shape[0], x:x + p.shape[1]] = p
        cv2.putText(canvas, t, (x + 14, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (235, 235, 240),
                    1, cv2.LINE_AA)
        x += p.shape[1]
    return canvas


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--left_video", required=True)
    p.add_argument("--right_video", required=True)
    p.add_argument("--out_path", required=True,
                   help="output .mp4 path")
    p.add_argument("--ckpt",
                   default="model/checkpoints/stereolite_finetune_indoor_best.pth")
    p.add_argument("--stride", type=int, default=2,
                   help="process every Nth frame; output fps = src_fps/stride. "
                        "Combine with --slowdown to slow playback.")
    p.add_argument("--slowdown", type=float, default=1.0,
                   help="playback time-stretch factor. 1.0 = real-time, "
                        "1.5 = 1.5× slower. Implemented by dividing the "
                        "encoded fps without changing frame count.")
    p.add_argument("--disp_lo", type=float, default=None,
                   help="hard override for colormap lower bound (px)")
    p.add_argument("--disp_hi", type=float, default=None,
                   help="hard override for colormap upper bound (px)")
    p.add_argument("--inf_h", type=int, default=512)
    p.add_argument("--inf_w", type=int, default=832)
    p.add_argument("--warmup_frames", type=int, default=24,
                   help="evenly-spaced frames sampled to fix the colormap range")
    p.add_argument("--max_seconds", type=float, default=None,
                   help="optional cap on processed clip duration in seconds "
                        "(measured at source fps; useful for trimming)")
    p.add_argument("--out_w", type=int, default=1080,
                   help="per-panel width in the output video")
    p.add_argument("--crf", type=int, default=20,
                   help="ffmpeg H.264 CRF quality (lower = better)")
    args = p.parse_args()

    capL = cv2.VideoCapture(args.left_video)
    capR = cv2.VideoCapture(args.right_video)
    if not (capL.isOpened() and capR.isOpened()):
        raise SystemExit("could not open one of the input videos")

    src_fps = capL.get(cv2.CAP_PROP_FPS)
    src_w = int(capL.get(cv2.CAP_PROP_FRAME_WIDTH))
    src_h = int(capL.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(capL.get(cv2.CAP_PROP_FRAME_COUNT))
    if args.max_seconds:
        total = min(total, int(round(args.max_seconds * src_fps)))
    print(f"input: {args.left_video}  {src_w}x{src_h} @ {src_fps:.2f} fps  "
          f"{total} frames  ({total/src_fps:.1f} s)")

    out_fps = (src_fps / args.stride) / max(args.slowdown, 1e-3)
    out_panel_h = int(round(src_h * args.out_w / src_w))
    out_canvas_w = args.out_w * 2                            # [L | disp]
    out_canvas_h = out_panel_h + 36                          # + caption strip
    print(f"output: {args.out_path}  panels {args.out_w}x{out_panel_h}  "
          f"canvas {out_canvas_w}x{out_canvas_h} @ {out_fps:.2f} fps")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device.type}  ckpt={args.ckpt}")
    model = StereoLite(StereoLiteConfig(
        backbone="mobilenet", use_dav2=False)).to(device)
    ck = torch.load(args.ckpt, map_location=device, weights_only=False)
    model.load_state_dict(ck["model"] if "model" in ck else ck, strict=True)
    model.eval()
    if "epe" in ck:
        print(f"  loaded checkpoint val EPE: {ck['epe']:.3f} px")

    # Warm up the colormap so colors don't pulse across the clip
    if args.disp_lo is not None and args.disp_hi is not None:
        lo, hi = args.disp_lo, args.disp_hi
        print(f"\ncolormap range manually fixed: [{lo:.1f}, {hi:.1f}] px")
    else:
        print(f"\nwarmup ({args.warmup_frames} sampled frames) ...")
        lo, hi = warmup_range(model, capL, capR, args.warmup_frames,
                               device, args.inf_h, args.inf_w)
        if args.disp_lo is not None: lo = args.disp_lo
        if args.disp_hi is not None: hi = args.disp_hi
        print(f"colormap range: [{lo:.1f}, {hi:.1f}] px")

    # Encode via ffmpeg pipe so we don't depend on the OpenCV codec
    # build (libopenh264 is sometimes missing). yuv420p needs even dims.
    canvas_w_even = out_canvas_w + (out_canvas_w & 1)
    canvas_h_even = out_canvas_h + (out_canvas_h & 1)
    out_path = Path(args.out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg", "-y", "-loglevel", "warning",
        "-f", "rawvideo", "-pix_fmt", "bgr24",
        "-s", f"{canvas_w_even}x{canvas_h_even}",
        "-r", f"{out_fps}",
        "-i", "-",
        "-c:v", "libx264", "-pix_fmt", "yuv420p",
        "-crf", str(args.crf), "-preset", "medium",
        "-movflags", "+faststart",
        str(out_path),
    ]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)

    # Streaming loop
    t0 = time.time()
    frame_idx = 0
    written = 0
    while True:
        if frame_idx >= total:
            break
        okL, L = capL.read()
        okR, R = capR.read()
        if not (okL and okR):
            break
        if frame_idx % args.stride == 0:
            disp = predict_one(model, L, R, device, args.inf_h, args.inf_w)
            disp_color = colorize_disp(disp, lo, hi)
            L_panel = cv2.resize(L, (args.out_w, out_panel_h),
                                  interpolation=cv2.INTER_AREA)
            D_panel = cv2.resize(disp_color, (args.out_w, out_panel_h),
                                  interpolation=cv2.INTER_AREA)
            canvas = label_panel(args.out_w, out_panel_h,
                                  [L_panel, D_panel],
                                  ["Left RGB",
                                   f"StereoLite disparity (px), "
                                   f"range [{lo:.0f}, {hi:.0f}]"])
            if canvas.shape[1] != canvas_w_even \
                    or canvas.shape[0] != canvas_h_even:
                canvas = cv2.copyMakeBorder(
                    canvas, 0, canvas_h_even - canvas.shape[0],
                    0, canvas_w_even - canvas.shape[1],
                    cv2.BORDER_CONSTANT, value=(12, 13, 17))
            proc.stdin.write(canvas.tobytes())
            written += 1
            if written % 100 == 0:
                el = time.time() - t0
                rate = written / el if el > 0 else 0
                eta = (total // args.stride - written) / max(rate, 1e-3)
                print(f"  written {written}/{total // args.stride} "
                      f"frames  ({rate:.1f} fps  ETA {eta/60:.1f} min)")
        frame_idx += 1

    capL.release(); capR.release()
    proc.stdin.close()
    proc.wait()
    sz = out_path.stat().st_size / 1e6 if out_path.exists() else 0.0
    print(f"\ndone. {written} frames written. {sz:.1f} MB at "
          f"{args.out_path}")


if __name__ == "__main__":
    main()
