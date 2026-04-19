"""Quick test: capture one fresh stereo pair from AR0144 and run pretrained
HITNet (Scene Flow finalpass). Also runs the existing StereoLite
checkpoint if found, so the user can compare on the same shot.

Saves per-model 3-window panel + a summary panel.
"""
from __future__ import annotations

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

INF_H, INF_W = 384, 640
BASELINE_M = 0.052
FOCAL_INF_PX = 960.0 * INF_W / 1280
DEPTH_NUM = FOCAL_INF_PX * BASELINE_M
DEPTH_MIN, DEPTH_MAX = 0.25, 5.0

OUT_DIR = os.path.join(PROJ, "model", "benchmarks", "camera_hitnet")


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
    os.makedirs(OUT_DIR, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("opening camera and capturing...")
    cap = open_cam()
    for _ in range(15):
        cap.read()
    ok, frame = cap.read()
    cap.release()
    mid = frame.shape[1] // 2
    Lb = frame[:, :mid].copy()
    Rb = frame[:, mid:].copy()
    cv2.imwrite(os.path.join(OUT_DIR, "captured_left.png"), Lb)
    cv2.imwrite(os.path.join(OUT_DIR, "captured_right.png"), Rb)
    print(f"captured shape={Lb.shape}; saved raw frames")

    Lt = to_tensor(Lb, device)
    Rt = to_tensor(Rb, device)

    # ---- HITNet pretrained (Scene Flow) ----
    from hitnet_baseline import HitnetBaseline
    print("\nloading HITNet pretrained (Scene Flow finalpass)...")
    h = HitnetBaseline(device=device)
    print(f"  HITNet: {h.n_params/1e6:.2f} M params")
    # warmup
    _ = h(Lt, Rt)
    if device.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.time()
    disp_h = h(Lt, Rt)
    if device.type == "cuda":
        torch.cuda.synchronize()
    ms_h = (time.time() - t0) * 1000
    dh_np = disp_h.squeeze().cpu().numpy()
    dh_np = np.clip(dh_np, 0, 192)
    print(f"  HITNet inference: {ms_h:.0f} ms, disp [{dh_np.min():.1f}, {dh_np.max():.1f}]")

    L_small = cv2.resize(Lb, (INF_W, INF_H))
    R_small = cv2.resize(Rb, (INF_W, INF_H))
    div = np.full((INF_H, 4, 3), (0, 255, 0), dtype=np.uint8)
    feed = annotate(np.hstack([L_small, div, R_small]),
                    f"Stereo Feed (Left | Right)  {INF_W}x{INF_H} per eye")
    h_disp_vis = annotate(colour_disp(dh_np),
                          f"HITNet Scene Flow pretrained ({h.n_params/1e6:.2f}M)   {ms_h:.0f} ms")
    h_depth_vis = annotate(colour_depth(dh_np),
                           f"Depth from HITNet  range {DEPTH_MIN}-{DEPTH_MAX}m   MiDaS INFERNO")

    # ---- Optional: also StereoLite if checkpoint exists ----
    tf_panel = None
    ckpt_path = os.path.join(PROJ, "model", "checkpoints", "student_d1.pth")
    if os.path.exists(ckpt_path):
        print("\nloading StereoLite (indoor-distilled)...")
        from d1_tile import StereoLite
        tf = StereoLite().to(device)
        ck = torch.load(ckpt_path, map_location=device, weights_only=False)
        tf.load_state_dict(ck["model"], strict=True)
        tf.train()    # GroupNorm = batch-size independent
        tn = sum(p.numel() for p in tf.parameters())
        _ = tf(Lt, Rt)
        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.time()
        with torch.no_grad():
            disp_t = tf(Lt, Rt)
        if device.type == "cuda":
            torch.cuda.synchronize()
        ms_t = (time.time() - t0) * 1000
        dt_np = disp_t.squeeze().cpu().numpy()
        print(f"  StereoLite: {ms_t:.0f} ms, disp [{dt_np.min():.1f}, {dt_np.max():.1f}]")
        tf_disp_vis = annotate(colour_disp(dt_np),
                               f"StereoLite (indoor-distilled, {tn/1e6:.2f}M)   {ms_t:.0f} ms")
        tf_depth_vis = annotate(colour_depth(dt_np),
                                f"Depth from StereoLite   MiDaS INFERNO")
        tf_panel = (tf_disp_vis, tf_depth_vis)

    # ---- Compose panels ----
    # HITNet 3-window panel
    bottom = np.hstack([h_disp_vis, h_depth_vis])
    composite = np.vstack([cv2.resize(feed, (bottom.shape[1], feed.shape[0])), bottom])
    p1 = os.path.join(OUT_DIR, "panel_hitnet.png")
    cv2.imwrite(p1, composite)
    print(f"saved {p1}")

    # If TileFM available: 4-row comparison
    if tf_panel is not None:
        tf_disp_vis, tf_depth_vis = tf_panel
        row_h = np.hstack([h_disp_vis, h_depth_vis])
        row_t = np.hstack([tf_disp_vis, tf_depth_vis])
        composite2 = np.vstack([cv2.resize(feed, (row_h.shape[1], feed.shape[0])),
                                 row_h, row_t])
        p2 = os.path.join(OUT_DIR, "panel_hitnet_vs_stereolite.png")
        cv2.imwrite(p2, composite2)
        print(f"saved {p2}")

        # Side-by-side disparity strip
        strip = np.hstack([
            annotate(L_small, "left input"),
            annotate(colour_disp(dh_np), f"HITNet  {ms_h:.0f}ms"),
            annotate(colour_disp(dt_np), f"TileFM  {ms_t:.0f}ms"),
        ])
        p3 = os.path.join(OUT_DIR, "strip_hitnet_vs_stereolite.png")
        cv2.imwrite(p3, strip)
        print(f"saved {p3}")


if __name__ == "__main__":
    main()
