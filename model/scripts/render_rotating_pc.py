"""Render rotating / panning / zooming GIFs of the model's predicted point
cloud on the top-EPE validation pairs, using Open3D's EGL-headless
OffscreenRenderer (Filament backend).

Per pair:
  1. Resize L, R to (512, 832) and run the fine-tuned StereoLite forward.
  2. Upsample disparity back to native (720, 1280) and rescale by sx.
  3. Pinhole-project to colored 3D points (AR0144 intrinsics).
  4. Open3D post-processing:
       - statistical_outlier_removal (drop depth-edge fliers)
       - voxel_down_sample (uniform density, splats look dense)
       - estimate_normals + orient toward camera (for lit shading)
  5. Render with OffscreenRenderer:
       - phase 1: hold opening view ~1.3 s
       - phase 2: smoothstep-eased 360° azimuth sweep ~4.5 s
       - phase 3: hold closing view ~1 s
  6. Stitch the PNG frames into a GIF.
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np
import open3d as o3d
import open3d.visualization.rendering as o3dr
import torch
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "designs"))

from pseudo_pairs_loader import list_pairs, split_pairs  # noqa: E402
from disparity_to_pointcloud import disparity_to_points, write_ply  # noqa: E402
from d1_tile import StereoLite, StereoLiteConfig  # noqa: E402


# Top-3 best-EPE val pairs (from progress.csv at step 9000)
TOP3 = [
    ("pair_01", "00038", 0.2579),
    ("pair_00", "01282", 0.2966),
    ("pair_07", "01077", 0.3154),
]


def predict_disparity(model, L_path: Path, R_path: Path, device,
                      inf_h: int = 512, inf_w: int = 832):
    L_native = cv2.imread(str(L_path))
    R_native = cv2.imread(str(R_path))
    H_n, W_n = L_native.shape[:2]
    L = cv2.resize(L_native, (inf_w, inf_h), interpolation=cv2.INTER_AREA)
    R = cv2.resize(R_native, (inf_w, inf_h), interpolation=cv2.INTER_AREA)
    Lt = torch.from_numpy(cv2.cvtColor(L, cv2.COLOR_BGR2RGB)).float() \
              .permute(2, 0, 1).unsqueeze(0).to(device)
    Rt = torch.from_numpy(cv2.cvtColor(R, cv2.COLOR_BGR2RGB)).float() \
              .permute(2, 0, 1).unsqueeze(0).to(device)
    with torch.no_grad():
        d_pred = model(Lt, Rt).squeeze().cpu().numpy().astype(np.float32)
    sx = W_n / inf_w
    d_native = cv2.resize(d_pred, (W_n, H_n),
                           interpolation=cv2.INTER_LINEAR) * sx
    L_rgb_native = cv2.cvtColor(L_native, cv2.COLOR_BGR2RGB)
    return d_native, L_rgb_native


def build_clean_cloud(pts: np.ndarray, cols: np.ndarray,
                       voxel: float = 0.004,
                       outlier_nb: int = 30,
                       outlier_std: float = 2.0) -> o3d.geometry.PointCloud:
    """Open3D-ify the numpy cloud and post-process for a denser, cleaner look."""
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts.astype(np.float64))
    pcd.colors = o3d.utility.Vector3dVector(cols.astype(np.float64))
    print(f"    raw points: {len(pcd.points):,}")

    pcd, _ = pcd.remove_statistical_outlier(
        nb_neighbors=outlier_nb, std_ratio=outlier_std)
    print(f"    post-outlier: {len(pcd.points):,}")

    if voxel > 0:
        pcd = pcd.voxel_down_sample(voxel_size=voxel)
        print(f"    post-voxel({voxel*1000:.1f}mm): {len(pcd.points):,}")

    pcd.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=0.02, max_nn=30))
    pcd.orient_normals_towards_camera_location([0.0, 0.0, 0.0])
    return pcd


def smoothstep(t: float) -> float:
    """Cubic smoothstep — eases in and out so motion looks intentional."""
    t = max(0.0, min(1.0, t))
    return t * t * (3 - 2 * t)


def orbit_eye(center: np.ndarray, radius: float, azim_deg: float,
              elev_deg: float) -> np.ndarray:
    """Position the camera on a sphere around `center`. Azim 0 = +X axis,
    measured CCW from above. Elev 0 = horizontal, +90 = directly above."""
    az = np.deg2rad(azim_deg)
    el = np.deg2rad(elev_deg)
    eye = center + radius * np.array(
        [np.cos(el) * np.cos(az),     # X
         -np.sin(el),                 # Y (image-down convention)
         np.cos(el) * np.sin(az)],    # Z (depth)
        dtype=np.float64)
    return eye.astype(np.float32)


def render_rotation_gif(pcd: o3d.geometry.PointCloud, out_gif: Path,
                         out_mp4: Path | None,
                         title: str,
                         W: int = 800, H: int = 680,
                         fps: int = 18,
                         hold_start_frames: int = 24,
                         rot_frames: int = 90,
                         hold_end_frames: int = 16,
                         elev_base: float = 12.0,
                         elev_amp: float = 6.0,
                         point_size: float = 3.5,
                         radius_scale: float = 1.45):
    """Render an azim sweep with hold-start, eased rotation, hold-end.

    Default: 24 + 90 + 16 = 130 frames at 18 fps ≈ 7.2 s.
    Always writes a GIF; if `out_mp4` is given and ffmpeg is on PATH,
    also writes a much-smaller H.264 MP4 (PowerPoint-friendly).
    """
    pts_np = np.asarray(pcd.points)
    center = pts_np.mean(axis=0).astype(np.float32)
    extent = np.linalg.norm(pts_np.max(axis=0) - pts_np.min(axis=0))
    radius = float(extent) * radius_scale

    ren = o3dr.OffscreenRenderer(W, H)
    ren.scene.set_background([0.04, 0.045, 0.055, 1.0])
    # We have vertex colors AND normals — use a lit shader for a soft splat
    mat = o3dr.MaterialRecord()
    mat.shader = "defaultLit"
    mat.point_size = point_size
    mat.base_color = [1.0, 1.0, 1.0, 1.0]
    ren.scene.add_geometry("cloud", pcd, mat)

    # Soft scene lighting — keep the shadow off (looks cleaner on tiny dots)
    ren.scene.set_lighting(
        o3dr.Open3DScene.LightingProfile.MED_SHADOWS,
        np.array([0.5, -0.7, 0.5], dtype=np.float32))
    ren.scene.scene.enable_sun_light(True)
    ren.scene.show_axes(False)

    up = np.array([0.0, -1.0, 0.0], dtype=np.float32)   # image Y is down

    # Phase 1: hold the opening view. The camera sits at world origin
    # looking toward +Z, so the eye must be at -Z relative to the cloud
    # centre to reproduce the camera's view. orbit_eye places the eye
    # at  z = radius * cos(el) * sin(az), so az = -90° → eye at -Z.
    azim_start = -90.0
    azim_end   = azim_start + 360.0

    frames: list[np.ndarray] = []
    for i in range(hold_start_frames):
        eye = orbit_eye(center, radius, azim_start, elev_base)
        ren.setup_camera(45.0, center, eye, up)
        frames.append(np.asarray(ren.render_to_image()))

    # Phase 2: eased 360° rotation, with mild elevation bob
    for i in range(rot_frames):
        t = i / max(rot_frames - 1, 1)
        s = smoothstep(t)
        az = azim_start + 360.0 * s
        el = elev_base + elev_amp * np.sin(2 * np.pi * t)
        eye = orbit_eye(center, radius, az, el)
        ren.setup_camera(45.0, center, eye, up)
        frames.append(np.asarray(ren.render_to_image()))

    # Phase 3: hold the closing view (back at start azimuth)
    for i in range(hold_end_frames):
        eye = orbit_eye(center, radius, azim_end, elev_base)
        ren.setup_camera(45.0, center, eye, up)
        frames.append(np.asarray(ren.render_to_image()))

    # Burn the title into every frame so the slide reader has context.
    title_h = 32
    try:
        font = ImageFont.truetype(
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 16)
    except OSError:
        font = ImageFont.load_default()

    out_gif.parent.mkdir(parents=True, exist_ok=True)

    # Write PNG frames to a temp dir for ffmpeg, also collect PIL frames
    # for the GIF writer. Two outputs from a single render pass.
    tmp = Path(tempfile.mkdtemp(prefix="o3drender_"))
    pil_frames: list[Image.Image] = []
    for i, f in enumerate(frames):
        img = Image.fromarray(f).convert("RGB")
        canvas = Image.new("RGB", (W, H + title_h), color=(10, 11, 14))
        canvas.paste(img, (0, title_h))
        ImageDraw.Draw(canvas).text((12, 7), title,
                                     font=font, fill=(232, 232, 240))
        pil_frames.append(canvas)
        canvas.save(str(tmp / f"f_{i:04d}.png"))

    # GIF (slide-friendly: adaptive 256-color palette via PIL)
    pil_frames[0].save(
        str(out_gif),
        save_all=True,
        append_images=pil_frames[1:],
        duration=int(1000 / fps),
        loop=0,
        optimize=True,
    )
    print(f"  -> {out_gif.name}  ({out_gif.stat().st_size/1e6:.1f} MB GIF, "
          f"{len(pil_frames)} frames @ {fps} fps = {len(pil_frames)/fps:.1f} s)")

    # MP4 via ffmpeg (much smaller than GIF, plays in PowerPoint and the web)
    if out_mp4 is not None and shutil.which("ffmpeg"):
        cmd = [
            "ffmpeg", "-y", "-loglevel", "error",
            "-framerate", str(fps),
            "-i", str(tmp / "f_%04d.png"),
            "-c:v", "libx264", "-pix_fmt", "yuv420p",
            "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2",   # H.264 needs even dims
            "-crf", "20", "-preset", "slow",
            str(out_mp4),
        ]
        subprocess.run(cmd, check=True)
        print(f"  -> {out_mp4.name}  "
              f"({out_mp4.stat().st_size/1e6:.1f} MB MP4)")

    shutil.rmtree(tmp, ignore_errors=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--pairs_dir",
                   default="/mnt/abrarssd/Datasets/stereo_samples_20260425_104147")
    p.add_argument("--ckpt",
                   default="model/checkpoints/stereolite_finetune_indoor_best.pth")
    p.add_argument("--out_dir", default=None,
                   help="default: <pairs_dir>/point_clouds_top3/")
    p.add_argument("--focal_px", type=float, default=1005.0)
    p.add_argument("--baseline_m", type=float, default=0.052)
    p.add_argument("--max_depth", type=float, default=15.0)
    # We use stride=1 here (full density). Open3D voxel_down_sample
    # then makes density uniform without losing structure.
    p.add_argument("--stride", type=int, default=1)
    p.add_argument("--voxel_mm", type=float, default=4.0,
                   help="voxel size for uniform downsampling, mm")
    p.add_argument("--point_size", type=float, default=3.5)
    p.add_argument("--inf_h", type=int, default=512)
    p.add_argument("--inf_w", type=int, default=832)
    p.add_argument("--fps", type=int, default=18)
    p.add_argument("--hold_start_frames", type=int, default=24)
    p.add_argument("--rot_frames", type=int, default=90)
    p.add_argument("--hold_end_frames", type=int, default=16)
    p.add_argument("--render_w", type=int, default=800)
    p.add_argument("--render_h", type=int, default=680)
    p.add_argument("--no_mp4", action="store_true",
                   help="skip MP4 export (default: writes both GIF and MP4)")
    args = p.parse_args()

    pairs_dir = Path(args.pairs_dir)
    out_dir = Path(args.out_dir) if args.out_dir else pairs_dir / "point_clouds_top3"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Reproduce val split with seed 0 to confirm basenames
    all_pairs = list_pairs(pairs_dir)
    _, val = split_pairs(all_pairs, n_val=50, seed=0)
    val_basenames = [Path(lp).stem for (lp, _, _) in val[:8]]
    print("val[:8] basenames:", val_basenames)

    # Load model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device.type}  ckpt={args.ckpt}")
    model = StereoLite(StereoLiteConfig(
        backbone="mobilenet", use_dav2=False)).to(device)
    ck = torch.load(args.ckpt, map_location=device, weights_only=False)
    sd = ck["model"] if "model" in ck else ck
    model.load_state_dict(sd, strict=True)
    model.eval()
    if "epe" in ck:
        print(f"  loaded checkpoint val EPE: {ck['epe']:.3f} px")

    for pair_id, base, epe in TOP3:
        L = pairs_dir / "left" / f"{base}.png"
        R = pairs_dir / "right" / f"{base}.png"
        print(f"\n[{pair_id}] {base}  EPE={epe:.3f} px")
        d_native, L_rgb_native = predict_disparity(
            model, L, R, device, inf_h=args.inf_h, inf_w=args.inf_w)
        pts, cols = disparity_to_points(
            L_rgb_native, d_native, f_px=args.focal_px,
            baseline_m=args.baseline_m, stride=args.stride,
            max_depth_m=args.max_depth)

        pcd = build_clean_cloud(pts, cols,
                                 voxel=args.voxel_mm / 1000.0)

        # Save the cleaned cloud as PLY too — it's much nicer than the raw
        ply = out_dir / f"{pair_id}_{base}_epe{epe:.3f}_clean.ply"
        o3d.io.write_point_cloud(str(ply), pcd, write_ascii=False)
        print(f"    -> {ply.name}  ({ply.stat().st_size/1e6:.1f} MB)")

        gif = out_dir / f"{pair_id}_{base}_epe{epe:.3f}.gif"
        mp4 = None if args.no_mp4 else \
            out_dir / f"{pair_id}_{base}_epe{epe:.3f}.mp4"
        title = (f"{pair_id} (basename {base})  "
                 f"|  StereoLite pred  |  EPE = {epe:.3f} px")
        render_rotation_gif(
            pcd, gif, mp4, title=title, fps=args.fps,
            W=args.render_w, H=args.render_h,
            hold_start_frames=args.hold_start_frames,
            rot_frames=args.rot_frames,
            hold_end_frames=args.hold_end_frames,
            point_size=args.point_size)

    print(f"\ndone. outputs in {out_dir}")


if __name__ == "__main__":
    main()
