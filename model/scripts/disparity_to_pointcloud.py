"""Convert disparity .npy + left RGB → coloured 3D point cloud (.ply).

The pinhole stereo geometry:
    Z = f * B / d                              # depth in metres
    X = (u - cx) * Z / f                       # rightward
    Y = (v - cy) * Z / f                       # downward
where f = focal length in pixels, B = stereo baseline in metres,
(cx, cy) = principal point. We assume the principal point is the image
centre and that the disparity / image are at the SAME resolution.

Defaults match the Waveshare AR0144 stereo USB camera (per-eye 1280×720,
horizontal FOV 65°, baseline 52 mm). The horizontal focal length in
pixels is fx = (W/2) / tan(HFOV/2) = 640 / tan(32.5°) ≈ 1005 px. We use
the horizontal focal length because disparity is a horizontal-pixel
shift, so it is the relevant scale for Z = f·B/d. Override via flags if
your camera differs.

Outputs:
    <out>.ply       3D coloured point cloud, openable in MeshLab,
                    CloudCompare, Open3D viewer, Blender, etc.
    <out>.png       (optional, --render) a 2D screenshot of the cloud
                    rendered from a fixed viewpoint via Open3D.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np


def disparity_to_points(L_rgb: np.ndarray, disp: np.ndarray, f_px: float,
                         baseline_m: float, cx: float | None = None,
                         cy: float | None = None,
                         min_disp: float = 1.0,
                         max_depth_m: float = 50.0,
                         stride: int = 1):
    """Convert (H, W, 3) RGB + (H, W) disparity → (N, 3) XYZ + (N, 3) RGB."""
    H, W = disp.shape
    if cx is None:
        cx = W / 2.0
    if cy is None:
        cy = H / 2.0
    if stride > 1:
        L_rgb = L_rgb[::stride, ::stride]
        disp = disp[::stride, ::stride]
        H, W = disp.shape
        cx /= stride
        cy /= stride
        # Note: f_px does NOT change with stride — the same focal length
        # describes the camera regardless of how we subsample the grid.

    valid = (disp > min_disp) & np.isfinite(disp)
    Z = np.zeros_like(disp, dtype=np.float32)
    Z[valid] = f_px * baseline_m / disp[valid]
    valid &= (Z > 0) & (Z < max_depth_m)

    yy, xx = np.meshgrid(np.arange(H), np.arange(W), indexing="ij")
    X = (xx - cx) * Z / f_px
    Y = (yy - cy) * Z / f_px

    pts = np.stack([X[valid], Y[valid], Z[valid]], axis=1).astype(np.float32)
    cols = L_rgb[valid].astype(np.float32) / 255.0  # 0-1 RGB
    return pts, cols


def write_ply(path: Path, pts: np.ndarray, cols: np.ndarray):
    """Plain-text PLY writer with per-vertex RGB."""
    n = len(pts)
    rgb_u8 = (np.clip(cols, 0, 1) * 255).astype(np.uint8)
    with open(path, "wb") as fp:
        header = (
            "ply\n"
            "format binary_little_endian 1.0\n"
            f"element vertex {n}\n"
            "property float x\n"
            "property float y\n"
            "property float z\n"
            "property uchar red\n"
            "property uchar green\n"
            "property uchar blue\n"
            "end_header\n"
        )
        fp.write(header.encode("ascii"))
        # Pack as struct array for speed
        dtype = np.dtype([("x", "f4"), ("y", "f4"), ("z", "f4"),
                           ("r", "u1"), ("g", "u1"), ("b", "u1")])
        rec = np.zeros(n, dtype=dtype)
        rec["x"] = pts[:, 0]
        rec["y"] = pts[:, 1]
        rec["z"] = pts[:, 2]
        rec["r"] = rgb_u8[:, 0]
        rec["g"] = rgb_u8[:, 1]
        rec["b"] = rgb_u8[:, 2]
        fp.write(rec.tobytes())


def render_point_cloud(pts: np.ndarray, cols: np.ndarray, out_png: Path,
                        width: int = 1200, height: int = 900):
    """Render the cloud as a PNG, headless. Two views (front, oblique).

    Uses matplotlib's 3D scatter so it works without an OpenGL context.
    Slower than Open3D but reliable on Wayland / SSH / containers.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

    # Subsample for plotting if huge
    if len(pts) > 80_000:
        idx = np.random.default_rng(0).choice(
            len(pts), size=80_000, replace=False)
        pts = pts[idx]
        cols = cols[idx]

    fig = plt.figure(figsize=(width / 100, height / 100), dpi=100,
                      facecolor="#0a0a0c")
    # Three viewpoints: front (camera view), top-down, oblique
    views = [
        (10, -90, "near-front (camera view)"),
        (60, -90, "top-down (looking down on scene)"),
        (20, -55, "oblique"),
    ]
    # Mapping: matplotlib's (X, Y, Z) ← (world X, world Z, -world Y)
    # so vertical axis = up in the world, horizontal = scene depth
    px = pts[:, 0]
    py = pts[:, 2]
    pz = -pts[:, 1]
    z_lo, z_hi = np.percentile(pts[:, 2], [1, 99])
    y_lo, y_hi = np.percentile(-pts[:, 1], [1, 99])
    x_lo, x_hi = np.percentile(pts[:, 0], [1, 99])

    for i, (elev, azim, title) in enumerate(views):
        ax = fig.add_subplot(1, 3, i + 1, projection="3d",
                              facecolor="#0a0a0c")
        ax.scatter(px, py, pz, c=cols, s=2.5, marker=".", linewidths=0,
                   alpha=0.95)
        ax.view_init(elev=elev, azim=azim)
        ax.set_title(title, color="#ddd", fontsize=10, pad=8)
        # Hide the 3D axis chrome for a cleaner cloud-only look
        ax.set_xlim(x_lo, x_hi)
        ax.set_ylim(z_lo, z_hi)
        ax.set_zlim(y_lo, y_hi)
        ax.set_axis_off()
        # Equal-aspect-ish proportional box
        ax.set_box_aspect((x_hi - x_lo, z_hi - z_lo, y_hi - y_lo))
    fig.suptitle(
        f"point cloud  ({len(pts):,} points  |  depth range "
        f"{z_lo:.1f}–{z_hi:.1f} m)",
        color="#fff", fontsize=11, y=0.97)
    fig.tight_layout()
    fig.savefig(str(out_png), dpi=110, facecolor="#0a0a0c",
                bbox_inches="tight", pad_inches=0.1)
    plt.close(fig)


def process_pair(left_path: Path, disp_path: Path, out_dir: Path,
                  f_px: float, baseline_m: float, stride: int,
                  render: bool, max_depth: float):
    L = cv2.imread(str(left_path))
    if L is None:
        raise FileNotFoundError(left_path)
    L_rgb = cv2.cvtColor(L, cv2.COLOR_BGR2RGB)
    disp = np.load(disp_path).astype(np.float32)
    if disp.shape[:2] != L_rgb.shape[:2]:
        # Resize disp to match image (rescale by width ratio if needed)
        sx = L_rgb.shape[1] / disp.shape[1]
        disp = cv2.resize(disp, (L_rgb.shape[1], L_rgb.shape[0]),
                            interpolation=cv2.INTER_LINEAR) * sx
    pts, cols = disparity_to_points(
        L_rgb, disp, f_px=f_px, baseline_m=baseline_m,
        stride=stride, max_depth_m=max_depth)
    base = left_path.stem
    out_dir.mkdir(parents=True, exist_ok=True)
    ply_path = out_dir / f"{base}.ply"
    write_ply(ply_path, pts, cols)
    print(f"  -> {ply_path}  ({len(pts):,} points, "
          f"{ply_path.stat().st_size/1e6:.1f} MB)")
    if render:
        png_path = out_dir / f"{base}.png"
        render_point_cloud(pts, cols, png_path)
        print(f"  -> {png_path}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--pairs_dir", required=True,
                   help="dir with left/ and disp_pseudo/ subfolders")
    p.add_argument("--out_dir", default=None,
                   help="default: <pairs_dir>/point_clouds/")
    p.add_argument("--basenames", nargs="*", default=None,
                   help="specific basenames to convert (e.g. 00500 01000). "
                        "Default: 8 evenly-spaced from clean_pairs.txt")
    p.add_argument("--n_default", type=int, default=8,
                   help="if --basenames not given, render N evenly-spaced")
    p.add_argument("--focal_px", type=float, default=1005.0,
                   help="horizontal focal length in pixels at 1280-wide. "
                        "Default = AR0144 stereo (HFOV 65°, fx ≈ 1005 px). "
                        "If your image is at a different width W, pass "
                        "fx_actual = 1005 * (W / 1280).")
    p.add_argument("--baseline_m", type=float, default=0.052,
                   help="stereo baseline in metres "
                        "(default = AR0144 stereo USB camera = 52 mm)")
    p.add_argument("--stride", type=int, default=2,
                   help="sample every Nth pixel (2 -> ~25% of points)")
    p.add_argument("--max_depth", type=float, default=20.0,
                   help="discard points farther than this many metres")
    p.add_argument("--render", action="store_true",
                   help="also save a PNG screenshot via Open3D")
    args = p.parse_args()

    pairs_dir = Path(args.pairs_dir)
    out_dir = Path(args.out_dir) if args.out_dir else pairs_dir / "point_clouds"

    # Pick basenames
    if args.basenames:
        bases = args.basenames
    else:
        clean_list = pairs_dir / "clean_pairs.txt"
        if clean_list.exists():
            all_bases = [l.strip() for l in clean_list.read_text().splitlines()
                          if l.strip()]
        else:
            all_bases = sorted(p.stem for p in
                                (pairs_dir / "left").glob("*.png"))
        if len(all_bases) > args.n_default:
            step = len(all_bases) / args.n_default
            bases = [all_bases[int(i * step)] for i in range(args.n_default)]
        else:
            bases = all_bases
    print(f"converting {len(bases)} pairs ...")

    for b in bases:
        L = pairs_dir / "left" / f"{b}.png"
        D = pairs_dir / "disp_pseudo" / f"{b}.npy"
        if not (L.exists() and D.exists()):
            print(f"  skip {b}: missing files")
            continue
        try:
            process_pair(L, D, out_dir, f_px=args.focal_px,
                          baseline_m=args.baseline_m, stride=args.stride,
                          render=args.render, max_depth=args.max_depth)
        except Exception as e:
            print(f"  FAIL {b}: {type(e).__name__}: {e}")

    print(f"\ndone. point clouds in {out_dir}")


if __name__ == "__main__":
    main()
