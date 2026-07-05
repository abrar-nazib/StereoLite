"""Fig 4.10: 3D reconstruction from the model's predicted disparity.

Runs best.pth on indoor pairs from the project's AR0144 stereo rig,
back-projects the predicted disparity to a colored point cloud through
the pinhole relations (Z = f B / d), cleans it with Open3D, and renders
three viewpoints per scene offscreen for a print figure.

Intrinsics per model/scripts/disparity_to_pointcloud.py (verified rig
values): per-eye 1280x720, HFOV 65 deg -> fx ~= 1005 px at W=1280,
baseline 0.052 m. At the 640-wide inference resolution fx scales to
1005 * 640/1280 = 502.5 px.

Fallback: if the Filament OffscreenRenderer fails (Wayland/EGL), a
matplotlib 3D scatter path renders the same clouds.
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image

ROOT = Path("/home/abrar/Research/stero_research_claude")
sys.path.insert(0, str(ROOT / "model/scripts"))
sys.path.insert(0, str(ROOT / "model/designs"))
import os
os.chdir(ROOT)
from train_full_sceneflow import _forward_pad16, build_model  # noqa: E402

RUN = ROOT / "model/benchmarks/20260704_fullsf_gev4onp_nc"
CAM = ROOT / "data/user_cam_1"
OUT = ROOT / "thesis/book/figures"
PLY_OUT = ROOT / "model/benchmarks/thesis_reconstruction"
PAIRS = ["00000", "00025"]
H, W = 384, 640
FX = 1005.0 * (W / 1280.0)   # 502.5 px at inference width
BASELINE = 0.052             # metres
MAX_DEPTH = 4.0              # metres, indoor scene cap

plt.rcParams.update({"font.family": "DejaVu Serif"})


def predict(model, device, pid):
    L = np.array(Image.open(CAM / "left" / f"{pid}.png")
                 .convert("RGB").resize((W, H)))
    R = np.array(Image.open(CAM / "right" / f"{pid}.png")
                 .convert("RGB").resize((W, H)))
    Lt = torch.from_numpy(L).float().permute(2, 0, 1)[None].to(device) / 255
    Rt = torch.from_numpy(R).float().permute(2, 0, 1)[None].to(device) / 255
    with torch.no_grad():
        d = _forward_pad16(model, Lt, Rt)[0, 0].cpu().numpy()
    return L, d


def to_cloud(L, disp):
    cx, cy = W / 2.0, H / 2.0
    valid = (disp > 1.0) & np.isfinite(disp)
    Z = np.zeros_like(disp)
    Z[valid] = FX * BASELINE / disp[valid]
    valid &= (Z > 0.2) & (Z < MAX_DEPTH)
    yy, xx = np.meshgrid(np.arange(H), np.arange(W), indexing="ij")
    X = (xx - cx) * Z / FX
    Y = (yy - cy) * Z / FX
    pts = np.stack([X[valid], Y[valid], Z[valid]], axis=1)
    cols = L[valid].astype(np.float64) / 255.0
    return pts, cols


def clean_cloud(pts, cols):
    import open3d as o3d
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts.astype(np.float64))
    pcd.colors = o3d.utility.Vector3dVector(cols)
    pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
    pcd = pcd.voxel_down_sample(voxel_size=0.008)
    return pcd


def render_offscreen(pcd, views):
    """Render the cloud from several viewpoints; white background."""
    import open3d as o3d
    import open3d.visualization.rendering as r
    RW, RH = 900, 640
    ren = r.OffscreenRenderer(RW, RH)
    ren.scene.set_background([1.0, 1.0, 1.0, 1.0])
    mat = r.MaterialRecord()
    mat.shader = "defaultUnlit"
    mat.point_size = 4.0
    ren.scene.add_geometry("pcd", pcd, mat)
    bbox = pcd.get_axis_aligned_bounding_box()
    c = bbox.get_center()
    ext = float(np.max(bbox.get_extent()))
    imgs = []
    for name, (dx, dy, dz) in views:
        eye = c + np.array([dx, dy, dz]) * ext * 0.62
        ren.scene.camera.look_at(c, eye, [0, -1, 0])
        img = np.asarray(ren.render_to_image())
        imgs.append((name, _crop_content(img)))
    return imgs


def _crop_content(img, pad=14, thresh=None):
    """Crop a uniform-background render to its content bounding box."""
    bg = img[2, 2].astype(int)
    mask = (np.abs(img.astype(int) - bg).max(axis=2) > 8)
    ys, xs = np.where(mask)
    if len(ys) == 0:
        return img
    y0, y1 = max(0, ys.min() - pad), min(img.shape[0], ys.max() + pad)
    x0, x1 = max(0, xs.min() - pad), min(img.shape[1], xs.max() + pad)
    return img[y0:y1, x0:x1]


def render_matplotlib(pcd, views):
    """Fallback: matplotlib 3D scatter from the same viewpoints."""
    pts = np.asarray(pcd.points)
    cols = np.asarray(pcd.colors)
    sel = np.random.RandomState(0).choice(
        len(pts), size=min(60000, len(pts)), replace=False)
    imgs = []
    for name, (elev, azim) in views:
        fig = plt.figure(figsize=(4.2, 3.2))
        ax = fig.add_subplot(111, projection="3d")
        ax.scatter(pts[sel, 0], pts[sel, 2], -pts[sel, 1], s=0.4,
                   c=cols[sel], depthshade=False)
        ax.view_init(elev=elev, azim=azim)
        ax.set_axis_off()
        ax.set_box_aspect((1, 1.2, 0.8))
        fig.tight_layout(pad=0)
        fig.canvas.draw()
        img = np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()
        plt.close(fig)
        imgs.append((name, img))
    return imgs


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, _ = build_model("gev4_opt_narrow_plane")
    ck = torch.load(RUN / "best.pth", map_location=device,
                    weights_only=False)
    model.load_state_dict(ck["model"])
    model.to(device).eval()
    print(f"loaded step {ck.get('step')}")

    PLY_OUT.mkdir(parents=True, exist_ok=True)
    import open3d as o3d

    rows = []
    for pid in PAIRS:
        L, disp = predict(model, device, pid)
        pts, cols = to_cloud(L, disp)
        pcd = clean_cloud(pts, cols)
        o3d.io.write_point_cloud(str(PLY_OUT / f"cam_{pid}.ply"), pcd)
        print(f"pair {pid}: {len(pcd.points)} points after cleaning")
        try:
            views = [("camera view", (0.0, 0.0, -1.6)),
                     ("oblique", (1.1, -0.5, -1.2)),
                     ("top-down", (0.0, -1.7, -0.35))]
            imgs = render_offscreen(pcd, views)
            print("  offscreen render ok")
        except Exception as e:
            print("  offscreen failed, matplotlib fallback:", e)
            views = [("camera view", (5, -90)), ("oblique", (25, -55)),
                     ("top-down", (75, -90))]
            imgs = render_matplotlib(pcd, views)
        rows.append((pid, L, imgs))

    # assemble figure: one row per scene = [left image | 3 views]
    ncol = 4
    fig, axes = plt.subplots(len(rows), ncol,
                             figsize=(6.3, 1.55 * len(rows)))
    if len(rows) == 1:
        axes = axes[None, :]
    for r_i, (pid, L, imgs) in enumerate(rows):
        axes[r_i, 0].imshow(L)
        axes[r_i, 0].axis("off")
        if r_i == 0:
            axes[r_i, 0].set_title("Left image", fontsize=8)
        for c_i, (name, img) in enumerate(imgs, start=1):
            axes[r_i, c_i].imshow(img)
            axes[r_i, c_i].axis("off")
            if r_i == 0:
                axes[r_i, c_i].set_title(name, fontsize=8)
    fig.tight_layout(pad=0.15)
    fig.savefig(OUT / "fig_4_10_reconstruction.pdf", bbox_inches="tight")
    fig.savefig(OUT / "fig_4_10_reconstruction.png", dpi=220,
                bbox_inches="tight", facecolor="white")
    print("saved fig_4_10")


if __name__ == "__main__":
    main()
