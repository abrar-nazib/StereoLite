"""Fig 4.10: 3D reconstruction from the model's predicted disparity.

Reuses the ready-made cleaned point clouds rendered during the
presentation from the trained checkpoint's disparity on the low-cost
AR0144 rig (stereo_samples_20260425_104147/point_clouds_top3). Each
cloud is the back-projection of the model's predicted disparity through
the calibrated rig geometry, statistical-outlier filtered and voxel
downsampled. This figure loads those .ply clouds and renders three
viewpoints per scene offscreen for a print figure, paired with the left
image; no re-inference is performed.

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
from PIL import Image

ROOT = Path(__file__).resolve().parents[4]
OUT = ROOT / "thesis/book/figures"
DS = Path("/media/abrar/AbrarSSD/Datasets/stereo_samples_20260425_104147")
PC = DS / "point_clouds_top3"

# (frame id for the left image, ready cleaned cloud)
SCENES = [
    ("01282", PC / "pair_00_01282_epe0.297_clean.ply"),
    ("00038", PC / "pair_01_00038_epe0.258_clean.ply"),
    ("01077", PC / "pair_07_01077_epe0.315_clean.ply"),
]

plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "Times", "TeX Gyre Termes", "DejaVu Serif"]})


def _crop_content(img, pad=14):
    """Crop a uniform-background render to its content bounding box."""
    bg = img[2, 2].astype(int)
    mask = (np.abs(img.astype(int) - bg).max(axis=2) > 8)
    ys, xs = np.where(mask)
    if len(ys) == 0:
        return img
    y0, y1 = max(0, ys.min() - pad), min(img.shape[0], ys.max() + pad)
    x0, x1 = max(0, xs.min() - pad), min(img.shape[1], xs.max() + pad)
    return img[y0:y1, x0:x1]


def _pad_to(img, aspect=4.0 / 3.0, bg=255):
    """White-pad an image to a fixed W:H aspect so grid cells align."""
    h, w = img.shape[:2]
    W, H = w, h
    if w / h > aspect:
        H = int(round(w / aspect))
    else:
        W = int(round(h * aspect))
    out = np.full((H, W, 3), bg, dtype=img.dtype)
    y0 = (H - h) // 2
    x0 = (W - w) // 2
    out[y0:y0 + h, x0:x0 + w] = img[..., :3]
    return out


def render_offscreen(pcd, views):
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


def render_matplotlib(pcd, views):
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
    import open3d as o3d
    rows = []
    for fid, ply in SCENES:
        pcd = o3d.io.read_point_cloud(str(ply))
        print(f"scene {fid}: {len(pcd.points)} points from {ply.name}")
        L = np.array(Image.open(DS / "left" / f"{fid}.png")
                     .convert("RGB").resize((640, 384)))
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
        rows.append((fid, L, imgs))

    ncol = 4
    fig, axes = plt.subplots(len(rows), ncol,
                             figsize=(6.3, 1.55 * len(rows)))
    if len(rows) == 1:
        axes = axes[None, :]
    for r_i, (fid, L, imgs) in enumerate(rows):
        axes[r_i, 0].imshow(_pad_to(L))
        axes[r_i, 0].axis("off")
        if r_i == 0:
            axes[r_i, 0].set_title("Left image", fontsize=8)
        for c_i, (name, img) in enumerate(imgs, start=1):
            axes[r_i, c_i].imshow(_pad_to(img))
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
