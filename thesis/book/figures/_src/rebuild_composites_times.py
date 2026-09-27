"""Rebuild thesis-owned composite figures with Times New Roman labels.

The original experiment folders are no longer all available locally. The
unchanged photographic and model-output panels were therefore extracted
losslessly from the existing vector PDFs and are stored under ``extracted``.
Only the surrounding typography and layout are rebuilt here.
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from PIL import Image

ROOT = Path(__file__).resolve().parents[4]
SRC = Path(__file__).resolve().parent / "extracted"
OUT = ROOT / "thesis/book/figures"

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "TeX Gyre Termes"],
    "font.size": 9,
    "axes.titlesize": 9,
})


def load(folder, index):
    return Image.open(SRC / folder / f"img-{index:03d}.png").convert("RGB")


def save(fig, stem, dpi=300):
    fig.savefig(OUT / f"{stem}.pdf", bbox_inches="tight")
    fig.savefig(OUT / f"{stem}.png", dpi=dpi, bbox_inches="tight",
                facecolor="white")
    plt.close(fig)


def convergence():
    fig, axes = plt.subplots(2, 6, figsize=(6.3, 2.05))
    titles = ["1k", "5k", "15k", "30k", "53k", "Ground truth"]
    for i, ax in enumerate(axes.flat):
        ax.imshow(load("fig_4_2", i))
        ax.axis("off")
        if i < 6:
            ax.set_title(titles[i], fontsize=10,
                         fontweight="bold" if i == 5 else "normal")
    fig.tight_layout(pad=0.12, w_pad=0.08, h_pad=0.08)
    save(fig, "fig_4_2_convergence")


def sceneflow_qualitative():
    fig, axes = plt.subplots(3, 4, figsize=(6.3, 3.25))
    titles = ["Left image", "Ground truth", "Prediction", "Absolute error"]
    for i, ax in enumerate(axes.flat):
        ax.imshow(load("fig_4_3_sf", i))
        ax.axis("off")
        if i < 4:
            ax.set_title(titles[i], fontsize=10)
    sm = ScalarMappable(norm=Normalize(0, 3), cmap="viridis")
    cbar = fig.colorbar(sm, ax=axes[:, -1], fraction=0.045, pad=0.02)
    cbar.set_label("Absolute error (px)", fontsize=8)
    cbar.ax.tick_params(labelsize=8)
    fig.subplots_adjust(left=0.01, right=0.93, top=0.92, bottom=0.02,
                        wspace=0.04, hspace=0.04)
    save(fig, "fig_4_3_sceneflow_qualitative")


def middlebury_qualitative():
    names = [
        "Adirondack\nD1 2.4%", "Recycle\nD1 2.9%", "Storage\nD1 4.2%",
        "Flowers\nD1 30.3%", "Jadeplant\nD1 34.9%",
    ]
    fig, axes = plt.subplots(5, 3, figsize=(6.3, 5.25))
    titles = ["Left image", "Ground truth", "Prediction"]
    for i, ax in enumerate(axes.flat):
        ax.imshow(load("fig_4_5", i))
        ax.axis("off")
        if i < 3:
            ax.set_title(titles[i], fontsize=9)
    for r, name in enumerate(names):
        axes[r, 0].text(-0.04, 0.5, name, transform=axes[r, 0].transAxes,
                        ha="right", va="center", fontsize=8.5)
    fig.subplots_adjust(left=0.18, right=0.995, top=0.94, bottom=0.01,
                        wspace=0.03, hspace=0.06)
    save(fig, "fig_4_5_mb14_qualitative")


def augmentation():
    titles = [
        "Native co-located crop\n(384 × 640)",
        "Asymmetric colour jitter\n(b/c/s in [0.6, 1.4])",
        "Right-only random erase\n(1–2 boxes, 50–100 px)",
        "Random scale and x-stretch\n(disparity rescaled)",
    ]
    fig, axes = plt.subplots(2, 4, figsize=(6.3, 2.55))
    for i, ax in enumerate(axes.flat):
        ax.imshow(load("fig_4_3_aug", i))
        ax.axis("off")
        if i < 4:
            ax.set_title(titles[i], fontsize=9.5)
    axes[0, 0].text(-0.08, 0.5, "Left", transform=axes[0, 0].transAxes,
                    rotation=90, ha="center", va="center", fontsize=10)
    axes[1, 0].text(-0.08, 0.5, "Right", transform=axes[1, 0].transAxes,
                    rotation=90, ha="center", va="center", fontsize=10)
    fig.suptitle("Training augmentations applied to a native co-located crop",
                 fontsize=11, fontweight="bold", y=0.995)
    fig.subplots_adjust(left=0.04, right=0.995, top=0.79, bottom=0.01,
                        wspace=0.03, hspace=0.03)
    save(fig, "fig_4_3_augmentation")


def input_protocol():
    fig, axes = plt.subplots(1, 2, figsize=(6.3, 2.2),
                             gridspec_kw={"width_ratios": [1.5, 1.0]})
    titles = [
        "Native 960 × 540 with random 384 × 640 crops\n"
        "(disparity retains its native scale)",
        "Global downscale to 640 × 384\n(disparity shrinks by two thirds)",
    ]
    for i, ax in enumerate(axes):
        ax.imshow(load("fig_3_10", i))
        ax.axis("off")
        ax.set_title(titles[i], fontsize=10)
    fig.tight_layout(pad=0.2)
    save(fig, "fig_3_10_input_protocol")


def reconstruction():
    fig, axes = plt.subplots(3, 4, figsize=(6.3, 4.55))
    titles = ["Left image", "Camera view", "Oblique view", "Top-down view"]
    for i, ax in enumerate(axes.flat):
        ax.imshow(load("fig_4_10", i))
        ax.axis("off")
        if i < 4:
            ax.set_title(titles[i], fontsize=9)
    fig.tight_layout(pad=0.12, w_pad=0.08, h_pad=0.08)
    save(fig, "fig_4_10_reconstruction")


def progression():
    fig = plt.figure(figsize=(6.3, 2.65))
    outer = fig.add_gridspec(2, 1, height_ratios=[1.0, 1.05], hspace=0.28)
    top = outer[0].subgridspec(1, 5, width_ratios=[1, 1, 1, 0.05, 2.0])
    top_titles = ["Left image", "Right image", "GEV fail-soft gate w"]
    for i in range(3):
        ax = fig.add_subplot(top[0, i])
        ax.imshow(load("fig_3_9", i))
        ax.axis("off")
        ax.set_title(top_titles[i], fontsize=8.5)
    note = fig.add_subplot(top[0, 4])
    note.axis("off")
    note.text(0, 0.95,
              "The pipeline starts from a coarse estimate and sharpens it "
              "scale by scale.\n\nAt quarter resolution, the independent "
              "GEV proposal is blended in only where the gate opens:\n\n"
              "w → 1: trust the GEV proposal\n"
              "w → 0: keep the recurrent estimate",
              ha="left", va="top", fontsize=8.2, wrap=True)
    bottom = outer[1].subgridspec(1, 6, width_ratios=[1, 1, 1, 1, 1, 0.06],
                                  wspace=0.08)
    titles = ["Init. at 1/16", "Refined at 1/8", "GEV proposal at 1/4",
              "Fused and refined at 1/4", "Full resolution"]
    panel_indices = [3, 4, 5, 7, 8]
    for i, panel_index in enumerate(panel_indices):
        ax = fig.add_subplot(bottom[0, i])
        ax.imshow(load("fig_3_9", panel_index))
        ax.axis("off")
        ax.set_title(titles[i], fontsize=8)
    cax = fig.add_subplot(bottom[0, 5])
    cbar = fig.colorbar(ScalarMappable(norm=Normalize(5, 50), cmap="turbo"),
                        cax=cax)
    cbar.set_label("Disparity (px)", fontsize=7)
    cbar.ax.tick_params(labelsize=7)
    fig.subplots_adjust(left=0.01, right=0.995, top=0.94, bottom=0.01)
    save(fig, "fig_3_9_progression")


def lidar_compare():
    full = Image.open(SRC / "fig_6_lidar" / "full.png").convert("RGB")
    w, h = full.size
    # Preserve the original vector-rendered point cloud but exclude its old
    # labels; the right panel is the extracted stereo reconstruction.
    lidar = full.crop((int(0.02*w), int(0.14*h), int(0.48*w), int(0.82*h)))
    stereo = load("fig_6_lidar", 0)
    fig, axes = plt.subplots(1, 2, figsize=(6.3, 2.7))
    for ax, image, title in zip(
        axes, [lidar, stereo],
        ["(a) LiDAR point cloud (KITTI)", "(b) Stereo point cloud (ours)"],
    ):
        ax.imshow(image)
        ax.axis("off")
        ax.set_title(title, fontsize=10)
    axes[0].text(0.5, -0.04,
                 "Accurate but sparse: 42,312 points over a 40 m scene",
                 transform=axes[0].transAxes, ha="center", va="top", fontsize=8)
    axes[1].text(0.5, -0.04,
                 "Dense surface over a 2 m indoor scene",
                 transform=axes[1].transAxes, ha="center", va="top", fontsize=8)
    fig.subplots_adjust(left=0.01, right=0.99, top=0.91, bottom=0.13,
                        wspace=0.05)
    save(fig, "fig_6_lidar_compare")


if __name__ == "__main__":
    convergence()
    sceneflow_qualitative()
    middlebury_qualitative()
    augmentation()
    input_protocol()
    reconstruction()
    progression()
    lidar_compare()
