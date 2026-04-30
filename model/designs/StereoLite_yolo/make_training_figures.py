"""Generate figures for the architecture document from Kaggle training artifacts.

Outputs:
    model/designs/StereoLite/training_curves.pdf       (4-panel loss/LR curves)
    model/designs/StereoLite/progress_grid.pdf         (same 2 pairs × 5 steps)
    model/designs/StereoLite/final_gallery.pdf         (all 6 val pairs at final step)
"""
from __future__ import annotations

import os
import csv

import cv2
import numpy as np
import matplotlib.pyplot as plt

ROOT = "/home/abrar/Research/stero_research_claude"
ART = os.path.join(ROOT, "data", "stereolite_v8_kaggle")
OUT = os.path.join(ROOT, "model", "designs", "StereoLite")


# ------------------------- training curves ------------------------------- #
def make_training_curves():
    steps, epoch, loss, l1f, l1cv, lr, ms = [], [], [], [], [], [], []
    with open(os.path.join(ART, "train_log.csv")) as fp:
        r = csv.DictReader(fp)
        for row in r:
            epoch.append(int(row["epoch"]))
            steps.append(int(row["step"]))
            loss.append(float(row["loss"]))
            l1f.append(float(row["l1_final"]))
            l1cv.append(float(row["l1_cv"]))
            lr.append(float(row["lr"]))
            ms.append(float(row["ms_per_step"]))

    steps = np.array(steps)
    loss = np.array(loss)
    l1f = np.array(l1f)
    l1cv = np.array(l1cv)
    lr = np.array(lr)

    fig, axes = plt.subplots(2, 2, figsize=(10, 5.5))

    # Multi-scale total loss
    ax = axes[0, 0]
    ax.plot(steps, loss, color="#2d6a4f", linewidth=1.6)
    ax.set_xlabel("training step")
    ax.set_ylabel("total loss")
    ax.set_title("(a) Total multi-scale loss", fontsize=10)
    ax.set_yscale("log")
    ax.grid(True, alpha=0.3, linewidth=0.5)

    # l1_final (L1 on full-res disparity)
    ax = axes[0, 1]
    ax.plot(steps, l1f, color="#c1121f", linewidth=1.6)
    ax.set_xlabel("training step")
    ax.set_ylabel(r"L1 on $\hat{d}$ (px)")
    ax.set_title("(b) L1 error on final disparity (training)", fontsize=10)
    ax.set_yscale("log")
    ax.grid(True, alpha=0.3, linewidth=0.5)

    # l1_cv (L1 on 1/8 cost-volume output)
    ax = axes[1, 0]
    ax.plot(steps, l1cv, color="#8a5a00", linewidth=1.6)
    ax.set_xlabel("training step")
    ax.set_ylabel("L1 on $d_{1/8}$ (px at 1/8 units)")
    ax.set_title("(c) L1 on 1/8 cost-volume output (training)", fontsize=10)
    ax.set_yscale("log")
    ax.grid(True, alpha=0.3, linewidth=0.5)

    # OneCycle LR schedule
    ax = axes[1, 1]
    ax.plot(steps, lr, color="#1a659e", linewidth=1.6)
    ax.set_xlabel("training step")
    ax.set_ylabel("learning rate")
    ax.set_title("(d) OneCycle LR schedule (peak $8\\!\\times\\!10^{-4}$)",
                  fontsize=10)
    ax.grid(True, alpha=0.3, linewidth=0.5)

    fig.suptitle(
        "StereoLite — Kaggle training curves   "
        "(30 epochs, T4 x2, effective batch 16, OneCycle)",
        fontsize=11, fontweight="bold", y=1.01)
    fig.tight_layout()

    pdf = os.path.join(OUT, "training_curves.pdf")
    png = os.path.join(OUT, "training_curves.png")
    fig.savefig(pdf, bbox_inches="tight")
    fig.savefig(png, dpi=170, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {pdf}")
    print(f"wrote {png}")

    # Return final-step stats for the LaTeX template
    return {
        "final_step": int(steps[-1]),
        "final_loss": float(loss[-1]),
        "final_l1f": float(l1f[-1]),
        "final_l1cv": float(l1cv[-1]),
    }


# ------------------------- sample panel composition ---------------------- #
ANNOT_H = 22  # header height on each tile (from training panels)


def _split_panel(panel_img):
    """A saved panel is N rows of (L, GT, pred). Split into list of tuples."""
    h, w, _ = panel_img.shape
    n_pairs = 6
    row_h = h // n_pairs
    col_w = w // 3
    rows = []
    for r in range(n_pairs):
        tiles = []
        for c in range(3):
            tile = panel_img[r * row_h:(r + 1) * row_h,
                             c * col_w:(c + 1) * col_w]
            tiles.append(tile)
        rows.append(tiles)
    return rows


def _strip_annot(tile):
    """Remove the black annotation header so the final figure is clean."""
    return tile[ANNOT_H:]


def _add_disparity_colorbar(fig, axes_list, lo=0, hi=120):
    """Attach a horizontal colorbar to the right of the disparity column,
    so readers can see this is disparity in PIXELS, not depth."""
    import matplotlib.cm as cm
    from matplotlib.colors import Normalize
    from matplotlib.cm import ScalarMappable
    # Use a TURBO mapping consistent with what we used to render the panels.
    sm = ScalarMappable(norm=Normalize(vmin=lo, vmax=hi),
                         cmap=cm.get_cmap("turbo"))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=axes_list, orientation="horizontal",
                         fraction=0.012, pad=0.02, aspect=50)
    cbar.set_label(
        "disparity $\\hat{d}$ (pixels)   "
        "— red = high disparity (close), blue = low (far)",
        fontsize=8.5)
    cbar.ax.tick_params(labelsize=7.5)


def _load_panel(step):
    return cv2.imread(os.path.join(ART, "samples", f"step_{step:06d}.png"))


def _tag(ax, text, color="black"):
    ax.text(0.02, 0.95, text, color=color, transform=ax.transAxes,
            fontsize=7.5, va="top", ha="left",
            bbox=dict(facecolor="white", edgecolor="none",
                       alpha=0.85, pad=1.5))


def make_progress_grid():
    """Two scenes (pair 0 and pair 3) across five training checkpoints."""
    steps_to_show = [500, 1500, 3000, 5000, 7500]
    pairs_to_show = [0, 3]

    fig, axes = plt.subplots(
        len(pairs_to_show) * 3, len(steps_to_show),
        figsize=(14, 2.0 * len(pairs_to_show) * 3),
        gridspec_kw=dict(wspace=0.03, hspace=0.08))

    for col, step in enumerate(steps_to_show):
        panel = _load_panel(step)
        rows = _split_panel(panel)
        for pr_i, pr in enumerate(pairs_to_show):
            L, GT, P = (cv2.cvtColor(_strip_annot(t), cv2.COLOR_BGR2RGB)
                        for t in rows[pr])
            axes[pr_i * 3 + 0, col].imshow(L)
            axes[pr_i * 3 + 1, col].imshow(GT)
            axes[pr_i * 3 + 2, col].imshow(P)

    # Hide axes + add column titles (step) + row labels (L / GT / pred)
    for col, step in enumerate(steps_to_show):
        axes[0, col].set_title(f"step {step}", fontsize=10, fontweight="bold")
    row_labels = []
    for pr in pairs_to_show:
        row_labels.extend([f"pair {pr}\nleft", "GT disparity",
                            f"predicted $\\hat{{d}}$"])
    for r, lbl in enumerate(row_labels):
        for c in range(len(steps_to_show)):
            axes[r, c].set_xticks([]); axes[r, c].set_yticks([])
        axes[r, 0].set_ylabel(lbl, fontsize=8.5, rotation=0, ha="right",
                               va="center", labelpad=22)

    fig.suptitle(
        "StereoLite — training progression on two held-out val pairs   "
        "(rows 2-3 and 5-6 are DISPARITY in pixels, NOT depth)",
        fontsize=10, fontweight="bold", y=0.995)
    # Attach colorbar across the bottom of all disparity rows
    cbar_axes = [axes[r, c] for r in range(len(axes))
                 for c in range(axes.shape[1])
                 if (r % 3) > 0]  # all rows except L
    _add_disparity_colorbar(fig, cbar_axes, lo=0, hi=120)
    pdf = os.path.join(OUT, "progress_grid.pdf")
    png = os.path.join(OUT, "progress_grid.png")
    fig.savefig(pdf, bbox_inches="tight")
    fig.savefig(png, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {pdf}")
    print(f"wrote {png}")


def make_final_gallery():
    """All six tracked val pairs at the final checkpoint (step 7500)."""
    panel = _load_panel(7500)
    rows = _split_panel(panel)

    fig, axes = plt.subplots(6, 3, figsize=(11, 16),
                              gridspec_kw=dict(wspace=0.03, hspace=0.08))
    col_titles = ["left image  $I_L$", "ground truth disparity",
                   r"predicted $\hat{d}$"]
    for c, title in enumerate(col_titles):
        axes[0, c].set_title(title, fontsize=10, fontweight="bold")

    for r, (L, GT, P) in enumerate(rows):
        for c, tile in enumerate((L, GT, P)):
            img = cv2.cvtColor(_strip_annot(tile), cv2.COLOR_BGR2RGB)
            axes[r, c].imshow(img)
            axes[r, c].set_xticks([]); axes[r, c].set_yticks([])
        axes[r, 0].set_ylabel(f"pair {r}", fontsize=9, rotation=0,
                               ha="right", va="center", labelpad=12)

    fig.suptitle("StereoLite — all six tracked val pairs at step 7500 "
                 "(end of training)   |   VAL EPE 1.54 px, bad1 25.2%   "
                 "(GT and prediction columns are DISPARITY, not depth)",
                 fontsize=10, fontweight="bold", y=0.995)
    # Attach colorbar to GT and pred columns
    cbar_axes = [axes[r, c] for r in range(axes.shape[0])
                 for c in (1, 2)]
    _add_disparity_colorbar(fig, cbar_axes, lo=0, hi=120)
    pdf = os.path.join(OUT, "final_gallery.pdf")
    png = os.path.join(OUT, "final_gallery.png")
    fig.savefig(pdf, bbox_inches="tight")
    fig.savefig(png, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {pdf}")
    print(f"wrote {png}")


if __name__ == "__main__":
    stats = make_training_curves()
    print(stats)
    make_progress_grid()
    make_final_gallery()
