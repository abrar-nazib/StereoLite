"""Fig 3.1: full StereoLite (gev4_opt_narrow_plane) architecture overview.

Designed 1:1 for a sideways (landscape) full-page thesis figure:
figsize 9.6 x 4.3 in, fonts 6.5 to 9.5 pt, printed unscaled.

Visual language follows the field's standard architecture figures
(IGEV Fig 3, RAFT-Stereo Fig 1): real image thumbnails at the ends,
3-D tensor prisms for cost volumes, color-coded blocks with a legend,
solid arrows for the main tensor flow, dashed arrows for guidance or
gating connections, crimson dots for supervised outputs.

Structure source: model/designs/StereoLite_yolo_ctx_gev4/
FINAL_MODEL_ARCHITECTURE.md sections 3 to 11, with the two trained
deltas of the _opt variant: narrow GEV band (+-16 around tile d) and
plane rendering in the upsample stage.

Thumbnails: genuine SceneFlow FT3D TEST/A/0000 t09 pair
(model/benchmarks/thesis_assets/) and the model's own step-53k
prediction for that scene (run images/val_00).
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, Rectangle
from PIL import Image

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / ".claude/skills/diagram-drawer/helpers"))
from diag_helpers import (  # noqa: E402
    C_CV, C_CV_EDGE, C_ENC, C_ENC_EDGE, C_GREY, C_GREY_EDGE, C_REF,
    C_REF_EDGE, C_SUP, arrow, block, box, cv_prism, loop_glyph, sup_dot, txt,
)

# context-stream color (lavender), distinct from encoder blue
C_CTX, C_CTX_EDGE = "#d5c6e0", "#7a5f9a"

ASSETS = ROOT / "model/benchmarks/thesis_assets"
RUN = ROOT / "model/benchmarks/20260704_fullsf_gev4onp_nc"
OUT = ROOT / "thesis/book/figures"
SCENE = "TEST_A_0000_t09"


def _disp_rgb(d: np.ndarray) -> np.ndarray:
    """Turbo-colormapped disparity, invalid (<=0) pixels dark."""
    valid = d > 0
    vmax = np.percentile(d[valid], 98) if valid.any() else 1.0
    norm = np.clip(d / max(vmax, 1e-6), 0, 1)
    rgb = plt.get_cmap("turbo")(norm)[..., :3]
    rgb[~valid] = 0.12
    return rgb


def _thumb(ax, img, x, y, w, h, label=None, label_fs=6.5):
    ax.imshow(img, extent=(x, x + w, y, y + h), zorder=4,
              interpolation="bilinear", aspect="auto")
    ax.add_patch(Rectangle((x, y), w, h, fill=False, edgecolor="#333",
                           linewidth=0.9, zorder=5))
    if label:
        txt(ax, x + w / 2, y - 0.17, label, fs=label_fs)


def main():
    left = np.array(Image.open(ASSETS / f"{SCENE}_left.png"))
    right = np.array(Image.open(ASSETS / f"{SCENE}_right.png"))
    pred = np.array(Image.open(RUN / "images/val_00/step_053000.png")
                    ).astype(np.float32) / 256.0

    fig = plt.figure(figsize=(9.6, 4.3))
    ax = fig.add_axes([0.005, 0.01, 0.99, 0.98])
    W, H = 16.8, 7.5
    ax.set_xlim(0, W); ax.set_ylim(0, H)
    ax.axis("off"); ax.set_aspect("equal")

    # ---------------- Band A: inputs, encoders, initialization ----------
    _thumb(ax, left, 0.25, 5.95, 1.5, 0.9, "Left image")
    _thumb(ax, right, 0.25, 4.55, 1.5, 0.9, "Right image")

    block(ax, 2.45, 4.95, 1.75, 1.85, color="enc",
          title="Shared feature\nencoder", title_fs=8.0,
          subtitle="YOLO26s layers 0-6\n(pretrained)\nf4 128 · f8 256 · f16 256",
          sub_fs=6.0)
    arrow(ax, 1.75, 6.4, 2.45, 6.25)
    arrow(ax, 1.75, 5.0, 2.45, 5.45)

    cv_prism(ax, 5.0, 5.6, 1.05, 0.85)
    txt(ax, 5.6, 5.35, "cost volume 1/16\n8 grp × 24 disp", fs=6.0)
    arrow(ax, 4.2, 6.0, 5.0, 6.0)
    txt(ax, 4.6, 6.15, "fL16, fR16", fs=5.8)

    block(ax, 6.7, 5.55, 1.35, 0.95, color="cv", title="3-D CNN",
          title_fs=7.5, subtitle="8-16-16-1\n+ softmax", sub_fs=6.0)
    arrow(ax, 6.25, 6.0, 6.7, 6.0)

    block(ax, 8.65, 5.55, 2.15, 0.95, color="grey",
          title="Tile-state init", title_fs=7.5,
          subtitle="T = (d, sx, sy, h, c)\nd0 soft-argmax · h ← ctx16",
          sub_fs=6.0)
    arrow(ax, 8.05, 6.0, 8.65, 6.0)

    txt(ax, W - 0.2, 7.25, "StereoLite · 2.96 M parameters",
        fs=7.5, ha="right", weight="bold")

    # ---------------- Band B: context stream + GEV branch ----------------
    box(ax, 2.45, 3.5, 1.75, 0.95, C_CTX, C_CTX_EDGE)
    txt(ax, 3.325, 4.13, "Context encoder", fs=7.5, weight="bold")
    txt(ax, 3.325, 3.78, "left image only\nGhostConv + SE → 32 ch", fs=6.0)
    # left image tap into context encoder (dashed elbow)
    arrow(ax, 1.6, 5.95, 2.45, 4.2, dashed=True, lw=0.9)

    # fL4/fR4 rail from encoder into the GEV prism (lands on top face)
    arrow(ax, 4.2, 5.3, 10.05, 4.68, dashed=True, lw=0.9)
    txt(ax, 6.8, 5.22, "fL4, fR4", fs=5.8)

    cv_prism(ax, 9.75, 3.75, 1.05, 0.85)
    txt(ax, 9.6, 4.1, "GEV 1/4\n8 grp · ±16\nband around d", fs=5.8,
        ha="right")

    block(ax, 11.55, 3.75, 1.6, 0.85, color="grey",
          title="Fail-soft fusion", title_fs=7.0,
          subtitle="w = σ(F(·)) · bias init -4", sub_fs=5.8)
    arrow(ax, 11.0, 4.17, 11.55, 4.17)

    # ---------------- Band C: recurrent coarse-to-fine ladder ------------
    block(ax, 4.6, 1.5, 1.55, 1.15, color="ref", title="ConvGRU ×2",
          title_fs=7.5, subtitle="1/16 · corr ±2\nwarp(fR), fL", sub_fs=5.8)
    loop_glyph(ax, 6.0, 2.55, r=0.10)
    block(ax, 7.25, 1.5, 1.55, 1.15, color="ref", title="ConvGRU ×3",
          title_fs=7.5, subtitle="1/8 · corr ±2\nwarp(fR), fL", sub_fs=5.8)
    loop_glyph(ax, 8.65, 2.55, r=0.10)
    block(ax, 10.7, 1.5, 1.55, 1.15, color="ref", title="ConvGRU ×3",
          title_fs=7.5, subtitle="1/4 · corr ±2\nwarp(fR), fL", sub_fs=5.8)
    loop_glyph(ax, 12.1, 2.55, r=0.10)

    # init state drops into the ladder (elbow route, clear of all blocks)
    ax.plot([9.7, 9.7, 4.4, 4.4], [5.55, 5.05, 5.05, 2.07],
            color="#333", lw=1.1, zorder=3)
    arrow(ax, 4.4, 2.07, 4.6, 2.07)
    txt(ax, 4.6, 3.3, "T$_0$", fs=6.5, ha="left")

    # plane-aware upsample arrows between GRU stages
    arrow(ax, 6.15, 2.07, 7.25, 2.07)
    txt(ax, 6.7, 1.83, "plane ↑2", fs=5.8)
    arrow(ax, 8.8, 2.07, 9.55, 2.07)
    txt(ax, 9.17, 1.83, "plane ↑2", fs=5.8)

    # fusion node between 1/8 and 1/4 stages
    ax.add_patch(Circle((9.9, 2.07), 0.2, facecolor="white",
                        edgecolor="#333", lw=1.1, zorder=5))
    txt(ax, 9.9, 2.07, "⊕", fs=9, zorder=6)
    arrow(ax, 10.1, 2.07, 10.7, 2.07)
    # gate output into fusion node
    ax.plot([12.35, 12.35, 9.9], [3.75, 3.0, 3.0], color="#333", lw=0.9,
            zorder=3, linestyle="--")
    arrow(ax, 9.9, 3.0, 9.9, 2.27, dashed=True, lw=0.9)
    txt(ax, 11.1, 3.15, "d ← d + w·(d$_{gev}$ − d)", fs=5.8)

    # narrow-band center tap: tile d into GEV
    arrow(ax, 9.3, 2.3, 9.9, 3.7, dashed=True, lw=0.8)
    txt(ax, 9.42, 2.95, "d", fs=6.0, style="italic")

    # upsample + output
    block(ax, 12.95, 1.5, 1.7, 1.15, color="grey",
          title="Plane render +\nconvex upsample", title_fs=7.0,
          subtitle="1/4 → 1/2 → full\nmasks from fL4, fL2", sub_fs=5.8)
    arrow(ax, 12.25, 2.07, 12.95, 2.07)
    _thumb(ax, _disp_rgb(pred), 15.15, 1.62, 1.45, 0.87,
           "Disparity (full res)")
    arrow(ax, 14.65, 2.07, 15.15, 2.07)

    # context rail feeding all three GRU stages
    ax.plot([3.325, 3.325, 11.47], [3.5, 1.0, 1.0], color=C_CTX_EDGE,
            lw=1.0, linestyle="--", zorder=3)
    for gx in (5.37, 8.02, 11.47):
        arrow(ax, gx, 1.0, gx, 1.5, dashed=True, lw=0.9, color=C_CTX_EDGE)
    txt(ax, 7.6, 0.82, "ctx16 · ctx8 · ctx4", fs=5.8, color=C_CTX_EDGE)

    # supervision dots on the inter-stage disparity outputs
    sup_dot(ax, 6.7, 2.32, "1/16", fs=6.0)
    sup_dot(ax, 9.17, 2.32, "1/8", fs=6.0)
    sup_dot(ax, 12.6, 2.32, "1/4", fs=6.0)
    sup_dot(ax, 10.5, 4.95, "gev", fs=6.0)
    sup_dot(ax, 14.9, 2.32, "1/2, full", fs=6.0)

    # ---------------- Legend ---------------------------------------------
    items = [
        (C_ENC, C_ENC_EDGE, "feature encoder"),
        (C_CV, C_CV_EDGE, "cost volume / 3-D conv"),
        (C_REF, C_REF_EDGE, "recurrent refinement"),
        (C_CTX, C_CTX_EDGE, "context stream"),
        (C_GREY, C_GREY_EDGE, "state / fusion / upsampling"),
    ]
    lx = 0.4
    for face, edge, lab in items:
        box(ax, lx, 0.22, 0.32, 0.26, face, edge)
        txt(ax, lx + 0.45, 0.35, lab, fs=6.2, ha="left")
        lx += 0.45 + len(lab) * 0.088 + 0.45
    ax.add_patch(Circle((lx + 0.15, 0.35), 0.05, facecolor=C_SUP,
                        edgecolor="#8a0000", lw=0.6))
    txt(ax, lx + 0.35, 0.35, "supervised output (multi-scale loss)",
        fs=6.2, ha="left")

    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / "fig_3_1_architecture.pdf", bbox_inches="tight")
    fig.savefig(OUT / "fig_3_1_architecture.png", dpi=220,
                bbox_inches="tight", facecolor="white")
    print("saved", OUT / "fig_3_1_architecture.png")


if __name__ == "__main__":
    main()
