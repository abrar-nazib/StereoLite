"""Render a side-by-side HITNet vs StereoLite-v8 architecture diagram.

Run:
    python3 model/designs/d1_tile/draw_arch_comparison.py

Output:
    model/designs/d1_tile/arch_comparison.png
    model/designs/d1_tile/arch_comparison.pdf
"""
from __future__ import annotations

import os

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch


# Palette — colour-blind-safe, distinct per block family
C_ENC = "#8ecae6"     # encoder (blue)
C_CV = "#ffd166"      # cost volume (yellow)
C_PROP = "#a8dadc"    # propagation / refinement (teal)
C_UP = "#c8b6ff"      # upsample (purple)
C_OUT = "#b7e4c7"     # output (green)
C_NOVEL = "#ffb3c1"   # highlight differences (pink)
C_TEXT = "#1a1a2e"


def box(ax, x, y, w, h, text, color, ec="#333", lw=1.2, fs=8.3):
    box = FancyBboxPatch((x, y), w, h,
                          boxstyle="round,pad=0.02,rounding_size=0.08",
                          linewidth=lw, edgecolor=ec, facecolor=color)
    ax.add_patch(box)
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
            fontsize=fs, color=C_TEXT, zorder=5)


def arrow(ax, x0, y0, x1, y1, color="#333", lw=1.4, style="-|>"):
    a = FancyArrowPatch((x0, y0), (x1, y1), arrowstyle=style,
                         mutation_scale=12, linewidth=lw, color=color, zorder=4)
    ax.add_patch(a)


def label(ax, x, y, text, fs=9, color=C_TEXT, ha="center", va="center",
          weight="normal"):
    ax.text(x, y, text, ha=ha, va=va, fontsize=fs, color=color, weight=weight)


def draw_hitnet(ax):
    """Left panel: HITNet (Tankovich et al. CVPR 2021)."""
    label(ax, 2.5, 12.5, "HITNet  (Tankovich et al., CVPR 2021)",
          fs=12, weight="bold")
    label(ax, 2.5, 12.05, "0.63 M params  |  multi-scale tile init + single-pass propagation",
          fs=8.5, color="#555")

    # Input
    box(ax, 0.1, 11.2, 2.0, 0.5, "left image  (3, H, W)", "#eee")
    box(ax, 2.9, 11.2, 2.0, 0.5, "right image (3, H, W)", "#eee")

    # Encoder
    box(ax, 1.0, 10.3, 3.0, 0.5, "U-Net encoder  (trained from scratch)", C_ENC)
    label(ax, 4.6, 10.55, "→ f4, f16, f64", fs=7.8, color="#444", ha="left")

    # Initialization at three scales
    box(ax, 0.2, 9.1, 1.5, 0.6, "init 1/64\nlocal CV + plane", C_CV)
    box(ax, 1.85, 9.1, 1.5, 0.6, "init 1/16\nlocal CV + plane", C_CV)
    box(ax, 3.5, 9.1, 1.5, 0.6, "init 1/4\nlocal CV + plane", C_CV)
    label(ax, 2.5, 8.75, "(d, dx, dy, conf) per tile — PLANE PARAMS ONLY",
          fs=7.5, color="#c92a2a", weight="bold")

    # Propagation at each scale
    box(ax, 0.2, 7.7, 1.5, 0.55, "propagate\n1/64 tiles", C_PROP)
    box(ax, 1.85, 7.7, 1.5, 0.55, "propagate\n1/16 tiles", C_PROP)
    box(ax, 3.5, 7.7, 1.5, 0.55, "propagate\n1/4 tiles", C_PROP)
    label(ax, 2.5, 7.35, "warp + 3x3 spatial fusion,  SINGLE PASS PER SCALE",
          fs=7.4, color="#555")

    # Cascade fusion
    box(ax, 0.3, 6.5, 4.4, 0.55,
         "cascade fuse: 1/64 seed → 1/16 init → 1/4 init",
         "#fff1c1")

    # Plane upsample
    box(ax, 0.3, 5.5, 4.4, 0.5,
         "plane-geometry upsample  1/4 → full  (d + dx·Δx + dy·Δy)", C_UP)

    # Output
    box(ax, 1.0, 4.5, 3.0, 0.55, "disparity  (1, H, W)", C_OUT)

    # Loss
    box(ax, 0.4, 3.3, 4.2, 0.85,
        "loss: L1 on d per tile + slant-plane reg\n(supervises tile params directly)",
        "#eee")

    # Arrows (simplified — show the vertical flow)
    for x in [0.95, 2.6, 4.25]:
        arrow(ax, x, 10.3, x, 9.7)
        arrow(ax, x, 9.1, x, 8.25)
    arrow(ax, 2.5, 7.7, 2.5, 7.05)
    arrow(ax, 2.5, 6.5, 2.5, 6.0)
    arrow(ax, 2.5, 5.5, 2.5, 5.05)

    # Input -> encoder
    arrow(ax, 1.1, 11.2, 2.0, 10.8)
    arrow(ax, 3.9, 11.2, 3.0, 10.8)


def draw_ours(ax):
    """Right panel: StereoLite v8 (ours)."""
    label(ax, 2.5, 12.5, "StereoLite v8  (ours)",
          fs=12, weight="bold")
    label(ax, 2.5, 12.05,
          "2.14 M params  |  single-scale init + iterative-per-scale refine + learned upsample",
          fs=8.5, color="#555")

    # Input
    box(ax, 0.1, 11.2, 2.0, 0.5, "left image  (3, H, W)", "#eee")
    box(ax, 2.9, 11.2, 2.0, 0.5, "right image (3, H, W)", "#eee")

    # Encoder — NOVEL
    box(ax, 1.0, 10.3, 3.0, 0.5,
        "MobileNetV2-100  (ImageNet-pretrained, fine-tuned)", C_NOVEL,
        ec="#c92a2a", lw=1.8)
    label(ax, 4.6, 10.55, "→ f2, f4, f8, f16", fs=7.8, color="#444", ha="left")

    # Single-scale init only
    box(ax, 1.6, 9.1, 1.8, 0.6, "init 1/16\nlocal CV + 3D agg", C_CV)
    label(ax, 2.5, 8.72,
          "(d, sx, sy, feat, conf) — PLANE + LEARNED TILE FEATURE",
          fs=7.5, color="#c92a2a", weight="bold")
    label(ax, 2.5, 8.43, "single-scale init (cheapest scale only)",
          fs=7.4, color="#555")

    # Iterative refine block at 1/16
    box(ax, 0.3, 7.7, 4.4, 0.55,
        "refine 1/16  ×2 iters  (RAFT-style recurrent update)", C_PROP,
        ec="#c92a2a", lw=1.6)

    # Upsample 16->8 (plane equation)
    box(ax, 0.3, 6.9, 4.4, 0.4,
        "plane-equation upsample  1/16 → 1/8", C_UP)

    # Iterative refine at 1/8
    box(ax, 0.3, 6.05, 4.4, 0.55,
        "refine 1/8   ×3 iters  (separate weights)", C_PROP,
        ec="#c92a2a", lw=1.6)

    # Upsample 8->4
    box(ax, 0.3, 5.25, 4.4, 0.4,
        "plane-equation upsample  1/8 → 1/4", C_UP)

    # Iterative refine at 1/4
    box(ax, 0.3, 4.4, 4.4, 0.55,
        "refine 1/4   ×3 iters  (separate weights)", C_PROP,
        ec="#c92a2a", lw=1.6)

    # Convex upsample cascade — NOVEL vs HITNet
    box(ax, 0.3, 3.6, 4.4, 0.5,
        "learned convex upsample  1/4 → 1/2 → full  (RAFT-style 9-neighbour mask)",
        C_NOVEL, ec="#c92a2a", lw=1.8)

    # Output
    box(ax, 1.0, 2.7, 3.0, 0.55, "disparity  (1, H, W)", C_OUT)

    # Multi-scale loss
    box(ax, 0.4, 1.5, 4.2, 0.85,
        "multi-scale loss: L1 + grad + bad-1 hinge at d32/d16/d8/d4/d_half/d_final\n"
        "+ edge-aware smoothness on d_final",
        "#eee")

    # Arrows
    arrow(ax, 2.5, 11.2, 2.5, 10.8)
    arrow(ax, 2.5, 10.3, 2.5, 9.7)
    arrow(ax, 2.5, 9.1, 2.5, 8.25)
    arrow(ax, 2.5, 7.7, 2.5, 7.3)
    arrow(ax, 2.5, 6.9, 2.5, 6.6)
    arrow(ax, 2.5, 6.05, 2.5, 5.65)
    arrow(ax, 2.5, 5.25, 2.5, 4.95)
    arrow(ax, 2.5, 4.4, 2.5, 4.1)
    arrow(ax, 2.5, 3.6, 2.5, 3.25)


def draw_legend(ax):
    """Bottom legend spanning both panels."""
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 1)
    ax.axis("off")
    y = 0.5
    items = [
        ("input / output", "#eee"),
        ("encoder", C_ENC),
        ("cost volume", C_CV),
        ("refinement", C_PROP),
        ("upsample", C_UP),
        ("disparity", C_OUT),
        ("distinct vs HITNet", C_NOVEL),
    ]
    w = 0.95
    x = 0.2
    for name, col in items:
        box_patch = FancyBboxPatch(
            (x, y - 0.18), 0.22, 0.36,
            boxstyle="round,pad=0.01,rounding_size=0.04",
            linewidth=0.8, edgecolor="#555", facecolor=col)
        ax.add_patch(box_patch)
        ax.text(x + 0.27, y, name, fontsize=8.5, va="center")
        x += w + 0.45


def draw_diff_panel(ax):
    """Summary of what distinguishes ours from HITNet — publication case."""
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis("off")

    label(ax, 5.0, 9.5, "Differentiators of StereoLite v8 vs HITNet",
          fs=12, weight="bold")

    rows = [
        ("aspect", "HITNet", "ours (v8)"),
        ("encoder",
         "custom U-Net, train from scratch",
         "MobileNetV2-100, ImageNet-pretrained, fine-tuned"),
        ("cost-volume stages",
         "init CV at 3 scales (1/64, 1/16, 1/4)",
         "init CV at 1 scale (1/16) only — cheaper"),
        ("tile state carried",
         "(d, dx, dy, conf)  — plane params only",
         "(d, sx, sy, feat, conf)  — + 16-ch LEARNED feature"),
        ("refinement per scale",
         "single pass (3x3 spatial fusion)",
         "recurrent iterative update (2/3/3 iters at 1/16, 1/8, 1/4)"),
        ("scale-to-scale transition",
         "cascade fuse (re-init CV at finer scale)",
         "plane-equation upsample (no re-init cost)"),
        ("final upsample",
         "plane geometry only (d + dx·Δx + dy·Δy)",
         "learned convex upsample cascade (RAFT-style, 1/4→1/2→full)"),
        ("supervision",
         "L1 on per-tile d + slanted-plane regulariser",
         "multi-scale L1 + Sobel grad + bad-1 hinge + edge-aware smooth"),
        ("params / inference",
         "0.63 M / ~20 ms (paper, KITTI)",
         "2.14 M / ~60 ms (RTX 3050, 512x832)"),
    ]

    # Table render
    x_col = [0.1, 2.0, 5.5]
    col_w = [1.85, 3.4, 4.45]
    y = 8.6
    row_h = 0.95
    for r, row in enumerate(rows):
        is_header = r == 0
        for c, txt in enumerate(row):
            xc = x_col[c]
            if is_header:
                box(ax, xc, y, col_w[c], 0.55, txt,
                    "#333", ec="#222", lw=1.0, fs=9.0)
                for patch in ax.patches[-1:]:
                    # header text colour: use explicit override
                    pass
            else:
                # highlight "ours" column
                col = C_NOVEL if c == 2 else "#f7f7f7"
                box(ax, xc, y - row_h + 0.05, col_w[c], row_h - 0.1, txt,
                    col, ec="#aaa", lw=0.8, fs=8.2)
        if is_header:
            y -= 0.65
        else:
            y -= row_h

    # Re-draw header text in white for contrast
    # (the box() helper wrote dark text; overlay white text)
    # Simpler: repaint header row using a local text call
    ax.text(x_col[0] + col_w[0] / 2, 8.88, "aspect",
            ha="center", va="center", fontsize=9, color="white", weight="bold")
    ax.text(x_col[1] + col_w[1] / 2, 8.88, "HITNet (Tankovich et al., CVPR 2021)",
            ha="center", va="center", fontsize=9, color="white", weight="bold")
    ax.text(x_col[2] + col_w[2] / 2, 8.88, "StereoLite v8  (ours)",
            ha="center", va="center", fontsize=9, color="white", weight="bold")


def main():
    fig = plt.figure(figsize=(14.0, 16.5))
    gs = fig.add_gridspec(3, 2, height_ratios=[18, 1, 10],
                          hspace=0.08, wspace=0.08)

    ax_l = fig.add_subplot(gs[0, 0])
    ax_l.set_xlim(0, 5)
    ax_l.set_ylim(3, 13)
    ax_l.axis("off")
    draw_hitnet(ax_l)

    ax_r = fig.add_subplot(gs[0, 1])
    ax_r.set_xlim(0, 5)
    ax_r.set_ylim(1, 13)
    ax_r.axis("off")
    draw_ours(ax_r)

    # Shared legend
    ax_leg = fig.add_subplot(gs[1, :])
    draw_legend(ax_leg)

    # Diff summary panel (spans both columns)
    ax_diff = fig.add_subplot(gs[2, :])
    draw_diff_panel(ax_diff)

    fig.suptitle(
        "Architecture comparison: HITNet vs StereoLite v8",
        fontsize=14, fontweight="bold", y=0.992)

    out_dir = os.path.dirname(os.path.abspath(__file__))
    png = os.path.join(out_dir, "arch_comparison.png")
    pdf = os.path.join(out_dir, "arch_comparison.pdf")
    fig.savefig(png, dpi=170, bbox_inches="tight", pad_inches=0.15)
    fig.savefig(pdf, bbox_inches="tight", pad_inches=0.15)
    print(f"wrote {png}")
    print(f"wrote {pdf}")


if __name__ == "__main__":
    main()
