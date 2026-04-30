"""Publication-quality architecture diagram for StereoLite.

Design principles distilled from 17 TIER-A stereo papers.
"""
from __future__ import annotations

import os

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle, \
    Rectangle, Polygon, Arc
import numpy as np


# --- palette ------------------------------------------------------------- #
C_ENC = "#a8c8e4"
C_ENC_EDGE = "#5a7fa0"
C_CV = "#ffc66d"
C_CV_EDGE = "#c8861c"
C_REF = "#95d5b2"
C_REF_EDGE = "#4a8a5f"
C_GREY = "#e9ecef"
C_GREY_EDGE = "#6c757d"
C_TEXT = "#1a1a2e"
C_SUP = "#c1121f"


def _box(ax, x, y, w, h, face, edge, lw=1.2, rounding=0.04, zorder=2):
    ax.add_patch(FancyBboxPatch(
        (x, y), w, h,
        boxstyle=f"round,pad=0.0,rounding_size={rounding}",
        linewidth=lw, edgecolor=edge, facecolor=face, zorder=zorder))


def _txt(ax, x, y, s, fs=8.5, weight="normal", color=C_TEXT,
         ha="center", va="center", zorder=6, style="normal"):
    ax.text(x, y, s, fontsize=fs, fontweight=weight, color=color,
            ha=ha, va=va, zorder=zorder, fontstyle=style)


def _arrow(ax, x0, y0, x1, y1, lw=1.2, color="#333",
           style="-|>", mutation=12, zorder=3):
    ax.add_patch(FancyArrowPatch(
        (x0, y0), (x1, y1), arrowstyle=style,
        mutation_scale=mutation, linewidth=lw, color=color, zorder=zorder))


def _sup_dot(ax, x, y, label, fs=8.0):
    ax.add_patch(Circle((x, y), 0.045, facecolor=C_SUP, edgecolor="#8a0000",
                         linewidth=0.6, zorder=7))
    ax.text(x, y + 0.14, f"$\\mathcal{{L}}_{{{label}}}$",
            fontsize=fs, ha="center", va="bottom", color=C_SUP, zorder=7)


def _cv_prism(ax, x, y, w, h, depth=0.22, face=C_CV, edge=C_CV_EDGE, lw=1.2):
    _box(ax, x, y, w, h, face, edge, lw=lw, rounding=0.02)
    top = Polygon([(x, y + h), (x + depth, y + h + depth * 0.45),
                   (x + w + depth, y + h + depth * 0.45), (x + w, y + h)],
                  closed=True, facecolor=face, edgecolor=edge, linewidth=lw,
                  zorder=2)
    ax.add_patch(top)
    right = Polygon([(x + w, y), (x + w + depth, y + depth * 0.45),
                     (x + w + depth, y + h + depth * 0.45), (x + w, y + h)],
                    closed=True, facecolor=face, edgecolor=edge, linewidth=lw,
                    zorder=2)
    ax.add_patch(right)


def _loop_glyph(ax, x, y, r=0.11, color="#333", lw=1.3):
    """Tight circular arrow glyph representing recurrent iteration."""
    ax.add_patch(Arc((x, y), r * 2, r * 2, angle=0, theta1=40, theta2=320,
                      color=color, linewidth=lw, zorder=6))
    ax.add_patch(FancyArrowPatch(
        (x + r * np.cos(np.radians(42)), y + r * np.sin(np.radians(42))),
        (x + r * np.cos(np.radians(22)), y + r * np.sin(np.radians(22))),
        arrowstyle="-|>", mutation_scale=7, linewidth=lw, color=color,
        zorder=7))


def main():
    # Widened canvas so the full pipeline fits without clipping conv-up 1/2→full
    # or the disparity thumbnail. Taller too, for vertically-stacked pill legend.
    fig = plt.figure(figsize=(17.5, 5.4))
    ax = fig.add_axes([0.01, 0.03, 0.98, 0.94])
    ax.set_xlim(0, 17.5)
    ax.set_ylim(0, 5.4)
    ax.axis("off")
    ax.set_aspect("equal")

    # ================== LEFT: input stereo pair ================== #
    L = mpimg.imread("/tmp/diag_left.png")
    R = mpimg.imread("/tmp/diag_right.png")
    D = mpimg.imread("/tmp/diag_disp.png")

    img_w, img_h = 1.20, 0.70
    img_x = 0.30

    # Stacked L / R thumbnails
    ax.imshow(L, extent=(img_x, img_x + img_w, 3.25, 3.25 + img_h),
              aspect="auto", zorder=2)
    ax.add_patch(Rectangle((img_x, 3.25), img_w, img_h, linewidth=0.9,
                             edgecolor="#333", facecolor="none", zorder=3))
    _txt(ax, img_x + img_w / 2, 3.18, "$I_L$  left",
         fs=9, va="top", weight="bold")

    ax.imshow(R, extent=(img_x, img_x + img_w, 2.00, 2.00 + img_h),
              aspect="auto", zorder=2)
    ax.add_patch(Rectangle((img_x, 2.00), img_w, img_h, linewidth=0.9,
                             edgecolor="#333", facecolor="none", zorder=3))
    _txt(ax, img_x + img_w / 2, 1.93, "$I_R$  right",
         fs=9, va="top", weight="bold")

    # ================== ENCODER ================== #
    enc_x = img_x + img_w + 0.60
    enc_y_center = 2.95  # center of the encoder vertical block
    enc_box_widths = [0.42, 0.36, 0.31, 0.28]
    enc_box_heights = [1.60, 1.30, 1.05, 0.85]
    labels_scale = ["1/2", "1/4", "1/8", "1/16"]
    labels_ch = ["16c", "24c", "32c", "96c"]

    enc_centers = []
    cur_x = enc_x
    for i, (w, h, s, ch) in enumerate(zip(enc_box_widths, enc_box_heights,
                                           labels_scale, labels_ch)):
        by = enc_y_center - h / 2
        _box(ax, cur_x, by, w, h, C_ENC, C_ENC_EDGE, lw=1.3)
        # Scale label BELOW block, well clear
        _txt(ax, cur_x + w / 2, by - 0.18, s, fs=9, weight="bold")
        # Channel label ABOVE block
        _txt(ax, cur_x + w / 2, by + h + 0.14, ch, fs=7.5, color="#666")
        enc_centers.append(cur_x + w / 2)
        cur_x = cur_x + w + 0.12
    enc_right_x = cur_x - 0.12  # right edge of last encoder block

    # Encoder title well below scale labels
    enc_title_x = (enc_x + enc_right_x) / 2
    _txt(ax, enc_title_x, 1.78, "MobileNetV2-100 (truncated)",
         fs=9, weight="bold", color=C_ENC_EDGE)
    _txt(ax, enc_title_x, 1.60, "ImageNet-pretrained, shared weights",
         fs=7.2, color="#666", style="italic")

    # Feeder arrows
    _arrow(ax, img_x + img_w + 0.02, 3.25 + img_h / 2,
           enc_x - 0.05, enc_y_center + 0.5, lw=1.1)
    _arrow(ax, img_x + img_w + 0.02, 2.00 + img_h / 2,
           enc_x - 0.05, enc_y_center - 0.5, lw=1.1)

    # ================== COST VOLUME ================== #
    cv_x = enc_right_x + 0.45
    cv_y = 2.55
    cv_w = 0.90
    cv_h = 1.15
    _cv_prism(ax, cv_x, cv_y, cv_w, cv_h)
    # Label INSIDE the prism front face
    _txt(ax, cv_x + cv_w / 2, cv_y + cv_h / 2 + 0.18, "local CV",
         fs=9.5, weight="bold", color="#703f00")
    _txt(ax, cv_x + cv_w / 2, cv_y + cv_h / 2 - 0.02, "1/16,  D=24",
         fs=8, color="#4a2e00")
    _txt(ax, cv_x + cv_w / 2, cv_y + cv_h / 2 - 0.22, "group-corr",
         fs=7.2, color="#5d3b00", style="italic")
    # 3D aggregator label BELOW prism, clear of the right-slant face
    _txt(ax, cv_x + cv_w / 2, cv_y - 0.20,
         "3D aggregator",
         fs=8.5, weight="bold", color=C_CV_EDGE)

    # Arrow from encoder's last block to cost volume
    _arrow(ax, enc_right_x + 0.02, enc_y_center,
           cv_x - 0.02, cv_y + cv_h / 2, lw=1.2)

    # ================== TILE STATE PILL ================== #
    pill_x = cv_x + cv_w + 0.70
    pill_y = cv_y + 0.30
    pill_w, pill_h = 1.55, 0.42
    _box(ax, pill_x, pill_y, pill_w, pill_h, "#ffffff", "#333", lw=1.0)
    stripes = ["$d$", "$s_x$", "$s_y$", "$f$", "$c$"]
    for j, s in enumerate(stripes):
        sx = pill_x + (pill_w / 5) * (j + 0.5)
        _txt(ax, sx, pill_y + pill_h / 2, s, fs=10, color="#333")
        if j < 4:
            ax.plot([pill_x + (pill_w / 5) * (j + 1)] * 2,
                    [pill_y + 0.06, pill_y + pill_h - 0.06],
                    color="#bbb", linewidth=0.6, zorder=6)
    # Title ABOVE pill
    _txt(ax, pill_x + pill_w / 2, pill_y + pill_h + 0.22,
         "tile-hypothesis state", fs=9, weight="bold", color="#333")
    # Legend BELOW pill — one item per line to avoid horizontal overflow.
    legend_lines = [
        "$d$:  disparity",
        "$s_x, s_y$:  plane slopes",
        "$f$:  16-ch feature",
        "$c$:  confidence",
    ]
    for i, line in enumerate(legend_lines):
        _txt(ax, pill_x + pill_w / 2, pill_y - 0.18 - i * 0.20,
             line, fs=7.3, color="#555", style="italic", va="top", ha="center")

    # Arrow from CV to pill
    _arrow(ax, cv_x + cv_w + 0.30, pill_y + pill_h / 2,
           pill_x - 0.02, pill_y + pill_h / 2, lw=1.1)

    # ================== ITERATIVE REFINEMENT (3 scales) ================== #
    # Narrower + taller blocks with vertically stacked content: title / loop
    # glyph / iteration count / param count.
    ref_y_center = pill_y + pill_h / 2
    ref_h = 1.30
    ref_y = ref_y_center - ref_h / 2
    ref_x0 = pill_x + pill_w + 0.85
    ref_w = 1.05
    ref_gap = 0.70

    iter_counts = ["2", "3", "3"]
    scales = ["1/16", "1/8", "1/4"]
    ref_param = ["0.13 M", "0.08 M", "0.07 M"]
    ref_centers = []
    for i, (it, sc, pm) in enumerate(zip(iter_counts, scales, ref_param)):
        rx = ref_x0 + i * (ref_w + ref_gap)
        _box(ax, rx, ref_y, ref_w, ref_h, C_REF, C_REF_EDGE, lw=1.4)
        # 1. Title near top
        _txt(ax, rx + ref_w / 2, ref_y + ref_h - 0.20,
             f"refine  {sc}", fs=10, weight="bold", color="#1a3d24")
        # 2. Loop glyph — standalone, horizontally centered
        glyph_y = ref_y + ref_h - 0.55
        _loop_glyph(ax, rx + ref_w / 2, glyph_y, r=0.15,
                    color="#1a3d24", lw=1.5)
        # 3. Iteration count BELOW glyph
        _txt(ax, rx + ref_w / 2, ref_y + ref_h - 0.93, f"×{it}",
             fs=12, weight="bold", color="#1a3d24")
        # 4. Param count at bottom
        _txt(ax, rx + ref_w / 2, ref_y + 0.15, pm,
             fs=7.5, color="#2e5339", style="italic")
        ref_centers.append((rx, rx + ref_w, ref_y_center))

    # Arrow pill → first refine
    _arrow(ax, pill_x + pill_w + 0.02, ref_y_center,
           ref_centers[0][0] - 0.02, ref_y_center, lw=1.2)

    # Plane-equation arrows between refine blocks
    for i in range(len(ref_centers) - 1):
        x0 = ref_centers[i][1]
        x1 = ref_centers[i + 1][0]
        _arrow(ax, x0 + 0.02, ref_y_center, x1 - 0.02, ref_y_center, lw=1.3)
        mid_x = (x0 + x1) / 2
        # Plane eq. label placed WELL ABOVE the arrow AND above the refine
        # block tops, so it never overlaps a green box.
        lab_y = ref_y + ref_h + 0.30
        _txt(ax, mid_x, lab_y, "plane eq.",
             fs=8, weight="bold", color="#333")
        _txt(ax, mid_x, lab_y - 0.18, "2× upsample",
             fs=7, color="#666", style="italic")
        # Thin vertical connector from label to arrow head for clarity
        ax.plot([mid_x, mid_x], [ref_y + ref_h + 0.05, ref_y_center + 0.07],
                color="#aaa", linewidth=0.6, linestyle=":", zorder=1)

    # Supervision dots ABOVE each refine block, between plane-eq. labels and
    # the meta caption. Meta box starts at y=4.55, so keep labels below 4.45.
    sup_y = ref_y + ref_h + 0.55
    sup_labels_ref = ["16", "8", "4"]
    for (x0, x1, _yc), lbl in zip(ref_centers, sup_labels_ref):
        _sup_dot(ax, (x0 + x1) / 2, sup_y, lbl, fs=8.5)

    # ================== CONVEX UPSAMPLE + DISPARITY ================== #
    cu_x = ref_centers[-1][1] + 0.55
    cu_w = 0.90
    cu_h = 0.75
    cu_y = ref_y_center - cu_h / 2

    trap_specs = [("conv-up\n1/4→1/2", 0), ("conv-up\n1/2→full", 1)]
    trap_centers = []
    for lbl, idx in trap_specs:
        tx = cu_x + idx * (cu_w + 0.25)
        pts = [(tx, cu_y + 0.12), (tx + cu_w, cu_y - 0.05),
               (tx + cu_w, cu_y + cu_h + 0.05), (tx, cu_y + cu_h - 0.12)]
        ax.add_patch(Polygon(pts, closed=True, facecolor=C_GREY,
                              edgecolor=C_GREY_EDGE, linewidth=1.3, zorder=2))
        _txt(ax, tx + cu_w / 2, cu_y + cu_h / 2, lbl, fs=7.5,
             color="#333")
        trap_centers.append(tx + cu_w / 2)

    # Connecting arrows
    _arrow(ax, ref_centers[-1][1] + 0.02, ref_y_center,
           cu_x - 0.02, ref_y_center, lw=1.2)
    _arrow(ax, cu_x + cu_w + 0.02, ref_y_center,
           cu_x + cu_w + 0.23, ref_y_center, lw=1.0)

    # Supervision dots above conv-ups
    _sup_dot(ax, trap_centers[0], sup_y, "2", fs=8.5)
    _sup_dot(ax, trap_centers[1], sup_y, "1", fs=8.5)

    # Disparity output thumbnail. Our model predicts DISPARITY d̂ (per-pixel
    # horizontal shift between left/right views). Depth Z is a downstream
    # geometric conversion: Z = f·B / d̂ (focal × baseline / disparity).
    disp_x = cu_x + (cu_w + 0.25) + cu_w + 0.35
    disp_w, disp_h = 1.30, 0.78
    disp_y = ref_y_center - disp_h / 2
    ax.imshow(D, extent=(disp_x, disp_x + disp_w, disp_y, disp_y + disp_h),
              aspect="auto", zorder=2)
    ax.add_patch(Rectangle((disp_x, disp_y), disp_w, disp_h, linewidth=0.9,
                             edgecolor="#333", facecolor="none", zorder=3))
    _txt(ax, disp_x + disp_w / 2, disp_y + disp_h + 0.16,
         "disparity  $\\hat{d}$", fs=10, weight="bold")
    # Depth-conversion note below the thumbnail
    _txt(ax, disp_x + disp_w / 2, disp_y - 0.14,
         "depth $Z = f \\cdot B / \\hat{d}$",
         fs=7.8, color="#444", style="italic", va="top")
    _txt(ax, disp_x + disp_w / 2, disp_y - 0.34,
         "(post-hoc, not part of the network)",
         fs=6.8, color="#777", style="italic", va="top")
    _arrow(ax, cu_x + (cu_w + 0.25) + cu_w + 0.02, ref_y_center,
           disp_x - 0.02, ref_y_center, lw=1.2)

    # ================== PHASE RAILS ================== #
    rail_y = 1.22
    phase_groups = [
        (enc_x - 0.05, enc_right_x + 0.05, "encoder  (1/2 → 1/16)"),
        (cv_x - 0.05, pill_x + pill_w + 0.05, "tile-hypothesis init"),
        (ref_centers[0][0] - 0.05, ref_centers[-1][1] + 0.05,
         "iterative refinement  (coarse → fine)"),
        (cu_x - 0.08, cu_x + (cu_w + 0.25) + cu_w + 0.08,
         "learned convex upsample"),
    ]
    for xa, xb, lbl in phase_groups:
        ax.plot([xa, xb], [rail_y, rail_y], linestyle="--",
                color="#9ea4aa", linewidth=0.9, zorder=1)
        for xt in (xa, xb):
            ax.plot([xt, xt], [rail_y - 0.05, rail_y + 0.05],
                    color="#9ea4aa", linewidth=0.9, zorder=1)
        _txt(ax, (xa + xb) / 2, rail_y - 0.25, lbl,
             fs=8, color="#444", weight="bold", style="italic")

    # ================== META + LEGEND ================== #
    # Meta caption, top-right. Meta row sits ABOVE the supervision rail.
    meta_x, meta_y = 10.4, 5.00
    meta_w, meta_h = 6.8, 0.30
    _box(ax, meta_x, meta_y, meta_w, meta_h, "#ffffff", "#888",
         lw=0.8, rounding=0.04)
    _txt(ax, meta_x + meta_w / 2, meta_y + meta_h / 2,
         "StereoLite  •  0.87 M params  •  54 ms @ 512×832 on RTX 3050  •  "
         "EPE 1.62 on Scene Flow Driving",
         fs=8.5, weight="bold", color="#222")

    # Legend bottom — spread evenly across full width
    leg_y = 0.45
    items = [
        ("CNN encoder (2D conv)", C_ENC, C_ENC_EDGE),
        ("cost volume / 3D agg.", C_CV, C_CV_EDGE),
        ("iterative refinement", C_REF, C_REF_EDGE),
        ("learned upsample", C_GREY, C_GREY_EDGE),
    ]
    step = 2.6
    leg_x0 = 0.40
    for i, (name, col, ec) in enumerate(items):
        bx = leg_x0 + i * step
        _box(ax, bx, leg_y - 0.10, 0.32, 0.24, col, ec, lw=1.0, rounding=0.02)
        _txt(ax, bx + 0.40, leg_y + 0.02, name, fs=8.2, ha="left", va="center")

    # Supervision legend on the right
    sup_x = leg_x0 + 4 * step
    _sup_dot(ax, sup_x, leg_y + 0.02, "k", fs=8.2)
    _txt(ax, sup_x + 0.20, leg_y + 0.02,
         "multi-scale supervision  $\\mathcal{L}_k$  "
         "(L1 + Sobel-grad + bad-1 hinge)",
         fs=8.2, ha="left", va="center")

    # ================== SAVE ================== #
    out_dir = os.path.dirname(os.path.abspath(__file__))
    png = os.path.join(out_dir, "stereolite_arch.png")
    pdf = os.path.join(out_dir, "stereolite_arch.pdf")
    fig.savefig(png, dpi=200, bbox_inches="tight", pad_inches=0.12,
                facecolor="white")
    fig.savefig(pdf, bbox_inches="tight", pad_inches=0.12, facecolor="white")
    print(f"wrote {png}")
    print(f"wrote {pdf}")


if __name__ == "__main__":
    main()
