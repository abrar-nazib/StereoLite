"""Vertical deployment-pipeline diagram:
   IMX-219 83 stereo camera -> Jetson Nano -> StereoLite model -> Socket stream.
"""
from __future__ import annotations

import os

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch


# Colour-coded by functional role
C_HW = "#a8c8e4"       # hardware (camera, Jetson)
C_HW_E = "#5a7fa0"
C_SW = "#95d5b2"       # software (model)
C_SW_E = "#4a8a5f"
C_OUT = "#ffc66d"      # output / stream
C_OUT_E = "#c8861c"
C_TEXT = "#1a1a2e"


def _box(ax, x, y, w, h, face, edge, lw=1.4):
    ax.add_patch(FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.0,rounding_size=0.06",
        linewidth=lw, edgecolor=edge, facecolor=face, zorder=2))


def _arrow(ax, x0, y0, x1, y1, lw=1.6):
    ax.add_patch(FancyArrowPatch(
        (x0, y0), (x1, y1), arrowstyle="-|>",
        mutation_scale=20, linewidth=lw, color="#333", zorder=3))


def _txt(ax, x, y, s, fs=10, weight="normal", color=C_TEXT,
         style="normal", ha="center", va="center"):
    ax.text(x, y, s, fontsize=fs, fontweight=weight, color=color,
            fontstyle=style, ha=ha, va=va, zorder=6)


def main():
    fig = plt.figure(figsize=(6.0, 9.0))
    ax = fig.add_axes([0.05, 0.03, 0.90, 0.94])
    ax.set_xlim(0, 6)
    ax.set_ylim(0, 9)
    ax.axis("off")
    ax.set_aspect("equal")

    # Layout — 4 boxes, vertical, evenly spaced
    box_w = 4.4
    box_h = 1.20
    box_x = 0.80
    centers_y = [7.40, 5.40, 3.40, 1.40]  # top -> bottom

    stages = [
        ("IMX-219 83° Stereo Camera",
         "MIPI-CSI dual sensor, 1280$\\times$720 @ 60 fps\n"
         "side-by-side stereo, 83° FOV",
         C_HW, C_HW_E),
        ("NVIDIA Jetson Nano",
         "host SoC, CUDA + TensorRT\n"
         "captures, rectifies, batches frames",
         C_HW, C_HW_E),
        ("StereoLite",
         "0.87 M params, INT8 / FP16\n"
         "predicts per-pixel disparity $\\hat{d}$",
         C_SW, C_SW_E),
        ("Socket Stream",
         "TCP / UDP output\n"
         "disparity frames to remote consumer",
         C_OUT, C_OUT_E),
    ]

    for cy, (title, sub, face, edge) in zip(centers_y, stages):
        _box(ax, box_x, cy - box_h / 2, box_w, box_h, face, edge)
        _txt(ax, box_x + box_w / 2, cy + 0.30, title,
             fs=12, weight="bold")
        _txt(ax, box_x + box_w / 2, cy - 0.18, sub,
             fs=9.5, color="#333", style="italic")

    # Vertical arrows between boxes, with data-flow labels on the right
    arrow_specs = [
        ("raw stereo frames", "1280×720, 60 fps"),
        ("rectified $I_L, I_R$", "$512 \\times 832$ tensors"),
        ("disparity $\\hat{d}$", "1 ch, full-res"),
    ]
    for i in range(3):
        y0 = centers_y[i] - box_h / 2
        y1 = centers_y[i + 1] + box_h / 2
        x = box_x + box_w / 2
        _arrow(ax, x, y0 - 0.05, x, y1 + 0.05)
        # Label on the right
        primary, secondary = arrow_specs[i]
        ymid = (y0 + y1) / 2
        _txt(ax, x + box_w / 2 + 0.10, ymid + 0.10, primary,
             fs=9, weight="bold", color="#333", ha="left")
        _txt(ax, x + box_w / 2 + 0.10, ymid - 0.10, secondary,
             fs=8, color="#666", style="italic", ha="left")

    # Title
    _txt(ax, 3.0, 8.65,
         "StereoLite — edge deployment pipeline",
         fs=14, weight="bold")

    # Save
    out_dir = os.path.dirname(os.path.abspath(__file__))
    png = os.path.join(out_dir, "deployment_pipeline.png")
    pdf = os.path.join(out_dir, "deployment_pipeline.pdf")
    fig.savefig(png, dpi=200, bbox_inches="tight", pad_inches=0.10,
                facecolor="white")
    fig.savefig(pdf, bbox_inches="tight", pad_inches=0.10, facecolor="white")
    print(f"wrote {png}")
    print(f"wrote {pdf}")


if __name__ == "__main__":
    main()
