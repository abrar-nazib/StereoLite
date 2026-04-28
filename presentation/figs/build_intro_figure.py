"""Introduction-slide infographic: stereo geometry schematic.

Produces ``intro_stereo_geometry.png`` — a clean labeled diagram
showing two horizontally offset cameras observing a point P, with the
baseline B, focal length f, image-plane projections x_L and x_R, and
the resulting depth Z = f·B/d.

The slide also embeds two photos (left view + depth heatmap) extracted
separately from one of our indoor benchmark pairs; this script only
draws the geometry."""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle, Polygon

OUT = Path(__file__).resolve().parent

CREAM = "#F4EFE6"
INK = "#1A1A1F"
SUBINK = "#5A5550"
ACCENT = "#C24A1C"
NAVY = "#14385C"
DOT_BLUE = "#3B6FB6"
DOT_GREEN = "#3F7A48"
DOT_AMBER = "#A07A1F"
PANEL = "#FBF7EE"

mpl.rcParams.update({
    "font.family": "DejaVu Serif",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.spines.left": False,
    "axes.spines.bottom": False,
})


def stereo_geometry():
    fig, ax = plt.subplots(figsize=(9.5, 3.0), facecolor=CREAM)
    ax.set_facecolor(CREAM)
    ax.set_xlim(0, 19); ax.set_ylim(0, 6)
    ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])

    # Title-bar
    ax.text(9.5, 5.55, "HOW STEREO COMPUTES DEPTH",
             fontsize=12, fontweight="bold", color=ACCENT,
             ha="center", va="center", family="DejaVu Sans Mono")

    # ---- Scene point P (at top, observed by both cameras) ----
    P_xy = (9.5, 4.55)
    ax.add_patch(Polygon([
        (P_xy[0] - 0.18, P_xy[1] - 0.18),
        (P_xy[0] + 0.18, P_xy[1] - 0.18),
        (P_xy[0],         P_xy[1] + 0.22),
    ], facecolor=ACCENT, edgecolor=INK, linewidth=0.8, zorder=4))
    ax.text(P_xy[0] + 0.45, P_xy[1] + 0.15, "P  (scene point)",
             fontsize=10, color=INK, va="center")

    # ---- Two cameras (lens icons) ----
    cam_y = 1.20
    cam_h = 0.50
    cam_w = 1.50
    L_x = 4.30
    R_x = 13.50

    def _draw_camera(cx, cy, label, color):
        # Body
        ax.add_patch(FancyBboxPatch((cx - cam_w / 2, cy), cam_w, cam_h,
            boxstyle="round,pad=0.02,rounding_size=0.08",
            linewidth=0.8, edgecolor=INK, facecolor=color, zorder=2))
        # Lens (small circle)
        from matplotlib.patches import Circle
        ax.add_patch(Circle((cx, cy + cam_h / 2), 0.16,
                              facecolor="#1A1A1F", edgecolor=INK,
                              linewidth=0.6, zorder=3))
        ax.text(cx, cy - 0.30, label, fontsize=10,
                 ha="center", va="top", color=INK, fontweight="bold")

    _draw_camera(L_x, cam_y, "LEFT camera", DOT_BLUE)
    _draw_camera(R_x, cam_y, "RIGHT camera", DOT_BLUE)

    # ---- Baseline B between cameras ----
    base_y = cam_y + cam_h / 2
    ax.annotate("", xy=(R_x - 0.85, base_y), xytext=(L_x + 0.85, base_y),
                  arrowprops=dict(arrowstyle="<->", color=ACCENT, lw=1.6),
                  zorder=2)
    ax.text((L_x + R_x) / 2, base_y + 0.30, "B  (baseline)",
             fontsize=11, color=ACCENT, ha="center", va="bottom",
             fontweight="bold")

    # ---- Image planes (small rectangles inside / between camera and P) ----
    # Just visualise where x_L and x_R land on each image plane.
    plane_y = cam_y + cam_h + 0.40   # plane sits above the camera body
    plane_h = 0.30
    plane_w = 1.50

    def _draw_plane(cx):
        ax.add_patch(Rectangle((cx - plane_w / 2, plane_y), plane_w, plane_h,
                                  facecolor=PANEL, edgecolor=SUBINK,
                                  linewidth=0.6, zorder=2))

    _draw_plane(L_x)
    _draw_plane(R_x)

    # Optical axes (vertical lines through camera centers)
    for cx in (L_x, R_x):
        ax.plot([cx, cx], [cam_y + cam_h, plane_y + plane_h + 0.10],
                 color=SUBINK, linewidth=0.5, linestyle=":", zorder=1)

    # Projection points x_L, x_R on each image plane
    # Choose offsets so x_L is to the right of left optical axis, x_R is
    # to the left of right optical axis (typical convention).
    xL_offset = 0.42
    xR_offset = -0.42
    xL_xy = (L_x + xL_offset, plane_y + plane_h / 2)
    xR_xy = (R_x + xR_offset, plane_y + plane_h / 2)
    for xy, color in [(xL_xy, ACCENT), (xR_xy, ACCENT)]:
        ax.plot(*xy, marker="o", color=color, markersize=6, zorder=4)

    ax.text(xL_xy[0], xL_xy[1] + 0.45, r"$x_L$",
             fontsize=11, color=ACCENT, ha="center", fontweight="bold")
    ax.text(xR_xy[0], xR_xy[1] + 0.45, r"$x_R$",
             fontsize=11, color=ACCENT, ha="center", fontweight="bold")

    # ---- Rays from P through image-plane points to camera centers ----
    # Each ray crosses the image plane at x_L / x_R then converges to the
    # camera optical center (centre of the camera body).
    L_optical = (L_x, cam_y + cam_h / 2)
    R_optical = (R_x, cam_y + cam_h / 2)
    for ipt, ocenter in [(xL_xy, L_optical), (xR_xy, R_optical)]:
        # P → image-plane point (in front of camera)
        ax.plot([P_xy[0], ipt[0]], [P_xy[1], ipt[1]],
                 color=DOT_GREEN, linewidth=1.4, zorder=2)
        # image-plane point → optical centre
        ax.plot([ipt[0], ocenter[0]], [ipt[1], ocenter[1]],
                 color=DOT_GREEN, linewidth=1.4, linestyle="--", zorder=2)

    # ---- Depth annotation (Z, vertical from baseline to P) ----
    ax.annotate("", xy=(P_xy[0], P_xy[1] - 0.08),
                  xytext=(P_xy[0], base_y + 0.10),
                  arrowprops=dict(arrowstyle="<->", color=NAVY, lw=1.4),
                  zorder=1)
    ax.text(P_xy[0] + 0.25, (P_xy[1] + base_y) / 2,
             r"$Z$  (depth)",
             fontsize=11, color=NAVY, va="center", fontweight="bold")

    # ---- Equation panel on the right side ----
    eq_x = 16.6
    eq_y = 3.10
    ax.add_patch(FancyBboxPatch((eq_x - 1.85, eq_y - 0.85), 3.50, 2.00,
        boxstyle="round,pad=0.04,rounding_size=0.10",
        linewidth=1.0, edgecolor=ACCENT, facecolor=PANEL, zorder=2))
    ax.text(eq_x, eq_y + 0.85, "DISPARITY",
             fontsize=9, fontweight="bold", color=ACCENT, ha="center",
             family="DejaVu Sans Mono")
    ax.text(eq_x, eq_y + 0.45, r"$d = x_L - x_R$",
             fontsize=12, color=INK, ha="center")
    ax.plot([eq_x - 1.55, eq_x + 1.55], [eq_y + 0.18, eq_y + 0.18],
             color=SUBINK, linewidth=0.5)
    ax.text(eq_x, eq_y - 0.10, "DEPTH",
             fontsize=9, fontweight="bold", color=ACCENT, ha="center",
             family="DejaVu Sans Mono")
    ax.text(eq_x, eq_y - 0.55,
             r"$Z = f \cdot B \, / \, d$",
             fontsize=14, color=INK, ha="center", fontweight="bold")

    # ---- Footnote: f = focal length ----
    ax.text(9.5, 0.35,
             "f = focal length (camera intrinsic)   ·   "
             "B = baseline (distance between cameras)   ·   "
             "d = disparity (horizontal pixel shift between left and right views)",
             fontsize=8.5, color=SUBINK, ha="center", va="center",
             style="italic")

    out = OUT / "intro_stereo_geometry.png"
    fig.savefig(out, bbox_inches="tight", pad_inches=0.10,
                 facecolor=CREAM, dpi=220)
    plt.close(fig)
    print(f"  -> {out}")
    return out


if __name__ == "__main__":
    stereo_geometry()
