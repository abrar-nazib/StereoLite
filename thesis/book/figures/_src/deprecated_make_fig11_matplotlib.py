"""Fig 1.1: two-view stereo geometry schematic (thesis style: white bg,
serif, vector PDF). Content mirrors presentation/figs/build_intro_figure
but restyled per WRITING_PLAN section 9."""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Rectangle

OUT = Path(__file__).resolve().parents[4] / "thesis/book/figures"
INK = "#1a1a1a"
BLUE = "#0072B2"
VERM = "#D55E00"
GREY = "#888888"

plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "Times", "TeX Gyre Termes", "DejaVu Serif"], "mathtext.fontset": "dejavuserif"})


def main():
    fig, ax = plt.subplots(figsize=(6.3, 2.9))
    ax.set_xlim(0, 19)
    ax.set_ylim(0, 8)
    ax.set_aspect("equal")
    ax.axis("off")

    # camera centres on the baseline
    oL, oR, ybase = 4.0, 10.0, 1.0
    P = (7.6, 6.8)

    for x, name in ((oL, "$O_L$"), (oR, "$O_R$")):
        ax.add_patch(Rectangle((x - 0.55, ybase - 0.55), 1.1, 0.75,
                               facecolor="#e9ecef", edgecolor=INK, lw=1.0))
        ax.plot([x], [ybase + 0.2], marker="o", ms=3, color=INK)
        ax.text(x, ybase - 0.95, name, ha="center", fontsize=10)

    # baseline
    ax.annotate("", xy=(oR - 0.6, ybase - 1.6), xytext=(oL + 0.6, ybase - 1.6),
                arrowprops=dict(arrowstyle="<->", color=GREY, lw=1.0))
    ax.text((oL + oR) / 2, ybase - 2.1, "baseline $B$", ha="center",
            fontsize=9, color=GREY)

    # image planes at focal distance f
    yplane = ybase + 1.8
    for x in (oL, oR):
        ax.plot([x - 1.5, x + 1.5], [yplane, yplane], color=INK, lw=1.6)
    ax.annotate("", xy=(oL - 1.85, yplane), xytext=(oL - 1.85, ybase + 0.2),
                arrowprops=dict(arrowstyle="<->", color=GREY, lw=0.9))
    ax.text(oL - 2.15, (yplane + ybase) / 2 + 0.1, "$f$", ha="right",
            fontsize=10, color=GREY)

    # scene point + rays
    ax.plot(*P, marker="*", ms=13, color=VERM, zorder=5)
    ax.text(P[0] + 0.35, P[1] + 0.15, "$P$", fontsize=11, color=VERM)
    for x, xl, lbl in ((oL, None, "$x_L$"), (oR, None, "$x_R$")):
        ax.plot([x, P[0]], [ybase + 0.2, P[1]], color=BLUE, lw=1.1, alpha=0.85)
        # projection = ray/plane intersection
        t = (yplane - (ybase + 0.2)) / (P[1] - (ybase + 0.2))
        xi = x + t * (P[0] - x)
        ax.plot([xi], [yplane], marker="o", ms=4.5, color=BLUE)
        ax.text(xi + (0.30 if x == oL else 0.30), yplane + 0.28, lbl,
                fontsize=10, color=BLUE)

    # depth arrow
    ax.annotate("", xy=(P[0], P[1] - 0.35), xytext=(P[0], ybase + 0.2),
                arrowprops=dict(arrowstyle="<->", color=VERM, lw=0.9,
                                linestyle=(0, (4, 3))))
    ax.text(P[0] - 0.35, (P[1] + ybase) / 2, "$Z$", ha="right",
            fontsize=11, color=VERM)

    # relation panel
    ax.text(15.6, 5.4, "$d = x_L - x_R$", fontsize=12, ha="center")
    ax.text(15.6, 3.9, r"$Z = \dfrac{f\,B}{d}$", fontsize=14, ha="center")
    ax.text(15.6, 2.3, "near object $\\rightarrow$ large $d$\n"
                       "far object $\\rightarrow$ small $d$",
            fontsize=8.5, ha="center", color=GREY)

    fig.tight_layout(pad=0.3)
    fig.savefig(OUT / "fig_1_1_stereo_geometry.pdf", bbox_inches="tight")
    fig.savefig(OUT / "fig_1_1_stereo_geometry.png", dpi=220,
                bbox_inches="tight", facecolor="white")
    print("saved fig_1_1")


if __name__ == "__main__":
    main()
