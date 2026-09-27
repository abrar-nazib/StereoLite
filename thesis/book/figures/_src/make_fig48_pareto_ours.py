"""Fig 4.8: final positioning, zero-shot Middlebury 2014 D1-all against
trainable parameters, SAME 23-scene protocol for all four points
(mb14_zero_shot.json + the repo's matched-protocol reference evals)."""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = Path(__file__).resolve().parents[4] / "thesis/book/figures"
plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "Times", "TeX Gyre Termes", "DejaVu Serif"]})

# (label, params M, D1-all %, marker, color)
POINTS = [
    ("StereoLite (ours)", 2.96, 10.86, "*", "#D55E00"),
    ("LiteAnyStereo", 7.60, 6.9, "s", "#0072B2"),
    ("IGEV-Stereo (16 it.)", 12.60, 5.0, "^", "#009E73"),
]


def main():
    fig, ax = plt.subplots(figsize=(4.6, 2.9))
    for lbl, p, d1, m, c in POINTS:
        sz = 280 if m == "*" else 90
        ax.scatter(p, d1, marker=m, s=sz, color=c, edgecolor="#1a1a1a",
                   linewidth=0.6, zorder=3)
        dy = 8 if lbl.startswith("Legacy") else 8
        ax.annotate(lbl, (p, d1), textcoords="offset points",
                    xytext=(7, dy - 4), fontsize=7.5,
                    fontweight="bold" if m == "*" else "normal", color=c)
    ax.set_xscale("log")
    ax.set_xlim(1.8, 22)
    ax.set_ylim(0, 14)
    ax.set_xlabel("trainable parameters (M, log scale)", fontsize=8.5)
    ax.set_ylabel("zero-shot MB14 D1-all (%)", fontsize=8.5)
    ax.tick_params(labelsize=7.5)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout(pad=0.3)
    fig.savefig(OUT / "fig_4_8_pareto_ours.pdf", bbox_inches="tight")
    fig.savefig(OUT / "fig_4_8_pareto_ours.png", dpi=220,
                bbox_inches="tight", facecolor="white")
    print("saved fig_4_8")


if __name__ == "__main__":
    main()
