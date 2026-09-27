"""Fig 3.11: parameter budget of the trained model. Module split computed
directly from best.pth (state-dict numel per top-level module), NOT from
any older variant's numbers."""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = Path(__file__).resolve().parents[4] / "thesis/book/figures"
plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "Times", "TeX Gyre Termes", "DejaVu Serif"]})

# from best.pth state dict (see commit message / this script's provenance)
SPLIT = [
    ("Shared encoder (YOLO26s)", 1.2376, "#a8c8e4"),
    ("Refinement 1/16", 0.6008, "#95d5b2"),
    ("Refinement 1/8", 0.6008, "#74c69d"),
    ("Refinement 1/4", 0.3796, "#52b788"),
    ("Convex upsampling", 0.0729, "#e9ecef"),
    ("Context encoder", 0.0230, "#d5c6e0"),
    ("GEV + fusion", 0.0412, "#ffc66d"),
    ("Tile init + plane gate", 0.0117, "#f8cecc"),
]


def main():
    fig, ax = plt.subplots(figsize=(4.8, 2.7))
    vals = [v for _, v, _ in SPLIT]
    colors = [c for _, _, c in SPLIT]
    labels = [f"{n}\n{v:.2f} M ({100*v/sum(vals):.0f}%)" if v > 0.3
              else "" for n, v, _ in SPLIT]
    wedges, _ = ax.pie(vals, colors=colors, startangle=90,
                       wedgeprops=dict(width=0.42, edgecolor="white"))
    ax.legend(wedges, [f"{n}: {v:.2f} M" for n, v, _ in SPLIT],
              loc="center left", bbox_to_anchor=(1.02, 0.5), fontsize=7,
              frameon=False)
    ax.text(0, 0, "2.96 M\ntotal", ha="center", va="center", fontsize=10,
            fontweight="bold")
    fig.tight_layout(pad=0.3)
    fig.savefig(OUT / "fig_3_11_param_budget.pdf", bbox_inches="tight")
    fig.savefig(OUT / "fig_3_11_param_budget.png", dpi=220,
                bbox_inches="tight", facecolor="white")
    print("saved fig_3_11")


if __name__ == "__main__":
    main()
