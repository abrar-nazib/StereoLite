"""Compact Gantt chart of the main experimental period (Dec to Jul)."""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = Path(__file__).resolve().parents[4] / "thesis/book/figures"
plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "Times", "TeX Gyre Termes", "DejaVu Serif"]})

# (task, start_month_index, duration_months) with month 0 = December 2025
TASKS = [
    ("Model selection and setup", 0.0, 1.7, "#0072B2"),
    ("Pre-training and fine-tuning", 0.5, 3.0, "#56B4E9"),
    ("Real-data capture", 1.5, 2.3, "#009E73"),
    ("Ablation studies", 2.5, 2.5, "#66C2A5"),
    ("Benchmarking and reconstruction", 3.5, 2.2, "#E69F00"),
    ("Result analysis and evaluation", 4.7, 1.8, "#D55E00"),
    ("Thesis writing and documentation", 5.0, 2.2, "#CC79A7"),
]
MONTHS = ["Dec", "Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul"]


def main():
    fig, ax = plt.subplots(figsize=(6.3, 3.0))
    for i, (name, s, d, c) in enumerate(TASKS):
        y = len(TASKS) - 1 - i
        ax.barh(y, d, left=s, height=0.55, color=c, edgecolor="#333",
                linewidth=0.5)
        ax.text(-0.1, y, name, ha="right", va="center", fontsize=9)
    ax.set_yticks([])
    ax.set_xlim(0, 8)
    ax.set_xticks([i + 0.5 for i in range(8)])
    ax.set_xticklabels(MONTHS, fontsize=9)
    ax.set_xlabel("December 2025 to July 2026", fontsize=10)
    ax.spines[["top", "right", "left"]].set_visible(False)
    for m in range(9):
        ax.axvline(m, color="#eeeeee", lw=0.6, zorder=0)
    fig.tight_layout(pad=0.3)
    fig.savefig(OUT / "fig_b_1_gantt.pdf", bbox_inches="tight")
    fig.savefig(OUT / "fig_b_1_gantt.png", dpi=220, bbox_inches="tight",
                facecolor="white")
    print("saved fig_b_1_gantt")


if __name__ == "__main__":
    main()
