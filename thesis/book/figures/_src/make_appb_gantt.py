"""Appendix B Gantt chart of the actual project timeline (Feb to Jul 2026)."""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = Path("/home/abrar/Research/stero_research_claude/thesis/book/figures")
plt.rcParams.update({"font.family": "DejaVu Serif"})

# (task, start_month_index, duration_months) with month 0 = February 2026
TASKS = [
    ("Paper collection and study", 0, 2.0, "#0072B2"),
    ("Review paper draft", 1.0, 2.0, "#56B4E9"),
    ("Architecture design and iteration", 1.5, 2.5, "#009E73"),
    ("Ablation campaign", 3.0, 1.5, "#66C2A5"),
    ("Production training run", 4.2, 0.6, "#E69F00"),
    ("Evaluation and gates", 4.6, 0.6, "#D55E00"),
    ("Thesis writing", 4.5, 1.3, "#CC79A7"),
]
MONTHS = ["Feb", "Mar", "Apr", "May", "Jun", "Jul"]


def main():
    fig, ax = plt.subplots(figsize=(6.0, 2.6))
    for i, (name, s, d, c) in enumerate(TASKS):
        y = len(TASKS) - 1 - i
        ax.barh(y, d, left=s, height=0.55, color=c, edgecolor="#333",
                linewidth=0.5)
        ax.text(-0.1, y, name, ha="right", va="center", fontsize=8)
    ax.set_yticks([])
    ax.set_xlim(0, 6)
    ax.set_xticks(range(7))
    ax.set_xticklabels(MONTHS + [""], fontsize=8)
    ax.set_xlabel("2026", fontsize=8)
    ax.spines[["top", "right", "left"]].set_visible(False)
    for m in range(7):
        ax.axvline(m, color="#eeeeee", lw=0.6, zorder=0)
    fig.tight_layout(pad=0.3)
    fig.savefig(OUT / "fig_b_1_gantt.pdf", bbox_inches="tight")
    fig.savefig(OUT / "fig_b_1_gantt.png", dpi=220, bbox_inches="tight",
                facecolor="white")
    print("saved fig_b_1_gantt")


if __name__ == "__main__":
    main()
