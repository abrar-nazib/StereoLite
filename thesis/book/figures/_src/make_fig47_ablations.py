"""Fig 4.7: ablation summary, four panels. All numbers verbatim from
model/benchmarks/EXPERIMENTS.md and the dual-axis comparison.md:
(a) augmentation lever (grand 9-arm, best val EPE),
(b) efficiency pass (eff_gev4_n100: EPE + latency),
(c) boundary blur fixes (blurfix_n500 + compose: bad-0.5),
(d) input protocol (native-axis EPE, comparison.md)."""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUT = Path("/home/abrar/Research/stero_research_claude/thesis/book/figures")
BLUE, VERM, GREY = "#0072B2", "#D55E00", "#999999"
plt.rcParams.update({"font.family": "DejaVu Serif"})


def main():
    fig, axes = plt.subplots(1, 4, figsize=(6.3, 2.0))

    # (a) augmentation (GRAND_COMPARISON_20260703: 2.778 -> 1.921)
    ax = axes[0]
    ax.bar(["off", "on"], [2.778, 1.921], color=[GREY, BLUE], width=0.6)
    ax.set_title("(a) augmentation", fontsize=8)
    ax.set_ylabel("best val EPE (px)", fontsize=7.5)
    for i, v in enumerate([2.778, 1.921]):
        ax.text(i, v + 0.03, f"{v:.2f}", ha="center", fontsize=6.5)

    # (b) efficiency pass (eff_gev4_n100)
    ax = axes[1]
    labels = ["base", "opt", "narrow"]
    epe = [2.908, 3.021, 2.906]
    ms = [46.6, 35.8, 30.2]
    x = np.arange(3)
    ax.bar(x - 0.2, epe, 0.38, color=BLUE, label="EPE (px)")
    ax2 = ax.twinx()
    ax2.bar(x + 0.2, ms, 0.38, color=VERM, label="latency (ms)")
    ax.set_xticks(x, labels, fontsize=7)
    ax.set_title("(b) efficiency pass", fontsize=8)
    ax.set_ylabel("EPE (px)", fontsize=7.5, color=BLUE)
    ax2.set_ylabel("ms", fontsize=7.5, color=VERM)
    ax2.tick_params(labelsize=6.5)

    # (c) blur fixes (blurfix_n500 + compose), bad-0.5 %
    ax = axes[2]
    labels = ["ctrl", "bndl", "bimod", "plane", "p+b", "allin"]
    vals = [55.36, 49.95, 43.26, 42.91, 44.40, 46.58]
    colors = [GREY, BLUE, BLUE, VERM, BLUE, BLUE]
    ax.bar(labels, vals, color=colors, width=0.65)
    ax.set_title("(c) blur fixes", fontsize=8)
    ax.set_ylabel("bad-0.5 (%)", fontsize=7.5)
    ax.tick_params(axis="x", labelsize=6, rotation=45)

    # (d) input protocol, native-axis EPE (comparison.md)
    ax = axes[3]
    labels = ["resize", "nat.crop", "nat.full"]
    vals = [6.672, 2.869, 2.967]
    ax.bar(labels, vals, color=[GREY, VERM, BLUE], width=0.6)
    ax.set_title("(d) input protocol", fontsize=8)
    ax.set_ylabel("native-axis EPE (px)", fontsize=7.5)
    for i, v in enumerate(vals):
        ax.text(i, v + 0.08, f"{v:.2f}", ha="center", fontsize=6.5)

    for a in list(axes):
        a.tick_params(labelsize=6.5)
        a.spines[["top", "right"]].set_visible(False)
    fig.tight_layout(pad=0.4)
    fig.savefig(OUT / "fig_4_7_ablations.pdf", bbox_inches="tight")
    fig.savefig(OUT / "fig_4_7_ablations.png", dpi=220,
                bbox_inches="tight", facecolor="white")
    print("saved fig_4_7")


if __name__ == "__main__":
    main()
