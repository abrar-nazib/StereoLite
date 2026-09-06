"""Readable thesis-width parameter and latency comparisons.

These plots use the representative methods already listed in Table 2.4.
They are drawn at the final LaTeX width so their Times New Roman labels
remain close to the book's normal type size after inclusion.
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[4]
OUT = ROOT / "thesis/book/figures"

METHODS = [
    ("PSMNet", 5.20, 2.32, 410.0),
    ("HITNet", 0.97, 1.98, 54.0),
    ("BGNet", 2.90, 2.51, 25.0),
    ("CoEx", 2.72, 2.13, 27.0),
    ("RAFT-Stereo", 11.23, 1.82, 380.0),
    ("IGEV-Stereo", 12.60, 1.59, 180.0),
    ("LightStereo-S", 3.44, 2.30, 17.0),
    ("DEFOM-Stereo", 47.30, 1.55, 316.0),
    ("FoundationStereo", 340.0, 1.46, 470.0),
]

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "TeX Gyre Termes"],
    "font.size": 10,
    "axes.labelsize": 11,
    "axes.titlesize": 12,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
})


def _finish(fig, stem):
    fig.savefig(OUT / f"{stem}.pdf", bbox_inches="tight")
    fig.savefig(OUT / f"{stem}.png", dpi=300, bbox_inches="tight",
                facecolor="white")
    plt.close(fig)


def parameter_plot():
    fig, ax = plt.subplots(figsize=(6.3, 2.35))
    for i, (name, params, d1, _) in enumerate(METHODS, start=1):
        ax.scatter(params, d1, s=58, color="#0072B2", edgecolor="black",
                   linewidth=0.6, zorder=3)
        ax.annotate(str(i), (params, d1), xytext=(0, 0),
                    textcoords="offset points", ha="center", va="center",
                    fontsize=8, color="white", fontweight="bold")
    ax.set_xscale("log")
    ax.set_xlabel("Trainable parameters (M, log scale)")
    ax.set_ylabel("KITTI 2015 D1-all (%, lower is better)")
    ax.set_title("Model size and accuracy of representative stereo methods")
    ax.grid(True, alpha=0.25, linestyle=":")
    ax.spines[["top", "right"]].set_visible(False)
    key = "   ".join(f"{i} {m[0]}" for i, m in enumerate(METHODS, 1))
    fig.text(0.5, 0.005, key, ha="center", va="bottom", fontsize=8,
             wrap=True)
    fig.subplots_adjust(bottom=0.29, left=0.13, right=0.98, top=0.86)
    _finish(fig, "fig_2_6a_param_pareto")


def latency_plot():
    # Keep the labels at the intended thesis size while using a shorter
    # canvas so the full-width panel can share the page with panel (a).
    fig, ax = plt.subplots(figsize=(6.3, 1.55))
    for i, (name, _, d1, latency) in enumerate(METHODS, start=1):
        ax.scatter(latency, d1, s=58, color="#009E73", edgecolor="black",
                   linewidth=0.6, zorder=3)
        ax.annotate(str(i), (latency, d1), xytext=(0, 0),
                    textcoords="offset points", ha="center", va="center",
                    fontsize=8, color="white", fontweight="bold")
    ax.axvline(33, color="#D55E00", linestyle="--", linewidth=1)
    ax.text(35, 2.47, "33 ms (30 fps)", color="#D55E00", fontsize=9)
    ax.set_xscale("log")
    ax.set_xlabel("Reported inference latency (ms, log scale)")
    ax.set_ylabel("KITTI 2015 D1-all (%, lower is better)")
    ax.set_title("Accuracy and reported latency of representative stereo methods")
    ax.grid(True, alpha=0.25, linestyle=":")
    ax.spines[["top", "right"]].set_visible(False)
    key = "   ".join(f"{i} {m[0]}" for i, m in enumerate(METHODS, 1))
    fig.text(0.5, 0.005, key, ha="center", va="bottom", fontsize=8,
             wrap=True)
    fig.subplots_adjust(bottom=0.42, left=0.13, right=0.98, top=0.86)
    _finish(fig, "fig_2_6b_latency_pareto")


if __name__ == "__main__":
    parameter_plot()
    latency_plot()
