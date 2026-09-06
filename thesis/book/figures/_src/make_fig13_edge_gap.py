"""Fig 1.3: the edge-deployment gap. Params (log) vs SceneFlow EPE
scatter with the edge target zone and StereoLite's marker.

Data: review_paper/figures/_data/method_data.py (each value carries a
source breadcrumb; mirrors papers/verified_performance.md). Our point:
2.963 M params, FT3D-TEST EPE 0.78 (native axis; the axis caveat goes in
the caption). Numbered dots + side legend (deck-proven layout), restyled
for print per WRITING_PLAN section 9.
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / "review_paper/figures/_data"))
from method_data import METHODS  # noqa: E402

OUT = ROOT / "thesis/book/figures"
INK = "#1a1a1a"

plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "Times", "TeX Gyre Termes", "DejaVu Serif"]})

KEEP = [
    ("FoundationStereo", "FoundationStereo"), ("DEFOM-Stereo", "DEFOM-Stereo"),
    ("MonSter", "MonSter"),
    ("RAFT-Stereo", "RAFT-Stereo"), ("IGEV-Stereo", "IGEV-Stereo"),
    ("CREStereo", "CREStereo"),
    ("PSMNet", "PSMNet"), ("GwcNet-g", "GwcNet"), ("ACVNet", "ACVNet"),
    ("MobileStereoNet-2D", "MobileStereoNet"), ("CoEx", "CoEx"),
    ("BGNet+", "BGNet+"), ("StereoNet", "StereoNet"),
    ("LightStereo-S", "LightStereo-S"), ("LightStereo-M", "LightStereo-M"),
    ("Fast-FoundationStereo", "Fast-FoundationStereo"),
]
# colorblind-safe family palette (Okabe-Ito)
FAM_COLOR = {"foundation": "#CC79A7", "iterative": "#0072B2",
             "3dcv": "#009E73", "efficient": "#E69F00",
             "efficient_iter": "#56B4E9"}
FAM_LABEL = {"foundation": "foundation model", "iterative": "iterative",
             "3dcv": "3D cost volume", "efficient": "efficient (pre 2024)",
             "efficient_iter": "efficient (2024+)"}


def main():
    fig = plt.figure(figsize=(6.3, 3.2))
    gs = fig.add_gridspec(1, 2, width_ratios=[2.1, 1.0], wspace=0.04)
    ax = fig.add_subplot(gs[0, 0])
    lax = fig.add_subplot(gs[0, 1])
    lax.axis("off")

    entries = []
    for i, (key, lbl) in enumerate(KEEP, start=1):
        m = METHODS.get(key)
        if not m or m.get("sf_epe") is None or m.get("params_m") is None:
            continue
        fam = m["family"]
        ax.scatter(m["params_m"], m["sf_epe"], c=FAM_COLOR[fam], s=130,
                   edgecolor=INK, linewidth=0.5, zorder=3)
        ax.text(m["params_m"], m["sf_epe"], str(i), fontsize=5.5,
                fontweight="bold", ha="center", va="center", zorder=4)
        entries.append((i, lbl, fam))

    # target zone: < 3 M params (edge budget)
    ax.axvspan(0.2, 3.0, color="#0072B2", alpha=0.07, zorder=1)
    ax.text(0.78, 0.34, "edge budget\n< 3 M", fontsize=7, color="#0072B2",
            ha="center")

    # ours
    ax.scatter([2.963], [0.78], marker="*", s=330, c="#D55E00",
               edgecolor=INK, linewidth=0.7, zorder=5)
    ax.annotate("StereoLite (ours)", (2.963, 0.78), fontsize=8,
                fontweight="bold", color="#D55E00",
                textcoords="offset points", xytext=(8, 8))

    ax.set_xscale("log")
    ax.set_xlabel("trainable parameters (M, log scale)", fontsize=8.5)
    ax.set_ylabel("SceneFlow EPE (px)", fontsize=8.5)
    ax.tick_params(labelsize=7.5)
    ax.spines[["top", "right"]].set_visible(False)

    # side legend: numbered names grouped by family
    y = 0.98
    for fam in FAM_COLOR:
        fam_entries = [(n, l) for n, l, f in entries if f == fam]
        if not fam_entries:
            continue
        lax.text(0.02, y, FAM_LABEL[fam], fontsize=7.5, fontweight="bold",
                 color=FAM_COLOR[fam], transform=lax.transAxes, va="top")
        y -= 0.062
        for n, l in fam_entries:
            lax.text(0.07, y, f"{n}. {l}", fontsize=7,
                     transform=lax.transAxes, va="top")
            y -= 0.052
        y -= 0.018

    fig.tight_layout(pad=0.3)
    fig.savefig(OUT / "fig_1_3_edge_gap.pdf", bbox_inches="tight")
    fig.savefig(OUT / "fig_1_3_edge_gap.png", dpi=220, bbox_inches="tight",
                facecolor="white")
    print("saved fig_1_3")


if __name__ == "__main__":
    main()
