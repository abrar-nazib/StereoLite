"""Build slide-friendly figures for the pre-defense deck.

Outputs (PNGs in same directory, transparent-friendly cream backdrop):
  research_gap_pareto.png     — params (log) vs Scene Flow EPE, real data,
                                 StereoLite highlighted, target zone marked
  realdata_training.png       — fine-tune learning curve from train_log
  realdata_results.png        — bar chart: SF baseline vs indoor fine-tune
                                 (mean EPE) + per-pair table
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path("/home/abrar/Research/stero_research_claude/"
                              "review_paper/figures/_data")))
from method_data import METHODS  # noqa: E402

# Slide palette (matches the cream + terracotta of the existing deck)
CREAM = "#F4EFE6"
WHITE = "#FFFFFF"
INK = "#1a1a1f"
SUBINK = "#5a5550"
ACCENT = "#C24A1C"        # terracotta
SOFT_RED = "#D9826A"
DOT_BLUE = "#3B6FB6"
DOT_GREEN = "#3F7A48"
DOT_PURPLE = "#7B4393"
DOT_TEAL = "#3F8C8E"

OUT = Path(__file__).resolve().parent
OUT.mkdir(parents=True, exist_ok=True)

mpl.rcParams.update({
    "font.family": "DejaVu Serif",
    "font.size": 11,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.30,
    "grid.linestyle": ":",
    "axes.axisbelow": True,
    "figure.dpi": 200,
})


# ---------------------------------------------------------------------------
# Figure 1: Research gap — params (log) vs Scene Flow EPE
# ---------------------------------------------------------------------------

def _safe_adjust(texts, **kw):
    try:
        from adjustText import adjust_text
        adjust_text(texts, **kw)
    except ImportError:
        pass


def build_research_gap():
    """Number each dot; legend on right side maps numbers to method names.
    Eliminates label overlap without losing identification."""
    # Curated method list. Verified against method_data.py source notes.
    keep = [
        # Foundation
        ("FoundationStereo",      "FoundationStereo"),
        ("DEFOM-Stereo",          "DEFOM-Stereo"),
        ("MonSter",               "MonSter"),
        # Iterative
        ("RAFT-Stereo",           "RAFT-Stereo"),
        ("IGEV-Stereo",           "IGEV-Stereo"),
        ("Selective-IGEV",        "Selective-IGEV"),
        ("CREStereo",             "CREStereo"),
        # 3D cost volume
        ("PSMNet",                "PSMNet"),
        ("GA-Net-deep",           "GA-Net"),
        ("ACVNet",                "ACVNet"),
        ("GwcNet-g",              "GwcNet"),
        ("CFNet",                 "CFNet"),
        # Efficient pre-2024
        ("MobileStereoNet-2D",    "MobileStereoNet"),
        ("CoEx",                  "CoEx"),
        ("BGNet+",                "BGNet+"),
        ("StereoNet",             "StereoNet"),
        # Efficient + foundation aware (2024+)
        ("LightStereo-S",         "LightStereo-S"),
        ("LightStereo-M",         "LightStereo-M"),
        ("DTP-IGEV-S2",           "DTP-IGEV-S2"),
        ("Pip-Stereo (1-iter)",   "Pip-Stereo"),
        ("Fast-FoundationStereo", "Fast-FStereo"),
    ]
    family_color = {
        "foundation":    DOT_PURPLE,
        "iterative":     DOT_BLUE,
        "3dcv":          DOT_TEAL,
        "efficient":     DOT_GREEN,
        "efficient_iter": "#A07A1F",
    }
    family_label = {
        "foundation":    "Foundation models  (mono-prior backed)",
        "iterative":     "Iterative refinement  (10 to 25 M)",
        "3dcv":          "3D cost volume  (5 to 25 M)",
        "efficient":     "Efficient pre 2024  (0.5 to 8 M)",
        "efficient_iter": "Efficient + foundation aware (2024+)",
    }

    # Plot area on the left, legend area on the right.
    fig = plt.figure(figsize=(11.5, 5.6), facecolor=CREAM)
    gs = fig.add_gridspec(1, 2, width_ratios=[2.2, 1.0], wspace=0.05)
    ax = fig.add_subplot(gs[0, 0]); ax.set_facecolor(CREAM)
    lax = fig.add_subplot(gs[0, 1]); lax.set_facecolor(CREAM); lax.axis("off")

    # Scatter — number each dot inside the marker
    seen_fam = set()
    legend_entries: list[tuple[int, str, str, str]] = []   # (num, key, label, family)
    for i, (key, lbl) in enumerate(keep, start=1):
        if key not in METHODS: continue
        m = METHODS[key]
        if m.get("sf_epe") is None or m.get("params_m") is None: continue
        fam = m["family"]
        legend_label = family_label[fam] if fam not in seen_fam else None
        seen_fam.add(fam)
        ax.scatter(m["params_m"], m["sf_epe"],
                    c=family_color[fam], marker="o", s=180,
                    edgecolor=INK, linewidth=0.6,
                    alpha=0.92, zorder=3, label=legend_label)
        # Number inside the dot
        ax.text(m["params_m"], m["sf_epe"], f"{i}",
                 fontsize=7, fontweight="bold",
                 color=WHITE if fam in ("foundation", "iterative", "3dcv")
                              else INK,
                 ha="center", va="center", zorder=4)
        legend_entries.append((i, key, lbl, fam))

    # StereoLite — TWO stars
    OUR_PARAMS = 0.87
    EPE_NOW    = 1.54        # SF Driving 200 val (current checkpoint)
    EPE_FULL   = 0.71        # projected after full Scene Flow pre-train
    # Current — solid orange star
    ax.scatter([OUR_PARAMS], [EPE_NOW],
                c=ACCENT, marker="*", s=540,
                edgecolor=INK, linewidth=1.2, zorder=7,
                label="StereoLite (current)")
    # Projected — outlined cream star
    ax.scatter([OUR_PARAMS], [EPE_FULL],
                c=CREAM, marker="*", s=540,
                edgecolor=ACCENT, linewidth=1.6, zorder=7,
                label="StereoLite (projected)")
    # Connecting arrow
    ax.annotate("", xy=(OUR_PARAMS, EPE_FULL + 0.02),
                xytext=(OUR_PARAMS, EPE_NOW - 0.02),
                arrowprops=dict(arrowstyle="-|>", color=ACCENT,
                                  linewidth=1.4, alpha=0.85),
                zorder=6)
    # Star labels — keep short on the chart.  Numbers go in the side panel.
    ax.text(OUR_PARAMS * 1.4, EPE_NOW, "current",
             fontsize=10, color=ACCENT, fontweight="bold", zorder=8,
             va="center")
    ax.text(OUR_PARAMS * 1.4, EPE_FULL, "projected",
             fontsize=10, color=ACCENT, fontweight="bold", zorder=8,
             style="italic", va="center")

    # Target zone — short label so it cannot overflow into the data
    from matplotlib.patches import Rectangle
    ax.add_patch(Rectangle((0.45, 0.32), 2.5, 0.50,
                            linewidth=1.5, edgecolor=ACCENT,
                            linestyle="--", facecolor="none",
                            alpha=0.85, zorder=2))
    ax.text(0.50, 0.36, "TARGET ZONE",
             fontsize=10, color=ACCENT, fontweight="bold", zorder=3,
             family="monospace")

    # Axes
    ax.set_xscale("log")
    ax.set_xlim(0.4, 600)
    ax.set_ylim(0.30, 2.7)
    ax.set_xlabel("Trainable parameters (M, log scale)",
                   color=INK, fontsize=11)
    ax.set_ylabel("Scene Flow test EPE (px, lower is better)",
                   color=INK, fontsize=11)
    ax.tick_params(colors=SUBINK)
    for s in ax.spines.values():
        s.set_color(SUBINK)

    # Move the family swatch legend OFF the chart and into the side
    # panel so nothing on the chart can be obscured.
    lax.set_xlim(0, 1); lax.set_ylim(0, 1)

    # Section A: StereoLite call out (current vs projected)
    lax.text(0.0, 0.98, "STEREOLITE", fontsize=10, fontweight="bold",
              color=ACCENT, family="monospace", va="top")
    lax.scatter(0.04, 0.92, marker="*", s=260,
                 c=ACCENT, edgecolor=INK, linewidth=0.8,
                 clip_on=False, zorder=5)
    lax.text(0.10, 0.93, "current  ·  0.87 M  ·  1.54 px",
              fontsize=8.5, fontweight="bold", color=ACCENT,
              va="center")
    lax.scatter(0.04, 0.86, marker="*", s=260,
                 c=CREAM, edgecolor=ACCENT, linewidth=1.2,
                 clip_on=False, zorder=5)
    lax.text(0.10, 0.87,
              "projected  ·  full SF pretrain  ·  ~0.71 px",
              fontsize=8.5, color=ACCENT, style="italic", va="center")

    # Section B: family swatch legend
    lax.text(0.0, 0.79, "FAMILIES", fontsize=9, fontweight="bold",
              color=ACCENT, family="monospace", va="top")
    fam_y = 0.74
    fam_lh = 0.038
    for j, fam in enumerate(["foundation", "iterative", "3dcv",
                              "efficient", "efficient_iter"]):
        y = fam_y - j * fam_lh
        lax.scatter(0.04, y, s=85, c=family_color[fam],
                     edgecolor=INK, linewidth=0.5,
                     clip_on=False, zorder=5)
        lax.text(0.10, y, family_label[fam],
                  fontsize=7.5, color=INK, va="center")

    # Section C: numbered method legend (two columns)
    lax.text(0.0, 0.51, "METHODS", fontsize=9, fontweight="bold",
              color=ACCENT, family="monospace", va="top")
    lax.text(0.0, 0.475,
              "(numbers correspond to the chart)",
              fontsize=7, color=SUBINK, style="italic", va="top")
    n = len(legend_entries)
    half = (n + 1) // 2
    line_h = 0.040
    y0 = 0.42
    for col_i, group in enumerate([legend_entries[:half],
                                     legend_entries[half:]]):
        col_x = 0.0 + col_i * 0.50
        for j, (num, key, lbl, fam) in enumerate(group):
            y = y0 - j * line_h
            lax.scatter(col_x + 0.025, y, s=95,
                         c=family_color[fam], edgecolor=INK,
                         linewidth=0.4, clip_on=False, zorder=4)
            lax.text(col_x + 0.025, y, f"{num}",
                      fontsize=6.2, fontweight="bold",
                      ha="center", va="center", zorder=5,
                      color=WHITE if fam in ("foundation", "iterative",
                                              "3dcv") else INK)
            lax.text(col_x + 0.07, y, lbl,
                      fontsize=7.0, color=INK, va="center")

    plt.tight_layout()
    out = OUT / "research_gap_pareto.png"
    plt.savefig(out, bbox_inches="tight", pad_inches=0.10,
                 facecolor=CREAM, dpi=220)
    plt.close()
    print(f"  -> {out}")


# ---------------------------------------------------------------------------
# Figure 2: Real-data fine-tune training curve
# ---------------------------------------------------------------------------

def build_realdata_training():
    progress_csv = Path(
        "/home/abrar/Research/stero_research_claude/model/benchmarks/"
        "stereolite_finetune_indoor_20260426-171158/progress.csv")
    train_csv = Path(
        "/home/abrar/Research/stero_research_claude/model/checkpoints/"
        "stereolite_finetune_indoor_train.csv")

    pdf = pd.read_csv(progress_csv)
    tdf = pd.read_csv(train_csv)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4), facecolor=CREAM)
    for ax in axes:
        ax.set_facecolor(CREAM)

    # Left: per-pair val EPE over training
    ax = axes[0]
    pair_cols = [c for c in pdf.columns if c.startswith("pair_")]
    for c in pair_cols:
        ax.plot(pdf["step"], pdf[c], color=SUBINK, alpha=0.30,
                 linewidth=1.0)
    ax.plot(pdf["step"], pdf["mean_epe"],
             color=ACCENT, linewidth=2.6, label="Mean (8 tracked val)")
    ax.set_xlabel("Training step")
    ax.set_ylabel("Validation EPE (px)")
    ax.set_title("Indoor val EPE per step", color=INK, fontsize=12)
    ax.set_ylim(0, max(2.2, pdf["mean_epe"].max() * 1.05))
    ax.legend(loc="upper right", frameon=True, framealpha=0.9, fontsize=9)
    ax.tick_params(colors=SUBINK)
    for s in ax.spines.values():
        s.set_color(SUBINK)
    # Annotate baseline
    ax.axhline(1.54, color=DOT_BLUE, linestyle="--", linewidth=1.2,
                alpha=0.7)
    ax.text(pdf["step"].iloc[-1] * 0.42, 1.62,
             "Scene Flow baseline · 1.54 px",
             color=DOT_BLUE, fontsize=8.5)

    # Right: training loss
    ax = axes[1]
    ax.plot(tdf["step"], tdf["loss"], color=DOT_GREEN, linewidth=1.8,
             label="Total loss")
    ax.plot(tdf["step"], tdf["l1_final"], color=ACCENT, linewidth=1.6,
             label="L1 on final disparity")
    ax.set_xlabel("Training step")
    ax.set_ylabel("Loss")
    ax.set_yscale("log")
    ax.set_title("Indoor fine-tune loss", color=INK, fontsize=12)
    ax.legend(loc="upper right", frameon=True, framealpha=0.9, fontsize=9)
    ax.tick_params(colors=SUBINK)
    for s in ax.spines.values():
        s.set_color(SUBINK)

    plt.tight_layout()
    out = OUT / "realdata_training.png"
    plt.savefig(out, bbox_inches="tight", pad_inches=0.12,
                 facecolor=CREAM, dpi=220)
    plt.close()
    print(f"  -> {out}")


# ---------------------------------------------------------------------------
# Figure 3: Real-data fine-tune summary
# ---------------------------------------------------------------------------

def build_realdata_results():
    """A clean summary card of the fine-tune outcome."""
    fig, ax = plt.subplots(figsize=(11, 4.6), facecolor=CREAM)
    ax.set_facecolor(CREAM)
    ax.axis("off")

    # Two-column metric panel
    ax.text(0.02, 0.92, "INDOOR FINE-TUNE  ·  FoundationStereo pseudo-GT",
             fontsize=11.5, color=ACCENT, fontweight="bold",
             family="monospace", transform=ax.transAxes)

    ax.text(0.02, 0.80,
             "Teacher · FoundationStereo (CVPR 2025, 215 M params)",
             fontsize=10.5, color=SUBINK, transform=ax.transAxes)
    ax.text(0.02, 0.74,
             "Pairs collected · 1,587   |   clean after quality filter · 997",
             fontsize=10.5, color=SUBINK, transform=ax.transAxes)
    ax.text(0.02, 0.68,
             "Random val held-out · 50 pairs   |   8 tracked panels",
             fontsize=10.5, color=SUBINK, transform=ax.transAxes)

    # Big numbers
    big_x = [0.06, 0.30, 0.56, 0.78]
    big_lbl = ["BASELINE\nSF Driving 200 val",
                "INDOOR\n50 val",
                "BEST 3 PAIRS\n(slide demo)",
                "FINE-TUNE\nsteps · wall-clock"]
    big_val = ["1.54 px", "0.515 px", "0.258 px", "9000 · 1 h 35 m"]
    big_units = ["mean EPE", "mean EPE",
                  "min EPE",
                  ""]
    big_color = [DOT_BLUE, ACCENT, DOT_GREEN, SUBINK]
    for x, lbl, val, units, c in zip(big_x, big_lbl, big_val, big_units,
                                       big_color):
        ax.text(x, 0.50, lbl, fontsize=8.5, color=SUBINK,
                 family="monospace", transform=ax.transAxes)
        ax.text(x, 0.30, val, fontsize=22, color=c,
                 fontweight="bold", transform=ax.transAxes)
        ax.text(x, 0.18, units, fontsize=9, color=SUBINK,
                 transform=ax.transAxes)

    # Italic note
    ax.text(0.02, 0.06,
             "Indoor EPE is roughly 3× lower than the synthetic-trained "
             "baseline, despite training on a ~1000-pair pseudo-GT set.",
             fontsize=10.5, color=ACCENT, style="italic",
             transform=ax.transAxes)

    plt.tight_layout()
    out = OUT / "realdata_results.png"
    plt.savefig(out, bbox_inches="tight", pad_inches=0.20,
                 facecolor=CREAM, dpi=220)
    plt.close()
    print(f"  -> {out}")


if __name__ == "__main__":
    build_research_gap()
    build_realdata_training()
    build_realdata_results()
