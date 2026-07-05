"""Fig 4.1: training dynamics of the 60k full Scene Flow run.

Two panels, designed 1:1 for \\textwidth (~6.3 in):
  (a) training loss (log y) + OneCycle LR (twin axis);
  (b) held-out val EPE and bad-1 vs step, best checkpoint marked at 53k.

Source: model/benchmarks/20260704_fullsf_gev4onp_nc/train.csv.
The run restarted across Modal preemptions, so train.csv contains
overlapping step ranges; dedupe keeps the LAST row per step.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path("/home/abrar/Research/stero_research_claude")
RUN = ROOT / "model/benchmarks/20260704_fullsf_gev4onp_nc"
OUT = ROOT / "thesis/book/figures"
BEST_STEP = 53000

# colorblind-safe pair (blue / vermillion) + neutral grey
C_LOSS = "#0072B2"
C_LR = "#999999"
C_EPE = "#0072B2"
C_BAD1 = "#D55E00"


def main():
    df = pd.read_csv(RUN / "train.csv")
    df = df.drop_duplicates(subset="step", keep="last").sort_values("step")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6.3, 2.5))

    # (a) loss + LR
    ax1.plot(df["step"] / 1000, df["loss"], color=C_LOSS, lw=0.7,
             alpha=0.35, label="_nolegend_")
    smooth = df["loss"].rolling(9, center=True, min_periods=1).mean()
    ax1.plot(df["step"] / 1000, smooth, color=C_LOSS, lw=1.4,
             label="training loss")
    ax1.set_yscale("log")
    ax1.set_xlabel("step (thousands)", fontsize=8)
    ax1.set_ylabel("training loss", fontsize=8, color=C_LOSS)
    ax1.tick_params(labelsize=7)
    ax1b = ax1.twinx()
    ax1b.plot(df["step"] / 1000, df["lr"], color=C_LR, lw=1.1, ls="--",
              label="learning rate")
    ax1b.set_ylabel("learning rate", fontsize=8, color="#666")
    ax1b.tick_params(labelsize=7)
    ax1b.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
    ax1b.yaxis.get_offset_text().set_fontsize(6)
    ax1.set_title("(a) loss and OneCycle schedule", fontsize=8.5)

    # (b) validation metrics
    v = df.dropna(subset=["val_epe"])
    ax2.plot(v["step"] / 1000, v["val_epe"], color=C_EPE, lw=1.3,
             label="val EPE (px)")
    ax2.set_xlabel("step (thousands)", fontsize=8)
    ax2.set_ylabel("val EPE (px)", fontsize=8, color=C_EPE)
    ax2.tick_params(labelsize=7)
    ax2b = ax2.twinx()
    ax2b.plot(v["step"] / 1000, v["val_bad1"], color=C_BAD1, lw=1.1,
              ls=":", label="val bad-1 (%)")
    ax2b.set_ylabel("val bad-1 (%)", fontsize=8, color=C_BAD1)
    ax2b.tick_params(labelsize=7)
    best = v.loc[(v["step"] - BEST_STEP).abs().idxmin()]
    ax2.scatter([best["step"] / 1000], [best["val_epe"]], marker="*",
                s=90, color="#000", zorder=5)
    ax2.annotate(f"best: {best['val_epe']:.3f} px @ {BEST_STEP // 1000}k",
                 (best["step"] / 1000, best["val_epe"]),
                 textcoords="offset points", xytext=(-8, 9),
                 fontsize=7, ha="right")
    ax2.set_title("(b) held-out validation (400 pairs, native axis)",
                  fontsize=8.5)

    for a in (ax1, ax2):
        a.spines[["top"]].set_visible(False)
    for a in (ax1b, ax2b):
        a.spines[["top"]].set_visible(False)

    fig.tight_layout(pad=0.6)
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / "fig_4_1_training_curves.pdf", bbox_inches="tight")
    fig.savefig(OUT / "fig_4_1_training_curves.png", dpi=220,
                bbox_inches="tight", facecolor="white")
    print("saved", OUT / "fig_4_1_training_curves.png")


if __name__ == "__main__":
    main()
