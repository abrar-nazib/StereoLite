"""Fig 4.9: accuracy under vertical rectification error. EPE and D1-all
against the right-image vertical offset. Source: the sweep report
rectification_robustness.json (400-pair FT3D-TEST subset, native axis)."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path("/home/abrar/Research/stero_research_claude")
REP = ROOT / "model/benchmarks/20260704_fullsf_gev4onp_nc/rectification_robustness.json"
OUT = ROOT / "thesis/book/figures"
plt.rcParams.update({"font.family": "DejaVu Serif"})
C_EPE, C_D1 = "#0072B2", "#D55E00"


def main():
    rep = json.loads(REP.read_text())
    offs = rep["offsets_px"]
    m = rep["metrics_by_offset"]
    epe = [m[str(o)]["epe"] for o in offs]
    d1 = [m[str(o)]["d1_all"] for o in offs]

    fig, ax = plt.subplots(figsize=(4.6, 2.8))
    ax.plot(offs, epe, "o-", color=C_EPE, lw=1.6, label="EPE (px)")
    ax.set_xlabel("right-image vertical offset (px)", fontsize=9)
    ax.set_ylabel("EPE (px)", fontsize=9, color=C_EPE)
    ax.tick_params(labelsize=8)
    ax2 = ax.twinx()
    ax2.plot(offs, d1, "s--", color=C_D1, lw=1.4, label="D1-all (%)")
    ax2.set_ylabel("D1-all (%)", fontsize=9, color=C_D1)
    ax2.tick_params(labelsize=8)
    ax.axvspan(0, 1.0, color="#009E73", alpha=0.08)
    ax.text(0.5, max(epe) * 0.92, "realistic\nresidual", fontsize=7,
            color="#009E73", ha="center")
    ax.spines[["top"]].set_visible(False)
    ax2.spines[["top"]].set_visible(False)
    fig.tight_layout(pad=0.3)
    fig.savefig(OUT / "fig_4_9_rectification.pdf", bbox_inches="tight")
    fig.savefig(OUT / "fig_4_9_rectification.png", dpi=220,
                bbox_inches="tight", facecolor="white")
    print("saved fig_4_9")


if __name__ == "__main__":
    main()
