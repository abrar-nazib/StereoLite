"""Fig 4.4: zero-shot Middlebury 2014 per-scene D1-all bars.

23 perfect-set scenes, sorted by D1, with our aggregate and the two
reference aggregates (LiteAnyStereo, IGEV-Stereo) as horizontal lines.
Reference aggregates come from the repo's matched-protocol reference
evals (CLAUDE.md cross-domain table).

Source: model/benchmarks/20260704_fullsf_gev4onp_nc/mb14_zero_shot.json.
Designed 1:1 for \\textwidth (~6.3 in).
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path("/home/abrar/Research/stero_research_claude")
RUN = ROOT / "model/benchmarks/20260704_fullsf_gev4onp_nc"
OUT = ROOT / "thesis/book/figures"

C_BAR = "#0072B2"
C_OURS = "#000000"
C_LAS = "#D55E00"
C_IGEV = "#009E73"
REF_LAS = 6.9    # LiteAnyStereo aggregate D1-all, same 23-scene protocol
REF_IGEV = 5.0   # IGEV-Stereo aggregate D1-all, same protocol


def main():
    rep = json.loads((RUN / "mb14_zero_shot.json").read_text())
    scenes = sorted(rep["per_scene"], key=lambda s: s["d1_all"])
    names = [s["scene"] for s in scenes]
    d1 = [s["d1_all"] for s in scenes]
    ours = rep["aggregate"]["d1_all"]

    fig, ax = plt.subplots(figsize=(6.3, 2.7))
    ax.bar(range(len(names)), d1, color=C_BAR, width=0.72)
    ax.axhline(ours, color=C_OURS, lw=1.2,
               label=f"ours, mean {ours:.1f}%")
    ax.axhline(REF_LAS, color=C_LAS, lw=1.1, ls="--",
               label=f"LiteAnyStereo, mean {REF_LAS:.1f}%")
    ax.axhline(REF_IGEV, color=C_IGEV, lw=1.1, ls=":",
               label=f"IGEV-Stereo, mean {REF_IGEV:.1f}%")
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=60, ha="right", fontsize=6)
    ax.set_ylabel("D1-all (%)", fontsize=8)
    ax.tick_params(axis="y", labelsize=7)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(fontsize=6.5, frameon=False, loc="upper left")
    fig.tight_layout(pad=0.5)

    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / "fig_4_4_mb14_perscene.pdf", bbox_inches="tight")
    fig.savefig(OUT / "fig_4_4_mb14_perscene.png", dpi=220,
                bbox_inches="tight", facecolor="white")
    print("saved", OUT / "fig_4_4_mb14_perscene.png")


if __name__ == "__main__":
    main()
