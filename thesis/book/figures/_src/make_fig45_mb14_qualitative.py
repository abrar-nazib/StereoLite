"""Fig 4.5: zero-shot Middlebury 2014 qualitative panels. Three easiest
+ two hardest scenes by D1 (per mb14_zero_shot.json): left, GT, ours.
Source images: the eval driver's --save_viz output (turbo, per-scene
shared scale, invalid GT dark)."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[4]
VIZ = ROOT / "model/benchmarks/viz_20260704_fullsf_gev4onp_nc"
RUN = ROOT / "model/benchmarks/20260704_fullsf_gev4onp_nc"
OUT = ROOT / "thesis/book/figures"

plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "Times", "TeX Gyre Termes", "DejaVu Serif"]})


def main():
    rep = json.loads((RUN / "mb14_zero_shot.json").read_text())
    scenes = sorted(rep["per_scene"], key=lambda s: s["d1_all"])
    picks = scenes[:3] + scenes[-2:]

    fig, axes = plt.subplots(len(picks), 3, figsize=(6.0, 1.28 * len(picks)))
    for r, s in enumerate(picks):
        name = s["scene"]
        for c, kind in enumerate(("left", "gt", "pred")):
            im = np.array(Image.open(VIZ / f"{name}_{kind}.png"))[..., ::-1]
            axes[r, c].imshow(im)
            axes[r, c].axis("off")
        axes[r, 0].text(-0.04, 0.5, f"{name}\nD1 {s['d1_all']:.1f}%",
                        transform=axes[r, 0].transAxes, fontsize=7,
                        ha="right", va="center")
        if r == 0:
            for c, t in enumerate(("Left image", "Ground truth", "Prediction")):
                axes[r, c].set_title(t, fontsize=8)
    fig.tight_layout(pad=0.15)
    fig.savefig(OUT / "fig_4_5_mb14_qualitative.pdf", bbox_inches="tight")
    fig.savefig(OUT / "fig_4_5_mb14_qualitative.png", dpi=220,
                bbox_inches="tight", facecolor="white")
    print("saved fig_4_5")


if __name__ == "__main__":
    main()
