"""Fig 3.10: native-crop vs global-resize input protocol, illustrated on
a genuine FT3D frame (thesis_assets). Left: native 960x540 frame with two
sampled 384x640 co-located crop windows. Right: the same frame globally
downscaled to 640x384 (disparity magnitudes shrink with it)."""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle
from PIL import Image

ROOT = Path("/home/abrar/Research/stero_research_claude")
OUT = ROOT / "thesis/book/figures"
IMG = ROOT / "model/benchmarks/thesis_assets/TEST_A_0001_t13_left.png"

plt.rcParams.update({"font.family": "DejaVu Serif"})


def main():
    im = np.array(Image.open(IMG))
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6.3, 2.15),
                                   gridspec_kw={"width_ratios": [1.5, 1.0]})
    ax1.imshow(im)
    for (x, y), c in (((60, 40), "#D55E00"), ((420, 130), "#0072B2")):
        ax1.add_patch(Rectangle((x, y), 640, 384, fill=False, edgecolor=c,
                                lw=1.8))
    ax1.set_title("native 960$\\times$540 + random 384$\\times$640 crops\n"
                  "(disparity keeps its native scale)", fontsize=8)
    ax1.axis("off")

    small = np.array(Image.open(IMG).resize((640, 384)))
    ax2.imshow(small)
    ax2.set_title("global downscale to 640$\\times$384\n"
                  "(disparity shrinks by 2/3)", fontsize=8)
    ax2.axis("off")
    fig.tight_layout(pad=0.3)
    fig.savefig(OUT / "fig_3_10_input_protocol.pdf", bbox_inches="tight")
    fig.savefig(OUT / "fig_3_10_input_protocol.png", dpi=220,
                bbox_inches="tight", facecolor="white")
    print("saved fig_3_10")


if __name__ == "__main__":
    main()
