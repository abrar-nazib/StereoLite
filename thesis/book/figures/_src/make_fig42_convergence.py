"""Fig 4.2: qualitative convergence filmstrip. Two tracked validation
scenes; prediction at five checkpoints + ground truth. Sources: the
run's per-1k tracked images (uint16 disparity * 256)."""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[4]
RUN = ROOT / "model/benchmarks/20260704_fullsf_gev4onp_nc/images"
OUT = ROOT / "thesis/book/figures"

SCENES = ["val_00", "val_12"]
STEPS = [1000, 5000, 15000, 30000, 53000]

plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "Times", "TeX Gyre Termes", "DejaVu Serif"]})


def _load(p: Path) -> np.ndarray:
    return np.array(Image.open(p)).astype(np.float32) / 256.0


def _rgb(d: np.ndarray, vmax: float) -> np.ndarray:
    rgb = plt.get_cmap("turbo")(np.clip(d / vmax, 0, 1))[..., :3]
    rgb[d <= 0] = 0.12
    return rgb


def main():
    ncol = len(STEPS) + 1
    fig, axes = plt.subplots(len(SCENES), ncol,
                             figsize=(6.3, 1.16 * len(SCENES)))
    for r, scene in enumerate(SCENES):
        gt = _load(RUN / scene / "gt.png")
        vmax = float(np.percentile(gt[gt > 0], 98))
        for c, step in enumerate(STEPS):
            ax = axes[r, c]
            ax.imshow(_rgb(_load(RUN / scene / f"step_{step:06d}.png"), vmax))
            ax.axis("off")
            if r == 0:
                ax.set_title(f"{step // 1000}k", fontsize=8)
        ax = axes[r, ncol - 1]
        ax.imshow(_rgb(gt, vmax))
        ax.axis("off")
        if r == 0:
            ax.set_title("GT", fontsize=8, fontweight="bold")
    fig.tight_layout(pad=0.15)
    fig.savefig(OUT / "fig_4_2_convergence.pdf", bbox_inches="tight")
    fig.savefig(OUT / "fig_4_2_convergence.png", dpi=220,
                bbox_inches="tight", facecolor="white")
    print("saved fig_4_2")


if __name__ == "__main__":
    main()
