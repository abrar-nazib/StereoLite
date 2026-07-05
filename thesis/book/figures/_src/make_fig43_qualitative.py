"""Fig 4.3: SceneFlow test qualitative grid. Three validation scenes x
(left image, ground truth, prediction at the best checkpoint, absolute
error map). Sources: the run's tracked images."""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

ROOT = Path("/home/abrar/Research/stero_research_claude")
RUN = ROOT / "model/benchmarks/20260704_fullsf_gev4onp_nc/images"
OUT = ROOT / "thesis/book/figures"

SCENES = ["val_00", "val_07", "val_12"]
BEST = 53000

plt.rcParams.update({"font.family": "DejaVu Serif"})


def _disp(p: Path) -> np.ndarray:
    return np.array(Image.open(p)).astype(np.float32) / 256.0


def _rgb(d: np.ndarray, vmax: float) -> np.ndarray:
    rgb = plt.get_cmap("turbo")(np.clip(d / vmax, 0, 1))[..., :3]
    rgb[d <= 0] = 0.12
    return rgb


def main():
    titles = ["Left image", "Ground truth", "Prediction", "Abs. error (px)"]
    fig, axes = plt.subplots(len(SCENES), 4, figsize=(6.3, 1.15 * len(SCENES)))
    last_im = None
    for r, scene in enumerate(SCENES):
        gt = _disp(RUN / scene / "gt.png")
        pred = _disp(RUN / scene / f"step_{BEST:06d}.png")
        vmax = float(np.percentile(gt[gt > 0], 98))
        err = np.abs(pred - gt)
        err[gt <= 0] = 0
        panels = [np.array(Image.open(RUN / scene / "left.png")),
                  _rgb(gt, vmax), _rgb(pred, vmax)]
        for c in range(3):
            axes[r, c].imshow(panels[c])
            axes[r, c].axis("off")
        last_im = axes[r, 3].imshow(err, cmap="magma", vmin=0, vmax=3)
        axes[r, 3].axis("off")
        if r == 0:
            for c, t in enumerate(titles):
                axes[r, c].set_title(t, fontsize=8)
    fig.tight_layout(pad=0.15)
    cbar = fig.colorbar(last_im, ax=axes[:, 3], fraction=0.05, pad=0.02)
    cbar.ax.tick_params(labelsize=6)
    fig.savefig(OUT / "fig_4_3_sceneflow_qualitative.pdf",
                bbox_inches="tight")
    fig.savefig(OUT / "fig_4_3_sceneflow_qualitative.png", dpi=220,
                bbox_inches="tight", facecolor="white")
    print("saved fig_4_3")


if __name__ == "__main__":
    main()
