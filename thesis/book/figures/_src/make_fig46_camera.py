"""Fig 4.6: real-camera qualitative panels. Runs the trained checkpoint
(best.pth, step 53k) on indoor pairs from the low-cost AR0144 stereo rig
(stereo_samples_20260425_104147, 1280x720 rectified) at 384x640 and shows
left image + predicted disparity for four scenes. The rig's factory
rectification is imperfect; the figure shows zero-shot visual
plausibility on real data. Local RTX 3050 inference (~seconds)."""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image

ROOT = Path("/home/abrar/Research/stero_research_claude")
sys.path.insert(0, str(ROOT / "model/scripts"))
sys.path.insert(0, str(ROOT / "model/designs"))
import os
os.chdir(ROOT)  # yolo26s.pt lives at repo root
from train_full_sceneflow import _forward_pad16, build_model  # noqa: E402

RUN = ROOT / "model/benchmarks/20260704_fullsf_gev4onp_nc"
CAM = Path("/media/abrar/AbrarSSD/Datasets/stereo_samples_20260425_104147")
OUT = ROOT / "thesis/book/figures"
PAIRS = ["00038", "01077", "01282", "00195"]
H, W = 384, 640

plt.rcParams.update({"font.family": "DejaVu Serif"})


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, _ = build_model("gev4_opt_narrow_plane")
    ck = torch.load(RUN / "best.pth", map_location=device,
                    weights_only=False)
    model.load_state_dict(ck["model"])
    model.to(device).eval()
    print(f"loaded step {ck.get('step')} on {device}")

    fig, axes = plt.subplots(2, len(PAIRS), figsize=(6.3, 2.4))
    for c, pid in enumerate(PAIRS):
        L = np.array(Image.open(CAM / "left" / f"{pid}.png")
                     .convert("RGB").resize((W, H)))
        R = np.array(Image.open(CAM / "right" / f"{pid}.png")
                     .convert("RGB").resize((W, H)))
        Lt = (torch.from_numpy(L).float().permute(2, 0, 1)[None]
              .to(device) / 255.0)
        Rt = (torch.from_numpy(R).float().permute(2, 0, 1)[None]
              .to(device) / 255.0)
        with torch.no_grad():
            pred = _forward_pad16(model, Lt, Rt)[0, 0].cpu().numpy()
        vmax = np.percentile(pred, 98)
        axes[0, c].imshow(L)
        axes[1, c].imshow(np.clip(pred / vmax, 0, 1), cmap="turbo")
        for r in (0, 1):
            axes[r, c].axis("off")
    axes[0, 0].set_ylabel("left", fontsize=8)
    fig.tight_layout(pad=0.15)
    fig.savefig(OUT / "fig_4_6_camera.pdf", bbox_inches="tight")
    fig.savefig(OUT / "fig_4_6_camera.png", dpi=220, bbox_inches="tight",
                facecolor="white")
    print("saved fig_4_6")


if __name__ == "__main__":
    main()
