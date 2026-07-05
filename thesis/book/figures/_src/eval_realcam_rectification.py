"""Real-camera rectification-tolerance eval.

The project's stereo rig is a low-cost (~$45) AR0144 camera whose
rectification is imperfect. We run the trained model on its captured
pairs and measure agreement against the FoundationStereo teacher's
pseudo-disparity on the same pairs. Low error on physically
imperfectly-rectified real data corroborates the synthetic
vertical-shift sweep: the model tolerates the residual misalignment a
cheap rig produces.

Protocol mirrors the Middlebury zero-shot driver: model runs at
384x640; the native 1280-wide pseudo-GT is resized to the inference
axis and scaled by sx = 640/1280 = 0.5; metrics on valid pixels only.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

ROOT = Path("/home/abrar/Research/stero_research_claude")
sys.path.insert(0, str(ROOT / "model/scripts"))
sys.path.insert(0, str(ROOT / "model/designs"))
import os
os.chdir(ROOT)
from train_full_sceneflow import _forward_pad16, build_model  # noqa: E402

RUN = ROOT / "model/benchmarks/20260704_fullsf_gev4onp_nc"
DS = Path("/media/abrar/AbrarSSD/Datasets/stereo_samples_20260425_104147")
H, W = 384, 640
SX = W / 1280.0  # native width 1280 -> inference width 640


def load_model(device):
    model, _ = build_model("gev4_opt_narrow_plane")
    ck = torch.load(RUN / "best.pth", map_location=device, weights_only=False)
    sd = ck.get("model", ck.get("state_dict", ck))
    model.load_state_dict(sd, strict=False)
    model.eval().to(device)
    return model


def resize_disp(disp, w, h, sx):
    d = Image.fromarray(disp).resize((w, h), Image.NEAREST)
    return np.asarray(d, dtype=np.float32) * sx


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_model(device)
    pairs = [p.strip() for p in (DS / "clean_pairs.txt").read_text().split()
             if p.strip()]
    print(f"clean pairs: {len(pairs)}")

    epes, bad1, bad2, d1 = [], [], [], []
    for i, pid in enumerate(pairs):
        Lp = DS / "left" / f"{pid}.png"
        Rp = DS / "right" / f"{pid}.png"
        Gp = DS / "disp_pseudo" / f"{pid}.npy"
        if not (Lp.exists() and Rp.exists() and Gp.exists()):
            continue
        L = np.array(Image.open(Lp).convert("RGB").resize((W, H)))
        R = np.array(Image.open(Rp).convert("RGB").resize((W, H)))
        Lt = torch.from_numpy(L).float().permute(2, 0, 1)[None].to(device) / 255
        Rt = torch.from_numpy(R).float().permute(2, 0, 1)[None].to(device) / 255
        with torch.no_grad():
            pred = _forward_pad16(model, Lt, Rt)[0, 0].cpu().numpy()
        try:
            gt = np.load(Gp).astype(np.float32)
        except Exception:
            continue
        gt = resize_disp(gt, W, H, SX)
        valid = (gt > 0.5) & (gt < 192.0) & np.isfinite(gt)
        if valid.sum() < 1000:
            continue
        err = np.abs(pred - gt)[valid]
        gtv = gt[valid]
        epes.append(err.mean())
        bad1.append((err > 1.0).mean() * 100)
        bad2.append((err > 2.0).mean() * 100)
        d1.append(((err > 3.0) & (err > 0.05 * gtv)).mean() * 100)
        if (i + 1) % 100 == 0:
            print(f"  {i+1}/{len(pairs)}  running EPE={np.mean(epes):.3f}")

    print("\n=== Real-camera agreement vs FoundationStereo teacher ===")
    print(f"pairs scored : {len(epes)}")
    print(f"EPE   : {np.mean(epes):.3f}")
    print(f"bad-1 : {np.mean(bad1):.2f}%")
    print(f"bad-2 : {np.mean(bad2):.2f}%")
    print(f"D1-all: {np.mean(d1):.2f}%")
    out = {
        "n": len(epes),
        "epe": float(np.mean(epes)),
        "bad1": float(np.mean(bad1)),
        "bad2": float(np.mean(bad2)),
        "d1": float(np.mean(d1)),
    }
    import json
    (ROOT / "model/benchmarks/thesis_reconstruction/realcam_eval.json").write_text(
        json.dumps(out, indent=2))
    print("saved realcam_eval.json")


if __name__ == "__main__":
    main()
