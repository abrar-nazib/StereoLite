"""IGEV-Stereo zero-shot Middlebury 2014 reference run.

Loads the official IGEV-Stereo Scene-Flow-trained checkpoint
(pretrained_models/sceneflow/sceneflow.pth, 12.6 M params, never seen
Middlebury) and evaluates on all 23 perfect-set MB14 scenes at 384x640.

This is the literature-side reference: how does a published iterative
edge-stereo method (IGEV-Stereo) generalize from SF Driving to MB14
under the SAME eval protocol we used for our chassis (legacy ckpt 5.53
EPE / 40.1% D1-all).

Usage:
    modal run model/scripts/modal/eval_igev_middlebury2014.py::main
"""
from __future__ import annotations

import modal


app = modal.App("eval-igev-mb14")
datasets_vol = modal.Volume.from_name("stereo-datasets")
results_vol = modal.Volume.from_name("widener-results", create_if_missing=True)
PROJECT_ROOT = "/home/abrar/Research/stero_research_claude"

image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("libgl1", "libglib2.0-0", "git", "unzip")
    .pip_install(
        "torch==2.11.0", "torchvision", "numpy<2",
        "opencv-python-headless", "Pillow", "matplotlib",
        "pandas", "timm==0.5.4", "scipy", "tqdm",
        # IGEV-Stereo's extractor.py accesses model.act1, model.bn1 directly,
        # which only works on timm < 1.0 (older flat-attribute API).
    )
    .add_local_dir(f"{PROJECT_ROOT}/model/scripts/modal/igev_stereo_repo",
                   "/workspace/igev_repo",
                   ignore=["**/__pycache__/**", "**/*.png", "**/*.jpg"])
)


@app.function(
    image=image,
    gpu="T4",
    volumes={"/data": datasets_vol, "/results": results_vol},
    timeout=3600,
)
def run_eval(iters: int = 16):
    import os, sys, time, json, zipfile, glob, types, shutil
    from pathlib import Path
    import numpy as np
    import cv2
    import torch

    sys.path.insert(0, "/workspace/igev_repo/core")
    sys.path.insert(0, "/workspace/igev_repo")
    from igev_stereo import IGEVStereo
    from utils.utils import InputPadder

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device}  iters={iters}")

    # IGEV-Stereo args (defaults match the official demo_imgs.py).
    args = types.SimpleNamespace(
        restore_ckpt="/results/igev_pretrained/sceneflow.pth",
        mixed_precision=False,
        precision_dtype="float32",
        valid_iters=iters,
        hidden_dims=[128, 128, 128],
        corr_levels=2,
        corr_radius=4,
        n_downsample=2,
        n_gru_layers=3,
        max_disp=192,
    )

    print(f"loading IGEV-Stereo from {args.restore_ckpt}")
    model = IGEVStereo(args)
    sd = torch.load(args.restore_ckpt, map_location="cpu", weights_only=False)
    # The checkpoint was saved under DataParallel — keys prefixed with "module."
    if any(k.startswith("module.") for k in sd):
        sd = {k.replace("module.", "", 1): v for k, v in sd.items()}
    model.load_state_dict(sd, strict=True)
    model = model.to(device).eval()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"loaded IGEV-Stereo, params = {n_params/1e6:.4f} M")

    # PFM reader for MB14 GT.
    def read_pfm(path):
        with open(path, "rb") as f:
            header = f.readline().decode("latin-1").rstrip()
            assert header in ("Pf", "PF"), f"bad PFM header {header}"
            color = (header == "PF")
            dim_line = f.readline().decode("latin-1")
            while dim_line.startswith("#"):
                dim_line = f.readline().decode("latin-1")
            w, h = (int(x) for x in dim_line.strip().split())
            scale = float(f.readline().decode("latin-1").rstrip())
            endian = "<" if scale < 0 else ">"
            data = np.fromfile(f, endian + "f")
            shape = (h, w, 3) if color else (h, w)
            data = np.reshape(data, shape)
            data = np.flipud(data)
        return data

    zip_root = Path("/data/middlebury/2014")
    perfect_zips = sorted(zip_root.glob("*-perfect.zip"))
    print(f"found {len(perfect_zips)} perfect-set zips")

    scratch = Path("/tmp/mb14_scratch_igev")
    scratch.mkdir(exist_ok=True)

    INF_H, INF_W = 384, 640
    per_scene = []
    epes, rmses, medians = [], [], []
    bad05s, bad1s, bad2s, bad3s, d1s = [], [], [], [], []
    inf_ms = []

    for zp in perfect_zips:
        scene_name = zp.stem.replace("-perfect", "")
        scene_dir = scratch / scene_name
        if scene_dir.exists():
            shutil.rmtree(scene_dir)
        scene_dir.mkdir()
        with zipfile.ZipFile(zp) as zf:
            zf.extractall(scene_dir)

        im0_candidates = list(scene_dir.rglob("im0.png"))
        if not im0_candidates:
            shutil.rmtree(scene_dir)
            continue
        im0_path = im0_candidates[0]
        im1_path = im0_path.parent / "im1.png"
        gt_path = next((p for p in [
            im0_path.parent / "disp0GT.pfm",
            im0_path.parent / "disp0.pfm",
        ] if p.exists()), None)
        if not im1_path.exists() or gt_path is None:
            shutil.rmtree(scene_dir)
            continue

        L = cv2.imread(str(im0_path))
        R = cv2.imread(str(im1_path))
        D_native = read_pfm(str(gt_path))
        H_n, W_n = D_native.shape

        sx = INF_W / W_n
        L_in = cv2.resize(L, (INF_W, INF_H), interpolation=cv2.INTER_AREA)
        R_in = cv2.resize(R, (INF_W, INF_H), interpolation=cv2.INTER_AREA)
        D = cv2.resize(D_native, (INF_W, INF_H), interpolation=cv2.INTER_NEAREST) * sx
        D[~np.isfinite(D) | (D < 0)] = 0.0
        valid = (D > 0).astype(np.float32)
        valid[D > 192.0] = 0.0
        n_valid = float(valid.sum())
        if n_valid < 100:
            shutil.rmtree(scene_dir)
            continue

        # IGEV expects RGB float tensors; padded to divis_by=32 (already is).
        Lt = torch.from_numpy(cv2.cvtColor(L_in, cv2.COLOR_BGR2RGB)).permute(2, 0, 1).unsqueeze(0).float().to(device)
        Rt = torch.from_numpy(cv2.cvtColor(R_in, cv2.COLOR_BGR2RGB)).permute(2, 0, 1).unsqueeze(0).float().to(device)
        padder = InputPadder(Lt.shape, divis_by=32)
        Lt_p, Rt_p = padder.pad(Lt, Rt)

        t0 = time.time()
        with torch.no_grad():
            disp = model(Lt_p, Rt_p, iters=args.valid_iters, test_mode=True)
        if device.type == "cuda":
            torch.cuda.synchronize()
        ms = (time.time() - t0) * 1000
        disp = padder.unpad(disp).squeeze().cpu().numpy()

        err = np.abs(disp - D)
        valid_bool = valid > 0
        epe = float((err * valid).sum() / n_valid)
        rmse = float(np.sqrt((err ** 2 * valid).sum() / n_valid))
        median = float(np.median(err[valid_bool])) if valid_bool.any() else float("nan")
        bad05 = 100 * float(((err > 0.5) & valid_bool).sum() / n_valid)
        bad1 = 100 * float(((err > 1.0) & valid_bool).sum() / n_valid)
        bad2 = 100 * float(((err > 2.0) & valid_bool).sum() / n_valid)
        bad3 = 100 * float(((err > 3.0) & valid_bool).sum() / n_valid)
        d1_mask = (err > 3.0) & (err > 0.05 * D) & valid_bool
        d1 = 100 * float(d1_mask.sum() / n_valid)

        print(f"  [{scene_name:14s}] EPE={epe:.3f}  bad_0.5={bad05:.1f}  bad_1={bad1:.1f}  bad_2={bad2:.1f}  bad_3={bad3:.1f}  D1={d1:.1f}  med={median:.3f}  ms={ms:.1f}  W_native={W_n}")

        per_scene.append({
            "scene": scene_name,
            "epe": epe, "rmse": rmse, "median": median,
            "bad_0.5": bad05, "bad_1.0": bad1, "bad_2.0": bad2,
            "bad_3.0": bad3, "d1_all": d1, "ms": ms,
            "W_native": W_n, "H_native": H_n,
        })
        epes.append(epe); rmses.append(rmse); medians.append(median)
        bad05s.append(bad05); bad1s.append(bad1); bad2s.append(bad2)
        bad3s.append(bad3); d1s.append(d1); inf_ms.append(ms)

        shutil.rmtree(scene_dir)

    def _mean(xs): return float(np.mean(xs)) if xs else float("nan")
    summary = {
        "model": "IGEV-Stereo",
        "ckpt": "sceneflow.pth (official)",
        "params_M": n_params / 1e6,
        "valid_iters": iters,
        "n_scenes": len(per_scene),
        "inference_resolution": [INF_H, INF_W],
        "training_data": "Scene Flow (full FlyingThings3D + Driving + Monkaa)",
        "test_data": "Middlebury 2014 perfect set, never seen during training",
        "aggregate": {
            "epe": _mean(epes), "rmse": _mean(rmses), "median": _mean(medians),
            "bad_0.5": _mean(bad05s), "bad_1.0": _mean(bad1s),
            "bad_2.0": _mean(bad2s), "bad_3.0": _mean(bad3s),
            "d1_all": _mean(d1s), "ms_mean": _mean(inf_ms),
        },
        "per_scene": per_scene,
    }
    print("\n=== AGGREGATE (mean over scenes) ===")
    for k, v in summary["aggregate"].items():
        print(f"  {k}: {v:.4f}")

    out_dir = Path("/results/middlebury2014_eval")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / f"mb14_zero_shot_igev_stereo_iter{iters}.json"
    with open(out_file, "w") as f:
        json.dump(summary, f, indent=2)
    results_vol.commit()
    print(f"\nreport saved to {out_file}")
    return summary


@app.local_entrypoint()
def main():
    print("IGEV-Stereo zero-shot Middlebury 2014 reference (T4, 16 iters)")
    summary = run_eval.remote(iters=16)
    print("\n=== DONE ===")
    print(f"  Aggregate over {summary['n_scenes']} scenes:")
    for k, v in summary["aggregate"].items():
        print(f"    {k}: {v:.4f}")
