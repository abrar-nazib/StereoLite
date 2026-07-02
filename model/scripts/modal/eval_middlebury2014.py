"""Zero-shot Middlebury 2014 evaluation of the legacy StereoLite checkpoint.

The checkpoint `stereolite_finetune_indoor_best.pth` was trained on
Scene Flow Driving + indoor real-life pseudo-GT (FoundationStereo teacher).
This run measures cross-dataset behaviour on Middlebury 2014 — a dataset
the model has never seen during training.

Methodology:
  - 23 "perfect" rectified scenes from MB2014.
  - Resize each pair to 384x640 (training resolution).
  - Disparity GT scaled by sx = 640/W_native to match the resize.
  - Mask occluded / saturated pixels (PFM stores them as +inf).
  - Report per-scene + aggregate EPE / RMSE / median / bad_{0.5,1,2,3} / D1-all.

T4 GPU, single container, ~3-5 min wall, ~$0.05.

Usage:
    modal run model/scripts/modal/eval_middlebury2014.py::main
"""
from __future__ import annotations

import modal


app = modal.App("eval-middlebury2014")
datasets_vol = modal.Volume.from_name("stereo-datasets")
results_vol = modal.Volume.from_name("widener-results", create_if_missing=True)
PROJECT_ROOT = "/home/abrar/Research/stero_research_claude"

image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("libgl1", "libglib2.0-0", "git", "unzip")
    .pip_install(
        "torch==2.11.0", "torchvision", "numpy<2",
        "opencv-python-headless", "Pillow", "matplotlib",
        "pandas", "ultralytics==8.3.40", "timm", "scipy",
    )
    .add_local_dir(f"{PROJECT_ROOT}/model", "/workspace/model",
                   ignore=["benchmarks/**/*",
                           "teachers/**/*", "kaggle/**/*",
                           "**/__pycache__/**"])  # NOTE: checkpoints kept (we need the ckpt)
    .add_local_python_source("modal")
)


@app.function(
    image=image,
    gpu="T4",
    volumes={"/data": datasets_vol, "/results": results_vol},
    timeout=3600,
)
def run_eval(ckpt_name: str = "stereolite_finetune_indoor_best.pth"):
    import os, sys, time, json, zipfile, glob
    from pathlib import Path
    import numpy as np
    import cv2
    import torch

    sys.path.insert(0, "/workspace/model/designs")
    sys.path.insert(0, "/workspace/model/scripts")
    from StereoLite.model import StereoLite, StereoLiteConfig

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device}  ckpt={ckpt_name}")

    # Load checkpoint (baked into the image under model/checkpoints).
    ckpt_path = f"/workspace/model/checkpoints/{ckpt_name}"
    if not Path(ckpt_path).exists():
        print(f"checkpoint not found at {ckpt_path}; available:")
        for p in glob.glob("/workspace/model/checkpoints/*"):
            print(f"  {p}")
        raise FileNotFoundError(ckpt_path)

    sd_full = torch.load(ckpt_path, map_location=device, weights_only=False)
    sd = sd_full["model"] if "model" in sd_full else sd_full
    # Detect backbone from the state-dict keys.
    has_mn2 = any(k.startswith("fnet.backbone.blocks.") for k in sd)
    backbone = "mobilenet" if has_mn2 else "ghost"
    print(f"detected backbone from ckpt: {backbone}")

    model = StereoLite(StereoLiteConfig(backbone=backbone)).to(device)
    model.load_state_dict(sd, strict=True)
    model.train()  # GroupNorm: train mode is safe at any batch
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"loaded StereoLite (backbone={backbone}), trainable params = {n_params/1e6:.4f} M")

    # Find all "perfect" zips on the master volume.
    zip_root = Path("/data/middlebury/2014")
    perfect_zips = sorted(zip_root.glob("*-perfect.zip"))
    print(f"found {len(perfect_zips)} perfect-set zips")

    # PFM reader — Middlebury format with +inf for invalid pixels.
    def read_pfm(path: str) -> np.ndarray:
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

    # Scratch dir on container's local disk (NOT the volume).
    scratch = Path("/tmp/mb14_scratch")
    scratch.mkdir(exist_ok=True)

    INF_H, INF_W = 384, 640
    per_scene = []
    epes, rmses, medians = [], [], []
    bad05s, bad1s, bad2s, bad3s, d1s = [], [], [], [], []

    diagnostics_printed = 0
    for zp in perfect_zips:
        scene_name = zp.stem.replace("-perfect", "")
        # Extract this scene's zip locally.
        scene_dir = scratch / scene_name
        if scene_dir.exists():
            import shutil
            shutil.rmtree(scene_dir)
        scene_dir.mkdir()
        with zipfile.ZipFile(zp) as zf:
            zf.extractall(scene_dir)

        # Try known MB14 layouts.
        # Standard: <scene>/im0.png, im1.png, disp0GT.pfm
        # Some: <scene>/<scene>-perfect/im0.png, ...
        # Some: <scene>-perfect/im0.png, ...
        im0_candidates = list(scene_dir.rglob("im0.png"))
        if not im0_candidates:
            # Try other naming patterns. List what we DO have.
            if diagnostics_printed < 2:
                files = sorted(p.relative_to(scene_dir).as_posix() for p in scene_dir.rglob("*") if p.is_file())
                print(f"  [{scene_name}] no im0.png. Files in archive ({len(files)}):")
                for f in files[:20]:
                    print(f"      {f}")
                diagnostics_printed += 1
            continue
        im0_path = im0_candidates[0]
        im1_path = im0_path.parent / "im1.png"
        # MB14 disparity GT names: disp0GT.pfm OR disp0.pfm OR disp0-n.pfm
        gt_candidates_local = [
            im0_path.parent / "disp0GT.pfm",
            im0_path.parent / "disp0.pfm",
            im0_path.parent / "disp0-n.pfm",
        ]
        gt_path = next((p for p in gt_candidates_local if p.exists()), None)
        if not im1_path.exists() or gt_path is None:
            if diagnostics_printed < 2:
                files = sorted(p.relative_to(scene_dir).as_posix() for p in scene_dir.rglob("*") if p.is_file())
                print(f"  [{scene_name}] im0 found at {im0_path.relative_to(scene_dir)} but missing im1/disp0GT. Files:")
                for f in files[:20]:
                    print(f"      {f}")
                diagnostics_printed += 1
            continue

        L = cv2.imread(str(im0_path))
        R = cv2.imread(str(im1_path))
        D_native = read_pfm(str(gt_path))
        H_n, W_n = D_native.shape

        # Resize to inference resolution. Disparity scales with width ratio.
        sx = INF_W / W_n
        L_in = cv2.resize(L, (INF_W, INF_H), interpolation=cv2.INTER_AREA)
        R_in = cv2.resize(R, (INF_W, INF_H), interpolation=cv2.INTER_AREA)
        D = cv2.resize(D_native, (INF_W, INF_H), interpolation=cv2.INTER_NEAREST) * sx
        D[~np.isfinite(D) | (D < 0)] = 0.0
        valid = (D > 0).astype(np.float32)
        # Cap extreme disparities (likely error in resize for very-near foreground).
        D_max_reasonable = 192.0
        valid[D > D_max_reasonable] = 0.0
        n_valid = float(valid.sum())
        if n_valid < 100:
            print(f"  [{scene_name}] too few valid pixels after resize, skip")
            continue

        Lt = torch.from_numpy(cv2.cvtColor(L_in, cv2.COLOR_BGR2RGB)).float().permute(2, 0, 1).unsqueeze(0).to(device)
        Rt = torch.from_numpy(cv2.cvtColor(R_in, cv2.COLOR_BGR2RGB)).float().permute(2, 0, 1).unsqueeze(0).to(device)

        t0 = time.time()
        with torch.no_grad():
            pred = model(Lt, Rt)
        if device.type == "cuda":
            torch.cuda.synchronize()
        ms = (time.time() - t0) * 1000

        pred_np = pred.squeeze().cpu().numpy()
        err = np.abs(pred_np - D)
        # Mask metrics by valid pixels.
        epe = float((err * valid).sum() / n_valid)
        rmse = float(np.sqrt((err ** 2 * valid).sum() / n_valid))
        # median over valid pixels only.
        valid_bool = valid > 0
        median = float(np.median(err[valid_bool])) if valid_bool.any() else float("nan")
        bad05 = 100 * float(((err > 0.5) & valid_bool).sum() / n_valid)
        bad1 = 100 * float(((err > 1.0) & valid_bool).sum() / n_valid)
        bad2 = 100 * float(((err > 2.0) & valid_bool).sum() / n_valid)
        bad3 = 100 * float(((err > 3.0) & valid_bool).sum() / n_valid)
        # KITTI D1: > 3 px AND > 5% of GT.
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
        bad3s.append(bad3); d1s.append(d1)

        # Free disk for next scene.
        import shutil
        shutil.rmtree(scene_dir)

    # Aggregate.
    def _mean(xs): return float(np.mean(xs)) if xs else float("nan")
    summary = {
        "ckpt": ckpt_name,
        "n_scenes": len(per_scene),
        "inference_resolution": [INF_H, INF_W],
        "training_data": "Scene Flow Driving + indoor real-life pseudo-GT (FoundationStereo teacher)",
        "test_data": "Middlebury 2014 perfect set (23 scenes, never seen during training)",
        "aggregate": {
            "epe": _mean(epes), "rmse": _mean(rmses), "median": _mean(medians),
            "bad_0.5": _mean(bad05s), "bad_1.0": _mean(bad1s),
            "bad_2.0": _mean(bad2s), "bad_3.0": _mean(bad3s),
            "d1_all": _mean(d1s),
        },
        "per_scene": per_scene,
        "params_train_M": n_params / 1e6,
    }
    print("\n=== AGGREGATE (mean over scenes) ===")
    for k, v in summary["aggregate"].items():
        print(f"  {k}: {v:.4f}")

    # Save report to results volume.
    out_dir = Path("/results/middlebury2014_eval")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / f"mb14_zero_shot_{ckpt_name.replace('.pth','')}.json"
    with open(out_file, "w") as f:
        json.dump(summary, f, indent=2)
    results_vol.commit()
    print(f"\nreport saved to /results/middlebury2014_eval/{out_file.name}")
    return summary


@app.local_entrypoint()
def main():
    print("Zero-shot Middlebury 2014 eval of the legacy StereoLite checkpoint")
    print("(trained on SF Driving + indoor real-life, never seen Middlebury)")
    summary = run_eval.remote()
    print("\n=== DONE ===")
    print(f"  Aggregate over {summary['n_scenes']} scenes:")
    for k, v in summary["aggregate"].items():
        print(f"    {k}: {v:.4f}")
