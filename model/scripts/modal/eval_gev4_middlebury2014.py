"""Zero-shot Middlebury 2014 eval of the THESIS checkpoint
(gev4_opt_narrow_plane, run 20260704_fullsf_gev4onp_nc, best.pth step 53k).

Cross-domain gate: the model trained on full SceneFlow (FT3D-TRAIN + Monkaa +
Driving, native-crop), never saw Middlebury. This slots our number into the existing MB14 reference
table (legacy chassis 40.1% D1-all / LiteAnyStereo 6.9% / IGEV 5.0%).

Protocol is IDENTICAL to eval_middlebury2014.py (the legacy driver that
produced the 40.1% number) so the comparison is apples-to-apples:
  - 23 "perfect" rectified scenes.
  - resize each pair to 384x640; disparity GT scaled by sx = 640/W_native.
  - mask invalid (PFM +inf), negative, and > 192 px disparities.
  - per-scene + aggregate EPE / RMSE / median / bad_{0.5,1,2,3} / D1-all.

The ONLY differences vs the legacy driver:
  - model built via train_full_sceneflow.build_model(arch) (not StereoLite);
  - input fed as RGB / 255.0 (the new chassis's normalization), not raw 0-255;
  - forward through _forward_pad16 (no-op at 384x640, kept for parity);
  - checkpoint read from widener-results:/fulltrain/<run>/best.pth (on the
    volume, not baked into the image).

T4 GPU, single container, ~3-5 min wall, ~$0.05.

Usage:
    modal run model/scripts/modal/eval_gev4_middlebury2014.py::main
"""
from __future__ import annotations

import modal

app = modal.App("eval-gev4-middlebury2014")
datasets_vol = modal.Volume.from_name("stereo-datasets")
cache_vol = modal.Volume.from_name("stereo-overfit-cache")
results_vol = modal.Volume.from_name("widener-results", create_if_missing=True)
PROJECT_ROOT = "/home/abrar/Research/stero_research_claude"

image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("libgl1", "libglib2.0-0", "git")
    .pip_install(
        "torch==2.11.0", "torchvision", "numpy<2",
        "opencv-python-headless", "Pillow", "matplotlib",
        "pandas", "ultralytics==8.3.40", "timm", "scipy", "zstandard",
    )
    .add_local_dir(f"{PROJECT_ROOT}/model", "/workspace/model",
                   ignore=["benchmarks/**/*", "checkpoints/*",
                           "teachers/**/*", "kaggle/**/*",
                           "**/__pycache__/**"])
)


@app.function(image=image, gpu="T4",
              volumes={"/data": datasets_vol, "/cache": cache_vol,
                       "/results": results_vol},
              cpu=8, memory=32768, timeout=3600, retries=0)
def run_eval(run_name: str, arch: str, ckpt: str,
             save_viz: bool = False) -> dict:
    import os, sys, time, json, zipfile, glob, shutil
    from pathlib import Path
    import numpy as np
    import cv2
    import torch

    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    os.chdir("/workspace")
    # gev4 encoder builds a YOLO26s backbone -> needs yolo26s.pt in cwd.
    if Path("/cache/yolo26s.pt").exists() and not Path("/workspace/yolo26s.pt").exists():
        os.symlink("/cache/yolo26s.pt", "/workspace/yolo26s.pt")
    sys.path.insert(0, "/workspace/model/scripts")
    sys.path.insert(0, "/workspace/model/designs")

    # reuse the trainer's EXACT model builder + forward.
    from train_full_sceneflow import build_model, _forward_pad16  # noqa: E402

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt_path = Path(f"/results/fulltrain/{run_name}/{ckpt}")
    print(f"device={device}  run={run_name}  arch={arch}  ckpt={ckpt_path}")

    model, cfg = build_model(arch)
    ck = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ck["model"])
    model.to(device).eval()
    best_step = ck.get("step")
    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"loaded step {best_step}; params={n_params:.4f} M")

    zip_root = Path("/data/middlebury/2014")
    perfect_zips = sorted(zip_root.glob("*-perfect.zip"))
    print(f"found {len(perfect_zips)} perfect-set zips")

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

    scratch = Path("/tmp/mb14_scratch")
    scratch.mkdir(exist_ok=True)

    INF_H, INF_W = 384, 640
    per_scene = []
    epes, rmses, medians = [], [], []
    bad05s, bad1s, bad2s, bad3s, d1s = [], [], [], [], []

    diagnostics_printed = 0
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
            if diagnostics_printed < 2:
                files = sorted(p.relative_to(scene_dir).as_posix()
                               for p in scene_dir.rglob("*") if p.is_file())
                print(f"  [{scene_name}] no im0.png. Files ({len(files)}):")
                for f in files[:20]:
                    print(f"      {f}")
                diagnostics_printed += 1
            continue
        im0_path = im0_candidates[0]
        im1_path = im0_path.parent / "im1.png"
        gt_candidates = [im0_path.parent / n for n in
                         ("disp0GT.pfm", "disp0.pfm", "disp0-n.pfm")]
        gt_path = next((p for p in gt_candidates if p.exists()), None)
        if not im1_path.exists() or gt_path is None:
            print(f"  [{scene_name}] missing im1/disp0GT, skip")
            continue

        L = cv2.imread(str(im0_path))
        R = cv2.imread(str(im1_path))
        D_native = read_pfm(str(gt_path))
        H_n, W_n = D_native.shape

        sx = INF_W / W_n
        L_in = cv2.resize(L, (INF_W, INF_H), interpolation=cv2.INTER_AREA)
        R_in = cv2.resize(R, (INF_W, INF_H), interpolation=cv2.INTER_AREA)
        D = cv2.resize(D_native, (INF_W, INF_H),
                       interpolation=cv2.INTER_NEAREST) * sx
        D[~np.isfinite(D) | (D < 0)] = 0.0
        valid = (D > 0).astype(np.float32)
        valid[D > 192.0] = 0.0
        n_valid = float(valid.sum())
        if n_valid < 100:
            print(f"  [{scene_name}] too few valid pixels, skip")
            continue

        # RGB / 255.0 -- the new chassis's normalization (batchify at
        # overfit_efficiency_ablation.py:239 feeds RGB float / 255).
        Lt = (torch.from_numpy(cv2.cvtColor(L_in, cv2.COLOR_BGR2RGB))
              .float().permute(2, 0, 1).unsqueeze(0).to(device) / 255.0)
        Rt = (torch.from_numpy(cv2.cvtColor(R_in, cv2.COLOR_BGR2RGB))
              .float().permute(2, 0, 1).unsqueeze(0).to(device) / 255.0)

        t0 = time.time()
        with torch.no_grad():
            pred = _forward_pad16(model, Lt, Rt)
        if device.type == "cuda":
            torch.cuda.synchronize()
        ms = (time.time() - t0) * 1000

        pred_np = pred.squeeze().cpu().numpy()
        err = np.abs(pred_np - D)
        vb = valid > 0
        epe = float((err * valid).sum() / n_valid)
        rmse = float(np.sqrt((err ** 2 * valid).sum() / n_valid))
        median = float(np.median(err[vb])) if vb.any() else float("nan")
        bad05 = 100 * float(((err > 0.5) & vb).sum() / n_valid)
        bad1 = 100 * float(((err > 1.0) & vb).sum() / n_valid)
        bad2 = 100 * float(((err > 2.0) & vb).sum() / n_valid)
        bad3 = 100 * float(((err > 3.0) & vb).sum() / n_valid)
        d1 = 100 * float(((err > 3.0) & (err > 0.05 * D) & vb).sum() / n_valid)

        print(f"  [{scene_name:14s}] EPE={epe:.3f}  bad_1={bad1:.1f}  "
              f"bad_2={bad2:.1f}  D1={d1:.1f}  med={median:.3f}  "
              f"ms={ms:.1f}  W_native={W_n}")

        if save_viz:
            # left + turbo GT + turbo prediction, shared per-scene scale
            viz_dir = Path(f"/results/middlebury2014_eval/viz_{run_name}")
            viz_dir.mkdir(parents=True, exist_ok=True)
            vmax = max(float(np.percentile(D[valid > 0], 98)), 1e-6)

            def _turbo(x):
                n = np.clip(x / vmax, 0, 1)
                t = (cv2.applyColorMap((n * 255).astype(np.uint8),
                                       cv2.COLORMAP_TURBO))
                t[valid <= 0] = 30
                return t

            cv2.imwrite(str(viz_dir / f"{scene_name}_left.png"), L_in)
            cv2.imwrite(str(viz_dir / f"{scene_name}_gt.png"), _turbo(D))
            cv2.imwrite(str(viz_dir / f"{scene_name}_pred.png"),
                        cv2.applyColorMap(
                            (np.clip(pred_np / vmax, 0, 1) * 255
                             ).astype(np.uint8), cv2.COLORMAP_TURBO))

        per_scene.append({
            "scene": scene_name, "epe": epe, "rmse": rmse, "median": median,
            "bad_0.5": bad05, "bad_1.0": bad1, "bad_2.0": bad2,
            "bad_3.0": bad3, "d1_all": d1, "ms": ms,
            "W_native": W_n, "H_native": H_n})
        epes.append(epe); rmses.append(rmse); medians.append(median)
        bad05s.append(bad05); bad1s.append(bad1); bad2s.append(bad2)
        bad3s.append(bad3); d1s.append(d1)

        shutil.rmtree(scene_dir)

    def _mean(xs): return float(np.mean(xs)) if xs else float("nan")
    summary = {
        "run_name": run_name, "arch": arch, "ckpt": ckpt,
        "best_step": best_step, "params_M": n_params,
        "n_scenes": len(per_scene),
        "inference_resolution": [INF_H, INF_W],
        "training_data": "full SceneFlow native-crop (FT3D-TRAIN + Monkaa + Driving); zero-shot on MB14",
        "test_data": "Middlebury 2014 perfect set (23 scenes, unseen)",
        "aggregate": {
            "epe": _mean(epes), "rmse": _mean(rmses), "median": _mean(medians),
            "bad_0.5": _mean(bad05s), "bad_1.0": _mean(bad1s),
            "bad_2.0": _mean(bad2s), "bad_3.0": _mean(bad3s),
            "d1_all": _mean(d1s)},
        "per_scene": per_scene}
    print("\n=== AGGREGATE (mean over scenes) ===")
    for k, v in summary["aggregate"].items():
        print(f"  {k}: {v:.4f}")

    out_dir = Path("/results/middlebury2014_eval")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / f"mb14_zero_shot_{run_name}.json"
    out_file.write_text(json.dumps(summary, indent=2))
    results_vol.commit()
    print(f"\nreport saved to /results/middlebury2014_eval/{out_file.name}")
    return summary


@app.local_entrypoint()
def main(run_name: str = "20260704_fullsf_gev4onp_nc",
         arch: str = "gev4_opt_narrow_plane", ckpt: str = "best.pth",
         save_viz: bool = False):
    print(f"Zero-shot MB14 eval of thesis checkpoint: {run_name} / {ckpt}")
    print("(trained on full SceneFlow: FT3D+Monkaa+Driving, never saw Middlebury)")
    summary = run_eval.remote(run_name, arch, ckpt, save_viz)
    print("\n=== DONE ===")
    print(f"  Aggregate over {summary['n_scenes']} scenes "
          f"(step {summary['best_step']}, {summary['params_M']:.3f} M):")
    for k, v in summary["aggregate"].items():
        print(f"    {k}: {v:.4f}")
