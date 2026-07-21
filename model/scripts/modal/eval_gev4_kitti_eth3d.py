"""Zero-shot KITTI 2012 / KITTI 2015 / ETH3D eval of the THESIS checkpoint
(gev4_opt_narrow_plane, run 20260704_fullsf_gev4onp_nc, best.pth step 53k).

Completes the zero-shot quartet next to the existing MB14 number (D1 10.9%,
eval_gev4_middlebury2014.py). Protocol is IDENTICAL to that driver so the
four datasets are comparable:
  - resize each pair to 384x640; disparity GT scaled by sx = 640/W_native
  - mask invalid, negative, and > 192 px disparities (after scaling)
  - RGB / 255.0 input, forward through _forward_pad16
  - per-pair + aggregate EPE / RMSE / median / bad_{0.5,1,2,3} / D1-all

Datasets (training splits with public GT, zero-shot: model never saw them):
  - KITTI 2012: 194 pairs, colored_0/colored_1, disp_occ (uint16 png / 256)
  - KITTI 2015: 200 pairs, image_2/image_3,   disp_occ_0 (uint16 png / 256)
  - ETH3D:      27 low-res two-view training pairs, im0/im1 + disp0GT.pfm

Zips/7z stay on the volume; extraction goes to container-local /tmp
(volume v1 inode rule). T4 GPU, single container, ~10 min, ~$0.10.

Usage:
    modal run model/scripts/modal/eval_gev4_kitti_eth3d.py::main
"""
from __future__ import annotations

import modal

app = modal.App("eval-gev4-kitti-eth3d")
datasets_vol = modal.Volume.from_name("stereo-datasets")
cache_vol = modal.Volume.from_name("stereo-overfit-cache")
results_vol = modal.Volume.from_name("widener-results", create_if_missing=True)
PROJECT_ROOT = "/home/abrar/Research/stero_research_claude"

image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("libgl1", "libglib2.0-0", "git", "p7zip-full")
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


def _read_pfm(path: str):
    import numpy as np
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
        return np.flipud(np.reshape(data, shape))


@app.function(image=image, gpu="T4",
              volumes={"/data": datasets_vol, "/cache": cache_vol,
                       "/results": results_vol},
              cpu=8, memory=32768, timeout=3600, retries=0)
def run_eval(run_name: str, arch: str, ckpt: str) -> dict:
    import json
    import os
    import subprocess
    import sys
    import time
    import zipfile
    from pathlib import Path

    import cv2
    import numpy as np
    import torch

    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    os.chdir("/workspace")
    if Path("/cache/yolo26s.pt").exists() and \
            not Path("/workspace/yolo26s.pt").exists():
        os.symlink("/cache/yolo26s.pt", "/workspace/yolo26s.pt")
    sys.path.insert(0, "/workspace/model/scripts")
    sys.path.insert(0, "/workspace/model/designs")

    from train_full_sceneflow import build_model, _forward_pad16  # noqa: E402

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt_path = Path(f"/results/fulltrain/{run_name}/{ckpt}")
    model, cfg = build_model(arch)
    ck = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ck["model"])
    model.to(device).eval()
    best_step = ck.get("step")
    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"device={device}  arch={arch}  step={best_step}  "
          f"params={n_params:.4f} M")

    INF_H, INF_W = 384, 640

    def eval_pair(L, R, D_native, name, stats):
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
            return None
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
        rec = {
            "pair": name,
            "epe": float((err * valid).sum() / n_valid),
            "rmse": float(np.sqrt((err ** 2 * valid).sum() / n_valid)),
            "median": float(np.median(err[vb])),
            "bad_0.5": 100 * float(((err > 0.5) & vb).sum() / n_valid),
            "bad_1.0": 100 * float(((err > 1.0) & vb).sum() / n_valid),
            "bad_2.0": 100 * float(((err > 2.0) & vb).sum() / n_valid),
            "bad_3.0": 100 * float(((err > 3.0) & vb).sum() / n_valid),
            "d1_all": 100 * float(((err > 3.0) & (err > 0.05 * D)
                                   & vb).sum() / n_valid),
            "ms": ms, "W_native": W_n, "H_native": H_n,
        }
        stats.append(rec)
        return rec

    def aggregate(stats):
        keys = ["epe", "rmse", "median", "bad_0.5", "bad_1.0",
                "bad_2.0", "bad_3.0", "d1_all"]
        return {k: float(np.mean([s[k] for s in stats])) for k in keys} \
            if stats else {}

    results = {}

    # ---------------- KITTI (2012 + 2015) ----------------
    KITTI = [
        ("kitti2012", "/data/kitti/data_stereo_flow.zip",
         "training/colored_0", "training/colored_1", "training/disp_occ"),
        ("kitti2015", "/data/kitti/data_scene_flow.zip",
         "training/image_2", "training/image_3", "training/disp_occ_0"),
    ]
    for tag, zpath, ldir, rdir, gdir in KITTI:
        if not Path(zpath).exists():
            print(f"[{tag}] MISSING {zpath}, skipping")
            continue
        scratch = Path(f"/tmp/{tag}")
        scratch.mkdir(exist_ok=True)
        t0 = time.time()
        with zipfile.ZipFile(zpath) as zf:
            wanted = [n for n in zf.namelist()
                      if n.startswith((ldir, rdir, gdir))
                      and n.endswith("_10.png")]
            zf.extractall(scratch, members=wanted)
        print(f"[{tag}] extracted {len(wanted)} files "
              f"in {time.time()-t0:.0f} s")
        stats = []
        gts = sorted((scratch / gdir).glob("*_10.png"))
        for gt_path in gts:
            fname = gt_path.name
            L = cv2.imread(str(scratch / ldir / fname))
            R = cv2.imread(str(scratch / rdir / fname))
            if L is None or R is None:
                continue
            gt_raw = cv2.imread(str(gt_path), cv2.IMREAD_UNCHANGED)
            D_native = gt_raw.astype(np.float32) / 256.0   # 0 = invalid
            eval_pair(L, R, D_native, fname, stats)
        agg = aggregate(stats)
        results[tag] = {"n_pairs": len(stats), "aggregate": agg,
                        "per_pair": stats}
        print(f"[{tag}] {len(stats)} pairs  EPE={agg.get('epe', 0):.3f}  "
              f"D1={agg.get('d1_all', 0):.2f}%")

    # ---------------- ETH3D two-view training ----------------
    eth_imgs = "/data/eth3d/two_view_training.7z"
    eth_gt = "/data/eth3d/two_view_training_gt.7z"
    if Path(eth_imgs).exists() and Path(eth_gt).exists():
        scratch = Path("/tmp/eth3d")
        scratch.mkdir(exist_ok=True)
        for a in (eth_imgs, eth_gt):
            subprocess.check_call(["7z", "x", "-y", a, f"-o{scratch}"],
                                  stdout=subprocess.DEVNULL)
        stats = []
        for gt_path in sorted(scratch.rglob("disp0GT.pfm")):
            scene = gt_path.parent
            L = cv2.imread(str(scene / "im0.png"))
            R = cv2.imread(str(scene / "im1.png"))
            if L is None or R is None:
                continue
            D_native = _read_pfm(str(gt_path)).astype(np.float32).copy()
            rec = eval_pair(L, R, D_native, scene.name, stats)
            if rec:
                print(f"  [eth3d {scene.name:22s}] EPE={rec['epe']:.3f}  "
                      f"bad_1={rec['bad_1.0']:.1f}  D1={rec['d1_all']:.1f}")
        agg = aggregate(stats)
        results["eth3d"] = {"n_pairs": len(stats), "aggregate": agg,
                            "per_pair": stats}
        print(f"[eth3d] {len(stats)} pairs  EPE={agg.get('epe', 0):.3f}  "
              f"D1={agg.get('d1_all', 0):.2f}%")
    else:
        print("[eth3d] archives missing, skipping")

    summary = {
        "run_name": run_name, "arch": arch, "ckpt": ckpt,
        "best_step": best_step, "params_M": n_params,
        "inference_resolution": [INF_H, INF_W],
        "protocol": "identical to eval_gev4_middlebury2014.py "
                    "(384x640, sx-scaled GT, mask >192 px, RGB/255)",
        "training_data": "full SceneFlow native-crop; zero-shot on all sets",
        "datasets": results,
    }
    print("\n=== ZERO-SHOT AGGREGATES ===")
    for tag, r in results.items():
        a = r["aggregate"]
        print(f"  {tag:10s} n={r['n_pairs']:3d}  epe={a['epe']:.3f}  "
              f"rmse={a['rmse']:.3f}  med={a['median']:.3f}  "
              f"bad1={a['bad_1.0']:.2f}  bad2={a['bad_2.0']:.2f}  "
              f"bad3={a['bad_3.0']:.2f}  D1={a['d1_all']:.2f}")

    out_dir = Path("/results/kitti_eth3d_eval")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / f"kitti_eth3d_zero_shot_{run_name}.json"
    out_file.write_text(json.dumps(summary, indent=2))
    results_vol.commit()
    print(f"report saved to /results/kitti_eth3d_eval/{out_file.name}")
    return summary


@app.local_entrypoint()
def main(run_name: str = "20260704_fullsf_gev4onp_nc",
         arch: str = "gev4_opt_narrow_plane", ckpt: str = "best.pth"):
    print(f"Zero-shot KITTI 2012/2015 + ETH3D eval: {run_name} / {ckpt}")
    summary = run_eval.remote(run_name, arch, ckpt)
    print("\n=== DONE ===")
    for tag, r in summary["datasets"].items():
        a = r["aggregate"]
        print(f"  {tag}: n={r['n_pairs']}  EPE={a['epe']:.3f}  "
              f"D1-all={a['d1_all']:.2f}%")
