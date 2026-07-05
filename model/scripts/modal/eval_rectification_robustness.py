"""Rectification-robustness sweep for thesis objective 2.

Evaluates best.pth on a subset of the SceneFlow FT3D test set while the
RIGHT image is shifted vertically by a set of offsets, emulating imperfect
rectification (a residual vertical misalignment between the two views).
The disparity ground truth is unchanged; only the right input moves. All
eight stereo metrics are reported per offset so Chapter 4 can show the
degradation curve. Native 960x540 pad16 axis, matching the headline eval.

Streaming (shard-by-shard, bs-sized windows) to stay memory-safe.

Usage:
    modal run model/scripts/modal/eval_rectification_robustness.py::run
"""
from __future__ import annotations

import modal

app = modal.App("eval-rectification-robustness")
shards_vol = modal.Volume.from_name("sceneflow-shards")
cache_vol = modal.Volume.from_name("stereo-overfit-cache")
results_vol = modal.Volume.from_name("widener-results")
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
              volumes={"/shards": shards_vol, "/cache": cache_vol,
                       "/results": results_vol},
              cpu=8, memory=32768, timeout=2 * 3600, retries=0)
def eval_remote(run_name: str, arch: str, ckpt: str, bs: int,
                max_pairs: int) -> dict:
    import json
    import os
    import pickle
    import sys
    from pathlib import Path

    import numpy as np
    import torch
    import torch.nn.functional as F

    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    os.chdir("/workspace")
    if Path("/cache/yolo26s.pt").exists() and not Path("/workspace/yolo26s.pt").exists():
        os.symlink("/cache/yolo26s.pt", "/workspace/yolo26s.pt")
    sys.path.insert(0, "/workspace/model/scripts")
    sys.path.insert(0, "/workspace/model/designs")

    from train_full_sceneflow import (  # noqa: E402
        decode_record, build_model, _forward_pad16, batchify, stereo_metrics)

    device = "cuda"
    shards_dir = Path("/shards/v1")
    out_dir = Path(f"/results/fulltrain/{run_name}")
    ckpt_path = out_dir / ckpt

    model, cfg = build_model(arch)
    sd = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(sd["model"])
    model.to(device).eval()
    print(f"loaded {ckpt_path} (step {sd.get('step')}); arch={arch}",
          flush=True)

    OFFSETS = [0.0, 0.5, 1.0, 2.0, 4.0]

    def vshift(img: torch.Tensor, s: float) -> torch.Tensor:
        """Shift image rows down by s pixels with replicate padding, via
        grid_sample (handles sub-pixel offsets)."""
        if s == 0.0:
            return img
        B, C, H, W = img.shape
        ys = torch.arange(H, device=img.device, dtype=img.dtype)
        xs = torch.arange(W, device=img.device, dtype=img.dtype)
        gy, gx = torch.meshgrid(ys - s, xs, indexing="ij")
        # normalize to [-1, 1]
        gyn = 2.0 * gy / max(H - 1, 1) - 1.0
        gxn = 2.0 * gx / max(W - 1, 1) - 1.0
        grid = torch.stack([gxn, gyn], dim=-1)[None].expand(B, -1, -1, -1)
        return F.grid_sample(img, grid, mode="bilinear",
                             padding_mode="border", align_corners=True)

    test_shards = sorted(shards_dir.glob("test_ft3d_*.pkl"))
    # accumulate metrics per offset
    agg = {o: [] for o in OFFSETS}
    n_pairs = 0
    with torch.no_grad():
        for sp in test_shards:
            if n_pairs >= max_pairs:
                break
            with open(sp, "rb") as f:
                recs = pickle.load(f)
            for i in range(0, len(recs), bs):
                if n_pairs >= max_pairs:
                    break
                window = [decode_record(r, native=True)
                          for r in recs[i:i + bs]]
                idxs = list(range(len(window)))
                L, R, D, V = batchify(window, idxs, device)
                for o in OFFSETS:
                    Rs = vshift(R, o)
                    pred = _forward_pad16(model, L, Rs)
                    for b in range(pred.shape[0]):
                        agg[o].append(
                            stereo_metrics(pred[b:b+1], D[b:b+1], V[b:b+1]))
                    del Rs, pred
                n_pairs += len(window)
                del window, L, R, D, V
            del recs
            print(f"  {n_pairs} pairs done", flush=True)

    keys = list(agg[OFFSETS[0]][0].keys())
    table = {}
    for o in OFFSETS:
        table[str(o)] = {k: float(np.nanmean([a[k] for a in agg[o]]))
                         for k in keys}
    print("\n=== RECTIFICATION ROBUSTNESS (native axis) ===", flush=True)
    print(f"pairs per offset: {n_pairs}", flush=True)
    for o in OFFSETS:
        m = table[str(o)]
        print(f"  vshift {o:>4} px: EPE={m['epe']:.3f}  bad1={m['bad_1.0']:.2f}"
              f"  bad2={m['bad_2.0']:.2f}  D1={m['d1_all']:.2f}", flush=True)

    report = {"run_name": run_name, "arch": arch, "ckpt": ckpt,
              "axis": "native_960x540_pad16", "n_pairs": n_pairs,
              "offsets_px": OFFSETS, "metrics_by_offset": table}
    rpath = Path(f"/results/fulltrain/{run_name}/rectification_robustness.json")
    rpath.write_text(json.dumps(report, indent=1))
    results_vol.commit()
    return report


@app.local_entrypoint()
def run(run_name: str = "20260704_fullsf_gev4onp_nc",
        arch: str = "gev4_opt_narrow_plane", ckpt: str = "best.pth",
        bs: int = 4, max_pairs: int = 400):
    print(f"rectification-robustness sweep: {run_name}/{ckpt}, "
          f"{max_pairs} pairs, offsets 0/0.5/1/2/4 px")
    out = eval_remote.remote(run_name, arch, ckpt, bs, max_pairs)
    print("\n=== RESULT ===")
    for o in out["offsets_px"]:
        m = out["metrics_by_offset"][str(o)]
        print(f"  {o} px: EPE {m['epe']:.3f}  bad-1 {m['bad_1.0']:.2f}  "
              f"bad-2 {m['bad_2.0']:.2f}  D1 {m['d1_all']:.2f}")
