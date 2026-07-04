"""Streaming full FT3D-TEST (4,370-pair) eval on best.pth — the PUBLISHABLE
SceneFlow test number.

Why this exists: the in-run final eval in train_full_sceneflow.py called
load_test_pairs(..., keys=None), which decoded ALL 4,370 native pairs into RAM
at once (~70 GB) and OOM-stalled the 64 GB container before it printed a number.
This driver does the same measurement but STREAMS: it decodes at most `bs` pairs
at a time, so peak host RAM stays tiny. Metrics are accumulated per image and
macro-averaged over the full 4,370 pairs (identical statistic to a single
evaluate() pass, just memory-safe).

Native axis (960x540 pad16), the run's protocol axis. Writes final_metrics_all
back into the run's meta.json on the results volume.

Usage:
    modal run model/scripts/modal/eval_full_testset.py::run \
        --run-name 20260704_fullsf_gev4onp_nc --arch gev4_opt_narrow_plane
"""
from __future__ import annotations

import modal

app = modal.App("eval-full-testset")
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
def eval_remote(run_name: str, arch: str, ckpt: str, bs: int) -> dict:
    import json
    import os
    import pickle
    import sys
    from pathlib import Path

    import numpy as np
    import torch

    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    os.chdir("/workspace")
    if Path("/cache/yolo26s.pt").exists() and not Path("/workspace/yolo26s.pt").exists():
        os.symlink("/cache/yolo26s.pt", "/workspace/yolo26s.pt")
    sys.path.insert(0, "/workspace/model/scripts")
    sys.path.insert(0, "/workspace/model/designs")

    # reuse the trainer's EXACT model builder, decode, batching, forward, metric
    from train_full_sceneflow import (  # noqa: E402
        decode_record, build_model, _forward_pad16, batchify, stereo_metrics)

    device = "cuda"
    shards_dir = Path("/shards/v1")
    out_dir = Path(f"/results/fulltrain/{run_name}")
    ckpt_path = out_dir / ckpt

    model, cfg = build_model(arch)
    ck = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ck["model"])
    model.to(device).eval()
    best_step = ck.get("step")
    print(f"loaded {ckpt_path} (step {best_step}); arch={arch}; "
          f"params={sum(p.numel() for p in model.parameters())/1e6:.4f} M",
          flush=True)

    test_shards = sorted(shards_dir.glob("test_ft3d_*.pkl"))
    print(f"{len(test_shards)} test shards; streaming at bs={bs} "
          f"(native 960x540 pad16 axis) ...", flush=True)

    agg = []           # one 8-metric dict per image, over the full test set
    n_pairs = 0
    with torch.no_grad():
        for si, sp in enumerate(test_shards):
            with open(sp, "rb") as f:
                recs = pickle.load(f)
            # stream within the shard: decode only a bs-sized window at a time
            for i in range(0, len(recs), bs):
                window = [decode_record(r, native=True)
                          for r in recs[i:i + bs]]
                idxs = list(range(len(window)))
                L, R, D, V = batchify(window, idxs, device)
                pred = _forward_pad16(model, L, R)
                for b in range(pred.shape[0]):
                    agg.append(stereo_metrics(pred[b:b+1], D[b:b+1], V[b:b+1]))
                n_pairs += len(window)
                del window, L, R, D, V, pred
            print(f"  shard {si+1}/{len(test_shards)} done "
                  f"({n_pairs} pairs)", flush=True)

    # nanmean: a frame with zero valid GT pixels (gt>0 & gt<192 empty) yields
    # nan EPE/bad-* (mean of an empty tensor); it cannot contribute a stereo
    # metric, so it is excluded rather than poisoning the macro-average. Count
    # how many frames were degenerate per metric for honesty.
    keys = list(agg[0].keys())
    nan_cnt = {k: int(np.sum(np.isnan([a[k] for a in agg]))) for k in keys}
    fm = {k: float(np.nanmean([a[k] for a in agg])) for k in keys}
    print(f"\nFULL 4,370-pair TEST (native, {n_pairs} pairs) on {ckpt}:")
    print("  " + "  ".join(f"{k}={v:.4f}" for k, v in fm.items()), flush=True)
    print(f"  degenerate frames excluded per metric (empty valid mask): "
          f"{ {k: c for k, c in nan_cnt.items() if c} }", flush=True)

    # write into meta.json (fills the field the OOM'd in-run eval never wrote)
    meta_path = out_dir / "meta.json"
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    meta["final_metrics_all"] = fm
    meta["final_axis"] = "native_960x540_pad16"
    meta["final_best_step"] = best_step
    meta["final_n_pairs"] = n_pairs
    meta["final_degenerate_frames"] = {k: c for k, c in nan_cnt.items() if c}
    meta["final_source"] = "eval_full_testset.py (streaming; in-run eval OOM'd)"
    meta_path.write_text(json.dumps(meta, indent=1))
    results_vol.commit()
    return {"metrics": fm, "n_pairs": n_pairs, "best_step": best_step}


@app.local_entrypoint()
def run(run_name: str = "20260704_fullsf_gev4onp_nc",
        arch: str = "gev4_opt_narrow_plane",
        ckpt: str = "best.pth", bs: int = 4):
    print(f"streaming full-testset eval: run={run_name} ckpt={ckpt} "
          f"arch={arch} bs={bs} — keep client alive (~15 min on T4).")
    out = eval_remote.remote(run_name, arch, ckpt, bs)
    m = out["metrics"]
    print(f"\n=== PUBLISHABLE SceneFlow FT3D-TEST ({out['n_pairs']} pairs, "
          f"best step {out['best_step']}) ===")
    print("  " + "  ".join(f"{k}={v:.4f}" for k, v in m.items()))
    print(f"written to widener-results:/fulltrain/{run_name}/meta.json")
