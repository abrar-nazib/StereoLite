"""Phase 1: 100-pair baseline for the ablation methodology shift.

Single A100 container, single config (costlookup + yolo26n_full +
ghostconv = current production winner). Trained on 100 fixed Scene
Flow pairs at batch=4 for up to 15,000 steps. Reads train.csv after to
identify the loss-plateau step count, which becomes the budget for the
Phase 2 9-variant sweep.

Usage:
    modal run model/scripts/modal/ablation_baseline_n100.py::main
"""
from __future__ import annotations

import modal


app = modal.App("ablation-baseline-n100")

cache_vol = modal.Volume.from_name("stereo-overfit-cache", create_if_missing=True)
results_vol = modal.Volume.from_name("widener-results", create_if_missing=True)

PROJECT_ROOT = "/home/abrar/Research/stero_research_claude"

image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("libgl1", "libglib2.0-0", "git")
    .pip_install(
        "torch==2.11.0",
        "torchvision",
        "numpy<2",
        "opencv-python-headless",
        "Pillow",
        "matplotlib",
        "pandas",
        "ultralytics==8.3.40",
        "timm",
        "scipy",
    )
    .add_local_dir(f"{PROJECT_ROOT}/model", "/workspace/model",
                   ignore=["benchmarks/**/*", "checkpoints/*",
                           "teachers/**/*", "kaggle/**/*",
                           "**/__pycache__/**"])
    .add_local_python_source("modal")
)


@app.function(
    image=image,
    gpu="A100-40GB",
    volumes={"/cache": cache_vol, "/results": results_vol},
    timeout=60 * 45,   # 45 min cap (15k steps at ~50ms each = 12-13 min)
)
def run_baseline_n100():
    import os, sys, subprocess, time
    from pathlib import Path

    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    sys.path.insert(0, "/workspace/model/designs")
    sys.path.insert(0, "/workspace/model/scripts")
    project_root = "/workspace"
    os.chdir(project_root)

    # YOLO weights + cache symlinks.
    for variant in ("yolo26n", "yolo26s"):
        src_w = f"/cache/{variant}.pt"
        dst_w = f"{project_root}/{variant}.pt"
        if Path(src_w).exists() and not Path(dst_w).exists():
            os.symlink(src_w, dst_w)
    cache_dir = f"{project_root}/.cache"
    Path(cache_dir).mkdir(parents=True, exist_ok=True)
    # Use the 100-pair cache (already uploaded earlier).
    src = "/cache/sf_overfit_pairs_v1_n100.pt"
    if not Path(src).exists():
        # Fall back to default name lookup; Modal volume has it.
        src = "/cache/sf_overfit_pairs_v1_n100.pt"
    dst = f"{cache_dir}/sf_overfit_pairs_v1_n100.pt"
    if not Path(dst).exists():
        os.symlink(src, dst)

    tag = "baseline_n100_ghostconv"
    print(f"[{tag}] Phase 1 baseline starting, 100 pairs, batch=4, up to 15000 steps")

    cmd = [
        "python3", "-u", "/workspace/model/scripts/overfit_arch_ablation.py",
        "--arch", "costlookup",
        "--backbone", "yolo26n",
        "--extend_to_full", "1",
        "--widener", "ghostconv",
        "--batch", "4",
        "--n_pairs", "100",
        "--steps", "15000",
        "--seed", "42",
        "--out_root", "/results/ablation_phase1",
        "--variant_tag", tag,
        "--show", "0",
        "--viz_interval_s", "120",
    ]
    env = os.environ.copy()
    env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    t0 = time.time()
    p = subprocess.Popen(cmd, env=env, stdout=subprocess.PIPE,
                         stderr=subprocess.STDOUT, text=True,
                         bufsize=1, cwd=project_root)
    for line in p.stdout:
        print(f"[{tag}] {line}", end="")
    rc = p.wait()
    elapsed = time.time() - t0
    results_vol.commit()
    return {"tag": tag, "rc": rc, "elapsed_s": round(elapsed, 1)}


@app.local_entrypoint()
def main():
    out = run_baseline_n100.remote()
    print(out)
