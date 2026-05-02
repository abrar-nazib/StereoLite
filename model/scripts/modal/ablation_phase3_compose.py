"""Phase 3: compose the Phase 2 winners on T4.

Tests whether `cascade_cv_4 + slope_aware_warp` (the two Phase 2 winners)
compose without interference. Apples-to-apples on T4 to avoid the
A100/cuDNN reproducibility drift seen in earlier sweeps.

4 configs, all on the ghostconv-winner chassis (costlookup_yolo26n_full
+ ghostconv), 100 pairs / 9000 steps / batch=4 / seed=42:

  1. baseline (control)               — anchor on T4
  2. cascade_cv_4 alone               — Phase 2 repro on T4
  3. slope_aware_warp alone           — Phase 2 repro on T4
  4. cascade_cv_4 + slope_aware_warp  — THE composition

Cost: ~$0.40 (4 parallel T4 × ~30 min × $0.59/hr).

Usage:
    modal run model/scripts/modal/ablation_phase3_compose.py::main
"""
from __future__ import annotations

import modal


app = modal.App("ablation-phase3-compose")
cache_vol = modal.Volume.from_name("stereo-overfit-cache", create_if_missing=True)
results_vol = modal.Volume.from_name("widener-results", create_if_missing=True)
PROJECT_ROOT = "/home/abrar/Research/stero_research_claude"

image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("libgl1", "libglib2.0-0", "git")
    .pip_install(
        "torch==2.11.0", "torchvision", "numpy<2",
        "opencv-python-headless", "Pillow", "matplotlib",
        "pandas", "ultralytics==8.3.40", "timm", "scipy",
    )
    .add_local_dir(f"{PROJECT_ROOT}/model", "/workspace/model",
                   ignore=["benchmarks/**/*", "checkpoints/*",
                           "teachers/**/*", "kaggle/**/*",
                           "**/__pycache__/**"])
    .add_local_python_source("modal")
)

# Just the combined variant — both cascade_cv_4 AND slope_aware_warp on.
# Compare against the existing Phase 2 baseline (already trained on A100
# with the same methodology; final EPE 0.8644).
CONFIGS = [
    (1, 1, "p3_cascade_plus_slopewarp"),
]


@app.function(
    image=image,
    gpu="T4",
    volumes={"/cache": cache_vol, "/results": results_vol},
    timeout=60 * 90,
)
def run_one(cascade_cv_4: int, slope_aware_warp: int, tag: str):
    import os, sys, subprocess, time
    from pathlib import Path

    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    sys.path.insert(0, "/workspace/model/designs")
    sys.path.insert(0, "/workspace/model/scripts")
    project_root = "/workspace"
    os.chdir(project_root)

    for variant in ("yolo26n", "yolo26s"):
        src_w = f"/cache/{variant}.pt"
        dst_w = f"{project_root}/{variant}.pt"
        if Path(src_w).exists() and not Path(dst_w).exists():
            os.symlink(src_w, dst_w)
    cache_dir = f"{project_root}/.cache"
    Path(cache_dir).mkdir(parents=True, exist_ok=True)
    src = "/cache/sf_overfit_pairs_v1_n100.pt"
    dst = f"{cache_dir}/sf_overfit_pairs_v1_n100.pt"
    if not Path(dst).exists():
        os.symlink(src, dst)

    print(f"[{tag}] cascade={cascade_cv_4} slopewarp={slope_aware_warp} on T4")

    cmd = [
        "python3", "-u", "/workspace/model/scripts/overfit_arch_ablation.py",
        "--arch", "costlookup",
        "--backbone", "yolo26n",
        "--extend_to_full", "1",
        "--widener", "ghostconv",
        "--cascade_cv_4", str(cascade_cv_4),
        "--slope_aware_warp", str(slope_aware_warp),
        "--batch", "4",
        "--n_pairs", "100",
        "--steps", "9000",
        "--seed", "42",
        "--out_root", "/results/ablation_phase3",
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
    print(f"Launching {len(CONFIGS)} Phase 3 composition variants in parallel on T4")
    results = list(run_one.starmap(CONFIGS))
    print("\n=== ALL DONE ===")
    for r in results:
        print(r)
