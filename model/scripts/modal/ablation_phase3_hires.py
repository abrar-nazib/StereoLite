"""Phase 3 hi-res: 3 T4 containers @ 512x832 on the 100-pair set.

Tests whether higher resolution training (1.7x area vs the 384x640
default) helps the Phase 2 winners. Same chassis (costlookup_yolo26n_full
+ ghostconv), 100 pairs, 9000 steps, batch=4, seed=42.

3 configs in parallel on T4:
  1. cascade_cv_4 alone @ 512x832
  2. slope_aware_warp alone @ 512x832
  3. cascade_cv_4 + slope_aware_warp combined @ 512x832

Cost estimate: 3 × ~45 min (T4 slower at higher res) × $0.59/hr = ~$1.30.

Compare against:
  - Phase 2 baseline @ 384x640 (control, EPE 0.8644)
  - Phase 2 cascade_cv_4 @ 384x640 (EPE 0.8385)
  - Phase 2 slope_aware_warp @ 384x640 (EPE 0.8435)
  - Phase 3 combined @ 384x640 (running now on T4)

Usage:
    modal run model/scripts/modal/ablation_phase3_hires.py::main
"""
from __future__ import annotations

import modal


app = modal.App("ablation-phase3-hires")
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


# (cascade_cv_4, slope_aware_warp, tag)
CONFIGS = [
    (1, 0, "p3_hires_cascade_only"),
    (0, 1, "p3_hires_slopewarp_only"),
    (1, 1, "p3_hires_cascade_plus_slopewarp"),
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
    # Hi-res cache, keyed on (n=100, 512x832).
    src = "/cache/sf_overfit_pairs_v1_n100_512x832.pt"
    dst = f"{cache_dir}/sf_overfit_pairs_v1_n100_512x832.pt"
    if not Path(dst).exists():
        os.symlink(src, dst)

    print(f"[{tag}] cascade={cascade_cv_4} slopewarp={slope_aware_warp} @ 512x832 on T4")

    cmd = [
        "python3", "-u", "/workspace/model/scripts/overfit_arch_ablation.py",
        "--arch", "costlookup",
        "--backbone", "yolo26n",
        "--extend_to_full", "1",
        "--widener", "ghostconv",
        "--cascade_cv_4", str(cascade_cv_4),
        "--slope_aware_warp", str(slope_aware_warp),
        "--height", "512",
        "--width", "832",
        "--batch", "4",
        "--n_pairs", "100",
        "--steps", "9000",
        "--seed", "42",
        "--out_root", "/results/ablation_phase3_hires",
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
    print(f"Launching {len(CONFIGS)} Phase 3 hi-res variants in parallel on T4 @ 512x832")
    results = list(run_one.starmap(CONFIGS))
    print("\n=== ALL DONE ===")
    for r in results:
        print(r)
