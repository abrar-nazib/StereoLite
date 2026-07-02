"""Diagnostic: pure-L1 overfit ceiling test.

Same chassis as tonight's leader (costlookup + ghostconv + cascade_cv_4 +
slope_aware_warp + GCE-in-TileInit) but with the loss stripped to a single
full-resolution L1 term — no multi-scale, no Sobel grad, no threshold-stack
hinge, no D1 hinge, no init loss.

Stereo-vision-expert-recommended diagnostic. Two outcomes:
  - EPE drops to 0.3-0.4 → the regularized stack_d1 / baseline loss is the
    overfit ceiling, not the architecture. The 0.81 plateau is by design.
  - EPE stays > 0.6 → structural ceiling at the 1/16 init quantization or
    missing per-iter signal. Follow-up: try max_disp=48 or add a 1/8 init.

T4, timeout=24h, 100 pairs, batch=4, 9000 steps, seed=42, 384x640.

Usage:
    modal run model/scripts/modal/ablation_pure_l1_diagnostic.py::main
"""
from __future__ import annotations

import modal


app = modal.App("ablation-pure-l1-diagnostic")
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


@app.function(
    image=image,
    gpu="T4",
    volumes={"/cache": cache_vol, "/results": results_vol},
    timeout=86400,  # 24h, no mid-run kills
)
def run_one():
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

    tag = "diag_pure_l1_combo"
    print(f"[{tag}] starting | costlookup + ghostconv + cascade_cv_4 + "
          f"slope_aware_warp + GCE-in-TileInit + PURE L1 ONLY on T4")

    cmd = [
        "python3", "-u", "/workspace/model/scripts/overfit_arch_ablation.py",
        "--arch", "costlookup",
        "--backbone", "yolo26n",
        "--extend_to_full", "1",
        "--widener", "ghostconv",
        "--cascade_cv_4", "1",
        "--slope_aware_warp", "1",
        "--init_gce", "1",
        "--pure_l1", "1",
        "--batch", "4",
        "--n_pairs", "100",
        "--steps", "9000",
        "--seed", "42",
        "--out_root", "/results/ablation_expert_review",
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
    print("Single T4 container, 24h timeout — pure-L1 overfit ceiling diagnostic.")
    result = run_one.remote()
    print("\n=== DONE ===")
    print(result)
