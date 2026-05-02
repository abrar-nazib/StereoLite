"""Apples-to-apples widener sweep on Modal A100 — PARALLEL.

9 configs each on their own A100-40GB, run in parallel via .starmap().
batch=4 (matches the prior 12-config RTX 3050 sweep), 3000 steps, 20 pairs.

Wall time: ~3-4 min total (cold start + ~2.5 min training).
Cost: ~$1 of $200 credit at Modal A100-40GB rates.

Usage:
    modal run --detach model/scripts/modal/widener_apples_to_apples.py::main

The outputs (meta.json, train.csv, viz/, checkpoint.pth) for each config
land in the `widener-results` Modal volume under
    /widener_modal_run/<tag>/
Pull back with `modal volume get widener-results widener_modal_run <local-path>`.
"""
from __future__ import annotations

import modal


app = modal.App("widener-apples-to-apples")

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


# 9 configs (arch, backbone, extend_to_full, widener, tag)
CONFIGS = [
    ("costlookup", "yolo26n", 1, "none",        "costlookup_yolo26n_full_none"),
    ("costlookup", "yolo26n", 1, "f2_only",     "costlookup_yolo26n_full_f2_only"),
    ("costlookup", "yolo26n", 1, "f2_f4",       "costlookup_yolo26n_full_f2_f4"),
    ("costlookup", "yolo26n", 1, "topdown_fpn", "costlookup_yolo26n_full_topdown_fpn"),
    ("costlookup", "yolo26n", 1, "dw",          "costlookup_yolo26n_full_dw"),
    ("costlookup", "yolo26n", 1, "ghostconv",   "costlookup_yolo26n_full_ghostconv"),
    ("costlookup", "yolo26s", 1, "none",        "costlookup_yolo26s_full_native"),
    ("costlookup", "ghost",   0, "none",        "costlookup_ghost_p1"),
    ("tilegru",    "yolo26n", 0, "none",        "tilegru_yolo26n_p1"),
]


@app.function(
    image=image,
    gpu="A100-40GB",
    volumes={"/cache": cache_vol, "/results": results_vol},
    timeout=60 * 30,  # 30 min cap per config (typical ~3 min)
)
def run_one(arch: str, backbone: str, extend_to_full: int,
            widener: str, tag: str):
    """Runs ONE config end-to-end on its own A100. batch=4 to match the
    prior 12-config sweep on RTX 3050 (apples-to-apples). Saves checkpoint
    + meta + train.csv + viz to the results volume."""
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
    src = "/cache/sf_overfit_pairs_v1.pt"
    dst = f"{cache_dir}/sf_overfit_pairs_v1.pt"
    if not Path(dst).exists():
        os.symlink(src, dst)

    print(f"[{tag}] starting on {os.environ.get('MODAL_GPU', 'A100')}")

    cmd = [
        "python3", "-u", "/workspace/model/scripts/overfit_arch_ablation.py",
        "--arch", arch,
        "--backbone", backbone,
        "--extend_to_full", str(extend_to_full),
        "--widener", widener,
        "--batch", "4",       # apples-to-apples with prior sweep
        "--n_pairs", "20",
        "--steps", "3000",
        "--seed", "42",
        "--out_root", "/results/widener_modal_run",
        "--variant_tag", tag,
        "--show", "0",
        "--viz_interval_s", "60",
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
    """Fire off all 9 configs in parallel via .starmap()."""
    print(f"Launching {len(CONFIGS)} configs in parallel on A100-40GB containers")
    results = list(run_one.starmap(CONFIGS))
    print("\n=== ALL CONFIGS DONE ===")
    for r in results:
        print(r)
