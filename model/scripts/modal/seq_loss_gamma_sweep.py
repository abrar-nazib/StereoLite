"""seq_loss γ sweep on Modal T4.

Phase 2's seq_loss variant ran with γ=0.9 (RAFT-Stereo's choice for 32
iters) and underperformed (-3.4% vs baseline). Hypothesis: with our
shorter iter chain (10 iters), γ=0.9 weights early iters too much.
Re-test with γ=0.5 / 0.6 / 0.7 to find the right schedule.

T4 GPU because we're ablating a small model and don't need A100 compute.
~$0.20-0.30 for 3 parallel containers.

Usage:
    modal run model/scripts/modal/seq_loss_gamma_sweep.py::main
"""
from __future__ import annotations

import modal


app = modal.App("seq-loss-gamma-sweep")
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

# (gamma, tag)
CONFIGS = [
    (0.5, "p2_seq_loss_g05"),
    (0.6, "p2_seq_loss_g06"),
    (0.7, "p2_seq_loss_g07"),
]


@app.function(
    image=image,
    gpu="T4",            # cheap GPU; we're launch-overhead-bound at this scale
    volumes={"/cache": cache_vol, "/results": results_vol},
    timeout=60 * 90,     # T4 is slower; 90 min cap
)
def run_one(gamma: float, tag: str):
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

    print(f"[{tag}] starting seq_loss with γ={gamma} on T4")

    cmd = [
        "python3", "-u", "/workspace/model/scripts/overfit_arch_ablation.py",
        "--arch", "costlookup",
        "--backbone", "yolo26n",
        "--extend_to_full", "1",
        "--widener", "ghostconv",
        "--loss_variant", "seq_loss",
        "--seq_loss_gamma", str(gamma),
        "--batch", "4",
        "--n_pairs", "100",
        "--steps", "9000",
        "--seed", "42",
        "--out_root", "/results/ablation_phase2",
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
    return {"tag": tag, "gamma": gamma, "rc": rc, "elapsed_s": round(elapsed, 1)}


@app.local_entrypoint()
def main():
    print(f"Launching {len(CONFIGS)} seq_loss γ variants in parallel on T4")
    results = list(run_one.starmap(CONFIGS))
    print("\n=== ALL DONE ===")
    for r in results:
        print(r)
