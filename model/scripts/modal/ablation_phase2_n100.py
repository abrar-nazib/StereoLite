"""Phase 2: 9-variant ablation sweep, parallel A100 containers.

All variants share the ghostconv winner chassis (costlookup_yolo26n_full +
ghostconv) and methodology (100 pairs, batch=4, 9000 steps = Phase 1
plateau). Each variant adds ONE intervention on top of the baseline,
then we measure on balanced metrics.

Variants (within edge envelope: < 2.5 M params, < 60 ms RTX 3050 fp16,
INT8-quantizable ops only):

LOSS-SIDE (zero param, zero latency):
  1. baseline           — control (= Phase 1 baseline)
  2. seq_loss           — RAFT γ-weighted L1 across iters
  3. slope_sup          — direct sx/sy supervision via finite-diff GT
  4. conf_aware         — confidence-aware focal-style penalty
  5. edge_smooth        — image-gradient-weighted disparity smoothness

ARCH-SIDE (small param add):
  6. slope_aware_warp   — A2, ~0 params, slope-corrected fR sample
  7. selective_gate     — A5, ~200 params, per-pixel "stop refining" gate
  8. cascade_cv_4       — A3, +68 k, narrow-range 3D-agg CV at 1/4
  9. context_branch     — A1, +242 k, parallel encoder fed into TileRefine

Usage:
    modal run model/scripts/modal/ablation_phase2_n100.py::main
"""
from __future__ import annotations

import modal


app = modal.App("ablation-phase2-n100")

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


# (loss_variant, slope_aware_warp, selective_gate, cascade_cv_4,
#  context_branch, tag)
CONFIGS = [
    ("baseline",    0, 0, 0, 0, "p2_baseline"),
    ("seq_loss",    0, 0, 0, 0, "p2_seq_loss"),
    ("slope_sup",   0, 0, 0, 0, "p2_slope_sup"),
    ("conf_aware",  0, 0, 0, 0, "p2_conf_aware"),
    ("edge_smooth", 0, 0, 0, 0, "p2_edge_smooth"),
    ("baseline",    1, 0, 0, 0, "p2_slope_aware_warp"),
    ("baseline",    0, 1, 0, 0, "p2_selective_gate"),
    ("baseline",    0, 0, 1, 0, "p2_cascade_cv_4"),
    ("baseline",    0, 0, 0, 1, "p2_context_branch"),
]


@app.function(
    image=image,
    gpu="T4",   # T4 ($0.59/hr): plenty of memory (16 GB) for our small model;
                # we're kernel-launch-bound at batch=4 anyway, so A100's
                # 312 TFLOPS is wasted compute. T4 is ~3.6x cheaper per hour.
                # Switch to A100-40GB for full Scene Flow training where
                # batch sizes are larger and compute matters.
    volumes={"/cache": cache_vol, "/results": results_vol},
    timeout=60 * 60,   # 60 min cap (T4 ~2x slower wall than A100)
)
def run_one(loss_variant: str, slope_aware_warp: int, selective_gate: int,
            cascade_cv_4: int, context_branch: int, tag: str):
    """Runs ONE Phase 2 variant on its own A100. batch=4 / n_pairs=100 /
    9000 steps to match the Phase 1 baseline plateau."""
    import os, sys, subprocess, time
    from pathlib import Path

    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    sys.path.insert(0, "/workspace/model/designs")
    sys.path.insert(0, "/workspace/model/scripts")
    project_root = "/workspace"
    os.chdir(project_root)

    # Symlink yolo weights + caches.
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

    print(f"[{tag}] Phase 2 variant starting")

    cmd = [
        "python3", "-u", "/workspace/model/scripts/overfit_arch_ablation.py",
        "--arch", "costlookup",
        "--backbone", "yolo26n",
        "--extend_to_full", "1",
        "--widener", "ghostconv",
        "--loss_variant", loss_variant,
        "--slope_aware_warp", str(slope_aware_warp),
        "--selective_gate", str(selective_gate),
        "--cascade_cv_4", str(cascade_cv_4),
        "--context_branch", str(context_branch),
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
    return {"tag": tag, "rc": rc, "elapsed_s": round(elapsed, 1)}


@app.local_entrypoint()
def main():
    print(f"Launching {len(CONFIGS)} Phase 2 variants in parallel on A100-40GB")
    results = list(run_one.starmap(CONFIGS))
    print("\n=== ALL CONFIGS DONE ===")
    for r in results:
        print(r)
