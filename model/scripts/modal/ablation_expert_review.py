"""Expert-recommended experiments on T4, with NO time pressure.

Two experiments per the stereo-vision-expert agent's review (2026-05-02):

EXPERIMENT 1 — Warm-start regression with explicit init loss (IGEV-style L_init).
  Hypothesis: a regressed init disparity (APC + 2-layer head) supervised
  directly with smooth-L1 should compress the iter chain.
  1 container:
    - costlookup + ghostconv + cascade_cv_4 + slope_aware_warp + init_regress
      + init_loss_weight=0.5

EXPERIMENT 2 — Per-iter sequence loss on the ConvGRU chassis (RAFT-Stereo style).
  Hypothesis: γ-weighted per-iter supervision is what makes few-iter GRU
  refinement converge. Sweep γ since the RAFT default 0.9 was tuned for
  N≈22-32 iters — our chain is N≈8.
  4 containers (compare to "same chassis without per-iter supervision"):
    - tilegru baseline (loss_variant=baseline, no seq_loss) — control
    - tilegru + seq_loss γ=0.6
    - tilegru + seq_loss γ=0.7
    - tilegru + seq_loss γ=0.8

Common methodology: 100 fixed Scene Flow pairs, batch=4, 9000 steps,
seed=42, 384x640, T4 GPU.

Container timeout = 24h (86400s) so nothing gets killed mid-training.
Cost estimate: 5 containers × ~30-50 min × $0.59/hr ≈ $2.

Usage:
    modal run model/scripts/modal/ablation_expert_review.py::main
"""
from __future__ import annotations

import modal


app = modal.App("ablation-expert-review")
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


# Each entry is a dict of CLI args for overfit_arch_ablation.py.
# `tag` is the per-variant subdir name on the results volume.
CONFIGS = [
    # ---- EXPERIMENT 1: warm-start init regression (IGEV L_init) ----
    {
        "tag": "exp1_costlookup_init_regress",
        "arch": "costlookup",
        "backbone": "yolo26n",
        "extend_to_full": 1,
        "widener": "ghostconv",
        "cascade_cv_4": 1,
        "slope_aware_warp": 1,
        "init_regress": 1,
        "init_loss_weight": 0.5,
        "loss_variant": "baseline",
        "seq_loss_gamma": 0.9,  # unused
    },

    # ---- EXPERIMENT 2: per-iter sequence loss + ConvGRU ----
    {
        "tag": "exp2_tilegru_baseline_no_seqloss",
        "arch": "tilegru",
        "backbone": "yolo26n",
        "extend_to_full": 1,
        "widener": None,  # tilegru doesn't support widener
        "cascade_cv_4": 0,
        "slope_aware_warp": 0,
        "init_regress": 0,
        "init_loss_weight": 0.0,
        "loss_variant": "baseline",
        "seq_loss_gamma": 0.9,  # unused
    },
    {
        "tag": "exp2_tilegru_seq_g06",
        "arch": "tilegru",
        "backbone": "yolo26n",
        "extend_to_full": 1,
        "widener": None,
        "cascade_cv_4": 0,
        "slope_aware_warp": 0,
        "init_regress": 0,
        "init_loss_weight": 0.0,
        "loss_variant": "seq_loss",
        "seq_loss_gamma": 0.6,
    },
    {
        "tag": "exp2_tilegru_seq_g07",
        "arch": "tilegru",
        "backbone": "yolo26n",
        "extend_to_full": 1,
        "widener": None,
        "cascade_cv_4": 0,
        "slope_aware_warp": 0,
        "init_regress": 0,
        "init_loss_weight": 0.0,
        "loss_variant": "seq_loss",
        "seq_loss_gamma": 0.7,
    },
    {
        "tag": "exp2_tilegru_seq_g08",
        "arch": "tilegru",
        "backbone": "yolo26n",
        "extend_to_full": 1,
        "widener": None,
        "cascade_cv_4": 0,
        "slope_aware_warp": 0,
        "init_regress": 0,
        "init_loss_weight": 0.0,
        "loss_variant": "seq_loss",
        "seq_loss_gamma": 0.8,
    },
]


@app.function(
    image=image,
    gpu="T4",
    volumes={"/cache": cache_vol, "/results": results_vol},
    timeout=86400,  # 24h; "without any time limit so we can do it completely"
)
def run_one(cfg: dict):
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

    tag = cfg["tag"]
    print(f"[{tag}] starting | arch={cfg['arch']} loss={cfg['loss_variant']} "
          f"γ={cfg['seq_loss_gamma']} init_regress={cfg['init_regress']} "
          f"init_w={cfg['init_loss_weight']} on T4")

    cmd = [
        "python3", "-u", "/workspace/model/scripts/overfit_arch_ablation.py",
        "--arch", cfg["arch"],
        "--backbone", cfg["backbone"],
        "--extend_to_full", str(cfg["extend_to_full"]),
        "--cascade_cv_4", str(cfg["cascade_cv_4"]),
        "--slope_aware_warp", str(cfg["slope_aware_warp"]),
        "--init_regress", str(cfg["init_regress"]),
        "--init_loss_weight", str(cfg["init_loss_weight"]),
        "--loss_variant", cfg["loss_variant"],
        "--seq_loss_gamma", str(cfg["seq_loss_gamma"]),
        "--batch", "4",
        "--n_pairs", "100",
        "--steps", "9000",
        "--seed", "42",
        "--out_root", "/results/ablation_expert_review",
        "--variant_tag", tag,
        "--show", "0",
        "--viz_interval_s", "120",
    ]
    if cfg.get("widener") not in (None, "none"):
        cmd.extend(["--widener", cfg["widener"]])

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
    print(f"Launching {len(CONFIGS)} expert-recommended ablations in parallel on T4")
    print("  - 1 warm-start (init_regress + L_init)")
    print("  - 4 ConvGRU sequence-loss variants (baseline + γ ∈ {0.6, 0.7, 0.8})")
    print("Container timeout = 24h. No mid-run kills.")
    results = list(run_one.map(CONFIGS))
    print("\n=== ALL DONE ===")
    for r in results:
        print(r)
