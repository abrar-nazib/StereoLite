"""Blur-fix validation, round 2: Fix 2 (plane render) + Fix 3 (bimodal
aux head) on the same 500-pair leak-proof cache as round 1. L40S.

Arms:
  plane    gev4_opt_narrow + plane-equation rendering + edge gate,
           gated slant supervision (--slant_w 0.3)   [treats decode cause]
  bimodal  gev4_opt_narrow + SMD-Nets bimodal aux head (--bimodal_w 0.4)
           [treats mean-regression cause; d_final untouched, d_bimodal
           reported separately]

Control = blurfix_n500/control500. Deciding metric: val bad-0.5 +
matched collages.

Blocking .map(); do NOT `modal run -d`.

Usage:
    modal run model/scripts/modal/ablation_blurfix2_gev4.py::main
"""
from __future__ import annotations

import modal

app = modal.App("ablation-blurfix2-gev4")
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
)

from datetime import datetime
RUN_NAME = f"{datetime.now():%Y%m%d}_blurfix_n500"  # date_tag convention
ARMS = {
    "plane": ["--arch", "gev4_opt_narrow_plane", "--slant_w", "0.3"],
    "bimodal": ["--arch", "gev4_opt_narrow_bimodal", "--bimodal_w", "0.4"],
}


@app.function(
    image=image,
    gpu="L40S",
    volumes={"/cache": cache_vol, "/results": results_vol},
    timeout=5 * 3600,
    retries=0,
)
def run_arm(tag: str):
    import os, subprocess, time, shutil
    from pathlib import Path

    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    project_root = "/workspace"
    os.chdir(project_root)
    if Path("/cache/yolo26s.pt").exists():
        os.symlink("/cache/yolo26s.pt", f"{project_root}/yolo26s.pt")

    cmd = [
        "python3", "-u",
        "/workspace/model/scripts/overfit_efficiency_ablation.py",
        "--n_pairs", "500", "--n_val", "50",
        "--max_steps", "20000", "--min_steps", "8000",
        "--patience", "4", "--eval_every", "500",
        "--batch", "8", "--lr", "2e-4", "--seed", "42",
        "--aug", "1", "--freeze_bn", "0",
        "--show", "0",
        "--pairs_cache", "/cache/eff_pairs_n500_windowed.pt",
        "--out_root", "/results/efficiency_gev4",
        "--run_name", f"{RUN_NAME}/{tag}",
    ] + ARMS[tag]
    t0 = time.time()
    p = subprocess.Popen(cmd, env=os.environ.copy(), stdout=subprocess.PIPE,
                         stderr=subprocess.STDOUT, text=True, bufsize=1,
                         cwd=project_root)
    tail = []
    for line in p.stdout:
        print(f"[{tag}] {line}", end="")
        tail.append(line)
    rc = p.wait()
    results_vol.commit()
    if not Path("/cache/yolo26s.pt").exists() and Path(f"{project_root}/yolo26s.pt").exists():
        shutil.copy(f"{project_root}/yolo26s.pt", "/cache/yolo26s.pt")
        cache_vol.commit()
    return {"tag": tag, "rc": rc, "elapsed_s": round(time.time() - t0, 1),
            "tail": "".join(tail[-6:])}


@app.local_entrypoint()
def main():
    print(f"round-2 arms on L40S, run={RUN_NAME} — keep client alive.")
    for res in run_arm.map(list(ARMS.keys())):
        print(f"\n=== {res['tag']} rc={res['rc']} {res['elapsed_s']}s ===\n"
              f"{res['tail']}")
