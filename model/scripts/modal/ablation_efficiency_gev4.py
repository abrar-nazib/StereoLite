"""Efficiency-fix validation on Modal: 3 arms parallel on A10 (24 GB).

Arms: gev4 (control) / gev4_opt (F1/F2/F4/F5/F7, metric-equivalence proven
on RTX 3050: max EPE delta 3.1e-5 px, 1.29x faster) / gev4_opt_narrow
(+F3 narrow GEV — the accuracy A/B this run decides).

Protocol: 100 Scene Flow Driving pairs (80 train / 20 val), up to 12000
steps with plateau early-stop, batch 8, lr 2e-4, input [0,1]. Full
artifacts per ablation-study-expert skill incl. the 6-tile annotated
collage per eval (1 GT + 3 train preds + 2 val preds).

GPU: A10 per user grant ("a little better GPU than T4", 2026-07-03).
Cost estimate: 3 x A10 x ~1 h ~= $3.3.

Blocking .map() — do NOT `modal run -d` this file (detach cancels pending
map inputs). Keep the client alive (background shell is fine).

Usage:
    modal run model/scripts/modal/ablation_efficiency_gev4.py::main
Pull results after:
    modal volume get widener-results efficiency_gev4/<run> model/benchmarks/
"""
from __future__ import annotations

import modal

app = modal.App("ablation-efficiency-gev4")
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

RUN_NAME = "eff_gev4_n100"


@app.function(
    image=image,
    gpu="A10",
    volumes={"/cache": cache_vol, "/results": results_vol},
    timeout=4 * 3600,
    retries=0,
)
def run_arm(arch: str):
    import os, subprocess, time, shutil
    from pathlib import Path

    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    project_root = "/workspace"
    os.chdir(project_root)

    src_w = "/cache/yolo26s.pt"
    if Path(src_w).exists():
        os.symlink(src_w, f"{project_root}/yolo26s.pt")

    cmd = [
        "python3", "-u",
        "/workspace/model/scripts/overfit_efficiency_ablation.py",
        "--arch", arch,
        "--n_pairs", "100", "--n_val", "20",
        "--max_steps", "12000", "--min_steps", "4000",
        "--patience", "4", "--eval_every", "500",
        "--batch", "8", "--lr", "2e-4", "--seed", "42",
        "--show", "0",
        "--pairs_cache", "/cache/eff_pairs_n100.pt",
        "--out_root", "/results/efficiency_gev4",
        "--run_name", RUN_NAME,
    ]
    t0 = time.time()
    p = subprocess.Popen(cmd, env=os.environ.copy(), stdout=subprocess.PIPE,
                         stderr=subprocess.STDOUT, text=True, bufsize=1,
                         cwd=project_root)
    tail = []
    for line in p.stdout:
        print(f"[{arch}] {line}", end="")
        tail.append(line)
    rc = p.wait()
    results_vol.commit()
    if not Path(src_w).exists() and Path(f"{project_root}/yolo26s.pt").exists():
        shutil.copy(f"{project_root}/yolo26s.pt", src_w)
        cache_vol.commit()
    return {"arch": arch, "rc": rc, "elapsed_s": round(time.time() - t0, 1),
            "tail": "".join(tail[-6:])}


@app.local_entrypoint()
def main():
    arms = ["gev4", "gev4_opt", "gev4_opt_narrow"]
    print(f"3 arms parallel on A10, run={RUN_NAME} — keep this client alive.")
    for res in run_arm.map(arms):
        print(f"\n=== {res['arch']} rc={res['rc']} {res['elapsed_s']}s ===\n"
              f"{res['tail']}")
