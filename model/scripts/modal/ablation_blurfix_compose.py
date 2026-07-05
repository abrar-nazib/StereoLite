"""Blur-fix composition: do the validated fixes stack? 2 arms on L40S.

Arms (control = 20260703_blurfix_n500/control500, same pairs/seed/harness
training path):
  pb     plane render + bimodal aux (the two decisive single fixes)
  allin  bundle1 + plane + bimodal (everything)

Pre-registered criterion (20260703_blurfix_n500/comparison.md): a
composition wins if final val bad-0.5 < 39.5 (beats plane's 42.91 beyond
the ~8% noise band); on a tie prefer pb over allin (bundle1 may not
compose). Same n500 leak-proof pairs cache.

Blocking .map(); do NOT `modal run -d`.

Usage:
    modal run model/scripts/modal/ablation_blurfix_compose.py::main
"""
from __future__ import annotations

from datetime import datetime

import modal

app = modal.App("ablation-blurfix-compose")
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

RUN_NAME = f"{datetime.now():%Y%m%d}_blurfix_compose"  # date_tag convention
ARMS = {
    "pb": ["--arch", "gev4_opt_narrow_pb",
           "--slant_w", "0.3", "--bimodal_w", "0.4"],
    "allin": ["--arch", "gev4_opt_narrow_allin",
              "--trunc_A", "1.0", "--init_ce_w", "0.3",
              "--slant_w", "0.3", "--bimodal_w", "0.4"],
}


@app.function(
    image=image,
    gpu="L40S",
    volumes={"/cache": cache_vol, "/results": results_vol},
    timeout=5 * 3600,
    retries=0,
)
def run_arm(args_pack: tuple):
    import os, subprocess, time, shutil
    from pathlib import Path

    tag, run_name = args_pack
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
        "--run_name", f"{run_name}/{tag}",
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
    print(f"composition arms on L40S, run={RUN_NAME} — keep client alive.")
    for res in run_arm.map([(t, RUN_NAME) for t in ARMS]):
        print(f"\n=== {res['tag']} rc={res['rc']} {res['elapsed_s']}s ===\n"
              f"{res['tail']}")
