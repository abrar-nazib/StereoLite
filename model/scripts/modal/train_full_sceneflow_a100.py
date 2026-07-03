"""Full SceneFlow training on A100-80GB — the thesis checkpoint run.

Locked config (thesis/THESIS_PLAN.md 3b): gev4_opt_narrow_plane
(--slant_w 0.3) + OpenStereo aug + OneCycle + canonical split
(sceneflow_split_v1: 35,454 train / 4,370 FT3D-TEST) via the shard volume
built by repack_sceneflow_shards.py (no bz2, no extraction: training starts
in minutes).

Two entrypoints:
  probe  ~10 min on the A100: ms/step + peak VRAM at batch 8..48 through
         the REAL train step (aug + loss + backward). Read the table, pick
         the batch (largest <= 85% VRAM), then launch train.
  train  the real run. Checkpoints latest.pth to the results volume every
         eval; relaunching the same command RESUMES from it, so a
         preemption/crash costs at most eval_every steps.

Blocking .remote(); do NOT `modal run -d`.

Usage:
    modal run model/scripts/modal/train_full_sceneflow_a100.py::probe
    modal run model/scripts/modal/train_full_sceneflow_a100.py::train \
        --batch 32 --steps 100000
    modal volume get widener-results fulltrain/<RUN_NAME> model/benchmarks/
"""
from __future__ import annotations

from datetime import datetime

import modal

app = modal.App("train-full-sceneflow")
shards_vol = modal.Volume.from_name("sceneflow-shards")
cache_vol = modal.Volume.from_name("stereo-overfit-cache")
results_vol = modal.Volume.from_name("widener-results", create_if_missing=True)
PROJECT_ROOT = "/home/abrar/Research/stero_research_claude"

image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("libgl1", "libglib2.0-0", "git")
    .pip_install(
        "torch==2.11.0", "torchvision", "numpy<2",
        "opencv-python-headless", "Pillow", "matplotlib",
        "pandas", "ultralytics==8.3.40", "timm", "scipy", "zstandard",
    )
    .add_local_dir(f"{PROJECT_ROOT}/model", "/workspace/model",
                   ignore=["benchmarks/**/*", "checkpoints/*",
                           "teachers/**/*", "kaggle/**/*",
                           "**/__pycache__/**"])
)

RUN_NAME = f"{datetime.now():%Y%m%d}_fullsf_gev4onp"  # date_tag convention


def _run(cmd: list[str]) -> int:
    import os
    import subprocess
    from pathlib import Path

    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    os.chdir("/workspace")
    if Path("/cache/yolo26s.pt").exists():
        if not Path("/workspace/yolo26s.pt").exists():
            os.symlink("/cache/yolo26s.pt", "/workspace/yolo26s.pt")
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE,
                         stderr=subprocess.STDOUT, text=True, bufsize=1)
    for line in p.stdout:
        print(line, end="", flush=True)
    return p.wait()


BASE_CMD = [
    "python3", "-u", "/workspace/model/scripts/train_full_sceneflow.py",
    "--shards_dir", "/shards/v1",
    "--arch", "gev4_opt_narrow_plane", "--slant_w", "0.3",
    "--aug", "1", "--amp", "bf16", "--seed", "42",
    "--val_manifest", "/workspace/model/configs/sceneflow_split_v1.json.gz",
]


@app.function(image=image, gpu="A100-80GB",
              volumes={"/shards": shards_vol, "/cache": cache_vol,
                       "/results": results_vol},
              cpu=12, memory=65536, timeout=2 * 3600, retries=0)
def probe_remote() -> int:
    rc = _run(BASE_CMD + ["--probe", "8,16,24,32,40,48",
                          "--out_dir", f"/results/fulltrain/{RUN_NAME}_probe"])
    results_vol.commit()
    return rc


@app.function(image=image, gpu="A100-80GB",
              volumes={"/shards": shards_vol, "/cache": cache_vol,
                       "/results": results_vol},
              cpu=12, memory=65536, timeout=24 * 3600, retries=0)
def train_remote(batch: int, steps: int, lr: str, eval_every: int) -> int:
    import threading
    import time

    # background committer so checkpoints/logs are visible + safe mid-run
    stop = threading.Event()

    def committer():
        while not stop.wait(300):
            results_vol.commit()

    t = threading.Thread(target=committer, daemon=True)
    t.start()
    rc = _run(BASE_CMD + [
        "--batch", str(batch), "--steps", str(steps), "--lr", lr,
        "--eval_every", str(eval_every), "--workers", "10", "--resume", "1",
        "--out_dir", f"/results/fulltrain/{RUN_NAME}",
    ])
    stop.set()
    time.sleep(1)
    results_vol.commit()
    return rc


@app.local_entrypoint()
def probe():
    print(f"A100 batch probe, run={RUN_NAME}_probe — keep client alive.")
    rc = probe_remote.remote()
    print(f"probe rc={rc}; table also at "
          f"widener-results:/fulltrain/{RUN_NAME}_probe/probe.json")


@app.local_entrypoint()
def train(batch: int = 32, steps: int = 100000, lr: str = "auto",
          eval_every: int = 2000):
    est_h = steps * batch / 30 / 3600  # rough @30 samples/s, refine via probe
    print(f"FULL TRAINING run={RUN_NAME}: batch={batch}, steps={steps}, "
          f"lr={lr} (~{est_h:.0f} h rough) — keep client alive; "
          f"relaunch to resume.")
    rc = train_remote.remote(batch, steps, lr, eval_every)
    print(f"train rc={rc}")
