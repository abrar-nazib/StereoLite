"""Sharp-tail hybrid arm: gev4_opt_narrow core + pre-rahi costlookup tail
(TileRefineCtx at 1/2 + plane-equation upsample instead of ConvexUpsample).

Single knob vs the gev4_opt_narrow arm of eff_gev4_n100 (same pairs cache,
seed, protocol). Deciding axes: bad-0.5 / visual sharpness on the matched
collages, at a measured latency cost (3050 fp32: 82.3 vs 61.4 ms).

Blocking .map(); do NOT `modal run -d`.

Usage:
    modal run model/scripts/modal/ablation_sharptail_gev4.py::main
    modal volume get widener-results efficiency_gev4/sharptail_n100 model/benchmarks/
"""
from __future__ import annotations

import modal

app = modal.App("ablation-sharptail-gev4")
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

RUN_NAME = "sharptail_n100"


@app.function(
    image=image,
    gpu="A10",
    volumes={"/cache": cache_vol, "/results": results_vol},
    timeout=4 * 3600,
    retries=0,
)
def run_arm():
    import os, subprocess, time
    from pathlib import Path

    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    project_root = "/workspace"
    os.chdir(project_root)
    if Path("/cache/yolo26s.pt").exists():
        os.symlink("/cache/yolo26s.pt", f"{project_root}/yolo26s.pt")

    cmd = [
        "python3", "-u",
        "/workspace/model/scripts/overfit_efficiency_ablation.py",
        "--arch", "gev4_opt_narrow_sharptail",
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
        print(f"[sharptail] {line}", end="")
        tail.append(line)
    rc = p.wait()
    results_vol.commit()
    return {"rc": rc, "elapsed_s": round(time.time() - t0, 1),
            "tail": "".join(tail[-6:])}


@app.local_entrypoint()
def main():
    print(f"1 arm on A10, run={RUN_NAME} — keep client alive.")
    res = run_arm.remote()
    print(f"\n=== sharptail rc={res['rc']} {res['elapsed_s']}s ===\n{res['tail']}")
