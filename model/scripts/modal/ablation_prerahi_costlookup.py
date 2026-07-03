"""Pre-rahi costlookup leader on the 80/20 held-out protocol: 2 arms on A10.

Arms:
  costlookup_y26n  the exact pre-rahi project leader (yolo26n + ghostconv
                   widener + extend_to_full + cascade_cv_4 + slope_aware_warp
                   + init_gce; 1.326 M) — EPE 0.811 on the LEGACY 100-pair
                   eval-on-train protocol (not comparable to this one)
  costlookup_y26s  same knobs on native yolo26s (2.208 M)

Same pairs cache as eff_gev4_n100 (pair_paths hash-identical) so results
are directly comparable with the gev4/gev4_opt/gev4_opt_narrow arms.

Blocking .map(); do NOT `modal run -d`.

Usage:
    modal run model/scripts/modal/ablation_prerahi_costlookup.py::main
    modal volume get widener-results efficiency_gev4/prerahi_n100 model/benchmarks/
"""
from __future__ import annotations

import modal

app = modal.App("ablation-prerahi-costlookup")
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

RUN_NAME = "prerahi_n100"


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
    for w in ("yolo26n.pt", "yolo26s.pt"):
        if Path(f"/cache/{w}").exists():
            os.symlink(f"/cache/{w}", f"{project_root}/{w}")

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
    for w in ("yolo26n.pt", "yolo26s.pt"):
        if not Path(f"/cache/{w}").exists() and Path(f"{project_root}/{w}").exists():
            shutil.copy(f"{project_root}/{w}", f"/cache/{w}")
            cache_vol.commit()
    return {"arch": arch, "rc": rc, "elapsed_s": round(time.time() - t0, 1),
            "tail": "".join(tail[-6:])}


@app.local_entrypoint()
def main():
    arms = ["costlookup_y26n", "costlookup_y26s"]
    print(f"2 arms parallel on A10, run={RUN_NAME} — keep this client alive.")
    for res in run_arm.map(arms):
        print(f"\n=== {res['arch']} rc={res['rc']} {res['elapsed_s']}s ===\n"
              f"{res['tail']}")
