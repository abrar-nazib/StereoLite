"""Train StereoLite on Middlebury from Modal.

This expects `download_middlebury.py` to have populated the master
`stereo-datasets` Volume with zips under `/data/middlebury/{2014,2021}`.
The zips are extracted into the container's local disk for each run; checkpoints
and samples are written to the persistent `stereolite-results` Volume.

Typical flow:

    modal run -d model/scripts/modal/download_middlebury.py::main \\
        --action download --year all

    modal run -d model/scripts/modal/train_middlebury.py::main \\
        --gpu A10 --steps 10000 --run-name yolo_ctx_gev4_mb_a10_b4

Monitor:

    modal app logs stereolite-middlebury-train --follow
    modal volume ls stereolite-results middlebury_runs

Pull results:

    modal volume get stereolite-results \\
        /middlebury_runs/yolo_ctx_gev4_mb_a100x2/best.pth ./best.pth
"""
from __future__ import annotations

from pathlib import Path

import modal


APP_NAME = "stereolite-middlebury-train"
DATA_VOL_NAME = "stereo-datasets"
RESULTS_VOL_NAME = "stereolite-results"
CACHE_VOL_NAME = "stereo-overfit-cache"


def _find_local_repo_root() -> Path:
    """Find repo root during local Modal image construction.

    Modal re-imports this file inside the remote container as
    `/root/train_middlebury.py`, where the original source-tree parent depth no
    longer exists. Returning `/workspace` there keeps top-level import safe; the
    real source tree has already been mounted into the image by the local run.
    """
    here = Path(__file__).resolve()
    for base in (Path.cwd(), *here.parents):
        if (base / "model" / "scripts" / "modal").is_dir():
            return base
    return Path("/workspace")


LOCAL_REPO_ROOT = _find_local_repo_root()

app = modal.App(APP_NAME)
data_vol = modal.Volume.from_name(DATA_VOL_NAME, create_if_missing=True)
results_vol = modal.Volume.from_name(RESULTS_VOL_NAME, create_if_missing=True)
cache_vol = modal.Volume.from_name(CACHE_VOL_NAME, create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("libgl1", "libglib2.0-0", "git", "unzip", "ca-certificates")
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
    .add_local_dir(
        str(LOCAL_REPO_ROOT / "model"),
        "/workspace/model",
        ignore=[
            "benchmarks/**/*",
            "checkpoints/**/*",
            "teachers/**/*",
            "kaggle/**/*",
            "**/__pycache__/**",
            "**/*.pyc",
        ],
    )
)


def _gpu_count(gpu: str) -> int:
    if ":" not in gpu:
        return 1
    tail = gpu.rsplit(":", 1)[-1]
    return int(tail) if tail.isdigit() else 1


@app.function(
    image=image,
    # Cheap default. The local entrypoint overrides this dynamically with
    # train.with_options(gpu=...). Keeping the decorator cheap prevents an
    # accidental A100 launch if args are omitted or override parsing changes.
    gpu="L4",
    volumes={"/data": data_vol, "/results": results_vol, "/cache": cache_vol},
    timeout=24 * 3600,
    cpu=4,
    memory=8192,
    scaledown_window=60,
)
def train(
    gpu: str = "L4",
    arch: str = "yolo_ctx_gev4",
    backbone: str = "yolo26s",
    run_name: str = "yolo_ctx_gev4_middlebury_modal",
    steps: int = 10000,
    batch: int = 4,
    height: int = 384,
    width: int = 640,
    train_mode: str = "mixed",
    crop_prob: float = 0.25,
    n_val: int = 20,
    val_every: int = 1000,
    sample_every: int = 1000,
    save_every: int = 5000,
    early_stop_patience: int = 10,
    early_stop_min_delta: float = 0.003,
    min_steps: int = 3000,
    lr: float = 2e-4,
    smooth_w: float = 0.02,
    max_train: int = 0,
):
    import glob
    import os
    import shutil
    import subprocess
    import sys
    import time
    import zipfile
    from pathlib import Path

    project_root = Path("/workspace")
    work_root = Path("/tmp/stereolite_middlebury")
    data_root = work_root / "middlebury"
    yolo_dir = Path("/tmp/yolo")
    yolo_dir.mkdir(parents=True, exist_ok=True)

    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    os.environ["OMP_NUM_THREADS"] = "4"
    os.environ["YOLO_CONFIG_DIR"] = str(yolo_dir)
    sys.path.insert(0, str(project_root / "model" / "scripts"))

    for variant in ("yolo26n", "yolo26s"):
        src = Path("/cache") / f"{variant}.pt"
        dst = project_root / f"{variant}.pt"
        if src.exists() and not dst.exists():
            dst.symlink_to(src)

    if data_root.exists():
        shutil.rmtree(data_root)
    data_root.mkdir(parents=True, exist_ok=True)

    zip_paths = sorted(glob.glob("/data/middlebury/**/*.zip", recursive=True))
    if not zip_paths:
        raise RuntimeError(
            "No Middlebury zips found in /data/middlebury. Run: "
            "modal run -d model/scripts/modal/download_middlebury.py::main "
            "--action download --year all"
        )

    print(f"extracting {len(zip_paths)} Middlebury zip files to {data_root}")
    for zp in zip_paths:
        zp_path = Path(zp)
        year = zp_path.parent.name
        if zp_path.name == "all.zip":
            target = data_root / year
        else:
            target = data_root / year / zp_path.stem
        target.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(zp_path) as zf:
            zf.extractall(target)

    from middlebury_loader import enumerate_middlebury

    pairs = enumerate_middlebury(str(data_root))
    print(f"middlebury pairs found: {len(pairs)}")
    if not pairs:
        raise RuntimeError(f"No im0/im1/disp0.pfm triples found under {data_root}")

    nproc = _gpu_count(gpu)
    cmd = [
        "torchrun",
        "--standalone",
        "--nproc_per_node", str(nproc),
        str(project_root / "model" / "scripts" / "train_arch_sceneflow.py"),
        "--dataset", "middlebury",
        "--arch", arch,
        "--backbone", backbone,
        "--data_root", str(data_root),
        "--out_root", "/results/middlebury_runs",
        "--run_name", run_name,
        "--steps", str(steps),
        "--batch", str(batch),
        "--height", str(height),
        "--width", str(width),
        "--train_mode", train_mode,
        "--crop_prob", str(crop_prob),
        "--n_val", str(n_val),
        "--max_train", str(max_train),
        "--val_every", str(val_every),
        "--sample_every", str(sample_every),
        "--save_every", str(save_every),
        "--early_stop_metric", "resize",
        "--early_stop_patience", str(early_stop_patience),
        "--early_stop_min_delta", str(early_stop_min_delta),
        "--min_steps", str(min_steps),
        "--lr", str(lr),
        "--smooth_w", str(smooth_w),
        "--num_workers", "6",
        "--amp", "1",
    ]
    print("launching:", " ".join(cmd))
    t0 = time.time()
    proc = subprocess.Popen(
        cmd,
        cwd=str(project_root),
        env=os.environ.copy(),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    assert proc.stdout is not None
    for line in proc.stdout:
        print(line, end="")
    rc = proc.wait()
    results_vol.commit()
    elapsed = time.time() - t0
    if rc != 0:
        raise RuntimeError(f"training failed with exit code {rc}")
    return {
        "run_name": run_name,
        "pairs": len(pairs),
        "gpu": gpu,
        "nproc": nproc,
        "elapsed_s": round(elapsed, 1),
        "results": f"/middlebury_runs/{run_name}",
    }


@app.local_entrypoint()
def main(
    gpu: str = "L4",
    arch: str = "yolo_ctx_gev4",
    backbone: str = "yolo26s",
    run_name: str = "yolo_ctx_gev4_middlebury_modal",
    steps: int = 10000,
    batch: int = 4,
    height: int = 384,
    width: int = 640,
    train_mode: str = "mixed",
    crop_prob: float = 0.25,
    n_val: int = 20,
    val_every: int = 1000,
    sample_every: int = 1000,
    save_every: int = 5000,
    early_stop_patience: int = 10,
    early_stop_min_delta: float = 0.003,
    min_steps: int = 3000,
    lr: float = 2e-4,
    smooth_w: float = 0.02,
    max_train: int = 0,
    wait: int = 0,
):
    print(f"Launching {arch}/{backbone} on Middlebury with {gpu}")
    kwargs = dict(
        gpu=gpu,
        arch=arch,
        backbone=backbone,
        run_name=run_name,
        steps=steps,
        batch=batch,
        height=height,
        width=width,
        train_mode=train_mode,
        crop_prob=crop_prob,
        n_val=n_val,
        val_every=val_every,
        sample_every=sample_every,
        save_every=save_every,
        early_stop_patience=early_stop_patience,
        early_stop_min_delta=early_stop_min_delta,
        min_steps=min_steps,
        lr=lr,
        smooth_w=smooth_w,
        max_train=max_train,
    )
    runner = train.with_options(gpu=gpu)
    if wait:
        result = runner.remote(**kwargs)
        print(result)
        return

    call = runner.spawn(**kwargs)
    print("Spawned Modal training job.")
    print(call)
    print("You can turn off this PC after this line; the job is running on Modal.")
    print("Monitor later with:")
    print(f"  modal app logs {APP_NAME} --follow")
    print("Results volume:")
    print(f"  modal volume ls {RESULTS_VOL_NAME} middlebury_runs/{run_name}")
