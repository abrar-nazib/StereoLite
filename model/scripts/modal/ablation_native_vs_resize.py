"""Native-crop vs resize input protocol: 2 arms on L40S, n500 Driving.

Question (user, 2026-07-04): is the global 960x540 -> 640x384 downscale
(INTER_AREA) responsible for the crispness ceiling? Published methods train
on random crops of NATIVE-resolution frames; our resize protocol was
inherited from the T4 overfit harness, not chosen.

Arms (single knob = input protocol; arch fixed at the locked full-training
config gev4_opt_narrow_plane + slant 0.3 + aug):
  control      full frame downscaled to 640x384 (legacy protocol)
  native_crop  random 384x640 crops of native 960x540 frames, native
               disparity magnitudes (matched pixel budget per sample)

Shared axes: BOTH arms get a final dual-axis eval on the same 50 val
frames at (a) native 960x540 (pad16) and (b) resized 640x384, plus a
final_native_collage.png for visual crispness comparison.

Pre-registered criterion (README written at launch): native_crop wins if
it beats control on bad-0.5 AND bad-1 on the NATIVE-axis val eval by more
than the ~8% relative noise band; EPE alone does not decide (crispness
question). Driving native disparities >192 px are masked by protocol; the
two arms therefore see different valid-pixel statistics -- that is the
thing being measured, not a confound.

Pairs: same (seq,t) keys and split as /cache/eff_pairs_n500_windowed.pt,
rebuilt at native resolution from the sceneflow-shards volume into
/cache/eff_pairs_n500_native.pt (builder function below, runs first).

Blocking .map(); do NOT `modal run -d`.

Usage:
    modal run model/scripts/modal/ablation_native_vs_resize.py::main
    modal volume get widener-results efficiency_gev4/20260704_native_vs_resize_n500 model/benchmarks/
"""
from __future__ import annotations

from datetime import datetime

import modal

app = modal.App("ablation-native-vs-resize")
cache_vol = modal.Volume.from_name("stereo-overfit-cache", create_if_missing=True)
results_vol = modal.Volume.from_name("widener-results", create_if_missing=True)
shards_vol = modal.Volume.from_name("sceneflow-shards")
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

RUN_NAME = f"{datetime.now():%Y%m%d}_native_vs_resize_n500"
NATIVE_CACHE = "/cache/eff_pairs_n500_native.pt"
ARMS = {
    "control": ["--input_mode", "resize"],
    "native_crop": ["--input_mode", "native_crop"],
    "native_full": ["--input_mode", "native_full"],
}


@app.function(
    image=image,
    volumes={"/cache": cache_vol, "/shards": shards_vol},
    cpu=8, memory=16384, timeout=1800, retries=0,
)
def build_native_cache():
    """Rebuild the n500 windowed pairs at NATIVE resolution from the
    driving shards, preserving key order and split. Idempotent."""
    import pickle
    from pathlib import Path

    import cv2
    import json
    import numpy as np
    import torch
    import zstandard

    if Path(NATIVE_CACHE).exists():
        blob = torch.load(NATIVE_CACHE, map_location="cpu", weights_only=False)
        print(f"native cache exists: {len(blob['train'])} train / "
              f"{len(blob['val'])} val -- skipping build")
        return {"built": False}

    src = torch.load("/cache/eff_pairs_n500_windowed.pt", map_location="cpu",
                     weights_only=False)
    need = {}   # key -> (split, position)
    for split in ("train", "val"):
        for i, p in enumerate(src[split]):
            key = f"frames_finalpass/{p['seq']}/left/{p['t']:04d}.png"
            need[key] = (split, i)
    print(f"need {len(need)} native pairs "
          f"({len(src['train'])} train / {len(src['val'])} val)")

    idx = json.loads(Path("/shards/v1/index.json").read_text())
    out = {"train": [None] * len(src["train"]), "val": [None] * len(src["val"])}
    found = 0
    for entry in idx["shards"]:
        if not entry["shard"].startswith("train_driving"):
            continue
        hits = [k for k in entry["keys"] if k in need]
        if not hits:
            continue
        with open(f"/shards/v1/{entry['shard']}", "rb") as f:
            recs = {r["key"]: r for r in pickle.load(f)}
        for key in hits:
            rec = recs[key]
            ims = []
            for k in ("left_png", "right_png"):
                im = cv2.imdecode(np.frombuffer(rec[k], np.uint8),
                                  cv2.IMREAD_COLOR)
                ims.append(torch.from_numpy(im[..., ::-1].copy())
                           .permute(2, 0, 1).to(torch.uint8))
            h, w = rec["shape"]
            d = np.abs(np.frombuffer(zstandard.decompress(rec["disp_z"]),
                                     "<f4").reshape(h, w))
            d = np.nan_to_num(d.astype(np.float32), nan=0.0, posinf=0.0)
            split, i = need[key]
            parts = key.split("/")
            out[split][i] = dict(seq="/".join(parts[1:-2]),
                                 t=int(parts[-1][:-4]),
                                 L=ims[0], R=ims[1],
                                 D=torch.from_numpy(d)[None].to(torch.float16))
            found += 1
        del recs
        print(f"{entry['shard']}: +{len(hits)} ({found}/{len(need)})")

    missing = [(s, i) for s in ("train", "val")
               for i, p in enumerate(out[s]) if p is None]
    if missing:
        raise RuntimeError(f"{len(missing)} pairs not found in shards: "
                           f"{missing[:5]}")
    # order + key equality check against the source cache
    for split in ("train", "val"):
        for a, b in zip(src[split], out[split]):
            assert (a["seq"], a["t"]) == (b["seq"], b["t"])
    torch.save(dict(train=out["train"], val=out["val"], seed=src.get("seed"),
                    n_pairs=src.get("n_pairs"), n_val=len(out["val"]),
                    split_protocol=str(src.get("split_protocol")) + "-native",
                    val_windows=src.get("val_windows")),
               NATIVE_CACHE)
    cache_vol.commit()
    print(f"native cache written: {NATIVE_CACHE}")
    return {"built": True, "pairs": found}


@app.function(
    image=image,
    gpu="L40S",
    volumes={"/cache": cache_vol, "/results": results_vol},
    timeout=5 * 3600,
    retries=0,
)
def run_arm(args_pack: tuple):
    import os
    import shutil
    import subprocess
    import time
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
        "--arch", "gev4_opt_narrow_plane", "--slant_w", "0.3",
        "--n_pairs", "500", "--n_val", "50",
        "--max_steps", "20000", "--min_steps", "8000",
        "--patience", "4", "--eval_every", "500",
        "--batch", "8", "--lr", "2e-4", "--seed", "42",
        "--aug", "1", "--freeze_bn", "0",
        "--show", "0",
        "--native_cache", NATIVE_CACHE,
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
            "tail": "".join(tail[-10:])}


@app.local_entrypoint()
def main(arms: str = "control,native_crop", run_name: str = ""):
    """arms: comma-separated subset of ARMS to launch. run_name: override to
    append late arms into an existing dated run dir."""
    tags = [t.strip() for t in arms.split(",") if t.strip()]
    unknown = [t for t in tags if t not in ARMS]
    if unknown:
        raise SystemExit(f"unknown arms: {unknown}")
    rn = run_name or RUN_NAME
    print(f"native-vs-resize on L40S, run={rn}, arms={tags} -- keep client alive.")
    res = build_native_cache.remote()
    print(f"cache builder: {res}")
    for r in run_arm.map([(t, rn) for t in tags]):
        print(f"\n=== {r['tag']} rc={r['rc']} {r['elapsed_s']}s ===\n"
              f"{r['tail']}")
