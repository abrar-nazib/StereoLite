"""InStereo2K real-GT fine-tune (Modal, A100), native_crop protocol.

Reads InStereo2K.zip from the stereo-datasets volume, extracts to the
container's local disk (never onto the volume: inode budget), then fine-tunes
best.pth on random 384x640 crops taken at ~960 wide density (matching the
SceneFlow native_crop training and the 960 inference density). Disparity is
PNG value / 100, invalid = 0 (official format).

    modal run model/scripts/modal/finetune_instereo2k.py::train \
        --steps 2500 --batch 20

Checkpoint -> widener-results:/realcam_finetune/finetune_instereo2k_ncrop_best.pth
Blocking .remote(); do NOT `modal run -d`.
"""
from __future__ import annotations
import modal

app = modal.App("finetune-instereo2k")
ds_vol = modal.Volume.from_name("stereo-datasets")
ckpt_vol = modal.Volume.from_name("realcam-finetune")   # holds /best.pth
cache_vol = modal.Volume.from_name("stereo-overfit-cache")
results_vol = modal.Volume.from_name("widener-results", create_if_missing=True)
PROJECT_ROOT = "/home/abrar/Research/stero_research_claude"

image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("libgl1", "libglib2.0-0", "git", "unzip")
    .pip_install("torch==2.11.0", "torchvision", "numpy<2",
                 "opencv-python-headless", "Pillow", "matplotlib", "pandas",
                 "ultralytics==8.3.40", "timm", "scipy")
    .add_local_dir(f"{PROJECT_ROOT}/model", "/workspace/model",
                   ignore=["benchmarks/**/*", "checkpoints/*", "teachers/**/*",
                           "kaggle/**/*", "**/__pycache__/**"])
)

MAX_DISP = 192.0
TARGET_W = 960        # resize each pair to this width (native_crop density)
CROP_H, CROP_W = 384, 640


@app.function(image=image, gpu="A100-40GB", timeout=4 * 3600,
              volumes={"/ds": ds_vol, "/ckpt": ckpt_vol, "/cache": cache_vol,
                       "/results": results_vol})
def train(steps: int = 2500, batch: int = 20, lr: float = 1e-4,
          eval_every: int = 150, seed: int = 42, slant_w: float = 0.3,
          n_val: int = 100):
    import glob
    import os
    import subprocess
    import sys
    import time
    import cv2
    import numpy as np
    import torch

    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    os.environ.setdefault("XFORMERS_DISABLED", "1")
    os.chdir("/workspace")
    if os.path.exists("/cache/yolo26s.pt") and not os.path.exists("/workspace/yolo26s.pt"):
        os.symlink("/cache/yolo26s.pt", "/workspace/yolo26s.pt")
    sys.path.insert(0, "/workspace/model/scripts")
    sys.path.insert(0, "/workspace/model/designs")
    from overfit_efficiency_ablation import build_model, loss_fn

    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    dev = "cuda"
    tot = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f"GPU {torch.cuda.get_device_name(0)}  {tot:.1f} GB", flush=True)

    # ---- extract InStereo2K to local disk (NOT the volume) ----
    zpath = "/ds/instereo2k/InStereo2K.zip"
    ex = "/tmp/is2k"
    os.makedirs(ex, exist_ok=True)
    print(f"extracting {zpath} ({os.path.getsize(zpath)/1e6:.0f} MB) ...", flush=True)
    subprocess.run(["unzip", "-q", "-o", zpath, "-d", ex], check=True)
    # The OneDrive export nests part2.zip/part3.zip/part4.zip; extract any
    # inner zips too so the scene folders surface.
    for p in glob.glob(f"{ex}/**/*.zip", recursive=True):
        subprocess.run(["unzip", "-q", "-o", p, "-d", ex], check=True)

    # ---- discover (left, right, left_disp) triples ----
    lefts = sorted(glob.glob(f"{ex}/**/left.png", recursive=True))
    triples = []
    for lp in lefts:
        d = os.path.dirname(lp)
        rp = os.path.join(d, "right.png")
        dp = os.path.join(d, "left_disp.png")
        if os.path.exists(rp) and os.path.exists(dp):
            triples.append((lp, rp, dp))
    print(f"found {len(triples)} InStereo2K pairs", flush=True)
    if len(triples) < 200:
        # fall back: some zips nest as *_L.png etc.; report a tree sample
        sample = subprocess.run(["find", ex, "-maxdepth", "3"], capture_output=True,
                                text=True).stdout.splitlines()[:40]
        print("STRUCTURE SAMPLE:\n" + "\n".join(sample), flush=True)
        raise RuntimeError(f"only {len(triples)} pairs found; check structure")

    idx = rng.permutation(len(triples))
    val_ids = idx[:n_val]
    train_ids = idx[n_val:]
    print(f"train {len(train_ids)}  val {len(val_ids)}", flush=True)

    def load_resized(lp, rp, dp):
        L = cv2.cvtColor(cv2.imread(lp), cv2.COLOR_BGR2RGB)
        R = cv2.cvtColor(cv2.imread(rp), cv2.COLOR_BGR2RGB)
        D = cv2.imread(dp, cv2.IMREAD_UNCHANGED).astype(np.float32) / 100.0
        Hn, Wn = D.shape[:2]
        sx = TARGET_W / Wn
        tw, th = TARGET_W, int(round(Hn * sx))
        L = cv2.resize(L, (tw, th), interpolation=cv2.INTER_AREA)
        R = cv2.resize(R, (tw, th), interpolation=cv2.INTER_AREA)
        D = cv2.resize(D, (tw, th), interpolation=cv2.INTER_NEAREST) * sx
        D[~np.isfinite(D) | (D < 0)] = 0.0
        return L, R, D

    def to_batch(ids, mode):
        Lb, Rb, Db = [], [], []
        for k in ids:
            L, R, D = load_resized(*triples[k])
            Hf, Wf = D.shape
            if mode == "rand":
                y0 = int(rng.integers(0, max(1, Hf - CROP_H + 1)))
                x0 = int(rng.integers(0, max(1, Wf - CROP_W + 1)))
            else:
                y0, x0 = max(0, (Hf - CROP_H) // 2), max(0, (Wf - CROP_W) // 2)
            Lb.append(L[y0:y0 + CROP_H, x0:x0 + CROP_W])
            Rb.append(R[y0:y0 + CROP_H, x0:x0 + CROP_W])
            Db.append(D[y0:y0 + CROP_H, x0:x0 + CROP_W])
        Lt = torch.from_numpy(np.stack(Lb)).to(dev).float().permute(0, 3, 1, 2) / 255.0
        Rt = torch.from_numpy(np.stack(Rb)).to(dev).float().permute(0, 3, 1, 2) / 255.0
        Dt = torch.from_numpy(np.stack(Db).astype(np.float32)).to(dev).unsqueeze(1)
        Vt = ((Dt > 0) & (Dt < MAX_DISP)).float()
        return Lt, Rt, Dt, Vt

    # ---- model ----
    model, cfg = build_model("gev4_opt_narrow_plane")
    ck = torch.load("/ckpt/best.pth", map_location="cpu")
    model.load_state_dict(ck["model"])
    model.to(dev).train()
    print(f"loaded base best.pth step {ck.get('step')}", flush=True)

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps, eta_min=lr * 0.05)
    scaler = torch.cuda.amp.GradScaler()

    @torch.no_grad()
    def val_epe():
        model.eval()
        errs, npx = 0.0, 0
        for i in range(0, len(val_ids), 8):
            L, R, D, V = to_batch(val_ids[i:i + 8], "center")
            with torch.cuda.amp.autocast(dtype=torch.float16):
                out = model(L, R, aux=True)
            errs += (torch.abs(out["d_final"].float() - D) * V).sum().item()
            npx += V.sum().item()
        model.train()
        return errs / max(npx, 1)

    base = val_epe()
    print(f"[step 0] val EPE (base) = {base:.4f}", flush=True)
    best = float("inf")
    peak = 0.0
    t0 = time.time()
    order = rng.permutation(len(train_ids))
    ptr = 0
    for step in range(1, steps + 1):
        if ptr + batch > len(train_ids):
            order = rng.permutation(len(train_ids))
            ptr = 0
        bi = [train_ids[j] for j in order[ptr:ptr + batch]]
        ptr += batch
        L, R, D, V = to_batch(bi, "rand")
        opt.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast(dtype=torch.float16):
            out = model(L, R, aux=True)
            loss = loss_fn(out, D, V, slant_w=slant_w)
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(opt)
        scaler.update()
        sched.step()
        peak = max(peak, torch.cuda.max_memory_allocated() / 1e9)
        if step % 50 == 0:
            print(f"[step {step}/{steps}] loss {loss.item():.4f} "
                  f"lr {sched.get_last_lr()[0]:.2e} peak {peak:.1f}GB "
                  f"({step*batch/(time.time()-t0):.0f} img/s)", flush=True)
        if step % eval_every == 0 or step == steps:
            e = val_epe()
            tag = ""
            if e < best:
                best = e
                os.makedirs("/results/realcam_finetune", exist_ok=True)
                torch.save({"model": model.state_dict(), "step": step,
                            "val_epe": e, "base_epe": base,
                            "arch": "gev4_opt_narrow_plane",
                            "finetune": "instereo2k_ncrop"},
                           "/results/realcam_finetune/finetune_instereo2k_ncrop_best.pth")
                results_vol.commit()
                tag = "  <- best, saved"
            print(f"[step {step}] val EPE = {e:.4f}  (best {best:.4f}){tag}", flush=True)

    print(f"DONE. base {base:.4f} -> best {best:.4f}  peak {peak:.1f}GB", flush=True)
    return {"base_epe": base, "best_epe": best, "pairs": len(triples)}


@app.local_entrypoint()
def main(steps: int = 2500, batch: int = 20):
    print("RESULT:", train.remote(steps=steps, batch=batch))
