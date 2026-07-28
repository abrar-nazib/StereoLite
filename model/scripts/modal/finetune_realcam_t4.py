"""Fine-tune the thesis checkpoint on REAL camera pairs (Modal, T4).

Takes the SceneFlow-trained gev4_opt_narrow_plane checkpoint (best.pth,
step 53k, FT3D-TEST EPE 0.78) and fine-tunes it on the 917 clean real
rig pairs (FoundationStereo pseudo-GT) so it sees the actual camera's
photometric and rectification statistics before the supervisor demo.

Data flow:
  - packed blobs (640x384, disparity in 640-width px) live on the
    `realcam-finetune` volume: /data/realcam_train.npz, realcam_val.npz
    (uploaded once with `modal volume put`).
  - base checkpoint baked into the image at /workspace/best.pth.
  - yolo26s.pt (encoder pretrain, needed at model construction) mounted
    from stereo-overfit-cache at /cache and symlinked into /workspace.
  - result checkpoint written to widener-results:/realcam_finetune/.

Blocking .remote(); do NOT `modal run -d`.

Usage:
    # one-time data upload (after packing locally):
    modal volume put realcam-finetune LOCAL/realcam_train.npz /realcam_train.npz -f
    modal volume put realcam-finetune LOCAL/realcam_val.npz   /realcam_val.npz   -f
    # fine-tune:
    modal run model/scripts/modal/finetune_realcam_t4.py::train --steps 2500 --batch 24
    # pull result:
    modal volume get widener-results realcam_finetune/finetune_realcam_best.pth \
        model/checkpoints/finetune_realcam_best.pth
"""
from __future__ import annotations

import modal

app = modal.App("finetune-realcam")
data_vol = modal.Volume.from_name("realcam-finetune", create_if_missing=True)
cache_vol = modal.Volume.from_name("stereo-overfit-cache")
results_vol = modal.Volume.from_name("widener-results", create_if_missing=True)
PROJECT_ROOT = "/home/abrar/Research/stero_research_claude"
BASE_CKPT = f"{PROJECT_ROOT}/model/benchmarks/20260704_fullsf_gev4onp_nc/best.pth"

image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("libgl1", "libglib2.0-0", "git")
    .pip_install(
        "torch==2.11.0", "torchvision", "numpy<2",
        "opencv-python-headless", "Pillow", "matplotlib",
        "pandas", "ultralytics==8.3.40", "timm", "scipy",
    )
    # add_local_* must be LAST (they are startup mounts, no build step after).
    # best.pth is uploaded to the realcam-finetune volume and read from /data.
    .add_local_dir(f"{PROJECT_ROOT}/model", "/workspace/model",
                   ignore=["benchmarks/**/*", "checkpoints/*",
                           "teachers/**/*", "kaggle/**/*",
                           "**/__pycache__/**"])
)

MAX_DISP = 192.0


# A100-40GB: this GEV+GRU model is sequential-heavy, so per-step time (not
# memory) dominates. The A100 cuts step time vs L4/T4 and fits a larger batch.
# Peak at batch 16 / 640x384 was 22.4 GB, so batch 20 sits comfortably in 40 GB.
@app.function(image=image, gpu="A100-40GB", timeout=4 * 3600,
              volumes={"/data": data_vol, "/cache": cache_vol,
                       "/results": results_vol})
def train(steps: int = 2500, batch: int = 24, lr: float = 1e-4,
          eval_every: int = 200, seed: int = 42, slant_w: float = 0.3,
          data_prefix: str = "realcam_ncrop", crop_h: int = 384,
          crop_w: int = 640,
          out_name: str = "finetune_realcam_ncrop_best.pth"):
    import os
    import sys
    import time

    import numpy as np
    import torch

    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    os.environ.setdefault("XFORMERS_DISABLED", "1")
    os.chdir("/workspace")
    # yolo26s.pt is needed when build_model constructs the encoder.
    if os.path.exists("/cache/yolo26s.pt") and not os.path.exists("/workspace/yolo26s.pt"):
        os.symlink("/cache/yolo26s.pt", "/workspace/yolo26s.pt")

    sys.path.insert(0, "/workspace/model/scripts")
    sys.path.insert(0, "/workspace/model/designs")
    from overfit_efficiency_ablation import build_model, loss_fn

    torch.manual_seed(seed)
    np.random.seed(seed)
    rng = np.random.default_rng(seed)
    dev = "cuda"

    tot = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f"GPU {torch.cuda.get_device_name(0)}  {tot:.1f} GB total", flush=True)

    # ---- data ----
    tr = np.load(f"/data/{data_prefix}_train.npz")
    va = np.load(f"/data/{data_prefix}_val.npz")
    Ltr, Rtr, Dtr = tr["L"], tr["R"], tr["D"]
    Lva, Rva, Dva = va["L"], va["R"], va["D"]
    N = Ltr.shape[0]
    Hf, Wf = Ltr.shape[1], Ltr.shape[2]
    print(f"train {N}  val {Lva.shape[0]}  frames {Ltr.shape[1:]}  "
          f"crop {crop_h}x{crop_w} ({'native_crop' if (Hf,Wf)!=(crop_h,crop_w) else 'full'})",
          flush=True)

    def to_batch(Ls, Rs, Ds, idx, mode):
        """Crop a crop_h x crop_w window (random for train, center for val),
        applying the SAME window to L, R and D (native_crop protocol)."""
        Lb, Rb, Db = [], [], []
        for k in idx:
            if mode == "rand":
                y0 = int(rng.integers(0, Hf - crop_h + 1))
                x0 = int(rng.integers(0, Wf - crop_w + 1))
            else:
                y0, x0 = (Hf - crop_h) // 2, (Wf - crop_w) // 2
            Lb.append(Ls[k, y0:y0 + crop_h, x0:x0 + crop_w])
            Rb.append(Rs[k, y0:y0 + crop_h, x0:x0 + crop_w])
            Db.append(Ds[k, y0:y0 + crop_h, x0:x0 + crop_w])
        L = torch.from_numpy(np.stack(Lb)).to(dev).float().permute(0, 3, 1, 2) / 255.0
        R = torch.from_numpy(np.stack(Rb)).to(dev).float().permute(0, 3, 1, 2) / 255.0
        D = torch.from_numpy(np.stack(Db).astype(np.float32)).to(dev).unsqueeze(1)
        V = ((D > 0) & (D < MAX_DISP)).float()
        return L, R, D, V

    # ---- model ----
    model, cfg = build_model("gev4_opt_narrow_plane")
    ck = torch.load("/data/best.pth", map_location="cpu")
    model.load_state_dict(ck["model"])
    model.to(dev).train()
    n_par = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"loaded base ckpt step {ck.get('step')}  params {n_par/1e6:.3f}M", flush=True)

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps, eta_min=lr * 0.05)
    scaler = torch.cuda.amp.GradScaler()

    @torch.no_grad()
    def val_epe():
        model.eval()
        errs, npx = 0.0, 0
        for i in range(0, Lva.shape[0], 8):
            j = list(range(i, min(i + 8, Lva.shape[0])))
            L, R, D, V = to_batch(Lva, Rva, Dva, j, "center")
            with torch.cuda.amp.autocast(dtype=torch.float16):
                out = model(L, R, aux=True)
            d = out["d_final"].float()
            e = (torch.abs(d - D) * V).sum().item()
            errs += e
            npx += V.sum().item()
        model.train()
        return errs / max(npx, 1)

    base_epe = val_epe()
    print(f"[step 0] val EPE (base, pre-finetune) = {base_epe:.4f}", flush=True)

    best = float("inf")
    perm = rng.permutation(N)
    ptr = 0
    peak = 0.0
    t0 = time.time()
    for step in range(1, steps + 1):
        if ptr + batch > N:
            perm = rng.permutation(N)
            ptr = 0
        idx = perm[ptr:ptr + batch]
        ptr += batch
        L, R, D, V = to_batch(Ltr, Rtr, Dtr, idx.tolist(), "rand")

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
            ips = step * batch / (time.time() - t0)
            print(f"[step {step}/{steps}] loss {loss.item():.4f} "
                  f"lr {sched.get_last_lr()[0]:.2e} peak {peak:.1f}GB "
                  f"({ips:.0f} img/s)", flush=True)

        if step % eval_every == 0 or step == steps:
            e = val_epe()
            tag = ""
            if e < best:
                best = e
                out_ck = {"model": model.state_dict(), "step": step,
                          "val_epe": e, "base_epe": base_epe, "cfg": ck.get("cfg"),
                          "arch": "gev4_opt_narrow_plane", "finetune": "realcam"}
                os.makedirs("/results/realcam_finetune", exist_ok=True)
                torch.save(out_ck, f"/results/realcam_finetune/{out_name}")
                results_vol.commit()
                tag = "  <- best, saved"
            print(f"[step {step}] val EPE = {e:.4f}  (best {best:.4f}){tag}", flush=True)

    print(f"DONE. base EPE {base_epe:.4f} -> best EPE {best:.4f} "
          f"(peak {peak:.1f}/{tot:.1f} GB = {100*peak/tot:.0f}% util)", flush=True)
    return {"base_epe": base_epe, "best_epe": best, "peak_gb": peak}


@app.local_entrypoint()
def main(steps: int = 2500, batch: int = 24, lr: float = 1e-4):
    r = train.remote(steps=steps, batch=batch, lr=lr)
    print("RESULT:", r)
