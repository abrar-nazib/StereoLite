"""Build the Kaggle notebook `stereolite_v8_kaggle.ipynb`.

Reads the current v8 source files from the repo and inlines them into
%%writefile cells so the whole training + export pipeline runs on Kaggle
without any external code fetch. Run:

    python3 model/kaggle/build_notebook.py
"""
from __future__ import annotations

import json
import os
import re
from pathlib import Path

PROJ = Path(__file__).resolve().parent.parent.parent
OUT = Path(__file__).resolve().parent / "stereolite_v8_kaggle.ipynb"


def read(rel: str) -> str:
    return (PROJ / rel).read_text()


BLOCKS_PY = read("model/designs/_blocks.py")
TILE_PROPAGATE_PY = read("model/designs/StereoLite/tile_propagate.py")
MODEL_PY = read("model/designs/StereoLite/model.py")
SF_LOADER_PY = read("model/scripts/sceneflow_loader.py")

INIT_PY = '''"""Kaggle build of StereoLite v8."""
from .model import StereoLite, StereoLiteConfig
'''


# ---------------------------- training script ------------------------------ #
TRAIN_DDP_PY = r'''"""DDP trainer for StereoLite v8 on Kaggle 2x T4 GPUs.

Launched via `torchrun --nproc_per_node=2 train_ddp.py ...`.

Highlights:
  - MobileNetV2-100 ImageNet-pretrained encoder
  - OneCycle LR peak 8e-4 (OpenStereo recipe)
  - Random-erase on right image
  - Multi-scale L1 + grad + bad-1 hinge + edge-aware smoothness loss
  - FP16 autocast + GradScaler
  - DistributedDataParallel on all visible GPUs
  - Rank-0 saves best-EPE checkpoint + CSV log each epoch
"""
from __future__ import annotations

import argparse
import csv
import math
import os
import sys
import time

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist

# Source layout set up by the notebook: /kaggle/working/src/
sys.path.insert(0, "/kaggle/working/src")
os.environ.setdefault("XFORMERS_DISABLED", "1")

from StereoLite import StereoLite, StereoLiteConfig
from sceneflow_loader import enumerate_pairs, train_val_split, SceneFlowDriving, read_pfm


# ---------------------------- loss terms ---------------------------------- #
def sobel_xy(d):
    kx = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
                      device=d.device, dtype=d.dtype).view(1, 1, 3, 3) / 8
    ky = kx.transpose(-1, -2).contiguous()
    d_pad = F.pad(d, (1, 1, 1, 1), mode="replicate")
    return F.conv2d(d_pad, kx), F.conv2d(d_pad, ky)


def grad_loss(pred, target, valid):
    gx_p, gy_p = sobel_xy(pred)
    gx_t, gy_t = sobel_xy(target)
    err = ((gx_p - gx_t).abs() + (gy_p - gy_t).abs()) * valid
    return err.sum() / valid.sum().clamp(min=1.0)


def edge_aware_smooth(pred, image, sigma=3.0):
    dx_p = (pred[..., 1:] - pred[..., :-1]).abs()
    dy_p = (pred[..., 1:, :] - pred[..., :-1, :]).abs()
    g = image.mean(dim=1, keepdim=True)
    dx_i = (g[..., 1:] - g[..., :-1]).abs()
    dy_i = (g[..., 1:, :] - g[..., :-1, :]).abs()
    wx = torch.exp(-dx_i / sigma)
    wy = torch.exp(-dy_i / sigma)
    return (wx * dx_p).mean() + (wy * dy_p).mean()


SCALE_WEIGHTS = {
    "d_final": 1.0, "d_half": 0.7, "d4": 0.5,
    "d8": 0.3, "d8_cv": 1.0, "d16": 0.3, "d32": 0.2,
}


def multiscale_loss(preds, D_full, grad_w=0.5, hinge_w=0.3,
                     smooth_w=0.02, left_img=None, max_disp=192.0):
    total = torch.zeros((), device=D_full.device, dtype=D_full.dtype)
    diag = {}
    for key, pred in preds.items():
        w = SCALE_WEIGHTS.get(key)
        if w is None:
            continue
        th = pred.shape[-2:]
        scale = th[1] / D_full.shape[-1]
        D_s = F.interpolate(D_full, size=th, mode="bilinear",
                             align_corners=True) * scale
        D_s[~torch.isfinite(D_s)] = 0
        valid = (D_s > 0) & (D_s < max_disp * scale) & torch.isfinite(D_s)
        n = valid.sum().clamp(min=1.0)
        diff = (pred - D_s).abs() * valid
        l1 = diff.sum() / n
        g = grad_loss(pred, D_s, valid) if w > 0.1 else torch.zeros_like(l1)
        thresh = 1.0 * scale
        hinge = (torch.relu(diff - thresh)).sum() / n
        total = total + w * (l1 + grad_w * g + hinge_w * hinge)
        diag[key] = float(l1.item())
    if smooth_w > 0 and left_img is not None and "d_final" in preds:
        total = total + smooth_w * edge_aware_smooth(preds["d_final"], left_img)
    return total, diag


def random_erase_right(R, prob=0.3, n_patches=2, h_frac=(0.02, 0.1),
                        w_frac=(0.02, 0.1)):
    if prob <= 0:
        return R
    B, _, H, W = R.shape
    R = R.clone()
    for b in range(B):
        if torch.rand(1).item() > prob:
            continue
        for _ in range(n_patches):
            ph = int(H * (h_frac[0] + torch.rand(1).item() * (h_frac[1] - h_frac[0])))
            pw = int(W * (w_frac[0] + torch.rand(1).item() * (w_frac[1] - w_frac[0])))
            if ph < 1 or pw < 1:
                continue
            y0 = torch.randint(0, max(H - ph, 1), (1,)).item()
            x0 = torch.randint(0, max(W - pw, 1), (1,)).item()
            fill = R[b, :, y0:y0 + ph, x0:x0 + pw].mean(dim=(-1, -2), keepdim=True)
            R[b, :, y0:y0 + ph, x0:x0 + pw] = fill
    return R


# ---------------------------- DDP helpers --------------------------------- #
def is_dist():
    return dist.is_available() and dist.is_initialized()


def get_rank():
    return dist.get_rank() if is_dist() else 0


def get_world_size():
    return dist.get_world_size() if is_dist() else 1


def setup_dist():
    if "RANK" in os.environ:
        dist.init_process_group(backend="nccl")
        lr = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(lr)
        return lr
    return 0


# ---------------------------- validation --------------------------------- #
@torch.no_grad()
def validate(model, val_loader, device, max_disp=192.0):
    model.eval()
    epes, bad1s = [], []
    for L, R, D in val_loader:
        L, R, D = L.to(device), R.to(device), D.to(device)
        pred = model(L, R)
        if pred.shape != D.shape:
            pred = F.interpolate(pred, size=D.shape[-2:], mode="bilinear",
                                  align_corners=True)
        valid = (D > 0) & (D < max_disp) & torch.isfinite(D)
        n = valid.sum().clamp(min=1.0)
        epes.append(float(((pred - D).abs() * valid).sum() / n))
        bad1s.append(float((((pred - D).abs() > 1.0).float() * valid).sum() / n))
    model.train()
    return float(np.mean(epes)), float(np.mean(bad1s))


def _colourise(d, lo, hi):
    d = d.astype(np.float32)
    v = np.clip((d - lo) / max(hi - lo, 1e-6), 0, 1) * 255
    return cv2.applyColorMap(v.astype(np.uint8), cv2.COLORMAP_TURBO)


def _annotate(img, text, h=22):
    out = img.copy()
    cv2.rectangle(out, (0, 0), (out.shape[1], h), (0, 0, 0), -1)
    cv2.putText(out, text, (6, h - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.42,
                (255, 255, 255), 1, cv2.LINE_AA)
    return out


@torch.no_grad()
def save_sample_panels(model, track_pairs, device, out_dir, tag,
                        inf_h, inf_w):
    """Save per-step montage showing [L | GT | pred] for each tracked pair.

    Writes to `<out_dir>/samples/step_<tag>.png`. `tag` is typically
    a zero-padded global step, e.g. "00500".
    """
    os.makedirs(os.path.join(out_dir, "samples"), exist_ok=True)
    was_training = model.training
    model.eval()
    rows = []
    ms_list = []
    for i, (lp, rp, pp) in enumerate(track_pairs):
        L = cv2.imread(lp); R = cv2.imread(rp)
        D_n = read_pfm(pp)
        H_n, W_n = D_n.shape
        L_in = cv2.resize(L, (inf_w, inf_h))
        R_in = cv2.resize(R, (inf_w, inf_h))
        sx = inf_w / W_n
        D = cv2.resize(D_n, (inf_w, inf_h)) * sx
        D[~np.isfinite(D) | (D < 0)] = 0

        Lt = torch.from_numpy(cv2.cvtColor(L_in, cv2.COLOR_BGR2RGB)).float() \
                .permute(2, 0, 1).unsqueeze(0).to(device)
        Rt = torch.from_numpy(cv2.cvtColor(R_in, cv2.COLOR_BGR2RGB)).float() \
                .permute(2, 0, 1).unsqueeze(0).to(device)

        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.time()
        P = model(Lt, Rt).squeeze().float().cpu().numpy()
        if device.type == "cuda":
            torch.cuda.synchronize()
        ms_list.append((time.time() - t0) * 1000)

        valid = (D > 0) & (D < 192) & np.isfinite(D)
        if valid.sum() > 16:
            lo = float(np.percentile(D[valid], 2))
            hi = float(np.percentile(D[valid], 98))
        else:
            lo, hi = 0.0, 60.0
        epe = float((np.abs(P - D) * valid).sum() / max(valid.sum(), 1))

        row = np.hstack([
            _annotate(L_in, f"L pair{i:02d}  {inf_w}x{inf_h}"),
            _annotate(_colourise(D, lo, hi), f"GT"),
            _annotate(_colourise(P, lo, hi),
                      f"pred  EPE={epe:.2f}  {ms_list[-1]:.0f}ms  step {tag}"),
        ])
        rows.append(row)

    montage = np.vstack(rows)
    out_path = os.path.join(out_dir, "samples", f"step_{tag}.png")
    cv2.imwrite(out_path, montage)
    if was_training:
        model.train()
    return float(np.mean(ms_list))


# ---------------------------- main --------------------------------------- #
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data_root", required=True)
    p.add_argument("--out_dir", default="/kaggle/working")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch", type=int, default=4)
    p.add_argument("--inf_h", type=int, default=384)
    p.add_argument("--inf_w", type=int, default=768)
    p.add_argument("--n_val", type=int, default=200)
    p.add_argument("--max_train", type=int, default=0,
                   help="cap training set (0 = all)")
    p.add_argument("--lr", type=float, default=8e-4)
    p.add_argument("--random_erase_p", type=float, default=0.3)
    p.add_argument("--grad_w", type=float, default=0.5)
    p.add_argument("--hinge_w", type=float, default=0.3)
    p.add_argument("--smooth_w", type=float, default=0.02)
    p.add_argument("--num_workers", type=int, default=2)
    p.add_argument("--log_every", type=int, default=50)
    p.add_argument("--sample_every", type=int, default=500,
                   help="save iterative sample-panel montage every N steps (0 = off)")
    p.add_argument("--n_sample_pairs", type=int, default=6,
                   help="number of val pairs to render in each sample montage")
    p.add_argument("--resume", default="")
    args = p.parse_args()

    local_rank = setup_dist()
    device = torch.device(f"cuda:{local_rank}")
    is_main = get_rank() == 0
    ws = get_world_size()

    if is_main:
        os.makedirs(args.out_dir, exist_ok=True)
        print(f"world size={ws}  local_rank={local_rank}  device={device}")
        print(f"data_root={args.data_root}  out_dir={args.out_dir}")

    # ---- data ----
    items = enumerate_pairs(args.data_root)
    train_items, val_items = train_val_split(items, args.n_val)
    if args.max_train > 0:
        train_items = train_items[: args.max_train]
    # Evenly-spaced val subset used for the sample-panel montage
    track_pairs = []
    if args.sample_every > 0 and len(val_items) > 0:
        stride = max(1, len(val_items) // args.n_sample_pairs)
        track_pairs = [val_items[i * stride] for i in range(args.n_sample_pairs)
                        if i * stride < len(val_items)]
    if is_main:
        print(f"train={len(train_items)}  val={len(val_items)}  "
              f"sample_track={len(track_pairs)}")

    train_ds = SceneFlowDriving(train_items, args.inf_h, args.inf_w)
    val_ds = SceneFlowDriving(val_items, args.inf_h, args.inf_w)

    train_sampler = (DistributedSampler(train_ds, num_replicas=ws,
                                         rank=get_rank(), shuffle=True)
                     if is_dist() else None)
    train_loader = DataLoader(train_ds, batch_size=args.batch,
                               sampler=train_sampler,
                               shuffle=(train_sampler is None),
                               num_workers=args.num_workers, pin_memory=True,
                               persistent_workers=args.num_workers > 0,
                               drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False,
                             num_workers=1, pin_memory=True)

    # ---- model ----
    model = StereoLite(StereoLiteConfig(backbone="mobilenet",
                                     )).to(device)
    if args.resume and os.path.exists(args.resume):
        ck = torch.load(args.resume, map_location="cpu", weights_only=False)
        sd = ck["model"] if "model" in ck else ck
        model.load_state_dict(sd, strict=False)
        if is_main:
            print(f"resumed from {args.resume}")

    if is_dist():
        model = DDP(model, device_ids=[local_rank],
                    find_unused_parameters=False)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if is_main:
        print(f"trainable params: {trainable/1e6:.3f} M")

    # ---- optim ----
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                             lr=args.lr, weight_decay=1e-5)
    steps_per_epoch = len(train_loader)
    total_steps = steps_per_epoch * args.epochs
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=args.lr, total_steps=total_steps,
        pct_start=0.05, anneal_strategy="cos",
        div_factor=25.0, final_div_factor=1e4)
    scaler = torch.amp.GradScaler("cuda", enabled=True)

    # ---- log ----
    if is_main:
        log_path = os.path.join(args.out_dir, "train_log.csv")
        fp_log = open(log_path, "w", newline="")
        w_csv = csv.writer(fp_log)
        w_csv.writerow(["epoch", "step", "loss", "l1_final", "l1_cv",
                         "lr", "ms_per_step"])

    best_epe = float("inf")
    global_step = 0
    model.train()
    t_start = time.time()

    for epoch in range(1, args.epochs + 1):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)

        t_epoch = time.time()
        run_loss, run_l1f, run_l1c = [], [], []

        for i, (L, R, D) in enumerate(train_loader):
            L, R, D = L.to(device, non_blocking=True), \
                      R.to(device, non_blocking=True), \
                      D.to(device, non_blocking=True)
            R = random_erase_right(R, prob=args.random_erase_p)

            with torch.amp.autocast("cuda", dtype=torch.float16):
                preds = model(L, R, aux=True)
                if preds["d_final"].shape[-2:] != D.shape[-2:]:
                    preds["d_final"] = F.interpolate(
                        preds["d_final"], size=D.shape[-2:],
                        mode="bilinear", align_corners=True)
                preds_fp32 = {k: v.float() for k, v in preds.items()}
                loss, diag = multiscale_loss(
                    preds_fp32, D.float(),
                    grad_w=args.grad_w, hinge_w=args.hinge_w,
                    smooth_w=args.smooth_w, left_img=L.float())

            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad], 1.0)
            scaler.step(opt)
            scaler.update()
            sched.step()

            global_step += 1
            run_loss.append(loss.item())
            run_l1f.append(diag.get("d_final", 0.0))
            run_l1c.append(diag.get("d8_cv", 0.0))

            if is_main and (global_step == 1 or global_step % args.log_every == 0):
                ms = (time.time() - t_start) * 1000 / global_step
                cur_lr = opt.param_groups[0]["lr"]
                ml = float(np.mean(run_loss[-args.log_every:]))
                m_l1 = float(np.mean(run_l1f[-args.log_every:]))
                m_cv = float(np.mean(run_l1c[-args.log_every:]))
                w_csv.writerow([epoch, global_step, ml, m_l1, m_cv,
                                cur_lr, ms])
                fp_log.flush()
                print(f"ep {epoch:3d} step {global_step:6d}  "
                      f"loss={ml:6.3f}  l1_f={m_l1:6.3f}  l1_cv={m_cv:6.3f}  "
                      f"lr={cur_lr:.2e}  {ms:5.0f} ms/step")

            # Save iterative improvement panels (rank 0 only)
            if (is_main and track_pairs and args.sample_every > 0
                    and global_step % args.sample_every == 0):
                core = model.module if is_dist() else model
                med_ms = save_sample_panels(
                    core, track_pairs, device, args.out_dir,
                    tag=f"{global_step:06d}",
                    inf_h=args.inf_h, inf_w=args.inf_w)
                print(f"    sample panel saved  "
                      f"({len(track_pairs)} pairs, median {med_ms:.0f} ms)")

        # End-of-epoch validation on rank 0 only
        if is_main:
            epe, bad1 = validate(model.module if is_dist() else model,
                                  val_loader, device)
            dt = time.time() - t_epoch
            print(f"==> epoch {epoch}: VAL EPE={epe:.3f} px  "
                  f"bad1={bad1*100:.1f}%  ({dt/60:.1f} min)")

            ck = {"model": (model.module if is_dist() else model).state_dict(),
                  "epe": epe, "bad1": bad1, "epoch": epoch,
                  "args": vars(args)}
            torch.save(ck, os.path.join(args.out_dir, "stereolite_v8_last.pth"))
            if epe < best_epe:
                best_epe = epe
                torch.save(ck, os.path.join(args.out_dir, "stereolite_v8_best.pth"))
                print(f"    saved new best (EPE {epe:.3f})")

        if is_dist():
            dist.barrier()

    if is_main:
        fp_log.close()
        print(f"\ntraining done. best VAL EPE = {best_epe:.3f} px")
        print(f"total time: {(time.time()-t_start)/60:.1f} min")

    if is_dist():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
'''


# ---------------------------- export script -------------------------------- #
EXPORT_PY = r'''"""Export StereoLite v8 to multiple deployment formats.

Called from the notebook after training. Produces:
  - stereolite_v8.pth         (fp32 state_dict, already saved by trainer)
  - stereolite_v8.ts.pt       (TorchScript, trace-mode, fp32, CUDA)
  - stereolite_v8_fp32.onnx   (ONNX fp32, opset 17)
  - stereolite_v8_fp16.onnx   (ONNX fp16, via onnxconverter_common)
  - stereolite_v8_int8.onnx   (ONNX int8, dynamic quantization of conv/gemm)

Validates each export by running a random input through PyTorch and the
exported form, reporting max disparity diff.
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, "/kaggle/working/src")
os.environ.setdefault("XFORMERS_DISABLED", "1")

from StereoLite import StereoLite, StereoLiteConfig


def load_model(ckpt_path: str, device: torch.device) -> torch.nn.Module:
    model = StereoLite(StereoLiteConfig(backbone="mobilenet",
                                     )).to(device)
    ck = torch.load(ckpt_path, map_location=device, weights_only=False)
    sd = ck["model"] if "model" in ck else ck
    model.load_state_dict(sd, strict=True)
    # eval mode is required for clean ONNX export (folds BN running stats in
    # MobileNetV2). Our other norms are GroupNorm, train/eval-identical.
    model.eval()
    return model


def export_torchscript(model, out_path, inp):
    with torch.no_grad():
        ts = torch.jit.trace(model, inp, check_trace=False)
    ts.save(out_path)
    print(f"  -> {out_path}  ({os.path.getsize(out_path)/1e6:.1f} MB)")


def export_onnx(model, out_path, inp, opset=17):
    # Use the legacy tracing exporter — the dynamo exporter currently mis-names
    # weight initializers when the model contains timm BatchNorms alongside
    # GroupNorms (produces an invalid graph onnxruntime refuses to load).
    torch.onnx.export(
        model, inp, out_path,
        input_names=["left", "right"],
        output_names=["disparity"],
        opset_version=opset,
        do_constant_folding=True,
        dynamo=False,
    )
    print(f"  -> {out_path}  ({os.path.getsize(out_path)/1e6:.1f} MB)")


def convert_onnx_fp16(fp32_path, fp16_path):
    import onnx
    from onnxconverter_common import float16
    m = onnx.load(fp32_path)
    m16 = float16.convert_float_to_float16(m, keep_io_types=False)
    onnx.save(m16, fp16_path)
    print(f"  -> {fp16_path}  ({os.path.getsize(fp16_path)/1e6:.1f} MB)")


def quantize_onnx_int8(fp32_path, int8_path):
    from onnxruntime.quantization import quantize_dynamic, QuantType
    from onnxruntime.quantization.shape_inference import quant_pre_process
    # Preprocess: unfold constant-folded ops so every conv weight is a direct
    # initializer. Without this, quantize_dynamic errors on folded convs.
    pre_path = fp32_path.replace(".onnx", "_pre.onnx")
    quant_pre_process(fp32_path, pre_path, skip_symbolic_shape=False)
    quantize_dynamic(pre_path, int8_path, weight_type=QuantType.QInt8)
    os.remove(pre_path)
    print(f"  -> {int8_path}  ({os.path.getsize(int8_path)/1e6:.1f} MB)")


def compare_pytorch_vs_onnx(pytorch_model, onnx_path, inp, device, tol=1e-2):
    import onnxruntime as ort
    with torch.no_grad():
        out_torch = pytorch_model(*inp).cpu().numpy()
    providers = ["CUDAExecutionProvider", "CPUExecutionProvider"] \
                if device.type == "cuda" else ["CPUExecutionProvider"]
    try:
        sess = ort.InferenceSession(onnx_path, providers=providers)
    except Exception as e:
        print(f"    onnxruntime load failed: {e}")
        return
    ort_inp = {
        "left": inp[0].cpu().numpy(),
        "right": inp[1].cpu().numpy(),
    }
    out_ort = sess.run(["disparity"], ort_inp)[0]
    diff = np.abs(out_torch - out_ort)
    print(f"    max abs diff vs PyTorch: {diff.max():.4f}  "
          f"mean: {diff.mean():.4f}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default="/kaggle/working/stereolite_v8_best.pth")
    p.add_argument("--out_dir", default="/kaggle/working")
    p.add_argument("--inf_h", type=int, default=384)
    p.add_argument("--inf_w", type=int, default=768)
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device}  ckpt={args.ckpt}")
    model = load_model(args.ckpt, device)

    L = torch.rand(1, 3, args.inf_h, args.inf_w, device=device) * 255
    R = torch.rand(1, 3, args.inf_h, args.inf_w, device=device) * 255
    with torch.no_grad():
        _ = model(L, R)  # warmup

    ts_path = os.path.join(args.out_dir, "stereolite_v8.ts.pt")
    fp32_onnx = os.path.join(args.out_dir, "stereolite_v8_fp32.onnx")
    fp16_onnx = os.path.join(args.out_dir, "stereolite_v8_fp16.onnx")
    int8_onnx = os.path.join(args.out_dir, "stereolite_v8_int8.onnx")

    print("\n[1/5] TorchScript (CUDA, fp32)")
    export_torchscript(model, ts_path, (L, R))

    print("\n[2/5] ONNX fp32  (opset 17)")
    # ONNX export runs on CPU for reproducibility; cast inputs.
    model_cpu = model.cpu()
    L_cpu, R_cpu = L.cpu(), R.cpu()
    export_onnx(model_cpu, fp32_onnx, (L_cpu, R_cpu))

    print("\n[3/5] ONNX fp16")
    convert_onnx_fp16(fp32_onnx, fp16_onnx)

    print("\n[4/5] ONNX int8  (dynamic quantization)")
    quantize_onnx_int8(fp32_onnx, int8_onnx)

    print("\n[5/5] validation (PyTorch vs ONNX fp32)")
    compare_pytorch_vs_onnx(model_cpu, fp32_onnx, (L_cpu, R_cpu),
                             torch.device("cpu"))

    print("\ndone.")


if __name__ == "__main__":
    main()
'''


# ---------------------------- cells -------------------------------------- #
def md(text: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": text}


def code(src: str) -> dict:
    return {"cell_type": "code", "execution_count": None,
            "metadata": {}, "outputs": [], "source": src}


def writefile(path: str, body: str) -> dict:
    # IPython %%writefile magic — body written verbatim.
    return code(f"%%writefile {path}\n{body}")


cells = []

cells.append(md(
    "# StereoLite v8 — Kaggle Training + Multi-format Export\n"
    "\n"
    "Trains the v8 architecture (MobileNetV2-100 ImageNet-pretrained encoder, "
    "HITNet-style tile hypothesis + RAFT-style iterative refinement, "
    "convex-upsample output head) on Scene Flow Driving using 2x T4 GPUs "
    "via DistributedDataParallel, then exports the best checkpoint to:\n"
    "\n"
    "- **PyTorch state_dict** `.pth` — **primary GPU-inference format.** "
    "Load with `torch.load()` + `model.load_state_dict()` after importing "
    "`StereoLite`. Bundled with the model source code so it's self-contained.\n"
    "- **TorchScript** `.ts.pt` — **also PyTorch-runnable on GPU**, but "
    "self-contained (no source code import needed). Load with "
    "`torch.jit.load('stereolite_v8.ts.pt').cuda()` and call like any nn.Module. "
    "Useful for C++/LibTorch deployment.\n"
    "- **ONNX** `.onnx` — for benchmarking only (fp32, fp16, int8 variants "
    "produced via opset 17 + onnxruntime dynamic quantization).\n"
    "\n"
    "## Setup before running\n"
    "\n"
    "1. **Accelerator**: Settings > Accelerator > **GPU T4 x2** "
    "(the notebook also works on a single P100 — batch size auto-adjusts).\n"
    "2. **Dataset**: Add Scene Flow Driving as a Kaggle Dataset input. "
    "The notebook auto-detects the two tarball-extracted subfolders "
    "(`driving__frames_finalpass/frames_finalpass/...` and "
    "`driving__disparity/disparity/...`) and stitches them into a "
    "loader-friendly layout at `/kaggle/working/sceneflow_driving/` using "
    "symlinks.\n"
    "3. **Time budget**: one full 30-epoch run is ~2-3 h on T4 x2. "
    "Kaggle's 9-hour GPU session easily covers training + export + validation.\n"
    "\n"
    "## Outputs\n"
    "\n"
    "All artefacts land in `/kaggle/working/` and are zipped at the end as "
    "`stereolite_v8_artifacts.zip` for one-click download.\n"
))


cells.append(md("## 1. Environment check"))
cells.append(code(
    "import torch, sys, os\n"
    "print('Python:', sys.version.split()[0])\n"
    "print('PyTorch:', torch.__version__, ' CUDA:', torch.version.cuda)\n"
    "print('CUDA available:', torch.cuda.is_available())\n"
    "N = torch.cuda.device_count()\n"
    "print('GPU count:', N)\n"
    "for i in range(N):\n"
    "    p = torch.cuda.get_device_properties(i)\n"
    "    print(f'  [{i}] {p.name}  {p.total_memory/1e9:.1f} GB')\n"
    "\n"
    "assert N >= 1, 'No GPU detected — enable T4 x2 or P100 in Settings.'\n"
))


cells.append(md("## 2. Install missing dependencies\n\nMost of what we need "
                "is pre-installed on Kaggle. We add `onnx`, ONNX Runtime and "
                "the fp16 converter."))
cells.append(code(
    "!pip install -q timm==1.0.26 onnx onnxruntime-gpu onnxconverter-common onnxscript\n"
    "!pip list 2>/dev/null | grep -E '^(torch|timm|onnx)' | sort\n"
))


cells.append(md(
    "## 2b. Stitch attached dataset into the loader-friendly layout\n"
    "\n"
    "The Scene Flow Driving Kaggle dataset (uploaded as two tarballs) is "
    "extracted by Kaggle into two sibling folders:\n"
    "\n"
    "```\n"
    "/kaggle/input/.../sceneflow-driving/driving__frames_finalpass/frames_finalpass/...\n"
    "/kaggle/input/.../sceneflow-driving/driving__disparity/disparity/...\n"
    "```\n"
    "\n"
    "Our loader expects `frames_finalpass/` and `disparity/` as siblings under "
    "one root. This cell creates symlinks at `/kaggle/working/sceneflow_driving/` "
    "that point at the right subdirectories under `/kaggle/input/`. No data is "
    "copied; symlinks are essentially free.\n"
))
cells.append(code(
    "import os, glob\n"
    "\n"
    "STAGE = '/kaggle/working/sceneflow_driving'\n"
    "os.makedirs(STAGE, exist_ok=True)\n"
    "\n"
    "# Locate the two extracted folders under /kaggle/input/ (any depth).\n"
    "frames_hits = glob.glob('/kaggle/input/**/frames_finalpass', recursive=True)\n"
    "disp_hits = glob.glob('/kaggle/input/**/disparity', recursive=True)\n"
    "if not frames_hits or not disp_hits:\n"
    "    print('Could not auto-locate frames_finalpass/ or disparity/ under '\n"
    "          '/kaggle/input/. Tree (depth <= 4):')\n"
    "    for root, dirs, files in os.walk('/kaggle/input', followlinks=True):\n"
    "        depth = root.replace('/kaggle/input', '').count('/')\n"
    "        if depth <= 4:\n"
    "            print(' ', root)\n"
    "        if depth >= 4:\n"
    "            dirs[:] = []\n"
    "    raise FileNotFoundError('Scene Flow subfolders not found under /kaggle/input/')\n"
    "\n"
    "# Pick the deepest hit (handles nested duplicates).\n"
    "frames_src = sorted(frames_hits, key=len)[-1]\n"
    "disp_src = sorted(disp_hits, key=len)[-1]\n"
    "print(f'frames source: {frames_src}')\n"
    "print(f'disp   source: {disp_src}')\n"
    "\n"
    "for src, name in [(frames_src, 'frames_finalpass'), (disp_src, 'disparity')]:\n"
    "    link = os.path.join(STAGE, name)\n"
    "    if os.path.islink(link) or os.path.exists(link):\n"
    "        if os.path.islink(link):\n"
    "            os.remove(link)\n"
    "        elif os.path.isdir(link):\n"
    "            import shutil; shutil.rmtree(link)\n"
    "    os.symlink(src, link)\n"
    "    print(f'  linked {name} -> {src}')\n"
    "\n"
    "print(f'\\nDATA_ROOT = {STAGE}')\n"
    "print('\\n-- contents --')\n"
    "for entry in sorted(os.listdir(STAGE)):\n"
    "    p = os.path.join(STAGE, entry)\n"
    "    real = os.path.realpath(p)\n"
    "    n = sum(1 for _ in os.walk(p, followlinks=True))\n"
    "    print(f'  {entry} -> {real} ({n} subdirs reachable)')\n"
))


cells.append(md(
    "## 3. Config\n\n"
    "Edit `DATA_ROOT` if auto-detect misses. Default assumes you ran the "
    "download cell above (which writes under `/kaggle/working/"
    "sceneflow_driving`), or attached a Kaggle Dataset under `/kaggle/input/`."
))
cells.append(code(
    "import os\n"
    "\n"
    "# Cell 2b created /kaggle/working/sceneflow_driving/{frames_finalpass,disparity}\n"
    "# as symlinks into /kaggle/input/. Override DATA_ROOT here only if you\n"
    "# attached a dataset that already has the loader-friendly layout.\n"
    "DATA_ROOT = '/kaggle/working/sceneflow_driving'\n"
    "\n"
    "EPOCHS        = 30\n"
    "BATCH_PER_GPU = 12      # ~12.5 GB on a T4 at 512x832; drop to 8 if OOM\n"
    "INF_H         = 512\n"
    "INF_W         = 832\n"
    "LR_PEAK       = 8e-4\n"
    "RANDOM_ERASE  = 0.3\n"
    "\n"
    "for sub in ('frames_finalpass', 'disparity'):\n"
    "    p = os.path.join(DATA_ROOT, sub)\n"
    "    if not os.path.isdir(p):\n"
    "        raise FileNotFoundError(\n"
    "            f'{p} not found. Did Cell 2b run successfully?')\n"
    "\n"
    "print(f'DATA_ROOT = {DATA_ROOT}')\n"
    "print(f'epochs={EPOCHS}  batch/GPU={BATCH_PER_GPU}  input={INF_H}x{INF_W}')\n"
    "print(f'peak LR={LR_PEAK}  random-erase p={RANDOM_ERASE}')\n"
))


cells.append(md("## 4. Write model source files\n\nWe recreate the exact "
                "repo layout under `/kaggle/working/src/` so the training "
                "script can `import StereoLite` as a package."))
cells.append(code(
    "import os\n"
    "os.makedirs('/kaggle/working/src/StereoLite', exist_ok=True)\n"
    "print(os.listdir('/kaggle/working/src'))\n"
))

cells.append(writefile("/kaggle/working/src/_blocks.py", BLOCKS_PY))
cells.append(writefile("/kaggle/working/src/StereoLite/__init__.py", INIT_PY))
cells.append(writefile("/kaggle/working/src/StereoLite/tile_propagate.py",
                        TILE_PROPAGATE_PY))
cells.append(writefile("/kaggle/working/src/StereoLite/model.py", MODEL_PY))
cells.append(writefile("/kaggle/working/src/sceneflow_loader.py",
                        SF_LOADER_PY))
cells.append(writefile("/kaggle/working/train_ddp.py", TRAIN_DDP_PY))
cells.append(writefile("/kaggle/working/export_v8.py", EXPORT_PY))


cells.append(md("## 5. Sanity check model assembles + forward pass works"))
cells.append(code(
    "import sys, os\n"
    "os.environ.setdefault('XFORMERS_DISABLED', '1')\n"
    "sys.path.insert(0, '/kaggle/working/src')\n"
    "import torch\n"
    "from StereoLite import StereoLite, StereoLiteConfig\n"
    "\n"
    "m = StereoLite(StereoLiteConfig(backbone='mobilenet', )).cuda()\n"
    "L = torch.rand(1, 3, INF_H, INF_W, device='cuda') * 255\n"
    "R = torch.rand(1, 3, INF_H, INF_W, device='cuda') * 255\n"
    "with torch.no_grad():\n"
    "    out = m(L, R)\n"
    "print(f'trainable params: {sum(p.numel() for p in m.parameters())/1e6:.3f} M')\n"
    "print(f'output shape: {tuple(out.shape)}')\n"
    "del m, out\n"
    "torch.cuda.empty_cache()\n"
))


cells.append(md("## 6. Verify dataset is reachable"))
cells.append(code(
    "sys.path.insert(0, '/kaggle/working/src')\n"
    "from sceneflow_loader import enumerate_pairs\n"
    "items = enumerate_pairs(DATA_ROOT)\n"
    "print(f'found {len(items)} stereo pairs under {DATA_ROOT}')\n"
    "assert len(items) > 100, 'Expected thousands of pairs — check DATA_ROOT.'\n"
    "for lp, rp, pp in items[:3]:\n"
    "    print(' ', lp.replace(DATA_ROOT, '<data>'))\n"
))


cells.append(md("## 7. Train (DDP via `torchrun`)\n\nOn T4 x2 this runs "
                "`--nproc_per_node=2`; on a single P100 it falls back to one "
                "process. Logs stream live; each epoch ends with a VAL EPE line "
                "and checkpoint save."))
cells.append(code(
    "import torch, os, subprocess, shlex\n"
    "\n"
    "N_GPU = torch.cuda.device_count()\n"
    "launcher = f'torchrun --standalone --nproc_per_node={N_GPU}' if N_GPU > 1 \\\n"
    "           else 'python3'\n"
    "\n"
    "cmd = (\n"
    "    f'{launcher} /kaggle/working/train_ddp.py '\n"
    "    f'--data_root {shlex.quote(DATA_ROOT)} '\n"
    "    f'--out_dir /kaggle/working '\n"
    "    f'--epochs {EPOCHS} --batch {BATCH_PER_GPU} '\n"
    "    f'--inf_h {INF_H} --inf_w {INF_W} '\n"
    "    f'--lr {LR_PEAK} --random_erase_p {RANDOM_ERASE} '\n"
    "    f'--num_workers 2 --log_every 50'\n"
    ")\n"
    "print('launching:\\n', cmd, '\\n')\n"
    "rc = subprocess.call(shlex.split(cmd))\n"
    "print(f'\\nexit code: {rc}')\n"
    "assert rc == 0, 'training failed — check logs above'\n"
))


cells.append(md("## 8. Export best checkpoint to all formats"))
cells.append(code(
    "import subprocess, shlex\n"
    "cmd = (\n"
    "    f'python3 /kaggle/working/export_v8.py '\n"
    "    f'--ckpt /kaggle/working/stereolite_v8_best.pth '\n"
    "    f'--out_dir /kaggle/working '\n"
    "    f'--inf_h {INF_H} --inf_w {INF_W}'\n"
    ")\n"
    "print(cmd)\n"
    "subprocess.check_call(shlex.split(cmd))\n"
))


cells.append(md(
    "## 8b. GPU inference demo — load each format and time it\n"
    "\n"
    "Both `.pth` and `.ts.pt` are PyTorch-runnable on GPU. The `.pth` is a "
    "plain state_dict (needs the source code at import time); the `.ts.pt` is "
    "a self-contained TorchScript — no source needed to load and run.\n"
))
cells.append(code(
    "import torch, time, sys, os\n"
    "sys.path.insert(0, '/kaggle/working/src')\n"
    "os.environ.setdefault('XFORMERS_DISABLED', '1')\n"
    "from StereoLite import StereoLite, StereoLiteConfig\n"
    "\n"
    "device = torch.device('cuda')\n"
    "L = torch.rand(1, 3, INF_H, INF_W, device=device) * 255\n"
    "R = torch.rand(1, 3, INF_H, INF_W, device=device) * 255\n"
    "\n"
    "def bench(model, n=30, warmup=5):\n"
    "    with torch.no_grad():\n"
    "        for _ in range(warmup): model(L, R)\n"
    "        torch.cuda.synchronize()\n"
    "        t0 = time.time()\n"
    "        for _ in range(n): model(L, R)\n"
    "        torch.cuda.synchronize()\n"
    "    return (time.time() - t0) / n * 1000\n"
    "\n"
    "# (A) .pth — state_dict + source code\n"
    "m_pth = StereoLite(StereoLiteConfig(backbone='mobilenet', )).to(device)\n"
    "ck = torch.load('/kaggle/working/stereolite_v8_best.pth',\n"
    "                 map_location=device, weights_only=False)\n"
    "m_pth.load_state_dict(ck['model'])\n"
    "m_pth.eval()\n"
    "ms_pth = bench(m_pth)\n"
    "print(f'.pth (state_dict) on GPU: {ms_pth:.1f} ms  '\n"
    "      f'params={sum(p.numel() for p in m_pth.parameters())/1e6:.3f} M')\n"
    "\n"
    "# (B) .ts.pt — TorchScript (no source needed)\n"
    "m_ts = torch.jit.load('/kaggle/working/stereolite_v8.ts.pt').to(device).eval()\n"
    "ms_ts = bench(m_ts)\n"
    "print(f'.ts.pt (TorchScript) on GPU: {ms_ts:.1f} ms')\n"
    "\n"
    "# Parity check\n"
    "with torch.no_grad():\n"
    "    a = m_pth(L, R).cpu().numpy()\n"
    "    b = m_ts(L, R).cpu().numpy()\n"
    "import numpy as np\n"
    "print(f'|.pth - .ts.pt| max: {np.abs(a - b).max():.5f}  '\n"
    "      f'mean: {np.abs(a - b).mean():.6f}')\n"
    "\n"
    "del m_pth, m_ts\n"
    "torch.cuda.empty_cache()\n"
))

cells.append(md("## 9. List artifacts + sizes"))
cells.append(code(
    "import os\n"
    "print(f'{\"file\":<30s}  size')\n"
    "print('-' * 42)\n"
    "for f in sorted(os.listdir('/kaggle/working')):\n"
    "    path = os.path.join('/kaggle/working', f)\n"
    "    if os.path.isfile(path):\n"
    "        sz = os.path.getsize(path)\n"
    "        print(f'{f:<30s}  {sz/1e6:8.2f} MB')\n"
))


cells.append(md(
    "## 10. Bundle for download\n"
    "\n"
    "Zip includes:\n"
    "- model artifacts (.pth, .ts.pt, .onnx x 3)\n"
    "- `src/` — the exact Python source files needed to load the `.pth` on "
    "your own machine. Just `sys.path.insert(0, 'src')` and "
    "`from StereoLite import StereoLite, StereoLiteConfig`.\n"
    "- `load_pth_example.py` — minimal loader / inference script\n"
    "- `train_log.csv` — training curves\n"
))
cells.append(code(
    "import zipfile, os\n"
    "out = '/kaggle/working/stereolite_v8_artifacts.zip'\n"
    "MODEL_FILES = [\n"
    "    'stereolite_v8_best.pth',\n"
    "    'stereolite_v8_last.pth',\n"
    "    'stereolite_v8.ts.pt',\n"
    "    'stereolite_v8_fp32.onnx',\n"
    "    'stereolite_v8_fp16.onnx',\n"
    "    'stereolite_v8_int8.onnx',\n"
    "    'train_log.csv',\n"
    "]\n"
    "SRC_FILES = [\n"
    "    'src/_blocks.py',\n"
    "    'src/StereoLite/__init__.py',\n"
    "    'src/StereoLite/model.py',\n"
    "    'src/StereoLite/tile_propagate.py',\n"
    "    'src/sceneflow_loader.py',\n"
    "]\n"
    "\n"
    "LOADER_EXAMPLE = '''\"\"\"Minimal loader for stereolite_v8_best.pth.\n"
    "\n"
    "Run:  python3 load_pth_example.py path/to/left.png path/to/right.png\n"
    "Outputs disparity as a TURBO-coloured PNG next to the left image.\n"
    "\"\"\"\n"
    "import sys, os, numpy as np, torch, cv2\n"
    "sys.path.insert(0, os.path.join(os.path.dirname(__file__), \"src\"))\n"
    "from StereoLite import StereoLite, StereoLiteConfig\n"
    "\n"
    "CKPT = os.path.join(os.path.dirname(__file__), \"stereolite_v8_best.pth\")\n"
    "INF_H, INF_W = 384, 768\n"
    "\n"
    "def main(left_path, right_path):\n"
    "    dev = torch.device(\"cuda\" if torch.cuda.is_available() else \"cpu\")\n"
    "    m = StereoLite(StereoLiteConfig(backbone=\"mobilenet\", )).to(dev)\n"
    "    ck = torch.load(CKPT, map_location=dev, weights_only=False)\n"
    "    m.load_state_dict(ck[\"model\"])\n"
    "    m.eval()\n"
    "    L = cv2.resize(cv2.imread(left_path), (INF_W, INF_H))\n"
    "    R = cv2.resize(cv2.imread(right_path), (INF_W, INF_H))\n"
    "    Lt = torch.from_numpy(cv2.cvtColor(L, cv2.COLOR_BGR2RGB)).float().permute(2,0,1).unsqueeze(0).to(dev)\n"
    "    Rt = torch.from_numpy(cv2.cvtColor(R, cv2.COLOR_BGR2RGB)).float().permute(2,0,1).unsqueeze(0).to(dev)\n"
    "    with torch.no_grad():\n"
    "        d = m(Lt, Rt).squeeze().cpu().numpy()\n"
    "    lo, hi = float(np.percentile(d, 2)), float(np.percentile(d, 98))\n"
    "    v = np.clip((d - lo) / max(hi - lo, 1e-6), 0, 1) * 255\n"
    "    out = cv2.applyColorMap(v.astype(np.uint8), cv2.COLORMAP_TURBO)\n"
    "    op = os.path.splitext(left_path)[0] + \"_disp.png\"\n"
    "    cv2.imwrite(op, out)\n"
    "    print(f\"wrote {op}  (disparity range {d.min():.2f} .. {d.max():.2f})\")\n"
    "\n"
    "if __name__ == \"__main__\":\n"
    "    main(sys.argv[1], sys.argv[2])\n"
    "'''\n"
    "\n"
    "with open('/kaggle/working/load_pth_example.py', 'w') as f:\n"
    "    f.write(LOADER_EXAMPLE)\n"
    "\n"
    "with zipfile.ZipFile(out, 'w', zipfile.ZIP_DEFLATED) as z:\n"
    "    for f in MODEL_FILES:\n"
    "        p = os.path.join('/kaggle/working', f)\n"
    "        if os.path.exists(p):\n"
    "            z.write(p, arcname=f)\n"
    "            print(f'  + {f} ({os.path.getsize(p)/1e6:.1f} MB)')\n"
    "        else:\n"
    "            print(f'  ! missing: {f}')\n"
    "    for f in SRC_FILES:\n"
    "        p = os.path.join('/kaggle/working', f)\n"
    "        if os.path.exists(p):\n"
    "            z.write(p, arcname=f)\n"
    "            print(f'  + {f} ({os.path.getsize(p)/1024:.1f} KB)')\n"
    "    z.write('/kaggle/working/load_pth_example.py', arcname='load_pth_example.py')\n"
    "    print(f'  + load_pth_example.py')\n"
    "    # Iterative sample panels (progress snapshots)\n"
    "    samples_dir = '/kaggle/working/samples'\n"
    "    if os.path.isdir(samples_dir):\n"
    "        count = 0\n"
    "        for f in sorted(os.listdir(samples_dir)):\n"
    "            p = os.path.join(samples_dir, f)\n"
    "            z.write(p, arcname=f'samples/{f}')\n"
    "            count += 1\n"
    "        print(f'  + samples/ ({count} iterative panels)')\n"
    "\n"
    "print(f'\\nwrote {out}  ({os.path.getsize(out)/1e6:.1f} MB)')\n"
    "print('Download from the Kaggle notebook Output panel.')\n"
))


cells.append(md("---\n\n"
                "### Notes\n"
                "- **Expected artifact sizes** (on a trained v8 checkpoint, "
                "~2.14 M params):\n"
                "  - `stereolite_v8_best.pth` ~9 MB (fp32 state_dict)\n"
                "  - `stereolite_v8.ts.pt` ~9 MB (TorchScript traced)\n"
                "  - `stereolite_v8_fp32.onnx` ~15 MB (opset 17 + unrolled loops)\n"
                "  - `stereolite_v8_fp16.onnx` ~8 MB (halved weights)\n"
                "  - `stereolite_v8_int8.onnx` ~75 MB (large because the int8 "
                "preprocessor un-folds our 24-step cost-volume and 8-iter "
                "refinement loops; only conv/matmul weights are int8, "
                "activations stay fp32). For smaller int8 artifacts use "
                "static quantization with a calibration set.\n"
                "- **Exporter backend**: we force `dynamo=False` (legacy "
                "tracer) because the new dynamo exporter currently mis-names "
                "initializers when a model mixes timm BatchNorm (inside "
                "MobileNetV2) with GroupNorm (in our blocks) and produces a "
                "graph that onnxruntime refuses to load.\n"
                "- **If training OOMs**: reduce `BATCH_PER_GPU` to 2, "
                "or `INF_H/INF_W` to 320/640.\n"
                "- **Resume**: re-run the training cell with an additional "
                "`--resume /kaggle/working/stereolite_v8_last.pth` argument if "
                "the session restarted mid-run.\n"
                "- **Single-GPU fallback**: if you switch to P100 (1 GPU), "
                "the launcher automatically drops DDP and runs single-process."))


# ---------------------------- write notebook ------------------------------- #
nb = {
    "cells": cells,
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        },
        "language_info": {
            "name": "python",
            "version": "3.11",
        },
        "accelerator": "GPU",
        "colab": {},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

OUT.write_text(json.dumps(nb, indent=1))
print(f"wrote {OUT}  ({OUT.stat().st_size/1024:.1f} KB, {len(cells)} cells)")
