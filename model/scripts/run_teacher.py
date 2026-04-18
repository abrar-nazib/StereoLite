"""Run FoundationStereo (ViT-Small variant, 11-33-40) on saved stereo pairs
and write per-pair pseudo-GT disparity to disk.

Output layout (alongside data/pairs/{left,right}/):
    data/pairs/disp_pseudo/00000.npy   (float32 disparity, full input resolution)
    data/pairs/disp_vis/00000.png      (turbo-coloured visualisation)
    data/pairs/teacher_log.txt         (timing + per-pair stats)

Resize: input pairs are read at native 1280x720 and downscaled to
INF_SIZE before passing to the teacher. Disparity is then rescaled to the
saved resolution by * (saved_W / inf_W) so that pseudo-GT and stored images
are on a common scale that students can use directly.

Tested-only: torch.no_grad, FP16 if --fp16 is set (helps fit on 3.68 GB
RTX 3050 Laptop), --valid_iters 16 to keep latency reasonable.
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import cv2
import numpy as np
import torch

PROJ = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
TEACHER_DIR = os.path.join(PROJ, "model", "teachers", "FoundationStereo")
sys.path.insert(0, TEACHER_DIR)

# xformers fallback for DINOv2 internals
os.environ.setdefault("XFORMERS_DISABLED", "1")


def load_teacher(ckpt_dir: str, device: torch.device, valid_iters: int = 16):
    from omegaconf import OmegaConf
    from core.foundation_stereo import FoundationStereo

    cfg_path = os.path.join(os.path.dirname(ckpt_dir), "cfg.yaml")
    cfg = OmegaConf.load(cfg_path)
    if "vit_size" not in cfg:
        cfg["vit_size"] = "vits"
    cfg["valid_iters"] = valid_iters
    cfg["mixed_precision"] = True
    cfg["low_memory"] = True
    cfg["hiera"] = 0
    cfg["scale"] = 1.0

    model = FoundationStereo(cfg).to(device).eval()
    sd = torch.load(ckpt_dir, map_location=device, weights_only=False)
    if "model" in sd:
        sd = sd["model"]
    missing, unexpected = model.load_state_dict(sd, strict=False)
    print(f"  loaded teacher (missing={len(missing)} unexpected={len(unexpected)})")
    return model, cfg


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--pairs_dir", default=os.path.join(PROJ, "data", "pairs"))
    p.add_argument("--ckpt", default=os.path.join(
        PROJ, "model", "teachers", "FoundationStereo",
        "pretrained_models", "11-33-40", "model_best_bp2.pth"))
    p.add_argument("--inf_h", type=int, default=384)
    p.add_argument("--inf_w", type=int, default=640)
    p.add_argument("--save_h", type=int, default=384)
    p.add_argument("--save_w", type=int, default=640)
    p.add_argument("--valid_iters", type=int, default=16)
    p.add_argument("--max_pairs", type=int, default=None)
    args = p.parse_args()

    out_npy = os.path.join(args.pairs_dir, "disp_pseudo")
    out_vis = os.path.join(args.pairs_dir, "disp_vis")
    os.makedirs(out_npy, exist_ok=True)
    os.makedirs(out_vis, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device.type}; loading teacher from {args.ckpt}")
    model, cfg = load_teacher(args.ckpt, device, args.valid_iters)

    left_dir = os.path.join(args.pairs_dir, "left")
    right_dir = os.path.join(args.pairs_dir, "right")
    files = sorted(os.listdir(left_dir))
    if args.max_pairs is not None:
        files = files[: args.max_pairs]

    log_path = os.path.join(args.pairs_dir, "teacher_log.txt")
    log_lines: list[str] = [
        f"# FoundationStereo teacher pseudo-GT log",
        f"# ckpt={args.ckpt}",
        f"# inf_size={args.inf_h}x{args.inf_w}, save_size={args.save_h}x{args.save_w}",
        f"# valid_iters={args.valid_iters}",
        f"# device={device.type}",
        "",
    ]

    times: list[float] = []
    for i, fname in enumerate(files):
        lp = os.path.join(left_dir, fname)
        rp = os.path.join(right_dir, fname)
        l = cv2.imread(lp)
        r = cv2.imread(rp)
        if l is None or r is None:
            print(f"  skip {fname}: read failed")
            continue
        # Resize to inference size; disparity is computed at this scale.
        lr = cv2.resize(l, (args.inf_w, args.inf_h), interpolation=cv2.INTER_AREA)
        rr = cv2.resize(r, (args.inf_w, args.inf_h), interpolation=cv2.INTER_AREA)
        lt = torch.from_numpy(cv2.cvtColor(lr, cv2.COLOR_BGR2RGB)).float()
        rt = torch.from_numpy(cv2.cvtColor(rr, cv2.COLOR_BGR2RGB)).float()
        lt = lt.permute(2, 0, 1).unsqueeze(0).to(device)
        rt = rt.permute(2, 0, 1).unsqueeze(0).to(device)

        try:
            t0 = time.time()
            with torch.no_grad():
                disp = model(lt, rt, iters=args.valid_iters, test_mode=True)
            if device.type == "cuda":
                torch.cuda.synchronize()
            dt = (time.time() - t0) * 1000
        except torch.cuda.OutOfMemoryError as exc:
            print(f"OOM at frame {i}: {exc}")
            torch.cuda.empty_cache()
            continue

        d_inf = disp.squeeze().cpu().numpy()                # (inf_h, inf_w)
        # Rescale disparity to save resolution
        if (args.save_h, args.save_w) != (args.inf_h, args.inf_w):
            scale_x = args.save_w / args.inf_w
            d_save = cv2.resize(d_inf, (args.save_w, args.save_h),
                                interpolation=cv2.INTER_LINEAR) * scale_x
        else:
            d_save = d_inf
        np.save(os.path.join(out_npy, fname.replace(".png", ".npy")),
                d_save.astype(np.float32))

        # Colour viz
        d = np.clip(d_save, 0, None)
        if d.max() > d.min():
            v = (d - d.min()) / (d.max() - d.min()) * 255
        else:
            v = np.zeros_like(d)
        cv2.imwrite(os.path.join(out_vis, fname),
                    cv2.applyColorMap(v.astype(np.uint8), cv2.COLORMAP_TURBO))

        times.append(dt)
        line = (f"{fname}  {dt:7.1f} ms  "
                f"disp[min={float(d_inf.min()):.1f} "
                f"max={float(d_inf.max()):.1f} "
                f"mean={float(d_inf.mean()):.1f}]")
        log_lines.append(line)
        if i % 25 == 0 or i == len(files) - 1:
            print(f"  [{i+1}/{len(files)}] {line}")

    if times:
        med = float(np.median(times))
        log_lines.append("")
        log_lines.append(f"# median per-pair latency: {med:.1f} ms ({1000/med:.2f} fps)")
        print(f"\nDone. median teacher latency: {med:.1f} ms ({1000/med:.2f} fps)")
    with open(log_path, "w") as fp:
        fp.write("\n".join(log_lines) + "\n")
    print(f"wrote {log_path}")


if __name__ == "__main__":
    main()
