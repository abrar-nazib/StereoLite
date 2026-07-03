"""Efficiency-fix validation: 3-arm overfit study on 100 Scene Flow Driving
pairs (80 train / 20 held-out val), plateau-based early stopping.

Arms (--arch):
  gev4             control — original StereoLite_yolo_ctx_gev4
  gev4_opt         F1/F2/F4/F5/F7 safe fixes (metric-equivalence proven on
                   RTX 3050: max EPE delta 3.1e-5 px, 1.29x faster fp32)
  gev4_opt_narrow  + F3 narrow GEV (33 bins around tile.d vs full 64) —
                   ACCURACY-AFFECTING, this run is its A/B

Protocol per ablation-study-expert skill + user spec (2026-07-03):
  - 100 random pairs across all 8 Driving sequences, seed 42, split 80/20
  - input 384x640, images fed as [0,1] (overfit-harness convention)
  - loss: msL1{1,.5,.3,.2,.1} + 0.5 grad + 0.2 bad1 + 0.15 gev4 branch
  - steps up to --max_steps (default 12000); PLATEAU STOP: after
    --min_steps, stop when the best val EPE hasn't improved by >1% over
    the last --patience evals
  - every eval: 6-tile annotated collage (1 GT + 3 train preds + 2 val
    preds, each pred annotated with its own EPE/RMSE/bad-0.5/1/2/3/D1)
    -> viz/collage_step_NNNNN.png, showing iterative improvement
  - artifacts: meta.json (full schema), train.csv (8 metrics / 100 steps),
    curve.png (loss + train/val EPE), checkpoint.pth (final) + best.pth
    (best val EPE), EXPERIMENTS.md rebuild

Example:
    python model/scripts/overfit_efficiency_ablation.py --arch gev4_opt \
        --run_name eff_20260703 --show 0
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import os
import platform as _platform
import random
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F

_SCRIPTS = Path(__file__).resolve().parent
_DESIGNS = _SCRIPTS.parent / "designs"
sys.path.insert(0, str(_SCRIPTS))
sys.path.insert(0, str(_DESIGNS))

from overfit_yolo_ablation import stereo_metrics, _colorize_disp  # noqa: E402

DATA_ROOT = Path(os.environ.get(
    "STEREO_DATA_ROOT", "/media/abrar/AbrarSSD/Datasets/sceneflow_driving"))
TRAIN_W, TRAIN_H = 640, 384
NATIVE_W = 960
MAX_DISP = 192.0
ALL_SEQS = [f"{fl}/{d}/{s}" for fl in ("35mm_focallength", "15mm_focallength")
            for d in ("scene_forwards", "scene_backwards")
            for s in ("slow", "fast")]


def build_model(arch: str):
    if arch == "gev4":
        from StereoLite_yolo_ctx_gev4.model import (
            StereoLiteYoloCtxGEV4, StereoLiteYoloCtxGEV4Config)
        cfg = StereoLiteYoloCtxGEV4Config()
        return StereoLiteYoloCtxGEV4(cfg), cfg
    if arch.startswith("costlookup"):
        # The exact pre-rahi project leader ("gce_in_tileinit_combo",
        # EPE 0.811 on the legacy 100-pair protocol): costlookup chassis
        # + extend_to_full + cascade_cv_4 + slope_aware_warp + init_gce.
        # y26n uses the ghostconv widener (the validated pairing);
        # y26s runs the native encoder (its historical comparison point).
        from StereoLite_costlookup.model import (
            StereoLite, StereoLiteConfig)
        common = dict(extend_to_full=True, cascade_cv_4=True,
                      slope_aware_warp=True, init_gce=True)
        if arch == "costlookup_y26n":
            cfg = StereoLiteConfig(backbone="yolo26n", widener="ghostconv",
                                   **common)
        elif arch == "costlookup_y26s":
            cfg = StereoLiteConfig(backbone="yolo26s", widener=None, **common)
        else:
            raise ValueError(arch)
        return StereoLite(cfg), cfg
    from StereoLite_yolo_ctx_gev4_opt.model import (
        StereoLiteYoloCtxGEV4, StereoLiteYoloCtxGEV4Config)
    if arch == "gev4_opt":
        cfg = StereoLiteYoloCtxGEV4Config()
    elif arch == "gev4_opt_narrow":
        cfg = StereoLiteYoloCtxGEV4Config(narrow_gev=True, gev_half_range=16)
    elif arch == "gev4_opt_narrow_sharptail":
        # narrow core + pre-rahi costlookup tail (1/2 refine + plane-eq up)
        cfg = StereoLiteYoloCtxGEV4Config(narrow_gev=True, gev_half_range=16,
                                          sharp_tail=True)
    elif arch == "gev4_opt_narrow_bundle1":
        # blur bundle-1 arch side: top-k=3 init (pair with --trunc_A 1.0
        # --init_ce_w 0.3 for the full bundle)
        cfg = StereoLiteYoloCtxGEV4Config(narrow_gev=True, gev_half_range=16,
                                          init_topk=3)
    else:
        raise ValueError(arch)
    return StereoLiteYoloCtxGEV4(cfg), cfg


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def _read_pfm(path):
    with open(path, "rb") as fh:
        assert fh.readline().decode().rstrip() in ("Pf", "PF")
        w, h = map(int, fh.readline().decode().split())
        scale = float(fh.readline().decode().rstrip())
        data = np.fromfile(fh, "<f" if scale < 0 else ">f")
        return np.flipud(data.reshape(h, w)).copy()


def _load_pair(seq: str, t: int):
    fp = DATA_ROOT / "frames_finalpass" / seq
    dp = DATA_ROOT / "disparity" / seq
    ims = []
    for side in ("left", "right"):
        im = cv2.imread(str(fp / side / f"{t:04d}.png"), cv2.IMREAD_COLOR)
        im = cv2.resize(im, (TRAIN_W, TRAIN_H), interpolation=cv2.INTER_AREA)
        ims.append(torch.from_numpy(im[..., ::-1].copy()).permute(2, 0, 1)
                   .to(torch.uint8))
    d = np.abs(_read_pfm(dp / "left" / f"{t:04d}.pfm")).astype(np.float32)
    d = cv2.resize(d, (TRAIN_W, TRAIN_H), interpolation=cv2.INTER_NEAREST)
    d = np.nan_to_num(d * (TRAIN_W / NATIVE_W), nan=0.0, posinf=0.0)
    return dict(seq=seq, t=t, L=ims[0], R=ims[1],
                D=torch.from_numpy(d)[None].to(torch.float16))


def load_or_build_pairs(args):
    """Random (seq, frame) samples with a LEAK-PROOF train/val split.

    Val = randomly-placed contiguous windows inside sequences; a
    +-`buffer` frame zone around every val window is excluded from the
    train pool. This prevents the near-duplicate-neighbor leak (val frame
    t with train frames t-1/t+1 from the same sequence), while both the
    window placement and the train sampling stay random (user spec,
    2026-07-03).
    """
    if args.pairs_cache and Path(args.pairs_cache).exists():
        blob = torch.load(args.pairs_cache, map_location="cpu",
                          weights_only=False)
        print(f"pairs cache: {args.pairs_cache} "
              f"({len(blob['train'])} train / {len(blob['val'])} val)"
              + (f" split={blob.get('split_protocol', 'legacy-random')}"))
        return blob["train"], blob["val"]

    rng = random.Random(args.seed)
    win, buffer = 5, 10
    n_windows = max(args.n_val // win, 1)

    def frame_exists(s, t):
        return ((DATA_ROOT / "frames_finalpass" / s / "left" / f"{t:04d}.png").exists()
                and (DATA_ROOT / "frames_finalpass" / s / "right" / f"{t:04d}.png").exists()
                and (DATA_ROOT / "disparity" / s / "left" / f"{t:04d}.pfm").exists())

    # discover per-sequence frame ranges (Driving frames are 1..N contiguous)
    seq_max = {}
    for s in ALL_SEQS:
        hi = 0
        for probe in (800, 500, 400, 300, 200, 100):
            if frame_exists(s, probe):
                hi = probe
                break
        while frame_exists(s, hi + 1):
            hi += 1
        seq_max[s] = hi

    # place val windows randomly (non-overlapping incl. buffers)
    val_windows = []
    tries = 0
    while len(val_windows) < n_windows and tries < 10000:
        tries += 1
        s = rng.choice(ALL_SEQS)
        start = rng.randint(1, max(seq_max[s] - win, 1))
        clash = any(s == s2 and abs(start - st2) < win + 2 * buffer
                    for s2, st2 in val_windows)
        if not clash:
            val_windows.append((s, start))
    val_keys = [(s, st + i) for s, st in val_windows for i in range(win)]
    excluded = {(s, t) for s, st in val_windows
                for t in range(st - buffer, st + win + buffer)}

    n_train = args.n_pairs - len(val_keys)
    train_keys, seen = [], set(val_keys) | excluded
    while len(train_keys) < n_train and len(seen) < sum(seq_max.values()):
        s = rng.choice(ALL_SEQS)
        t = rng.randint(1, seq_max[s])
        if (s, t) in seen or not frame_exists(s, t):
            seen.add((s, t))
            continue
        seen.add((s, t))
        train_keys.append((s, t))

    print(f"loading {len(train_keys)} train + {len(val_keys)} val pairs "
          f"({n_windows} val windows of {win}, buffer {buffer}) ...")
    train = [_load_pair(s, t) for s, t in train_keys]
    val = [_load_pair(s, t) for s, t in val_keys]
    if args.pairs_cache:
        Path(args.pairs_cache).parent.mkdir(parents=True, exist_ok=True)
        torch.save(dict(train=train, val=val, seed=args.seed,
                        n_pairs=args.n_pairs, n_val=len(val_keys),
                        split_protocol=f"windowed-val w{win} buf{buffer}",
                        val_windows=val_windows),
                   args.pairs_cache)
        print(f"pairs cache written: {args.pairs_cache}")
    return train, val


def batchify(pairs, idxs, device):
    L = torch.stack([pairs[i]["L"] for i in idxs]).to(device).float() / 255.0
    R = torch.stack([pairs[i]["R"] for i in idxs]).to(device).float() / 255.0
    D = torch.stack([pairs[i]["D"] for i in idxs]).to(device).float()
    V = ((D > 0) & (D < MAX_DISP)).float()
    return L, R, D, V


# ---------------------------------------------------------------------------
# OpenStereo-style train-time augmentation (docs/openstereo_findings.md #1).
# Ported from external_models/OpenStereo stereo_trans.py semantics:
#   - StereoColorJitter: b/c/s in [0.6,1.4], hue +-0.5; ASYMMETRIC_PROB 0.2
#   - RandomErase: p=0.5, 1-2 rects 50-100 px, RIGHT image only, mean fill
#   - RandomScale: 2^U(-0.2,0.4) p=0.8 + anisotropic stretch 2^U(-.2,.2)
#     on x, disparity multiplied by scale_x; recrop to TRAIN_HxTRAIN_W
# Applied on [0,1] float tensors, train batches only.
# ---------------------------------------------------------------------------

def _color_jitter(img, rng):
    b = rng.uniform(0.6, 1.4)
    c = rng.uniform(0.6, 1.4)
    s = rng.uniform(0.6, 1.4)
    mean = img.mean(dim=(-1, -2), keepdim=True)
    grey = img.mean(dim=-3, keepdim=True)
    out = ((img * b - mean * b) * c + mean * b)          # brightness+contrast
    out = out * s + grey * (1 - s)                        # saturation
    return out.clamp(0, 1)


def augment_batch(L, R, D, V, rng):
    B, _, H, W = L.shape
    for b in range(B):
        # -- color jitter (asymmetric 20% of the time) --
        if rng.random() < 0.8:
            if rng.random() < 0.2:
                L[b] = _color_jitter(L[b], rng)
                R[b] = _color_jitter(R[b], rng)
            else:
                jb = rng.uniform(0.6, 1.4); jc = rng.uniform(0.6, 1.4)
                js = rng.uniform(0.6, 1.4)
                for img in (L, R):
                    mean = img[b].mean(dim=(-1, -2), keepdim=True)
                    grey = img[b].mean(dim=-3, keepdim=True)
                    o = ((img[b] * jb - mean * jb) * jc + mean * jb)
                    img[b] = (o * js + grey * (1 - js)).clamp(0, 1)
        # -- right-image eraser --
        if rng.random() < 0.5:
            mean_c = R[b].mean(dim=(-1, -2), keepdim=True)
            for _ in range(rng.randint(1, 2)):
                eh = rng.randint(50, 100); ew = rng.randint(50, 100)
                y0 = rng.randint(0, max(H - eh, 1))
                x0 = rng.randint(0, max(W - ew, 1))
                R[b, :, y0:y0+eh, x0:x0+ew] = mean_c
        # -- random scale + anisotropic stretch, disparity *= scale_x --
        if rng.random() < 0.8:
            sc = 2.0 ** rng.uniform(-0.2, 0.4)
            stx = 2.0 ** rng.uniform(-0.2, 0.2)
            sx_f, sy_f = sc * stx, sc
            nH, nW = max(int(H * sy_f), H // 2), max(int(W * sx_f), W // 2)
            def rs(x, mode):
                return F.interpolate(x[b:b+1], size=(nH, nW), mode=mode,
                                     align_corners=False if mode == "bilinear" else None)
            Lz = rs(L, "bilinear"); Rz = rs(R, "bilinear")
            Dz = rs(D, "nearest") * (nW / W)
            Vz = rs(V, "nearest")
            if nH >= H and nW >= W:
                y0 = rng.randint(0, nH - H); x0 = rng.randint(0, nW - W)
                L[b] = Lz[0, :, y0:y0+H, x0:x0+W]
                R[b] = Rz[0, :, y0:y0+H, x0:x0+W]
                D[b] = Dz[0, :, y0:y0+H, x0:x0+W]
                V[b] = Vz[0, :, y0:y0+H, x0:x0+W]
            else:  # downscale: pad back to full size, pad region invalid
                pl = torch.zeros_like(L[b]); pr = torch.zeros_like(R[b])
                pd = torch.zeros_like(D[b]); pv = torch.zeros_like(V[b])
                hh, ww = min(nH, H), min(nW, W)
                pl[:, :hh, :ww] = Lz[0, :, :hh, :ww]
                pr[:, :hh, :ww] = Rz[0, :, :hh, :ww]
                pd[:, :hh, :ww] = Dz[0, :, :hh, :ww]
                pv[:, :hh, :ww] = Vz[0, :, :hh, :ww]
                L[b], R[b], D[b], V[b] = pl, pr, pd, pv
    V = (V * ((D > 0) & (D < MAX_DISP)).float())
    return L, R, D, V


def freeze_bn(model):
    n = 0
    for m in model.modules():
        if isinstance(m, torch.nn.modules.batchnorm._BatchNorm):
            m.eval()
            for p in m.parameters():
                p.requires_grad = False
            n += 1
    return n


# ---------------------------------------------------------------------------
# Loss (identical across arms)
# ---------------------------------------------------------------------------

def ms_l1(pred, gt, valid, scale):
    if scale != 1.0:
        gt = F.interpolate(gt, scale_factor=1.0 / scale, mode="nearest") / scale
        valid = F.interpolate(valid, scale_factor=1.0 / scale, mode="nearest")
        if pred.shape[-2:] != gt.shape[-2:]:
            pred = F.interpolate(pred, size=gt.shape[-2:], mode="bilinear",
                                 align_corners=True)
    return ((pred - gt).abs() * valid).sum() / valid.sum().clamp(min=1)


def grad_consistency(pred, gt, valid):
    def dx(x):
        return x[..., :, 1:] - x[..., :, :-1]

    def dy(x):
        return x[..., 1:, :] - x[..., :-1, :]

    vx = valid[..., :, 1:] * valid[..., :, :-1]
    vy = valid[..., 1:, :] * valid[..., :-1, :]
    lx = ((dx(pred) - dx(gt)).abs() * vx).sum() / vx.sum().clamp(min=1)
    ly = ((dy(pred) - dy(gt)).abs() * vy).sum() / vy.sum().clamp(min=1)
    return lx + ly


def bad1_hinge(pred, gt, valid):
    return (F.relu((pred - gt).abs() - 1.0).clamp(max=2.0) * valid).sum() / \
        valid.sum().clamp(min=1)


def ms_l1_trunc(pred, gt, valid, scale, A=1.0):
    """HITNet-style truncated L1 (Eq. 12): per-scale error capped at A px
    (in that scale's units) — a boundary tile committed to one surface is
    not pulled toward the mean by the other surface's gradient."""
    gt = F.interpolate(gt, scale_factor=1.0 / scale, mode="nearest") / scale
    valid = F.interpolate(valid, scale_factor=1.0 / scale, mode="nearest")
    if pred.shape[-2:] != gt.shape[-2:]:
        pred = F.interpolate(pred, size=gt.shape[-2:], mode="bilinear",
                             align_corners=True)
    err = (pred - gt).abs().clamp(max=A)
    return (err * valid).sum() / valid.sum().clamp(min=1)


def init_ce_loss(logits, gt, valid, scale=16.0, max_disp=24):
    """Distribution-shaping CE on the TileInit volume: subpixel-aware
    two-bin target at the GT disparity (deviation from HITNet's
    contrastive-on-raw-costs Eq. 10 — we shape aggregated logits with CE,
    same peakedness intent, documented in docs/deblurring_plan.md)."""
    gt_s = F.interpolate(gt, scale_factor=1.0 / scale, mode="nearest") / scale
    v = F.interpolate(valid, scale_factor=1.0 / scale, mode="nearest")
    v = (v > 0.5) & (gt_s[:, 0:1] < max_disp - 1)
    gt_c = gt_s[:, 0].clamp(0, max_disp - 1 - 1e-4)
    lo = gt_c.floor().long()
    w_hi = gt_c - lo.float()
    logp = F.log_softmax(logits, dim=1)
    nll = -(logp.gather(1, lo.unsqueeze(1)).squeeze(1) * (1 - w_hi)
            + logp.gather(1, (lo + 1).clamp(max=max_disp - 1)
                          .unsqueeze(1)).squeeze(1) * w_hi)
    m = v[:, 0]
    return (nll * m).sum() / m.sum().clamp(min=1)


def loss_fn(out, D, V, trunc_A: float = 0.0, init_ce_w: float = 0.0):
    if trunc_A > 0:
        coarse = (0.3 * ms_l1_trunc(out["d4"], D, V, 4.0, trunc_A)
                  + 0.2 * ms_l1_trunc(out["d8"], D, V, 8.0, trunc_A)
                  + 0.1 * ms_l1_trunc(out["d16"], D, V, 16.0, trunc_A))
    else:
        coarse = (0.3 * ms_l1(out["d4"], D, V, 4.0)
                  + 0.2 * ms_l1(out["d8"], D, V, 8.0)
                  + 0.1 * ms_l1(out["d16"], D, V, 16.0))
    loss = (1.0 * ms_l1(out["d_final"], D, V, 1.0)
            + 0.5 * ms_l1(out["d_half"], D, V, 2.0)
            + coarse
            + 0.5 * grad_consistency(out["d_final"], D, V)
            + 0.2 * bad1_hinge(out["d_final"], D, V))
    if "d4_gev" in out:
        loss = loss + 0.15 * ms_l1(out["d4_gev"], D, V, 4.0)
    if init_ce_w > 0 and out.get("init_logits") is not None:
        loss = loss + init_ce_w * init_ce_loss(out["init_logits"], D, V)
    return loss


# ---------------------------------------------------------------------------
# Eval + user-spec collage
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate(model, pairs, device, bs=4):
    model.eval()
    agg = []
    for i in range(0, len(pairs), bs):
        idxs = list(range(i, min(i + bs, len(pairs))))
        L, R, D, V = batchify(pairs, idxs, device)
        pred = model(L, R)
        for b in range(pred.shape[0]):
            agg.append(stereo_metrics(pred[b:b+1], D[b:b+1], V[b:b+1]))
    model.train()
    return {k: float(np.mean([a[k] for a in agg])) for k in agg[0]}


def _annot(img, lines, color=(255, 255, 0)):
    for i, ln in enumerate(lines):
        cv2.putText(img, ln, (8, 22 + 20 * i), cv2.FONT_HERSHEY_SIMPLEX,
                    0.52, color, 1, cv2.LINE_AA)
    return img


@torch.no_grad()
def make_collage(model, train_pairs, val_pairs, device, step, args):
    """User spec: 6 tiles = 1 GT + 5 preds (3 train, 2 val), each pred
    annotated with its full per-image metric set. Layout 2x3:
        [GT(train#0) | pred train#0 | pred train#1]
        [pred train#2 | pred VAL#0  | pred VAL#1 ]
    Shared TURBO colormap; vmax from GT(train#0) p99 for the top-left
    reference pair, per-image GT p99 for the rest."""
    model.eval()
    picks = [("train", train_pairs[0]), ("train", train_pairs[1]),
             ("train", train_pairs[2]), ("val", val_pairs[0]),
             ("val", val_pairs[1])]
    tiles = []
    gt0 = train_pairs[0]["D"][0].float().numpy()
    vmax0 = max(np.percentile(gt0[gt0 > 0], 99), 1.0)
    gt_tile = _annot(_colorize_disp(gt0, vmax=vmax0),
                     [f"GT (train #0) {train_pairs[0]['seq'].split('/')[0][:4]}#{train_pairs[0]['t']}",
                      f"step {step}"], color=(255, 255, 255))
    tiles.append(gt_tile)
    for split, p in picks:
        L, R, D, V = batchify([p], [0], device)
        pred = model(L, R)
        m = stereo_metrics(pred, D, V)
        dnp = pred[0, 0].float().cpu().numpy()
        g = D[0, 0].cpu().numpy()
        vmax = max(np.percentile(g[g > 0], 99), 1.0)
        tile = _colorize_disp(dnp, vmax=vmax)
        col = (0, 255, 0) if split == "train" else (0, 200, 255)
        tile = _annot(tile, [
            f"{split.upper()} {p['seq'].split('/')[0][:4]}#{p['t']}",
            f"EPE {m['epe']:.3f}  RMSE {m['rmse']:.2f}  med {m['median_ae']:.3f}",
            f"bad0.5 {m['bad_0.5']:.1f}  bad1 {m['bad_1.0']:.1f}",
            f"bad2 {m['bad_2.0']:.1f}  bad3 {m['bad_3.0']:.1f}  D1 {m['d1_all']:.1f}",
        ], color=col)
        tiles.append(tile)
    top = np.concatenate(tiles[:3], axis=1)
    bot = np.concatenate(tiles[3:], axis=1)
    model.train()
    return np.concatenate([top, bot], axis=0)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arch", required=True,
                    choices=["gev4", "gev4_opt", "gev4_opt_narrow",
                             "gev4_opt_narrow_sharptail",
                             "gev4_opt_narrow_bundle1",
                             "costlookup_y26n", "costlookup_y26s"])
    ap.add_argument("--trunc_A", type=float, default=0.0,
                    help="HITNet truncated L1 cap (px, per-scale units) on "
                         "the d4/d8/d16 terms; 0 = off")
    ap.add_argument("--init_ce_w", type=float, default=0.0,
                    help="weight of the TileInit distribution-shaping CE "
                         "loss; 0 = off")
    ap.add_argument("--n_pairs", type=int, default=100)
    ap.add_argument("--n_val", type=int, default=20)
    ap.add_argument("--max_steps", type=int, default=12000)
    ap.add_argument("--min_steps", type=int, default=4000)
    ap.add_argument("--patience", type=int, default=4,
                    help="evals without >1%% val-EPE improvement before stop")
    ap.add_argument("--eval_every", type=int, default=500)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--show", type=int, default=1)
    ap.add_argument("--out_root", default="model/benchmarks")
    ap.add_argument("--run_name", default=None)
    ap.add_argument("--pairs_cache", default=None)
    ap.add_argument("--dump_pairs_only", type=int, default=0)
    ap.add_argument("--aug", type=int, default=0,
                    help="OpenStereo triplet on train batches (asym jitter, "
                         "right eraser, scale/stretch w/ disparity rescale)")
    ap.add_argument("--freeze_bn", type=int, default=0,
                    help="freeze all encoder BatchNorm (OpenStereo FREEZE_BN)")
    args = ap.parse_args()

    torch.manual_seed(args.seed); np.random.seed(args.seed)
    random.seed(args.seed)
    device = "cuda"

    train_pairs, val_pairs = load_or_build_pairs(args)
    if args.dump_pairs_only:
        print("pairs cache dumped; exiting."); return
    print(f"[{args.arch}] {len(train_pairs)} train / {len(val_pairs)} val")

    run = args.run_name or f"efficiency_{datetime.now():%Y%m%d-%H%M%S}"
    out_dir = Path(args.out_root) / run / args.arch
    (out_dir / "viz").mkdir(parents=True, exist_ok=True)

    model, cfg = build_model(args.arch)
    model = model.to(device)
    if args.freeze_bn:
        nbn = freeze_bn(model)
        print(f"freeze_bn: {nbn} BatchNorm modules frozen")
        # evaluate()/make_collage() call model.train() on exit, which would
        # flip BN back to train mode — make the freeze sticky.
        _orig_train = model.train
        def _train_keep_bn(mode: bool = True):
            _orig_train(mode)
            if mode:
                freeze_bn(model)
            return model
        model.train = _train_keep_bn
    n_train_p = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"params: {n_train_p/1e6:.4f} M")
    aug_rng = random.Random(args.seed + 7)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)
    scaler = torch.amp.GradScaler("cuda")

    started_at = datetime.now().isoformat(timespec="seconds")
    csv_rows = [("step", "loss", "epe", "rmse", "bad_0.5", "bad_1.0",
                 "bad_2.0", "bad_3.0", "d1_all", "val_epe", "val_bad1",
                 "val_d1", "lr", "elapsed_s")]
    best_val = float("inf"); best_step = 0; evals_since_best = 0
    val_hist = []
    t0 = time.time()
    step = 0
    nan_streak = 0
    while step < args.max_steps:
        step += 1
        idxs = [random.randrange(len(train_pairs)) for _ in range(args.batch)]
        L, R, D, V = batchify(train_pairs, idxs, device)
        if args.aug:
            L, R, D, V = augment_batch(L, R, D, V, aug_rng)
        opt.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda"):
            out = model(L, R, aux=True)
            loss = loss_fn(out, D, V, trunc_A=args.trunc_A,
                           init_ce_w=args.init_ce_w)
        if not torch.isfinite(loss):
            nan_streak += 1
            if nan_streak >= 50:
                raise RuntimeError(f"non-finite loss x{nan_streak} @ step {step}")
            continue
        nan_streak = 0
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(opt); scaler.update()

        if step % 100 == 0:
            with torch.no_grad():
                bm = stereo_metrics(out["d_final"].detach().float(), D, V)
            print(f"step {step:5d}  loss {float(loss.detach()):.4f}  "
                  f"epe {bm['epe']:.3f}  bad1 {bm['bad_1.0']:.1f}%  "
                  f"{(time.time()-t0)/step*1000:.0f} ms/step", flush=True)
            csv_rows.append((step, f"{float(loss.detach()):.5f}",
                             f"{bm['epe']:.4f}", f"{bm['rmse']:.4f}",
                             f"{bm['bad_0.5']:.3f}", f"{bm['bad_1.0']:.3f}",
                             f"{bm['bad_2.0']:.3f}", f"{bm['bad_3.0']:.3f}",
                             f"{bm['d1_all']:.3f}", "", "", "",
                             f"{args.lr:g}", f"{time.time()-t0:.1f}"))

        if step % args.eval_every == 0:
            vm = evaluate(model, val_pairs, device)
            val_hist.append((step, vm["epe"]))
            csv_rows.append((step, "", "", "", "", "", "", "", "",
                             f"{vm['epe']:.4f}", f"{vm['bad_1.0']:.3f}",
                             f"{vm['d1_all']:.3f}", f"{args.lr:g}",
                             f"{time.time()-t0:.1f}"))
            print(f"  val@{step}: epe={vm['epe']:.4f} bad1={vm['bad_1.0']:.2f} "
                  f"D1={vm['d1_all']:.2f}", flush=True)
            panel = make_collage(model, train_pairs, val_pairs, device,
                                 step, args)
            cv2.imwrite(str(out_dir / "viz" / f"collage_step_{step:06d}.png"),
                        panel)
            if args.show:
                cv2.imshow(args.arch, panel); cv2.waitKey(1)

            if vm["epe"] < best_val * 0.99:      # >1% relative improvement
                best_val = vm["epe"]; best_step = step; evals_since_best = 0
                torch.save(model.state_dict(), out_dir / "best.pth")
            else:
                evals_since_best += 1
            if step >= args.min_steps and evals_since_best >= args.patience:
                print(f"PLATEAU at step {step}: best val EPE {best_val:.4f} "
                      f"@ {best_step}; {evals_since_best} evals w/o >1% gain",
                      flush=True)
                break

    # final metrics on both splits + latency
    final_train = evaluate(model, train_pairs, device)
    final_val = evaluate(model, val_pairs, device)
    Lb, Rb, _, _ = batchify(val_pairs, [0], device)
    model.eval()
    with torch.no_grad():
        for _ in range(10): model(Lb, Rb)
        torch.cuda.synchronize(); ts = []
        for _ in range(50):
            s0 = time.perf_counter(); model(Lb, Rb); torch.cuda.synchronize()
            ts.append((time.perf_counter() - s0) * 1000)
    lat = {"mean": float(np.mean(ts)), "median": float(np.median(ts)),
           "p95": float(np.percentile(ts, 95))}
    print(f"FINAL[val]   [{args.arch}] " +
          "  ".join(f"{k}={v:.3f}" for k, v in final_val.items()))
    print(f"FINAL[train] [{args.arch}] " +
          "  ".join(f"{k}={v:.3f}" for k, v in final_train.items()))
    print(f"latency: {lat['median']:.1f} ms median  |  stopped @ {step}")

    meta = dict(
        run=run, arch=args.arch, variant=args.arch,
        harness="overfit_efficiency_ablation",
        steps=step, max_steps=args.max_steps, lr=args.lr, batch=args.batch,
        height=TRAIN_H, width=TRAIN_W, n_pairs=args.n_pairs,
        n_train=len(train_pairs), n_val=len(val_pairs), seed=args.seed,
        plateau=dict(min_steps=args.min_steps, patience=args.patience,
                     best_val_epe=best_val, best_step=best_step),
        device="cuda", gpu=torch.cuda.get_device_name(0),
        pytorch=torch.__version__, platform=_platform.platform(),
        started_at=started_at,
        finished_at=datetime.now().isoformat(timespec="seconds"),
        params_total_M=round(sum(p.numel() for p in model.parameters())/1e6, 4),
        params_train_M=round(n_train_p / 1e6, 4),
        encoder_out_channels=list(model.fnet.out_channels),
        arch_config=dataclasses.asdict(cfg),
        loss_formulation="msL1{1,.5,.3,.2,.1}+0.5grad+0.2bad1+0.15gev4",
        input_scale="[0,1]",
        pair_paths=[dict(split="train", seq=p["seq"], t=p["t"])
                    for p in train_pairs] +
                   [dict(split="val", seq=p["seq"], t=p["t"])
                    for p in val_pairs],
        peak_gpu_mem_GB=round(torch.cuda.max_memory_allocated()/1e9, 3),
        final_metrics_all=final_val,
        final_metrics_train=final_train,
        latency_ms=lat,
        args=vars(args),
    )
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2))
    torch.save(model.state_dict(), out_dir / "checkpoint.pth")

    with open(out_dir / "train.csv", "w") as fh:
        fh.write("\n".join(",".join(str(c) for c in r) for r in csv_rows))
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        rows = [r for r in csv_rows[1:] if r[1] != ""]
        xs = [r[0] for r in rows]; ls = [float(r[1]) for r in rows]
        es = [float(r[2]) for r in rows]
        vx = [s for s, _ in val_hist]; vy = [e for _, e in val_hist]
        fig, ax = plt.subplots(figsize=(8, 4.5))
        ax.plot(xs, ls, lw=1, color="tab:blue", label="loss")
        ax.set_xlabel("step"); ax.set_ylabel("loss", color="tab:blue")
        ax2 = ax.twinx()
        ax2.plot(xs, es, lw=1, color="tab:orange", label="train EPE")
        ax2.plot(vx, vy, "o-", lw=1.4, ms=3, color="tab:red", label="val EPE")
        ax2.set_ylabel("EPE (px)")
        ax2.legend(loc="upper right")
        ax.set_title(f"{args.arch}  best val EPE {best_val:.3f} @ {best_step}")
        ax.grid(alpha=0.3)
        fig.tight_layout(); fig.savefig(out_dir / "curve.png", dpi=110)
        plt.close(fig)
    except Exception as e:
        print(f"curve.png skipped: {e}")

    lat_s = f"mean {lat['mean']:.1f} / median {lat['median']:.1f} / p95 {lat['p95']:.1f} ms"
    rows_md = "\n".join(f"| {k} | {final_val[k]:.4g} | {final_train[k]:.4g} |"
                        for k in final_val)
    (out_dir / "README.md").write_text(
        f"# Efficiency ablation arm: {args.arch}\n\n"
        f"**Run:** {started_at} -> {meta['finished_at']}  |  **GPU:** {meta['gpu']}\n\n"
        f"- stopped at step {step}/{args.max_steps} "
        f"(plateau: best val EPE {best_val:.4f} @ step {best_step})\n"
        f"- {len(train_pairs)} train / {len(val_pairs)} val pairs, seed {args.seed}, "
        f"batch {args.batch}, lr {args.lr}, {TRAIN_H}x{TRAIN_W}, input [0,1]\n"
        f"- params {meta['params_train_M']} M, peak {meta['peak_gpu_mem_GB']} GB\n\n"
        f"| Metric | Val (20) | Train (80) |\n|---|---|---|\n{rows_md}\n\n"
        f"Latency ({meta['gpu']}): {lat_s}\n")

    subprocess.run([sys.executable,
                    str(_SCRIPTS / "build_experiments_summary.py")], check=False)


if __name__ == "__main__":
    main()
