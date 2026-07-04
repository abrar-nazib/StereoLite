"""Full-SceneFlow trainer for the locked thesis config (gev4_opt_narrow_plane).

Protocol:
  - data: shard files produced by modal/repack_sceneflow_shards.py
    (canonical split sceneflow_split_v1: train = FT3D-TRAIN + Monkaa +
    Driving 35,454 pairs; test = FT3D-TEST 4,370 pairs, finalpass)
  - input protocol (--input_mode):
      resize       384x640 global downscale (legacy; the 20260703 run)
      native_crop  random 384x640 co-located crops of native 960x540
                   frames, native disparity magnitudes. Validated by the
                   20260704_native_vs_resize_n500 ablation (crop beats
                   resize on the native axis by -32% bad-0.5 / -37% bad-1
                   with zero cost on the resized axis). Crop protocol
                   matches OpenStereo RandomCrop (same x,y window in both
                   views, external_models/OpenStereo stereo_trans.py:59);
                   Y_JITTER stays off, as in every OpenStereo SceneFlow
                   config (synthetic data is perfectly rectified) and as
                   in the validated ablation arm. No published method
                   shifts the right crop horizontally (that would offset
                   GT disparity); do not add one.
  - loss/aug/model: imported from overfit_efficiency_ablation.py so the
    training path is EXACTLY the ablation-validated one (aug order
    crop -> colorjitter/erase/scale, identical to the native_crop arm)
  - schedule: OneCycle over --steps; bf16 autocast (A100) or fp16+scaler
  - validation: fixed seed-42 400-pair subset of FT3D-TEST every
    --eval_every steps. In native_crop mode the val axis is NATIVE
    960x540 pad16 (OpenStereo evals SceneFlow the same way: RightTopPad
    544x960) and a 640x384 resized-axis eval is logged alongside for
    continuity with the legacy run. FULL 4,370-pair test on the best
    checkpoint at the end (the reported number).
  - checkpoints: latest.pth (resume-able, atomic) every eval, best.pth on
    val-EPE improvement, PLUS checkpoints/step_XXXXXX.pth (model-only)
    every --ckpt_every steps so no training state is ever lost again.
  - tracked images (--track_train/--track_val, native mode): fixed 50
    train + 50 val pairs; per eval, each pair's native-axis prediction is
    written to images/<split>_<idx>/step_XXXXXX.png (uint16, px*256,
    KITTI encoding) with a step_XXXXXX.json carrying the full 8-metric
    set for that image. GT + left view written once at startup. No
    collages; they are built locally later from these files.
  - probe mode: --probe "8,16,24,32,40,48" measures ms/step + peak VRAM per
    batch size through the real train step, then exits (GPU-efficiency rule:
    pick the largest batch <= 85% VRAM)

Example (Modal driver: modal/train_full_sceneflow_a100.py):
    python model/scripts/train_full_sceneflow.py \
        --shards_dir /shards/v1 --out_dir /results/fulltrain/RUN \
        --arch gev4_opt_narrow_plane --slant_w 0.3 --aug 1 \
        --input_mode native_crop --steps 60000 --batch 32 \
        --eval_every 1000 --lr auto --amp bf16
"""
from __future__ import annotations

import argparse
import json
import pickle
import random
import sys
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader, IterableDataset

_SCRIPTS = Path(__file__).resolve().parent
sys.path.insert(0, str(_SCRIPTS))
sys.path.insert(0, str(_SCRIPTS.parent / "designs"))

from overfit_efficiency_ablation import (  # noqa: E402
    MAX_DISP, TRAIN_H, TRAIN_W, _forward_pad16, augment_batch, batchify,
    build_model, downscale_pair, evaluate, loss_fn, make_collage)
from overfit_yolo_ablation import stereo_metrics  # noqa: E402

NATIVE_W = 960


def decode_record(rec: dict, native: bool = False) -> dict:
    import zstandard
    ims = []
    for k in ("left_png", "right_png"):
        im = cv2.imdecode(np.frombuffer(rec[k], np.uint8), cv2.IMREAD_COLOR)
        if not native:
            im = cv2.resize(im, (TRAIN_W, TRAIN_H),
                            interpolation=cv2.INTER_AREA)
        ims.append(torch.from_numpy(im[..., ::-1].copy()).permute(2, 0, 1)
                   .to(torch.uint8))
    h, w = rec["shape"]
    d = np.abs(np.frombuffer(zstandard.decompress(rec["disp_z"]),
                             "<f4").reshape(h, w))
    if not native:
        d = cv2.resize(d, (TRAIN_W, TRAIN_H), interpolation=cv2.INTER_NEAREST)
        d = d * (TRAIN_W / NATIVE_W)
    d = np.nan_to_num(d, nan=0.0, posinf=0.0)
    parts = rec["key"].split("/")
    return dict(seq="/".join(parts[1:-2]), t=int(parts[-1][:-4]),
                L=ims[0], R=ims[1],
                D=torch.from_numpy(d.astype(np.float32))[None].to(torch.float16))


class ShardStream(IterableDataset):
    """Infinite stream over train shards: per-epoch shard-order + intra-shard
    shuffle, shards split across DataLoader workers. One shard resident per
    worker at a time (~1.8 GB).

    input_mode=native_crop: decode at native resolution and yield a random
    co-located TRAIN_HxTRAIN_W crop of L/R/D (same window in both views;
    stereo geometry is preserved under identical translation). Byte-level
    protocol match to batchify_native_crop in the validated ablation arm.

    yjitter: RAFT/IGEV-style rectification-error simulation — the RIGHT
    crop window is shifted vertically by a random integer in [-2, +2] px
    (IGEV core/utils/augmentor.py:154-161; default ON in every RAFT-family
    SceneFlow recipe, OFF in OpenStereo/LightStereo configs). Left and
    disparity keep the unjittered window; x is never shifted.
    """

    def __init__(self, shard_paths: list[str], seed: int,
                 input_mode: str = "resize", yjitter: int = 0):
        self.paths = sorted(shard_paths)
        self.seed = seed
        self.input_mode = input_mode
        self.yjitter = yjitter

    def __iter__(self):
        info = torch.utils.data.get_worker_info()
        wid = info.id if info else 0
        nw = info.num_workers if info else 1
        crng = random.Random(self.seed * 104729 + wid)  # crop rng, per worker
        epoch = 0
        while True:
            order = list(self.paths)
            random.Random(self.seed + epoch).shuffle(order)
            for sp in order[wid::nw]:
                with open(sp, "rb") as f:
                    recs = pickle.load(f)
                random.Random(self.seed + epoch * 7919 + wid).shuffle(recs)
                for rec in recs:
                    if self.input_mode == "native_crop":
                        s = decode_record(rec, native=True)
                        H, W = s["L"].shape[-2:]
                        j = self.yjitter
                        y0 = crng.randint(j, H - TRAIN_H - j)
                        x0 = crng.randint(0, W - TRAIN_W)
                        yR = y0 + (crng.randint(-j, j) if j else 0)
                        s["L"] = s["L"][..., y0:y0 + TRAIN_H,
                                        x0:x0 + TRAIN_W].clone()
                        s["R"] = s["R"][..., yR:yR + TRAIN_H,
                                        x0:x0 + TRAIN_W].clone()
                        s["D"] = s["D"][..., y0:y0 + TRAIN_H,
                                        x0:x0 + TRAIN_W].clone()
                        yield s
                    else:
                        yield decode_record(rec)
            epoch += 1


def _safe_sched_step(sched):
    """OneCycle raises once stepped past total_steps (possible when NaN-skip
    steps consumed schedule ticks); ride the floor LR instead of dying."""
    try:
        sched.step()
    except ValueError:
        pass


def _atomic_save(obj, path: Path):
    """torch.save via tmp+rename so the background volume committer can never
    snapshot a half-written checkpoint."""
    import os
    tmp = path.with_suffix(".tmp")
    torch.save(obj, tmp)
    os.replace(tmp, path)


def _make_tail_sched(opt, lr: float, leg: int):
    """Fresh cosine tail lr -> 0. initial_lr must be set explicitly: the
    scheduler reads base_lrs from it, and setdefault keeps OneCycle's stale
    warmup floor otherwise."""
    for gp in opt.param_groups:
        gp["lr"] = lr
        gp["initial_lr"] = lr
    return torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=leg,
                                                      eta_min=0.0)


def to_device(batch, device):
    L = batch["L"].to(device, non_blocking=True).float() / 255.0
    R = batch["R"].to(device, non_blocking=True).float() / 255.0
    D = batch["D"].to(device, non_blocking=True).float()
    V = ((D > 0) & (D < MAX_DISP)).float()
    return L, R, D, V


def load_test_pairs(shards_dir: Path, keys: set[str] | None,
                    native: bool = False) -> list[dict]:
    """Load FT3D-TEST pairs (all, or only `keys`) into harness pair dicts."""
    out = []
    for sp in sorted(shards_dir.glob("test_ft3d_*.pkl")):
        with open(sp, "rb") as f:
            recs = pickle.load(f)
        for rec in recs:
            if keys is None or rec["key"] in keys:
                out.append(decode_record(rec, native=native))
    return out


def load_tracked_train(shards_dir: Path, n: int, seed: int) -> list[dict]:
    """Fixed native-resolution train pairs for per-eval image tracking:
    one shard per source (ft3d/monkaa/driving), seeded sample, ~n/3 each.
    Deterministic across restarts (sorted shard list + fixed seed)."""
    rng = random.Random(seed)
    srcs = ["ft3d", "monkaa", "driving"]
    counts = [n // 3 + (1 if i < n % 3 else 0) for i in range(3)]
    picks = []
    for src, k in zip(srcs, counts):
        sps = sorted(shards_dir.glob(f"train_{src}_*.pkl"))
        if not sps or k == 0:
            continue
        with open(sps[0], "rb") as f:
            recs = pickle.load(f)
        recs = sorted(recs, key=lambda r: r["key"])
        for rec in rng.sample(recs, min(k, len(recs))):
            picks.append(decode_record(rec, native=True))
    return picks[:n]


def _disp_to_u16(d: np.ndarray) -> np.ndarray:
    """KITTI-style uint16 encoding: px * 256 (decode: png / 256.0)."""
    return np.clip(d * 256.0, 0, 65535).astype(np.uint16)


def init_tracked_dirs(images_dir: Path, tracked: list[dict], prefix: str):
    """Per tracked pair: images/<prefix>_<idx>/ with left.png (8-bit BGR),
    gt.png (uint16 px*256) and info.json, written once."""
    for i, p in enumerate(tracked):
        d = images_dir / f"{prefix}_{i:02d}"
        d.mkdir(parents=True, exist_ok=True)
        if (d / "gt.png").exists():
            continue
        left = p["L"].permute(1, 2, 0).numpy()[..., ::-1]  # RGB -> BGR
        cv2.imwrite(str(d / "left.png"), np.ascontiguousarray(left))
        cv2.imwrite(str(d / "gt.png"), _disp_to_u16(p["D"][0].float().numpy()))
        (d / "info.json").write_text(json.dumps(
            {"seq": p["seq"], "t": p["t"], "split": prefix,
             "native_hw": list(p["L"].shape[-2:]),
             "disp_encoding": "uint16 = disparity_px * 256",
             "valid_protocol": f"gt > 0 and gt < {MAX_DISP}"}, indent=1))


@torch.no_grad()
def save_tracked(model, tracked: list[dict], prefix: str, images_dir: Path,
                 step: int, device, bs: int = 8):
    """Per eval: native-axis prediction PNG (uint16 px*256) + full per-image
    metric JSON for every tracked pair."""
    model.eval()
    for i0 in range(0, len(tracked), bs):
        idxs = list(range(i0, min(i0 + bs, len(tracked))))
        L, R, D, V = batchify(tracked, idxs, device)
        pred = _forward_pad16(model, L, R)
        for j, i in enumerate(idxs):
            m = stereo_metrics(pred[j:j + 1].float(), D[j:j + 1],
                               V[j:j + 1])
            d = images_dir / f"{prefix}_{i:02d}"
            cv2.imwrite(str(d / f"step_{step:06d}.png"),
                        _disp_to_u16(pred[j, 0].float().cpu().numpy()))
            (d / f"step_{step:06d}.json").write_text(json.dumps(
                {"step": step,
                 **{k: round(float(v), 5) for k, v in m.items()}}, indent=1))
    model.train()


def run_probe(model, loader, device, args):
    """Measure ms/step + peak VRAM through the real train step per batch."""
    total = torch.cuda.get_device_properties(0).total_memory
    print(f"GPU: {torch.cuda.get_device_name(0)}, "
          f"{total/1e9:.1f} GB; target peak <= 85% = {0.85*total/1e9:.1f} GB")
    it = iter(loader)
    pool = [next(it) for _ in range(64)]  # reusable decoded samples
    rows = []
    for B in [int(b) for b in args.probe.split(",")]:
        opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        try:
            times = []
            for i in range(20):
                sel = random.sample(pool, min(B, len(pool)))
                batch = {k: torch.stack([s[k] for s in sel]) for k in
                         ("L", "R", "D")}
                while batch["L"].shape[0] < B:  # tile if B > pool
                    batch = {k: torch.cat([batch[k], batch[k]])[:B]
                             for k in batch}
                L, R, D, V = to_device(batch, device)
                if args.aug:
                    L, R, D, V = augment_batch(L, R, D, V, random.Random(i))
                torch.cuda.synchronize(); t = time.time()
                opt.zero_grad(set_to_none=True)
                with torch.amp.autocast("cuda", dtype=torch.bfloat16
                                        if args.amp == "bf16" else torch.float16):
                    out = model(L, R, aux=True)
                    loss = loss_fn(out, D, V, slant_w=args.slant_w,
                                   trunc_A=args.trunc_A,
                                   init_ce_w=args.init_ce_w,
                                   bimodal_w=args.bimodal_w)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                torch.cuda.synchronize()
                if i >= 5:
                    times.append(time.time() - t)
            peak = torch.cuda.max_memory_allocated()
            ms = float(np.median(times)) * 1000
            rows.append((B, ms, B / ms * 1000, peak / 1e9,
                         100 * peak / total))
            print(f"batch {B:3d}: {ms:7.1f} ms/step  "
                  f"{B/ms*1000:6.1f} samples/s  peak {peak/1e9:5.1f} GB "
                  f"({100*peak/total:.0f}%)", flush=True)
        except torch.cuda.OutOfMemoryError:
            print(f"batch {B:3d}: OOM", flush=True)
            rows.append((B, None, None, None, None))
            break
        except RuntimeError as e:  # kernel/dispatch failures: record, continue
            print(f"batch {B:3d}: RuntimeError: {e}", flush=True)
            rows.append((B, None, None, None, None))
            torch.cuda.empty_cache()
    ok = [r for r in rows if r[1] is not None and r[4] <= 85.0]
    if ok:
        best = max(ok, key=lambda r: r[2])
        print(f"\nRECOMMENDED: batch {best[0]} "
              f"({best[2]:.1f} samples/s, {best[4]:.0f}% VRAM); "
              f"suggested peak lr = {min(8e-4, 2e-4*best[0]/8):.1e}")
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shards_dir", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--arch", default="gev4_opt_narrow_plane")
    ap.add_argument("--slant_w", type=float, default=0.3)
    ap.add_argument("--trunc_A", type=float, default=0.0)
    ap.add_argument("--init_ce_w", type=float, default=0.0)
    ap.add_argument("--bimodal_w", type=float, default=0.0)
    ap.add_argument("--aug", type=int, default=1)
    ap.add_argument("--input_mode", choices=["resize", "native_crop"],
                    default="resize",
                    help="resize: legacy 640x384 global downscale; "
                         "native_crop: random 384x640 crops of native "
                         "960x540 (ablation-validated winner)")
    ap.add_argument("--yjitter", type=int, default=0,
                    help="native_crop only: shift the RIGHT crop window "
                         "vertically by a random int in [-N, +N] px "
                         "(RAFT/IGEV rectification-error aug; 0 = off, "
                         "matching the validated ablation arm)")
    ap.add_argument("--steps", type=int, default=100000)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", default="auto",
                    help="peak OneCycle lr; auto = min(8e-4, 2e-4*batch/8)")
    ap.add_argument("--amp", choices=["bf16", "fp16"], default="bf16")
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--eval_every", type=int, default=2000)
    ap.add_argument("--ckpt_every", type=int, default=0,
                    help="save model-only checkpoints/step_XXXXXX.pth every "
                         "N steps (0 = off); rounds to eval boundaries")
    ap.add_argument("--track_train", type=int, default=0,
                    help="native mode: fixed train pairs tracked per eval "
                         "(images/train_XX/step_*.png + .json)")
    ap.add_argument("--track_val", type=int, default=0,
                    help="native mode: fixed val pairs tracked per eval")
    ap.add_argument("--val_manifest", default=None,
                    help="split manifest (for val_subset keys)")
    ap.add_argument("--val_bs", type=int, default=0,
                    help="eval batch size (0 = auto: 4 native / 8 resized)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--resume", type=int, default=1)
    ap.add_argument("--probe", default=None,
                    help='e.g. "8,16,24,32,40,48": measure + exit')
    ap.add_argument("--final_full_eval", type=int, default=1)
    ap.add_argument("--auto_extend", type=int, default=1,
                    help="after OneCycle ends, keep training in low-LR "
                         "cosine tails while val EPE still improves")
    ap.add_argument("--extend_leg", type=int, default=5000)
    ap.add_argument("--extend_cap", type=int, default=20000)
    ap.add_argument("--extend_lr", type=float, default=1e-4)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    device = "cuda"
    native = args.input_mode == "native_crop"
    shards_dir = Path(args.shards_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "viz").mkdir(exist_ok=True)
    if args.ckpt_every:
        (out_dir / "checkpoints").mkdir(exist_ok=True)

    train_shards = [str(p) for p in shards_dir.glob("train_*.pkl")]
    assert train_shards, f"no train shards in {shards_dir}"
    loader = DataLoader(ShardStream(train_shards, args.seed, args.input_mode,
                                    args.yjitter),
                        batch_size=args.batch, num_workers=args.workers,
                        pin_memory=True, prefetch_factor=4,
                        persistent_workers=args.workers > 0, drop_last=True)

    model, cfg = build_model(args.arch)
    model = model.to(device).train()
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    if args.probe:
        # probe uses a 1-worker loader just to fetch real samples
        probe_loader = DataLoader(ShardStream(train_shards, args.seed,
                                              args.input_mode, args.yjitter),
                                  batch_size=1, num_workers=2,
                                  collate_fn=lambda x: x[0])
        rows = run_probe(model, probe_loader, device, args)
        (out_dir / "probe.json").write_text(json.dumps(
            {"gpu": torch.cuda.get_device_name(0), "amp": args.amp,
             "aug": args.aug, "arch": args.arch,
             "input_mode": args.input_mode,
             "rows": [{"batch": r[0], "ms_step": r[1], "samples_s": r[2],
                       "peak_gb": r[3], "peak_pct": r[4]} for r in rows]},
            indent=1))
        return

    lr = min(8e-4, 2e-4 * args.batch / 8) if args.lr == "auto" else float(args.lr)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=lr, total_steps=args.steps, pct_start=0.05,
        cycle_momentum=False)
    scaler = torch.amp.GradScaler("cuda", enabled=args.amp == "fp16")
    amp_dtype = torch.bfloat16 if args.amp == "bf16" else torch.float16

    # validation pairs: fixed 400-pair subset of FT3D-TEST.
    # native mode: primary val axis is NATIVE 960x540 pad16 (matches
    # OpenStereo's SceneFlow eval protocol); a resized-axis eval is kept
    # alongside for continuity with the legacy resize run's val curve.
    val_keys = None
    if args.val_manifest:
        import gzip
        man = json.loads(gzip.decompress(Path(args.val_manifest).read_bytes()))
        val_keys = set(man["val_subset"])
    val_pairs = load_test_pairs(shards_dir, val_keys, native=native)
    val_pairs_rs = ([downscale_pair(p) for p in val_pairs] if native
                    else val_pairs)
    print(f"val pairs: {len(val_pairs)} (subset={val_keys is not None}, "
          f"axis={'native+resized' if native else 'resized'})")

    # tracked images (native mode): 50 train + 50 val fixed pairs
    tracked_train, tracked_val = [], []
    images_dir = out_dir / "images"
    if native and (args.track_train or args.track_val):
        tracked_train = load_tracked_train(shards_dir, args.track_train,
                                           args.seed)
        tracked_val = val_pairs[:args.track_val]
        init_tracked_dirs(images_dir, tracked_train, "train")
        init_tracked_dirs(images_dir, tracked_val, "val")
        print(f"tracked images: {len(tracked_train)} train / "
              f"{len(tracked_val)} val -> {images_dir}")

    start_step, best_val = 0, float("inf")
    val_log: list = []          # (step, val_epe) — drives auto-extension
    phase, target, extend_used = "main", args.steps, 0
    leg_prev_best = float("inf")   # best val EPE when the current tail began
    ckpt_path = out_dir / "latest.pth"
    if args.resume and ckpt_path.exists():
        ck = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["opt"])
        scaler.load_state_dict(ck["scaler"])
        start_step, best_val = ck["step"], ck["best_val"]
        val_log = ck.get("val_log", [])
        phase = ck.get("phase", "main")
        target = ck.get("target", args.steps)
        extend_used = ck.get("extend_used", 0)
        leg_prev_best = ck.get("leg_prev_best", float("inf"))
        if phase == "tail":  # rebuild the tail scheduler before loading state
            sched = _make_tail_sched(opt, args.extend_lr, args.extend_leg)
        sched.load_state_dict(ck["sched"])
        print(f"resumed from step {start_step} (best val EPE {best_val:.4f}, "
              f"phase={phase}, target={target})")

    meta = {"args": vars(args), "arch": args.arch, "lr_peak": lr,
            "params_train_M": n_params / 1e6,
            "input_mode": args.input_mode,
            "yjitter": args.yjitter,
            "crop_hw": [TRAIN_H, TRAIN_W] if native else None,
            "val_axis": "native_960x540_pad16" if native else "resized_640x384",
            "disp_png_encoding": "uint16 = disparity_px * 256",
            "tracked_train_keys": [(p["seq"], p["t"]) for p in tracked_train],
            "tracked_val_keys": [(p["seq"], p["t"]) for p in tracked_val],
            "split": "sceneflow_split_v1 (train 35454 / test 4370)",
            "n_train_shards": len(train_shards),
            "started_at": datetime.now().isoformat(timespec="seconds"),
            "gpu": torch.cuda.get_device_name(0)}
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=1))
    print(f"params {n_params/1e6:.4f} M | peak lr {lr:.1e} | "
          f"batch {args.batch} | {args.steps} steps | amp {args.amp} | "
          f"input {args.input_mode}")

    csv_path = out_dir / "train.csv"
    if not csv_path.exists():
        csv_path.write_text("step,loss,lr,ms_step,val_epe,val_bad05,"
                            "val_bad1,val_bad2,val_d1,elapsed_s,"
                            "val_epe_rs,val_bad05_rs,val_bad1_rs,"
                            "val_bad2_rs,val_d1_rs\n")
    aug_rng = random.Random(args.seed + 7)
    it = iter(loader)
    t0, t_step, nan_streak = time.time(), time.time(), 0
    loss_ema = None
    collage_train_pairs = None  # 3 raw train samples, stashed at step 1

    step = start_step
    while step < target:
        step += 1
        batch = next(it)
        if collage_train_pairs is None:
            collage_train_pairs = [
                dict(L=batch["L"][i].clone(), R=batch["R"][i].clone(),
                     D=batch["D"][i].clone(), seq="train", t=i)
                for i in range(min(3, batch["L"].shape[0]))]
        L, R, D, V = to_device(batch, device)
        if args.aug:
            L, R, D, V = augment_batch(L, R, D, V, aug_rng)
        opt.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda", dtype=amp_dtype):
            out = model(L, R, aux=True)
            loss = loss_fn(out, D, V, trunc_A=args.trunc_A,
                           init_ce_w=args.init_ce_w,
                           slant_w=args.slant_w, bimodal_w=args.bimodal_w)
        if not torch.isfinite(loss):
            nan_streak += 1
            if nan_streak >= 50:
                raise RuntimeError(f"non-finite loss x{nan_streak} @ {step}")
            _safe_sched_step(sched)
            continue
        nan_streak = 0
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(opt)
        scaler.update()
        _safe_sched_step(sched)
        lv = float(loss.detach())
        loss_ema = lv if loss_ema is None else 0.98 * loss_ema + 0.02 * lv

        if step % 100 == 0:
            ms = (time.time() - t_step) / 100 * 1000
            t_step = time.time()
            print(f"step {step:6d}  loss {loss_ema:.4f}  "
                  f"lr {sched.get_last_lr()[0]:.2e}  {ms:.0f} ms/step",
                  flush=True)
            with open(csv_path, "a") as f:
                f.write(f"{step},{loss_ema:.5f},{sched.get_last_lr()[0]:.2e},"
                        f"{ms:.1f},,,,,,{time.time()-t0:.0f},,,,,\n")

        if step % args.eval_every == 0 or step == target:
            with torch.no_grad():  # fp32 val, same protocol as the ablations
                nbs = args.val_bs or (4 if native else 8)
                vm = evaluate(model, val_pairs, device, bs=nbs)
                vr = (evaluate(model, val_pairs_rs, device,
                               bs=max(nbs, 2) * 2) if native else vm)
            val_log.append((step, float(vm["epe"])))
            print(f"  VAL @ {step}: epe {vm['epe']:.4f}  "
                  f"bad0.5 {vm['bad_0.5']:.2f}  bad1 {vm['bad_1.0']:.2f}  "
                  f"D1 {vm['d1_all']:.2f}"
                  + (f"  | resized epe {vr['epe']:.4f}" if native else "")
                  + ("  ** new best" if vm["epe"] < best_val else ""),
                  flush=True)
            with open(csv_path, "a") as f:
                f.write(f"{step},{loss_ema:.5f},{sched.get_last_lr()[0]:.2e},"
                        f",{vm['epe']:.4f},{vm['bad_0.5']:.3f},"
                        f"{vm['bad_1.0']:.3f},{vm['bad_2.0']:.3f},"
                        f"{vm['d1_all']:.3f},{time.time()-t0:.0f},"
                        f"{vr['epe']:.4f},{vr['bad_0.5']:.3f},"
                        f"{vr['bad_1.0']:.3f},{vr['bad_2.0']:.3f},"
                        f"{vr['d1_all']:.3f}\n")
            if vm["epe"] < best_val:
                best_val = vm["epe"]
                _atomic_save({"model": model.state_dict(), "step": step,
                              "val_metrics": vm, "val_metrics_resized": vr,
                              "cfg": args.arch}, out_dir / "best.pth")
            _atomic_save({"model": model.state_dict(),
                          "opt": opt.state_dict(), "sched": sched.state_dict(),
                          "scaler": scaler.state_dict(), "step": step,
                          "best_val": best_val, "val_log": val_log,
                          "phase": phase, "target": target,
                          "extend_used": extend_used,
                          "leg_prev_best": leg_prev_best}, ckpt_path)
            if args.ckpt_every and step % args.ckpt_every == 0:
                _atomic_save({"model": model.state_dict(), "step": step,
                              "val_metrics": vm, "val_metrics_resized": vr,
                              "cfg": args.arch},
                             out_dir / "checkpoints" / f"step_{step:06d}.pth")
            try:
                if tracked_train or tracked_val:
                    tv0 = time.time()
                    tbs = args.val_bs or 8
                    save_tracked(model, tracked_train, "train", images_dir,
                                 step, device, bs=tbs)
                    save_tracked(model, tracked_val, "val", images_dir,
                                 step, device, bs=tbs)
                    print(f"  tracked images saved "
                          f"({time.time()-tv0:.1f}s)", flush=True)
                elif not native:  # legacy collage path (resize mode only)
                    col = make_collage(model, collage_train_pairs,
                                       val_pairs[:2], device, step, args)
                    cv2.imwrite(str(out_dir / "viz" /
                                    f"collage_{step:06d}.png"), col)
            except Exception as e:  # artifact writes must never kill the run
                print(f"  tracked/collage failed: {e}")
            meta["best_val_epe"] = best_val
            meta["last_step"] = step
            meta["phase"] = phase
            meta["extend_used"] = extend_used
            (out_dir / "meta.json").write_text(json.dumps(meta, indent=1))

        # --- auto-extension: after the planned horizon, keep training in
        # low-LR cosine tails while the val curve still pays. Plateau logic
        # is only valid here (low, decaying LR), never mid-OneCycle. ---
        if (step == target and args.auto_extend
                and extend_used < args.extend_cap):
            if phase == "main":
                # window rule: did best-val improve >1% within the last 4
                # evals of the main schedule?
                epes = [e for _, e in val_log]
                improving = (len(epes) < 5
                             or min(epes[-4:]) < min(epes[:-4]) * 0.99)
                reason = ">1% gain in the last 4 main-phase evals"
            else:
                # leg rule: did THIS tail beat the best it started from by
                # >0.5%? (window rule would double-count the main finish)
                improving = best_val < leg_prev_best * 0.995
                reason = ">0.5% gain within the last tail"
            if improving:
                leg_prev_best = best_val
                target += args.extend_leg
                extend_used += args.extend_leg
                phase = "tail"
                sched = _make_tail_sched(opt, args.extend_lr, args.extend_leg)
                print(f"AUTO-EXTEND ({reason}) -> +{args.extend_leg} steps "
                      f"at lr {args.extend_lr:.1e} (target {target}, "
                      f"extension used {extend_used}/{args.extend_cap})",
                      flush=True)
            else:
                print(f"NO EXTEND: val plateaued "
                      f"(best {best_val:.4f}); stopping at {step}.",
                      flush=True)

    # final: FULL 4,370-pair test on the best checkpoint (the paper number).
    # native mode: native axis (the run's protocol axis).
    if args.final_full_eval:
        ck = torch.load(out_dir / "best.pth", map_location=device,
                        weights_only=False)
        model.load_state_dict(ck["model"])
        full_pairs = load_test_pairs(shards_dir, None, native=native)
        print(f"FULL TEST: {len(full_pairs)} pairs "
              f"(axis={'native' if native else 'resized'}) ...")
        with torch.no_grad():
            fm = evaluate(model, full_pairs, device,
                          bs=args.val_bs or (4 if native else 8))
        print("FINAL full-test metrics: "
              + "  ".join(f"{k}={v:.4f}" for k, v in fm.items()), flush=True)
        meta["final_metrics_all"] = fm
        meta["final_axis"] = "native_960x540_pad16" if native else "resized"
        meta["final_best_step"] = ck["step"]
        meta["finished"] = datetime.now().isoformat(timespec="seconds")
        (out_dir / "meta.json").write_text(json.dumps(meta, indent=1))


if __name__ == "__main__":
    main()
