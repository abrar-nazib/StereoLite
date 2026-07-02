"""Multi-way architecture A/B overfit comparison.

Architectures (legacy, GhostConv encoder fixed for fairness):
    current  : StereoLite_yolo with backbone=ghost.
               TileRefine iterated at 1/16 (×2), 1/8 (×3), 1/4 (×3),
               then ConvexUpsample 1/4 → 1/2 → full.

    v1_iter  : StereoLite_v1_iter.
               Same TileRefine iterations as current, PLUS additional
               iterations at 1/2 (×2). No ConvexUpsample. Plane equation
               upsample 1/2 → full.

    v2_hitnet: StereoLite_v2_hitnet.
               Replaces all TileRefine with HITNet's exact propagation
               block (residual blocks + dilated convs + local cost-volume
               augmentation, no BN, leaky ReLU). Single pass per scale at
               1/16, 1/8, 1/4, 1/2. Plane equation upsample throughout.
               No ConvexUpsample.

Architectures (newer, backbone + extend_to_full configurable):
    costlookup : YOLO encoder + local cost-lookup-driven TileRefine.
    tilegru    : YOLO encoder + ConvGRU on tile.feat (no cost lookup).
    raftlike   : YOLO encoder + cost-lookup-driven ConvGRU (combo).
    yolo_ctx   : YOLO matching encoder + dedicated context-encoder
                 stream (RAFT-Stereo Fig 1 bottom stream). Cost-lookup-
                 driven ConvGRU on the tile feat slot, initialised from
                 the projected context features at 1/16.
    yolo_ctx_gate: yolo_ctx plus a lightweight context/confidence gate
                   that scales residual tile updates.
    yolo_ctx_gev4: yolo_ctx_gate plus a fail-soft 1/4 Geometry Encoding
                   Volume before final 1/4 refinement.
    yolo_ctx_hstereo: yolo_ctx_gev4 plus true 1/2-resolution TileRefineCtx
                      before the final upsample.
    yolo_ctx_guided: yolo_ctx_gev4 plus fail-soft full-resolution guided
                     propagation for sharper boundaries.
    yolo_ctx_init4: yolo_ctx_gate plus a fresh 1/4-resolution cost-volume
                    warm start before the final 1/4 refinement stage.
    yolo_ctx_sru: yolo_ctx_gate plus a Selective-Stereo-style recurrent
                  update with 1x1 edge and 3x3 smooth branches.
    yolo_ctx_sharp: yolo_ctx_gate plus a lightweight full-resolution
                    boundary residual tail.
    yolo_ctx_hrrefine: yolo_ctx_gate plus a small half-resolution residual
                       refinement loop before final full-resolution output.
    yolo_geomctx: yolo_ctx plus slant-aware local lookup and a context /
                  confidence gate that blends ConvGRU updates with local
                  stereo evidence.

Run:
    python3 model/scripts/overfit_arch_ablation.py --arch current
    python3 model/scripts/overfit_arch_ablation.py --arch raftlike --backbone yolo26s
    python3 model/scripts/overfit_arch_ablation.py --arch yolo_ctx --backbone yolo26s
    python3 model/scripts/overfit_arch_ablation.py --arch yolo_ctx_gate --backbone yolo26s
    python3 model/scripts/overfit_arch_ablation.py --arch yolo_ctx_gev4 --backbone yolo26s
    python3 model/scripts/overfit_arch_ablation.py --arch yolo_ctx_guided --backbone yolo26s
    python3 model/scripts/overfit_arch_ablation.py --arch yolo_ctx_init4 --backbone yolo26s
    python3 model/scripts/overfit_arch_ablation.py --arch yolo_ctx_sru --backbone yolo26s
    python3 model/scripts/overfit_arch_ablation.py --arch yolo_ctx_sharp --backbone yolo26s
    python3 model/scripts/overfit_arch_ablation.py --arch yolo_ctx_hrrefine --backbone yolo26s
    python3 model/scripts/overfit_arch_ablation.py --arch yolo_geomctx --backbone yolo26s

Or run all back-to-back into one timestamped directory:
    TS=$(date +%Y%m%d-%H%M%S)
    OUT=model/benchmarks/arch_ablation_$TS
    for a in current costlookup tilegru raftlike yolo_ctx; do
        python3 model/scripts/overfit_arch_ablation.py --arch $a --out_root $OUT
    done

Outputs:
    model/benchmarks/arch_ablation_<TS>/<arch>/{train.csv,curve.png,README.md,viz/}
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import platform
import random
import sys
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "model" / "designs"))

# Reuse helpers from overfit_yolo_ablation.py
sys.path.insert(0, str(ROOT / "model" / "scripts"))
from overfit_yolo_ablation import (  # noqa: E402
    SF_ROOT_CANDIDATES, BENCH_ROOT, CACHE_PATH,
    load_or_cache_pairs, stereo_metrics, render_panel,
)


def _cache_path_for(n_pairs: int, height: int = 384, width: int = 640):
    """Derive a per-(n_pairs, resolution) cache file so larger sweeps
    don't clobber the default 20-pair cache. n=20 at default resolution
    keeps the original path for backward compat with prior runs.
    Higher-res training writes a separate cache file."""
    from pathlib import Path
    if n_pairs == 20 and height == 384 and width == 640:
        return CACHE_PATH
    cache_dir = CACHE_PATH.parent
    cache_dir.mkdir(parents=True, exist_ok=True)
    if height == 384 and width == 640:
        # Backward-compat name when resolution is the default
        return cache_dir / f"sf_overfit_pairs_v1_n{n_pairs}.pt"
    return cache_dir / f"sf_overfit_pairs_v1_n{n_pairs}_{height}x{width}.pt"


_NEW_ARCHES = ("costlookup", "tilegru", "raftlike", "yolo_ctx",
               "yolo_ctx_gate", "yolo_ctx_gev4", "yolo_ctx_hstereo",
               "yolo_ctx_guided",
               "yolo_ctx_init4", "yolo_ctx_sru", "yolo_ctx_sharp",
               "yolo_ctx_hrrefine", "yolo_geomctx")

# Phase-2 architecture flags injected from main()'s argparse before
# build_model() is called. Avoids changing build_model's signature for
# every variant.
_arch_flags: dict[str, bool] = {}


def build_model(arch: str, backbone: str = "ghost",
                extend_to_full: bool = False,
                widener: str | None = None):
    """Return (StereoLite_instance, StereoLiteConfig).

    `backbone` and `extend_to_full` are honoured for the costlookup /
    tilegru / raftlike variants. The legacy archs (current, v1_iter,
    v2_hitnet) ignore them — they're each fixed to a specific config.
    """
    if arch == "current":
        from StereoLite_yolo.model import StereoLite, StereoLiteConfig
        cfg = StereoLiteConfig(backbone="ghost")
    elif arch == "v1_iter":
        from StereoLite_v1_iter.model import StereoLite, StereoLiteConfig
        cfg = StereoLiteConfig()
    elif arch == "v2_hitnet":
        from StereoLite_v2_hitnet.model import StereoLite, StereoLiteConfig
        cfg = StereoLiteConfig()
    elif arch == "costlookup":
        from StereoLite_costlookup.model import StereoLite, StereoLiteConfig
        cfg_kwargs = dict(backbone=backbone, extend_to_full=extend_to_full,
                           widener=widener)
        # Phase-2 architecture ablation flags (passed via outer scope).
        for k in ("slope_aware_warp", "selective_gate", "cascade_cv_4",
                   "context_branch", "init_regress", "init_gce"):
            v = _arch_flags.get(k)
            if v is not None:
                cfg_kwargs[k] = v
        cfg = StereoLiteConfig(**cfg_kwargs)
    elif arch == "tilegru":
        from StereoLite_tilegru.model import StereoLite, StereoLiteConfig
        cfg = StereoLiteConfig(backbone=backbone, extend_to_full=extend_to_full)
    elif arch == "raftlike":
        from StereoLite_raftlike.model import StereoLite, StereoLiteConfig
        cfg = StereoLiteConfig(backbone=backbone, extend_to_full=extend_to_full)
    elif arch == "yolo_ctx":
        from StereoLite_yolo_ctx.model import (
            StereoLiteYoloCtx, StereoLiteYoloCtxConfig)
        cfg = StereoLiteYoloCtxConfig(backbone=backbone)
        return StereoLiteYoloCtx(cfg), cfg
    elif arch == "yolo_ctx_gate":
        from StereoLite_yolo_ctx_gate.model import (
            StereoLiteYoloCtxGate, StereoLiteYoloCtxGateConfig)
        cfg = StereoLiteYoloCtxGateConfig(backbone=backbone)
        return StereoLiteYoloCtxGate(cfg), cfg
    elif arch == "yolo_ctx_gev4":
        from StereoLite_yolo_ctx_gev4.model import (
            StereoLiteYoloCtxGEV4, StereoLiteYoloCtxGEV4Config)
        cfg = StereoLiteYoloCtxGEV4Config(backbone=backbone)
        return StereoLiteYoloCtxGEV4(cfg), cfg
    elif arch == "yolo_ctx_hstereo":
        from StereoLite_yolo_ctx_hstereo.model import (
            StereoLiteYoloCtxHStereo, StereoLiteYoloCtxHStereoConfig)
        cfg = StereoLiteYoloCtxHStereoConfig(backbone=backbone)
        return StereoLiteYoloCtxHStereo(cfg), cfg
    elif arch == "yolo_ctx_guided":
        from StereoLite_yolo_ctx_guided.model import (
            StereoLiteYoloCtxGuided, StereoLiteYoloCtxGuidedConfig)
        cfg = StereoLiteYoloCtxGuidedConfig(backbone=backbone)
        return StereoLiteYoloCtxGuided(cfg), cfg
    elif arch == "yolo_ctx_init4":
        from StereoLite_yolo_ctx_init4.model import (
            StereoLiteYoloCtxInit4, StereoLiteYoloCtxInit4Config)
        cfg = StereoLiteYoloCtxInit4Config(backbone=backbone)
        return StereoLiteYoloCtxInit4(cfg), cfg
    elif arch == "yolo_ctx_sru":
        from StereoLite_yolo_ctx_sru.model import (
            StereoLiteYoloCtxSRU, StereoLiteYoloCtxSRUConfig)
        cfg = StereoLiteYoloCtxSRUConfig(backbone=backbone)
        return StereoLiteYoloCtxSRU(cfg), cfg
    elif arch == "yolo_ctx_sharp":
        from StereoLite_yolo_ctx_sharp.model import (
            StereoLiteYoloCtxSharp, StereoLiteYoloCtxSharpConfig)
        cfg = StereoLiteYoloCtxSharpConfig(backbone=backbone)
        return StereoLiteYoloCtxSharp(cfg), cfg
    elif arch == "yolo_ctx_hrrefine":
        from StereoLite_yolo_ctx_hrrefine.model import (
            StereoLiteYoloCtxHRRefine, StereoLiteYoloCtxHRRefineConfig)
        cfg = StereoLiteYoloCtxHRRefineConfig(backbone=backbone)
        return StereoLiteYoloCtxHRRefine(cfg), cfg
    elif arch == "yolo_geomctx":
        from StereoLite_yolo_geomctx.model import (
            StereoLiteYoloGeomCtx, StereoLiteYoloGeomCtxConfig)
        cfg = StereoLiteYoloGeomCtxConfig(
            backbone=backbone,
            slant_patch_radius=int(_arch_flags.get("slant_patch_radius", 1)))
        return StereoLiteYoloGeomCtx(cfg), cfg
    else:
        raise ValueError(f"unknown arch: {arch}")
    return StereoLite(cfg), cfg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arch",
                    choices=["current", "v1_iter", "v2_hitnet",
                              "costlookup", "tilegru", "raftlike",
                              "yolo_ctx", "yolo_ctx_gate",
                              "yolo_ctx_gev4", "yolo_ctx_hstereo",
                              "yolo_ctx_guided",
                              "yolo_ctx_init4", "yolo_ctx_sru",
                              "yolo_ctx_sharp",
                              "yolo_ctx_hrrefine", "yolo_geomctx"],
                    required=True)
    ap.add_argument("--backbone",
                    choices=["ghost", "yolo26n", "yolo26s"],
                    default="ghost",
                    help="encoder backbone (only used for costlookup/"
                          "tilegru/raftlike). Legacy archs ignore this.")
    ap.add_argument("--extend_to_full", type=int, default=0,
                    help="1 = TileRefine all the way to 1/2 + plane-eq to "
                          "full (no ConvexUpsample). 0 = ConvexUpsample "
                          "1/4 -> 1/2 -> full. Only used for new archs.")
    ap.add_argument("--widener", type=str, default=None,
                    choices=[None, "none", "f2_only", "f2_f4", "all_to_s",
                              "dw", "mbconv", "ghostconv",
                              "topdown_fpn", "bifpn", "gn_replace"],
                    help="optional feature widener applied to the yolo26n "
                          "encoder outputs (only meaningful for costlookup). "
                          "See model/designs/_wideners.py for definitions.")
    ap.add_argument("--loss_variant", type=str, default="baseline",
                    choices=["baseline", "seq_loss", "slope_sup",
                              "conf_aware", "edge_smooth",
                              "boundary_focus", "seq_boundary"],
                    help="Phase-2 loss-side ablation variant. baseline = "
                          "the prior multi-scale L1 + grad + bad-1 hinge "
                          "(matches the 12-config sweep). Others add a "
                          "single extra term on top.")
    ap.add_argument("--seq_loss_gamma", type=float, default=0.9,
                    help="γ exponent for the RAFT-style γ-weighted "
                          "per-iter sequence loss (only used if "
                          "--loss_variant=seq_loss). Lower γ = more "
                          "weight on the FINAL iter; higher γ = more "
                          "uniform across iters.")
    ap.add_argument("--guided_start_step", type=int, default=3000,
                    help="yolo_ctx_guided only: train the GEV4/base path with "
                         "guided refinement bypassed before this step. This "
                         "keeps the sharpness tail from slowing early EPE.")
    # Phase-2 architecture-side ablation flags.
    ap.add_argument("--slope_aware_warp", type=int, default=0,
                    help="A2 — warp fR by slope-corrected disparity")
    ap.add_argument("--selective_gate", type=int, default=0,
                    help="A5 — per-pixel 'stop refining' gate in TileRefine")
    ap.add_argument("--cascade_cv_4", type=int, default=0,
                    help="A3 — narrow-range 3D-agg CV at 1/4")
    ap.add_argument("--context_branch", type=int, default=0,
                    help="A1 — small parallel context encoder fed into TileRefine")
    ap.add_argument("--init_regress", type=int, default=0,
                    help="Expert-recommended: APC + 2-layer regression head "
                          "in TileInit (costlookup variant only). Pair with "
                          "--init_loss_weight > 0 to actually supervise it.")
    ap.add_argument("--init_gce", type=int, default=0,
                    help="Expert-recommended: wire CoEx-style "
                          "GuidedCostExcitation into TileInit's 3D aggregator "
                          "(costlookup variant only). Free accuracy per CoEx.")
    ap.add_argument("--pure_l1", type=int, default=0,
                    help="Diagnostic: strip the loss to a single full-res L1 "
                          "term (drops multi-scale, grad consistency, hinges, "
                          "and all Phase-2 extras). Tests the overfit ceiling "
                          "of the architecture without regularizer interference.")
    ap.add_argument("--init_loss_weight", type=float, default=0.0,
                    help="Smooth-L1 supervision weight on the post-init "
                          "disparity (out_dict['d32']). Implements the IGEV "
                          "L_init pattern. 0 = off; 0.5 is a reasonable start.")
    ap.add_argument("--slant_patch_radius", type=int, default=1,
                    help="yolo_geomctx only: 1 = 3x3 slanted support patch, "
                          "0 = center-only lookup ablation.")
    ap.add_argument("--variant_tag", type=str, default="",
                    help="optional override for the per-variant subdir. "
                          "Defaults to arch name.")
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--height", type=int, default=384)
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--n_pairs", type=int, default=20)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out_root", type=str, default="")
    ap.add_argument("--show", type=int, default=1,
                    help="1 = open OpenCV window with live panel; 0 = headless")
    ap.add_argument("--viz_interval_s", type=float, default=15.0)
    ap.add_argument("--viz_pair_idx", type=int, default=-1,
                    help="-1 = choose one seeded-random pair for viz; "
                         ">=0 = fixed pair index.")
    ap.add_argument("--viz_rotate", type=int, default=0,
                    help="1 = rotate the saved/live viz pair over the "
                         "loaded training pairs.")
    args = ap.parse_args()

    # Reproducibility patch.
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # `use_deterministic_algorithms(True)` forces a decomposed path for
    # bilinear interpolate that allocates ~2x more activation memory at
    # high resolutions, which OOMs on RTX 3050 for the new arches at 1/2
    # res. Seeded RNG + cudnn.deterministic still give run-to-run
    # reproducibility for the legacy archs; we relax only the ATen-level
    # enforcement, and only for the new (memory-heavier) arches.
    if args.arch in _NEW_ARCHES:
        torch.use_deterministic_algorithms(False)
    else:
        torch.use_deterministic_algorithms(True, warn_only=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device != "cuda":
        print("[WARN] no GPU detected; running on CPU will be very slow.")

    # Output dir.
    if args.out_root:
        out_root = Path(args.out_root)
    else:
        ts = datetime.now().strftime("%Y%m%d-%H%M%S")
        out_root = BENCH_ROOT / f"arch_ablation_{ts}"
    # variant_tag wins if set; else encode (arch, backbone, extend) for the
    # new archs to disambiguate sweep variants.
    if args.variant_tag:
        tag = args.variant_tag
    elif args.arch in _NEW_ARCHES:
        suffix = "_full" if args.extend_to_full else ""
        widener_suf = (f"_w_{args.widener}" if args.widener
                        and args.widener != "none" else "")
        tag = f"{args.arch}_{args.backbone}{suffix}{widener_suf}"
    else:
        tag = args.arch
    out = out_root / tag
    out.mkdir(parents=True, exist_ok=True)

    # Load pairs.
    Ls, Rs, Ds, valid, pair_paths = load_or_cache_pairs(
        args.n_pairs, (args.height, args.width),
        cache_path=_cache_path_for(args.n_pairs, args.height, args.width))
    # For n_pairs > ~30 the upfront .to(device) eats too much GPU memory on
    # the RTX 3050 (each pair is ~7 MB across L+R+D+valid; 100 pairs = ~786 MB
    # before training even starts). Keep on pinned CPU memory and ship each
    # mini-batch to GPU on demand. For small N (<=30) keep the old fast path.
    cpu_residency = args.n_pairs > 30
    if cpu_residency:
        if device == "cuda":
            Ls = Ls.pin_memory(); Rs = Rs.pin_memory()
            Ds = Ds.pin_memory(); valid = valid.pin_memory()
    else:
        Ls = Ls.to(device); Rs = Rs.to(device)
        Ds = Ds.to(device); valid = valid.to(device)
    print(f"  L shape={tuple(Ls.shape)}  D range=[{Ds[valid > 0].min():.1f}, "
          f"{Ds[valid > 0].max():.1f}]")

    # Pass Phase-2 arch flags into build_model via module-level dict.
    _arch_flags["slope_aware_warp"] = bool(args.slope_aware_warp)
    _arch_flags["selective_gate"] = bool(args.selective_gate)
    _arch_flags["cascade_cv_4"] = bool(args.cascade_cv_4)
    _arch_flags["context_branch"] = bool(args.context_branch)
    _arch_flags["init_regress"] = bool(args.init_regress)
    _arch_flags["init_gce"] = bool(args.init_gce)
    _arch_flags["slant_patch_radius"] = int(args.slant_patch_radius)

    # Build model.
    model, cfg = build_model(args.arch,
                              backbone=args.backbone,
                              extend_to_full=bool(args.extend_to_full),
                              widener=args.widener)
    model = model.to(device)
    n_total = sum(p.numel() for p in model.parameters())
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"model: arch={args.arch}, total={n_total/1e6:.3f} M, "
          f"train={n_train/1e6:.3f} M")

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)

    meta = {
        "arch": args.arch,
        "steps": args.steps,
        "lr": args.lr,
        "batch": args.batch,
        "height": args.height,
        "width": args.width,
        "n_pairs": args.n_pairs,
        "seed": args.seed,
        "device": device,
        "gpu": torch.cuda.get_device_name(0) if device == "cuda" else None,
        "pytorch": torch.__version__,
        "platform": platform.platform(),
        "started_at": datetime.now().isoformat(timespec="seconds"),
        "params_total_M": round(n_total / 1e6, 4),
        "params_train_M": round(n_train / 1e6, 4),
        "encoder_out_channels": list(model.fnet.out_channels),
        "pair_paths": pair_paths,
        "viz_pair_idx": args.viz_pair_idx,
        "viz_rotate": bool(args.viz_rotate),
        "guided_start_step": args.guided_start_step,
    }
    (out / "meta.json").write_text(json.dumps(meta, indent=2))

    # Training loop.
    csv_path = out / "train.csv"
    cf = open(csv_path, "w", newline="")
    w = csv.writer(cf)
    w.writerow(["step", "loss", "epe", "rmse", "bad_1.0", "bad_2.0",
                "d1_all", "lr", "elapsed_s"])

    t0 = time.time()
    N = Ls.shape[0]
    rng = np.random.default_rng(args.seed)
    peak_mem = 0
    model.train()
    viz_dir = out / "viz"
    viz_dir.mkdir(exist_ok=True)
    last_viz_t = 0.0
    win_name = f"overfit {args.arch}"
    viz_rng = np.random.default_rng(args.seed + 1009)
    if args.viz_pair_idx < 0:
        viz_pair0 = int(viz_rng.integers(0, N))
    else:
        viz_pair0 = max(0, min(args.viz_pair_idx, N - 1))
    viz_count = 0

    def get_viz_batch(pair_idx: int):
        Lviz = Ls[pair_idx:pair_idx + 1]
        Rviz = Rs[pair_idx:pair_idx + 1]
        Dviz = Ds[pair_idx:pair_idx + 1]
        Vviz = valid[pair_idx:pair_idx + 1]
        if cpu_residency and device == "cuda":
            Lviz = Lviz.to(device, non_blocking=True)
            Rviz = Rviz.to(device, non_blocking=True)
            Dviz = Dviz.to(device, non_blocking=True)
            Vviz = Vviz.to(device, non_blocking=True)
        return Lviz, Rviz, Dviz, Vviz

    show_ok = bool(args.show) and bool(os.environ.get("DISPLAY"))

    def ms_l1(pred, gt, val, scale):
        if pred.shape[-2:] != gt.shape[-2:]:
            pred = F.interpolate(pred, size=gt.shape[-2:],
                                  mode="bilinear", align_corners=False) * scale
        return ((pred - gt).abs() * val).sum() / val.sum().clamp(min=1)

    def grad_consistency(pred, gt, val):
        gx_p = pred[..., :, 1:] - pred[..., :, :-1]
        gx_g = gt[..., :, 1:] - gt[..., :, :-1]
        gy_p = pred[..., 1:, :] - pred[..., :-1, :]
        gy_g = gt[..., 1:, :] - gt[..., :-1, :]
        vx = val[..., :, 1:] * val[..., :, :-1]
        vy = val[..., 1:, :] * val[..., :-1, :]
        lx = ((gx_p - gx_g).abs() * vx).sum() / vx.sum().clamp(min=1)
        ly = ((gy_p - gy_g).abs() * vy).sum() / vy.sum().clamp(min=1)
        return lx + ly

    def bad1_hinge(pred, gt, val):
        err = (pred - gt).abs()
        hinge = (err - 1.0).clamp(min=0) ** 2
        return (hinge * val).sum() / val.sum().clamp(min=1)

    def to_unit_image(x):
        return x / 255.0 if x.detach().amax() > 2.0 else x

    # ---------- Phase-2 ablation extras (loss-side variants) ----------
    def seq_loss(out_dict, gt, val, gamma=0.9):
        """RAFT-style γ-weighted L1 across every TileRefine iteration at
        every scale. Each list `iter_stages_<s>` holds tile.d at every iter
        — element 0 is post-init/upsample, element i>=1 is after the i-th
        refine pass. Later iters get higher weight; supervises early iters."""
        total = 0.0
        wsum = 0.0
        scale_keys = [(16, "iter_stages_16"), (8, "iter_stages_8"),
                      (4, "iter_stages_4"),  (2, "iter_stages_2")]
        all_stages = [(s, d) for s, k in scale_keys for d in out_dict.get(k, [])]
        N = len(all_stages)
        for i, (sc, d) in enumerate(all_stages):
            weight = gamma ** (N - 1 - i)
            if d.shape[-2:] != gt.shape[-2:]:
                d_up = F.interpolate(d, size=gt.shape[-2:],
                                      mode="bilinear", align_corners=False) * sc
            else:
                d_up = d
            term = ((d_up - gt).abs() * val).sum() / val.sum().clamp(min=1)
            total = total + weight * term
            wsum += weight
        return total / max(wsum, 1e-6)

    def slope_sup_loss(out_dict, gt, val):
        """Direct supervision of sx, sy from finite-difference GT gradient.
        Currently slopes (sx, sy) are unsupervised — only flow through
        plane-eq upsample. This term gives them an explicit target."""
        # Compute target slopes at full resolution from GT
        gx_gt = gt[..., :, 1:] - gt[..., :, :-1]                    # (B,1,H,W-1)
        gy_gt = gt[..., 1:, :] - gt[..., :-1, :]                    # (B,1,H-1,W)
        # Use d_full as predicted gradient surrogate (slope behaviour
        # in-betweens scales is hard to supervise; cleanest target is
        # via the final disparity prediction's gradient).
        d_full = out_dict["d_final"]
        gx_p = d_full[..., :, 1:] - d_full[..., :, :-1]
        gy_p = d_full[..., 1:, :] - d_full[..., :-1, :]
        vx = val[..., :, 1:] * val[..., :, :-1]
        vy = val[..., 1:, :] * val[..., :-1, :]
        lx = ((gx_p - gx_gt).abs() * vx).sum() / vx.sum().clamp(min=1)
        ly = ((gy_p - gy_gt).abs() * vy).sum() / vy.sum().clamp(min=1)
        # This is structurally similar to grad_consistency; the difference
        # is intent: slope_sup targets the *gradient-of-disparity* directly
        # instead of as a side-effect of multi-scale L1 + grad consistency.
        return lx + ly

    def conf_aware_loss(out_dict, gt, val, tau_correct=1.0):
        """Train conf head with proxy targets: pixels where |d_pred - GT|
        < tau_correct are 'correct', conf should be high. Else low.
        Loss: BCE on conf vs binary correctness target.
        Conf is at 1/4 resolution (last tile state's conf field upsampled
        to full); we read it from out_dict if present, else skip."""
        # We don't currently have conf in aux dict. For this ablation,
        # use the model's last tile state - we'd need to surface it.
        # Quick proxy: penalise where d_full is wrong (encourage low conf
        # on hard pixels), via |err| as a soft target weight.
        # (Real conf head loss requires extra plumbing; this term is a
        # lightweight proxy.)
        d_full = out_dict["d_final"]
        err = (d_full - gt).abs()
        # Target conf: 1 if correct (err < tau), 0 otherwise. Continuous
        # version: target = exp(-err / tau).
        target = torch.exp(-err / tau_correct)
        # Without an exposed conf head, supervise the *disparity* harder
        # where err > tau (focal-style weighting). This is a stand-in.
        focal_w = (1 - target).clamp(0, 1)
        return ((err * focal_w) * val).sum() / val.sum().clamp(min=1)

    def edge_smooth_loss(d, left, val, alpha=10.0):
        """First-order edge-aware smoothness: |∇d| * exp(-α |∇L|).
        Penalises disparity discontinuities except where the image has
        edges (preserves real depth boundaries, smooths flat regions)."""
        # Image gradient (mean over channels)
        L_grey = to_unit_image(left).mean(dim=1, keepdim=True)
        gx_L = (L_grey[..., :, 1:] - L_grey[..., :, :-1]).abs()
        gy_L = (L_grey[..., 1:, :] - L_grey[..., :-1, :]).abs()
        gx_d = (d[..., :, 1:] - d[..., :, :-1]).abs()
        gy_d = (d[..., 1:, :] - d[..., :-1, :]).abs()
        wx = torch.exp(-alpha * gx_L)
        wy = torch.exp(-alpha * gy_L)
        vx = val[..., :, 1:] * val[..., :, :-1]
        vy = val[..., 1:, :] * val[..., :-1, :]
        lx = (gx_d * wx * vx).sum() / vx.sum().clamp(min=1)
        ly = (gy_d * wy * vy).sum() / vy.sum().clamp(min=1)
        return lx + ly

    def boundary_focus_loss(pred, gt, val, left):
        """Extra pressure on thin structures and disparity discontinuities.

        Plain EPE is dominated by road/wall pixels. Tree leaves, poles, and
        far object boundaries occupy few pixels, so a model can look soft
        while still scoring well. This term upweights GT-disparity edges and
        image edges, and also matches second-order detail with a Laplacian.
        """
        gx_g = (gt[..., :, 1:] - gt[..., :, :-1]).abs()
        gy_g = (gt[..., 1:, :] - gt[..., :-1, :]).abs()
        disp_edge = F.pad(gx_g, (0, 1, 0, 0)) + F.pad(gy_g, (0, 0, 0, 1))
        L_grey = to_unit_image(left).mean(dim=1, keepdim=True)
        gx_l = (L_grey[..., :, 1:] - L_grey[..., :, :-1]).abs()
        gy_l = (L_grey[..., 1:, :] - L_grey[..., :-1, :]).abs()
        img_edge = F.pad(gx_l, (0, 1, 0, 0)) + F.pad(gy_l, (0, 0, 0, 1))

        # Clamp keeps the loss from being hijacked by huge foreground jumps.
        w_edge = 1.0 + 4.0 * disp_edge.clamp(max=2.0) + 1.5 * img_edge.clamp(max=0.2)
        l1 = (((pred - gt).abs() * w_edge) * val).sum() / (
            (w_edge * val).sum().clamp(min=1))

        k = pred.new_tensor([[0.0, 1.0, 0.0],
                             [1.0, -4.0, 1.0],
                             [0.0, 1.0, 0.0]]).view(1, 1, 3, 3)
        lap_p = F.conv2d(pred, k, padding=1)
        lap_g = F.conv2d(gt, k, padding=1)
        lap = (((lap_p - lap_g).abs() * w_edge) * val).sum() / (
            (w_edge * val).sum().clamp(min=1))
        return l1 + 0.35 * lap

    def guided_guard_loss(out_dict, gt, val, left):
        """Keep guided full-res propagation from damaging easy regions.

        The guided head is useful exactly at boundaries, but EPE explodes if it
        freely perturbs flat road/wall/sky regions. Use GT disparity edges plus
        image edges as a permissive boundary mask, then penalise final-vs-base
        changes and high gates away from those boundaries.
        """
        if "d_pre_guided" not in out_dict or "guided_gate" not in out_dict:
            return gt.new_zeros(())
        d0 = out_dict["d_pre_guided"]
        d1 = out_dict["d_final"]
        gate = out_dict["guided_gate"]
        if d0.shape[-2:] != gt.shape[-2:]:
            d0 = F.interpolate(d0, size=gt.shape[-2:],
                               mode="bilinear", align_corners=False)
        if gate.shape[-2:] != gt.shape[-2:]:
            gate = F.interpolate(gate, size=gt.shape[-2:],
                                 mode="bilinear", align_corners=False)

        gx_g = F.pad((gt[..., :, 1:] - gt[..., :, :-1]).abs(),
                     (0, 1, 0, 0))
        gy_g = F.pad((gt[..., 1:, :] - gt[..., :-1, :]).abs(),
                     (0, 0, 0, 1))
        grey = to_unit_image(left).mean(dim=1, keepdim=True)
        gx_l = F.pad((grey[..., :, 1:] - grey[..., :, :-1]).abs(),
                     (0, 1, 0, 0))
        gy_l = F.pad((grey[..., 1:, :] - grey[..., :-1, :]).abs(),
                     (0, 0, 0, 1))
        # Saturate quickly: any real GT/image boundary gets a free pass,
        # while smooth regions remain protected.
        boundary = (2.5 * (gx_g + gy_g).clamp(max=1.0)
                    + 5.0 * (gx_l + gy_l).clamp(max=0.2)).clamp(0.0, 1.0)
        smooth = (1.0 - boundary) * val
        denom = smooth.sum().clamp(min=1)
        delta = (d1 - d0).abs()
        delta_pen = (delta * smooth).sum() / denom
        gate_pen = (gate * smooth).sum() / denom
        return delta_pen + 0.25 * gate_pen

    def run_model(left, right, aux: bool, guided_active: bool):
        if args.arch == "yolo_ctx_guided":
            return model(left, right, aux=aux, use_guided=guided_active)
        return model(left, right, aux=aux)

    # AMP autocast for the new (memory-heavy) arches — cuts activation
    # memory ~50% with negligible accuracy impact in our experience.
    use_amp = (device == "cuda") and (args.arch in _NEW_ARCHES)
    scaler = torch.amp.GradScaler("cuda") if use_amp else None

    for step in range(1, args.steps + 1):
        # Allow batch > n_pairs by sampling with replacement; otherwise sample
        # without replacement (matches prior runs).
        replace_sampling = args.batch > N
        idx = rng.choice(N, size=args.batch, replace=replace_sampling)
        if cpu_residency and device == "cuda":
            L = Ls[idx].to(device, non_blocking=True)
            R = Rs[idx].to(device, non_blocking=True)
            D = Ds[idx].to(device, non_blocking=True)
            V = valid[idx].to(device, non_blocking=True)
        else:
            L = Ls[idx]; R = Rs[idx]; D = Ds[idx]; V = valid[idx]

        opt.zero_grad(set_to_none=True)
        guided_active = not (
            args.arch == "yolo_ctx_guided"
            and step < max(1, args.guided_start_step)
        )
        if use_amp:
            with torch.amp.autocast("cuda", dtype=torch.float16):
                # `with_iter_stages` is only accepted by the costlookup /
                # tilegru designs. raftlike and yolo_ctx don't expose per-
                # iter stage outputs, so passing the kwarg is a TypeError.
                if args.arch in ("costlookup", "tilegru"):
                    out_dict = model(L, R, aux=True,
                                      with_iter_stages=(
                                          args.loss_variant == "seq_loss"))
                else:
                    out_dict = run_model(L, R, aux=True,
                                         guided_active=guided_active)
                d_full = out_dict["d_final"]
                d_half = out_dict["d_half"]
                d4 = out_dict["d4"]
                d8 = out_dict["d8"]
                d16 = out_dict["d16"]
                if args.pure_l1:
                    # Diagnostic: pure full-resolution L1 only. No multi-scale,
                    # no grad consistency, no hinges, no Phase-2 extras, no
                    # init loss. Strictly tests the architecture's overfit
                    # ceiling without regularizer interference.
                    loss = ms_l1(d_full, D, V, 1.0)
                else:
                    loss = (
                        1.0 * ms_l1(d_full, D, V, 1.0)
                        + 0.5 * ms_l1(d_half, D, V, 2.0)
                        + 0.3 * ms_l1(d4, D, V, 4.0)
                        + 0.2 * ms_l1(d8, D, V, 8.0)
                        + 0.1 * ms_l1(d16, D, V, 16.0)
                        + 0.5 * grad_consistency(d_full, D, V)
                        + 0.2 * bad1_hinge(d_full, D, V)
                    )
                    # Phase-2 ablation: optional extra loss term
                    if args.loss_variant in ("seq_loss", "seq_boundary"):
                        loss = loss + 0.3 * seq_loss(out_dict, D, V,
                                                      gamma=args.seq_loss_gamma)
                    elif args.loss_variant == "slope_sup":
                        loss = loss + 0.3 * slope_sup_loss(out_dict, D, V)
                    elif args.loss_variant == "conf_aware":
                        loss = loss + 0.3 * conf_aware_loss(out_dict, D, V)
                    elif args.loss_variant == "edge_smooth":
                        loss = loss + 0.1 * edge_smooth_loss(d_full, L, V)
                    elif args.loss_variant == "boundary_focus":
                        loss = loss + 0.35 * boundary_focus_loss(d_full, D, V, L)
                    if args.loss_variant == "seq_boundary":
                        loss = loss + 0.25 * boundary_focus_loss(d_full, D, V, L)
                    # Expert-recommended: L_init supervision on the post-init
                    # disparity (d32) with smooth-L1. Implements IGEV's L_init.
                    if "d4_gev" in out_dict:
                        loss = loss + 0.15 * ms_l1(out_dict["d4_gev"], D, V, 4.0)
                    if "d_pre_guided" in out_dict and guided_active:
                        loss = loss + 0.40 * ms_l1(
                            out_dict["d_pre_guided"], D, V, 1.0)
                        loss = loss + 0.20 * guided_guard_loss(out_dict, D, V, L)
                    if args.init_loss_weight > 0:
                        loss = loss + args.init_loss_weight * ms_l1(
                            out_dict["d32"], D, V, 16.0)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
        else:
            if args.arch in ("costlookup", "tilegru"):
                out_dict = model(L, R, aux=True,
                                  with_iter_stages=(
                                      args.loss_variant == "seq_loss"))
            else:
                out_dict = run_model(L, R, aux=True,
                                     guided_active=guided_active)
            d_full = out_dict["d_final"]
            d_half = out_dict["d_half"]
            d4 = out_dict["d4"]
            d8 = out_dict["d8"]
            d16 = out_dict["d16"]
            if args.pure_l1:
                # Diagnostic: pure full-resolution L1 only.
                loss = ms_l1(d_full, D, V, 1.0)
            else:
                loss = (
                    1.0 * ms_l1(d_full, D, V, 1.0)
                    + 0.5 * ms_l1(d_half, D, V, 2.0)
                    + 0.3 * ms_l1(d4, D, V, 4.0)
                    + 0.2 * ms_l1(d8, D, V, 8.0)
                    + 0.1 * ms_l1(d16, D, V, 16.0)
                    + 0.5 * grad_consistency(d_full, D, V)
                    + 0.2 * bad1_hinge(d_full, D, V)
                )
                if args.loss_variant in ("seq_loss", "seq_boundary"):
                    loss = loss + 0.3 * seq_loss(out_dict, D, V,
                                                  gamma=args.seq_loss_gamma)
                elif args.loss_variant == "slope_sup":
                    loss = loss + 0.3 * slope_sup_loss(out_dict, D, V)
                elif args.loss_variant == "conf_aware":
                    loss = loss + 0.3 * conf_aware_loss(out_dict, D, V)
                elif args.loss_variant == "edge_smooth":
                    loss = loss + 0.1 * edge_smooth_loss(d_full, L, V)
                elif args.loss_variant == "boundary_focus":
                    loss = loss + 0.35 * boundary_focus_loss(d_full, D, V, L)
                if args.loss_variant == "seq_boundary":
                    loss = loss + 0.25 * boundary_focus_loss(d_full, D, V, L)
                if "d4_gev" in out_dict:
                    loss = loss + 0.15 * ms_l1(out_dict["d4_gev"], D, V, 4.0)
                if "d_pre_guided" in out_dict and guided_active:
                    loss = loss + 0.40 * ms_l1(
                        out_dict["d_pre_guided"], D, V, 1.0)
                    loss = loss + 0.20 * guided_guard_loss(out_dict, D, V, L)
                if args.init_loss_weight > 0:
                    loss = loss + args.init_loss_weight * ms_l1(
                        out_dict["d32"], D, V, 16.0)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

        if device == "cuda":
            peak_mem = max(peak_mem, torch.cuda.max_memory_allocated())

        if step == 1 or step % 50 == 0 or step == args.steps:
            with torch.no_grad():
                mtr = stereo_metrics(d_full, D, V)
            w.writerow([
                step, f"{loss.item():.5f}",
                f"{mtr['epe']:.5f}", f"{mtr['rmse']:.5f}",
                f"{mtr['bad_1.0']:.3f}", f"{mtr['bad_2.0']:.3f}",
                f"{mtr['d1_all']:.3f}",
                f"{opt.param_groups[0]['lr']:.6f}",
                f"{time.time() - t0:.2f}",
            ])
            cf.flush()
            print(f"  step {step:>4d}  loss={loss.item():.4f}  "
                  f"EPE={mtr['epe']:.3f}  bad1={mtr['bad_1.0']:.1f}%  "
                  f"bad2={mtr['bad_2.0']:.1f}%  "
                  f"({time.time()-t0:.1f}s, peak={peak_mem/1e9:.2f} GB)",
                  flush=True)

        now = time.time()
        if (now - last_viz_t) >= args.viz_interval_s or step == args.steps:
            last_viz_t = now
            if args.viz_rotate:
                viz_pair = (viz_pair0 + viz_count) % N
                viz_count += 1
            else:
                viz_pair = viz_pair0
            Lviz, Rviz, Dviz, Vviz = get_viz_batch(viz_pair)
            with torch.no_grad():
                pred_v = run_model(Lviz, Rviz, aux=False,
                                   guided_active=guided_active)
                mtr_v = stereo_metrics(pred_v, Dviz, Vviz)
            stats = {
                "step": f"{step}/{args.steps}",
                "loss": f"{loss.item():.4f}",
                "EPE": f"{mtr_v['epe']:.3f} px",
                "RMSE": f"{mtr_v['rmse']:.3f} px",
                "median": f"{mtr_v['median_ae']:.3f} px",
                "bad-0.5": f"{mtr_v['bad_0.5']:.1f}%",
                "bad-1.0": f"{mtr_v['bad_1.0']:.1f}%",
                "bad-2.0": f"{mtr_v['bad_2.0']:.1f}%",
                "bad-3.0": f"{mtr_v['bad_3.0']:.1f}%",
                "D1-all": f"{mtr_v['d1_all']:.1f}%",
                "lr": f"{opt.param_groups[0]['lr']:.4g}",
                "elapsed": f"{(now - t0):.0f} s",
                "peak": f"{peak_mem/1e9:.2f} GB",
                "arch": args.arch,
                "pair": str(viz_pair),
            }
            panel = render_panel(Lviz[0], Dviz[0, 0], pred_v[0, 0], stats)
            cv2.imwrite(str(viz_dir / f"step_{step:05d}.png"), panel)
            if show_ok:
                cv2.imshow(win_name, panel)
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    print("[viz] q pressed: aborting early.")
                    break

    elapsed = time.time() - t0
    cf.close()

    # Final eval — chunk through pairs so we never blow up GPU memory.
    model.eval()
    chunk = max(1, min(args.batch, 4))
    preds = []
    with torch.no_grad():
        for i in range(0, N, chunk):
            sl = slice(i, min(i + chunk, N))
            if cpu_residency and device == "cuda":
                Lc = Ls[sl].to(device, non_blocking=True)
                Rc = Rs[sl].to(device, non_blocking=True)
                preds.append(run_model(
                    Lc, Rc, aux=False,
                    guided_active=args.guided_start_step <= args.steps).cpu())
            else:
                preds.append(run_model(
                    Ls[sl], Rs[sl], aux=False,
                    guided_active=args.guided_start_step <= args.steps))
        all_pred = torch.cat(preds, dim=0)
        # Compute metrics on whichever device the GT lives on
        final_metrics = stereo_metrics(all_pred, Ds, valid)
        final_epe = final_metrics["epe"]

    # Inference latency benchmark. Wrapped so any failure here doesn't
    # destroy the meta.json — training metrics are the priority.
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True
    torch.use_deterministic_algorithms(False)
    inf_bench = {}
    arr = None
    try:
        if device == "cuda":
            torch.cuda.synchronize()
        with torch.no_grad():
            L1 = Ls[:1]; R1 = Rs[:1]
            # Under cpu_residency, Ls/Rs live on pinned CPU; the model is
            # on GPU. Move the single-pair benchmark inputs to the device.
            if cpu_residency and device == "cuda":
                L1 = L1.to(device, non_blocking=True)
                R1 = R1.to(device, non_blocking=True)
            for _ in range(10):
                _ = run_model(
                    L1, R1, aux=False,
                    guided_active=args.guided_start_step <= args.steps)
            if device == "cuda":
                torch.cuda.synchronize()
            per_call = []
            for _ in range(100):
                t = time.time()
                _ = run_model(
                    L1, R1, aux=False,
                    guided_active=args.guided_start_step <= args.steps)
                if device == "cuda":
                    torch.cuda.synchronize()
                per_call.append((time.time() - t) * 1000.0)
        arr = np.asarray(per_call)
    except Exception as e:
        print(f"[inference benchmark] failed: {type(e).__name__}: {e}",
              flush=True)
    if arr is not None:
        inf_bench = {
            "n_warmup": 10, "n_timed": 100,
            "input_size_HW": [args.height, args.width],
            "batch_size": 1,
            "ms_mean": round(float(arr.mean()), 3),
            "ms_std": round(float(arr.std()), 3),
            "ms_median": round(float(np.median(arr)), 3),
            "ms_p95": round(float(np.percentile(arr, 95)), 3),
            "fps_mean": round(1000.0 / float(arr.mean()), 2),
        }
        print(f"[inference] mean={inf_bench['ms_mean']} ms  "
              f"median={inf_bench['ms_median']} ms  "
              f"p95={inf_bench['ms_p95']} ms  fps={inf_bench['fps_mean']}")
    else:
        inf_bench = {"failed": True}

    # Save checkpoint so we can reuse the trained model later (compare against
    # other checkpoints, run the held-out eval again, fine-tune, etc.).
    ckpt_path = out / "checkpoint.pth"
    try:
        ckpt = {
            "model": model.state_dict(),
            "cfg": vars(cfg) if hasattr(cfg, "__dict__") else dict(cfg.__dict__),
            "args": vars(args),
            "params_train_M": round(n_train / 1e6, 4),
            "final_metrics_all": {k: float(v) for k, v in final_metrics.items()},
            "inference_bench": inf_bench,
            "tag": tag,
        }
        torch.save(ckpt, ckpt_path)
        print(f"[checkpoint] saved -> {ckpt_path} "
              f"({ckpt_path.stat().st_size/1e6:.1f} MB)")
    except Exception as e:
        print(f"[checkpoint] save failed: {type(e).__name__}: {e}")

    meta["finished_at"] = datetime.now().isoformat(timespec="seconds")
    meta["elapsed_s"] = round(elapsed, 2)
    meta["peak_gpu_mem_gb"] = round(peak_mem / 1e9, 3) if peak_mem else None
    meta["final_epe_all"] = round(final_epe, 4)
    meta["final_metrics_all"] = {k: round(v, 4) for k, v in final_metrics.items()}
    meta["inference_bench"] = inf_bench
    (out / "meta.json").write_text(json.dumps(meta, indent=2))

    # Curves plot.
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        steps, losses, epes, bad1s = [], [], [], []
        with open(csv_path) as cfr:
            r = csv.DictReader(cfr)
            for row in r:
                steps.append(int(row["step"]))
                losses.append(float(row["loss"]))
                epes.append(float(row["epe"]))
                bad1s.append(float(row["bad_1.0"]))
        fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
        axes[0].plot(steps, losses, "b-", label="loss", alpha=0.7)
        axes[0].set_xlabel("step"); axes[0].set_ylabel("loss", color="b")
        axes[0].set_yscale("log"); axes[0].grid(alpha=0.3)
        ax1b = axes[0].twinx()
        ax1b.plot(steps, epes, "r-", label="EPE (full)", alpha=0.7)
        ax1b.set_ylabel("EPE (px)", color="r")
        axes[1].plot(steps, bad1s, "g-", label="bad-1.0 %", alpha=0.7)
        axes[1].set_xlabel("step"); axes[1].set_ylabel("bad-1.0 (%)")
        axes[1].set_yscale("log"); axes[1].grid(alpha=0.3); axes[1].legend()
        fig.suptitle(f"{args.arch}: overfit on {args.n_pairs} SF pairs "
                     f"(final EPE = {final_epe:.3f}, "
                     f"bad-1.0 = {final_metrics['bad_1.0']:.1f}%)")
        fig.tight_layout()
        fig.savefig(out / "curve.png", dpi=120)
        plt.close(fig)
    except Exception as e:
        print(f"plot failed: {e}")

    # Per-variant README.
    readme = f"""# Overfit ablation: {args.arch}

**Run:** {meta['started_at']} → {meta['finished_at']}  ({meta['elapsed_s']:.1f} s)
**GPU:** {meta['gpu']}

## Configuration

- arch: `{args.arch}`
- steps: {args.steps}, lr: {args.lr}, batch: {args.batch}
- input: {args.height}×{args.width}, {args.n_pairs} fixed pairs (seed {args.seed})
- model: {meta['params_total_M']} M total, {meta['params_train_M']} M trainable
- encoder out_channels (1/2, 1/4, 1/8, 1/16): {meta['encoder_out_channels']}
- peak GPU memory: {meta['peak_gpu_mem_gb']} GB

## Result (full-res, all {args.n_pairs} pairs, eval mode)

| Metric | Value |
|---|---|
| EPE (px) | **{final_metrics['epe']:.4f}** |
| RMSE (px) | {final_metrics['rmse']:.4f} |
| Median AE (px) | {final_metrics['median_ae']:.4f} |
| bad-0.5 (%) | {final_metrics['bad_0.5']:.2f} |
| bad-1.0 (%) | **{final_metrics['bad_1.0']:.2f}** |
| bad-2.0 (%) | {final_metrics['bad_2.0']:.2f} |
| bad-3.0 (%) | {final_metrics['bad_3.0']:.2f} |
| D1-all (%) | {final_metrics['d1_all']:.2f} |

## Inference latency

- mean: **{inf_bench['ms_mean']} ms** ({inf_bench['fps_mean']} FPS)
- median: {inf_bench['ms_median']} ms
- p95: {inf_bench['ms_p95']} ms
"""
    (out / "README.md").write_text(readme)
    print(f"\n=== {args.arch}: final EPE = {final_epe:.4f} ===")
    print(f"results saved to {out}")
    if show_ok:
        try:
            cv2.destroyWindow(win_name)
        except cv2.error:
            pass

    # Auto-update the master experiments summary so it stays current.
    try:
        import subprocess
        subprocess.run(
            [sys.executable, str(ROOT / "model" / "scripts"
                                  / "build_experiments_summary.py")],
            check=False, capture_output=True)
    except Exception as e:
        print(f"experiments summary update failed: {e}")


if __name__ == "__main__":
    main()
