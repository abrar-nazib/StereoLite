"""StereoLite_costlookup — current chassis + per-iter local cost lookup.

Adds RAFT-style cost-volume lookup to TileRefine: at each iter, samples
fR at (current d) ± half_range and concatenates the resulting groupwise
correlation slice to the standard TileRefine input.

Two phases controlled by `extend_to_full`:
  - False: TileRefine runs at 1/16, 1/8, 1/4 + ConvexUpsample 1/4 → 1/2 → full
           (matches the locked chassis, just adds cost lookup inside TileRefine)
  - True : TileRefine runs at 1/16, 1/8, 1/4, 1/2 + plane-eq upsample 1/2 → full
           (no ConvexUpsample anywhere)

Backbone: "ghost" or "yolo26n" (YOLO26s also accepted).
"""
from __future__ import annotations

from dataclasses import dataclass
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from _blocks import (GhostConv, SqueezeExcitation, _safe_gn)
from _wideners import build_widener, WIDENER_TARGETS
from StereoLite_yolo.yolo_encoder import YoloTruncatedEncoder

from .tile_propagate import (TileState, TileInit, TileRefineCostLookup,
                              TileUpsample)


def _gn(ch: int, groups: int = 8) -> nn.GroupNorm:
    return _safe_gn(ch, groups)


class GhostStage(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, stride: int = 1):
        super().__init__()
        self.down = GhostConv(in_ch, out_ch, k=3, s=stride)
        self.refine = GhostConv(out_ch, out_ch, k=3, s=1)
        self.se = SqueezeExcitation(out_ch)

    def forward(self, x):
        return self.se(self.refine(self.down(x)))


class TileFeatureEncoder(nn.Module):
    def __init__(self, base: int = 24):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, base, 7, stride=2, padding=3, bias=False),
            _gn(base), nn.SiLU(inplace=True),
        )
        self.s4 = GhostStage(base, 2 * base, stride=2)
        self.s8 = GhostStage(2 * base, 3 * base, stride=2)
        self.s16 = GhostStage(3 * base, 4 * base, stride=2)
        self.out_channels = (base, 2 * base, 3 * base, 4 * base)

    def forward(self, x):
        x = x / 255.0
        f2 = self.stem(x)
        f4 = self.s4(f2)
        f8 = self.s8(f4)
        f16 = self.s16(f8)
        return f2, f4, f8, f16


class ConvexUpsample(nn.Module):
    def __init__(self, feat_ch: int, scale: int = 2, hidden: int = 48):
        super().__init__()
        self.scale = scale
        self.mask = nn.Sequential(
            nn.Conv2d(feat_ch, hidden, 3, padding=1, bias=False),
            _gn(hidden), nn.SiLU(inplace=True),
            nn.Conv2d(hidden, 9 * scale * scale, 1),
        )

    def forward(self, disp, feat):
        B, _, H, W = disp.shape
        s = self.scale
        m = self.mask(feat).view(B, 1, 9, s, s, H, W).softmax(dim=2)
        up = F.unfold(disp * s, kernel_size=3, padding=1)
        up = up.view(B, 1, 9, 1, 1, H, W)
        out = (m * up).sum(dim=2)
        out = out.permute(0, 1, 4, 2, 5, 3).contiguous()
        return out.view(B, 1, s * H, s * W)


@dataclass
class StereoLiteConfig:
    base_ch: int = 24
    tile_feat_ch: int = 16
    iters_16: int = 2
    iters_8: int = 3
    iters_4: int = 3
    iters_2: int = 2                # only used if extend_to_full=True
    init_max_disp: int = 24
    init_groups: int = 8
    refine_hidden: int = 48
    cost_lookup_half_range: int = 2
    cost_lookup_groups: int = 8
    backbone: str = "ghost"
    backbone_pretrained: bool = True
    extend_to_full: bool = False
    # Optional feature-widener applied to the encoder outputs.
    # See model/designs/_wideners.py for the catalogue. None = no widener.
    widener: str | None = None
    # ---- Phase-2 ablation knobs (default off; baseline = ghostconv winner) ----
    slope_aware_warp: bool = False        # A2 — slope-corrected fR sample
    selective_gate: bool = False          # A5 — per-pixel "stop refining" gate
    cascade_cv_4: bool = False            # A3 — narrow-range 3D-agg CV at 1/4
    context_branch: bool = False          # A1 — small parallel encoder fed
                                           # into TileRefine alongside fL/fR
    context_branch_ch: int = 24           # base channels for the context encoder
    # ---- Expert-recommended warm-start init regression (IGEV-style) ----
    init_regress: bool = False            # add APC + 2-layer head to TileInit;
                                           # supervise d_init with smooth-L1 in the
                                           # trainer (--init_loss_weight > 0).


class StereoLite(nn.Module):
    def __init__(self, cfg: StereoLiteConfig | None = None):
        super().__init__()
        self.cfg = cfg or StereoLiteConfig()
        b = self.cfg.base_ch
        if self.cfg.backbone in ("yolo26n", "yolo26s"):
            self.fnet = YoloTruncatedEncoder(variant=self.cfg.backbone)
        else:
            self.fnet = TileFeatureEncoder(base=b)

        # Optional feature widener applied to encoder outputs.
        # gn_replace also modifies self.fnet in place.
        if self.cfg.widener and self.cfg.widener != "none":
            self.widener = build_widener(self.cfg.widener,
                                          self.fnet.out_channels,
                                          encoder=self.fnet)
            ch2, ch4, ch8, ch16 = self.widener.out_channels
        else:
            self.widener = None
            ch2, ch4, ch8, ch16 = self.fnet.out_channels

        g = self.cfg.init_groups
        while ch16 % g != 0 and g > 1:
            g -= 1
        self.init_tile = TileInit(feat_ch=ch16,
                                   max_disp=self.cfg.init_max_disp,
                                   groups=g,
                                   feat_out=self.cfg.tile_feat_ch,
                                   regress=self.cfg.init_regress)

        # Context branch (A1) — small parallel encoder mimicking the matching
        # encoder's strides. Provides per-stage context features that get
        # concatenated into TileRefine's input alongside fL/fR_warp.
        if self.cfg.context_branch:
            cb = self.cfg.context_branch_ch
            # Reuse TileFeatureEncoder for the context encoder; it already
            # gives 4 stages at strides (2, 4, 8, 16) with channels
            # (cb, 2*cb, 3*cb, 4*cb). Tiny by construction.
            self.cnet = TileFeatureEncoder(base=cb)
            ctx2, ctx4, ctx8, ctx16 = self.cnet.out_channels
            ctx_chs = (ctx2, ctx4, ctx8, ctx16)
        else:
            self.cnet = None
            ctx_chs = (0, 0, 0, 0)

        def make_refine(feat_ch, ctx_ch=0):
            return TileRefineCostLookup(
                feat_ch=feat_ch, tile_feat_ch=self.cfg.tile_feat_ch,
                hidden=self.cfg.refine_hidden,
                half_range=self.cfg.cost_lookup_half_range,
                groups=self.cfg.cost_lookup_groups,
                slope_aware_warp=self.cfg.slope_aware_warp,
                selective_gate=self.cfg.selective_gate,
                context_ch=ctx_ch)

        self.refine_16 = make_refine(ch16, ctx_chs[3])
        self.refine_8 = make_refine(ch8, ctx_chs[2])
        self.refine_4 = make_refine(ch4, ctx_chs[1])
        self.up_16_to_8 = TileUpsample(scale_factor=2)
        self.up_8_to_4 = TileUpsample(scale_factor=2)

        # A3 cascade_cv_4 — narrow-range 3D-aggregated cost volume between
        # iters at 1/4 (BGNet/CFNet style). Refines the disparity field
        # mid-iteration with a denser matching signal at fine scale.
        if self.cfg.cascade_cv_4:
            from StereoLite.cost_volume import CascadeRefinementVolume
            self.cascade_4 = CascadeRefinementVolume(
                feat_ch=ch4, half_range=4, groups=8, agg_ch=24)
        else:
            self.cascade_4 = None

        if self.cfg.extend_to_full:
            self.refine_2 = make_refine(ch2, ctx_chs[0])
            self.up_4_to_2 = TileUpsample(scale_factor=2)
            self.up_2_to_1 = TileUpsample(scale_factor=2)
            self.up_final_4_to_2 = None
            self.up_final_2_to_1 = None
        else:
            self.refine_2 = None
            self.up_4_to_2 = None
            self.up_2_to_1 = None
            self.up_final_4_to_2 = ConvexUpsample(feat_ch=ch4, scale=2)
            self.up_final_2_to_1 = ConvexUpsample(feat_ch=ch2, scale=2)

    def forward(self, left, right, aux: bool = False,
                with_iter_stages: bool = False):
        feats = self.fnet(torch.cat([left, right], dim=0))
        if self.widener is not None:
            feats = self.widener(feats)
        fL2,  fR2  = feats[0].chunk(2, dim=0)
        fL4,  fR4  = feats[1].chunk(2, dim=0)
        fL8,  fR8  = feats[2].chunk(2, dim=0)
        fL16, fR16 = feats[3].chunk(2, dim=0)

        # Context branch (A1) — runs only on the LEFT image, in parallel.
        if self.cnet is not None:
            cfeats = self.cnet(left)
            ctx2, ctx4, ctx8, ctx16 = cfeats
        else:
            ctx2 = ctx4 = ctx8 = ctx16 = None

        tile = self.init_tile(fL16, fR16)
        t16_stages = [tile]
        for _ in range(self.cfg.iters_16):
            tile = self.refine_16(tile, fL16, fR16, context=ctx16)
            t16_stages.append(tile)

        tile = self.up_16_to_8(tile, target_hw=fL8.shape[-2:])
        t8_stages = [tile]
        for _ in range(self.cfg.iters_8):
            tile = self.refine_8(tile, fL8, fR8, context=ctx8)
            t8_stages.append(tile)

        tile = self.up_8_to_4(tile, target_hw=fL4.shape[-2:])
        t4_stages = [tile]
        for i in range(self.cfg.iters_4):
            # A3: insert cascade CV refinement between iter 1 and iter 2
            if self.cascade_4 is not None and i == 1:
                d_ref = self.cascade_4(fL4, fR4, tile.d)
                tile = TileState(d=d_ref, sx=tile.sx, sy=tile.sy,
                                  feat=tile.feat, conf=tile.conf)
            tile = self.refine_4(tile, fL4, fR4, context=ctx4)
            t4_stages.append(tile)

        if self.cfg.extend_to_full:
            tile = self.up_4_to_2(tile, target_hw=fL2.shape[-2:])
            t2_stages = [tile]
            for _ in range(self.cfg.iters_2):
                tile = self.refine_2(tile, fL2, fR2, context=ctx2)
                t2_stages.append(tile)
            d_half = tile.d
            tile_full = self.up_2_to_1(tile, target_hw=left.shape[-2:])
            d_full = tile_full.d
        else:
            d_half = self.up_final_4_to_2(tile.d, fL4)
            d_full = self.up_final_2_to_1(d_half, fL2)

        if d_full.shape[-2:] != left.shape[-2:]:
            d_full = F.interpolate(d_full, size=left.shape[-2:],
                                    mode="bilinear", align_corners=True)

        if aux:
            d = {
                "d_final": d_full,
                "d_half": d_half,
                "d4": t4_stages[-1].d,
                "d8": t8_stages[-1].d,
                "d8_cv": t16_stages[-1].d,
                "d16": t16_stages[-1].d,
                "d32": t16_stages[0].d,
            }
            if with_iter_stages:
                # Per-iter outputs for RAFT-style γ-weighted sequence loss.
                # Only populated on demand because keeping references to
                # every iteration's `tile.d` retains activation memory.
                t2_stages_list = locals().get("t2_stages", [])
                d["iter_stages_16"] = [t.d for t in t16_stages]
                d["iter_stages_8"]  = [t.d for t in t8_stages]
                d["iter_stages_4"]  = [t.d for t in t4_stages]
                d["iter_stages_2"]  = [t.d for t in t2_stages_list]
            return d
        return d_full
