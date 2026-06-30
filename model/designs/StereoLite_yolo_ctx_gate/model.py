"""StereoLite_yolo_ctx_gate — YOLO26 matching encoder + dedicated context encoder.

The matching encoder (YOLO26s truncated at 1/16) is unchanged from the
`StereoLite_yolo/` sibling. The novelty is the bottom stream in the
RAFT-Stereo Fig 1 architecture: a separate, shallower encoder that runs
on the LEFT image only and produces per-pixel context features that
feed the persistent GRU hidden state across iterations.

Pipeline:
  Left image ──► ContextEncoder ──► ctx4  (B, ctx_ch, H/4,  W/4)
                              │
                              ├─► upsample to 1/16 → init h (tile.feat)
                              ├─► upsample to 1/8  → per-iter input ctx8
                              └─► upsample to 1/4  → per-iter input ctx4

  (L, R) ──► YoloTruncatedEncoder ──► fL/R at 1/2, 1/4, 1/8, 1/16

  Cost-volume init at 1/16 → h overwritten with ctx16 → iterate with
  TileRefineCtx(fL16, fR16, ctx=ctx16) × N16.

  Plane upsample 1/16 → 1/8 → iterate with TileRefineCtx(..., ctx=ctx8).
  Plane upsample 1/8  → 1/4 → iterate with TileRefineCtx(..., ctx=ctx4).

  Final: ConvexUpsample 1/4 → 1/2 → full, exactly as `StereoLite_yolo`.

Budget target: ~2-3 M trainable (matches the mid-tier envelope).
YOLO26s encoder ≈ 1.23 M, ContextEncoder (small) ≈ 0.05-0.10 M,
init + GRU update heads + convex upsample ≈ 0.5-0.8 M.
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

from StereoLite_yolo.yolo_encoder import YoloTruncatedEncoder

from .tile_propagate import (TileState, TileInit, TileRefineCtx, TileUpsample)


def _gn(ch: int, groups: int = 8) -> nn.GroupNorm:
    return _safe_gn(ch, groups)


def _to_unit_image(x: torch.Tensor) -> torch.Tensor:
    return x / 255.0 if x.detach().amax() > 2.0 else x


# ----------------------------------------------------------------------- #
# Context encoder (RAFT-Stereo Fig 1, bottom stream)
# ----------------------------------------------------------------------- #
class ContextEncoder(nn.Module):
    """Lightweight GhostConv encoder on the LEFT image, output at 1/4.

    Returns (B, ctx_ch, H/4, W/4). ~50-100 k params depending on
    base/out channel choice. The model bilinear-upsamples these features
    to whatever scale the GRU needs (1/16, 1/8, 1/4).

    Why 1/4 (not 1/16): context in RAFT-Stereo is at 1/4 because the
    hidden state must capture long-range image structure (edges, large
    homogeneous regions, occluding contours) and downsampling past 1/4
    loses that. Bilinear upsample from 1/4 to 1/16 is cheap.
    """

    def __init__(self, base: int = 24, out_ch: int = 32):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, base, 7, stride=2, padding=3, bias=False),
            _gn(base), nn.SiLU(inplace=True),
        )
        # 1/4: GhostConv stage
        self.s4 = nn.Sequential(
            GhostConv(base, 2 * base, k=3, s=2),
            SqueezeExcitation(2 * base),
            GhostConv(2 * base, 2 * base, k=3, s=1),
            SqueezeExcitation(2 * base),
        )
        # Project to out_ch
        self.proj = nn.Sequential(
            nn.Conv2d(2 * base, out_ch, 1, bias=False),
            _gn(out_ch), nn.SiLU(inplace=True),
        )
        self.out_channels = out_ch

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Accept either [0, 1] benchmark tensors or [0, 255] raw tensors.
        x = _to_unit_image(x)
        x = self.stem(x)         # 1/2
        x = self.s4(x)           # 1/4
        x = self.proj(x)         # 1/4, out_ch
        return x


# ----------------------------------------------------------------------- #
# Convex upsample (copied verbatim from StereoLite_yolo)
# ----------------------------------------------------------------------- #
class ConvexUpsample(nn.Module):
    def __init__(self, feat_ch: int, scale: int = 2, hidden: int = 48):
        super().__init__()
        self.scale = scale
        self.mask = nn.Sequential(
            nn.Conv2d(feat_ch, hidden, 3, padding=1, bias=False),
            _gn(hidden), nn.SiLU(inplace=True),
            nn.Conv2d(hidden, 9 * scale * scale, 1),
        )

    def forward(self, disp: torch.Tensor, feat: torch.Tensor) -> torch.Tensor:
        B, _, H, W = disp.shape
        s = self.scale
        m = self.mask(feat).view(B, 1, 9, s, s, H, W).softmax(dim=2)
        up = F.unfold(disp * s, kernel_size=3, padding=1)
        up = up.view(B, 1, 9, 1, 1, H, W)
        out = (m * up).sum(dim=2)
        out = out.permute(0, 1, 4, 2, 5, 3).contiguous()
        return out.view(B, 1, s * H, s * W)


@dataclass
class StereoLiteYoloCtxGateConfig:
    base_ch: int = 24
    tile_feat_ch: int = 32
    ctx_ch: int = 32
    iters_16: int = 2
    iters_8: int = 3
    iters_4: int = 3
    init_max_disp: int = 24
    init_groups: int = 8
    refine_hidden: int = 48
    cost_half_range: int = 2
    backbone: str = "yolo26s"     # mid-tier default; can switch to yolo26n
    backbone_pretrained: bool = True


class StereoLiteYoloCtxGate(nn.Module):
    def __init__(self, cfg: StereoLiteYoloCtxGateConfig | None = None):
        super().__init__()
        self.cfg = cfg or StereoLiteYoloCtxGateConfig()

        # Matching encoder (left+right, shared)
        self.fnet = YoloTruncatedEncoder(variant=self.cfg.backbone)
        ch2, ch4, ch8, ch16 = self.fnet.out_channels

        # Context encoder (LEFT image only)
        self.ctxnet = ContextEncoder(base=self.cfg.base_ch,
                                      out_ch=self.cfg.ctx_ch)

        # Cost-volume init at 1/16 (h emitted as zeros, model replaces)
        g = self.cfg.init_groups
        while ch16 % g != 0 and g > 1:
            g -= 1
        self.init_tile = TileInit(feat_ch=ch16,
                                   max_disp=self.cfg.init_max_disp,
                                   groups=g,
                                   feat_out=self.cfg.tile_feat_ch)

        # Per-scale refine heads — one instance per scale because channel
        # counts differ; weights are not shared.
        self.refine_16 = TileRefineCtx(
            feat_ch=ch16, tile_feat_ch=self.cfg.tile_feat_ch,
            hidden=self.cfg.refine_hidden,
            half_range=self.cfg.cost_half_range, groups=g,
            ctx_ch=self.cfg.ctx_ch)
        self.refine_8 = TileRefineCtx(
            feat_ch=ch8, tile_feat_ch=self.cfg.tile_feat_ch,
            hidden=self.cfg.refine_hidden,
            half_range=self.cfg.cost_half_range, groups=g,
            ctx_ch=self.cfg.ctx_ch)
        self.refine_4 = TileRefineCtx(
            feat_ch=ch4, tile_feat_ch=self.cfg.tile_feat_ch,
            hidden=self.cfg.refine_hidden,
            half_range=self.cfg.cost_half_range, groups=g,
            ctx_ch=self.cfg.ctx_ch)

        # Plane upsamples (no trainable weights)
        self.up_16_to_8 = TileUpsample(scale_factor=2)
        self.up_8_to_4 = TileUpsample(scale_factor=2)

        # Final learned 4x upsample 1/4 → full, two 2x convex steps
        self.up_final_4_to_2 = ConvexUpsample(feat_ch=ch4, scale=2)
        self.up_final_2_to_1 = ConvexUpsample(feat_ch=ch2, scale=2)

    def forward(self, left: torch.Tensor, right: torch.Tensor,
                aux: bool = False):
        left_in = _to_unit_image(left)
        right_in = _to_unit_image(right)
        # --- Matching features (L+R shared encoder, batched) ---
        feats = self.fnet(torch.cat([left_in, right_in], dim=0))
        fL2,  fR2  = feats[0].chunk(2, dim=0)
        fL4,  fR4  = feats[1].chunk(2, dim=0)
        fL8,  fR8  = feats[2].chunk(2, dim=0)
        fL16, fR16 = feats[3].chunk(2, dim=0)

        # --- Context features (LEFT image only) at 1/4 ---
        ctx4 = self.ctxnet(left_in)
        # Bilinear-upsample context to each scale the GRU needs it at.
        ctx16 = F.interpolate(ctx4, size=fL16.shape[-2:], mode="bilinear",
                              align_corners=False)
        ctx8  = F.interpolate(ctx4, size=fL8.shape[-2:],  mode="bilinear",
                              align_corners=False)

        # --- 1/16: init h from ctx16, then iterate ---
        tile = self.init_tile(fL16, fR16)
        tile = TileState(d=tile.d, sx=tile.sx, sy=tile.sy,
                          feat=ctx16, conf=tile.conf)
        t16_stages = [tile]
        for _ in range(self.cfg.iters_16):
            tile = self.refine_16(tile, fL16, fR16, ctx16)
            t16_stages.append(tile)

        # --- 1/16 → 1/8 via plane equation, then iterate ---
        tile = self.up_16_to_8(tile, target_hw=fL8.shape[-2:])
        t8_stages = [tile]
        for _ in range(self.cfg.iters_8):
            tile = self.refine_8(tile, fL8, fR8, ctx8)
            t8_stages.append(tile)

        # --- 1/8 → 1/4, then iterate ---
        tile = self.up_8_to_4(tile, target_hw=fL4.shape[-2:])
        t4_stages = [tile]
        for _ in range(self.cfg.iters_4):
            tile = self.refine_4(tile, fL4, fR4, ctx4)
            t4_stages.append(tile)

        # --- Final convex upsample 1/4 → 1/2 → full ---
        d_half = self.up_final_4_to_2(tile.d, fL4)
        d_full = self.up_final_2_to_1(d_half, fL2)
        if d_full.shape[-2:] != left.shape[-2:]:
            d_full = F.interpolate(d_full, size=left.shape[-2:],
                                    mode="bilinear", align_corners=True)

        if aux:
            return {
                "d_final": d_full,
                "d_half": d_half,
                "d4": tile.d,
                "d8": t8_stages[-1].d,
                "d8_cv": t16_stages[-1].d,
                "d16": t16_stages[-1].d,
                "d32": t16_stages[0].d,
            }
        return d_full


# Compatibility aliases for existing yolo_ctx import patterns.
StereoLiteYoloCtx = StereoLiteYoloCtxGate
StereoLiteYoloCtxConfig = StereoLiteYoloCtxGateConfig
