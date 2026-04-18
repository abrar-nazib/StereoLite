"""LiteFMStereo — Lightweight Foundation-Model Stereo.

Pipeline:
  GhostConv+SE encoder (shared) -> Ghost pyramid f4, f8, f16, f32
  DAv2-Small frozen (INT8 at deploy):
      raw block-11 features   -> 1/14 -> project 384 -> 64 -> interp 1/16
      DPT path_3 features     -> 1/8  -> project 64 -> 24  -> interp 1/8

  Fusion:
      f16_fused = Ghost f16  ⊕  DAv2 raw projection
      f8_fused  = Ghost f8   ⊕  DAv2 path_3 projection

  Coarse-to-fine disparity:
      1/32  head32 warp-regress (from zero init)
      1/16  head16 warp-regress on fused features
      1/8   cost volume (group-wise correlation + hourglass) — HARD init
            head8 refines CV output
      1/4   head4 refinement
      full  bilinear upsample

Training mode (`aux=True`): returns dict of all intermediate disparities
for multi-scale supervision. Required to prevent monocular-shortcut
collapse (where the network ignores matching and predicts the scene-mean
depth gradient).

Budget: ~1.72 M trainable + 24.8 M frozen DAv2. INT8 helper at deploy.
"""
from __future__ import annotations

from dataclasses import dataclass

import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from _blocks import (GhostConv, SqueezeExcitation, RepVGGBlock,
                     NeighborhoodAttention2d, _safe_gn)

from .dav2_backbone import DAv2SmallFrozen
from .cost_volume import GroupwiseCostVolume1D8, CascadeRefinementVolume


def _gn(ch: int, groups: int = 8) -> nn.GroupNorm:
    return _safe_gn(ch, groups)


class GhostStage(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, stride: int = 1):
        super().__init__()
        self.down = GhostConv(in_ch, out_ch, k=3, s=stride)
        self.refine = GhostConv(out_ch, out_ch, k=3, s=1)
        self.se = SqueezeExcitation(out_ch)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.se(self.refine(self.down(x)))


class TileFeatureEncoder(nn.Module):
    def __init__(self, base: int = 24):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, base, 7, stride=2, padding=3, bias=False),
            _gn(base),
            nn.SiLU(inplace=True),
        )
        self.s4 = GhostStage(base, 2 * base, stride=2)
        self.s8 = GhostStage(2 * base, 3 * base, stride=2)
        self.s16 = GhostStage(3 * base, 4 * base, stride=2)
        self.s32 = GhostStage(4 * base, 4 * base, stride=2)

    def forward(self, x: torch.Tensor):
        x = x / 255.0
        f2 = self.stem(x)            # 1/2
        f4 = self.s4(f2)
        f8 = self.s8(f4)
        f16 = self.s16(f8)
        f32 = self.s32(f16)
        return f2, f4, f8, f16, f32


def _horizontal_warp(fR: torch.Tensor, disp: torch.Tensor) -> torch.Tensor:
    B, _, H, W = fR.shape
    yy, xx = torch.meshgrid(
        torch.arange(H, device=fR.device, dtype=fR.dtype),
        torch.arange(W, device=fR.device, dtype=fR.dtype),
        indexing="ij",
    )
    gx = (xx.view(1, H, W) - disp.squeeze(1)) / max(W - 1, 1) * 2 - 1
    gy = (yy.view(1, H, W) / max(H - 1, 1) * 2 - 1).expand_as(gx)
    grid = torch.stack([gx, gy], dim=-1)
    return F.grid_sample(fR, grid, align_corners=True, padding_mode="border")


class FullResRefinement(nn.Module):
    """Tiny warp-regress refinement at full output resolution. Takes the
    upsampled disparity plus left/right image features, computes a warped
    right feature, and outputs a residual correction. ~12 k params.

    Operates in PyTorch native FP32 (not AMP) to keep residuals stable,
    but the outer model can still run under autocast; this layer will
    promote internally as needed.
    """

    def __init__(self, hidden: int = 16):
        super().__init__()
        self.img_feat = nn.Sequential(
            nn.Conv2d(3, hidden, 3, padding=1, bias=False),
            _gn(hidden),
            nn.SiLU(inplace=True),
        )
        self.refine = nn.Sequential(
            nn.Conv2d(2 * hidden + 1, hidden, 3, padding=1, bias=False),
            _gn(hidden),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden, hidden, 3, padding=1, bias=False),
            _gn(hidden),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden, 1, 3, padding=1, bias=True),
        )

    def forward(self, left: torch.Tensor, right: torch.Tensor,
                disp: torch.Tensor) -> torch.Tensor:
        fL = self.img_feat(left / 255.0)
        fR = self.img_feat(right / 255.0)
        fR_w = _horizontal_warp(fR, disp)
        x = torch.cat([fL, fR_w, disp], dim=1)
        return disp + self.refine(x)


class ConvexUpsample(nn.Module):
    """RAFT-style learned upsample. Predicts a 9-neighbor soft mask per
    subpixel position, then upsamples disparity via weighted sum of 3x3
    neighborhoods. Much sharper than bilinear at edges.

    In:  disp (B, 1, H, W) at low-res, feat (B, C, H, W) same spatial.
    Out: disp at (B, 1, scale*H, scale*W).
    """

    def __init__(self, feat_ch: int, scale: int = 4, hidden: int = 64):
        super().__init__()
        self.scale = scale
        self.mask = nn.Sequential(
            nn.Conv2d(feat_ch, hidden, 3, padding=1),
            nn.SiLU(inplace=True),
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


class HypothesisHead(nn.Module):
    def __init__(self, in_ch: int, hidden: int = 48):
        super().__init__()
        self.rep1 = RepVGGBlock(2 * in_ch, hidden)
        self.rep2 = RepVGGBlock(hidden, hidden)
        self.head = nn.Conv2d(hidden, 4, 1)

    def forward(self, fL: torch.Tensor, fR: torch.Tensor,
                d_init: torch.Tensor | None):
        if d_init is None:
            d_init = torch.zeros(fL.shape[0], 1, *fL.shape[-2:],
                                 device=fL.device, dtype=fL.dtype)
        fR_w = _horizontal_warp(fR, d_init)
        x = torch.cat([fL, fR_w], dim=1)
        x = self.rep1(x)
        x = self.rep2(x)
        out = self.head(x)
        d_abs = F.softplus(out[:, 0:1] + d_init)
        slope = out[:, 1:3]
        conf = torch.sigmoid(out[:, 3:4])
        return d_abs, torch.cat([slope, conf], dim=1)


@dataclass
class LiteConfig:
    base_ch: int = 32
    use_dav2: bool = True
    dav2_raw_proj_ch: int = 64        # raw block-11 projection for 1/16 fusion
    dav2_path3_proj_ch: int = 24      # DPT path_3 projection for 1/8 fusion
    cv_max_disp: int = 24
    cv_groups: int = 8
    cv_agg_ch: int = 48
    cascade_half_range: int = 8       # ±8 disparity search at 1/4
    cascade_agg_ch: int = 24
    full_res_refine_hidden: int = 16


class LiteFMStereo(nn.Module):
    def __init__(self, cfg: LiteConfig | None = None):
        super().__init__()
        self.cfg = cfg or LiteConfig()
        b = self.cfg.base_ch
        self.fnet = TileFeatureEncoder(base=b)

        if self.cfg.use_dav2:
            self.dav2 = DAv2SmallFrozen()
            self.dav2_proj16 = nn.Conv2d(
                self.dav2.embed_dim, self.cfg.dav2_raw_proj_ch, 1)
            self.dav2_proj8 = nn.Conv2d(
                self.dav2.dpt_ch, self.cfg.dav2_path3_proj_ch, 1)
            in_ch_16 = 4 * b + self.cfg.dav2_raw_proj_ch
            in_ch_8 = 3 * b + self.cfg.dav2_path3_proj_ch
        else:
            self.dav2 = None
            in_ch_16 = 4 * b
            in_ch_8 = 3 * b

        self.head32 = HypothesisHead(4 * b)
        self.head16 = HypothesisHead(in_ch_16)
        self.head8 = HypothesisHead(in_ch_8)
        self.head4 = HypothesisHead(2 * b)
        # 1/4 -> 1/2 learned upsample, then refine at 1/2 with stem features
        self.up4_to_2 = ConvexUpsample(feat_ch=2 * b, scale=2)
        self.head2 = HypothesisHead(b)          # refinement at 1/2
        # 1/2 -> full learned upsample (preserves sub-pixel edges)
        self.up2_to_full = ConvexUpsample(feat_ch=b, scale=2)
        # Full-resolution residual refinement with explicit warp-matching
        self.refine_full = FullResRefinement(
            hidden=self.cfg.full_res_refine_hidden)

        # Cost volume at 1/8 on fused features
        groups = self.cfg.cv_groups
        if in_ch_8 % groups != 0:
            for g in range(groups, 0, -1):
                if in_ch_8 % g == 0:
                    groups = g
                    break
        self.cv_groups = groups
        self.costvol = GroupwiseCostVolume1D8(
            feat_ch=in_ch_8, max_disp=self.cfg.cv_max_disp,
            groups=groups, agg_ch=self.cfg.cv_agg_ch)

        # Cascade refinement at 1/4 — narrow-range CV that commits to
        # sub-8px disparities given a coarse 1/8 estimate.
        f4_ch = 2 * b
        cas_groups = self.cfg.cv_groups
        if f4_ch % cas_groups != 0:
            for g in range(cas_groups, 0, -1):
                if f4_ch % g == 0:
                    cas_groups = g
                    break
        self.cascade_cv = CascadeRefinementVolume(
            feat_ch=f4_ch, half_range=self.cfg.cascade_half_range,
            groups=cas_groups, agg_ch=self.cfg.cascade_agg_ch)

    def forward(self, left: torch.Tensor, right: torch.Tensor,
                aux: bool = False):
        # Ghost features for L and R. Batch L+R into a single forward pass
        # where possible to halve DAv2 compute.
        B = left.shape[0]
        fL2, fL4, fL8, fL16, fL32 = self.fnet(left)
        fR2, fR4, fR8, fR16, fR32 = self.fnet(right)

        if self.dav2 is not None:
            # One DAv2 forward on the stacked L|R batch, then split.
            lr = torch.cat([left, right], dim=0)
            out = self.dav2(lr)
            raw_lr = out["raw"]
            p3_lr = out["path_3"]
            # Project once, split after projection.
            raw16 = self.dav2_proj16(
                F.interpolate(raw_lr, size=fL16.shape[-2:],
                              mode="bilinear", align_corners=False))
            p3_8 = self.dav2_proj8(
                F.interpolate(p3_lr, size=fL8.shape[-2:],
                              mode="bilinear", align_corners=False))
            dL16, dR16 = raw16[:B], raw16[B:]
            dL8, dR8 = p3_8[:B], p3_8[B:]
            fL16_fused = torch.cat([fL16, dL16], dim=1)
            fR16_fused = torch.cat([fR16, dR16], dim=1)
            fL8_fused = torch.cat([fL8, dL8], dim=1)
            fR8_fused = torch.cat([fR8, dR8], dim=1)
        else:
            fL16_fused, fR16_fused = fL16, fR16
            fL8_fused, fR8_fused = fL8, fR8

        # 1/32 coarse
        d32, _ = self.head32(fL32, fR32, None)

        # 1/16 warp-regress on fused features
        d16_init = F.interpolate(d32 * 2, size=fL16.shape[-2:],
                                  mode="bilinear", align_corners=True)
        d16, _ = self.head16(fL16_fused, fR16_fused, d16_init)

        # 1/8: cost volume HARD init, then head8 refinement
        d8_cv = self.costvol(fL8_fused, fR8_fused)
        d8, _ = self.head8(fL8_fused, fR8_fused, d8_cv)

        # 1/4 cascade refinement volume — commits to sub-grid disparities
        d4_init = F.interpolate(d8 * 2, size=fL4.shape[-2:],
                                 mode="bilinear", align_corners=True)
        d4_cas = self.cascade_cv(fL4, fR4, d4_init)
        d4, _ = self.head4(fL4, fR4, d4_cas)

        # 1/4 -> 1/2 convex upsample, then refine at 1/2 with stem features
        d2_init = self.up4_to_2(d4, fL4)
        d2, _ = self.head2(fL2, fR2, d2_init)

        # 1/2 -> full convex upsample (sharp boundaries)
        d_full_raw = self.up2_to_full(d2, fL2)
        if d_full_raw.shape[-2:] != left.shape[-2:]:
            d_full_raw = F.interpolate(d_full_raw, size=left.shape[-2:],
                                        mode="bilinear", align_corners=True)
        # Full-resolution residual refinement with explicit matching
        d_final = self.refine_full(left, right, d_full_raw)

        if aux:
            return {
                "d_final": d_final,
                "d_full_raw": d_full_raw,
                "d2": d2,
                "d4": d4,
                "d4_cas": d4_cas,
                "d8": d8,
                "d8_cv": d8_cv,
                "d16": d16,
                "d32": d32,
            }
        return d_final


# Backward-compat alias
TileHypothesisStereo = LiteFMStereo
TileConfig = LiteConfig
