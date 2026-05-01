"""StereoLite — Variant A (iterative all the way, no ConvexUpsample).

Architecture experiment (vs the StereoLite baseline):
  - Removes both ConvexUpsample modules entirely.
  - Adds plane upsample 1/4 → 1/2 + TileRefine iterations at 1/2 resolution.
  - Final 1/2 → full uses plane equation upsample only (no full-res
    refinement because the encoder doesn't produce full-res features).

Pipeline:
  Input (L, R) ──► encoder ──► f2, f4, f8, f16

  1/16: TileInit + TileRefine × 2
  1/16 → 1/8 plane upsample, TileRefine × 3
  1/8 → 1/4 plane upsample, TileRefine × 3
  1/4 → 1/2 plane upsample, TileRefine × 2   (NEW)
  1/2 → full plane upsample only             (NEW; no fL1, so no refine)

This variant tests whether HITNet's "iterate at every scale" approach
produces sharper output than RAFT-style "iterate at coarse, then learned
mask upsample". The point of comparison is StereoLite_yolo with
backbone=ghost (the baseline) and StereoLite_v2_hitnet (HITNet's exact
propagation block).
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

from .tile_propagate import (TileState, TileInit, TileRefine, TileUpsample)


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
    """Returns features at 1/2, 1/4, 1/8, 1/16."""

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
        self.out_channels = (base, 2 * base, 3 * base, 4 * base)

    def forward(self, x: torch.Tensor):
        x = x / 255.0
        f2 = self.stem(x)
        f4 = self.s4(f2)
        f8 = self.s8(f4)
        f16 = self.s16(f8)
        return f2, f4, f8, f16


@dataclass
class StereoLiteConfig:
    base_ch: int = 24
    tile_feat_ch: int = 16
    iters_16: int = 2
    iters_8: int = 3
    iters_4: int = 3
    iters_2: int = 2
    init_max_disp: int = 24
    init_groups: int = 8
    refine_hidden: int = 48
    backbone: str = "ghost"
    backbone_pretrained: bool = True


class StereoLite(nn.Module):
    def __init__(self, cfg: StereoLiteConfig | None = None):
        super().__init__()
        self.cfg = cfg or StereoLiteConfig()
        b = self.cfg.base_ch
        # Only ghost is supported here; the experiment fixes the encoder
        # to keep variation strictly to the refinement+upsample pipeline.
        assert self.cfg.backbone == "ghost", \
            "Variant A is fixed to ghost encoder for fair architecture A/B"
        self.fnet = TileFeatureEncoder(base=b)

        ch2, ch4, ch8, ch16 = self.fnet.out_channels

        g = self.cfg.init_groups
        while ch16 % g != 0 and g > 1:
            g -= 1
        self.init_tile = TileInit(feat_ch=ch16,
                                   max_disp=self.cfg.init_max_disp,
                                   groups=g,
                                   feat_out=self.cfg.tile_feat_ch)

        # Refinement heads at 1/16, 1/8, 1/4, AND 1/2.
        self.refine_16 = TileRefine(feat_ch=ch16,
                                     tile_feat_ch=self.cfg.tile_feat_ch,
                                     hidden=self.cfg.refine_hidden)
        self.refine_8 = TileRefine(feat_ch=ch8,
                                    tile_feat_ch=self.cfg.tile_feat_ch,
                                    hidden=self.cfg.refine_hidden)
        self.refine_4 = TileRefine(feat_ch=ch4,
                                    tile_feat_ch=self.cfg.tile_feat_ch,
                                    hidden=self.cfg.refine_hidden)
        self.refine_2 = TileRefine(feat_ch=ch2,
                                    tile_feat_ch=self.cfg.tile_feat_ch,
                                    hidden=self.cfg.refine_hidden)

        self.up_16_to_8 = TileUpsample(scale_factor=2)
        self.up_8_to_4 = TileUpsample(scale_factor=2)
        self.up_4_to_2 = TileUpsample(scale_factor=2)
        self.up_2_to_1 = TileUpsample(scale_factor=2)

    def forward(self, left: torch.Tensor, right: torch.Tensor,
                aux: bool = False):
        feats = self.fnet(torch.cat([left, right], dim=0))
        fL2,  fR2  = feats[0].chunk(2, dim=0)
        fL4,  fR4  = feats[1].chunk(2, dim=0)
        fL8,  fR8  = feats[2].chunk(2, dim=0)
        fL16, fR16 = feats[3].chunk(2, dim=0)

        tile = self.init_tile(fL16, fR16)
        t16_stages = [tile]
        for _ in range(self.cfg.iters_16):
            tile = self.refine_16(tile, fL16, fR16)
            t16_stages.append(tile)

        tile = self.up_16_to_8(tile, target_hw=fL8.shape[-2:])
        t8_stages = [tile]
        for _ in range(self.cfg.iters_8):
            tile = self.refine_8(tile, fL8, fR8)
            t8_stages.append(tile)

        tile = self.up_8_to_4(tile, target_hw=fL4.shape[-2:])
        t4_stages = [tile]
        for _ in range(self.cfg.iters_4):
            tile = self.refine_4(tile, fL4, fR4)
            t4_stages.append(tile)

        # NEW: refine at 1/2 too.
        tile = self.up_4_to_2(tile, target_hw=fL2.shape[-2:])
        t2_stages = [tile]
        for _ in range(self.cfg.iters_2):
            tile = self.refine_2(tile, fL2, fR2)
            t2_stages.append(tile)
        d_half = tile.d

        # NEW: 1/2 → full via plane equation only (no full-res features).
        tile_full = self.up_2_to_1(tile, target_hw=left.shape[-2:])
        d_full = tile_full.d
        if d_full.shape[-2:] != left.shape[-2:]:
            d_full = F.interpolate(d_full, size=left.shape[-2:],
                                    mode="bilinear", align_corners=True)

        if aux:
            return {
                "d_final": d_full,
                "d_half": d_half,
                "d4": t4_stages[-1].d,
                "d8": t8_stages[-1].d,
                "d8_cv": t16_stages[-1].d,
                "d16": t16_stages[-1].d,
                "d32": t16_stages[0].d,
            }
        return d_full
