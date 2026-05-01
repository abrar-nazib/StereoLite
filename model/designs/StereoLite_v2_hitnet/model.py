"""StereoLite — Variant B (HITNet-exact propagation block, single-pass per scale).

Architecture experiment (vs the StereoLite baseline):
  - Replaces our iterated 3-layer-conv TileRefine with HITNet's exact
    propagation block: residual blocks (no BN), dilated convolutions,
    1×1 channel reduction, leaky ReLU. Augmented input includes a local
    cost volume from warping at three disparity offsets (d-1, d, d+1).
  - Single pass per scale (no iteration).
  - Removes ConvexUpsample entirely.
  - Plane-equation upsample between scales and 1/2 → full.

Pipeline:
  Input (L, R) ──► encoder ──► f2, f4, f8, f16

  1/16: TileInit + HITNetPropagate × 1
  1/16 → 1/8 plane upsample, HITNetPropagate × 1
  1/8 → 1/4 plane upsample, HITNetPropagate × 1
  1/4 → 1/2 plane upsample, HITNetPropagate × 1
  1/2 → full plane upsample (no full-res features available, no propagate)

This is the closest faithful adaptation of HITNet to our chassis given
that we work at feature-map resolution rather than HITNet's 4×4-image-tile
resolution. The reference is Tankovich et al. CVPR 2021, Section 3.4.
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

from .tile_propagate import (TileState, TileInit, TileUpsample)
from .hitnet_propagate import HITNetPropagate


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
    init_max_disp: int = 24
    init_groups: int = 8
    propagate_hidden: int = 32
    backbone: str = "ghost"
    backbone_pretrained: bool = True


class StereoLite(nn.Module):
    def __init__(self, cfg: StereoLiteConfig | None = None):
        super().__init__()
        self.cfg = cfg or StereoLiteConfig()
        b = self.cfg.base_ch
        assert self.cfg.backbone == "ghost", \
            "Variant B is fixed to ghost encoder for fair architecture A/B"
        self.fnet = TileFeatureEncoder(base=b)

        ch2, ch4, ch8, ch16 = self.fnet.out_channels

        g = self.cfg.init_groups
        while ch16 % g != 0 and g > 1:
            g -= 1
        self.init_tile = TileInit(feat_ch=ch16,
                                   max_disp=self.cfg.init_max_disp,
                                   groups=g,
                                   feat_out=self.cfg.tile_feat_ch)

        # HITNet-style single-pass propagation modules, one per scale.
        # Different feature channel counts mean separate weights per scale.
        self.prop_16 = HITNetPropagate(feat_ch=ch16,
                                        tile_feat_ch=self.cfg.tile_feat_ch,
                                        hidden=self.cfg.propagate_hidden)
        self.prop_8 = HITNetPropagate(feat_ch=ch8,
                                       tile_feat_ch=self.cfg.tile_feat_ch,
                                       hidden=self.cfg.propagate_hidden)
        self.prop_4 = HITNetPropagate(feat_ch=ch4,
                                       tile_feat_ch=self.cfg.tile_feat_ch,
                                       hidden=self.cfg.propagate_hidden)
        self.prop_2 = HITNetPropagate(feat_ch=ch2,
                                       tile_feat_ch=self.cfg.tile_feat_ch,
                                       hidden=self.cfg.propagate_hidden)

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

        # 1/16 init + propagate (single pass).
        tile = self.init_tile(fL16, fR16)
        t16_init = tile
        tile = self.prop_16(tile, fL16, fR16)
        t16_prop = tile

        # 1/16 → 1/8.
        tile = self.up_16_to_8(tile, target_hw=fL8.shape[-2:])
        tile = self.prop_8(tile, fL8, fR8)
        t8_prop = tile

        # 1/8 → 1/4.
        tile = self.up_8_to_4(tile, target_hw=fL4.shape[-2:])
        tile = self.prop_4(tile, fL4, fR4)
        t4_prop = tile

        # 1/4 → 1/2.
        tile = self.up_4_to_2(tile, target_hw=fL2.shape[-2:])
        tile = self.prop_2(tile, fL2, fR2)
        d_half = tile.d

        # 1/2 → full via plane upsample only.
        tile_full = self.up_2_to_1(tile, target_hw=left.shape[-2:])
        d_full = tile_full.d
        if d_full.shape[-2:] != left.shape[-2:]:
            d_full = F.interpolate(d_full, size=left.shape[-2:],
                                    mode="bilinear", align_corners=True)

        if aux:
            return {
                "d_final": d_full,
                "d_half": d_half,
                "d4": t4_prop.d,
                "d8": t8_prop.d,
                "d8_cv": t16_prop.d,
                "d16": t16_prop.d,
                "d32": t16_init.d,
            }
        return d_full
