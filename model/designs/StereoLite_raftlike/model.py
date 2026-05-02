"""StereoLite_raftlike — current chassis + cost lookup + ConvGRU.

The closest tile-resolution analogue of RAFT-Stereo's update loop:
  - Per-iter local correlation lookup around current d
  - ConvGRU on the tile feat slot (hidden state)
  - Plane-aware propagation across scales (slope-aware upsample of h)

Two phases controlled by `extend_to_full`:
  - False: TileRefine at 1/16, 1/8, 1/4 + ConvexUpsample to full
  - True : TileRefine at 1/16, 1/8, 1/4, 1/2 + plane-eq upsample to full
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

from .tile_propagate import (TileState, TileInit, TileRefineRAFTLike,
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
    iters_2: int = 2
    init_max_disp: int = 24
    init_groups: int = 8
    refine_hidden: int = 48
    cost_lookup_half_range: int = 2
    cost_lookup_groups: int = 8
    backbone: str = "ghost"
    backbone_pretrained: bool = True
    extend_to_full: bool = False


class StereoLite(nn.Module):
    def __init__(self, cfg: StereoLiteConfig | None = None):
        super().__init__()
        self.cfg = cfg or StereoLiteConfig()
        b = self.cfg.base_ch
        if self.cfg.backbone in ("yolo26n", "yolo26s"):
            self.fnet = YoloTruncatedEncoder(variant=self.cfg.backbone)
        else:
            self.fnet = TileFeatureEncoder(base=b)

        ch2, ch4, ch8, ch16 = self.fnet.out_channels

        g = self.cfg.init_groups
        while ch16 % g != 0 and g > 1:
            g -= 1
        self.init_tile = TileInit(feat_ch=ch16,
                                   max_disp=self.cfg.init_max_disp,
                                   groups=g,
                                   feat_out=self.cfg.tile_feat_ch)

        def make_refine(feat_ch):
            return TileRefineRAFTLike(
                feat_ch=feat_ch, tile_feat_ch=self.cfg.tile_feat_ch,
                hidden=self.cfg.refine_hidden,
                half_range=self.cfg.cost_lookup_half_range,
                groups=self.cfg.cost_lookup_groups)

        self.refine_16 = make_refine(ch16)
        self.refine_8 = make_refine(ch8)
        self.refine_4 = make_refine(ch4)
        self.up_16_to_8 = TileUpsample(scale_factor=2)
        self.up_8_to_4 = TileUpsample(scale_factor=2)

        if self.cfg.extend_to_full:
            self.refine_2 = make_refine(ch2)
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

    def forward(self, left, right, aux: bool = False):
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

        if self.cfg.extend_to_full:
            tile = self.up_4_to_2(tile, target_hw=fL2.shape[-2:])
            t2_stages = [tile]
            for _ in range(self.cfg.iters_2):
                tile = self.refine_2(tile, fL2, fR2)
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
