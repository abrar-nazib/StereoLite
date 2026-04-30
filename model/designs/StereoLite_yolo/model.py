"""StereoLite v7 — tile-hypothesis propagation with iterative refinement.

Pipeline:
  Input (L, R) ──► GhostConv+SE encoder (shared) ──► f2, f4, f8, f16

  1/16: TileInit — tiny local cost volume + soft-argmin seeds
        (d, sx=0, sy=0, feat, conf). Iterative refine × N16.

  Upsample to 1/8 via plane equation (d + slope * offset) → iterate × N8.

  Upsample to 1/4 via plane equation → iterate × N4.

  Final: plane upsample 1/4 → 1/2 → full, with a single convex-mask-style
  learned upsample for sub-pixel boundary sharpness.

Key architectural changes vs v6:
  - Slopes (sx, sy) are now USED during upsampling (plane equation)
  - Iterative refinement: each scale runs the refine head N times
  - No 3D cost-volume aggregator at 1/8 (just local CV at 1/16 init)
  - No cascade CV (replaced by iteration)

Budget target: ~1.2-1.5 M trainable, ~150 ms inference on RTX 3050 at
512×832. Jetson INT8 target: ~25-40 ms.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from _blocks import (GhostConv, SqueezeExcitation, _safe_gn)

from .tile_propagate import (TileState, TileInit, TileRefine, TileUpsample)
from .yolo_encoder import YoloTruncatedEncoder


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
        f2 = self.stem(x)             # 1/2,  base ch
        f4 = self.s4(f2)              # 1/4,  2*base
        f8 = self.s8(f4)              # 1/8,  3*base
        f16 = self.s16(f8)            # 1/16, 4*base
        return f2, f4, f8, f16


class MobileNetV2Encoder(nn.Module):
    """ImageNet-pretrained MobileNetV2-100 features at 1/2, 1/4, 1/8, 1/16.

    Uses timm's features_only API and explicitly truncates blocks beyond the
    last requested stage. timm's FeatureListNet builds the full backbone and
    hooks intermediate outputs — the deeper unhooked blocks (1/32 stage)
    waste ~1 M params, consume forward compute, and break DDP because their
    parameters never receive gradients. Slicing self.backbone.blocks fixes
    all three problems.

    ImageNet normalisation is applied inside the module (input is expected
    in [0, 255] BGR→RGB-converted floats, to match the rest of the codebase).
    """

    def __init__(self, pretrained: bool = True):
        super().__init__()
        import timm
        self.backbone = timm.create_model(
            "mobilenetv2_100", pretrained=pretrained,
            features_only=True, out_indices=(0, 1, 2, 3))
        # Drop unused 1/32 blocks. The deepest hook is at the end of the
        # last block whose feature stride <= 16. For mobilenetv2_100 this
        # is index 4 in self.backbone.blocks (0..6 total). Slicing keeps
        # blocks 0..4 inclusive.
        last_kept = self._deepest_used_block_idx()
        self.backbone.blocks = self.backbone.blocks[: last_kept + 1]
        ch = self.backbone.feature_info.channels()
        self.out_channels = tuple(ch)
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1) * 255.0
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1) * 255.0
        self.register_buffer("mean", mean, persistent=False)
        self.register_buffer("std", std, persistent=False)

    def _deepest_used_block_idx(self) -> int:
        """Largest index in self.backbone.blocks whose output reduction is
        still <= 16 (our coarsest scale). Anything deeper is at 1/32 and we
        don't use it."""
        # feature_info.info is a list of dicts with 'reduction' and 'module'
        # ('module' looks like 'blocks.4', 'blocks.6', etc.).
        info = self.backbone.feature_info.info
        deepest = 0
        for entry in info:
            if entry.get("reduction", 0) <= 16:
                # Parse 'blocks.<N>' to get N
                module = entry.get("module", "")
                if module.startswith("blocks."):
                    deepest = max(deepest, int(module.split(".")[1]))
        return deepest

    def forward(self, x: torch.Tensor):
        x = (x - self.mean) / self.std
        f2, f4, f8, f16 = self.backbone(x)
        return f2, f4, f8, f16


class ConvexUpsample(nn.Module):
    """Learned 2x upsample of a (B, 1, H, W) disparity guided by per-pixel
    CNN features. Each fine pixel is a weighted sum of 9 coarse neighbours."""

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
class StereoLiteConfig:
    base_ch: int = 24
    tile_feat_ch: int = 16         # channels carried in the tile feat state
    # Iterations at each scale:
    iters_16: int = 2
    iters_8: int = 3
    iters_4: int = 3
    # Cost volume at init
    init_max_disp: int = 24        # at 1/16, covers 24*16=384 full-res px
    init_groups: int = 8
    # Refine head hidden channels
    refine_hidden: int = 48
    # Backbone: "ghost" (custom GhostConv) or "mobilenet" (ImageNet-pretrained)
    backbone: str = "ghost"
    backbone_pretrained: bool = True


class StereoLite(nn.Module):
    def __init__(self, cfg: StereoLiteConfig | None = None):
        super().__init__()
        self.cfg = cfg or StereoLiteConfig()
        b = self.cfg.base_ch
        if self.cfg.backbone == "mobilenet":
            self.fnet = MobileNetV2Encoder(pretrained=self.cfg.backbone_pretrained)
        elif self.cfg.backbone in ("yolo26n", "yolo26s"):
            self.fnet = YoloTruncatedEncoder(variant=self.cfg.backbone)
        else:
            self.fnet = TileFeatureEncoder(base=b)

        # Channel counts per scale from the encoder
        ch2, ch4, ch8, ch16 = self.fnet.out_channels

        # Groups need to divide feat_ch for the init cost volume
        g = self.cfg.init_groups
        while ch16 % g != 0 and g > 1:
            g -= 1
        self.init_tile = TileInit(feat_ch=ch16,
                                   max_disp=self.cfg.init_max_disp,
                                   groups=g,
                                   feat_out=self.cfg.tile_feat_ch)

        # Refinement heads — one instance per scale (weights not shared
        # because feature channel counts differ).
        self.refine_16 = TileRefine(feat_ch=ch16,
                                     tile_feat_ch=self.cfg.tile_feat_ch,
                                     hidden=self.cfg.refine_hidden)
        self.refine_8 = TileRefine(feat_ch=ch8,
                                    tile_feat_ch=self.cfg.tile_feat_ch,
                                    hidden=self.cfg.refine_hidden)
        self.refine_4 = TileRefine(feat_ch=ch4,
                                    tile_feat_ch=self.cfg.tile_feat_ch,
                                    hidden=self.cfg.refine_hidden)

        # Plane upsamples (no trainable weights, just geometry)
        self.up_16_to_8 = TileUpsample(scale_factor=2)
        self.up_8_to_4 = TileUpsample(scale_factor=2)

        # Final learned 4x upsample 1/4 → full, guided by 1/2 features
        # (done in two steps: 1/4 → 1/2 using f4 features, 1/2 → full
        # using f2). Two small convex modules.
        self.up_final_4_to_2 = ConvexUpsample(feat_ch=ch4, scale=2)
        self.up_final_2_to_1 = ConvexUpsample(feat_ch=ch2, scale=2)

    def _tile_d_at_scale(self, tile: TileState, scale: int) -> torch.Tensor:
        """Return the tile's d in 1/scale-px units (what the tile stores)."""
        return tile.d

    def forward(self, left: torch.Tensor, right: torch.Tensor,
                aux: bool = False):
        # Batch L+R through the encoder in a single forward: the GPU runs
        # both views in parallel inside the same kernel launches, roughly
        # halving encoder latency vs two sequential calls.
        feats = self.fnet(torch.cat([left, right], dim=0))
        fL2,  fR2  = feats[0].chunk(2, dim=0)
        fL4,  fR4  = feats[1].chunk(2, dim=0)
        fL8,  fR8  = feats[2].chunk(2, dim=0)
        fL16, fR16 = feats[3].chunk(2, dim=0)

        # 1/16: init + iterate
        tile = self.init_tile(fL16, fR16)
        t16_stages = [tile]
        for _ in range(self.cfg.iters_16):
            tile = self.refine_16(tile, fL16, fR16)
            t16_stages.append(tile)

        # 1/16 → 1/8 via plane equation
        tile = self.up_16_to_8(tile, target_hw=fL8.shape[-2:])
        t8_stages = [tile]
        for _ in range(self.cfg.iters_8):
            tile = self.refine_8(tile, fL8, fR8)
            t8_stages.append(tile)

        # 1/8 → 1/4
        tile = self.up_8_to_4(tile, target_hw=fL4.shape[-2:])
        t4_stages = [tile]
        for _ in range(self.cfg.iters_4):
            tile = self.refine_4(tile, fL4, fR4)
            t4_stages.append(tile)

        # Final: learned upsample 1/4 → 1/2 → full, each 2x. Disparity
        # unit conversion: multiplying by 2 each time converts from
        # coarse-unit disparity to fine-unit.
        d_half = self.up_final_4_to_2(tile.d, fL4)
        d_full = self.up_final_2_to_1(d_half, fL2)
        if d_full.shape[-2:] != left.shape[-2:]:
            d_full = F.interpolate(d_full, size=left.shape[-2:],
                                    mode="bilinear", align_corners=True)

        if aux:
            return {
                "d_final": d_full,
                # expose a few iteration outputs for multi-scale loss
                "d_half": d_half,
                "d4": tile.d,               # after last 1/4 iteration
                "d8": t8_stages[-1].d,      # after last 1/8 iteration
                "d8_cv": t16_stages[-1].d,  # backward-compat key for loss
                "d16": t16_stages[-1].d,
                "d32": t16_stages[0].d,     # init at 1/16
            }
        return d_full


