"""HITNet-inspired plane-tile hypothesis propagation for StereoLite v7.

Each tile carries a plane hypothesis:
    d    : disparity at the tile centre
    sx   : slope in x direction (sub-pixel disparity change per pixel to right)
    sy   : slope in y direction
    feat : learned per-tile features (propagated through iterations)
    conf : scalar confidence in [0, 1]

Key operations:
    - TileInit:     build a tile state from a local cost volume at coarse scale
    - TileRefine:   update (d, sx, sy, feat, conf) via warp-regress given L, R
                    features at the current scale. Warp uses the plane
                    equation, so matching benefits from sub-pixel gradients.
    - TileUpsample: apply the plane equation to produce 2x denser tiles at
                    the next scale up.

Plane equation for upsampling:
    for a parent tile centre at (y, x) with slope (sx, sy) and disparity d,
    its four children at offsets (dy, dx) in { -0.25, 0.25 } × { -0.25, 0.25 }
    get disparity d + sx * dx * parent_scale + sy * dy * parent_scale.
    (We implement this via explicit per-child offsets during upsample.)
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class TileState:
    """Plane-tile hypothesis at some scale. All tensors are (B, C, H, W)
    where (H, W) is the tile-grid resolution."""
    d: torch.Tensor              # (B, 1, H, W) absolute disparity in 1/k-px units
    sx: torch.Tensor             # (B, 1, H, W) slope x (dispar change per pixel)
    sy: torch.Tensor             # (B, 1, H, W) slope y
    feat: torch.Tensor           # (B, Cf, H, W) learned per-tile features
    conf: torch.Tensor           # (B, 1, H, W) in [0, 1]


def _horizontal_warp(fR: torch.Tensor, disp: torch.Tensor) -> torch.Tensor:
    """Same warp primitive as the rest of the code."""
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


def _plane_warp_disp(tile: TileState) -> torch.Tensor:
    """For warping fR, use the plane-sampled disparity: each pixel's
    effective disparity includes a sub-pixel correction from the slope.
    Here we just return d — the slope-corrected sampling happens inside
    refine via slope-informed feature fusion. Keeping warp simple.
    """
    return tile.d


def _safe_gn(ch: int, max_groups: int = 8) -> nn.GroupNorm:
    g = 1
    for cand in range(min(max_groups, ch), 0, -1):
        if ch % cand == 0:
            g = cand
            break
    return nn.GroupNorm(num_groups=g, num_channels=ch)


class TileInit(nn.Module):
    """Initialise tiles at the coarsest scale (1/16) using a small local
    cost volume + 3D soft-argmin. Outputs (d, sx=0, sy=0, feat, conf)."""

    def __init__(self, feat_ch: int, max_disp: int = 24, groups: int = 8,
                 feat_out: int = 16):
        super().__init__()
        assert feat_ch % groups == 0
        self.max_disp = max_disp
        self.groups = groups
        self.feat_out = feat_out

        # Tiny 3D aggregator (~50 k params)
        self.agg = nn.Sequential(
            nn.Conv3d(groups, 16, 3, padding=1, bias=False),
            _safe_gn(16), nn.SiLU(inplace=True),
            nn.Conv3d(16, 16, 3, padding=1, bias=False),
            _safe_gn(16), nn.SiLU(inplace=True),
            nn.Conv3d(16, 1, 3, padding=1, bias=True),
        )
        # Per-tile feature extractor from left features only.
        self.feat_head = nn.Sequential(
            nn.Conv2d(feat_ch, feat_out, 3, padding=1, bias=False),
            _safe_gn(feat_out), nn.SiLU(inplace=True),
        )
        self.register_buffer(
            "disp_idx",
            torch.arange(max_disp, dtype=torch.float32).view(1, max_disp, 1, 1),
            persistent=False,
        )

    def forward(self, fL: torch.Tensor, fR: torch.Tensor) -> TileState:
        B, C, H, W = fL.shape
        cg = C // self.groups
        fL_g = fL.view(B, self.groups, cg, H, W)
        cv = fL.new_zeros((B, self.groups, self.max_disp, H, W))
        for d in range(self.max_disp):
            if d == 0:
                fR_s = fR
            else:
                fR_s = fL.new_zeros(fR.shape)
                fR_s[:, :, :, d:] = fR[:, :, :, :-d]
            fR_g = fR_s.view(B, self.groups, cg, H, W)
            cv[:, :, d] = (fL_g * fR_g).mean(dim=2)

        logits = self.agg(cv).squeeze(1)                     # (B, D, H, W)
        prob = F.softmax(logits, dim=1)
        d = (prob * self.disp_idx).sum(dim=1, keepdim=True)
        # Confidence from the peak mass — how sharp the distribution is
        conf = prob.max(dim=1, keepdim=True).values          # (B, 1, H, W)
        sx = torch.zeros_like(d)
        sy = torch.zeros_like(d)
        feat = self.feat_head(fL)
        return TileState(d=d, sx=sx, sy=sy, feat=feat, conf=conf)


class TileRefine(nn.Module):
    """Update a tile state given left/right CNN features at the tile scale.
    Warps fR using d, concatenates with fL, prior feat, slopes, and conf,
    then predicts residuals for d, sx, sy, feat, conf. Residuals are added.
    """

    def __init__(self, feat_ch: int, tile_feat_ch: int, hidden: int = 48):
        super().__init__()
        in_ch = 2 * feat_ch + tile_feat_ch + 4   # fL, fR_warp, feat, d, sx, sy, conf
        self.trunk = nn.Sequential(
            nn.Conv2d(in_ch, hidden, 3, padding=1, bias=False),
            _safe_gn(hidden), nn.SiLU(inplace=True),
            nn.Conv2d(hidden, hidden, 3, padding=1, bias=False),
            _safe_gn(hidden), nn.SiLU(inplace=True),
            nn.Conv2d(hidden, hidden, 3, padding=1, bias=False),
            _safe_gn(hidden), nn.SiLU(inplace=True),
        )
        self.head_d = nn.Conv2d(hidden, 1, 1)
        self.head_sx = nn.Conv2d(hidden, 1, 1)
        self.head_sy = nn.Conv2d(hidden, 1, 1)
        self.head_conf = nn.Conv2d(hidden, 1, 1)
        self.head_feat = nn.Conv2d(hidden, tile_feat_ch, 1)

    def forward(self, tile: TileState, fL: torch.Tensor,
                fR: torch.Tensor) -> TileState:
        fR_w = _horizontal_warp(fR, _plane_warp_disp(tile))
        x = torch.cat([fL, fR_w, tile.feat, tile.d, tile.sx, tile.sy,
                        tile.conf], dim=1)
        h = self.trunk(x)
        d = F.softplus(self.head_d(h) + tile.d)              # keep positive
        sx = tile.sx + self.head_sx(h) * 0.1                 # small residual
        sy = tile.sy + self.head_sy(h) * 0.1
        conf = torch.sigmoid(self.head_conf(h) + 2.0 * tile.conf - 1.0)
        feat = tile.feat + self.head_feat(h)
        return TileState(d=d, sx=sx, sy=sy, feat=feat, conf=conf)


class TileUpsample(nn.Module):
    """Plane-aware upsample: the four children at (±0.25 tile) inherit
    d + slope * offset. We emit the 4 children tiles into a 2x denser grid.

    Slopes and features are just 2x bilinear upsampled (they change less
    rapidly than d across a scale boundary). Confidence is also bilinear.
    The disparity uses the plane equation for sub-pixel accuracy.
    """

    def __init__(self, scale_factor: int = 2):
        super().__init__()
        self.scale = scale_factor

    def forward(self, tile: TileState,
                target_hw: Optional[tuple[int, int]] = None) -> TileState:
        s = self.scale
        if target_hw is None:
            target_hw = (tile.d.shape[-2] * s, tile.d.shape[-1] * s)

        # Bilinear upsample of slopes, features, confidence
        sx_up = F.interpolate(tile.sx, size=target_hw, mode="bilinear",
                              align_corners=False)
        sy_up = F.interpolate(tile.sy, size=target_hw, mode="bilinear",
                              align_corners=False)
        feat_up = F.interpolate(tile.feat, size=target_hw, mode="bilinear",
                                 align_corners=False)
        conf_up = F.interpolate(tile.conf, size=target_hw, mode="bilinear",
                                 align_corners=False)

        # Plane-equation upsample of d: first bilinear the parent d (*s to
        # rescale the disparity units to the finer scale), then add a
        # sub-pixel correction from slopes.
        d_bilinear = F.interpolate(tile.d, size=target_hw, mode="bilinear",
                                    align_corners=False) * s

        # Build per-pixel offsets in the fine grid: each fine pixel sits
        # at a sub-parent-tile position. Offset in parent-tile units is in
        # {-0.25, +0.25} for a 2x upsample. We synthesise this pattern.
        device = tile.d.device
        H_f, W_f = target_hw
        # Offsets within each 2x2 block, in tile-local units (center-relative)
        dx_block = torch.tensor([[-0.25, 0.25], [-0.25, 0.25]],
                                device=device, dtype=tile.d.dtype)
        dy_block = torch.tensor([[-0.25, -0.25], [0.25, 0.25]],
                                device=device, dtype=tile.d.dtype)
        dx = dx_block.repeat(H_f // 2, W_f // 2)
        dy = dy_block.repeat(H_f // 2, W_f // 2)
        if dx.shape != (H_f, W_f):
            # Handle odd dims by pad/truncate (rare)
            dx = F.pad(dx, (0, W_f - dx.shape[1], 0, H_f - dx.shape[0]))
            dy = F.pad(dy, (0, W_f - dy.shape[1], 0, H_f - dy.shape[0]))
        dx = dx.view(1, 1, H_f, W_f)
        dy = dy.view(1, 1, H_f, W_f)

        # Slopes are in coarse units (disparity change per pixel at coarse
        # scale). Multiply by s to convert to fine units, then by the fine
        # per-pixel offset.
        d_plane = d_bilinear + (sx_up * s) * dx + (sy_up * s) * dy

        return TileState(d=d_plane, sx=sx_up, sy=sy_up, feat=feat_up,
                          conf=conf_up)
