"""Tile propagation for StereoLite_costlookup.

Adds RAFT-style per-iteration local correlation lookup to TileRefine. At
each iter, given the current per-tile disparity d, samples fR at
d + Δ for Δ ∈ {-half_range, ..., +half_range}, computes the group-wise
correlation with fL at each offset, and concatenates the resulting
(2*half_range+1) × groups channels to the standard TileRefine input.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class TileState:
    d: torch.Tensor
    sx: torch.Tensor
    sy: torch.Tensor
    feat: torch.Tensor
    conf: torch.Tensor


def _horizontal_warp(fR: torch.Tensor, disp: torch.Tensor,
                     sx: torch.Tensor | None = None,
                     sy: torch.Tensor | None = None) -> torch.Tensor:
    """Warp fR by disparity. If sx/sy are provided (slope-aware warp),
    bilinearly-sample fR at TWO disparities — one slope-extrapolated
    forward and one back — and average. Effectively integrates fR over
    the pixel's footprint along the local plane. For flat surfaces
    (sx=sy=0) this collapses to the standard warp.
    """
    B, _, H, W = fR.shape
    yy, xx = torch.meshgrid(
        torch.arange(H, device=fR.device, dtype=fR.dtype),
        torch.arange(W, device=fR.device, dtype=fR.dtype),
        indexing="ij",
    )
    d = disp.squeeze(1)
    gy = (yy.view(1, H, W) / max(H - 1, 1) * 2 - 1).expand(B, H, W)

    if sx is None and sy is None:
        gx = (xx.view(1, H, W) - d) / max(W - 1, 1) * 2 - 1
        grid = torch.stack([gx, gy], dim=-1)
        return F.grid_sample(fR, grid, align_corners=True,
                             padding_mode="border")

    # Slope-aware: sample at d - 0.5*sx and d + 0.5*sx along x (forward+back
    # slope-extrapolated) and average. The half-pixel offset matches the
    # tile-cell footprint at the current scale.
    sx_v = sx.squeeze(1) if sx is not None else 0.0
    d_lo = d - 0.5 * sx_v
    d_hi = d + 0.5 * sx_v
    gx_lo = (xx.view(1, H, W) - d_lo) / max(W - 1, 1) * 2 - 1
    gx_hi = (xx.view(1, H, W) - d_hi) / max(W - 1, 1) * 2 - 1
    grid_lo = torch.stack([gx_lo, gy], dim=-1)
    grid_hi = torch.stack([gx_hi, gy], dim=-1)
    fR_lo = F.grid_sample(fR, grid_lo, align_corners=True,
                          padding_mode="border")
    fR_hi = F.grid_sample(fR, grid_hi, align_corners=True,
                          padding_mode="border")
    return 0.5 * (fR_lo + fR_hi)


def _safe_gn(ch: int, max_groups: int = 8) -> nn.GroupNorm:
    g = 1
    for cand in range(min(max_groups, ch), 0, -1):
        if ch % cand == 0:
            g = cand
            break
    return nn.GroupNorm(num_groups=g, num_channels=ch)


def _correlation_lookup(fL: torch.Tensor, fR: torch.Tensor,
                        d_center: torch.Tensor,
                        half_range: int, groups: int) -> torch.Tensor:
    """Local groupwise correlation around a per-pixel disparity estimate.

    fL, fR: (B, C, H, W). d_center: (B, 1, H, W) in current-scale px.
    Returns (B, G * D, H, W) where D = 2*half_range+1 — flattened so it
    can be concatenated as channels to the TileRefine input.
    """
    B, C, H, W = fL.shape
    cg = C // groups
    D = 2 * half_range + 1
    device, dtype = fL.device, fL.dtype
    yy, xx = torch.meshgrid(
        torch.arange(H, device=device, dtype=dtype),
        torch.arange(W, device=device, dtype=dtype), indexing="ij")
    gy = (yy / max(H - 1, 1) * 2 - 1).view(1, H, W).expand(B, H, W)
    fL_g = fL.view(B, groups, cg, H, W)
    out = fL.new_zeros((B, groups, D, H, W))
    for i in range(D):
        d_off = (i - half_range)
        d_total = d_center.squeeze(1) + d_off
        gx = (xx.view(1, H, W) - d_total) / max(W - 1, 1) * 2 - 1
        grid = torch.stack([gx, gy], dim=-1)
        fR_w = F.grid_sample(fR, grid, align_corners=True,
                             padding_mode="border")
        fR_g = fR_w.view(B, groups, cg, H, W)
        out[:, :, i] = (fL_g * fR_g).mean(dim=2)
    return out.view(B, groups * D, H, W)


class TileInit(nn.Module):
    """Same as the StereoLite TileInit — local cost volume + soft-argmin."""

    def __init__(self, feat_ch: int, max_disp: int = 24, groups: int = 8,
                 feat_out: int = 16):
        super().__init__()
        assert feat_ch % groups == 0
        self.max_disp = max_disp
        self.groups = groups
        self.feat_out = feat_out
        self.agg = nn.Sequential(
            nn.Conv3d(groups, 16, 3, padding=1, bias=False),
            _safe_gn(16), nn.SiLU(inplace=True),
            nn.Conv3d(16, 16, 3, padding=1, bias=False),
            _safe_gn(16), nn.SiLU(inplace=True),
            nn.Conv3d(16, 1, 3, padding=1, bias=True),
        )
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
        logits = self.agg(cv).squeeze(1)
        prob = F.softmax(logits, dim=1)
        d = (prob * self.disp_idx).sum(dim=1, keepdim=True)
        conf = prob.max(dim=1, keepdim=True).values
        sx = torch.zeros_like(d)
        sy = torch.zeros_like(d)
        feat = self.feat_head(fL)
        return TileState(d=d, sx=sx, sy=sy, feat=feat, conf=conf)


class TileRefineCostLookup(nn.Module):
    """TileRefine + per-iter local correlation lookup.

    Channels added to input: groups_eff * (2*half_range+1).
    """

    def __init__(self, feat_ch: int, tile_feat_ch: int, hidden: int = 48,
                 half_range: int = 2, groups: int = 8):
        super().__init__()
        # Pick groups that divides feat_ch
        g = groups
        while feat_ch % g != 0 and g > 1:
            g -= 1
        self.groups_eff = g
        self.half_range = half_range
        self.feat_ch = feat_ch
        cost_ch = g * (2 * half_range + 1)
        in_ch = 2 * feat_ch + tile_feat_ch + 4 + cost_ch
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
        fR_w = _horizontal_warp(fR, tile.d)
        cost = _correlation_lookup(fL, fR, tile.d,
                                   self.half_range, self.groups_eff)
        x = torch.cat([fL, fR_w, tile.feat, tile.d, tile.sx, tile.sy,
                        tile.conf, cost], dim=1)
        h = self.trunk(x)
        d = F.softplus(self.head_d(h) + tile.d)
        sx = tile.sx + self.head_sx(h) * 0.1
        sy = tile.sy + self.head_sy(h) * 0.1
        conf = torch.sigmoid(self.head_conf(h) + 2.0 * tile.conf - 1.0)
        feat = tile.feat + self.head_feat(h)
        return TileState(d=d, sx=sx, sy=sy, feat=feat, conf=conf)


class TileUpsample(nn.Module):
    """Plane-equation upsample (same as StereoLite)."""

    def __init__(self, scale_factor: int = 2):
        super().__init__()
        self.scale = scale_factor

    def forward(self, tile: TileState,
                target_hw: Optional[tuple[int, int]] = None) -> TileState:
        s = self.scale
        if target_hw is None:
            target_hw = (tile.d.shape[-2] * s, tile.d.shape[-1] * s)
        sx_up = F.interpolate(tile.sx, size=target_hw, mode="bilinear",
                              align_corners=False)
        sy_up = F.interpolate(tile.sy, size=target_hw, mode="bilinear",
                              align_corners=False)
        feat_up = F.interpolate(tile.feat, size=target_hw, mode="bilinear",
                                 align_corners=False)
        conf_up = F.interpolate(tile.conf, size=target_hw, mode="bilinear",
                                 align_corners=False)
        d_bilinear = F.interpolate(tile.d, size=target_hw, mode="bilinear",
                                    align_corners=False) * s
        device = tile.d.device
        H_f, W_f = target_hw
        dx_block = torch.tensor([[-0.25, 0.25], [-0.25, 0.25]],
                                device=device, dtype=tile.d.dtype)
        dy_block = torch.tensor([[-0.25, -0.25], [0.25, 0.25]],
                                device=device, dtype=tile.d.dtype)
        dx = dx_block.repeat(H_f // 2, W_f // 2)
        dy = dy_block.repeat(H_f // 2, W_f // 2)
        if dx.shape != (H_f, W_f):
            dx = F.pad(dx, (0, W_f - dx.shape[1], 0, H_f - dx.shape[0]))
            dy = F.pad(dy, (0, W_f - dy.shape[1], 0, H_f - dy.shape[0]))
        dx = dx.view(1, 1, H_f, W_f)
        dy = dy.view(1, 1, H_f, W_f)
        d_plane = d_bilinear + (sx_up * s) * dx + (sy_up * s) * dy
        return TileState(d=d_plane, sx=sx_up, sy=sy_up, feat=feat_up,
                          conf=conf_up)
