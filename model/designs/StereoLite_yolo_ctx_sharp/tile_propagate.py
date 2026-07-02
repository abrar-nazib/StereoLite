"""Tile propagation for StereoLite_yolo_ctx_sharp.

This is a deliberately small modification of `StereoLite_yolo_ctx`: keep
the same cost lookup and ConvGRU, but add a cheap context/confidence gate
that scales each residual update. The goal is better overfit and
generalization without the large latency hit from slanted patch sampling.

  - `TileInit` is unchanged from the raftlike sibling (a tiny local cost
    volume at 1/16 + 3D soft-argmin). The hidden state `h` (= tile.feat)
    is INITIALISED from the projected context features at 1/16, not from
    a random conv head. This is the load-bearing difference vs `raftlike`.

  - `TileRefineCtx` mirrors `TileRefineRAFTLike` but takes a `ctx`
    argument (the context features at the current scale's resolution,
    bilinearly upsampled from the context encoder's 1/4 output). The
    per-iteration input context is
        x = [fL, fR_warp, d, sx, sy, conf, cost_lookup, ctx]
    The GRU update is unchanged.

  - `TileUpsample` is unchanged. Cross-scale hidden state propagation
    still uses the plane-equation-aware bilinear upsample of tile.feat.
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


def _safe_gn(ch: int, max_groups: int = 8) -> nn.GroupNorm:
    g = 1
    for cand in range(min(max_groups, ch), 0, -1):
        if ch % cand == 0:
            g = cand
            break
    return nn.GroupNorm(num_groups=g, num_channels=ch)


def _norm_feat(x: torch.Tensor) -> torch.Tensor:
    return F.normalize(x, p=2, dim=1, eps=1e-6)


def _correlation_lookup(fL: torch.Tensor, fR: torch.Tensor,
                        d_center: torch.Tensor,
                        half_range: int, groups: int) -> torch.Tensor:
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
    """Cost-volume init at 1/16. Hidden state h is set later (in the
    model) from the context encoder — this module just emits h=zeros at
    the right shape; the model overwrites it before the first refine."""

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
        # h = zeros at feat_out channels; the model replaces this with the
        # projected context at 1/16 before the first refine call.
        feat = torch.zeros((B, self.feat_out, H, W), device=fL.device,
                           dtype=fL.dtype)
        return TileState(d=d, sx=sx, sy=sy, feat=feat, conf=conf)


class TileRefineCtx(nn.Module):
    """Cost-lookup-driven ConvGRU refine with gated residual updates.

    State: h = tile.feat (tile_feat_ch channels).
    Per-iteration input context:
        x = [fL, fR_warp, d, sx, sy, conf, cost_lookup, ctx_proj]
    where `ctx_proj` is the context-encoder output at the current scale's
    spatial resolution, bilinearly upsampled from the encoder's 1/4
    output by the model and passed in as `ctx`.

    The GRU update is the standard minimal ConvGRU (z, r, q gates). A small
    gate then decides how strongly to apply the predicted geometry update.
    """

    def __init__(self, feat_ch: int, tile_feat_ch: int, hidden: int = 48,
                 half_range: int = 2, groups: int = 8,
                 ctx_ch: int | None = None):
        super().__init__()
        g = groups
        while feat_ch % g != 0 and g > 1:
            g -= 1
        self.groups_eff = g
        self.half_range = half_range
        self.ctx_ch = ctx_ch if ctx_ch is not None else tile_feat_ch
        cost_ch = g * (2 * half_range + 1)
        ctx_in = 2 * feat_ch + 4 + cost_ch + self.ctx_ch
        gru_in = tile_feat_ch + ctx_in

        self.conv_z = nn.Conv2d(gru_in, tile_feat_ch, 3, padding=1)
        self.conv_r = nn.Conv2d(gru_in, tile_feat_ch, 3, padding=1)
        self.conv_q = nn.Conv2d(gru_in, tile_feat_ch, 3, padding=1)

        self.head_trunk = nn.Sequential(
            nn.Conv2d(tile_feat_ch, hidden, 3, padding=1, bias=False),
            _safe_gn(hidden), nn.SiLU(inplace=True),
            nn.Conv2d(hidden, hidden, 3, padding=1, bias=False),
            _safe_gn(hidden), nn.SiLU(inplace=True),
        )
        self.head_d = nn.Conv2d(hidden, 1, 1)
        self.head_sx = nn.Conv2d(hidden, 1, 1)
        self.head_sy = nn.Conv2d(hidden, 1, 1)
        self.head_conf = nn.Conv2d(hidden, 1, 1)
        self.update_gate = nn.Sequential(
            nn.Conv2d(tile_feat_ch + self.ctx_ch + 5, hidden, 3, padding=1,
                      bias=False),
            _safe_gn(hidden), nn.SiLU(inplace=True),
            nn.Conv2d(hidden, 4, 1),
        )

    def forward(self, tile: TileState, fL: torch.Tensor,
                fR: torch.Tensor,
                ctx: torch.Tensor) -> TileState:
        """ctx: (B, ctx_ch, H, W) context features at the current scale."""
        fR_w = _horizontal_warp(fR, tile.d)
        cost = _correlation_lookup(fL, fR, tile.d,
                                    self.half_range, self.groups_eff)
        ctx_vec = torch.cat([fL, fR_w, tile.d, tile.sx, tile.sy, tile.conf,
                              cost, ctx], dim=1)
        h = tile.feat
        hctx = torch.cat([h, ctx_vec], dim=1)
        z = torch.sigmoid(self.conv_z(hctx))
        r = torch.sigmoid(self.conv_r(hctx))
        rh_ctx = torch.cat([r * h, ctx_vec], dim=1)
        q = torch.tanh(self.conv_q(rh_ctx))
        h_new = (1.0 - z) * h + z * q

        trunk = self.head_trunk(h_new)
        dd = self.head_d(trunk)
        dsx = self.head_sx(trunk) * 0.1
        dsy = self.head_sy(trunk) * 0.1
        dconf = self.head_conf(trunk)

        cost_view = cost.view(cost.shape[0], self.groups_eff,
                              2 * self.half_range + 1,
                              cost.shape[-2], cost.shape[-1])
        score = cost_view.mean(dim=1)
        prob = F.softmax(score, dim=1)
        cost_peak = prob.max(dim=1, keepdim=True).values
        cost_entropy = -(prob * prob.clamp_min(1e-6).log()).sum(
            dim=1, keepdim=True)

        gate_in = torch.cat([
            h_new, ctx, tile.conf, cost_peak, cost_entropy,
            dd.abs(), (dsx.abs() + dsy.abs()),
        ], dim=1)
        g_d, g_sx, g_sy, g_conf = torch.sigmoid(
            self.update_gate(gate_in)).chunk(4, dim=1)

        d = F.softplus(tile.d + g_d * dd)
        sx = tile.sx + g_sx * dsx
        sy = tile.sy + g_sy * dsy
        conf = torch.sigmoid(g_conf * dconf + 2.0 * tile.conf - 1.0)
        return TileState(d=d, sx=sx, sy=sy, feat=h_new, conf=conf)


class TileUpsample(nn.Module):
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
