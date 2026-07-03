"""Tile propagation for StereoLite_yolo_ctx_gev4_opt — efficiency-optimized
copy of StereoLite_yolo_ctx_gev4/tile_propagate.py.

Optimizations (all designed to be output-equivalent to the original; see
convert_state_dict() for loading original checkpoints):

  F1  TileInit cost volume built from zero-copy padded views instead of
      per-bin alloc+zero+copy (bitwise identical).
  F2  _correlation_lookup does ONE batched grid_sample (offsets stacked
      along the grid's H axis) instead of D sequential calls, and returns
      the offset-0 full-channel sample so TileRefineCtx skips its duplicate
      _horizontal_warp (identical sampling math). Meshgrids cached.
  F4  Static input channels (fL, ctx) of the GRU convs are convolved ONCE
      per scale invocation and reused across iterations (linear split of
      the convolution; mathematically exact, fp-summation-order differences
      only). Controlled by cfg.opt_static_ctx.
  F5  conv_z+conv_r fused into one conv (same input), and the four 1x1
      output heads fused into one 4-channel conv (bitwise identical given
      concatenated weights).
  F7  TileUpsample plane-offset constants cached per output size.
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


def _safe_gn(ch: int, max_groups: int = 8) -> nn.GroupNorm:
    g = 1
    for cand in range(min(max_groups, ch), 0, -1):
        if ch % cand == 0:
            g = cand
            break
    return nn.GroupNorm(num_groups=g, num_channels=ch)


_GRID_CACHE: dict = {}


def _base_grid(B: int, H: int, W: int, device, dtype):
    """Cached (1,H,W) pixel-coordinate grids."""
    key = (H, W, device, dtype)
    if key not in _GRID_CACHE:
        yy, xx = torch.meshgrid(
            torch.arange(H, device=device, dtype=dtype),
            torch.arange(W, device=device, dtype=dtype), indexing="ij")
        _GRID_CACHE[key] = (xx.unsqueeze(0), yy.unsqueeze(0))
    xx, yy = _GRID_CACHE[key]
    return xx, yy


def _correlation_lookup_batched(fL: torch.Tensor, fR: torch.Tensor,
                                d_center: torch.Tensor,
                                half_range: int, groups: int):
    """F2: one grid_sample for all D offsets, stacked along the grid H axis.

    Returns (cost [B, G*D, H, W], fR_w [B, C, H, W]) where fR_w is the
    offset-0 sample (identical to the original _horizontal_warp of fR by
    d_center with border padding + align_corners=True).
    """
    B, C, H, W = fL.shape
    cg = C // groups
    D = 2 * half_range + 1
    dtype = fL.dtype
    xx, yy = _base_grid(B, H, W, fL.device, dtype)
    gy = (yy / max(H - 1, 1) * 2 - 1).expand(B, H, W)

    d = d_center.squeeze(1)  # (B, H, W)
    # grids for offsets -hr..+hr, stacked along H: (B, D*H, W, 2)
    gxs = []
    for i in range(D):
        d_total = d + (i - half_range)
        gxs.append((xx.expand(B, H, W) - d_total) / max(W - 1, 1) * 2 - 1)
    gx_all = torch.cat(gxs, dim=1)                      # (B, D*H, W)
    gy_all = gy.repeat(1, D, 1)                         # (B, D*H, W)
    grid = torch.stack([gx_all, gy_all], dim=-1)        # (B, D*H, W, 2)
    # cudnn disabled for this op: the stacked-offset grid makes the output
    # B x C x (D*H) x W, which crosses cuDNN's 32-bit indexing ceiling around
    # batch 40 at 384x640 (CUDNN_STATUS_NOT_SUPPORTED, A100 probe 2026-07-04).
    # The native CUDA kernel handles it; values identical.
    with torch.backends.cudnn.flags(enabled=False):
        fR_all = F.grid_sample(fR.contiguous(), grid.contiguous(),
                                align_corners=True,
                                padding_mode="border")  # (B, C, D*H, W)
    fR_all = fR_all.view(B, C, D, H, W)

    fR_w = fR_all[:, :, half_range]                     # offset-0 sample
    fL_g = fL.view(B, groups, cg, 1, H, W)
    corr = (fL_g * fR_all.view(B, groups, cg, D, H, W)).mean(dim=2)
    return corr.reshape(B, groups * D, H, W), fR_w


def _groupwise_corr_volume_padded(fL: torch.Tensor, fR: torch.Tensor,
                                  max_disp: int, groups: int) -> torch.Tensor:
    """F1: shift-by-d as zero-copy narrow() views on a left-zero-padded fR.

    Bitwise identical to the original per-bin alloc/zero/copy loop: left
    zero-padding reproduces the out-of-range-equals-zero semantics.
    """
    B, C, H, W = fL.shape
    g = groups
    while C % g != 0 and g > 1:
        g -= 1
    cg = C // g
    fL_g = fL.view(B, g, cg, H, W)
    fR_pad = F.pad(fR, (max_disp - 1, 0))
    cv = fL.new_empty((B, g, max_disp, H, W))
    for d in range(max_disp):
        fR_view = fR_pad.narrow(-1, max_disp - 1 - d, W).view(B, g, cg, H, W)
        cv[:, :, d] = (fL_g * fR_view).mean(dim=2)
    return cv


class TileInit(nn.Module):
    """Cost-volume init at 1/16 (F1-optimized volume construction).

    `topk` > 0 enables CoEx-style top-k soft-argmin (softmax + expectation
    over only the k best-scoring bins per pixel) — commits to a dominant
    mode at bimodal boundary pixels instead of the between-modes mean.
    Must be trained in (CoEx: retrofitting k at eval time degrades).
    The post-aggregation logits are stashed on `self.last_logits` so the
    trainer can add a distribution-shaping init loss (blur bundle-1).
    """

    def __init__(self, feat_ch: int, max_disp: int = 24, groups: int = 8,
                 feat_out: int = 16, topk: int = 0):
        super().__init__()
        assert feat_ch % groups == 0
        self.max_disp = max_disp
        self.groups = groups
        self.feat_out = feat_out
        self.topk = topk
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
        self.last_logits = None

    def forward(self, fL: torch.Tensor, fR: torch.Tensor) -> TileState:
        B, C, H, W = fL.shape
        cv = _groupwise_corr_volume_padded(fL, fR, self.max_disp, self.groups)
        logits = self.agg(cv).squeeze(1)
        self.last_logits = logits
        if self.topk and self.topk < self.max_disp:
            # top-k soft-argmin: expectation over the k best bins only.
            vals, idx = logits.topk(self.topk, dim=1)
            prob_k = F.softmax(vals, dim=1)
            d = (prob_k * idx.to(prob_k.dtype)).sum(dim=1, keepdim=True)
            conf = prob_k.max(dim=1, keepdim=True).values
        else:
            prob = F.softmax(logits, dim=1)
            d = (prob * self.disp_idx).sum(dim=1, keepdim=True)
            conf = prob.max(dim=1, keepdim=True).values
        sx = torch.zeros_like(d)
        sy = torch.zeros_like(d)
        feat = torch.zeros((B, self.feat_out, H, W), device=fL.device,
                           dtype=fL.dtype)
        return TileState(d=d, sx=sx, sy=sy, feat=feat, conf=conf)


class TileRefineCtx(nn.Module):
    """ConvGRU tile refine, F2/F4/F5-optimized, output-equivalent to the
    original TileRefineCtx.

    Parameter layout (fused/split) differs from the original; use
    convert_state_dict() to load original checkpoints.

    Channel layout of the original gru_in concat:
        [h(thc) | fL(fc) | fR_w(fc) | d,sx,sy,conf(4) | cost(cc) | ctx(cx)]
    Split for F4: static = fL + ctx; dynamic = fR_w + 4 + cost; h separate.
    """

    def __init__(self, feat_ch: int, tile_feat_ch: int, hidden: int = 48,
                 half_range: int = 2, groups: int = 8,
                 ctx_ch: int | None = None, opt_static_ctx: bool = True):
        super().__init__()
        g = groups
        while feat_ch % g != 0 and g > 1:
            g -= 1
        self.groups_eff = g
        self.half_range = half_range
        self.feat_ch = feat_ch
        self.tile_feat_ch = tile_feat_ch
        self.ctx_ch = ctx_ch if ctx_ch is not None else tile_feat_ch
        self.cost_ch = g * (2 * half_range + 1)
        self.dyn_ch = feat_ch + 4 + self.cost_ch
        self.opt_static_ctx = opt_static_ctx

        thc = tile_feat_ch
        # F5: z and r fused (same input). F4: each split into h/static/dyn.
        self.zr_h = nn.Conv2d(thc, 2 * thc, 3, padding=1, bias=True)
        self.zr_static = nn.Conv2d(feat_ch + self.ctx_ch, 2 * thc, 3,
                                   padding=1, bias=False)
        self.zr_dyn = nn.Conv2d(self.dyn_ch, 2 * thc, 3, padding=1, bias=False)
        self.q_h = nn.Conv2d(thc, thc, 3, padding=1, bias=True)
        self.q_static = nn.Conv2d(feat_ch + self.ctx_ch, thc, 3,
                                  padding=1, bias=False)
        self.q_dyn = nn.Conv2d(self.dyn_ch, thc, 3, padding=1, bias=False)

        self.head_trunk = nn.Sequential(
            nn.Conv2d(thc, hidden, 3, padding=1, bias=False),
            _safe_gn(hidden), nn.SiLU(inplace=True),
            nn.Conv2d(hidden, hidden, 3, padding=1, bias=False),
            _safe_gn(hidden), nn.SiLU(inplace=True),
        )
        # F5: four 1x1 heads fused into one 4-channel conv
        # (out order: d, sx, sy, conf).
        self.heads = nn.Conv2d(hidden, 4, 1)
        self.update_gate = nn.Sequential(
            nn.Conv2d(tile_feat_ch + self.ctx_ch + 5, hidden, 3, padding=1,
                      bias=False),
            _safe_gn(hidden), nn.SiLU(inplace=True),
            nn.Conv2d(hidden, 4, 1),
        )

    def precompute_static(self, fL: torch.Tensor, ctx: torch.Tensor):
        """F4: per-scale static contributions, reused across iterations."""
        st = torch.cat([fL, ctx], dim=1)
        return self.zr_static(st), self.q_static(st)

    def forward(self, tile: TileState, fL: torch.Tensor, fR: torch.Tensor,
                ctx: torch.Tensor, static: tuple | None = None) -> TileState:
        cost, fR_w = _correlation_lookup_batched(
            fL, fR, tile.d, self.half_range, self.groups_eff)
        dyn = torch.cat([fR_w, tile.d, tile.sx, tile.sy, tile.conf, cost],
                        dim=1)
        if static is None:
            static = self.precompute_static(fL, ctx)
        zr_st, q_st = static

        h = tile.feat
        zr = self.zr_h(h) + zr_st + self.zr_dyn(dyn)
        z, r = torch.sigmoid(zr).chunk(2, dim=1)
        q = torch.tanh(self.q_h(r * h) + q_st + self.q_dyn(dyn))
        h_new = (1.0 - z) * h + z * q

        trunk = self.head_trunk(h_new)
        dd, dsx_raw, dsy_raw, dconf = self.heads(trunk).chunk(4, dim=1)
        dsx = dsx_raw * 0.1
        dsy = dsy_raw * 0.1

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
        self._const_cache: dict = {}

    def _plane_offsets(self, H_f, W_f, device, dtype):
        key = (H_f, W_f, device, dtype)
        if key not in self._const_cache:
            dx_block = torch.tensor([[-0.25, 0.25], [-0.25, 0.25]],
                                    device=device, dtype=dtype)
            dy_block = torch.tensor([[-0.25, -0.25], [0.25, 0.25]],
                                    device=device, dtype=dtype)
            dx = dx_block.repeat(H_f // 2, W_f // 2)
            dy = dy_block.repeat(H_f // 2, W_f // 2)
            if dx.shape != (H_f, W_f):
                dx = F.pad(dx, (0, W_f - dx.shape[1], 0, H_f - dx.shape[0]))
                dy = F.pad(dy, (0, W_f - dy.shape[1], 0, H_f - dy.shape[0]))
            self._const_cache[key] = (dx.view(1, 1, H_f, W_f),
                                      dy.view(1, 1, H_f, W_f))
        return self._const_cache[key]

    def forward(self, tile: TileState,
                target_hw: Optional[tuple[int, int]] = None) -> TileState:
        s = self.scale
        if target_hw is None:
            target_hw = (tile.d.shape[-2] * s, tile.d.shape[-1] * s)
        up = lambda x: F.interpolate(x, size=target_hw, mode="bilinear",
                                     align_corners=False)
        sx_up, sy_up = up(tile.sx), up(tile.sy)
        feat_up, conf_up = up(tile.feat), up(tile.conf)
        d_bilinear = up(tile.d) * s
        H_f, W_f = target_hw
        dx, dy = self._plane_offsets(H_f, W_f, tile.d.device, tile.d.dtype)
        d_plane = d_bilinear + (sx_up * s) * dx + (sy_up * s) * dy
        return TileState(d=d_plane, sx=sx_up, sy=sy_up, feat=feat_up,
                          conf=conf_up)


# ---------------------------------------------------------------------------
# Checkpoint conversion: original TileRefineCtx layout -> optimized layout
# ---------------------------------------------------------------------------

def convert_refine_keys(sd: dict, prefix: str, feat_ch: int,
                        tile_feat_ch: int, cost_ch: int, ctx_ch: int) -> dict:
    """Slice/concat one original TileRefineCtx's weights into the optimized
    layout. Original concat order along in_channels:
        [h(thc) | fL(fc) | fR_w(fc) | 4 | cost(cc) | ctx(cx)]
    """
    out = {}
    thc, fc, cc, cx = tile_feat_ch, feat_ch, cost_ch, ctx_ch
    dyn = fc + 4 + cc
    sl_h = slice(0, thc)
    sl_fL = slice(thc, thc + fc)
    sl_dyn = slice(thc + fc, thc + fc + dyn)
    sl_ctx = slice(thc + fc + dyn, thc + fc + dyn + cx)

    def split(w):
        return w[:, sl_h], torch.cat([w[:, sl_fL], w[:, sl_ctx]], 1), w[:, sl_dyn]

    wz, bz = sd[f"{prefix}conv_z.weight"], sd[f"{prefix}conv_z.bias"]
    wr, br = sd[f"{prefix}conv_r.weight"], sd[f"{prefix}conv_r.bias"]
    zh, zs, zd = split(wz)
    rh, rs, rd = split(wr)
    out[f"{prefix}zr_h.weight"] = torch.cat([zh, rh], 0)
    out[f"{prefix}zr_h.bias"] = torch.cat([bz, br], 0)
    out[f"{prefix}zr_static.weight"] = torch.cat([zs, rs], 0)
    out[f"{prefix}zr_dyn.weight"] = torch.cat([zd, rd], 0)

    wq, bq = sd[f"{prefix}conv_q.weight"], sd[f"{prefix}conv_q.bias"]
    qh, qs, qd = split(wq)
    out[f"{prefix}q_h.weight"] = qh
    out[f"{prefix}q_h.bias"] = bq
    out[f"{prefix}q_static.weight"] = qs
    out[f"{prefix}q_dyn.weight"] = qd

    out[f"{prefix}heads.weight"] = torch.cat(
        [sd[f"{prefix}head_{k}.weight"] for k in ("d", "sx", "sy", "conf")], 0)
    out[f"{prefix}heads.bias"] = torch.cat(
        [sd[f"{prefix}head_{k}.bias"] for k in ("d", "sx", "sy", "conf")], 0)

    for k in list(sd.keys()):
        if k.startswith(prefix) and (
                k.startswith(f"{prefix}head_trunk") or
                k.startswith(f"{prefix}update_gate")):
            out[k] = sd[k]
    return out
