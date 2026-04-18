"""Group-wise correlation cost volume with a 1-level 3D hourglass
aggregator for LiteFMStereo.

Operates at 1/8 scale. For each candidate disparity d = 0..D-1:
  shift fR left by d (zero-fill on the left), compute group-wise inner
  product with fL, yielding one cost slice per disparity. Stack into a
  (B, G, D, H, W) 4D tensor. Aggregate with a 3D hourglass (down-then-up
  with a skip), softmax over D, soft-argmin → disparity at 1/8 scale in
  1/8-pixel units (multiply by 8 for full-res disparity).

Intended budget ~1 M params total.

Deployment: the 3D convolutions are INT8-friendly (standard ops in both
TensorRT and ONNX). GroupNorm replaces BatchNorm for batch-size
independence at inference.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def _gn3d(ch: int, max_groups: int = 8) -> nn.GroupNorm:
    g = 1
    for cand in range(min(max_groups, ch), 0, -1):
        if ch % cand == 0:
            g = cand
            break
    return nn.GroupNorm(num_groups=g, num_channels=ch)


class ConvGnSiLU3D(nn.Module):
    def __init__(self, cin: int, cout: int, k: int = 3,
                 stride: tuple[int, int, int] | int = 1):
        super().__init__()
        pad = k // 2
        self.conv = nn.Conv3d(cin, cout, k, stride=stride,
                              padding=pad, bias=False)
        self.norm = _gn3d(cout)
        self.act = nn.SiLU(inplace=True)

    def forward(self, x):
        return self.act(self.norm(self.conv(x)))


class DeconvGnSiLU3D(nn.Module):
    def __init__(self, cin: int, cout: int, k: int = 3,
                 stride: tuple[int, int, int] = (1, 2, 2)):
        super().__init__()
        pad = k // 2
        out_pad = tuple((s - 1) if s > 1 else 0 for s in stride)
        self.conv = nn.ConvTranspose3d(cin, cout, k, stride=stride,
                                       padding=pad, output_padding=out_pad,
                                       bias=False)
        self.norm = _gn3d(cout)
        self.act = nn.SiLU(inplace=True)

    def forward(self, x):
        return self.act(self.norm(self.conv(x)))


class ResBlock3D(nn.Module):
    def __init__(self, ch: int):
        super().__init__()
        self.c1 = ConvGnSiLU3D(ch, ch, 3)
        self.c2 = nn.Conv3d(ch, ch, 3, padding=1, bias=False)
        self.n2 = _gn3d(ch)
        self.act = nn.SiLU(inplace=True)

    def forward(self, x):
        r = self.c1(x)
        r = self.n2(self.c2(r))
        return self.act(x + r)


def _groupwise_correlation(fL: torch.Tensor, fR: torch.Tensor,
                            max_disp: int, groups: int) -> torch.Tensor:
    """Build (B, G, D, H, W) group-wise correlation cost volume.

    fL, fR: (B, C, H, W) with C % groups == 0.
    Returns mean-over-channels correlation per group, per disparity.
    """
    B, C, H, W = fL.shape
    assert C % groups == 0, f"channels {C} not divisible by groups {groups}"
    cg = C // groups
    # Pre-group reshape: (B, G, cg, H, W)
    fL_g = fL.view(B, groups, cg, H, W)
    cv = fL.new_zeros((B, groups, max_disp, H, W))
    for d in range(max_disp):
        if d == 0:
            fR_shift = fR
        else:
            fR_shift = fL.new_zeros(fR.shape)
            fR_shift[:, :, :, d:] = fR[:, :, :, :-d]
        fR_g = fR_shift.view(B, groups, cg, H, W)
        cv[:, :, d] = (fL_g * fR_g).mean(dim=2)
    return cv


class GuidedCostExcitation(nn.Module):
    """CoEx-style guided cost excitation: channel-wise gate on the 3D
    aggregator features, derived from the left image features. Suppresses
    irrelevant feature channels and sharpens the matching signal.
    (~few k params; known to improve EPE by 3-8% in the CoEx paper.)
    """

    def __init__(self, guide_ch: int, agg_ch: int):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Conv2d(guide_ch, agg_ch, 1, bias=True),
            nn.Sigmoid(),
        )

    def forward(self, cv_feat: torch.Tensor, fL_guide: torch.Tensor) -> torch.Tensor:
        # cv_feat: (B, agg_ch, D, H, W).  fL_guide: (B, guide_ch, H, W).
        g = self.gate(fL_guide)                    # (B, agg_ch, H, W)
        return cv_feat * g.unsqueeze(2)             # broadcast over D


class GroupwiseCostVolume1D8(nn.Module):
    """Group-wise correlation + 1-level 3D hourglass aggregator at 1/8,
    with CoEx-style Guided Cost Excitation after the first 3D layer.

    Input: fL, fR at 1/8 scale, shape (B, C, H/8, W/8). C must be
    divisible by `groups`.
    Output: disparity at 1/8 scale in 1/8-px units, shape (B, 1, H/8, W/8).
    """

    def __init__(self, feat_ch: int, max_disp: int = 24, groups: int = 8,
                 agg_ch: int = 48):
        super().__init__()
        assert feat_ch % groups == 0
        self.max_disp = max_disp
        self.groups = groups

        self.in_conv = ConvGnSiLU3D(groups, agg_ch, 3)
        self.gce = GuidedCostExcitation(guide_ch=feat_ch, agg_ch=agg_ch)
        self.res_a = ResBlock3D(agg_ch)
        self.down = ConvGnSiLU3D(agg_ch, 2 * agg_ch, 3, stride=(1, 2, 2))
        self.res_mid = ResBlock3D(2 * agg_ch)
        self.up = DeconvGnSiLU3D(2 * agg_ch, agg_ch, 3, stride=(1, 2, 2))
        self.res_b = ResBlock3D(agg_ch)
        self.out_conv = nn.Conv3d(agg_ch, 1, 3, padding=1, bias=True)

        self.register_buffer(
            "disp_idx",
            torch.arange(max_disp, dtype=torch.float32).view(1, max_disp, 1, 1),
            persistent=False,
        )

    def forward(self, fL: torch.Tensor, fR: torch.Tensor) -> torch.Tensor:
        cv = _groupwise_correlation(fL, fR, self.max_disp, self.groups)
        x0 = self.in_conv(cv)
        x0 = self.gce(x0, fL)                      # guided excitation
        x0 = self.res_a(x0)
        x1 = self.down(x0)
        x1 = self.res_mid(x1)
        y0 = self.up(x1)
        if y0.shape[-2:] != x0.shape[-2:]:
            y0 = F.interpolate(y0, size=x0.shape[-3:], mode="trilinear",
                               align_corners=False)
        y0 = y0 + x0
        y0 = self.res_b(y0)
        logits = self.out_conv(y0).squeeze(1)
        prob = F.softmax(logits, dim=1)
        disp = (prob * self.disp_idx).sum(dim=1, keepdim=True)
        return disp


def _groupwise_correlation_around(fL: torch.Tensor, fR: torch.Tensor,
                                   d_center: torch.Tensor,
                                   half_range: int, groups: int) -> torch.Tensor:
    """Build a narrow-range cost volume CENTERED on a per-pixel disparity
    estimate d_center. This is the "cascade" / refinement volume from
    BGNet/CFNet — covers 2*half_range+1 disparity samples around the
    coarse estimate, which lets the CV commit to sub-grid disparities at
    finer resolution without paying for a full D dim.

    fL, fR: (B, C, H, W) at 1/4.
    d_center: (B, 1, H, W) in 1/4-px units — the coarse estimate.
    Returns (B, G, 2*half_range+1, H, W).
    """
    B, C, H, W = fL.shape
    cg = C // groups
    D = 2 * half_range + 1
    device, dtype = fL.device, fL.dtype
    # Flow-sample fR at continuous offset d_center + offset(d).
    yy, xx = torch.meshgrid(
        torch.arange(H, device=device, dtype=dtype),
        torch.arange(W, device=device, dtype=dtype), indexing="ij")
    gy = (yy / max(H - 1, 1) * 2 - 1).view(1, H, W).expand(B, H, W)
    fL_g = fL.view(B, groups, cg, H, W)
    out = fL.new_zeros((B, groups, D, H, W))
    for i in range(D):
        d_off = (i - half_range)
        d_total = d_center.squeeze(1) + d_off                   # (B, H, W)
        gx = (xx.view(1, H, W) - d_total) / max(W - 1, 1) * 2 - 1
        grid = torch.stack([gx, gy], dim=-1)
        fR_w = F.grid_sample(fR, grid, align_corners=True, padding_mode="border")
        fR_g = fR_w.view(B, groups, cg, H, W)
        out[:, :, i] = (fL_g * fR_g).mean(dim=2)
    return out


class CascadeRefinementVolume(nn.Module):
    """Narrow-range cascade cost volume at 1/4 scale. Refines a coarse
    disparity estimate (upsampled from 1/8) by searching ±half_range
    disparities around it. Output is a sub-grid disparity correction,
    making the 1/4 estimate pixel-accurate up to the 1/4-resolution
    Nyquist limit — meaningfully sharper than soft-argmin of a coarse
    24-level 1/8 CV alone.

    Input: fL, fR at 1/4, coarse d_init at 1/4 (in 1/4-px units).
    Output: refined disparity (B, 1, H/4, W/4) in 1/4-px units.
    """

    def __init__(self, feat_ch: int, half_range: int = 4, groups: int = 8,
                 agg_ch: int = 24):
        super().__init__()
        assert feat_ch % groups == 0
        self.half_range = half_range
        self.groups = groups
        D = 2 * half_range + 1

        # Smaller 2D-3D hybrid aggregator since D is small (9 slots here).
        self.in_conv = ConvGnSiLU3D(groups, agg_ch, 3)
        self.res_a = ResBlock3D(agg_ch)
        self.res_b = ResBlock3D(agg_ch)
        self.out_conv = nn.Conv3d(agg_ch, 1, 3, padding=1, bias=True)

        self.register_buffer(
            "off_idx",
            (torch.arange(D, dtype=torch.float32) - half_range).view(1, D, 1, 1),
            persistent=False,
        )

    def forward(self, fL: torch.Tensor, fR: torch.Tensor,
                d_coarse: torch.Tensor) -> torch.Tensor:
        cv = _groupwise_correlation_around(
            fL, fR, d_coarse, self.half_range, self.groups)
        x = self.in_conv(cv)
        x = self.res_a(x)
        x = self.res_b(x)
        logits = self.out_conv(x).squeeze(1)
        prob = F.softmax(logits, dim=1)
        offset = (prob * self.off_idx).sum(dim=1, keepdim=True)
        return d_coarse + offset
