"""HITNet's exact propagation block (Tankovich et al., CVPR 2021, Sec 3.4).

Implements the architecture described in HITNet's paper for the
propagation network U_l:

    a_l,x,y = [tile_hypothesis, φ(d-1), φ(d), φ(d+1)]
    (Δh, w) = U_l(a_l)

where φ(d) is the local cost vector from warping the right feature map
at disparity d, and U_l is a CNN with:
  - 1×1 conv + leaky ReLU to reduce channels at the input
  - Residual blocks (no batch normalisation, paper says "[20] without BN")
  - Dilated convolutions to increase receptive field, following [25]

The block is applied **once per scale** (no iteration), in contrast to
RAFT-style iterated refinement.

The "tile hypothesis" we propagate is (d, sx, sy, feat, conf), matching
the rest of our chassis. Output is a residual update to all five.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .tile_propagate import TileState


def _horizontal_warp(fR: torch.Tensor, disp: torch.Tensor) -> torch.Tensor:
    """Same warp primitive used elsewhere in the codebase."""
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


class HITNetResBlock(nn.Module):
    """Residual block as described in HITNet sec 3.4: dilated convs, no BN,
    leaky ReLU. Each block is two 3×3 dilated convs with a residual add."""

    def __init__(self, ch: int, dilation: int = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(ch, ch, 3, padding=dilation,
                                dilation=dilation, bias=True)
        self.conv2 = nn.Conv2d(ch, ch, 3, padding=dilation,
                                dilation=dilation, bias=True)
        self.act = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.act(self.conv1(x))
        h = self.conv2(h)
        return self.act(x + h)


class HITNetPropagate(nn.Module):
    """HITNet's per-scale propagation block (single pass, not iterated).

    Inputs:
        tile  : current tile state at this scale
        fL    : left features at this scale
        fR    : right features at this scale

    Internal pipeline (from paper):
        1. Build local cost volume aug: warp fR by d-1, d, d+1; per-channel
           absolute differences vs fL → 3 × feat_ch cost slices.
        2. Concatenate with the tile hypothesis (d, sx, sy, conf, feat).
        3. 1×1 conv reduces to `hidden` channels with leaky ReLU.
        4. Stack of residual blocks with dilations [1, 2, 4, 1].
        5. Per-output 1×1 convs → Δd, Δsx, Δsy, Δfeat, Δconf.
    """

    def __init__(self, feat_ch: int, tile_feat_ch: int = 16,
                 hidden: int = 32,
                 dilations: tuple[int, ...] = (1, 2, 4, 1)):
        super().__init__()
        # Augmented input: 3 cost slices (each feat_ch dim, from |fL - fR_warp|)
        # plus the tile hypothesis itself: d, sx, sy, conf, feat (1+1+1+1+tile_feat_ch).
        in_ch = 3 * feat_ch + 4 + tile_feat_ch
        self.reduce = nn.Sequential(
            nn.Conv2d(in_ch, hidden, 1, bias=True),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.blocks = nn.Sequential(*[
            HITNetResBlock(hidden, dilation=d) for d in dilations
        ])
        self.head_d = nn.Conv2d(hidden, 1, 1)
        self.head_sx = nn.Conv2d(hidden, 1, 1)
        self.head_sy = nn.Conv2d(hidden, 1, 1)
        self.head_feat = nn.Conv2d(hidden, tile_feat_ch, 1)
        self.head_conf = nn.Conv2d(hidden, 1, 1)

    def _cost(self, fL: torch.Tensor, fR: torch.Tensor,
              disp: torch.Tensor) -> torch.Tensor:
        """Per-channel L1 cost between fL and fR warped by `disp`."""
        fR_w = _horizontal_warp(fR, disp)
        return (fL - fR_w).abs()

    def forward(self, tile: TileState, fL: torch.Tensor,
                fR: torch.Tensor) -> TileState:
        # 1) Local cost volume augmentation: warp at d-1, d, d+1.
        cost_m1 = self._cost(fL, fR, tile.d - 1.0)
        cost_0  = self._cost(fL, fR, tile.d)
        cost_p1 = self._cost(fL, fR, tile.d + 1.0)

        # 2) Augmented input.
        x = torch.cat([
            cost_m1, cost_0, cost_p1,
            tile.d, tile.sx, tile.sy, tile.conf, tile.feat,
        ], dim=1)

        # 3) Channel reduction.
        h = self.reduce(x)
        # 4) Residual stack.
        h = self.blocks(h)

        # 5) Residual updates to the tile state.
        d = F.softplus(self.head_d(h) + tile.d)
        sx = tile.sx + self.head_sx(h) * 0.1
        sy = tile.sy + self.head_sy(h) * 0.1
        conf = torch.sigmoid(self.head_conf(h) + 2.0 * tile.conf - 1.0)
        feat = tile.feat + self.head_feat(h)
        return TileState(d=d, sx=sx, sy=sy, feat=feat, conf=conf)
