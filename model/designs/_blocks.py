"""Shared modern building blocks for d1 and d3.

All blocks use GroupNorm (not BatchNorm) so eval-mode and train-mode behave
identically at any batch size. INT8-friendly choices throughout.
"""
from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


def _safe_gn(ch: int, max_groups: int = 8) -> nn.GroupNorm:
    """Pick the largest divisor of ch that is <= max_groups (and >= 1)."""
    g = max_groups
    while g > 1 and ch % g != 0:
        g -= 1
    return nn.GroupNorm(g, ch)


# ----------------------------------------------------------------------- #
# GhostConv (Han et al., CVPR 2020) — half "primary" + half cheap depthwise
# ----------------------------------------------------------------------- #
class GhostConv(nn.Module):
    """Output channels = primary (out//2) ⊕ cheap depthwise of primary.

    Roughly halves the conv parameters vs a vanilla 3×3 conv at similar
    accuracy. Activations split out cleanly for INT8.
    """

    def __init__(self, in_ch: int, out_ch: int, k: int = 3, s: int = 1,
                 ratio: int = 2, gn_groups: int = 8):
        super().__init__()
        primary = out_ch // ratio
        cheap = out_ch - primary
        self.primary = nn.Conv2d(in_ch, primary, k, stride=s, padding=k // 2, bias=False)
        self.norm1 = _safe_gn(primary, gn_groups)
        self.cheap = nn.Conv2d(primary, cheap, 3, padding=1, groups=primary, bias=False)
        self.norm2 = _safe_gn(cheap, gn_groups)
        self.act = nn.SiLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        p = self.act(self.norm1(self.primary(x)))
        c = self.act(self.norm2(self.cheap(p)))
        return torch.cat([p, c], dim=1)


# ----------------------------------------------------------------------- #
# Squeeze-and-Excitation (Hu et al., CVPR 2018) — channel attention
# ----------------------------------------------------------------------- #
class SqueezeExcitation(nn.Module):
    def __init__(self, ch: int, r: int = 8):
        super().__init__()
        hidden = max(ch // r, 8)
        self.fc1 = nn.Conv2d(ch, hidden, 1)
        self.fc2 = nn.Conv2d(hidden, ch, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        s = F.adaptive_avg_pool2d(x, 1)
        s = F.silu(self.fc1(s), inplace=True)
        s = torch.sigmoid(self.fc2(s))
        return x * s


# ----------------------------------------------------------------------- #
# RepVGG block (Ding et al., CVPR 2021) — train multi-branch / deploy 3x3
# ----------------------------------------------------------------------- #
class RepVGGBlock(nn.Module):
    """Three parallel branches at training time: 3×3 conv, 1×1 conv,
    identity. They sum into one output. At deployment time the three are
    fused into a single 3×3 conv (free accuracy/latency win).

    For our validation rounds we keep the train-time multi-branch path
    (no fusion). The fusion routine is left as a TODO for deployment.
    """

    def __init__(self, in_ch: int, out_ch: int, stride: int = 1, gn_groups: int = 8):
        super().__init__()
        self.in_ch = in_ch
        self.out_ch = out_ch
        self.stride = stride
        self.conv3 = nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False)
        self.conv1 = nn.Conv2d(in_ch, out_ch, 1, stride=stride, padding=0, bias=False)
        self.norm3 = _safe_gn(out_ch, gn_groups)
        self.norm1 = _safe_gn(out_ch, gn_groups)
        self.has_identity = stride == 1 and in_ch == out_ch
        if self.has_identity:
            self.norm_id = _safe_gn(out_ch, gn_groups)
        self.act = nn.SiLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.norm3(self.conv3(x)) + self.norm1(self.conv1(x))
        if self.has_identity:
            y = y + self.norm_id(x)
        return self.act(y)


# ----------------------------------------------------------------------- #
# Lightweight neighborhood (linear) attention for tile propagation
# ----------------------------------------------------------------------- #
class NeighborhoodAttention2d(nn.Module):
    """Linear-attention version of HITNet's 3×3 tile-propagation step.

    Instead of a fixed 3×3 average over the 8 neighbours, each cell attends
    to its k×k neighbourhood with learned scores derived from a feature
    projection. Linear-attention kernel φ(x) = elu(x) + 1 keeps it cheap.
    """

    def __init__(self, ch: int, kernel: int = 3, head_dim: int = 16):
        super().__init__()
        self.kernel = kernel
        self.proj_qkv = nn.Conv2d(ch, 3 * head_dim, 1, bias=False)
        self.out = nn.Conv2d(head_dim, ch, 1, bias=False)
        self.head_dim = head_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        qkv = self.proj_qkv(x)                                          # (B, 3D, H, W)
        q, k, v = qkv.chunk(3, dim=1)
        q = F.elu(q) + 1
        k = F.elu(k) + 1

        # Unfold k and v over a kernel×kernel window for local linear attn.
        kh = self.kernel
        pad = kh // 2
        k_loc = F.unfold(k, kernel_size=kh, padding=pad)                # (B, D*K, H*W)
        v_loc = F.unfold(v, kernel_size=kh, padding=pad)
        k_loc = k_loc.view(B, self.head_dim, kh * kh, H * W)
        v_loc = v_loc.view(B, self.head_dim, kh * kh, H * W)
        q_flat = q.view(B, self.head_dim, 1, H * W)

        scores = (q_flat * k_loc).sum(dim=1, keepdim=True)              # (B,1,K,HW)
        weights = scores / (scores.sum(dim=2, keepdim=True) + 1e-6)
        out = (weights * v_loc).sum(dim=2)                              # (B, D, HW)
        out = out.view(B, self.head_dim, H, W)
        return self.out(out)


# ----------------------------------------------------------------------- #
# Pure-PyTorch selective-scan (Mamba S6 core), 1D over a chosen dim
# ----------------------------------------------------------------------- #
class SelectiveScan1d(nn.Module):
    """A simplified Mamba-S6 selective scan over one spatial dimension.

    Input  : x of shape (B, C, L) where L is the scan length.
    Output : same shape; aggregated context with input-dependent gating.

    For our stereo SGM-replacement use, the caller reshapes a cost volume
    (B, D, H, W) to (B*H, D, W) and back to scan along W (left-right), and
    similarly for the other three directions. Compute is sequential (slow
    on GPU, just like Python-loop SGM) but it is end-to-end differentiable
    and the operator is structurally identical to the official Mamba scan.
    """

    def __init__(self, ch: int, state_dim: int = 16, dt_rank: int = 4):
        super().__init__()
        self.ch = ch
        self.state_dim = state_dim
        self.dt_rank = dt_rank
        self.x_proj = nn.Conv1d(ch, dt_rank + 2 * state_dim, 1, bias=False)
        self.dt_proj = nn.Conv1d(dt_rank, ch, 1, bias=True)
        # Stable A-init: log(-A) where A < 0 for stability.
        A = torch.arange(1, state_dim + 1, dtype=torch.float32).repeat(ch, 1)  # (C, N)
        self.A_log = nn.Parameter(torch.log(A))
        self.D = nn.Parameter(torch.ones(ch))
        self.gate = nn.Conv1d(ch, ch, 1, bias=True)

    def forward(self, x: torch.Tensor, reverse: bool = False,
                chunk_size: int = 16) -> torch.Tensor:
        """Chunked-parallel selective scan for h(t) = dA(t)·h(t-1) + dBu(t).

        Within each chunk we do a short sequential scan (chunk_size ≪ L),
        parallel across all chunks. Across chunks we propagate state with
        n_chunks sequential ops. Total CUDA launches ≈ chunk_size + n_chunks
        instead of L per direction. Numerically stable (no exp of cumulative
        log) for the sequence lengths we care about.
        """
        B, C, L = x.shape
        if reverse:
            x = x.flip(-1)

        proj = self.x_proj(x)                                            # (B, dt_rank + 2N, L)
        dt, B_t, C_t = torch.split(proj, [self.dt_rank, self.state_dim, self.state_dim], dim=1)
        dt = F.softplus(self.dt_proj(dt))                                # (B, C, L)

        A = -torch.exp(self.A_log).to(x.dtype)                           # (C, N), negative

        B_t = B_t.permute(0, 2, 1).unsqueeze(1)                          # (B, 1, L, N)
        C_t = C_t.permute(0, 2, 1).unsqueeze(1)                          # (B, 1, L, N)

        deltaA = torch.exp(dt.unsqueeze(-1) * A.view(1, C, 1, self.state_dim))   # (B, C, L, N)
        deltaB_u = dt.unsqueeze(-1) * B_t * x.unsqueeze(-1)              # (B, C, L, N)

        # Pad L to a multiple of chunk_size.
        pad = (chunk_size - L % chunk_size) % chunk_size
        if pad:
            deltaA = F.pad(deltaA, (0, 0, 0, pad), value=1.0)
            deltaB_u = F.pad(deltaB_u, (0, 0, 0, pad), value=0.0)
        Lp = L + pad
        n_chunks = Lp // chunk_size

        dA = deltaA.view(B, C, n_chunks, chunk_size, self.state_dim)
        dBu = deltaB_u.view(B, C, n_chunks, chunk_size, self.state_dim)

        # 1) Within each chunk: short sequential scan starting from h=0.
        h = torch.zeros(B, C, n_chunks, self.state_dim, device=x.device, dtype=x.dtype)
        within = []
        for t in range(chunk_size):
            h = dA[:, :, :, t] * h + dBu[:, :, :, t]
            within.append(h)
        within = torch.stack(within, dim=3)                              # (B, C, n_chunks, chunk_size, N)
        chunk_total_A = torch.prod(dA, dim=3)                            # (B, C, n_chunks, N)
        chunk_total_b = within[:, :, :, -1]                              # (B, C, n_chunks, N)

        # 2) Across chunks: sequential propagation of chunk-start state.
        h_start = torch.zeros(B, C, self.state_dim, device=x.device, dtype=x.dtype)
        starts = [h_start]
        for i in range(n_chunks - 1):
            h_start = chunk_total_A[:, :, i] * h_start + chunk_total_b[:, :, i]
            starts.append(h_start)
        starts = torch.stack(starts, dim=2)                              # (B, C, n_chunks, N)

        # 3) Combine: full h(t) = cumprod(dA[:t]) * h_start + within(t)
        cum_dA = torch.cumprod(dA, dim=3)                                # (B, C, n_chunks, chunk_size, N)
        h_full = cum_dA * starts.unsqueeze(3) + within                   # same shape
        h_full = h_full.view(B, C, Lp, self.state_dim)[:, :, :L]

        y = (h_full * C_t).sum(dim=-1)                                   # (B, C, L)
        y = y + x * self.D.view(1, C, 1)
        y = y * torch.sigmoid(self.gate(x))

        if reverse:
            y = y.flip(-1)
        return y


def scan_4_dirs(scanner: SelectiveScan1d, cv: torch.Tensor) -> torch.Tensor:
    """Apply a SelectiveScan1d in four directions over a (B, D, H, W) cost
    volume and sum the four scans. Sequential: slow but functionally
    identical to a CUDA Mamba kernel.
    """
    B, D, H, W = cv.shape
    # L→R / R→L scans operate over W (treat each row as an independent batch)
    cv_w = cv.permute(0, 2, 1, 3).reshape(B * H, D, W)
    lr = scanner(cv_w, reverse=False).view(B, H, D, W).permute(0, 2, 1, 3)
    rl = scanner(cv_w, reverse=True).view(B, H, D, W).permute(0, 2, 1, 3)
    # T→B / B→T over H
    cv_h = cv.permute(0, 3, 1, 2).reshape(B * W, D, H)
    tb = scanner(cv_h, reverse=False).view(B, W, D, H).permute(0, 2, 3, 1)
    bt = scanner(cv_h, reverse=True).view(B, W, D, H).permute(0, 2, 3, 1)
    return lr, rl, tb, bt
