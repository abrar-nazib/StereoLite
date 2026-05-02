"""Feature wideners for the yolo26n encoder.

The YOLO26n truncated stack has thinner f2 (16 ch) than ghost (24 ch) and
flat f8/f16 (128/128) where it was supposed to be refined by the FPN/PAN
neck (which we removed). These wideners are tested against ghost / yolo26s
on yolo26n's output features to see if widening / refining the per-stage
channels can close the quality gap that the locked chassis sees on yolo26n.

Tier 1: plain 1×1 channel adapters (f2_only, f2_f4, all_to_s)
Tier 2: refinement-with-widening blocks (DW, MBConv, GhostConv)
Tier 3: cross-scale fusion (top-down FPN, BiFPN)
Tier 4: norm-replacement (GN replaces BN inside the encoder)

`yolo26s_native` is not implemented here; it's expressed by setting
backbone="yolo26s" with widener=none.

All widener wrappers are nn.Modules taking a 4-tuple (f2, f4, f8, f16)
and returning a 4-tuple of the same length. They expose `out_channels`
as a tuple of 4 ints describing the post-widener stage channel counts.
"""
from __future__ import annotations

from typing import Sequence
import torch
import torch.nn as nn
import torch.nn.functional as F


def _safe_gn(ch: int, max_groups: int = 8) -> nn.GroupNorm:
    g = 1
    for cand in range(min(max_groups, ch), 0, -1):
        if ch % cand == 0:
            g = cand
            break
    return nn.GroupNorm(num_groups=g, num_channels=ch)


# -------------------- Tier 1: 1×1 adapters --------------------

def _adapter_1x1(in_ch: int, out_ch: int) -> nn.Module:
    if in_ch == out_ch:
        return nn.Identity()
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, 1, bias=False),
        _safe_gn(out_ch),
        nn.SiLU(inplace=True),
    )


class WidenerNone(nn.Module):
    """Pass-through. Returns inputs unchanged."""
    def __init__(self, in_channels: Sequence[int]):
        super().__init__()
        self.out_channels = tuple(in_channels)

    def forward(self, feats):
        return feats


class WidenerTier1(nn.Module):
    """Per-stage 1×1 Conv + GroupNorm + SiLU adapters."""
    def __init__(self, in_channels: Sequence[int],
                 target_channels: Sequence[int]):
        super().__init__()
        assert len(in_channels) == len(target_channels) == 4
        self.out_channels = tuple(target_channels)
        self.adapters = nn.ModuleList([
            _adapter_1x1(i, o) for i, o in zip(in_channels, target_channels)
        ])

    def forward(self, feats):
        return tuple(a(f) for f, a in zip(feats, self.adapters))


# -------------------- Tier 2: refinement+widening blocks --------------------

class _DWBlock(nn.Module):
    """1×1 expand → DW 3×3 → 1×1 project (with GN+SiLU between)."""
    def __init__(self, in_ch: int, out_ch: int, expand: int = 2):
        super().__init__()
        mid = out_ch * expand if in_ch < out_ch else in_ch
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_ch, mid, 1, bias=False),
            _safe_gn(mid), nn.SiLU(inplace=True),
        )
        self.dw = nn.Sequential(
            nn.Conv2d(mid, mid, 3, padding=1, groups=mid, bias=False),
            _safe_gn(mid), nn.SiLU(inplace=True),
        )
        self.proj = nn.Sequential(
            nn.Conv2d(mid, out_ch, 1, bias=False),
            _safe_gn(out_ch),
        )

    def forward(self, x):
        return self.proj(self.dw(self.conv1(x)))


class WidenerTier2DW(nn.Module):
    """Per-stage DW refinement block widened to target channels."""
    def __init__(self, in_channels: Sequence[int],
                 target_channels: Sequence[int]):
        super().__init__()
        self.out_channels = tuple(target_channels)
        self.blocks = nn.ModuleList([
            _DWBlock(i, o) for i, o in zip(in_channels, target_channels)
        ])

    def forward(self, feats):
        return tuple(b(f) for f, b in zip(feats, self.blocks))


class _SE(nn.Module):
    def __init__(self, ch: int, r: int = 4):
        super().__init__()
        m = max(8, ch // r)
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(ch, m, 1, bias=True), nn.SiLU(inplace=True),
            nn.Conv2d(m, ch, 1, bias=True), nn.Sigmoid(),
        )

    def forward(self, x):
        return x * self.fc(self.gap(x))


class _MBConv(nn.Module):
    """MobileNetV3-style inverted residual: 1×1 expand → DW 3×3 → SE → 1×1 project."""
    def __init__(self, in_ch: int, out_ch: int, expand_ratio: int = 6):
        super().__init__()
        mid = max(in_ch, out_ch) * expand_ratio
        self.expand = nn.Sequential(
            nn.Conv2d(in_ch, mid, 1, bias=False),
            _safe_gn(mid), nn.SiLU(inplace=True),
        )
        self.dw = nn.Sequential(
            nn.Conv2d(mid, mid, 3, padding=1, groups=mid, bias=False),
            _safe_gn(mid), nn.SiLU(inplace=True),
        )
        self.se = _SE(mid, r=4)
        self.proj = nn.Sequential(
            nn.Conv2d(mid, out_ch, 1, bias=False),
            _safe_gn(out_ch),
        )

    def forward(self, x):
        return self.proj(self.se(self.dw(self.expand(x))))


class WidenerTier2MBConv(nn.Module):
    """Per-stage MBConv block widened to target channels."""
    def __init__(self, in_channels: Sequence[int],
                 target_channels: Sequence[int]):
        super().__init__()
        self.out_channels = tuple(target_channels)
        # Smaller expand for f8/f16 to keep params reasonable
        ratios = [6, 6, 4, 4]
        self.blocks = nn.ModuleList([
            _MBConv(i, o, expand_ratio=r)
            for i, o, r in zip(in_channels, target_channels, ratios)
        ])

    def forward(self, feats):
        return tuple(b(f) for f, b in zip(feats, self.blocks))


class _GhostConv(nn.Module):
    """YOLO-style GhostConv: half the channels via cheap 3×3 DW from the
    primary 1×1 conv. Cheap widening."""
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        primary_ch = (out_ch + 1) // 2
        ghost_ch = out_ch - primary_ch
        self.primary = nn.Sequential(
            nn.Conv2d(in_ch, primary_ch, 1, bias=False),
            _safe_gn(primary_ch), nn.SiLU(inplace=True),
        )
        self.ghost = nn.Sequential(
            nn.Conv2d(primary_ch, ghost_ch, 3, padding=1, groups=primary_ch,
                      bias=False),
            _safe_gn(ghost_ch), nn.SiLU(inplace=True),
        )

    def forward(self, x):
        p = self.primary(x)
        g = self.ghost(p)
        return torch.cat([p, g], dim=1)


class WidenerTier2Ghost(nn.Module):
    """Per-stage GhostConv widening."""
    def __init__(self, in_channels: Sequence[int],
                 target_channels: Sequence[int]):
        super().__init__()
        self.out_channels = tuple(target_channels)
        self.blocks = nn.ModuleList([
            _GhostConv(i, o) for i, o in zip(in_channels, target_channels)
        ])

    def forward(self, feats):
        return tuple(b(f) for f, b in zip(feats, self.blocks))


# -------------------- Tier 3: cross-scale fusion --------------------

class WidenerTier3TopDownFPN(nn.Module):
    """Mini top-down FPN. Lateral 1×1 to a unified channel count, then
    f16 → upsample → add to f8 → upsample → add to f4 → upsample → add
    to f2. Single 3×3 smoothing conv after each fusion.

    Output channels are (fpn_ch, fpn_ch, fpn_ch, fpn_ch).
    """
    def __init__(self, in_channels: Sequence[int], fpn_ch: int = 96):
        super().__init__()
        self.out_channels = (fpn_ch, fpn_ch, fpn_ch, fpn_ch)
        self.lateral = nn.ModuleList([
            nn.Conv2d(c, fpn_ch, 1, bias=False) for c in in_channels
        ])
        self.smooth = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(fpn_ch, fpn_ch, 3, padding=1, bias=False),
                _safe_gn(fpn_ch), nn.SiLU(inplace=True),
            ) for _ in range(4)
        ])

    def forward(self, feats):
        f2, f4, f8, f16 = feats
        l2 = self.lateral[0](f2)
        l4 = self.lateral[1](f4)
        l8 = self.lateral[2](f8)
        l16 = self.lateral[3](f16)
        # top-down
        p16 = self.smooth[3](l16)
        p8 = self.smooth[2](l8 + F.interpolate(p16, size=l8.shape[-2:],
                                                  mode="nearest"))
        p4 = self.smooth[1](l4 + F.interpolate(p8, size=l4.shape[-2:],
                                                  mode="nearest"))
        p2 = self.smooth[0](l2 + F.interpolate(p4, size=l2.shape[-2:],
                                                  mode="nearest"))
        return (p2, p4, p8, p16)


class WidenerTier3BiFPN(nn.Module):
    """Bidirectional FPN: top-down then bottom-up, single iteration with
    weighted feature fusion (EfficientDet-lite). Cheaper than full BiFPN."""
    def __init__(self, in_channels: Sequence[int], fpn_ch: int = 96):
        super().__init__()
        self.out_channels = (fpn_ch, fpn_ch, fpn_ch, fpn_ch)
        self.lateral = nn.ModuleList([
            nn.Conv2d(c, fpn_ch, 1, bias=False) for c in in_channels
        ])
        # 7 fusion convs: 3 top-down + 4 bottom-up (re-use lateral for top)
        self.td_smooth = nn.ModuleList([
            self._smooth(fpn_ch) for _ in range(3)
        ])
        self.bu_smooth = nn.ModuleList([
            self._smooth(fpn_ch) for _ in range(4)
        ])
        # Learnable scalar weights for each fusion (initialised to 1)
        self.w_td = nn.Parameter(torch.ones(3, 2))
        self.w_bu = nn.Parameter(torch.ones(4, 2))
        self.eps = 1e-3

    @staticmethod
    def _smooth(ch):
        return nn.Sequential(
            nn.Conv2d(ch, ch, 3, padding=1, bias=False, groups=ch),
            nn.Conv2d(ch, ch, 1, bias=False),
            _safe_gn(ch), nn.SiLU(inplace=True),
        )

    def _fuse(self, w, *xs):
        w = F.relu(w)
        ws = w / (w.sum() + self.eps)
        return sum(wi * x for wi, x in zip(ws, xs))

    def forward(self, feats):
        f2, f4, f8, f16 = feats
        l2 = self.lateral[0](f2)
        l4 = self.lateral[1](f4)
        l8 = self.lateral[2](f8)
        l16 = self.lateral[3](f16)
        # Top-down (start at l16, fuse downward to l2)
        td16 = l16
        td8 = self.td_smooth[2](self._fuse(
            self.w_td[2], l8, F.interpolate(td16, size=l8.shape[-2:],
                                              mode="nearest")))
        td4 = self.td_smooth[1](self._fuse(
            self.w_td[1], l4, F.interpolate(td8, size=l4.shape[-2:],
                                              mode="nearest")))
        td2 = self.td_smooth[0](self._fuse(
            self.w_td[0], l2, F.interpolate(td4, size=l2.shape[-2:],
                                              mode="nearest")))
        # Bottom-up (start at td2, propagate up — fuse with td/lateral)
        p2 = self.bu_smooth[0](td2)
        p4 = self.bu_smooth[1](self._fuse(
            self.w_bu[1], td4, l4,
            F.interpolate(p2, size=td4.shape[-2:], mode="area")))
        p8 = self.bu_smooth[2](self._fuse(
            self.w_bu[2], td8, l8,
            F.interpolate(p4, size=td8.shape[-2:], mode="area")))
        p16 = self.bu_smooth[3](self._fuse(
            self.w_bu[3], td16, l16,
            F.interpolate(p8, size=td16.shape[-2:], mode="area")))
        return (p2, p4, p8, p16)


# -------------------- Tier 4: norm-replacement --------------------

def apply_gn_replace_(module: nn.Module, max_groups: int = 8) -> None:
    """In-place: walk the module tree and replace every BatchNorm2d with
    a GroupNorm of matching channel count. Loaded BN weights/biases are
    transferred to the GN affine params (running stats are dropped — GN
    has no running stats).

    This addresses the "BN drift on small batches" hypothesis for the
    yolo26n encoder. The pretrained BN affine values still get a chance
    to influence the new GN normalization, just without the instability
    of running stats updating on tiny batches.
    """
    for name, child in module.named_children():
        if isinstance(child, nn.BatchNorm2d):
            ch = child.num_features
            g = 1
            for cand in range(min(max_groups, ch), 0, -1):
                if ch % cand == 0:
                    g = cand
                    break
            gn = nn.GroupNorm(num_groups=g, num_channels=ch,
                                eps=child.eps, affine=child.affine)
            if child.affine:
                gn.weight.data.copy_(child.weight.data)
                gn.bias.data.copy_(child.bias.data)
            setattr(module, name, gn)
        else:
            apply_gn_replace_(child, max_groups)


# -------------------- factory --------------------

# Targets per widener (output channels at f2, f4, f8, f16)
WIDENER_TARGETS = {
    "none":          None,                      # no widening
    "f2_only":       (32, 64, 128, 128),
    "f2_f4":         (32, 96, 128, 128),
    "all_to_s":      (32, 128, 256, 256),
    "dw":            (32, 128, 256, 256),
    "mbconv":        (32, 128, 256, 256),
    "ghostconv":     (32, 128, 256, 256),
    "topdown_fpn":   (96, 96, 96, 96),
    "bifpn":         (96, 96, 96, 96),
    "gn_replace":    None,                      # no widening; encoder mod only
}


def build_widener(widener_type: str, in_channels: Sequence[int],
                  encoder: nn.Module | None = None) -> nn.Module:
    """Build the widener module. For `gn_replace`, also modifies the
    encoder in place to swap BN → GN."""
    if widener_type not in WIDENER_TARGETS:
        raise ValueError(f"unknown widener type: {widener_type!r}; "
                         f"choices: {list(WIDENER_TARGETS)}")
    targets = WIDENER_TARGETS[widener_type]

    if widener_type == "none":
        return WidenerNone(in_channels)
    if widener_type == "gn_replace":
        if encoder is None:
            raise ValueError("gn_replace needs the encoder argument")
        apply_gn_replace_(encoder)
        return WidenerNone(in_channels)
    if widener_type in ("f2_only", "f2_f4", "all_to_s"):
        return WidenerTier1(in_channels, targets)
    if widener_type == "dw":
        return WidenerTier2DW(in_channels, targets)
    if widener_type == "mbconv":
        return WidenerTier2MBConv(in_channels, targets)
    if widener_type == "ghostconv":
        return WidenerTier2Ghost(in_channels, targets)
    if widener_type == "topdown_fpn":
        return WidenerTier3TopDownFPN(in_channels, fpn_ch=96)
    if widener_type == "bifpn":
        return WidenerTier3BiFPN(in_channels, fpn_ch=96)
    raise AssertionError("unreachable")
