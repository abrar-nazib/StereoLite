"""StereoLite_yolo_ctx_gev4_opt — efficiency-optimized yolo_ctx_gev4.

Output-equivalent optimizations F1/F2/F4/F5/F7 (see tile_propagate.py)
plus one flag-gated ACCURACY-AFFECTING variant:

  F3 (cfg.narrow_gev=True): the 1/4 GEV samples 2*gev_half_range+1 bins
     centered on the incoming tile disparity (detached) instead of a full
     64-bin volume — the validated cascade_cv_4 narrowing pattern. NOT
     output-equivalent; must win a matched A/B before adoption.

`convert_state_dict()` maps original yolo_ctx_gev4 checkpoints into this
layout so output equivalence can be verified weight-for-weight.
Wiring otherwise identical to StereoLite_yolo_ctx_gev4/model.py.
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

from .tile_propagate import (TileState, TileInit, TileRefineCtx, TileUpsample,
                             _groupwise_corr_volume_padded, _base_grid,
                             convert_refine_keys)


def _gn(ch: int, groups: int = 8) -> nn.GroupNorm:
    return _safe_gn(ch, groups)


# ----------------------------------------------------------------------- #
# Context encoder (RAFT-Stereo Fig 1, bottom stream)
# ----------------------------------------------------------------------- #
class ContextEncoder(nn.Module):
    """Lightweight GhostConv encoder on the LEFT image, output at 1/4.

    Returns (B, ctx_ch, H/4, W/4). ~50-100 k params depending on
    base/out channel choice. The model bilinear-upsamples these features
    to whatever scale the GRU needs (1/16, 1/8, 1/4).

    Why 1/4 (not 1/16): context in RAFT-Stereo is at 1/4 because the
    hidden state must capture long-range image structure (edges, large
    homogeneous regions, occluding contours) and downsampling past 1/4
    loses that. Bilinear upsample from 1/4 to 1/16 is cheap.
    """

    def __init__(self, base: int = 24, out_ch: int = 32):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, base, 7, stride=2, padding=3, bias=False),
            _gn(base), nn.SiLU(inplace=True),
        )
        # 1/4: GhostConv stage
        self.s4 = nn.Sequential(
            GhostConv(base, 2 * base, k=3, s=2),
            SqueezeExcitation(2 * base),
            GhostConv(2 * base, 2 * base, k=3, s=1),
            SqueezeExcitation(2 * base),
        )
        # Project to out_ch
        self.proj = nn.Sequential(
            nn.Conv2d(2 * base, out_ch, 1, bias=False),
            _gn(out_ch), nn.SiLU(inplace=True),
        )
        self.out_channels = out_ch

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x is in [0, 255]; normalise to [0, 1] like the original GEV4 run.
        x = x / 255.0
        x = self.stem(x)         # 1/2
        x = self.s4(x)           # 1/4
        x = self.proj(x)         # 1/4, out_ch
        return x


# ----------------------------------------------------------------------- #
# Convex upsample (copied verbatim from StereoLite_yolo)
# ----------------------------------------------------------------------- #
class ConvexUpsample(nn.Module):
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


# F1: cost-volume construction shared with tile_propagate (padded views).
_groupwise_corr_volume = _groupwise_corr_volume_padded


def _groupwise_corr_volume_around(fL: torch.Tensor, fR: torch.Tensor,
                                  d_center: torch.Tensor,
                                  half_range: int, groups: int) -> torch.Tensor:
    """F3: narrow group-wise volume sampled at d_center + {-hr..+hr} via one
    batched grid_sample (same mechanics as _correlation_lookup_batched but
    keeping the (B, G, D, H, W) volume shape for the 3D regularizer)."""
    B, C, H, W = fL.shape
    g = groups
    while C % g != 0 and g > 1:
        g -= 1
    cg = C // g
    D = 2 * half_range + 1
    xx, yy = _base_grid(B, H, W, fL.device, fL.dtype)
    gy = (yy / max(H - 1, 1) * 2 - 1).expand(B, H, W)
    d = d_center.squeeze(1)
    gxs = [(xx.expand(B, H, W) - (d + (i - half_range))) / max(W - 1, 1) * 2 - 1
           for i in range(D)]
    grid = torch.stack([torch.cat(gxs, dim=1), gy.repeat(1, D, 1)], dim=-1)
    fR_all = F.grid_sample(fR, grid, align_corners=True,
                           padding_mode="zeros").view(B, C, D, H, W)
    fL_g = fL.view(B, g, cg, 1, H, W)
    return (fL_g * fR_all.view(B, g, cg, D, H, W)).mean(dim=2)


class GeometryEncoding4(nn.Module):
    """Tiny IGEV-style 1/4 cost regularizer.

    Input: group-wise correlation volume (B, G, D, H/4, W/4).
    Output: soft-argmin disparity at 1/4 scale, confidence, and a 2D geometry
    feature obtained by expectation over the disparity distribution.
    """

    def __init__(self, feat_ch: int, max_disp: int = 64, groups: int = 8,
                 geo_ch: int = 16, hidden: int = 16,
                 narrow: bool = False, half_range: int = 16):
        super().__init__()
        g = groups
        while feat_ch % g != 0 and g > 1:
            g -= 1
        self.groups = g
        self.max_disp = max_disp
        self.narrow = narrow
        self.half_range = half_range
        n_bins = (2 * half_range + 1) if narrow else max_disp
        self.reg = nn.Sequential(
            nn.Conv3d(g, hidden, 3, padding=1, bias=False),
            _gn(hidden), nn.SiLU(inplace=True),
            nn.Conv3d(hidden, hidden, 3, padding=1, bias=False),
            _gn(hidden), nn.SiLU(inplace=True),
            nn.Conv3d(hidden, hidden, 3, padding=1, bias=False),
            _gn(hidden), nn.SiLU(inplace=True),
        )
        self.logits = nn.Conv3d(hidden, 1, 3, padding=1)
        self.geo_proj = nn.Sequential(
            nn.Conv2d(hidden, geo_ch, 1, bias=False),
            _gn(geo_ch), nn.SiLU(inplace=True),
        )
        idx = (torch.arange(n_bins, dtype=torch.float32) - half_range
               ) if narrow else torch.arange(max_disp, dtype=torch.float32)
        self.register_buffer("disp_idx", idx.view(1, n_bins, 1, 1),
                             persistent=False)

    def forward(self, fL: torch.Tensor, fR: torch.Tensor,
                d_center: torch.Tensor | None = None):
        if self.narrow:
            assert d_center is not None
            cv = _groupwise_corr_volume_around(
                fL, fR, d_center.detach(), self.half_range, self.groups)
        else:
            cv = _groupwise_corr_volume(fL, fR, self.max_disp, self.groups)
        gv = self.reg(cv)
        logits = self.logits(gv).squeeze(1)
        prob = F.softmax(logits, dim=1)
        d_exp = (prob * self.disp_idx.to(prob.dtype)).sum(dim=1, keepdim=True)
        d = (d_center.detach() + d_exp) if self.narrow else d_exp
        if self.narrow:
            d = F.relu(d)
        conf = prob.max(dim=1, keepdim=True).values
        geo = (gv * prob.unsqueeze(1)).sum(dim=2)
        geo = self.geo_proj(geo)
        return d, conf, geo


@dataclass
class StereoLiteYoloCtxGEV4Config:
    base_ch: int = 24
    tile_feat_ch: int = 32
    ctx_ch: int = 32
    iters_16: int = 2
    iters_8: int = 3
    iters_4: int = 3
    init_max_disp: int = 24
    gev4_max_disp: int = 64
    gev4_ch: int = 16
    gev4_hidden: int = 16
    init_groups: int = 8
    refine_hidden: int = 48
    cost_half_range: int = 2
    backbone: str = "yolo26s"     # mid-tier default; can switch to yolo26n
    backbone_pretrained: bool = True
    # ---- efficiency knobs ----
    opt_static_ctx: bool = True   # F4: hoist static (fL, ctx) conv work
    narrow_gev: bool = False      # F3: GEV around tile.d (ACCURACY-AFFECTING)
    gev_half_range: int = 16      # F3 bins = 2*hr+1 (33 vs original 64)
    # ---- sharp tail (pre-rahi costlookup output path; ACCURACY-AFFECTING) ----
    # True: drop ConvexUpsample; plane-upsample 1/4 -> 1/2, run TileRefineCtx
    # at 1/2 for iters_2 passes, plane-upsample 1/2 -> full. Targets the
    # bad-0.5 / visual-sharpness axis (hard plane-equation discontinuities,
    # finest matching evidence at 1/2 instead of 1/4) at a latency cost.
    sharp_tail: bool = False
    iters_2: int = 2
    # ---- blur bundle-1 (docs/deblurring_plan.md Fix 1; ACCURACY-AFFECTING) ----
    init_topk: int = 0            # 3 = CoEx top-k soft-argmin at TileInit
    # ---- blur Fix 2: plane-equation rendering + learned edge gate ----
    # Renders full-res disparity from the 1/4 slanted-tile state via the
    # HITNet plane equation (hard tile-boundary discontinuities), blended
    # with the ConvexUpsample path by a per-pixel gate conditioned on the
    # left image + both candidates. Gate bias starts at the convex path
    # (fail-soft). Requires slant supervision to be trustworthy (harness
    # --slant_w with robust plane-fit GT + |d_err|<1 gating).
    plane_render: bool = False
    # ---- blur Fix 3: SMD-Nets-style bimodal Laplacian aux head ----
    # 1x1-conv MLP over bilinearly-upsampled 1/4 features + tile state ->
    # (pi, mu1_off, mu2_off, b1, b2) at full res. AUXILIARY: supervised by
    # NLL (harness --bimodal_w) but d_final stays the main path; the
    # bimodal mode-pick output is exposed as "d_bimodal" for eval.
    bimodal_head: bool = False


class StereoLiteYoloCtxGEV4(nn.Module):
    def __init__(self, cfg: StereoLiteYoloCtxGEV4Config | None = None):
        super().__init__()
        self.cfg = cfg or StereoLiteYoloCtxGEV4Config()

        # Matching encoder (left+right, shared)
        self.fnet = YoloTruncatedEncoder(variant=self.cfg.backbone)
        ch2, ch4, ch8, ch16 = self.fnet.out_channels

        # Context encoder (LEFT image only)
        self.ctxnet = ContextEncoder(base=self.cfg.base_ch,
                                      out_ch=self.cfg.ctx_ch)

        # Cost-volume init at 1/16 (h emitted as zeros, model replaces)
        g = self.cfg.init_groups
        while ch16 % g != 0 and g > 1:
            g -= 1
        self.init_tile = TileInit(feat_ch=ch16,
                                   max_disp=self.cfg.init_max_disp,
                                   groups=g,
                                   feat_out=self.cfg.tile_feat_ch,
                                   topk=self.cfg.init_topk)

        # Per-scale refine heads — one instance per scale because channel
        # counts differ; weights are not shared.
        mk = dict(tile_feat_ch=self.cfg.tile_feat_ch,
                  hidden=self.cfg.refine_hidden,
                  half_range=self.cfg.cost_half_range, groups=g,
                  ctx_ch=self.cfg.ctx_ch,
                  opt_static_ctx=self.cfg.opt_static_ctx)
        self.refine_16 = TileRefineCtx(feat_ch=ch16, **mk)
        self.refine_8 = TileRefineCtx(feat_ch=ch8, **mk)
        self.refine_4 = TileRefineCtx(feat_ch=ch4, **mk)
        self.gev4 = GeometryEncoding4(
            feat_ch=ch4, max_disp=self.cfg.gev4_max_disp,
            groups=g, geo_ch=self.cfg.gev4_ch,
            hidden=self.cfg.gev4_hidden,
            narrow=self.cfg.narrow_gev,
            half_range=self.cfg.gev_half_range)
        self.gev4_fuse = nn.Sequential(
            nn.Conv2d(self.cfg.ctx_ch + self.cfg.gev4_ch + 4,
                      self.cfg.refine_hidden, 3, padding=1, bias=False),
            _gn(self.cfg.refine_hidden), nn.SiLU(inplace=True),
            nn.Conv2d(self.cfg.refine_hidden, 1, 1),
        )
        self.gev4_ctx = nn.Conv2d(self.cfg.gev4_ch, self.cfg.ctx_ch, 1)
        # Fail-soft: start very close to ctx_gate, then learn to use GEV.
        nn.init.constant_(self.gev4_fuse[-1].bias, -4.0)

        # Plane upsamples (no trainable weights)
        self.up_16_to_8 = TileUpsample(scale_factor=2)
        self.up_8_to_4 = TileUpsample(scale_factor=2)

        if self.cfg.sharp_tail:
            # costlookup-style tail: refine at 1/2, plane-eq to full.
            self.refine_2 = TileRefineCtx(feat_ch=ch2, **mk)
            self.up_4_to_2 = TileUpsample(scale_factor=2)
            self.up_2_to_1 = TileUpsample(scale_factor=2)
            self.up_final_4_to_2 = None
            self.up_final_2_to_1 = None
        else:
            self.refine_2 = None
            # Final learned 4x upsample 1/4 → full, two 2x convex steps
            self.up_final_4_to_2 = ConvexUpsample(feat_ch=ch4, scale=2)
            self.up_final_2_to_1 = ConvexUpsample(feat_ch=ch2, scale=2)

        if self.cfg.plane_render:
            # per-pixel gate between plane-rendered and convex-upsampled
            # disparity; input = [left(3), d_plane(1), d_convex(1)]
            self.plane_gate = nn.Sequential(
                nn.Conv2d(5, 16, 3, padding=1, bias=False),
                _gn(16), nn.SiLU(inplace=True),
                nn.Conv2d(16, 1, 1),
            )
            nn.init.constant_(self.plane_gate[-1].bias, -2.0)  # start convex
        else:
            self.plane_gate = None

        if self.cfg.bimodal_head:
            # SMD-Nets-style aux head at full res over bilinearly upsampled
            # 1/4 features + tile state (continuous-query analogue).
            bm_in = ch4 + self.cfg.ctx_ch + 4  # fL4 + ctx4 + (d, sx, sy, conf)
            self.bimodal = nn.Sequential(
                nn.Conv2d(bm_in, 32, 1, bias=False),
                _gn(32), nn.SiLU(inplace=True),
                nn.Conv2d(32, 32, 1, bias=False),
                _gn(32), nn.SiLU(inplace=True),
                nn.Conv2d(32, 5, 1),   # pi, mu1_off, mu2_off, log_b1, log_b2
            )
        else:
            self.bimodal = None

    def forward(self, left: torch.Tensor, right: torch.Tensor,
                aux: bool = False):
        # --- Matching features (L+R shared encoder, batched) ---
        feats = self.fnet(torch.cat([left, right], dim=0))
        fL2,  fR2  = feats[0].chunk(2, dim=0)
        fL4,  fR4  = feats[1].chunk(2, dim=0)
        fL8,  fR8  = feats[2].chunk(2, dim=0)
        fL16, fR16 = feats[3].chunk(2, dim=0)

        # --- Context features (LEFT image only) at 1/4 ---
        ctx4 = self.ctxnet(left)
        # Bilinear-upsample context to each scale the GRU needs it at.
        ctx16 = F.interpolate(ctx4, size=fL16.shape[-2:], mode="bilinear",
                              align_corners=False)
        ctx8  = F.interpolate(ctx4, size=fL8.shape[-2:],  mode="bilinear",
                              align_corners=False)

        # --- 1/16: init h from ctx16, then iterate ---
        tile = self.init_tile(fL16, fR16)
        tile = TileState(d=tile.d, sx=tile.sx, sy=tile.sy,
                          feat=ctx16, conf=tile.conf)
        d32 = tile.d
        # F4: per-scale static conv contributions, computed once.
        st16 = self.refine_16.precompute_static(fL16, ctx16)
        for _ in range(self.cfg.iters_16):
            tile = self.refine_16(tile, fL16, fR16, ctx16, static=st16)
        d16 = tile.d

        # --- 1/16 → 1/8 via plane equation, then iterate ---
        tile = self.up_16_to_8(tile, target_hw=fL8.shape[-2:])
        d8_cv = tile.d
        st8 = self.refine_8.precompute_static(fL8, ctx8)
        for _ in range(self.cfg.iters_8):
            tile = self.refine_8(tile, fL8, fR8, ctx8, static=st8)
        d8 = tile.d

        # --- 1/8 → 1/4, then iterate ---
        tile = self.up_8_to_4(tile, target_hw=fL4.shape[-2:])
        d_gev4, conf_gev4, geo4 = self.gev4(
            fL4, fR4, d_center=tile.d if self.cfg.narrow_gev else None)
        gev_delta = (d_gev4 - tile.d).abs()
        fuse_in = torch.cat([ctx4, geo4, tile.conf, conf_gev4,
                             gev_delta, tile.d], dim=1)
        gev_w = torch.sigmoid(self.gev4_fuse(fuse_in))
        tile = TileState(
            d=F.softplus(tile.d + gev_w * (d_gev4 - tile.d)),
            sx=tile.sx * (1.0 - gev_w),
            sy=tile.sy * (1.0 - gev_w),
            feat=ctx4 + 0.1 * self.gev4_ctx(geo4),
            conf=torch.maximum(tile.conf, conf_gev4),
        )
        st4 = self.refine_4.precompute_static(fL4, ctx4)
        for _ in range(self.cfg.iters_4):
            tile = self.refine_4(tile, fL4, fR4, ctx4, static=st4)
        d4_out = tile.d

        if self.cfg.sharp_tail:
            # --- costlookup tail: plane-up to 1/2, refine at 1/2 with real
            #     stereo evidence, plane-equation upsample to full (hard
            #     tile-boundary discontinuities, no convex blending) ---
            ctx2 = F.interpolate(ctx4, size=fL2.shape[-2:], mode="bilinear",
                                 align_corners=False)
            tile = self.up_4_to_2(tile, target_hw=fL2.shape[-2:])
            st2 = self.refine_2.precompute_static(fL2, ctx2)
            for _ in range(self.cfg.iters_2):
                tile = self.refine_2(tile, fL2, fR2, ctx2, static=st2)
            d_half = tile.d
            tile_full = self.up_2_to_1(tile, target_hw=left.shape[-2:])
            d_full = tile_full.d
        else:
            # --- Final convex upsample 1/4 → 1/2 → full ---
            d_half = self.up_final_4_to_2(d4_out, fL4)
            d_full = self.up_final_2_to_1(d_half, fL2)
        if d_full.shape[-2:] != left.shape[-2:]:
            d_full = F.interpolate(d_full, size=left.shape[-2:],
                                    mode="bilinear", align_corners=True)

        gate_mean_pr = None
        if self.plane_gate is not None:
            # HITNet plane-equation rendering: each 1/4 tile spawns its 4x4
            # full-res patch via d + sx*dx + sy*dy (offsets in 1/4-pixel
            # units), x4 disparity unit conversion. Nearest upsample keeps
            # tile-boundary discontinuities HARD.
            H, W = left.shape[-2:]
            d_nn = F.interpolate(tile.d, size=(H, W), mode="nearest")
            sx_nn = F.interpolate(tile.sx, size=(H, W), mode="nearest")
            sy_nn = F.interpolate(tile.sy, size=(H, W), mode="nearest")
            xs = torch.arange(W, device=left.device, dtype=d_nn.dtype)
            ys = torch.arange(H, device=left.device, dtype=d_nn.dtype)
            offx = ((xs % 4) - 1.5).view(1, 1, 1, W) / 4.0
            offy = ((ys % 4) - 1.5).view(1, 1, H, 1) / 4.0
            d_plane = 4.0 * (d_nn + sx_nn * offx + sy_nn * offy)
            g = torch.sigmoid(self.plane_gate(
                torch.cat([left, d_plane, d_full], dim=1)))
            gate_mean_pr = g.mean()
            d_full = g * d_plane + (1.0 - g) * d_full

        bimodal_out = None
        if self.bimodal is not None:
            H, W = left.shape[-2:]
            up = lambda x: F.interpolate(x, size=(H, W), mode="bilinear",
                                         align_corners=False)
            bm_in = torch.cat([up(fL4), up(ctx4), up(tile.d) * 4.0,
                               up(tile.sx), up(tile.sy), up(tile.conf)],
                              dim=1)
            pi_l, mu1o, mu2o, lb1, lb2 = self.bimodal(bm_in).chunk(5, dim=1)
            base = up(tile.d) * 4.0
            pi = torch.sigmoid(pi_l)
            mu1, mu2 = base + mu1o, base + mu2o
            bimodal_out = {
                "pi": pi, "mu1": mu1, "mu2": mu2,
                "b1": F.softplus(lb1) + 0.1,
                "b2": F.softplus(lb2) + 0.1,
                # SMD-Nets inference: pick the dominant mode, never average.
                "d_bimodal": torch.where(pi > 0.5, mu1, mu2),
            }

        if aux:
            return {
                "d_final": d_full,
                "d_half": d_half,
                "d4": d4_out,
                "d4_gev": d_gev4,
                "gev4_w": gev_w,
                "d8": d8,
                "d8_cv": d16,
                "d16": d16,
                "d32": d32,
                "init_logits": self.init_tile.last_logits,
                "sx4": tile.sx, "sy4": tile.sy,
                "plane_gate_mean": gate_mean_pr,
                "bimodal": bimodal_out,
            }
        return d_full


def convert_state_dict(orig_sd: dict,
                       cfg: "StereoLiteYoloCtxGEV4Config | None" = None) -> dict:
    """Map an original StereoLite_yolo_ctx_gev4 checkpoint into the optimized
    parameter layout (F4 split + F5 fusion). Everything outside the three
    TileRefineCtx blocks passes through unchanged. Only valid for
    narrow_gev=False (the GEV weights transfer as-is at max_disp=64)."""
    cfg = cfg or StereoLiteYoloCtxGEV4Config()
    # channel dims must match the encoder used at training time
    from StereoLite_yolo.yolo_encoder import _VARIANT_INFO
    ch2, ch4, ch8, ch16 = _VARIANT_INFO[cfg.backbone][0]
    g = cfg.init_groups
    while ch16 % g != 0 and g > 1:
        g -= 1
    cost_ch = g * (2 * cfg.cost_half_range + 1)
    out = {}
    refine_dims = {"refine_16.": ch16, "refine_8.": ch8, "refine_4.": ch4}
    handled_prefixes = tuple(refine_dims.keys())
    for prefix, fc in refine_dims.items():
        out.update(convert_refine_keys(orig_sd, prefix, feat_ch=fc,
                                       tile_feat_ch=cfg.tile_feat_ch,
                                       cost_ch=cost_ch, ctx_ch=cfg.ctx_ch))
    for k, v in orig_sd.items():
        if not k.startswith(handled_prefixes):
            out[k] = v
    return out


# Compatibility aliases for existing yolo_ctx import patterns.
StereoLiteYoloCtx = StereoLiteYoloCtxGEV4
StereoLiteYoloCtxConfig = StereoLiteYoloCtxGEV4Config
