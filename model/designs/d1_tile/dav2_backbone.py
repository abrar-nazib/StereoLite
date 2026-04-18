"""Frozen DAv2-Small feature backbone for LiteFMStereo.

Exposes two useful feature maps:
  * `raw`   : block-11 ViT patch features at stride 14 (384 channels). Used
              to enrich 1/16 features with semantic priors.
  * `path_3`: DPT-head fused multi-scale features at stride 8 (64 channels
              for vits). Used to enrich 1/8 features going into the cost
              volume — matching operations benefit from DPT's spatial
              detail more than raw ViT patches.

Loaded frozen (24.8 M non-trainable params). At Jetson deploy time the
same weights are quantized to INT8 via TensorRT; since weights never
change, PTQ is deterministic.
"""
from __future__ import annotations

import os
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F


PROJ = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
FS_DIR = os.path.join(PROJ, "model", "teachers", "FoundationStereo")
FS_CKPT = os.path.join(FS_DIR, "pretrained_models", "11-33-40", "model_best_bp2.pth")


_DAV2_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_DAV2_IMAGENET_STD = (0.229, 0.224, 0.225)


class DAv2SmallFrozen(nn.Module):
    """DAv2-Small ViT + DPT head, frozen. Returns a dict:
        'raw'    : (B, embed_dim, Hp, Wp) at patch stride (/14)
        'path_3' : (B, dpt_ch, Hp8, Wp8) at DPT stride /8

    Input: (B, 3, H, W) RGB in [0, 255]. Internal resize to multiple of 14.
    """

    def __init__(self, ckpt_path: str = FS_CKPT,
                 all_block_indices: tuple[int, ...] = (2, 5, 8, 11),
                 embed_dim: int = 384,
                 dpt_ch: int = 64,
                 patch_size: int = 14):
        super().__init__()
        self.embed_dim = embed_dim
        self.dpt_ch = dpt_ch
        self.patch_size = patch_size
        self.all_block_indices = list(all_block_indices)

        mean = torch.tensor(_DAV2_IMAGENET_MEAN).view(1, 3, 1, 1)
        std = torch.tensor(_DAV2_IMAGENET_STD).view(1, 3, 1, 1)
        self.register_buffer("mean", mean, persistent=False)
        self.register_buffer("std", std, persistent=False)

        if FS_DIR not in sys.path:
            sys.path.insert(0, FS_DIR)
        # xFormers enabled — provides memory-efficient attention for DINOv2
        from core.extractor import DepthAnythingFeature

        self.dav2 = DepthAnythingFeature(encoder="vits")

        ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        sd = ck["model"] if "model" in ck else ck
        prefix = "feature.dino."
        own_sd = {k[len(prefix):]: v for k, v in sd.items() if k.startswith(prefix)}
        missing, unexpected = self.dav2.load_state_dict(own_sd, strict=False)
        if len(missing) > 50:
            raise RuntimeError(
                f"DAv2 ckpt load looks wrong: missing={len(missing)} "
                f"unexpected={len(unexpected)}")

        for p in self.dav2.parameters():
            p.requires_grad = False
        self.dav2.eval()

    def train(self, mode: bool = True):
        super().train(mode)
        self.dav2.eval()
        for p in self.dav2.parameters():
            p.requires_grad = False
        return self

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        x = x / 255.0
        x = (x - self.mean) / self.std

        H, W = x.shape[-2:]
        H14 = max(self.patch_size * round(H / self.patch_size), self.patch_size)
        W14 = max(self.patch_size * round(W / self.patch_size), self.patch_size)
        if (H14, W14) != (H, W):
            x = F.interpolate(x, size=(H14, W14), mode="bilinear",
                              align_corners=False)
        Hp, Wp = H14 // self.patch_size, W14 // self.patch_size

        vit = self.dav2.depth_anything.pretrained
        features = vit.get_intermediate_layers(
            x, self.all_block_indices, return_class_token=True)
        # features is a tuple of (patch_tokens, cls_token) per block.

        # Raw block-11 features at 1/14
        patch_11 = features[-1][0]                                 # (B, N, C)
        raw = patch_11.transpose(1, 2).reshape(
            patch_11.shape[0], self.embed_dim, Hp, Wp).float()

        # DPT head: returns (out, path_1, path_2, path_3, path_4, disp)
        dpt_out = self.dav2.depth_anything.depth_head(
            features, Hp, Wp, return_intermediate=True)
        path_3 = dpt_out[3].float()                                # 1/8 stride

        return {"raw": raw, "path_3": path_3}
