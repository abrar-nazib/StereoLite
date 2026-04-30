"""Truncated YOLO26 backbone as a stereo feature encoder.

Loads `yolo26n.pt` or `yolo26s.pt` via ultralytics, slices the first 7
modules of the backbone (layers 0-6 of `model.model.model`), and exposes
features at strides (2, 4, 8, 16) — the same API as
`MobileNetV2Encoder` / `TileFeatureEncoder` so it drops into StereoLite.

Layer mapping (from a forward-trace, input 640x640):
    [0] Conv stride-2  -> /2,  ch=(16 / 32)   <- f2 (raw stem)
    [1] Conv stride-2  -> /4
    [2] C3k2           -> /4,  ch=(64 / 128)  <- f4
    [3] Conv stride-2  -> /8
    [4] C3k2           -> /8,  ch=(128 / 256) <- f8
    [5] Conv stride-2  -> /16
    [6] C3k2           -> /16, ch=(128 / 256) <- f16

The (n / s) channel pair is for yolo26n / yolo26s. Layers 7+ build the
1/32 stage, neck (FPN+PAN), and detection head — all dropped.

Layers 0-6 form a strict linear chain (every layer's `from` is -1), so
sequential forward works. Confirmed by tracing the loaded model.
"""
from __future__ import annotations

import torch
import torch.nn as nn


_VARIANT_INFO = {
    # variant: (out_channels, expected_truncated_params)
    "yolo26n": ((16, 64, 128, 128), 309_600),
    "yolo26s": ((32, 128, 256, 256), 1_233_088),
}


class YoloTruncatedEncoder(nn.Module):
    """Truncated YOLO26 backbone exposing (f2, f4, f8, f16).

    Args:
        variant: "yolo26n" or "yolo26s".
        weights_path: optional explicit path to the .pt checkpoint. If None,
            ultralytics will auto-download `{variant}.pt` to the current
            working directory.
    """

    out_channels: tuple[int, int, int, int]

    def __init__(self, variant: str = "yolo26n",
                 weights_path: str | None = None,
                 freeze: bool = False):
        super().__init__()
        if variant not in _VARIANT_INFO:
            raise ValueError(
                f"variant must be one of {list(_VARIANT_INFO)}, got {variant!r}")
        self.variant = variant
        self.out_channels = _VARIANT_INFO[variant][0]

        from ultralytics import YOLO
        ckpt = weights_path or f"{variant}.pt"
        full = YOLO(ckpt).model.eval()

        # Keep only the first 7 modules (linear chain, all from=-1).
        # Wrapping in nn.ModuleList preserves the parent module hierarchy
        # used by the layers' forward methods.
        self.layers = nn.ModuleList([full.model[i] for i in range(7)])

        # ultralytics loads params with requires_grad=False (inference
        # mode default). Re-enable so the encoder is trainable end-to-end,
        # unless the caller explicitly asks to freeze.
        for p in self.parameters():
            p.requires_grad = not freeze

    def forward(self, x: torch.Tensor):
        # f2 = layer 0 output; f4 = layer 2 output; f8 = layer 4; f16 = layer 6.
        f2  = self.layers[0](x)
        x4_ = self.layers[1](f2)
        f4  = self.layers[2](x4_)
        x8_ = self.layers[3](f4)
        f8  = self.layers[4](x8_)
        x16 = self.layers[5](f8)
        f16 = self.layers[6](x16)
        return f2, f4, f8, f16
