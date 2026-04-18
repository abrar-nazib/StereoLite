"""Wrapper to load and run pretrained TinyHITNet (Scene Flow checkpoint) as
a baseline for cycle.py side-by-side comparisons.

Usage (programmatic):
    from hitnet_baseline import HitnetBaseline
    h = HitnetBaseline(device='cuda')
    disp = h(left_tensor_uint8, right_tensor_uint8)   # both (1,3,H,W) uint8/float

Inputs are expected as (B, 3, H, W) RGB tensors (any precision, scaled 0-255).
The wrapper handles HITNet's *2-1 normalisation internally.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import torch
import torch.nn as nn

PROJ = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
HITNET_DIR = os.path.join(PROJ, "model", "teachers", "TinyHITNet")
HITNET_CKPT = os.path.join(HITNET_DIR, "ckpt", "hitnet_sf_finalpass.ckpt")


class HitnetBaseline:
    """Loads TinyHITNet pretrained on Scene Flow finalpass."""

    def __init__(self, device: torch.device | str = "cuda", which: str = "HITNet_SF"):
        self.device = torch.device(device)
        # Add HITNet dir to path so its `models` and `dataset` subpackages import.
        if HITNET_DIR not in sys.path:
            sys.path.insert(0, HITNET_DIR)
        from models import build_model

        class _A:
            pass

        a = _A()
        a.model = which
        self.model = build_model(a).to(self.device).eval()

        ckpt = torch.load(HITNET_CKPT, map_location=self.device, weights_only=False)
        if "state_dict" in ckpt:
            sd = {k.replace("model.", "", 1): v for k, v in ckpt["state_dict"].items()
                  if k.startswith("model.")}
            self.model.load_state_dict(sd, strict=False)
        else:
            self.model.load_state_dict(ckpt, strict=False)
        self.n_params = sum(p.numel() for p in self.model.parameters())

    @torch.no_grad()
    def __call__(self, left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
        """left/right: (B, 3, H, W) RGB uint8 or float in [0, 255].
        Returns disparity (B, 1, H, W) at the same resolution as input."""
        if left.dtype != torch.float32:
            left = left.float()
            right = right.float()
        left = left.to(self.device) / 255.0
        right = right.to(self.device) / 255.0
        # HITNet expects in [-1, 1]
        left_n = left * 2 - 1
        right_n = right * 2 - 1
        out = self.model(left_n, right_n)
        if isinstance(out, dict):
            disp = out["disp"]
        elif isinstance(out, (list, tuple)):
            disp = out[0]
        else:
            disp = out
        if disp.dim() == 3:
            disp = disp.unsqueeze(1)
        return disp


if __name__ == "__main__":
    import time
    device = torch.device("cuda")
    h = HitnetBaseline(device=device)
    print(f"loaded HITNet SF: {h.n_params/1e6:.2f} M params")
    L = torch.rand(1, 3, 384, 640, device=device) * 255
    R = torch.rand(1, 3, 384, 640, device=device) * 255
    for _ in range(3):
        d = h(L, R)
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(5):
        d = h(L, R)
    torch.cuda.synchronize()
    print(f"latency: {(time.time()-t0)/5*1000:.1f} ms / forward")
    print(f"output: {tuple(d.shape)}, range [{d.min().item():.1f}, {d.max().item():.1f}]")
