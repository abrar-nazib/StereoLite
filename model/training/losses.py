"""Loss functions and common metrics used by every approach in this project."""
from __future__ import annotations

import torch


def sequence_loss(preds: list[torch.Tensor], gt: torch.Tensor,
                  valid: torch.Tensor | None = None,
                  gamma: float = 0.9, max_disp: float = 700.0) -> torch.Tensor:
    """Exponentially-weighted L1 over a list of intermediate disparity preds.

    Matches the RAFT-Stereo training recipe.
    """
    n = len(preds)
    if valid is None:
        valid = (gt.abs() < max_disp).float()
    loss = torch.tensor(0.0, device=gt.device, dtype=gt.dtype)
    for i, p in enumerate(preds):
        w = gamma ** (n - 1 - i)
        err = ((p - gt).abs() * valid)
        loss = loss + w * err.mean()
    return loss


def epe(pred: torch.Tensor, gt: torch.Tensor,
        valid: torch.Tensor | None = None) -> torch.Tensor:
    err = (pred - gt).abs()
    if valid is not None:
        return (err * valid).sum() / valid.sum().clamp(min=1.0)
    return err.mean()


def bad_px(pred: torch.Tensor, gt: torch.Tensor, threshold: float = 1.0,
           valid: torch.Tensor | None = None) -> torch.Tensor:
    err = (pred - gt).abs()
    bad = (err > threshold).float()
    if valid is not None:
        return (bad * valid).sum() / valid.sum().clamp(min=1.0)
    return bad.mean()
