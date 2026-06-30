"""StereoLite_yolo_ctx_gate design package.

Small ablation variant of `StereoLite_yolo_ctx`: the matching encoder,
context encoder, cost lookup, and ConvGRU are unchanged, but the predicted
tile residuals are scaled by a learned context/confidence gate.
"""

from .model import StereoLiteYoloCtxGate, StereoLiteYoloCtxGateConfig

StereoLiteYoloCtx = StereoLiteYoloCtxGate
StereoLiteYoloCtxConfig = StereoLiteYoloCtxGateConfig

