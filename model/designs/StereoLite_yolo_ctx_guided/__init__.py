"""StereoLite_yolo_ctx_guided design package.

`yolo_ctx_gev4` with fail-soft full-resolution guided propagation.
"""

from .model import StereoLiteYoloCtxGuided, StereoLiteYoloCtxGuidedConfig

StereoLiteYoloCtx = StereoLiteYoloCtxGuided
StereoLiteYoloCtxConfig = StereoLiteYoloCtxGuidedConfig
