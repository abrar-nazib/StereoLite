"""StereoLite_yolo_ctx_hrrefine design package.

YOLO context-gated tile model with an extra half-resolution residual
refinement loop before the final full-resolution upsample.
"""

from .model import StereoLiteYoloCtxHRRefine, StereoLiteYoloCtxHRRefineConfig

StereoLiteYoloCtx = StereoLiteYoloCtxHRRefine
StereoLiteYoloCtxConfig = StereoLiteYoloCtxHRRefineConfig
