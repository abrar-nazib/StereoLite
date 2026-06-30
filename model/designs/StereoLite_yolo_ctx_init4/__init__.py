"""StereoLite_yolo_ctx_init4 design package.

`yolo_ctx_gate` plus a fresh 1/4-resolution cost warm start before the final
1/4 recurrent refinement stage.
"""

from .model import StereoLiteYoloCtxInit4, StereoLiteYoloCtxInit4Config

StereoLiteYoloCtx = StereoLiteYoloCtxInit4
StereoLiteYoloCtxConfig = StereoLiteYoloCtxInit4Config
