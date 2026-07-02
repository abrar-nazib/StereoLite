"""StereoLite_yolo_ctx_sru design package.

Selective recurrent update variant of `StereoLite_yolo_ctx_gate`.
"""

from .model import StereoLiteYoloCtxSRU, StereoLiteYoloCtxSRUConfig

StereoLiteYoloCtx = StereoLiteYoloCtxSRU
StereoLiteYoloCtxConfig = StereoLiteYoloCtxSRUConfig
