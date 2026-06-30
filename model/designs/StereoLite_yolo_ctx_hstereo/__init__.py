"""StereoLite_yolo_ctx_hstereo design package.

GEV4 with an added true 1/2-resolution stereo refinement stage.
"""

from .model import StereoLiteYoloCtxHStereo, StereoLiteYoloCtxHStereoConfig

StereoLiteYoloCtx = StereoLiteYoloCtxHStereo
StereoLiteYoloCtxConfig = StereoLiteYoloCtxHStereoConfig
