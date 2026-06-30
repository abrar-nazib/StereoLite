"""StereoLite_yolo_ctx_sharp design package.

Boundary-aware sibling of `StereoLite_yolo_ctx_gate`. It keeps the gated
context recurrent stereo core and adds a lightweight full-resolution sharp
refinement tail after convex upsampling.
"""

from .model import StereoLiteYoloCtxSharp, StereoLiteYoloCtxSharpConfig

StereoLiteYoloCtx = StereoLiteYoloCtxSharp
StereoLiteYoloCtxConfig = StereoLiteYoloCtxSharpConfig
