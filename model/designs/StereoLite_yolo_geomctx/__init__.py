"""StereoLite_yolo_geomctx design package.

Experimental context-gated slanted tile recurrent stereo. This variant
keeps the YOLO26 matching encoder and dedicated RAFT-style context encoder
from `StereoLite_yolo_ctx`, then makes the tile geometry active inside the
local lookup and gates recurrent updates against local stereo evidence.
"""

from .model import StereoLiteYoloGeomCtx, StereoLiteYoloGeomCtxConfig

# Compatibility aliases for simple script swaps.
StereoLiteYoloCtx = StereoLiteYoloGeomCtx
StereoLiteYoloCtxConfig = StereoLiteYoloGeomCtxConfig

