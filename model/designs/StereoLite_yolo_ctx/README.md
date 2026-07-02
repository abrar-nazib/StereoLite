# StereoLite_yolo_ctx

YOLO26 matching encoder + dedicated **context-encoder stream** inspired by
RAFT-Stereo Fig 1 (the bottom stream in the architecture diagram).

## What it changes vs `StereoLite_yolo/`

The existing `StereoLite_yolo/` chassis re-uses the matching features
`fL` as the context fed to the per-iteration GRU update. RAFT-Stereo uses
a separate encoder for that role. This sibling does the same:

1. A new `ContextEncoder` runs on the **left image only** and emits
   `ctx_ch=32` channels at 1/4 resolution (`ContextEncoder` in
   `model.py`). ~50-100 k params.
2. Those context features are bilinear-upsampled to 1/16, 1/8, 1/4 and:
   - **Initialise the GRU hidden state** (the tile `feat` slot) at 1/16.
   - **Feed the per-iteration GRU input** at every scale.
3. The matching encoder, cost-volume init, plane upsample, convex
   upsample, and iteration counts are unchanged from `StereoLite_yolo/`.

## Why this might help

Cross-domain failure (Middlebury 2014 zero-shot: EPE 5.53 / D1-all 40.1%
on the legacy chassis vs 6.9% for LiteAnyStereo) is the load-bearing
problem. The hypothesis is that the hidden state should carry
**long-range left-image structure** independent of the current disparity
hypothesis — occluding contours, repeated texture, large homogeneous
regions. The matching features are corrupted near occlusions and
repetitive texture because they mix L and R. A dedicated context stream
on L only gives the GRU a stable signal for "what is the structure of
this image" vs "how does this row match".

## Files

- `model.py` — `StereoLiteYoloCtx` + `ContextEncoder` + `ConvexUpsample`.
- `tile_propagate.py` — `TileState`, `TileInit`, `TileRefineCtx`
  (cost-lookup + ConvGRU + context concat), `TileUpsample`.

## How to use

```python
from model.designs.StereoLite_yolo_ctx.model import (
    StereoLiteYoloCtx, StereoLiteYoloCtxConfig)

m = StereoLiteYoloCtx(StereoLiteYoloCtxConfig(backbone="yolo26s"))
d = m(left, right)              # (B, 1, H, W) disparity at full res
d_aux = m(left, right, aux=True)  # also d4, d8, d16, d32 for multi-scale loss
```

## Param budget

Target: ~2-3 M trainable (mid-tier). YOLO26s encoder is ~1.23 M;
context encoder + GRU heads + convex upsample add ~0.7-1.0 M.

## Sibling relationship

- `StereoLite/` — base chassis, edge tier (GhostConv encoder, ~0.87 M).
- `StereoLite_yolo/` — mid tier with YOLO26s encoder, plain tile refine
  (no GRU, no context stream), ~2.06 M.
- `StereoLite_tilegru/` — adds ConvGRU on tile.feat, re-uses fL as GRU
  context, no dedicated context stream.
- `StereoLite_raftlike/` — adds cost-lookup + ConvGRU, still re-uses
  fL as GRU context.
- **`StereoLite_yolo_ctx/` (this folder)** — adds a dedicated
  context-encoder stream on top of the cost-lookup + ConvGRU pattern.

Don't edit `StereoLite/` or `StereoLite_yolo/` for this experiment; this
folder is the canonical place.
