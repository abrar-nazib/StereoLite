# StereoLite_yolo_geomctx

YOLO26 matching encoder + RAFT-style left context stream + geometry-aware
slanted tile recurrent refinement.

This is the test variant for the research idea:

> use HITNet's explicit slanted tile state, but update it with a compact
> context-guided recurrent module instead of copying the full HITNet pipeline.

## What it changes vs `StereoLite_yolo_ctx/`

1. **Slant-aware local lookup**
   `sx` and `sy` now affect the local matching lookup. For each tile, the
   updater samples a small support patch with:

   ```text
   d_patch(dx, dy) = d + sx * dx + sy * dy
   ```

   and evaluates local disparity offsets around that slanted plane.

2. **Local stereo rescue**
   The slanted lookup is converted into a small local disparity correction
   using softmax over the offset scores. This gives the model a cheap
   high-resolution correction source without implementing HITNet's full
   multi-scale exhaustive initializer.

3. **Context/confidence gate**
   A learned gate blends the ConvGRU proposal with the local stereo rescue:

   ```text
   tile_out = gate * recurrent_update + (1 - gate) * local_rescue
   ```

   The gate sees the context feature, old confidence, local confidence,
   and disagreement between the recurrent and local disparity proposals.

## Ablation knobs

- `slant_patch_radius=1`: default 3x3 slanted support patch.
- `slant_patch_radius=0`: center-only lookup; useful to estimate the value
  of true slant-aware support.
- `cost_half_range=2`: searches offsets `[-2, -1, 0, 1, 2]` around the
  current slanted plane.
- `iters_16`, `iters_8`, `iters_4`: recurrent update budget per scale.

## How to use

```python
from model.designs.StereoLite_yolo_geomctx.model import (
    StereoLiteYoloGeomCtx, StereoLiteYoloGeomCtxConfig)

cfg = StereoLiteYoloGeomCtxConfig(backbone="yolo26s")
m = StereoLiteYoloGeomCtx(cfg)
d = m(left, right)
aux = m(left, right, aux=True)
```

Compatibility aliases are also exported from the package:

```python
from model.designs.StereoLite_yolo_geomctx import (
    StereoLiteYoloCtx, StereoLiteYoloCtxConfig)
```

## Paper framing

This should be described as **context-gated slanted tile recurrent stereo**,
not as a HITNet reimplementation. It borrows HITNet's explicit geometric
state, RAFT's context-guided recurrence pattern, and the StereoLite/YOLO
edge-budget chassis.

