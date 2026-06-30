# StereoLite_yolo_ctx_hrrefine

Boundary-focused variant of `StereoLite_yolo_ctx_gate`.

## Idea

`yolo_ctx_gate` is accurate, but its final disparity can look soft because the
last iterative stereo reasoning happens at 1/4 resolution. This variant keeps
that stable core and adds a small residual refinement loop at 1/2 resolution
before the final full-resolution convex upsample.

```text
1/4 ctx-gated tile refinement
        |
convex upsample to 1/2
        |
local 1/2 correlation + context GRU -> bounded residual correction
        |
convex upsample to full resolution
```

## Why This Variant Exists

- Tests whether visual blur comes from stopping iterative refinement too early.
- Uses local stereo evidence at 1/2 resolution without copying the full HITNet
  pipeline.
- Keeps residual correction bounded so the strong coarse prediction remains
  stable.
- Exposes `edge_half` and `d_half_pre_refine` in `aux=True` for debugging.

## Usage

```python
from model.designs.StereoLite_yolo_ctx_hrrefine.model import (
    StereoLiteYoloCtxHRRefine, StereoLiteYoloCtxHRRefineConfig)

model = StereoLiteYoloCtxHRRefine(
    StereoLiteYoloCtxHRRefineConfig(backbone="yolo26s"))
```

Benchmark:

```bash
python model/scripts/overfit_arch_ablation.py \
  --arch yolo_ctx_hrrefine \
  --backbone yolo26s \
  --steps 3000 \
  --batch 4 \
  --show 0
```
