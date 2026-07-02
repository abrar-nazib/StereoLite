# StereoLite_yolo_ctx_sharp

Boundary-aware refinement variant built on top of `StereoLite_yolo_ctx_gate`.

## Idea

The recurrent stereo core reaches good EPE but produces soft object borders.
This variant keeps the fast gated-context architecture and adds a small
full-resolution residual tail:

```text
d_full + left image + upsampled f2 features -> edge mask + residual
d_sharp = d_full + edge_mask * residual
```

The residual is bounded by `sharp_max_residual`, so the tail is encouraged to
fix boundary detail rather than rewrite the whole disparity map.

## Expected Benefit

- sharper object boundaries,
- less foreground/background bleeding,
- better thin structures,
- small parameter increase compared with `yolo_ctx_gate`.

## Usage

```python
from model.designs.StereoLite_yolo_ctx_sharp.model import (
    StereoLiteYoloCtxSharp, StereoLiteYoloCtxSharpConfig)

model = StereoLiteYoloCtxSharp(
    StereoLiteYoloCtxSharpConfig(backbone="yolo26s"))
```

The aux output includes:

- `d_pre_sharp`: disparity before the sharp tail,
- `edge_full`: predicted refinement mask.

