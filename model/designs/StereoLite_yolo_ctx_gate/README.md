# StereoLite_yolo_ctx_gate

Minimal gated-update variant of `StereoLite_yolo_ctx`.

## Idea

The previous geometry-heavy `yolo_geomctx` improved some overfit numbers but
cost too much latency. This variant keeps the fast `yolo_ctx` architecture and
adds only one cheap mechanism: a learned gate that scales each residual update.

```text
cost lookup + ctx + ConvGRU -> residuals
ctx/conf/cost sharpness -> update gate
tile_new = tile_old + gate * residual
```

## Why This Variant Exists

- Keep latency close to `yolo_ctx`.
- Avoid the 3x3 slanted patch lookup from `yolo_geomctx`.
- Let the network suppress bad updates in ambiguous regions.
- Keep the ablation clean enough for a thesis/paper table.

## Usage

```python
from model.designs.StereoLite_yolo_ctx_gate.model import (
    StereoLiteYoloCtxGate, StereoLiteYoloCtxGateConfig)

model = StereoLiteYoloCtxGate(
    StereoLiteYoloCtxGateConfig(backbone="yolo26s"))
```

The package also exports compatibility aliases:

```python
from model.designs.StereoLite_yolo_ctx_gate import (
    StereoLiteYoloCtx, StereoLiteYoloCtxConfig)
```

