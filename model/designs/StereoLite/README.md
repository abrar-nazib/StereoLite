# StereoLite — edge tier (GhostConv encoder)

Lightweight HITNet-paradigm tile-hypothesis network with a small RAFT-style
iterative refinement loop and convex-mask final upsample. Targets the
< 1 M param / Raspberry Pi 5 / Jetson Nano (~1-2 TOPS) deployment slot.

## Pipeline

```
(L, R) ─► GhostConv encoder ─► f2, f4, f8, f16  (shared L/R weights)

  1/16: TileInit (8-group cost volume, max_disp=24)
        → seeds (d, sx=0, sy=0, feat, conf) per tile
        → TileRefine × 2 (warps fR by current d, residual updates)

  1/16 → 1/8 via plane-equation upsample (uses slopes sx, sy)
        → TileRefine × 3

  1/8 → 1/4 via plane-equation upsample
        → TileRefine × 3

  1/4 → 1/2 via ConvexUpsample (RAFT-style 9-neighbour learned mask)
  1/2 → full via ConvexUpsample
```

Total iterative refinement passes: 8 (2 + 3 + 3).

## Numbers (current)

| | |
|---|---|
| Trainable params | 0.874 M (full chassis) / 0.538 M (overfit-harness config) |
| Inference latency, RTX 3050, 384×640, B=1 | ~23.5 ms (42.5 FPS) |
| Inference latency, RTX 3050, 512×832, B=1 | ~54 ms |
| Best val EPE (InStereo2K, post-finetune) | 1.54 px |

## Files

| File | Role |
|---|---|
| `model.py` | `StereoLite`, `StereoLiteConfig`, `TileFeatureEncoder` (GhostConv default), `MobileNetV2Encoder` (alternative), `ConvexUpsample` |
| `tile_propagate.py` | `TileState`, `TileInit`, `TileRefine`, `TileUpsample` |
| `cost_volume.py` | `GroupwiseCostVolume1D8` (1/16, 8 groups, max_disp=24) |
| `arch_refs/` | Reference architecture PDFs (RAFT, IGEV, BANet, etc.) used when sketching diagrams |
| `stereolite_architecture_doc.{tex,pdf}` | Formal architecture spec |

## Backbone choices

`StereoLiteConfig.backbone` selects the encoder:

- `"ghost"` (default) — GhostConv stack, 24/48/72/96 channels at 1/2 to 1/16
- `"mobilenet"` — timm `mobilenetv2_100`, ImageNet-pretrained, 1/32 stage truncated

The mid-tier YOLO26s variant lives in the sibling folder
[`../StereoLite_yolo/`](../StereoLite_yolo/) — never edit this folder for
encoder-swap experiments; spawn a new sibling instead.

## Training entry points

See [`../../scripts/`](../../scripts/) — `train_sceneflow.py` for full Scene
Flow training, `train_finetune_indoor.py` for indoor pseudo-GT finetune,
`overfit_yolo_ablation.py` for the matched-overfit A/B harness.
