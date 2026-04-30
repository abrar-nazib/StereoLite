# StereoLite — mid tier (YOLO26s-truncated encoder)

Same chassis as the edge tier, with the GhostConv encoder swapped for a
truncated YOLO26s backbone. Targets the ~2.5 M param / BeagleY-AI / Jetson
Orin Nano (~4-6 TOPS) deployment slot, where the extra encoder capacity
buys meaningfully better fine-detail preservation.

The original `model/designs/StereoLite/` is the deployed edge baseline and
must not be edited for encoder experiments. This sibling folder is the
canonical pattern for an architectural variant — copy it for future
encoder swaps.

## What's different from edge tier

Only the encoder. `tile_propagate.py`, `cost_volume.py`, refinement
iteration counts, plane-equation upsample, and convex-mask upsample are
all unchanged.

| | Edge (`StereoLite/`) | Mid (this folder) |
|---|---|---|
| Encoder | GhostConv stack (24/48/72/96 ch) | YOLO26s truncated (32/128/256/256 ch) |
| Trainable params | 0.874 M | 2.061 M |
| Latency, RTX 3050, 384×640 | 23.5 ms (42.5 FPS) | 25.8 ms (38.7 FPS) |
| Status | trained, deployed | selected via matched overfit; full Scene Flow training pending |

YOLO26n is also wired (`backbone="yolo26n"`) and still under
investigation. In matched runs so far it underperforms GhostConv on
average metrics and YOLO26s on threshold metrics, but the run-to-run
behaviour is wobbly and the cause (initialisation, COCO-pretrain
transfer, channel-jump 16→64 in the truncated stack) hasn't been
isolated yet. Treated as an open question, not as a dropped variant.

## Files

| File | Role |
|---|---|
| `model.py` | `StereoLite` with `backbone="yolo26s"` (or `"yolo26n"`) wired to `YoloTruncatedEncoder` |
| `yolo_encoder.py` | `YoloTruncatedEncoder` wrapping ultralytics YOLO26 model.model[:7], emits taps at 1/2, 1/4, 1/8, 1/16 |
| `tile_propagate.py`, `cost_volume.py` | Copies of the edge-tier modules; do not diverge without recording the reason here |
| `arch_refs/` | Reference architecture PDFs |

## Why a sibling folder, not a config flag

Future architectural variants (new TileInit, GRU TileRefine, context
encoder, etc.) will each get their own sibling so we can A/B them with the
matched-overfit harness without polluting the deployed edge chassis. See
[`../../benchmarks/OVERFIT_METHODOLOGY.md`](../../benchmarks/OVERFIT_METHODOLOGY.md).
