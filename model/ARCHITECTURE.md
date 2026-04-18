# `latest.pth` — Reference Asset Documentation

This document describes the trained checkpoint at
`model/checkpoints/latest.pth`. The checkpoint is kept as an optional
distillation teacher (see `model/README.md`); **none of the three candidate
designs in `model/designs/` inherit from it**.

All numbers below were extracted directly from the checkpoint by
`model/scripts/inspect_checkpoint.py`.

## Training Metadata

| Field | Value |
|---|---|
| Model name (from args) | `stereo_lite_v3` |
| Architecture family | RAFT-Stereo (3-level hierarchical ConvGRU) |
| Training step | 9,000 of 100,000 |
| Epoch | 62 |
| Training loss | 9.42 |
| Train EPE | **0.585 px** |
| Train 1-px accuracy | **94.06%** |
| Validation EPE | not yet run (`best_val_epe = inf`) |
| Optimizer | AdamW-class, 248 parameter states, 1 param group |
| File size | 133.7 MB |
| Total parameters | 11,122,295 (11.12 M) |

## Training Hyperparameters

| Parameter | Value |
|---|---|
| Dataset | InStereo2K |
| Crop size | 320 × 736 |
| Spatial scale augmentation | [-0.2, 0.4] |
| Batch size | 8 |
| Max disparity | 192 |
| Learning rate | 2e-4 |
| Weight decay | 1e-5 |
| Mixed precision | True |
| Train iterations (per step) | 22 |
| Val iterations | 32 |

## Architecture Shape

| Parameter | Value |
|---|---|
| `hidden_dims` | `[128, 128, 128]` (three-level GRU, 128 ch each) |
| `corr_levels` | 4 (correlation pyramid depth) |
| `corr_radius` | 4 (lookup radius → 9 samples per level) |
| `n_downsample` | 2 (features at 1/4 input resolution) |
| `n_gru_layers` | 3 (hierarchical GRU across 1/8, 1/16, 1/32) |
| `context_norm` | batch (BatchNorm in context encoder) |

## Why this asset is not a chassis for new work

The edge-stereo SOTA sits at **0.6–3 M parameters**. `latest.pth` at 11 M is
an order of magnitude heavier than models like HITNet (0.63 M), BGNet
(2.28 M), CoEx (2.7 M), or LightStereo-S (~1.8 M). Building new compression
work on top of a RAFT-Stereo chassis therefore starts from the wrong place —
the first 8 M of parameters to shed would just be RAFT overhead.

For that reason, new designs are based on paradigms that do not use an
all-pairs correlation volume or an iterative GRU refinement loop. See
`model/designs/`.

## Use of this checkpoint going forward

- As an **optional distillation teacher** if a foundation stereo teacher
  (FoundationStereo, DEFOM) is not available locally.
- As the baseline in the final paper's comparison table (so readers can
  see the "starting point" even though it is not the architecture).
- As a sanity-check target when building data loaders — any new design
  should match its mean disparity range on the same scene before training
  is considered converged.

## Inspection

```bash
source venv/bin/activate
python3 model/scripts/inspect_checkpoint.py
```
