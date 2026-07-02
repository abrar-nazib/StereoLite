# StereoLite_yolo_ctx_hstereo

GEV4 plus a true high-resolution stereo stage.

## What Changed

Earlier sharp/guided variants tried to repair boundaries after the final
disparity was already produced. That is weak because image-only refinement can
sharpen edges, but it cannot verify stereo correspondence.

This variant keeps the useful GEV4 base and changes the final path:

```text
1/4 GEV + tile refinement
        |
plane upsample tile state to 1/2
        |
TileRefineCtx on fL2/fR2 with local cost lookup
        |
ConvexUpsample 1/2 -> full
```

The goal is to improve thin structures and sample-level EPE by doing actual
stereo matching at 1/2 resolution before the last learned upsample.

## Usage

```bash
/home/lelouch/miniconda3/envs/ML/bin/python model/scripts/overfit_arch_ablation.py \
  --arch yolo_ctx_hstereo \
  --backbone yolo26s \
  --steps 10000 \
  --batch 4 \
  --height 384 \
  --width 640 \
  --n_pairs 100 \
  --show 0 \
  --viz_rotate 1 \
  --loss_variant boundary_focus
```
