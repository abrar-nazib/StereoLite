# StereoLite_yolo_ctx_gev4

Usable-model attempt built on the best `yolo_ctx_gate` baseline.

## Idea

The previous variants changed the tail or recurrent operator and hurt EPE.
This branch keeps the stable `ctx_gate` path and adds a small IGEV-style
geometry encoding volume at 1/4 resolution.

```text
ctx_gate coarse-to-fine tile refinement
        |
1/4 group-wise correlation volume
        |
tiny 3D regularizer -> soft-argmin disparity + geometry feature
        |
fail-soft blend into 1/4 tile state
        |
normal ctx_gate 1/4 refinement + convex upsample
```

The blend gate is initialized with a negative bias, so the model starts close
to `yolo_ctx_gate` and only learns to use the GEV path where it helps.

## Usage

```bash
python model/scripts/overfit_arch_ablation.py \
  --arch yolo_ctx_gev4 \
  --backbone yolo26s \
  --steps 3000 \
  --batch 4 \
  --show 0
```
