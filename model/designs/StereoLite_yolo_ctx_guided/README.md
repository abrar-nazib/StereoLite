# StereoLite_yolo_ctx_guided

Sharpness-focused variant built on `StereoLite_yolo_ctx_gev4`.

## Idea

The model keeps the successful pieces:

- YOLO26 stereo encoder
- RAFT-style left context stream
- ctx-gated iterative tile refinement
- 1/4 GEV warm evidence
- convex upsampling

Then it adds one fail-soft full-resolution module:

```text
convex full-res disparity
        |
RGB + 1/2 features + local gradients
        |
predict 5x5 local propagation weights + residual + gate
        |
edge-aware local disparity selection
```

This is inspired by CSPN / guided depth refinement and the same edge-preserving
principle behind BGNet-style upsampling: at a boundary, choose from local
foreground/background disparity candidates instead of averaging across them.

The current default is deliberately conservative: one propagation pass, max
residual `0.75 px`, and a near-closed initial gate. The training scripts also
add a guided guard loss, which penalises final-vs-pre-guided changes in smooth
regions while allowing corrections near image/GT disparity boundaries. This is
meant to keep the visual edge gain without creating the high-EPE outliers seen
in the first guided run.

## Usage

```bash
/home/lelouch/miniconda3/envs/ML/bin/python model/scripts/overfit_arch_ablation.py \
  --arch yolo_ctx_guided \
  --backbone yolo26s \
  --steps 12000 \
  --batch 4 \
  --height 384 \
  --width 640 \
  --n_pairs 100 \
  --show 0 \
  --viz_rotate 1 \
  --loss_variant boundary_focus
```

The auxiliary output includes:

- `d_pre_guided`: full-res disparity before guided propagation
- `guided_gate`: learned propagation strength
