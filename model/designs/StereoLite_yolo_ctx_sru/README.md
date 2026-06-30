# StereoLite_yolo_ctx_sru

Selective recurrent update variant of `StereoLite_yolo_ctx_gate`.

## Idea

The previous `sharp` and `hrrefine` variants tried to repair softness after
the main iterative field had already been produced. This variant attacks the
earlier cause: over-smoothing inside the recurrent tile update.

Each `TileRefineCtx` step runs two update branches:

```text
1x1 GRU branch -> thin structures / boundaries
3x3 GRU branch -> smooth regions / context propagation
context attention -> per-pixel branch blend
```

It keeps the existing context/confidence update gate from `yolo_ctx_gate`.

## Why This Variant Exists

- Inspired by Selective-Stereo's SRU idea.
- Targets both EPE and visual sharpness without adding a slow final tail.
- Keeps the successful `yolo_ctx_gate` pipeline and only changes the recurrent
  update operator.

## Usage

```bash
python model/scripts/overfit_arch_ablation.py \
  --arch yolo_ctx_sru \
  --backbone yolo26s \
  --steps 3000 \
  --batch 4 \
  --show 0
```

For boundary-heavy overfit experiments:

```bash
python model/scripts/overfit_arch_ablation.py \
  --arch yolo_ctx_sru \
  --backbone yolo26s \
  --steps 3000 \
  --batch 4 \
  --show 0 \
  --loss_variant boundary_focus
```
