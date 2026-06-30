# StereoLite_yolo_ctx_init4

Conservative improvement over `StereoLite_yolo_ctx_gate`.

## Idea

The failed `sharp`, `hrrefine`, and `sru` variants suggest the recurrent core
should not be disturbed. This variant keeps the `ctx_gate` update path and adds
fresh matching evidence at 1/4 resolution before the last refinement stage.

```text
1/16 cost init -> 1/16 refine
        |
plane upsample -> 1/8 refine
        |
plane upsample -> 1/4
        |
fresh 1/4 cost init + learned blend
        |
normal ctx_gate 1/4 refine
        |
convex upsample to full
```

## Why This Variant Exists

- Thin/far structures may be damaged by a 1/16-only initialization.
- A 1/4 cost init gives the model sharper local evidence before final tile
  refinement.
- The successful `yolo_ctx_gate` GRU and update gate remain unchanged.

## Usage

```bash
python model/scripts/overfit_arch_ablation.py \
  --arch yolo_ctx_init4 \
  --backbone yolo26s \
  --steps 3000 \
  --batch 4 \
  --show 0
```
