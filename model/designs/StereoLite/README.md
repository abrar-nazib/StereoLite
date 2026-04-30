# Design 1 — Foundation-Seeded Tile-Hypothesis Stereo

## Base paradigm
HITNet (Tankovich et al., CVPR 2021). Tile-hypothesis propagation: every 4×4
image tile carries a slanted-plane hypothesis `(d, d/dx, d/dy, w)` that
propagates coarse-to-fine via local message passing. **No all-pairs
correlation volume. No iterative GRU.** Single forward pass.

| Property | HITNet baseline |
|---|---|
| Parameters | 0.63 M |
| Input resolution | 1024×384 (KITTI native) |
| Jetson Orin Nano | ~30 FPS at 1080p |
| KITTI D1-all | 2.40 |

## Contribution on top of HITNet

1. **Foundation-seeded initial hypotheses.** Run a frozen DAv2-Small on the
   left image once. Fit a cheap per-image affine against a 1/32 scale
   cost-slice to convert relative depth into a scale-aware initial disparity.
   Use this to initialise the coarsest-level tile hypotheses instead of the
   learned-zero that HITNet uses.
2. **Attention-gated tile propagation.** Replace HITNet's hand-designed 3×3
   propagation with a small 4-head linear-attention block over each tile's
   eight neighbours. Adds ~0.1 M params; lets propagation respect tile
   confidence.
3. **Foundation-teacher disparity distillation.** A FoundationStereo or
   DEFOM teacher supervises the final tile disparity plus the per-tile
   confidence head. Training-only cost.

## Paper claim
*Foundation-seeded tile-hypothesis stereo matches RAFT-Stereo accuracy at
1/9th the parameters by seeding tile hypotheses with a monocular foundation
prior and distilling from a foundation stereo teacher.*

## Target budget
- Parameters: **~1.2 M**
- Latency (Orin Nano, 1280×720): **<35 ms**

## Ablations the paper will need
- HITNet baseline / + foundation init / + attention propagation / + teacher KD
- Teacher choice: FoundationStereo vs DEFOM vs IGEV
- Mono backbone ablation: DAv2-Small vs MiDaS-small vs none

## What is not yet implemented
Everything. This directory contains only this README until you greenlight the
design. The RAFT chassis we had was deleted (see project history).
