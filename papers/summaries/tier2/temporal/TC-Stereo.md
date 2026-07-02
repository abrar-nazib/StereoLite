# TC-Stereo: Temporally Consistent Stereo Matching

**Authors:** Jiaxi Zeng, Chengtang Yao, Yuwei Wu, Yunde Jia (Beijing Institute of Technology)
**Venue:** ECCV 2024 (arXiv 2407.11950)
**Tier:** 2 (video stereo as temporal disparity completion + dual-space refinement; found during TempTile novelty sweep)
**PDF:** `papers/raw/temporal/TC-Stereo_Zeng_ECCV2024.pdf`

---

## Core Idea
Reformulate video stereo as **temporal disparity completion + continuous iterative refinement**. Project the previous frame's disparity to the current viewpoint using camera pose (semi-dense after warping), complete it to a dense initialization with a completion module, then run a few RAFT-style iterations with a **dual-space refinement** that operates in both disparity space and **disparity-gradient space** (local planar hypotheses: per-pixel d, dd/du, dd/dv used to propagate disparity to neighbours). Temporal state features from the completion module and past refinement hidden state are fused for a temporally coherent state.

## What is carried across frames, and how
- Previous **disparity map** (pose-projected: inverse-project to 3D, rigid transform T_{t-1 to t}, re-project, forward warp; holes where occluded/out-of-view), previous **refinement hidden state**, and **state features** from the completion module, fused into the current state.
- A **cost-evidence gate filters the projected disparity**: only points whose cost-volume top-1 vs top-2 margin exceeds theta = 0.3 survive into the semi-dense map. This is a cost-gated acceptance of temporal hypotheses, at pixel granularity, applied once at initialization.
- Disparity gradients (slopes) are estimated per pixel **within each frame** by sampling neighbouring points in disparity space; they guide intra-frame propagation but are **recomputed every iteration and never transported across time**, and never transformed under pose.

## Dynamic objects / disocclusions
Static-scene assumption for the warp (acknowledged limitation); moving objects are corrected by the iterative refinement (shown qualitatively on a moving car). Disocclusion holes are filled by the completion module. No learned motion field, no explicit per-region reset; under large motion / bad pose it implicitly degrades toward a from-scratch RAFT-style search.

## Benchmark Numbers
| Metric | Value |
|--------|-------|
| KITTI (N=5 iters) | D1-all **1.46%** at **0.09 s/frame** vs RAFT-Stereo N=5: 1.82% at 0.38 s |
| Speedup | ~4x over RAFT-Stereo at matched accuracy (A40-class GPU) |
| Params | not reported (RAFT-family chassis) |

## Relevance to Edge Stereo / TempTile audit
- Together with XR-Stereo, the closest deep prior: analytic pose warp of disparity + hidden state, cost-margin gate on temporal hypotheses, few iterations per frame, and slanted-plane (gradient-space) reasoning **within** the frame.
- The pieces TempTile claims that TC-Stereo does not do: slopes as part of the *carried* temporal state; analytic covariant transformation of plane parameters (TC-Stereo warps only scalar disparity; gradients are re-derived from scratch each frame); tile-granular fail-soft reinitialization (its theta-gate only filters the initialization once, with no per-region reinit of a persistent hypothesis); stateless no-GRU chassis at sub-2.5 M params (TC-Stereo carries a GRU hidden state and is RAFT-class, ~11 FPS full-res).
