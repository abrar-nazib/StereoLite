# CODD: Temporally Consistent Online Depth Estimation in Dynamic Scenes

**Authors:** Zhaoshuo Li, Wei Ye, Dilin Wang, Francis X. Creighton, Russell H. Taylor, Ganesh Venkatesh, Mathias Unberath (JHU + Meta Reality Labs)
**Venue:** WACV 2023 (arXiv 2111.09337)
**Tier:** 2 (online per-pixel SE3 alignment + learned reset/fusion gating on top of HITNet)
**PDF:** `papers/raw/temporal/CODD_Li_WACV2023.pdf`

---

## Core Idea
A modular online framework: per-frame **stereo network** (HITNet) + **motion network** (RAFT3D-style, predicts a dense per-pixel SE3 field) + **fusion network** (learned per-pixel reset and fusion weights). The previous frame's fused output is aligned into the current frame via the SE3 field and blended with the current per-frame estimate, giving temporally consistent metric depth in **dynamic** scenes with only past frames.

## What is carried across frames, and how
- **Memory state m in R^{H x W x (3+C+1)}**: per-pixel semantic features (RGB + C-channel features from the stereo network) + the fused disparity, at full image resolution, from the **immediately preceding frame only**.
- Alignment is **learned, not analytic**: the motion network (built on RAFT3D, GRU + Gauss-Newton, K = 1-16 iterations depending on motion magnitude) predicts a per-pixel SE3 transformation T in SE(3)^{H x W} covering camera AND object motion; the previous state is back-projected, transformed, and re-rendered into the current view by differentiable rendering. A visibility mask, motion confidence (sigmoid), and flow magnitude are appended as cues.

## Dynamic objects / fail-soft mechanism (the interesting part)
Dynamics are handled head-on by the per-pixel SE3 field. Then the fusion network computes per-pixel **w_reset** (outlier rejection, supervised to fire when the warped estimate's error exceeds the fresh estimate's by tau_reset = 5 px) and **w_fusion** (aggregation weight, tau_fusion = 1 px):
d_F = (1 - w_reset * w_fusion) d_stereo + w_reset * w_fusion d_motion.
When the temporal estimate is deemed unreliable, the pixel **falls back to the fresh per-frame stereo estimate**. Input cues: 3-channel L/R feature-distance disparity confidence, pixel-to-patch self-correlation (local smoothness), pixel-to-patch cross-correlation (inter-frame disagreement), flow magnitude/confidence, visibility mask. This is the closest published relative of confidence-gated per-region fail-soft reinitialization, but it operates at **output level** (blending two finished disparity maps), not inside the matching pipeline.

## Benchmark Numbers
| Metric | Value |
|--------|-------|
| FlyingThings3D TEPE | HITNet alone 0.812 → CODD **0.741**; EPE 0.607 → 0.595 |
| TartanAir TEPEr | 9.04 → **6.21** (-31%) |
| KITTI Depth TEPE | 0.289 → 0.258 |
| Params | **9.3 M total = stereo 0.6 M (HITNet) + motion 8.5 M + fusion 0.2 M** |
| Speed | **25 FPS at 640x480 (Titan RTX)**: stereo 26 ms + motion 13 ms + fusion 0.3 ms |

Naive forward-only propagation of the warped past (motion-only row) is WORSE than per-frame stereo (TEPE 0.875 vs 0.812): un-gated temporal reuse propagates errors. The gating is what makes temporal reuse pay off.

## Relevance to Edge Stereo / TempTile audit
- Proof that gated temporal fusion adds only ~13.3 ms and 8.7 M params over a HITNet per-frame base, and that the **motion network is 91% of the overhead**. An analytic pose transform (when ego-motion dominates) would delete nearly all of that cost, which is precisely the gap an analytic plane-covariant transform exploits.
- Carries HITNet's *output* (disparity + features), never its internal slanted-tile states; slopes are neither carried nor transformed.
- The w_reset / w_fusion mechanism must be cited as the nearest prior for per-region fail-soft; differentiate on granularity (pixel vs tile), location (output blending vs hypothesis reinit inside matching, gated by cost-volume evidence), and cost (learned 8.5 M motion net vs analytic transform).
