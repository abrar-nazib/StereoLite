# TemporalStereo: Efficient Spatial-Temporal Stereo Matching Network

**Authors:** Youmin Zhang, Matteo Poggi, Stefano Mattoccia (University of Bologna)
**Venue:** IROS 2023 (arXiv 2211.13755)
**Tier:** 2 (first supervised stereo net with cached temporal context; pose-based candidate warping)
**PDF:** `papers/raw/temporal/TemporalStereo_Zhang_IROS2023.pdf`

---

## Core Idea
A coarse-to-fine sparse-cost-volume network (3 stages at 1/16, 1/8, 1/4) that, when a stereo video plus camera pose is available, switches to a "temporal mode": past disparities, past top-K costs, and past backbone features are cached and warped into the current frame to enrich the current sparse candidate set. One model, trained once on videos, runs seamlessly in both single-pair and temporal mode.

## What is carried across frames, and how
- **Local Map:** full-resolution disparity maps of the last N_key = 3 keyframes (keyframe promoted when relative motion exceeds |t| > 0.1 m or |R| > 15 deg). Each cached disparity value is analytically updated under the relative pose T_j→t: back-project pixel to 3D with the stereo camera model, rigidly transform, recompute d_proj = b f_x / Z, then forward-warp coordinates with differentiable Softmax Splatting. Warped disparities join the stage-2 candidate set D2 (only stage 2, the accuracy/speed sweet spot).
- **Past Costs:** the previous frame's final stage-3 cost volume (top-K = 2 candidates plus their costs) is pose-updated and forward-warped the same way, downsampled, and concatenated into the current stage-1/2 aggregation via the Statistical Fusion module.
- **Temporal Shift (TSM):** cached backbone feature maps are channel-shifted into the current features. Zero extra parameters, zero pose requirement, works even with identity pose.

## Dynamic objects / robustness
No explicit motion segmentation. Rigid (pose-only) warping is wrong on moving objects, but temporal cues only *augment* candidates and costs; the current stereo pair always dominates. Empirically robust: KITTI 2015 D1-FG actually improves in temporal mode (2.85 → 2.78), and results degrade gracefully with noisy pose (survives sigma_R = 10 deg, sigma_t = 0.5 m; DROID-SLAM pose matches GT pose performance).

## Benchmark Numbers
| Metric | Value |
|--------|-------|
| SceneFlow EPE (single-pair) | 0.53 (equal to HITNet's 0.53) |
| KITTI 2015 D1-all | 2.05 (single-pair) → **1.81 (temporal)** |
| KITTI 2012 3PE-Refl | 6.99 → **6.14** (temporal) |
| TartanAir EPE all / occ | 0.647 / 1.899 (single) → 0.601 / 1.615 (temporal, W=8) |
| Runtime | ~45 ms single-pair on RTX 3090; temporal overhead only **+4 ms** |
| Params | not stated; backbone is **EfficientNetV2-S**, so clearly research-scale, not edge-scale |

Temporal cues are also portable: bolted onto CoEx (+14.6%) and StereoNet (+26.1%) on TartanAir.

## Relevance to Edge Stereo / TempTile audit
- Closest published precedent for **analytic pose warping of cached stereo state** (disparity values and costs, via back-project + rigid transform + splat). It transforms only the scalar d per pixel; slopes/plane parameters do not exist in its representation.
- The "temporal candidates only augment, current evidence dominates" design is an implicit fail-soft, but there is no explicit confidence- or cost-gated reinitialization and no adaptive per-frame compute.
- Not edge-class: EfficientNetV2-S backbone, RTX 3090 benchmarks, no embedded deployment.
