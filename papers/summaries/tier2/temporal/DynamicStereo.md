# DynamicStereo: Consistent Dynamic Depth from Stereo Videos

**Authors:** Nikita Karaev, Ignacio Rocco, Benjamin Graham, Natalia Neverova, Andrea Vedaldi, Christian Rupprecht (Meta AI + VGG Oxford)
**Venue:** CVPR 2023 (arXiv 2305.02296)
**Tier:** 2 (offline transformer for temporal consistency; introduces Dynamic Replica dataset and the TEPE metric)
**PDF:** `papers/raw/temporal/DynamicStereo_Karaev_CVPR2023.pdf`

---

## Core Idea
Treat video stereo as **joint inference over a temporal window**, not frame-to-frame state propagation. A transformer encoder-decoder processes T frames of a stereo video together: divided Space-Stereo-Time (SST) attention at 1/16 exchanges information across space, view, and time; a coarse-to-fine decoder (1/16 → 1/8 → 1/4) runs RAFT-style iterative updates with a **3D (space-time) convolutional GRU** so every update sees neighbouring frames. Also introduces **Dynamic Replica**: 524 stereo videos (1280x720, 300 frames each) of animated people and animals in Replica scans, and the **TEPE** temporal end-point-error metric.

## What is carried across frames, and how
Nothing is propagated online. All T frames (train T=5, test T=20 with overlapping sliding windows of 10) are processed jointly; temporal coupling comes purely from learned attention (SST block, quadratic attention over time) and the separable 3D convolutions inside the GRU. **No camera pose, no optical flow, no warping of any kind.** This makes it an offline / near-offline method; overlapping-window inference causes low-frequency oscillation at window scale (stated limitation).

## Dynamic objects / disocclusions
Handled implicitly: the model is trained on non-rigid content (Dynamic Replica) and learns to pool across time without a motion model. No explicit occlusion mask, no rigidity assumption, no failure gating.

## Benchmark Numbers
| Metric | Value |
|--------|-------|
| Sintel Clean delta-3px / TEPE (SF-trained) | 6.10 / **0.77** vs RAFT-Stereo 6.12 / 0.92, CODD 8.68 / 1.44 |
| Dynamic Replica delta-1px / TEPE (DR+SF) | 3.32 / **0.075** |
| Runtime | **1.20 s/frame** at 1280x720 (vs RAFT-Stereo 0.83, CODD 1.04) |
| Params | not stated; trained on 8x V100-32GB for ~4 days |

Ablations: separate update blocks per resolution beat weight sharing; 3D (space-time) GRU convolution beats 2D especially on TEPE (0.823 vs 1.05 Sintel).

## Relevance to Edge Stereo / TempTile audit
- Anti-edge by construction: joint window inference, transformer attention, ~1.2 s/frame. Useful only as an accuracy/consistency reference and for its dataset + TEPE metric (both worth adopting for any temporal-stereo evaluation).
- Zero overlap with plane/tile state currency: no carried state, no analytic transform, no gating, no adaptive compute. Cite it for the *problem definition* (temporal consistency, TEPE) and the Dynamic Replica benchmark, not for mechanism.
