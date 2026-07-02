# XR-Stereo: Stereo Matching in Time: 100+ FPS Video Stereo Matching for Extended Reality

**Authors:** Ziang Cheng, Jiayu Yang, Hongdong Li (Tencent XR Vision Labs + ANU)
**Venue:** WACV 2024 (arXiv 2309.04183)
**Tier:** 2 (amortizes RAFT iterations across time; the closest prior to any temporal-warm-start edge stereo)
**PDF:** `papers/raw/temporal/XR-Stereo_Cheng_WACV2024.pdf`

---

## Core Idea
Take RAFT-Stereo (real-time variant) and unroll its iterative GRU cost aggregation **across the temporal dimension**: instead of 7-20 GRU iterations per frame from scratch, warp the previous frame's disparity and GRU hidden state into the current view (using known camera pose) and run only **1-5 GRU iterations per frame**. With 1 iteration/frame it matches RAFT-Stereo RT at 20 iterations while running 108 fps. Also contributes the XR-Stereo synthetic dataset: 60K photo-realistic indoor stereo pairs at 640x480, 30 Hz, with real 6-DoF HMD head trajectories.

## What is carried across frames, and how
- **Dense disparity map D_{t-1}** plus the **GRU hidden state Z_{t-1}** (RAFT-Stereo RT resolutions).
- Warped analytically in disparity space with a single 4x4 stereo transformation **T_geo = Q [R|t] Q^{-1}** that maps (u, v, d) homogeneous stereo coordinates from the previous to the current left camera (Q is the disparity-to-depth matrix built from f, c_x, c_y, baseline). Forward warping uses Softmax Splatting weighted by disparity so the nearest surface wins on collisions; disocclusion holes are set to zero and repaired by the current-frame GRU iteration(s).
- First frame: a small init network predicts disparity from context features; hidden state starts at zero. Error converges after ~15 frames (0.5 s at 30 Hz).

## Dynamic objects / disocclusions
The rigid warp assumes a static scene, and their dataset **contains no moving objects** (stated limitation). Occlusion handled by splatting priority; holes zero-filled; robustness to fast ego-motion demonstrated up to 20x playback speed (the no-warp fast variant collapses there, EPE 3.18 vs 1.68 at 6x). Pose noise tolerated to roughly SLAM-level (~0.3 deg / 0.5 mm inter-frame); extreme noise degrades it.

## Benchmark Numbers
| Metric | Value |
|--------|-------|
| XR-Stereo dataset EPE | 1.48 (1 iter, 108 fps), 1.42 (5 iters, 57 fps); fast no-warp variant 1.67 at **134 fps** (RTX 3090 Ti, 640x480) |
| vs RAFT-Stereo RT | 20-iter RAFT: EPE 1.70 at 20 fps. Ours is better AND 5x faster |
| KITTI VO (real, trained from scratch) | EPE 1.66 (5 iters) vs RAFT RT 1.77 (20 iters) |
| Embedded deployment | **30 fps on Qualcomm XR2 HMD** (ONNX float, no quantization) |
| Params | not reported (RAFT-Stereo RT chassis; fast variant halves encoder channels) |

## Relevance to Edge Stereo / TempTile audit
- **The strongest single prior** for "warm-start stereo state across frames + 1-2 refinement iterations per frame + low-power target". Anticipates temporal iteration amortization and analytic pose warping of network state.
- Crucial differences: the carried state is a **dense disparity + GRU hidden tensor** (stateful RAFT chassis), not slanted-plane tile hypotheses; the warp transforms point-wise disparity only, never plane slopes; iteration count is a fixed offline choice (1/2/5), not drift-triggered; there is **no confidence/cost gating**, only zero-fill of holes; params are RAFT-class, not sub-2.5 M.
- Warping a GRU hidden state through splatting is semantically dubious (features are not covariant quantities); a stateless plane-parameter state avoids exactly this.
