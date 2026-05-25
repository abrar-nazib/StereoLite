# Real-Time Semantic Stereo Matching (RTS2Net)

**Authors:** Pier Luigi Dovesi, Matteo Poggi, Lorenzo Andraghetti, Miquel Marti, Hedvig Kjellstrom, Alessandro Pieropan, Stefano Mattoccia (Univrses AB, KTH, University of Bologna)
**Venue:** ICRA 2020 (arXiv:1910.00541v2, February 2020)
**Tier:** 3 (joint disparity + semantic seg, FIRST real-time multi-task stereo network with explicit Jetson TX2 deployment data; the AnyNet-derived efficient baseline)

---

## Core Idea
RTS2Net is the first semantic-stereo network that actually runs in real time on an embedded device. The design follows AnyNet (Wang et al. 2019): a shared encoder feeds two task-specific decoders, each producing coarse-to-fine outputs at 1/16, 1/8, 1/4 resolution that can be early-stopped at any intermediate stage to trade accuracy for latency. The novelty is that both the disparity decoder *and* the semantic decoder share this pyramidal structure, plus a final **Synergy Disparity Refinement** module that concatenates the two task outputs into a hybrid volume and produces a refined disparity. The whole thing is parameterised by a base-feature width `c`; setting `c = 1` recovers AnyNet's disparity-only configuration, while `c = 4` to `c = 32` trade quality for speed.

## Architecture
- **Shared encoder (Sect. III-B, p. 2-3):** two initial 3x3 convs producing `c` features at 1/2, then four blocks each containing 2x2 max-pool + two 3x3 convs, extracting `2c, 4c, 8c, 16c` features at 1/4, 1/8, 1/16, 1/32 respectively. BN + ReLU after every conv. Total: very thin tower with `c` as the width knob.
- **Disparity branch (Sect. III-C, p. 3):** three coarse-to-fine stages at 1/16, 1/8, 1/4.
  - Stage 1: build distance-based CV by subtracting right features from left features over `d_max = 12` (covers 192 px at full res). Regularise with three 3D conv blocks (16, 16, 1 features) + BN + ReLU + soft-argmin.
  - Stages 2 and 3: upsample previous disparity, warp right features by it, build residual CV with `d_max = +-2` (i.e. +-16 px at full res), 3D conv decoder (4, 4, 1 features) + soft-argmin. Residual added to upsampled estimate.
- **Semantic branch (Sect. III-D, p. 3):** symmetric three-stage design at 1/16, 1/8, 1/4. Per stage emits per-class probability map; probabilities are upsampled across stages and summed residually. Class argmax at each stage produces a semantic map. Includes a 1/32 path for broader context.
- **Synergy Disparity Refinement (Sect. III-E, p. 4):** purple block in Fig. 2. Per stage: (1) compress semantic embedding to match disparity-CV dimensionality, (2) concatenate compressed semantic features with the disparity volume (reorganised so disparity is on the channel axis) plus upsampled refined disparity from the previous stage, (3) three 2D conv layers produce a residual added back into the volume, soft-argmin produces the refined disparity.
- **Objective (Eq. 2, p. 4):** `L = sum_{st=1..3} W_st * (W_d * L_d_st + W_s * L_s_st + W_dr * L_dr_st)`. Stage weights `W_st = 1/4, 1/2, 1`. Task weights `W_d, W_s, W_dr = 1, 2, 2`. Class-imbalance reweighting (Eq. 3) for the cross-entropy loss; coarse-annotation reweighting (Eq. 4) when CityScapes' coarse labels are used.

## Main Innovation
**The first real-time joint semantic stereo network with measured Jetson TX2 numbers.** This is RTS2Net's headline. The architectural moves (shared encoder + parallel pyramidal decoders + synergy refinement) are derivative of AnyNet (disparity) and standard segmentation pyramids; the contribution is showing that the two tasks can fit inside a 6-7 FPS embedded-class budget while still beating real-time stereo baselines (MADNet, StereoNet) on KITTI. Secondary contribution: a clean ablation across `c = 1, 4, 8, 16, 32` that shows where the speed-quality knee actually sits.

## Key Benchmark Numbers
- **Params:** not in paper.
- **GFLOPs:** not in paper.
- **Latency / FPS / target GPU (Tab. II, p. 5, KITTI 2015 val):**
  - `c = 1`: TX2 **8.3 FPS** / 2080 Ti 60.5 FPS.
  - `c = 4`: TX2 7.4 FPS / 2080 Ti 60.5 FPS.
  - `c = 8`: TX2 **6.3 FPS** (~160 ms/inference) / 2080 Ti 60.4 FPS.
  - `c = 16`: TX2 4.5 FPS / 2080 Ti 60.4 FPS.
  - `c = 32`: TX2 2.3 FPS / 2080 Ti 42.2 FPS.
  - Bottleneck at `c = 8` is the disparity subnetwork (120 ms / 160 ms total on TX2, Sect. IV-B, p. 5).

**Stereo + semantic, KITTI 2015 val (Tab. II, p. 5):**
- RTS2Net `c = 1`: EPE **1.12** / D1-all **5.57%** / mIoU **58.86%** / pAcc 80.86%.
- RTS2Net `c = 8`: EPE **0.84** / D1-all 3.33% / mIoU **62.22%** / pAcc 90.64%.
- RTS2Net `c = 32`: EPE **0.74** / D1-all 2.62% / mIoU **69.62%** / pAcc 93.57%.

**Stereo, KITTI 2015 leaderboard (Tab. V, p. 6) with `c = 32`:**
- RTS2Net: D1-bg **3.09%** / D1-fg 5.91% / D1-all **3.56%** / runtime 0.02 s on 2080 Ti.
- MADNet (real-time): 3.75 / 9.20 / 4.66 / 0.02 s. **RTS2Net beats MADNet by 1.1 D1-all at matched latency.**
- StereoNet: 4.30 / 7.45 / 4.83 / 0.02 s. **RTS2Net wins by 1.27 D1-all.**
- GANet (heavy): 1.48 / 3.46 / 1.81 / 1.80 s.
- PSMNet: 1.86 / 4.62 / 2.32 / 0.41 s.

**Semantic, KITTI 2015 leaderboard (Tab. VI, p. 6) with `c = 32`:**
- RTS2Net: IoU-class **57.67%** / iIoU-class 27.42 / IoU-cat 82.85% / iIoU-cat 60.72 / runtime 0.02 s (0.008 s if disparity head skipped).
- SegStereo: 59.10 / 28.00 / 81.31 / 60.26 / 0.60 s. **RTS2Net 1.4 IoU-class worse but 30x faster.**
- SDNet: 51.14 / 17.74 / 79.62 / 50.45 / 0.20 s.

**Anytime inference (Tab. IV, p. 5):** RTS2Net Stage 1 at 17.2 FPS / 8.00% D1-all, Stage 2 at 10.9 FPS / 4.70% (KITTI 10 Hz-compatible), Stage 3 at 6.3 FPS / 3.33%.

## Mutual-Task Coupling: Load-Bearing or Decorative?
Tab. III (p. 5) is the relevant ablation at `c = 8` on KITTI 2015 val:
| Configuration | EPE | D1-all | mIoU | pAcc |
|---|---|---|---|---|
| Disparity only | 0.91 | 3.98% | n/a | n/a |
| Disparity + semantic (no synergy refinement) | 0.90 | 3.90% | 64.21% | 91.56% |
| Disparity + semantic + synergy refinement | 0.91 (0.84 refined) | 3.91 (3.33 refined) | 62.22% | 90.64% |

Verdict: **Multi-task itself buys essentially nothing on stereo** (EPE 0.91 -> 0.90 = -0.01, D1-all 3.98 -> 3.90 = -0.08). The **synergy refinement module is what does the work**: refined disparity goes to 0.84 / 3.33% (-0.07 EPE, -0.65 D1-all). And the synergy module *costs* a small amount on segmentation: mIoU drops from 64.21 to 62.22 (-1.99 mIoU). So:

- **Pure joint training (parallel branches, no synergy):** stereo improvement is in the noise (-0.08 D1-all), segmentation gets free quality.
- **+ Synergy refinement:** stereo improves substantially (-0.65 D1-all), segmentation pays a 2-point mIoU tax.

This is the cleanest evidence in the whole multi-task stereo lineage that **the architecture that *re-injects* the semantic output into the disparity pipeline (the synergy module) is what generates the stereo benefit**, not the mere sharing of features in a parallel two-branch design. Same finding qualitatively as DispSegNet / SGNet / SemStereo, but with the most explicit ablation isolating the mechanism.

**Other lesson from Tab. II:** doubling `c` from 1 to 8 lifts the stereo-disparity benefit of going multi-task from 0.02 EPE (AnyNet vs RTS2Net `c=1`) to 0.07 EPE (`c=8`) to 0.12 EPE (`c=32`). The semantic branch is more useful when the feature dimensionality is large enough to encode multiple categories — confirming that for an edge-tier (small `c`) network, the semantic coupling buys little.

## Relevance to Our Project
- **First directly comparable architecture.** RTS2Net `c = 8` at 6.3 FPS on Jetson TX2 (160 ms / inference) is in the same envelope as StereoLite's target of "Jetson Orin Nano, < 33 ms (30 FPS), 4 GB peak". TX2 is ~1.3 TOPS vs Orin Nano's ~6 TOPS, so RTS2Net `c = 8` would land around 32-50 ms on Orin Nano. This is the only multi-task stereo paper whose deployment numbers we can actually compare ours against.
- **The "small `c` is enough for stereo, larger `c` is needed for semantics" finding is critical.** Tab. II shows EPE 1.12 -> 0.74 from `c = 1` to `c = 32` (a 34% reduction) but mIoU 58.86% -> 69.62% (an 18% relative gain). **Semantic segmentation needs more channels than stereo at the same accuracy budget.** Implication for us: if we ever bolt a segmentation head onto StereoLite for cross-task supervision, the segmentation head needs its *own* deeper / wider feature path; sharing our 24-72 channel feature tree will starve it.
- **The synergy refinement design is portable.** The pattern "concatenate compressed semantic class probabilities with the disparity CV channel axis, then 3 conv layers + soft-argmin" is implementable in our chassis at maybe 0.05-0.1 M extra params. The Tab. III ablation says it can buy 0.07 EPE / 0.65 D1-all on KITTI 2015 at this architectural budget. Worth a 30-minute A/B on the 100-pair harness if and when we add a segmentation head.
- **Pre-train on Cityscapes coarse + fine, not on Scene Flow.** Tab. I (p. 4) demonstrates that 60 epochs coarse CS + 75 epochs fine CS + 800 epochs KITTI beats 40 epochs Scene Flow + 800 epochs KITTI by 0.04 EPE / 0.53 D1-all on KITTI 2015 val. This is a free training-recipe lever, not architectural — would directly apply to any stereo network we train, even without a segmentation branch. Worth investigating whether our Scene Flow Driving pretrain could be replaced by Cityscapes pretrain for the KITTI / driving deployment regime.
- **Anytime inference is a free deployment feature.** RTS2Net Stage 1 / 2 / 3 lets the same network serve power-constrained and accuracy-constrained scenarios. Our StereoLite already exposes "iteration counts are config-tunable" but does not yet expose this at runtime; RTS2Net's stage-stopping is the design we should imitate when we wire up dynamic-compute.
- **The synergy module pays a -2 mIoU tax for stereo gains** (Tab. III). For our project we do not have a segmentation head to lose, so the tax does not exist; only the +0.07 EPE benefit is potentially in play (assuming we ever add semantics). Net: low-cost, low-risk experiment to defer until we have a need for semantics in the deployment story.

## Limitations / What This Paper Doesn't Solve
- **No params / GFLOPs reported.** Have to infer model size from `c` scaling; the paper does not give numbers. Architectural-cost analysis requires reimplementation.
- **AnyNet-style distance-cost-volume (`d - L feat - R_warped feat`) is less expressive than concat / group-wise CVs.** This is a known limitation of the AnyNet family; RTS2Net inherits it. The final D1-all of 3.56% on KITTI 2015 (Tab. V) confirms it lags state-of-the-art heavy networks by 1.2-1.7 percentage points.
- **Synergy module is the only place semantics actually helps stereo.** Tab. III shows pure parallel multi-task gives -0.01 EPE / -0.08 D1-all — basically noise. So "joint training is mutually beneficial" is a slogan; the real benefit requires the explicit synergy block.
- **Foreground D1 still much worse than state-of-the-art.** D1-fg 5.91% vs PSMNet 4.62% vs GANet 3.46%. The real-time constraint costs ~30-70% on foreground D1 — the regime that matters most for safety-critical driving.
- **No Scene Flow / cross-dataset / cross-domain evaluation.** Trained and evaluated entirely on Cityscapes + KITTI; we cannot tell whether the joint training also helps cross-domain generalisation (the central question for our project).
- **Tab. III is on `c = 8` only.** The synergy benefit at `c = 1` (the deployable-on-Pi config) is not measured — so we cannot tell whether the synergy refinement is *also* load-bearing at the tightest parameter budget, or whether it requires `c >= 8` worth of features to work.
