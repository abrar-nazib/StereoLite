# S3M-Net: Joint Learning of Semantic Segmentation and Stereo Matching for Autonomous Driving

**Authors:** Zhiyuan Wu, Yi Feng, Chuang-Wei Liu, Fisher Yu, Qijun Chen, Rui Fan (Tongji University, ETH Zurich)
**Venue:** IEEE Transactions on Intelligent Vehicles (TIV) 2024, Vol. 9, No. 2 (arXiv:2401.11414v2)
**Tier:** 3 (joint segmentation + stereo via shared RGB encoder; baseline for TiCoSS)

---

## Core Idea
RGB features for stereo matching and RGB features for semantic segmentation share most of the same information; running two separate encoders wastes both compute and the regularisation benefit of joint learning. S3M-Net wires a single RAFT-Stereo-style joint encoder into both tasks: the same shared feature pyramid drives a multi-level GRU disparity head AND a feature-fusion-adaptation (FFA) module that remaps shared features into the segmentation space and concatenates them with disparity-derived features. The whole pipeline is trained end-to-end under one semantic-consistency-guided (SCG) loss that re-weights per-pixel loss by how spatially "isolated" a class label is, emphasizing inter-class boundaries.

## Architecture
- **Joint encoder** = RAFT-Stereo encoder (residual blocks + downsampling), shared across stereo and segmentation branches. Outputs F_L = {F_1^L, ..., F_n^L} and F_R, both at multiple scales (Sect. III-A, p. 3).
- **Stereo branch:** 3D correlation volume C_1(i,j,k) = F_n^L(i,j,:) . F_n^R(i,k,:), with C_m built by 1D avg-pool on C_{m-1}. Multi-level GRU updater (RAFT-Stereo paradigm) produces disparity sequence D = {D_1, ..., D_n} from initial D_0 = 0 (Sect. III-B, p. 3-4; Eq. 1 p. 3).
- **FFA module** (Sect. III-C, p. 4): adapts shared features F_L to the semantic space and fuses them with disparity-derived features. Channel-count progression in the remapping operation R: 64 -> 256 -> 512 (Sect. III-C). Disparity is encoded via a ResNet-152 backbone, reaching 1024 and 2048 channels post-encoding (i.e., the disparity branch into segmentation is heavyweight).
- **Decoder:** SNE-RoadSeg densely-connected skip-connection decoder; 3x3 kernel, stride 1, padding 1 (Sect. III-D, p. 4).
- **Fusion operation:** ablated across 6 choices in Tab. V p. 10 (Addition / Concat / CFM / DDPM / SA Gate / SWS). Winner is plain element-wise **Addition**.

## Main Innovation
The **Semantic Consistency-Guided (SCG) loss**, not the architecture. Specifically: build a 3D one-hot volume V^3D from the ground-truth segmentation map (Eq. 5 p. 4), average-pool per channel to get an inter-class volume V^I (Eq. 6), apply V^N(p) = exp(-(2 V^I(p) - 1)^2) to get a normalisation (Eq. 7 p. 4), then take W(p) = max_c V_c^N(p) as a per-pixel weight (Eq. 8 p. 5). This W is highest at *boundaries* (where average-pooling gives V^I near 0.5, exp argument near 0) and lowest inside large piecewise-constant regions. The same W is then plugged into both the segmentation cross-entropy and the stereo regression loss (Eqs. 10-11 p. 5) with mixing weight alpha = 0.1. So the network is told: spend your gradient budget on inter-class boundaries, where both tasks fail in correlated ways.

## Key Benchmark Numbers

**Efficiency (Sect. V, p. 11):**
- 0.66 FPS on RTX 3090 at 1248 x 384 input (i.e., ~1.5 s/image).
- Trainable parameter count: not in paper (the FFA disparity encoder is ResNet-152, so total is large).
- Hardware: NVIDIA RTX 3090, batch=1, AdamW (eps=1e-8, wd=1e-5), lr=2e-4 (Sect. IV-B, p. 5).

**Semantic segmentation, vKITTI2 (Tab. I, p. 6):**
- S3M-Net (w/ SCG): Acc 98.32, mAcc 88.24, mIoU 84.18, fwIoU 96.98, mFSc 98.31.
- vs SegFormer mIoU 64.98, RoadFormer mIoU 80.83, Mask2Former mIoU 57.14.
- S3M-Net w/o SCG: mIoU 84.25 -> nearly identical to with-SCG, marginal -0.07.

**Semantic segmentation, KITTI 2015 (Tab. II, p. 6):**
- S3M-Net (w/ SCG): mIoU 57.80, mAcc 65.90, mFSc 91.80.
- S3M-Net w/o SCG: mIoU 54.33, mAcc 62.48 -> +3.47 mIoU from SCG loss.
- vs RoadFormer mIoU 55.13, SegFormer mIoU 51.39, DeepLabv3+ mIoU 42.79.

**Stereo matching, vKITTI2 (Tab. III, p. 9):**
- S3M-Net (w/ SCG): EPE 0.38 px, PEP>1 5.56%, PEP>3 2.55%.
- S3M-Net w/o SCG: EPE 0.39, PEP>1 5.59, PEP>3 2.55 -> SCG delta is -0.01 px EPE.
- vs RAFT-Stereo (the architectural parent): EPE 0.40, PEP>1 5.88.
- vs IGEV-Stereo: EPE 0.47.

**Stereo matching, KITTI 2015 (Tab. IV, p. 9):**
- S3M-Net (w/ SCG): EPE 0.55 px, PEP>1 10.02%, PEP>3 1.62%.
- S3M-Net w/o SCG: EPE 0.56, PEP>1 10.33 -> SCG delta is -0.01 px EPE.
- vs RAFT-Stereo: EPE 0.60.

**Fusion-strategy ablation (Tab. V, p. 10), KITTI 2015:**
- Addition: mIoU 54.33, mFSc 90.65 (winner).
- Concatenation: mIoU 48.40.
- CFM: mIoU 48.77.
- DDPM: mIoU 49.51.
- SA Gate: mIoU 52.10.
- SWS: mIoU 49.64.

**Datasets used:**
- vKITTI2: 500 train / 200 val pairs (Sect. IV-A, p. 5).
- KITTI 2015: 200 GT-annotated pairs split 70/30 train/test.

## Mutual-Task Coupling: Load-Bearing or Decorative?

This is the test case for "does joint learning of stereo + seg actually help either task?" The numbers are clearer than the prose.

- **SCG loss on segmentation:** KITTI 2015 mIoU 54.33 -> 57.80 (+3.47). vKITTI2 mIoU 84.25 -> 84.18 (-0.07). On large/easy data the loss does literally nothing; on small/hard real data it adds +3.47 mIoU. **Conditionally load-bearing.**
- **SCG loss on stereo:** vKITTI2 EPE 0.39 -> 0.38 (-0.01 px). KITTI 2015 EPE 0.56 -> 0.55 (-0.01 px). **Decorative for stereo.** The improvements over RAFT-Stereo (0.40 -> 0.39 on vKITTI, 0.60 -> 0.56 on KITTI) come from the shared encoder regularisation, not the SCG weighting per se.
- **Shared encoder regularisation:** Comparing S3M-Net-w/o-SCG against single-task RAFT-Stereo (vKITTI EPE 0.40 vs 0.39; KITTI EPE 0.60 vs 0.56) suggests the joint training gives a +0.01 to +0.04 px boost on stereo regardless of the loss. Modest but real.
- **The fusion strategy matters more than the SCG loss.** Tab. V p. 10 shows mIoU swinging from 48.40 (concatenation) to 54.33 (addition) - a 6-point swing on KITTI 2015 - far larger than the 3.47 mIoU SCG gain. The "magic" is mostly the FFA module + simple addition fusion.
- **The cross-task signal is asymmetric.** Segmentation gets a clean win from the joint chassis. Stereo gets a marginal one. The paper effectively builds a segmentation network that happens to also produce a disparity map for free, not a stereo network that benefits from segmentation.

Verdict: **Marginal-to-decorative for stereo, conditionally load-bearing for segmentation on small real datasets.** SCG loss earns its keep on KITTI 2015 (+3.47 mIoU) but does nothing on vKITTI2. The joint chassis gives stereo a tiny constant boost over single-task RAFT-Stereo across both datasets that comes from shared-encoder regularisation, not from the seg labels supervising the stereo branch.

## Relevance to Our Project
- **Architectural shape is wrong for us.** S3M-Net uses ResNet-152 inside the FFA module - that alone is ~60 M params, an order of magnitude over the entire StereoLite mid-tier (2.06 M). The "shared encoder + two heads" *idea* is reusable, but the channel widths and depths are not.
- **SCG loss is the cheapest steal.** Building W from a segmentation pseudo-label (e.g., from SAM2 or a SegFormer-mini) and plugging it into our `stack_d1` stereo loss is a 30-line change in `training/losses.py` and costs zero inference compute. Worth A/B'ing on the 100-pair Modal harness: does boundary-weighted L1 help cross-domain MB14 zero-shot? The bad-X metrics are exactly where boundary weighting should help.
- **FFA addition-fusion ablation reinforces an existing prior.** Concatenation and gated fusion underperform plain addition for cross-task feature merging (Tab. V p. 10). When we wire any auxiliary branch into StereoLite (depth pseudo-GT, segmentation pseudo-GT), start with addition and only escalate to gated fusion if addition fails.
- **Cross-domain story is missing.** All numbers are vKITTI2 / KITTI 2015. No SceneFlow, no Middlebury 2014, no ETH3D zero-shot. We cannot tell whether SCG-loss boundary weighting helps the textureless / repetitive-structure failure mode (the MB14 catastrophic failure CLAUDE.md highlights).
- **0.66 FPS rules out direct deployment.** The shared-encoder *concept* could be re-implemented at our budget, but their actual checkpoint is unusable on edge hardware.

## Limitations / What This Paper Doesn't Solve
- **Disparity branch is RAFT-Stereo without iteration-count gains.** The chassis is structurally identical to RAFT-Stereo, and the small EPE deltas (-0.01 px from SCG) suggest the joint-learning lever has only weak coupling into the stereo loss.
- **Real-time deployment ruled out:** Sect. V, p. 11: "S3M-Net achieves a processing speed of 0.66 FPS... further computational efficiency optimizations are necessary".
- **Requires both labels.** Same as TiCoSS - both stereo GT and segmentation GT, on the same scenes, which is rare. No semi-supervised story.
- **vKITTI2 ceiling effect.** Most methods already exceed 80% mIoU on vKITTI2, leaving little room for the SCG loss to demonstrate boundary improvement; the -0.07 mIoU delta there is noise. The paper's claim of universal SCG benefit isn't well-supported by its own numbers.
