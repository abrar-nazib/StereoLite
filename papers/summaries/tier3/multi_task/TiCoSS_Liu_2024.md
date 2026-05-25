# TiCoSS: Tightening the Coupling between Semantic Segmentation and Stereo Matching within A Joint Learning Framework

**Authors:** Guanfeng Tang, Zhiyuan Wu, Jiahang Li, Ping Zhong, Wei Ye, Xieyuanli Chen, Huimin Lu, Rui Fan (Tongji University, NUDT, King's College London, Central South University)
**Venue:** IEEE Transactions on Automation Science and Engineering (TASE) 2025 (arXiv:2407.18038v4, July 2025)
**Tier:** 3 (joint segmentation + stereo, focuses on tightening coupling between branches; direct successor to S3M-Net)

---

## Core Idea
Most "joint" stereo + segmentation networks share an encoder, sum features once, and hope two heads cooperate. TiCoSS argues the cooperation is loose because heterogeneous features (RGB context, disparity geometry) get fused without filtering, and because per-task losses are computed independently. The paper attacks all three coupling surfaces at once: (1) a gated, tightly-coupled encoder fusion strategy (TGF), (2) a hierarchical deep supervision (HDS) strategy that pipes the highest-resolution fused features into every side branch, and (3) a coupling-tightening (CT) loss that adds a disparity-inconsistency-aware (DIA) term and a KL deep-supervision-consistency (DSCC) term on top of the S3M-Net SCG loss.

## Architecture
- **Duplex encoder**, one branch per modality (RGB and predicted disparity). The first three RGB layers share weights with the stereo-matching network's encoder (Sect. III-B, p. 5).
- Output scales: s/2, s/4, s/8, s/16 (Fig. 1, p. 4).
- **Selective Inheritance Gate (SIG)** at every layer; Eq. (1) p. 4. The gate G_i in [0,1]^(H_i x W_i) controls how much of the previous-layer fused feature is inherited; the remainder comes from the current-layer encoded feature. Built on top of Gated Fully Fusion (GFF, AAAI 2020).
- **Stereo branch** is taken unchanged from S3M-Net: feature pyramid + 3D correlation pyramid + multi-level GRU updater (RAFT-Stereo style). Sect. III-B p. 5 explicitly says they "adopt the stereo matching approach used in S3M-Net".
- **Hierarchical Deep Supervision (HDS):** the shallowest fused feature F_F^1 is downsampled by a Feature Dynamic Alignment (FDA) block (l stacked 3x3 stride-2 conv + BN + ReLU) and concatenated into every side auxiliary classifier. Sect. III-C, p. 5.
- **Coupling Tightening (CT) loss** = alpha * L_DIA + beta * L_DSCC + L_SCG + L_SM, with alpha = 1.5, beta = 1.0 selected via Fig. 7 sweep on KITTI. Eq. (4) p. 5.

## Main Innovation
The three changes are individually small; the contribution is the *combination* applied to a single chassis. Of the three, the **selective inheritance gate (SIG) inside the encoder** is the most concrete novelty: it is the first time gated fusion has been used as an *intermediate-encoding-stage* operator on heterogeneous (RGB + disparity) features rather than as a late-fusion decoder-side operator. Combined with the SCG-based weight map from S3M-Net, the design explicitly trains the encoder to ignore disparity-feature noise instead of indiscriminately summing it into RGB context.

## Key Benchmark Numbers

**Efficiency (Sect. IV-F, p. 11):**
- 385.05 M parameters, 308.86 GFLOPs at 512x256 input.
- Inference 0.30 s/image on RTX 3090 + Intel i7-13700KF (i.e., ~3.3 FPS).
- 5.82 GB GPU memory.
- Hardware reference: NVIDIA RTX 3090 (Sect. IV-B, p. 6).

**Semantic segmentation, KITTI 2015 (Tab. I, p. 8):**
- TiCoSS: Acc 91.90, mAcc 71.97, mIoU 63.63, fwIoU not in table (Tab IV mIoU 47.66), Pre 92.43, Rec 94.10, mFSc 92.90.
- S3M-Net baseline: mIoU 57.80, mAcc 65.90 -> +5.83 mIoU, +6.07 mAcc.
- vs SegFormer (NeurIPS'21) mIoU 51.39, Mask2Former mIoU 45.87, RoadFormer+ mIoU 57.69, DFormer (ICLR'24) mIoU 58.18.

**Semantic segmentation, vKITTI2 (Tab. II, p. 8):**
- TiCoSS: mIoU 88.46, mAcc 91.66.
- S3M-Net: mIoU 84.18, mAcc 88.24 -> +4.28 mIoU.

**Semantic segmentation, Cityscapes (Tab. III, p. 9):**
- TiCoSS: mIoU 68.36, mAcc 81.76.
- S3M-Net: mIoU 62.59 -> +5.77 mIoU.

**Stereo matching, KITTI 2015 (Tab. V, p. 10):**
- TiCoSS: EPE 0.54 px, PEP>1 10.39%, PEP>3 1.60%.
- S3M-Net: EPE 0.55, PEP>1 10.02, PEP>3 1.62 (so TiCoSS marginally better on EPE / PEP>3, marginally worse on PEP>1).
- vs IGEV-Stereo: EPE 0.62, PEP>1 12.15, PEP>3 1.99 (TiCoSS beats it).
- vs RAFT-Stereo: EPE 0.60.

**Stereo matching, vKITTI2 (Tab. V, p. 10):**
- TiCoSS: EPE 0.34 px, PEP>1 5.43%, PEP>3 2.58%.
- S3M-Net: EPE 0.38 -> -10.5% EPE.

## Mutual-Task Coupling: Load-Bearing or Decorative?

This is the one paper of the three where coupling is *explicitly* the contribution, but the ablation deltas are honest about what each piece is doing.

- **Encoder coupling (TGF):** Tab. VI p. 10. Baseline (S3M-Net w/o SCG loss): mIoU 54.33. + TGF on both branches: mIoU 59.06 (+4.73). + GFF (the prior-art fusion): mIoU 58.03. So TGF beats GFF by +1.03 mIoU. **Load-bearing for segmentation, marginal vs prior art.**
- **CT loss removal (Tab. IX p. 11):** Starting from baseline (TGF + HDS, no extra loss): mIoU 62.36. + DIA only: 63.04. + DSCC only: 62.88. + SCG only: 62.75. + all three: 63.63. Delta from full CT vs zero CT: **+1.27 mIoU.** Removing any single one of {DIA, DSCC, SCG} costs less than 1 mIoU. **Marginal at best.**
- **All three contributions (Tab. X p. 11):** TGF alone -> 59.06 mIoU. TGF + HDS -> 62.36. TGF + HDS + CT -> 63.63. So TGF is doing +4.73, HDS is doing +3.30, CT is doing +1.27.
- **Stereo branch impact:** This is the smoking gun. Tab. V p. 10. TiCoSS over S3M-Net on KITTI 2015: EPE 0.54 vs 0.55 (-0.01), PEP>1 10.39 vs 10.02 (*worse* by 0.37). On vKITTI2: EPE 0.34 vs 0.38, PEP>1 5.43 vs 5.56. Paper itself admits in Sect. IV-D-2: "the stereo matching performance of TiCoSS is slightly better than that of S3M-Net... improvements of 3.64% in EPE and 2.47% in PEP 3.0 on the KITTI dataset" (i.e., a delta of 0.02 px EPE).

Verdict: **Load-bearing for segmentation (TGF gate is real), window-dressing for stereo.** TiCoSS spends 385 M params and 309 GFLOPs to push segmentation mIoU up by ~6 over S3M-Net while moving stereo EPE by 0.01-0.04 px. The "tightened coupling" narrative is real for the segmentation direction (RGB encoder learns to filter disparity-encoded noise) but the stereo branch effectively rides along unchanged, and the loss-side coupling (DIA + DSCC) contributes only +1.27 mIoU on KITTI - small enough that it could be replaced by tuning weights on a single loss without losing the headline number.

## Relevance to Our Project
- **Param budget is incompatible.** 385.05 M params and 308.86 GFLOPs at 512x256 is two orders of magnitude over StereoLite's 2.5 M-param mid-tier envelope and >70 times slower than our 60 ms fp16 budget. Cannot port directly.
- **The SIG gate is the one transferable idea.** A tiny gated fusion (sigmoid gate G in [0,1]^(H,W), broadcast across channels) costs <0.05 M params and could replace the element-wise summation in any feature-mixing point of our chassis. If we ever add a YOLO26 detection head, SIG between detection features and stereo features is a cheap thing to A/B test on our 100-pair overfit harness.
- **Dataset story is brutal.** TiCoSS trains on KITTI 2015 (140 images) and vKITTI2 (500 images). Drones / mobile robots do not have these labels. We would need to generate semantic pseudo-labels via something like SAM2 or DINO-based open-vocab segmenters before this idea is even testable on our data.
- **DIA loss has a self-contained side use.** The left-right disparity inconsistency map W_N (Eq. 5-7 p. 5-6) is a free per-pixel confidence signal that does *not* depend on the segmentation branch existing. We could reuse it in the StereoLite loss as a weighted L1 over inconsistent regions and get the regularisation without paying for the segmentation head.
- **The "tight coupling" narrative is dataset-bound.** All wins were measured on Driving (KITTI / vKITTI / Cityscapes). Zero zero-shot Middlebury 2014 / ETH3D numbers, so we cannot tell whether the gating actually helps cross-domain stereo - which is the only stereo question StereoLite currently cares about (CLAUDE.md "Cross-domain catastrophic failure" lesson).

## Limitations / What This Paper Doesn't Solve
- **No real-time deployment.** 0.30 s/image on RTX 3090 is unusable for drone navigation; the conclusion (Sect. V, p. 11) explicitly punts this to future work: "we plan to further improve the efficiency of the framework... resource-constrained hardware."
- **Stereo branch is unchanged from S3M-Net.** The "coupling" idea is one-directional: better RGB features for segmentation. The disparity head still bottoms out at 0.54 px EPE on KITTI, no better than vanilla S3M-Net.
- **Annotation requirement is severe.** Sect. V p. 11: "TiCoSS still requires both semantic and disparity annotations, and collecting data with such ground truth remains a labor-intensive process." This kills any deployment-style retraining cycle.
- **The +9% mIoU headline claim is dataset-cherry-picked.** The +10.57% on KITTI (Tab. I) is real, but vKITTI2 gain is +4.28 and Cityscapes is +5.77. The 9% figure averages with KITTI 2015 mIoU as the lever.
