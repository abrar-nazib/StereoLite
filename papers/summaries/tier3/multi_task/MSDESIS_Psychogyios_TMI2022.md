# MSDESIS: Multi-task stereo disparity estimation and surgical instrument segmentation

**Authors:** Dimitrios Psychogyios, Evangelos Mazomenos, Francisco Vasconcelos, Danail Stoyanov (Wellcome/EPSRC Centre for Interventional and Surgical Sciences, UCL)
**Venue:** IEEE Transactions on Medical Imaging (TMI) 2022, DOI 10.1109/TMI.2022.3181229
**Tier:** 3 (joint stereo + binary segmentation in a surgical / endoscopic setting; primarily a *training-pipeline* contribution rather than a new block)

---

## Core Idea
In stereoscopic endoscopy, stereo training data is almost nonexistent (the SCARED challenge had ~17 usable key-frames after cleaning) but monocular *segmentation* data is comparatively abundant (RIS has ~2400 annotated frames). MSDESIS proposes a modular shared-encoder, two-head architecture (one disparity head, one segmentation head) and a multi-phase training schedule that lets segmentation supervision on monocular data act as **domain adaptation for the disparity head** without ever needing surgical stereo GT. The headline finding is that fine-tuning only the segmentation head + shared encoder on RIS reduces SCARED disparity EPE by 77.73% and depth MAE by 61.73% relative to the FlyingThings3D pretrain.

## Architecture
- **Shared feature encoder** at scales S_i, i in (2,3,4,5,6). Two variants tested:
  - **Lightweight** (modified MADNet): channels 16/32/64/96/128 across scales, 0.5 M encoder params (Sect. III-A.1, p. 3).
  - **ResNet34:** channels 64/64/128/256/512, 21 M encoder params (Sect. III-A.1, p. 3).
- **Modifications to MADNet encoder:** drop the s/64 output scale; add batch-norm between LeakyReLU(0.1) and the 3x3 convs; first conv per block is stride-2 (Sect. III-A.1, p. 3).
- **Disparity head** (Sect. III-A.2, p. 4): 2D *cascade cost volume* at scales S_2..S_6. At each scale i, right features F_r^i are warped by the up-sampled previous-scale disparity D'_{i+1}, then a bidirectional shallow cost volume with **+/-2 disparity search range** is built and concatenated with F_l^i. At S_6 only (no previous disparity), a unidirectional cost volume of depth 320/32 + 1 = 11 is built. Full-resolution refinement at S_1 uses a 2D hourglass module on D_2 (no cost volume) - the paper notes this gives ~2 px tolerance to vertical rectification error.
- **Segmentation head** (Sect. III-A.3, p. 4): U-Net-style with skip connections from s/2 down. Upsampling uses 4x4 transposed conv stride 2. Each block = 2x 3x3 conv + BN + LeakyReLU(0.1). Final sigmoid + threshold 0.5.
- **Max disparity search range: 320 pixels** (configured via cost-volume depth at S_6; Sect. III-A.2).
- **Element-wise multiply + mean across channels** instead of dot product in cost-volume construction, to avoid fp16 overflow in mixed-precision training (Sect. III-A.2).

## Main Innovation
Not the architecture - the *training scheme*. The paper's true contribution is showing that **monocular segmentation supervision can act as a stereo domain adaptation signal via the shared encoder**, eliminating the need for surgical stereo GT. Concretely: phase 1 pretrains on FlyingThings3D under full disparity supervision; phase 2 fine-tunes by *only* supervising segmentation (with monocular RIS data, no stereo GT, no disparity loss). The shared encoder shifts toward surgical-domain statistics, and the disparity head - which was never touched - rides that domain shift into 61% lower depth error on SCARED. This is a clean inversion of the usual "more labels of task X to help task X" recipe: more labels of task Y, on a different dataset, with no joint stereo pairs, help task X.

## Key Benchmark Numbers

**Efficiency (Tab. III, p. 12):**
- Lightweight encoder: 0.47 M params, 1243 MB GPU, encoder-only inference time 2.27 s / 1000 frames at 1280x1024.
- Lightweight disparity head: 2.56 M params, 1897 MB.
- Lightweight segmentation head: 1.20 M params, 1393 MB.
- **Lightweight multi-task total: 3.30 M params, 1933 MB GPU, 37.23 s / 1000 frames at 1280x1024** -> **~22 FPS** including data loading on Tesla V100 (Sect. VI-F, p. 12; abstract).
- ResNet34 multi-task: 29.49 M params, 2775 MB, 79.56 s / 1000 frames.
- Hardware: Nvidia DGX Station V100 (32 GB), single GPU, mixed precision (Sect. V-B, p. 7).
- Comparison: RAFT-Stereo 683.27 s / 1000 frames, CFNet 567.32 s, DeepPrunner 321.77 s (all single-task, same 1280x1024 input) -> MSDESIS lightweight multi-task is **~18 times faster than RAFT-Stereo** and produces both outputs.

**Disparity on FlyingThings3D pretrain (Sect. VI-A, p. 8):**
- ResNet34-ph1-disp: EPE 1.37 px, Bad3 5.54%.
- Lightweight-ph1-disp: EPE 1.29 px, Bad3 6.39%. (Notable: lightweight beats ResNet34 on EPE.)

**Disparity / depth on SCARED test, all variants (Tab. I, p. 9):**

| Variant | Light EPE / Bad3 / Depth | ResNet34 EPE / Bad3 / Depth |
|---|---|---|
| ph1-disp (pretrain only) | 17.02 / 38.76% / 8.91 mm | 67.16 / 69.98% / 25.61 mm |
| ph2-disp (only disparity FT) | 2.92 / 28.62% / 2.85 mm | 4.41 / 35.46% / 3.72 mm |
| **ph2-seg (only segmentation FT, no stereo GT)** | **3.79 / 31.47% / 3.41 mm** | **9.16 / 40.50% / 6.37 mm** |
| ph2-multitask | 4.45 / 33.71% / 3.91 mm | 7.22 / 38.37% / 5.84 mm |
| ph3-multi2disp | 3.04 / 29.37% / 2.98 mm | 4.78 / 41.01% / 4.28 mm |
| ph3-multi2seg | 4.11 / 33.11% / 3.61 mm | 6.13 / 36.43% / 5.08 mm |
| ph3-seg2disp | 3.08 / 28.85% / 2.93 mm | 3.08 / 32.57% / 3.03 mm |
| **ph3-disp2seg (best overall)** | **3.46 / 31.62% / 3.18 mm** | 9.89 / 40.85% / 6.37 mm |

**Segmentation on RIS test (Tab. I p. 9 + Tab. II p. 11):**
- light-ph2-seg: 89.08% mIoU.
- light-ph3-disp2seg: **89.15% mIoU** (best lightweight).
- resnet34-ph3-multi2seg: **90.46% mIoU** (best ResNet).
- vs TernausNet (MIT submission): 88.80%.
- vs ST-MTL (RIS prior SOTA): 91.00% -> MSDESIS is 1.85% below SOTA but 18x faster.

**Reduction from segmentation-only domain adaptation (Sect. VI-B, p. 9):**
- Light: depth 8.91 -> 3.41 mm = **-61.73%**; EPE 17.02 -> 3.79 = **-77.73%**.
- ResNet34: depth 25.61 -> 6.37 = -75.13%; EPE 67.16 -> 9.16 = -86.36%.

**Loss weights (Sect. V-B.2, p. 7):**
- Multi-task L_mt = alpha_mt * L_ss-disp + (1 - alpha_mt) * L_seg, alpha_mt = 0.2.
- Self-supervised disparity L_ss-disp = beta_ss * (alpha_ss * L_ph + (1-alpha_ss) * L_ssim) + (1-beta_ss) * L_smooth, with alpha_ss = 0.9, beta_ss = 0.7.
- WBCE class weight beta = (1 - 0.15) / 0.15 (tool pixels cover ~15% of frame area).

## Mutual-Task Coupling: Load-Bearing or Decorative?

This is the *only* one of the three papers where cross-task coupling demonstrably moves the needle by a huge factor, and it's also the most counter-intuitive coupling direction.

- **Segmentation supervision -> stereo accuracy (no shared loss, no stereo GT):** light-ph1-disp EPE 17.02 -> light-ph2-seg EPE 3.79 (Tab. I p. 9). That is a 77.73% EPE drop driven *entirely* by gradient flow through the shared encoder when supervising segmentation. **Massively load-bearing.**
- **Joint multi-task vs single-task disparity:** light-ph2-multitask EPE 4.45 vs light-ph2-disp EPE 2.92 (Tab. I p. 9). Joint training is *worse* than disparity-only training on disparity when surgical stereo GT is available. The joint chassis loses 1.53 px EPE. **Coupling is not always helpful when both labels exist.**
- **Joint multi-task vs single-task segmentation:** light-ph2-multitask mIoU 66.50 vs light-ph2-seg mIoU 89.08 (Tab. I p. 9). The seg head is **worse by 22.6 mIoU** when trained jointly on the smaller RIS_12 stereo-rectified subset vs alone on the full RIS. The paper attributes this to "RIS_12 has significantly fewer data" (Sect. VI-B p. 9), not the joint training. The Fig. 6 ablation (Sect. VI-E p. 11) confirms: trained on the same RIS_12, multi-task ~= seg-only on mIoU but multi-task is *better* on disparity.
- **Phase 3 (segmentation -> stereo subsequent training):** light-ph3-seg2disp EPE 3.08, slightly *worse* than light-ph2-disp 2.92 but with the added benefit that the model can now also predict segmentation. The "best overall" light-ph3-disp2seg achieves 3.18 mm depth + 89.15% mIoU - within 0.3 mm of disparity-best AND within 2 mIoU of segmentation-best, on the same network.
- **MADNet encoder vs ResNet34 in domain shift:** Tab. I p. 9 row "ph1-disp" - on FlyingThings3D the two backbones are nearly tied (EPE 1.29 vs 1.37) but on out-of-domain SCARED the lightweight wins decisively (EPE 17.02 vs 67.16). The bigger backbone overfit FlyingThings3D and generalises *worse*. This is an important data point for our own backbone-selection work.

Verdict: **Genuinely load-bearing** when the coupling is used as a domain-adaptation mechanism (segmentation labels carry the encoder into the surgical domain), with -77.73% EPE deltas that no other architectural change in the three papers comes close to matching. **Marginal-to-negative** when used as a vanilla joint-supervision regularizer (the ph2-multitask numbers are typically worse than the per-task fine-tunes). The win is in the *training pipeline*, not in joint loss minimisation.

## Relevance to Our Project
- **The training-pipeline insight transfers directly to our LiteAnyStereo path.** Our 3-stage KD plan (CLAUDE.md "Current research direction") follows the exact MSDESIS template: pretrain on synthetic stereo (FlyingThings3D for them, SceneFlow Driving for us), then domain-adapt via a different supervisory signal that does NOT require stereo GT (segmentation labels for them, FoundationStereo pseudo-labels for us). MSDESIS provides empirical backing that this works: -77.73% EPE on cross-domain SCARED.
- **Lightweight backbone wins on out-of-domain (OOD) generalisation.** Tab. I p. 9: ph1-disp shows the 0.5 M-param MADNet encoder outperforms the 21 M-param ResNet34 by 50 px EPE on SCARED despite being nearly tied on FlyingThings3D. This is direct support for our two-tier "smaller is more generalisable" instinct and our concern about yolo26s overfitting on Driving (CLAUDE.md edge-tier story).
- **The architecture is in our param ballpark.** 3.30 M params multi-task at 22 FPS on V100 is roughly comparable to our 2.06 M yolo26s mid-tier; their 0.5 M encoder + 2 M disparity head is right at our edge-tier 0.87 M envelope. Their pyramidal MADNet variant + 2D cost-volume cascade design is a credible *alternative* chassis to test against StereoLite.
- **The cascade cost volume with ±2 search at fine scales is reusable.** Our current cost volume rebuilds full max_disp at each scale; MSDESIS only searches +/-2 once the previous-scale disparity is available. This is exactly the "search range compression" trick we should A/B test for the mid-tier (Sect. III-A.2 p. 4).
- **Useful negative result for joint training.** The ph2-multitask numbers (worse than per-task fine-tunes) are a warning: if we ever wire a segmentation pseudo-label head onto StereoLite, expect the stereo metrics to *regress* unless we follow MSDESIS's phase-3 sequential training.
- **Surgical-domain specifics don't transfer.** Tool segmentation is binary; navigation segmentation is multi-class. The class-imbalance WBCE weight beta = (1-0.15)/0.15 will not apply directly. But the *structure* of the multi-phase pipeline is what we want.

## Limitations / What This Paper Doesn't Solve
- **No iterative refinement.** The disparity head is a single forward pass; no GRU, no iterative cost-volume re-sampling. This is fine for surgical (small disparity ranges, smooth surfaces) but is exactly the regime the cross-domain stereo literature suggests iterative methods would close further.
- **Self-supervised disparity term is fragile.** The photometric + SSIM + smoothness self-supervision in L_ss-disp does not handle occlusions or fast-moving objects in time-unsynchronized stereo channels (Sect. VI-B, p. 9). This is exactly why ph2-seg beats ph2-multitask on disparity in their setup.
- **Surgical evaluation only.** No KITTI, SceneFlow, or Middlebury numbers; we cannot directly compare against natural-scene stereo baselines.
- **Not state-of-the-art on either task.** 3.18 mm depth MAE vs SCARED best 2.33 mm (DeepPrunner), 89.15% mIoU vs RIS best 91.00% (ST-MTL). The contribution is the *speed* and the *adaptation pipeline*, not raw accuracy.
- **Disparity head's pure-Python cost-volume construction is the inference bottleneck.** Sect. VI-F p. 12: "most of the inference time in the disparity head is spent during the cost volume construction phase which is the same regardless of the variant and is implemented in pure python code." Edge deployment would require a CUDA-kernel rewrite.
