# SGNet: Semantics Guided Deep Stereo Matching

**Authors:** Shuya Chen, Zhiyu Xiang, Chengyu Qiao, Yiman Chen, Tingming Bai (Zhejiang University)
**Venue:** Asian Conference on Computer Vision (ACCV) 2020
**Tier:** 3 (joint disparity + semantic seg, three-pronged "guidance" modules on top of PSMNet; the cleanest semantic-stereo ablation on KITTI in this lineage)

---

## Core Idea
SGNet wraps PSMNet with three additive "semantic guidance" modules whose only job is to refine disparity. (1) A **confidence module** correlates left/right disparity features and left/right semantic features separately, multiplies the two correlation maps, and uses the result as a per-disparity confidence map to weight the early cost volume — the rationale is "if two pixels do not share a semantic label, they are unlikely to be the correct match". (2) A **residual module** takes the initial disparity, splits it into C category-channels by multiplying with the softmax of the segmentation probability map, then runs depthwise conv per channel and integrates with a pointwise conv, producing a per-category residual correction. (3) A **loss module** adds two semantic-supervision terms, one for boundaries (only fire where the disparity *also* has a boundary) and one for inner-region smoothness (suppressed where the disparity gradient exceeds a threshold lambda = 3 px).

## Architecture
- **Disparity backbone:** PSMNet exactly as in Chang & Chen 2018 — feature extractor + spatial pyramid pooling + 3D cost volume (`D_max = 192`) + stacked hourglass cost aggregation + three regression outputs disp1, disp2, disp3 (Sect. 3.1, p. 4).
- **Semantic branch:** shares shallow layers with disparity, then two additional ResNet blocks at 256 channels + PSPNet-style pyramid pooling + classification head (Sect. 3.2, p. 5).
- **Confidence module (Fig. 3a, p. 6):** at 1/4 resolution, feed disparity feature pair and semantic feature pair into separate correlation layers per Eq. 1. Multiply the two correlation outputs (chosen via Tab. 1 ablation, "Disp-cor x Seg-cor" beats addition by 0.063 percentage points on 3px). Three 3D conv layers + residual structure + sigmoid produces confidence values that multiply into disp1's cost volume (Sect. 3.4, p. 6-7).
- **Residual module (Fig. 2, p. 5):** input is disp3 (H x W) and semantic probability (H x W x C). Element-wise multiply to give H x W x C "category-wise raw disparity". Depthwise conv (separately per category), pointwise conv to integrate, transposed conv to give the disparity residual added back into disp3 to produce disp4 (the final output) (Sect. 3.3, p. 5-6).
- **Loss module:**
  - `L_bdry` (Eq. 4, p. 8): `|grad^2 sem| * e^{-|grad^2 d|}` with a category mask `m_b` that *excludes* road / sidewalk / vegetation / terrain (Eq. 5, p. 8). The intent is "supervise boundary terms only where semantic boundaries should also be disparity boundaries."
  - `L_sm` (Eq. 6, p. 9): `|grad^2 d| * e^{-|grad^2 sem|}` with a mask `m_s` that excludes pixels where `|grad d| > lambda = 3`. The intent is "demand smoothness only where the disparity itself is already smooth."
  - Total: `L = L_disp + 1.0 * L_sem + 0.5 * L_bdry + 0.5 * L_sm` (Eq. 8, p. 9). `L_disp` is multi-output: `0.5 * L_disp1 + 0.7 * L_disp2 + 1.0 * L_disp4` (Eq. 9, p. 9). disp3 is unsupervised; disp4 = disp3 + residual is the test-time output.

## Main Innovation
The most original of the three modules is the **confidence module** that uses cross-task correlation consistency (semantic-correlation x disparity-correlation) as a per-disparity reliability score — this is *not* feature concatenation (the SegStereo / DispSegNet move) but a meta-signal about how trustworthy the cost volume is. The residual module's per-category depthwise conv is a cheap way to give the network the "different categories smooth differently" prior without spending many params: it costs one (C-channel) depthwise + one C-to-1 pointwise, a few hundred parameters per scale.

## Key Benchmark Numbers
- **Params:** not in paper.
- **GFLOPs:** not in paper.
- **Latency / FPS / target GPU:** **0.674 s/image** on **NVIDIA TITAN 1080 Ti** (Tab. 3, p. 11) for full Baseline-CRL, vs 0.671 s for the PSMNet baseline. ~1.5 FPS.

**KITTI 2015 leaderboard test, ALL D1-bg / D1-fg / D1-all, NOC D1-bg / D1-fg / D1-all (Tab. 4, p. 12):**
- SGNet: **1.63 / 3.76 / 1.99 / 1.46 / 3.40 / 1.78**.
- AANet+: 1.65 / 3.96 / 2.03 / 1.49 / 3.66 / 1.85.
- EdgeStereo-V2: 1.84 / 3.30 / 2.08 / 1.69 / 2.94 / 1.89.
- SSPCV-Net: 1.75 / 3.89 / 2.11 / 1.61 / 3.40 / 1.91.
- SegStereo: 1.88 / 4.07 / 2.25 / 1.76 / 3.70 / 2.08.
- PSMNet baseline: 1.86 / 4.62 / 2.32 / 1.71 / 4.31 / 2.14.

**KITTI 2012 leaderboard (Tab. 5, p. 14), Noc / All for 2/3/4/5 px:**
- SGNet: 2.22 / 2.89 / 1.38 / 1.85 / 1.05 / 1.40 / 0.86 / 1.15. Beats PSMNet on every Noc metric.

**Virtual KITTI 2 val (Tab. 3, p. 11):** SGNet baseline-CRL **3.874% 3px / 0.5892 EPE**, vs PSMNet baseline 4.108 / 0.6237. **Delta = -0.234 / -0.035.**

**Scene Flow:** mentioned only as a pretrain dataset (Sect. 4, p. 9-10), no reported numbers.

**Semantic:** mIoU 48.12% and mAcc 55.25% on KITTI 2015 val (Sect. 4.1, p. 12). No comparison to standalone PSPNet baseline.

**Latency:** training on a single 1080 Ti with batch=2 — the model is heavy.

## Mutual-Task Coupling: Load-Bearing or Decorative?
Tab. 3 (p. 11) gives the cleanest module-by-module ablation in the multi-task stereo lineage:

KITTI 2015 val (Scene Flow + KITTI 2015 finetuning):
| Variant | Conf | Res | Loss | 3px (%) | EPE (px) |
|---|---|---|---|---|---|
| Baseline (PSMNet) | | | | 1.415 | 0.6341 |
| Baseline-C | yes | | | 1.371 | 0.6275 |
| Baseline-R | | yes | | 1.368 | 0.6253 |
| Baseline-CR | yes | yes | | 1.328 | 0.6203 |
| **Baseline-CRL** | yes | yes | yes | **1.299** | **0.6198** |

Deltas from baseline:
- Confidence alone: -0.044 / -0.0066.
- Residual alone: -0.047 / -0.0088.
- Confidence + Residual: -0.087 / -0.0138.
- + Loss module: -0.116 / -0.0143. **The loss module gives only 0.029% 3px and 0.0005 EPE on top of C+R — basically noise.**

Verdict: **Load-bearing for stereo at the level of 0.1 percentage point on 3px-error, which is the single-digit-noise range on KITTI 2015 val (40 images).** The confidence module and residual module each carry their weight; the loss module is borderline cosmetic — it could be removed and the paper's headline KITTI 2015 D1-all of 1.99% would be essentially unchanged. SGNet's *honest* contribution is "two small architectural blocks each worth ~0.05% on 3px-error," not "three contributions each worth a section title."

The 1.999% D1-all on KITTI 2015 test (Tab. 4) is also small vs prior art: AANet+ at 2.03% is within 0.04 D1-all, no semantic branch at all. So **the semantic branch buys 0.04 D1-all over a non-semantic state-of-the-art network of similar architectural budget.** This is exactly the regime where a confounder (extra parameters, longer training schedule, or PSMNet-specific tuning) could explain the entire delta.

## Relevance to Our Project
- **The category-wise depthwise-conv residual is the most transferable idea.** Multiplying disparity by softmax(seg) to get a C-channel volume, then doing a tiny depthwise conv per channel, is ~C * (3*3) + C * 1 = O(10*C) parameters per scale. At C = 8 (our overfit class budget for indoor scenes), this is <100 params per scale. Could be tested as a "per-class refinement gate" in the TileRefine block on the 100-pair harness.
- **Cross-correlation confidence is interesting but expensive.** Two separate correlation layers (one on disparity features, one on semantic features) at H/4 x W/4 doubles cost-volume construction time. For our 1/16 single-CV chassis this is not free. Defer.
- **The boundary / smooth loss masks could replace the bad-1 hinge in our cocktail.** SGNet's `m_b` (only supervise boundary error where semantic boundary exists *and* the pixel is not road/sidewalk/vegetation/terrain) and `m_s` (only supervise smoothness where `|grad d| < lambda`) are exactly the kind of selective hinge that a tuned `stack_d1` lacks. Translatable to our regime: replace "semantic boundary" with "edge from Sobel filter on the image" and "non-road class" with "non-bottom-half-of-image" (cheap driving-scene prior).
- **The PSMNet backbone is heavy.** PSMNet alone is 5.2 M params and 0.41 s on Titan-Xp; SGNet at 0.674 s on a 1080 Ti is heavier still. Cannot port the architecture itself.
- **The "joint-training barely helps" finding aligns with TiCoSS and DispSegNet.** Each multi-task paper finds the semantic coupling buys roughly 0.05-0.2 percentage points on KITTI D1 vs a similarly-sized non-semantic baseline. The pattern is consistent: semantic supervision is a small regulariser, not a quantum leap.
- **No cross-domain (Cityscapes / MB14 / ETH3D) numbers.** Same blind spot as the prior two papers.

## Limitations / What This Paper Doesn't Solve
- **Loss module is cosmetic.** Only 0.029 percentage points on 3px-error beyond C+R (Tab. 3, p. 11). The paper presents it as one of three contributions, but the ablation does not support that framing.
- **No Scene Flow / cross-dataset eval.** Scene Flow is used only as pretrain; the paper does not report Scene Flow test numbers, so we cannot tell how SGNet's semantic guidance behaves on the only large-scale "non-driving" synthetic stereo dataset.
- **Semantic quality is undermeasured.** mIoU = 48.12% on KITTI 2015 val (Sect. 4.1) but no comparison to a standalone PSPNet at matched parameter count. Joint training might be hurting segmentation as it did in DispSegNet.
- **40-image validation split** is the basis for every ablation number. 0.04 percentage-point deltas on 40 images is in the noise — most of the ablation table is statistically indistinguishable.
- **No params / FLOPs / memory reported.** Makes deployment cost analysis impossible.
- **Inference time only on 1080 Ti.** No edge / embedded discussion.
