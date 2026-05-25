# Multi-Task Learning for Dense Prediction Tasks: A Survey

**Authors:** Simon Vandenhende, Stamatios Georgoulis, Wouter Van Gansbeke, Marc Proesmans, Dengxin Dai, Luc Van Gool (KU Leuven PSI, ETH Zurich CVL)
**Venue:** IEEE TPAMI 2022
**Tier:** 3 (survey of deep multi-task learning architectures and optimization for dense prediction; first comprehensive review structured around encoder-focused vs decoder-focused taxonomy, with apples-to-apples re-implementation on NYUD-v2 and PASCAL)

---

## Core Idea
By 2021 the multi-task learning (MTL) literature for dense prediction had splintered into two loosely-connected camps (architecture design and optimization / loss balancing), with inconsistent benchmarks across papers. This survey proposes (a) a new architecture taxonomy that replaces the historical soft-vs-hard-parameter-sharing split with **encoder-focused vs decoder-focused**, distinguishing where in the network task interactions happen (Sec. 2.1.4, p. 3-4); (b) a unified optimization-strategy taxonomy that catalogues uncertainty weighting, GradNorm, DWA, DTP, MGDA and gradient-sign dropout; and (c) a *clean re-implementation* of every major architecture and optimization method on NYUD-v2 (semseg + depth) and PASCAL-Context (semseg + parsing + saliency + normals + edges), so the methods can be ranked head-to-head (Sec. 4, p. 11-16).

## Taxonomy / What the Survey Covers
**Architecture taxonomy (Sec. 2, Fig. 1 p. 2):**
- **Encoder-focused** (information shared during feature extraction, independent task-specific heads):
  - MTL baseline (shared encoder, ASPP-style task heads)
  - Cross-Stitch Networks (Misra et al. 2016): linear combination of activations across single-task networks
  - Sluice Networks (extension allowing skip-connection and subspace sharing)
  - NDDR-CNN (1x1 conv-based feature fusion instead of cross-stitch's linear combination)
  - MTAN (single shared backbone + task-specific attention modules)
  - Branched MTL networks: FAFS, Vandenhende et al. 2020, BMTAS, LTB (NAS-style search over branch points)
- **Decoder-focused** (initial predictions made, then refined through cross-task distillation):
  - PAD-Net: spatial-attention multi-modal distillation between task-specific heads (Eq. 2, p. 6)
  - PAP-Net: pixel-affinity-based distillation (Eq. 3, p. 6)
  - JTRL: sequential / recursive prediction at increasingly higher resolutions (only 2 tasks)
  - PSD: separates inter- and intra-task patterns via pixel affinities
  - MTI-Net: multi-scale multi-modal distillation over HRNet-18 backbone
- **Other**: ASTMT (adversarial gradient training to remove task-specific signal from shared layers)

**Optimization taxonomy (Sec. 3, Tab. 1 p. 10):** Uncertainty weighting (Kendall et al., homoscedastic), GradNorm (balance loss magnitudes and learning pace), DWA (loss-ratio-based weighting), DTP (Dynamic Task Prioritization, push difficult tasks), MGDA (multi-objective optimization to a Pareto front), gradient sign dropout (mask conflicting gradient signs), adversarial / modulation / heuristic schemes (Sec. 3.2, p. 10).

**Evaluation criterion**: `Delta_MTL` (Eq. 10, p. 11), the average per-task relative drop vs the single-task baseline, sign-flipped for "lower-is-better" metrics. Single-task baselines themselves are grid-searched over batch size and learning rate so the comparison is fair.

## Main Innovation
The taxonomy itself is the contribution. Two specific lessons that matter for our work:
1. **Encoder-vs-decoder split is the right axis**, because encoder-focused designs share *intermediate* features and decoder-focused designs share *near-output* features that are already disentangled by task. The latter is more parameter-efficient because the cross-task interactions happen on per-task feature maps with smaller channel counts (Sec. 4.3.2, p. 15).
2. **Apples-to-apples re-implementation forces honest numbers.** Several optimization methods that look great in their original papers (uncertainty weighting, MGDA, GradNorm in some settings) underperform fixed grid-searched loss weights when single-task baselines are themselves carefully tuned (Sec. 4.4, p. 16).

## Headline Numbers Reported in the Survey

**NYUD-v2 (Tab. 5a, p. 13)** semseg IoU + depth RMSE; single-task ResNet-50 baseline is 43.9 / 0.585:

| Family | Model | FLOPs (G) | Params (M) | Seg IoU | Depth RMSE | `Delta_MTL` |
|---|---|---|---|---|---|---|
| ST | ResNet-50 | 192 | 80 | 43.9 | 0.585 | 0.00 |
| Encoder | MTL Baseline | 133 | 56 | 44.4 | 0.587 | +0.41 |
| Encoder | MTAN | 197 | 72 | 45.0 | 0.584 | +1.32 |
| Encoder | Cross-Stitch | 192 | 80 | 44.2 | 0.570 | +1.61 |
| Encoder | NDDR-CNN | 207 | 102 | 44.2 | 0.573 | +1.38 |
| Decoder | JTRL | 660 | 295 | 46.4 | 0.501 | +10.02 |
| Decoder | PAP-Net | 4800 | 52 | 50.4 | 0.530 | +12.10 |
| Decoder | PAD-Net (single-scale) | 256 | 52 | 50.2 | 0.582 | +7.43 |
| Decoder | MTI-Net (HRNet-18) | 16 | 27 | 38.6 | 0.593 | +8.95 |

**PASCAL-Context (Tab. 5b, p. 13)** 5-task dictionary; single-task ResNet-18 baseline is 66.2 / 59.9 / 13.9 / 66.3 / 68.8:
- MTL baseline (ResNet-18): -2.86%
- MTAN: -2.39%
- Cross-Stitch: +0.60%
- NDDR-CNN: +0.39%
- PAD-Net (ResNet-18): **-5.62%** (cannot handle large task dictionary; lacks skip connections in this re-implementation)
- MTI-Net (HRNet-18): **+1.13%** (only model that breaks the ST baseline on this dataset)

**Optimization comparison, MTL baseline with ResNet-50 backbone (Tab. 5c, p. 13):** on NYUD-v2, GradNorm wins (+1.45%), uncertainty (-0.23%) and DWA (-0.28%) underperform fixed grid-searched weights (+0.41%). On PASCAL (Tab. 5d), no automated method beats fixed loss weights; MGDA collapses to -6.81% because it masks out the edge-detection loss with a small weight.

## Multi-Task Coupling as a Phenomenon: What the Survey Concludes
This survey is the cleanest source on the question. Its conclusions on cross-task gradient flow:
- **Task dictionary determines whether coupling helps or hurts.** A small set of well-correlated tasks (semseg + depth on NYUD-v2) gets a large positive `Delta_MTL` from almost every architecture. A larger, more diverse dictionary (5 tasks on PASCAL with semseg, parsing, normals, saliency, edges) drives most encoder-focused models *negative* (Sec. 4.2, p. 12).
- **Decoder-focused beats encoder-focused on dense prediction.** Even the simplest decoder-focused architecture (PAD-Net at +7.43% with 52 M params) beats every encoder-focused architecture on NYUD-v2 by a wide margin. The reason given (Sec. 4.3.2, p. 15) is that task features near the output are already disentangled, so the cross-task signal is cleaner.
- **More gradient cooperation is not always better.** Surprisingly, MGDA (which only updates along directions that are *non-conflicting* across all task gradients) under-performs simple grid-searched loss weights (Tab. 5c, NYUD: +0.02% vs +0.41%). The authors hypothesize that *some* gradient competition between tasks helps escape local minima (Sec. 4.4, p. 16).
- **Negative transfer is real and well-documented.** Almost every encoder-focused model on PASCAL has negative `Delta_MTL`. The mechanism is not gradient-direction conflict per se; it is that diverse-task dictionaries force the shared encoder to allocate capacity poorly.
- **Loss-magnitude imbalance matters more than gradient direction.** When the edge-detection loss is 100x smaller than the semantic-segmentation loss (Sec. 4.4 PASCAL discussion), MGDA assigns it negligible weight and the task collapses; fixed grid-searched weights handle this better. The authors recommend balancing magnitudes first, gradient signs second.

## Relevance to Our Project
- **Decoder-focused over encoder-focused** is the takeaway most directly usable for our hypothetical stereo + detection / segmentation chassis. If we ever wire YOLO26-detection and stereo into one network, the survey says: keep two task-specific heads and add cross-task distillation between them (PAD-Net or MTI-Net style), rather than fusing in the encoder.
- **The HRNet-18 + MTI-Net combination at +8.95% on NYUD-v2 with 27 M params and 16 G FLOPs** is the most compute-efficient working MTL setup in the table. Roughly 10x our edge envelope but at the right shape (multi-scale backbone + per-scale distillation). MTI-Net's distillation-at-multiple-scales pattern could potentially be subsetted to 1/16 + 1/8 + 1/4 (matching StereoLite's pyramid) on a sub-million-param chassis.
- **Loss balancing wisdom directly transfers.** Our `stack_d1` production loss (multi-scale L1 + grad + threshold-stack hinge + KITTI D1-relative hinge, all hand-weighted) is consistent with the survey's recommendation: fixed grid-searched weights beat automatic schemes. We should not waste cycles on uncertainty weighting or GradNorm for StereoLite.
- **Task-dictionary observation is a planning input.** If we add semseg or detection to StereoLite, the *closely-related* pair (stereo + segmentation, like S3M-Net / TiCoSS) is more likely to net positive than a diverse set (stereo + detection + segmentation + normals).
- **Drone application is implicit.** None of the surveyed architectures fit a Jetson Orin Nano envelope. The survey predates the modern edge-MTL literature (YOLOP, HybridNets, Sparse U-PDP). The closest practical chassis would be MTI-Net's HRNet-18 base, distilled aggressively.

## Limitations
- **Pre-transformer survey.** Published 2022 but covers literature up to 2021 (no InvPT, no TaskPrompter, no transformer-based MTL). For the lineage HKUST is building, you need a follow-up paper.
- **Edge / latency is absent.** FLOPs and parameter counts are reported but not actual ms latency on any hardware. The survey is silent on real-time deployment.
- **Task dictionary is narrow.** Only NYUD-v2 (2-task) and PASCAL-Context (5-task). No driving, no aerial, no robot manipulation, no large-scale benchmark.
- **No knowledge distillation or self-supervised MTL coverage.** Just supervised MTL. Modern MTL has moved heavily into KD (LiteAnyStereo-style 3-stage) and self-supervised cross-task signals (e.g. depth predicting normals for free).
- **Re-implementation choices favour the authors' own work** (MTI-Net is from this group). PAD-Net's PASCAL implementation lacks skip connections (Sec. 4.3.2, p. 15) which the survey acknowledges as a cause of the -5.62% score. So the PAD-Net number is plausibly worse than the original would achieve. The survey says this honestly but it is a confound worth knowing.
