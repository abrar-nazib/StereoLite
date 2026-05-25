# MTI-Net: Multi-Scale Task Interaction Networks for Multi-Task Learning

**Authors:** Simon Vandenhende, Stamatios Georgoulis, Luc Van Gool (KU Leuven/ESAT-PSI, ETH Zurich/CVL)
**Venue:** ECCV 2020 (arXiv:2001.06902v5, July 2020)
**Tier:** 3 (foundational multi-task dense-prediction CNN; explicitly attacks PAD-Net / PAP-Net's single-scale distillation limitation)

---

## Core Idea
PAD-Net and PAP-Net both distill cross-task information at a **single scale** (typically 1/4 or 1/8 of input), implicitly assuming task interactions are scale-invariant. MTI-Net measures this assumption empirically (Fig. 2b, p. 6) by computing pixel-affinity correspondence between task pairs at different kernel dilations and shows that **task affinity changes with receptive-field size**: two tasks that share local patterns may not share global ones, and vice versa. MTI-Net therefore distills cross-task information at *every* scale of a multi-scale backbone (HRNet or FPN), with explicit feature propagation from lower scales (global context) to higher scales (local detail) before final aggregation.

## Architecture
- **Multi-scale backbone** (Sect. 2.3, p. 7): off-the-shelf HRNet-18/HRNet48-V2 or FPN-ResNet-18/50. Yields features at four scales: 1/4, 1/8, 1/16, 1/32.
- **Initial task predictions at every scale** (Sect. 2.3, p. 7): each scale has its own set of K task-specific heads (basic ResNet blocks), producing per-task feature maps F_{k,s}^i for task k at scale s. This is the *front-end* and gives deep supervision at every scale.
- **Multi-scale multi-modal distillation** (Eq. 1, p. 7): at each scale s independently, refine each task feature F_{k,s}^i by adding a spatial-attention-gated sum of other tasks' features:
  F_{k,s}^o = F_{k,s}^i + sum_{l != k} sigma(W_{k,l,s} F_{l,s}^i) ⊙ (W'_{k,l,s} F_{l,s}^i)
  where sigma is a sigmoid spatial attention mask. Same operator as PAD-Net Module C, but applied independently at four scales.
- **Feature Propagation Module (FPM)** (Sect. 2.4, Fig. 3, p. 8): higher-resolution scales (1/4) have limited receptive field, so their initial predictions are weak. FPM passes distilled task features from a lower scale (e.g. 1/16) up to the next higher scale (1/8) before that scale's head produces its initial predictions. FPM internals: (i) feature harmonisation block — concat all N task features, apply a learned non-linear function f, softmax-along-task-axis to produce per-task attention mask, recombine; (ii) squeeze-and-excitation refinement to gate the shared representation per task; (iii) add as residual to original task features.
- **Feature aggregation unit** (Sect. 2.5, p. 9): distilled task features at all four scales are upsampled to 1/4 and concatenated; final per-task heads decode the aggregated multi-scale features into the final outputs.
- **Auxiliary tasks** (Sect. 2.5, p. 9; Tab. 1 p. 9): can be added at the front-end of the network only (PASCAL: surface normals + saliency as aux to seg/parts/edges).

## Main Innovation
**Three pieces compounded:**
1. **Multi-scale multi-modal distillation** — apply PAD-Net's attention-gated cross-task message passing not once at a fixed scale but four times in parallel at 1/4, 1/8, 1/16, 1/32. Justified empirically (Fig. 2b p. 6) by the task-affinity-vs-dilation measurement.
2. **Feature Propagation Module (FPM)** — explicit cross-scale information path from coarse-to-fine within the *front-end*, so higher-resolution scales receive global context before they make their initial predictions. This is the load-bearing piece (Tab. 2a p. 11).
3. **Multi-scale aggregation** — final per-task heads see distilled features from every scale, not just one, providing both local detail (1/4) and global structure (1/32).

This stack is *complementary* to PAD-Net; if you replace MTI-Net's FPM and aggregation with PAD-Net's single-scale distillation, you get PAD-Net's headline numbers and not MTI-Net's.

## Key Benchmark Numbers

**NYUD-v2 ablation (Tab. 2a, p. 11), HRNet-18 backbone, segmentation + depth main / edges + normals auxiliary:**

| Method | Seg mIoU | Depth rmse | Multi-task delta (%) |
|---|---|---|---|
| Single task | 33.18 | 0.667 | +0.00 |
| Naive MTL (shared encoder + heads) | 32.09 | 0.668 | **-1.71** (negative transfer) |
| PAD-Net (single-scale distillation) | 32.80 | 0.660 | -0.02 |
| MTI-Net w/o FPM | 34.38 | 0.640 | +3.85 |
| **MTI-Net w/ FPM** | **35.12** | **0.620** | **+6.40** |
| MTI-Net w/ FPM + edges aux | 36.22 | 0.600 | +9.57 |
| MTI-Net w/ FPM + edges + normals aux | **37.49** | 0.607 | **+10.91** |

**PASCAL (Tab. 5, p. 12), R18-FPN backbone, 5 tasks (seg/parts/saliency/edges/normals):**
- MTI-Net delta_m ↑ +3.84% over single-task; PAD-Net delta_m = -3.08% (negative transfer); ASTMT (R26-DLv3+) delta_m = -3.42%.
- MTI-Net is **the first method in the lineage to report consistent positive multi-task delta** across this large a task set.

**NYUD-v2 vs prior state-of-the-art (Tab. 6, p. 13), HRNet48-V2 backbone:**
- Depth rmse 0.529 (best in table), rel 0.138, delta_1 0.830, delta_2 0.969, delta_3 0.993.
- Seg mIoU 49.0, mean-acc 62.9, pixel-acc 75.3 (slightly below PAP-Net mIoU 50.4 but with much fewer FLOPs).

**Computational resource analysis (Tab. 7, p. 13), HRNet-18 on NYUD-v2:**
- Single-task baseline (2 tasks): 8.0 M params, 22.0 G FLOPS.
- Naive MTL: 50% fewer params, 45% fewer FLOPS, -1.71% multi-task delta.
- PAD-Net: 15% fewer params, **+204% FLOPS** (because distillation is at 1/4), -0.02% delta.
- **MTI-Net: +57% params, -13% FLOPS, +6.40% delta.**

MTI-Net uses *more* params than the single-task baseline (because of per-scale heads + FPM + aggregator) but **fewer FLOPs** than PAD-Net by 5x because most distillation happens at 1/32 and 1/16 scales where spatial dims are small.

## Multi-Task Coupling: Load-Bearing or Decorative?

**Load-bearing — and decisively so, with the cleanest evidence of any paper in this lineage.**

The Tab. 2a NYUD-v2 ablation is the smoking gun:

- **Naive MTL: delta = -1.71%** (worse than single-task — classic negative transfer).
- **PAD-Net (single-scale distillation): delta = -0.02%** (single-scale distillation just barely undoes the negative transfer but provides essentially no benefit).
- **MTI-Net w/o FPM: delta = +3.85%** (multi-scale distillation alone is worth +3.87% over PAD-Net).
- **MTI-Net w/ FPM: delta = +6.40%** (FPM adds another +2.55%).
- **MTI-Net w/ FPM + 2 auxiliaries: delta = +10.91%** (auxiliary tasks add another +4.51%).

So the per-component contributions on NYUD-v2 are: multi-scale distillation +3.87%, FPM +2.55%, auxiliary tasks +4.51%. Each piece is doing real work.

**Ablation on number of scales (Tab. 3, p. 12):** going from 1 scale (= PAD-Net) to 2 scales (1/4 + 1/8) gives +3.80%; adding 1/16 gives +5.53%; adding 1/32 gives +6.40%. Monotone improvement with scale count, confirming the central thesis.

**FPM directionality (Tab. 4, p. 12):** Adding initial-prediction heads at successive scales 1/32 -> 1/16 -> 1/8 -> 1/4 improves multi-task delta monotonically from -1.87% to +6.40%, validating that lower-scale predictions are *informing* higher-scale predictions via FPM.

Verdict: **every architectural piece earns its keep with measurable deltas in the 2-5% range, and the total stack converts negative transfer into +10.91% positive transfer on the same backbone.** This is the strongest case for "multi-task coupling is load-bearing" anywhere in this lineage.

## Relevance to Our Project
- **The multi-scale-task-affinity insight is directly applicable to stereo.** Disparity is a multi-scale signal by construction — coarse disparity at 1/16 plus fine refinement at 1/4 — and so is semantic segmentation or detection if we add one. The MTI-Net argument that "task affinity changes with receptive field" means our stereo and any auxiliary task should fuse at *every* scale we use, not just once.
- **StereoLite's tile-propagation is already multi-scale.** Our current pipeline runs TileRefine at 1/16, 1/8, 1/4 with TileUpsample between scales. MTI-Net's FPM is structurally the same pattern: lower scale -> upper scale information flow before each scale produces its predictions. If we add a seg head, applying MTI-Net-style multi-scale distillation onto our existing chassis is structurally natural — we'd add per-scale 1x1 conv "harmonisation + attention" blocks between disparity features and seg features at 1/16, 1/8, 1/4.
- **Param/FLOPs profile is the right shape for edge.** Tab. 7 (p. 13): MTI-Net uses +57% params but -13% FLOPs vs single-task. The flat-FLOPs property is because most distillation happens at low spatial resolution (1/16, 1/32). For a YOLO26 + stereo joint model at 384x640, four-scale distillation would cost dozens of GFLOPs total — could plausibly fit our 60 ms budget.
- **The "single-scale distillation = PAD-Net = no benefit" finding kills any naive seg-head bolt-on.** Tab. 2a is unambiguous: a single fusion module between two task branches gives essentially nothing on NYUD-v2 (-0.02% delta). Multi-scale fusion is the minimum bar for a meaningful joint network. If we ever add a seg head without going multi-scale, we should expect zero benefit.
- **The 2-aux-task gain (+4.51%) is the cheapest port.** Even without doing full MTI-Net, just adding a contour-detection head (free from disparity edges) or a saliency head (free from a SAM2 pseudo-label) as auxiliary supervision at the front-end gives a measurable boost.

## Limitations
- **HRNet-18 backbone is the smallest evaluated.** No experiments on MobileNet- or GhostNet-scale backbones. The 8 M HRNet-18 single-task baseline is 9x our edge envelope.
- **Per-scale task heads multiply linearly with K.** Each of K tasks has its own head at each of 4 scales — 4K heads total. For K=2 that's manageable; for K=5 (PASCAL setting) the head bank starts to dominate the param count (+57% over single-task).
- **No cross-domain evaluation.** All NYUD-v2 results train and test on NYUD-v2. PASCAL is within-dataset. The cross-domain generalisation question (which is what kills our chassis on MB14) is not asked.
- **Auxiliary-task selection is empirical.** Tab. 2a shows edges + normals together gives best NYUD-v2 mIoU but adding normals alone slightly *hurts* segmentation. No principle for which auxiliary task to add for which main task — still hand-tuned.
- **The "task affinity vs dilation" plot (Fig. 2b) is suggestive, not definitive.** Three task pairs on one dataset; affinity curves all monotonically decreasing with dilation. The strong claim "task interactions are scale-dependent" is empirically supported but not theoretically grounded.
- **FPM upsamples by concatenation + non-linear function; no learned upsampling.** This is fine on indoor scenes but may produce blocky boundaries on outdoor scenes (Cityscapes-style). Not evaluated.
