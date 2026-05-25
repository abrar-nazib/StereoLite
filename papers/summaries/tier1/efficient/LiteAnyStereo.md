# Lite Any Stereo: Efficient Zero-Shot Stereo Matching

**Authors:** Junpeng Jing, Weixun Luo, Ye Mao, Krystian Mikolajczyk (Imperial College London)
**Venue:** arXiv 2025 (accepted CVPR 2026)
**Priority:** 9/10 — **most directly relevant paper to our edge model goals**
**arXiv:** https://arxiv.org/abs/2511.16555
**WebPage:** https://tomtomtommi.github.io/LiteAnyStereo/
**Code:** https://github.com/tomtomtommi/LiteAnyStereo (cloned to `.claude/skills/stereo-vision-expert/reference_impls/lite_any_stereo/` — `liteanystereo.py`, `aggregation.py`, `fnet.py`, `submodule.py`, `profile_speed.py`)

## VERIFIED FACTS (instantiated from official code 2026-05-02)

| Attribute | Value | Source |
|---|---|---|
| **Trainable parameters** | **7.604 M** | `model = LiteAnyStereo(); sum(p.numel() for p in model.parameters() if p.requires_grad)` — paper does NOT state |
| `cost_agg_2d` (ConvNeXt 2D blocks) | **5.731 M** | per-module sum — dominant component, 75% of total |
| `fnet` (MobileNetV2 + multi-scale heads) | **1.846 M** | per-module sum |
| All other (refines + stems) | ~25 k | per-module sum |
| **MACs at KITTI 1242×375** | **33 G** | Paper Tab. 3 + Tab. 5 |
| **Latency at 4K on GTX 1080 fp32** | **21 ms** | Paper Fig. 1 caption |
| **Latency at 384×640 on RTX 3050 fp16** (verified locally, with cuDNN warmup) | **70 ms mean / 73 ms median / 87 ms p95** | Tested 2026-05-02 |
| **Latency at 480×768 on RTX 3050 fp16** | **57 ms mean / 56 ms median** | Tested 2026-05-02 |
| **Peak GPU memory at 384×640 fp16** | **0.69 GB** | Tested 2026-05-02 |
| **Latency on Jetson / Orin / mobile NPU** | **NOT MEASURED** | Not in paper, not in code; **unknown** |

**Real-time on edge — honest caveats:**
- Paper's only edge-relevant claim is "21 ms at 4K on GTX 1080" (a desktop GPU, ~6.4 TFLOPS fp16, 320 W).
- RTX 3050 (~9 TFLOPS fp16) is roughly equivalent to Jetson AGX Orin in compute, but **5× faster than Jetson Orin Nano** (~6 TFLOPS fp16 with much narrower memory bus).
- Naive extrapolation: 70 ms RTX 3050 fp16 → likely **200-350 ms on Jetson Orin Nano fp16 raw PyTorch**. NOT real-time on Orin Nano.
- TensorRT INT8 typically gives 2-4× speedup on Jetson; **borderline real-time at 50-90 ms** on Orin Nano with full optimization, untested.
- The 5.73 M `cost_agg_2d` ConvNeXt block is the obvious compression target — halving its width should land closer to 30-50 ms on Orin Nano with TensorRT.

---

## Core Problem & Motivation

LiteAnyStereo attacks a widely-held assumption: **"efficient stereo models cannot generalize zero-shot."** Recent foundation-model approaches (FoundationStereo, DEFOM-Stereo, Stereo Anywhere) achieve remarkable cross-domain generalization — but at enormous computational cost:

- **FoundationStereo:** 12,824G MACs (impossible on edge)
- **Selective-IGEV:** 3,619G MACs (far from real-time)

Meanwhile, efficient methods (MobileStereoNet, BANet, LightStereo) are fast but generalize poorly. The community had accepted this trade-off as fundamental.

**LiteAnyStereo's central claim:** This trade-off is NOT fundamental — it arises from **limited training data and suboptimal training procedures**, not from model capacity. The solution requires two things prior efficient models lacked:

1. **A cost aggregation design** that can leverage large-scale training without overfitting
2. **A training strategy** that bridges the sim-to-real gap without needing a heavy foundation model backbone

---

## Architecture

### Feature Extraction
- **MobileNetV2** (ImageNet pre-trained), shared weights for left and right images
- Multi-scale features at {1/4, 1/8, 1/16, 1/32} resolution
- All scales upsampled to 1/4 via residual upsampling blocks
- **Same backbone as BANet-2D** — the differences are in aggregation and training, making these the controlled variables

### Cost Volume Construction
Standard normalized correlation at 1/4 resolution:
$$C(d, h, w) = \frac{1}{N_c} \langle F^{1/4}_L(h, w), F^{1/4}_R(h, w-d) \rangle$$

- **$C(d, h, w)$** = matching cost at disparity hypothesis $d$, row $h$, column $w$
- **$N_c$** = number of feature channels (normalization)
- **$F^{1/4}_L, F^{1/4}_R$** = left/right feature maps at 1/4 resolution
- **$d$** = integer disparity hypothesis in $[0, D_{max}/4 - 1]$, $D_{max} = 192$
- Volume shape: $[D_{max}/4, H/4, W/4]$

### Hybrid 3D-2D Cost Aggregation — THE Core Innovation

Rather than pure 2D (BANet, LightStereo) or pure 3D (IGEV-Stereo), LiteAnyStereo uses a **serial 3D-then-2D design**:

$$C_{agg} = G_{2D}(G_{3D}(C))$$

- **$G_{3D}$** = **tiny** 3D convolutional block using 3×3×3 kernels
- **$G_{2D}$** = larger 2D convolutional block using **ConvNeXt layers**
- **Critically: the 3D component is only 4.8% of total MACs**

**Why serial 3D→2D works:**
- **3D convolutions are essential** for modeling the joint (disparity, spatial) structure — pure 2D methods lose cross-disparity context
- **But 3D operations are expensive**, so use minimally as a "context initialization" step
- **Most computation is delegated to efficient 2D ConvNeXt** layers for spatial refinement

This is a **principled separation of concerns**: 3D for joint structure, 2D for spatial pattern refinement. The ablation shows this strictly beats bilateral parallel aggregation (BANet's approach) at the same compute budget for zero-shot generalization.

### Disparity Estimation

$$d = \sum_{d=0}^{D_{max}/4 - 1} d \cdot \sigma(C_{agg}(d))$$

Standard soft-argmax regression at 1/4 resolution, followed by **convex upsampling** to recover full resolution.

### Total Computation: 33G MACs

No monocular prior, no ViT, no cross-attention, no iterative refinement. Just efficient correlation + hybrid aggregation + convex upsampling.

---

## The Three-Stage Training Pipeline (Second Key Innovation)

**This is arguably more important than the architecture.** The paper validates that the training recipe also improves other efficient backbones (LightStereo-M, BANet-2D).

### Stage 1: Synthetic Supervised Pre-training
- **1.8M synthetic pairs** from diverse sources
- Standard smooth L1 loss on disparity
- Learns basic stereo matching from synthetic diversity
- 150K training iterations

### Stage 2: Self-Distillation with Perturbation
- **Teacher:** frozen copy of Stage 1 model, sees **clean** image pairs
- **Student:** same model being trained, sees **perturbed** image pairs (color jitter, blur, etc.)
- **Feature alignment loss:**

$$\mathcal{L}_{feat} = 1 - \frac{1}{HW} \sum_i \cos(F_i, F'_i)$$

- **$F_i$** = feature vector from teacher at position $i$ (clean input)
- **$F'_i$** = feature vector from student at position $i$ (perturbed input)
- **$\cos(\cdot, \cdot)$** = cosine similarity
- Forces domain-invariant representations — student must produce features that match teacher even under perturbation
- **No external model needed** for this stage
- 50K iterations

**Critical finding:** Fixed teacher weights outperform EMA and hard-copy updates. The teacher must remain **anchored to Stage 1 knowledge**, not drift with the student.

### Stage 3: Knowledge Distillation from FoundationStereo
- **0.5M real-world stereo pairs** (unlabeled)
- **Teacher:** frozen FoundationStereo (expensive, runs once to generate pseudo-labels)
- **Student:** LiteAnyStereo (cheap, trained to mimic teacher's disparity output)
- Pseudo-labels transfer FoundationStereo's zero-shot generalization capability into the lightweight model
- 100K iterations

**Remarkable result:** The student **surpasses the teacher** on DrivingStereo weather subsets (8.74% vs 10.71% D1) — evidence of genuine generalization, not just mimicry.

---

## Benchmark Results (Zero-Shot, Million-Scale Training)

| Method | KITTI 2012 D1 | KITTI 2015 D1 | ETH3D Bad 1.0 | Middlebury Bad 2.0 | MACs (G) |
|--------|---------------|---------------|---------------|-------------------|----------|
| LightStereo-M* (same training) | 4.10 | 4.97 | 5.33 | 10.85 | 33 |
| BANet-2D* (same training) | 3.90 | 4.71 | 5.92 | 10.05 | 36 |
| StereoAnything-L† | 4.00 | 4.81 | 3.81 | 9.82 | 84 |
| **Selective-IGEV** (accurate) | 3.20 | 4.50 | 3.40 | 7.50 | **3,619** |
| **FoundationStereo** (SOTA) | 2.51 | 2.83 | 0.49 | 1.12 | **12,824** |
| **LiteAnyStereo** | **3.04** | **3.87** | **3.53** | **7.51** | **33** |

**Headline results:**

- **Beats Selective-IGEV on KITTI 2012, KITTI 2015, and Middlebury** with **109× less compute** (33G vs 3,619G MACs)
- **Matches Selective-IGEV exactly on Middlebury** (7.51 vs 7.50)
- **Outperforms BANet-2D on ALL four benchmarks** at similar MAC budget

**DrivingStereo Weather (student beats teacher):**
| Method | Weather D1-all |
|--------|---------------|
| FoundationStereo | 10.71 |
| **LiteAnyStereo** | **8.74** |

The student generalizes **better than the teacher** on extreme out-of-distribution weather conditions.

### Inference Time

| Hardware | Latency | Memory (2K res) |
|----------|---------|----------------|
| A100 | **17ms** | — |
| RTX 4090 | 19ms | — |
| GTX 1080 | **21ms** at 4K | 2.5 GB |
| A5000 | 23ms | — |

**21 ms on a GTX 1080 at 4K resolution** — this is remarkable efficiency.

---

## Ablation Highlights

### Cost aggregation architecture (KITTI 2012 D1%)

| Design | KITTI-12 | KITTI-15 |
|--------|----------|----------|
| 2D only | 5.02 | 5.01 |
| Bilateral parallel (BANet-style) | 5.10 | — |
| 2D → 3D | 4.73 | — |
| **3D → 2D (chosen)** | **4.78** | **4.64** |
| Interleaved | 4.61 | — |

**Key finding:** Parallel bilateral aggregation (BANet's approach) actually **underperforms** the serial 3D→2D design. Interleaved is marginally better but more complex; 3D→2D is selected for simplicity.

### 3D proportion (critical ablation)

The paper tested different 3D/total MAC ratios. **The smallest tested (4.8%) is optimal** — more 3D hurts. This counterintuitive finding validates the "3D as context initialization only" philosophy.

### 2D layer type

| 2D Block | KITTI-12 |
|----------|----------|
| MobileNetV2 | 4.78 |
| **ConvNeXt** | **4.38** |

ConvNeXt layers outperform MobileNetV2 blocks by ~0.4% on KITTI-12 — better expressive capacity while remaining hardware-friendly.

### Training strategy progression

| Stages | KITTI-12 | KITTI-15 | ETH3D | Middlebury |
|--------|----------|----------|-------|-----------|
| Stage 1 only | 4.05 | 4.55 | 4.43 | 8.49 |
| Stages 1+2 | 3.66 | 4.53 | 4.69 | 7.03 |
| **All 3 stages** | **3.04** | **3.87** | **3.53** | **7.51** |

Each stage contributes measurably. Stage 3 (KD from FoundationStereo) provides the largest across-the-board improvement.

---

## Strengths & Limitations

**Strengths:**
- **First efficient model to match/exceed accurate non-prior methods' zero-shot generalization** at 1% of their compute
- **Training pipeline is architecture-agnostic** — also improves LightStereo-M and BANet-2D
- **Real-time on GPUs** (17-23ms) and deployable on mobile
- **Student surpasses teacher on OOD weather** — proves genuine generalization, not mimicry
- **No monocular prior needed** — avoids ViT dependency
- **ConvNeXt is quantization-friendly**
- **Explicitly disproves** the "efficient = no zero-shot" assumption with careful ablation
- **Minimal 3D (4.8%) is provably optimal** — a reusable architectural finding

**Limitations:**
- **Still behind prior-based methods on indoor benchmarks** (FoundationStereo: 0.49 vs 3.53 on ETH3D Bad 1.0)
- **Middlebury performance is weakest** relative to SOTA (7.51 vs 1.12 for FoundationStereo)
- **No edge-hardware latency reported** in the paper. Only "21 ms at 4K on GTX 1080" (desktop). Local RTX 3050 fp16 measurement: 70 ms at 384×640 (verified 2026-05-02). Likely 200-350 ms on Jetson Orin Nano without TensorRT INT8.
- **Paper does not state parameter count.** Verified from the public code: **7.604 M trainable**, with 5.73 M concentrated in the `cost_agg_2d` ConvNeXt blocks.
- **Requires FoundationStereo for Stage 3 training** — creates a dependency on teacher availability and inference cost during training
- **No iterative refinement** — single-pass, lacks the inductive bias that Pip-Stereo demonstrated is important
- **Interleaved aggregation design is marginally better** — the 3D→2D choice may not be globally optimal

---

## Relevance to Our Edge Model

**LiteAnyStereo is arguably the most directly relevant paper to our edge model goals.** Specific takeaways:

### Directly Adoptable

1. **Adopt the three-stage training pipeline wholesale** — Stage 1 on large synthetic, Stage 2 self-distillation with perturbation, Stage 3 KD from DEFOM-Stereo or FoundationStereo on unlabeled real data. This is how we bridge sim-to-real without a heavy backbone.

2. **Hybrid 3D→2D aggregation is better than bilateral for generalization** — for within-domain benchmarks, BANet's bilateral approach is competitive, but for zero-shot (our primary goal), serial 3D→2D wins.

3. **Keep the 3D block tiny (~4.8% of MACs)** — the ablation is unambiguous. Use 3D only as cross-disparity context initialization.

4. **ConvNeXt over MobileNetV2 blocks in aggregation** — better expressive capacity, quantization-friendly.

5. **33G MACs target** — LiteAnyStereo proves this compute budget is sufficient for zero-shot generalization.

6. **FoundationStereo / DEFOM-Stereo as Stage 3 teacher** — we already have both PDFs. Either can serve as the pseudo-label generator for training our edge model.

### Proposed Integration with Our Architecture

Combining LiteAnyStereo's training recipe with other Tier 1 findings:

```
ARCHITECTURE (from all Tier 1 efficient papers):
  [MobileNetV4/RepViT-NAS Backbone]   ← Pip-Stereo
         ↓
  [Local Correlation + 2D-1D Search]  ← CREStereo
         ↓
  [Serial 3D→2D Hybrid Aggregation]   ← LiteAnyStereo
  (tiny 3D block + ConvNeXt 2D)
         ↓
  [Bilateral Split inside 2D stage]   ← BANet (optional)
         ↓
  [Small GEV from 3D block]           ← IGEV-Stereo
         ↓
  [Warm Start + SRU × 2-4 iters]      ← Selective-Stereo + Pip-Stereo
         ↓
  [CUBG Bilateral Grid Upsampling]    ← BGNet
         ↓
  Final disparity

TRAINING (from LiteAnyStereo):
  Stage 1: Synthetic supervised (1.8M+ pairs)
  Stage 2: Self-distillation with perturbation (teacher = frozen Stage 1)
  Stage 3: KD from DEFOM-Stereo / FoundationStereo on unlabeled real data
          + optional DTP structural pruning
```

**Expected performance:** At 33-50G MACs, this should achieve:
- KITTI D1-all: 3-4% (zero-shot) / ~1.5% (fine-tuned)
- <33ms on Jetson Orin Nano at 320×640
- Strong cross-domain generalization via Stage 3 KD

---

## Connections to Other Papers

| Paper | Relationship |
|-------|-------------|
| **BANet** | Direct competitor — same backbone, LiteAnyStereo beats it via different aggregation and training |
| **LightStereo** | Direct competitor — similar MAC budget, LiteAnyStereo better generalization |
| **FoundationStereo** | Stage 3 teacher — provides pseudo-labels that transfer zero-shot capability |
| **DEFOM-Stereo** | Alternative Stage 3 teacher option |
| **Selective-IGEV** | Compared as accurate baseline — LiteAnyStereo beats it at 109× less compute |
| **ConvNeXt** | Provides the 2D block architecture |
| **MobileNetV2** | Backbone choice (same as BANet for controlled comparison) |
| **Pip-Stereo** | Complementary — Pip-Stereo shows iterative methods generalize better, but LiteAnyStereo proves non-iterative can work with right training |
