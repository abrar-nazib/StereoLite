# Separable Convolutions for Optimizing 3D Stereo Networks

**Authors:** Rafia Rahim, Faranak Shamsafar, Andreas Zell (University of Tübingen)
**Venue:** ICIP 2021
**Priority:** 9/10 — foundational paper for 3D convolution efficiency in stereo
**arXiv:** https://arxiv.org/abs/2108.10216

---

## Core Problem & Motivation

Deep 3D stereo networks outperform 2D networks and classical methods, but the accuracy improvement comes at enormous computational cost. The authors provide a **concrete empirical measurement** that makes the problem stark:

**3D convolutions consume ~94% of total operations** in state-of-the-art stereo networks (GANet11, GANetdeep, PSMNet). The 3D cost volume regularization is the overwhelming bottleneck, not feature extraction or disparity regression.

Specifically, in PSMNet at 540×960 resolution:
- **Total operations: 256 GMACs**
- **3D cost aggregation: 241.6 GMACs (94%)**
- Feature extraction: 14.4 GMACs
- Disparity regression + overhead: 0.1 GMACs

**The goal:** Reduce this 94% bottleneck via depthwise-separable 3D convolutions, without losing accuracy.

---

## The Key Concept: Separable Convolutions for 3D

A standard **3D convolution** uses a kernel of shape $k_h \times k_w \times k_d \times c_i$ where $k_h \times k_w$ is spatial and $k_d$ is the disparity dimension. Typical $k = 3$, so the kernel is $3 \times 3 \times 3 \times c_i$.

**Feature-wise Separable Convolutions (FwSCs)** — the V1 analog:
Decompose the 3D convolution into **two steps**:

1. **Depthwise 3D spatial convolution:** Apply separate $k \times k \times k$ filters to each of the $c_i$ input channels independently. No channel mixing.
2. **Pointwise ($1 \times 1 \times 1$) convolution:** Mix across channels to produce $c_o$ output channels.

**MACs reduction:** For input $c_i \times d \times h \times w$ and output $c_o$ channels:

$$\text{MACs}_{\text{std}} = k^3 \cdot c_i \cdot c_o \cdot d \cdot h \cdot w$$

$$\text{MACs}_{\text{FwSC}} = (k^3 \cdot c_i + c_i \cdot c_o) \cdot d \cdot h \cdot w$$

**Reduction factor:**
$$\frac{\text{MACs}_{\text{FwSC}}}{\text{MACs}_{\text{std}}} = \frac{1}{c_o} + \frac{1}{k^3}$$

- For $k = 3$, $c_o = 64$: reduction = $\frac{1}{64} + \frac{1}{27} \approx 0.053$ → **~18.9× fewer MACs**
- The reduction scales with $c_o$: more output channels = larger relative savings

**Feature-Disparity-wise Separable Convolutions (FDwSCs)** — a more aggressive variant:
Further decompose the depthwise 3D step into a sequence of **1D convolutions** along height, width, and disparity separately. Each 1D depthwise convolution operates on one spatial axis at a time.

This is the **3D analog of ResNeXt / factorized convolutions**: $k \times 1 \times 1$, $1 \times k \times 1$, $1 \times 1 \times k$ → three sequential 1D depthwise convolutions replace one 3D depthwise convolution.

**FDwSC reduction factor:** larger than FwSC but with slightly more operations in total due to the sequential nature.

### The "Plug-and-Run" Strategy

Both FwSCs and FDwSCs are designed to be **drop-in replacements** for any 3D convolution in an existing stereo network. No retraining from scratch needed — just replace 3D convs with the separable variants and fine-tune. The paper demonstrates this plug-and-run strategy on three popular networks: GANet11, GANetdeep, and PSMNet.

---

## Detailed Computational Cost Analysis (Table 1)

The paper provides a breakdown of where the 3D convolutions dominate:

| Model | Feature Extraction MACs | 3D Cost Aggregation MACs | Total MACs | 3D % of Total |
|-------|------------------------|--------------------------|------------|--------------|
| **GANet11** | 14.4 | 241.6 | 256 | **94.4%** |
| **GANetdeep** | 14.4 | ~290 | ~304 | **95.4%** |
| **PSMNet** | 14.4 | 241.6 | 256 | **94.4%** |

This quantifies the "3D conv bottleneck" claim: in all three networks, more than 94% of operations are in 3D convolutions, making them the unique target for optimization.

---

## Benchmark Results

### Scene Flow (Table 3, EPE in pixels)

| Base Model | Variant | EPE | GMACs | Reduction |
|-----------|---------|-----|-------|-----------|
| GANet11 | Original | 0.88 | 256 | baseline |
| GANet11 | + FwSC | **0.87** | 39.5 | **6.5×** |
| GANet11 | + FDwSC | 1.06 | **36.4** | **7.0×** |
| GANetdeep | Original | 0.78 | ~304 | baseline |
| GANetdeep | + FwSC | **0.80** | 45.7 | **6.6×** |
| GANetdeep | + FDwSC | 0.95 | **42.5** | **7.2×** |
| PSMNet | Original | 1.09 | 256 | baseline |
| PSMNet | + FwSC | **0.99** | 39.4 | **6.5×** |
| PSMNet | + FDwSC | 1.16 | **36.3** | **7.0×** |

**Critical finding:** PSMNet + FwSC actually **improves** over the PSMNet baseline (0.99 vs 1.09 EPE) — replacing 3D convs with separable ones can increase accuracy while massively reducing compute. GANet11 with FwSC maintains accuracy within 0.01 EPE.

### KITTI 2015 (fine-tuned)

| Base Model | Variant | D1-all | GMACs | Reduction |
|-----------|---------|--------|-------|-----------|
| GANet11 | Original | 1.95 | 256 | baseline |
| GANet11 | + FwSC | **1.96** | 39.5 | 6.5× |
| PSMNet | Original | 2.32 | 256 | baseline |
| PSMNet | + FwSC | 2.31 | 39.4 | 6.5× |

**Matches baseline accuracy at 6.5× less compute.** The plug-and-run approach works on real benchmarks.

### Parameter Reduction

The paper also reports parameter reductions of up to **3.5×** (not just MACs). This is significant because parameter count affects memory footprint on edge devices, not just compute.

---

## FwSC vs FDwSC: Which to Use?

**FwSCs (feature-wise separable):**
- Decomposes only along the channel dimension
- **Up to 6.5-7× MAC reduction**
- **Preserves or improves accuracy** vs baseline
- Generally the better choice

**FDwSCs (feature-disparity-wise separable):**
- Decomposes along channel AND disparity dimensions
- **Up to 7× MAC reduction** (slightly more aggressive)
- **Sometimes hurts accuracy** (0.17-0.21 EPE loss)
- Only use when FwSC's compute budget is insufficient

**Paper's recommendation:** Start with FwSCs. Only use FDwSCs when extreme compression is needed and some accuracy loss is acceptable.

---

## Strengths & Limitations

**Strengths:**
- **Quantifies the 3D bottleneck** — the 94% finding is a headline result that motivates the entire field's efficiency research
- **Plug-and-run simplicity** — no architectural redesign needed; just replace 3D convs
- **Demonstrated across 3 networks** (GANet11, GANetdeep, PSMNet) — generalizable
- **Accuracy maintained or improved** with FwSCs in most cases
- **~7× MAC reduction** is a massive one-line change
- **Parameter reduction** (up to 3.5×) benefits memory-constrained edge devices
- **Fundamental observation** that has guided subsequent edge stereo research (MobileStereoNet, BANet, LightStereo all build on this)

**Limitations:**
- **Short paper** (5 pages, ICIP workshop format) — doesn't explore every corner
- **Only tested on 3D-cost-volume architectures** — the work predates iterative methods (RAFT-Stereo)
- **No direct edge device benchmarks** (Jetson, mobile NPU) — only MACs and desktop GPU
- **FDwSC accuracy loss** is significant (0.17+ EPE) — only FwSC is "free"
- **Depthwise 3D convolutions may have poor hardware support** on some NPUs (cache-unfriendly)
- **No inference latency reported** — only MACs (a proxy for speed)

---

## Relevance to Our Edge Model

**Separable-Stereo's insight is foundational for every 3D convolution in our architecture.**

### Directly adoptable

1. **Use FwSCs everywhere we have 3D convolutions** — this includes:
   - IGEV-Stereo's 3D UNet regularization (most important target)
   - FoundationStereo's AHCF hourglass
   - Any cost volume aggregation network we build
   - Applied to IGEV-Stereo's GEV construction, this gives **~6.5× MAC reduction** for that stage

2. **Verify with Scene Flow EPE** — the paper's key finding is that FwSCs preserve or improve EPE. We should verify this on our own architectures.

3. **Use the 94% bottleneck finding** to prioritize where to optimize — don't waste effort optimizing 6% of the compute.

4. **Avoid FDwSCs** unless extreme compression is needed — they have real accuracy costs.

5. **Plug-and-run with fine-tuning** — we don't need to train from scratch when adopting FwSCs.

### Combined with other Tier 1 techniques

Separable-Stereo stacks multiplicatively with other efficiency techniques:

| Combined with | Compound effect |
|---------------|----------------|
| **BGNet** (1/8 resolution 3D) | 64× volume reduction × 6.5× per-layer reduction = **~400× effective** |
| **LightStereo's channel boost** | LightStereo eliminates 3D entirely in the aggregation; Separable-Stereo applies where 3D is essential |
| **IGEV-Stereo's CGEV** | Replace the 3D UNet with FwSC version; GEV still works because it's an architectural concept |
| **Pip-Stereo's PIP** | Orthogonal — Separable for per-layer compression, PIP for iteration compression |
| **DTP's distillation** | Orthogonal — Separable for architecture, DTP for training |

### Hardware caveat

Depthwise **3D** convolutions may have poor cache behavior on mobile NPUs. On NPUs where depthwise-separable 3D is unsupported, the fallback strategies are:

1. **LightStereo's channel boost 2D blocks** (eliminates 3D entirely)
2. **BGNet's bilateral grid** (does 3D at tiny sizes, easier to fit in cache)
3. **BANet's bilateral aggregation** (pure 2D with detail/smooth split)

But on **Jetson Orin Nano** (which has good 3D support), FwSCs should directly translate to wall-clock speedup.

### For Our Edge Model

The proposed architecture uses IGEV-Stereo's GEV concept combined with separable 3D convolutions:

```
Group-wise Correlation Volume
    ↓
[3D UNet with FwSC layers]  ← Separable-Stereo
(small hourglass at 1/8 resolution — combining BGNet's insight)
    ↓
[GEV + soft-argmin warm start]  ← IGEV-Stereo
    ↓
[Single-layer GRU with SRU + PIP]  ← Selective-Stereo + Pip-Stereo
    ↓
[CUBG upsampling]  ← BGNet
```

The 3D UNet uses FwSCs throughout, giving ~6.5× MAC reduction while maintaining accuracy.

---

## Connections to Other Papers

| Paper | Relationship |
|-------|-------------|
| **MobileNet V1** | Direct inspiration — depthwise separable 2D convs extended to 3D |
| **PSMNet / GANet11 / GANetdeep** | Direct targets — FwSCs plug into these baselines |
| **MobileStereoNet (WACV 2022)** | **Direct successor** by the same group — extends and supersedes Separable-Stereo |
| **BGNet** | Complementary — BGNet reduces volume resolution, Separable-Stereo reduces per-layer compute |
| **LightStereo** | Alternative approach — eliminates 3D entirely rather than making it cheaper |
| **BANet** | Also uses MobileNetV2 blocks but for 2D aggregation with bilateral split |
| **IGEV-Stereo / IGEV++** | Direct beneficiaries — their 3D UNets can use FwSCs for ~6.5× speedup |
| **FoundationStereo** | Uses APC (Axial-Planar Convolution) as an alternative factorization |
| **Distill-then-Prune** | Orthogonal compression axis — separable for architecture, distillation for training |
