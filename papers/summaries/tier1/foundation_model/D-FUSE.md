# D-FUSE: Diving into the Fusion of Monocular Priors for Generalized Stereo Matching

**Authors:** Yao et al.
**Venue:** ICCV 2025
**Priority:** 8/10

---

## Core Problem & Motivation

Fusing monocular depth priors with stereo matching improves generalization, but existing methods suffer from three fundamental problems:
1. **Misalignment** — affine-invariant relative monocular depth vs. absolute binocular disparity
2. **Over-confidence** — iterative updates get stuck in local optima when monocular features are naively fused
3. **Noisy early iterations** — early disparity estimates misguide the fusion

D-FUSE is the first paper to systematically analyze and solve all three issues.

## Key Innovation: Binary Local Ordering Maps

Instead of directly aligning relative depth with absolute disparity (complex, compute-heavy), D-FUSE converts both representations into **binary ordering maps** encoding whether neighboring pixels are closer or farther:

$$M_O(u,v) = \{\sigma(D(u',v') - D(u,v))\} \quad \text{for } (u',v') \in \mathcal{N}_{(u,v)}$$

- **$M_O(u,v)$** = binary ordering map at pixel $(u,v)$ — a vector of values near 0 or 1
- **$D(u,v)$** = depth (or disparity) at the center pixel
- **$D(u',v')$** = depth at a neighboring pixel
- **$\sigma(\cdot)$** = sigmoid function — outputs near 1 if neighbor is farther, near 0 if closer
- **$\mathcal{N}_{(u,v)}$** = local neighborhood (LBP-like pattern)

**Why this works:** Relative orderings are naturally invariant to scale and shift — a pixel that's "closer than its neighbor" in relative depth is also "closer than its neighbor" in absolute disparity. This unifies the two representations without any alignment.

![Figure 2: D-FUSE pipeline. Three modules: Monocular Encoder (frozen Depth Anything V2), Iterative Local Fusion (binary ordering maps + Beta-distributed guidance + re-weighted GRU updates), and Global Fusion (scale/shift registration + confidence-weighted combination).](../../../figures/DFUSE_fig2_pipeline.png)

## Architecture: Three Modules

### 1. Monocular Encoder
Frozen Depth Anything V2 extracts features and monocular depth from the left image. A two-stream conv module generates initial hidden states and context features.

### 2. Iterative Local Fusion
- **Stream 1:** Computes binary ordering maps from both monocular depth and current binocular disparity using fixed-weight LBP convolutions. Produces guidance signal $G$ modeled as a Beta distribution.
- **Stream 2:** Multi-level GRU with cost volume lookup (standard RAFT-Stereo)
- The guidance $G$ re-weights the disparity update to avoid local optima

### 3. Global Fusion (post-iterative)
Registers monocular depth to binocular disparity via pixel-wise scale and shift:

$$\tilde{D}_m = a \cdot D_m + b$$

$$D_f = c \cdot D_d^T + (1-c) \cdot \tilde{D}_m$$

- **$D_m$** = monocular depth prediction, **$D_d^T$** = optimized binocular disparity
- **$a, b$** = pixel-wise scale and shift parameters learned by a conv network
- **$c$** = learned confidence map — where stereo is reliable, $c \to 1$; where monocular is better, $c \to 0$

## Benchmark Results (Zero-Shot)

| Dataset | EPE | Bad-τ |
|---------|-----|-------|
| KITTI 2015 | **1.12** | bad 3.0: **5.60** |
| Middlebury (H) | **1.15** | bad 2.0: **8.39** |
| ETH3D | **0.25** | bad 1.0: **1.88** |
| Booster (Q) | **2.26** | bad 2.0: **11.02** |

SOTA zero-shot across all five real-world datasets. Particularly strong on Booster (transparent/reflective): 10-point improvement on bad 2.0 for transparent regions.

**Inference time:** 0.40s (only 0.08s overhead over RAFT-Stereo baseline — the fusion modules are very lightweight).

## Relevance to Our Edge Model

**Directly applicable.** Binary ordering maps use fixed-weight convolutions (no learnable parameters, near-zero compute). The global fusion module (a few conv layers for scale/shift + confidence) is also lightweight. The key insight — that only 0.08s overhead is needed for monocular fusion — suggests this approach could work with a small distilled backbone on edge devices.
