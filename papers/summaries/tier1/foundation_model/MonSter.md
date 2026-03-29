# MonSter: Marry Monodepth to Stereo Unleashes Power

**Authors:** Junda Cheng et al.
**Venue:** CVPR 2025
**Priority:** 9/10
**Code:** https://github.com/Junda24/MonSter

---

## Core Problem & Motivation

Monocular depth models provide strong geometric priors in ill-posed regions but suffer from scale/shift ambiguity. Even after global alignment, substantial per-pixel errors remain. MonSter's insight: rather than using monocular depth as a simple initialization, reformulate the problem as **recovering per-pixel scale and shift** using stereo matching as the corrective signal, creating a bidirectional synergy.

## Architecture: Bidirectional Mutual Refinement

![Figure 5: MonSter architecture. Dual-branch design: Monocular branch (frozen DepthAnythingV2 ViT-L) produces relative depth. Stereo branch (IGEV-based) produces initial stereo disparity. SGA (Stereo Guided Alignment) refines monocular scale using stereo cues. MGR (Mono Guided Refinement) refines stereo disparity using monocular priors. These alternate for N2 iterations.](../../../figures/MonSter_fig5_architecture.png)

### Three Main Components

**1. Monocular Branch:** Frozen DepthAnythingV2 (ViT-L + DPT decoder). ViT encoder is **shared** with stereo branch via a Feature Transfer Network (conv layers that downsample ViT features into a multi-scale pyramid).

**2. Stereo Branch:** IGEV architecture — constructs Geometry Encoding Volume, N1 ConvGRU iterations for initial stereo disparity.

**3. Mutual Refinement Module** (the core novelty):

**Step 1 — Global Scale-Shift Alignment:**
$$s_G, t_G = \arg\min_{s_G, t_G} \sum_{i \in \Omega} (s_G \cdot D_M(i) + t_G - D_S^0(i))^2$$
- **$D_M(i)$** = inverse monocular depth at pixel $i$
- **$D_S^0(i)$** = initial stereo disparity
- **$\Omega$** = pixels where stereo disparity is between 20th and 90th percentile (filters unreliable extremes)
- **$s_G, t_G$** = global scale and shift — solved via least squares

**Step 2 — Stereo Guided Alignment (SGA):** Iteratively refines monocular disparity using stereo cues. A confidence-based flow residual map measures matching reliability:
$$F_S^j(x,y) = \|F_3^L(x,y) - F_3^R(x - D_S^j, y)\|_1$$
- **$F_S^j$** = flow residual — low values = high confidence in stereo match
- Condition-guided ConvGRU predicts per-pixel residual shift $\Delta t$ to update monocular disparity

**Step 3 — Mono Guided Refinement (MGR):** Symmetric to SGA — uses refined monocular disparity to guide stereo refinement in ill-posed regions via a separate ConvGRU.

SGA and MGR alternate for $N_2$ iterations, creating a mutual improvement loop.

### Training Loss
$$\mathcal{L} = \sum_{i=0}^{N_1-1} \gamma^{N_1+N_2-i} |d_i - d_{gt}|_1 + \sum_{i=N_1}^{N_1+N_2-1} \gamma^{N_1+N_2-i} (|D_S^{i-N_1} - d_{gt}|_1 + |D_M^{i-N_1} - d_{gt}|_1)$$
- Both stereo and mono branches are supervised — enforcing that both improve

## Benchmark Results

**Ranks 1st simultaneously on ETH3D, Middlebury, KITTI 2015, and KITTI 2012.**

| Dataset | Best Metric | Improvement over IGEV |
|---------|-----------|----------------------|
| ETH3D | Bad 1.0 All: **0.45** | 70.20% |
| Middlebury | Bad 4.0 Noc: **1.18** | 64.56% |
| KITTI 2015 | D1-all: **1.37** | — |
| Scene Flow | EPE: **0.37** | 21.28% |

## Model Size & Speed

| Variant | Params | Inference Time |
|---------|--------|---------------|
| MonSter (32 iters) | 356.1M (335.3M frozen ViT-L) | 0.64s |
| MonSter (4 iters) | 356.1M | 0.34s |
| **RT-MonSter** | ~20M (stereo+refinement only) | **47ms** (~21 FPS) |

**RT-MonSter** uses coarse-to-fine cascade with local cost volumes and single-layer ConvGRU. **Only 2GB GPU memory.** Best accuracy among all real-time methods.

## Critical Ablation Finding

**With only 4 iterations (N1=N2=2), MonSter achieves EPE 0.42 in 0.34s — a 10.64% improvement over IGEV at 32 iterations (0.47 in 0.37s).** Mutual refinement with monocular priors provides such good initialization that fewer iterations suffice.

## Relevance to Edge Model

- **Mutual refinement reduces iteration count** — 4 iterations with mono priors beats 32 without. Critical for edge latency.
- **RT-MonSter is near edge-feasible** — 47ms / 2GB. With backbone replacement + quantization, sub-30ms on Jetson may be achievable.
- **Feature sharing via transfer network** — lightweight conv layers decouple ViT from stereo pipeline. Can swap in distilled backbone.
- **Confidence-based flow residual** — L1 norm computation, nearly free, provides automatic matching confidence.
