# AIO-Stereo: All-in-One: Transferring Vision Foundation Models into Stereo Matching

**Authors:** Zhou et al.
**Venue:** AAAI 2025
**Priority:** 8/10

---

## Core Problem & Motivation

Iterative stereo methods focus on refining the update mechanism but neglect the quality of the feature encoder, which struggles in dark areas and textureless regions due to limited training data. Vision Foundation Models (VFMs) like DINOv2, SAM, and Depth Anything V2 have strong representations but naively fusing their heterogeneous features into a CNN stereo backbone causes architectural mismatch and knowledge conflicts.

## Key Innovation: Multi-VFM Distillation with Selective Gating

AIO-Stereo is the first to combine **three heterogeneous VFMs** as teacher networks, each contributing complementary information:
- **DINOv2** — strong on foreground/salient regions
- **SAM** — strong on edges/boundaries
- **Depth Anything V2** — strong on dark/textureless areas

Critically, **VFMs are frozen teachers used only during training** — at inference, only the lightweight CNN stereo backbone runs. No VFM inference cost at deployment.

![Figure 2: AIO-Stereo architecture. Three frozen VFMs (DINO, SAM, Depth Anything) provide features at three stages. Dual-Level Selective Knowledge Transfer (DLSKT) modules distill their knowledge into the CNN context network via expert networks and KeepTopK gating.](../../../figures/AIO_fig2_architecture.png)

## Architecture

Built on Selective-IGEV / RAFT-Stereo backbone with a **Dual-Level Selective Knowledge Transfer (DLSKT)** module inserted at each of three residual block stages:

1. **Feature Alignment (FA) network** — aligns each VFM's features to a common space via MSE loss
2. **Expert networks** — one per VFM, distill VFM knowledge into lightweight representations
3. **Selective gating** — pixel-wise Mixture-of-Experts with KeepTopK selects which VFM expert to trust at each location

### Key Equation — Selective Gating

$$g_i = \text{KeepTopK}(\text{Softmax}(G_i(f_i \mid \psi_i)), k, \text{dim}=0)$$

- **$g_i$** = gating weights for stage $i$ — determines which VFM expert contributes at each pixel
- **$G_i$** = gating network with parameters $\psi_i$
- **$f_i$** = intermediate CNN feature at block $i$
- **$k$** = number of experts retained per pixel (top-k selection)
- The softmax ensures weights sum to 1; KeepTopK zeroes out all but the top $k$ experts

### Training Loss

$$\mathcal{L}_{AIO} = \mathcal{L}_P + \mathcal{L}_{KD} = \mathcal{L}_P + \sum_{j=1}^{3} \gamma_{KD}^{4-j} \cdot \mathcal{L}_{KD,j}$$

- **$\mathcal{L}_P$** = standard prediction loss (weighted L1 across iterations)
- **$\mathcal{L}_{KD,j}$** = knowledge distillation loss at stage $j$ (MSE between aligned VFM and CNN features)
- **$\gamma_{KD}$** = decay factor — later stages get higher weight

## Benchmark Results

| Dataset | Metric | AIO-Stereo |
|---------|--------|-----------|
| Middlebury | bad2.0 | **2.36** (rank 1) |
| ETH3D | bad1.0 | **0.94** |
| KITTI 2015 | D1-all | **1.54** |

## Relevance to Our Edge Model

**Highly relevant.** Since VFMs are only needed during training (knowledge distillation), the deployed model remains a lightweight CNN. This is exactly the paradigm we need: distill foundation model knowledge into our edge model's encoder during training, then deploy without any VFM overhead. The selective distillation idea could be adapted to distill from a single compact VFM.
