# Selective-Stereo: Adaptive Frequency Information Selection for Stereo Matching

**Authors:** Xianqi Wang, Gangwei Xu, Hao Jia, Xin Yang (Huazhong University of Science and Technology)
**Venue:** CVPR 2024
**Priority:** 8/10
**Code:** https://github.com/Windsrain/Selective-Stereo

---

## Core Problem & Motivation

RAFT-Stereo and IGEV-Stereo use a **single GRU update operator** with a fixed kernel size. This creates an irreconcilable tension:

- **Edge regions** require **high-frequency information** — small receptive fields with small kernels capture fine boundaries and sharp disparity transitions
- **Smooth/textureless regions** require **low-frequency information** — large receptive fields with large kernels are needed to span flat areas and avoid false matches

A single GRU can only use one kernel size, so it cannot satisfy both. The result:
- RAFT-Stereo and IGEV-Stereo **lose fine edge detail** in high-frequency regions
- They **produce false matches or blur** in textureless smooth areas
- IGEV-Stereo's aggregated cost volume partially helps smooth regions but **makes things worse for edges** (the network compensates by seeking high-frequency detail with small kernels)

**Selective-Stereo's insight:** The network needs to **simultaneously maintain multiple receptive fields** during iteration and **adaptively select** which frequency to use on a per-pixel, per-region basis.

---

## Architecture

![Figure 2: Selective-Stereo overview. Feature Network and Context Network extract multi-level features. The Contextual Spatial Attention (CSA) module generates multi-level attention maps from context information. These maps guide the Selective Recurrent Units (SRU) at each iteration, fusing two GRU branches with different kernel sizes based on spatial frequency needs.](../../../figures/Selective_fig2_overview.png)

### Key Modules: CSA + SRU

The method is **plug-and-play** — drops into any iterative stereo base (RAFT-Stereo, IGEV-Stereo, DLNR) without architectural surgery. The paper primarily demonstrates on top of IGEV-Stereo ("Selective-IGEV").

### Contextual Spatial Attention (CSA) Module

![Figure 3: Detailed architecture of CSA (left) and SRU (right). CSA processes context features through Channel Attention Enhancement (CAE: max-pool + avg-pool → 2 conv layers → sigmoid) then Spatial Attention Extractor (SAE: another max-pool + avg-pool + conv + sigmoid), producing a spatial attention map A. SRU has two GRU branches (small kernel and large kernel) that run in parallel; their outputs are fused using A (for small kernel, high-frequency) and (1-A) (for large kernel, low-frequency).](../../../figures/Selective_fig3_csa_sru.png)

CSA runs **once before iterations begin** to produce attention maps used throughout. It has two sub-modules:

**Sub-module 1: Channel Attention Enhancement (CAE)**
- Input: context information map $c \in \mathbb{R}^{C \times H \times W}$
- Apply max-pool and avg-pool across spatial dimensions → $t_{avg}, t_{max} \in \mathbb{R}^{C \times 1 \times 1}$
- Two separate conv layers applied to each → element-wise add → sigmoid
- Output: channel weights $M_c \in [0,1]^{C \times 1 \times 1}$
- Apply element-wise to input: highlights channels with **high feature values** (high-frequency info)

**Sub-module 2: Spatial Attention Extractor (SAE)**
- Continue from CAE output
- Apply max-pool and avg-pool **across channel dimension**
- Concatenate → single conv layer → sigmoid
- Output: spatial attention map $A \in [0,1]^{1 \times H \times W}$

**Intuition:** Regions with high feature values in the context map tend to be **high-frequency** (edges, object boundaries). Smooth/textureless regions have **low feature values**. The CSA attention map $A$ spatially encodes this frequency preference — high $A$ where high-frequency info is needed, low $A$ where low-frequency info is needed.

### Selective Recurrent Unit (SRU)

The SRU replaces the standard GRU with **two parallel GRU branches** with different kernel sizes:

**Small kernel branch (1×1):**
- Tiny receptive field
- Captures high-frequency edge information
- Produces hidden state $h_k^s$

**Large kernel branch (3×3):**
- Larger receptive field
- Captures low-frequency smooth-region information
- Produces hidden state $h_k^l$

Both branches use standard GRU equations:

$$z_k = \sigma(\text{Conv}([h_{k-1}, x_k], W_z))$$

$$r_k = \sigma(\text{Conv}([h_{k-1}, x_k], W_r))$$

$$\tilde{h}_k = \tanh(\text{Conv}([r_k \odot h_{k-1}, x_k], W_h))$$

$$h_k = (1 - z_k) \odot h_{k-1} + z_k \odot \tilde{h}_k$$

Every variable:
- **$x_k$** = input: concatenation of disparity, correlation lookup, hidden information, context
- **$h_{k-1}$** = previous hidden state
- **$z_k$** = update gate (how much new information)
- **$r_k$** = reset gate (how much of previous state to forget)
- **$\tilde{h}_k$** = candidate hidden state
- **$h_k$** = new hidden state
- **$W_z, W_r, W_h$** = convolutional weights (different kernel sizes for small vs large branch)
- **$\sigma$** = sigmoid, **$\odot$** = element-wise product

**SRU hidden state fusion:**

$$h_k = A \odot h_k^s + (1 - A) \odot h_k^l \quad \text{(4)}$$

- **$h_k^s$** = hidden state from **small-kernel** GRU branch (high-frequency)
- **$h_k^l$** = hidden state from **large-kernel** GRU branch (low-frequency)
- **$A$** = attention map from CSA, values in $[0, 1]$
- **$(1 - A)$** = **contrary attention** — complement; high where $A$ is low
- **$h_k$** = fused hidden state passed to next iteration

**Where $A$ is high (edge regions):** the small-kernel branch dominates → preserves fine edges
**Where $A$ is low (smooth regions):** the large-kernel branch dominates → spans textureless areas

This is **adaptive frequency-based selection**: the network uses the right receptive field at every pixel.

### Training Loss

$$\mathcal{L} = \sum_{i=1}^{N} \gamma^{N-i} \Vert d_i - d_{gt}\Vert _1 \quad \text{(6)}$$

Standard exponentially-weighted L1 loss over all iterations, $\gamma = 0.9$, $N = 22$ during training, 32 during inference.

---

## Key Innovations vs Prior Work

### vs RAFT-Stereo

| Aspect | RAFT-Stereo | Selective-Stereo |
|--------|-------------|-----------------|
| GRU update | Single GRU, single kernel | Two parallel GRUs: small + large kernel |
| Receptive field | Fixed, single | Multiple, adaptive per pixel |
| Attention guidance | None | CSA provides frequency-aware map |
| Edge regions | Loses detail | Preserved via small-kernel branch |
| Smooth regions | False matches | Large-kernel branch selected |
| Convergence | 32 iterations needed | Same quality in 8 iterations |

### vs IGEV-Stereo

| Aspect | IGEV-Stereo | Selective-IGEV |
|--------|-------------|---------------|
| Cost volume | Aggregated GEV (over-smooths edges) | Same CGEV, but edge detail recovered via SRU |
| Parameters | 12.60M | **11.65M** (smaller!) |
| Scene Flow EPE | 0.47 | **0.44** |
| KITTI 2015 D1-all | 1.44 | **1.33** |
| Runtime | 0.47s | **0.24s** (faster due to fewer iterations) |

---

## Benchmark Results

**Ranked 1st on KITTI 2012, KITTI 2015, ETH3D, and Middlebury** at time of submission.

### Scene Flow

| Method | EPE |
|--------|-----|
| RAFT-Stereo | 0.53 |
| IGEV-Stereo | 0.47 |
| Selective-RAFT | 0.47 |
| **Selective-IGEV** | **0.44** |

### KITTI 2015

| Method | D1-bg | D1-fg | D1-all | Runtime |
|--------|-------|-------|--------|---------|
| RAFT-Stereo | 1.58 | 3.05 | 1.82 | 0.38s |
| IGEV-Stereo | 1.44 | 2.83 | 1.67 | 0.47s |
| **Selective-IGEV** | **1.33** | **2.61** | **1.55** | **0.24s** |

**Selective-IGEV is faster AND more accurate than IGEV-Stereo** — faster because it converges in fewer iterations.

### Middlebury

| Method | Bad 2.0 | Bad 1.0 | Bad 4.0 | AvgErr |
|--------|---------|---------|---------|--------|
| RAFT-Stereo | 4.74 | 9.37 | 2.75 | 1.27 |
| IGEV-Stereo | 4.83 | 9.41 | 3.33 | 2.89 |
| **Selective-IGEV** | **2.51** | **6.53** | **1.36** | **0.91** |

Substantial improvement on Middlebury — the attention-based frequency selection particularly helps high-resolution indoor scenes with fine details.

---

## Ablation Highlights

### Module effectiveness (Scene Flow)

| Configuration | EPE | >1px | Params |
|---------------|-----|------|--------|
| Baseline (RAFT-Stereo) | 0.53 | 6.08 | 11.12M |
| + SRU only (no CSA) | 0.50 | 5.38 | 11.65M |
| + SRU + CSA (**inverted**) | 0.50 | 5.58 | 11.65M |
| **+ SRU + CSA (correct)** | **0.47** | **5.32** | **11.65M** |

**The inverted CSA ablation is a strong test:** using $(1-A)$ for the small-kernel branch and $A$ for the large-kernel branch (the opposite assignment) actually hurts performance (5.58 vs 5.38). This validates that CSA correctly identifies high vs low frequency regions.

### Universality across base architectures

| Method | EPE | Params |
|--------|-----|--------|
| RAFT-Stereo → Selective-RAFT | 0.53 → 0.47 | 11.12 → 11.65M |
| IGEV-Stereo → Selective-IGEV | 0.47 → 0.44 | 12.60 → 11.65M |
| DLNR → Selective-DLNR | 0.49 → 0.46 | 57.37 → 58.09M |

Consistent improvement across all three base networks, confirming the **plug-and-play** claim.

### Kernel size choice

| Small + Large kernels | EPE | >1px |
|----------------------|-----|------|
| 1×1 + 1×5 | 0.48 | 5.41 |
| 3×3 + 1×5 | 0.48 | 5.30 |
| **1×1 + 3×3** | **0.47** | **5.32** |

Counter-intuitively, the **smallest pair (1×1 + 3×3)** works best — large kernels don't help much and cost more.

### Iteration count

| Model | 1 iter | 3 iters | 8 iters | 32 iters |
|-------|--------|---------|---------|----------|
| RAFT-Stereo | 2.08 | 0.87 | 0.58 | 0.53 |
| Selective-RAFT | 1.37 | **0.81** | 0.58 | 0.47 |
| **Selective-IGEV** | **0.65** | **0.56** | **0.48** | **0.44** |

**Selective-RAFT achieves RAFT-Stereo's 32-iter quality in 8 iterations** — 4× reduction in required iterations.

---

## Strengths & Limitations

**Strengths:**
- Elegant solution to a real architectural bottleneck (fixed receptive field)
- Only **+0.53M parameters** overhead — very lightweight
- **Faster convergence** means lower inference time despite more work per iteration
- Plug-and-play: works on RAFT-Stereo, IGEV-Stereo, DLNR with consistent gains
- **Smaller than IGEV-Stereo yet faster and more accurate** (Selective-IGEV: 11.65M/0.24s vs IGEV: 12.60M/0.47s)
- Excellent ablation methodology — the "inverted CSA" test is compelling

**Limitations:**
- Still only 2 discrete kernel sizes — not truly continuous frequency adaptation
- Memory cost scales with number of branches (adding more frequencies expensive)
- CSA runs once before iterations — doesn't update as disparity evolves
- No edge deployment target — 0.24s on RTX 3090 is still too slow for mobile/Jetson
- Self-attention not explored (acknowledged as future work)

---

## Relevance to Our Edge Model

**Directly applicable:**

1. **SRU concept at reduced scale** — for edge, an SRU with two lightweight branches using **depthwise separable convolutions** (1×1 + 3×3) preserves the multi-frequency benefit at minimal overhead. Maps directly onto our "separable 3D conv" direction.

2. **CSA as precomputed attention** — CSA runs once before iterations, amortizing its cost. On edge, this one-time precompute is affordable even with compressed context networks.

3. **Fewer iterations with better initialization** — the finding that SRU achieves same quality in 4× fewer iterations is gold for edge latency budgets.

4. **1×1 + 3×3 is optimal** — **smaller kernels work best**. No need for expensive large-kernel branches.

5. **Plug into IGEV-Stereo base** — Selective-IGEV already combines SRU with IGEV-Stereo. Our edge model building on IGEV-Stereo should include SRU.

**Cautions:**

- Dual GRU branches double per-iteration compute — net benefit depends on iteration count reduction
- CSA's channel + spatial attention submodules may be expensive on mobile NPUs lacking efficient attention support. A simplified version (channel attention only, or single lightweight conv) might be needed.
- Paper doesn't report ONNX/TensorRT export — needs validation on edge

**Proposed edge variant:** SRU with depthwise separable 1×1 + 3×3 branches + simplified CSA (channel attention only) on top of a lightweight IGEV-Stereo backbone. Adds ~0.2M params, expects 4× iteration reduction → net latency win.

---

## Connections to Other Papers

| Paper | Relationship |
|-------|-------------|
| **RAFT-Stereo** | Base architecture (Selective-RAFT variant) |
| **IGEV-Stereo** | Primary base architecture (Selective-IGEV is the main model) |
| **DLNR** | Third base architecture validated |
| **CBAM** | CSA module is inspired by CBAM's channel + spatial attention |
| **IGEV++** | Contemporary successor to IGEV-Stereo addressing different issues (multi-range) |
| **DEFOM-Stereo** | Orthogonal improvement — could combine DEFOM's mono features with Selective's SRU |
| **DLNR** | LSTM-based, shown compatible with SRU concept (replace LSTM with selective-LSTM) |
