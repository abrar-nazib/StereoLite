# Synthesis: The Iterative Paradigm Evolution (RAFT-Stereo → IGEV++)

> Consolidation of 5 Tier 1 iterative-method papers spanning 2021-2025.

---

## Papers Covered

| # | Paper | Year | Key Contribution | Backbone GRU | Top KITTI-15 D1-all |
|---|-------|------|-----------------|-------------|---------------------|
| 1 | **RAFT-Stereo** | 2021 | Iterative paradigm + correlation pyramid | 3-level ConvGRU | 1.82% |
| 2 | **CREStereo** | 2022 | Cascaded + Adaptive Group Correlation + 2D-1D search | 3-level ConvGRU | 1.69% |
| 3 | **IGEV-Stereo** | 2023 | Combined Geometry Encoding Volume (GEV + APC) | 3-level ConvGRU | 1.59% |
| 4 | **Selective-Stereo** | 2024 | Selective Recurrent Unit (SRU) + CSA | Selective GRU | 1.55% |
| 5 | **IGEV++** | 2025 | Multi-range GEV + APM + SGFF | 3-level ConvGRU | 1.43% |

---

## The Grand Narrative: Four Waves of Iterative Stereo

### Wave 1 (2021): Establishing the Paradigm — RAFT-Stereo

**Problem:** 3D cost volume methods (PSMNet, GA-Net) were expensive and couldn't handle high resolution.

**Solution:** Adapt RAFT's optical flow iterative refinement to stereo:
- **1D all-pairs correlation volume** (single matmul, no 3D convs)
- **Multi-level ConvGRU** for propagating context across the image
- **Slow-fast GRU** for efficient inference
- **Warm-start from zero disparity, iteratively refine**

**Why this changed everything:** No more 3D convolutions. Handles megapixel images. Strong cross-dataset generalization. Flexible speed-accuracy via iteration count.

**Remaining limitations:**
- Assumes perfect rectification
- All-pairs correlation expensive at very high resolution
- Slow convergence (32 iterations)
- Local matching only — struggles with textureless/occluded regions

### Wave 2 (2022): Practical Robustness — CREStereo

**Problem:** RAFT-Stereo's all-pairs correlation has memory issues at megapixel resolution, and assumes perfect rectification (which consumer cameras violate).

**Solution:**
- **Local correlation window** ($H \times W \times 9$ instead of $H \times W \times W$) — 50× memory reduction
- **2D-1D alternating search** — handles real-world rectification errors
- **Deformable search** — content-adaptive matching in occluded/textureless regions
- **Group-wise correlation** for richer feature matching
- **Cascaded coarse-to-fine** with shared weights across scales
- **Stacked cascades** for any inference resolution without retraining

**Surprising finding:** Local correlation **strictly beats** all-pairs correlation (~2.3× lower Middlebury Bad 2.0). The combination of local window + iterative resampling is a better propagation mechanism.

### Wave 3 (2023): Bridging to Cost-Filtering — IGEV-Stereo

**Problem:** Pure correlation (RAFT-Stereo, CREStereo) is local and context-free. In textureless, occluded, or reflective regions, it collapses.

**Solution:** Combine iterative refinement with **cost-filtering's strength** (non-local geometry from 3D convolutions):
- Build a **Geometry Encoding Volume (GEV)** by running a lightweight 3D UNet on correlations **once** (not per iteration)
- Combine with all-pairs correlations into the **CGEV** — geometry context + local detail
- **Warm-start from soft argmin on GEV** — dramatically better than $d_0 = 0$
- **Dual supervision**: explicitly train the GEV output ($\mathcal{L}_{init}$) alongside final iterations

**Critical result:** At 1 iteration, IGEV-Stereo achieves EPE 0.66 vs RAFT-Stereo's 2.16 — a **69% reduction** from warm start alone. At 3 iterations, it already beats RAFT-Stereo at 32 iterations.

**Why this matters:** IGEV-Stereo demonstrated that cost-filtering and iterative methods are **complementary**, not competing. The 3D UNet adds only 0.58M params and 1ms but provides most of the accuracy gain.

### Wave 4a (2024): Fixing Over-Smoothing — Selective-Stereo

**Problem:** Every iterative method (RAFT-Stereo, CREStereo, IGEV-Stereo) uses a single GRU with one fixed kernel size. This creates tension:
- Edge regions need **small kernels** (high frequency)
- Smooth regions need **large kernels** (low frequency)
- A single GRU can only use one → either blurs edges or fails on textureless

**Solution:**
- **Selective Recurrent Unit (SRU)**: two parallel GRU branches with 1×1 and 3×3 kernels
- **Contextual Spatial Attention (CSA)**: precomputed per-pixel attention map from context features
- Fuse via $h_k = A \odot h_k^{small} + (1-A) \odot h_k^{large}$ — edges use small kernel, smooth regions use large

**Key finding:** This is **plug-and-play** — drops into RAFT-Stereo, IGEV-Stereo, or DLNR with consistent gains at +0.53M params. Selective-IGEV achieves **better accuracy AND faster inference** than IGEV-Stereo (converges in 8 iterations instead of 16).

### Wave 4b (2025): Large-Disparity Specialization — IGEV++

**Problem:** IGEV-Stereo v1's fixed single-range GEV (192px max) fails on large disparities. Naively extending the range explodes memory.

**Solution:**
- **Multi-range GEV (MGEV)**: three volumes at 192, 384, 768px ranges
- **Adaptive Patch Matching (APM)**: for medium/large ranges, use sparse sampling (stride-2, stride-4) with learned 4-pixel patch weights — 48 candidates per range regardless of range size
- **Selective Geometry Feature Fusion (SGFF)**: per-pixel spatial weights to fuse features from all three volumes based on local content
- **RT-IGEV++** variant: single-range, single-level GRU, 96-channel hidden state → **48ms on KITTI** (best among real-time methods)

---

## Architectural Evolution Table

| Feature | RAFT-Stereo | CREStereo | IGEV-Stereo | Selective-Stereo | IGEV++ |
|---------|:----------:|:---------:|:-----------:|:---------------:|:------:|
| Correlation volume | All-pairs | Local window (9 cands) | GEV + APC (CGEV) | Same as base | MGEV (3 ranges, APM) |
| Volume size (typical) | $H·W·W$ | $H·W·9$ | $H·W·D$ + APC | Same as base | $H·W·48·3$ (sparse) |
| 3D convolutions | ❌ | ❌ | ✅ Small 3D UNet | Base-dependent | ✅ 3× 3D UNet |
| Warm start | ❌ ($d_0 = 0$) | Cascade upsampling | ✅ Soft argmin on GEV | Base-dependent | ✅ Soft argmin on $G^s$ |
| Iterations needed | 22-32 | 8-16 per cascade stage | 3 (matches RAFT-32) | 4-8 (50% fewer) | 2 (beats RAFT-32) |
| Multi-frequency GRU | ❌ | ❌ | ❌ | ✅ SRU + CSA | ❌ (orthogonal) |
| 2D search | ❌ | ✅ (alternating) | ❌ | ❌ | ❌ |
| Backbone | ResNet | CNN | MobileNetV2 | Base-dependent | MobileNetV2 / DAv2-L |
| Real-time variant | Slow-Fast GRU | ❌ | ❌ | Selective-IGEV (0.24s) | RT-IGEV++ (48ms) |

---

## Convergence Comparison (Scene Flow EPE vs Iterations)

| Model | 1 iter | 2 iters | 3 iters | 8 iters | 32 iters |
|-------|--------|---------|---------|---------|----------|
| RAFT-Stereo | 2.16 | 1.21 | 0.95 | 0.66 | 0.61 |
| IGEV-Stereo | 0.66 | 0.62 | **0.58** | 0.51 | 0.47 |
| Selective-RAFT | 1.37 | 0.82 | 0.81 | 0.58 | 0.47 |
| **Selective-IGEV** | 0.65 | 0.60 | 0.56 | **0.48** | **0.44** |
| **IGEV++** | — | **0.83** | — | — | — |

**Two insights:**
1. **Warm start dominates** — IGEV-Stereo's 1-iteration result (0.66) already beats RAFT-Stereo's 32-iteration result (0.61)
2. **Selective-IGEV is the current sweet spot** — best quality at every iteration count, and converges 4× faster than RAFT-Stereo

---

## Common Architectural Elements (What Every Iterative Method Needs)

Despite their differences, all 5 papers share core elements:

### 1. Feature Encoder + Context Encoder (separate)
- Both are 2D CNNs applied to input images
- Feature encoder used for cost volume construction
- Context encoder (left image only) provides initial GRU hidden state and per-iteration context injection

### 2. Cost Volume (with or without filtering)
- Some form of correlation between left and right features
- RAFT-Stereo: pure all-pairs, no filtering
- CREStereo: local window, no filtering
- IGEV-Stereo/IGEV++: correlation + 3D UNet filtering
- Selective-Stereo: same as base, with SRU on top

### 3. ConvGRU Update Operator
- Multi-level GRUs (typically 3: 1/4, 1/8, 1/16) with cross-connections
- Context features additively injected into gate computations
- Correlation lookup at current disparity fed to GRU input
- Residual disparity update: $d_{k+1} = d_k + \Delta d_k$

### 4. Convex Upsampling
- Final 1/4-resolution disparity upsampled via convex combination of 3×3 neighborhood
- Weights predicted by highest-resolution GRU
- Some variants (IGEV) use fuller features for upsampling weights

### 5. Exponentially-Weighted L1 Loss
- $\mathcal{L} = \sum_{i=1}^{N} \gamma^{N-i} \|d_i - d_{gt}\|_1$ with $\gamma = 0.9$
- Supervises every intermediate iteration
- IGEV/IGEV++ add auxiliary loss on initial disparity ($\mathcal{L}_{init}$)

---

## Key Equations Shared Across the Lineage

### Correlation volume (Eq. 1 in RAFT-Stereo)

$$\mathbf{C}_{ijk} = \sum_h \mathbf{f}^l_{hij} \cdot \mathbf{f}^r_{hik}$$

Survives in every paper in some form:
- **RAFT-Stereo**: used directly
- **CREStereo**: made local (only $k \in [j-r, j+r]$) + grouped
- **IGEV-Stereo**: combined with 3D-regularized GEV
- **IGEV++**: extended to multi-range with APM

### Group-wise correlation (Eq. 1 in IGEV-Stereo, Eq. 1 in IGEV++)

$$C^s(g, d^s, x, y) = \frac{1}{N_c/N_g} \langle f^g_{l,4}(x,y), f^g_{r,4}(x-d^s, y) \rangle$$

Borrowed from GwcNet (2019). Used by CREStereo, IGEV-Stereo, IGEV++.

### ConvGRU Update (same structure in all papers)

$$z_k = \sigma(\text{Conv}([h_{k-1}, x_k], W_z) + c_z)$$
$$r_k = \sigma(\text{Conv}([h_{k-1}, x_k], W_r) + c_r)$$
$$\tilde{h}_k = \tanh(\text{Conv}([r_k \odot h_{k-1}, x_k], W_h) + c_h)$$
$$h_k = (1 - z_k) \odot h_{k-1} + z_k \odot \tilde{h}_k$$

Identical structure from RAFT-Stereo through IGEV++. Only variant: Selective-Stereo runs this twice (small + large kernel) and fuses with attention.

### Residual disparity update

$$d_{k+1} = d_k + \Delta d_k$$

The absolute foundation — residual accumulation is why iteration works.

### Exponentially-weighted loss

$$\mathcal{L} = \sum_{i=1}^{N} \gamma^{N-i} \Vert d_i - d_{gt}\Vert _1, \quad \gamma = 0.9$$

Unchanged from RAFT-Stereo. IGEV variants add auxiliary $\mathcal{L}_{init}$.

---

## Benchmark Evolution

### KITTI 2015 D1-all over time

```
2021  RAFT-Stereo       1.82%  |████████████████████░░░░
2022  CREStereo         1.69%  |█████████████████░░░░░░░
2023  IGEV-Stereo       1.59%  |███████████████░░░░░░░░░
2024  Selective-IGEV    1.55%  |██████████████░░░░░░░░░░
2025  IGEV++            1.43%  |████████████░░░░░░░░░░░░
```

### Scene Flow EPE over time

```
2021  RAFT-Stereo       0.53
2022  CREStereo         0.40
2023  IGEV-Stereo       0.47
2024  Selective-IGEV    0.44
2025  IGEV++            0.43
```

### Inference Time (960×540, similar GPU)

```
RAFT-Stereo       0.36s  | 22 iterations
CREStereo         0.41s  | not reported, slow
IGEV-Stereo       0.18s  | 16 iterations, best speed/accuracy
Selective-IGEV    0.24s  | 8 iterations (fewer needed)
IGEV++            0.28s  | 22 iterations with DAv2 backbone
RT-IGEV++         0.048s | 6 iterations, single-level GRU
```

---

## The Common Weaknesses (What's Still Missing)

1. **All methods are Tier 1-accuracy but not edge-fast** — the best are still ~0.18-0.28s on desktop GPUs, far from Jetson Orin Nano's ~33ms target.

2. **3D convolutions remain a problem** — IGEV-Stereo and IGEV++ depend on 3D UNets. Poor NPU support.

3. **No monocular priors** — all 5 papers are pure stereo. They lose to foundation-model methods (DEFOM-Stereo, MonSter) on zero-shot generalization and ill-posed regions.

4. **Fixed disparity range at test time** — IGEV++ addresses this but at the cost of more volumes. None handle arbitrary range as elegantly as would be ideal.

5. **Limited 2D search** — only CREStereo handles rectification errors (2D-1D alternating). All others assume perfect rectification.

---

## The Path Forward: What Our Edge Model Should Inherit

Taking the best from each paper:

### From RAFT-Stereo
- **Multi-level ConvGRU framework** (but reduce to 1 or 2 levels for edge)
- **Residual disparity update** (fundamental)
- **Convex upsampling**
- **Slow-fast variant** (update coarse levels more often)

### From CREStereo
- **Local correlation window** (NOT all-pairs — the surprising finding)
- **2D-1D alternating search** (nearly free robustness to rectification errors)
- **Group-wise correlation** (better matching per compute)
- **Cascaded inference** (optional, for high-res)

### From IGEV-Stereo
- **Combined Geometry Encoding Volume** — 3D regularization + local correlations
- **Warm start from soft argmin on GEV** (eliminates wasted iterations)
- **Dual supervision** (auxiliary $\mathcal{L}_{init}$)
- **MobileNetV2 backbone** (already edge-aware)

### From Selective-Stereo
- **Selective Recurrent Unit (SRU)** — 1×1 + 3×3 branches with attention fusion
- **Contextual Spatial Attention** (precomputed once)
- **Smaller kernels work best** (1×1 + 3×3)

### From IGEV++
- **Adaptive Patch Matching** (sparse sampling with learned weights)
- **Selective Geometry Feature Fusion** (spatially adaptive multi-scale fusion)
- **RT-IGEV++ design** as template: single backbone, single-range GEV, single-level GRU, 96-channel hidden state, 6 iterations → 48ms
- **$\lambda_s = 1.0, \lambda_m = 0.5, \lambda_l = 0.2$** weighting scheme for multi-scale supervision

### Not From These Papers (from foundation-model era)
- **Monocular depth initialization** (DEFOM-Stereo) — replaces need for multi-range volumes
- **Foundation model feature distillation** (Fast-FoundationStereo) — provides robust priors without inference-time VFM
- **Bidirectional refinement** (MonSter) — optional; SGA/MGR mutual refinement

---

## Proposed Edge Model Architecture

Based on this synthesis:

```
Stereo Pair
    ↓
[Distilled CNN Backbone]  ← MobileNetV4 / EfficientViT
(feature + context, shared)
    ↓
[Distilled Mono Depth Head]  ← From DEFOM-Stereo line
    ↓
[Local Correlation + Group-wise]  ← From CREStereo (not all-pairs!)
(with 2D-1D alternating)
    ↓
[Lightweight GEV]  ← From IGEV-Stereo (with separable 3D conv)
(1/8 resolution, not 1/4)
    ↓
[Warm Start]  ← Soft argmin on GEV AND distilled mono depth
    ↓
[SRU × 4 iterations]  ← From Selective-Stereo
(1×1 + 3×3 branches + CSA)
    ↓
[Convex Upsampling]  ← From RAFT-Stereo
    ↓
Disparity Map
    ↓
[ONNX → TensorRT]  ← Jetson Orin Nano / mobile NPU
```

**Target specs:**
- **<5M parameters** (vs RT-IGEV++'s ~3.5M)
- **<33ms on Jetson Orin Nano** (vs RT-IGEV++'s 48ms on desktop 3090)
- **KITTI D1-all within 15% of IGEV++** (target ~1.6-1.8%)
- **Strong zero-shot generalization** via foundation distillation

This represents the **convergence of the iterative lineage with the foundation-model era** — inheriting structural efficiency from IGEV++ while adding distilled monocular priors from DEFOM-Stereo.
