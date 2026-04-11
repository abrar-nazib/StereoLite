# IGEV++: Iterative Multi-range Geometry Encoding Volumes for Stereo Matching

**Authors:** Gangwei Xu, Xianqi Wang, Zhaoxing Zhang, Junda Cheng, Chunyuan Liao, Xin Yang (HUST)
**Venue:** TPAMI 2025
**Priority:** 9/10
**Code:** https://github.com/gangweiX/IGEV-plusplus

---

## Core Problem & Motivation

IGEV-Stereo v1 (CVPR 2023) built a **single-range** Geometry Encoding Volume (GEV) with a fixed maximum disparity of 192px. This has three fundamental limitations:

### Limitation 1: Large disparity failure

A fixed-range cost volume cannot effectively handle disparities beyond its pre-defined maximum. Naively increasing the range causes memory/compute explosion. PCWNet's EPE **degrades by 94.87%** (from 0.78 to 1.52) when extending from Disp<192 to Disp<768. IGEV v1 simply cannot compete at large disparities.

### Limitation 2: Ill-posed region ambiguity

Even IGEV v1 combines GEV with all-pairs correlations (APC), but APC is local and context-free. In textureless, occluded, or reflective regions, local correlations collapse into ambiguity.

### Limitation 3: Single-granularity cost representation

A single-scale GEV must choose: fine resolution (good for small disparities) OR wide range (good for large disparities). Never both. Fine-textured details at small disparity and coarse-textured large-disparity regions conflict in a single volume.

**IGEV++'s insight:** Different disparity magnitudes require specialized cost representations. Build **three separate geometry volumes** (small, medium, large range) and fuse them adaptively based on local content.

---

## Architecture

![Figure 3: IGEV++ architecture. The Feature Network extracts multi-scale features. Multi-range Geometry Encoding Volumes (MGEV) are built at three disparity ranges (192, 384, 768px) with Adaptive Patch Matching (APM) for the medium and large ranges. Each volume is regularized by a 3D CNN. Selective Geometry Feature Fusion (SGFF) adaptively combines features from all three volumes based on spatial content at each GRU iteration.](../../../figures/IGEVpp_fig3_architecture.png)

### Five Main Components

#### A. Feature Network
- **MobileNetV2** backbone (same as IGEV v1)
- Features at 1/4, 1/8, 1/16, 1/32 resolution
- Separate Context Network produces multi-scale context features (128 channels)

#### B. Multi-range Geometry Encoding Volumes (MGEV) — The Core Innovation

Three cost volumes for three disparity ranges:

**Small-range volume $C^s$ (D^s < 192px)** — standard group-wise correlation:

$$C^s(g, d^s, x, y) = \frac{1}{N_c/N_g} \langle f^g_{l,4}(x,y), f^g_{r,4}(x - d^s, y) \rangle \quad \text{(1)}$$

Variables:
- **$g \in \{0, ..., N_g-1\}$** = group index, $N_g = 8$
- **$d^s \in D^s = \{0, 1, ..., D^s/4 - 1\}$** = disparity candidates (192/4 = 48 at 1/4 resolution)
- **$(x, y)$** = spatial coordinates at 1/4 resolution
- **$N_c / N_g$** = channels per group (normalizer)
- **$f^g_{l,4}, f^g_{r,4}$** = $g$-th channel group of left/right features at 1/4 resolution
- **$\langle \cdot, \cdot \rangle$** = inner product over $N_c/N_g$ channels

**Large-range volume $C^l$ (D^l < 768px)** — uses **Adaptive Patch Matching (APM)**:

$$C^l(g, d^l, x, y) = \frac{1}{N_c/N_g} \langle f^g_{l,4}(x,y), p^g(x - d^l, y) \rangle \quad \text{(2)}$$

$$p^g(x - d^l, y) = \sum_{i=0}^{3} \omega_i f^g_{r,4}(x - (d^l + i), y) \quad \text{(2b)}$$

Variables:
- **$d^l \in D^l = \{0, 4, 8, ..., D^l/4 - 4\}$** = **sparse** candidates at stride 4 — only 48 total (768/4/4 = 48), same count as small-range!
- **$p^g$** = adaptive patch representation of right features at position $(x - d^l, y)$
- **$\omega_i$** = **learned adaptive weights** for the 4 neighboring pixels, $i \in \{0, 1, 2, 3\}$
- The 4-pixel weighted patch compensates for the stride-4 sparse sampling

**Key insight:** Stride-4 sampling covers 768px with only 48 candidates, keeping memory tractable. The learned patch weights make each sparse candidate represent a 4-pixel-wide region accurately. Without APM, stride-4 sampling would miss many correct matches; with APM, the effective receptive field along disparity is recovered.

**Medium-range volume $C^m$ (D^m < 384px)** — similar construction with stride-2 APM, 48 candidates covering 384px.

**3D Regularization (applied to each volume):**

$$G^l = R(C^l) \quad \text{(3)}$$

- **$R$** = lightweight 3D UNet: 3 downsampling + 3 upsampling blocks
- Each down block: two 3×3×3 3D convs (channel counts 16, 32, 48)
- Each up block: one 4×4×4 3D transposed conv + two 3×3×3 3D convs

**Cost volume excitation (CoEx-style):**

$$C_i' = \sigma(f_{i,l}) \odot C_i \quad \text{(4)}$$

- **$\sigma$** = sigmoid activation
- **$f_{i,l}$** = left features at scale $i$, used as channel-wise attention
- **$\odot$** = element-wise product
- Applied at multiple scales $i = 4, 8, 16, 32$ during 3D regularization

#### C. Initial Disparity Maps (Warm Start)

Three initial disparity maps via soft argmin:

$$d_0^s = \sum_{d^s \in D^s} d^s \cdot \text{Softmax}(G^s(d^s)) \quad \text{(5)}$$

Similarly for $d_0^m$ and $d_0^l$. The small-range estimate $d_0^s$ is used to **initialize all GRU iterations** (it's most accurate for the dominant near-range of typical scenes).

#### D. Selective Geometry Feature Fusion (SGFF)

Rather than naive concatenation, SGFF uses **learned spatially adaptive fusion**:

$$f_d = \text{Conv}(\text{Concat}(d_k^s, d_k^m, d_k^l)) \quad \text{(6a)}$$

$$s_j = \sigma(\text{Conv}(\text{Concat}(f_{l,4}, f_d))), \quad j \in \{s, m, l\} \quad \text{(6b)}$$

Variables:
- **$d_k^s, d_k^m, d_k^l$** = current disparity maps from each range at iteration $k$
- **$f_d$** = fused disparity feature
- **$s_j$** = spatial weight map for range $j$, learned from **both image content and current disparity state**

Then the fused geometry feature:

$$f_G = s_s \odot f_G^s + s_m \odot f_G^m + s_l \odot f_G^l \quad \text{(7)}$$

Variables:
- **$f_G^s, f_G^m, f_G^l$** = indexed geometry features from $G^s, G^m, G^l$ at the current disparity estimate
- **$s_s, s_m, s_l$** = spatial weight maps from Eq. 6b

**Per-pixel selection:** Flat/large-disparity regions lean on coarse-grained $G^l$; textured/small-disparity regions lean on fine-grained $G^s$. The SGFF weights are computed fresh at each iteration based on the evolving disparity estimate.

#### E. ConvGRU Update Operator

Standard RAFT-Stereo ConvGRU (same 3-level multi-scale structure as IGEV-Stereo):

$$x_k = [\text{Encoder}_g(f_G), \text{Encoder}_d(d_{k-1}), d_{k-1}] \quad \text{(8a)}$$

$$z_k = \sigma(\text{Conv}([h_{k-1}, x_k], W_z) + c_z) \quad \text{(8b)}$$

$$r_k = \sigma(\text{Conv}([h_{k-1}, x_k], W_r) + c_r) \quad \text{(8c)}$$

$$\tilde{h}_k = \tanh(\text{Conv}([r_k \odot h_{k-1}, x_k], W_h) + c_h) \quad \text{(8d)}$$

$$h_k = (1 - z_k) \odot h_{k-1} + z_k \odot \tilde{h}_k \quad \text{(8e)}$$

Variables: same as IGEV-Stereo (context features $c_z, c_r, c_h$ additively injected into gates). Disparity update:

$$d_k = d_{k-1} + \Delta d_k \quad \text{(9)}$$

### Loss Function (three components)

**Regularization loss** on each range's initial disparity:

$$\mathcal{L}_{reg}^s = \text{Smooth}_{L1}(d_0^s - d_{gt}), \quad \mathcal{L}_{reg}^m = ..., \quad \mathcal{L}_{reg}^l = ... \quad \text{(10a)}$$

$$\mathcal{L}_{reg} = \lambda_s \mathcal{L}_{reg}^s + \lambda_m \mathcal{L}_{reg}^m + \lambda_l \mathcal{L}_{reg}^l \quad \text{(10b)}$$

- **Weights:** $\lambda_s = 1.0, \lambda_m = 0.5, \lambda_l = 0.2$ — small-range weighted most (most common scene disparities)

**Iteration loss** with exponential decay:

$$\mathcal{L}_{iter} = \sum_{i=1}^{N} \gamma^{N-i} \Vert d_i - d_{gt}\Vert _1 \quad \text{(11)}$$

**Total:**

$$\mathcal{L}_{total} = \mathcal{L}_{reg} + \mathcal{L}_{iter} \quad \text{(12)}$$

---

## Key Innovations vs IGEV-Stereo v1 and RAFT-Stereo

### vs IGEV-Stereo v1

| Aspect | IGEV v1 | IGEV++ |
|--------|---------|--------|
| Disparity range | Fixed D < 192px | Three ranges: 192, 384, 768px |
| Cost volume | Single GEV (dense) | Multi-range MGEV (3 volumes) |
| Large-disp handling | Fails above 192px | Dedicated large-range volume |
| Cost construction | Group-wise correlation | Group-wise (small) + APM (medium/large) |
| Feature fusion | Single scale | Selective adaptive fusion (SGFF) |
| Initial disparity | Soft argmin over single GEV | Three soft argmins; $d_0^s$ starts GRU |
| Memory (Scene Flow) | ~1.65 GB | **~1.57 GB** (despite 3 volumes!) |

Remarkably, IGEV++ uses **less memory** than IGEV v1 because the medium/large-range volumes use sparse stride-2/stride-4 sampling (48 candidates each instead of 192).

### vs RAFT-Stereo

| Aspect | RAFT-Stereo | IGEV++ |
|--------|-------------|--------|
| Cost representation | All-pairs correlation only | MGEV (geometry-informed) + APC |
| Geometry encoding | None | 3D CNN regularized at 3 scales |
| Large disparities | Degrades significantly | Robust up to 768px |
| Convergence | Slow (32 iters) | Fast (2 iters beat RAFT-32) |
| Ill-posed regions | Struggles | MGEV provides non-local context |

---

## Benchmark Results

### Scene Flow — Multi-range

| Method | Disp<192 | Disp<384 | Disp<512 | Disp<768 | Memory |
|--------|----------|----------|----------|----------|--------|
| PCWNet | 0.78 | 3.30 | 3.87 | 12.62 | 11.62 GB |
| GwcNet | 0.76 | 2.40 | 3.69 | 3.78 | 10.04 GB |
| RAFT-Stereo | 0.67 | 3.41 | 3.79 | 3.83 | 1.65 GB |
| GMStereo | 0.64 | 2.55 | 2.87 | 2.96 | 1.56 GB |
| **IGEV++** | **0.43** | **1.81** | **2.16** | **2.21** | **1.57 GB** |

**Best across all disparity ranges AND least memory** — a remarkable combination.

### KITTI 2012/2015 (22 iterations + DepthAnythingV2-Large backbone)

KITTI 2012: 2-noc = 1.36, 3-noc = 0.89, 3-all = 1.13 — **best in class**
KITTI 2015: D1-all = 1.43 — **best accuracy method**

### Middlebury v3

| Method | Bad 1.0 | Bad 2.0 | AvgErr |
|--------|---------|---------|--------|
| RAFT-Stereo | 7.04 | 3.41 | 0.18 |
| GMStereo | 5.94 | 2.83 | 0.08 |
| Selective-IGEV | 3.06 | 1.23 | 0.12 |
| **IGEV++** | **2.98** | **1.14** | **0.13** |

### Zero-Shot Middlebury (trained on Scene Flow only)

| Method | Half-res Bad 2.0 | Full-res Bad 2.0 |
|--------|-----------------|------------------|
| PSMNet | 25.1 | 39.5 |
| RAFT-Stereo | 12.6 | 18.3 |
| DLNR | 9.5 | 14.5 |
| **IGEV++** | **7.8** | **12.7** |

### RT-IGEV++ (Real-time variant)

**RT-IGEV++: 48ms on KITTI resolution** — best among all published real-time methods.

Design changes for RT variant:
- Single backbone (no separate context network)
- Single-range GEV (D < 192px, matching IGEV v1)
- Single-level ConvGRU (not 3-level)
- Hidden state reduced from 128 → 96 channels
- 6 inference iterations (vs 22 for full model)

---

## Ablation Highlights

### Component ablation on Scene Flow

| Model | Disp<192 EPE | Disp<384 EPE | Disp<768 EPE |
|-------|-------------|-------------|-------------|
| Baseline (RAFT-Stereo) | 0.54 | 0.68 | 0.87 |
| + GEV (single range) | 0.46 | 0.59 | 0.84 |
| + MGEV (no APM) | 0.46 | 0.49 | 0.71 |
| + MGEV + APM | 0.45 | 0.53 | 0.71 |
| **+ SGFF (full IGEV++)** | **0.43** | **0.56** | **0.67** |

**Key findings:**
- MGEV alone provides the biggest jump at large disparities (0.84 → 0.71)
- SGFF provides final improvements at **all ranges**, including small-disparity (0.45 → 0.43)
- The multi-range approach strictly dominates any single-range variant

### Range specialization

| Single range used | Disp<192 | Disp<384 | Disp<768 |
|-------------------|----------|----------|----------|
| Small-range only | 0.47 | 0.87 | 1.06 |
| Medium-range only | 0.51 | 0.67 | 0.82 |
| Large-range only | 0.54 | 0.69 | 0.76 |
| **MGEV (combined)** | **0.43** | **0.56** | **0.67** |

Each single range is best in its "home" region. MGEV beats all of them at every range — the combination is strictly better.

### Convergence

With **only 2 GRU iterations**, IGEV++ achieves EPE 0.83 — **better than RAFT-Stereo at 32 iterations** (0.98).

---

## Model Size & Speed

| Configuration | Memory | Runtime |
|---------------|--------|---------|
| IGEV++ (22 iters, DAv2-L backbone) | 1.57 GB | 280ms @ KITTI |
| **RT-IGEV++ (6 iters)** | — | **48ms @ KITTI** |
| IGEV++ on Scene Flow | 1.57 GB | (not reported) |

---

## Strengths & Limitations

**Strengths:**
- SOTA across all disparity ranges (small to large) simultaneously
- **Memory-efficient despite multi-range** — APM keeps it at 1.57 GB
- **Fast convergence** — 2 iterations beat 32 RAFT-Stereo iterations
- Robust to ill-posed regions (non-local context from 3D regularization)
- **Real-time variant available** (RT-IGEV++ at 48ms)
- Best zero-shot generalization on large-disparity scenes
- Handles medical/surgical scenes (SCARED dataset)

**Limitations:**
- **Three 3D CNN regularization networks** — more complex than IGEV v1 despite sparser sampling
- **Large-model inference is slow** — 280ms with DAv2-L backbone
- Depends on heavy backbone (DAv2-L) for top KITTI results
- Medium/large volumes use stride-4 sampling → 4px precision at large disparities (relies on GRU refinement for fine-tuning)
- SGFF weights computed per-iteration → accumulated latency over 22 iterations

---

## Relevance to Our Edge Model

**Directly adoptable ideas:**

1. **Multi-range philosophy** — different disparity ranges need specialized volumes. For edge, reduce from 3 ranges to 2 (small 192px + large 768px) with shallower 3D UNets per range.

2. **Adaptive Patch Matching (APM)** — directly applicable to edge. APM reduces large-range volume from 192 to 48 candidates (4× reduction) at minimal quality cost. ONNX-exportable as gather + weighted sum.

3. **SGFF as lightweight learned fusion** — two conv layers, minimal params, spatially adaptive selection. Keep for edge.

4. **Faster convergence (2 iterations viable)** — RT-IGEV++'s design already validates this direction (6 inference iterations).

5. **RT-IGEV++ is a near-drop-in edge architecture** — 48ms single backbone, single-range GEV, single-level GRU, 96-channel hidden state. This is close to what our edge model should look like.

**Adaptations for edge:**

- **Replace MobileNetV2 with MobileNetV4/EfficientViT** for better NPU efficiency
- **Separable 3D convolutions** in the 3D UNet (Separable-Stereo approach, ~8× FLOP reduction)
- **Two ranges instead of three** (small 192px + large 768px)
- **Distill monocular depth** into the initial disparity (DEFOM-Stereo approach) — eliminates need for large-range GEV entirely
- **Quantize to INT8** — the 3D convolutions are quantization-friendly (ReLU, no attention)

**Key takeaway:** **RT-IGEV++ + DEFOM-style mono initialization + separable 3D convs + MobileNetV4 backbone** is a highly promising starting architecture for our edge model. The multi-range idea may not even be needed if mono priors provide a good enough warm start.

---

## Connections to Other Papers

| Paper | Relationship |
|-------|-------------|
| **IGEV-Stereo v1** | Direct predecessor — IGEV++ extends single-range to multi-range |
| **RAFT-Stereo** | Same GRU update framework |
| **Selective-Stereo** | Contemporary — addresses different limitation (fixed receptive field) |
| **MonSter** | Uses IGEV backbone + bidirectional mono refinement |
| **PCWNet** | Compared as baseline — IGEV++ memory is 7× less |
| **DEFOM-Stereo** | Potential combination: DEFOM's mono priors replace IGEV++'s need for large-range volume |
| **Separable-Stereo** | Separable 3D convs would cut IGEV++'s 3D regularization cost |
