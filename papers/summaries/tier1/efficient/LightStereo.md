# LightStereo: Channel Boost Is All You Need for Efficient 2D Cost Aggregation

**Authors:** Xianda Guo et al.
**Venue:** ICRA 2025
**Priority:** 8/10
**Code:** https://github.com/XiandaGuo/OpenStereo

---

## Core Problem & Motivation

All prior competitive lightweight stereo methods face a dilemma:

- **4D cost volume + 3D CNN aggregation** — accurate but slow (>100ms) and memory-hungry (GCNet, PSMNet, GA-Net)
- **3D cost volume + 2D CNN aggregation** — fast but significantly less accurate. MobileStereoNet-2D achieves only EPE = 1.11 on SceneFlow, a large accuracy gap vs 3D methods.

**LightStereo's question:** Can the accuracy of 2D cost aggregation be fundamentally improved without reverting to expensive 3D convolutions?

**The answer:** Yes — by focusing computation on the **disparity channel dimension** of the 3D cost volume, not on spatial H×W expansion. This is the "channel boost" insight.

---

## Architecture

Four components: multi-scale feature extraction → cost volume → 2D aggregation with channel boost → disparity regression.

### Component 1: Multi-Scale Feature Extraction
- **MobileNetV2** backbone (ImageNet pretrained) extracts features at 1/4, 1/8, 1/16, 1/32 scales
- U-Net-style upsampling blocks restore all features to 1/4 scale
- Separately, the **MSCA module** processes left image features at 1/4, 1/8, 1/16 for attention guidance

### Component 2: Cost Volume Computation

Standard correlation volume:

$$C_{cor}(d, h, w) = \frac{1}{C} \sum_{c=1}^{C} f_{l,4}(c, h, w) \cdot f_{r,4}(c, h, w-d)$$

- **$d$** = disparity level, integer in $[0, D-1]$
- **$h, w$** = spatial position at 1/4 resolution
- **$C$** = number of feature channels
- **$f_{l,4}(c, h, w)$** = left feature vector component at channel $c$, position $(h, w)$
- **$f_{r,4}(c, h, w-d)$** = right feature at shifted position (shifted left by $d$ pixels)
- **Mean over $c$** converts per-channel products to a single scalar matching cost
- Output shape: 3D volume $(D/4) \times (H/4) \times (W/4)$ — each entry is a **single scalar** similarity score

**Key reframing:** The 3D cost volume has $D/4$ as a **channel** dimension and $(H/4) \times (W/4)$ as spatial. LightStereo treats this like a 2D image with $D/4$ channels, allowing standard 2D convolutions and depthwise-separable blocks to process it.

### Component 3: 2D Cost Aggregation (The Channel Boost)

The 3D cost volume is processed by an encoder-decoder using **inverted residual blocks (MobileNetV2-style V2 blocks)** at three scales (1/4, 1/8, 1/16). MSCA attention is applied at each scale.

**The V2 block — THE core innovation:**

**Step 1 — Expand (channel boost):**
$$y = \sigma(W_{expand} * C)$$

- **$C$** = input cost volume, shape $(D_s, H_s, W_s)$ where $D_s$ is disparity channels
- **$W_{expand}$** = $1 \times 1$ convolution expanding disparity channels from $D_s$ to $t \times D_s$
- **$t$** = expansion factor (typically 4-8)
- **$\sigma$** = ReLU6 activation
- **$y$** has shape $(t \cdot D_s, H_s, W_s)$ — **temporarily inflated disparity dimension**

**Step 2 — Depthwise spatial processing:**
$$z = \sigma(W_{depthwise} * y)$$

- **$W_{depthwise}$** = $3 \times 3$ depthwise convolution (one filter per channel, no cross-channel mixing)
- Applied independently to each of the $t \cdot D_s$ channels over the spatial $H_s \times W_s$ plane
- Processes the **inflated** representation where the richer disparity distribution lives

**Step 3 — Project back:**
$$\text{out} = W_{project} * z$$

- **$W_{project}$** = $1 \times 1$ convolution (no activation — linear projection, following MobileNetV2 convention)
- Reduces channels back from $t \cdot D_s$ to $D_s$

**Step 4 — Skip connection** (when input and output dimensions match):
$$\text{out} = \text{out} + x$$

### Why Channel Boost Works

**Prior 2D aggregation methods** (MobileStereoNet-2D) applied depthwise spatial convolutions directly on $D$ channels without expanding. This limits how richly the cost distribution can be represented at any point in the network.

**The expansion phase in the V2 block creates the "channel boost":** temporarily inflating the disparity dimension to $t \cdot D$ allows the subsequent $3 \times 3$ depthwise convolution to operate on a **richer, higher-dimensional representation** of the cost distribution before projecting back.

**Fundamental insight:** this is different from increasing **spatial** kernel size. The paper shows that spatial expansion (larger $k \times k$ kernels) is **counterproductive** — accuracy drops. Disparity-dimension expansion is what matters.

### Component 4: Multi-Scale Convolutional Attention (MSCA)

A lightweight attention module producing spatial maps to guide cost aggregation:

- Extracts features from left image at 1/4, 1/8, 1/16 scales
- Applies **depthwise separable strip convolutions** with kernel sizes $1 \times 1$, $7 \times 1$, $1 \times 7$, $11 \times 1$, $1 \times 11$, $21 \times 1$, $1 \times 21$ **in parallel**
- Strip decomposition (e.g., $7 \times 1 + 1 \times 7$ replacing $7 \times 7$) is computationally lightweight
- Concatenates outputs, mixes with $1 \times 1$ channel-mixing conv
- **Multiplied element-wise with the aggregated cost volume** → semantic image guidance (object-level context, disparity discontinuities)

MSCA provides image-guided stopping at depth boundaries — halts cost propagation where image edges indicate object boundaries.

### Component 5: Disparity Regression (standard soft-argmax)

$$\hat{d} = \sum_{d=0}^{D_{max}} d \cdot \sigma(c_d)$$

- **$c_d$** = predicted cost at disparity level $d$
- **$\sigma(\cdot)$** = softmax over the disparity dimension → probability distribution
- **$\hat{d}$** = expected value of disparity

### Training Loss (Smooth L1)

$$\mathcal{L}(d, \hat{d}) = \frac{1}{N} \sum_{i=1}^{N} \text{smoothL1}(d_i - \hat{d}_i)$$

### Model Variants

| Variant | Blocks per scale | Expansion $t$ | Backbone |
|---------|-----------------|---------------|----------|
| **LightStereo-S** | (1, 2, 4) | 4 | MobileNetV2 |
| **LightStereo-M** | (4, 8, 14) | 4 | MobileNetV2 |
| **LightStereo-L** | (8, 16, 32) | 8 | MobileNetV2 |
| **LightStereo-H** | (8, 16, 32) | 8 | EfficientNetV2 |

---

## Benchmark Results

### SceneFlow (EPE, RTX 3090)

| Method | FLOPs (G) | Params (M) | EPE | Time (ms) |
|--------|-----------|------------|-----|-----------|
| StereoNet | 85.93 | 0.40 | 1.10 | 20 |
| 2D-MobileStereoNet | 128.84 | 2.23 | 1.11 | 73 |
| CoEx | 53.39 | 2.72 | 0.67 | 36 |
| Fast-ACVNet+ | 93.08 | 3.20 | 0.59 | 27 |
| HITNet | 50.23 | 0.42 | 0.55 | 36 |
| IINet | 90.16 | 19.54 | 0.54 | 26 |
| **LightStereo-S** | **22.71** | 3.44 | 0.73 | **17** |
| LightStereo-M | 36.36 | 7.64 | 0.62 | 23 |
| LightStereo-L | 91.85 | 24.29 | 0.59 | 37 |
| **LightStereo-H** | 159.26 | 45.63 | **0.51** | 54 |

**LightStereo-S is the fastest (17ms) and lowest FLOPs (22.71G) among methods with competitive accuracy.** LightStereo-H achieves the best EPE (0.51) overall.

### KITTI 2015

| Method | D1-all (%) | Time (ms) |
|--------|-----------|-----------|
| Fast-ACVNet+ | 2.01 | 45 |
| HITNet | 1.98 | 54 |
| LightStereo-M | 2.04 | 23 |
| LightStereo-L | 1.93 | 34 |
| **LightStereo-H** | **1.82** | 49 |

**LightStereo-H ranks 1st on KITTI 2015 among real-time methods** (under 100ms). LightStereo-S at 17ms beats StereoNet at 20ms with better accuracy.

### Cross-Domain Generalization (SceneFlow → real)

| Method | KITTI12 D1 | KITTI15 D1 | Middlebury 2px |
|--------|-----------|-----------|----------------|
| IINet | 11.6 | 8.5 | 19.57 |
| LightStereo-S | 11.6 | 9.0 | 19.63 |
| **LightStereo-M** | **7.0** | **6.6** | 17.69 |
| LightStereo-H | 7.2 | 7.3 | **14.27** |

LightStereo-M shows strong generalization — **38% reduction in D1 error** vs IINet on KITTI12 zero-shot.

### Runtime Breakdown (LightStereo-S, 17.83ms total)

| Stage | Time |
|-------|------|
| Feature extraction (MobileNetV2) | **10.39ms (58%)** |
| Cost volume | 1.98ms |
| Cost aggregation (channel boost) | 3.98ms |
| Disparity regression | 1.48ms |

**Feature extraction dominates runtime** — not the aggregation network. This is a critical finding: further optimizing the aggregation head has diminishing returns; backbone efficiency is the real bottleneck.

---

## Ablation Highlights

### Conv block selection

| Block type | EPE | Time |
|-----------|-----|------|
| Regular $3 \times 3$ conv | 0.7652 | 16.70ms |
| V1 (depthwise no expansion) | 0.7801 | 54.21ms (needs more blocks) |
| **V2 (inverted residual, channel boost)** | **0.7144** | **22.93ms** |
| ViT block (EfficientViT) | 0.7149 | 51.14ms |

**V2 uniquely achieves best accuracy AND competitive speed.** ViT matches V2 on accuracy but is 2× slower.

### Larger spatial kernels hurt

The paper shows $5 \times 5$, $7 \times 7$, $11 \times 11$ regular convolutions all give **worse EPE with higher FLOPs**. Spatial expansion is the wrong axis. **Disparity-dimension expansion** (channel boost) is what works.

### Expansion factor $t$ sensitivity

| $t$ | EPE | FLOPs |
|-----|-----|-------|
| 2 | 0.7557 | 26G |
| **4** | **0.7144** | **36G** |
| 8 | 0.6779 | 57G |
| 16 | 0.6650 | 93G |

**Diminishing returns after $t = 4$.** The paper uses $t = 4$ for S/M and $t = 8$ for L/H.

### MSCA contribution

Adding MSCA drops EPE from 0.7144 → **0.6809** with minimal cost (+0.54G FLOPs, +0.21ms). Most impactful accuracy gain at lowest cost.

---

## Strengths & Limitations

**Strengths:**
- **Lowest compute budget** at competitive accuracy among lightweight methods (22.71G FLOPs, 17ms)
- **Principled insight** — disparity-dimension expansion, not spatial expansion
- **MSCA** provides image guidance at near-zero cost
- **Strong generalization** despite lightweight design (LightStereo-M 6.6% D1 on KITTI15 zero-shot)
- **Four model variants** for different compute budgets
- **KITTI 2015 rank 1** among real-time methods

**Limitations:**
- **Accuracy still trails SOTA non-real-time methods** significantly (HITNet 1.98 vs IGEV 1.38)
- **Correlation cost volume is standard** — learned or group-wise might improve, not explored
- **No iterative refinement** — produces single estimate, which limits accuracy on thin structures
- **LightStereo-H is not actually lightweight** (54ms, 45.63M params) — EfficientNetV2 backbone undoes efficiency gains
- **Feature extraction dominates runtime** (10-27ms) — backbone is the bottleneck, not aggregation
- **No TensorRT benchmarks** — inference times are on RTX 3090, not edge hardware
- **Zero-shot generalization is modest** — much worse than iterative methods like Pip-Stereo (4.35 vs 13.08 DrivingStereo D1-all)

---

## Relevance to Our Edge Model

### Directly adoptable

1. **Channel boost (V2 block) as cost aggregation** — if our edge model uses 2D aggregation (for NPU compatibility), LightStereo's inverted residual with disparity-dimension expansion is the right block. Directly replaces the 3D CNN regularization stage.

2. **MSCA for image guidance** — the strip convolution attention module provides semantic guidance at near-zero cost. Drop-in addition to any aggregation stage.

3. **Backbone is the bottleneck, not aggregation** — the runtime breakdown (58% feature extraction) is a critical finding. For our edge model, prioritize backbone compression (MobileNetV4 / EfficientViT distillation) over further optimizing the aggregation head.

4. **Expansion factor $t = 4$** — diminishing returns beyond this. Use as default.

5. **Spatial kernel expansion is counterproductive** — stick with $3 \times 3$ depthwise convolutions. Don't use $5 \times 5$ or $7 \times 7$.

### Cautions

- **LightStereo has no iterative refinement** — Pip-Stereo demonstrates this catastrophically fails cross-domain generalization (non-iterative 13-93 D1-all vs iterative ~4 D1-all on DrivingStereo). **Our edge model MUST keep iteration.**

- **LightStereo's accuracy is behind iterative methods.** For our model, use LightStereo's efficient aggregation as **one of multiple components**, not as the whole architecture.

### Proposed Integration

For our edge model, use LightStereo's **channel-boosted V2 aggregation** as a replacement for IGEV-Stereo's 3D UNet regularization:

```
Correlation Volume (3D, scalar per disparity)
    ↓
[Channel-boost V2 blocks]  ← LightStereo
(at 1/4, 1/8, 1/16 with MSCA guidance)
    ↓
[GEV extraction via soft argmin]  ← IGEV-Stereo
    ↓
[SRU iterations with PIP compression]  ← Selective-Stereo + Pip-Stereo
```

This combines LightStereo's efficient 2D aggregation with the iterative refinement (which cannot be removed per Pip-Stereo's findings).

---

## Connections to Other Papers

| Paper | Relationship |
|-------|-------------|
| **MobileStereoNet-2D** | Direct competitor — LightStereo fixes its accuracy gap via channel boost |
| **StereoNet** | Earlier lightweight method — LightStereo-S beats it on speed AND accuracy |
| **CoEx** | Uses similar guided cost volume excitation (inspired MSCA) |
| **Fast-ACVNet+** | Competitor in the real-time category |
| **HITNet** | Competing tile-based efficient method |
| **IINet** | Implicit 2D network — competitor with good accuracy but slower |
| **BGNet** | Complementary — BGNet optimizes upsampling, LightStereo optimizes aggregation |
| **MobileNetV2** | Source of the inverted residual block concept |
| **Pip-Stereo** | Shows that non-iterative methods (LightStereo) cannot generalize cross-domain — critical complement |
