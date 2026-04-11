# CREStereo: Practical Stereo Matching via Cascaded Recurrent Network with Adaptive Correlation

**Authors:** Jiankun Li, Peisen Wang, Pengfei Xiong, Tao Cai, Ziwei Yan, Lei Yang, Jiangyu Liu, Haoqiang Fan, Shuaicheng Liu (Megvii / Tencent / UESTC)
**Venue:** CVPR 2022
**Priority:** 9/10
**Code:** https://github.com/megvii-research/CREStereo

---

## Core Problem & Motivation

RAFT-Stereo proved that iterative recurrent refinement works, but it has four practical weaknesses that matter for real deployment:

1. **Fine structure loss at high resolution** — RAFT-Stereo operates at a single resolution during iteration. Consumer photos need thin-structure accuracy (wires, nets, hair, whiskers) where errors are perceptually devastating (e.g., bokeh rendering fails).
2. **Assumes perfect rectification** — Real smartphone stereo pairs often have vertical misalignment due to mismatched lenses (wide + telephoto), sensor orientation, and distortion. RAFT-Stereo's strict 1D horizontal correlation fails here.
3. **All-pairs memory cost** — The $H \times W \times W$ correlation volume is quadratic in width. Prohibitive at megapixel resolution.
4. **Training data mismatch** — Standard synthetic datasets lack thin structures, repetitive textures, and transparent/reflective surfaces.

CREStereo addresses all four with a **cascaded coarse-to-fine architecture using an Adaptive Group Correlation Layer (AGCL)**.

---

## Architecture

![Figure 2: CREStereo training architecture (left) and stacked cascaded architecture for inference (right). A feature pyramid is built at 1/16, 1/8, 1/4 scales. Three cascade stages run sequentially, each applying the Recurrent Update Module (RUM) at that scale. Shared weights across all three stages. For high-resolution inference, a grid of RUMs is stacked across multiple image pyramid levels.](../../../figures/CREStereo_fig2_cascade_architecture.png)

### Three Main Components

**1. Feature Extraction:** Shared-weight CNN producing a 3-level feature pyramid at 1/16, 1/8, 1/4 of input resolution. At the first cascade stage (1/16), **LoFTR-style self-attention and cross-attention** with linear attention bootstrap global context.

**2. Cascaded Recurrent Refinement:** Three cascade stages at 1/16, 1/8, 1/4 scales. Disparity initializes at zero for Stage 1, then each subsequent stage uses the upsampled output of the previous stage as initialization. All three stages share the **same RUM weights**.

**3. Stacked Cascades for Inference:** For high-resolution inference (e.g., 1536×2048 Middlebury), multiple image pyramid levels feed into a grid of RUMs — same trained weights apply at all resolutions without retraining.

### Adaptive Group Correlation Layer (AGCL) — The Core Innovation

![Figure 3: Left - the Recurrent Update Module (RUM) iterates AGCL + GRU. Right - AGCL construction: right features pass through cross-attention, grouping, sampling at the current disparity estimate, deformable offset addition, and grouped correlation with left features.](../../../figures/CREStereo_fig3_rum_agcl.png)

AGCL has four interlocking mechanisms:

#### Mechanism A: Local correlation (not all-pairs)

Standard local correlation:

$$\text{Corr}(x, y, d) = \frac{1}{C} \sum_{i=1}^{C} F_1(i, x, y) \cdot F_2(i, x', y') \quad \text{(1)}$$

Every element:
- **$(x, y)$** = pixel position in the left feature map
- **$d$** = index into $D$ candidate displacements (usually $D = 2r + 1 = 9$ with $r=4$)
- **$C$** = total number of feature channels (the sum averages via $1/C$)
- **$F_1(i, x, y)$** = channel $i$ of left feature at $(x, y)$
- **$F_2(i, x', y')$** = channel $i$ of right feature at displaced position
- **$x' = x + f(d)$** = horizontally shifted position with $f(d) \in [-r, r]$
- **$y' = y + g(d)$** = vertically shifted position; $g(d) = 0$ in 1D mode
- **$\text{Corr}(x, y, d)$** = the output local correlation volume — shape $H \times W \times D$ (only $D = 9$ candidates, not $W$)

**Critical efficiency gain:** The local volume is $H \times W \times D$ where $D \ll W$ (e.g., 9 vs 512). This is a ~50× memory reduction compared to RAFT-Stereo's all-pairs volume at high resolution.

#### Mechanism B: 2D-1D Alternate Search

![Figure 4: Adaptive local correlation search. Top: 2D mode with 3×3 grid and deformable offsets (arrows show learned shifts from regular grid positions). Bottom: 1D mode with 9 horizontal positions. Both modes produce the same D=9 correlation values so they share downstream GRU weights.](../../../figures/CREStereo_fig4_search_window.png)

- **1D mode:** $g(d) = 0$, $f(d) \in \{-4, ..., 4\}$ — 9 candidates along the horizontal epipolar line. Use when rectification is good.
- **2D mode:** $k \times k$ grid where $k = \sqrt{2r+1} \approx 3$ — 9 candidates in a 3×3 neighborhood. Handles rectification error.

The two modes **alternate across iterations** in the RUM. Both produce $D = 9$ correlation values, so the downstream GRU can share weights between modes. This is the fix for the "practical stereo" problem (real smartphone cameras have imperfect rectification).

#### Mechanism C: Deformable Search Window

Extending deformable convolutions to correlation:

$$\text{Corr}(x, y, d) = \frac{1}{C} \sum_{i=1}^{C} F_1(i, x, y) \cdot F_2(i, x'', y'') \quad \text{(2)}$$

Every element:
- **$x'' = x + f(d) + dx$** = candidate position with learned horizontal offset $dx$
- **$y'' = y + g(d) + dy$** = candidate position with learned vertical offset $dy$
- **$dx, dy$** = **learned per-pixel, per-candidate offsets**. The network predicts offsets $o \in \mathbb{R}^{2 \times (2r+1) \times h \times w}$ from the left features and current disparity — two offset components (dx, dy) for each of 9 candidate positions, at each spatial location.

**Why this helps:** In occluded or textureless regions, a fixed search window finds equally plausible but wrong candidates. The deformable window allows the network to **reshape** the search pattern based on content, focusing candidates on informative positions.

#### Mechanism D: Group-wise Correlation

Inspired by GwcNet, features are split into $G$ groups along the channel dimension. Each group computes its own local correlation sub-volume (via Eq. 1 or 2), and all $G$ sub-volumes are concatenated:

- Input feature channels: $C$
- Grouped features: $C/G$ channels per group
- Per-group correlation: $H \times W \times D$
- Concatenated output: $H \times W \times (G \cdot D)$

Different feature subspaces capture different matching cues, yielding a richer representation than a single dot product over all channels.

### Loss Function

$$\mathcal{L} = \sum_s \sum_{i=1}^{n} \gamma^{n-i} \Vert d_{gt} - \mu_s(f_i^s)\Vert _1 \quad \text{(3)}$$

Every element:
- **$s$** = cascade stage index (1/16, 1/8, 1/4)
- **$i$** = iteration index within the RUM at stage $s$; $n$ total iterations
- **$\gamma = 0.9$** = exponential decay — later iterations weighted more
- **$f_i^s$** = intermediate disparity prediction at stage $s$, iteration $i$
- **$\mu_s$** = upsampling operator rescaling to full resolution
- **$d_{gt}$** = ground-truth disparity
- Supervises every intermediate prediction at every scale

---

## Key Innovations vs RAFT-Stereo

| Aspect | RAFT-Stereo | CREStereo |
|--------|-------------|-----------|
| Cost volume | All-pairs 1D: $H \times W \times W$ | Local-window: $H \times W \times 9$ |
| Search direction | Strictly horizontal | 2D-1D alternating |
| Search pattern | Fixed | Deformable (content-adaptive) |
| Feature grouping | Single dot product | $G$ groups concatenated |
| Global context | None | Self + cross-attention at first stage |
| Multi-scale | Single-scale iteration | 3-level cascade with shared weights |
| High-res inference | OOM on Middlebury | Stacked cascade grid, any resolution |
| Training data | Standard synthetic | Custom hard-case dataset |

---

## Benchmark Results

### Middlebury v3

| Method | Bad 2.0 | Bad 1.0 | AvgErr | A95 |
|--------|---------|---------|--------|-----|
| LEAStereo | 7.15 | 20.84 | 1.43 | 2.65 |
| HITNet | 6.46 | 13.3 | 1.71 | 4.26 |
| RAFT-Stereo | 4.74 | 9.37 | 1.27 | 2.29 |
| **CREStereo** | **3.71** | **8.25** | **1.15** | **1.58** |

Ranked **1st** on Middlebury at publication — surpassing RAFT-Stereo by 21.7% on Bad 2.0.

### ETH3D

| Method | Bad 1.0 | Bad 0.5 | AvgErr |
|--------|---------|---------|--------|
| RAFT-Stereo | 2.44 | 7.04 | 0.18 |
| **CREStereo** | **0.98** | **3.58** | **0.13** |

Ranked **1st** on ETH3D — 59.8% improvement over RAFT-Stereo on Bad 1.0.

### Smartphone Evaluation (Holopix50K)

| Method | mxIoU | mxIoUbd |
|--------|-------|---------|
| HSMNet | 91.70% | 60.17% |
| RAFT-Stereo | 94.58% | 69.26% |
| **CREStereo** | **97.50%** | **72.61%** |

Best practical performance on consumer smartphone photos.

---

## Ablation Highlights

### Local vs All-Pairs Correlation (the surprising finding)

| Configuration | Middlebury Bad2.0 | ETH3D Bad1.0 |
|--------------|------------------|--------------|
| 1D all-pairs (RAFT-Stereo) | 44.41 | 6.03 |
| 2D all-pairs | 47.38 | 6.17 |
| 1D local only | 19.87 | 3.13 |
| 2D local only | 20.70 | 3.33 |
| 1D+2D alternating | 19.23 | 3.05 |
| 3-level cascade | **12.67** | **2.01** |

**Local correlation is dramatically better than all-pairs** (~2.3× lower Bad 2.0 on Middlebury). Counter-intuitive but validated: local windows + iterative resampling act as a propagation mechanism that outperforms brute-force all-pairs search.

### AGCL Component Contribution

| Configuration | Middlebury Bad2.0 | ETH3D Bad1.0 |
|--------------|------------------|--------------|
| Without deformable, group-wise, or attention | 6.86 | 1.26 |
| + group-wise | 6.84 | 1.22 |
| + deformable | 6.82 | 1.20 |
| + attention | 6.49 | 1.22 |
| **Full AGCL** | **6.46** | **1.03** |

All three AGCL mechanisms contribute, with attention providing the largest single gain on Middlebury.

---

## Model Size & Speed

**The paper does not report parameter counts or inference latency** — an unusual omission. The Conclusion explicitly states: *"A limitation of our method is that the model is not yet efficient enough to run in current mobile applications."*

Training: 8× GTX 2080Ti, batch size 16, 300K iterations at 384×512.

---

## Strengths & Limitations

**Strengths:**
- SOTA on both Middlebury and ETH3D simultaneously
- Robust to real-world rectification errors (2D mode)
- Memory-efficient local correlation (10-50× reduction over all-pairs)
- Single weight set works across all resolutions via stacked cascades
- Fine-structure recovery for practical photography

**Limitations:**
- No reported inference speed
- Explicitly admits mobile-incompatibility
- KITTI not a primary target (competitive but not rank-1)
- Multi-stage cascade accumulates compute
- Deformable offset sub-network adds per-iteration overhead

---

## Relevance to Our Edge Model

**Directly adoptable:**

1. **Local correlation window (Eq. 1)** — 9 candidates instead of all-pairs is a pure win. No reason to use all-pairs on edge.
2. **2D-1D alternating search** — near-zero cost for robustness to real-world rectification errors on device-mounted cameras. Critical for mobile/robot deployment.
3. **Group-wise correlation** — negligible compute, better matching quality.
4. **3-level cascade → 2-level for edge** — reduce cascade depth and iterations per level for latency budget.
5. **Custom synthetic training data** — hard-case coverage (thin structures, repetitive textures, transparent/reflective) matters more than architecture choices for real-world quality.

**Replace/simplify for edge:**

- **Cross-attention at first stage** → replace with lightweight depthwise separable block or omit (ablation shows modest contribution)
- **Deformable offsets** → replace with fixed dilation pattern if the sub-network cost is prohibitive
- **GRU update block** → use distilled compact variant with fewer channels

**Key takeaway for our edge model:** **Do NOT inherit RAFT-Stereo's all-pairs correlation.** CREStereo proves local correlation + cascade is strictly better and dramatically cheaper. AGCL (possibly simplified) is the correct correlation module to build on.

---

## Connections to Other Papers

| Paper | Relationship |
|-------|-------------|
| **RAFT-Stereo** | Direct baseline — CREStereo addresses its practical limitations |
| **GwcNet** | Group-wise correlation concept borrowed from here |
| **LoFTR** | Self/cross-attention with positional encoding at first cascade stage |
| **DCN v2** | Deformable correlation extends deformable convolutions |
| **IGEV-Stereo** | Contemporary work — hybrid 3D aggregation + iterative |
| **DEFOM-Stereo** | Compatible: CREStereo's cascade can be replaced by monocular depth initialization |
