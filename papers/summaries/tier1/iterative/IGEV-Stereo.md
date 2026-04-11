# IGEV-Stereo: Iterative Geometry Encoding Volume for Stereo Matching

**Authors:** Gangwei Xu, Xianqi Wang, Xiaohuan Ding, Xin Yang (Huazhong University of Science and Technology)
**Venue:** CVPR 2023
**Priority:** 9/10
**Code:** https://github.com/gangweiX/IGEV

---

## Core Problem & Motivation

RAFT-Stereo and its descendants (CREStereo) use **only all-pairs correlation** as the cost representation. This is local and context-free: each entry records the dot-product similarity between a left pixel and a right pixel at a given offset, with no aggregation across spatial neighbors or the disparity dimension. This produces a noisy, locally ambiguous cost signal that:

- **Cannot distinguish true matches in textureless regions** — no spatial context propagates
- **Fails in occluded areas** — no valid right-image match exists
- **Collapses under repetitive structures** — multiple disparities appear equally plausible
- **Forces the GRU to invent context** from hidden state alone — requires ~32 iterations to converge (~440ms)
- **Starts from $d_0 = 0$** — many iterations are wasted climbing out of a degenerate initialization

Meanwhile, **cost-filtering methods** (PSMNet, GwcNet, ACVNet) with 3D convolutions have the opposite strengths and weaknesses:

| Property | Cost-filtering (3D CNN) | Iterative (RAFT) |
|----------|------------------------|------------------|
| Non-local geometry | Strong | Absent |
| Ill-posed regions | Good | Poor |
| High resolution | Poor (3D conv memory) | Good |
| Boundary detail | Can over-smooth | Good |
| Compute cost | High | Low |

**IGEV-Stereo's insight:** These approaches are **complementary, not competing**. Build a lightweight 3D CNN that's only run ONCE (not per iteration) to encode non-local geometry, then combine it with local all-pairs correlations for iterative GRU refinement. Best of both worlds at minimal cost.

![Figure 2: The over-smoothing problem of raw correlation vs the sharp-but-noisy problem of plain iteration. (a) Left image. (b) Disparity from all-pairs correlation (RAFT-Stereo) — noisy road surfaces and reflective hoods. (c) Disparity from GEV alone — globally smooth but loses fine detail at object boundaries. (d) Final IGEV-Stereo disparity — combines global structure from GEV with local detail from APC.](../../../figures/IGEV_fig2_apc_vs_gev.png)

---

## Architecture

![Figure 3: IGEV-Stereo overview. The network first builds a Geometry Encoding Volume (GEV) by encoding geometry and context information through 3D CNN, and combines it with All-Pairs Correlations (APC) to form a Combined Geometry Encoding Volume (CGEV). The initial disparity starts from GEV via soft argmin (warm start). The CGEV is then indexed at the current disparity and fed into ConvGRUs for iterative refinement.](../../../figures/IGEV_fig3_architecture.png)

### Four Main Stages

#### Stage 1: Feature Extraction (two networks)

**Feature Network:**
- **MobileNetV2** backbone (ImageNet pre-trained) — already lightweight, unlike RAFT-Stereo's ResNet
- Multi-scale features $\{f_{l,i}, f_{r,i}\}$ at $i = 4, 8, 16, 32$ (1/4, 1/8, 1/16, 1/32 resolution)
- $f_{l,4}$ and $f_{r,4}$ are used to build the cost volume
- All four scales of $f_{l,i}$ serve as guidance signals for 3D regularization

**Context Network:**
- Same design as RAFT-Stereo — residual blocks with downsampling
- Applied only to left image
- Produces context features at 1/4, 1/8, 1/16 resolution, 128 channels each
- Initializes GRU hidden states AND is injected at every iteration

#### Stage 2: Combined Geometry Encoding Volume (CGEV) — The Key Innovation

**Step 2a: Group-wise correlation volume** $C_{corr}$

$$C_{corr}(g, d, x, y) = \frac{1}{N_c/N_g} \langle f^g_{l,4}(x, y), f^g_{r,4}(x-d, y) \rangle \quad \text{(1)}$$

Every element:
- **$g$** = group index, $g \in \{1, ..., N_g\}$ with $N_g = 8$ groups
- **$d$** = disparity index at 1/4 resolution, $d \in \{0, ..., D/4 - 1\}$
- **$(x, y)$** = spatial coordinates in the left feature map
- **$N_c$** = total number of feature channels
- **$N_c / N_g$** = channels per group (used as normalizer)
- **$f^g_{l,4}(x,y)$** = the $g$-th channel group of left features at position $(x,y)$
- **$f^g_{r,4}(x-d, y)$** = the $g$-th channel group of right features shifted left by $d$ (i.e., at the candidate matching position)
- **$\langle \cdot, \cdot \rangle$** = inner product over the $N_c/N_g$ channels in that group
- Output shape: $[N_g, D, H/4, W/4]$

**Step 2b: 3D regularization → Geometry Encoding Volume**

$$C_G = R(C_{corr}) \quad \text{(2)}$$

- **$C_G$** = Geometry Encoding Volume — 3D-regularized cost volume with injected non-local context
- **$R$** = lightweight 3D UNet: 3 downsampling blocks + 3 upsampling blocks
- Each down block: two $3 \times 3 \times 3$ 3D convs; channel counts 16, 32, 48
- Each up block: one $4 \times 4 \times 4$ 3D transposed conv + two $3 \times 3 \times 3$ 3D convs
- **Only adds 0.58M parameters** — the 3D UNet is deliberately small

**Step 2c: Guided cost volume excitation (CoEx-style)**

$$C_i' = \sigma(f_{l,i}) \odot C_i \quad \text{(3)}$$

- **$C_i'$** = excited cost volume at scale $i$
- **$\sigma(\cdot)$** = sigmoid activation
- **$f_{l,i}$** = left image features at scale $i$, used as channel-wise attention weights
- **$\odot$** = element-wise product
- **$C_i$** = cost volume at scale $i$ during 3D UNet processing

The left image's appearance modulates which cost volume channels are amplified — semantic guidance for 3D regularization.

**Step 2d: Building the CGEV**

Combine $C_G$ (smooth, geometry-aware) with raw APC (local detail):

1. Pool $C_G$ along the disparity axis to get $C_G^p$ (two-level pyramid: original + half-resolution)
2. Build all-pairs correlations $C_A$ and similarly pool to get $C_A^p$
3. Concatenate: **CGEV = $\{C_G, C_A, C_G^p, C_A^p\}$**

The CGEV simultaneously contains:
- **$C_G$**: non-local geometry at full disparity resolution
- **$C_A$**: local fine-grained matching at full disparity resolution
- **$C_G^p$**: geometry-aware costs at half resolution (wider context)
- **$C_A^p$**: local correlations at half resolution

#### Stage 3: Warm-Start Initial Disparity from GEV

$$d_0 = \sum_{d=0}^{D-1} d \times \text{Softmax}(C_G(d)) \quad \text{(4)}$$

- **$d_0$** = initial disparity map at 1/4 resolution
- **$D$** = maximum disparity at 1/4 resolution
- **$C_G(d)$** = geometry encoding volume slice at disparity $d$
- **$\text{Softmax}$** = applied across the disparity dimension → probability distribution
- The sum is the **soft argmin**: expected value of disparity under the distribution

**Why this matters:** RAFT-Stereo starts from $d_0 = 0$ and must iterate many times to reach a reasonable disparity. IGEV-Stereo starts from a soft-argmin estimate derived from the 3D-regularized GEV — a geometrically-informed warm start that dramatically accelerates convergence.

#### Stage 4: ConvGRU-based Iterative Refinement

**Geometry feature lookup at each iteration:**

$$G_f = \sum_{i=-r}^{r} \text{Concat}\{C_G(d_k + i), C_A(d_k + i), C_G^p(d_k/2 + i), C_A^p(d_k/2 + i)\} \quad \text{(5)}$$

- **$G_f$** = geometry features retrieved from CGEV for this iteration
- **$d_k$** = current disparity estimate at iteration $k$
- **$r = 4$** = indexing radius (9 values sampled)
- Each of the four volume components contributes $(2r+1) = 9$ lookups → total $4 \times 9 = 36$ values per pixel

**ConvGRU update** (same structure as RAFT-Stereo, 3 levels at 1/4, 1/8, 1/16):

$$x_k = [\text{Encoder}_g(G_f), \text{Encoder}_d(d_k), d_k]$$

$$z_k = \sigma(\text{Conv}([h_{k-1}, x_k], W_z) + c_z)$$

$$r_k = \sigma(\text{Conv}([h_{k-1}, x_k], W_r) + c_r)$$

$$\tilde{h}_k = \tanh(\text{Conv}([r_k \odot h_{k-1}, x_k], W_h) + c_h) \quad \text{(6)}$$

$$h_k = (1 - z_k) \odot h_{k-1} + z_k \odot \tilde{h}_k$$

- **$c_z, c_r, c_h$** = context features from Context Network, **additively injected into gate computations** — this is a key difference from standard GRUs. Explicit scene context modulates the recurrent dynamics at every step.
- All other variables as in standard GRU

**Disparity update:**

$$d_{k+1} = d_k + \Delta d_k \quad \text{(7)}$$

### Training Loss (Dual Supervision)

$$\mathcal{L}_{init} = \text{Smooth}_{L1}(d_0 - d_{gt}) \quad \text{(8)}$$

$$\mathcal{L}_{stereo} = \mathcal{L}_{init} + \sum_{i=1}^{N} \gamma^{N-i} \Vert d_i - d_{gt}\Vert _1 \quad \text{(9)}$$

- **$\mathcal{L}_{init}$** = auxiliary loss on the soft-argmin initial disparity — **forces the 3D regularization network to learn meaningful geometry on its own**, not just produce an intermediate representation
- **$\gamma = 0.9$** = exponential weighting — later iterations weighted more
- **$N = 22$** = training iterations

**This dual supervision is critical:** Without $\mathcal{L}_{init}$, the GEV might collapse into an arbitrary intermediate representation that the GRU corrects later. The explicit supervision ensures the GEV output is already a reasonable disparity estimate.

---

## Key Innovations vs RAFT-Stereo and CREStereo

### vs RAFT-Stereo

| Aspect | RAFT-Stereo | IGEV-Stereo |
|--------|-------------|-------------|
| Cost representation | Raw APC only | CGEV (GEV + APC at 2 pyramid levels) |
| Geometry encoding | None | Explicit 3D UNet |
| Initial disparity | $d_0 = 0$ | Soft argmin on GEV (warm start) |
| Convergence | 32 iterations needed | 3 iterations match RAFT-32 |
| Backbone | ResNet | MobileNetV2 (ImageNet pretrained) |
| Upsampling | Hidden state alone | Hidden state + $f_{l,2}$ features |
| KITTI 2015 D1-all | 1.82% | **1.59%** |
| Runtime (960×540) | 0.36s | **0.18s** |

### vs CREStereo

- CREStereo uses multi-scale cascade with purely correlation-based matching
- IGEV-Stereo uses **single-scale with 3D-regularized geometry volume** — simpler but more powerful
- IGEV-Stereo is ~2× faster (0.18s vs 0.41s) AND more accurate (1.59% vs 1.69% on KITTI 2015 D1-all)

---

## Benchmark Results

### Scene Flow
- **EPE: 0.47** (vs 0.48 ACVNet, 0.56 RAFT-Stereo) — SOTA at publication

### KITTI 2015 (ranked 1st at publication, 16 iterations)
| Method | D1-bg | D1-fg | D1-all | Runtime |
|--------|-------|-------|--------|---------|
| PSMNet | 1.86 | 4.62 | 2.32 | 0.41s |
| GwcNet | 1.74 | 3.93 | 2.11 | 0.32s |
| RAFT-Stereo | 1.58 | 3.05 | 1.82 | 0.38s |
| CREStereo | 1.45 | 2.86 | 1.69 | 0.41s |
| **IGEV-Stereo** | **1.38** | **2.67** | **1.59** | **0.18s** |

**Fastest AND most accurate** among top-10 methods.

### Reflective Regions (KITTI 2012)

| Method | Iter | 2-noc | 3-noc |
|--------|------|-------|-------|
| RAFT-Stereo | 8 | 9.98 | 6.64 |
| RAFT-Stereo | 32 | 8.41 | 5.40 |
| IGEV-Stereo | 8 | 8.30 | 4.88 |
| IGEV-Stereo | 32 | 7.29 | 4.11 |

**IGEV-Stereo at 8 iterations outperforms RAFT-Stereo at 32** in reflective regions — direct proof that GEV's non-local geometry helps where local matching fails.

---

## Ablation Highlights

### Component ablation (Scene Flow, 32 iters)

| Components | EPE | >3px | Params |
|------------|-----|------|--------|
| Baseline (APC only) | 0.56 | 2.85 | 12.02M |
| + GEV | 0.51 | 2.68 | 12.60M |
| + GEV + Init from GEV | 0.50 | 2.62 | 12.60M |
| + GEV + Init + $\mathcal{L}_{init}$ | 0.48 | 2.51 | 12.60M |
| **Full (CGEV)** | **0.47** | **2.47** | **12.60M** |

**The GEV adds only 0.58M params** but provides most of the accuracy gain. Every component contributes, with $\mathcal{L}_{init}$ being essential.

### Iteration efficiency

| Model | 1 iter | 3 iters | 8 iters | 32 iters |
|-------|--------|---------|---------|----------|
| RAFT-Stereo | 2.16 | 0.95 | 0.66 | 0.61 |
| **IGEV-Stereo** | **0.66** | **0.58** | **0.50** | **0.47** |

**At 1 iteration, IGEV-Stereo beats RAFT-Stereo by 69.4%** (0.66 vs 2.16 EPE). At 3 iterations, IGEV-Stereo already surpasses RAFT-Stereo's 32-iteration result. The warm start alone accounts for most of this advantage.

### GEV resolution

| GEV Resolution | EPE | Params |
|---------------|-----|--------|
| 1/8 | 0.49 | 12.71M |
| 1/4 | **0.47** | 12.60M |

**1/8 GEV is only 0.02 EPE worse** — validated operating point for cheaper edge variants.

---

## Strengths & Limitations

**Strengths:**
- Elegant theoretical framing: cost-filtering and iterative methods are complementary
- Tiny overhead: 0.58M params, ~1ms for the 3D UNet
- Dramatic convergence speedup: 3 iterations beat 32 RAFT iterations
- Best zero-shot generalization on Middlebury (ETH3D slightly worse than RAFT-Stereo)
- Fastest among top-10 KITTI methods

**Limitations:**
- 3D convolutions still present (poor NPU support)
- Fixed disparity range baked into GEV
- Slight ETH3D generalization gap vs RAFT-Stereo (3.6% vs 3.2%)

---

## Relevance to Our Edge Model

**IGEV-Stereo is arguably the best base architecture for our edge model.** Reasons:

1. **Convergence in 3 iterations** — directly aligns with our <33ms latency target. RAFT-Stereo's 32 iterations are wasteful.
2. **1/8 GEV ablation validates cheaper config** — 0.49 EPE at 1/8 with only +0.11M params is a free speedup
3. **MobileNetV2 backbone already** — swap for MobileNetV4/EfficientViT for NPU-friendly inference
4. **3D UNet can be replaced by separable 3D convs** — Separable-Stereo approach reduces FLOPs ~8×
5. **Warm-start from GEV + distilled mono depth** — combining IGEV's GEV initialization with DEFOM-Stereo's mono prior could eliminate the need for many iterations entirely

**Proposed edge variant:** IGEV-Stereo with 1/8 GEV using separable 3D convs, MobileNetV4 backbone, single-level GRU with 4 iterations, DEFOM-style mono initialization. Target: <15ms on Jetson Orin Nano.

---

## Connections to Other Papers

| Paper | Relationship |
|-------|-------------|
| **RAFT-Stereo** | Direct baseline — IGEV-Stereo adds GEV on top of RAFT's GRU framework |
| **GwcNet** | Group-wise correlation comes from here |
| **CoEx** | Guided cost volume excitation (Eq. 3) borrowed from CoEx |
| **PSMNet / GC-Net** | Soft argmin for initial disparity comes from these 3D cost volume methods |
| **IGEV++** | Direct successor — extends to multi-range volumes |
| **Selective-Stereo** | Uses IGEV-Stereo as base, adds SRU to fix over-smoothing |
| **MonSter** | Built on IGEV backbone with added mono-stereo mutual refinement |
| **DEFOM-Stereo** | Could be combined — DEFOM's mono init + IGEV's GEV could both provide warm start |
