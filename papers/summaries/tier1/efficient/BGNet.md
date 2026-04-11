# BGNet: Bilateral Grid Learning for Stereo Matching Networks

**Authors:** Bin Xu, Yuhua Xu, Xiaoli Yang, Wei Jia, Yulan Guo (Orbbec, Hefei University of Technology, Sun Yat-sen University)
**Venue:** CVPR 2021
**Priority:** 9/10 — THE key efficiency technique for our edge model
**Code:** https://github.com/3DCVdeveloper/BGNet

---

## Core Problem & Motivation

The fundamental efficiency bottleneck in stereo matching networks is the **cost aggregation step**. 3D cost volume methods (GC-Net, PSMNet, GA-Net) achieve strong accuracy but apply 3D convolutions at **1/3 or 1/4 resolution**, leading to unacceptable latency:

- **GANet**: 1.8 seconds per image pair
- **PSMNet**: 0.41 seconds per image pair
- Both are too slow for real-time applications (AR, robotics, autonomous driving)

Lightweight alternatives (StereoNet) do aggregation at **very low resolution** (1/8 or 1/16) where 3D convolutions are cheap, then upsample via **bilinear interpolation + hierarchical refinement**. This is fast but inaccurate — bilinear upsampling cannot recover sharp disparity edges, and the refinement stages don't fully compensate.

**BGNet's insight:** The problem is essentially **edge-preserving upsampling** of a low-resolution disparity map guided by a high-resolution reference image. This is exactly what the **bilateral filter** solves classically, and what the **bilateral grid** (Chen et al. 2007) does efficiently. Adapt the bilateral grid to deep learning as a parameter-free, learned edge-preserving upsampling module → perform expensive cost aggregation at low resolution (fast) but get high-resolution sharp output (accurate).

---

## Background: The Bilateral Grid (Classical)

Before diving into BGNet's innovation, understand the classical bilateral grid:

**Bilateral filter concept:** At each pixel, average nearby pixels weighted by both:
1. Spatial proximity (Gaussian in $(x, y)$)
2. Range proximity (Gaussian in intensity $I$)

This produces edge-preserving smoothing — pixels across edges have very different intensity, so they don't contribute.

**Bilateral grid acceleration:** Chen et al. (2007) showed that the bilateral filter becomes a simple 3D linear operation if you lift the image into a **3D grid** indexed by $(x, y, I)$:

1. **Splat:** Distribute each pixel's value into the 3D grid at location $(x, y, I)$
2. **Blur:** Apply a simple 3D Gaussian blur in the grid
3. **Slice:** For each original pixel, lookup the grid value at $(x, y, I)$ via trilinear interpolation

The key property: **convolution in the bilateral grid = bilateral filtering in the image**. This is dramatically cheaper than brute-force bilateral filtering.

**BGNet's insight:** Replace "intensity $I$" with "**disparity $d$**" and use the bilateral grid as an **edge-preserving disparity upsampling module**. The disparity map is the "value" being filtered, and the reference image features provide the guidance signal.

---

## Architecture: Cost Volume Upsampling in Bilateral Grid (CUBG)

![Figure 1: The Cost Volume Upsampling in Bilateral Grid (CUBG) module. Inputs: a low-resolution cost volume and high-resolution image features. Output: a high-resolution cost volume. The low-res cost volume is reinterpreted as a bilateral grid, a guidance map is learned from the image features via 1x1 convolutions, and the slicing layer produces the high-res cost volume via trilinear interpolation.](../../../figures/BGNet_fig1_CUBG_module.png)

The CUBG module is the core innovation. It takes:
- **Low-resolution cost volume** $\mathcal{C}_\mathcal{L}$ at e.g. 1/8 resolution (where 3D aggregation is cheap)
- **High-resolution image features** (full guidance)

And produces:
- **High-resolution cost volume** $\mathcal{C}_H$ via bilateral grid slicing

### Step 1: Reinterpret Cost Volume as Bilateral Grid

A cost volume has 4 dimensions: $(x, y, d, c)$ — spatial, disparity, channels. A bilateral grid has 4 dimensions: $(x, y, g, c)$ — spatial, guidance value, channels. These are structurally identical!

BGNet reinterprets the low-resolution cost volume $\mathcal{C}_\mathcal{L}$ as a bilateral grid $\mathcal{B}$ directly. No conversion needed — just a reinterpretation. A 3D convolution of size $3 \times 3 \times 3$ is applied to $\mathcal{B}$ to make it a proper bilateral grid (letting the network learn the best grid structure).

### Step 2: Learn the Guidance Map

The **guidance map** $G$ tells the slicing layer, for each high-resolution pixel, which disparity level in the grid to look up. BGNet learns $G$ from the high-resolution image features via two $1 \times 1$ convolutions:

$$G = \text{Conv}_{1 \times 1}(\text{Conv}_{1 \times 1}(\text{features}))$$

**Key property:** Because each pixel's guidance value depends on its own feature vector, **sharp edges can be obtained automatically** — neighboring pixels with different features produce different guidance values, leading to different grid lookups.

### Step 3: Slicing Layer — The Core Operation

The slicing layer produces the high-resolution cost volume via trilinear interpolation from the bilateral grid:

$$\mathcal{C}_H(x, y, d) = \mathcal{B}(sx, sy, sd, s_G G(x, y)) \quad \text{(1)}$$

Every element:
- **$\mathcal{C}_H(x, y, d)$** = the output high-resolution cost at position $(x, y)$ for disparity $d$
- **$\mathcal{B}$** = the bilateral grid (reinterpreted low-resolution cost volume after $3 \times 3 \times 3$ conv)
- **$s \in (0, 1)$** = the spatial ratio — ratio between high-res and low-res spatial dimensions (e.g., $s = 1/8$ when upsampling from 1/8 to full resolution)
- **$sx, sy$** = spatial lookup position in the grid (scale the high-res pixel $(x, y)$ by $s$ to get low-res coordinates)
- **$sd$** = disparity lookup in the grid (scaled similarly)
- **$s_G \in (0, 1)$** = the ratio between gray level of the grid ($l_{grid}$) to gray level of the guidance map ($l_{guid}$) — handles the mapping between guidance map values and grid indices
- **$G(x, y)$** = guidance map value at position $(x, y)$
- **$s_G \cdot G(x, y)$** = the grid index along the guidance dimension — **this is where edge-preservation happens**: different pixels look up different grid slices based on their features
- **$\mathcal{B}(\cdot, \cdot, \cdot, \cdot)$** = trilinear interpolation in the 4D grid

**Why this works (intuition):**
- Within a smooth region (e.g., a road surface), neighboring high-res pixels have similar features → similar guidance values → they look up the same grid slice → smooth output
- Across an edge (e.g., car boundary), neighboring high-res pixels have different features → different guidance values → they look up different grid slices → sharp discontinuity preserved

This is the same mathematical trick as the classical bilateral grid, but **learned end-to-end** — the guidance map generator and the grid content are both optimized by backprop.

### Key Properties

1. **Parameter-free slicing layer** — the slice operation has zero learnable parameters. All learning goes into the guidance map generator and the $3 \times 3 \times 3$ grid conv.
2. **Grid size:** typically $W/8 \times H/8 \times D_{max}/8 \times 32$, where $D_{max}$ is max disparity
3. **Plug-and-play:** embeds seamlessly into any existing 3D cost volume architecture (GCNet, PSMNet, GANet, DeepPruner, FADNet)

---

## BGNet Architecture (End-to-End)

![Figure 2: BGNet overview. Feature extraction (shared weights) → cost volume construction → 3D convolutions on low-resolution cost volume → CUBG slicing layer (inside dashed box) → high-resolution cost volume → residual disparity refinement. All expensive operations happen at 1/8 resolution.](../../../figures/BGNet_fig2_architecture.png)

BGNet is a complete end-to-end network built around CUBG. Four main stages:

### Stage 1: Feature Extraction

- ResNet-like architecture (similar to PSMNet and GANet)
- First 3 conv layers with $3 \times 3$ kernels, strides 2, 1, 1 downsample from input to 1/2 resolution
- Next several layers with strides 1, 2, 2, 1 take features to 1/8 resolution
- Two hourglass networks for large receptive fields and rich semantic information
- All feature maps at 1/8 resolution concatenated → 352 channels for cost volume construction
- $\mathbf{f}_l, \mathbf{f}_r$ = final left/right feature maps

### Stage 2: Cost Volume Construction

**Group-wise correlation cost volume** (from GwcNet, Guo 2019):

For each spatial location $(x, y)$ at disparity level $d$, split feature channels into $N_g = 44$ groups. Each group independently computes a correlation:

$$c_g(x, y, d) = \langle \mathbf{f}_l^g(x, y), \mathbf{f}_r^g(x - d, y) \rangle$$

The $N_g$ group correlations are stacked. This combines the advantages of concatenation volumes (rich features) and correlation volumes (explicit matching signal).

### Stage 3: Cost Aggregation (at 1/8 resolution, CHEAP)

One hourglass architecture with 3D convolutions operating on the 1/8-resolution cost volume. Only ONE hourglass (vs 3 in PSMNet) for efficiency:
- Two 3D convs reduce channels from 44 to 16
- U-Net-like 3D aggregation with element-wise summation for skip connections (cheaper than concatenation)
- Output: aggregated cost volume $\mathcal{C}_\mathcal{L}$ at 1/8 resolution

**Critical insight:** All the expensive 3D convolution work happens at **1/8 resolution**. At this resolution, the cost volume is $64 \times$ smaller than at full resolution, so 3D conv costs drop by the same factor.

### Stage 4: CUBG Upsampling + Regression

The aggregated low-res cost volume is upsampled to full resolution via the CUBG module (Eq. 1). Then soft argmin regression produces the disparity map:

$$\mathbf{D}_{pred}(x, y) = \sum_{d=0}^{D_{max}} d \times \text{softmax}(\mathcal{C}_H(x, y, d)) \quad \text{(2)}$$

- **$\mathbf{D}_{pred}(x, y)$** = predicted disparity at pixel $(x, y)$
- **$D_{max}$** = maximum disparity candidate
- **$\text{softmax}$** = applied over the disparity dimension → probability distribution
- The sum computes the expected value (soft argmin) — sub-pixel accurate

### Loss Function

$$L = \sum_p \mathcal{L}(\mathbf{D}_{pred}(p) - \mathbf{D}_{gt}(p)) \quad \text{(3)}$$

Where $\mathcal{L}$ is the **smooth L1** (Huber) loss:

$$\mathcal{L}(x) = \begin{cases} 0.5 x^2 & \text{if } |x| < 1 \\ |x| - 0.5 & \text{otherwise} \end{cases}$$

Smooth L1 is quadratic near zero (stable gradients for small errors) and linear far from zero (robust to outliers).

---

## Embedding CUBG into Other Networks

A key contribution: CUBG is **plug-and-play**. The paper demonstrates CUBG embedded into four existing architectures:

| Base Network | Original Cost Volume Resolution | With CUBG | Suffix |
|--------------|-------------------------------|-----------|--------|
| GCNet | 1/2 | 1/8 | **GCNet-BG** |
| PSMNet | 1/4 | 1/8 | **PSMNet-BG** |
| GANet | 1/3 | 1/8 | **GANet-BG** |
| DeepPrunerFast | PatchMatch + 1/4 | 1/8 | **DeepPrunerFast-BG** |

Each modified network replaces its original bilinear upsampling with the CUBG module. **All achieve 4-29× speedup while maintaining comparable accuracy** — a remarkable result.

---

## Benchmark Results

### Speedup of Existing Networks via CUBG

| Network | Original Time | With CUBG | Speedup | KITTI D1-all |
|---------|--------------|-----------|---------|--------------|
| GCNet | 0.85s | 0.03s | **28.3×** | 2.87 → 2.70 |
| PSMNet | 0.41s | 0.04s | **10.3×** | 2.32 → 2.33 |
| GANet | 1.8s | 0.06s | **30.0×** | 1.81 → 1.97 |
| DeepPrunerFast | 0.06s | 0.022s | **2.7×** | 2.59 → 2.48 |

**GANet gets a 30× speedup with only 0.16 points of accuracy loss on KITTI** — a stunning efficiency gain.

### BGNet Standalone

| Metric | Value |
|--------|-------|
| **Runtime (KITTI)** | **25ms (40 FPS)** |
| Parameters | ~2.9M |
| KITTI 2015 D1-all | 2.78% |
| Scene Flow EPE | 1.17 |

**BGNet runs in real-time at 40 FPS** on a single 2080Ti GPU at KITTI resolution.

### Comparison to Real-Time Competitors (KITTI 2015)

| Method | D1-all | FPS | Notes |
|--------|--------|-----|-------|
| StereoNet | 4.83 | 60 | Lower accuracy |
| DeepPrunerFast | 2.59 | 16 | Slower |
| AANet | 2.55 | 16 | Slower |
| **BGNet** | **2.78** | **40** | Real-time with competitive accuracy |

BGNet sits at the Pareto frontier of speed vs accuracy.

---

## Ablation Highlights

### CUBG vs Bilinear Upsampling (Scene Flow)

| Configuration | EPE | >1 px error | Time |
|---------------|-----|-------------|------|
| PSMNet (direct 1/4 aggregation) | 1.17 | ~12% | 0.41s |
| PSMNet with 1/8 aggregation + bilinear upsample | 1.42 | 14% | ~0.05s |
| PSMNet-BG (1/8 aggregation + CUBG) | 1.20 | ~12% | 0.04s |

**Replacing bilinear upsampling with CUBG recovers almost all accuracy lost from the resolution reduction** — the key finding. CUBG is genuinely edge-preserving.

### Qualitative Edge Preservation

Figure 3 in the paper shows qualitative comparisons where CUBG preserves sharp object boundaries (vehicles, pedestrians, road edges) that bilinear upsampling blurs. The difference is especially visible in regions near depth discontinuities.

---

## Strengths & Limitations

**Strengths:**
- **Massive speedup** — 2.7-30× across four different base networks
- **Minimal accuracy loss** — often less than 0.2 points on KITTI
- **Parameter-free slicing layer** — the core operation adds zero learnable parameters
- **Plug-and-play** — no architectural surgery required to add to existing networks
- **Real-time on KITTI** — BGNet standalone at 40 FPS
- **Mathematically principled** — grounded in classical bilateral grid theory, not an ad-hoc heuristic

**Limitations:**
- **Still uses 3D convolutions** — just at lower resolution. 3D convs remain poorly supported on mobile NPUs.
- **Not the absolute SOTA** — KITTI D1-all of 2.78% is competitive but behind iterative methods (RAFT-Stereo, IGEV-Stereo at ~1.6%)
- **Single-scale guidance** — the guidance map is computed from a single feature resolution; multi-scale guidance might help at very different object scales
- **Memory for the bilateral grid** — the grid is $W/8 \times H/8 \times D_{max}/8 \times 32$ which is still substantial

---

## Relevance to Our Edge Model

**BGNet's bilateral grid is arguably the single most important efficiency technique for our edge model.** Here's why:

### The Core Trade-off BGNet Solves

Every edge stereo network faces a dilemma:
- **Aggregate at full resolution** → accurate but slow
- **Aggregate at low resolution** → fast but lose edge detail

BGNet's CUBG resolves this dilemma: aggregate at low resolution (fast 3D convs on tiny volumes) but upsample with edge preservation (sharp final output). **This is exactly the trade-off our edge model must navigate.**

### Direct Integration into Our Architecture

Our proposed edge architecture from the iterative synthesis already includes bilateral grids:

```
[Local Correlation + Group-wise]  ← from CREStereo
(with 2D-1D alternating)
    ↓
[Lightweight GEV]  ← from IGEV-Stereo (with separable 3D conv)
(1/8 resolution, not 1/4)  ← BGNet's insight
    ↓
[Warm Start + CUBG upsampling]  ← from BGNet for edge-preserving high-res output
```

### Specific Benefits

1. **Replaces bilinear upsampling** — the CUBG module eliminates the final upsampling accuracy loss that plagues lightweight designs like StereoNet
2. **Enables 1/8-resolution aggregation** — all expensive 3D conv work happens on $64\times$-smaller tensors
3. **Combines with separable 3D convs (Separable-Stereo)** — the 1/8 resolution aggregation itself can use depthwise separable 3D convs for additional ~8× FLOP reduction
4. **Parameter-free slicing** — no extra learnable parameters beyond the guidance map generator (two $1 \times 1$ convs)
5. **Edge-aware naturally** — the guidance map generates sharp discontinuities from the input features without explicit edge detection

### Proposed Edge Model Integration

- **Cost volume aggregation at 1/8 resolution** with depthwise-separable 3D convs (combining BGNet + Separable-Stereo)
- **CUBG upsampling** from 1/8 to 1/4 resolution (not to full — iterative refinement handles the last mile)
- **Iterative refinement** at 1/4 resolution with SRU (Selective-Stereo) for 4-6 iterations
- **Convex upsampling** from 1/4 to full (RAFT-Stereo / IGEV-Stereo style)

This combines **BGNet's cheap aggregation**, **Separable-Stereo's tiny 3D convs**, **IGEV-Stereo's warm start from GEV**, **Selective-Stereo's SRU**, and **RAFT-Stereo's convex upsampling** — a hybrid edge architecture that inherits the best efficiency technique from each paper.

---

## Connections to Other Papers

| Paper | Relationship |
|-------|-------------|
| **Chen et al. 2007 (Bilateral Grid)** | Classical foundation that BGNet adapts for deep learning |
| **StereoNet** | Competitor in lightweight category — BGNet beats it on accuracy at similar speed |
| **PSMNet / GCNet / GANet** | Base networks demonstrated to benefit from CUBG (4-30× speedup) |
| **DeepPrunerFast** | Another fast baseline that CUBG improves |
| **Separable-Stereo** | Complementary efficiency technique — separable 3D convs can go inside the 1/8 aggregation stage |
| **IGEV-Stereo** | Uses MobileNetV2 backbone; could integrate CUBG to further speed up the GEV step |
| **RT-IGEV++** | RT variant already operates at reduced resolution; CUBG would make the final upsampling sharper |
| **LightStereo / BANet** | Contemporary lightweight methods with different efficiency strategies |
| **Fast-FoundationStereo** | Foundation model compression approach — BGNet's insight complements distillation |
