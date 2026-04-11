# BANet: Bilateral Aggregation Network for Mobile Stereo Matching

**Authors:** Gangwei Xu, Jiaxin Liu, Xianqi Wang, Junda Cheng, Yong Deng, Jinliang Zang, Yurui Chen, Xin Yang (HUST, Autel Robotics, Optics Valley Lab)
**Venue:** ICCV 2025
**Priority:** 8/10
**Code:** https://github.com/gangweiX/BANet

---

## Core Problem & Motivation

SOTA stereo methods typically use costly **3D convolutions** to aggregate a full cost volume — accurate but not mobile-friendly. Directly applying 2D convolutions for cost aggregation causes **edge blurring, detail loss, and mismatches in textureless regions**. Complex deformable convolutions or iterative warping can partially mitigate this but are **not mobile-friendly**, making deployment on mobile devices nearly impossible.

![Figure 1: Top — comparison with real-time methods on KITTI 2015 D1-all vs Scene Flow EPE. BANet-2D sits at the Pareto frontier. Bottom — qualitative: MobileStereoNet-2D blurs edges, loses details, causes mismatches in textureless regions. BANet-2D preserves sharp edges and fine details using pure 2D convolutions. Latency measured for 512×512 input on Qualcomm Snapdragon 8 Gen 3.](../../../figures/BANet_fig1_comparison.png)

**Core insight:** The problem is that real stereo scenes contain **two distinct region types with opposing needs**:
- **High-frequency detailed regions (edges, fine structures)** — need sharp, local aggregation that preserves boundaries
- **Low-frequency smooth regions (textureless walls, floors, roads)** — need wide-area aggregation that fills in missing information

A single aggregation network forced to handle both either blurs edges (from averaging too much) or fails on textureless areas (from averaging too little). **BANet's solution: separate the cost volume into two branches and aggregate each with specialized sub-networks.**

---

## Architecture

![Figure 2: BANet overview. Feature extraction → correlation → Scale-aware Spatial Attention produces detailed map A and smooth map (1-A). The cost volume is split into a Detailed Cost Volume and a Smooth Cost Volume via element-wise Hadamard product with these maps. Two parallel aggregation branches (Detailed Aggregation and Smooth Aggregation) process each independently. Outputs are fused for the final disparity.](../../../figures/BANet_fig2_architecture.png)

### Stage 1: Feature Extraction + Correlation
Standard components — CNN backbone extracting features, standard correlation volume computation.

### Stage 2: Scale-aware Spatial Attention (SSA)

This is one of two key contributions. SSA produces a **spatial attention map $\mathbf{A}$** that identifies high-frequency (detailed) vs low-frequency (smooth) regions.

**Intuition:** Fine-scale image features capture high-frequency details; coarse-scale features encompass smoother contexts. By **contrasting multi-scale left image features**, we can identify where detail vs smoothness dominates.

**Multi-scale feature processing:**
- Extract left image features at 1/16, 1/8, 1/4 resolutions: $\mathbf{F}_{l,16}, \mathbf{F}_{l,8}, \mathbf{F}_{l,4}$
- Upsample 1/16 and 1/8 to 1/4 via learned convs: $\mathbf{F}_{l,16}^{up}, \mathbf{F}_{l,8}^{up}$
- Compress each scaled feature to the same channel count with $\text{Conv}$
- Concatenate all three

**Equation:**
$$\mathbf{S} = \text{Concat}([\text{Conv}(\mathbf{F}_{l,16}^{up}), \text{Conv}(\mathbf{F}_{l,8}^{up}), \text{Conv}(\mathbf{F}_{l,4})])$$
$$\mathbf{A} = \sigma(\text{Conv}(\mathbf{S})) \quad \text{(4)}$$

- **$\mathbf{S}$** = multi-scale feature stack at 1/4 resolution
- **$\sigma$** = sigmoid activation — produces values in $[0, 1]$
- **$\mathbf{A}$** = spatial attention map. **High** values indicate **detailed regions**; **low** values indicate **smooth regions**
- **$(1 - \mathbf{A})$** = the **contrary map** — highlights smooth regions

**Why this works:** When the 1/4 features differ significantly from the upsampled 1/16 features, it means there's high-frequency information not captured at coarse scale — i.e., a detailed region. When they're similar, the region is smooth enough that the coarser view is sufficient.

### Stage 3: Bilateral Aggregation

Split the full cost volume $\mathbf{C}_{cor}$ into two disjoint volumes:

$$\mathbf{C}_d = \mathbf{A} \odot \mathbf{C}_{cor}$$
$$\mathbf{C}_s = (1 - \mathbf{A}) \odot \mathbf{C}_{cor} \quad \text{(1)}$$

- **$\mathbf{C}_d$** = detailed cost volume — masked by $\mathbf{A}$, keeping costs in high-frequency regions
- **$\mathbf{C}_s$** = smooth cost volume — masked by $(1 - \mathbf{A})$, keeping costs in low-frequency regions
- **$\odot$** = Hadamard (element-wise) product — broadcast across disparity dimension

**Two parallel aggregation branches:**

$$\mathbf{C}'_d = \mathbf{G}_d(\mathbf{C}_d)$$
$$\mathbf{C}'_s = \mathbf{G}_s(\mathbf{C}_s) \quad \text{(2)}$$

- **$\mathbf{G}_d$** = **detailed aggregation** branch — targets high-frequency details, preserves edges. Uses a series of **inverted residual blocks** (MobileNet V2 style) with smaller receptive fields.
- **$\mathbf{G}_s$** = **smooth aggregation** branch — targets low-frequency regions with wider receptive fields. Same block structure as $\mathbf{G}_d$ but **does NOT share weights** — each learns its own specialized filters.

**Detailed branch structure:** 4 blocks at 1/4, 6 blocks at 1/8, 8 blocks at 1/16 resolution. Channels: 32, 64, 128. Uses inverted residual blocks with point-wise → depth-wise → point-wise 2D convolutions, expansion factor 4.

**Smooth branch:** Same channels and block counts but different learned weights, targeting wider-context aggregation.

### Stage 4: Fusion

$$\mathbf{C}_{agg} = \mathbf{A} \odot \mathbf{C}'_d + (1 - \mathbf{A}) \odot \mathbf{C}'_s \quad \text{(3)}$$

- **$\mathbf{C}_{agg}$** = final aggregated cost volume
- Each region uses the output of the branch that specialized in its type
- Detailed regions get sharp output from $\mathbf{G}_d$; smooth regions get well-spread output from $\mathbf{G}_s$

### Extension to 3D: BANet-3D

The bilateral aggregation concept extends naturally to 3D convolutions. **BANet-3D** uses 3×3×3 3D convolutions in 3 downsampling + 3 upsampling blocks within each branch. Each upsampling block uses one 4×4×4 3D transposed convolution followed by two 3×3×3 3D convolutions. BANet-3D achieves **SOTA accuracy** among real-time methods on high-end GPUs.

---

## Key Innovations

1. **Bilateral cost volume separation** — the cost volume is explicitly split into detail/smooth sub-volumes based on learned image content. This is the first stereo method to do explicit frequency-based cost volume separation at the architecture level.

2. **Scale-aware Spatial Attention** — the multi-scale contrast (fine features vs upsampled coarse features) is a principled way to identify detail vs smoothness, not an ad-hoc heuristic.

3. **Pure 2D for mobile, 3D for high-end** — same architecture, swap 2D blocks for 3D blocks depending on target hardware.

4. **Contrary attention fusion** — using $\mathbf{A}$ and $(1 - \mathbf{A})$ as complementary masks ensures every pixel is handled by exactly one specialized branch (weighted sum), not averaged ambiguously.

---

## Benchmark Results

### KITTI 2015 (the headline result)

**BANet-2D achieves 35.3% higher accuracy than MobileStereoNet-2D** on the KITTI 2015 leaderboard, with **faster runtime** on mobile devices (Qualcomm Snapdragon 8 Gen 3).

Specific numbers from Figure 1:
- **BANet-2D:** 45ms on Snapdragon 8G3
- **MobileStereoNet-2D:** 157ms on Snapdragon 8G3 (3.5× slower)

### Scene Flow

BANet-2D sits at the Pareto frontier of 2D cost aggregation methods.

### BANet-3D on KITTI

Achieves the **highest accuracy among all published real-time methods on high-end GPUs**, demonstrating the bilateral aggregation concept's generality.

---

## Strengths & Limitations

**Strengths:**
- **Principled content-aware aggregation** — explicit frequency-based cost volume separation is a novel architectural idea
- **Pure 2D convolutions** — mobile-friendly, deployable on Snapdragon 8 Gen 3 at 45ms
- **Scales to 3D** — same concept works for high-end GPUs via BANet-3D
- **Significant mobile speedup** — 3.5× faster than MobileStereoNet-2D on mobile
- **Scale-aware attention is lightweight** — multi-scale contrast is cheap to compute
- **Accuracy gap over prior 2D methods** — 35.3% D1 improvement on KITTI is substantial

**Limitations:**
- **Two parallel branches** — doubles the aggregation compute vs single-branch methods (though each branch is smaller)
- **Still not iterative** — produces single-pass estimate; loses the cross-domain generalization advantage that Pip-Stereo demonstrated iterative methods have
- **Attention map is computed once** — doesn't evolve with the disparity estimate (unlike iterative refinement)
- **Doesn't use foundation models** — no monocular prior integration
- **Specific mobile hardware** — tested on Snapdragon 8 Gen 3, not Jetson/Orin

---

## Relevance to Our Edge Model

**BANet is directly applicable as the cost aggregation stage of our edge model.** Specific takeaways:

### Directly adoptable

1. **Bilateral cost volume split** — explicitly separating detail and smooth regions is a clean architectural pattern that compounds with other efficiency techniques. Can be combined with:
   - **BGNet's bilateral grid upsampling** (complementary — BGNet handles spatial edge-preserving upsampling)
   - **LightStereo's channel boost** (can be used inside each aggregation branch)
   - **Separable-Stereo's V2 blocks** (can be used for the individual branch convolutions)

2. **Scale-aware Spatial Attention (SSA)** — the multi-scale contrast mechanism is lightweight and directly usable. Provides semantically meaningful region classification for free.

3. **Pure 2D aggregation for mobile** — BANet-2D proves that with the right architectural split, 2D-only aggregation can match 3D accuracy. Important for NPU deployment.

### Caution

- **BANet is non-iterative** — by itself, it would inherit LightStereo's generalization weaknesses. **Combine with iterative refinement** (Pip-Stereo's PIP + Selective-Stereo's SRU) to keep cross-domain robustness.

### Proposed Integration

```
Correlation Volume
    ↓
[Scale-aware Spatial Attention]  ← BANet
(produces A and 1-A from multi-scale left features)
    ↓
[Bilateral Split: C_detailed | C_smooth]  ← BANet
    ↓
[Detailed Agg] || [Smooth Agg]  ← BANet with V2 blocks (Separable-Stereo)
    ↓
[Fused Cost Volume]  ← BANet's Eq. 3
    ↓
[GEV extraction + warm start]  ← IGEV-Stereo
    ↓
[SRU iterations + PIP compression]  ← Selective-Stereo + Pip-Stereo
```

This combines BANet's content-aware aggregation with the iterative refinement that Pip-Stereo demonstrated is essential for zero-shot generalization.

---

## Connections to Other Papers

| Paper | Relationship |
|-------|-------------|
| **MobileStereoNet-2D** | Direct competitor — BANet-2D beats it by 35.3% on KITTI at lower mobile latency |
| **LightStereo** | Alternative 2D aggregation approach — LightStereo uses channel boost, BANet uses content-aware split |
| **BGNet** | Complementary — BGNet optimizes upsampling, BANet optimizes aggregation; stackable |
| **Separable-Stereo** | Complementary — V2 blocks can be used inside BANet's aggregation branches |
| **IGEV-Stereo** | Can benefit from bilateral aggregation as an alternative to single-branch 3D UNet |
| **Pip-Stereo** | Natural complement — BANet's fast aggregation + Pip-Stereo's compressed iteration |
| **CoEx** | Uses similar guided cost volume excitation concept |
