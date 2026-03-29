# Stereo Anywhere: Robust Zero-Shot Deep Stereo Matching Even Where Either Stereo or Mono Fail

**Authors:** Luca Bartolomei, Fabio Tosi, Matteo Poggi, Stefano Mattoccia (University of Bologna)
**Venue:** CVPR 2025
**Priority:** 9/10
**Code:** https://github.com/bartn8/stereoanywhere

---

## Core Problem & Motivation

Stereo and mono are **complementary**: stereo provides geometric constraints (resolves scale, perspective illusions) but fails on textureless/non-Lambertian surfaces. Mono provides contextual priors (handles textureless/reflective) but suffers from scale ambiguity and is fooled by perspective illusions. Stereo Anywhere fuses both so that when one fails, the other compensates.

## Architecture: Dual-Branch with Clean Separation

![Figure 2: Stereo Anywhere architecture. Seven components: (1) frozen RAFT-Stereo feature encoder, (2) stereo correlation volume with mono-guided truncation, (3) frozen VFM producing mono depth for both views, (4) normal-based mono correlation volume, (5) differentiable monocular scaling, (6) learned context encoder on mono depth, (7) dual-lookup GRU refinement.](../../../figures/StereoAnywhere_fig2_architecture.png)

### Key Design: Two Parallel Cost Volumes

**Stereo Correlation Volume** $V_S$ — standard dot-product correlation between left/right features. **Truncated** using mono priors to suppress false matches behind mirrors/glass.

**Monocular Correlation Volume** $V_M$ — correlation between **surface normals** derived from mono depth:
$$(\mathbf{V}_M)_{ijk} = \sum_h (\nabla_L)_{hij} \cdot (\nabla_R)_{hik}$$
- **$\nabla_L, \nabla_R$** = surface normals estimated from monocular depth maps $M_L, M_R$ via spatial gradient
- Normals are scale-invariant — naturally compatible between relative mono depth and absolute disparity
- Segmented into N=8 relative depth bins to prevent foreground-background confusion

### Differentiable Monocular Scaling

$$\hat{s}, \hat{t} = \arg\min_{s,t} \sum_{L,R} \|\sqrt{\hat{C}} \odot [(s \cdot M + t) - \hat{D}]\|_F$$
- **$\hat{s}, \hat{t}$** = global scale and shift aligning mono depth to stereo disparity
- **$\hat{C}$** = confidence map from entropy of the mono correlation volume (sqrt for weighting)
- **$\hat{D}$** = coarse disparity from softargmax on $V_M$
- Solved via weighted least squares, jointly for both views for left-right consistency

### Iterative Refinement
Starting from scaled mono disparity $\tilde{M}_L = \hat{s} \cdot M_L + \hat{t}$, dual-lookup GRU alternates between stereo ($G_S$) and mono ($G_M$) correlation lookups at each iteration.

## Unique Contributions

**Volume truncation for mirrors:** Fuzzy-logic mask detects where mono and stereo disagree about surface location. Suppresses stereo correlations that "see through" reflective surfaces.

**VFM-agnostic design:** Stereo Anywhere treats the VFM as a black box producing depth maps. Works with DAv2, DepthPro, MoGe, Lotus — all outperform baseline.

**MonoTrap dataset:** 26 scenes of perspective illusions (painted checkerboards, murals) where mono fails but stereo succeeds. Proves the architecture doesn't blindly trust mono.

## Benchmark Results

| Dataset | Stereo Anywhere | RAFT-Stereo | Improvement |
|---------|----------------|-------------|-------------|
| Middlebury bad>2 | **6.96** | 11.15 | 37.6% |
| ETH3D bad>1 | **1.66** | 2.59 | 35.9% |
| KITTI 2015 bad>3 | **3.93** | 5.44 | 27.8% |
| Booster bad>2 | **9.01** | 17.84 | 49.5% |

Nearly halves errors on non-Lambertian surfaces (Booster). Ranks 1st on Booster leaderboard (fine-tuned).

## Model Size & Speed

| Resolution | Total Time | Memory |
|-----------|-----------|--------|
| 512x512 | 0.24s | 1.34 GB |
| 1024x1024 | 0.63s | 6.31 GB |

~50% overhead vs RAFT-Stereo due to the mono branch.

## Relevance to Edge Model

- **Normal-based mono correlation is lightweight** — spatial gradients + dot products, no heavy processing
- **Global scale/shift** (single solve) is cheap — provides good initialization reducing iterations
- **Volume truncation** for mirrors is virtually free — applicable to any edge model
- **VFM-agnostic** — can swap in a tiny mono model (DAv2-Small or distilled variant)
- **Challenge:** Dual cost volumes double memory. Could use sparse sampling or bilateral grids, or alternate lookups (stereo on even iterations, mono on odd).
