# PromptStereo: Zero-Shot Stereo Matching via Structure and Motion Prompts

**Authors:** Wang et al.
**Venue:** CVPR 2026
**Priority:** 8/10

---

## Core Problem & Motivation

Existing methods integrate foundation models only at the feature extraction stage. The iterative refinement stage — the GRU — is still trained from scratch, limiting its zero-shot generalization. GRUs have three fundamental limitations:
1. Trained from scratch — can't inherit VFM priors
2. Hidden states constrained by reset/update gates — limited capacity for extreme disparities
3. Compress external inputs through direct convolution — causing ambiguous guidance

## Key Innovation: Prompt Recurrent Unit (PRU)

PromptStereo **replaces the GRU entirely** with a DPT decoder architecture from Depth Anything V2, initialized with pre-trained weights. Two types of prompts are injected as additive residuals:

- **Structure Prompt (SP):** Encodes geometric discrepancy $|\hat{d}_k - \hat{d}_M|$ between normalized current disparity and monocular depth — tells the network *where* stereo and mono disagree
- **Motion Prompt (MP):** Encodes stereo motion cues from local cost volume lookup — provides matching-specific information

**PRU is both more accurate AND faster than GRU** (0.36s vs 0.64s for MonSter baseline) — a rare combination.

![Figure 2: PromptStereo overview. Standard feature encoder + cost volume produces initial disparity. Depth Anything V2 produces relative depth. Affine-Invariant Fusion creates fused starting point. Chain of Prompt Recurrent Units (PRU) iteratively refines, with structure and motion prompts injected at the highest resolution level.](../../../figures/PromptStereo_fig2_overview.png)

## Architecture

Built on MonSter baseline, with same feature extraction and cost volume. Key changes:

### Affine-Invariant Fusion (AIF) — initialization

$$\hat{d} = \frac{d - t(d)}{s(d)}, \quad \text{where } t(d) = \text{median}(d), \quad s(d) = \frac{1}{N}\sum|d - t(d)|$$

- **$\hat{d}$** = normalized disparity (affine-invariant)
- **$t(d)$** = median of the disparity map (robust center estimate)
- **$s(d)$** = mean absolute deviation from median (robust scale estimate)
- Both monocular depth and stereo disparity are normalized this way, making them directly comparable

$$d'_M = s(d_0) \cdot \hat{d}_M + t(d_0)$$

- **$d'_M$** = monocular depth projected into stereo disparity coordinate system
- Uses the stereo estimate's scale and shift to map normalized monocular depth to absolute disparity

$$d_F = c \cdot d_0 + (1-c) \cdot d'_M$$

- **$c$** = learned confidence map — fuses stereo initialization and projected monocular depth

### PRU Update (replaces GRU)

$$\hat{h}_k^i = h_k^i + \text{ConvBlock}(P_S) + \text{ConvBlock}(P_M^k)$$

$$h_{k+1}^i = (1 - z_k) \cdot h_k^i + z_k \cdot \hat{h}_k^i$$

- **$P_S$** = Structure Prompt — static across iterations (geometric discrepancy doesn't change)
- **$P_M^k$** = Motion Prompt at iteration $k$ — changes each iteration (depends on current disparity lookup)
- **$z_k$** = update gate only (no reset gate — simpler than GRU)
- Prompts are added as **residuals** (not concatenated) to avoid distorting the pre-trained DPT features

## Benchmark Results (Zero-Shot)

| Dataset | PromptStereo (SceneFlow) | PromptStereo (Unlimited) |
|---------|-------------------------|-------------------------|
| KITTI 2015 EPE | 1.09 | 0.88 |
| Middlebury-T (H) bad2.0 | **6.03** | **3.36** |
| ETH3D bad1.0 | **1.56** | **0.97** |
| Booster (Q) bad2.0 | 12.13 | **3.67** |

SOTA zero-shot generalization across all datasets. Reduces Middlebury 2021 error by ~50% vs MonSter. Outperforms FoundationStereo on Middlebury 2021 and Booster.

**Inference time:** 0.36s (faster than MonSter at 0.64s and Stereo Anywhere at 0.65s).

## Relevance to Our Edge Model

Two key takeaways:
1. **PRU is faster than GRU while being more accurate** — if a lightweight version of the DPT decoder can be distilled, it could serve as the iterative update module in our edge model
2. **Affine-Invariant Fusion** (median-MAD normalization + linear projection + confidence weighting) is extremely cheap and provides robust initialization that reduces iterations needed — critical for edge latency
