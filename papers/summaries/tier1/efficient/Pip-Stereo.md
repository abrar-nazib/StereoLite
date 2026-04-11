# Pip-Stereo: Progressive Iterations Pruner for Iterative Optimization-based Stereo Matching

**Authors:** Jintu Zheng, Qizhe Liu, HuangXin Xu, Zhuojie Chen (ARIDGE XPENG)
**Venue:** CVPR 2026
**Priority:** 9/10 — this is the closest prior art to our edge iterative design

---

## Core Problem & Motivation

All prior stereo efficiency work optimizes the **cost volume computation** (BGNet, LightStereo, Fast-ACVNet, etc.). Nobody addresses the **RNN/GRU recurrent update loop itself** as a hardware bottleneck. Yet iterative refinement is the dominant paradigm (RAFT-Stereo, IGEV, Selective-IGEV) and has two fundamental problems on edge hardware:

1. **Static-graph-with-iterative-loops** creates complex control flow that **breaks operator fusion** and makes the model acutely **sensitive to quantization noise**.
2. **High-resolution RNN inference is memory-bandwidth-bound** — hidden-state reads/writes to HBM (GPU) or LPDDR (edge NPU) dominate latency and scale quadratically with resolution.

### The Empirical Motivation

Pip-Stereo demonstrates that iterative updates are:
- **Spatially sparse:** By iteration 32, **fewer than 1% of pixels are still being updated**
- **Temporally redundant:** The "hit ratio" (fraction of pixels updated at the same location as the previous iteration) **exceeds 0.99 for IGEV after just 10 iterations**, and for RAFT after ~15 iterations
- **Almost all subsequent iterations echo already-settled pixels** — pure waste of compute

### Gap vs Prior Efficient Iterative Methods

- **RT-IGEV++** drops to 6 iterations but **EPE degrades to 0.52 on SceneFlow** vs IGEV's 0.49
- **Pip-Stereo reaches 1 iteration** at EPE **0.45** — better than RT-IGEV++ at 6 iterations, better than original IGEV at 12 iterations, 11.5× faster than DEFOM-Stereo on Orin NX

---

## Architecture: Three-Part Contribution

Pip-Stereo builds on Selective-IGEV and introduces three innovations:

### Contribution 1: Monocular Prior Transfer (MPT) — training-time only

Unlike DEFOM-Stereo/MonSter which embed Depth Anything V2 ViT permanently at inference (5-14 second latency), MPT uses a **teacher-student framework**:

- **Teacher:** Depth Anything V2-Large (frozen ViT)
- **Student:** NAS-searched RepViT backbone (lightweight, mobile-friendly)
- **Alignment losses:** Two feature-level correspondence losses:
  - Multi-resolution context feature alignment
  - Cost volume embedding alignment (from the 3D regularization network)
- **Re-parameterization blocks** are used so that multi-branch training-time blocks fold into single-path convolutions at inference
- **Teacher is discarded at inference** — only the student encoder runs, with monocular knowledge baked into its weights

This solves DEFOM-Stereo's fundamental edge-deployment problem: you can have monocular depth priors **without running any ViT at inference**.

### Contribution 2: Progressive Iteration Pruning (PIP) — the core innovation

The key observation: naive iteration truncation causes EPE to jump by +32.1% at 1 iteration on IGEV-style models. You cannot simply cut iterations.

**The PIP recipe:**

1. Train the network at **$T = 24-32$ iterations** ("More iterations RNN" = Mi-RNN)
2. Progressively halve: 32 → 16 → 8 → 4 → 2 → 1 ("Fewer iterations RNN" = Fi-RNN at each step)
3. At each halving, only the GRU update block weights are finetuned; all other weights are **frozen**
4. The Fi-RNN is **initialized from the Mi-RNN** weights

Each Fi-RNN learns to approximate the **$r$-fold composition** of the Mi-RNN's update operator $F^{(r)} = F \circ F \circ \ldots \circ F$ — a coarse-grained operator that compresses $r$ fine-grained steps into one.

### The PIP Loss Functions

**Mi-RNN dynamics (Eq. 1):**
$$z_{t+1}^{\text{Mi-RNN}} = F_\theta(z_t^{\text{Mi-RNN}}), \quad t = 0, \ldots, T-1$$

- **$z_t \in \mathbb{R}^d$** = GRU hidden state at iteration $t$ ($d$-dimensional)
- **$T$** = total iterations in the "more" model
- **$F_\theta$** = ConvGRU update function parameterized by $\theta$

**Fi-RNN dynamics (Eq. 2):**
$$z_{s+1}^{\text{Fi-RNN}} = G_\phi(z_s^{\text{Fi-RNN}}), \quad s = 0, \ldots, S-1$$

- **$S = T/r$** = reduced iteration count
- **$r$** = compression ratio (= 2 at each halving step)
- **$G_\phi$** = pruned GRU update function, initialized from $F_\theta$

**Block-aggregated output of Mi-RNN (Eq. 3):**
$$\bar{d}_s^{\text{Mi-RNN}} = \frac{1}{r} \sum_{i=1}^{r} \Psi(z_{r(s-1)+i}^{\text{Mi-RNN}}), \quad s = 1, \ldots, S$$

- **$\Psi(\cdot)$** = the disparity output mapping (head converting hidden state to disparity delta $\Delta d$)
- **$\bar{d}_s^{\text{Mi-RNN}}$** = average disparity update over the $r$-step block — the "coarsened" trajectory of the full model

**Cumulative alignment loss (Eq. 4) — the core PIP loss:**

$$\mathcal{L}_{cum} = \sum_{s=1}^{S} \Vert \sum_{k=1}^{s} d_k^{\text{Fi-RNN}} - \sum_{k=1}^{s} \bar{d}_k^{\text{Mi-RNN}}\Vert _2^2$$

- Aligns the **cumulative disparity trajectory** of the pruned model to the full model at each coarse step
- Forces the Fi-RNN to match the **entire refinement evolution** of the Mi-RNN, not just the final output
- This is the key distillation signal — the Fi-RNN learns to emulate the full refinement trajectory at a coarser time scale

**Final disparity supervision (Eq. 5):**

$$\mathcal{L}_{final} = \Vert d_S^{\text{Fi-RNN}} - \Psi(z_T^{\text{Mi-RNN}})\Vert _2^2$$

- Direct alignment of the pruned model's last output to the full model's last output

**Hidden state matching (Eq. 6):**

$$\mathcal{L}_{hid} = \sum_{s=1}^{S} \Vert z_s^{\text{Fi-RNN}} - z_{rs}^{\text{Mi-RNN}}\Vert _2^2$$

- Aligns hidden states at each coarse step to the Mi-RNN hidden state at the corresponding stride-$r$ position
- Enforces **skip-step equivalence** — the Fi-RNN's hidden state evolution should mimic taking strides of size $r$ through the Mi-RNN's trajectory

**Total PIP loss (Eq. 7):**
$$\mathcal{L} = \mathcal{L}_{cum} + \mathcal{L}_{final} + \mathcal{L}_{hid}$$

**Result:** After 5 halvings (32→1), EPE only increases by **+0.07** vs the 32-iteration model, compared to **+0.18 without PIP**.

### Contribution 3: FlashGRU — hardware-aware sparse CUDA kernel

A structured-sparse, I/O-aware CUDA kernel replacing standard ConvGRU:

- Uses an **importance map** (from Selective-IGEV's attention module) to select the **top-k pixels** for update (e.g., top 30% → 70% sparsity)
- Pre-allocates **contiguous GPU SRAM buffers** for hidden states
- Uses a **static multi-resolution coordinate index-mapping table** to pack sparse pixels contiguously
- **Fuses** sigmoid/tanh kernels with GEMM operations
- **Minimizes HBM write-backs** during GRU unrolling

**Benefit scales with resolution** — most valuable at 2K+ images:

| Resolution | Native | FlashGRU | Speedup |
|-----------|--------|----------|---------|
| 320×736 | 28.8ms | 14.8ms | 1.94× |
| 640×1472 | 45.2ms | 15.2ms | 2.97× |
| **1280×2944 (2K)** | **122.4ms** | **16.8ms** | **7.28×** |

At 2K resolution the memory peak drops from 4105MB to 957MB (4.3× memory reduction).

---

## Benchmark Results

### In-Domain (Orin NX, 384×1344, FP32)

| Method | Iters | SceneFlow EPE | ETH3D Bad-1(Noc) | KITTI15 D1-all | Latency (s) |
|--------|-------|---------------|------------------|----------------|-------------|
| RAFT-Stereo | 32 | 0.72 | 2.44 | 1.82 | 2.95 |
| IGEV-Stereo | 12 | 0.49 | 1.12 | 1.59 | 1.29 |
| Selective-IGEV | 12 | 0.44 | 1.23 | 1.55 | 1.61 |
| DEFOM-Stereo | 32 | 0.42 | 0.70 | 1.33 | **5.05** |
| MonSter | 32 | 0.37 | 0.46 | 1.41 | 7.63 |
| RT-IGEV++ | 6 | 0.52 | 3.81 | 1.79 | 0.39 |
| HITNet(L) | — | 0.55 | 2.79 | 1.98 | 0.44 |
| **Pip-Stereo** | **1** | **0.45** | **0.35** | **1.44** | **0.44** |

**Headline result:** Pip-Stereo at 1 iteration matches Selective-IGEV's 12-iteration accuracy while being **3.7× faster** on Orin NX, and is **11.5× faster than DEFOM-Stereo** with better ETH3D accuracy.

**RTX 4090 at 320×640 FP16:** 19ms (52 FPS)
**Orin NX at 320×640 FP16:** 75ms (13.3 FPS)

### Zero-Shot Generalization (DrivingStereo, D1-all)

| Method | Avg D1-all | Latency |
|--------|-----------|---------|
| RT-IGEV++ | 8.04 | 0.39s |
| RT-MonSter++ | 4.18 | 0.79s |
| HITNet(L) | **93.52** | 0.44s |
| IINet | 27.70 | 0.29s |
| LightStereo(S) | 13.08 | 0.13s |
| **Pip-Stereo** | **4.35** | 0.44s |

**This is the most striking result.** At the same latency as HITNet, Pip-Stereo achieves 4.35 vs 93.52 D1-all — a **~20× better generalization**. Pip-Stereo approaches MonSter++ (2.69) while being 17× faster.

**Critical insight:** Non-iterative real-time methods (HITNet, IINet, LightStereo) catastrophically fail at cross-domain generalization. Iterative refinement provides an inductive bias that cannot be replicated — **we must preserve iteration, just make it cheap.**

---

## Ablation Highlights

| Variant | Iters | EPE | Latency |
|---------|-------|-----|---------|
| Selective-IGEV baseline | 12 | 0.44 | 1.61s |
| Baseline, 1 iter (no MPT/PIP) | 1 | 0.64 | 0.60s |
| +MPT, PipStereo-i32, 1 iter | 1 | 0.56 | 0.44s |
| +MPT, PipStereo-i32, 32 iters | 32 | 0.38 | 3.40s |
| **+MPT +PIP, 1 iter** | **1** | **0.45** | **0.44s** |

**Key findings:**
1. **MPT alone** reduces EPE from 0.44 → 0.38 at 32 iters (13.6% improvement from monocular priors)
2. **Without PIP at 1 iter:** EPE = 0.56 (+32.1% degradation)
3. **With PIP at 1 iter:** EPE = 0.45 (+15.5% degradation — **PIP halves the accuracy loss**)
4. **PIP is less effective on zero-initialized RAFT-Stereo** — requires regularized initial disparity (IGEV-style) for best results

---

## Strengths & Limitations

**Strengths:**
- **Only paper to attack the RNN hardware bottleneck** directly rather than replacing it
- **PIP is model-agnostic** — demonstrated on Selective-IGEV, RAFT-Stereo, and FoundationStereo
- **MPT eliminates monocular encoder at inference** (unlike DEFOM-Stereo/MonSter)
- **FlashGRU scales with resolution** — most beneficial exactly where edge latency is worst
- **Enormous generalization gap** vs non-iterative real-time methods (4.35 vs 13-93 D1-all)
- **Real Orin NX hardware measurements**, not just FLOPs
- **Strong combination of all pieces** — MPT + PIP + FlashGRU compound to unique efficiency

**Limitations:**
- **PIP is less effective on RAFT-style zero initialization** — requires IGEV-style regularized init
- **Final accuracy still trails best large models** (EPE 0.45 vs MonSter's 0.37)
- **No explicit parameter count reported**
- **FlashGRU benefit at 1 iteration is marginal** — recommends 4-iter variants for FlashGRU's full speedup
- **NAS search adds training complexity** (10K genetic algorithm generations)
- **Not real-time at 384×1344** on Orin NX (440ms ≈ 2.3 FPS) — only real-time at 320×640 (75ms)

---

## Relevance to Our Edge Model

**Pip-Stereo is the closest prior art to our goal.** It is essentially a prototype of what our edge DEFOM-Stereo variant should be. Specific takeaways:

### Directly adoptable

1. **PIP as training recipe** — Train at N=32 iterations, then apply successive halving with $\mathcal{L}_{cum} + \mathcal{L}_{final} + \mathcal{L}_{hid}$ loss. Our edge model should do exactly this to reach 1-4 inference iterations.

2. **MPT without monocular encoder at inference** — solves DEFOM-Stereo's biggest deployment problem. Distill Depth Anything V2 priors into a lightweight RepViT student via feature alignment losses, fold re-parameterization blocks at inference.

3. **RepViT + NAS backbone** — strong mobile backbone candidate. Genetic search over 10% of FoundationStereo Dataset is computationally tractable.

4. **FlashGRU design principle** — structured sparsity + I/O-aware CUDA kernel. We should implement or adapt this for our GRU update block.

5. **Cumulative trajectory matching (Eq. 4)** is a novel distillation objective that specifically preserves iterative refinement characteristics while compressing iterations.

### Cautions

- **Use IGEV-style regularized initial disparity**, not RAFT-Stereo zero init — PIP won't work as well with zero init
- **PIP is a training-time technique** — no inference overhead, so this is pure win
- **FlashGRU CUDA kernel is GPU-specific** — may need re-implementation for mobile NPUs (SNPE, TFLite)

### Key Insight

**Iterative refinement cannot be replaced** without losing generalization. The generalization gap between Pip-Stereo (1 iter, 4.35 D1-all) and LightStereo-S (non-iterative, 13.08 D1-all) is enormous. Our edge model MUST keep iteration, but can aggressively compress it via PIP.

### Proposed Edge Model Architecture (updated from iterative synthesis)

```
Stereo Pair
    ↓
[NAS-searched RepViT Backbone]  ← From Pip-Stereo
(with MPT distillation from Depth Anything V2, train-time only)
    ↓
[Local Correlation + Group-wise]  ← From CREStereo
    ↓
[Lightweight GEV]  ← From IGEV-Stereo (1/8 resolution)
(with BGNet-style bilateral grid upsampling)
    ↓
[Warm Start from GEV soft argmin]  ← IGEV-style (not zero init!)
    ↓
[SRU × 1-4 iterations]  ← Selective-Stereo's SRU + Pip-Stereo's PIP
(with FlashGRU if deploying on GPU)
    ↓
[Convex Upsampling]  ← RAFT-Stereo
    ↓
Disparity Map
```

**Target:** <5M params, <33ms on Jetson Orin Nano at 320×640, strong zero-shot generalization.

---

## Connections to Other Papers

| Paper | Relationship |
|-------|-------------|
| **Selective-IGEV** | Direct base architecture — Pip-Stereo builds on this |
| **IGEV-Stereo** | Provides regularized initial disparity that PIP requires |
| **DEFOM-Stereo** | Monocular prior idea, but MPT removes the ViT at inference |
| **MonSter** | Bidirectional mono-stereo fusion — Pip-Stereo is simpler, training-time only |
| **RT-IGEV++** | Fair speed comparison — Pip-Stereo is more accurate at similar latency |
| **Fast-FoundationStereo** | Same compression philosophy (distill + prune) but different target teacher |
| **RepViT** | Backbone choice |
| **Depth Anything V2** | Teacher for MPT |
| **FoundationStereo Dataset (FSD)** | Used for NAS search sub-sample |
