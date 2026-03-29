# Synthesis: Foundation-Model Era Stereo Matching (2025-2026)

> Consolidation of 8 Tier 1 foundation-model era papers.

---

## Papers Covered

| # | Paper | Venue | Approach | Params | Speed |
|---|-------|-------|----------|--------|-------|
| 1 | **DEFOM-Stereo** | CVPR 2025 | CNN+ViT fusion + Scale Update | 15-47M | 0.26-0.32s |
| 2 | **FoundationStereo** | CVPR 2025 | Side-Tuning + APC + Disparity Transformer | ~400M | 0.7s |
| 3 | **MonSter** | CVPR 2025 | Bidirectional mutual refinement (SGA+MGR) | 356M | 0.64s |
| 4 | **Stereo Anywhere** | CVPR 2025 | Dual cost volumes (stereo + normal-based mono) | ~350M | 0.63s |
| 5 | **AIO-Stereo** | AAAI 2025 | Multi-VFM distillation (train-time only) | ~15M deploy | ~0.24s |
| 6 | **D-FUSE** | ICCV 2025 | Binary ordering maps + global fusion | ~11M + ViT | 0.40s |
| 7 | **PromptStereo** | CVPR 2026 | PRU replaces GRU with prompted DPT decoder | ~350M | 0.36s |
| 8 | **Fast-FoundationStereo** | CVPR 2026 | Divide-and-conquer compression | **14.6M** | **49ms** |

---

## The Three Paradigms of Mono-Stereo Fusion

Reading all 8 papers reveals three distinct approaches to integrating monocular foundation models with stereo matching:

### Paradigm 1: Feature-Level Fusion
**Papers:** DEFOM-Stereo, FoundationStereo, AIO-Stereo

**Approach:** Extract features from a VFM (ViT backbone) and fuse them with CNN stereo features. The VFM's depth output may or may not be used directly.

| Method | How VFM features enter | VFM depth used? | At inference? |
|--------|----------------------|-----------------|---------------|
| DEFOM-Stereo | Addition (CNN + aligned ViT DPT features) | Yes (initialization + scale update) | Yes (frozen ViT) |
| FoundationStereo | Side-tuning (CNN adapts ViT features) | No (only latent features) | Yes (frozen ViT) |
| AIO-Stereo | Distillation via DLSKT gating | No | **No** (VFMs are teachers only) |

**Key insight:** AIO-Stereo is the only method where VFMs are NOT needed at inference — the knowledge is distilled into the CNN during training. This is the most edge-friendly paradigm.

### Paradigm 2: Scale-and-Shift Correction
**Papers:** DEFOM-Stereo, MonSter, D-FUSE, Stereo Anywhere

**Approach:** The VFM produces a depth map with unknown scale/shift. The stereo network recovers the correct scale using geometric matching.

| Method | Scale correction approach | Scope |
|--------|-------------------------|-------|
| DEFOM-Stereo | Scale Update Module (learned per-pixel scale via Scale Lookup) | Per-pixel, iterative |
| MonSter | Global least-squares → SGA per-pixel refinement | Global → per-pixel |
| D-FUSE | Binary ordering maps + global scale/shift registration | Local structure + global |
| Stereo Anywhere | Differentiable weighted least-squares (entropy-weighted) | Global |

**Key insight:** Per-pixel scale correction (DEFOM, MonSter SGA) is more powerful than global (Stereo Anywhere, D-FUSE), but also more expensive. For edge deployment, a good global initialization might suffice with fewer per-pixel refinement iterations.

### Paradigm 3: Architecture-Level Integration
**Papers:** PromptStereo, Stereo Anywhere

**Approach:** The iterative refinement architecture itself is redesigned to incorporate VFM priors structurally.

| Method | Architectural change | Effect |
|--------|---------------------|--------|
| PromptStereo | Replaces GRU with DPT decoder (PRU), injects structure/motion prompts | Faster AND more accurate than GRU |
| Stereo Anywhere | Dual cost volumes with separate stereo+mono lookups per iteration | Complementary failure handling |

**Key insight:** PromptStereo's finding that PRU is both faster and better than GRU is significant — the DPT decoder architecture may be inherently better suited for iterative refinement than a standard ConvGRU.

---

## Comparative Benchmark Analysis

### Zero-Shot Generalization (all trained on SceneFlow or mixed synthetic)

| Method | KITTI 2015 D1 | Middlebury BP-2 | ETH3D BP-1 | Speed |
|--------|--------------|----------------|------------|-------|
| RAFT-Stereo (baseline) | 5.5 | 12.6 | 3.3 | 0.22s |
| DEFOM-Stereo (ViT-L) | 4.99 | 5.91 | 2.35 | 0.32s |
| FoundationStereo | **2.8** | **1.1** | **0.5** | 0.7s |
| MonSter | 3.41 | 9.33 | 0.99 | 0.64s |
| Stereo Anywhere | 3.93 | 6.96 | 1.66 | 0.63s |
| D-FUSE | 5.60 | 8.39 | 1.88 | 0.40s |
| PromptStereo | 4.59 | 6.03 | **1.56** | 0.36s |
| **Fast-FoundationStereo** | 3.25 | 4.80 | 0.62 | **0.049s** |
| RT-IGEV (no VFM) | 4.00 | 12.75 | 1.63 | 0.045s |

**FoundationStereo has the best absolute accuracy** (by far on Middlebury/ETH3D). **Fast-FoundationStereo achieves the best speed-accuracy trade-off** — 10x faster than FoundationStereo with modest accuracy loss, and dramatically better than any other real-time method.

---

## Design Principles for Our Edge Model

Synthesizing lessons from all 8 papers:

### 1. Feature distillation is non-negotiable
AIO-Stereo and Fast-FoundationStereo prove that distilling VFM knowledge into a lightweight CNN is both feasible and essential. Training from scratch (even with the same architecture) produces dramatically worse zero-shot generalization. **Our edge model must use foundation model distillation during training.**

### 2. Scale correction is the key innovation
DEFOM-Stereo's Scale Update and MonSter's SGA both show that correcting the scale ambiguity of monocular depth is what unlocks the fusion benefit. Without it, depth initialization can actually hurt (DEFOM ablation: DI alone hurts on some datasets). **Our edge model needs at least a lightweight scale correction mechanism.**

### 3. Fewer iterations with better initialization
MonSter achieves EPE 0.42 with only 4 iterations (vs IGEV's 0.47 with 32) when monocular priors provide good initialization. PromptStereo's PRU is faster per iteration than GRU. **Our target: 4-6 iterations with distilled mono initialization.**

### 4. The cost volume is the memory bottleneck
FoundationStereo's 4D cost volume causes 18.5GB at full resolution. All methods struggle with memory. **Our edge model should use bilateral grids (BGNet) or sparse sampling (DeepPruner) to avoid full cost volume construction.**

### 5. No need for VFM at inference on edge
AIO-Stereo proves VFMs can be training-time teachers only. Fast-FoundationStereo distills the backbone. D-FUSE shows minimal overhead (0.08s) with a frozen VFM but that's still too much for edge. **Our edge model should NOT run any VFM at inference — all knowledge distilled into the CNN.**

---

## Proposed Edge Model Architecture

Based on the synthesis of all 8 papers:

```
                    TRAINING ONLY
                    ┌─────────────────────┐
                    │ Frozen Depth Any. V2 │──→ Feature distillation loss
                    │ (teacher, discarded) │──→ Depth pseudo-labels
                    └─────────────────────┘
                              ↕ MSE
┌─────────────────────────────────────────────────────────┐
│                    DEPLOYED MODEL                        │
│                                                          │
│  Stereo Pair → [Distilled Encoder] → Correlation Volume  │
│                  (MobileNetV4,         (bilateral grid    │
│                   <5M params)           from BGNet)       │
│                        ↓                                  │
│              [Distilled Mono Head] → Scale Init            │
│                  (from DPT,             ↓                  │
│                   few conv layers)    [Lite GRU]           │
│                                      (4-6 iters,          │
│                                       separable 3D,       │
│                                       scale + delta)      │
│                        ↓                                  │
│                   Disparity Output                        │
│                        ↓                                  │
│               [ONNX → TensorRT]                           │
└─────────────────────────────────────────────────────────┘

Target: <5M params, <33ms on Jetson Orin Nano, <50ms on mobile NPU
```

### Key Design Choices Informed by Each Paper

| Design Decision | Informed By |
|----------------|------------|
| Distill VFM into CNN backbone | AIO-Stereo, Fast-FoundationStereo |
| Scale Update mechanism | DEFOM-Stereo |
| Bilateral grid cost volume | BGNet (efficiency section) |
| 4-6 GRU iterations max | MonSter (4-iter ablation) |
| Mono depth initialization | DEFOM-Stereo, Stereo Anywhere |
| No VFM at inference | AIO-Stereo principle |
| Separable 3D convolutions | Separable-Stereo |
| Structured pruning of GRU | Fast-FoundationStereo |
| Pseudo-label training data | Fast-FoundationStereo pipeline |
| Normal-consistency filtering | Fast-FoundationStereo, D-FUSE |

---

## Open Questions for Our Model

1. **Can DEFOM-Stereo's Scale Lookup work with bilateral grids?** Scale Lookup indexes the correlation volume at scaled disparity positions — does this conflict with the bilateral grid's spatially adaptive structure?

2. **What's the minimum mono model for useful initialization?** DEFOM uses ViT-S (15M). AIO uses distilled features (no mono model at inference). Can a 1-2M param distilled depth head provide useful scale priors?

3. **PRU vs GRU for edge?** PromptStereo shows PRU (DPT decoder) is faster and better than GRU. But the DPT decoder has more parameters per iteration. Is there a sweet spot?

4. **How much accuracy do we lose at <5M params?** Fast-FoundationStereo achieves 14.6M. RT-IGEV achieves 4.17M but without VFM priors. What's achievable at 5M with distillation?
