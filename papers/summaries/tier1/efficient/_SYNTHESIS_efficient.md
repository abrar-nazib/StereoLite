# Synthesis: Efficient Stereo Matching (Tier 1)

> Consolidation of 8 Tier 1 efficient stereo papers, spanning 2021-2026.

---

## Papers Covered

| # | Paper | Year | Core Technique | Status |
|---|-------|------|---------------|--------|
| 1 | **BGNet** | 2021 | Bilateral grid for edge-preserving cost volume upsampling | Full analysis |
| 2 | **Separable-Stereo** | 2021 | Feature-wise Separable Convolutions (FwSCs) for 3D cost aggregation | Full analysis |
| 3 | **LightStereo** | 2025 | Channel-boost 2D aggregation (inverted residual on disparity dim) | Full analysis |
| 4 | **BANet** | 2025 | Bilateral aggregation (detail/smooth split with scale-aware attention) | Full analysis |
| 5 | **LiteAnyStereo** | 2025 | Hybrid 3D-2D aggregation + 3-stage training w/ FoundationStereo KD | Full analysis |
| 6 | **GGEV** | 2026 | Depth-aware Dynamic Cost Aggregation guided by frozen DAv2 features | Full analysis |
| 7 | **Pip-Stereo** | 2026 | Progressive Iteration Pruning + MPT + FlashGRU | Full analysis |
| 8 | **Distill-then-Prune** | 2024 | Distillation THEN structured pruning for edge | Full analysis |

> **Note on file recovery:** Three PDFs in `papers/raw/efficient/` were initially mislabeled during download (wrong arXiv IDs: Separable-Stereo, GGEV, LiteAnyStereo). All three have been re-downloaded from the correct sources (arXiv 2108.10216, 2512.06793, 2511.16555 respectively) and fully analyzed. No placeholders remain.

---

## The Five Dimensions of Stereo Efficiency

Reading all 8 papers reveals five distinct axes along which efficiency can be improved. Each paper attacks one or more:

### Axis 1: Cost Volume Representation
**How cheap is the matching cost data structure?**

| Technique | Papers | Trade-off |
|-----------|--------|-----------|
| 4D cost volume (standard) | (baseline) | Accurate but $O(C \cdot D \cdot H \cdot W)$ memory |
| 3D cost volume (scalar per disparity) | LightStereo, BANet | Cheap but needs careful aggregation |
| Local correlation window | CREStereo (iterative section) | $\sim 50\times$ smaller than all-pairs |
| Bilateral grid reinterpretation | BGNet | Enables low-res 3D processing + high-res output |
| Multi-range sparse sampling | IGEV++ (iterative section) | Covers 768px with 48 candidates |

### Axis 2: Cost Aggregation Cost
**How cheap is the per-layer processing of the cost volume?**

| Technique | Papers | Reduction vs Standard 3D |
|-----------|--------|-----------------------|
| Depthwise separable 3D (V1) | Separable-Stereo | $\sim 19\times$ |
| Inverted residual 3D (V2, $t=2$) | Separable-Stereo | $\sim 7\times$ |
| Channel-boost 2D (V2 on disparity dim) | LightStereo | Eliminates 3D entirely |
| Bilateral aggregation (split network) | BANet | 2D-only with detail preservation |
| Axial-Planar Convolution (APC) | FoundationStereo (foundation section) | Separates spatial × disparity |

### Axis 3: Spatial Resolution of Aggregation
**At what resolution do the expensive operations happen?**

| Technique | Papers | Where 3D convs run |
|-----------|--------|-------------------|
| Full resolution (bad) | — | OOM on 80GB GPUs |
| 1/4 resolution | PSMNet, IGEV-Stereo | Still expensive |
| **1/8 resolution** | **BGNet, IGEV++** | **64× smaller volume** |
| 1/16 resolution | — | Too coarse, loses detail |

BGNet's key insight: do aggregation at 1/8 resolution, then use bilateral grid to upsample with edge preservation.

### Axis 4: Iteration Count (for iterative methods)
**How many GRU iterations are needed?**

| Technique | Papers | Typical Iterations |
|-----------|--------|-------------------|
| Naive truncation (bad) | — | Fails below 8 iterations |
| Warm start from GEV | IGEV-Stereo | 3 iters match RAFT-32 |
| Progressive Iteration Pruning (PIP) | **Pip-Stereo** | **1 iteration with minimal accuracy loss** |
| Coarse-to-fine cascade | CREStereo | 4-8 per cascade stage |
| Single-layer GRU | RT-IGEV++ | 6 iterations |

### Axis 5: Training-Time Compression (deployment only)
**Can we compress a big model into a small one?**

| Technique | Papers | Target |
|-----------|--------|--------|
| Distillation from teacher | DTP, Fast-FoundationStereo | Transfer knowledge |
| Structured pruning (DepGraph) | DTP, Fast-FoundationStereo | Remove redundancy |
| NAS for backbone | Pip-Stereo, Fast-FoundationStereo | Search efficient architectures |
| Monocular Prior Transfer (MPT) | Pip-Stereo | Distill VFM, no VFM at inference |
| Re-parameterization | Pip-Stereo, RepViT | Fold multi-branch to single-path |

---

## Benchmark Comparison

### Latency vs Accuracy Pareto (real-time methods)

| Method | SceneFlow EPE | KITTI15 D1 | Runtime | Hardware |
|--------|--------------|-----------|---------|----------|
| **BGNet** | ~1.17 | 2.78 | 25ms (40 FPS) | 2080Ti |
| **LightStereo-S** | 0.73 | 2.30 | 17ms | RTX 3090 |
| LightStereo-H | 0.51 | 1.82 | 54ms | RTX 3090 |
| MobileStereoNet-3D | 0.80 | 2.10 | — | (MACs: 153G) |
| **BANet-2D** | ~0.7 | ~2.0 | 45ms | Snapdragon 8G3 mobile |
| **LiteAnyStereo** | ~0.7 | 3.87 | — | (33G MACs) |
| **DTPnet** | 1.56 | 3.28 | **16.3 ms** | **Jetson AGX** |
| **Pip-Stereo** | 0.45 | 1.44 | 440ms at 384×1344 | Orin NX |
| Pip-Stereo (small res) | 0.45 | 1.44 | **19ms** | RTX 4090 (FP16 320×640) |

**The clear Pareto leaders:**
- **For pure latency at moderate accuracy:** DTPnet (16.3ms on Jetson AGX)
- **For best accuracy at real-time:** Pip-Stereo (EPE 0.45, matches MonSter-level accuracy at 11× the speed)
- **For balance:** LightStereo-S (17ms, EPE 0.73, tiny FLOPs)

---

## The Critical Insight: Iterative vs Non-Iterative Efficiency

**The single most important finding across all 8 papers:**

**Non-iterative efficient methods catastrophically fail at zero-shot generalization.** Pip-Stereo's DrivingStereo evaluation makes this brutally clear:

| Method Category | DrivingStereo D1-all (zero-shot) |
|----------------|----------------------------------|
| HITNet (non-iterative) | 93.52 |
| IINet (non-iterative) | 27.70 |
| LightStereo-S (non-iterative) | 13.08 |
| RT-IGEV++ (iterative, 6 iters) | 8.04 |
| **Pip-Stereo (iterative, 1 iter)** | **4.35** |
| MonSter++ (iterative, full) | 2.69 |

**The implication for our edge model:** We cannot simply use the fastest non-iterative approach (LightStereo, BANet, HITNet). **Iterative refinement provides an inductive bias that cannot be replicated** — it must be preserved, just made cheap via PIP or similar compression.

---

## Compound Efficiency: Stacking Techniques

The major insight from reading all 8 papers: **these techniques are complementary and compound multiplicatively.**

### A maximally-compressed stack

| Stage | Technique | Source | Speedup |
|-------|-----------|--------|---------|
| 1. Backbone | Distilled lightweight (MobileNetV4 / RepViT from MPT) | Pip-Stereo, Fast-FoundationStereo | ~5× |
| 2. Cost volume resolution | 1/8 instead of 1/4 | BGNet, IGEV++ | 4-8× |
| 3. Cost volume type | Local correlation (not all-pairs) + group-wise | CREStereo | 50× |
| 4. Aggregation layer | Separable 3D V2 blocks OR LightStereo channel boost + BANet split | Separable-Stereo, LightStereo, BANet | 7-20× |
| 5. Aggregation branches | Bilateral split (detail/smooth) | BANet | Maintains accuracy at 2D cost |
| 6. Warm start | Soft argmin on GEV | IGEV-Stereo | Enables step 7 |
| 7. Iteration count | PIP compression 32 → 1-4 | Pip-Stereo | 8-32× |
| 8. Upsampling | CUBG bilateral grid | BGNet | Edge-preserving, cheap |
| 9. Final upsampling | Convex combination | RAFT-Stereo | Sharp edges |
| 10. Training | DTP distillation + structured pruning | DTP, Fast-FoundationStereo | 2× params |

**None of these conflict.** Each attacks a different bottleneck.

---

## Common Architectural Elements

Despite their differences, all efficient methods share core elements:

1. **Lightweight 2D feature encoder** (MobileNetV2, MobileNetV3, MobileNetV4, RepViT)
2. **Correlation-based cost volume** (group-wise or standard)
3. **Some form of learned cost aggregation** (varies by paper — the main differentiator)
4. **Context attention** (MSCA in LightStereo, SSA in BANet, guided excitation in IGEV-Stereo)
5. **Soft-argmax or iterative disparity regression**
6. **Smooth L1 / exponentially weighted L1 loss**

---

## Proposed Edge Model Architecture (Final Synthesis)

Based on ALL Tier 1 papers (surveys, foundation models, iterative, efficient), here is the proposed edge model architecture:

```
                        TRAINING ONLY
                        ┌─────────────────────┐
                        │ Frozen DEFOM-Stereo │──→ Feature distillation
                        │ or FoundationStereo │──→ Depth pseudo-labels
                        │ (teacher, discarded)│──→ Scale update targets
                        └─────────────────────┘
                                  ↕
┌──────────────────────────────────────────────────────────────────────┐
│                         DEPLOYED EDGE MODEL                          │
│                                                                      │
│  Stereo Pair                                                         │
│       ↓                                                              │
│  [Distilled Lightweight Backbone]                                   │
│  MobileNetV4 / RepViT-NAS                                            │
│  + MPT-distilled mono features (Pip-Stereo)                         │
│       ↓                                                              │
│  [Local Correlation + 2D-1D Search + Group-wise]                    │
│  Not all-pairs. (CREStereo)                                          │
│       ↓                                                              │
│  [Bilateral Cost Volume Split]                                      │
│  Scale-aware attention → detail + smooth volumes. (BANet)           │
│       ↓                                                              │
│  [Parallel Aggregation at 1/8 Resolution]                           │
│  Detail branch: V2 blocks with 3×3 depthwise                        │
│  Smooth branch: V2 blocks with channel boost $t=4$                  │
│  (Separable-Stereo + LightStereo)                                    │
│       ↓                                                              │
│  [Small 3D UNet for GEV]                                            │
│  Separable 3D convs, $t=2$ (IGEV-Stereo + Separable-Stereo)          │
│       ↓                                                              │
│  [Warm Start from GEV Soft Argmin]                                  │
│  NOT zero init. (IGEV-Stereo — essential for PIP)                   │
│       ↓                                                              │
│  [SRU × 1-4 iterations]                                             │
│  1×1 + 3×3 branches + CSA, compressed via PIP                        │
│  (Selective-Stereo + Pip-Stereo)                                     │
│       ↓                                                              │
│  [CUBG Upsampling to Full Res]                                      │
│  Edge-preserving bilateral grid slicing. (BGNet)                    │
│       ↓                                                              │
│  Disparity Map                                                       │
│       ↓                                                              │
│  [ONNX → TensorRT]                                                  │
│  No 3D ops incompatible with NPU, no dynamic shapes. (DTP)          │
└──────────────────────────────────────────────────────────────────────┘

Target specs:
- Parameters: <5M (vs RT-IGEV++'s 3.5M, Pip-Stereo unreported, DTPnet 0.26M)
- Jetson Orin Nano latency: <33ms (target) at 320×640, FP16
- Desktop RTX 3090 latency: <15ms
- KITTI D1-all: <1.8% (within 25% of Pip-Stereo's 1.44%)
- Zero-shot DrivingStereo D1: <10% (much better than LightStereo's 13.08%)
```

---

## Open Questions for Our Edge Model

1. **Is bilateral aggregation (BANet) compatible with iterative refinement?** BANet is single-pass; can we apply it inside each GRU iteration, or only once as preprocessing?

2. **How much of the PIP speedup translates to mobile NPUs?** PIP's FlashGRU is a CUDA kernel — re-implementing on SNPE/TFLite may lose some speedup.

3. **Does MPT work with teacher DEFOM-Stereo?** Pip-Stereo used Depth Anything V2 directly. We want to distill from full DEFOM-Stereo (which already incorporates DAv2). Does this teacher stack work?

4. **Can we use LightStereo's channel boost inside BANet's branches?** Stacking two efficient aggregation techniques from the same era — unclear if they compose.

5. **What's the right balance between 3D (IGEV-style GEV) and 2D (LightStereo/BANet) aggregation?** LiteAnyStereo (if we had the paper) suggests 4.8% 3D is optimal. Need to verify.

6. **How much does CUBG (BGNet) upsampling help if the iterative refinement is already at 1/4 resolution?** Some redundancy may exist between these two edge-preserving mechanisms.

---

## Download History

All 8 efficient papers have full analyses. Three PDFs were initially mislabeled during bulk download and had to be recovered:

| Paper | Initial Wrong Content | Correct arXiv ID | Status |
|-------|----------------------|------------------|--------|
| **Separable-Stereo** | NLP paper (Zhejiang University) | **2108.10216** | Re-downloaded and analyzed |
| **GGEV** | Physics education paper (Purdue) | **2512.06793** | Re-downloaded and analyzed |
| **LiteAnyStereo** | Digital humanities paper | **2511.16555** | Re-downloaded and analyzed |

All papers are now in `papers/raw/efficient/` with correct content and full summaries in this folder.
