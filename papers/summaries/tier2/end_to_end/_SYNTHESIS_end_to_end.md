# Synthesis: End-to-End Stereo Architectures (Tier 2)

> 9 papers covering the evolution from first learned matching costs to the culmination of the 3D cost volume era (2015-2022).

---

## Papers Covered

| # | Paper | Year | Core Contribution |
|---|-------|------|-------------------|
| 1 | **MC-CNN** | 2016 | First learned matching cost (CNN + classical SGM) |
| 2 | **DispNet-C** | 2016 | First end-to-end 2D network + SceneFlow dataset |
| 3 | **GC-Net** | 2017 | First 4D concatenation cost volume + soft argmin |
| 4 | **PSMNet** | 2018 | SPP + stacked hourglass 3D CNN |
| 5 | **GA-Net** | 2019 | SGA/LGA layers replace 3D convs (100× cheaper) |
| 6 | **GWCNet** | 2019 | Group-wise correlation — the modern standard |
| 7 | **AANet** | 2020 | Deformable ISA + CSA — eliminate 3D convs entirely |
| 8 | **CFNet** | 2021 | Cross-domain generalization via cascade + uncertainty |
| 9 | **ACVNet** | 2022 | Attention-filtered concatenation volume (end of 3D era) |

---

## The Evolutionary Timeline

```
2016  MC-CNN          Learned cost + SGM (no end-to-end)
  ↓
2016  DispNet-C       End-to-end 2D network + correlation layer
  ↓                   + SceneFlow dataset (universal pretraining)
  ↓
2017  GC-Net          First 3D concatenation cost volume
  ↓                   + soft argmin (still universal)
  ↓
2018  PSMNet          SPP + stacked hourglass 3D CNN
  ↓                   (the 2018-2021 baseline)
  ↓
2019  GA-Net          SGA/LGA replace 3D convs
  ↓   GWCNet          Group-wise correlation (still dominant)
  ↓
2020  AANet           Deformable 2D aggregation (no 3D convs)
  ↓
2021  CFNet           Cross-domain + cascade + uncertainty
  ↓
2022  ACVNet          Attention-filtered concatenation (era ceiling)
  ↓
  RAFT-Stereo takes over → iterative paradigm begins
```

---

## Key Architectural Evolution

### Cost Volume Type (the defining design axis)

| Era | Type | Examples | Trade-off |
|-----|------|----------|-----------|
| **2016** | Patch similarity (not a volume) | MC-CNN | Flexible but slow, requires SGM |
| **2016** | 1D correlation | DispNet-C | Fast, efficient, less expressive |
| **2017** | 4D concatenation | GC-Net, PSMNet, GA-Net | Rich but expensive |
| **2019** | Group-wise correlation | **GWCNet** | Best balance — still standard |
| **2022** | Attention-filtered concat | ACVNet | Ceiling of the 3D era |

### Cost Aggregation Strategy

| Method | Strategy | Cost |
|--------|----------|------|
| GC-Net, PSMNet | Dense 3D CNN hourglass | Very high |
| GA-Net | SGA/LGA differentiable SGM | ~1/100 of 3D conv |
| AANet | Deformable ISA + HRNet CSA | ~1/130 of 3D conv |
| ACVNet | Attention-filtered + shallow 3D | Much lighter than PSMNet |

---

## The Three Fundamental Techniques That Survived

Every modern method still uses these:

1. **Soft argmin regression** (from GC-Net 2017):
   $$\hat{d} = \sum_d d \cdot \sigma(-c_d)$$

2. **Group-wise correlation** (from GWCNet 2019): used by IGEV-Stereo, IGEV++, BANet, LightStereo, CREStereo, and every modern method

3. **SceneFlow pretraining** (from DispNet-C 2016): universal pretraining dataset even in 2025

---

## Historical Patterns

**Pattern 1: Each era's "bottleneck" became the next era's innovation target**
- 2016-17: matching cost is hand-crafted → learn it (MC-CNN)
- 2017-18: 3D convs are expensive → reduce them (GA-Net, AANet)
- 2019-21: concatenation volumes are expensive → use group-wise correlation (GWCNet)
- 2022: all the above → attention-filter the volume (ACVNet)

**Pattern 2: The 3D convolution "bottleneck" was NOT fundamental**
- GA-Net (2019): replaced 3D convs with SGA/LGA — worked
- AANet (2020): eliminated 3D convs entirely with deformable 2D — worked
- But RAFT-Stereo (2021) ultimately won by **avoiding** the cost volume paradigm altogether

**Pattern 3: Cross-domain generalization emerged as a distinct objective**
- 2016-2020: accuracy-racing on KITTI leaderboard
- 2021 (CFNet): **first paper to explicitly target cross-domain**, won RVC 2020
- Post-2021: generalization becomes standard evaluation criterion

---

## Relevance to Our Edge Model

### Techniques to Inherit

| Technique | From | Why |
|-----------|------|-----|
| **Group-wise correlation** | GWCNet | Best cost volume / compute ratio |
| **SPP feature extraction** | PSMNet | Multi-scale context nearly free |
| **SGA/LGA differentiable SGM** | GA-Net | Replaces expensive 3D convs |
| **Deformable ISA + CSA** | AANet | Content-adaptive aggregation |
| **Cascade + uncertainty** | CFNet | Coarse-to-fine with confidence |
| **Attention-filtered volume** | ACVNet | Reduce aggregation burden |

### Techniques to Avoid

- **Full 4D concatenation volumes** (GC-Net/PSMNet) — too expensive
- **Dense 3D CNN hourglass** (PSMNet) — 25+ 3D conv layers too slow
- **Fixed disparity range baked into architecture** (all pre-2021) — inflexible

### Key Insight

**ACVNet represents the ceiling of the 3D cost volume era** — after it, RAFT-Stereo's iterative paradigm took over and subsequent efficient models (LightStereo, BANet, Pip-Stereo) built on the iterative approach. Our edge model should **NOT** try to extend the 3D cost volume line further; instead, use the insights from this era (group-wise correlation, efficient aggregation, attention filtering) as **components** within an iterative framework.
