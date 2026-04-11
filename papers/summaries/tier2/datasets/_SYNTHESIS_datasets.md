# Synthesis: Stereo Matching Datasets (Tier 2)

> 8 datasets that define the stereo matching evaluation landscape.

---

## Datasets Covered

| # | Dataset | Year | Type | Primary Role |
|---|---------|------|------|--------------|
| 1 | **KITTI 2012** | 2012 | Real outdoor driving | First driving benchmark |
| 2 | **KITTI 2015** | 2015 | Real outdoor driving | **Primary stereo benchmark** |
| 3 | **Middlebury v3** | 2014 | Real indoor | High-res precision + cross-domain |
| 4 | **ETH3D** | 2017 | Mixed (grayscale) | Multi-view + two-view |
| 5 | **Scene Flow** | 2016 | Synthetic | **Universal pretraining** |
| 6 | **DrivingStereo** | 2019 | Real driving (large-scale) | Weather variation |
| 7 | **Booster** | 2022 | Real indoor | Non-Lambertian surfaces |
| 8 | **Spring** | 2023 | Synthetic (high-res) | Modern high-detail benchmark |

---

## Coverage Matrix

| Property | KITTI12/15 | Middlebury | ETH3D | Scene Flow | Driving Stereo | Booster | Spring |
|----------|:---------:|:----------:|:-----:|:----------:|:--------------:|:-------:|:------:|
| **Real images** | ✅ | ✅ | ✅ | ❌ | ✅ | ✅ | ❌ |
| **Outdoor** | ✅ | ❌ | ✅ (partial) | ✅ (Driving subset) | ✅ | ❌ | ✅ |
| **Indoor** | ❌ | ✅ | ✅ (partial) | ✅ (Monkaa) | ❌ | ✅ | ✅ |
| **High resolution** | ❌ | ✅ | ✅ | ❌ | ❌ | ✅ | ✅ |
| **Large disparity** | ✅ | ✅ | ❌ (small) | ✅ | ✅ | ❌ | ✅ |
| **Weather variations** | ❌ | ❌ | ❌ | ❌ | ✅ | ❌ | ❌ |
| **Non-Lambertian** | ❌ (partial) | ❌ (partial) | ❌ | ❌ | ❌ | ✅ | ❌ |
| **Large-scale** | ❌ | ❌ | ❌ | ✅ | ✅ | ❌ | ❌ |
| **Dense GT** | ❌ (LiDAR) | ✅ (SL) | ✅ (laser) | ✅ (synth) | ✅ (accum) | ✅ (SL) | ✅ (synth) |

## The Standard Benchmarking Workflow

Every modern stereo paper follows this pattern:

1. **Pretrain on Scene Flow** (~200K iterations, dense ground truth, diverse synthetic data)
2. **Fine-tune on target dataset** (KITTI 2012/2015 most common)
3. **Evaluate zero-shot cross-domain** on Middlebury + ETH3D (without finetuning)
4. **Optional:** report Booster for non-Lambertian, DrivingStereo for weather

## Evaluation Metrics Summary

| Dataset | Primary Metric | Threshold |
|---------|---------------|-----------|
| KITTI 2012 | **Out-Noc 3px** / EPE | 3 pixel |
| KITTI 2015 | **D1-all** | > 3px AND > 5% of GT |
| Middlebury | **bad-1** / bad-2 | 1 or 2 pixel |
| ETH3D | **bad-1** | 1 pixel |
| Scene Flow | **EPE** | — |
| DrivingStereo | EPE / D1 | 3 pixel |
| Booster | **bad-2 per class** | 2 pixel |
| Spring | EPE / outliers | varies |

**KITTI uses 3px threshold** (sparse LiDAR GT forgives small errors).
**Middlebury/ETH3D use 1-2px** (much stricter, subpixel accuracy).

## Cross-Domain Generalization: The Modern Gold Standard

The **Robust Vision Challenge (RVC)** requires training on Scene Flow only and testing on KITTI + Middlebury + ETH3D with a **single model, no fine-tuning**. This is the most important zero-shot evaluation:

- **CFNet (2021)** won RVC 2020
- **CREStereo++ (2023)** won RVC 2022
- **FoundationStereo / DEFOM-Stereo (2025)** dominate modern zero-shot

Our edge model should target **competitive RVC numbers** as the primary generalization metric.

---

## Scale Comparison

| Dataset | # Training Pairs | Notes |
|---------|-----------------|-------|
| Middlebury v3 | ~30 | Tiny but high quality |
| ETH3D | 27 | Very small |
| KITTI 2012 | 194 | Small |
| KITTI 2015 | 200 | Small |
| **Scene Flow** | **35,454** | The pretraining standard |
| **DrivingStereo** | **182,000+** | Largest driving dataset |

**Modern training strategies** (StereoAnything, Pip-Stereo's MPT) combine:
- Scene Flow (synthetic diversity)
- TartanAir (large-scale synthetic)
- CREStereo synthetic dataset
- DrivingStereo (large-scale real)
- **Pseudo-stereo from monocular images** (DepthAnythingV2 + inpainting, 53M+ pairs)

---

## Relevance to Our Edge Model

### Mandatory Benchmarks
1. **KITTI 2015 D1-all** — THE primary metric every paper reports
2. **Scene Flow EPE** — standard pretraining benchmark
3. **Middlebury + ETH3D** — zero-shot cross-domain evaluation
4. **DrivingStereo weather** — robustness validation (critical for ADAS deployment)

### Optional But Valuable
5. **Booster** — non-Lambertian surface handling (demonstrates monocular prior integration)
6. **KITTI 2012** — secondary driving benchmark
7. **Spring** — high-resolution stress test (only if our edge model supports 2K input)

### Training Data Strategy

Following **StereoAnything's curriculum**, our edge model should train on:
```
Scene Flow (35K)
  +
CREStereo synthetic (~200K)
  +
TartanAir (~1M)
  +
Pseudo-stereo from monocular (DepthAnythingV2 + RealFill, ~53M)
  +
DrivingStereo labeled (~180K)
```

**No need** to include Middlebury or ETH3D in training — keep them as held-out zero-shot tests.

### Target Benchmark Numbers

| Dataset | Target | Tier |
|---------|--------|------|
| Scene Flow EPE | < 0.6 | Competitive |
| **KITTI 2015 D1-all** | **< 2.0%** | **Real-time SOTA** |
| Middlebury bad-2 (zero-shot) | < 12% | Top tier |
| ETH3D bad-1 (zero-shot) | < 3% | Top tier |
| DrivingStereo weather D1 | < 10% | Real-time SOTA |
| Booster bad-2 All | < 20% | Foundation-informed |

**Achieving all of these simultaneously on Jetson Orin Nano at <33ms is our edge model goal.**
