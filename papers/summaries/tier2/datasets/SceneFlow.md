# Scene Flow Datasets (FlyingThings3D, Monkaa, Driving)

**Authors:** Nikolaus Mayer, Eddy Ilg, Philip Hausser, Philipp Fischer, Daniel Cremers, Alexey Dosovitskiy, Thomas Brox
**Venue:** CVPR 2016 (introduced with DispNet-C)
**Tier:** 2 (THE universal pretraining dataset)

---

## Dataset Overview

| Property | Value |
|----------|-------|
| **Scene type** | **Synthetic** — three subsets from different Blender animations |
| **Size** | **~35,454 stereo pairs total** across subsets |
| **Resolution** | 960×540 |
| **GT acquisition** | **Synthetic rendering** — perfect pixel-accurate ground truth |
| **GT density** | Dense |
| **Includes** | Disparity, optical flow, disparity change (scene flow) |

### Three Subsets

| Subset | Size | Content |
|--------|------|---------|
| **FlyingThings3D** | ~27K pairs | **Abstract** — flying random 3D objects against textured backgrounds |
| **Monkaa** | ~8K pairs | **Animated movie** — cartoon animals and characters |
| **Driving** | ~4.3K pairs | **Driving simulation** — virtual streets |

## Main Characteristics
- **Completely synthetic** — rendered from Blender animations
- **Dense ground truth** for disparity (and optical flow, scene flow)
- **Huge variety** across the three subsets
- **Standard pretraining target** before fine-tuning on real data

## Evaluation Metrics
- **EPE (End-Point Error)** — primary metric
- **bad-1 / bad-3:** percentage of pixels with error > N pixels
- **Usually reported on the FlyingThings3D test split**

## Role in the Ecosystem
**THE universal pretraining dataset for stereo matching, used by literally every modern method.** The standard workflow is:

1. **Pretrain on Scene Flow** (long training, learn general matching)
2. **Fine-tune on target dataset** (KITTI, Middlebury, ETH3D, etc.)

**Without Scene Flow pretraining, stereo networks generalize poorly.** This pattern is used by:
- MC-CNN, DispNet-C, GC-Net, PSMNet, GA-Net, GWCNet, AANet, CFNet, ACVNet, CREStereo
- RAFT-Stereo, IGEV-Stereo, IGEV++, Selective-Stereo, MoCha-Stereo
- DEFOM-Stereo, FoundationStereo, MonSter, Stereo Anywhere, BridgeDepth
- Pip-Stereo, LiteAnyStereo, GGEV

**Every single one.** Scene Flow is the de facto standard.

## Why It Works (Despite Being Synthetic)
- **Huge scale** — 35K pairs provides enough diversity
- **Dense ground truth** — 100% pixel coverage vs KITTI's ~50%
- **FlyingThings3D's abstract objects** force the network to learn general matching (not memorize scenes)
- **Already normalized** — the network learns the core correspondence problem

## Relevance to Our Edge Model
**Mandatory.** Our edge model will:
- **Pretrain on Scene Flow** as the first training stage
- **Report Scene Flow EPE** as a standard metric
- **Use it for ablations** — faster than fine-tuning cycles on KITTI

Target: **Scene Flow EPE < 0.6** would be competitive with LightStereo-M and RT-IGEV++ levels. With distillation from DEFOM-Stereo teacher, **EPE < 0.5** is achievable.

**Note:** Scene Flow alone is insufficient for SOTA zero-shot generalization — following StereoAnything's approach, we should mix Scene Flow with:
- CREStereo synthetic dataset
- TartanAir
- Pseudo-stereo from monocular images (DepthAnythingV2 + RealFill)
