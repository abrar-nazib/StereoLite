# Panoptic Perception for Autonomous Driving: A Survey

**Authors:** Yunge Li, Lanyu Xu (Oakland University, USA)
**Venue:** ACM survey (manuscript, June 2025; ACM Computing Surveys submission, listed as 2025 but covers literature through 2024)
**Tier:** 3 (survey of multi-task panoptic perception in autonomous driving; reviews 28 panoptic networks across image, point-cloud, and multi-modal inputs; complements the Vandenhende MTL survey by focusing on the driving-specific task set: detection + lane segmentation + drivable area segmentation + depth estimation, rather than the dense-prediction PASCAL / NYUD-v2 dictionary)

---

## Core Idea
Autonomous driving traditionally compartmentalizes perception into separate detection, segmentation, and depth modules, each running its own network. Panoptic perception is the multi-task alternative: a single network with a shared backbone and neck plus multiple task-specific heads (Fig. 10, p. 15). The survey argues this is motivated by (a) robustness through cross-task signal sharing, (b) efficiency from shared computation, and (c) better downstream decision-making (Sec. 1.1, p. 2). The contribution is a structured review of 28 panoptic networks divided by input modality (image / LiDAR point cloud / multi-modal fusion), plus a head-to-head comparison of accuracy, latency, and resource utilization on BDD100K and nuScenes.

## Taxonomy / What the Survey Covers
**Modality-based categorization (Sec. 3, Fig. 1 p. 3):**
- **Image-based panoptic networks** (~16 reviewed; most common): MultiNet, DLT-Net, YOLOP, HybridNets, YOLOPv2, CenterPNets, Sparse U-PDP, BEVFormer, BEVFormer v2, BEVerse, PETRv2, M2BEV, etc. Backbones: VGG (MultiNet, DLT-Net), ResNet (YOLOP variants), CSPDarkNet / EfficientNet (HybridNets, YOLOPv2), transformer-based DETR / BEVFormer variants for BEV.
- **Point-cloud-based panoptic networks** (5 reviewed): LiDARMTL (UNet3D backbone), AOP-Net (dual-task 3D backbone + ConvMLP), LiDARFormer (VoxelNet + transformer), LiDARMultiNet (3D sparse conv), SphereFormer.
- **Multi-modal fusion panoptic networks** (BEVFusion, CALICO): project image and LiDAR features to a shared BEV space, then run task-specific heads.

**Task set covered**: 2D / 3D object detection, semantic segmentation, instance segmentation, lane segmentation, drivable-area segmentation, depth estimation, BEV map segmentation. The "panoptic" framing pulls multiple of these into one network rather than the strict instance-vs-stuff distinction used in COCO panoptic.

**Datasets and benchmarks reviewed (Sec. 2.2, Tab. 2 p. 10):** COCO, ADE20K, Mapillary Vistas, IDD, Cityscapes, BDD100K, KITTI, ApolloScape, WoodScape, Waymo Open Dataset, nuScenes, Lyft Level 5. Survey breaks each down by image vs video, weather / time-of-day diversity, available modalities (mono / stereo / fisheye / 3D LiDAR), and supported task annotations.

**Sensors covered (Tab. 1, p. 4):** monocular, stereo, fisheye, 1D / 2D / 3D LiDAR; advantages and disadvantages for each. Stereo cameras are noted as "complex algorithms required" and "consume more computing power" but as a way to "estimate depth information" while keeping RGB context.

**Architecture decomposition (Sec. 3, p. 15)**: every panoptic network has a backbone (feature extractor), neck (cross-layer fusion / FPN-style refinement), and a set of heads (one per task with task-specific loss). Section 3.2-3.4 walks through each modality's backbone choices, neck choices, and head specializations.

## Main Innovation
The survey itself is the contribution; not novel architecture. Two threads worth pulling for our work:
1. **Unified BDD100K + nuScenes comparison table** (Tabs. 10-12, pp. 25-27) of FPS / params / GFLOPs / mAP / mIoU on driving-relevant tasks for 20+ models. This is the kind of compute-vs-accuracy table that Vandenhende's survey lacked.
2. **Multi-task vs concatenated-single-task efficiency analysis** (Fig. 11, p. 28): summing the latency of three best-in-class single-task models gives ~65 ms / 18 FPS on RTX 3090, while YOLOP / HybridNets / Sparse U-PDP each do all three tasks in under 65 ms with smaller param count. Empirical confirmation that the MTL "shared backbone" wins in practice on this task dictionary.

## Headline Numbers Reported in the Survey

**Image-based panoptic models on BDD100K (Tab. 10, pp. 25-26):** FPS on RTX 3090, params (M), GFLOPs, AP50 for detection, mIoU-d for drivable area, IoU-l for lane segmentation.

| Model | FPS | Params (M) | GFLOPs | AP50 | mIoU-d | IoU-l |
|---|---|---|---|---|---|---|
| YOLOv5s (single-task) | 82.0 | 7.2 | 16.5 | 77.2 | - | - |
| ENet-SAD (single-task lane) | 50.6 | 1.0 | - | - | - | 16.0 |
| DeepLabV3+ (single-task drivable) | 23.4 | 15.4 | 30.7 | - | 90.9 | 29.8 |
| SegFormer (single-task) | 30.8 | 7.2 | 12.1 | - | 92.3 | 31.7 |
| DLT-Net (MTL) | 9.3 | - | - | 68.4 | 71.3 | - |
| MultiNet (MTL) | 8.6 | - | - | 60.2 | 71.6 | - |
| **YOLOP** (MTL) | 24.0 | 7.9 | 18.6 | 76.5 | 91.5 | 26.2 |
| **HybridNets** (MTL) | 26.0 | 12.8 | 15.6 | 77.3 | 90.5 | 31.6 |
| **YOLOPv2** (MTL) | - | 38.9 | - | 83.4 | 93.2 | 27.3 |
| CenterPNets | - | 28.6 | - | 81.6 | 92.8 | 32.1 |

YOLOPv2 (38.9 M) beats every single-task baseline on all three tasks simultaneously. HybridNets at 12.8 M / 26 FPS is the sweet spot for current edge-leaning MTL.

**3D detection on nuScenes (Tab. 11, p. 26):** mAP / NDS for camera-only (C), LiDAR-only (L), and multi-modal (C+L).
- BEVFormer (C): 0.412 / 0.520 at 68.7 G params (note: appears to be measured differently, with params listed as 68.7 but GFLOPs 1303.5).
- BEVFormer v2 (C): 0.556 / 0.634.
- LiDARFormer (L): **0.715 / 0.743** (highest mAP among reviewed).
- BEVFusion (C+L, multi-modal MTL): 0.702 / 0.729 at 8.4 FPS.

**BEV segmentation, nuScenes (Tab. 12, p. 27):** mIoU across 20+ semantic classes. LiDARFormer 81.0%, LiDARMultiNet 81.4%, SphereFormer **81.9%** lead the LiDAR-only segmentation. Camera-only PETRv2 60.3%, BEVFormer 48.7%. Multi-modal BEVFusion 62.7%.

**Real-time / efficiency analysis (Sec. 4.2, Fig. 11 p. 28):** combining three single-task models for full panoptic perception sums to ~65 ms on RTX 3090. The MTL chassis YOLOP, HybridNets, Sparse U-PDP each run all three tasks in under 65 ms with smaller total parameter footprint. **Empirical confirmation that MTL is the right paradigm for resource-constrained driving perception.**

## Multi-Task Coupling as a Phenomenon: What the Survey Concludes
Sec. 5 ("Challenges and Future Directions") is the survey's most direct statement on cross-task gradient flow. Three named challenges:

1. **Weight balance**: tasks need different magnitudes of attention. Static loss weights leave easy tasks under-optimized; dynamic weighting (GradNorm) helps but is sensitive to gradient-norm spikes (Sec. 5.2, p. 30).
2. **Task relevance**: cross-task synergy depends on the *pair*. The survey notes that pedestrian-detection and vehicle-detection benefit each other in urban scenes but diverge in rural scenes, so static "task-relevance assumptions" fail in deployment (Sec. 5, p. 29-30).
3. **Negative transfer**: the canonical MTL pathology. The survey reviews three concrete remedies in this space:
   - **Bayesian meta-learning** for balancing learning processes across varied tasks (cited from Sec. 5).
   - **M3ViT**: sparsely-activated mixture-of-experts layers inside a ViT backbone, selectively activated per task at inference. This is described as effective for mitigating gradient conflict because it physically separates parameter updates per task.
   - **Prompt-learning (VE-Prompt)**: visual exemplars as task-specific prompts that guide the shared backbone to task-relevant features. This directly maps to the TaskPrompter mechanism from a different research lineage.

The survey's overall conclusion on coupling: **MTL works in driving panoptic perception when (a) the task dictionary is well-correlated (detection + segmentation + depth share spatial structure) and (b) the architecture uses a shared backbone with task-specific heads, not a fully fused single output**. The 2D-image-based MTL table is the strongest empirical evidence: YOLOPv2 and Sparse U-PDP exceed single-task models on all three tasks simultaneously.

## Relevance to Our Project
- **YOLOP / HybridNets architecture is a direct blueprint** for a stereo + segmentation + drivable-area MTL chassis on Jetson Orin Nano-class hardware. HybridNets at 12.8 M params / 26 FPS RTX 3090 is the relevant cost reference; scaled to our YOLO26s mid-tier (2.06 M trainable), a HybridNets-paradigm stereo chassis would land plausibly under 30 ms.
- **The "shared backbone + task heads" pattern is what we should adopt** if we ever combine StereoLite with detection or lane-segmentation. The survey's empirical evidence (Fig. 11) directly supports this over separate networks.
- **BEV-based methods (BEVFormer, M2BEV) are out of scope.** They require multi-camera input, calibration, and a global ego-pose; drone work uses a single forward-facing rig.
- **Drone application directly maps.** The survey is written for ground vehicles but the task dictionary (detection + drivable-area + depth) is the right starting point for drone obstacle avoidance and corridor navigation. DroNet (RAL 2018, reviewed separately) is the historical precedent in the same space at a tenth of the parameter count.
- **Stereo cameras under-served in panoptic surveys**: the survey acknowledges stereo (Sec. 2.1.1, p. 5) but every multi-task chassis it reviews uses monocular RGB or LiDAR. There is a gap: a stereo-first panoptic chassis (depth from rectified stereo + segmentation from same encoder + light detection head) is not represented in the surveyed literature. This is a research direction.
- **GradNorm and MGDA appear in both this survey and Vandenhende's**, with consistent conclusions: useful in principle, fragile in practice, fixed grid-searched weights are usually competitive.

## Limitations
- **2024-era cutoff with weak transformer coverage.** The latest BEVFormer v2 and YOLOPv2 are included but the lineage of transformer-MTL for dense prediction (InvPT, TaskPrompter, MViT) is missing. This is a driving-perception survey, not a general MTL survey.
- **No standardized benchmark across models.** Tab. 10 numbers come from different papers with different test splits; the survey acknowledges this. FPS comparisons use RTX 3090 but most original papers reported on V100 or A100, so the FPS column has consistency issues despite the disclaimer.
- **Depth estimation is mentioned as a task but never benchmarked.** The big comparison tables (10-12) cover detection, lane, drivable-area, BEV segmentation. Depth on KITTI / DDAD / DrivingStereo would be the natural fourth column and it is absent. For stereo work, this is a gap.
- **Real-time claims are RTX 3090 figures, not edge-device.** The 65 ms / 18 FPS comparison is on a 24 GB desktop GPU. None of the panoptic models are benchmarked on Jetson Orin Nano or comparable hardware.
- **Coverage of small-network designs is thin.** The survey is biased toward higher-end multi-modal architectures; no LightStereo, MobileStereo, or comparable sub-10 M-param chassis. For edge work, this is the wrong end of the size spectrum.
- **Negative-transfer remediation is reviewed but not benchmarked.** M3ViT and VE-Prompt are described but there are no head-to-head numbers showing how much they actually reduce task interference. Conclusions on "what works" are qualitative.
