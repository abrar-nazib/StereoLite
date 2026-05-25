# AurigaNet: A Real-Time Multi-Task Network for Enhanced Urban Driving Perception

**Authors:** Kiarash Ghasemzadeh (University of Alberta / Shahid Beheshti University), Sedigheh Dehghani (Shahid Beheshti University)
**Venue:** arXiv:2024/2025 preprint (PDF dated 2026 in our corpus; venue not yet listed; code at https://github.com/KiaRational/AurigaNet)
**Tier:** 3 (YOLOP-family panoptic driving: three heads, drivable-area *instance* segmentation, object detection, lane detection; no stereo / no depth)

---

## Core Idea
AurigaNet extends the YOLOP/HybridNets recipe with one specific upgrade: turn the drivable-area output from semantic segmentation into **end-to-end instance segmentation** so that left lane vs right lane vs forward-lane drivable regions can be separated for path planning, without DBSCAN or other post-hoc clustering (Sect. 1, p. 1-2). This is achieved by adding a feature-embedding branch trained with a discriminative loss and using deformable convolutions to adapt receptive fields to irregular lane geometry, then running mean-shift on the embeddings at inference (Sect. 3.5, p. 8). The detection and lane heads are conventional YOLOv5-style.

## Architecture
- **Shared encoder (Sect. 3.1, p. 4)**: CSPDarknet backbone (same as YOLOP) + neck with SPPF and FPN. Image input 640x640.
- **Detection head (Sect. 3.2.1, p. 5)**: YOLOv5-style anchor-based multi-scale head over PAN+FPN feature maps, three anchors per cell, predicts position / size / class / confidence.
- **Drivable-area instance head (Sect. 3.2.2, p. 5)**: two parallel decoders.
  - **Binary segmentation head**: bottom FPN layer (W/8 x H/8 x C) -> one transpose convolution + multiple C3 layers -> (W/4 x H/4 x 1), per-pixel drivable-vs-background.
  - **Feature embedding head**: first upsampled feature map (W/4 x H/4 x C) -> downsample -> stack of **deformable convolutions** (Eq. learned 2D offsets per kernel location) -> (W/8 x H/8 x 8) embedding map per pixel.
- **Lane head (Sect. 3.2.3, p. 5)**: binary segmentation, same wiring as drivable-area binary head, output (W/4 x H/4 x 1).
- **Loss**: discriminative loss (clusters same-instance embeddings tightly, pushes apart different-instance embeddings) on the embedding head + standard CE on binary segmentation + YOLO detection loss. Hyperparameter table in Sect. 4.2 (p. 9-10) shows Adam lr=1e-4, 250 epochs, warmup 3 epochs, batch 16.
- **Mean-shift clustering at inference (Sect. 3.5, p. 8)**: vMF (von Mises-Fisher) mean shift on the normalised embeddings to separate drivable-area instances without a pre-set cluster count. Eq. 9 gives the kappa-weighted update.

## Main Innovation
The **end-to-end instance-segmentation drivable-area head** is the contribution. Specifically the *triplet* of (a) feature-embedding branch trained with discriminative loss, (b) deformable convolutions on the embedding branch (Fig. 6 visualises 729 sampling locations per activation, showing they cluster on instance boundaries), (c) vMF mean-shift at inference instead of pre-set-k clustering. The rest (CSPDarknet encoder, YOLOv5 detection head, binary segmentation heads) is unchanged from YOLOP / HybridNets. Sect. 1 explicitly says "AurigaNet combines two key components: (i) a discriminative loss... (ii) deformable convolutions... Together, these design choices enable robust instance-level separation".

## Key Benchmark Numbers
- **Dataset**: BDD100K, 70k train / 10k val / 20k test at 640x640 (Sect. 4.1, p. 9).
- **Training hardware**: Intel i5-13600K + RTX 4080 + 64 GB RAM (Tab. 1, p. 10).
- **Inference hardware**: RTX 4080 (desktop) and Jetson Orin NX (Tab. 5, p. 12), 6-core Cortex-A78AE + 1024-core Ampere GPU, 8 GB LPDDR5, 10-20 W power envelope.
- **Params, FPS (Tab. 4, p. 11)**:
  - AurigaNet: **9.09 M params, 217 FPS on 4080, 5.077 FPS on Orin NX (FP32)**.
  - YOLOP: 7.90 M, 362 FPS / 4.002 FPS.
  - HybridNets: 12.83 M, 139.98 FPS / 1.986 FPS.
  - All in FP32; paper notes TensorRT FP16/INT8 would give 1.5-2x speedup but is not measured.
- **Drivable area, IoU / Accuracy / mAP50 (Tab. 3, p. 11)**:
  - AurigaNet: **85.2 / 97.7 / 87.25** (mAP50 is for instance segmentation; YOLOP/HybridNets do not have this since they are semantic-only).
  - YOLOP: 84.5 / 97.3 / - .
  - HybridNets: 83.4 / 96.3 / - .
  - PSPNet: 83.5 / 94.9 / - .
- **Lane detection, IoU / Accuracy (Tab. 3, p. 11)**:
  - AurigaNet: **60.80 / 98.77**.
  - HybridNets: 31.60 / 85.40.
  - YOLOP: 26.20 / 70.5.
  - Enet-SAD: 16.02 / 36.56.
  - 30-point IoU margin over HybridNets, the previous best (paper's flagship claim).
- **Traffic object detection, mAP@0.5:0.95 / Recall (Tab. 3, p. 11)**:
  - AurigaNet: **47.6 / 75.9**.
  - HybridNets: 44.7 / 92.8.
  - YOLOP: 43.1 / 89.2.
  - YOLOv5s: 42.5 / 62.5.
  - (AurigaNet wins on mAP but loses on Recall: 75.9 vs 92.8 for HybridNets, ~17 points fewer detections.)

## Multi-Task Coupling: Load-Bearing or Decorative?
**Mixed, leaning decorative.** The paper does not isolate "joint training vs separate training" or "with/without lane head, does drivable-area accuracy change". The three heads share a CSPDarknet encoder + FPN/SPPF/PAN, which is the standard compute-amortisation argument; the loss is summed across tasks with no shown ablation of task weights. The architectural novelty (deformable conv + discriminative loss + mean-shift) lives entirely inside the drivable-area instance head, so removing it would not cost the other two heads anything, and the headline numbers for lane (60.80 IoU vs HybridNets 31.60) and detection (47.6 mAP vs 44.7) are not attributed to any cross-task synergy in the ablation. Quoting the conclusion (Sect. 5, p. 12): "the integration of a discriminative loss function and deformable convolutions has further refined the network's capabilities, particularly in complex driving scenarios". Both upgrades are *intra-head*. The "multi-task" framing is presentational; functionally this is a YOLOv5 detector + a binary-seg head + an instance-seg head, sharing an encoder.

## Relevance to Our Project
- **Most direct edge-deployment benchmark we have on Jetson Orin NX for MTL.** Tab. 5 (p. 12) and Tab. 4 (p. 11) give the only FP32 Jetson Orin NX numbers in this corpus that are comparable to our edge envelope. AurigaNet 9.09 M / 5.08 FPS, YOLOP 7.90 M / 4.00 FPS, HybridNets 12.83 M / 1.99 FPS. FP32. These FPS numbers are catastrophically below the 30 FPS minimum for drone control; they are why we need TensorRT INT8 + width reduction.
- **9.09 M is 3.6x our edge tier (2.5 M).** Even with INT8 + 2x speedup, AurigaNet on Orin NX would land at ~10 FPS, still below our target.
- **Drivable-area instance segmentation is genuinely useful for drone landing.** If we ever add an MTL head to StereoLite-mid, separating left/right/forward drivable regions (e.g. parallel roads in a parking lot) is more useful than a binary drivable-vs-not mask. The discriminative-loss + mean-shift recipe (Sect. 3.5, p. 8) is the cheapest path.
- **Deformable convolutions are a Jetson INT8 risk.** Standard TensorRT INT8 plugin support for deformable conv is partial; INT8 deployment of AurigaNet's embedding head is non-trivial. CLAUDE.md's "INT8-only ops" hard constraint would auto-drop deformable conv unless we accept FP16 for that subgraph.
- **No stereo / depth branch.** Same gap as YOLOP and YOLOPv2: this is a left-image-only chassis. Wiring StereoLite's disparity head onto the same encoder is the obvious next experiment.

## Limitations
- **No real-time on Orin NX in FP32.** 5.08 FPS is unusable. Paper acknowledges this and punts to "deployment can leverage NVIDIA TensorRT to quantize models to FP16 or INT8, typically yielding 1.5 to 2x higher throughput" (Sect. 4.4.4, p. 11), but does not measure it. The "real-time" in the title refers to RTX 4080 at 217 FPS.
- **Recall regression on detection.** 75.9 vs HybridNets 92.8 is a 17-point Recall drop in exchange for +2.9 mAP. For safety-critical autonomous driving, lower Recall is the wrong direction.
- **No cross-domain eval.** Trained and tested on BDD100K only; no zero-shot Cityscapes / nuImages / KITTI.
- **No ablation of the discriminative loss vs deformable conv vs mean-shift.** The three contributions are bundled and never disentangled, so we cannot tell which one carries the +30-point lane IoU.
- **Deformable convolution computational cost is not reported.** GFLOPs are absent from the paper; only params and FPS are given.
- **Inference-side mean-shift adds latency.** vMF mean shift on (W/8 x H/8 x 8) embeddings for every drivable instance has unspecified cost; the 5.08 FPS Orin NX number presumably includes it, but the iteration count and tolerance are not stated.
