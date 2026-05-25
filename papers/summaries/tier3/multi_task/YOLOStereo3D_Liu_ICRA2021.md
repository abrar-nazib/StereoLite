# YOLOStereo3D: A Step Back to 2D for Efficient Stereo 3D Detection

**Authors:** Yuxuan Liu, Lujia Wang, Ming Liu (HKUST, CAS-SIAT)
**Venue:** IEEE ICRA 2021
**Tier:** 3 (stereo + 3D object detection; deliberately monocular-detector-flavored, sidesteps 3D-volume reasoning for edge-class efficiency)

---

## Core Idea
Every other stereo 3D detector (DSGN, LIGA, PLUMENet, the PL family) climbed up the volumetric ladder; YOLOStereo3D explicitly **steps back to 2D**, treating stereo 3D detection as a *monocular 3D detector enhanced with stereo features*. The pipeline is a M3D-RPN-style one-stage 2D anchor predictor (12 regressed values per anchor: 2D bbox + 3D center + 3D size + 2-channel orientation) augmented by a lightweight stereo-matching module that fuses correlation-based cost volumes at scales 1/4, 1/8, 1/16 into the left feature stream. Trains on **one** 1080Ti, runs at **>10 FPS**.

## Architecture
- **Backbone**: ResNet-34 Siamese, shared weights (Sect. IV-A, p. 5).
- **Input**: 288 x 1280 (top 100 pixels of KITTI image cropped off; speeds training/inference).
- **Lightweight cost volume (Sect. III-B.1, p. 3)**: normalized dot-product (correlation) cost volume, not concatenation. For input feature maps [1, 64, 72, 320], correlation CV forward pass takes ~7 ms vs ~200 ms for concat-based CV on a 1080Ti. Output shape [B, max_disp, H, W] (no extra channel dim like concat-CV).
- **Densely Connected Ghost Module (Sect. III-B.2, p. 3-4)**: Han et al.'s GhostNet "cheap operation" (depthwise conv) generates extra feature maps. YOLOStereo3D dense-concatenates the original input with the ghost-module output to triple the channels before downsampling. Rebalances channel budget between thin stereo features and thicker semantic features.
- **Hierarchical multi-scale stereo fusion (Sect. III-B.3, p. 4)**: Three light cost volumes:
  - 1/4 scale: max_disp 96, correlation-based.
  - 1/8 scale: max_disp 192, correlation-based.
  - 1/16 scale: concatenation-based CV (smaller, preserves more semantic info from right image after 1x1 channel reduction).
  Each CV is fed into a Ghost module, downsampled, concatenated with the smaller-scale features.
- **Detection head**: M3D-RPN one-stage 2D anchor head with statistical priors. Each anchor regresses [x2d, y2d, w2d, h2d, cx, cy, z, w3d, h3d, l3d, sin(2 alpha), cos(2 alpha)] + a classification channel for |alpha| > pi/2 (Sect. III-A.1, p. 3). 3D priors computed per-class from training-set statistics.
- **Anchor filtering**: dense anchors are projected to 3D with mean depth z; anchors far from the ground plane are filtered during training (Fig. 2, p. 4).
- **Auxiliary disparity supervision (Sect. III-C.1, p. 4)**: a decoder upsamples final stereo features to W/4 x H/4 and predicts disparity. Supervised by **sparse** OpenCV block-matching disparities (not LiDAR!) during training only. Disabled at inference. Uses stereo focal loss (Eq. for L_SF, p. 4-5) with sigma=0.5.
- **Final loss**: focal loss (classification) + smooth-L1 (regression) + stereo focal loss (auxiliary disparity).
- **NMS**: standard (not specified explicitly).

## Main Innovation
The paper's central methodological argument is **"don't build a 3D cost volume, build small 2D correlation cost volumes and concat them across scales into a 2D head"**. This pulls inference time down 10x to 50x vs DSGN. The supporting tricks are: (a) **densely-connected Ghost modules** rebalance the thin stereo feature (low-D output of correlation CV) against the thicker monocular feature stream so the network does not "skew toward monocular features"; (b) **hierarchical fusion across 1/4, 1/8, 1/16 with different max_disp at each scale** (96, 192, small concat-CV) gives multi-scale stereo cues without the cost of a single deep 3D CV; (c) **OpenCV block-matching disparity as the auxiliary GT**, intentionally coarse, since the paper shows the network only needs slight regularization to avoid collapsing to a monocular optimum (Sect. V-A.4, p. 6).

## Key Benchmark Numbers
**Latency**: **~80 ms / pair on a single 1080Ti** including file I/O (Sect. IV-A, p. 5). Roughly **12.5 FPS**.

**Hardware footprint**: trained on a SINGLE NVIDIA 1080Ti GPU; ~7 GB GPU memory at batch=4 (Sect. IV-A, p. 5). Paper emphasizes "significantly less than other SOTA stereo detection algorithms."

**Parameters / GFLOPs**: Not reported as absolute numbers. ResNet-34 baseline + thin correlation-CV stack would be in the low single-digit M trainable range, but no explicit count.

**KITTI test, Car, AP3D IoU=0.7 (Tab. I, p. 5):** Easy 65.68, Moderate 41.25, Hard 30.42. Time 0.08 s.
- vs DSGN: 73.50 / 52.18 / 45.14 at 0.67 s (DSGN beats by +10.93 AP Mod at 8.4x the latency).
- vs RT3DStereo (real-time stereo): 29.90 / 23.28 / 18.96 at 0.08 s (YOLOStereo3D beats by +17.97 AP Mod at same latency).
- vs Pseudo-LiDAR++: 61.11 / 42.43 / 36.99 at 0.40 s (comparable Mod AP, 5x faster).

**KITTI test, Pedestrian, AP3D IoU=0.5 (Tab. II, p. 5):** Easy 28.49, Mod 19.75, Hard 16.48. **Beats DSGN** (20.53 / 15.55 / 14.15) here despite simpler architecture.

**KITTI test, Cyclist, AP3D IoU=0.5:** Not reported.

**KITTI test, BEV AP**: Not directly tabulated as a separate column; only 3D AP is the primary metric.

**KITTI val ablation (Tab. IV, p. 6, Car AP3D IoU=0.7):**
- Full YOLOStereo3D: 72.06 / 46.58 / 35.53.
- w/o disparity supervision: 62.58 / 39.09 / 30.34 (-9.5 AP Easy). Disparity supervision is the single most important component.
- w/o Channel Expand (no Ghost module): 64.16 / 39.96 / 30.02 (-7.9 AP Easy).
- w/o anchor prior: 65.09 / 41.38 / 30.90.
- w/o scale-16 CV: 68.64 / 44.54 / 33.95.
- w/o scale-8 CV: 70.80 / 45.71 / 35.86.

**Stereo EPE / disparity error**: Not reported. The auxiliary disparity GT is OpenCV block-matching (coarse, sparse), so the paper doesn't claim its disparity output is a useful product; "the disparity estimation branch is disabled to improve efficiency" at inference.

**Monocular variant (Tab. III, p. 5)**: YOLOMono3D (same backbone, no right image, no fusion module) gives 19.24 / 12.37 / 8.67 AP3D at 0.05 s. Stereo gives +33% AP for +60% latency.

## Joint-Task Coupling: Stereo + Detection in One Net or Two?
**One network, joint training, but stereo is a feature-side enhancement rather than a parallel task.** The disparity branch is auxiliary; it exists at training time and gets disabled at inference (Sect. III-C.1, p. 4: "During evaluation and testing, this disparity estimation branch is disabled to improve efficiency"). The stereo features are FUSED into the 2D detection branch via the multi-scale Ghost modules, not consumed as a separate prediction. Tab. IV ablation: **without disparity supervision**, AP3D drops 9.5 AP Easy / 7.5 AP Moderate / 5.2 AP Hard. So stereo supervision is critical for training, but the network does not output disparity as a deliverable. The author argues this is the point: "the network may not be guided to produce local features useful in stereo matching... could be trapped in a local minimum similar to that of a monocular detection network" (Sect. III-C.1, p. 4). Diagnostically: stereo's role here is **a regularizer that prevents collapse to a monocular optimum**, not a parallel prediction. This is a substantially weaker form of coupling than DSGN/LIGA/PLUMENet, but it's a deliberate engineering tradeoff for 12 FPS.

## Relevance to Our Project
- **Closest spiritual neighbor to StereoLite in this paper set.** Single-GPU training (1080Ti), real-time (~12 FPS), lightweight cost volume, 2D-centric design. The architectural philosophy ("step back to 2D, use stereo as feature enhancement, not as volumetric reasoner") is exactly the StereoLite philosophy.
- **Correlation CV >> concat CV for edge.** YOLOStereo3D's measurement (200 ms vs 7 ms on 1080Ti for the same input shape) is consistent with our own observation that StereoLite's 1/16 group-wise correlation CV is cheap. The trick of running correlation at multiple scales (1/4, 1/8, 1/16) is what we already do in tile-refinement; YOLOStereo3D shows the same idea works for detection.
- **Auxiliary supervision idea is exportable.** OpenCV block-matching as a "free" disparity GT is exactly the recipe we want for fast iteration: no human/LiDAR annotation, no FoundationStereo teacher pass needed, just a CPU stereo block-matcher. For drone bbox training without ground-truth depth, we could use OpenCV block matching as a regularizer on the disparity branch while a bbox head learns separately.
- **3D bbox over class map for drone planning.** YOLOStereo3D outputs full 7-DoF 3D bboxes (x, y, z, w, h, l, theta) at >10 FPS. This is the right output for navigation planning. Segmentation gives pixel labels, which a planner can't directly use to decide "is there a tree branch 3 m to my left at height 4 m".
- **Anchor-based 3D priors specific to autonomous-vehicle scenes.** The per-class 3D mean (h=1.56, w=1.6, l=3.9 for KITTI cars) hard-codes vehicle priors. For drone obstacle classes (other drones, branches, buildings) we'd need new priors, and most drone obstacles are not as well-separable by hand-tuned priors as KITTI cars are.
- **Different from segmentation-based perception how?** Two ways: (a) instance-level output (each detection is one 3D bbox with a class label, no per-pixel grouping required), (b) explicit 3D geometry baked into the anchors. Segmentation gives "class probability at pixel (u,v)" and demands the downstream stack to group + localize in 3D itself.
- **Edge-deployment realism check.** 80 ms on 1080Ti maps to roughly 250 to 400 ms on Orin Nano without TensorRT INT8. Likely 60 to 120 ms on Orin Nano with INT8. So YOLOStereo3D-class detection is within striking distance of our 60 ms budget on mid-tier silicon; it is the only one of these five papers where that statement holds.

## Limitations / What This Paper Doesn't Solve
- **Asymmetric stereo treatment.** Sect. VI, p. 7 admits: "Information loss in the right image is significant. As a result, when an object is occluded in the left image but is more visible in the right image, the model could be significantly sub-optimal."
- **Performance gap vs DSGN/LIGA at the high end.** 41.25 Mod AP3D vs DSGN 52.18 vs LIGA 64.66. There is a real accuracy ceiling for the 2D-centric design. The author's framing is that this is a deliberate accuracy/speed trade-off, but the gap is large.
- **No cross-domain eval.** KITTI-only. Like all the other 2020-2021 stereo-3D-det papers, no zero-shot Middlebury / ETH3D / Nuscenes. Brittle to camera baselines other than KITTI's 54 cm.
- **No disparity at inference.** The auxiliary disparity branch is disabled at inference, so the network produces 3D bboxes but no dense depth map. For a drone application that needs both "where are objects" and "how far is everything else (e.g., walls, branches)", you'd need a separate stereo network running in parallel.
- **Block-matching as disparity GT is a known weak signal.** The paper says it works because the network "only needs slight regularizations", but this likely caps the upper bound of the disparity branch's usefulness. Replacing OpenCV BM with FoundationStereo pseudo-disparity would change the answer.
- **Anchor-based heads age poorly.** Modern object detectors are anchor-free (FCOS, CenterPoint, DETR). YOLOStereo3D's heavy use of per-class statistical anchor priors is brittle to dataset shifts and to new classes.
- **Parameters / GFLOPs / model size not quantified.** Cannot directly compare param budget to StereoLite / LiteAnyStereo without re-instantiating the code.
