# Yolo-SGN: Binocular Stereo Vision-Based Relative Positioning Algorithm for Drone Swarm

**Authors:** Qing Cheng, Yazhe Wang (Civil Aviation Flight University of China, Guanghan)
**Venue:** Scientific Reports (Nature) 2025, 15:3402, DOI 10.1038/s41598-025-86981-1
**Tier:** 3 (lightweight YOLOv5s variant + classical ORB-based stereo triangulation for UAV-to-UAV localization)

---

## Core Idea
For UAV swarm formation flying, relative 3D positioning has to run in ~10 ms on embedded hardware (Jetson TX2-class). The authors argue the bottleneck is **not** the stereo matching algorithm itself; classical ORB on a small bounding-box region is already fast enough; but the per-frame cost of running a full-image keypoint matcher across the entire stereo pair. Their solution: run a tiny YOLO detector first to localize the target UAV in both images, then run ORB *only inside those boxes*. The detector (Yolo-SGN) is a lightweight YOLOv5s with SF-GS backbone + C3Ghost neck + NextConv (7x7 depthwise) blocks. **Important context for this summary group: Yolo-SGN itself is a single-task object detector**, not a YOLOP/HybridNets-style shared-encoder multi-head network. The "multi-task" framing here is system-level; detection plus classical stereo triangulation; not joint deep-learning training.

## Architecture
- **Backbone (replaces YOLOv5s CSPDarknet):** "SF-GS"; SF blocks built from channel split + 3x3 grouped conv + channel shuffle (inspired by ShuffleNet and VOVNet) interleaved with GS downsampling blocks (max-pool + 1x1 conv on one branch, 3x3 stride-2 conv on the other, concatenated). Pages 4 to 5.
- **Neck:** PANet, with **C3 modules replaced by C3Ghost** (Ghost convolution from GhostNet; primary 3x3 conv produces half the features, the other half via cheap 1x1 / 3x3 / 5x5 linear transforms). Page 5 to 6.
- **Head:** standard YOLOv5s anchor-based detection head, **three detection scales**. Each Conv in the neck pathway is replaced with **NextConv** (7x7 depthwise + 1x1 pointwise, ConvNeXt-style). Page 7.
- **Loss:** **EIoU** (Efficient IoU) replaces CIoU for the box term; splits the aspect-ratio penalty into width-difference and height-difference relative to the minimum bounding rectangle, avoiding CIoU's gradient-direction conflict. Pages 8 to 9.
- **Stereo backend:** ORB (FAST corners + BRIEF descriptors + Hamming match + RANSAC outlier removal) applied **only inside the Yolo-SGN bounding boxes** in both rectified left and right images, followed by triangulation using calibrated baseline (~230 mm) and focal length (~525 px).
- **Input:** 640x640 detection input; 1280x720 native stereo capture.

## Main Innovation
**System-level coupling of a tight detection model with a sparse classical stereo matcher**, plus three orthogonal lightening tricks for the detector (SF-GS backbone + C3Ghost neck + NextConv 7x7-depthwise blocks), unified by EIoU loss. The detector is not architecturally novel on its own; every block is borrowed from ShuffleNet / GhostNet / ConvNeXt / VOVNet; but the **65.5% parameter reduction and 62.7% FLOPs reduction versus YOLOv5s with a simultaneous +1.8% mAP gain** (Table 4 versus baseline; abstract) demonstrates that careful block-by-block substitution beats a from-scratch lightweight design. Pairing this with ORB-in-bbox cuts feature-matching cost to **~1/4** of full-frame ORB (Table 5, page 17).

## Key Benchmark Numbers
- **Params:** **2.42 M** (Table 3, page 15 and Table 4, page 15); vs YOLOv5s baseline 7.02 M (-65.5%).
- **FLOPs:** **5.9 GFLOPs**; vs YOLOv5s 15.8 G (-62.7%).
- **Detection accuracy on Drone-Fly dataset:** **mAP@0.5 = 0.916, mAP@0.5:0.95 = 0.668** (Table 4, page 14). Beats YOLOv5s (0.898 / 0.592), YOLOv4 (0.881 / 0.598), YOLOv3 (0.855 / 0.563), Faster R-CNN (0.884 / 0.572). Slightly below YOLOv8n on mAP@0.5 (0.917) but +0.027 on mAP@0.5:0.95.
- **Compared to prior author's YOLOv5s-ngn:** Yolo-SGN gives +0.031 on mAP@0.5:0.95 with comparable params (2.42 M vs 2.39 M, +0.03 M).
- **Stereo matching latency (Table 5, page 17):** ORB alone: 100 point-pair matches, **28 ms**. Yolo-SGN + ORB: 7 matches inside the bbox, **7 ms** (~4x speedup).
- **Embedded latency (Table 7, page 17):** on Jetson TX2 (dual Denver 2 + quad A57 + 256-core Pascal GPU, 7.5/15 W): **25 FPS, 40 ms per frame, 50% CPU, 1100 MB memory** end-to-end (detection + ORB + triangulation).
- **Localization accuracy:** static UAV at 5 m, **max distance error 3.21%, mean 1.52%** (page 18); dynamic lateral fly-by at 10 m, mean errors X 3.8% / Y 2.1% / Z 1.7%.
- **Training data:** Drone-Fly, a ~30,000-image custom dataset combining the Det-Fly dataset + public sources + self-collected drone imagery; 70/30 train/val split; 1280x720 resolution.

## Multi-Task Coupling: Load-Bearing or Decorative?
**Not applicable in the YOLOP sense; this is not a multi-task deep-learning architecture.** Yolo-SGN performs **single-task object detection only**; the stereo "matching task" is done by classical ORB outside the network. There is no shared encoder feeding multiple decoder heads, no joint loss balancing competing gradients, no representation sharing between detection and stereo. The only ablation in the paper (Table 3, page 15) measures the impact of *block-level lightening choices* on detection mAP; not multi-task coupling. So the question "does joint training help?" cannot be asked here. The system-level coupling (detection bbox prunes ORB search space) is **load-bearing** in the deployment sense; without it, total per-frame latency would be 28 ms ORB + detection time instead of 40 ms total, but the 28 ms cost would also hit a slower mid-range CPU much harder. The detector and the matcher are decoupled at the gradient level entirely.

## Relevance to Our Project
- **Direct precedent for our YOLO26n encoder size class.** Yolo-SGN at 2.42 M / 5.9 GFLOPs is essentially the same operating point as our YOLO26n-truncated edge encoder (0.81 M trainable in StereoLite, but the YOLO26n proper is similar in scale). The detection-only ablations (Table 3) show that GhostConv (C3Ghost), NextConv depthwise 7x7, and SF-GS channel-split tricks are individually worth 0.3 to 0.6% mAP; all directly applicable to our encoder if we tune for drone perception.
- **Edge envelope hit: 25 FPS on TX2.** This is within our deployment band (~40 ms / 25 FPS on Jetson Orin Nano-class for the mid tier). Their **full stereo pipeline** including triangulation fits in 40 ms; useful sanity-check for what "real-time stereo + perception" looks like at TX2 class.
- **Drone-to-drone localization is exactly our pitched application.** The intro talks about drones for warehousing / formation flight (Slide 4 of `presentation/build_v5_deck.py`); Yolo-SGN's deployment use case overlaps directly. Worth citing as motivation in the review paper's edge section.
- **Caveat: their stereo is ORB-on-bbox, not dense disparity.** This works for "where is the other drone" (one 3D point) but not for "what is the depth of every pixel" (dense disparity for obstacle avoidance). Our StereoLite produces dense disparity; different operating point. Don't confuse the two.

## Limitations / What This Paper Doesn't Solve
- **Not actually a multi-task deep learning paper.** Stereo is classical (ORB triangulation); the deep model is single-task detection. Comparing it head-to-head with YOLOP, HybridNets, TwinLiteNet on the multi-task axis is misleading.
- **Sparse, not dense, depth.** Output is 3D coordinate(s) of detected targets, not a dense disparity map. Cannot be used for free-space mapping or general obstacle avoidance; only for "where is the other UAV".
- **Validation only on Drone-Fly (custom dataset).** No KITTI / BDD100K / Middlebury benchmark; cannot cross-compare with the rest of the stereo literature.
- **Two-rectified-image assumption is strict.** The ORB-in-bbox trick requires the same target to land in both images' bboxes; false negatives in either view (occlusion, lighting, partial overlap; explicitly acknowledged in the Discussion) silently break the matching.
- **No INT8 / TensorRT optimization reported.** 25 FPS on TX2 is FP32 PyTorch; with INT8 + TRT this could plausibly hit 60+ FPS, but the paper does not measure that.
