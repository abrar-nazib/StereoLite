# Pseudo-LiDAR from Visual Depth Estimation: Bridging the Gap in 3D Object Detection for Autonomous Driving

**Authors:** Yan Wang, Wei-Lun Chao, Divyansh Garg, Bharath Hariharan, Mark Campbell, Kilian Q. Weinberger (Cornell University)
**Venue:** CVPR 2019
**Tier:** 3 (bridge paper: dense stereo disparity becomes a 3D point cloud, then LiDAR-style 3D detectors run on it; the entire stereo branch is upstream of detection and is *not* trained jointly with the detector)

---

## Core Idea
The headline argument is one sentence: "the major cause for the performance gap between stereo and LiDAR is not a discrepancy in depth accuracy, but a poor choice of representations of the 3D information for ConvNet-based 3D object detection systems operating on stereo" (Sect. 1, p. 1-2). Prior image-based 3D detectors fed depth back into a 2D CNN as an extra channel, which is fatal because (a) far objects shrink, and (b) 2D pixel neighbourhoods mix points that are far apart in 3D. The fix is two lines of code: back-project every pixel into a 3D point using the calibrated focal length and baseline (Eq. 1-4, p. 3), call the resulting cloud "pseudo-LiDAR", and feed it to any LiDAR-based 3D detector unchanged.

## Architecture
- **Stereo disparity stage**: PSMNet (CVPR'18), DispNet (with and without correlation layer), or SPS-Stereo. The release-trained PSMNet is found to use validation pairs in pretraining, so a clean variant called PSMNet* is retrained on Scene Flow then finetuned on 3712 KITTI detection training images using LiDAR-projected pseudo-disparity GT (Sect. 4.2, p. 5).
- **Pseudo-LiDAR generation (Sect. 3, p. 3)**: depth z = f * b / disp; x = (u - cu) * z / fu; y = (v - cv) * z / fv. Points above 1 m relative to the fictitious LiDAR plane are clipped to mimic the ground-mounted Velodyne's vertical FOV.
- **3D detection backend, two flavours**:
  - **F-PointNet** consumes the 3D points directly via PointNet inside frustums from a 2D detector.
  - **AVOD** rasterises pseudo-LiDAR to a BEV image and fuses with the front-view RGB tower.
- **No joint training between stereo and detection**, the two stages share no gradients. Stereo is a fixed depth oracle to the downstream detector.

## Main Innovation
**Representation, not algorithm.** Tab. 2 (p. 7) is the smoking-gun ablation: same disparity (DispNet) feeding AVOD as frontal-depth-channels gives 19.5 / 9.8 APBEV / AP3D (moderate, IoU=0.7), vs 36.5 / 26.2 when the *same* disparity is converted to pseudo-LiDAR. Same network, same depth quality, +17 points by changing the data layout. The paper proves the point further with Fig. 3: a single uniform 2D conv applied to the depth map (then back-projected) smears far-away cars across tens of meters in 3D, while the same conv on the BEV pseudo-LiDAR preserves object shape. This is the contribution. Everything else (the 2x stereo-vs-mono gap, etc.) follows from it.

## Key Benchmark Numbers
- **Dataset**: KITTI 3D object detection (Sect. 4.1, p. 5). Split 3712 train / 3769 val per Chen et al.; car category by default; IoU thresholds 0.5 and 0.7.
- **No params / GFLOPs / FPS table in the paper.** Discussion section (Sect. 5, p. 8) admits "the classification of all objects in one image takes on the order of 1s" with no real-time engineering done.
- **Hardware not stated for inference**; training is on PSMNet's standard config (Scene Flow + KITTI stereo finetune).
- **KITTI val, AP_BEV / AP3D (IoU=0.7), moderate (Tab. 1, p. 7)**:
  - AVOD stereo pseudo-LiDAR (PSMNet\*): **56.8 / 45.3**
  - F-PointNet stereo pseudo-LiDAR (PSMNet\*): 51.8 / 39.8
  - MLF-Stereo (prior SOTA): 19.5 / 9.8
  - AVOD LiDAR + Mono: 86.5 / 73.5
  - F-PointNet LiDAR + Mono: 82.2 / 68.8
  - 3DOP stereo: 9.5 / 5.1
- **Headline claim (Abstract)**: for cars within 30 m at IoU 0.7, accuracy rises from 22% (prior SOTA) to 74%, more than tripling.
- **KITTI test, AVOD stereo (Tab. 5, p. 8)**: APBEV / AP3D moderate at IoU 0.7 = **47.2 / 37.2**, vs LiDAR+Mono AVOD 83.8 / 71.9.
- **Pedestrian/cyclist (Tab. 4, p. 7)**: stereo pseudo-LiDAR with F-PointNet, moderate APBEV / AP3D = pedestrians 34.9 / 27.4, cyclists 29.9 / 25.2 (vs LiDAR+Mono pedestrians 60.6 / 56.5).

## Multi-Task Coupling: Load-Bearing or Decorative?
There is **no joint coupling at all** in the trained-together sense. The stereo network and the detector are trained independently and connected only via the back-projection formula. Quoting Sect. 4.3: "pseudo-LiDAR is applicable and highly beneficial to two 3D object detection algorithms with very different architectures, suggesting its wide compatibility". The contribution is precisely that the link can be a dumb geometric transformation, no shared features, no shared losses. Tab. 3 (p. 7) varies the stereo network across DispNet-S, DispNet-C, PSMNet, PSMNet* and the detection AP scales monotonically with disparity quality, so the *only* coupling is the disparity error injected into the point cloud. Verdict: **decorative as multi-task learning, load-bearing as a representation choice.** This is a single-task stereo network feeding a single-task 3D detector; the paper's value is showing that the connection between them must be 3D, not a frontal-view depth channel.

## Relevance to Our Project
- **Direct downstream consumer of any StereoLite output.** Our edge tier emits dense disparity at ~25 ms on RTX 3050. With known intrinsics and baseline (live CCB stereo camera has these calibrated), the back-projection in Eq. 1-4 is ~0.5 ms on Jetson and produces a pseudo-LiDAR cloud that any deployed Frustum-PointNet/AVOD-Lite/PointPillars-tiny variant can consume. Pseudo-LiDAR is the cleanest way to bolt 3D object boxes onto StereoLite without retraining StereoLite or jointly training a detector.
- **YOLO26 + StereoLite + Pseudo-LiDAR is a sane drone perception stack**. YOLO26n/s emits 2D boxes from the left image, StereoLite emits dense disparity, pseudo-LiDAR turns disparity into a 3D cloud, frustum-PointNet inside each 2D box gives 3D position. All four pieces run independently; the integration is geometry plus per-object frustum slicing.
- **Detector training cost is the gotcha**, not stereo cost. The paper finetunes AVOD on pseudo-LiDAR generated from KITTI training pairs; the detector must be retrained per-camera-rig (different baseline => different 3D distribution). For drones with non-KITTI baselines (10-20 cm vs 0.54 m), the detector would need its own data.
- **Cross-domain extension is non-trivial.** PSMNet* was finetuned with KITTI-LiDAR-projected disparity targets. Our StereoLite is trained on Scene Flow Driving + indoor pseudo-GT; its zero-shot Middlebury 2014 EPE 5.5 (CLAUDE.md) implies that the pseudo-LiDAR cloud built from StereoLite would have ~5 m / pixel-level systematic error at 30 m range, which the detector then sees as a corrupted point cloud. Pseudo-LiDAR's gain over MLF is entirely contingent on the stereo network being accurate; for our chassis it would expose the cross-domain stereo failure as 3D detection failure.

## Limitations
- **No real-time engineering.** Sect. 5 (p. 8): "in this paper we did not focus on real-time image processing and the classification of all objects in one image takes on the order of 1s". PSMNet alone is ~400 ms on a Titan X; the BEV-AVOD detector adds ~80 ms. The full stack is unusable on edge hardware as-is.
- **Distant-object error is the hard limit.** "Stereo algorithms are known to have larger depth errors for far-away objects" (Sect. 4.3, p. 7). AP at IoU 0.7 hard drops to 39.0 (AVOD stereo) vs 67.1 (AVOD LiDAR) for cars beyond ~40 m. Not solved by the representation change, only by higher-resolution imagery or post-hoc densification.
- **Calibration-dependent**. Eq. 1 requires known focal length, baseline, and principal point. Drift in calibration is invisible at the disparity level but appears as systematic 3D position bias in pseudo-LiDAR.
- **Pedestrian and cyclist gap is much wider than for cars.** Tab. 4 (p. 7) shows stereo pseudo-LiDAR APBEV moderate at 34.9 (pedestrian) vs 60.6 (LiDAR), a 25-point gap; for cars the gap is closer to 30 points but at much higher absolute accuracy. Small / thin objects amplify stereo depth error.
- **The 200-stereo-2015-pair contamination warning (Sect. 4.2, p. 5)** is a useful methodological reminder, the released PSMNet model overlapped train and detection-val. Anybody reusing PSMNet on KITTI without PSMNet*-style finetune is reporting inflated numbers.
