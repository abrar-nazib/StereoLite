# DSGN: Deep Stereo Geometry Network for 3D Object Detection

**Authors:** Yilun Chen, Shu Liu, Xiaoyong Shen, Jiaya Jia (CUHK, SmartMore)
**Venue:** CVPR 2020
**Tier:** 3 (joint stereo + 3D object detection; first one-stage end-to-end stereo 3D detector, predecessor to LIGA-Stereo / DSGN++ / PLUMENet)

---

## Core Idea
Most stereo-based 3D detectors of the time were two-stage pseudo-LiDAR pipelines: a stereo network predicts a disparity map, the map is unprojected to a sparse point cloud, then a LiDAR-style 3D detector runs on that cloud. DSGN argues this is suboptimal because (a) the disparity-to-point-cloud step throws away the dense matching distribution, (b) two networks cannot be jointly optimized, and (c) disparity-space (image-centric) cost volumes are imbalanced in 3D (objects 40 m and 39 m away differ by less than 0.25 px on KITTI). DSGN constructs a single end-to-end differentiable network that lifts stereo features into a plane-sweep volume (PSV) in camera frustum space, then warps it via the camera intrinsics to a 3D geometric volume (3DGV) in world coordinates, and runs both depth regression (on PSV) and 3D detection (on 3DGV) jointly.

## Architecture
- **2D feature extractor**: PSMNet-style Siamese network, ResNet-ish with basic-block counts modified from {3, 16, 3, 3} to {3, 6, 12, 4} (shifts capacity to conv4 / conv5 for semantics); conv1 widens 32 to 64; basic-block output channels widen 128 to 192; SPP module concatenates conv_4 and conv_5 outputs (Sect. 3.2.1, p. 3-4).
- **Plane-sweep volume (PSV)**: shape (W_I/4, H_I/4, D_I/4, 64) where W_I=1248, H_I=384, D_I=192. Built by concatenating left feature F_L with reprojected right feature F_R at equally-spaced depth planes (Sect. 4.1, p. 6).
- **3D aggregation**: one 3D hourglass module (PSMNet uses three; cut for memory) plus extra 3D convs to squeeze to a matching cost volume of shape (W_I/4, H_I/4, D_I/4, 1). Soft-argmin (Eq. 2, p. 5) gives the depth map.
- **3D geometric volume (3DGV)**: discretizes the world to (W_V=300, H_V=20, D_V=192) voxels of size 0.2 m each over [-30.4, 30.4] x [-1, 3] x [2, 40.4] m. Built by trilinearly warping the last 32-D PSV feature through the inverse camera intrinsic (Eq. 1, p. 4). The "Last Features" choice (64-D latent) beats binary "Occupancy" by +16.4 AP3D (Tab. 6, p. 9).
- **3D detection head**: anchors at each (x, z) BEV cell; gradual downsample along the height dimension to a BEV feature map; FCOS-inspired distance-based target assignment and centerness branch (Sect. 3.2.4, p. 5). 4 orientations (0, pi/2, pi, 3pi/2); anchor sizes (h=1.56, w=1.6, l=3.9) for Car. NMS with IoU 0.6.
- **Multi-task loss**: Loss = L_depth (smooth-L1 on supervised LiDAR depth, p. 5) + L_cls (focal) + L_reg (smooth-L1 on 8-corner distance, p. 5) + L_centerness (BCE).

## Main Innovation
The plane-sweep volume to 3D-geometric volume transform is the load-bearing piece. Crucially, the *last latent 64-D feature* of the PSV gets warped into 3DGV, not the scalar cost or the explicit occupancy. The ablation in Tab. 6 (p. 9) shows 37.86 AP3D (occupancy) vs 43.24 (probability) vs 54.27 (last features), confirming that semantic information leaking through the latent channels into 3DGV is what makes the joint pipeline work. The second innovation is supervising depth in PSV space (where pixel correspondence is natural) while supervising detection in 3DGV space (where objects have view-invariant shape).

## Key Benchmark Numbers
**Parameters / FLOPs / memory:** Paper does not report a single trainable-parameter count, GFLOPs, or model size figure (Sect. 4 is benchmark-only). Trained on 4 NVIDIA Tesla V100 (32 GB), batch=4 total (one stereo pair per GPU), 50 epochs, 17 h wall-clock.

**Latency:** 0.682 s per stereo pair on a single V100 (Sect. 4.3.2, p. 8). Roughly 1.5 FPS. The 2D feature extractor alone is 0.113 s.

**KITTI 3D AP, Car, IoU 0.7, test set (Tab. 1, p. 7):** Easy 73.50, Moderate 52.18, Hard 45.14. BEV AP: Easy 82.90, Moderate 65.05, Hard 56.60. 2D AP: 95.53 / 86.43 / 78.75.

**KITTI 3D AP, Pedestrian, IoU 0.5, test set (Tab. 9, p. 11):** Easy 20.53, Moderate 15.55, Hard 14.15.

**KITTI 3D AP, Cyclist, IoU 0.5, test set (Tab. 9, p. 11):** Easy 27.76, Moderate 18.17, Hard 16.21.

**Stereo EPE / depth error:** Mean absolute depth error on KITTI val 0.5586 m / median 0.1104 m when jointly trained with detection (Tab. 4, p. 8). When trained with depth only, mean error drops to 0.5279 m. No raw EPE in pixels reported.

## Joint-Task Coupling: Stereo + Detection in One Net or Two?
**One network, jointly optimized.** The whole pipeline is end-to-end differentiable: gradients from L_det flow through 3DGV, through the warping operator (trilinear interpolation, fully differentiable), back into PSV and the 2D extractor. The ablation in Tab. 3 (p. 8) quantifies this: "IMG to 3DV" (no PSV intermediate, no depth supervision) gets 11.03 AP3D on stereo. Adding 3DV supervision: 42.57 AP. Going through PSV with PSV supervision: 54.27 AP. So +43.24 AP3D comes from joint depth supervision compared with the no-supervision baseline. Conversely, jointly training depth + detection raises the depth error from 0.5337 m (depth-only) to 0.5606 m, then 0.5586 m for DSGN, meaning the detection loss does slightly degrade depth - a 4-5% relative cost on depth in exchange for a +7.86 AP3D gain over the PSMNet-PSV baseline that uses the identical training pipeline. Tight coupling, both directions.

## Relevance to Our Project
- **Wrong family for our chassis.** DSGN is a one-stage end-to-end voxel pipeline, but the cost is 0.682 s per pair on a V100. Trying to compress this to 60 ms fp16 on Orin Nano is a different research project, not a port.
- **The volumetric paradigm is hostile to edge deployment.** 3DGV is (300, 20, 192) at 32 channels = 36.9 M voxel features; 3D convs on it dominate cost. Drone-class hardware (4 GB, 4-6 TOPS, INT8-only) cannot host that. The relevant lesson is opposite: stereo-into-3D-volume is fundamentally an automotive-class architecture.
- **The differentiable PSV-to-3DGV warp is a transferable primitive.** If we ever wanted to add a 3D-object head to StereoLite, we could warp our 1/16 disparity feature into a tiny (W=80, H=8, D=48) 3D grid via the same trilinear operator (one tensor op, no params), then a 2D BEV head. That would be the DSGN-shaped path to drone-bbox prediction without paying for a full PSV.
- **Different from segmentation-based perception how?** A semantic segmentation head says "pixel (u, v) is class person but its 3D location is unknown until you read disparity". DSGN says "voxel (x, y, z) is a car at heading theta" directly. For drone obstacle avoidance, the bbox+heading output is much closer to what a planner consumes; a semantic map still requires a separate "what disparity should I trust" decision per pixel.
- **The "depth and detection are bidirectionally beneficial" finding is well-validated here.** Tab. 4 + Tab. 3 (p. 8) together show that joint training raises detection by ~10 AP while only costing 4% relative on mean depth error. This argues strongly for a coupled head if we ever add one, instead of bolting a detector on top of frozen StereoLite.

## Limitations / What This Paper Doesn't Solve
- **No real-time path.** 0.682 s / 1.5 FPS on a V100 is roughly 50-200x off any embedded-deployment target.
- **No 1/32 head, no foundation-model-era performance.** This is 2020-era; current iterative (IGEV-Stereo) and foundation models (FoundationStereo, MonSter) post substantially better cross-domain stereo and didn't exist when DSGN was designed.
- **KITTI-only evaluation.** No zero-shot Middlebury, ETH3D, or DrivingStereo numbers. Cross-domain generalization is not measured; the architecture's strong dependency on hand-tuned 3D grid bounds ([-30.4, 30.4] x [-1, 3] x [2, 40.4] m) makes it brittle outside KITTI's specific camera + scene geometry.
- **Single-class training caveat.** A separate network is trained for Pedestrian and Cyclist because only 1/3 of KITTI images have those labels (Sect. 4.1, p. 6). The reported per-class numbers are not from one unified model.
- **Scene Flow pretraining absent here but used by competitors.** DSGN was trained from scratch on ~7 K KITTI pairs only; PL / PL++ used Scene Flow's ~30 K pairs as pretraining (paper marks them with *). Direct apples-to-apples comparison is muddled.
