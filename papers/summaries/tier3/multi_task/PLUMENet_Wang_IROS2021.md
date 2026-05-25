# PLUMENet: Efficient 3D Object Detection from Stereo Images

**Authors:** Yan Wang, Bin Yang, Rui Hu, Ming Liang, Raquel Urtasun (Uber ATG, Cornell)
**Venue:** IEEE/RSJ IROS 2021
**Tier:** 3 (stereo + 3D object detection; efficiency-focused successor to DSGN, introduces the PLUME 3D feature volume in metric space)

---

## Core Idea
PLUMENet sets out to be the efficient end-to-end stereo 3D detector. Prior pseudo-LiDAR (PL, PL++) pipelines suffer "representation mismatch" because depth is estimated in image space while detection is performed in metric space, and DSGN's joint pipeline pays for that unification with a 670 ms-per-frame 3D-conv stack. PLUMENet's answer is to build a **single pseudo-LiDAR feature volume (PLUME) directly in 3D metric space** filled with bilinearly-sampled stereo features, then reason on it with a hybrid **3D-BEV** network that collapses 3D convs to 2D BEV convs as soon as the height-dimension reasoning is done. The result: 4x faster than DSGN at comparable or higher BEV detection AP.

## Architecture
- **Stereo image network (Fig. 4, p. 4)**: 2D Siamese CNN with SPP (spatial pyramid pooling) + FPN. The finest scale is at **full input resolution** (not 1/4 like DSGN). Encoder: conv0 (3x3 x 3, 32 ch), maxpool0 (stride 2), then four residual stages with 3, 6, 2, 2 (Small/Middle) or 3, 6, 6, 6 (Large) basic blocks. Channels for Small/Middle/Large = 8 / 32 / 32 in early layers, 16 / 64 / 64 mid, 32 / 128 / 128 late (Tab. V, p. 9).
- **PLUME (Fig. 5, p. 4):** voxel grid in 3D world space [-32, 32] x [2, 62.8] x [-1, 2] m, voxel size 0.2 m. Each voxel center (x, y, z) is projected into BOTH left and right image features by camera intrinsics; the per-voxel feature is the concatenation of [left-feature, right-feature, voxel-coordinates]. Crucially: no disparity dimension; the cost volume is implicit in the two image-feature concatenations per voxel.
- **3D-BEV hybrid network (Fig. 6, p. 4):** two 3D conv layers first reason about height; then height + feature dimensions are flattened, giving a 2D BEV feature map; a 2D hourglass increases receptive field in depth/width. This is the key efficiency trick.
- **Occupancy header**: 2 conv layers; output D x W with H channels, reshaped to 3D occupancy + sigmoid for per-voxel BCE supervision (Eq. 1, p. 5).
- **Detection header**: PIXOR-style BEV detector. Encoder = 5 blocks (first: two 3x3 stride-1 with 32 channels; remainder: bottleneck residuals with 3, 6, 6, 3 layers). Decoder = FPN with 4x downsampling. Prediction = (w, h) size, (u, v) BEV location, theta orientation (Sect. III-D, p. 4-5).
- **Output is BEV bbox, not 3D bbox**. Paper notes you can swap a 3D head in.
- **Loss**: L = L_focal (cls) + L_smooth-l1 (box regression), plus L_BCE on occupancy (Eq. 1, p. 5).
- **NMS** assumed (PIXOR-standard).
- **Training**: stage-wise. First train backbone + occupancy head with depth loss, 50 epochs; then freeze backbone, train detection head with detection loss, 50 epochs (Sect. III-E, p. 5). 4x RTX 5000, batch 8.

## Main Innovation
The **3D-BEV hybrid network**: two 3D convs to handle height, then flatten height-and-channel into 2D and do all subsequent work in BEV. Ablation in Tab. IV (p. 6): pure 3D network 82.2 / 68.2 / 64.8 AP3D in 230 ms; pure BEV network 75.6 / 62.6 / 56.3 in 110 ms; 3D-BEV hybrid 83.5 / 68.5 / 62.8 in **150 ms**. Hybrid beats pure BEV by 6 AP3D Easy at the cost of 40 ms, and matches pure 3D at 80 ms savings. Second innovation: feeding the volume from **full-resolution** image features (no 4x downsample like DSGN/LIGA), which Tab. IV shows is worth +5.2 AP3D Easy vs half-size features.

## Key Benchmark Numbers
**Latency (Tab. II + III, p. 5-6, NVIDIA Tesla V100):**
- PLUMENet-Small: **80 ms** (12.5 FPS). Real-time on V100.
- PLUMENet-Middle: **150 ms** (6.7 FPS). The "default" config.
- PLUMENet-Large: **530 ms** (1.9 FPS).
- vs DSGN 670 ms, CDN 600 ms; PLUMENet-Middle is 4x faster.

**Parameters / GFLOPs:** Not reported. Three sizes differ by channel widths (Small uses 8/16/32 ch in residual stages; Middle/Large use 32/64/128) per Tab. V.

**KITTI test, BEV AP (Car, sorted at Moderate, Tab. II, p. 5):**
- PLUMENet-Middle: Easy 83.0, Moderate 66.3, Hard 56.7. Reports BEV only on test.

**KITTI val, BEV AP at IoU=0.7 (Car, Tab. III, p. 6):**
- PLUMENet-Small: 74.4 / 61.7 / 55.8 (80 ms).
- PLUMENet-Middle: 83.5 / 68.5 / 62.8 (150 ms).
- PLUMENet-Large: 84.7 / 71.1 / 65.1 (530 ms).
- vs DSGN: 83.2 / 63.9 / 57.8 (670 ms).

**KITTI val, BEV AP at IoU=0.5 (Tab. III, p. 6):**
- PLUMENet-Middle: 91.0 / 85.9 / 80.5.
- PLUMENet-Large: 91.3 / 86.6 / 81.6.

**3D AP at IoU=0.7:** Not reported in the main tables. PLUMENet uses BEV detection as the primary output. The paper says you can swap a 3D head in, but only BEV numbers are benchmarked.

**Stereo EPE / disparity quality:** Not reported. The "depth" supervision is binary voxel-occupancy BCE (Eq. 1, p. 5), not a disparity / depth-map metric. The teacher signal comes from "at least one LiDAR point in the voxel".

## Joint-Task Coupling: Stereo + Detection in One Net or Two?
**One network, stage-wise trained (not jointly).** Sect. III-E (p. 5) explicitly: "we first train the network backbone (stereo image network and 3D-BEV network) plus the occupancy header with depth estimation loss, and then **fix the backbone's weights** and train the detection header with object detection loss." So while the *architecture* is unified end-to-end (gradient path exists), the *training procedure* is two-stage with the backbone frozen during stage 2. The depth/occupancy task supervises feature learning (stage 1), then the detection head consumes those frozen features (stage 2). The contrast with DSGN, which trains both losses jointly, is deliberate, presumably to stabilize early-training when depth signal is noisy. Stage 1 essentially turns the backbone into a 3D-feature extractor pre-trained on occupancy, then stage 2 does pure detection learning on top. Tab. IV "image feature fusion" entry (image features warped into BEV, weighted by occupancy, fused with detection features) only adds +0.4 AP3D Moderate, so the coupling beyond the shared backbone is weak.

## Relevance to Our Project
- **First end-to-end stereo 3D detector to break 100 ms on V100.** This is the architectural family that came closest to "real-time" before iterative-stereo + foundation-model approaches changed the game. PLUMENet-Small at 80 ms / V100 still translates to ~500 to 800 ms on Jetson Orin Nano, still off our 60 ms target by ~10x.
- **The 3D-BEV hybrid is a strong design principle.** "Use 3D convs only where the height dimension matters, then collapse to 2D ASAP" is a transferable rule. If we ever built a BEV head on top of StereoLite, we'd flatten our 1/16 disparity feature into BEV with one trilinear warp + 2D BEV agg, not a 3D-conv stack. This is essentially the recipe PLUMENet validates with Tab. IV.
- **Full-resolution image features in the stereo network matter even more than in pure stereo.** Tab. IV shows +5 AP3D from full-size vs half-size (a sign that "we lose far-away cars when we downsample"). For drone perception of small objects (other drones, hovering targets), this lesson is doubly important.
- **Stage-wise vs joint training trade-off.** PLUMENet's design choice (freeze backbone for stage 2) is the opposite of DSGN's joint-loss design. For our project, joint training has worked better historically (loss-cocktail sweep, etc.); stage-wise is the natural fit if we ever add a detection head onto a *pre-trained* StereoLite checkpoint, which is the kind of incremental change we typically do.
- **Different from segmentation how?** PLUMENet outputs BEV bboxes that a planner can consume directly: oriented rectangles with (x, y, w, h, theta). Segmentation gives per-pixel labels with no notion of "object". For drone path-planning the BEV bbox is the right level of abstraction. A semantic map is intermediate at best.

## Limitations / What This Paper Doesn't Solve
- **BEV-only output.** No 3D AP reported; the paper repeatedly says you "can" swap a 3D head in but never measures it. This is a step DOWN from DSGN/LIGA in output granularity.
- **Occupancy supervision needs LiDAR.** L_depth (Eq. 1) is binary BCE: 1 if a voxel contains a LiDAR point, 0 otherwise. No stereo-only training mode is documented. So PLUMENet is unrunnable in any drone scenario without a calibrated LiDAR overlay during training.
- **Stage-wise training pinpoints a limit.** If you can't co-train, the backbone learns features for occupancy alone, then detection has to make do; PLUMENet's modest +0.4 AP from image-feature fusion (Tab. IV) hints the backbone has not absorbed object-shape priors.
- **No cross-dataset / zero-shot eval.** KITTI val and test are the only numbers. Stereo + BEV is heavily KITTI-coupled in volume bounds, voxel size, and camera intrinsics.
- **Real-time only on V100.** "Real-time" in the paper means 80 ms on a V100, not on edge hardware. The 2D-BEV-hourglass + PIXOR head + 3D conv pair are all still chunky in TFLOPs.
- **No published parameter count.** Three sizes are described by channel widths only; absolute parameter / GFLOP figures missing.
- **Stereo metrics absent.** No disparity EPE, bad-px, KITTI 2015 leaderboard entry. The stereo branch is treated as a black-box feature extractor.
