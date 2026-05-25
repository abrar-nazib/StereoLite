# DSGN++: Exploiting Visual-Spatial Relation for Stereo-based 3D Detectors

**Authors:** Yilun Chen, Shijia Huang, Shu Liu, Bei Yu, Jiaya Jia (CUHK, SmartMore, SmartSens)
**Venue:** IEEE TPAMI 2022 / arXiv 2204.03039
**Tier:** 3 (direct successor to DSGN; renews three pipeline components against the same KITTI 3D detection task)

---

## Core Idea
DSGN++ takes apart DSGN's three pipeline bottlenecks one at a time. (1) 2D-to-3D info bottleneck: PSV/3DGV channel width (CV = 32) is dwarfed by the volume dimensions (W_V x H_V x D_V), starving the volumetric reasoner of features. (2) Geometric inductive bias mismatch: front-view (PSV) and top-view (3DGV) volumes have different voxel-occupancy profiles by depth, so each is good at a different class (PSV handles small pedestrians better; 3DGV handles cars better at distance). (3) Foreground sparsity in 3D: only a tiny fraction of voxels are foreground, biasing gradients. The fixes are Depth-wise Plane Sweeping (D-PS), Dual-view Stereo Volume (DSV), and Stereo-LiDAR Copy-Paste (SLCP).

## Architecture
- **Siamese 2D extractor**: ResNet-34 backbone (paper uses both DSGN's modified ResNet and a slimmer ResNet-18 variant in efficiency study). Output channels widened to C_I = 96 for D-PS to slice into (Sect. 4.1, p. 7).
- **Depth-wise Plane Sweeping (D-PS)**: instead of compressing 2D channels to 32 before volume construction, keep CI = 96 and slice CV channels via a sliding window along the channel axis whose offset depends on disparity = f_u x baseline / d (Eq. 4, p. 5). A "Cyclic Slicing" step reorders channels to keep continuity across nearby depth planes (Sect. 3.2, p. 4). Same FLOPs as classic plane sweeping, more memory.
- **Dual-view Stereo Volume (DSV)**: D-PSV (front view, camera-frustum-shaped voxels) + D-3DGV (top view, world-coordinate cubes) get aggregated. D-PSV is 3D-warped into world space, concatenated with D-3DGV, then passed through a 3D hourglass (Fig. 1 + Sect. 3.3, p. 5-6).
- **Front-Surface Depth Head (FSD)**: depth supervision applied in the FRUSTUM space (after warping DSV back), not in 3D space. Replaces DSGN's PSV-side smooth-L1 with a focused front-surface loss; PLUME's voxel-occupancy loss is rejected as inferior (Sect. 3.3, p. 6).
- **Stereo-LiDAR Copy-Paste (SLCP, Sect. 3.4, p. 6-7)**: paste foreground 3D object boxes from one scene to another, project to BOTH stereo views with the target camera intrinsics, crop the 2D patches via S->T projection, paste; remove background points behind pasted objects to keep uni-peak depth.
- **3D detection head**: SECOND-style BEV detector (Sect. 4.1, p. 7).
- **Volume dimensions**: PSV (W_I/4, H_I/4, D_I/4, 64); D_I=192 for DSGN-style, 288 for L-DSGN-style. 3DGV (W_V=300, H_V=20, D_V=288), 0.2 m voxels.

## Main Innovation
Of the three changes, **D-PS is the conceptually load-bearing one**: it dissolves the 2D-to-3D channel bottleneck so that the volumetric reasoner sees wider feature signatures per voxel without inflating 3D-conv FLOPs. The ablation hyperparameter sweep (Tab. 5 area in Sect. 4.3.1, p. 9) gives D-PS at 66.42 AP3D (FV, alpha=0.1) vs Group-PS at 65.48 AP3D, and the smoothness factor alpha lets the channel-shift rate be tuned per disparity range. SLCP is the second standout; it is the first multi-modal (stereo + LiDAR) copy-paste that preserves cross-modal alignment at sub-pixel precision, addressing the "foreground voxels are 1% of the volume" gradient-imbalance problem head-on.

## Key Benchmark Numbers
**Latency (Tab. 7, p. 12, NVIDIA RTX 2080Ti, batch=1, ResNet-34):** DSGN++ full DSV 0.281 s (~3.6 FPS). FV-only DSGN++ 0.198 s. TV-only DSGN++ 0.202 s. R18-DSGN++ (slim) 0.178 s. Per-stage breakdown: 2D extractor 2 x 0.058 = 0.116 s, PSV 0.036 s, 3DGV 0.045 s, DSV+3D 0.044 s, BEV head 0.012 s.

**Parameters / GFLOPs:** Paper does not report a single parameter count or GFLOPs figure. R18-DSGN++ states the "backbone network removing about half of the parameters" but no absolute count.

**KITTI test, Car, IoU 0.7, AP3D (Tab. 1, p. 8):** Easy 83.21, Moderate 67.37, Hard 59.91. AP_BEV: Easy 88.55, Moderate 78.94, Hard 69.74. AP_2D: 98.08 / 95.70 / 88.27. Beats LIGA-Stereo (Mod 64.66 AP3D) by +2.71 AP3D without cross-modal distillation.

**KITTI test, Pedestrian, IoU 0.5, AP3D (Tab. 2, p. 8):** Easy 43.05, Moderate 32.74, Hard 29.54.

**KITTI test, Cyclist, IoU 0.5, AP3D (Tab. 2, p. 8):** Easy 62.82, Moderate 43.90, Hard 39.21.

**KITTI val, Car AP3D (Mod, Tab. 3, p. 9):** DSGN++ on DSGN baseline 61.62 (vs DSGN baseline 56.09, +5.53); DSGN++ on L-DSGN baseline 69.12 (vs 63.58, +5.54).

**Stereo EPE:** Not reported. The paper does not report disparity-space metrics; only foreground-object depth error in the qualitative Fig. 7 (avg depth error 0.57 m with SLCP, 0.63 m without).

## Joint-Task Coupling: Stereo + Detection in One Net or Two?
**One network. Bidirectionally coupled.** The Front-Surface Depth Head sits on the *same* DSV feature volume that the 3D detection head reads, so the same parameters serve both losses. The paper explicitly chose the FSD over PLUME-style separate occupancy supervision because joint-feature supervision works better (Sect. 3.3, p. 6: "the geometric supervision of sub-voxel depth values in the front view provides stronger supervision than discretized voxel occupancy learning"). The W/O depth-supervision ablation (Tab. 3, p. 9: "L-DSGN w/o LiDAR sup." 55.22 -> "DSGN++ on L-DSGN w/o L Sup." 66.08, +10.86 AP3D) shows DSGN++'s improvements help even without explicit depth supervision; the improvement from L Sup adds another ~3 AP. So: structural coupling (shared DSV features) is what makes the gains transfer to the no-LiDAR setting.

## Relevance to Our Project
- **Not edge-deployable in any form.** 281 ms on RTX 2080Ti is roughly 5x our 60 ms budget on a desktop-class GPU. Even R18-DSGN++ at 178 ms is too heavy, and Orin Nano is ~10x slower than 2080Ti.
- **D-PS is the one idea worth borrowing IF we ever go volumetric.** The depth-wise channel slicing is a clever way to give a wide 2D feature stack to a 3D reasoner without paying for wide 3D channels. For a tile-based chassis like StereoLite this doesn't apply directly (we don't have a volumetric reasoner), but if we ever add a tiny BEV head, D-PS's idea of feeding wide image features to a narrow 3D grid is the transferable trick.
- **SLCP requires LiDAR labels we don't have.** Outdoor drone perception is the inverse situation: lots of stereo, no LiDAR-aligned 3D bboxes. SLCP cannot be replicated in our setting.
- **Different from segmentation how?** Segmentation gives per-pixel class only. DSGN++ outputs (x, y, z, w, h, l, theta) per object in BEV. The output is much more direct for a planner: a drone navigation stack can take 9 numbers per object and plan, vs a class map plus disparity plus per-pixel grouping logic.
- **Cross-domain unknown.** Paper trains and evaluates on KITTI only; no zero-shot Middlebury / ETH3D. The strong dependency on D-PS's stereo-baseline-conditioned channel shift (Eq. 4, p. 5) means changing baseline at test time likely breaks it.

## Limitations / What This Paper Doesn't Solve
- **Real-time deployment** unaddressed. Conclusion (Sect. 5, p. 12): "we leave the further code optimization to future work." Even with CUDA-optimized D-PS the architecture is bounded by 3D conv cost.
- **Parameter count silence.** Paper does not report parameters or GFLOPs in any table. R18-DSGN++ removes "about half" the backbone params but the absolute number is never quoted.
- **No stereo-only metrics.** Disparity EPE, bad-px rates, KITTI stereo leaderboard - none reported. The stereo network is treated purely as a means to the detection end.
- **KITTI overfitting risk.** Hyperparameters of the volume grid (300 x 20 x 288), the alpha = 0.1 D-PS smoothness, and the FSD head are all tuned on KITTI val. No held-out cross-dataset eval.
- **SLCP needs a LiDAR-instrumented dataset.** Stereo-only datasets (Scene Flow, Middlebury) have no LiDAR points to paste, so SLCP is a KITTI/NuScenes-only contribution.
- **Failure modes catalogued (Fig. 6, p. 11):** missing occluded objects, missing distant objects (>50 m), wrong orientation/dimension predictions. Distant-object accuracy still drops sharply with depth despite D-PS.
