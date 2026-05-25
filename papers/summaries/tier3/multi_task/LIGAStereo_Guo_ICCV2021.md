# LIGA-Stereo: Learning LiDAR Geometry Aware Representations for Stereo-based 3D Detector

**Authors:** Xiaoyang Guo, Shaoshuai Shi, Xiaogang Wang, Hongsheng Li (CUHK-SenseTime Joint Lab, CUHK)
**Venue:** ICCV 2021
**Tier:** 3 (stereo + 3D object detection; introduces cross-modal LiDAR-to-stereo feature distillation built on the DSGN chassis)

---

## Core Idea
LIGA-Stereo starts from DSGN and asks "what stops stereo-based 3D detectors from matching LiDAR detectors?" The diagnosis: stereo features inherit erroneous depth, especially in textureless, occluded, and distant regions, so the high-level geometry-aware features (surface normals, accurate boundaries) that LiDAR detectors learn never form in stereo. The fix is **feature imitation**: train a LiDAR-based teacher (SECOND), then distill its intermediate 3D/BEV features into the stereo student via an L2 imitation loss over foreground voxels. Second, a separate **auxiliary 2D detection head** is bolted onto the same semantic features so that semantic gradients no longer depend on the 3D-volume detour (which is noisy when depth is wrong).

## Architecture
- **Stereo branch (Fig. 2a, p. 3):**
  - 2D extractor: DSGN-style 2D backbone with blocks reduced from {3, 6, 12, 4} to {3, 4, 6, 3} (i.e., ResNet-34) + SPP + small U-Net for full-resolution feature output (Sect. 3.4, p. 5-6).
  - Plane-sweep volume Vst (Eq. 1, p. 3); aggregator outputs ~Vst and depth distribution Pst.
  - V3d (Eq. 2, p. 4): concatenation of resampled Vst with semantic features F_sem masked by Pst at each (x, y, z) voxel.
  - 3D-detection-head channels halved from 64 to 32; 3D hourglass on V3d removed (Sect. 3.4, p. 6).
- **BEV head**: SECOND head from OpenPCDet, replacing DSGN's FCOS head.
- **LiDAR teacher (Fig. 2c, p. 3):** SECOND with last sparse-conv stride changed 2->1 so its 1/4 BEV feature shape matches the student.
- **Imitation loss (Eq. 4, p. 4):** L2 between student feature `g(F_im)` and channel-normalized teacher feature, masked by M_fg (1 inside any GT box, 0 outside) and the LiDAR sparse mask M_sp. F_im in {V3d, F_BEV, ~F_BEV}; final choice = {V3d, ~F_BEV} (Tab. 6, p. 8).
- **Auxiliary 2D head (Fig. 2e):** five stride-2 convs on F_sem feeding an ATSS detection head; positive-sample assignment modified to use re-projected 3D centers instead of 2D bbox centers (Sect. 3.3, p. 5).
- **Modified losses**: uni-modal KL depth loss (Eq. 6, p. 6) replaces DSGN's smooth-L1, plus rotated 3D-IoU loss; total: L = L_depth + L_det + lambda_im L_im + lambda_2d L_2d, with lambda_im=1.0, lambda_2d=1.0.
- **Grid:** detection area [-30, 30] x [-1, 3] x [2, 59.6] m, voxel 0.2 m. Input 320 x 1248 (top of image cropped).

## Main Innovation
**Cross-modal feature imitation, foreground-masked, applied to intermediate 3D/BEV features.** Distilling soft labels of the LiDAR teacher (Hinton-style) did not help (Sect. 2, p. 2: "we found benefits little"); imitating intermediate features works. The foreground mask M_fg is mandatory: without it, imitation loss is dominated by the empty 99%+ of the BEV grid and gives no gain (Tab. 6 "w/o M_fg" row, p. 8: 62.01 AP3D vs 65.67 AP3D with mask).

## Key Benchmark Numbers
**Latency (Tab. 5, p. 8):** 0.35 s per pair on NVIDIA TITAN Xp. Paper notes TITAN Xp is "expected to be half the speed of NVIDIA V100", so V100-equivalent would be ~0.18 s. About 2.9 FPS on TITAN Xp.

**Memory (Tab. 5, p. 8):** 10 GB training, 4.9 GB inference (down from DSGN's 29 GB / 6 GB).

**Parameters / GFLOPs:** Not reported as absolute numbers. Backbone described as ResNet-34 + small U-Net + halved-channel stereo aggregator.

**KITTI test, Car, AP3D IoU=0.7 (Tab. 2, p. 5):** Easy 81.39, Moderate 64.66, Hard 57.22. AP_BEV: 88.15 / 76.78 / 67.40. AP_2D: 96.43 / 93.82 / 86.19. Beats DSGN (52.18 Mod AP3D) by +12.48 AP3D Mod. Closes gap to LiDAR teacher SECOND (78.57 Mod AP3D on test) to within ~14 AP.

**KITTI val, Car AP3D, R=11 (Tab. 1, p. 5):** Easy 84.92, Mod 67.06, Hard 63.80. IoU 0.5: Easy 97.06, Mod 89.97, Hard 87.94, nearly matching the LiDAR teacher SECOND (98.12, 90.17, 89.64). The gap at IoU 0.5 is only 0.2% mAP.

**KITTI test, Pedestrian, AP3D IoU=0.5 (Tab. 3, p. 6):** 40.46 / 30.00 / 27.07.

**KITTI test, Cyclist, AP3D IoU=0.5 (Tab. 3, p. 6):** 54.44 / 36.86 / 32.06.

**Stereo EPE:** Not reported. No disparity-space metrics are reported anywhere in the paper.

## Joint-Task Coupling: Stereo + Detection in One Net or Two?
**Two networks, one direction: distillation only at train time.** The LiDAR teacher and the stereo student are SEPARATE networks (Fig. 2). The teacher is trained first, then frozen, and at student-training time it is run forward only to produce F_lidar features that pull the student's F_im toward them via L_im. At inference the teacher is absent: only the stereo student runs. **Within the student itself**, stereo and detection are jointly trained (depth loss + det loss + imitation loss + aux-2D loss), so stereo features and detection features share a backbone. The ablation in Tab. 4 (p. 6) decomposes the contributions: baseline-with-tricks (b.) = 62.74 Mod AP3D; + imitation (c.) = 65.67 (+2.93); + 2D supervision (d.) = 63.32 (+0.58); both (e.) = 65.64; + ImageNet pretrain on top (g.) = 67.71 (+1.10 above e.). So imitation is the load-bearing piece (+2.93), 2D-aux head is small (+0.58 alone), and they don't strictly add (their combination is +2.90, not +3.51).

## Relevance to Our Project
- **Cross-modal distillation idea transfers; the architecture does not.** 350 ms on TITAN Xp ~~ 175 ms on V100 ~~ 1 second on Orin Nano. Forget the chassis for edge deployment.
- **The "use a stronger teacher" principle is exactly what LiteAnyStereo does for stereo.** LIGA's recipe is "train SECOND, distill into DSGN". LiteAnyStereo's recipe is "train FoundationStereo, distill into a 7 M MNV2+ConvNeXt chassis". Same paradigm, different teacher modality. Our project has the FoundationStereo teacher pipeline already wired (`model/scripts/run_teacher.py`), so the leap to "drone object bbox via stereo-feature distillation" would need only an object-bbox teacher (which exists abundantly for outdoor; less so for our indoor real data).
- **Foreground masking transfers DIRECTLY to our regime.** Any distillation we run from FoundationStereo to StereoLite should mask the background; FoundationStereo's depth is most reliable on textured foreground, which is also where we care for navigation. This is a free architectural lesson.
- **Two-headed model: 2D detection + stereo.** If we ever build a YOLO26 + StereoLite multi-task model, LIGA's auxiliary-2D-head trick (cheap, +0.58 AP, regularizes semantic features) is a natural template.
- **Different from segmentation how?** Segmentation does NOT teach stereo features about object identity; LIGA's 2D-aux head and 3D-imitation loss both inject object-level priors into the volumetric reasoner. For drone bbox + path-planning, this is exactly what we want, and segmentation alone would be insufficient.

## Limitations / What This Paper Doesn't Solve
- **Still not real-time.** 350 ms on TITAN Xp; conclusion (Sect. 5, p. 8) does not even mention deployment.
- **Pedestrian and Cyclist 2D-only-test is poor** (88 / 52 / 36 AP for car/ped/cyc when 2D loss is sole supervision, p. 8). The 2D head is a regularizer, not a standalone detector.
- **Requires a LiDAR-trained teacher** to even start training. For drone outdoor stereo without LiDAR labels this whole recipe is blocked. For our project with the FoundationStereo *depth* teacher already wired, we'd need a parallel **object** teacher.
- **Distillation across modalities is brittle.** Tab. 6 (p. 8) shows that imitating V3d alone gives 65.83 Mod AP3D, imitating ~F_BEV alone 65.10; imitating all three (V3d, F_BEV, ~F_BEV) gives 63.49 (WORSE than imitating any single feature). More distillation supervision is not monotonically better.
- **No stereo benchmark numbers.** EPE / D1 / bad-px on Scene Flow, KITTI 2015 stereo set, Middlebury, ETH3D are all absent. The stereo network is a means to the detection end.
- **KITTI-only training.** No cross-dataset eval. Likely brittle outside KITTI's specific camera + scene.
