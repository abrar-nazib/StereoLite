# SSPCV-Net: Semantic Stereo Matching with Pyramid Cost Volumes

**Authors:** Zhenyao Wu, Xinyi Wu, Xiaoping Zhang, Song Wang, Lili Ju (University of South Carolina, Wuhan University, Farsee2 Technology)
**Venue:** ICCV 2019, pp. 7484-7493
**Tier:** 3 (joint disparity + semantic seg, focuses on multi-scale cost-volume design augmented with one semantic cost volume; key predecessor to SGNet and SemStereo)

---

## Core Idea
PSMNet builds a single cost volume from multi-scale spatial pyramid pooling features. SSPCV-Net argues that flattening multiple scales into one cost volume loses level-specific structure and proposes to build **one cost volume per scale** instead (a "pyramid of cost volumes"), plus a single *semantic* cost volume from a PSPNet sub-branch. All four cost volumes are then merged by a recursive 3D feature-fusion module ("3D multi-cost aggregation") to produce the final disparity. A semantic-boundary loss (`|grad sem| * e^{-|grad d|}`) supervises the disparity to be smooth except at semantic boundaries.

## Architecture
- **Backbone:** ResNet-50 with dilated convolutions (DeepLab-style) for feature extraction (Sect. 3.1, p. 3).
- **Spatial branch:** adaptive average pooling to three scales (1/4, 1/8, 1/16 of input), each followed by a 1x1 conv to reduce channels; each scale forms its own GC-Net-style concatenation cost volume of size `C x alpha*W x alpha*H x alpha*D`, for `alpha in {1/4, 1/8, 1/16}` (Sect. 3.2.1, p. 3-4). `D_max = 256` for Scene Flow, 192 for KITTI.
- **Semantic branch:** PSPNet sub-network with shared shallow features; features before the classification layer are used to build a single semantic cost volume of size `C x 1/4 W x 1/4 H x 1/4 D` (matching the largest spatial CV) (Sect. 3.2.2, p. 4).
- **3D multi-cost aggregation:** recursive bottom-up fusion: low-level CV upsampled, fused with next-higher via a 3D Feature Fusion Module (FFM = residual sum + 3D adaptive pool + fc-ReLU-fc-sigmoid SE-style weight + multiply + add). Each fusion stage is preceded by a 3D hourglass. Final fused volume fused with semantic CV, then bilinear-upsampled to `1 x W x H x D` (Sect. 3.3, p. 4).
- **Disparity regression:** softmax over disparity dimension + soft-argmin (Eq. 1, p. 5).
- **Loss:** `L = alpha * L_disp + (1 - alpha) * L_bdry` with `alpha = 0.9` on Scene Flow / KITTI 2015. `L_disp` is smooth-L1; `L_bdry` is `|grad sem| * e^{-|grad d|}` (Eq. 2-4, p. 5).

## Main Innovation
The PSMNet-era contribution is a clean architectural separation between *spatial-pyramid features* (which mix scales before the cost volume) and *spatial-pyramid cost volumes* (which build a cost volume *per* scale and fuse the CVs themselves). Empirically the latter wins by ~1 percentage point on KITTI D1-all (Tab. 1, p. 6). The semantic cost volume is a smaller, additive contribution: built once at 1/4 scale, fused as the last step. The SE-style FFM (sigmoid-gated weighted sum of two CVs) is a minor but reused architectural element.

## Key Benchmark Numbers
- **Params:** not in paper.
- **GFLOPs:** not in paper.
- **Latency / FPS / target GPU:** not in paper (only mentions "two Nvidia 1080 GPUs" for training, Sect. 4.2 p. 5; runtime per-image is omitted).

**Scene Flow (Tab. 2, p. 7):**
- SSPCV-Net: **EPE 0.87 px, D1-all 3.1%**.
- PSMNet: EPE 1.09, D1-all 4.2. GC-Net: 1.84 / 9.7. SegStereo: 1.45 / 3.5.

**KITTI 2015 leaderboard, ALL D1-bg / D1-fg / D1-all, NOC D1-bg / D1-fg / D1-all (Tab. 3, p. 7):**
- SSPCV-Net: **1.75 / 3.89 / 2.11 / 1.61 / 3.40 / 1.91**.
- SegStereo: 1.88 / 4.07 / 2.25 / 1.76 / 3.70 / 2.08.
- PSMNet: 1.86 / 4.62 / 2.32 / 1.71 / 4.31 / 2.14.

**KITTI 2012, Out-Noc / Out-All at 2/3/4/5 px (Tab. 4, p. 8):**
- SSPCV-Net: 2.47 / 3.09 / 1.47 / 1.90 / 1.08 / 1.41 / 0.87 / 1.14.
- PSMNet: 2.44 / 3.01 / 1.49 / 1.89 / 1.12 / 1.42 / 0.90 / 1.15.

**Segmentation (KITTI 2015):** mIoU per class **56.43%** and 82.21% per category (Sect. 4.4, p. 7). No comparison to dedicated segmentation networks (PSPNet, DeepLab) is given.

**Cityscapes:** qualitative only; "compared methods... cost volumes channel set to 16" for fairness (Sect. 4.4, p. 8). No quantitative cross-domain numbers.

## Mutual-Task Coupling: Load-Bearing or Decorative?
Tab. 1 (p. 6) is the only ablation source. Trained on KITTI 2015 directly (no Scene Flow pretrain), Scene Flow val EPE / KITTI 2015 val D1:
- Single spatial CV baseline: 2.12 / 2.63.
- + Semantic branch (no joint train): 1.76 / 2.42. **Delta = -0.36 EPE, -0.21 D1.**
- + Semantic branch (joint-train): 1.78 / 2.37. (joint-train barely moves the needle vs frozen-segmentation)
- + Spatial pyramid CVs (replacing single CV): 1.21 / 2.11. **Delta = -0.55 / -0.26.**
- + Dilated convolution: 1.04 / 1.99. **Delta = -0.17 / -0.12.**
- - FFM (replaced with concat): 1.07 / 2.10.
- - L_bdry (only disparity loss in joint training): 1.01 / 1.93.
- **Full SSPCV-Net: 0.98 / 1.85.**

So the win is: pyramid CVs do ~0.55 EPE, semantic branch does ~0.36 EPE, dilated conv does ~0.17. The **semantic component is the second-biggest knob** but the spatial pyramid CV idea is the bigger contributor. The boundary loss `L_bdry` is the *smallest* contribution (0.03 EPE / 0.08 D1) and yet it is one of the named "contributions" of the paper.

Verdict: **Load-bearing for stereo on Driving-style scenes (KITTI/SF) — the semantic branch is doing real work (+0.36 EPE), more than the boundary loss but less than the pyramid CVs.** However, the paper does not measure segmentation quality vs a dedicated segmentation baseline at matched parameter budget, so it is impossible to tell whether the joint training is *net* helpful or just an architectural justification for adding more parameters. The Cityscapes generalisation experiment (Sect. 4.4) is qualitative only — they ducked the only test that would have settled it.

## Relevance to Our Project
- **Pyramid-of-cost-volumes ports cleanly to our chassis.** Our current `StereoLite_costlookup` chassis builds a single 1/16 group-wise CV with `max_disp = 24`. SSPCV-Net's pyramid (1/4 + 1/8 + 1/16) hierarchically fused via a recursive hourglass is exactly what the `p2_cascade_cv_4` Phase-2 winner (EPE 0.838) is gesturing toward. The recursive FFM (sigmoid-gated weighted sum) is a < 0.05 M-param block we could test inside our `MultiScaleCostFusion` head as a "select between hi-res and lo-res CVs" gate.
- **SE-style FFM is the most transferable single block.** Adaptive pool + fc-ReLU-fc-sigmoid SE-weighting on 3D cost volumes is implementable in ~30 lines of PyTorch; it is the same primitive that drove our earlier `SqueezeExcitation` block in `_blocks.py`. Worth testing as a CV-fusion gate.
- **Param budget incompatible.** ResNet-50 + three 3D cost-volume hourglasses is in the 30-60 M parameter range. The semantic branch alone (PSPNet head) adds another ~20 M. Not portable as-is to the 2.5 M mid-tier envelope.
- **Boundary loss is cheap and worth borrowing.** `|grad sem| * e^{-|grad d|}` (Eq. 4, p. 5) is structurally identical to DispSegNet's smoothness regulariser. Could be tested against our `loss_stack_d1` cocktail using Sobel of the *image* as a cheap proxy for semantic gradient (one-line change, no new parameters).
- **The dataset story is the same trap as TiCoSS.** All numbers are Driving (Scene Flow / KITTI 2015 / KITTI 2012). The Cityscapes evaluation is qualitative-only. No zero-shot MB14 / ETH3D. We cannot tell whether the pyramid CV + semantic cocktail actually helps cross-domain — which is the only stereo question our project cares about.

## Limitations / What This Paper Doesn't Solve
- **No real-time discussion.** Three 3D-CV hourglasses + recursive FFM is heavy; the paper does not report runtime per image, and "two 1080 GPUs" for training implies a chunky model. Cannot be deployed on edge without massive surgery.
- **Cityscapes generalisation is qualitative-only** (Sect. 4.4, p. 8). The only cross-domain comparison the paper offers is "look at the picture, it has clearer boundaries". No EPE / D1 numbers.
- **Semantic quality is only weakly measured.** KITTI 2015 segmentation gets 56.43% mIoU (Sect. 4.4, p. 7) but is not compared to a dedicated segmentation network. The semantic branch may be hurting segmentation quality compared to a standalone PSPNet at matched params.
- **No standalone "segmentation alone" baseline.** It is impossible to tell whether the semantic CV is adding capacity that any extra parameters would have added, or whether it specifically encodes useful structure.
- **No params / FLOPs reported.** Makes any deployment-cost analysis impossible without re-implementation.
