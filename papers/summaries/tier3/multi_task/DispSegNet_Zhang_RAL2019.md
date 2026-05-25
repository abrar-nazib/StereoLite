# DispSegNet: Leveraging Semantics for End-to-End Learning of Disparity Estimation from Stereo Imagery

**Authors:** Junming Zhang, Katherine A. Skinner, Ram Vasudevan, Matthew Johnson-Roberson (University of Michigan)
**Venue:** IEEE Robotics and Automation Letters (RAL) 2019, vol. 4 no. 2, pp. 1162-1169 (arXiv:1809.04734v2, January 2019)
**Tier:** 3 (joint disparity + semantic segmentation, unsupervised stereo with segment-embedding refinement; cited by SGNet and TiCoSS as the canonical "fuse segment embedding into refinement" baseline)

---

## Core Idea
DispSegNet builds a two-task CNN that estimates disparity unsupervised (via photometric warping + left-right consistency) while jointly predicting semantic segmentation supervised on KITTI/Cityscapes labels. The key design move is a **two-stage refinement** where the initial disparity from a 3D cost volume is concatenated with the semantic segment embedding and passed through a residual block before final loss. The semantic supervision is used both as a feature input to refinement and as a regulariser inside the smoothness loss (the smoothness term is multiplied by `e^{-|grad f_L|}` where `f_L` is shallow semantic features, so smoothness fires only inside semantically homogeneous regions).

## Architecture
- **Siamese backbone:** ResNet-50 shared between L and R, features taken at 1/4 of input for disparity, deeper layers for segmentation (Sect. III-A, p. 3). First conv is 7x7; all other kernels are 3x3.
- **Cost volume:** 5D concatenation cost volume of L and R features (`B x (D_max+1) x H x W x C`), `D_max = 192`. Both left-cost and right-cost volumes built so disparity is computed in each view (Sect. III-B, p. 3).
- **Initial disparity estimator:** 8-layer 3D encoder-decoder with two-layer 3D residual blocks on skip connections + 3D transpose conv on the decoder side; soft-argmin produces initial `d_init` (p. 3).
- **PSP segmentation branch:** PSP module applied at 1/8 resolution, pooling at 1/2, 1/4, 1/8 of feature-map size, 1x1 conv reducing to 1/4 of input channels, bilinear-upsample + concat + 1x1 mixing conv (Sect. III-D, p. 4).
- **Refinement block:** initial disparity + resized segment embedding concatenated, processed by a residual 2D block, then summed back into the initial disparity to produce `d_ref` (Sect. III-C, p. 3).
- **Post-processing:** left-right consistency check (threshold = 1 px) + median-filtered mask + leftward valid-pixel interpolation, no global optimisation (Sect. III-F, p. 5).

## Main Innovation
First *unsupervised* stereo network to use semantic-segment embeddings inside a residual refinement loop and inside the smoothness regulariser simultaneously. Concurrent with SegStereo (ECCV 2018) but with three explicit differences (Sect. II, p. 2): (1) unsupervised disparity training (SegStereo is supervised); (2) full 5D concatenation cost volume rather than SegStereo's correlation layer (claimed less information loss); (3) segment embedding enters both the refinement features and the smoothness loss term, not only the feature path.

## Key Benchmark Numbers
- **Params:** not in paper.
- **GFLOPs:** not in paper.
- **Latency / FPS / target GPU:** 0.9 s on NVIDIA Titan-X (Tab. II runtime column, p. 7), reported alongside GC-Net (0.9 s) and PSMNet (0.41 s). No FPS quoted directly; the implied rate is ~1.1 FPS at the unstated KITTI resolution.

**Stereo, KITTI 2015 unsupervised validation, NOC / All percent-erroneous (Tab. I, p. 6):**
- DispSegNet (CS + K + pp): **5.20 / 5.67%**.
- DispSegNet (K + pp): 5.29 / 5.69.
- DispSegNet (K, no pp): 5.93 / 6.32.
- Best prior unsupervised baseline (Luo et al. 2018): 6.31 / 6.63. SegStereo unsupervised: 7.70 / 8.79.

**Stereo, KITTI 2015 test set (Tab. II, p. 7), NOC D1-bg / D1-fg / D1-all, All D1-bg / D1-fg / D1-all:**
- DispSegNet: **3.86 / 15.89 / 5.84 / 4.20 / 16.97 / 6.33**.
- DispNet (supervised): 4.11 / 3.72 / 4.05 / 4.32 / 4.41 / 4.34. DispSegNet beats DispNet on D1-bg but loses badly on D1-fg.
- PSMNet (supervised): 1.71 / 4.31 / 2.14 / 1.86 / 4.62 / 2.32.

**EPE / RMSE / Scene Flow numbers:** not in paper (no Scene Flow training reported).

**Semantic, KITTI 2015 (Sect. IV-E, p. 6):** baseline mIoU 47.6%; after disparity refinement coupling, mIoU drops slightly to **46.9%** on 40 val images. **The shared training is net-negative for segmentation.**

## Mutual-Task Coupling: Load-Bearing or Decorative?
The ablation (Tab. IV, p. 7) is unusually clean about this:
- Without smoothness loss, without seg loss (only `L_p^init + L_c^init + L_r^init + L_p^ref + L_c^ref`): NOC 7.04 / All 8.60.
- Add semantic smoothness (`L_s^ref` but no `L_seg`): NOC 6.70 / All 8.14. **Delta = -0.34 NOC.**
- Add `L_seg` supervision on top: NOC 5.99 / All 6.42. **Delta = -0.71 NOC.**
- Full model: NOC 5.93 / All 6.32.

So **the segmentation supervision contributes ~0.7 NOC % out of the 7.04 -> 5.93 total improvement, roughly half of the improvement.** The semantic *smoothness* term gives a further ~0.3% by stopping the smoothness loss from blurring small objects. The per-class analysis in Tab. III, p. 7 confirms the mechanism honestly: with smoothness alone, error rates on small classes (poles, traffic signs) actually *go up*; with seg supervision added on top, small-class errors drop substantially. This is the cleanest "load-bearing for stereo" evidence in the multi-task stereo literature.

Verdict: **Load-bearing for stereo, but at the cost of segmentation quality (mIoU drops 0.7 points after joint training).** The coupling is real and asymmetric in the opposite direction of TiCoSS: here, segmentation pays a tax so that stereo wins. The mechanism (preventing the smoothness term from smearing across object boundaries) is interpretable and reproducible, not magic.

## Relevance to Our Project
- **Architectural cost.** ResNet-50 Siamese plus a 3D encoder-decoder cost-volume regressor is on the order of 30-50 M params (the paper does not report). Two orders of magnitude over StereoLite's 2.5 M mid-tier envelope. Cannot port directly.
- **The semantic-gated smoothness idea is portable and cheap.** The trick `e^{-|grad f_L|}` modulating smoothness is a one-line addition to our `loss_stack_d1` cocktail; it requires only a per-pixel "is this an object boundary" signal. We could substitute Sobel of the *image* for `f_L` (the cheapest variant) and test it on the 100-pair overfit harness with zero new parameters. Worth a 15-minute A/B.
- **Unsupervised pretraining lesson.** DispSegNet trains on Cityscapes with no disparity GT using `L_p + L_c + L_r` (photometric + LR-consistency + edge-aware regularisation). For our distillation pipeline (Stage-3 KD from FoundationStereo), this same photometric warping loss could be a Stage-0 self-supervised warm-up before the FoundationStereo pseudo-GT distillation step. Free signal from any unlabelled stereo footage.
- **The cost of segmentation supervision is real.** This paper is honest that joint training hurts segmentation quality even though it helps stereo. Drones / mobile robots that need *both* tasks at the edge probably do not want this trade. For us, who only need stereo, the multi-task framework provides no obvious benefit beyond the smoothness regulariser idea.
- **Foreground failure mode is documented.** DispSegNet's D1-fg = 16.97% (Tab. II) vs. supervised methods at 4-5% confirms that unsupervised photometric loss collapses on textureless / reflective foreground (car windshields), regardless of semantic coupling. Echoes our own MB14 catastrophic-failure finding (CLAUDE.md): textureless surfaces are a structural problem the loss formulation cannot fix.

## Limitations / What This Paper Doesn't Solve
- **Foreground regions remain catastrophic.** D1-fg = 16.97% vs 4-5% for supervised methods (Tab. II, p. 7); the paper attributes this to occlusion + reflection in foreground objects (cars), which kills the photometric loss. Semantic embedding does not rescue these regions.
- **No Scene Flow training reported.** Only KITTI 2015 / Cityscapes results, so cross-dataset generalisation (which our MB14 eval revealed as the critical question) is unmeasurable.
- **Joint training hurts segmentation.** mIoU drops from 47.6 to 46.9 after disparity refinement is wired in (Sect. IV-E, p. 6). The paper concedes: "the disparity loss forces features to be different even within a semantic class."
- **Maximum batch size = 1 on a Titan-X** (Sect. IV-B, p. 5) at 256x512 crop; this is a heavyweight model even at the time.
- **Edge / real-time deployment never addressed.** 0.9 s per inference on a Titan-X is ~3-10x too slow even for non-real-time autonomous-vehicle pipelines.
