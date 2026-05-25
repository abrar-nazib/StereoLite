# InvPT: Inverted Pyramid Multi-task Transformer for Dense Scene Understanding

**Authors:** Hanrong Ye, Dan Xu (HKUST, Department of Computer Science and Engineering)
**Venue:** ECCV 2022
**Tier:** 3 (transformer-based multi-task dense prediction; first MTL transformer for joint spatial + cross-task interaction; sets up the lineage that TaskPrompter later extends)

---

## Core Idea
Existing multi-task dense-prediction networks model spatial and cross-task interactions inside CNN locality, which prevents true global reasoning across the image and across tasks at the same time (Sec. 1, p. 2). InvPT argues that for pixel-level multi-task scene understanding the network needs to perform self-attention jointly over (i) all spatial positions and (ii) all tasks, and to do so on *high-resolution* feature maps rather than at the heavily-downsampled ViT bottleneck. The contribution is a transformer-decoder design (Fig. 2, p. 4) that produces task-specific dense features through an "inverted pyramid": instead of dropping resolution toward the output, it progressively doubles the spatial size across three stages while computing efficient pooled-key self-attention.

## Architecture
- **Encoder** is a shared ViT or Swin (Sec. 3.2, p. 5). Configurations explored: Swin-T/B/L and ViT-B/L (Tab. 2, p. 12).
- **Preliminary decoders** (Sec. 3.3, p. 6): two Conv-BN-ReLU blocks per task; emit task-specific feature `F_t^d` and preliminary prediction `P_t`. These are concatenated and projected to a common channel dim `C_0 = 768`, then flattened and concatenated across all T tasks to form `F^c` in `R^{T H_0 W_0 x C_0}`.
- **InvPT decoder**: 3 stages (Fig. 3, p. 6). Stage 0 keeps `H_0 x W_0`. Stages 1 and 2 use the **UP-Transformer block** (Fig. 4, p. 8) which (a) reshape-and-2x-upsample each task's feature map via a `Conv-BN-ReLU`, (b) compute pooled-key self-attention with `kernel_size = 2^(s+1)` to keep the global attention tractable at high resolution, and (c) inject **Attention Message Passing (AMP)** by bilinearly upsampling the prior stage's attention map (Eq. 1-2, p. 9).
- **Encoder Feature Aggregation (EFA)**: feature sequences from the first three transformer-encoder stages are projected and added into the corresponding decoder stage; multi-scale residual connections back into a transformer decoder.
- Output: feature maps from all 3 decoder stages are bilinearly aligned and summed, then split per task; each task head is a 1x1 conv (Sec. 3.4, p. 7).

## Main Innovation
Three pieces, deployed together:
1. **Joint spatial + all-task self-attention** on a single token sequence of length `T H W` (instead of doing them serially as PAD-Net / ATRC do).
2. **Inverted pyramid in the decoder**, which doubles resolution between transformer stages instead of halving it. This is the load-bearing trick because dense prediction quality depends on output resolution, but vanilla MSA at high res is `O((THW)^2)`. They control cost with pooled keys/values (`Pool(F'_s, k_s)`, `k_s = 2^(s+1)`).
3. **AMP across stages** (Eq. 2, p. 9) so the high-resolution attention at stage `s` does not have to be learned from scratch; it inherits the structure of the coarse attention from stage `s-1`.

## Key Benchmark Numbers
**Ablation, Swin-T encoder (Tab. 1, p. 11):**
- Baseline (MT, no UP-Transformer): NYUD-v2 mIoU 41.06, RMSE 0.6350; PASCAL semseg mIoU 70.92, normal mErr 14.63.
- + UTB: NYUD-v2 mIoU 43.18 (+2.12), RMSE 0.5643 (-0.0707).
- + UTB + AMP: PASCAL semseg 73.29, parsing 61.78.
- + UTB + AMP + EFA (full): NYUD-v2 mIoU **44.27**, RMSE **0.5589**, mErr 20.46, odsF 76.10; PASCAL semseg **73.93**, parsing **62.73**, saliency 84.24, normal 14.15, boundary 72.60.
- Overall MTL gain `delta_m` vs single-task: **+2.59%** on NYUD-v2, **+1.76%** on PASCAL-Context.

**State-of-the-art comparison, larger encoders (Tab. 3, p. 14):**
- NYUD-v2, ViT-L encoder: semseg mIoU **53.56**, depth RMSE **0.5183**, normal mErr **19.04**, boundary odsF **78.10**. Previous best (ATRC, CNN era) was 46.33 / 0.5363 / 20.18 / 77.94. Headline gain: **+7.23 mIoU** on semantic segmentation.
- PASCAL-Context, ViT-L encoder: semseg **79.03**, parsing **67.61**, saliency **84.81**, normal **14.15**, boundary **73.00**. Previous best (ATRC-BMTAS) at 67.67 / 62.93 / 82.29 / 14.24 / 72.42. Headline gain: **+11.36 mIoU** on semseg, **+4.68 mIoU** on parsing.

**Encoder ablation, PASCAL-Context (Tab. 2, p. 12):** moving from Swin-T (73.93 mIoU) to Swin-L (78.53) to ViT-L (79.03) all help semseg / parsing monotonically. Boundary saturates around 73.

## Multi-Task Coupling: Load-Bearing or Decorative?
The MT vs ST comparison in Tab. 1 is the cleanest answer. Single-task Swin-T baseline gets 43.29 mIoU on NYUD-v2; naive multi-task baseline collapses to 41.06 (negative transfer of -2.23). Full InvPT recovers to 44.27 (now +0.98 over the ST model), and the global `delta_m` is **+2.59%**. So the global self-attention coupling actively cancels out the negative transfer of vanilla MTL and adds a small net positive on top. The ablation also shows that pure UTB (no cross-stage messaging) already restores most of the loss; AMP and EFA each add fractions of a point. **Coupling is load-bearing in the sense that without it MTL hurts; the *additional* multi-task gains from cross-stage attention are real but small** (boundary and saliency improvements are sub-1-point).

## Relevance to Our Project
- **Param budget is incompatible.** ViT-L InvPT is in the hundreds of millions of parameters (not stated explicitly but inferable from the 24-layer ViT-L backbone). Even the Swin-T ablation chassis is far above StereoLite's 0.87 M edge envelope and ~2.5 M mid-tier envelope. Cannot port directly to drone hardware.
- **The Inverted Pyramid is a transferable design pattern**, not a recipe. The "expand resolution toward output instead of contracting it" idea matches what StereoLite's TileUpsample plus ConvexUpsample already do in a stereo-specific way: keep coarse-scale heavy computation, do fine-scale work cheaply on top. We are not learning a new paradigm here.
- **Pooled-key MSA (`kernel_size = 2^(s+1)`)** is the relevant engineering trick: if we ever add a transformer block to StereoLite (e.g. for cross-task or cross-view attention), this is the right way to keep cost down at 1/4 or 1/2 resolution.
- **Dataset gap is severe.** NYUD-v2 (795 train) and PASCAL-Context (10,581 train) provide co-labeled semseg + depth + normals + parsing + saliency + boundary. Our drone scenario has none of these jointly. To even test InvPT-style MTL on drone data, we would need to pseudo-label via SAM2 / DepthAnything / FoundationStereo, which is a separate project.
- **InvPT does not address cross-domain generalisation**, the open question for our chassis (MB14 zero-shot 40.1% D1-all). All InvPT experiments are within-dataset.

## Limitations
- **Computational cost not reported.** No FLOPs, no params, no latency numbers in the main paper. For an edge survey this is the most important missing data point.
- **Stage count fixed at 3** (Fig. 7, p. 12). The plot shows monotone improvement from 1 to 3 stages but no test at 4+, so we do not know whether the design is saturated or could go further at higher cost.
- **Negative-transfer cancellation, not amplification.** The ST -> MT delta on NYUD-v2 is -2.23 mIoU and InvPT pushes back to +0.98. So the multi-task contribution over the strong single-task baseline is **+0.98 mIoU**, not +7.23. The big headline number is over the prior MTL state-of-the-art, which started from a worse baseline.
- **No cross-domain study.** Generalisation is qualitatively tested on DAVIS (Fig. 9, p. 14) but with no quantitative numbers. Cannot judge whether the global attention learned on PASCAL transfers to driving / drone domains.
- **Authors acknowledge task competition at high capacity** (Sec. 4.1, p. 12): scaling encoder past Swin-B saturates or hurts boundary and saliency. The MTL framework does not fully resolve task interference.
