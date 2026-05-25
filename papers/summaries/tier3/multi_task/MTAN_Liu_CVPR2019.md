# End-to-End Multi-Task Learning with Attention (MTAN)

**Authors:** Shikun Liu, Edward Johns, Andrew J. Davison (Imperial College London, Department of Computing)
**Venue:** CVPR 2019 (arXiv:1803.10704v2, April 2019)
**Tier:** 3 (foundational multi-task dense-prediction CNN; introduces the "shared backbone + per-task soft-attention selector" recipe and the Dynamic Weight Average loss balancer)

---

## Core Idea
Cross-Stitch / NDDR / Sluice all use *two parallel backbones* with feature-fusion units between them. That doubles the backbone compute regardless of how clever the fusion is. MTAN flips the design: keep a **single shared backbone** as a "global feature pool", and attach **per-task soft-attention modules** at every block of that backbone. Each attention module produces a per-channel sigmoid mask over the shared features, picking which features each task needs. Attention modules are tiny (1x1 convs + BN + sigmoid), so the parameter overhead is roughly K * (small mask network) instead of K * (full backbone). The shared backbone is forced to learn a generalisable feature pool because both tasks consume from it; the attention masks specialise that pool per task.

## Architecture
- **Shared backbone** (Sect. 3.1, Fig. 2, p. 2): typically VGG-16 encoder + SegNet symmetric decoder for dense prediction tasks. Each backbone block is a Conv-BN-ReLU stack ending in pool/upsamp.
- **Task-specific attention modules**: one per backbone block per task. For block j of task i, the attention output is `a_hat_i^(j) = a_i^(j) ⊙ p^(j)`, where p^(j) is the shared feature, ⊙ is element-wise multiplication, and a_i^(j) in [0,1] is the learned task-attention mask (Eq. 1, p. 2). Mask network: [1x1 conv, BN, ReLU] -> [1x1 conv, BN, Sigmoid]. For blocks j >= 2, the mask network input is the concatenation of the shared feature u^(j) and the previous-block task-specific feature f^(j-1)(a_hat_i^(j-1)) (Eq. 2, p. 2).
- **Loss objective** (Sect. 3.3, Eq. 3, p. 3): standard weighted sum L_tot = sum_k lambda_k L_k. For dense tasks: cross-entropy for semantic seg (Eq. 4 p. 3); L1 norm for depth (Eq. 5 p. 3); element-wise normalised dot product for surface normals (Eq. 6 p. 3).
- **Dynamic Weight Average (DWA)** (Sect. 4.1.3, Eq. 7, p. 4): a new adaptive task-balancing method that, unlike GradNorm, requires only the task losses (not gradients). Define w_k(t-1) = L_k(t-1) / L_k(t-2); then lambda_k(t) = K * exp(w_k(t-1)/T) / sum_i exp(w_i(t-1)/T). Temperature T=2 empirically; large T -> equal weights. Cheap and effective drop-in for GradNorm.
- **Backbones evaluated**: SegNet (VGG-16-based) for dense prediction; Wide ResNet (depth 28, widening 4) for multi-domain classification on Visual Decathlon.

## Main Innovation
**Two things bundled into one paper:**

1. **Shared-backbone + per-task attention mask**, replacing the dual-backbone Cross-Stitch / NDDR paradigm. The shared backbone amortises encoder compute across K tasks; the attention masks are essentially free in latency and small in params. Compared to a Cross-Stitch / Sluice network of matched capacity, MTAN reports ~1.65-1.77x params vs ~2-3x for prior MTL methods (Tab. 2-3, p. 5-6) while delivering better numbers.

2. **Dynamic Weight Average (DWA)** loss balancer that needs only loss values (not gradients), unlike GradNorm. This makes it trivial to integrate in any training loop. Its empirical effect (Tab. 2/3) is modest but it consistently beats equal-weights and matches Weight Uncertainty.

## Key Benchmark Numbers

**CityScapes 7-class semantic segmentation + depth (Tab. 2, p. 5), SegNet/VGG-16 backbone, 128x256 input:**
- Single task (One Task): mIoU 51.09, pix-acc 90.69, depth abs-err 0.0158, rel-err 34.17.
- **MTAN (Equal Weights):** mIoU 53.04, pix-acc 91.11, abs-err 0.0144, rel-err 33.63. Param count 1.65x single-task.
- MTAN (Uncert. Weights): mIoU **53.86**, pix-acc 91.10.
- Cross-Stitch (Equal Weights): mIoU 50.08, pix-acc 90.33, abs-err 0.0154. Param count ~2x.
- Multi-Task Dense (shared encoder, task-specific decoders): mIoU 51.91. Param count 3.63x.

MTAN beats Cross-Stitch by **+2-3 mIoU** at *less* than Cross-Stitch's param count.

**NYU-v2 13-class seg + depth + surface normals (Tab. 3, p. 6), SegNet/VGG-16, 288x384 input:**
- MTAN (Equal Weights): mIoU **17.72**, pix-acc 55.32; depth abs-err **0.5906**, rel-err 0.2577; SN mean 31.44 deg, median 25.37, within-11.25 23.17%.
- Cross-Stitch (Equal Weights): mIoU 14.71, pix-acc 50.23, abs-err 0.6481, mean 33.56. **MTAN wins on every metric.**
- Single-task STAN baseline (single-task attention net): mIoU 15.73, pix-acc 52.89.

MTAN gains over Cross-Stitch: **+3.01 mIoU, +5.09 pix-acc, +9% abs-err improvement, +2.12 deg mean SN.** This is the largest single-paper margin over Cross-Stitch in the lineage.

**Param efficiency (Tab. 3):** MTAN 1.77x single-task; Cross-Stitch ~3x; Multi-Task Dense 4.95x.

**Visual Decathlon (Tab. 4 right, p. 7), 10-domain classification, Wide-ResNet-28-4:**
- MTAN score 2941, parameter factor 1.74; Piggyback 2838 @ 1.28x; Parallel SVD 3398 @ 1.5x; Scratch 1625 @ 10x. MTAN is competitive and parameter-efficient.

## Multi-Task Coupling: Load-Bearing or Decorative?

**Load-bearing — and notably, MTAN provides the cleanest *visual* evidence that the attention masks specialise per task.**

- **Removing attention (the "Multi-Task Dense" baseline in Tab. 2/3) drops mIoU from 53.04 to 51.91** on CityScapes and from 17.72 to 16.06 on NYU-v2. So the attention modules are doing ~1-2 mIoU of work over a shared-encoder + task-specific-decoder baseline.
- **Replacing attention with Cross-Stitch** drops to mIoU 50.08 / 14.71 (Tab. 2/3). MTAN wins by 3-5 mIoU against Cross-Stitch despite using fewer params.
- **STAN baseline (single-task attention, i.e. MTAN with K=1)** gets mIoU 15.73 on NYU-v2 vs MTAN's 17.72. The +2 mIoU comes specifically from *sharing* features across tasks via the global pool, not from attention itself.
- **Robustness across loss weighting schemes (Fig. 3, p. 5):** MTAN's training curves under Equal Weights, DWA, and Weight Uncertainty are nearly identical, while Cross-Stitch's curves diverge across weighting schemes. MTAN is much less sensitive to loss balancing than its predecessors — a real practical win.
- **Attention masks are visibly task-specialised (Fig. 5, p. 7):** the depth-task mask has much higher contrast than the semantic mask; semantic masks emphasise object boundaries while depth masks emphasise smooth surface regions. The masks are not collapsing to identity.

The DWA balancing scheme contribution is real but more modest: Tab. 2/3 show DWA matches or slightly trails Weight Uncertainty on dense tasks. The headline contribution is the **architecture**, not the loss balancer.

Verdict: **the shared-pool + attention-mask design is the load-bearing piece. Cross-stitch-style dual backbones are *strictly dominated* by MTAN at lower param count.** That is the strongest claim made in this lineage.

## Relevance to Our Project
- **MTAN is the architectural template most relevant to our edge-deployment constraints.** Unlike Cross-Stitch / NDDR (dual backbone), MTAN keeps **one** backbone. For us, that means: one YOLO26-truncated encoder shared between stereo and any auxiliary head, with per-task attention masks at each scale. The attention overhead is roughly 2 * (1x1 conv, C->C/4) + (1x1, C/4->C) = ~3 * C^2/4 params per scale per task. At C=64 and 3 scales and K=2 tasks, that's 3 * 4096/4 * 3 * 2 = ~18 K params — trivial against our 2.5 M budget.
- **Latency story is favourable.** Two attention modules per backbone block, each is two 1x1 convs + BN + sigmoid + element-wise mul. The forward pass adds roughly 5-10% over the bare backbone, vs the 100% cost of running a second YOLO26 encoder for Cross-Stitch.
- **DWA is essentially free and might help our loss-weight sensitivity.** Right now `stack_d1` is a hand-tuned linear sum of L1 + grad + threshold-stack + D1-hinge. DWA's recipe (lambda_k(t) = softmax-over-tasks of loss-ratio) could re-balance these dynamically during training. It needs no gradient access, no hyperparameters except temperature T=2. Single-task DWA equivalent for our multi-scale stereo losses: re-weight scale 1/16, 1/8, 1/4 supervision based on each scale's loss ratio across epochs.
- **The "single shared backbone + per-task attention" pattern is what LiteAnyStereo does implicitly.** LiteAnyStereo has one encoder, one cost-agg branch, one disparity head — but the attention-style gating between disparity-feature and context-feature within its agg blocks is structurally similar. MTAN's framing makes this an explicit, transferable design pattern.
- **The robustness-to-loss-weighting finding is the most practically useful claim.** Fig. 3 (p. 5) shows MTAN's training is stable across loss-weighting choices. For us, this means if we ever add a seg head and have to balance disparity loss vs seg loss, MTAN-style attention is more forgiving than alternatives. Cross-Stitch is fussy about loss balance; MTAN is not.

## Limitations
- **All experiments are at small input resolutions.** CityScapes is 128x256 (4x downsampled from 1024x2048); NYU-v2 is 288x384. The published numbers do not transfer cleanly to full-resolution edge deployment.
- **SegNet/VGG-16 backbone is dated.** No ResNet, no HRNet, no modern efficient backbone evaluated. Whether MTAN's attention pattern transfers to a GhostNet- or YOLO26-style backbone is open. The shape of the attention module (1x1, BN, ReLU, 1x1, BN, sigmoid) should compose with anything, but it's untested.
- **No latency or FLOPs measurement.** Param counts are reported (Tab. 2/3) but inference time isn't. The "parameter-efficient" claim is on storage, not compute.
- **No cross-domain or zero-shot eval.** Same dataset for train and test on every experiment. The MB14 zero-shot question (what kills our chassis) is not asked.
- **Per-block attention may be overkill.** The paper inserts an attention module at *every* SegNet block (~13 in encoder + 13 in decoder). No ablation on whether 2-3 attention modules at the most informative scales would suffice — a likely big latency win that the paper does not explore.
- **DWA is a soft contribution.** The deltas from DWA over Equal Weights in Tab. 2/3 are typically < 1 mIoU and within run-to-run noise. The architecture is doing the work; the loss balancer is a polish.
