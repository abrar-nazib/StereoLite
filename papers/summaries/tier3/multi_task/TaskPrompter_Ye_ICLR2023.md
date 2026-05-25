# TaskPrompter: Spatial-Channel Multi-Task Prompting for Dense Scene Understanding

**Authors:** Hanrong Ye, Dan Xu (HKUST, Department of Computer Science and Engineering)
**Venue:** ICLR 2023
**Tier:** 3 (transformer-based multi-task dense prediction; direct successor to InvPT from the same group; replaces InvPT's decoder-side cross-task module with per-layer prompt tokens that interact with patch tokens at every transformer layer)

---

## Core Idea
InvPT (ECCV 2022) and its peers separate three learning objectives into different modules: task-generic representations in the shared encoder, task-specific representations in a decoder, and cross-task interactions in a custom block. TaskPrompter argues this decoupling wastes capacity, because (a) you have to hand-design three module families and (b) capacity that could be shared across all three goals is fragmented (Sec. 1, p. 2). The fix is to fold all three into one mechanism: a small set of **T learnable task-prompt tokens** that travel through every transformer layer alongside the image patch tokens, and that interact with the patch tokens via **two complementary attention paths** at each layer, one along spatial dimensions and one along channel dimensions.

## Architecture
- **Prompt Embedding** (Sec. 3.1, p. 4): T learnable C-dim tokens are concatenated with N = H x W patch tokens, forming `Z_0 in R^{(T+N) x C}` (Eq. 1, p. 4). Backbone is ViT-Base (12 layers) or ViT-Large (24 layers), ImageNet-22K pretrained.
- **Spatial-Channel Task Prompt Learning module** (Sec. 3.2, Fig. 2 p. 5) replaces the standard MSA inside every transformer layer:
  - **Spatial Task Prompt Learning**: project task prompts to `P^s in R^{T x C}`; concatenate with patch tokens; run standard MSA (Eq. 3, p. 5). Task prompts attend to all patch tokens spatially, and vice versa.
  - **Channel Task Prompt Learning**: project task prompts to `P^c in R^{T x N}` via a 2-layer MLP that swaps the channel and spatial axes; compute cross-attention `Q^c = P^c W_q^c`, `K^c = (X W_k^c)^T`, `V^c = (X W_v^c)^T` (Eq. 4, p. 5). The key/value/query are then reshaped into spatial windows of size `N_head^c = N_win^h x N_win^w` to preserve local spatial adjacency along the "channel" dimension during head-partition.
  - The two updated task prompts are fused: `P' = P^{s'} + f_{N->C}(P^{c'})` (Eq. 5, p. 6).
- **Dense Spatial-Channel Task Prompt Decoding** (Sec. 3.3, p. 6, Fig. 3): for the final transformer layer, extract the task-to-patch attention slice from both `A^s` and `A^c`. The spatial slice `A_t^{p->s}` and channel slice `A_t^{p->c}` are Hadamard-producted with reshaped patch features `X'^s` and `X'^c` to yield two prompted task features `F_t^s` and `F_t^c` (Eqs. 6-7, p. 6). They are concatenated channelwise and fused with a 3x3 conv-BN-GELU (Eq. 8, p. 7).
- **Cross-task Reweighting**: project the `T x T` block of `A^s` (task-prompt-to-task-prompt attention) via a 2-layer MLP to weights `A^{p->p}`, then `F <- A^{p->p} F` rebalances task features.
- **Hierarchical Prompting (HP)**: apply the Decoder to multiple transformer layers (every 3rd in TaskPrompter-Base, 4 levels total) and sum the resulting task features before the final 3x3 conv head.

## Main Innovation
The per-layer prompt mechanism is the contribution. Two transferable design choices:
1. **Channel-axis cross-attention** between learnable task tokens and patch tokens, with window partition that preserves spatial adjacency along the channel axis (Sec. 3.2, p. 5). This is unusual in dense prediction transformers, where MSA is almost always along the spatial axis only.
2. **Affinity decoding** (Eqs. 6-7): the network's attention map itself, never trained for "decoding", is reused as a spatial/channel mask for extracting per-task dense features. Avoids learning a separate decoder.

## Key Benchmark Numbers
**Ablation on PASCAL-Context (Tab. 1, p. 8; ViT-Base baseline, MTL gain `delta_m` vs single-task models):**

| Variant | Semseg mIoU | Parsing mIoU | Saliency maxF | Normal mErr | Boundary odsF | MTL Gain |
|---|---|---|---|---|---|---|
| STL Model | 78.42 | 68.36 | 85.34 | 13.87 | 73.90 | 0 (ref) |
| TaskPrompter Baseline (MT) | 75.04 | 64.82 | 84.59 | 14.17 | 68.00 | -4.11 |
| + SPrompt | 75.95 | 65.60 | 84.91 | 13.97 | 70.50 | -2.59 |
| + SPrompt + CPrompt | 76.46 | 66.25 | 85.00 | 13.97 | 71.10 | -2.11 |
| + SPrompt + CPrompt + RW | 76.83 | 66.31 | 85.00 | 13.94 | 71.30 | -1.90 |
| **Full (+ HP)** | **79.00** | **67.00** | **85.05** | **13.47** | **73.50** | **+0.15** |

The full design is the first one in the family to push past zero on the MTL gain metric.

**Scaling, TaskPrompter-Large (ViT-L, 24 layers, Tab. 2, p. 8):**
- PASCAL-Context: semseg **80.89**, parsing **68.89**, saliency 84.83, normal 13.72, boundary 73.50.
- NYUD-v2: semseg **55.30**, depth RMSE **0.5152**, normal mErr **18.47**, boundary odsF **78.20**.

**State-of-the-art comparison (Tab. 3, p. 9):**
- vs InvPT (the prior leader from same group) on PASCAL: +1.86 semseg, +1.28 parsing.
- vs InvPT on NYUD-v2: +1.74 semseg, depth -0.0031 RMSE, normal -0.57 mErr.
- vs ATRC (best pre-transformer): semseg +13.22 (PASCAL) and +8.97 (NYUD), parsing +5.96 (PASCAL).

## Multi-Task Coupling: Load-Bearing or Decorative?
The cleanest signal in this paper. Tab. 1 starts at MTL gain -4.11 (vanilla MT baseline is much worse than separately-trained ST models on this 5-task dictionary). Each new piece chips away at the gap: SPrompt -> -2.59, +CPrompt -> -2.11, +RW -> -1.90, +HP -> **+0.15**. So the *whole package* is necessary to break even, and the **+4.26 cumulative recovery** is what the paper is actually buying. Of that, HP (hierarchical prompting at multiple layers) is doing the heaviest lift (+2.05 on its own, larger than every other piece combined). **Coupling is load-bearing**, but the productive coupling is the per-layer placement of prompt tokens through the entire transformer stack, not the spatial-vs-channel attention split.

## Relevance to Our Project
- **Param budget is incompatible at ViT-Base / ViT-Large scale.** Direct deployment on Jetson Orin Nano or Raspberry Pi 5 is out.
- **The transferable idea is "task tokens as a coupling primitive"**, not "spatial-channel decomposition". For our two-tier StereoLite + (future) detection head plan, a single shared backbone with one or two prompt tokens that route features into a stereo head vs a detection head is cheaper than full per-task decoders. This is a research direction, not a port.
- **Channel-axis attention with window partition** is potentially relevant for our cost-volume aggregator. The disparity axis in stereo plays the same structural role that "channel" plays here: it is a small dimension over which we want to attend without paying full MSA cost. Worth a sanity-check ablation on a small stereo head.
- **Affinity-decoding pattern** (reusing attention maps for output decoding instead of training a separate decoder) is parameter-efficient; if we ever try a vision-language conditioning on StereoLite, this is the cheap way to do per-task feature extraction.
- **Drone tie-in is indirect.** Drones with on-board panoptic tasks (segment + detect + depth) would benefit from a TaskPrompter-style chassis, but only after foundation-model pretraining is in place. Outside our current edge-compression scope.

## Limitations
- **No FLOPs / params / latency in the main paper.** Same gap as InvPT; the multi-task gain over ST is measured in mIoU but not in compute.
- **Negative MTL gain is the starting point.** Tab. 1 shows that *vanilla multi-task* on this 5-task dictionary is -4.11 vs single-task. So the design is not so much "multi-task helps" as "multi-task hurts by default, and we found a way to break even". This is more honest than InvPT's framing but worth flagging.
- **The "spatial + channel" framing is partially marketing.** Channel-task-prompt learning collapses to spatial cross-attention with reshaped keys (Sec. 3.2, p. 5). The contribution is "two attention paths that share computation across layers", not a new tensor-decomposition.
- **ViT-only.** No Swin or ConvNeXt result. We do not know whether the prompt mechanism transfers to hierarchical backbones, which are the ones we would actually deploy.
- **Scaling exploration is limited.** Only Base and Large tested (Tab. 2, p. 8). The standard ViT-Huge would say whether the design saturates.
