# Cross-stitch Networks for Multi-task Learning

**Authors:** Ishan Misra, Abhinav Shrivastava, Abhinav Gupta, Martial Hebert (CMU Robotics Institute)
**Venue:** CVPR 2016 (arXiv:1604.03539v1, April 2016)
**Tier:** 3 (foundational multi-task CNN; the paper that introduced learnable feature-mixing units between two task branches)

---

## Core Idea
"Split-architecture" multi-task networks force the designer to pick a single layer at which the trunk forks into two heads. The optimal split depends on the task pair and dataset, so the only honest baseline is brute force over every possible split layer. Cross-stitch units replace that discrete search with a learnable per-channel 2x2 linear mixer placed between two parallel single-task networks; the mixer alpha parameters interpolate continuously between "fully shared" and "fully separate", and gradient descent picks the right amount of sharing per layer per channel.

## Architecture
- Two parallel AlexNet/FCN-32s backbones (one per task), each pretrained as a single-task network (Sect. 5, p. 4-5).
- Cross-stitch units inserted **after every pooling layer and after fc6/fc7** (pool1, pool2, pool5, fc6, fc7 in AlexNet). Convolution-activation insertions were tried and lost to pooling-activation insertions (Sect. 5, p. 5).
- **Unit math (Eq. 1, p. 3):** for activation maps x_A, x_B at location (i,j) of layer l, the cross-stitched output is `[x_tilde_A; x_tilde_B] = [[alpha_AA, alpha_AB]; [alpha_BA, alpha_BB]] * [x_A; x_B]`. One unit per **channel** for SemSeg/SN (96 units after pool1, etc); one unit per **layer** for Det/Attr (per-channel was unstable for those tasks).
- **Backprop (Eq. 2-3, p. 4):** alpha gradients are first-order; the off-diagonal alphas (alpha_AB, alpha_BA) gate cross-task flow.
- **Initialisation (Sect. 5.1, Tab. 1, p. 5):** alphas initialised as a convex combination with (alpha_S, alpha_D) = (0.9, 0.1) — strong self-bias to start.
- **Learning rate (Sect. 5.2, Tab. 2, p. 5):** alphas use 100x the base LR; higher than 1000x makes training diverge.
- **Network init (Sect. 5.3, Tab. 3, p. 6):** sub-networks A and B must start from task-specific fine-tuned weights, not from raw ImageNet, otherwise the alphas have no signal to latch onto.

## Main Innovation
The unit itself is two scalars (alpha_S same-task, alpha_D different-task) per channel per layer, but the conceptual contribution is **making the shared-versus-task-specific decision a continuous learnable parameter instead of a discrete architectural choice**. Every later multi-task work in this lineage (Sluice, NDDR, MTAN, PAD-Net's distillation, MTI-Net's multi-scale distillation) inherits this framing. The alpha visualisation (Tab. 4, p. 7) shows the unit *re-discovers* the brute-force "best split at conv4" finding without enumeration: pool5 alphas show high self-bias (task-specific), pool1 alphas are more balanced (shared).

## Key Benchmark Numbers

**NYU-v2 (40 classes, SemSeg + Surface Normals; FCN-32s on AlexNet, Tab. 5 p. 8):**
- Surface Normal mean angle error 34.1 deg, median 18.2 deg, within-11.25-deg 39.0%.
- Semantic seg pixacc 47.2, mIoU 19.3, fwIoU 34.0.
- Best brute-force baseline (Split conv4): mean 34.7 / median 19.1 / mIoU 19.2 — cross-stitch matches or beats it on every metric.
- One-task baseline: mean 34.8 / median 19.0; mIoU 18.4.
- Ensemble (2 single-task nets, 2x params): mIoU 18.9, mean 34.4 — beaten by cross-stitch despite having more capacity.

**PASCAL VOC 2008 (Det + Attr; Fast R-CNN AlexNet, Tab. 6 p. 8):**
- Cross-stitch: Detection mAP 45.2, Attributes mAP 63.0.
- Best brute-force split: 44.8 mAP det / 61.0 mAP attr.
- One-task: 44.9 / 60.9. MTL-shared baseline catastrophically worse: 42.7 / 54.1.

**Param count:** ~2x a single-task net (effectively two AlexNets glued by trivial mixers); alphas themselves are negligible (~96 + 256 + 4096*2 = tiny). Sect. 6.1 p. 6 notes this 2x-param cost honestly.

**Data-starved category gain (Fig. 5/6, p. 7-8):** +4.6% mAP averaged over the 10 attribute classes with the fewest training instances; +4.3% over the 20 lowest. Bigger gains for tiny-data classes (saddle, sail, propeller, flower up to +19% per class), small gains for data-rich classes. This is the paper's strongest experimental finding.

## Multi-Task Coupling: Load-Bearing or Decorative?

**Load-bearing — but the strongest evidence is regularisation, not representational sharing.**

- The headline NYU-v2 deltas (mIoU 18.4 -> 19.3, mean SN error 34.8 -> 34.1) are modest absolute numbers but matter because they beat the **brute-force-best Split-conv4** baseline (mIoU 19.2). That's the key: the contribution is not "MTL helps" (everyone already knew that) but "you don't need to enumerate splits — the alphas converge to as-good-or-better".
- The data-starved gain (+4.6% mAP on rare-attribute classes) is where the unit is genuinely doing work that cannot be reproduced by tuning a single network.
- Removing cross-stitch and falling back to MTL-shared (a single trunk with two heads) collapses Det/Attr mAP from 45.2/63.0 to 42.7/54.1 (Tab. 6) — a 9-point drop on attributes. So the *mechanism* of letting each task choose its own mix is load-bearing, but the *novelty* over "two separate networks plus a fancy initialiser" is closer to 0.5-1 point on the dense-prediction tasks.
- Tab. 4 visualisations (p. 7) are the smoking gun for "decorative on some layers": at pool5 most channels have alpha_S near 1 and alpha_D near 0, meaning **the network learned to ignore cross-task flow at the late layers**. The unit is only earning its keep in the lower-mid layers (pool1-pool2) and on data-starved heads.

## Relevance to Our Project
- **Direct port to StereoLite YOLO26 + seg head is plausible.** A cross-stitch unit between a YOLO26 detection feature and a stereo feature at a matched spatial scale (say 1/8 or 1/16) would cost ~ C scalar alphas (C = channel count, typically 64-128). At 128 channels per scale and 3 scales that is ~400-800 extra trainable parameters — invisible against our 2.5 M mid-tier budget. The cost is the second forward pass through a second backbone, which doubles encoder latency. This is the killer for edge deployment, not the unit itself.
- **The brute-force-split insight is the keeper.** Even without using cross-stitch units, the paper's empirical finding (Fig. 2, p. 2) — "the best split layer depends on the task pair" — is exactly what we'd hit if we tried to bolt a seg head onto YOLO26: there is no a priori reason to fork at the same depth as the stereo head forks. Treat fork depth as a hyperparameter, not a fixed design.
- **Initialisation matters more than the unit.** Tab. 3 (p. 6) shows that random-init two-task training is worse than single-task initialisation followed by joint cross-stitch fine-tuning. If we ever add a seg head, the right recipe is: train stereo to convergence, train seg to convergence on the same encoder, *then* glue them with a mixer and joint-fine-tune. This contradicts the "train everything from scratch" default.
- **Data-starved categories are the use case.** Our stereo data is already small; if we ever add cross-domain transfer between SceneFlow + KITTI + Middlebury via an "auxiliary domain ID" task, the +4-6% mAP gain on rare categories is the kind of regularisation effect we'd be hoping for.

## Limitations
- **2x parameter cost is unavoidable.** Each task gets its own full backbone. For K tasks the cost scales linearly, which is why the MTAN / MTI-Net / NDDR-CNN successors all attack this point (shared trunk + per-task attention/dimensionality-reduction layer).
- **AlexNet-era backbone.** Numbers are pre-ResNet, pre-VGG-19, no skip connections, no batchnorm. The absolute mIoU 19.3 on NYU-v2 looks weak today (modern methods 50+); the *relative* gains are what carry.
- **No latency or FLOPs report.** The paper claims the units are cheap but never measures inference time. 2x backbones means ~2x latency in practice.
- **Tasks must take the same input.** Sect. 3, p. 3 explicitly: "we only consider tasks which take the same single input, e.g. an image as opposed to an image and a depth-map". Cross-stitching a stereo network (which needs L and R) with a mono detection network (which needs only L) is not directly handled by the paper.
- **No cross-domain or zero-shot eval.** All experiments are train/test on the same dataset. The cross-domain robustness story — central to StereoLite's current trajectory — is untouched here.
