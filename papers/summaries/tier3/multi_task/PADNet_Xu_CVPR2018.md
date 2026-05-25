# PAD-Net: Multi-Tasks Guided Prediction-and-Distillation Network for Simultaneous Depth Estimation and Scene Parsing

**Authors:** Dan Xu, Wanli Ouyang, Xiaogang Wang, Nicu Sebe (University of Trento, University of Sydney, CUHK)
**Venue:** CVPR 2018 (arXiv:1805.04409v1, May 2018)
**Tier:** 3 (foundational multi-task dense-prediction CNN; introduced the predict-then-distill paradigm with explicit auxiliary tasks)

---

## Core Idea
Most prior MTL networks predict all tasks **directly** from a shared trunk and hope the loss balance is right. PAD-Net inverts the pipeline: a front-end CNN first predicts a *bank of intermediate auxiliary tasks* (depth, surface normal, semantic, contour), and then a back-end **multi-modal distillation module** uses those intermediate predictions as multi-modal input features to produce the final main-task outputs (depth + parsing). The intermediate predictions act as both deep supervision for the front-end and as rich multi-modal evidence for the back-end. At inference the network still takes only RGB; the multi-modal "channels" are predicted internally.

## Architecture
- **Front-end encoder** (Sect. 3.2, p. 3): VGG-16 or ResNet-50/101 backbone. Multi-scale feature aggregation: shallower-layer feature maps are downsampled and concatenated with the last-scale feature map, with dilated convolution to expand receptive field. Resolution after the front-end is 1/8 of input.
- **Multi-task prediction module** (Sect. 3.3, p. 3): four parallel deconv branches, each producing a task-specific feature map at 1/4 input resolution. Two main-task branches (depth, parsing) get N=512 channels; two auxiliary branches (surface normal, contour) get N/2=256 channels. Score maps for all four tasks are computed by separate 1x1 conv heads and supervised with their own losses (L1 to L4 in Fig. 2 p. 3).
- **Multi-modal distillation module** (Sect. 3.4, Fig. 3, p. 4): three variants A/B/C, applied to the four intermediate score maps Y1..Y4 (depth, normal, contour, semantic). Score maps are first lifted back to feature tensors F_i^t via 1x1 conv. Then:
  - **Module A (naive concat):** F_i^o = CONCAT(F_i^1, ..., F_i^T). One shared distilled feature for both heads.
  - **Module B (per-task message passing, Eq. 1 p. 4):** F_i^{o,k} = F_i^k + sum_{t != k} W_{t,k} ⊗ F_i^t. Each main task receives a distinct distilled feature; cross-task contribution is a learned 3x3 conv on the other task's feature map.
  - **Module C (attention-gated message passing, Eq. 2-3 p. 4):** Module B's cross-task term is element-wise multiplied by an attention gate G_i^k = sigmoid(W_g^k ⊗ F_i^k). Final form: F_i^{o,k} = F_i^k + sum_{t != k} G_i^k ⊙ (W_t ⊗ F_i^t). Module C is the winning variant.
- **Decoder** (Sect. 3.5, p. 5): two deconv layers (4x upsample) followed by a final 1x1 conv per main task. Two losses for the main tasks (L5 depth, L6 parsing) on top of the four intermediate losses, total 6 supervised losses.
- **Supervision** (Sect. 3.6, p. 5): cross-entropy for contour, softmax for parsing, Euclidean for both depth and surface normal. Joint loss L_all = sum_i w_i * L_i with auxiliary task weights 0.8.

## Main Innovation
The **predict-then-distill** paradigm with **attention-gated cross-task message passing** (Module C). Two pieces matter:
1. Forcing the front-end to explicitly produce intermediate predictions of *related* tasks (normal, contour) — not just the final tasks — means the front-end feature space is regularised to encode those modalities, which depth and parsing can then exploit downstream as "free" multi-modal inputs.
2. The attention gate G_i^k means each task can *selectively reject* unhelpful cross-task signal at pixel granularity instead of indiscriminately fusing everything.

This is the architectural template that MTI-Net (multi-scale extension), PAP-Net (affinity-based extension), and almost every later MTL work directly inherits.

## Key Benchmark Numbers

**NYUD-v2 depth estimation (Tab. 4 p. 7), ResNet-50 backbone, 795 training images:**
- PAD-Net: rel 0.120, log10 0.055, rms 0.582 m, delta<1.25 0.817, delta<1.25^2 0.954, delta<1.25^3 0.987.
- Best prior single-task method (Xu et al. 2017 with 95K training images): rel 0.121, log10 0.052, rms 0.586. **PAD-Net matches state of the art with 120x less training data.**
- Joint baselines: Joint HCRF rel 0.220, Jafari et al. rel 0.157 — PAD-Net dominates.

**NYUD-v2 40-class scene parsing (Tab. 3 p. 6), ResNet-50:**
- PAD-Net: mIoU 0.502, mean accuracy 0.623, pixel accuracy 0.752.
- Best prior: RefineNet-Res152 mIoU 0.465. PAD-Net beats it by **+3.7 mIoU** with a shallower backbone.
- FCN-HHA (uses depth as extra input) mIoU 0.340. PAD-Net beats it by 16 points using **only RGB**.

**Cityscapes scene parsing (Tab. 5 p. 7), ResNet-101 backbone, fine-only training:**
- PAD-Net: IoU class 0.803, iIoU class 0.588, IoU category 0.908.
- Best prior PSPNet: 0.784 / 0.567 / 0.906. **+1.9 IoU class.**

## Multi-Task Coupling: Load-Bearing or Decorative?

**Strongly load-bearing — every removal causes a clean, measurable drop.**

The diagnostic tables (Sect. 4.2, p. 6) explicitly ablate each piece:

- **Direct multi-task baseline (Front-end + DE + SP, no distillation):** NYUD-v2 depth rel 0.260, mIoU 0.294. **Compared to single-task Front-end+DE (rel 0.265), direct MTL provides almost zero benefit** — the classic "negative transfer" trap. So just adding a second head is decorative.
- **+ Distillation Module A (naive concat):** rel drops 0.260 -> 0.248, mIoU 0.294 -> 0.308. Already +1.4 mIoU.
- **+ Module B (per-task message passing):** rel 0.230, mIoU 0.317. Big jump from A — message passing is doing real work.
- **+ Module C (attention-gated):** rel 0.221, mIoU 0.325. Another +0.8 mIoU and a clear win over Module B on depth.
- **+ Module C + simultaneous DE+SP:** rel 0.214, mIoU 0.331. Best result; another +0.6 mIoU from joint final supervision.

Total swing from "direct MTL baseline" to "full PAD-Net": **rel 0.260 -> 0.214 (-17.7%)** and **mIoU 0.294 -> 0.331 (+12.6% relative)**. That is a huge effect for a multi-task ablation.

The Tab. 6 (p. 6) MTDN-inp{0,2,3,full} sweep is the second proof: removing each auxiliary task (contour, normal) drops mIoU by 0.5 to 1.0 points. The contour + normal heads are not decoration — they are load-bearing inputs to the distillation module.

Verdict: **the distillation module is the contribution and it earns its 6 losses + 4 intermediate heads of overhead.** The attention gate is the marginal piece (Module B -> C is +0.8 mIoU) but the *prediction-then-distillation* skeleton itself is what drives the headline numbers.

## Relevance to Our Project
- **Auxiliary-task supervision is the cheapest transferable idea.** Even if we never bolt a full distillation module onto StereoLite, having the front-end produce *intermediate* supervised outputs (e.g. a coarse semantic mask at 1/8 from the YOLO26 encoder, in addition to disparity) would impose extra regularisation on the encoder for free. The PAD-Net + DE + SP -> + Module C trajectory (rel 0.221 -> 0.214) confirms that even small auxiliary signal helps.
- **Param/latency cost is prohibitive at our envelope.** ResNet-50 front-end + four parallel deconv branches at N=512 channels + Module C + two decoders is far past our 2.5 M / 60 ms ceiling. The full PAD-Net likely sits at ~50-80 M params. Would need a *heavy* distillation of the design — perhaps Module A only, at N=64-96 channels.
- **The "predict normal and contour for free from depth GT" trick is genuinely free for us.** Sect. 3.1 p. 3: "the surface normal and the contours can be directly inferred from depth and semantic labels". For stereo, disparity GT trivially yields a coarse occlusion mask and a 1/disparity gradient (depth-edge map). Two free auxiliary supervisions at zero annotation cost.
- **Stereo is a stronger geometry signal than mono depth.** PAD-Net assumes RGB input only and has to predict depth as an internal modality. We have *measured* disparity from the cost volume — much higher quality than the front-end depth prediction PAD-Net distills from. The cross-task path from disparity to a hypothetical seg head should be more useful than PAD-Net's depth-as-input path is for parsing.
- **The 3-stage KD pipeline in LiteAnyStereo is conceptually a descendant of PAD-Net's distillation idea**, applied across teacher/student instead of across tasks. The "use the powerful network to predict pseudo-modalities that the student then learns to use" pattern is the same.

## Limitations
- **No real-time path.** ResNet-50 + four deconv branches + Module C + two decoders is ~10-20 FPS on a Titan X (paper does not report latency). Cityscapes results use ResNet-101.
- **Auxiliary tasks must be derivable from existing GT.** PAD-Net is silent on what to do when the auxiliary signal you want (e.g. semantics) does not have GT in your dataset. For stereo deployment data (no semantic labels) this is the bottleneck.
- **Distillation operates at a single scale (1/4 resolution).** This is the failure mode MTI-Net (2020) explicitly attacks: task interactions can differ across receptive-field sizes, and a single-scale distillation misses that. Confirmed empirically by MTI-Net Tab. 3 (Pad-Net = single-scale distillation gets -0.02% multi-task delta on NYUD-v2 while multi-scale gets +6.4%).
- **No cross-domain numbers.** All NYUD-v2 and Cityscapes results are within-dataset. The depth+parsing model trained on Cityscapes is never evaluated on KITTI or Mapillary, which is exactly the failure mode we care about for stereo deployment.
- **The "we don't need extra annotation" claim is partly misleading** — they do need semantic GT to derive contours. For stereo, where we want detection + disparity, the equivalent would require detection GT to derive an objectness contour map. Not free in our setting.
