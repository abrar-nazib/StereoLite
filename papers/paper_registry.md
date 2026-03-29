# Paper Registry: Stereo Matching Research

> Extracted from [Awesome-Deep-Stereo-Matching](https://github.com/fabiotosi92/Awesome-Deep-Stereo-Matching)
> Total unique papers: ~190 | Duplicate entries across categories: ~50+
> Last updated: 2026-03-28

## Priority Scoring Criteria

| Score | Meaning | Guideline |
|-------|---------|-----------|
| 10 | Essential | Directly foundational to our edge model or defines the field |
| 9 | Critical | Key architecture/technique we must understand deeply |
| 8 | Very Important | Significant innovation relevant to our approach |
| 7 | Important | Valuable for review paper and understanding paradigm evolution |
| 6 | Moderately Important | Relevant technique or dataset for comparison |
| 5 | Useful | Provides context, benchmarks, or comparison points |
| 4 | Somewhat Relevant | Niche but worth mentioning in review |
| 3 | Low Priority | Peripheral to main goals |
| 2 | Very Low | Tangentially related (beyond-RGB, events, etc.) |
| 1 | Minimal | Brief mention at most |
| 0 | Skip | Duplicate or not relevant |

---

## TIER 1: MUST-READ (Priority 9-10) — 25 papers

These papers are absolutely essential for our project goals.

### Surveys & Foundations (read first)

| # | Paper | Authors | Venue/Year | Priority | Why |
|---|-------|---------|------------|----------|-----|
| 1 | **A Survey on Deep Stereo Matching in the Twenties** | Tosi et al. | IJCV 2025 | **10** | The definitive modern survey — our review paper builds on this |
| 2 | **A taxonomy and evaluation of dense two-frame stereo correspondence algorithms** | Scharstein & Szeliski | IJCV 2002 | **10** | The foundational taxonomy paper that defined the field |
| 3 | **SGM: Stereo processing by semiglobal matching and mutual information** | Hirschmuller | TPAMI 2007 | **9** | Classical baseline everyone compares against |
| 4 | **On the synergies between machine learning and binocular stereo** | Poggi et al. | TPAMI 2021 | **9** | Comprehensive ML+stereo survey |
| 5 | **Computer Vision: Algorithms and Applications (Ch. 12)** | Szeliski | Book, 2nd Ed | **8** | Textbook reference for fundamentals |

### Foundation-Model Era (our primary focus)

| # | Paper | Authors | Venue/Year | Priority | Why |
|---|-------|---------|------------|----------|-----|
| 6 | **DEFOM-Stereo: Depth Foundation Model Based Stereo Matching** | Jiang et al. | CVPR 2025 | **10** | Our primary model inspiration |
| 7 | **FoundationStereo: Zero-Shot Stereo Matching** | Wen et al. | CVPR 2025 | **10** | SOTA zero-shot; key competitor/reference |
| 8 | **MonSter: Marry Monodepth to Stereo Unleashes Power** | Cheng et al. | CVPR 2025 | **9** | Mono+stereo fusion — same paradigm as DEFOM |
| 9 | **Stereo Anywhere: Robust Zero-Shot Deep Stereo Matching** | Bartolomei et al. | CVPR 2025 | **9** | Zero-shot robustness via mono priors |
| 10 | **AIO-Stereo: All-in-One: Transferring Vision Foundation Models** | Zhou et al. | AAAI 2025 | **8** | Foundation model transfer to stereo |
| 11 | **D-FUSE: Diving into the Fusion of Monocular Priors** | Yao et al. | ICCV 2025 | **8** | Systematic study of mono prior fusion |
| 12 | **PromptStereo: Zero-Shot via Structure and Motion Prompts** | Wang et al. | CVPR 2026 | **8** | Latest zero-shot approach |

### Iterative Architectures (RAFT lineage — our backbone)

| # | Paper | Authors | Venue/Year | Priority | Why |
|---|-------|---------|------------|----------|-----|
| 13 | **RAFT-Stereo: Multilevel Recurrent Field Transforms** | Lipson et al. | 3DV 2021 | **10** | The iterative paradigm foundation — our architecture basis |
| 14 | **CREStereo: Practical Stereo Matching via Cascaded Recurrent Network** | Li et al. | CVPR 2022 | **9** | Key iteration on RAFT with adaptive correlation |
| 15 | **IGEV-Stereo: Iterative Geometry Encoding Volume** | Xu et al. | CVPR 2023 | **9** | Combined geometry encoding + iterative updates |
| 16 | **Selective-Stereo: Adaptive Frequency Information Selection** | Wang et al. | CVPR 2024 | **8** | Frequency-aware refinement of iterative matching |
| 17 | **IGEV++: Iterative Multi-range Geometry Encoding Volumes** | Xu et al. | TPAMI 2025 | **9** | Latest IGEV evolution, strong SOTA baseline |

### Efficient/Edge Architectures (directly relevant to our model)

| # | Paper | Authors | Venue/Year | Priority | Why |
|---|-------|---------|------------|----------|-----|
| 18 | **BGNet: Bilateral Grid Learning for Stereo Matching** | Xu et al. | CVPR 2021 | **9** | Bilateral grid — key efficiency technique we may adopt |
| 19 | **Separable-Stereo: Separable Convolutions for 3D Stereo** | Rahim et al. | ICIP 2021 | **9** | Depthwise separable 3D convs — directly relevant to our design |
| 20 | **Fast-FoundationStereo: Real-Time Zero-Shot Stereo** | Wen et al. | CVPR 2026 | **9** | Edge-optimized version of FoundationStereo — closest to our goal |
| 21 | **LightStereo: Channel Boost for Efficient 2D Cost Aggregation** | Guo et al. | ICRA 2025 | **8** | Latest efficient 2D cost volume approach |
| 22 | **BANet: Bilateral Aggregation Network for Mobile Stereo** | Xu et al. | ICCV 2025 | **8** | Mobile-targeted stereo matching |
| 23 | **LiteAnyStereo: Efficient Zero-Shot Stereo Matching** | Jing et al. | arXiv 2025 | **8** | Lightweight zero-shot — very aligned with our goal |
| 24 | **GGEV: Generalized Geometry Encoding Volume for Real-time** | Liu et al. | AAAI 2026 | **8** | Real-time IGEV variant |
| 25 | **Pip-Stereo: Progressive Iterations Pruner** | Zheng et al. | CVPR 2026 | **8** | Iteration pruning for efficiency — directly applicable |

---

## TIER 2: HIGH PRIORITY (Priority 7-8) — 35 papers

### Foundational End-to-End Architectures

| # | Paper | Authors | Venue/Year | Priority | Why |
|---|-------|---------|------------|----------|-----|
| 26 | **DispNet-C** (SceneFlow paper) | Mayer et al. | CVPR 2016 | **8** | First end-to-end stereo + introduced SceneFlow dataset |
| 27 | **GC-Net: End-to-end learning of geometry and context** | Kendall et al. | ICCV 2017 | **8** | Introduced 3D cost volume paradigm |
| 28 | **PSMNet: Pyramid Stereo Matching Network** | Chang et al. | CVPR 2018 | **8** | SPP + stacked hourglass — widely cited baseline |
| 29 | **GA-Net: Guided Aggregation Net** | Zhang et al. | CVPR 2019 | **7** | Semi-global + local guided aggregation |
| 30 | **GWCNet: Group-wise Correlation Stereo Network** | Guo et al. | CVPR 2019 | **7** | Group-wise correlation cost volume |
| 31 | **CFNet: Cascade and Fused Cost Volume** | Shen et al. | CVPR 2021 | **7** | Multi-scale cascade cost volume |
| 32 | **ACVNet: Attention Concatenation Volume** | Xu et al. | CVPR 2022 | **7** | Attention-based cost volume construction |
| 33 | **MC-CNN: Stereo matching by training a CNN to compare patches** | Zbontar & LeCun | JMLR 2016 | **8** | Pioneering learned matching cost |

### Efficient Architectures

| # | Paper | Authors | Venue/Year | Priority | Why |
|---|-------|---------|------------|----------|-----|
| 34 | **StereoNet: Guided hierarchical refinement for real-time** | Khamis et al. | ECCV 2018 | **7** | Early real-time stereo network |
| 35 | **DeepPruner: Learning efficient stereo via differentiable patchmatch** | Duggal et al. | ICCV 2019 | **7** | Adaptive cost volume sampling |
| 36 | **AnyNet: Anytime stereo image depth estimation on mobile** | Wang et al. | ICRA 2019 | **7** | Mobile stereo with anytime prediction |
| 37 | **HITNet: Hierarchical Iterative Tile Refinement** | Tankovich et al. | CVPR 2021 | **7** | Google's real-time stereo approach |
| 38 | **AANet: Adaptive Aggregation Network** | Xu et al. | CVPR 2020 | **7** | Efficient stereo without 3D convolutions |
| 39 | **MobileStereoNet: Lightweight Deep Networks** | Shamsafar et al. | WACV 2022 | **7** | Purpose-built mobile architecture |
| 40 | **CoEX: Correlate-and-Excite for Real-Time Stereo** | Bangunharcana et al. | IROS 2021 | **7** | Cost volume excitation for speed |
| 41 | **ADStereo: Efficient Stereo with Adaptive Downsampling** | Wang et al. | TIP 2023 | **7** | Adaptive downsampling strategy |
| 42 | **Distill-then-Prune for Edge Devices** | Pan et al. | ICRA 2024 | **7** | Distillation+pruning for edge — directly relevant |
| 43 | **FadNet: Fast and Accurate Network** | Wang et al. | ICRA 2020 | **7** | Fast disparity estimation |
| 44 | **AAFS: Attention-Aware Feature Aggregation on Edge** | Chang et al. | ACCV 2020 | **7** | Edge-device targeted attention |

### Transformer & Advanced Architectures

| # | Paper | Authors | Venue/Year | Priority | Why |
|---|-------|---------|------------|----------|-----|
| 45 | **STTR: Revisiting Stereo from Sequence-to-Sequence with Transformers** | Li et al. | ICCV 2021 | **7** | First transformer for stereo |
| 46 | **GMStereo/UniMatch: Unifying Flow, Stereo and Depth** | Xu et al. | TPAMI 2023 | **8** | Unified architecture — important for understanding trends |
| 47 | **CroCo v2: Improved Cross-View Completion Pre-training** | Weinzaepfel et al. | ICCV 2023 | **7** | Cross-view pre-training for stereo |
| 48 | **NMRF: Neural Markov Random Field for Stereo** | Guan et al. | CVPR 2024 | **7** | Novel MRF-based approach |
| 49 | **BridgeDepth: Bridging Monocular and Stereo with Latent Alignment** | Guan et al. | ICCV 2025 | **7** | Mono-stereo bridge via latent alignment |

### NAS for Stereo

| # | Paper | Authors | Venue/Year | Priority | Why |
|---|-------|---------|------------|----------|-----|
| 50 | **LEAStereo: Hierarchical NAS for Deep Stereo** | Cheng et al. | NeurIPS 2020 | **7** | NAS-designed efficient stereo |
| 51 | **EASNet: Searching elastic and accurate network** | Wang et al. | ECCV 2022 | **7** | Elastic architecture search |

### Key Iterative Variants

| # | Paper | Authors | Venue/Year | Priority | Why |
|---|-------|---------|------------|----------|-----|
| 52 | **CREStereo++: Uncertainty Guided Adaptive Warping** | Jing et al. | ICCV 2023 | **7** | Improved CREStereo with uncertainty |
| 53 | **DLNR: High-Frequency Stereo Matching** | Zhao et al. | CVPR 2023 | **7** | High-frequency detail recovery |
| 54 | **MoCha-Stereo: Motif Channel Attention** | Chen et al. | CVPR 2024 | **7** | Channel attention for stereo matching |
| 55 | **ICGNet: Intra-view and Cross-view Geometric Knowledge** | Gong et al. | CVPR 2024 | **7** | Geometric knowledge for stereo |
| 56 | **Stereo Anything: Unifying with Large-Scale Mixed Data** | Guo et al. | arXiv 2024 | **7** | Large-scale mixed training approach |
| 57 | **GREAT-Stereo: Global Regulation via Attention Tuning** | Li et al. | ICCV 2025 | **7** | Attention tuning approach |

### Datasets (essential for benchmarking)

| # | Paper | Authors | Venue/Year | Priority | Why |
|---|-------|---------|------------|----------|-----|
| 58 | **KITTI 2012** | Geiger et al. | CVPR 2012 | **8** | Primary autonomous driving benchmark |
| 59 | **KITTI 2015** | Menze et al. | CVPR 2015 | **8** | Updated KITTI with scene flow |
| 60 | **Middlebury v3** | Scharstein et al. | GCPR 2014 | **8** | Gold-standard indoor benchmark |
| 61 | **ETH3D** | Schops et al. | CVPR 2017 | **7** | Multi-view stereo benchmark |
| 62 | **Freiburg SceneFlow** | Mayer et al. | CVPR 2016 | **8** | Standard synthetic training set |
| 63 | **Spring: High-Resolution High-Detail** | Mehl et al. | CVPR 2023 | **7** | Modern high-res benchmark |
| 64 | **Booster: Open Challenges in Deep Stereo** | Ramirez et al. | CVPR 2022 | **7** | Challenging real-world benchmark |

---

## TIER 3: MODERATE PRIORITY (Priority 5-6) — 45 papers

### Domain Shift & Generalization

| # | Paper | Authors | Venue/Year | Priority | Why |
|---|-------|---------|------------|----------|-----|
| 65 | **DSM-Net: Domain-invariant Stereo Matching** | Zhang et al. | ECCV 2020 | **6** | Domain-invariant features |
| 66 | **FCStereo: Feature Consistency for Domain Generalized Stereo** | Zhang et al. | CVPR 2022 | **6** | Feature consistency perspective |
| 67 | **GraftNet: Domain Generalized Stereo** | Liu et al. | CVPR 2022 | **6** | Broad-spectrum features |
| 68 | **HVT: Hierarchical Visual Transformation** | Chang et al. | CVPR 2023 | **6** | Visual transformation for generalization |
| 69 | **MRL-Stereo: Masked Representation Learning** | Rao et al. | CVPR 2023 | **5** | Masked learning for generalization |
| 70 | **DKT-Stereo: Robust Synthetic-to-Real Transfer** | Zhang et al. | CVPR 2024 | **6** | Knowledge transfer for domain shift |

### Self-Supervised & Missing GT

| # | Paper | Authors | Venue/Year | Priority | Why |
|---|-------|---------|------------|----------|-----|
| 71 | **MonoDepth: Unsupervised Monocular Depth with L-R Consistency** | Godard et al. | CVPR 2017 | **6** | Foundational self-supervised depth |
| 72 | **NeRF-Supervised Deep Stereo** | Tosi et al. | CVPR 2023 | **6** | Novel supervision via NeRF |
| 73 | **Reversing-Stereo: Self-supervised via monocular distillation** | Aleotti et al. | ECCV 2020 | **5** | Mono-to-stereo distillation |

### Over-Smoothing & Refinement

| # | Paper | Authors | Venue/Year | Priority | Why |
|---|-------|---------|------------|----------|-----|
| 74 | **NDR: Neural Disparity Refinement** | Aleotti et al. | 3DV 2021 | **6** | Arbitrary-resolution refinement |
| 75 | **NDR v2: Neural Disparity Refinement** | Tosi et al. | TPAMI 2024 | **6** | Extended version |
| 76 | **SMD-Nets: Stereo Mixture Density Networks** | Tosi et al. | CVPR 2021 | **6** | Multi-modal disparity distribution |
| 77 | **ADL: Adaptive Multi-Modal Cross-Entropy Loss** | Xu et al. | CVPR 2024 | **5** | Loss function for edge-preserving |
| 78 | **LaC: Local similarity pattern and cost self-reassembling** | Liu et al. | AAAI 2022 | **5** | Edge-aware cost volume |
| 79 | **Stereo Risk: Continuous Modeling Approach** | Liu et al. | ICML 2024 | **5** | Continuous disparity estimation |

### Additional Iterative/Foundational

| # | Paper | Authors | Venue/Year | Priority | Why |
|---|-------|---------|------------|----------|-----|
| 80 | **LoS: Local Structure-guided Stereo** | Li et al. | CVPR 2024 | **6** | Structure-guided matching |
| 81 | **MC-Stereo: Multi-peak Lookup and Cascade Search** | Feng et al. | 3DV 2024 | **5** | Multi-peak search range |
| 82 | **Any-Stereo: Arbitrary Scale Disparity** | Liang et al. | AAAI 2024 | **5** | Scale-arbitrary estimation |
| 83 | **EAI-Stereo: Error Aware Iterative Network** | Zhao et al. | ACCV 2022 | **5** | Error-aware iteration |
| 84 | **MGS-Stereo: Multi-Scale Geometric-Structure** | Dai et al. | TIP 2025 | **5** | Multi-scale geometric features |

### More Efficient Approaches

| # | Paper | Authors | Venue/Year | Priority | Why |
|---|-------|---------|------------|----------|-----|
| 85 | **DecNet: Decomposition Model for Stereo** | Yao et al. | CVPR 2021 | **5** | Cost volume decomposition |
| 86 | **IINet: Implicit Intra-inter Information Fusion** | Li et al. | AAAI 2024 | **5** | Real-time matching |
| 87 | **PCVNet: Parameterized Cost Volume** | Zeng et al. | ICCV 2023 | **5** | Parameterized cost volume |
| 88 | **MABNet: Lightweight multibranch adjustable bottleneck** | Xing et al. | ECCV 2020 | **5** | Lightweight design |
| 89 | **StereoVAE: Lightweight stereo on embedded GPUs** | Chang et al. | ICRA 2023 | **5** | VAE-based lightweight approach |
| 90 | **NVStereoNet: Importance of stereo for accurate depth** | Smolyanskiy et al. | CVPRW 2018 | **5** | NVIDIA's edge stereo |
| 91 | **PBCStereo: Pure Binary Convolutional Operations** | Cai et al. | ACCV 2022 | **5** | Binary convolutions for stereo |

### Transformer/Advanced Variants

| # | Paper | Authors | Venue/Year | Priority | Why |
|---|-------|---------|------------|----------|-----|
| 92 | **FormerStereo: Foundation Models for Domain Generalized Stereo** | Zhang et al. | ECCV 2024 | **6** | Foundation model representations |
| 93 | **ViTAStereo: Vision Foundation Model's Strengths** | Zhang et al. | T-IV 2024 | **6** | ViT adaptation for stereo |
| 94 | **ELFNet: Evidential Local-global Fusion** | Lou et al. | ICCV 2023 | **5** | Evidential fusion |
| 95 | **GOAT: Global Occlusion-Aware Transformer** | Liu et al. | WACV 2024 | **5** | Occlusion handling with transformers |
| 96 | **Chitransformer: Reliable Stereo From Cues** | Su et al. | CVPR 2022 | **5** | Cue integration via transformers |
| 97 | **CEST: Context-enhanced stereo transformer** | Guo et al. | ECCV 2022 | **5** | Context enhancement |

### Additional Datasets

| # | Paper | Authors | Venue/Year | Priority | Why |
|---|-------|---------|------------|----------|-----|
| 98 | **TartanAir** | Wang et al. | IROS 2020 | **6** | Large-scale synthetic for training |
| 99 | **DrivingStereo** | Yang et al. | CVPR 2019 | **5** | Large-scale driving stereo |
| 100 | **Cityscapes** | Cordts et al. | CVPR 2016 | **5** | Urban scene understanding |
| 101 | **InStereo2K** | Bao et al. | SCIS 2020 | **5** | Indoor stereo dataset |
| 102 | **FSD (FoundationStereo Dataset)** | Wen et al. | CVPR 2025 | **6** | Foundation model training data |
| 103 | **WMGStereo: What Makes Good Synthetic Training Data** | Yan et al. | arXiv 2025 | **6** | Synthetic data quality study |

### Confidence Estimation (brief coverage)

| # | Paper | Authors | Venue/Year | Priority | Why |
|---|-------|---------|------------|----------|-----|
| 104 | **On the Confidence of Stereo Matching in Deep-Learning Era** | Poggi et al. | TPAMI 2022 | **6** | Confidence survey |
| 105 | **SEDNet: Error distribution for joint disparity and uncertainty** | Chen et al. | CVPR 2023 | **5** | Joint disparity+uncertainty |

### Codebases & Benchmarks

| # | Paper | Authors | Venue/Year | Priority | Why |
|---|-------|---------|------------|----------|-----|
| 106 | **OpenStereo: Comprehensive Benchmark** | Xianda et al. | arXiv 2023 | **6** | Unified benchmark framework |
| 107 | **Exploring Pre-trained Features for Stereo** | Zhang et al. | IJCV 2024 | **6** | Feature analysis |

---

## TIER 4: LOWER PRIORITY (Priority 3-4) — 45 papers

### Matching Cost (historical)

| # | Paper | Authors | Venue/Year | Priority | Why |
|---|-------|---------|------------|----------|-----|
| 108 | **Deep Embed** | Chen et al. | ICCV 2015 | **4** | Early learned matching cost |
| 109 | **Content CNN** | Luo et al. | CVPR 2016 | **4** | Efficient learned matching |
| 110 | **SDC: Stacked Dilated Convolution** | Schuster et al. | CVPR 2019 | **4** | Unified descriptor network |
| 111 | **CBMV: Coalesced Bidirectional Matching Volume** | Batsos et al. | CVPR 2018 | **3** | Bidirectional matching volume |

### Optimization (historical)

| # | Paper | Authors | Venue/Year | Priority | Why |
|---|-------|---------|------------|----------|-----|
| 112 | **Sgm-Nets: SGM with neural networks** | Seki et al. | CVPR 2017 | **4** | Learned SGM penalties |
| 113 | **SGM-Forest** | Schonberger et al. | ECCV 2018 | **3** | SGM proposal fusion |

### Refinement (historical)

| # | Paper | Authors | Venue/Year | Priority | Why |
|---|-------|---------|------------|----------|-----|
| 114 | **DRR: Detect, Replace, Refine** | Gidaris et al. | CVPR 2017 | **3** | Early refinement approach |
| 115 | **LRCR: Left-Right Comparative Recurrent** | Jie et al. | CVPR 2018 | **3** | Recurrent refinement |

### Additional 3D Architectures

| # | Paper | Authors | Venue/Year | Priority | Why |
|---|-------|---------|------------|----------|-----|
| 116 | **CasStereo: Cascade Cost Volume** | Gu et al. | CVPR 2020 | **4** | Cascade multi-view stereo |
| 117 | **HSMNet: Hierarchical deep stereo on high-res** | Yang et al. | CVPR 2019 | **4** | High-resolution stereo |
| 118 | **PDSNet: Practical Deep Stereo** | Tulyakov et al. | NeurIPS 2018 | **4** | Practical stereo |
| 119 | **PCWNet: Pyramid Combination and Warping** | Shen et al. | ECCV 2022 | **4** | Pyramid warping cost volume |
| 120 | **Coatrsnet: Convolution and attention by region separation** | Junda et al. | IJCV 2024 | **4** | Conv+attention fusion |

### Additional 2D Architectures

| # | Paper | Authors | Venue/Year | Priority | Why |
|---|-------|---------|------------|----------|-----|
| 121 | **EdgeStereo** | Song et al. | ACCV 2018 | **4** | Edge-aware stereo |
| 122 | **CRL: Cascade Residual Learning** | Pang et al. | CVPRW 2017 | **3** | Cascade residual |
| 123 | **iResNet** | Liang et al. | CVPR 2018 | **3** | Feature constancy |
| 124 | **HD3: Hierarchical Discrete Distribution** | Yin et al. | ICCV 2019 | **4** | Distribution-based matching |
| 125 | **Bi3D: Stereo via Binary Classifications** | Badki et al. | CVPR 2020 | **4** | Binary classification approach |

### Online/Offline Adaptation

| # | Paper | Authors | Venue/Year | Priority | Why |
|---|-------|---------|------------|----------|-----|
| 126 | **MadNet: Real-Time Self-Adaptive Deep Stereo** | Tonioni et al. | CVPR 2019 | **4** | Self-adaptive stereo |
| 127 | **FedStereo: Federated Online Adaptation** | Poggi et al. | CVPR 2024 | **4** | Federated learning for stereo |
| 128 | **AdaStereo: Simple and Efficient Adaptive Stereo** | Song et al. | CVPR 2021 | **4** | Domain adaptation |
| 129 | **CST-Stereo: Consistency-aware Self-Training** | Zhou et al. | CVPR 2025 | **4** | Self-training for adaptation |
| 130 | **ITSA: Information-Theoretic Shortcut Avoidance** | Chuah et al. | CVPR 2022 | **4** | Shortcut avoidance |

### Multi-Task

| # | Paper | Authors | Venue/Year | Priority | Why |
|---|-------|---------|------------|----------|-----|
| 131 | **SegStereo: Exploiting Semantic for Disparity** | Yang et al. | ECCV 2018 | **4** | Joint segmentation+stereo |
| 132 | **Multi-Task Learning Using Uncertainty** | Kendall et al. | CVPR 2018 | **4** | Multi-task uncertainty weighting |
| 133 | **RAFT-3D: Scene Flow using Rigid-Motion** | Teed et al. | CVPR 2021 | **4** | RAFT extended to scene flow |

### Temporal Consistency

| # | Paper | Authors | Venue/Year | Priority | Why |
|---|-------|---------|------------|----------|-----|
| 134 | **DynamicStereo: Consistent Dynamic Depth** | Karaev et al. | CVPR 2023 | **4** | Temporal consistency in stereo |
| 135 | **TemporalStereo: Efficient Spatial-Temporal** | Zhang et al. | IROS 2023 | **4** | Efficient temporal stereo |
| 136 | **XR-Stereo: 100+ FPS Video Stereo** | Cheng et al. | WACV 2024 | **4** | Video stereo for XR |
| 137 | **Temporally Consistent Stereo Matching** | Zeng et al. | ECCV 2024 | **4** | Temporal consistency |

### Adverse Weather & Challenges

| # | Paper | Authors | Venue/Year | Priority | Why |
|---|-------|---------|------------|----------|-----|
| 138 | **FoggyStereo: Stereo with Fog Volume** | Yao et al. | CVPR 2022 | **3** | Fog handling |
| 139 | **RobuSTereo: Robust Zero-Shot under Adverse Weather** | Wang et al. | ICCV 2025 | **4** | Weather robustness |

---

## TIER 5: LOW PRIORITY (Priority 1-2) — 60+ papers

These are specialized papers (event cameras, thermal, polarimetric, LiDAR fusion, stereo generation, etc.) that need only brief mention in the review.

### Beyond-RGB / Event Cameras (~20 papers)
Priority 1-2 each. Include: Event-IntensityStereo, SE-CFF, SCSNet, DTC-SPADE, ADES, SAFE, TemporalEventStereo, etc.

### Cross-Spectral (~8 papers)
Priority 2-3 each. Include: CS-Stereo, UCSS, RGB-MS, DPS-Net, CrossSP, Gated-RCCB, ThermoStereoRT.

### Depth-Guided Sensor / LiDAR Fusion (~15 papers)
Priority 2-3 each. Include: LidarStereoFusion, GSD, LidarStereoNet, Pseudo-LiDAR++, VPP, SDG-Depth, etc.

### Pattern Projection / Active Stereo (~6 papers)
Priority 2-3 each. Include: ActiveStereoNet, Polka Lines, Activezero/++, MonoStereoFusion, ASGrasp.

### Scene Flow (~12 papers)
Priority 2-3 each. Include: FlowNet3.0, DRISF, SENSE, DWARF, CamLiFlow, M-FUSE, etc.

### Stereo Image/Video Generation (~18 papers)
Priority 1-2 each. Include: StereoDiffusion, StereoCrafter, SpatialMe, StereoWorld, GenStereo, etc.

### Applications (NeRF, Object Detection, etc.)
Priority 2-3 each. Include: DSGN, IDA-3D, LIGA-Stereo, StereoNeRF, GS2Mesh, etc.

### Datasets (specialized, lower priority)
Priority 2-4 each. Include: WSVD, Flickr1024, ApolloScape, Holopix50k, A2D2, WHU-Stereo, SID, etc.

### Confidence Estimation (detailed)
Priority 2-4 each. Include: CCNN, LGC/ConfNet, ENS, GCP, LEV, FA, MPN, UCN, LAF, CRNN, CVA, etc.

---

## Identified Duplicate Entries

The following papers appear in **multiple categories** in the Awesome list. They should be analyzed ONCE under their primary category:

| Paper | Times Listed | Primary Category | Also Appears In |
|-------|-------------|-----------------|-----------------|
| **NDR** | 5x | Refinement | Over-smoothing, Asymmetric, Continuous, Geometric Cues |
| **NDR v2** | 5x | Refinement | Over-smoothing, Asymmetric, Continuous, Geometric Cues |
| **DEFOM-Stereo** | 2x | Iterative | Monocular-Stereo Integration |
| **FoundationStereo** | 3x | Iterative | Monocular-Stereo Integration, Dataset |
| **MonSter** | 2x | Iterative | Monocular-Stereo Integration |
| **Stereo Anywhere** | 3x | Iterative | Monocular-Stereo Integration, ToM |
| **D-FUSE** | 3x | Iterative | Monocular-Stereo Integration, ToM |
| **AIO-Stereo** | 2x | Iterative | Monocular-Stereo Integration |
| **PromptStereo** | 2x | Iterative | Monocular-Stereo Integration |
| **Pip-Stereo** | 2x | Efficient | Monocular-Stereo Integration |
| **Fast-FoundationStereo** | 3x | Efficient | Knowledge Transfer, Lightweight |
| **LoS** | 3x | Iterative | Joint Uncertainty, Monocular-Stereo |
| **DynamicStereo** | 3x | Iterative | Transformer, Temporal Consistency |
| **XR-Stereo** | 3x | Iterative | Dataset, Temporal Consistency |
| **SEDNet** | 3x | 3D Architecture | Joint Uncertainty, Confidence |
| **HITNet** | 2x | Lightweight | Normal-Assisted |
| **CasStereo** | 2x | 3D Architecture | Efficient Cost Volume |
| **SMD-Nets** | 2x | Over-smoothing | Continuous Estimation |
| **Multi-Task Learning Using Uncertainty** | 3x | Multi-Task | Joint Semantics, Confidence |
| **RCN** | 2x | Refinement | Joint Uncertainty, Confidence |
| **ACN** | 2x | Joint Uncertainty | Confidence |
| **BridgeDepthFlow** | 2x | Joint Flow | Self-Supervised |
| **UnOS** | 2x | Joint Flow | Self-Supervised |
| **Feature-Level Collaboration** | 2x | Joint Flow | Self-Supervised |
| **StereoFlowGAN** | 2x | Joint Flow | Offline Adaptation |
| **MadNet** | 2x | Lightweight | Online Adaptation |
| **NeRF-Supervised Stereo** | 3x | Dataset | Cross-Framework, Mono-to-Synthetic |
| **SAG** | 2x | Cross-Framework | Mono-to-Synthetic |
| **MonoDepth** | 2x | Self-Supervised | Monocular Depth |
| **CS-Stereo** | 2x | Dataset Beyond-RGB | Cross-Spectral Networks |
| **RGB-MS** | 2x | Dataset Beyond-RGB | Cross-Spectral Networks |
| **DPS-Net** | 2x | Dataset Polarimetric | Cross-Spectral Networks |
| **Gated-RCCB** | 2x | Cross-Spectral | Gated Networks |
| **GatedStereo** | 2x | Dataset | Gated Networks |
| **ASGrasp** | 2x | Pattern Projection | ToM |
| **D3RoMa** | 2x | Depth-Guided | ToM |
| **SCOD** | 3x | Dataset | Transformer, Confidence |
| **StereoWorld** | 2x | Stereo Gen | (exact duplicate entry) |
| **DispNet-CSS / FlowNet3.0** | 2x | 2D Architecture | Scene Flow (same paper) |
| **Freiburg SceneFlow / DispNet-C** | 2x | Dataset | 2D Architecture (same paper) |
| **CamLiFlow** | 2x | Scene Flow | Depth-Guided Sensor |
| **O1** | 2x | Optimization | Confidence |
| **GCP** | 2x | Optimization | Confidence |
| **PBCP** | 2x | Optimization | Confidence |
| **SGM-Forest** | 2x | Optimization | Confidence |
| **LevStereo/LEV** | 2x | Optimization | Confidence |
| **ENS23/ENS7** | 2x | Confidence ML | Confidence CV |
| **BiDA-Stereo** | 2x | Iterative | Temporal Consistency |
| **Temporally-Consistent Stereo** | 2x | Iterative | Temporal Consistency |
| **TemporalStereo** | 2x | Efficient | Temporal Consistency |
| **DepthFocus** | 2x | Transformer | ToM |
| **AcfNet** | 2x | Joint Uncertainty | Over-Smoothing |
| **VPP/VPP-Extended** | 2x | Depth-Guided | (related pair) |

**Total duplicated appearances: ~100+ entries represent ~50 duplicate groups**
**Unique papers after deduplication: ~190**

---

## Reading Order Recommendation

### Phase 1: Foundations (Week 1)
1. Scharstein & Szeliski 2002 (taxonomy)
2. Hirschmuller 2007 (SGM)
3. Szeliski Ch. 12 (textbook)
4. Tosi et al. IJCV 2025 (modern survey)
5. Poggi et al. TPAMI 2021 (ML+stereo survey)

### Phase 2: Architecture Evolution (Week 2-3)
6. MC-CNN (learned matching cost)
7. DispNet-C (first end-to-end)
8. GC-Net (3D cost volume)
9. PSMNet (pyramid 3D)
10. GA-Net (guided aggregation)
11. GWCNet (group-wise correlation)
12. AANet (efficient 2D)
13. RAFT-Stereo (iterative paradigm shift)
14. CREStereo (practical iterative)
15. IGEV-Stereo (geometry encoding)
16. Selective-Stereo (frequency-aware)
17. IGEV++ (SOTA iterative)

### Phase 3: Foundation Model Era (Week 3-4)
18. GMStereo/UniMatch (unified approach)
19. CroCo v2 (pre-training)
20. DEFOM-Stereo (our inspiration)
21. FoundationStereo (zero-shot SOTA)
22. MonSter (mono+stereo)
23. Stereo Anywhere (robustness)
24. AIO-Stereo (VFM transfer)
25. D-FUSE (mono prior fusion study)

### Phase 4: Efficiency (Week 4-5)
26. StereoNet (first real-time)
27. AnyNet (mobile)
28. DeepPruner (adaptive sampling)
29. BGNet (bilateral grids)
30. Separable-Stereo (separable 3D conv)
31. HITNet (tile refinement)
32. MobileStereoNet (mobile design)
33. LightStereo (efficient 2D)
34. BANet (mobile bilateral)
35. Fast-FoundationStereo (real-time zero-shot)
36. LiteAnyStereo (efficient zero-shot)
37. GGEV (real-time IGEV)
38. Pip-Stereo (iteration pruning)
39. Distill-then-Prune (edge compression)

### Phase 5: Remaining Categories (Week 5-6)
- Domain shift, self-supervised, confidence, multi-task, beyond-RGB (skim/summarize)
