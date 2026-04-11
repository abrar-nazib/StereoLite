# Synthesis: Transformer & MRF Stereo (Tier 2)

> 9 papers exploring attention-based alternatives to correlation volumes (2021-2025).

---

## Papers Covered

| # | Paper | Year | Type | Core Idea |
|---|-------|------|------|-----------|
| 1 | **STTR** | 2021 | Pure transformer | Sequence-to-sequence matching + optimal transport |
| 2 | **ChiTransformer** | 2022 | Self-supervised | Optic-chiasma cross-attention, fisheye support |
| 3 | **CEST** | 2022 | STTR++ | Context Enhanced Path for hazardous regions |
| 4 | **GMStereo** | 2023 | Unified | Single transformer for flow+stereo+depth |
| 5 | **CroCo v2** | 2023 | Pre-training | Cross-view completion self-supervised pre-training |
| 6 | **ELFNet** | 2023 | Fusion | Evidential fusion of STTR + PCWNet |
| 7 | **GOAT** | 2024 | Transformer+iterative | Occlusion-gated global attention |
| 8 | **NMRF** | 2024 | MRF | Neural Markov Random Field with sparse proposals |
| 9 | **BridgeDepth** | 2025 | Bidirectional | Mono-stereo latent alignment |

---

## Three Waves of Transformer Stereo

### Wave 1 (2021-2022): Feasibility
**STTR** demonstrated that pure transformers can do stereo matching, introducing:
- Sequence-to-sequence matching along epipolar lines
- Optimal transport for uniqueness + occlusion detection
- No fixed disparity range

**ChiTransformer** extended to self-supervised depth via optic-chiasma cross-attention.

### Wave 2 (2022-2023): Fixing Failure Modes
- **CEST:** adds Context Enhanced Path for textureless regions
- **ELFNet:** evidential fusion of transformer + cost-volume branches
- **GMStereo:** unifies flow + stereo + depth in single architecture

### Wave 3 (2024-2025): Foundation Priors + MRF Hybrids
- **CroCo v2:** cross-view completion pre-training (foundation approach)
- **GOAT:** occlusion-gated attention + iterative refinement (hybrid)
- **NMRF:** neural MRF with k=4 sparse disparity proposals (not iterative)
- **BridgeDepth:** bidirectional mono-stereo alignment (NMRF + DAv2)

---

## Two Competing Paradigms

### Paradigm A: Replace Cost Volume with Attention
- **STTR, CEST, GMStereo:** attention directly computes matching
- **Pro:** no fixed disparity range, natural occlusion handling
- **Con:** quadratic memory in image size

### Paradigm B: Sparse Candidate Labels + Graph Inference
- **NMRF, BridgeDepth:** k=2-4 disparity hypotheses per pixel
- **Pro:** much lower memory than dense cost volume
- **Con:** requires sophisticated neural potential functions

**Neither paradigm has definitively beaten iterative methods (RAFT-Stereo / IGEV-Stereo) on KITTI**, but they offer different trade-offs.

---

## Key Insights

### Insight 1: Global context matters more than it seems
Every transformer paper (STTR, CEST, GMStereo, GOAT) addresses the fact that **correlation volumes are inherently local**. The transformer's global attention provides the global context that iterative methods struggle to aggregate.

**But:** global attention is expensive. Successful methods use **restricted/windowed attention**:
- STTR: epipolar-line only
- GOAT: occlusion-gated
- GMStereo: local 2×2 windows in refinement
- CEST: low-resolution parallel pathway

### Insight 2: Foundation pre-training works
**CroCo v2** proved that cross-view completion pre-training produces features that work for both stereo and optical flow **without any task-specific design**. This directly motivated DEFOM-Stereo and FoundationStereo.

### Insight 3: Sparse label spaces are viable
**NMRF's k=4 hypotheses** and **BridgeDepth's k=2 hypotheses** show that you don't need a dense D-dimensional cost volume. This is a key edge efficiency technique.

### Insight 4: Bidirectional mono-stereo alignment beats one-way
**BridgeDepth** shows that continuously refining both mono AND stereo (vs DEFOM-Stereo's one-way mono → stereo injection) gives better results on specular/transparent surfaces.

---

## Benchmark Comparison

| Method | KITTI 2015 D1-all | ETH3D bad 1.0 | Params |
|--------|-------------------|---------------|--------|
| STTR | 2.01% | — | — |
| ELFNet | — | — | 2× networks |
| GOAT | 1.84% | — | — |
| GMStereo | **2.64%** EPE (test, flow/stereo trade) | 1.83% | **4.2M** |
| CroCo v2 | — | **1.14%** | 88M (ViT-Base) |
| NMRF | **1.59%** | 7.5% | — |
| BridgeDepth | **1.13%** | **0.50%** | — |

**BridgeDepth (2025)** achieves the best ETH3D zero-shot **matching FoundationStereo** at much lower cost.

---

## Relevance to Our Edge Model

### Directly Adoptable

1. **Axial-Attention** (CEST) — linear complexity alternative to full self-attention, edge-friendly
2. **Sparse hypothesis spaces** (NMRF k=4, BridgeDepth k=2) — dramatic memory savings
3. **Cross-view completion pre-training** (CroCo v2) — train once, fine-tune for stereo
4. **Occlusion-gated attention** (GOAT) — global attention only where needed
5. **Parallel low-res context pathway** (CEST CEP) — cheap textureless region handling

### High-Value But Requires Adaptation

1. **GMStereo's unified architecture** — multi-task edge model (stereo + flow) valuable for robotics
2. **BridgeDepth's bidirectional alignment** — requires DAv2 ViT (too heavy) but concept transferable via MPT distillation

### Avoid for Edge

1. **Full STTR/ChiTransformer attention** — quadratic memory
2. **ELFNet's dual-network fusion** — doubles compute
3. **ViT-Base encoders** (CroCo v2) — 88M params too heavy

---

## The Big Picture

**Transformers did NOT win on KITTI** — iterative methods (RAFT, IGEV) dominate. However, transformers contributed **critical ideas** adopted by iterative methods:
- **Cross-view completion pre-training** (CroCo v2) → DEFOM-Stereo, FoundationStereo
- **Sparse hypothesis spaces** (NMRF) → edge efficiency techniques
- **Global context via attention** (GOAT, GREAT-Stereo) → hybrid approaches
- **Bidirectional alignment** (BridgeDepth) → next-generation mono-stereo fusion

**The convergence:** modern methods (DEFOM-Stereo, Pip-Stereo) use **ViT features distilled into lightweight encoders** — inheriting the transformer's representation learning benefits without the quadratic attention cost. This is the direction for our edge model.
