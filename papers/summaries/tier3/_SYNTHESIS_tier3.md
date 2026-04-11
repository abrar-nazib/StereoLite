# Synthesis: Tier 3 Papers — Training Data, Refinement, Uncertainty, Generalization

> 10 papers selected to fill specific gaps left by Tier 1 (foundation/iterative/efficient) and Tier 2 (architecture lineage/datasets). These cover the **support infrastructure** that makes modern stereo work: training data strategies, refinement, uncertainty, and cross-domain generalization.

---

## Papers Covered

| # | Paper | Year | Theme | Why Tier 3 (not 1/2) |
|---|-------|------|-------|----------------------|
| 1 | **MonoDepth** | 2017 | Self-supervised | Foundational *concept* for self-supervision; superseded by NeRF-Stereo for stereo training |
| 2 | **NeRF-Supervised Stereo** | 2023 | Self-supervised | Modern data factory; conceptually critical, not architecturally novel |
| 3 | **TartanAir** | 2020 | Synthetic dataset | Mandatory training corpus; not central enough for Tier 2 |
| 4 | **DKT-Stereo** | 2024 | Continual learning | Training-loop only — high impact, no architecture |
| 5 | **DSM-Net** | 2020 | Domain generalization | Seminal cross-domain paper; building block for HVT |
| 6 | **HVT** | 2023 | Domain generalization | Training-time augmentation for cross-domain |
| 7 | **SMD-Nets** | 2021 | Output representation | Bimodal output head |
| 8 | **NDR** | 2021 | Refinement | Continuous-resolution refinement |
| 9 | **SEDNet** | 2023 | Uncertainty | Calibrated uncertainty at +190 params |
| 10 | **RAFT-3D** | 2021 | Scene flow | Conceptual ancestor of DEFOM's Scale Update Module |

---

## Thematic Groupings

### Theme 1: Training Data Without Labels
**MonoDepth → NeRF-Supervised → (Pseudo-stereo from monocular foundation models)**

The arc of stereo training data has moved from "expensive labeled stereo rigs" toward "anyone with a phone can generate unlimited training data":

| Era | Method | Data Source | Cost |
|-----|--------|-------------|------|
| 2012-2017 | KITTI / Middlebury | Real stereo + LiDAR / SL | High (specialized rigs) |
| 2017 | **MonoDepth** | Real stereo pairs (no labels) | Medium |
| 2016-2020 | Scene Flow / TartanAir | Blender / Unreal synthetic | Medium |
| 2023 | **NeRF-Supervised** | Phone + COLMAP + Instant-NGP | **Low** |
| 2024+ | StereoAnything / Pip-Stereo MPT | DepthAnythingV2 + inpainting | **Trivial** (53M+ pairs) |

**Implication for our edge model:** Target-domain adaptation is no longer a labeled-data problem. We can fine-tune to any deployment site with a phone walk-through + 1 day of NeRF/pseudo-data generation.

### Theme 2: Cross-Domain Generalization
**DSM-Net (DN + SGF) → HVT (learned augmentation) → DKT-Stereo (fine-tune protocol)**

These three papers cover the **complete generalization stack**:

1. **DSM-Net** — *architectural* cross-domain (Domain Normalization replaces BN, SGF for non-local aggregation)
2. **HVT** — *training augmentation* cross-domain (learned hierarchical visual transformations)
3. **DKT-Stereo** — *fine-tuning* cross-domain (filter+ensemble GT and pseudo-labels to prevent catastrophic forgetting)

| Method | Where it acts | Cost at inference |
|--------|---------------|-------------------|
| **DSM-Net** | Architecture (DN + SGF layers) | Small overhead |
| **HVT** | Training augmentation only | **Zero** |
| **DKT-Stereo** | Training loop only (3× forward passes) | **Zero** |

**All three are stackable** and address complementary stages of the training pipeline. For our edge model, we should adopt:
- **DN** instead of BN in the encoder
- **HVT** as a training-time augmentation
- **DKT-Stereo** protocol whenever fine-tuning on a target domain

### Theme 3: Output Representation & Uncertainty
**SMD-Nets (bimodal Laplacian) → NDR (continuous query) → SEDNet (KL-calibrated uncertainty)**

Stereo output is no longer a single scalar per pixel. Modern outputs include:
- **Bimodal mixtures** (SMD-Nets) — sharp boundaries via foreground/background mode separation
- **Continuous-coordinate MLP heads** (SMD-Nets, NDR) — query disparity at any sub-pixel location, free super-resolution
- **Calibrated uncertainty** (SEDNet) — distribution-matched aleatoric uncertainty

**All three add ~200 parameters or less.** They're essentially free output-side upgrades.

| Method | Output | Param overhead | Purpose |
|--------|--------|---------------|---------|
| Standard | scalar disparity | 0 | Baseline |
| **SMD-Nets** | bimodal (π, μ₁, μ₂, b₁, b₂) | ~190 | Sharp boundaries |
| **NDR** | continuous(x, y) → disparity | small MLP | Arbitrary resolution |
| **SEDNet** | scalar + uncertainty | ~190 | Calibrated confidence |

**For our edge model, all three are stackable on top of any backbone with no compute cost.**

### Theme 4: Iterative Geometric Optimization
**RAFT (2D flow) → RAFT-Stereo (1D disparity) → RAFT-3D (SE(3) scene flow) → DEFOM-Stereo (depth-prior + scale update)**

RAFT-3D shows that the iterative paradigm scales beyond stereo to **full scene flow**, with the **Dense-SE3 layer** as a learned differentiable optimization step. This pattern — embedding a per-pixel differentiable optimization inside an iterative refinement loop — is **exactly** what DEFOM-Stereo's Scale Update Module does.

**For our edge model:** The Dense-SE3 decomposition (per-pixel independent 6-variable Gauss-Newton) is the template for designing **edge-friendly differentiable optimization layers**. Any geometric constraint that decomposes per-pixel given a soft grouping costs ~O(N) at inference.

---

## Cross-Cutting Insights

### 1. The "Distribution-Match" Pattern
SEDNet's KL divergence loss matches the **distribution shape** of predictions to actual errors, not pixel-level values. This pattern recurs throughout modern stereo:
- **HVT** matches feature distributions via discriminator
- **DSM-Net's DN** matches per-pixel feature norm distributions across domains
- **DKT-Stereo** matches the distribution of student vs. EMA-teacher predictions

**Lesson:** When you can't supervise pixel-by-pixel, supervise the *distribution*.

### 2. Training-Loop Innovations Beat Architecture Innovations for Generalization
HVT, DKT-Stereo, and the NeRF-Supervised loss are all **architecture-agnostic** improvements that beat architecture-specific cross-domain methods. The lesson: **the training recipe is doing more work than the architecture** in cross-domain stereo.

For our edge model, this means: spend effort optimizing the **training recipe** (data mix, augmentation, fine-tuning protocol) before tweaking the architecture for generalization.

### 3. The "Plug-and-Play" Refinement Pattern
NDR established that **any low-resolution disparity** (classical or deep) can be refined by a separate, generalizable module. SMD-Nets's continuous query head is the same pattern. SEDNet's uncertainty subnetwork is again the same pattern.

**For our edge model:** Architect the system as **lightweight matcher + universal post-processing heads**, not as a monolithic end-to-end network. This makes:
- Each component independently improvable
- Inference latency tunable (skip refinement for speed)
- Training cheaper (each module trained separately)

### 4. Synthetic Pretraining is Already a Solved Problem
The combination of **Scene Flow + TartanAir + CREStereo synthetic + pseudo-stereo from monocular** is already the de facto industry standard. There's nothing to invent here for our edge model — just adopt the StereoAnything curriculum.

---

## Relevance to Our Edge Model: Concrete Adoptions

### Architecture additions (essentially free)
1. **Domain Normalization** (from DSM-Net) — replace BN in encoder; no extra params
2. **SMD-Nets bimodal head** — +190 params, sharp boundaries
3. **SEDNet uncertainty head** — +190 params, calibrated confidence
4. **NDR-style continuous refinement** — small MLP, optional HR upsampling

### Training pipeline (zero inference cost)
1. **TartanAir + Scene Flow + CREStereo synth + pseudo-stereo** training mix
2. **HVT learned visual transformations** as augmentation
3. **NeRF-Supervised training loss** for domain adaptation when needed
4. **DKT-Stereo protocol** for any target-domain fine-tuning

### Conceptual takeaways
1. The **iterative refinement + differentiable optimization** pattern (RAFT-3D's Dense-SE3 → DEFOM's Scale Update) is the right inductive bias to preserve at edge
2. **Distribution-level supervision** (KL loss in SEDNet, discriminator in HVT) generalizes better than pixel-level supervision
3. **Lightweight matcher + heavy refinement heads** is cheaper than monolithic HR networks

---

## What These 10 Papers Do NOT Cover (and Where to Look)

| Topic | Where | Tier |
|-------|-------|------|
| Foundation-model integration (mono prior) | DEFOM-Stereo, FoundationStereo, MonSter | Tier 1 |
| Iterative refinement details | RAFT-Stereo, IGEV-Stereo, IGEV++ | Tier 1 |
| Edge architecture techniques | BGNet, LightStereo, BANet, Pip-Stereo, Distill-then-Prune | Tier 1 |
| Cost volume construction | PSMNet, GA-Net, GWCNet, ACVNet, CFNet | Tier 2 |
| Transformer stereo | STTR, GMStereo, CroCo v2 | Tier 2 |
| Dataset details | KITTI, Middlebury, ETH3D, Scene Flow, Spring, Booster | Tier 2 |

---

## Bottom Line

Tier 3 papers are the **infrastructure layer** of modern stereo: how to get training data, how to generalize, how to refine, how to estimate uncertainty. They are **architecturally light** but **practically essential** — a competitive edge stereo model **must** adopt at least 60% of these ideas (training mix + DN + uncertainty head + DKT-Stereo protocol) regardless of which backbone it uses.
