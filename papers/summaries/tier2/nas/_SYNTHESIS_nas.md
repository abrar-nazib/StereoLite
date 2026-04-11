# Synthesis: Neural Architecture Search for Stereo (Tier 2)

> 2 papers applying NAS to stereo matching architectures (2020-2022).

---

## Papers Covered

| # | Paper | Year | Strategy | Key Result |
|---|-------|------|----------|-----------|
| 1 | **LEAStereo** | 2020 | Hierarchical NAS (find one arch) | 1.81M params, rank 1 KITTI |
| 2 | **EASNet** | 2022 | Once-for-all (find a family) | 4.5× faster than LEAStereo |

---

## The Two NAS Philosophies

### LEAStereo: "Find the Best Architecture"
- **Goal:** search for a single optimal architecture using domain priors
- **Method:** DARTS-style bi-level optimization + task-specific op candidates
- **Search space:** cell-level DAGs + network-level resolution trellis
- **Cost:** 10 GPU days
- **Result:** 1.81M params, 0.3s runtime, rank 1 on KITTI 2015 at submission

### EASNet: "Find a Family of Architectures"
- **Goal:** train one supernet that produces many sub-networks for heterogeneous deployment
- **Method:** progressive shrinking over 4 elastic dimensions (K, D, W, S)
- **Search space:** ~$10^{13}$ sub-networks
- **Cost:** 48 GPU days (higher than LEAStereo)
- **Result:** fastest sub-network is 4.5× faster than LEAStereo with lower EPE

---

## Common Findings

### 1. Domain priors dramatically reduce search cost
**Both papers exploit the stereo pipeline structure** (feature extraction → cost volume → aggregation → regression) to constrain the search space. Naive NAS over generic operations is impractical — stereo-specific inductive biases are essential.

### 2. Residual cells outperform direct cells
LEAStereo's ablation shows residual cells (which include the cell input in the output) consistently improve accuracy.

### 3. Joint search is better than separate search
Both papers search the full pipeline (feature + matching net) jointly rather than separately — joint optimization finds better architectures.

### 4. NAS can find compact architectures matching hand-designed SOTA
LEAStereo with **1.81M parameters** matches GANet-deep (43.34M parameters) on KITTI — **20× parameter reduction** with comparable accuracy.

---

## Why NAS Stereo Stopped After 2022

**RAFT-Stereo (2021) and the iterative paradigm arrived just as NAS-stereo was maturing.** By 2022:
- Iterative methods outperformed NAS-designed 3D cost volume models on accuracy
- Hand-designed iterative architectures (CREStereo, IGEV-Stereo) were simpler and more effective than NAS outputs
- Foundation model approaches (DEFOM-Stereo, FoundationStereo) shifted the emphasis from architecture search to representation learning

**EASNet is the last major NAS-stereo paper** before iterative methods rendered pure volumetric approaches secondary for accuracy. It remains SOTA for deployable-across-devices stereo NAS.

---

## Relevance to Our Edge Model

### Directly Applicable

**EASNet's elastic scale dimension** is the single most valuable idea for our edge model:
- A single model that can run at S=2 on Jetson Orin Nano and S=4 on server GPU
- **No retraining** for different deployment targets
- Exactly what heterogeneous edge deployment requires

**Progressive shrinking training strategy** is immediately applicable:
1. Train full network first (K=7, D=4, W=8, S=4)
2. Sequential fine-tuning stages for elastic K, D, W, S
3. Final supernet supports many sub-networks

### Combine With Tier 1 Efficiency Techniques

EASNet's AANet cost volume + elastic training can be combined with:
- **Separable-Stereo's 3D V2 blocks** (Tier 1) — apply to ISA aggregation
- **BGNet's bilateral grid upsampling** (Tier 1) — replace convex upsampling
- **Pip-Stereo's MPT distillation** (Tier 1) — add monocular priors without ViT at inference
- **LightStereo's channel boost** (Tier 1) — use V2 blocks in 2D aggregation

### Lessons Learned

**Both papers validate that:**
1. Compact architectures (sub-5M params) can match much larger hand-designed networks
2. Joint pipeline search outperforms component-wise search
3. Task-specific ops and residual connections matter

**But they also confirm:**
1. NAS alone is NOT enough — iterative methods + foundation priors dominate in 2025
2. The best edge models use **distillation + hand-designed efficient components**, not NAS search

### Our Strategy

**Do NOT run NAS for our edge model.** Instead:
1. Use LEAStereo's compact architecture as a **baseline comparison point**
2. Adopt **EASNet's elastic scale dimension** for deployment flexibility
3. Combine with **iterative paradigm** (RAFT/IGEV) + **Pip-Stereo's MPT** + **Separable-Stereo's V2 blocks**
4. The architecture is hand-designed using insights from Tier 1 papers; NAS is not needed
