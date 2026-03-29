# Stereo Vision Research: From Survey to Edge-Optimized Model

A comprehensive stereo matching research project spanning paper analysis, survey writing, and novel model development.

## Project Goals

| Phase | Goal | Status |
|-------|------|--------|
| 1 | Systematically analyze ~190 papers from [Awesome-Deep-Stereo-Matching](https://github.com/fabiotosi92/Awesome-Deep-Stereo-Matching) | In Progress |
| 2 | Write a comprehensive review paper covering stereo vision from fundamentals to SOTA | Planned |
| 3 | Build an edge-optimized stereo model inspired by DEFOM-Stereo (CVPR 2025) | Planned |

## Paper Analysis Summary

**Source:** [fabiotosi92/Awesome-Deep-Stereo-Matching](https://github.com/fabiotosi92/Awesome-Deep-Stereo-Matching)

| Metric | Count |
|--------|-------|
| Total entries in awesome list | ~240+ |
| Unique papers (after dedup) | ~190 |
| Duplicate entries across categories | ~50 groups |
| Papers at priority 9-10 (must-read) | 25 |
| Papers at priority 7-8 (high priority) | 35 |
| Papers at priority 5-6 (moderate) | 45 |
| Papers at priority 3-4 (lower) | 45 |
| Papers at priority 1-2 (skim) | 40+ |

See [`papers/paper_registry.md`](papers/paper_registry.md) for the full ranked list with justifications.

## Key Papers by Category

### Foundation-Model Era (our focus, CVPR 2025-2026)
| Paper | Priority | Key Insight |
|-------|----------|-------------|
| **DEFOM-Stereo** | 10 | Depth foundation model (Depth Anything V2) + stereo via scale correction |
| **FoundationStereo** | 10 | Zero-shot stereo via large-scale training + foundation features |
| **Fast-FoundationStereo** | 9 | Real-time version of FoundationStereo |
| **MonSter** | 9 | Monocular depth married to stereo matching |
| **Stereo Anywhere** | 9 | Robust zero-shot where either stereo or mono fail |
| **D-FUSE** | 8 | Systematic study of monocular prior fusion |
| **PromptStereo** | 8 | Zero-shot via structure and motion prompts |

### Iterative Architecture Lineage (our backbone)
| Paper | Priority | Key Innovation |
|-------|----------|----------------|
| **RAFT-Stereo** | 10 | Recurrent all-pairs field transforms for stereo |
| **CREStereo** | 9 | Cascaded recurrent + adaptive correlation |
| **IGEV-Stereo** | 9 | Geometry encoding volume + iterative updates |
| **IGEV++** | 9 | Multi-range geometry encoding |
| **Selective-Stereo** | 8 | Frequency-adaptive information selection |

### Edge/Efficient Architectures (our deployment target)
| Paper | Priority | Efficiency Technique |
|-------|----------|---------------------|
| **BGNet** | 9 | Bilateral grid for cost volume |
| **Separable-Stereo** | 9 | Depthwise separable 3D convolutions |
| **LightStereo** | 8 | Channel boost for 2D cost aggregation |
| **BANet** | 8 | Bilateral aggregation for mobile |
| **LiteAnyStereo** | 8 | Efficient zero-shot stereo |
| **GGEV** | 8 | Generalized geometry encoding for real-time |
| **Pip-Stereo** | 8 | Progressive iteration pruning |
| **Distill-then-Prune** | 7 | Distillation + pruning for edge |

## Project Structure

```
stero_research_claude/
|
+-- CLAUDE.md                    # AI assistant instructions
+-- README.md                    # This file
+-- awesome_list_raw.md          # Raw content from Awesome-Deep-Stereo-Matching
|
+-- papers/
|   +-- paper_registry.md        # Complete ranked paper list with priorities
|   +-- raw/                     # Downloaded PDFs organized by category
|   |   +-- surveys/
|   |   +-- matching_cost/
|   |   +-- optimization/
|   |   +-- refinement/
|   |   +-- end_to_end_2d/
|   |   +-- end_to_end_3d/
|   |   +-- iterative/
|   |   +-- transformer/
|   |   +-- mrf/
|   |   +-- efficient/
|   |   +-- nas/
|   |   +-- self_supervised/
|   |   +-- domain_shift/
|   |   +-- multi_task/
|   |   +-- confidence/
|   |   +-- foundation_model/
|   |   +-- beyond_rgb/
|   +-- metadata/                # Extracted metadata, BibTeX, summaries
|
+-- analysis/
|   +-- taxonomy/                # Paper categorization and evolution trees
|   +-- trends/                  # Temporal analysis, technique evolution
|   +-- comparisons/             # Benchmark results, method comparisons
|
+-- review_paper/
|   +-- sections/                # LaTeX sections
|   +-- figures/                 # Diagrams, architecture visualizations
|   +-- main.tex                 # Master LaTeX document
|
+-- model/
|   +-- core/                    # Core architecture (encoder, cost volume, GRU, refinement)
|   +-- modules/                 # Novel layer implementations
|   +-- configs/                 # Training and inference configs
|   +-- scripts/                 # train.py, evaluate.py, export_onnx.py, benchmark.py
|   +-- benchmarks/              # Latency/accuracy profiling
|
+-- data/                        # Dataset management scripts and symlinks
+-- tools/                       # Utility scripts (scraping, PDF parsing, plotting)
```

## Field Evolution Timeline

```
2002  Scharstein & Szeliski taxonomy
  |
2007  SGM (Hirschmuller) ---- classical baseline
  |
2015  MC-CNN (Zbontar) ------ first learned matching cost
  |
2016  DispNet-C ------------- first end-to-end CNN
  |
2017  GC-Net ---------------- 3D cost volume paradigm
  |
2018  PSMNet, GA-Net -------- mature 3D cost volume era
  |
2019  GWCNet, DeepPruner ---- group correlation, efficient pruning
  |
2020  AANet, LEAStereo ------ 2D efficient, NAS-designed
  |
2021  RAFT-Stereo ----------- iterative paradigm shift (!!!)
  |   BGNet, HITNet --------- real-time approaches
  |
2022  CREStereo, ACVNet ----- practical iterative, attention cost vol
  |
2023  IGEV-Stereo ----------- geometry encoding volume
  |   CroCo v2, GMStereo ---- transformer/unified approaches
  |
2024  Selective-Stereo ------- frequency-aware iterative
  |   NMRF, LoS ------------- alternative paradigms
  |   IGEV++ ---------------- multi-range encoding
  |
2025  DEFOM-Stereo ---------- foundation model + stereo (!!!)
  |   FoundationStereo ------ zero-shot stereo
  |   MonSter, Stereo Any --- mono+stereo fusion era
  |   BANet, LightStereo ---- mobile/edge efficient
  |
2026  PromptStereo ---------- prompt-based zero-shot
  |   Fast-FoundationStereo - real-time zero-shot
  |   Pip-Stereo, GGEV ------ efficient iterative
```

## Edge Model Design Direction

Our model aims to combine insights from the foundation-model era with edge-device constraints:

```
Input Stereo Pair
       |
  [Efficient CNN Encoder]  ---- MobileNetV4 / EfficientViT / FastViT
       |                         (replace heavy ViT backbone)
       |
  [Distilled Mono Prior]   ---- Lightweight Depth Anything distillation
       |                         (fewer channels, smaller resolution)
       |
  [Fused Cost Volume]      ---- Bilateral grid (BGNet) or
       |                         sparse adaptive sampling
       |
  [Lite GRU Updates]       ---- Fewer iterations (3-5 vs 12-32)
       |                         with better initialization from mono prior
       |                         + separable 3D convs (Separable-Stereo)
       |
  [Scale Correction]       ---- Lightweight scale update module
       |                         (from DEFOM-Stereo concept)
       |
  Disparity Output
       |
  [ONNX/TensorRT Export]   ---- Jetson Orin Nano, mobile NPU
```

**Target specs:**
- Latency: <50ms on Jetson Orin Nano (540x960 resolution)
- Accuracy: Within 10-15% of DEFOM-Stereo on KITTI/ETH3D
- Model size: <30M parameters (vs ~150M+ for full DEFOM-Stereo)

## Next Steps

1. **Phase 1a:** Download and read the 25 must-read papers (Tier 1)
2. **Phase 1b:** Extract key architecture diagrams, benchmark numbers, and innovations
3. **Phase 2a:** Build taxonomy tree and trend analysis
4. **Phase 2b:** Begin review paper outline with section assignments
5. **Phase 3a:** Set up model codebase following RAFT-Stereo conventions
6. **Phase 3b:** Implement baseline edge model and benchmark

## Environment Setup

```bash
conda create -n stereo python=3.10
conda activate stereo
pip install -r requirements.txt  # (to be created)
```

## References

- Awesome List: https://github.com/fabiotosi92/Awesome-Deep-Stereo-Matching
- Tosi et al. Survey: https://link.springer.com/content/pdf/10.1007/s11263-024-02331-0.pdf
- CVPR 2024 Tutorial: https://sites.google.com/view/stereo-twenties
