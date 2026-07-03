# Thesis Plan — "Edge-Deployable Deep Stereo Matching" (yolo_ctx_gev4)

Working plan for the RUET MTE B.Sc. thesis book. Authors: Nazib Abrar
(2008026), Md. Raihanul Haque Rahi (2008011). Supervisor: Md Zunaid Hossen,
Lecturer, Dept. of Mechatronics Engineering. Format authority:
`.claude/skills/ruet-thesis-expert/SKILL.md`. Branch: `thesis`.

Central claim to defend: **runs on edge hardware and provides usable results.**

---

## 1. What we already have (asset inventory, verified on disk)

### Evidence / data
| Asset | Location | Serves |
|---|---|---|
| Full 89k-step Scene Flow Driving training run (4200 train / 200 val): final val EPE 0.963 px, D1 7.88%, 2.9623 M params, RTX 4070 | `model/checkpoints/yolo_ctx_gev4_full_retry_es/{meta.json,train.csv}` | Ch4 core result |
| 20-pair overfit reference: EPE 0.261, 24.6 ms / 40.7 FPS RTX 4070 | `model/benchmarks/yolo_ctx_gev4_yolo26s/` | Ch4 capacity check |
| ~10-variant architecture ablation story (gate/gev4/guided/sharp/hrrefine/sru/init4/geomctx) | `model/benchmarks/EXPERIMENTS.md` | Ch4 design justification (PO4) |
| PDF-verified baseline numbers (HITNet, BGNet, CoEx, RAFT, IGEV, LightStereo, ...) | `papers/verified_performance.md` | Ch2/Ch4 tables |
| 72 verified BibTeX entries | `review_paper/references.bib` | all chapters |

### Figures (ready)
- Ch1: stereo geometry (`presentation/figs/intro_stereo_geometry.png`), application/hardware photos (`presentation/photos/`: jetson, AR0144, drone/robot/AR, lidar/realsense), example input/depth pair.
- Ch2: timeline, taxonomy (TikZ source!), two Pareto plots, family-contribution (`review_paper/figures/`).
- Ch4: gev4 overfit curve + ~150 labeled collages; 5 full-run milestone panels (`step_0{2,4,6}0000/85000/89000_resize.png`); real-camera panels (old arch, flagged).
- Generators for everything above under `presentation/figs/build_*.py`, `review_paper/figures/_data/*.py`, `model/scripts/build_viz_filmstrip.py`.

### Equations
- **gev4's own formal spec `model/designs/StereoLite_yolo_ctx_gev4/FINAL_MODEL_ARCHITECTURE.md`** (encoder, context stream, tile init, GRU refinement, plane propagation, GEV + fail-soft gate, convex upsample, loss, metric definitions) — primary Ch3 source.
- 22 canonical background equations (`.claude/skills/stereo-vision-expert/equations.md`) + 11 review-paper equations — Ch2 source.

### STALE — must not be presented as gev4
`presentation/figs/stage1..4_*.png` (draw the OLD ghost chassis), `stereolite_architecture_doc.tex` (old 0.874 M model), `training_v8_top3.gif`, `data/stereolite_v8_kaggle/` ONNX exports, indoor-finetune montages, the "StereoLite (Ours)" row in verified_performance.md. Regenerate or clearly relabel.

---

## 2. What is missing (gap list)

### A. MUST-HAVE for the edge claim (blockers first)
| # | Item | Effort | Depends on |
|---|---|---|---|
| A1 | **Trained gev4 checkpoint into repo** (exists only on Rahi's machine; repo has meta+csv, no .pth) | 0.5 h | **Rahi — ask today** |
| A2 | **Edge-device latency** (Jetson/RPi, fp16 + INT8 if possible; tegrastats power/mem). Fallback: soften claim to "designed for edge, real-time on laptop GPU" + Ch5 limitation | 1-2 d | A1 + hardware access |
| A3 | **MB14 zero-shot on gev4** (adapt `eval_middlebury2014.py` line-61 import; ~$0.05 Modal T4). Decides the whole Ch4/Ch5 narrative — prior chassis collapsed 40.1% D1 here | 2-4 h | A1 |
| A4 | **Real-camera qualitative panels with gev4** (re-wire `live_stereolite.py`; CCB camera / `data/user_cam_1/`) | 3-6 h | A1 + camera |
| A5 | Matched-protocol baseline table (gev4 vs IGEV vs LiteAnyStereo on identical MB14 protocol; reference JSONs already exist) | 2-3 h | A3 |
| A6 | Inference memory measurement (only 7.6 GB TRAINING peak recorded — unusable for edge claims) | 1 h | A1 |

### B. Chapter completeness (see SKILL §3-§5)
- B1 `thesis/book/` tree from template + format decisions (0.5 d, no deps).
- B2-B6 the five chapters (see writing plan below).
- B7 Complex Engineering Problem justification (K3/K4/K5/K8), folded into Ch1/Ch3.
- B8 Appendix A tech specs (meta.json `cfg` block is ready-made content).
- B9 Appendix B: expense table (Modal spend, camera, boards) + Gantt — nothing exists; deck's Time Plan slide is the seed.
- B10 Appendix C: **similarity report + AI report via supervisor (start now — external clock)**, ethics checklist, CoI, CRediT (2 students), communication statement.
- B11 Appendix D: PO/KPA tracker (after chapters drafted).
- B12 Appendix E: publications (survey draft or "none").
- B13 Frontmatter (title/declaration/certificate/acdarknowledgments/abstract ≤300 w — abstract written LAST).

### C. Nice-to-have (pick 2-3)
ETH3D zero-shot (data already on Modal volume) · FlyingThings3D standard test EPE ·
INT8/ONNX study · power measurement · RTX 3050 gev4 latency ·
3D point-cloud figure (`disparity_to_pointcloud.py`, lands well with MTE examiners) ·
ablation filmstrips · old-chassis vs gev4 head-to-head.

---

## 3. Architecture efficiency findings (optimizable WITHOUT accuracy loss)

Compute anatomy at 384×640: total ≈ 47 GMAC; the 1/4 GEV block alone ≈ 35-40%;
128 Python-loop bodies, 48 grid_sample calls, 16 meshgrid builds per forward.

| # | Finding | Saving | Accuracy risk |
|---|---|---|---|
| F1 | Cost-volume loops (64-iter GEV + 24-iter TileInit) alloc+zero+copy a full feature map per bin; replace with padded zero-copy views | ~5-15% total latency (more on Jetson; ~1 GB DRAM traffic removed) | **none, bitwise** |
| F2 | 48 grid_samples: 8 are exact duplicates of lookup offset-0; 40 lookup calls batchable into 8 via grid stacking; cache meshgrids | ~3-8% latency + ONNX graph 48→8 GridSample nodes | **none, identical math** |
| F4 | Static channels (fL + ctx ≈ 44% of GRU input at 1/4) re-convolved every iteration; hoist once per scale (RAFT's precomputed-context trick) | ~8-10% total MACs + biggest per-iter allocs gone | none (mathematically exact) |
| F5 | Fuse conv_z+conv_r and the 4 output heads (same inputs) | 1-3% latency | **none, bitwise** |
| F7 | Dead stage lists at inference, discarded zeros feat, dead `_norm_feat`, rebuilt upsample constants | few MB + launches | none |
| F3 | GEV narrows from full 64 bins to ~±16 around tile.d (validated cascade_cv_4 pattern) | **~25-30% of total MACs, ~100 MB activations** | needs matched A/B + MB14 check |
| F6 | **Normalization contract landmine**: overfit harness feeds [0,1], full trainer feeds [0,255]; gev4 ctx stream double-divides (stem runs at GroupNorm's eps floor); checkpoints not portable across pipelines; INT8 hazard | correctness, not speed | settle ONE contract before any retrain/export |
| F8 | Export blockers: F.unfold in ConvexUpsample (replace with 9 shifted slices — identical), GroupNorm on NPU toolchains, Conv3d on NPUs | export-time | flag now |

**Thesis-shaping insight:** F1+F2+F4+F5+F7 are retrain-free and together worth
~15-25% on-device latency. Implemented + measured before/after ON THE EDGE
DEVICE, they become a Ch3/Ch4 section ("optimization of the network for edge
inference") that directly evidences PO3/PO4/PO5 — the optimization work IS
thesis content, not a detour. F3 is a proper ablation (run under the
ablation-study-expert protocol) and the biggest lever if it holds.

---

## 4. Writing plan (phases; ~3-4 weeks calendar)

### Phase 0 — unblock (today)
1. Ask **Rahi** for the full gev4 checkpoint (A1).
2. Ask **supervisor** about similarity + AI report process (B10) and Jetson/edge
   hardware availability (A2).
3. Create `thesis/book/` from the template with the resolved format (B1).

### Phase 1 — evidence sprint (week 1; parallel with Phase 2 writing)
4. A3 MB14 zero-shot (decides narrative) → A5 baseline table → A6 memory.
5. A4 real-camera panels with gev4.
6. Efficiency pass F1+F2+F5+F7 (bitwise-safe) → re-measure latency (T4 + local
   + edge device when available) → before/after table.
7. If hardware: A2 edge latency (+ C4 power). Else: lock the fallback wording NOW,
   before Ch1 is drafted.
8. Optional stretch: F3 GEV-narrowing A/B (ablation protocol, MB14-checked), C6
   point-cloud figure.

### Phase 2 — chapters (weeks 1-3; no dependency on Phase 1 except Ch4 numbers)
Order: **Ch3 → Ch2 → Ch4 → Ch1 → Ch5 → frontmatter/abstract last.**
- **Ch3 Methodology** (2-4 d): problem statement + assumptions + research
  questions (formalize for the first time); gev4 architecture from
  FINAL_MODEL_ARCHITECTURE.md with REGENERATED per-stage diagrams (current
  stage*.png are the old chassis); training setup from meta.json; overfit
  ablation methodology; the efficiency optimizations (F-items). POs 2/3/5.
- **Ch2 Literature Review** (2-3 d): distill review_paper sections + tier1/2
  summaries; classical→deep→iterative→efficient arc; theory/math (epipolar
  geometry, cost volumes, soft-argmin, GRU refinement — equations.md). PO2.
  Citations: numeric [n] in order of appearance; author-year list entries.
- **Ch4 Results & Discussions** (2-3 d after A3-A6): full-training curves +
  val table; ablation story; baseline comparison; efficiency before/after;
  real-camera qualitative; REQUIRED societal/environmental/sustainability
  subsection (PO6/PO7; edge-vs-cloud energy, low-cost sensing vs LiDAR for
  local robotics/agriculture); implications. PO4/PO6/PO7.
- **Ch1 Introduction** (1-2 d): problem + significance, background, objectives
  + scope (written to match what Ch4 actually shows), rationale, outline, CEP
  justification (K3/K4/K5/K8). PO1/PO2.
- **Ch5 Conclusions** (0.5-1 d): objective attainment, HONEST limitations
  (whatever A2/A3 left unmeasured), future work (3-stage KD, INT8, temporal).
  PO12.

### Phase 3 — appendices + frontmatter (week 3)
Appendix A (specs) → B (expenses + Gantt) → C (ethics pack — external clock)
→ D (PO/KPA tracker vs finished chapters) → E → frontmatter → abstract.

### Phase 4 — assembly + QA (week 4)
Compile loop per SKILL §8; checklist §9 per chapter; page-level orphan check;
similarity/AI reports attached; supervisor review pass.

### Risk register
1. **Checkpoint unavailable** → only the overfit checkpoint exists; thesis
   quality drops a grade band. Mitigate: ask today; worst case re-train on
   Modal A100 (~$10-15, 1 day — legitimate A100 use).
2. **gev4 collapses on MB14** (prior chassis did) → narrative shifts to
   "in-domain + real-camera usable; cross-domain identified and analyzed as
   limitation with the KD path forward". Still a defensible thesis; know early.
3. **No edge hardware** → fallback wording locked before Ch1; C3 INT8/ONNX
   study partially substitutes.
4. **AI report friction** → disclosure is mandatory; start the conversation
   with the supervisor now, not at submission.
