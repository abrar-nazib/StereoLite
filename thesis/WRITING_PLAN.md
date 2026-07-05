# Thesis Writing Plan (detailed, executable)

Step-by-step production plan for the RUET MTE B.Sc. thesis book
**"AI-Enhanced Stereo Matching for High-Accuracy Depth Mapping and 3D
Reconstruction"** (title FIXED by the user, 2026-07-05; not open for
revision).
Written 2026-07-05 against the `ruet-thesis-expert` skill (format authority),
the `research-linguistics-expert` skill (prose authority), and a full asset
inventory of the repo. This document is the instruction set for whichever
AI/human session writes each piece: every section names its sources, figures,
tables, equations, prose patterns, and PO evidence. Strategic context and
evidence status live in `THESIS_PLAN.md`; this file is the HOW.

---

## 0. Binding authorities and global rules (read before writing anything)

Every writing session MUST:

1. **Invoke `ruet-thesis-expert`** (Skill tool) before touching any thesis
   file. It owns format, structure, PO/KPA mapping, frontmatter templates,
   and the resolved format conflicts. Non-negotiables repeated here:
   - Department is **Mechatronics** everywhere (template placeholder says
     Mechanical; always replace).
   - 2 authors: Nazib Abrar (2008026), Md. Raihanul Haque Rahi (2008011).
     Md Zunaid Hossen (Lecturer, MTE) is SUPERVISOR, never co-author.
   - Geometry: left margin 3 cm (binding), right/top/bottom 2.54 cm.
   - Times New Roman 12 pt body, single spacing, justified.
   - Headings: chapter 16 pt bold ("Chapter N" + title, two lines, centered),
     section 13 pt bold, subsection 12 pt normal. **No sub-subsections.**
   - Chapter-number style: `Chapter 1` (no zero padding), identical in ToC
     and body.
   - Page numbers: none on title page + Certificate; roman `i` from
     Declaration; arabic `1` from Chapter 1; bottom centered, plain.
   - Figure caption BELOW, `Figure C.F: title`. Table caption ABOVE,
     `Table C.T: title`. Equations auto-numbered per chapter `(C.E)`;
     reference with `\eqref` / `\ref`, never hard-coded numbers.
   - Citations: **pure IEEE numeric, DECIDED (user, 2026-07-05).** Numeric
     `[n]` in order of first appearance, IEEE-style reference entries
     (initials first, quoted titles), via BibTeX with the template's
     `Reference.bib` + an IEEE bibliography style. Matches the accepted
     June-2025 precedent. Apply 100% consistently; no hybrid entries.
2. **Invoke `research-linguistics-expert`** before composing prose. Register:
   modern efficient/edge bucket by default; flagship register for Ch 2
   background; early register only for field history. Key patterns are cited
   per-section below (7-move abstract, 4-paragraph intro funnel, connective
   inventory, where-clause equation convention, self-contained captions).
3. **Numbers discipline:** every benchmark/param/latency number cites a repo
   source. The sources of truth, in precedence order:
   - Our results: `model/benchmarks/20260704_fullsf_gev4onp_nc/meta.json`
     (`final_metrics_all`), `mb14_zero_shot.json`, `train.csv`,
     `model/benchmarks/EXPERIMENTS.md`, `THESIS_PLAN.md` §2 (latency,
     memory, Jetson projections with the asterisk convention).
   - Other methods: `papers/verified_performance.md` FIRST, then
     `review_paper/figures/_data/method_data.py` (41 methods, each with a
     `source` breadcrumb), then the tier1/tier2 summary of the paper.
   - Architecture facts: `model/designs/StereoLite_yolo_ctx_gev4/
     FINAL_MODEL_ARCHITECTURE.md` (the authoritative 20-section spec) and
     the actual code in `model/designs/StereoLite_yolo_ctx_gev4_opt/`.
   - NEVER assert a number from memory. If a needed number has no source,
     mark it `[TODO: measure]` and log it in `thesis/book/NOTES.md`.
4. **No `--` or `---` anywhere in prose** (body, captions, headings). Ranges
   use "to"; asides use commas/colons; CLI flags in verbatim blocks exempt.
5. **Honesty:** undergrad thesis reports what was built and measured.
   Missing measurements (Jetson on-device latency) appear as calculated
   values with an asterisk and a methodology sentence, per `THESIS_PLAN.md`
   §2A, and again as a limitation in Ch 5. "Recommended wording" for claims
   is pre-vetted in `FINAL_MODEL_ARCHITECTURE.md` §18: use it.
6. **PO evidence:** while drafting each section, append one line per
   demonstrated PO to `thesis/book/po_tracker_notes.md`
   (`section | PO | one-sentence evidence`). Appendix D is assembled from
   this file at the end. Mapping is fixed: Ch1 PO1+PO2, Ch2 PO2,
   Ch3 PO2+PO3+PO5, Ch4 PO4+PO6+PO7, Ch5 PO12, App A PO5, App B PO11,
   App C PO8+PO9+PO10.
7. **Figure/diagram work** uses the `diagram-drawer` skill and its helper
   module `.claude/skills/diagram-drawer/helpers/diag_helpers.py` for
   architecture figures, and the thesis figure style contract in §9 of this
   plan for every generated figure.

---

## 1. Step B1: scaffold `thesis/book/` (first concrete action, ~0.5 day)

The compilable skeleton is `thesis/Thesis Writing Instructions/Resources/MTE
B.Sc. thesis paper template/` (verified contents: `Thesis.tex` master,
`Reference.bib`, `frontmatter/{declaration,acknowledgments,abstract}.tex`,
`mainmatter/ch1.tex`, `images/`; ch2..ch6 includes exist but are commented
out). Do NOT edit the template in place. Steps:

1. `cp -r` the template to `thesis/book/`, delete build artifacts
   (`*.aux .log .toc .lof .lot .bbl .blg .out .nlo .synctex.gz Thesis.pdf`).
2. Edit `Thesis.tex`:
   - geometry: left 3 cm, others 2.54 cm (override whatever the template
     sets); 12 pt report class; single spacing (remove `\onehalfspacing`
     if present).
   - Add missing frontmatter files: `frontmatter/titlepage.tex` (outer,
     unnumbered), `frontmatter/titlepage_inner.tex` (optional, per the
     2025 precedent's two-title-page pattern), `frontmatter/certificate.tex`
     (unnumbered), then Declaration (roman i starts here), Acknowledgments,
     Abstract, ToC, LoF, LoT, List of Symbols & Abbreviations. Order:
     Certificate then Declaration (precedent order).
   - Create `mainmatter/ch2.tex` .. `mainmatter/ch5.tex`, uncomment five
     `\include` lines (the template stubs six chapters; we use FIVE per the
     Key Instructions: Introduction, Literature Review, Methodology,
     Results and Discussions, Conclusions and Future Work).
   - Appendices: `\appendix` + `appendices/app_a.tex` .. `app_e.tex`.
   - Page-numbering switches: `\pagenumbering{gobble}` for title +
     certificate, `\pagenumbering{roman}` at Declaration,
     `\pagenumbering{arabic}` at Ch 1.
3. Create `thesis/book/figures/` (all thesis figures land here; never link
   into `presentation/` or `review_paper/` directly, copy the built PDF/PNG
   in) and `thesis/book/figures/_src/` for thesis-specific generator
   scripts (§9).
4. Create `thesis/book/NOTES.md` (decisions log: citation style, title,
   any supervisor rulings) and `thesis/book/po_tracker_notes.md`.
5. Populate title page and certificate with the verbatim blocks from
   `ruet-thesis-expert` §2 (motto line, RUET 17 pt, logo, dept 16 pt,
   "A thesis report on" 14 pt + bold title, partial-fulfilment sentence,
   Supervised by / Submitted by columns, Month Year). Certificate signature
   blocks: Supervisor / both Students / Countersigned Head / External
   Examiner with name blank.
6. Compile gate: `pdflatex` x2 (+ `bibtex` once a first citation exists)
   from `thesis/book/`; verify roman-to-arabic transition, ToC populated,
   zero `??`. Commit the scaffold.

---

## 2. The narrative arc (what "masterclass" means here)

A reader with engineering background but zero stereo knowledge must be able
to read cover-to-cover and come out understanding: (a) what stereo vision is
and the geometry that makes it depth-capable; (b) how the field evolved from
hand-crafted matching through deep learning to foundation models; (c) why
edge deployment is a distinct, hard problem and the technique families the
literature uses for it; (d) exactly how our model works block by block;
(e) what we measured, honestly. The chapters implement this arc:

- Ch 1 hooks with the application constraint (depth on 5 to 25 W devices),
  states the problem, objectives, scope.
- Ch 2 IS the masterclass: geometry and classical pipeline, then the six
  visible eras, then a deep dive on HOW efficient models are built
  (compression technique families), each era anchored by 1 to 3
  representative architectures WITH redrawn diagrams and their canonical
  equations.
- Ch 3 narrates our model the way Ch 2 narrated others: one full-pipeline
  figure, then one subsection per block, each with a diagram, its equations,
  and its working procedure; then the training protocol and the efficiency
  optimization work.
- Ch 4 presents raw results, interprets them, positions them against the
  literature on matched protocols, and covers the societal/environmental
  angle the OBE mapping requires.
- Ch 5 closes the loop against the Ch 1 objectives, states limitations
  plainly, and lays out future work (3-stage KD, INT8/TensorRT, on-device
  measurement, temporal extension).

Page budget (target ~90 pages body + ~20 front/appendix):
Ch1 7 · Ch2 24 · Ch3 22 · Ch4 18 · Ch5 5 · appendices 12 · frontmatter 8.

---

## 3. Chapter 1: Introduction (~7 pages)

**Prose pattern:** 4-paragraph funnel per `research-linguistics-expert`
patterns.md §2, stretched to chapter length. Assert, don't hedge.

### 3.1 Section breakdown

| § | Title | Content | Research sources |
|---|---|---|---|
| 1.1 | Background and Motivation | What depth perception buys a machine; passive stereo vs LiDAR vs structured light (cost, power, form factor); the depth-from-disparity relation stated early (Eq 1.1); applications: mobile robots, drones, AR, embedded rigs | `review_paper/sections/01_introduction.tex` + `02_background.tex`; tier1 `surveys/Tosi_Survey_IJCV2025.md`; deck slide 3-5 content (`presentation/build_v5_deck.py` rebuild_introduction*, rebuild_problem_statement) |
| 1.2 | Problem Statement | Accurate deep stereo exists (foundation models, 340 M params) but is undeployable on edge; edge devices = 1 to 6 TOPS, 4 GB, 5 to 25 W; the accuracy-latency-memory triangle; the concessive pivot sentence (`However,` naming the gap) | `THESIS_PLAN.md` state-of-play; `papers/verified_performance.md` (FoundationStereo size/latency); review_paper §03 |
| 1.3 | Research Objectives and Scope | **The two thesis objectives are FIXED by the user (2026-07-05), verbatim:** (1) "To design a computationally efficient stereo matching pipeline that leverages AI-based disparity refinement to enhance depth estimation on resource-limited platforms." (2) "To design an architecture that can withstand camera imperfections in terms of rectification." Sub-goals under them (< 3 M params, synthetic training, zero-shot validation, edge latency, real-camera qualification) may be listed as scope items but the objectives themselves are these two. Scope: passive binocular stereo, supervised synthetic training; excluded: active sensing, self-supervision, deployment SDK productization | Write so Ch 5 answers both one-to-one. **Objective 2 evidence gap: see §11a rectification-robustness experiment (REQUIRED before Ch 4 is final)** |
| 1.4 | Rationale and Significance | Why a NEW small architecture rather than shrinking a big one; the Complex Engineering Problem justification woven in here: state the K-profile K3 (engineering fundamentals), K4 (specialist stereo knowledge), K5 (design of the network), K8 (research-literature grounding) explicitly, one sentence each | `ruet-thesis-expert` §5 CEP; `Complex Engineering Problem-demostration.pdf` in Resources if wording is needed |
| 1.5 | Thesis Organization | One tight paragraph per chapter | Write LAST for this chapter |

### 3.2 Figures

| ID | Figure | Source/tool | Status |
|---|---|---|---|
| Fig 1.1 | Two-camera stereo geometry (P, baseline B, x_L/x_R, Z = fB/d panel) | `presentation/figs/build_intro_figure.py::stereo_geometry` | ADAPT: regenerate with thesis style contract (§9): white background, serif, PDF output |
| Fig 1.2 | Real stereo pair and its disparity map (hook figure: "this is what the thesis produces") | Regenerate from the trained checkpoint on a camera pair (`model/benchmarks/gev4_camera_smoke/` panels, or re-run the smoke script with `best.pth`) | NEW (5 min once camera panels regenerated) |
| Fig 1.3 | The edge-deployment gap: params (log) vs accuracy scatter with the target zone and our model's star | `presentation/figs/build_slide_figs.py::build_research_gap`, updated with gev4 numbers (2.963 M, SF-TEST EPE 0.78) | ADAPT (data-driven; update star coords + restyle) |

Photos of application domains (drone/robot/AR from `presentation/photos/`)
are OPTIONAL; if used, verify provenance/attribution first; prefer omitting
over unattributed stock imagery.

### 3.3 Equations

| ID | Equation | Source |
|---|---|---|
| Eq 1.1 | Z = f·B/d (depth from disparity) | Any textbook form; introduce with purpose clause + where-clause naming f, B, d per patterns.md §6 |

### 3.4 PO evidence to log
PO1 (engineering knowledge: geometry, sensing trade-offs), PO2 (problem
analysis: the constraint triangle, objectives formulation). CEP K-profile
statement in §1.4.

---

## 4. Chapter 2: Literature Review (~24 pages, the masterclass)

**Prose pattern:** flagship register; present tense for a paper's claim,
past for its specific finding; `[SYSTEM] [ref] [verb]s [mechanism]`
compression for minor works, fuller treatment (diagram + equations) for the
~10 anchor architectures. Every architecture diagram is REDRAWN in our
style (never pasted from a paper PDF): reference PNGs for fidelity live in
`model/designs/StereoLite/arch_refs/` (18 diagrams: RAFT-Stereo, IGEV,
IGEV++, Selective, CoEx, CGI-Stereo, PSMNet, GANet, NMRF, BANet, DEFOM,
FoundationStereo, StereoAnywhere, ...) and official source code in
`.claude/skills/stereo-vision-expert/reference_impls/` (raft_stereo,
igev_stereo, selective_igev/raft, coex, lite_any_stereo). When a wiring
detail matters, cite the reference_impl file, not memory.

**Mandatory research step before writing each subsection:** read the listed
summary files end-to-end; for anchor models also skim the arch_ref PNG and,
if equations are reproduced, the tier1/tier2 summary's equation section.
The `stereo-vision-expert` skill/agent is the arbiter for any claim the
summaries leave ambiguous.

### 4.1 Section breakdown

| § | Title | Content | Research sources (read first) |
|---|---|---|---|
| 2.1 | Stereo Matching Fundamentals | Rectification, epipolar constraint, disparity; the Scharstein-Szeliski 4-step taxonomy (matching cost, aggregation, optimization, refinement); classical costs (SAD, NCC, census); SGM as the classical apex with its energy function | tier1 `surveys/Scharstein_Taxonomy_IJCV2002.md`, `surveys/Hirschmuller_SGM_TPAMI2007.md`; `papers/CONCEPTS.md`; review_paper §02.1 |
| 2.2 | The Deep Learning Transition (2015 to 2017) | MC-CNN (learned matching cost, rest classical); DispNetC (first end-to-end + correlation layer); GC-Net (4D cost volume + 3D conv + soft-argmin: the template every later network follows) | tier2 `end_to_end/{MC-CNN,DispNetC,GCNet}.md`; tier1 `surveys/Poggi_Synergies_TPAMI2021.md` for the transition narrative |
| 2.3 | The 3D Cost-Volume Era (2018 to 2021) | PSMNet (SPP + stacked hourglass, the workhorse baseline); GwcNet (group-wise correlation, DIRECTLY relevant: our TileInit uses it); GA-Net (learned SGM aggregation); brief: AANet, ACVNet, CFNet | tier2 `end_to_end/{PSMNet,GWCNet,GANet,AANet,ACVNet,CFNet}.md` |
| 2.4 | Iterative Refinement Era (2021 to 2024) | RAFT-Stereo (all-pairs correlation pyramid + ConvGRU + convex upsampling: THREE mechanisms our model inherits in modified form, say so explicitly); IGEV (Geometry Encoding Volume: the direct ancestor of our GEV block); CREStereo, Selective-Stereo one paragraph each | tier1 `iterative/{RAFT-Stereo,IGEV-Stereo,CREStereo,Selective-Stereo}.md`; `reference_impls/raft_stereo/` + `igev_stereo/` for wiring facts |
| 2.5 | Foundation-Model Era (2024 to 2026) | The latest trend: FoundationStereo, DEFOM-Stereo, MonSter, Stereo-Anywhere; the common recipe (monocular prior + massive data); accuracy vs deployability trade-off; why these motivate rather than solve the edge problem | tier1 `foundation_model/{FoundationStereo,DEFOM-Stereo,MonSter,StereoAnywhere}.md` + `_SYNTHESIS_foundation_model.md`; review_paper §03 |
| 2.6 | Efficient and Edge-Deployable Stereo | THE load-bearing section (~7 pages). Organize by compression technique family, not by paper (mirror review_paper §04's taxonomy): (a) backbone substitution (MobileNet/Ghost/YOLO-derived encoders: MobileStereoNet, our lineage); (b) cost-volume compression (cascade, bilateral grid BGNet, pruning DeepPruner, tile hypotheses HITNet); (c) iterative-loop compression (LightStereo, DTP); (d) knowledge distillation (LiteAnyStereo 3-stage KD, Distill-then-Prune); (e) architectural compression (CoEx guided excitation, CGI-Stereo); (f) adaptive compute (AnyNet anytime, MADNet online adaptation); (g) NAS (LEAStereo, briefly). Each family: definition, 1 to 2 anchor papers with mechanism + numbers, what it costs in accuracy | tier1 `efficient/*` (18 files: BGNet, CoEx, LightStereo, LiteAnyStereo, MADNet, CGI-Stereo, Pip-Stereo, ...); tier3 `efficient/{HITNet,StereoNet,AnyNet,MobileStereoNet,DeepPruner}.md`; **review_paper §04 (compression taxonomy) is the skeleton, rewrite for thesis register, do not copy**; §06 for edge-hardware operator costs |
| 2.7 | Cross-Domain Generalization | Why synthetic-trained models fail on real domains; architecture-level (domain-invariant features: DSMNet, FCStereo, GraftNet) vs training-recipe-level (data mixing, KD from foundation teachers: StereoAnything, LiteAnyStereo); the evidence that training pipeline, not iteration, carries generalization | tier3 `domain_shift/*`; tier1 `efficient/LiteAnyStereo.md`; review_paper §05; CLAUDE.md cross-domain insight section |
| 2.8 | Benchmarks and Datasets | SceneFlow (FlyingThings3D/Monkaa/Driving, 35k+ training pairs), KITTI 2012/2015, Middlebury 2014 (the eval3 protocol: 15 train GT public + 15 test hidden; zero-shot convention evaluates on the GT-public scenes), ETH3D; metric definitions deferred to Ch 3 | tier2 `datasets/*`; review_paper §02.3; MB14 protocol facts from this session's eval work |
| 2.9 | Summary and Research Gap | Synthesis table of the anchor methods (params, SF EPE, KITTI D1, latency, hardware); the gap sentence pattern: "the surveyed literature does not directly address [sub-3 M, tile-based, GEV-fused, real-time-on-Orin]"; NO roadmap prescriptions (review-honesty rule) | `papers/verified_performance.md` + `method_data.py` for the table |

### 4.2 Figures (redraw = diagram-drawer skill + diag_helpers, thesis style §9)

| ID | Figure | Source/tool | Status |
|---|---|---|---|
| Fig 2.1 | Classical 4-step stereo pipeline (Scharstein taxonomy) | NEW, simple 4-box flow, ASCII-to-matplotlib via diag_helpers | NEW (small) |
| Fig 2.2 | Historical evolution timeline 2016 to 2026, six paradigm lanes | `review_paper/figures/_data/make_timeline.py` output `fig_timeline.pdf` | REUSE (regenerate with thesis style; the script is data-driven) |
| Fig 2.3 | Generic deep stereo pipeline (encoder, cost volume, aggregation, regression + the iterative loop variant) | NEW via diag_helpers (`cv_prism`, `loop_glyph` exist for exactly this) | NEW |
| Fig 2.4 | Anchor architectures, redrawn simplified: (a) PSMNet (3D CV paradigm), (b) RAFT-Stereo (iterative GRU paradigm), (c) HITNet (tile-hypothesis paradigm), (d) LiteAnyStereo (KD-trained efficient paradigm). One row each, shared visual language so paradigm differences pop | NEW, 4-panel; fidelity from `arch_refs/` PNGs + reference_impls | NEW (biggest single figure task in Ch 2, ~half day) |
| Fig 2.5 | Compression technique taxonomy tree (7 families) | `review_paper/figures/fig_taxonomy_tikz.tex` (native TikZ) | REUSE (recompile inside thesis; check dvipsnames xcolor pitfall from CLAUDE.md) |
| Fig 2.6 | Accuracy vs params Pareto AND accuracy vs latency Pareto (two panels), ~40 methods, family-coded shapes | `make_param_pareto.py` + `make_pareto.py` | REUSE (regenerate, thesis style, add our model's marker) |

### 4.3 Equations (each: purpose clause, display, where-clause; cite source paper)

| ID | Equation | Anchors section | Source for correctness |
|---|---|---|---|
| Eq 2.1 | SGM energy E(D) with P1/P2 smoothness penalties | 2.1 | tier1 SGM summary |
| Eq 2.2 | Correlation layer (DispNetC) / dot-product matching cost | 2.2 | tier2 DispNetC summary |
| Eq 2.3 | Soft-argmin disparity regression (GC-Net) | 2.2 | tier2 GCNet summary |
| Eq 2.4 | Group-wise correlation (GwcNet): the exact op our TileInit adopts | 2.3 | tier2 GWCNet summary |
| Eq 2.5 | ConvGRU update equations (RAFT-Stereo) | 2.4 | tier1 RAFT-Stereo summary + `reference_impls/raft_stereo/update.py` |
| Eq 2.6 | Geometry Encoding Volume construction (IGEV) | 2.4 | tier1 IGEV summary + `reference_impls/igev_stereo/geometry.py` |
| Eq 2.7 | Generic KD loss (teacher-student disparity distillation, LiteAnyStereo form) | 2.6/2.7 | tier1 LiteAnyStereo summary |

Cap at ~7 reproduced equations; more turns the review into a derivation dump.

### 4.4 Tables

| ID | Table | Source |
|---|---|---|
| Tab 2.1 | Datasets (name, size, real/synthetic, GT source, resolution, role) | review_paper `_tables/tab_datasets.tex` content, reformatted |
| Tab 2.2 | Anchor-method comparison (12 to 15 rows: params, SF EPE, KITTI D1, latency+hardware, technique family) | `verified_performance.md` + `method_data.py`; every cell keeps its source breadcrumb in a bib citation |

### 4.5 PO evidence
PO2 throughout (systematic literature analysis, gap identification). Log per
subsection.

---

## 5. Chapter 3: Methodology (~22 pages)

**Prose pattern:** method register: flat present tense, assert mechanisms,
`To [SUB-GOAL], we exploit [COMPONENT] to [FUNCTION]`, tensor dims inline in
prose, every equation introduced then where-claused. The single source of
truth is `model/designs/StereoLite_yolo_ctx_gev4/FINAL_MODEL_ARCHITECTURE.md`
(20 sections; §§3-11 are the block specs, §12 the loss, §16 metric
definitions, §18 the pre-vetted thesis-safe claim wording). Cross-check any
detail against the code in `model/designs/StereoLite_yolo_ctx_gev4_opt/`
(`model.py`, `tile_propagate.py`) because the _opt folder is what actually
trained; cite file:line in NOTES.md when the doc and code disagree.

### 5.1 Section breakdown

| § | Title | Content | Sources |
|---|---|---|---|
| 3.1 | Problem Formulation and Design Requirements | Formal task statement (rectified pair to dense disparity); the hard envelope as design requirements: < 3 M params, real-time on ~4 to 6 TOPS, < 200 MB inference memory, INT8-exportable ops; research questions (RQ1 accuracy at budget, RQ2 zero-shot generalization, RQ3 edge latency); assumptions (rectified input, max disparity 192) | `THESIS_PLAN.md` methodology memory (100-pair baseline + envelope); FINAL_MODEL_ARCHITECTURE §1-2 |
| 3.2 | System Overview | The full pipeline narrated once, end-to-end, against Fig 3.1; one paragraph per stage; the three inherited-and-modified mechanisms named with their Ch 2 ancestry (group-wise cost from GwcNet, ConvGRU + convex upsample from RAFT-Stereo, GEV from IGEV, tile-plane hypotheses from HITNet) | FINAL_MODEL_ARCHITECTURE §3 (end-to-end data flow) |
| 3.3 | Shared Feature Encoder and Context Encoder | YOLO26s-derived shared encoder, fL/fR at 1/4, 1/8, 1/16; the separate left-only context encoder (ctx16/8/4, 32 ch out) and WHY matching features must not double as context (RAFT lesson) | §4, §5 of the arch doc |
| 3.4 | Tile Initialization and Cost Volume | 8-group 24-hypothesis correlation volume at 1/16 (Eq 3.1 group-wise score), 3D aggregation 8-16-16-1, softmax, soft-argmax init (Eq 3.2), confidence c0; tile state tuple T = (d, sx, sy, h, c) defined here | §6 |
| 3.5 | Recurrent Tile Refinement | Warp + local correlation lookup (half_range 2, 5 offsets); input assembly (Eq 3.3); ConvGRU quartet (Eq 3.4); 4 sigmoid update gates, softplus disparity; schedule 2/3/3 across 1/16, 1/8, 1/4, weights unshared and why | §7 |
| 3.6 | Plane-Aware Cross-Scale Propagation | The slanted-plane child-disparity equation (Eq 3.5): slopes are used, not stored; contrast with naive bilinear upsampling | §8 |
| 3.7 | Quarter-Resolution Geometry Encoding Volume and Fail-Soft Fusion | GEV4: 8-group 64-hypothesis volume, narrow variant ±16 around tile disparity (the validated efficiency knob), three 3x3x3 convs; expectations d_gev, c_gev, g_gev (Eq 3.6); the learned fusion gate w with bias init -4 (Eq 3.7) and the fail-soft design argument | §9; efficiency finding F3 in THESIS_PLAN §3 |
| 3.8 | Plane Rendering and Learned Convex Upsampling | Plane-equation rendering with edge gate (the blur fix, cite the ablation that selected it); two 2x convex-upsample stages with 9-weight softmax masks | §10; blur-fix run `20260703_blurfix_n500` |
| 3.9 | Training Objective | The full multi-scale loss (Eq 3.8): weighted L1 across scales + GEV aux 0.15 + gradient 0.50 + threshold 0.20 + D1 0.20 + edge-smoothness 0.02 + gated slant supervision (slant_w 0.3); name each term's job in one sentence | §12; trainer `model/scripts/train_full_sceneflow.py::loss_fn` provenance |
| 3.10 | Datasets and Training Protocol | SceneFlow canonical split (35,454 train = FT3D-TRAIN + Monkaa + Driving; 4,370 test = FT3D-TEST, finalpass); native 384x640 co-located random crops (vs legacy resize, cite the native-vs-resize ablation); OpenStereo augmentation triplet (color jitter, right-only erase, random scale); OneCycle 60k steps, peak 8e-4, batch 32, bf16, A100-80GB; input contract [0,1]; val protocol (400-pair fixed subset, native 960x540 pad16 axis) | meta.json `args`; `20260704_native_vs_resize_n500` comparison.md; grand-comparison aug evidence |
| 3.11 | Design-Space Exploration Methodology | How architecture decisions were MADE: the matched-overfit harness protocol (fixed pairs, fixed seed, full 8-metric reporting), controlled single-knob ablations, pre-registered win criteria, cross-dataset gate; this section is pure PO4-style methodology and sets up the Ch 4 ablation results | `model/benchmarks/OVERFIT_METHODOLOGY.md`; ablation-study-expert protocol |
| 3.12 | Optimization of the Network for Edge Inference | The F1-F7 efficiency pass as an engineering method: cost-volume zero-copy views, grid_sample batching, static-channel hoisting, head fusion, dead-code removal, narrow GEV; equivalence-proof discipline (bitwise/matched-A/B); export blockers identified (F.unfold, GroupNorm, Conv3d) and mitigations | `THESIS_PLAN.md` §3 (F1-F8 table); this is thesis content, not a detour |
| 3.13 | Evaluation Metrics | EPE, RMSE, median AE, bad-{0.5,1,2,3}, D1-all, formally defined (Eq 3.9 group); latency and memory measurement protocol (batch 1, resolution, warmup, fp16/fp32) | FINAL_MODEL_ARCHITECTURE §16 |
| 3.14 | Implementation and Deployment Environment | PyTorch 2.11/cu128; Modal A100 training + T4 eval; RTX 3050 local bench; the CCB stereo camera rig; target device class (Jetson Orin Nano) | meta.json; THESIS_PLAN §2A |

### 5.2 Figures (the biggest figure workload; all NEW via diag_helpers)

No gev4 architecture diagram exists anywhere yet (verified). Build them in
this order; each per-block figure shares the visual language of Fig 3.1 so
the reader can locate every block in the overview.

| ID | Figure | Content spec | Status |
|---|---|---|---|
| Fig 3.1 | **Full gev4_opt_narrow_plane architecture** (the thesis centerpiece, full-width, possibly landscape) | L/R images, shared YOLO26s encoder (3 scales), left context encoder as a visually distinct stream, TileInit cost prism at 1/16, tile-state pill (d, sx, sy, h, c), GRU refine loops x2/3/3 with scale ladder, plane-propagation arrows between scales, GEV4 prism at 1/4 + fusion gate, plane rendering + convex upsample to full res, supervision dots on every supervised output | NEW (~1 day incl. iteration; source spec FINAL_MODEL_ARCHITECTURE §3-11; use diag_helpers block/cv_prism/loop_glyph/sup_dot palette) |
| Fig 3.2 | Encoder + context encoder detail (channel counts, strides, which scales feed what) | NEW |
| Fig 3.3 | Tile initialization: group-wise volume construction + 3D aggregation + soft-argmax (adapt `stage2_init` concept to gev4 reality) | ADAPT/NEW |
| Fig 3.4 | One refinement iteration unrolled: warp, corr lookup, input assembly, ConvGRU, gated update (adapt `stage3_refine` concept) | ADAPT/NEW |
| Fig 3.5 | Plane-aware propagation: parent tile to 4 children with slope terms (small schematic) | NEW |
| Fig 3.6 | GEV4 + fail-soft fusion: narrow band around tile disparity, gate w | NEW |
| Fig 3.7 | Convex upsample mechanics (adapt `stage4_upsample`, still valid) | ADAPT |
| Fig 3.8 | Supervision map: which outputs receive which loss terms at which scales (adapt `supervision_diagram` with gev4 §12 weights) | ADAPT |
| Fig 3.9 | Training pipeline flowchart: shards, native-crop sampling, augmentation triplet, OneCycle, val loop (adapt `methodology_pipeline`) | ADAPT |
| Fig 3.10 | Native-crop vs resize input protocol illustration (one SceneFlow frame showing crop windows vs global downscale) | NEW (small, screenshot-based) |
| Fig 3.11 | Parameter budget donut for gev4_opt_narrow_plane (compute real split from `model.py` module param counts; adapt `budget_diagram`) | ADAPT (recompute numbers, never reuse v8 split) |

### 5.3 Equations (from FINAL_MODEL_ARCHITECTURE, keep doc's notation)

Eq 3.1 group-wise matching score C(g,d,y,x) · Eq 3.2 soft-argmax d0 +
confidence c0 · Eq 3.3 recurrent input assembly x = [fL, warp(fR,d), d, sx,
sy, c, corr, ctx] · Eq 3.4 ConvGRU quartet (z, r, q, h') · Eq 3.5 plane
child disparity d_child = 2·bilinear(d) + 2·sx·dx + 2·sy·dy · Eq 3.6 GEV
expectations (d_gev, c_gev, g_gev) · Eq 3.7 fusion gate w + d_fused ·
Eq 3.8 total training loss L with all term weights · Eq 3.9 metric
definitions (EPE, bad-t, D1-all; group as aligned block or 3 equations).

### 5.4 Tables

| ID | Table | Source |
|---|---|---|
| Tab 3.1 | Full architecture configuration (channels, groups, hypotheses, iteration schedule, param count 2.9623 M) | FINAL_MODEL_ARCHITECTURE §11 config table |
| Tab 3.2 | Training hyperparameters (optimizer, schedule, batch, crop, augmentation, steps, hardware) | meta.json `args` block |
| Tab 3.3 | Efficiency optimizations F1-F7 (finding, saving, equivalence status) | THESIS_PLAN §3 table, reworded for thesis |

### 5.5 PO evidence
PO2 (formulation), PO3 (design/development: §3.2-3.9, §3.12), PO5 (modern
tools: PyTorch, Modal, mixed precision, §3.14). CEP K5/K8 reinforced.

---

## 6. Chapter 4: Results and Discussions (~18 pages)

**Prose pattern:** past tense for what was done/found; every claim carries
its number in the same sentence; `As shown in Table N` pointers; honest-cost
sentences for every trade-off; ablation-delta phrasing per patterns.md §8.
The asterisk convention for calculated Jetson numbers is stated once in a
table footnote and respected everywhere.

### 6.1 Section breakdown

| § | Title | Content | Sources |
|---|---|---|---|
| 4.1 | Experimental Setup Summary | One page: what was trained (locked config), on what, evaluated how; pointer back to §3.10-3.14 | meta.json |
| 4.2 | Training Dynamics | The 60k run: loss + LR schedule, dual-axis val EPE curve, best-at-53k semantics, preemption robustness note; qualitative convergence filmstrip | `train.csv` (679 rows, cols documented); `images/` tracked folders |
| 4.3 | In-Domain Accuracy: SceneFlow Test | THE headline: full FT3D-TEST 4,370 pairs, native axis: EPE 0.781, bad-1 8.92, bad-2 5.34, bad-3 4.00, D1 3.40, RMSE 3.64, median 0.130; 4 degenerate frames excluded (state it); comparison table vs published methods at matched-ish protocol WITH the axis caveat honestly stated (our native eval vs their standard protocol; do not present as same-protocol unless it is) | meta.json `final_metrics_all`; `verified_performance.md` for baseline rows |
| 4.4 | Zero-Shot Cross-Domain: Middlebury 2014 | 23 perfect scenes, 384x640 protocol: EPE 1.71, D1-all 10.86; the reference ladder (legacy chassis 40.1, ours 10.86, LiteAnyStereo 6.9 with KD + 7.6 M params, IGEV 5.0 with 12.6 M + 16 iters); per-scene table (Adirondack 2.38 D1 to the hard scenes); interpretation: training pipeline + native crops closed the collapse, KD gap remains and is future work | `mb14_zero_shot.json` (aggregate + per_scene); CLAUDE.md reference numbers |
| 4.4b | Robustness to Rectification Imperfection | Evidence for thesis objective 2: vertical-misalignment sweep (see §11a). Perturb the right image by +-{0.5, 1, 2, 4} px vertical shift (and optionally small rotation) on the 400-pair val subset; report EPE/D1 degradation curve for our model, ideally alongside one reference model at 0 and 2 px for context; interpret which architectural elements confer the tolerance (2D local correlation lookup in TileRefine, GEV band, learned fusion) WITHOUT overclaiming: the claim is measured tolerance, not designed-in immunity, unless the measurements support more | New experiment (§11a); `eval` harness variant |
| 4.5 | Ablation Studies | Present the design evidence: (a) augmentation lever (grand comparison: best val EPE 2.778 to 1.921, D1 -22.6%); (b) efficiency A/B (gev4 46.6 ms to gev4_opt_narrow 30.2 ms at equal accuracy); (c) blur-fix round (plane rendering bad-0.5 -22.5%) AND the composition negative result (plane+bimodal WORSE than plane alone: report it, negative results are results); (d) native-vs-resize (native-axis EPE 6.67 to 2.87). Each: compact table + 2-paragraph interpretation | `EXPERIMENTS.md` sections; run folders named in THESIS_PLAN §3b |
| 4.6 | Efficiency and Deployment Measurements | RTX 3050 measured: fp32 61.4 / fp16 49.8 ms (gev4_opt_narrow), plane variant ~62.5 ms fp32; inference memory 0.26 GB fp16 / 0.35 GB fp32; Jetson Orin Nano PROJECTIONS with asterisk + the projection methodology sentence (eager-to-TRT x1.8, fp16-to-INT8 x1.2, device factor x1.15: ~30 ms*, ~33 FPS*); footnote: replaced by real measurements when hardware is available | THESIS_PLAN §2A (A2, A6) + Jetson projection block |
| 4.7 | Qualitative Results | SceneFlow test panels (good/typical/failure cases with GT), MB14 panels (best + worst scenes), real-camera panels (no GT: visual plausibility, edge sharpness); tie back to the plane-rendering fix visually | `images/` folders; `gev4_camera_smoke/` regenerated with best.pth |
| 4.8 | Societal, Environmental and Sustainability Aspects | REQUIRED by OBE (PO6/PO7): edge inference vs cloud (privacy, no video egress; energy per inference at 7 to 15 W vs datacenter round-trip); passive stereo vs LiDAR (no emitted radiation, lower BOM cost, no rare materials); training cost transparency (one A100 day, ~$15 Modal spend); accessibility angle (low-cost robotics in developing regions) | Write from first principles + device spec sheets; keep claims modest and cited or clearly reasoned |
| 4.9 | Discussion | Answer RQ1-RQ3 explicitly; position on the Pareto figure; what the results imply about the "iteration vs training pipeline" question for generalization; honest boundary: no on-device number yet, no KITTI leaderboard submission, single-run (no seed variance) | Synthesis; FINAL_MODEL_ARCHITECTURE §18 wording |

### 6.2 Figures

| ID | Figure | Tool | Status |
|---|---|---|---|
| Fig 4.1 | Training curves: train loss + OneCycle LR (twin axis) and val EPE/bad-1 vs step with best-checkpoint marker at 53k | NEW script `thesis/book/figures/_src/make_training_curves.py` reading `train.csv` (dedupe keep-last-per-step across restart legs!) | NEW |
| Fig 4.2 | Convergence filmstrip: 2 to 3 tracked scenes, prediction at steps {1k, 5k, 15k, 30k, 53k} + GT | `model/scripts/build_viz_filmstrip.py` or a small custom mosaic over `images/val_XX/step_*.png` | NEW (tool exists) |
| Fig 4.3 | SceneFlow qualitative grid: 4 scenes x (left, GT, ours, error map) | NEW small script over `images/` | NEW |
| Fig 4.4 | MB14 per-scene D1 bar chart (23 scenes, sorted, reference lines for LiteAnyStereo/IGEV aggregate) | NEW from `mb14_zero_shot.json` per_scene | NEW |
| Fig 4.5 | MB14 qualitative: 3 best + 2 hardest scenes (left, GT, ours) | Requires saving predictions: extend `eval_gev4_middlebury2014.py` with a `--save_viz` flag (writes turbo-colormapped pred + GT per scene to the results volume), re-run (~$0.05) | NEW (small driver edit + rerun) |
| Fig 4.6 | Real-camera panels: 3 to 4 indoor pairs, left + predicted disparity | Regenerate `gev4_camera_smoke` with `best.pth` (script exists per THESIS_PLAN A4) | REGENERATE |
| Fig 4.7 | Ablation summary: grouped bar chart (aug lever, efficiency latency, blur-fix bad-0.5, native-vs-resize EPE), one panel per ablation | NEW from EXPERIMENTS.md numbers | NEW |
| Fig 4.8 | Final Pareto positioning: params vs MB14 D1 (or SF EPE) with our star, LiteAnyStereo, IGEV, HITNet, BGNet, LightStereo | Extend `method_data.py` locally with our row; adapt `build_research_gap` | ADAPT |

### 6.3 Tables

| ID | Table | Source |
|---|---|---|
| Tab 4.1 | SceneFlow FT3D-TEST full results (all 8 metrics, 4,370 pairs, axis stated) | meta.json final_metrics_all |
| Tab 4.2 | Method comparison on SceneFlow (ours + 8 to 10 published; params, EPE, latency + hardware column; protocol caveats in footnotes) | verified_performance.md |
| Tab 4.3 | MB14 zero-shot ladder (legacy, ours, LiteAnyStereo, IGEV: all 8 metrics + params + T4 ms) | mb14_zero_shot.json + CLAUDE.md reference table |
| Tab 4.4 | MB14 per-scene results (23 rows, EPE + D1) | mb14_zero_shot.json per_scene; consider appendix if too long |
| Tab 4.5 | Ablation: efficiency A/B (3 arms x metrics + latency) | EXPERIMENTS.md eff_gev4_n100 |
| Tab 4.6 | Ablation: blur-fix + composition (control/bundle1/plane/bimodal/pb/allin) | EXPERIMENTS.md blurfix runs |
| Tab 4.7 | Ablation: native-vs-resize (3 arms, both eval axes) | 20260704_native_vs_resize_n500 |
| Tab 4.8 | Latency and memory (3050 fp32/fp16 measured; Orin Nano projected*; the asterisk footnote lives here) | THESIS_PLAN §2A |

### 6.4 PO evidence
PO4 (investigation: §4.2-4.5, §4.9), PO6 (society: §4.8), PO7
(environment/sustainability: §4.8).

---

## 7. Chapter 5: Conclusions and Future Work (~5 pages)

| § | Title | Content |
|---|---|---|
| 5.1 | Summary of Findings and Contributions | Mirror Ch 1 objectives one-to-one: objective, what was done, the number that proves it. Contribution list per patterns.md §3 (3 to 4 "We" bullets, last bullet is the result claim). Use FINAL_MODEL_ARCHITECTURE §18 recommended wording |
| 5.2 | Limitations | Honest hedge register: no on-device Jetson measurement yet (projections carry asterisks); single training run, no seed variance; MB14 gap to KD-trained models; max disparity 192 cap; synthetic-only supervision; no KITTI leaderboard submission |
| 5.3 | Future Work | (a) 3-stage KD pipeline from a foundation teacher (the LiteAnyStereo evidence says this is the biggest lever); (b) INT8/TensorRT export + real Orin Nano measurement (export blockers already identified in §3.12); (c) KITTI + ETH3D zero-shot quartet completion; (d) temporal extension (TempTile direction, shelved with math validated); (e) publication path |

No figures. PO12 (life-long learning: future-work reasoning shows awareness
of the field's trajectory). One reserved forward claim maximum ("we believe
... will be useful as ...", RAFT-Stereo pattern).

---

## 8. Appendices

| App | Content | Sources / actions |
|---|---|---|
| A: Technical Specifications | Full model config (dump FINAL_MODEL_ARCHITECTURE §11 + meta.json cfg as formatted tables); software stack versions; hardware specs (A100, T4, RTX 3050, camera rig, Orin Nano target spec); reproduction pointers (the three Modal drivers by name) | Ready-made content; PO5 |
| B: Project Management and Financial Aspects | Expense table (Modal spend: itemize training ~$15, ablations, eval runs ~$0.05 each; camera hardware; SSD); Gantt chart of the actual project timeline (Feb to Jul 2026: paper study, review paper, architecture iterations, ablation campaign, production run, writing); seed from the deck's Time Plan slide | NEW Gantt via matplotlib (`figures/_src/make_gantt.py`); PO11 |
| C: Ethics, Teamwork and Communication | (i) Similarity report (Turnitin via SUPERVISOR account). **TIMING DECIDED (user, 2026-07-05): similarity + AI report are obtained AFTER the thesis book is completely written**, so they are the LAST step of Phase 4 QA, not a day-0 external clock; leave placeholder pages in the appendix until then; (ii) **AI report + AI-usage disclosure: mandatory, this thesis is AI-assisted; disclose scope honestly** (drafting assistance, code assistance, analysis tooling); (iii) Research Ethics Compliance Checklist + Conflict-of-Interest statement + signatures; (iv) CRediT contribution statement for the 2 students (draft from the actual division: Abrar X, Rahi Y; confirm with both); (v) communication statement (supervisor meetings cadence) | ruet-thesis-expert §4; PO8/9/10 |
| D: PO and KPA Attainment Tracker | Assemble from `po_tracker_notes.md` accumulated during writing; per-PO: where demonstrated + one-line evidence; CEP K3/K4/K5/K8 statement | Last thing written before QA |
| E: List of Publications | The review-paper draft ("Compression Techniques for Deep Stereo Matching", 17-page IEEE-format manuscript, status: draft/in preparation) or "none" if supervisor prefers | Confirm with supervisor |

---

## 9. Figure production pipeline and style contract

All thesis figures obey ONE style so the book reads as a single artifact:

- **Print-first:** white background (NOT the deck's cream), vector PDF
  primary output + 300 dpi PNG fallback, serif text (match Times body),
  base font 9 to 10 pt at final printed size.
- **Palette:** the diagram-drawer block palette (encoder blue, cost-volume
  yellow, refinement green, supervision crimson, frozen grey dashed) for
  architecture figures; colorblind-safe categorical palette for data charts
  (consult the `dataviz` skill before each chart); family shape-coding
  (star/circle/square/triangle/diamond) for multi-method scatter plots per
  the CLAUDE.md figure-design lessons.
- **One generator per figure** in `thesis/book/figures/_src/`, each writing
  `thesis/book/figures/fig_<chapter>_<slug>.{pdf,png}`. Adapted scripts get
  COPIED into `_src/` and modified there; never mutate `presentation/` or
  `review_paper/` scripts for thesis needs.
- **Caption discipline:** self-contained (dataset, split, metric named every
  time); figure captions = bold noun phrase + one dataflow sentence; table
  captions = bold noun phrase + protocol sentence (patterns.md §7).
- **Diagram redraw rule:** other papers' architectures are redrawn in our
  visual language from `arch_refs/` references; never screenshot a paper
  figure into the thesis.
- **Glyphs/icons:** small pictorial glyphs (camera, GPU chip, robot, image
  thumbnails as placeholders) may be generated via the OpenRouter API
  (key in `/home/abrar/Research/stero_research_claude/.env`,
  `OPENROUTER_API_KEY`) using an image model, then embedded into matplotlib
  figures via `OffsetImage`. Keep glyphs monochrome/duotone so they read as
  diagram elements, not stock art. Real image thumbnails (left/right/
  disparity) always come from actual run artifacts, never generated.
- **Visual iteration loop (mandatory for architecture figures):** render to
  PNG, LOOK at it (Read tool), fix overlaps/spacing/arrow crossings,
  re-render; minimum 2 iterations before accepting. Benchmark the look
  against the paper figures in `arch_refs/` (RAFT-Stereo, IGEV, CoEx
  style: left-to-right dataflow, labeled tensor shapes, distinct block
  colors per stage, loop arrows for recurrence).
- Estimated figure workload: ~26 figures total; ~10 reuse/adapt (fast),
  ~16 new. The two big-ticket items are Fig 3.1 (full architecture, ~1 day)
  and Fig 2.4 (4-paradigm panel, ~0.5 day). Everything else is 0.5 to 2 h
  each. Budget ~5 working days of figure work total, parallelizable with
  prose.

---

## 10. QA gates

**Per chapter (before marking done):** the ruet-thesis-expert §9 checklist
verbatim (Mechatronics everywhere; heading fonts; caption placement;
PO notes logged; numbers cited; no dashes; compiles clean) PLUS:
- Every `[SLOT]` template filled, no placeholder text left.
- Every figure referenced in prose before it appears; every table pointed
  to with "As shown in Table N".
- Register check against patterns.md (method sections assert; results carry
  numbers; only Ch 5 hedges).
- grep the diff for `--`, ` – `, ` — `: zero hits in prose.

**Final assembly (week 4):**
1. Full compile x3 + bibtex; zero `??`, zero duplicate labels.
2. Page-by-page skim of the PDF at low zoom: orphan pages (< 60 words),
   float pileups, caption widows.
3. Roman/arabic numbering transitions correct; ToC/LoF/LoT/nomenclature
   complete and matching body.
4. Citation audit: every `[n]` resolves; order of first appearance holds;
   entry style consistent 100%.
5. Number audit: re-derive every table cell from its named source file one
   final time (meta.json, mb14_zero_shot.json, EXPERIMENTS.md,
   verified_performance.md).
6. Abstract last-write check: ≤ 300 words, 7-move structure, contains the
   headline numbers (2.96 M params, SF-TEST EPE 0.78, MB14 zero-shot D1
   10.9%, 3050 fp16 ~50 ms, Orin projection asterisked), one honest
   limitation sentence.
7. Ethics appendix complete: similarity report + AI report physically
   attached, signatures collected.

---

## 11. Execution order and dependencies

```
Day 0 (now):        DECIDED by user 2026-07-05: title fixed, objectives
                    fixed, pure IEEE citations, similarity+AI report
                    deferred to after the book is complete. No supervisor
                    blockers remain for starting.
Day 1:              B1 scaffold thesis/book/ + compile gate + commit.
Day 1-2:            Ch 4 TABLES first (numbers frozen, zero prose risk):
                    Tab 4.1-4.8 as .tex with source comments.
                    Fig 4.1 training curves + Fig 4.4 MB14 bars (data-driven,
                    fast). Regenerate camera panels with best.pth (A4).
Day 2-5:            Ch 3 prose (arch doc is ready; highest certainty).
                    In parallel: Fig 3.1 full-architecture diagram, then
                    Fig 3.2-3.11.
Day 5-9:            Ch 2 prose (biggest chapter; read summaries per §4.1
                    table as you go). In parallel: Fig 2.1-2.6 + the
                    eval driver --save_viz edit + MB14 qualitative rerun.
Day 9-11:           Ch 4 prose around the finished tables/figures.
Day 11-12:          Ch 1 (funnel; write after 2/3/4 exist so forward
                    references are real). Ch 5 (mirrors Ch 1 objectives).
Day 12-14:          Appendices A-E (D assembled from po_tracker_notes.md;
                    B needs the Gantt + expense pull from Modal billing).
Day 14:             Frontmatter: acknowledgments, declaration, certificate
                    finalization; ABSTRACT WRITTEN LAST.
Day 15-16:          QA gates (§10), full-book read-through, supervisor
                    draft handoff.
Slack:              Jetson arrival at any point: swap asterisked values in
                    Tab 4.8 + §4.6 + abstract, delete the projection
                    methodology sentence, add measurement protocol.
```

### 11a. Required experiment: rectification-robustness sweep (objective 2 evidence)

Thesis objective 2 ("withstand camera imperfections in terms of
rectification") currently has NO measured evidence in the repo. Before Ch 4
is finalized, run a cheap eval-only sweep (no training):

1. New Modal driver (clone `eval_full_testset.py` skeleton):
   `model/scripts/modal/eval_rectification_robustness.py`. Load `best.pth`,
   evaluate the fixed 400-pair val subset (or full test if cheap enough) at
   right-image vertical shifts of {0, 0.5, 1, 2, 4} px (torch.roll or
   grid_sample subpixel shift, replicate-pad edges). Optional second axis:
   +-0.2 deg right-image rotation. Report all 8 metrics per perturbation.
2. Output: degradation table + one line chart (Fig 4.x EPE vs vertical
   offset). Cost ~$0.10 T4, < 30 min.
3. Honest framing rules: if degradation is graceful (e.g. EPE < 2x at 2 px),
   objective 2 is supported by measurement; if it is not, report it as a
   limitation and let objective 2 rest on the architectural argument only
   (local 2D refinement) with the measured caveat. NEVER claim robustness
   without this table.
4. Optional strengthener: same sweep on real camera pairs (visual panels
   only, no GT), showing prediction stability under deliberate
   de-rectification.

Dependency notes:
- Ch 4 now blocks on ONE new experiment (§11a, ~30 min compute). Everything
  else unchanged. Optional nice-to-haves (ETH3D
  zero-shot ~$0.05, KITTI-15 training-split zero-shot if data lands) slot
  into Tab 4.3 as extra columns IF run; do not gate writing on them.
- The MB14 qualitative rerun (Fig 4.5) and camera regeneration (Fig 4.6)
  are the only compute tasks, both < 30 min + < $0.10 total.
- Writing sessions should be per-chapter (or per-half-chapter for Ch 2) to
  keep each session's context focused; every session starts by reading
  this plan's relevant section + the two skills.

---

## 12. Decisions log (settled + still open)

**SETTLED (user, 2026-07-05):**
1. **Title:** "AI-Enhanced Stereo Matching for High-Accuracy Depth Mapping
   and 3D Reconstruction". Fixed.
2. **Objectives:** the two verbatim objectives in §3.1 above. Fixed.
3. **Citation style:** pure IEEE numeric. Fixed.
4. **Similarity + AI report:** after the book is completely written (end of
   Phase 4), not before.

**STILL OPEN (log answers in thesis/book/NOTES.md):**
5. **Model public name:** RECOMMENDATION stands: brand the final model
   "StereoLite" in the thesis (one name, defined once in Ch 3); internal
   variant slugs (gev4_opt_narrow_plane) only in Appendix A.
6. **Appendix E:** list the review-paper draft or not.
7. **MB14 per-scene table:** Ch 4 body vs Appendix A (length call at
   layout time).
8. **Two title pages** (2025 precedent) vs one (Book Template): default two.
