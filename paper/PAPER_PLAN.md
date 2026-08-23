# StereoLite research paper: detailed writing plan

Branch `research-paper-release`. Written 2026-08-23 before drafting, after
(a) the `research-linguistics-expert` skill and its corpus (efficient/edge
bucket), (b) five sonnet readers over 15 model-introducing papers, (c) three
sonnet readers over the 113-page thesis, (d) a haiku artefact inventory, and
(e) direct re-verification of every headline number against the run JSONs
under `model/benchmarks/20260704_fullsf_gev4onp_nc/`.

Notes from the readers live in the session scratchpad
(`paper_notes/readers_{A..E}_*.md`, `thesis_{A,B,C}_*.md`,
`artefact_inventory.md`).

## 1. Positioning (what the paper claims, and what it does not)

One sentence: **a 2.96 M parameter tile-plane recurrent stereo network,
trained on SceneFlow alone, that transfers zero-shot to four real benchmarks,
tolerates up to one pixel of rectification error, and runs INT8 at 36.3 ms on
a Jetson Orin Nano.**

What the 15 papers taught us about positioning:

| Observation from corpus | Consequence for our paper |
|---|---|
| BGNet, LiteAnyStereo, Pip-Stereo, BANet never print a parameter count; HITNet Tab 7 and LightStereo Tab I do, and read better | Print 2.96 M in the abstract, intro, and every results table row |
| None of the 15 (except Pip-Stereo's untraceable abstract number and BANet's Snapdragon breakdown) measure on an embedded device | The measured Orin Nano INT8 number is the differentiator; pin device + resolution + precision + batch in one sentence every time it appears |
| RAFT-Stereo Tab 1, IGEV Tab 7, GGEV Tab 1, LiteAnyStereo Tab 3 all put zero-shot generalization in ONE master table with per-dataset thresholds stated in the caption | Build one master zero-shot table; state thresholds and our protocol in the caption |
| IGEV Tab 7 prints RAFT beating it on ETH3D without hiding it; RAFT-Stereo prints its KITTI second place | Print the ETH3D bad-1 and MB14 numbers where we are weak, plainly |
| LiteAnyStereo Sec 4.5 is a 3-sentence Limitations paragraph naming root cause + symptom metric | Write a short Limitations paragraph exactly in that shape |
| GGEV Fig 6/7 isolate architecture from iteration count; Pip-Stereo Tab 2 claims iteration is "indispensable" for generalization | Engage Pip-Stereo's claim directly: 8 coarse updates on a tile state land in the efficient-CNN zero-shot band without a 16 to 32 iteration schedule |
| CREStereo's "no fine-tuning" Middlebury/ETH3D still mixes 2% target data | State explicitly that no KITTI/ETH3D/MB14 pixel is ever seen in training |

Honesty constraints that bind every sentence:
- Zero-shot numbers are training-split evaluations under our own protocol,
  not leaderboard submissions. Say so in the abstract, the table caption, and
  Limitations.
- "outperforms" only with the beaten method named and the margin given;
  "state-of-the-art" nowhere (no table supports it).
- No en-dash `--` and no em-dash `---` in prose. Ranges use "to".
- Every number traces to a file: `meta.json`, `mb14_zero_shot.json`,
  `kitti_eth3d_zero_shot.json`, `rectification_robustness.json`,
  `thesis_reconstruction/realcam_eval.json`, thesis ablation tables, or a
  competitor PDF page:table.

## 2. Format and venue target

`\documentclass[conference]{IEEEtran}`, two column, IEEE conference style
(ICRA / IROS register: the closest siblings in the corpus are LightStereo
ICRA 2025, CoEx IROS 2021, AnyNet ICRA 2019). Target 8 pages of body plus
references. Authors: Nazib Abrar, Md Raihanul Haque Rahi, Md Zunaid Hossen,
Department of Mechatronics Engineering, RUET (the supervisor is a co-author on
the paper; this differs from the thesis book, where the supervisor is not an
author).

Title (proposed): **StereoLite: Real-Time Zero-Shot Stereo Matching at an
Embedded Parameter Budget**.

## 3. Section skeleton with per-section content and length

| # | Section | Len | Content (every number sourced) |
|---|---|---|---|
| 0 | Abstract | 10 sent. | Seven moves: constraint-framed context; gap (models that generalize are large; compact nets collapse off-domain); propose StereoLite 2.96 M; mechanism 1 (slanted tile-plane state from one coarse group-wise volume, refined by ConvGRU, 8 updates, 3 scales); mechanism 2 (narrow-band GEV through a learned fail-soft gate); headline result (D1-all 4.33 / 3.93 / 3.96 / 10.9 on K12 / K15 / ETH3D / MB14; 0.78 px FT3D); deployment (36.3 ms, 27.5 FPS, INT8 Orin Nano, 384x640); rectification tolerance (EPE 1.03 to 1.53 px up to 1 px offset); protocol caveat; code pointer |
| 1 | Introduction | 0.9 p | 4-paragraph funnel: P1 stereo = f B / d, platforms without LiDAR, shared accelerator; P2 lineage 3D-CV to iterative to foundation, concessive turn on size vs generalization, efficient nets reported in-domain; P3 the three borrowed mechanisms (HITNet tile state, RAFT GRU, IGEV GEV) each with its cost, and the open question (does the composition survive an edge budget cross-domain); P4 proposal + 4 contribution bullets (composition + INT8-clean operator set; zero-shot quartet under one protocol; Orin Nano deployment via 6-change 1.74x pass + INT8; rectification tolerance incl. 997-pair real rig 1.45 px vs FoundationStereo). Fig 1 = edge-gap scatter |
| 2 | Related Work | 0.8 p | 4 paragraphs with bold run-in heads: Efficient networks (StereoNet, AnyNet, MADNet, DeepPruner, BGNet, CoEx, MobileStereoNet, HITNet, LightStereo, BANet, Pip-Stereo); Iterative refinement (RAFT-Stereo, CREStereo, IGEV, Selective, GGEV); Tile and plane representations (HITNet); Zero-shot generalization (DSMNet, HVT, FoundationStereo, DEFOM, MonSter, Stereo Anywhere, Fast-FoundationStereo, LiteAnyStereo 3-stage KD). Close each paragraph with the one-clause contrast to StereoLite |
| 3 | Method | 2.2 p | 3.1 Overview + Fig 2 (architecture, figure*) + tiny scale table; 3.2 Encoders (YOLO26s truncated 7 blocks, 128/256/256 ch at 1/4, 1/8, 1/16; 32-ch left context); 3.3 Tile initialization: Eq gwc (G=8, d in 0..23), Eq init soft-argmin; 3.4 Recurrent tile refinement: Eq GRU input, Eq ConvGRU, schedule 2/3/3, local corr 5 offsets +-2 px, 48-ch head, 4 sigmoid gates, Fig 3 (one iteration); 3.5 Plane-aware propagation: Eq prop; 3.6 Narrow-band GEV + fail-soft gate: Eq gev, Eq failsoft, bias -4 (w0 = 0.018), 33 hypotheses +-16, Fig 4; 3.7 Plane rendering + two convex 2x stages; 3.8 Objective: Eq loss (11 terms, weights as in thesis eq:loss), valid 0 < d < 320; 3.9 Training protocol: SceneFlow finalpass 35,454 pairs, 60k steps, batch 32, bf16, AdamW wd 1e-5, OneCycle peak 8e-4, native 384x640 crops from 960x540, 3 augmentations, best at step 53k, one A100 80 GB |
| 4 | Experiments | 3.2 p | 4.1 Setup: metrics Eq (EPE, bad-t, D1), FT3D full test 4,370 pairs native; zero-shot protocol (resize 384x640, GT x sx, mask > 192 px, per-image macro-average, training splits, not leaderboard); latency protocol (batch 1, 384x640, 10 warm-up, 100 timed). 4.2 In-domain (Tab 1 eight metrics; Fig 5 curves; Fig 6 qualitative). 4.3 Zero-shot quartet (Tab 2 master; Fig 7 MB14 per-scene; Fig 8 MB14 qualitative). 4.4 Same-protocol MB14 comparison vs LiteAnyStereo / IGEV (Tab 3) + Fig 9 Pareto. 4.5 Context against published SceneFlow-only zero-shot numbers (Tab 4, standard protocol as reported; our row flagged as closest-equivalent metric; explicit mismatch caveat). 4.6 SceneFlow published methods (Tab 5, indicative only). 4.7 Rectification robustness (Tab 6, Fig 10) + real-rig agreement 997 pairs 1.45 px / 11.1% + Fig 11 camera + reconstruction. 4.8 Deployment: 6-change pass table (Tab 7, 106.7 to 61.4 ms, 1.74x), INT8 operator swaps, Tab 8 latency/memory (61.4 / 49.8 / 36.3 ms). 4.9 Ablations: Tab 9 multi-block (encoder; refinement scheme; recurrent family subset; objective subset; recipe) + Fig 12 ablation bars |
| 5 | Limitations | 0.25 p | LiteAnyStereo-shaped: protocol (training splits, 192 px cap), MB14 gap localized to thin repeated structures and near-cap disparities, ETH3D bad-1 15.2 under our resized protocol, tolerance only to ~1 px offset, real-rig number is agreement not GT, single training run, no occlusion/LR consistency |
| 6 | Conclusion | 0.2 p | Restate composition + four numbers + one reserved forward claim (distillation from a foundation teacher as the next lever, since the deficit is localized) |

## 4. Figures (reuse thesis vector assets; one new figure)

| Fig | Source file (thesis/book/figures) | Width | Caption core |
|---|---|---|---|
| 1 | `fig_1_3_edge_gap.pdf` | column | Params (log) vs SceneFlow EPE with edge budget shaded; StereoLite marker |
| 2 | `fig_3_1_architecture_preview.png` (1760x900) | figure* | Five stages; supervised outputs marked |
| 3 | `fig_3_4_refinement_preview.png` (1275x816) | column | One recurrent iteration: warp, local correlation, gated residuals |
| 4 | `fig_3_6_gev_fusion_preview.png` (1171x290) | column | Narrow-band GEV fused through fail-soft gate |
| 5 | `fig_4_1_training_curves.pdf` | column | Loss + val EPE over 60k steps, floor at 53k |
| 6 | `fig_4_3_sceneflow_qualitative.pdf` | figure* | Left / GT / prediction / error map, FT3D held-out |
| 7 | `fig_4_4_mb14_perscene.pdf` | column | Per-scene D1-all sorted; 16/23 inside the reference band |
| 8 | `fig_4_5_mb14_qualitative.pdf` (tall) | column | Easiest and hardest MB14 scenes |
| 9 | `fig_4_8_pareto_ours.pdf` | column | MB14 D1-all vs params, three zero-shot models |
| 10 | `fig_4_9_rectification.pdf` | column | EPE and D1-all vs vertical offset |
| 11 | `fig_4_6_camera.pdf` + `fig_4_10_reconstruction.pdf` | figure* (two subfigures) | Real rig disparity + metric point clouds |
| 12 | `fig_4_7_ablations.pdf` | figure* or column | Augmentation, efficiency, blur fixes, input protocol |
| NEW | zero-shot quartet bar figure (matplotlib, generated from the two JSONs) | column | EPE / bad-2 / D1-all per dataset; makes the "KITTI and ETH3D sit near in-domain, MB14 is the weak axis" point visually |

Not possible without a GPU/Modal environment on this machine (no venv, no
torch, no modal CLI at the moment): a four-domain qualitative teaser with
KITTI and ETH3D predictions (the eval driver does not save images). Listed
as an optional upgrade; the paper does not depend on it.

## 5. Tables (all values already verified)

| Tab | Content | Source |
|---|---|---|
| 1 | FT3D full test, 8 metrics: 0.781 / 3.64 / 0.130 / 15.32 / 8.92 / 5.34 / 4.00 / 3.40 | `meta.json:final_metrics_all` |
| 2 | Zero-shot quartet, one checkpoint, our protocol: K12 194 pairs EPE 0.823 bad-1 16.53 bad-2 6.96 bad-3 4.34 D1 4.33; K15 200 pairs 0.823 / 17.99 / 6.42 / 3.93 / 3.93; ETH3D 27 pairs 0.930 / 15.18 / 6.47 / 3.96 / 3.96; MB14 23 scenes 1.71 / 23.9 / 14.5 / 11.2 / 10.9 | `kitti_eth3d_zero_shot.json`, `mb14_zero_shot.json` |
| 3 | MB14 identical protocol: ours 2.96 M / 86 ms / 1.71 / 10.9; LiteAnyStereo 7.60 M / 64 ms / 1.17 / 6.9; IGEV 16 it. 12.60 M / 305 ms / 0.86 / 5.0 (full bad-0.5..bad-3 columns) | thesis tab:mb14_comparison (our re-evaluation) |
| 4 | Published SceneFlow-only zero-shot context (standard thresholds, as reported): RAFT-Stereo (K15 5.74, MB q 9.36, ETH3D 3.28; RAFT Tab 1 p5), IGEV (MB q 6.2, ETH3D 3.6; IGEV Tab 7 p7), GGEV and RT-IGEV, BGNet+, CoEx, DeepPrunerFast rows (GGEV Tab 1 p5), LightStereo-S (K12 11.6, K15 9.0, MB half 19.63; Tab VII p6), LiteAnyStereo SF-only (K12 5.45, K15 6.45, ETH3D 15.38, MB 13.13; Tab 3 p6), HITNet SF-only (K12 6.44, K15 6.49; supp Tab 5 p15). Our closest-equivalent row: K12 bad-3 4.34, K15 bad-3 3.93, MB14 bad-2 14.5 at 640-px width (about quarter resolution), ETH3D bad-1 15.18. Caption states the protocol mismatch explicitly | reader reports; to be re-verified against PDFs in the reference-check pass |
| 5 | SceneFlow published methods, params / EPE / latency / hardware (PSMNet, HITNet L, CoEx, RAFT-Stereo, IGEV, LightStereo-S, FoundationStereo, ours 0.78 at 49.8 ms fp16 RTX 3050) | `papers/verified_performance.md` |
| 6 | Rectification sweep 0 / 0.5 / 1 / 2 / 4 px: EPE 1.03 / 1.22 / 1.53 / 2.43 / 5.07; D1 4.29 / 4.60 / 6.19 / 15.83 / 41.17 (+ bad-1, bad-2) | `rectification_robustness.json` |
| 7 | Six-change optimization pass with saving and equivalence class; combined 106.7 to 61.4 ms (1.74x) | thesis tab:efficiency_findings |
| 8 | Latency and memory: RTX 3050 fp32 61.4 ms / 0.35 GB; fp16 49.8 / 0.26; Orin Nano INT8 TensorRT 36.3 / 27.5 FPS / ~0.15 GB (estimated) | thesis tab:latency |
| 9 | Ablations, four blocks: (a) encoder GhostConv 0.538 M 0.625 / YOLO26n 0.808 M 0.712 / YOLO26s 2.061 M 0.528; (b) refinement: iterated 0.578, + 1/2 scale 0.573 at 32.5 ms, single pass 0.844; (c) recurrent family: base 0.424, gate + YOLO26s 0.319, + SRU 0.474, + raw 1/4 init 0.455, + GEV fail-soft 0.261 (engineering record, budgets differ); (d) objective: L1 0.663, L1+grad 0.618, seq-weighted 1.005, stack 0.672, stack+D1 0.591; (e) recipe: augmentation 2.778 to 1.921, native crop 6.672 to 2.869 native-axis, plane rendering bad-0.5 55.36 to 42.91, efficiency pass 46.6 to 30.2 ms at EPE 2.908 to 2.906 | thesis tab:ablation_* |

## 6. Equations (9, all from thesis Ch3/Ch4, auto-numbered, `\eqref` only)

gwc, init (soft-argmin + confidence), GRU input, ConvGRU, plane propagation,
GEV expectation, fail-soft fusion, loss, metrics. Each introduced by a purpose
clause and followed by a `where` sentence (corpus rule).

## 7. References

Start from `paper/refs.bib` (34 PDF-verified entries). Add and verify against
`papers/raw/` first pages: GGEV (Liu, AAAI 2026), Fast-FoundationStereo (Wen,
CVPR 2026), OpenStereo (Guo et al., arXiv 2023; augmentation recipe), ACVNet /
Fast-ACVNet only if a row uses it, IGEV++ only if the RT-IGEV row is kept,
Waveshare AR0144 and NVIDIA Orin Nano as `@misc` web references. Reference
check pass (sonnet agent): open every cited PDF, confirm title / venue / year /
authors on page 1, and confirm every competitor number in Tab 3, 4, 5 at its
page:table. Drop any row whose number cannot be re-verified.

## 8. Pipeline after the plan

1. Assemble `paper/figures/` (copy the 12 thesis assets; generate the new
   quartet figure with matplotlib from the two JSONs).
2. Write `paper/main.tex` from scratch following Sections 3 to 6 above,
   drafting each prose unit against `patterns.md` templates (abstract 7
   moves, intro funnel, contribution bullets with We + verb, method in flat
   present tense, results in past tense with table pointers, captions as
   bold noun phrase + protocol).
3. Compile (pdflatex, bibtex, pdflatex x2) from `paper/`; check undefined
   refs = 0, page count, orphan pages (< 60 words), float placement.
4. Humanize pass with the `humanizer` skill over the full prose; re-check
   dashes, AI vocabulary, rule-of-three padding, and that no number changed.
5. Reference-check pass (agent) as in Section 7; fix or drop.
6. Final compile; commit incrementally at each verified milestone
   (figures, draft compiles, humanized, references verified).
