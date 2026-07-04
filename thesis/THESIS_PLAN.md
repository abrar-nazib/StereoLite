# Thesis Plan — "Edge-Deployable Deep Stereo Matching" (yolo_ctx_gev4)

Working plan for the RUET MTE B.Sc. thesis book. Authors: Nazib Abrar
(2008026), Md. Raihanul Haque Rahi (2008011). Supervisor: Md Zunaid Hossen,
Lecturer, Dept. of Mechatronics Engineering. Format authority:
`.claude/skills/ruet-thesis-expert/SKILL.md`. Branch: `thesis`.

Central claim to defend: **runs on edge hardware and provides usable results.**

## PRODUCTION RUN COMPLETE (2026-07-05) — thesis checkpoint acquired

The native-crop full Scene Flow run `20260704_fullsf_gev4onp_nc` finished all
60,000 steps. This is the thesis checkpoint (critical-path item A1, now done).

**Publishable headline number (report this, NOT the val-subset EPE):**
full FlyingThings3D-TEST, all 4,370 pairs, native 960x540 pad16 axis, on
`best.pth` (step 53,000):

| EPE | bad-1.0 | bad-2.0 | bad-3.0 | D1-all | RMSE | median AE |
|---|---|---|---|---|---|---|
| **0.7807** | 8.92% | 5.34% | **4.00%** | **3.40%** | 3.64 | 0.13 |

- 4 degenerate frames (zero valid pixels: all GT > 192 px) excluded via nanmean.
- best.pth = step 53k, not 60k: LR floored (3.2e-9) and val flat at 0.80 to 0.81
  for the last ~7k steps, so the cosine tail gave diminishing returns.
  **Architecture is frozen for the thesis; more steps would not help.**
- The 400-pair val-subset best was 0.7896; the full test came in slightly better
  (0.7807), so there is no negative selection surprise. The val number is a
  model-selection statistic and belongs only in the training-curve discussion.
- Model: 2.9631 M params. Config: `gev4_opt_narrow_plane --slant_w 0.3`,
  native_crop input, OneCycle 60k, bf16, batch 32, A100-80GB.

**Where the data lives (nothing lost):**
- Modal volume `widener-results:/fulltrain/20260704_fullsf_gev4onp_nc/`
  (best.pth, latest.pth, checkpoints/step_*.pth x60, images/, train.csv, meta.json).
- Local mirror `model/benchmarks/20260704_fullsf_gev4onp_nc/` (3.8 GB, gitignored):
  best.pth, latest.pth, 60 per-1k checkpoints, 100 tracked-image folders
  (each: gt.png + left.png + info.json + step_XXXXXX.{png,json} with 8-metric
  JSON per 1k), train.csv, viz/, meta.json (`final_metrics_all` written).
- Final number reproducible via `model/scripts/modal/eval_full_testset.py`.

**Incident + fix (2026-07-05):** the in-run final 4,370-pair eval OOM-stalled a
64 GB container by decoding all pairs at once (~70 GB), so it hung after step
60k and never wrote `final_metrics_all` (the "hang"). Stopped the app; the
number was produced by a streaming eval (`modal/eval_full_testset.py`). Both the
trainer's final block and the standalone driver now stream shard-by-shard and
use nanmean; the bug cannot recur.

**Remaining post-run gates (unchanged):** MB14 zero-shot on this checkpoint
(mandatory), resized-checkpoint native-inference ablation, real-camera panels,
RTX 3050 latency bench, Jetson Orin Nano reading when the borrowed board arrives.

---

**State of play (rewritten 2026-07-04, updated same day).** All ablation
gates are closed. The working architecture is **`gev4_opt_narrow_plane`**
(efficiency-optimized gev4 + narrow GEV + plane-equation rendering with gated
slant supervision) and the full-training recipe is locked (Section 3b).
Rahi's full checkpoint is NOT coming; the Modal full Scene Flow training run
is the critical path that produces the thesis checkpoint. Everything
downstream (MB14 zero-shot, camera panels, baseline table, Ch4 numbers)
waits on that one run.

**Input-protocol correction (2026-07-04).** The first full run
(`20260703_fullsf_gev4onp`, legacy 640x384 global-downscale input) reached
val EPE 0.7630 at step 32k before a Modal preemption that crossed midnight
UTC restarted it into a fresh out_dir (date-stamped run-name bug, now fixed:
run_name is a replayed function input). In parallel, the
`20260704_native_vs_resize_n500` ablation showed the downscale protocol
cripples native-resolution inference (control at native axis: EPE 6.67, D1
35.8% vs native_crop 2.87 / 16.5%, with zero cost on the resized axis).
Verdict: the resized run is NOT completed or relaunched. The thesis
checkpoint comes from a **native-crop rerun** (Section 3b). The salvaged
resized-protocol model is kept as an ablation asset:

- `model/checkpoints/fullsf_gev4onp_best_ep32k_epe0763.pth` (best, step 32k,
  resized-axis val EPE 0.7630) and `fullsf_gev4onp_latest_ep38k.pth`
  (resumable, step 38k).
- **Planned Ch4 ablation:** run native-resolution (960x540 pad16) inference
  with this resized-trained checkpoint and compare against the native-crop
  full checkpoint on the same 400-pair FT3D-TEST subset plus MB14. This
  scales the n500 protocol finding ("resize training cannot serve native
  inference") to full-data evidence, and doubles as the justification for
  the input-protocol choice in Ch3.

---

## 1. What we already have (asset inventory, verified on disk)

### Evidence / data
| Asset | Location | Serves |
|---|---|---|
| Full 89k-step Scene Flow Driving training run (4200 train / 200 val): final val EPE 0.963 px, D1 7.88%, 2.9623 M params, RTX 4070 | `model/checkpoints/yolo_ctx_gev4_full_retry_es/{meta.json,train.csv}` | Ch4 core result |
| 20-pair overfit reference: EPE 0.261, 24.6 ms / 40.7 FPS RTX 4070 | `model/benchmarks/yolo_ctx_gev4_yolo26s/` | Ch4 capacity check |
| ~10-variant architecture ablation story (gate/gev4/guided/sharp/hrrefine/sru/init4/geomctx) | `model/benchmarks/EXPERIMENTS.md` | Ch4 design justification (PO4) |
| Efficiency A/B (gev4 vs opt vs opt_narrow, 80/20, equivalence-proven 1.74x fp32) | `model/benchmarks/20260703_eff_gev4_n100/` (see `GRAND_COMPARISON_20260703.md`) | Ch3 optimization section (PO3/PO5) |
| 9-arm grand comparison: augmentation triplet -30.8% val EPE (largest lever measured), freeze_bn rejected, pre-rahi costlookup closed out | `model/benchmarks/GRAND_COMPARISON_20260703.md` | Ch4 training-recipe justification |
| Blur study: 2 root causes diagnosed (decode low-pass + L1/soft-argmin mean regression), literature-grounded fixes, corpus report | `docs/deblurring_plan.md` | Ch4 discussion (PO2/PO4) |
| Blur-fix A/B on 500-pair leak-proof windowed split: bundle1 -9.8% bad-0.5, **plane -22.5% (ADOPTED)**, bimodal -21.9%/best D1 | `model/benchmarks/20260703_blurfix_n500/` | Ch4 boundary-sharpness section |
| Composition A/B (pb / allin): REJECTED — neither beat plane alone on bad-0.5 (44.4 / 46.6 vs 42.9; criterion < 39.5 unmet); fixes anti-synergize | `model/benchmarks/20260703_blurfix_compose/` | Ch4 negative-result evidence (PO4) |
| OpenStereo findings report (StereoBase = IGEV + recipe; aug triplet provenance) | `docs/openstereo_findings.md` | Ch2/Ch4 |
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

### A. MUST-HAVE for the edge claim — status as of 2026-07-03 (user-instructed
### acquisition strategy: calculated edge values NOW marked with an asterisk (*),
### swapped for real readings when the borrowed Jetson arrives)

| # | Item | Status |
|---|---|---|
| A1 | Trained gev4 checkpoint | **NOT coming from Rahi** (his `checkpoint.pth` is the 20-pair overfit; pipeline validation only). **WE TRAIN IT** — Modal A100 full Scene Flow run with the locked config (Section 3b). ~$10-15, ~1 day wall clock. This is the single critical-path item |
| A2 | Edge-device latency | **Calculated* now, real later.** RTX 3050 MEASURED (2026-07-03, batch 1, 384x640, eager PyTorch): gev4 fp32 106.7 / fp16 75.4 ms; **gev4_opt_narrow fp32 61.4 / fp16 49.8 ms (1.74x)**; plane variant ~62.5 ms fp32 (rendering overhead ~1 ms). Orin Nano projection below. Real Jetson readings when the user borrows the device; every projected number carries * in the thesis until swapped |
| A3 | MB14 zero-shot on the trained checkpoint | Run on Modal immediately after A1 finishes; driver adaptation (legacy import at eval_middlebury2014.py:61) can be prepared during training |
| A4 | Real-camera qualitative panels | **Pipeline VALIDATED** (2026-07-03): gev4 runs end-to-end on `/media/abrar/AbrarSSD/Datasets/user_cam_1/` (60 indoor pairs + FoundationStereo pseudo-GT reference); smoke panels in `model/benchmarks/gev4_camera_smoke/`. Regenerate with the trained checkpoint, same script |
| A5 | Matched-protocol baseline table | After A3; IGEV + LiteAnyStereo reference JSONs already exist on the results volume; add LightStereo-S from the OpenStereo zoo |
| A6 | Inference memory | **DONE**: 0.26 GB (fp16) / 0.35 GB (fp32) measured on 3050 — replaces the misleading 7.6 GB training peak |

### Jetson Orin Nano projection (*calculated 2026-07-03 — replace with real readings)

Measured anchor: fp16 eager PyTorch on RTX 3050 Laptop = 74.7 ms.
Method: (i) eager -> TensorRT engine speedup for a launch-bound model,
conservative x1.8; (ii) fp16 -> INT8 x1.2; (iii) device penalty = mean of
compute ratio (3050 fp16 ~17.5 TFLOPS vs Orin Nano INT8 dense 20 TOPS ~ x0.9)
and bandwidth ratio (192 vs 68 GB/s, halved traffic at INT8 ~ x1.4) = x1.15.
74.7 / 1.8 / 1.2 x 1.15 ~= 40 ms.

| Quantity | Value (asterisked in thesis) |
|---|---|
| Orin Nano INT8 TensorRT latency, 384x640 (original gev4, anchor 74.7 ms fp16) | **~45 ms* (range 35-60 ms*)** |
| gev4_opt (safe fixes, MEASURED 3050 fp16 57.9 ms) | **~35 ms* (~29 FPS*)** |
| **gev4_opt_narrow (MEASURED 3050 fp16 49.8 ms / fp32 61.4 ms, 1.74x)** | **~30 ms* (~33 FPS*) — AT the real-time target** |
| Inference memory (INT8) | **~0.12-0.15 GB*** |
| Power envelope | 7-15 W (device spec, not projection) |

Write these into Ch4 as calculated estimates with the methodology sentence and
the asterisk convention; a table footnote states real measurements replace
them when hardware is available.

### 3b. Ablation verdicts + LOCKED full-training config (all gates closed 2026-07-04)

Chronology of evidence (all under the ablation-study-expert protocol, all in
`EXPERIMENTS.md`):

1. **Efficiency A/B** (eff_gev4_n100): safe fixes accuracy-equivalent
   (+0.45% best-val EPE, noise); narrow GEV holds accuracy (-2.0%, noise)
   at 1.74x fp32. ADOPTED gev4_opt_narrow. The earlier bad-0.5 watch item
   is superseded by the plane fix below.
2. **Grand 9-arm** (GRAND_COMPARISON_20260703.md, pair-set hash-verified
   across 4 runs): OpenStereo augmentation triplet = largest single lever
   in project history (best val EPE 2.778 -> 1.921, -30.8%, 6x noise floor,
   D1 -22.6%, train EPE HIGHER = regularization signature). freeze_bn: no
   effect alone, above-noise WORSE with aug — rejected. Pre-rahi costlookup
   8-11% behind gev4 family — closed (its plane-upsample sharpness insight
   was harvested into the plane fix). Sharptail hybrid rejected.
3. **Blur-fix round 1+2** (20260703_blurfix_n500, 500-pair leak-proof
   windowed split, control bad-0.5 = 55.4): bundle1 -9.8% bad-0.5;
   **plane rendering + gated slant supervision -22.5% (42.9) — decisive,
   sharpest collages of the project**; bimodal aux head -21.9% + best D1
   via training signal alone.
4. **Composition round** (20260703_blurfix_compose): pb (plane+bimodal)
   bad-0.5 44.4, allin (+bundle1) 46.6 — both WORSE than plane alone;
   pre-registered win criterion (< 39.5) unmet. Fixes anti-synergize
   (compete for the same boundary pixels). **ADOPT plane alone**; bimodal
   and bundle1 dropped. Negative result goes in Ch4.

5. **Native-vs-resize input protocol** (20260704_native_vs_resize_n500,
   L40S, 3 arms, dual-axis final eval): random native 384x640 crops beat
   the legacy global downscale by -32% bad-0.5 / -37% bad-1 / -57% EPE on
   the native axis while tying it on the resized axis; whole-native-frame
   training loses on the resized axis and is ~3x slower per step. Crop
   protocol verified against published practice (OpenStereo RandomCrop,
   RAFT/IGEV augmentor): co-located windows in both views, no horizontal
   right-image shift exists in any surveyed method; vertical jitter of the
   right crop (+-2 px) is a RAFT-family default but OFF in OpenStereo
   SceneFlow configs and OFF in our validated arm.

**LOCKED FULL-TRAINING CONFIG (every element evidence-backed):**

| Element | Choice | Evidence |
|---|---|---|
| Architecture | `gev4_opt_narrow_plane` (--slant_w 0.3) | items 1 + 3 |
| Augmentation | OpenStereo triplet ON | item 2 |
| freeze_bn | OFF | item 2 |
| Input protocol | random native 384x640 co-located crops (`--input_mode native_crop`); val axis = native 960x540 pad16 with a resized-axis eval logged alongside | item 5 |
| Data | full Scene Flow (~35k pairs), held-out val | scale-up from n500 protocol |
| Schedule | OneCycle, 60k steps, val every 1k | item 2 + 20260703 run trajectory (best at 32k of 40k under resize; native crops see more pixels per epoch) |
| Data preservation | per-1k model checkpoints (`checkpoints/step_*.pth`) + best/latest separate; 100 tracked images (50 train + 50 val) with per-image uint16 disparity PNG + full 8-metric JSON per eval | 2026-07-04 preemption incident: never lose training state again |
| GPU | A100 (full-data training per project rule; T4/L40S are for ablations) | Modal rules |
| Input contract | [0,1] at the model boundary (F6 landmine settled) | Section 3 F6 |
| Post-run gates | MB14 zero-shot (mandatory), resized-checkpoint native-inference ablation, camera panels, 3050 bench | CLAUDE.md lesson + input-protocol correction above |

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
| F3 | GEV narrows from full 64 bins to ~±16 around tile.d (validated cascade_cv_4 pattern) | **~25-30% of total MACs, ~100 MB activations** | **VALIDATED** (matched A/B: -2.0% best-val EPE, within noise; MB14 re-check rides on the full-training checkpoint) |
| F6 | **Normalization contract landmine**: overfit harness feeds [0,1], full trainer feeds [0,255]; gev4 ctx stream double-divides (stem runs at GroupNorm's eps floor); checkpoints not portable across pipelines; INT8 hazard | correctness, not speed | settle ONE contract before any retrain/export |
| F8 | Export blockers: F.unfold in ConvexUpsample (replace with 9 shifted slices — identical), GroupNorm on NPU toolchains, Conv3d on NPUs | export-time | flag now |

**Thesis-shaping insight:** F1-F7 are now all implemented, equivalence-proven,
and measured (1.74x fp32 on the 3050). Measured again before/after ON THE EDGE
DEVICE, they become a Ch3/Ch4 section ("optimization of the network for edge
inference") that directly evidences PO3/PO4/PO5 — the optimization work IS
thesis content, not a detour. The blur study (diagnose -> literature ->
controlled fix -> composition negative result) is a second such section and
the most contribution-shaped piece of the thesis.

---

## 4. Writing plan (phases; ~3-4 weeks calendar)

### Phase 0 — unblock (status 2026-07-04)
1. ~~Efficiency pass (test-first)~~ **DONE** — F1-F7 implemented,
   equivalence-proven, adopted (gev4_opt_narrow).
2. ~~Ablation gates~~ **DONE** — aug/freeze_bn/blur-fix/composition all
   closed; config locked (Section 3b).
3. Ask **supervisor** about similarity + AI report process (B10) — external
   clock, START NOW. Jetson: user borrowing from a friend; calculated*
   values stand in until then.
4. Create `thesis/book/` from the template with the resolved format (B1).

### Phase 1 — evidence sprint (week 1; parallel with Phase 2 writing)
5. **LAUNCH the full Scene Flow training run** (A1, locked config, Modal
   A100, date_tag run folder). THE critical-path item; everything in Ch4
   waits on it. Prepare + dry-run the trainer before burning A100 hours
   (input contract F6, checkpoint/resume, val cadence, tracked images).
   Status 2026-07-04: first attempt (resize protocol) preempted at 38k,
   best 0.7630@32k salvaged; native-crop relaunch is the production run
   (60k steps, per-1k checkpoints + tracked-image archive).
6. While it trains: adapt the MB14 driver for gev4_opt_narrow_plane (A3);
   scaffold the baseline table (A5, incl. LightStereo-S).
7. When the checkpoint lands: MB14 zero-shot (mandatory gate) → baseline
   table → regenerate camera panels (A4) → 3050 latency re-bench of the
   trained model.
8. Real Jetson readings when the borrowed device arrives; swap all * values.
9. Optional stretch: C6 point-cloud figure from the camera panels.

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
1. **Full-training run fails or underdelivers** (the n500 protocol trains at
   val EPE ~1.8; full data + longer schedule must land near the 0.963 px of
   the 89k-step reference run or better) → budget one relaunch; keep the
   reference run's numbers as the documented floor.
2. **gev4 collapses on MB14** (prior chassis did) → narrative shifts to
   "in-domain + real-camera usable; cross-domain identified and analyzed as
   limitation with the KD path forward". Still a defensible thesis; know early.
   Mitigation already in place: augmentation triplet is the single strongest
   known generalization lever.
3. **No edge hardware** → fallback wording locked before Ch1; C3 INT8/ONNX
   study partially substitutes.
4. **AI report friction** → disclosure is mandatory; start the conversation
   with the supervisor now, not at submission.
5. **Anti-synergy resurfaces at scale** (plane fix validated at n500, not
   35k) → the full trainer logs bad-0.5 at every val; if the plane arm's
   edge disappears, the control config is one flag away.
