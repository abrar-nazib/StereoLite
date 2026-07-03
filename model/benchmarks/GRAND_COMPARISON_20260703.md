# Grand comparison, 2026-07-03: efficiency / pre-rahi / recipe / sharptail (n=100 overfit)

Nine arms across four runs, all on the SAME 100-pair Scene Flow Driving set
(80 train / 20 val, seed 42, 384x640, batch 8, AdamW lr 2e-4,
loss `msL1{1,.5,.3,.2,.1}+0.5grad+0.2bad1+0.15gev4`, eval every 500,
plateau stop = patience 4, max 12000 steps, harness
`overfit_efficiency_ablation`).

**Pair-set equality: VERIFIED.** sha256 of `pair_paths` (sorted JSON) is
`7b3ef1fb055cbd19` (n=100) in all nine meta.json files. Accuracy numbers are
directly comparable.

> **WARNING (cross-GPU latency):** `prerahi_n100` trained/benched on
> **NVIDIA A10G**; the other three runs on **NVIDIA A10**. The "cloud ms"
> column is NOT comparable between the prerahi group and the rest. Use the
> RTX 3050 local column (all five architectures measured on the same local
> 3050) for cross-family latency. The sharptail 3050 number (82.3 ms fp32)
> was measured in-session and is not backed by a json file on disk; treat as
> provisional.

**Noise floor** (control's own post-plateau val-EPE oscillation, from
train.csv):

- `gev4` (control for Q-a/b): post-best samples 2.867, 2.869, 2.879, 2.908
  around best 2.834; band width 0.074 px, about 2.6% relative.
- `gev4_opt_narrow` (control for Q-c/d/e): post-best samples 2.822, 2.839,
  2.915, 2.906 around best 2.778; band width 0.137 px, about 4.9% relative.

So the working threshold is the skill's ~5% relative rule, empirically
confirmed at 0.07 to 0.14 px EPE on this val split.

## The 9-arm table

Val metrics = `final_metrics_all` (20-pair val split at the stopped step).
Best val EPE = `plateau.best_val_epe`. Cloud ms = `latency_ms.mean` from
meta.json (A10 except prerahi = A10G). 3050 ms from
`eff_gev4_n100/rtx3050_latency.json` and `prerahi_n100/rtx3050_latency.json`;
recipe arms share `gev4_opt_narrow` 3050 latency (identical inference
architecture).

| Q | Arm | Params M | Best val EPE (step) | Final val EPE | RMSE | MedAE | bad-0.5 | bad-1 | bad-2 | bad-3 | D1-all | Final train EPE | Stop step | Cloud ms | 3050 fp32/fp16 ms |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| a | **gev4 (control, eff)** | 2.962 | 2.834 (3000) | 2.908 | 7.130 | 0.485 | 43.12 | 30.32 | 21.12 | 16.68 | 16.31 | 1.228 | 5000 | 46.6 (A10) | 106.7 / 75.4 |
| a | gev4_opt | 2.962 | 2.847 (3500) | 3.021 | 7.215 | 0.510 | 44.15 | 32.09 | 22.74 | 18.17 | 17.80 | 1.193 | 5500 | 35.9 (A10) | 83.2 / 57.9 |
| b | gev4_opt_narrow | 2.962 | 2.778 (4500) | 2.906 | 6.973 | 0.528 | 46.22 | 32.29 | 22.22 | 17.31 | 16.88 | 1.209 | 6500 | 30.2 (A10) | 61.4 / 49.8 |
| c | costlookup_y26n | 1.327 | 3.000 (2500) | 3.063 | 7.247 | 0.533 | 45.16 | 31.42 | 21.90 | 17.45 | 17.09 | 1.141 | 4500 | 44.9 (A10G) | 66.4 / 55.5 |
| c | costlookup_y26s | 2.209 | 3.095 (2500) | 3.065 | 7.248 | 0.547 | 45.42 | 31.95 | 21.83 | 17.25 | 16.98 | 1.168 | 4500 | 38.9 (A10G) | 66.9 / 57.6 |
| d | sharptail (gev4_opt_narrow + costlookup tail) | 3.103 | 2.862 (2000) | 2.868 | 6.973 | 0.509 | 45.31 | 32.03 | 22.11 | 17.37 | 17.03 | 1.364 | 4000 | 39.0 (A10) | 82.3 (in-session) / n.m. |
| e | recipe: aug | 2.962 | 1.921 (11000) | 2.012 | 5.019 | 0.464 | 44.54 | 29.48 | 18.59 | 13.88 | 13.07 | 1.294 | 12000 (no plateau) | 31.2 (A10) | 61.4 / 49.8 (shared) |
| e | recipe: freeze_bn | 2.958 | 2.774 (2500) | 2.834 | 6.867 | 0.514 | 45.67 | 32.09 | 22.02 | 17.05 | 16.64 | 1.176 | 4500 | 29.0 (A10) | 61.4 / 49.8 (shared) |
| e | recipe: aug_freeze_bn | 2.958 | 2.153 (5500) | 2.138 | 5.336 | 0.466 | 44.57 | 28.94 | 18.88 | 14.37 | 13.63 | 1.366 | 7500 | 31.8 (A10) | 61.4 / 49.8 (shared) |

Footnotes:
- freeze_bn arms show 2.9579 M trainable vs 2.9623 M: the 4.4 K delta is
  the frozen BN affine parameters; inference architecture is unchanged.
- gev4 / gev4_opt / gev4_opt_narrow report identical param counts: the
  opt (static context) and narrow-GEV changes are compute-path changes,
  not parameter changes.
- Controls per question: (a, b) = `gev4` and chain; (c, d, e) =
  `eff_gev4_n100/gev4_opt_narrow`.

## Verdicts

### (a) Safe efficiency fixes (gev4_opt vs gev4): SETTLED, EQUIVALENT

One knob: `opt_static_ctx=true`. Best val EPE 2.847 vs 2.834
(+0.013, +0.4%, deep inside the 2.6% control noise band). Final-step
metrics read slightly worse (final EPE +3.9%, D1 +1.49 pp) but these are
single post-plateau tail samples at different stop steps; the deciding
metric (best val EPE) is a wash. Latency: cloud 35.9 vs 46.6 ms
(-23%), 3050 fp32 83.2 vs 106.7 (-22%), fp16 57.9 vs 75.4 (-23%).
**Verdict: equivalent accuracy, unambiguous speed win. Adopt.**

### (b) Narrow GEV (gev4_opt_narrow vs gev4_opt): HOLDS

One knob: `narrow_gev=true` (gev_half_range 16). Best val EPE 2.778 vs
2.847 (-0.069, -2.4%, within noise, nominally better). Final val EPE
-3.8%, RMSE -3.4%, all within noise. One watch item: final bad-0.5
46.22 vs 44.15 (+2.07 pp, +4.7% relative, at the noise edge) and MedAE
+0.018; a mild sub-pixel drift that should be re-checked under full
training. Latency: cloud -16%, 3050 fp32 61.4 vs 83.2 (-26%), fp16
-14%. **Verdict: holds. No above-noise accuracy cost, large latency
win. `gev4_opt_narrow` is the efficiency-track chassis.**

### (c) Pre-rahi costlookup vs gev4 family: gev4 keeps the accuracy lead; costlookup keeps the param budget

Deltas vs `gev4_opt_narrow` (control):

| Arm | d best val EPE | d final EPE | d bad-0.5 | d bad-1 | d bad-3 | d D1 | d params |
|---|---|---|---|---|---|---|---|
| costlookup_y26n | +0.222 (+8.0%) | +0.158 (+5.4%) | -1.06 pp (-2.3%) | -0.87 pp (-2.7%) | +0.14 pp | +0.21 pp | -1.64 M (-55%) |
| costlookup_y26s | +0.317 (+11.4%) | +0.159 (+5.5%) | -0.80 pp (-1.7%) | -0.34 pp | -0.06 pp | +0.10 pp | -0.75 M (-25%) |

- On the deciding metric (best val EPE) both costlookup arms are worse by
  more than the 5% noise band: **the gev4 family accuracy lead is real.**
- costlookup does win final bad-0.5 (and bad-1) against gev4_opt_narrow
  by about 1 pp, but that is 2 to 3% relative, **within noise**: a
  sharpness hint, not a validated win. Against base gev4 (43.12 bad-0.5)
  costlookup loses. Mixed at best.
- Within costlookup, **y26n dominates y26s**: better best val EPE (3.000
  vs 3.095) at 40% fewer params. y26s buys nothing here.
- Latency (3050, same device): costlookup_y26n 66.4/55.5 vs
  gev4_opt_narrow 61.4/49.8; slightly slower despite 55% fewer params.
  Cloud numbers are cross-GPU (A10G vs A10), not compared.

**Verdict: gev4_opt_narrow wins accuracy above noise at comparable local
latency. costlookup_y26n stays alive only as the sub-1.5 M param-budget
option (+8% EPE cost). costlookup_y26s: dominated, drop.**

### (d) Sharptail hybrid: REJECTED

One knob: `sharp_tail=true` (+ its `iters_2=2`) on gev4_opt_narrow.
Best val EPE 2.862 vs 2.778 (+3.0%, within noise, nominally worse), and
it plateaued at step 2000 vs the control's 4500: the tail added
capacity that the optimizer stopped exploiting almost immediately.
Final-step metrics are mixed within noise (final EPE -1.3%, D1 +0.15 pp,
bad-0.5 -0.91 pp). Costs are strictly worse: params +0.141 M (+4.8%),
cloud latency 39.0 vs 30.2 ms (+29%), 3050 fp32 82.3 (in-session) vs
61.4 (+34%). Under the matched-budget rule, a within-noise accuracy
result purchased at +29 to 34% latency is a loss.
**Verdict: rejected. (Note the honest reading is "no above-noise gain at
strictly higher cost", not "worse on every metric"; the final-step
deltas are mixed noise.)**

### (e) THE RECIPE RESULT: aug is a decisive above-noise win; freeze_bn is a no-op alone and a drag in composition

Deltas vs `gev4_opt_narrow` (control, best 2.778 / final 2.906):

| Arm | Best val EPE | d vs control | Final val EPE | Final D1 | Final bad-3 | Final train EPE | Stop |
|---|---|---|---|---|---|---|---|
| control | 2.778 | 0 | 2.906 | 16.88 | 17.31 | 1.209 | 6500 |
| aug | **1.921** | **-0.857 (-30.8%)** | **2.012** | **13.07 (-3.81 pp)** | **13.88 (-3.43 pp)** | 1.294 (+7%) | 12000, never plateaued |
| freeze_bn | 2.774 | -0.004 (-0.15%) | 2.834 | 16.64 | 17.05 | 1.176 | 4500 |
| aug_freeze_bn | 2.153 | -0.625 (-22.5%) | 2.138 | 13.63 | 14.37 | 1.366 | 7500 |

- **aug**: -30.8% best val EPE, six times the noise floor. The gain is
  concentrated in outlier metrics (D1 -22.6%, bad-3 -19.8%, bad-2 -16.3%,
  RMSE -28%) while sub-pixel bad-0.5 moves only -3.6% (within noise).
  Final **train** EPE is HIGHER than control (1.294 vs 1.209): classic
  regularization signature, aug trades memorization for val
  generalization. It ran all 12000 steps without triggering the plateau
  patience and set its best at step 11000: **the curve had not
  flattened; the true aug ceiling is below 1.92.**
- **freeze_bn alone**: best 2.774 vs 2.778, -0.15%. No effect. Same
  plateau behaviour as control.
- **aug_freeze_bn**: best 2.153, better than control by -22.5% but
  **worse than aug alone by +0.232 (+12.0%, above noise)**, and it
  plateaued at 5500 where aug was still improving at 11000. Freezing BN
  denies the network the statistics adaptation that aug's input
  distribution shift apparently needs.
- **Recommendation for the full Scene Flow training config: aug ON,
  freeze_bn OFF.** Budget extra steps; with aug the plateau detector
  needs a longer horizon.

## Adoption list (full Scene Flow training config)

1. **Chassis: `gev4_opt_narrow`** (yolo26s + static ctx + narrow GEV).
   Rationale: (a) opt is accuracy-equivalent at -22% latency, (b) narrow
   holds within noise at a further -26% 3050 fp32.
2. **Augmentation: ON** (the recipe_n100 `aug` pipeline). -30.8% best val
   EPE on the deciding metric, regularization signature confirmed.
3. **freeze_bn: OFF.** No effect alone; costs 12% vs aug-alone when
   composed.
4. **sharp_tail: OFF.** Rejected (Q-d).
5. **costlookup_y26s: dropped.** costlookup_y26n retained only as the
   param-budget fallback (< 1.5 M) if the deployment envelope tightens.
6. Plateau/stopping: extend max steps and patience under aug; the aug arm
   hit the 12000-step wall still improving.

## Pre-registered follow-ups

1. **Aug ceiling run**: rerun `recipe_n100/aug` with max_steps 24000 (or
   fold into the full training run). Deciding metric: best val EPE;
   success = any further gain > 5% relative below 1.921; within 5% =
   "12000 was effectively converged".
2. **Aug decomposition** (single-knob): split the aug pipeline into its
   components (geometric crop/flip vs photometric jitter), control =
   aug-all. Deciding metric: best val EPE. Purpose: know which component
   carries the 30% before it interacts with full-data training.
3. **bad-0.5 drift check**: after full training with aug on
   gev4_opt_narrow, compare bad-0.5 against a gev4_opt reference to
   confirm the narrow-GEV sub-pixel drift (Q-b watch item) did not
   compound.
4. **MB14 zero-shot eval** after the full Scene Flow run (mandatory per
   protocol, about $0.05 on Modal T4). Overfit numbers here validate
   capacity and recipe, not cross-domain behaviour.
5. **Latency re-measure** of the adopted config (fp16, plus INT8 path)
   on the reference devices; the sharptail 82.3 ms fp32 in-session number
   and the recipe arms' shared-latency assumption should be replaced by
   fresh on-disk json.
