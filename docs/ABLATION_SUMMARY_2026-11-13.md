# Consolidated Final LPIPS ablation evidence — 13 November 2026

Purpose: one table set answering the advisor's three priorities (role of the
residual mask, noise level/DDIM steps, ablation and baselines) from runs that
already exist. **No new inference, training, threshold fitting or test access
was performed for the original 13 Nov summary**. Later subsections marked
"added" record their own runs; only the 14 Nov test-CI subsection touched test.
All values are copied from producer aggregate
JSON on the private `dacn-bmad-team/dacn-runs` Hub repo; JSON member hashes were
checked against the producer manifests (see
[evidence](evidence/2026-11-13-hf-producer-sync.json) and
[09 Nov sweep CSV](evidence/2026-11-10-ddim-ablation-table.csv)).

Shared contract unless noted: Gaussian EMA epoch30 checkpoint `d3e06d20…b946`,
dataset revision `350c74f2…b3b7`, one training seed and one sampling seed policy
(base17/UUID-derived), spatial LPIPS0.1 SqueezeNet on grayscale repeated RGB
`[-1,1]`, `normalize=False`, full 240 crop with no foreground filtering.
Pixel metrics are global micro over the crop; F1 equals micro Dice. Image score
is the mean of the top 576 pixels. Thresholds are max-F1 on the 40-image
calibration half (largest threshold on ties) and are then applied unchanged.
"Holdout" means the registered 43-image patient-group validation holdout, which
has been inspected repeatedly; it is exploratory, not independent confirmation.

"Final LPIPS" = DDIM20 from t300 → pseudo-mask from
`LPIPS_initial × clip(L1/P95(L1))`, pooled-95th-percentile threshold fitted on
20 normal calibration slices, dilation 3 → masked DDIM30 from t50, one pass →
raw LPIPS(input, final) with no residual multiplier.

## 1. Role of the pseudo-mask, residual term and re-sampling

| Configuration | Cal. AP | Holdout AP | Holdout Dice | Test AP | Test AUROC | Test Dice | Test image AUROC |
|---|---:|---:|---:|---:|---:|---:|---:|
| Initial LPIPS (DDIM20, no phase 2) | 0.2381 | 0.1579 | 0.2469 | 0.2246 | 0.9200 | 0.2650 | 0.7213 |
| Naive-stitch LPIPS (same residual mask, no re-sampling) | – | 0.1806 | 0.2968 | 0.2695 | 0.9314 | 0.3330 | 0.7191 |
| Mask from raw initial LPIPS only (no residual), then re-sample | 0.2698 | 0.2873 | 0.3504 | 0.2900 | 0.9257 | 0.3472 | 0.7949 |
| **Final LPIPS (LPIPS×residual mask, re-sample)** | 0.3130 | 0.3123 | 0.3712 | **0.3182** | 0.9388 | 0.3643 | **0.8060** |
| Final reconstruction, score LPIPS × final residual | 0.2477 | 0.1968 | 0.2725 | – | – | – | – |
| Brightness control (no model) | 0.1466 | 0.3895 | 0.2681 | **0.5114** | **0.9615** | **0.4170** | 0.6436 |

Reading on this cohort (one seed; test CIs for the four frozen arms are in the
subsection below):

- Re-sampling inside the same mask adds about +0.05 test AP over naive stitching.
- Including the normalized residual in the *mask* adds about +0.03 test AP over
  an LPIPS-only mask; ranking agrees between holdout and test for this pair.
- Multiplying the residual into the *final score* lowers AP; the residual is
  useful for localisation of the re-sampled region, not as a score factor.
- Brightness remains higher on pixel AP/AUROC/Dice; Final LPIPS has the
  highest image AUROC of these rows and much lower normal-pixel FPR (test
  0.0098 vs 0.1077). Do not claim the diffusion method beats brightness overall.
- Test rows after the original four-arm test (LPIPS-only mask, 12 Nov; DDIM50,
  11 Nov) are exploratory extensions on an already inspected test set.

### Patient-group CIs for the frozen four arms on test (added 14 Nov, AF-09/AF-10)

`test-ci-full-20261114T042202Z` (code `e1ff7118`, T4) recomputed the frozen
four-arm protocol with the unchanged adapter and frozen thresholds, for
uncertainty only. Gate: every published test metric reproduced within 1e-6
(max abs error 5.2e-7). 3665 slices (640 normal / 3025 abnormal), 767 patient
groups, paired percentile bootstrap, 1000/1000 valid replicates, seed 17. No
threshold, method or configuration changed. Evidence:
[results](evidence/2026-11-14-test-ci-results.json),
[manifest](evidence/2026-11-14-test-ci-manifest.json).

| Arm | Pixel AP [95% CI] | Δ AP vs Final | Dice [95% CI] | Δ Dice vs Final | Image AUROC [95% CI] | Δ AUROC vs Final | Image AP [95% CI] | Δ image AP vs Final |
|---|---|---|---|---|---|---|---|---|
| **Final LPIPS** | 0.3182 [0.3031, 0.3333] | – | 0.3643 [0.3515, 0.3764] | – | 0.8060 [0.7764, 0.8330] | – | 0.9459 [0.9302, 0.9591] | – |
| Initial LPIPS | 0.2246 [0.2108, 0.2383] | [-0.1029, -0.0846] | 0.2650 [0.2524, 0.2770] | [-0.1066, -0.0915] | 0.7213 [0.6812, 0.7558] | [-0.1171, -0.0526] | 0.9139 [0.8924, 0.9343] | [-0.0469, -0.0171] |
| Naive-stitch LPIPS | 0.2695 [0.2546, 0.2842] | [-0.0575, -0.0406] | 0.3330 [0.3207, 0.3448] | [-0.0377, -0.0248] | 0.7191 [0.6812, 0.7541] | [-0.1165, -0.0583] | 0.9128 [0.8909, 0.9333] | [-0.0484, -0.0196] |
| Brightness control | 0.5114 [0.4830, 0.5431] | [0.1612, 0.2279] | 0.4170 [0.3972, 0.4379] | [0.0316, 0.0755] | 0.6436 [0.6033, 0.6873] | [-0.2132, -0.1073] | 0.9013 [0.8804, 0.9225] | [-0.0650, -0.0245] |

Reading:

- Final LPIPS beats Initial and Naive-stitch on every metric; all deltas exclude 0.
  Re-sampling inside the mask, not the mask alone, is what makes the gain.
- Brightness has higher pixel AP (+0.16 to +0.23) and Dice (+0.03 to +0.08)
  beyond uncertainty. Final LPIPS has higher image AUROC (+0.11 to +0.21) and
  image AP beyond uncertainty. The two methods win at different levels; neither
  dominates.
- The intervals cover patient resampling only. They are conditional on one
  trained model, one sampling seed and this cohort. The test set had already
  been inspected, so they quantify uncertainty and do not confirm the method
  independently.

### Pseudo-mask rule ablation — validation only (added 13 Nov, AF-07)

Hypothesis fixed before running: the p95/dilation3 mask re-samples about 45% of
normal brain foreground; a tighter mask should lower normal-tissue LPIPS and FPR,
retaining AP only while lesion recall stays high. Only the mask rule changes
(pooled normal-calibration percentile × dilation); cached DDIM20 t300 initials
and heatmaps, masked DDIM30 from t50, seeds and LPIPS are fixed. The control
replays the producer exactly (max abs error 0.0). Run
`mask-rule-full-20261113T165856Z`, [evidence](evidence/2026-11-13-mask-rule-results.json);
mask and final-quality columns use all 83 slices, metrics the 43-image holdout,
delta from a paired patient-group bootstrap (999 valid replicates).

| Rule | Mask threshold | Mask lesion recall | Normal foreground masked | Final normal LPIPS | Lesion/normal LPIPS | Cal. AP | Holdout AP | Holdout AUROC | Holdout Dice | Normal FPR | Image AUROC | Delta AP vs p95/d3 [95% CI] |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| **p95/d3 (control)** | 0.1334 | 0.814 | 0.448 | 0.1313 | 2.07 | 0.3130 | 0.3123 | 0.9490 | 0.3712 | 0.0058 | 0.9145 | – |
| p90/d3 | 0.0855 | 0.945 | 0.753 | 0.1567 | 1.79 | 0.3077 | 0.3119 | 0.9487 | 0.3807 | 0.0053 | 0.8925 | [-0.0217, +0.0180] |
| p90/d1 | 0.0855 | 0.643 | 0.376 | 0.1532 | 1.77 | 0.2822 | 0.3133 | 0.9458 | 0.3710 | 0.0055 | 0.8904 | [-0.0170, +0.0250] |
| p95/d1 | 0.1334 | 0.472 | 0.187 | 0.1190 | 2.16 | 0.2810 | 0.3322 | 0.9468 | 0.3819 | 0.0046 | 0.8991 | [-0.0147, +0.0433] |
| p98/d3 | 0.1825 | 0.600 | 0.202 | 0.0888 | 2.76 | 0.2859 | 0.3148 | 0.9417 | 0.3768 | 0.0051 | 0.8838 | [-0.0226, +0.0165] |
| p98/d1 | 0.1825 | 0.311 | 0.075 | 0.0722 | 3.12 | 0.2857 | 0.3286 | 0.9392 | 0.3904 | 0.0039 | 0.8860 | [-0.0323, +0.0544] |
| p99/d3 | 0.2141 | 0.449 | 0.109 | 0.0606 | 3.53 | 0.2788 | 0.2912 | 0.9275 | 0.3561 | 0.0049 | 0.8706 | [-0.0610, +0.0069] |
| p99/d1 | 0.2141 | 0.219 | 0.038 | 0.0459 | 4.24 | 0.2933 | 0.3291 | 0.9323 | 0.3852 | 0.0022 | 0.8838 | [-0.0503, +0.0758] |

The normal-tissue part of the hypothesis holds: tighter masks cut normal
coverage (0.448 → 0.038), normal-tissue LPIPS (0.131 → 0.046) and FPR
(0.0058 → 0.0022), and raise lesion/normal contrast (2.07 → 4.24). Lesion
recall falls with them (0.814 → 0.219), and pixel AP does not change reliably:
all delta AP and delta Dice intervals include zero. The control has the highest
calibration AP and image ROC-AUC, so the mask rule is a normal-FPR versus
lesion-coverage trade-off rather than the AP bottleneck on this holdout. No rule
is adopted.

## 2. Noise level (phase-1 start) and DDIM step split — validation only

Final LPIPS arm, DDIM20 phase 1 + DDIM30 phase 2 from t50, unless stated.

| Phase-1 start t | Cal. AP | Holdout AP | Holdout AUROC | Holdout Dice | Holdout image AUROC | Holdout normal FPR |
|---:|---:|---:|---:|---:|---:|---:|
| 100 | 0.2646 | 0.2277 | 0.9282 | 0.2778 | 0.8553 | 0.0065 |
| 200 | 0.3067 | 0.2914 | 0.9413 | 0.3472 | 0.8904 | 0.0046 |
| 300 | 0.3130 | 0.3123 | 0.9490 | 0.3712 | 0.9145 | 0.0058 |
| 500 | 0.3202 | 0.3211 | 0.9524 | 0.3734 | 0.8640 | 0.0068 |
| 800 | 0.1297 | 0.1861 | 0.9113 | 0.2661 | 0.7039 | 0.0490 |

Same total 50 U-Net calls at t300, split between phases:

| Phase 1 / phase 2 steps | Cal. AP | Holdout AP | Holdout Dice | Holdout image AUROC |
|---|---:|---:|---:|---:|
| 10 / 40 | 0.3165 | 0.2902 | 0.3614 | 0.8969 |
| 20 / 30 | 0.3130 | 0.3123 | 0.3712 | 0.9145 |
| 25 / 25 | 0.3145 | 0.3126 | 0.3707 | 0.9057 |
| 30 / 20 | 0.3122 | 0.3162 | 0.3770 | 0.9013 |

The 30/20 split was also run at t100/200/500/800 and follows the same shape
(holdout AP 0.2308/0.2910/0.3288/0.1925). Pixel AP rises from t100 to a plateau
at t300–500 and collapses at t800; image AUROC peaks at t300.

### Patient-group bootstrap of the sweep (added 13 Nov)

Paired percentile bootstrap over the 25 holdout patient groups, 999 valid
replicates, seed17, producer maps and thresholds fixed; reference Final LPIPS
t300 DDIM20+30. "Delta" is candidate minus reference
([evidence](evidence/2026-11-13-sweep-bootstrap-results.json), run
`sweep-bootstrap-full-20261113T161403Z`).

| Candidate | Pixel AP [95% CI] | Delta AP [95% CI] | Delta Dice [95% CI] |
|---|---|---|---|
| Final t300 20/30 (reference) | 0.3123 [0.0990, 0.4847] | – | – |
| Final t100 20/30 | 0.2277 [0.0760, 0.3283] | [-0.1976, +0.0009] | [-0.1627, -0.0119] |
| Final t200 20/30 | 0.2914 [0.0928, 0.4211] | [-0.0796, +0.0223] | [-0.0503, +0.0193] |
| Final t500 20/30 | 0.3211 [0.1247, 0.4579] | [-0.1028, +0.0660] | [-0.0513, +0.0466] |
| Final t800 20/30 | 0.1861 [0.0748, 0.3071] | [-0.2380, -0.0279] | [-0.2215, -0.0027] |
| Final t300 10/40 | 0.2902 [0.0896, 0.4992] | [-0.0376, +0.0161] | [-0.0290, +0.0157] |
| Final t300 25/25 | 0.3126 [0.0996, 0.4770] | [-0.0155, +0.0098] | [-0.0072, +0.0052] |
| Final t300 30/20 | 0.3162 [0.1026, 0.4756] | [-0.0155, +0.0134] | [-0.0008, +0.0116] |
| Initial LPIPS t300 | 0.1579 [0.0618, 0.3424] | [-0.2048, -0.0397] | [-0.1744, -0.0515] |
| Final × final residual t300 | 0.1968 [0.0729, 0.2715] | [-0.2337, -0.0210] | [-0.1497, -0.0074] |
| Brightness | 0.3895 [0.1038, 0.5833] | [-0.2329, +0.2874] | [-0.2980, +0.1131] |

On this holdout, t800 is reliably worse and t100 is worse on Dice. t200, t300
and t500 cannot be separated, and neither can the four DDIM step splits. The
wide absolute AP intervals (about 0.10–0.48) show that the 43-image holdout
cannot rank configurations whose AP differs by a few hundredths; the
Initial→Final and residual-in-score effects are the only sweep contrasts that
clearly exclude zero. Intervals are conditional on one checkpoint, one sampling
seed and the repeatedly inspected holdout; they are not test results.

### Noise level → reconstruction quality → detection (added 13 Nov)

Descriptive aggregates from the same producer arrays (all 83 validation slices,
DDIM20+30; [evidence](evidence/2026-11-13-sweep-quality-results.json), run
`sweep-quality-full-20261113T162803Z`). L1 is |input − reconstruction| on the
`[-1,1]` scale over foreground (input > −0.95); "normal" is foreground of normal
slices. Final AP and FPR are the producer's 43-holdout values.

| t | Initial normal L1 | Initial lesion/normal L1 | Final lesion/normal LPIPS | Mask lesion recall | Normal foreground masked | Final holdout AP | Holdout normal FPR |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 100 | 0.084 | 0.96 | 2.08 | 0.828 | 0.506 | 0.2277 | 0.0065 |
| 200 | 0.114 | 1.07 | 2.11 | 0.817 | 0.464 | 0.2914 | 0.0046 |
| 300 | 0.137 | 1.20 | 2.07 | 0.814 | 0.448 | 0.3123 | 0.0058 |
| 500 | 0.183 | 1.62 | 2.00 | 0.779 | 0.414 | 0.3211 | 0.0068 |
| 800 | 0.329 | 1.48 | 1.73 | 0.621 | 0.291 | 0.1861 | 0.0490 |

Reading, consistent with the under-/over-noising hypothesis:

- **Too little noise (t100):** the initial reconstruction reproduces the lesion
  (lesion error no larger than normal error, ratio 0.96), so the residual part of
  the mask heatmap carries little lesion signal and detection is lowest among
  t100–500.
- **Plateau (t200–500):** lesion "healing" increases (ratio 1.07 → 1.62) while
  normal-tissue error grows moderately; after phase 2 the lesion/normal LPIPS
  contrast stays about 2.0–2.1 and AP is not separable by bootstrap.
- **Too much noise (t800):** normal tissue is no longer reconstructed faithfully
  (normal L1 0.329), lesion contrast drops (1.73), mask lesion recall falls to
  0.62 and normal-pixel FPR rises about eightfold.
- **Mask is coarse at every t:** 41–51% of normal-slice brain foreground is
  masked (pixel precision about 0.10–0.12), so phase 2 re-samples a large share of
  the brain. This makes the mask rule (percentile, dilation, heatmap) the next
  candidate for a hypothesis-driven, validation-only ablation (AF-07).
- The 30/20 split and the 43-holdout subset show the same ordering. Final-phase
  L1 is zero outside the mask by construction, so LPIPS contrast rather than
  final L1 is the detection-relevant quantity.

### Complete noise curve with t50 and t400 (added 14 Nov, AF-05)

t50 and t400 were added with a producer-equivalent runner whose t300 control
replays every producer array exactly (run `noise-ext-full-20261114T034843Z`,
[evidence](evidence/2026-11-14-noise-ext-results.json)). DDIM20+30, Final LPIPS:

| Phase-1 t | Initial normal L1 | Initial lesion/normal L1 | Final lesion/normal LPIPS | Mask lesion recall | Normal foreground masked | Cal. AP | Holdout AP | Holdout Dice | Image AUROC | Delta AP vs t300 [95% CI] | Source |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|
| 50 | 0.061 | 0.91 | 2.06 | 0.806 | 0.509 | 0.2040 | 0.1912 | 0.2438 | 0.8333 | [-0.2640, -0.0114] | 14 Nov run |
| 100 | 0.084 | 0.96 | 2.08 | 0.828 | 0.506 | 0.2646 | 0.2277 | 0.2778 | 0.8553 | [-0.1976, +0.0009] | 09 Nov producer |
| 200 | 0.114 | 1.07 | 2.11 | 0.817 | 0.464 | 0.3067 | 0.2914 | 0.3472 | 0.8904 | [-0.0796, +0.0223] | 09 Nov producer |
| **300** | 0.137 | 1.20 | 2.07 | 0.814 | 0.448 | 0.3130 | 0.3123 | 0.3712 | 0.9145 | – (reference) | 14 Nov run |
| 400 | 0.159 | 1.39 | 2.02 | 0.800 | 0.430 | 0.3060 | 0.3479 | 0.3920 | 0.9035 | [-0.0013, +0.0544] | 14 Nov run |
| 500 | 0.183 | 1.62 | 2.00 | 0.779 | 0.414 | 0.3202 | 0.3211 | 0.3734 | 0.8640 | [-0.1028, +0.0660] | 09 Nov producer |
| 800 | 0.329 | 1.48 | 1.73 | 0.621 | 0.291 | 0.1297 | 0.1861 | 0.2661 | 0.7039 | [-0.2380, -0.0279] | 09 Nov producer |

The curve is an inverted U. Too little noise (t50, t100) leaves the lesion in
the initial reconstruction; too much (t800) destroys normal-tissue fidelity.
t200–t500 form a plateau: t400 has the highest holdout AP (0.3479) but a lower
calibration AP and image ROC-AUC than t300, and its interval touches zero.
t300 is kept; the 30/20 split gives the same picture.

**Not yet measured:** phase-2 start other than t50; total DDIM budget other
than 50 calls; more than one sampling seed. Mask
percentile/dilation sensitivity is in Section 1.

### Seeds, DDIM budget, phase-2 start and LPIPS-only mask (added 14 Nov)

Each variant changes one component of the t300 control and was run with the
producer-equivalent runner (control replay error 0.0; t400 and the 12 Nov
LPIPS-only mask reproduced). Run `variants-full-20261114T041258Z`,
[evidence](evidence/2026-11-14-variants-results.json).

| Variant | AF | Cal. AP | Holdout AP | Holdout AUROC | Holdout Dice | Normal FPR | Image AUROC | Mask lesion recall | Normal foreground masked | Delta AP vs control [95% CI] | Delta Dice [95% CI] |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---|
| **Control t300, 20+30, phase 2 t50, seed 17** | – | 0.3130 | 0.3123 | 0.9490 | 0.3712 | 0.0058 | 0.9145 | 0.814 | 0.448 | – | – |
| t400 (seed 17) | AF-05 | 0.3060 | 0.3479 | 0.9532 | 0.3920 | 0.0069 | 0.9035 | 0.800 | 0.430 | [-0.0013, +0.0544] | [-0.0092, +0.0430] |
| Seed 18, t300 | AF-11 | 0.3677 | 0.2868 | 0.9434 | 0.3461 | 0.0037 | 0.8640 | 0.820 | 0.477 | [-0.0718, +0.0230] | [-0.0613, +0.0071] |
| Seed 19, t300 | AF-11 | 0.3471 | 0.3070 | 0.9415 | 0.3489 | 0.0033 | 0.8904 | 0.798 | 0.465 | [-0.0618, +0.0165] | [-0.0861, +0.0348] |
| Seed 18, t400 | AF-11 | 0.2944 | 0.3238 | 0.9487 | 0.3728 | 0.0047 | 0.8728 | 0.807 | 0.454 | [-0.0446, +0.0448] | [-0.0345, +0.0175] |
| Seed 19, t400 | AF-11 | 0.3353 | 0.3340 | 0.9512 | 0.3870 | 0.0040 | 0.9013 | 0.789 | 0.444 | [-0.0349, +0.0503] | [-0.0132, +0.0518] |
| DDIM budget 10+15 | AF-07 | 0.3155 | 0.2770 | 0.9454 | 0.3550 | 0.0058 | 0.8947 | 0.773 | 0.453 | [-0.0534, +0.0178] | [-0.0367, +0.0116] |
| DDIM budget 40+50 | AF-07 | 0.3078 | 0.3328 | 0.9504 | 0.3833 | 0.0055 | 0.9013 | 0.837 | 0.447 | [-0.0078, +0.0344] | [-0.0016, +0.0222] |
| Phase-2 start t30 | AF-07 | 0.3030 | 0.3034 | 0.9473 | 0.3660 | 0.0042 | 0.8991 | 0.814 | 0.448 | [-0.0250, +0.0018] | [-0.0162, +0.0089] |
| Phase-2 start t100 | AF-07 | 0.3170 | 0.3182 | 0.9515 | 0.3765 | 0.0081 | 0.9013 | 0.814 | 0.448 | [-0.0158, +0.0216] | [-0.0163, +0.0167] |
| LPIPS-only pseudo-mask | AF-02 | 0.2698 | 0.2873 | 0.9332 | 0.3504 | 0.0062 | 0.8662 | 0.690 | 0.242 | [-0.0576, +0.0028] | [-0.0576, -0.0027] |

Sampling-seed summary (Final LPIPS pixel AP; seeds 17/18/19):

| Partition | t300 | t400 | t400 − t300 per seed |
|---|---|---|---|
| 43-image holdout | 0.3123 / 0.2868 / 0.3070 (mean 0.3021, SD 0.0135) | 0.3479 / 0.3238 / 0.3340 (mean 0.3353, SD 0.0121) | +0.0356 / +0.0370 / +0.0270 |
| 40-image calibration | 0.3130 / 0.3677 / 0.3471 (mean 0.3426, SD 0.0277) | 0.3060 / 0.2944 / 0.3353 (mean 0.3119, SD 0.0211) | −0.0070 / −0.0733 / −0.0118 |

Sampling seeds alone move AP by about one to three hundredths, the same size as
the differences between plateau configurations. t400 wins on the holdout and
loses on calibration for all three seeds, so no noise level in t200–t500 can be
preferred on this validation set. DDIM budget and phase-2 start do not change AP
or Dice beyond uncertainty. The LPIPS-only mask lowers Dice reliably, which
supports keeping the residual term inside the mask.

## 3. Alternative scores on the same Final reconstruction — validation only

All rows reuse cached Final LPIPS maps or layers; zero new diffusion calls.

| Score | Cal. AP | Holdout AP | Holdout Dice | Holdout image AUROC |
|---|---:|---:|---:|---:|
| Final LPIPS (reference) | 0.3130 | 0.3123 | 0.3712 | 0.9145 |
| CGFF context-gated layer fusion | 0.3129 | 0.3133 | 0.3702 | 0.9123 |
| PraNet-style fixed layer fusion, balanced | 0.3170 | 0.3023 | 0.3722 | 0.9167 |
| PraNet-style, middle-focused | 0.3078 | 0.3164 | 0.3723 | 0.9123 |
| PraNet-style, deep-focused | 0.3377 | 0.2884 | 0.3644 | 0.9189 |
| PraNet-style, shallow-focused | 0.2820 | 0.2723 | 0.3512 | 0.9145 |
| Single LPIPS layer 3 | 0.1637 | 0.3574 | 0.3716 | 0.8662 |
| Single LPIPS layer 5 | 0.2941 | 0.1592 | 0.2307 | 0.8465 |
| Normal-calibrated equal-mean LPIPS | 0.3033 | 0.3072 | 0.3736 | 0.9189 |
| Spatially normal-calibrated LPIPS (SNC) | 0.0432 | 0.0673 | 0.1386 | 0.7917 |
| Sobel gradient difference | 0.1003 | 0.0960 | 0.1807 | 0.3706 |
| Final L1 residual | 0.1138 | 0.0996 | 0.2001 | 0.4408 |

No alternative improves both halves. Single-layer rankings reverse between
calibration and holdout (layer 3: 0.164 → 0.357; layer 5: 0.294 → 0.159),
which shows the variance of 40/43-image comparisons. Treat SNC, Sobel,
PraNet-style and CGFF as negative or neutral results.

Fixed 50/50 calibration-CDF fusion of LPIPS and residual on the *initial*
DDIM20 reconstruction gave holdout AP 0.1531 vs raw initial LPIPS 0.1579
([DDIM20 score fusion](DDIM20_SCORE_FUSION_VALIDATION.md)).

### Weighted LPIPS + residual fusion on the Final reconstruction (added 14 Nov, AF-03)

Hypothesis fixed before running: additive fusion in a common rank scale avoids
the suppression of the LPIPS × residual product, so a small residual weight
keeps or raises AP. Scores mapped by midrank CDFs fitted on normal calibration
pixels; fused = (1 − w)·CDF(LPIPS) + w·CDF(L1). Run
`final-fusion-full-20261114T034226Z`, [evidence](evidence/2026-11-14-final-fusion-results.json),
paired bootstrap vs raw LPIPS (999 valid replicates).

| Residual weight | Cal. AP | Holdout AP | Holdout Dice | Image AUROC | Delta AP [95% CI] |
|---:|---:|---:|---:|---:|---|
| raw LPIPS (reference) | 0.3130 | 0.3123 | 0.3712 | 0.9145 | – |
| 0 (CDF only) | 0.3130 | 0.3123 | 0.3712 | 0.9145 | [−0.0001, +0.0000] |
| 0.25 | 0.3155 | 0.2525 | 0.3514 | 0.9232 | [−0.1691, +0.0039] |
| 0.5 | 0.2721 | 0.2143 | 0.2817 | 0.9189 | [−0.2463, −0.0032] |
| 0.75 | 0.2081 | 0.1701 | 0.2163 | 0.9232 | [−0.3196, −0.0159] |
| 1 (L1 only) | 0.1138 | 0.0996 | 0.2001 | 0.4627 | [−0.3987, −0.0425] |

Not supported for pixel AP: no weight raises holdout AP, and w ≥ 0.5 lowers it
beyond bootstrap uncertainty. A small residual weight slightly raises image
ROC-AUC. Raw Final LPIPS remains the reference score.

### Brain-mask post-processing and brightness hybrids (added 14 Nov, validation)

`hybrid-full-20261114T103245Z` (code `2c76db10`, Modal CPU, 6 min) used the cached
t300 Final LPIPS and brightness maps; there were no model calls and the test set was
not used. The hypotheses, 18 arms and decision rule are fixed in
`dacn.evaluation.postprocess_hybrid` and were not changed after the run. Brain mask:
input > −0.95, holes filled, eroded 3 px; it keeps 99.4% / 98.4% of lesion pixels
(calibration/holdout). Both unprocessed references replay the producer exactly
(error 0.0). Paired patient-group bootstrap on the 43-image holdout, 999/1000 valid
replicates. Evidence: [results](evidence/2026-11-14-hybrid-results.json),
[manifest](evidence/2026-11-14-hybrid-manifest.json).

| Arm | Cal. AP | Holdout AP [95% CI] | Δ AP vs Final | Δ AP vs brightness | Holdout pixel AUROC | Holdout Dice | Δ Dice vs Final | Image AUROC | Δ image AUROC vs Final | Rule |
|---|---:|---|---|---|---:|---:|---|---:|---|:---:|
| final_lpips / none | 0.313 | 0.312 [0.099, 0.485] | – | [-0.287, +0.233] | 0.949 | 0.371 | – | 0.914 | – |  |
| final_lpips / brain | 0.318 | 0.314 [0.102, 0.485] | [-0.002, +0.004] | [-0.287, +0.234] | 0.945 | 0.372 | [-0.000, +0.002] | 0.914 | [+0.000, +0.000] |  |
| final_lpips / brain_median5 | 0.324 | 0.318 [0.103, 0.495] | [+0.002, +0.013] | [-0.283, +0.244] | 0.946 | 0.377 | [-0.002, +0.013] | 0.912 | [-0.012, +0.000] |  |
| brightness / none | 0.147 | 0.390 [0.104, 0.583] | [-0.233, +0.287] | – | 0.950 | 0.268 | [-0.298, +0.113] | 0.680 | [-0.549, +0.036] |  |
| brightness / brain | 0.155 | 0.397 [0.109, 0.591] | [-0.225, +0.293] | [-0.001, +0.015] | 0.945 | 0.276 | [-0.293, +0.117] | 0.680 | [-0.549, +0.040] |  |
| brightness / brain_median5 | 0.174 | 0.429 [0.134, 0.615] | [-0.200, +0.332] | [+0.017, +0.076] | 0.946 | 0.278 | [-0.293, +0.126] | 0.724 | [-0.498, +0.058] |  |
| lpips_x_brightness / none | 0.496 | 0.656 [0.365, 0.783] | [+0.153, +0.449] | [+0.109, +0.426] | 0.979 | 0.588 | [+0.059, +0.324] | 0.932 | [-0.060, +0.129] | ✓ |
| lpips_x_brightness / brain | 0.497 | 0.653 [0.362, 0.777] | [+0.154, +0.445] | [+0.106, +0.426] | 0.971 | 0.589 | [+0.060, +0.323] | 0.930 | [-0.060, +0.122] | ✓ |
| lpips_x_brightness / brain_median5 | 0.504 | 0.669 [0.379, 0.791] | [+0.175, +0.464] | [+0.120, +0.453] | 0.972 | 0.597 | [+0.062, +0.337] | 0.934 | [-0.049, +0.129] | ✓ |
| lpips_bright_cdf_w025 / none | 0.501 | 0.624 [0.325, 0.751] | [+0.130, +0.392] | [+0.067, +0.400] | 0.971 | 0.584 | [+0.093, +0.295] | 0.901 | [-0.104, +0.092] | ✓ |
| lpips_bright_cdf_w025 / brain | 0.501 | 0.622 [0.325, 0.748] | [+0.130, +0.389] | [+0.066, +0.398] | 0.964 | 0.584 | [+0.095, +0.295] | 0.901 | [-0.104, +0.092] | ✓ |
| lpips_bright_cdf_w025 / brain_median5 | 0.507 | 0.635 [0.343, 0.758] | [+0.148, +0.403] | [+0.078, +0.422] | 0.964 | 0.592 | [+0.098, +0.307] | 0.904 | [-0.100, +0.098] | ✓ |
| lpips_bright_cdf_w050 / none | 0.434 | 0.623 [0.315, 0.776] | [+0.017, +0.465] | [+0.131, +0.341] | 0.978 | 0.547 | [-0.046, +0.347] | 0.866 | [-0.220, +0.107] | ✓ |
| lpips_bright_cdf_w050 / brain | 0.436 | 0.621 [0.315, 0.773] | [+0.019, +0.461] | [+0.127, +0.342] | 0.970 | 0.546 | [-0.049, +0.347] | 0.866 | [-0.220, +0.107] | ✓ |
| lpips_bright_cdf_w050 / brain_median5 | 0.443 | 0.641 [0.352, 0.790] | [+0.045, +0.486] | [+0.142, +0.378] | 0.971 | 0.568 | [-0.030, +0.368] | 0.879 | [-0.196, +0.115] | ✓ |
| lpips_bright_cdf_w075 / none | 0.310 | 0.548 [0.239, 0.721] | [-0.088, +0.433] | [+0.092, +0.230] | 0.970 | 0.406 | [-0.203, +0.259] | 0.796 | [-0.383, +0.075] |  |
| lpips_bright_cdf_w075 / brain | 0.314 | 0.548 [0.236, 0.720] | [-0.085, +0.430] | [+0.090, +0.233] | 0.962 | 0.410 | [-0.200, +0.261] | 0.794 | [-0.391, +0.078] |  |
| lpips_bright_cdf_w075 / brain_median5 | 0.317 | 0.570 [0.269, 0.745] | [-0.067, +0.457] | [+0.105, +0.274] | 0.964 | 0.438 | [-0.178, +0.293] | 0.820 | [-0.322, +0.095] |  |

Reading:

- **H1 (post-processing): small effect.** The brain mask alone changes nothing
  reliable. Adding the median filter gives Final +0.002 to +0.013 AP and brightness
  +0.017 to +0.076 AP. It removes edge false positives but does not change the
  ranking between methods.
- **H2 (hybrid): large effect.** LPIPS × brightness, with no extra parameters,
  roughly doubles holdout pixel AP (0.656 vs 0.312 Final and 0.390 brightness). It
  beats both beyond uncertainty, raises Dice by +0.06 to +0.32, keeps image AUROC
  (0.932 vs 0.914; Δ CI includes 0), and also raises calibration AP from 0.313 to
  0.496. The calibration gain matters because it shows the result is not a quirk of
  the holdout. Rank fusion with w ≤ 0.5 gives similar gains; w = 0.75 drifts toward
  brightness and fails the rule.
- Interpretation: Final LPIPS localises "what the model cannot reconstruct", which
  includes structures that are not lesions. Brightness keeps only the hyperintense
  part, which matches FLAIR lesion appearance. This is the same idea as counting only
  positive residuals on FLAIR.
- Caveats: 18 arms were evaluated, so some intervals exclude 0 by chance, although
  these deltas are far from 0. The holdout has been inspected repeatedly. This run
  did not test whether the diffusion reconstruction is needed, i.e. Initial LPIPS ×
  brightness or L1 × brightness. That ablation is required before the gain can be
  credited to Final.

### Which reconstruction error to gate by brightness (added 14 Nov, validation)

`gate-full-20261114T105141Z` (code `37c1cbe6`, Modal CPU, 74 s) tested whether the
hybrid gain needs the diffusion re-sampling. Each cached producer map was multiplied
by the same brightness map, with no post-processing and no test access. Rule fixed
before running (`GATE_*` in `dacn.evaluation.postprocess_hybrid`): the re-sampling is
credited only if Initial LPIPS × brightness is below Final LPIPS × brightness, with
the holdout Δ AP CI excluding 0 and calibration agreeing. Replays: producer
references 0.0; Final LPIPS × brightness reproduces the hybrid run (0.0). 999/1000
valid replicates. Evidence: [results](evidence/2026-11-14-gate-results.json),
[manifest](evidence/2026-11-14-gate-manifest.json).

| Map | Cal. AP | Holdout AP [95% CI] | Δ AP vs Final LPIPS × brightness | Holdout pixel AUROC | Holdout Dice | Δ Dice | Image AUROC | Δ image AUROC |
|---|---:|---|---|---:|---:|---|---:|---|
| final_lpips__none | 0.313 | 0.312 [0.099, 0.485] | [-0.449, -0.153] | 0.949 | 0.371 | [-0.324, -0.059] | 0.914 | [-0.129, +0.060] |
| brightness__none | 0.147 | 0.390 [0.104, 0.583] | [-0.426, -0.109] | 0.950 | 0.268 | [-0.411, -0.181] | 0.680 | [-0.537, -0.030] |
| final_lpips_x_brightness | 0.496 | 0.656 [0.365, 0.783] | – | 0.979 | 0.588 | – | 0.932 | – |
| initial_lpips_x_brightness | 0.421 | 0.530 [0.185, 0.681] | [-0.256, -0.014] | 0.969 | 0.486 | [-0.203, -0.025] | 0.818 | [-0.281, +0.015] |
| final_l1_x_brightness | 0.189 | 0.181 [0.061, 0.286] | [-0.567, -0.250] | 0.835 | 0.280 | [-0.382, -0.192] | 0.592 | [-0.561, -0.136] |
| initial_l1_x_brightness | 0.146 | 0.168 [0.059, 0.278] | [-0.589, -0.261] | 0.918 | 0.248 | [-0.420, -0.211] | 0.406 | [-0.776, -0.259] |
| naive_stitch_l1_x_brightness | 0.193 | 0.184 [0.063, 0.291] | [-0.566, -0.247] | 0.836 | 0.285 | [-0.379, -0.188] | 0.583 | [-0.586, -0.136] |

Reading:

- **The rule credits the re-sampling,** but with a small margin. On holdout, Final
  LPIPS × brightness beats Initial LPIPS × brightness by +0.126 AP (CI of
  Initial − Final [−0.256, −0.014]). Calibration agrees (0.496 vs 0.421), and so do
  Dice and image AUROC.
- **The perceptual term is essential.** Every L1 residual gated by brightness stays
  near AP 0.17–0.19, below brightness alone. Gating works on LPIPS maps, not on
  intensity residuals.
- Holdout AP builds up as follows: brightness 0.390 → Initial LPIPS × brightness
  0.530 → Final LPIPS × brightness 0.656. Brightness, the perceptual error and the
  masked re-sampling each contribute.
- Same caveats as the hybrid subsection: the holdout has been inspected repeatedly,
  there is one model and one sampling seed, and no test access. The
  Initial-vs-Final interval ends close to 0.

### Single frozen test evaluation of Final LPIPS × brightness (added 14 Nov)

After the gate ablation, the configuration (Final LPIPS × brightness, no
post-processing) and its validation-calibration thresholds (pixel 0.13542, image
0.17669) were frozen in code `a5911a35`. Then one test evaluation was run:
`hybrid-test-full-20261114T105717Z`, CPU, using the identity-checked test-CI map
chunks, with no model calls and no fitting on test. Gate: the unprocessed maps
reproduce the published test metrics (≤ 1.6e-14). 767 patient groups, 1000/1000
valid replicates. Pixel AUROC has no bootstrap interval; the reference values are from the published 10 Nov aggregate. This is the **fifth test access**; the configuration was chosen
after several validation development rounds in this session. Evidence:
[results](evidence/2026-11-14-hybrid-test-results.json),
[manifest](evidence/2026-11-14-hybrid-test-manifest.json).

| Arm | Pixel AP [95% CI] | Δ AP vs frozen hybrid | Pixel AUROC | Dice [95% CI] | Δ Dice | Image AUROC [95% CI] | Δ AUROC | Image AP [95% CI] | Δ image AP |
|---|---|---|---:|---|---|---|---|---|---|
| **Final LPIPS × brightness (frozen)** | 0.6505 [0.6263, 0.6731] | – | 0.9759 | 0.5943 [0.5797, 0.6089] | – | 0.8397 [0.8129, 0.8637] | – | 0.9594 [0.9475, 0.9697] | – |
| Final LPIPS | 0.3182 [0.3031, 0.3333] | [-0.3504, -0.3142] | 0.9388 | 0.3643 [0.3515, 0.3764] | [-0.2413, -0.2180] | 0.8060 [0.7764, 0.8330] | [-0.0596, -0.0086] | 0.9459 [0.9302, 0.9591] | [-0.0239, -0.0043] |
| Brightness control | 0.5114 [0.4830, 0.5431] | [-0.1676, -0.1119] | 0.9615 | 0.4170 [0.3972, 0.4379] | [-0.1937, -0.1600] | 0.6436 [0.6033, 0.6873] | [-0.2315, -0.1600] | 0.9013 [0.8804, 0.9225] | [-0.0740, -0.0426] |

Frozen hybrid point metrics: pixel AUROC 0.9759, precision 0.6641, recall 0.5377, macro lesion-slice Dice 0.5356, normal-slice pixel FPR 0.0101.

Reading:

- The validation gain carries over to test: pixel AP 0.651 (validation holdout
  0.656). The hybrid beats Final LPIPS and brightness on every pixel and image
  metric, with all deltas beyond uncertainty. It is the first configuration in this
  project that beats the brightness control at pixel level.
- The image-level gain over Final is small (AUROC +0.009 to +0.060); the pixel-level
  gains are large.
- Scope: one trained model, one sampling seed, one cohort; the test set has been
  inspected before. The gate ablation for the re-sampling contribution
  (Initial vs Final LPIPS × brightness) exists on validation only.

### Sampling-seed robustness of Final LPIPS × brightness (added 14 Nov, validation)

`seeds-full-20261114T120916Z` (code `31d89644`, Modal CPU, 4 min) reused the
SHA-checked t300 DDIM20+30 maps for sampling seed bases 17, 18 and 19 from
`variants-full-20261114T041258Z`. There were no model calls and no test access; the
trained model is unchanged. Thresholds were refit on calibration for each seed. Rules
fixed in code before running (`SEED_*` in `dacn.evaluation.postprocess_hybrid`).
Replays: the producer references and the seed-17 hybrid are exact (0.0). 999/1000
valid replicates per seed. Every Δ is the other arm minus the hybrid, so a CI below 0
means the hybrid wins. Evidence:
[results](evidence/2026-11-14-seeds-results.json),
[manifest](evidence/2026-11-14-seeds-manifest.json).

| Seed | Final LPIPS × brightness: cal. AP / holdout AP [95% CI] | Δ brightness − hybrid | Δ Final LPIPS − hybrid (same seed) | Initial LPIPS × brightness: cal. / holdout AP | Δ Initial × brightness − hybrid | Holdout Dice | Holdout pixel AUROC | Image AUROC |
|---|---|---|---|---|---|---:|---:|---:|
| 17 | 0.496 / 0.656 [0.365, 0.783] | [-0.426, -0.109] | [-0.449, -0.153] | 0.421 / 0.530 | [-0.256, -0.014] | 0.588 | 0.979 | 0.932 |
| 18 | 0.531 / 0.610 [0.287, 0.763] | [-0.348, -0.065] | [-0.430, -0.112] | 0.461 / 0.482 | [-0.231, -0.023] | 0.550 | 0.976 | 0.899 |
| 19 | 0.519 / 0.659 [0.387, 0.777] | [-0.416, -0.124] | [-0.459, -0.156] | 0.460 / 0.529 | [-0.275, -0.042] | 0.599 | 0.976 | 0.910 |

Reading:

- **Seed-robust:** the hybrid beats brightness and Final LPIPS beyond uncertainty for
  all three sampling seeds. Holdout AP is 0.642 ± 0.027 (mean ± SD), against a seed SD
  of about 0.013 for Final LPIPS alone. The hybrid varies more in absolute AP but stays
  far above both references.
- **The re-sampling credit holds for all three seeds:** Initial × brightness is lower
  beyond uncertainty and calibration agrees each time. The margins stay modest
  (CI upper bounds −0.014 to −0.042).
- This covers sampling-seed variability only. Variability from retraining the model
  (AF-11) has not been measured.

### LPIPS backbone and normalised brightness gate (added 14 Nov, validation)

`refine-full-20261114T125015Z` (code `436b03da`, Modal CPU, 3 min) recomputed
spatial LPIPS with SqueezeNet, AlexNet and VGG16 (lpips 0.1, same input contract) on
the cached Final reconstructions. Each map was crossed with four gates: none, raw
brightness, brightness CDF fitted on normal-calibration brain pixels, and per-slice
brightness rank. The reference is the frozen hybrid. There were no DDPM calls and no
test access. Hypotheses and rule are fixed in `dacn.evaluation.score_refinement`.
Replays: CPU SqueezeNet maps vs producer max error 4.8e-7; frozen hybrid exact (0.0).
999/1000 valid replicates. Each Δ is arm − frozen hybrid. Evidence:
[results](evidence/2026-11-14-refine-results.json),
[manifest](evidence/2026-11-14-refine-manifest.json).

| Arm (backbone / gate) | Cal. AP | Holdout AP [95% CI] | Δ AP − frozen hybrid | Holdout pixel AUROC | Holdout Dice | Δ Dice | Image AUROC | Δ image AUROC | Rule |
|---|---:|---|---|---:|---:|---|---:|---|:---:|
| frozen_hybrid | 0.496 | 0.656 [0.365, 0.783] | – | 0.979 | 0.588 | – | 0.932 | – |  |
| squeeze / none | 0.313 | 0.312 [0.099, 0.485] | [-0.449, -0.153] | 0.949 | 0.371 | [-0.324, -0.059] | 0.914 | [-0.129, +0.060] |  |
| squeeze / brightness | 0.496 | 0.656 [0.365, 0.783] | [-0.000, +0.000] | 0.979 | 0.588 | [+0.000, +0.000] | 0.932 | [+0.000, +0.000] |  |
| squeeze / normal_cdf | 0.466 | 0.648 [0.351, 0.785] | [-0.057, +0.035] | 0.980 | 0.579 | [-0.051, +0.025] | 0.914 | [-0.058, +0.005] |  |
| squeeze / slice_rank | 0.682 | 0.663 [0.346, 0.796] | [-0.090, +0.105] | 0.985 | 0.594 | [-0.101, +0.142] | 0.908 | [-0.164, +0.063] |  |
| alex / none | 0.153 | 0.286 [0.064, 0.490] | [-0.512, -0.086] | 0.937 | 0.308 | [-0.415, -0.108] | 0.822 | [-0.266, +0.020] |  |
| alex / brightness | 0.282 | 0.594 [0.253, 0.742] | [-0.154, -0.003] | 0.975 | 0.534 | [-0.108, -0.023] | 0.803 | [-0.281, -0.015] |  |
| alex / normal_cdf | 0.270 | 0.579 [0.252, 0.738] | [-0.147, -0.027] | 0.976 | 0.522 | [-0.130, -0.019] | 0.789 | [-0.283, -0.037] |  |
| alex / slice_rank | 0.479 | 0.629 [0.238, 0.763] | [-0.166, +0.108] | 0.983 | 0.585 | [-0.141, +0.118] | 0.811 | [-0.298, -0.002] |  |
| vgg / none | 0.279 | 0.253 [0.090, 0.399] | [-0.536, -0.151] | 0.941 | 0.288 | [-0.424, -0.092] | 0.875 | [-0.244, +0.060] |  |
| vgg / brightness | 0.456 | 0.595 [0.301, 0.726] | [-0.143, +0.019] | 0.974 | 0.551 | [-0.086, +0.008] | 0.862 | [-0.168, +0.005] |  |
| vgg / normal_cdf | 0.426 | 0.575 [0.256, 0.729] | [-0.195, +0.023] | 0.975 | 0.534 | [-0.129, +0.012] | 0.840 | [-0.223, +0.004] |  |
| vgg / slice_rank | 0.691 | 0.629 [0.358, 0.722] | [-0.133, +0.077] | 0.982 | 0.535 | [-0.130, +0.055] | 0.866 | [-0.255, +0.035] |  |

Reading:

- **No arm passes the rule; the frozen hybrid stands.**
- **H1 (backbone): not supported.** AlexNet and VGG16 are below SqueezeNet both raw
  (holdout AP 0.286 / 0.253 vs 0.312) and gated by brightness (0.594 / 0.595 vs
  0.656). AlexNet × brightness is lower beyond uncertainty. SqueezeNet stays.
- **H2 (normalised gate): not supported.** The normal-calibration CDF gate changes
  nothing reliable (0.648). The per-slice rank gate is the one notable signal:
  calibration AP rises from 0.496 to 0.682, but holdout AP barely moves (0.663,
  Δ CI [−0.090, +0.105]) and image AUROC is slightly lower. Calibration and holdout
  disagree, as with t400, so this is not adopted. It could be revisited with a
  different partition or more data.
- Together with AF-03 and the 6 Nov normal-calibrated LPIPS, this is the third
  LPIPS-side refinement without a reliable gain. The remaining headroom seems to lie
  less in the LPIPS network than in the gate and the reconstruction.

### Error analysis, pooled-83 comparison and reconstruction ensembles (added 14 Nov, validation)

`diagnose-full-20261114T132851Z` (code `4ffa2eab`, Modal CPU, 4.5 min) used only
cached SHA-checked maps: producer t200/t300/t500, variants t400 and seeds 18/19. There
were no model calls and no test access. Hypotheses and rules are fixed in
`dacn.evaluation.hybrid_diagnostics`. Replays: producer references 0.0, frozen hybrid
0.0. 999–1000 valid replicates. The pooled bootstrap covers 50 patient groups.
Evidence: [results](evidence/2026-11-14-diagnose-results.json),
[manifest](evidence/2026-11-14-diagnose-manifest.json).

**A. Where the frozen hybrid fails** (holdout at calibration thresholds; calibration
in parentheses):

| Arm | Lesions 1-49 px: detected / pixel recall | Lesions 50-199 px: detected / pixel recall | Lesions 200-999 px: detected / pixel recall | Lesions >=1000 px: detected / pixel recall | FP inside brain | FP at brightness 0–0.50 / 0.50–1.00 | FP in components < 10 px |
|---|---|---|---|---|---|---|---|
| Frozen hybrid | 0.22 / 0.26 (n=32; cal. 0.08 / 0.08) | 0.67 / 0.39 (n=3; cal. 0.00 / 0.00) | 0.86 / 0.47 (n=7; cal. 1.00 / 0.35) | 1.00 / 0.63 (n=18; cal. 1.00 / 0.59) | 0.99 | 0.23 / 0.77 | 0.027 |
| Final LPIPS | 0.44 / 0.28 (n=32; cal. 0.15 / 0.09) | 0.67 / 0.53 (n=3; cal. 0.00 / 0.00) | 0.57 / 0.30 (n=7; cal. 0.80 / 0.34) | 1.00 / 0.51 (n=18; cal. 1.00 / 0.59) | 0.99 | 0.90 / 0.10 | 0.011 |
| Brightness | 0.34 / 0.47 (n=32; cal. 0.27 / 0.18) | 0.67 / 0.63 (n=3; cal. 1.00 / 0.68) | 1.00 / 0.96 (n=7; cal. 0.80 / 0.56) | 1.00 / 0.78 (n=18; cal. 1.00 / 0.70) | 0.94 | 0.08 / 0.92 | 0.016 |

- **Small lesions are missed.** Only 22% of lesions under 50 px are detected on holdout
  (8% on calibration). Final LPIPS alone detects more of them (44%) but with 3.6×
  more false-positive pixels.
- **Large lesions are found but under-segmented:** pixel recall is about 0.6. This
  points to the reconstruction or the phase-2 mask, which may not cover the whole
  lesion. It is not a detection problem.
- **False positives:** 99% are inside the brain, mostly at mid-to-high brightness.
  Speckle is only 2–3%. This explains why brain masking, median filtering and
  small-component removal cannot help. The brightness gate removes most of Final
  LPIPS's dark false positives (90% of them sit below brightness 0.5).

**B and C. Pooled-83 contrasts and ensembles** (Δ = arm − frozen hybrid):

| Arm | Cal. AP | Holdout AP | Δ holdout AP − frozen | Pooled-83 AP | Δ pooled AP − frozen | Holdout pixel AUROC | Holdout Dice | Pooled image AUROC | Δ pooled image AUROC |
|---|---:|---:|---|---:|---|---:|---:|---:|---|
| frozen_hybrid | 0.496 | 0.656 | – | 0.588 | – | 0.979 | 0.588 | 0.830 | – |
| brightness | 0.147 | 0.390 | [-0.426, -0.109] | 0.272 | [-0.429, -0.187] | 0.950 | 0.268 | 0.556 | [-0.427, -0.139] |
| final_t300 | 0.313 | 0.312 | [-0.449, -0.153] | 0.312 | [-0.391, -0.110] | 0.949 | 0.371 | 0.833 | [-0.090, +0.086] |
| final_t400 | 0.306 | 0.348 | [-0.413, -0.131] | 0.332 | [-0.367, -0.097] | 0.953 | 0.392 | 0.827 | [-0.089, +0.083] |
| hybrid_t400 | 0.491 | 0.668 | [-0.024, +0.045] | 0.597 | [-0.034, +0.046] | 0.981 | 0.587 | 0.834 | [-0.042, +0.050] |
| slice_rank | 0.682 | 0.663 | [-0.090, +0.105] | 0.666 | [-0.035, +0.213] | 0.985 | 0.594 | 0.880 | [-0.046, +0.143] |
| normal_cdf | 0.466 | 0.648 | [-0.057, +0.035] | 0.568 | [-0.052, +0.012] | 0.980 | 0.579 | 0.819 | [-0.036, +0.015] |
| seed_ensemble | 0.561 | 0.711 | [+0.027, +0.103] | 0.646 | [+0.033, +0.092] | 0.982 | 0.635 | 0.861 | [-0.018, +0.086] |
| noise_ensemble | 0.527 | 0.677 | [-0.007, +0.052] | 0.615 | [+0.006, +0.052] | 0.982 | 0.600 | 0.869 | [+0.006, +0.080] |

| Pre-specified contrast | Pooled-83 Δ AP 95% CI | Resolved |
|---|---|:---:|
| final_t400 - final_t300 | [-0.011, +0.042] | no |
| hybrid_t400 - frozen_hybrid | [-0.034, +0.046] | no |
| slice_rank - frozen_hybrid | [-0.035, +0.213] | no |
| normal_cdf - frozen_hybrid | [-0.052, +0.012] | no |

- **Both ensembles pass the pre-registered rule.** The seed ensemble (mean of 3 sampling
  seeds) is the stronger one: holdout AP 0.711 vs 0.656 (Δ CI [+0.027, +0.103]),
  calibration 0.561 vs 0.496, pooled 0.646 vs 0.588 (Δ [+0.033, +0.092]), Dice 0.635. The
  noise-level ensemble (t200–t500) passes more narrowly (pooled Δ [+0.006, +0.052]) and
  also raises pooled image AUROC (Δ [+0.006, +0.080]). This supports the hypothesis
  that sampling noise limits the hybrid.
- **Pooling all 83 slices did not resolve the ambiguous contrasts.** t400 vs t300 and
  the slice-rank gate (Δ [−0.035, +0.213]) remain undecided. More data, not a
  different split, is needed to decide them.
- Cost: the seed ensemble needs 3 full DDIM20+30 runs per image, so test inference
  takes 3× as long. It has not been evaluated on test. A test run would be the sixth
  test access and needs a team decision.

## 4. Learned non-diffusion baselines — validation holdout only

| Model / score | Holdout pixel AP | Holdout Dice | Holdout image AUROC |
|---|---:|---:|---:|
| ConvAE z256, residual L1 | 0.1173 | 0.2033 | 0.5088 |
| ConvVAE z256, residual L1 | 0.1486 | 0.2311 | 0.5154 |
| ConvVAE z256, spatial LPIPS | 0.0463 | 0.0369 | 0.4189 |
| Final LPIPS (diffusion) | 0.3123 | 0.3712 | 0.9145 |

ConvAE/ConvVAE checkpoint selection used normal slices from both validation
halves; these rows are development comparisons, not independent holdout
results. No ConvAE/ConvVAE test evaluation exists.

## 5. Open items for the advisor's priorities

1. Patient-group paired bootstrap: done on validation for the sweep, its t300
   controls, mask rules, fusion, seeds/budget/phase-2 variants and the LPIPS-only
   mask (6–14 Nov). Test-set intervals for the frozen four arms: done 14 Nov
   (Section 1). Not done for the Section 3 alternative scores (negative
   results) or for the exploratory LPIPS-only mask/DDIM50 test rows.
2. Noise level → reconstruction quality → detection: done descriptively from
   the cached per-t arrays (13 Nov, Section 2); no intervals for these aggregates.
3. Sweep completion done 14 Nov: t50/t400, DDIM budget, phase-2 start and
   sampling seeds 18/19 at t300 and t400. Mask-rule ablation done (13 Nov,
   Section 1); weighted LPIPS+residual fusion on the Final reconstruction done
   (14 Nov, Section 3).
4. Hybrid follow-up (14 Nov): LPIPS × brightness passes the pre-registered
   rule on validation, and the gate ablation credits the re-sampling: Initial
   LPIPS × brightness is lower beyond uncertainty, and L1 gates are far lower
   (Section 3). Frozen and evaluated once on test (fifth access, 14 Nov): pixel
   AP 0.651, beating Final and brightness beyond uncertainty (Section 3).
5. No further test runs until a single configuration is frozen from
   validation; report the three post-freeze test uses as exploratory. The
   14 Nov CI run reused the frozen four-arm protocol unchanged, to quantify
   uncertainty only. It is a fourth access to the test set, and no selection
   was made from it.
6. Source for the 11–12 November runs is only in private source snapshots;
   see `BLK-20261113-01` in ISSUES.md.
