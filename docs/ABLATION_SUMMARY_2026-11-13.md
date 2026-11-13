# Consolidated Final LPIPS ablation evidence — 13 November 2026

Purpose: one table set answering the advisor's three priorities (role of the
residual mask, noise level/DDIM steps, ablation and baselines) from runs that
already exist. **No new inference, training, threshold fitting or test access
was performed for this summary.** All values are copied from producer aggregate
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
| Brightness control (no model) | 0.1466 | 0.3895 | 0.2681 | **0.5114** | **0.9615** | **0.4170** | 0.6437 |

Reading on this cohort (one seed, no confidence intervals yet):

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
at t300–500 and collapses at t800; image AUROC peaks at t300. Differences of
about 0.01 AP inside the plateau are smaller than the calibration/holdout
disagreement seen in Section 3 and should not select a configuration without
patient-group confidence intervals.

**Not yet measured:** phase-2 start other than t50; total DDIM budget other
than 50 calls; t400; more than one sampling seed; reconstruction-quality
measures (e.g. normal-slice error) as a function of t.

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
([DDIM20 score fusion](DDIM20_SCORE_FUSION_VALIDATION.md)). Weighted fusion on the Final
reconstruction has not been measured.

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

1. Patient-group paired bootstrap intervals for Sections 1–3, using
   `dacn.evaluation.group_bootstrap` on the fixed maps and thresholds.
2. Noise level → reconstruction quality → detection analysis from the cached
   per-t arrays, if they contain reconstructions.
3. Validation-only completion of the sweep: phase-2 start, total DDIM budget,
   t400 and extra sampling seeds.
4. No further test runs until a single configuration is frozen from
   validation; report the three post-freeze test uses as exploratory.
5. Source for the 11–12 November runs is only in private source snapshots;
   see `BLK-20261113-01` in ISSUES.md.
