# DACN roadmap: perceptual scoring and mask-guided diffusion

User-selected2026-11-06. Keep Gaussian DDPM EMA30 central and preserve the
existing preprocessing/data/sampling/calibration protocol. No retraining now.

1. **Active: explanatory scoring ablation.** Measure LPIPS-only and inspect
   existing maps. Separate perceptual distance from residual weighting using the
   same image/reconstruction pairs. This is not a new reconstruction experiment.
2. **Active: pseudo-mask error analysis.** Describe missed tumor pixels and false
   positive locations, with private map panels and aggregate region diagnostics.
3. **Later: one specific mask-guided diffusion change.** Specify exactly which
   mask construction/expansion rule changes and why, using literature, normal
   statistics or development/calibration evidence. Freeze algorithm, parameters,
   selection rule and primary metric before comparative evaluation. No redesign
   or parameter selection based on ranking designs on the inspected holdout.
   The current audit implements no new mask method and runs no re-sampling.
4. **Later: controlled comparison and independent confirmation.** Compare with
   Gaussian DDPM and the AutoDDPM adaptation, retaining naive-stitch controls to
   isolate the incremental role of re-sampling. Lock code/checkpoint/calibration
   thresholds/seeds/metric definitions before an independently confirmed result.

The present43image holdout has been examined repeatedly, including before this
roadmap. It is exploratory evidence, even if no additional tuning is performed.
Changing the random split or using another inference seed does not make these
same images an independent validation cohort. Repeated seeds measure sampling
variability, not independent data confirmation. Keep the3665image final test
sealed during method development. A final authorized evaluation after the full
method is frozen, or a genuinely unseen patient-disjoint external cohort with
compatible labels/preprocessing, can provide independent confirmation. Do not
use such results to select another design and still call them a final test.

## Frozen step1/2 audit protocol (before its GPU runs)

Reuse existing83sample Gaussian EMA30 ancestral t300 baseline and AutoDDPM LPIPS
artifacts from`autoddpm-lpips-full83-t4-20261106T061833Z`;verify cache/index/result
checksums and exact image/initial reconstruction/mask/label ordering. Original
results SHA`6fb277491308c63b2cc52fa614fc912486266a9c3322f7f099aff850cb51a7e0`,
private maps SHA`f4b37638c5f6b2aca92ecaa072402847e4f18f371b4c4b178483d50ef1f4bce2`.
Require the same frozen LPIPS state SHA
`18fed4d1fc5c04f2918f64df3a2463344108576d1888f0d75538ca6d5c4dffe6`.

Two **new predeclared score arms**:

- LPIPS-only: spatial SqueezeNet0.1 perceptual distance on240crop, grayscale
  repeatedRGB, input[-1,1],normalize=False. Clamp any negative numerical output
  to0, matching the existing product definition; no image-wise rescaling.
- Normalized-absolute-only:clip(abs(input-initial)/per-image95thpercentile,0,1),
  exactly the residual term used by the existing product. No tumor labels enter
  either score. All8original arms remain reported controls, not reselected winners.

Recompute the product from the two terms and require max error<=1e-7 against
every selected cached product map. Replay the original pseudo-mask threshold
0.13774138689041138 and dilation3 and require exact binary mask equality.
No new DDPM U-Net calls, reverse chain, re-sampling, training, masks or timestep
search. Full audit performs83LPIPS image-pair calls only. Bounded2image smoke
first verifies replay without measuring AP.

Use identical firstStratifiedGroupKFold(2,shuffle=True,random_state17) partition:
40calibration/43group holdout. Fit both new score pixel/image maxF1 thresholds
on40calibration;persist before holdout metrics. Continuous scores for pixelAP,
micro240crop including background;image mean top576pixels. Record pixel/image
AUROC/AP/P/R/F1/Dice and empty-mask normal pixelFPR. These explain contributions
on the current cohort, not confirmed independent gains. No weight optimization.

Diagnostic bins fixed before the run, used only to **describe errors**:

- Lesion geometry:distance-to-lesion-boundary<=3pixels vs core>3pixels.
- Eight-connected lesion components:areas1–25,26–100,101–400,>400pixels;
  record wholly missed component counts and macro component recall.
- Image-derived disjoint proxies:background(input<=-0.95),foreground boundary
  within5pixels of background/cropedge,remaining foreground with high gradient,
  remaining foreground. Gradient=magnitude of reflect-Sobel/8,high threshold
  pooled95thpercentile overforeground of20normal calibration slices only.
  These are **not anatomical labels**. Cannot identify a named brain structure
  or tissue type from these counts alone;normal here means no annotated lesion.
- For pseudo-mask and calibrated predictions of LPIPS-only,normalized residual,
  existing initial product and existing weighted final residual,report FP counts,
  within-region FP rate,share of total FP and normal-image-only FP rate, separately
  oncalibration andholdout. Also lesion boundary/core misses and component bins.
  Ground-truth masks enter these post-hoc diagnostics only,never inference/fitting
  of gradient or pseudo-mask. Do not translate a diagnostic bin into a new mask
  rule tuned on the holdout. Report different operating-point recalls honestly.

Generate4private panels: firstnormal and firstabnormal sample by existing order
within each partition,not top/worst selected by holdout outcomes. Panels show
input,initialDDPM,LPIPS-only,normalized residual,product,pseudo-mask,TP/FP/FN
overlay and cached finalweighted residual. Auto color scales illustrate shape;
they are not evidence of equal score magnitudes. Retain arrays for exact analysis.
Private images and identifiers never enter Git or local aggregate downloads.
Producing panels is not equivalent to expert anatomical visual review.

## Commands and provenance

```bash
.venv/bin/modal run --profile phutungcogioihaitan --detach \
  src/dacn/cloud/modal_perceptual_audit.py::run_perceptual_audit_remote \
  --evaluation-id perceptual-audit-smoke-UNIQUE \
  --code-revision GIT_HEAD_SHA --working-tree-dirty --mode smoke
```

After successful smoke use fresh`perceptual-audit-full83-UNIQUE` and`--mode full`.
T4 timeout30minutes,private team Volumes/HF,exact source snapshot and hashes.
Fit/score/mask arrays remain private;only aggregate JSON is downloaded. Fresh IDs
only. Record actual attempts inRESULT_TRACKING;check full results on user request.
No step3 method or independent confirmation run is authorized by this audit itself.

## Actual attempts

- Smoke`perceptual-audit-smoke-t4-20261106T065148Z` completed/uploaded private
  revision`402250a5f04037a03881be94cabff9ef8a82e09b`;2pairs,original product
  replay maxerror0,pseudo-maskexactmatchTrue,aggregate result checksum verified.
  Total5.726s before upload,peak45882880B,no AP. Source snapshot
  `4689409edc64b047b0822dfe966e377fda4e0fcd4c4f398eab19a11365f2e297`.
- Full`perceptual-audit-full83-t4-20261106T065328Z` launched detached after smoke:
  [Modal app](https://modal.com/apps/phutungcogioihaitan/main/ap-VmGU48SYKuWh8W685WGCf6).
  Completed/uploaded private revision`3b82d8b2bbfa62d887dbb2f687a790905c351113`.
  Total25.922s before upload,peak45882880B,83pairs,zeroDDPM calls.
  Product replayerror0 and exact pseudo-mask equality across83. ResultSHA
  `224e17448d671bf96de619efe4196a1d0a2992bc07e29915a7c0499df63ba0fa`
  verified against manifest. No new mask method chosen;independent confirmation
  remains a later frozen-protocol evaluation.

## Step1/2 measured findings

Metrics below use the same43group holdout with40calibration thresholds. The
two new arms use the same LPIPS state and image/reconstruction inputs as the
existing product. No new mask, reconstruction or model was selected.

| Score | Pixel AP | Micro Dice | Recall | Normal FPR | Image AP |
|---|---:|---:|---:|---:|---:|
| Gaussian absolute | 0.101066 | 0.173678 | 0.314702 | 0.058079 | 0.430006 |
| Normalized absolute only | 0.102264 | 0.167640 | 0.299978 | 0.057595 | 0.558140 |
| LPIPS only | 0.266899 | 0.339668 | 0.422469 | 0.007686 | 0.868242 |
| LPIPS x normalized absolute | 0.204298 | 0.274354 | 0.280623 | 0.010727 | 0.887665 |
| Weighted final re-sampled residual | 0.154785 | 0.226256 | 0.290408 | 0.026640 | 0.814398 |
| Brightness | 0.389506 | 0.268051 | 0.788577 | 0.134935 | 0.770309 |

LPIPS-only improves pixelAP over both product and weightedfinal. The same ordering
occurs oncalibration:0.316582 >0.285192 >0.196476. This isolates score composition
on this fixedcheckpoint/cohort and supports a perceptual-scoring contribution;
it does not establish independent generalization or better restoration. Product
imageAP is still higher than LPIPS-only. Normalized-absolute image score is
constant1 because its top5percent pixels are clipped1 while aggregation uses
only top1percent;therefore imageAUROC0.5 is expected. Do not adopt that as an
image detector. Brightness retains higherpixelAP despite lowerF1 at its separate
threshold;different recall prevents a matched-sensitivity claim.

Pseudo-mask misses9048/55694lesionpixels(16.2459percent):

| Lesion geometry | Missed / eligible pixels | Within-region miss rate | Share of all misses |
|---|---:|---:|---:|
| Boundary within3pixels | 2497 / 13693 | 18.2356% | 27.5973% |
| Core over3pixels | 6551 / 42001 | 15.5972% | 72.4027% |

Misses are not just around the boundary. The core's larger absolute contribution
does not mean a higher within-core miss probability. Pseudo-mask wholly misses
3/29connectedcomponents area1–25pixels and3/6area26–100;no whollymissed component
in3area101–400 or22area>400. These are2D connected label components,not counts
of independent tumors/patients. LPIPS final calibrated detection threshold misses
20/29smallestcomponents;better AP does not solve small-lesion sensitivity.

| Pseudo-mask FP proxy region | FP pixels | Share of all FP | Within-region normal-image mask rate |
|---|---:|---:|---:|
| Background | 17424 | 5.7504% | 0.6951% |
| Foreground boundary5pixels | 53260 | 17.5772% | 43.2058% |
| Interior highgradient | 5613 | 1.8524% | 78.0517% |
| Other foreground interior | 226710 | 74.8201% | 41.8644% |

FP counts are dominated by otherforegroundinterior,not solely sharp edges. The
highgradient proxy has high rate over a small area. Region sizes matter;do not
interpret FP share as a conditional error probability. Foreground>-0.95 and
gradient regions are heuristics;no anatomy labels support statements about a
specific structure. Four fixed example PNG panels were produced and retained
privately for team review;this aggregate check did not visually inspect or
download them. Counts/strata consistency and replay checks passed.

Full requiredpixel/image AUROC/P/R/F1 and provenance are inRESULT_TRACKING.md.
No step3 design is selected by the holdout ordering. Any later construction or
expansion rule needs a fixed development/calibration criterion and independent
confirmation after method freeze,with the finaltest remaining sealed until then.
