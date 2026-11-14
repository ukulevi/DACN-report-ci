# Frozen final-test results — 10/11/2026

Status: **completed, uploaded**,3665/3665test slices (640normal,3025abnormal).
Method/checkpoint/sampler,4arms, thresholds and seed policy frozen before test;
no threshold refitting or method selection after reading test. This is the final
held-out test, separate from repeatedly inspected43validation holdout slices.
One trained model seed17 and one UUID-derived sampling seed policy.

Dataset revision350c74f2313e4e73ca65e7ffc6e9dcf74040b3b7;
checkpointd3e06d2082e18234899b201718abd59078c8a800d66da3cc5d8b9e0d9ca9b946.
Frozen implementation commit0d3a1aa, source snapshot
f4f037fe9fef6fefc4d41cb2dafd94512d8caef0650641671a194480d454c02c.
New Modal T4 runtime independently accepted against historical validation;
not claimed identical to old Colab. Thresholds and native LPIPS states unchanged.

## Pixel-level global micro, full240crop

AP is average precision; Dice equals micro F1. Thresholds from original40
validation calibration, score>=threshold. No foreground filtering.

| Arm | ROC-AUC | AP | Precision | Recall | F1/Dice | Macro lesion-slice Dice | Normal pixel FPR |
|---|---:|---:|---:|---:|---:|---:|---:|
| Final LPIPS | .938799 | .318226 | .356107 | .372863 | .364293 | .291848 | .009782 |
| Initial LPIPS | .920021 | .224625 | .267751 | .262345 | .265020 | .210363 | .014741 |
| Naive-stitch LPIPS | .931432 | .269516 | .298738 | .376180 | .333016 | .280682 | .019917 |
| Brightness | .961536 | .511434 | .284699 | .778848 | .416977 | .544774 | .107685 |

## Image-level, mean top576pixels

| Arm | ROC-AUC | AP | Precision | Recall | F1 |
|---|---:|---:|---:|---:|---:|
| Final LPIPS | .805989 | .945863 | .873994 | .933223 | .902638 |
| Initial LPIPS | .721256 | .913943 | .849675 | .951074 | .897520 |
| Naive-stitch LPIPS | .719082 | .912759 | .854599 | .952066 | .900704 |
| Brightness | .643650 | .901341 | .831881 | .978182 | .899119 |

## Interpretation and limits

Final LPIPS improves pixel AP/Dice over initial and naive-stitch controls on this
cohort, and has the highest image ROC-AUC among the four arms. Brightness has
higher pixel ROC-AUC/AP/Dice; do not claim diffusion beats brightness overall.
Final's normal-pixel FPR is lower at the separately frozen thresholds; discuss
precision/recall trade-offs rather than selecting new operating points from test.

Abnormal-image prevalence3025/3665=82.54%; image F1 alone is misleading. Algebraic
all-positive image F1 from these class counts is6050/6690=.904335, slightly above
Final LPIPS .902638. This is an analytic class-count comparison, not a new model
run or test-driven tuning. One seed, MRI2D/one contrast and rendered PNGs limit
generalization; no clinical diagnostic claim or unmeasured confidence interval.

## Runtime, continuation and artifacts

183250U-Net/10995LPIPS calls correspond to completed-output trajectories;
additional uncommitted work at cancellation is not measured in these counts.
Adapter time summed from stored shards6439.784976s, median1.846989s/image,
p952.006275s/image. Includes cold loads at session boundaries and scores/transfers;
excludes data/checksum/shard I/O, old-shard verification and global metrics.
Peak CUDA allocated112928768B on the last T4 process; not reserved/total GPU RAM.
Do not treat adapter sum as total elapsed job time.

Original14400s timeout saved2631shards. First continuation reached2649before
premature operator cancellation; that mistake and evidence are preserved.
Second continuation reverified/reused existing outputs and completed3665samples.
Same run/attempt/protocol and frozen inputs, no new training or seed.
Inference last progress16:28:12+07; terminal aggregate completed
2026-11-10T09:53:27.113391Z, then uploaded. Scalar metrics use every test pixel
with exact ties, never averages of shard AUC/AP. Display ROC/PR curves are
subsampled for plotting, without changing scalar metrics.

- [Private HF aggregate/provenance/curves](https://huggingface.co/dacn-bmad-team/dacn-runs/tree/ddc715a7d17fbc00e176068c1d1ef91ae9e7835b/evaluations/final4-test-20261110-b)
- [Aggregate evidence](evidence/2026-11-10-fourarm-test.json)
- [Terminal manifest](evidence/2026-11-10-fourarm-test-manifest.json)
- [Freeze/acceptance](FINAL_EVALUATION_2026-11-10.md)

Producer aggregate SHA verified against terminal manifest. Private per-sample
shards and masks remain on shared Modal Volume. Report package in ignored
runs/final-20261110/report includes PNG/PDF charts, training curve, report Markdown
and private DOCX with validation high/low-Dice examples; no images committed.
A4/Times New Roman is a general layout; school template has not been supplied.

## Addendum 14/11/2026 — patient-group CIs and the post-freeze hybrid

The 10/11 record above is unchanged. Two later test analyses use the same test split
and must be reported with their access history. Details:
[ABLATION_SUMMARY §1 and §3](ABLATION_SUMMARY_2026-11-13.md),
[RESULT_TRACKING](RESULT_TRACKING.md).

1. **Patient-group CIs for the four frozen arms** (`test-ci-full-20261114T042202Z`). The
   frozen protocol was recomputed for uncertainty only, with no change, and reproduced
   the table above within 5.2e-7. 767 groups, 1000 replicates. Final LPIPS pixel AP 0.3182
   [0.3031, 0.3333]. Final beats Initial and Naive-stitch on every metric beyond
   uncertainty. Brightness keeps higher pixel AP (Δ [+0.161, +0.228]); Final keeps higher
   image AUROC.
2. **Final LPIPS × brightness** (`hybrid-test-full-20261114T105717Z`). The configuration
   and its thresholds were frozen on validation (code `a5911a35`) after several
   development rounds on 14/11 (validation hybrid and gate ablations). It was evaluated
   once on test, the fifth test access, with no fitting on test. Result:

   | Arm | Pixel AP [95% CI] | Pixel AUROC | Dice | Image AUROC | Normal pixel FPR |
   |---|---|---:|---:|---:|---:|
   | **Final LPIPS × brightness** | **0.6505** [0.6263, 0.6731] | **0.9759** | **0.5943** | **0.8397** | 0.0101 |
   | Final LPIPS | 0.3182 [0.3031, 0.3333] | 0.9388 | 0.3643 | 0.8060 | 0.0098 |
   | Brightness | 0.5114 [0.4830, 0.5431] | 0.9615 | 0.4170 | 0.6436 | 0.1077 |

   Pixel AP, Dice, image AUROC and image AP are higher than both Final LPIPS and
   brightness beyond uncertainty. This replaces the statement above that diffusion does
   not beat brightness, **for this hybrid only**. Final LPIPS alone still does not beat
   brightness at pixel level. Validation shows that both the masked re-sampling
   (Initial LPIPS × brightness 0.530 vs 0.656 holdout AP) and LPIPS (L1 × brightness
   ≈ 0.18) contribute.

3. **Seed ensemble of the hybrid** (`ens-test-full-20261114T163509Z`). The mean of
   Final LPIPS over sampling seeds 17/18/19, × brightness. It was chosen on validation by
   a rule fixed beforehand and frozen in code `df81f801`, then evaluated once on test (the
   sixth test access). Pixel AP **0.7009** [0.6769, 0.7242], pixel AUROC 0.9805, Dice 0.6290,
   image AUROC 0.8782, normal-slice FPR 0.0072. Every metric is better than the
   single-seed hybrid beyond uncertainty (AP Δ [+0.045, +0.055]). It costs three
   reconstructions per image.

Scope: one trained model, one sampling seed, one cohort. The test set had been
inspected before the hybrid was frozen, so the hybrid result is a post-freeze
evaluation, not an untouched hold-out confirmation.
