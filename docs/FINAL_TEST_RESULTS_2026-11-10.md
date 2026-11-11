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
