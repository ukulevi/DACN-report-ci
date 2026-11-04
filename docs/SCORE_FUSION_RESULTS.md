# Epoch30 score fusion: exploratory group holdout

User-requested experiment completed 2026-11-02 using Gaussian EMA epoch30,
t*=300, ancestral sampling seed17, all83 cached validation reconstructions.
No new GPU inference or training. Held-out test remains unused.

## Procedure fixed before execution

First split from StratifiedGroupKFold(2, shuffle=True, random_state=17), using
patient-group UUIDs from the pinned dataset index. Calibration:40 images,
25 groups,20 normal/20 abnormal. Internal holdout:43 images,25 different groups,
19 normal/24 abnormal. Asserted group disjointness and label/cache alignment.
Cache ordering follows the full-validation evaluator's dataset-index order.

For fusion, map each component through its empirical midrank CDF fitted only on
calibration pixels. Mixture = w*brightness_rank+(1-w)*reconstruction_score_rank,
w in {0.25,0.5,0.75}. Raw baseline scores remain unnormalized. Fit pixel/image
max-F1 thresholds only on calibration; select candidate by calibration pixel AP.
Freeze the candidate/thresholds to JSON before computing holdout metrics.
All metrics use the original240x240 ROI without foreground gating.

## Results

Pixel-level global micro metrics on the same43 holdout images:

| Score | Calibration AP | Holdout AP | AUROC | Precision | Recall | F1/Dice |
|---|---:|---:|---:|---:|---:|---:|
| Brightness only | 0.146595 | 0.389506 | 0.950197 | 0.161469 | 0.788577 | 0.268051 |
| Absolute difference | 0.100985 | 0.101066 | 0.876854 | 0.119934 | 0.314702 | 0.173678 |
| Local SSIM | 0.101263 | 0.102402 | 0.888870 | 0.110476 | 0.257712 | 0.154654 |
| 25% brightness + difference | 0.173266 | 0.190454 | 0.907222 | 0.223038 | 0.306676 | 0.258254 |
| 50% brightness + difference | **0.195268** | 0.262977 | 0.934268 | 0.275226 | 0.377455 | 0.318334 |
| 75% brightness + difference | 0.183504 | 0.320651 | 0.947496 | 0.277565 | 0.493859 | 0.355389 |
| 25% brightness + SSIM | 0.137320 | 0.139217 | 0.911235 | 0.155113 | 0.290714 | 0.202291 |
| 50% brightness + SSIM | 0.157727 | 0.190554 | 0.933237 | 0.194575 | 0.423098 | 0.266563 |
| 75% brightness + SSIM | 0.159537 | 0.272557 | 0.949881 | 0.247022 | 0.550239 | 0.340970 |

The preregistered calibration selector chooses50% brightness+absolute difference,
not the best-looking holdout row. It achieves holdout AP0.262977 /F1 0.318334,
versus brightness AP0.389506 /F1 0.268051. Higher F1 at a frozen calibration
threshold can coexist with lower AP; these measure different properties.
Selected fusion empty-normal FPR=0.033398 and macro lesion Dice=0.407599;
brightness FPR=0.134935 and macro lesion Dice=0.558978. Aggregate JSON retains
all image metrics, exact thresholds and all candidates' macro Dice/FPR.

## Interpretation

### Follow-up: combining reconstruction maps across timesteps

User-requested fixed equal-weight mean of calibration-CDF-normalized maps at
t100/200/300 was tested using the same group split and frozen calibration
threshold rule. Cached images/masks/labels match exactly across all timesteps.

| Method | Holdout AP | Holdout F1 |
|---|---:|---:|
| t300 absolute difference | 0.101066 | 0.173678 |
| Mean rank-normalized difference t100/200/300 | 0.091536 | 0.155345 |
| t300 local SSIM | 0.102402 | 0.154654 |
| Mean rank-normalized SSIM t100/200/300 | 0.092199 | 0.132428 |

Combining these maps equally did not improve either metric. It does not prove
that every multi-timestep method fails; no additional weighting was tuned on
holdout. This differs from the earlier epsilon-error experiment. Script:
`scripts/evaluate_multitimestep_maps.py`; aggregate JSON is in ignored
`runs/visualizations/ddpm-e30-t300/multitimestep_rank_mean_v1/`.
CPU96.510s; no new GPU inference. Exact baseline values reproduced.

Fusion improves over reconstruction alone, but none of these six mixtures beats
brightness-only AP on this holdout. Therefore this experiment does not establish
that diffusion adds useful ranking information beyond brightness. Do not claim
AP improvement by comparing the holdout0.39 with the earlier full-validation0.27:
they are different samples. Calibration brightness AP0.15 versus holdout0.39
also demonstrates substantial group heterogeneity and limited stability.

Full validation was already inspected during prior experiments: this internal
holdout is exploratory, not a pristine final test. One split/seed, no bootstrap
intervals, no claims of significance. Do not silently retune weights on the
holdout. Further confirmation would require a separately registered grouped
cross-validation/nested calibration procedure or an eventual frozen final test.

## Provenance and reproducibility

Dataset revision350c74f2313e4e73ca65e7ffc6e9dcf74040b3b7; source run
ddpm-partial-t4-b8-e30-cont-20261102T034644Z. Cache SHA256
8278ad18543dd87f698c36b510564b9fcd4445d4a1b0ae5d7a859093e2a4e959.
Validation index SHA256
57ad2e31df3ed4e625da00efaecd6fddc1e2a132329dfba6e60867caf1802db3.
Results JSON SHA256
198af731e7c0a79f1f2d16b4b14293bfc2138cf1a5ac4925bd8d4a6df3f56456.
CPU runtime20.628s; inference latency/VRAM N/A (no inference in this experiment).
Private index/arrays/results stay under ignored runs/; no group UUIDs in this report.

```bash
.venv/bin/python scripts/evaluate_score_fusion.py \
  --cache runs/visualizations/ddpm-e30-t300/reconstructions.npz \
  --index runs/visualizations/ddpm-e30-t300/validation_index.csv \
  --output runs/visualizations/ddpm-e30-t300/fusion_group_holdout_new
```

Script refuses existing outputs. Output includes frozen_selection.json and
results.json; no UUIDs or patient-level images are emitted to logs.
