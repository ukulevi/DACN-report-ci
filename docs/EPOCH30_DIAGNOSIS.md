# Gaussian epoch 30: validation diagnosis

## Scope and provenance

User-requested diagnosis on 2026-11-02, using Gaussian run
`ddpm-partial-t4-b8-e30-cont-20261102T034644Z`, EMA epoch30, ancestral reverse,
sampling seed17, dataset revision `350c74f2313e4e73ca65e7ffc6e9dcf74040b3b7`.
All measurements below use the 83 validation slices, not held-out test.
The t*=300 cache is from completed evaluation
`ddpm-val-t4b8-e30-t300-20261102T071516Z`, SHA256
`8278ad18543dd87f698c36b510564b9fcd4445d4a1b0ae5d7a859093e2a4e959`.
The diagnostic script reproduces the recorded AP/AUROC/F1 on this cache.
Only aggregate results are recorded here. Private arrays stay in ignored runs/.

## Completed CPU experiments on the t*=300 cache

All metrics keep the full 240x240 evaluation ROI. Thresholds are separately
selected by validation-max-F1, with the registered largest-threshold tie break.
This is exploratory validation analysis, not frozen test performance.

| Score | Pixel AUROC | Pixel AP | Precision | Recall | F1/Dice |
|---|---:|---:|---:|---:|---:|
| Absolute reconstruction difference | 0.884132 | 0.099834 | 0.118175 | 0.404081 | 0.182870 |
| Local one-minus-SSIM | 0.895647 | 0.100741 | 0.107193 | 0.382770 | 0.167483 |
| Input brightness only, no reconstruction | 0.943925 | 0.272325 | 0.265567 | 0.302829 | 0.282976 |
| Absolute difference with foreground gating | 0.903959 | 0.108038 | 0.124594 | 0.404091 | 0.190463 |
| SSIM with foreground gating | 0.900629 | 0.103202 | 0.107484 | 0.403228 | 0.169726 |

Brightness is rendered PNG intensity, not native MRI intensity. This simple
control is an empirical finding on these validation images, not a proposed
clinical detector or evidence of generalization. It shows that the current
reconstruction discrepancy is a poorer ranking signal than input brightness.

Foreground is only a diagnostic heuristic: input > -0.95. Interior means distance
from its boundary >5 pixels; it is not an anatomical brain segmentation.
Lesion prevalence is 1.9611% of all ROI pixels. Only 1 of 93,756 lesion pixels is
outside this foreground, so this gating did not substantially drop annotated
lesions. Outside-foreground scores are lowered below all inside scores; metric
ROI stays full size. Gating is not silently substituted into the baseline.

| Region | Mean absolute difference | Mean SSIM score | Share of absolute-difference FP | Share of SSIM FP |
|---|---:|---:|---:|---:|
| Annotated lesion | 0.221920 | 0.799976 | N/A | N/A |
| Nonlesion foreground interior | 0.139206 | 0.703140 | 62.89% | 83.47% |
| Nonlesion foreground boundary (5px) | 0.231605 | 0.596670 | 31.26% | 14.81% |
| Nonlesion background | 0.010581 | 0.028354 | 5.85% | 1.72% |

At the original SSIM threshold, 25.40% of nonlesion interior pixels are called
positive, versus 38.28% recall on lesion pixels. SSIM score 10/50/90 percentiles
are 0.429/0.802/1.143 for lesions and 0.355/0.694/1.045 for nonlesion interior.
The substantial overlap limits pixel ranking and threshold discrimination.
For absolute difference, mean boundary error exceeds mean lesion error.

## Interpretation and limits

- Directly supported: normal-structure reconstruction errors contribute many
  false positives; background removal alone provides only small AP/F1 gains.
- Directly supported: low lesion prevalence plus overlapping lesion/nonlesion
  scores produces low precision. Prevalence alone is not a sufficient excuse:
  the brightness control performs better under exactly the same class balance.
- Not established: whether the primary cause is excessive corruption at t*=300,
  learned weights, accumulated sampling error, limited architecture or the
  reconstruction objective. Cached final images cannot isolate these causes.
- No lesion-free target exists for abnormal inputs; these diagnostics cannot
  measure true healthy-restoration accuracy. Raw noise MSE does not evaluate it.
- One training/sampling seed and no patient-bootstrap intervals. Do not infer
  statistical significance or compare directly with published BMAD test numbers.

## Controlled GPU experiment launched next

Hold checkpoint, EMA, seed, validation samples, ancestral sampler, score and
threshold rule fixed; vary only t* to 100 and 200. Existing t*=300 is reused.
Expected inference-plus-score cost is roughly one prior t*=300 run (~12 minutes),
plus setup and scoring overhead; this is an estimate, not measured new runtime.
Private reconstruction caches are enabled for subsequent paired analysis.

Evaluation ID: `ddpm-val-e30-t100-200-diag-20261102T094115Z`.
If smaller t* improves AP and reduces nonlesion errors, excessive corruption is
supported as a contributing factor. If AP stays low, lowering t* alone is
insufficient; this still does not isolate a model-capacity failure.
Completed validation results (same checkpoint, score definition and seed):

| t* | SSIM AP | SSIM F1 | Absolute difference AP | Absolute difference F1 |
|---|---:|---:|---:|---:|
| 100 | 0.076971 | 0.128203 | 0.070977 | 0.134604 |
| 200 | 0.088429 | 0.141339 | 0.083184 | 0.154103 |
| 300 | 0.100741 | 0.167483 | 0.099834 | 0.182870 |

Lowering t* alone does not resolve the ranking problem; t*=300 is best among
these tested settings for both AP and F1. This does not prove that higher t*
will improve results or that every reconstruction method fails. Further
diagnosis should address the score/normal-structure errors rather than assume
that excessive t*=300 corruption is the sole cause. No new training is justified
by these results alone. Uploaded evaluation revision `67626af0`; complete
metrics/runtime/thresholds are recorded in RESULT_TRACKING.

## Reproduce CPU diagnostics

```bash
.venv/bin/python scripts/diagnose_ddpm_cache.py \
  --cache runs/visualizations/ddpm-e30-t300/reconstructions.npz \
  --expected-sha256 8278ad18543dd87f698c36b510564b9fcd4445d4a1b0ae5d7a859093e2a4e959 \
  --output runs/visualizations/ddpm-e30-t300/diagnostic_summary_new.json
```

The script refuses an existing output path and checks the source cache hash.
