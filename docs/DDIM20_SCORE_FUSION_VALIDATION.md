# Four DDIM20 score sums — validation extension, 11 November 2026

User requests LPIPS+Residual, LPIPS+LocalSSIM, Residual+LocalSSIM and all three.
The following fixed definition is registered before execution. This is an
exploratory validation extension after the completed frozen final test; no test
rerun, model training, parameter/weight sweep or replacement of the final method.

## Inputs and fixed score definitions

Reuse original83validation input/initial-DDIM20 pairs from the same immutable
cache SHA25622656e1f63a0047d3dbe32f7833eea8e303988414e5942dec1ba5a1d9ed1ef19.
Gaussian EMA30, DDIM20 fromt300, eta0, seed17/UUID-derived original sampling.
Dataset revision350c74f2313e4e73ca65e7ffc6e9dcf74040b3b7. Reuse the checksum-pinned
40calibration/43holdout patient-group membership, never derive a new fold.

R is cached initial raw absolute residual (ddpm_l1); L is cached initial spatial
LPIPS (initial_lpips_only). S is existing 1−localSSIM on the same input/initial
reconstruction:11×11Gaussian,sigma1.5,reflect,population moments,range2 signed
[-1,1], full240crop, no clipping or foreground filtering. Computing S uses CPU;
no new diffusion/LPIPS/model inference. Verify original cache/index/partition
bytes before deserialization. Derived arrays and fitted pixel distributions
remain private on the shared Modal Volume and private HF model repository.

## Normalization and sums

For each raw score fit midrank empirical CDF F on pixels of normal calibration
images only (expected20images). Values map to [0,1]; ties use average ranks,
values outside the reference extrema map to0/1. No per-image normalization,
reference updates from holdout, test access or fitted fusion weights.
Let r=F_R(R), l=F_L(L), s=F_S(S). The four maps are:

| Arm | Fixed score |
|---|---|
| LPIPS + Residual | (l+r)/2 |
| LPIPS + Local SSIM | (l+s)/2 |
| Residual + Local SSIM | (r+s)/2 |
| LPIPS + Local SSIM + Residual | (l+s+r)/3 |

Arithmetic means implement equal-weight sums (scale differs only by a fixed
constant); these are not the historical LPIPS×normalized-residual products.
No ranking-based choice of the best arm from43holdout; report all four.

Fit separate pixel/image max-F1 thresholds for each map on40calibration, prefer
largest threshold on exact ties. Save fitted CDF arrays privately, and persist
all thresholds/definitions before computing any new holdout results. Report
image/pixel ROC-AUC, Precision, Recall, F1; pixel AP/micro Dice, macro lesion Dice
and normal pixel FPR. Image score mean top576pixels, score>=threshold, full240crop.
Recompute raw R/L/S aggregate baselines in the same run as integrity controls;
these are the same three score types, not new model training experiments.

Any comparison against raw single scores measures both normalization and fusion,
not a clean causal effect of adding one component. Normal-CDF saturation can
create ties; report actual results without switching normalizers after holdout.
Previously inspected validation and a single seed limit conclusions; do not
claim clinical performance or definitive generalization from these combinations.

## Reproduction

Core: dacn.evaluation.ddim20_score_fusion. CPU Modal entrypoint shares the scorer
lifecycle in src/dacn/cloud/modal_ddim20_local_ssim.py with --score-family fusion.
Fresh evaluation ID, actual code snapshot/version inventory and durable start/
terminal manifests are required. Local controller downloads only aggregate
metrics/manifest; no private maps or original images. Produced results.json is
source of numeric results; none are entered here before run completion.

## Measured results — completed/uploaded
All83validation samples scored;20normal calibration images fit each CDF (1,152,000reference pixels/score). Thresholds use40calibration; the following results use43group-holdout images only. Raw-score integrity controls reproduce36previous image/pixel metric fields within1e-12.

### Pixel global micro on240crop
| Score | ROC-AUC | AP | Precision | Recall | F1 |
|---|---:|---:|---:|---:|---:|
| LPIPS + Residual | 0.917980 | 0.153122 | 0.177131 | 0.279527 | 0.216849 |
| LPIPS + Local SSIM | 0.915213 | 0.157358 | 0.176062 | 0.248375 | 0.206058 |
| Residual + Local SSIM | 0.886712 | 0.100392 | 0.108451 | 0.270388 | 0.154809 |
| LPIPS + SSIM + Residual | 0.911881 | 0.142310 | 0.162471 | 0.247316 | 0.196110 |
| Residual only (raw) | 0.874136 | 0.090936 | 0.101839 | 0.333698 | 0.156053 |
| Local SSIM only (raw) | 0.880550 | 0.095772 | 0.105750 | 0.182641 | 0.133945 |
| Spatial LPIPS only (raw) | 0.926110 | 0.157876 | 0.194958 | 0.336464 | 0.246871 |

### Image mean top576pixels
| Score | ROC-AUC | AP | Precision | Recall | F1 |
|---|---:|---:|---:|---:|---:|
| LPIPS + Residual | 0.734649 | 0.804783 | 0.636364 | 0.875000 | 0.736842 |
| LPIPS + Local SSIM | 0.728070 | 0.792304 | 0.600000 | 0.875000 | 0.711864 |
| Residual + Local SSIM | 0.324561 | 0.455594 | 0.558140 | 1.000000 | 0.716418 |
| LPIPS + SSIM + Residual | 0.692982 | 0.750643 | 0.666667 | 0.750000 | 0.705882 |
| Residual only (raw) | 0.250000 | 0.431879 | 0.558140 | 1.000000 | 0.716418 |
| Local SSIM only (raw) | 0.368421 | 0.493883 | 0.558140 | 1.000000 | 0.716418 |
| Spatial LPIPS only (raw) | 0.787281 | 0.854501 | 0.611111 | 0.916667 | 0.733333 |

### Operating points from calibration only
| Fusion | Pixel threshold | Image threshold | Macro lesion Dice | Normal pixel FPR |
|---|---:|---:|---:|---:|
| LPIPS + Residual | 0.948955729 | 0.964800378 | 0.226063126 | 0.023980263 |
| LPIPS + Local SSIM | 0.955628472 | 0.966236088 | 0.208720980 | 0.021614583 |
| Residual + Local SSIM | 0.925848524 | 0.971618153 | 0.206405070 | 0.057015716 |
| LPIPS + SSIM + Residual | 0.939471065 | 0.969965212 | 0.212168141 | 0.026532346 |

No tested fixed fusion exceeds raw Initial LPIPS on pixel AP or Dice in this cohort. LPIPS+SSIM has the largest AP among the four fusions; LPIPS+Residual has the largest Dice among the four. These are descriptive observations, not a new holdout-selected primary method or reason to retune weights. Residual+SSIM image threshold predicts all43holdout images positive.

Two CPU/8192MiB, no GPU; zero new U-Net/LPIPS calls. CPU SSIM3.944479504s, CDF/combination5.030068275s; evaluator39.880807316s after durable start commit/before upload, including global metrics/I-O. VRAM N/A. One original reconstruction/training seed17.

[Immutable private artifacts](https://huggingface.co/dacn-bmad-team/dacn-runs/tree/78e025c24c4ce327e63200fb66679ab762d2ca79/evaluations/ddim20-fusion-val83-20261111T063345Z) include results, frozen_calibration, source snapshot, private fitted CDFs and maps. [Aggregate evidence](evidence/2026-11-11-ddim20-score-fusion-results.json) and [terminal manifest](evidence/2026-11-11-ddim20-score-fusion-manifest.json) contain safe metrics and provenance only.
