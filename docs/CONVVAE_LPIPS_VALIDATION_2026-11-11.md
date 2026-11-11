# ConvVAE LPIPS validation — 11 November 2026

Completed83/83cached input/reconstruction pairs; bestepoch15 normal-validation checkpoint, latent256, one training seed17. No retraining or new VAE calls. Raw spatial LPIPS0.1 SqueezeNet on grayscale repeatRGB240crop[-1,1], normalizeFalse, clamp0; no residual fusion or foreground masking. Registered40calibration/43patient-disjoint internal-validation partition; max-F1 pixel/image thresholds fitted on40only, persisted before43metrics.

**Limitation:** normal-validation checkpoint selection includes both partitions; exploratory development validation, not independent model holdout. No test access. Prior Residual/SSIM rows below use exactly the same cached reconstructions and membership. No post-test configuration reselection is performed.

## 43 validation images after calibration

Pixel-level (global micro240crop)

| Score | ROC-AUC | Precision | Recall | F1 | AP | Threshold |
|---|---:|---:|---:|---:|---:|---:|
| Residual / L1 | 0.907751 | 0.183201 | 0.312942 | 0.231108 | 0.148551 | 0.353132248 |
| Local SSIM | 0.899503 | 0.171834 | 0.268377 | 0.209519 | 0.132878 | 0.998757358 |
| Spatial LPIPS | 0.784404 | 0.024985 | 0.070223 | 0.036856 | 0.046261 | 0.478287041 |

Image-level (top576mean)

| Score | ROC-AUC | Precision | Recall | F1 | AP | Threshold |
|---|---:|---:|---:|---:|---:|---:|
| Residual / L1 | 0.515351 | 0.585366 | 1.000000 | 0.738462 | 0.638061 | 0.569602118 |
| Local SSIM | 0.548246 | 0.571429 | 1.000000 | 0.727273 | 0.622346 | 1.034944747 |
| Spatial LPIPS | 0.418860 | 0.611111 | 0.916667 | 0.733333 | 0.487240 | 0.523297736 |

## 40 calibration images (threshold fitting)

Pixel-level (global micro240crop)

| Score | ROC-AUC | Precision | Recall | F1 | AP | Threshold |
|---|---:|---:|---:|---:|---:|---:|
| Residual / L1 | 0.943612 | 0.253746 | 0.488125 | 0.333911 | 0.213604 | 0.353132248 |
| Local SSIM | 0.947505 | 0.233707 | 0.452998 | 0.308339 | 0.206173 | 0.998757358 |
| Spatial LPIPS | 0.854490 | 0.068351 | 0.318717 | 0.112562 | 0.057834 | 0.478287041 |

Image-level (top576mean)

| Score | ROC-AUC | Precision | Recall | F1 | AP | Threshold |
|---|---:|---:|---:|---:|---:|---:|
| Residual / L1 | 0.515000 | 0.527778 | 0.950000 | 0.678571 | 0.502456 | 0.569602118 |
| Local SSIM | 0.510000 | 0.500000 | 1.000000 | 0.666667 | 0.552954 | 1.034944747 |
| Spatial LPIPS | 0.610000 | 0.558824 | 0.950000 | 0.703704 | 0.579382 | 0.523297736 |

On43validation LPIPS has lower pixel AP/F1 than both existing scores and image AUROC0.41886, below0.5. Do not invert the score post-hoc or interpret high image F1 as strong discrimination; false positives remain substantial. LPIPS is not an improvement for these fixed ConvVAE reconstructions. This is descriptive evidence from one seed, not a significance claim.

[Modal](https://modal.com/apps/phutungcogioihaitan/main/ap-4rJW8bMKp9yfo91i0D44M9), [private immutable artifacts](https://huggingface.co/dacn-bmad-team/dacn-runs/tree/6232ff98127d92f09afa4d389366bcdaad9b61e9/evaluations/convvae-lpips-val83-20261111-a). Exact sourcec1643be2fac59d61d78fea4f0813f760470121f4e3fcca8fa6ae0e427ae74d59, pinned dataset350c74f2313e4e73ca65e7ffc6e9dcf74040b3b7, checkpointb85619af1f53f74f09136c26c92e5ac960f9ebdea7d84329cc582c1d56f0ab8e. Runtime and LPIPS state hashes, macro lesion Dice, empty-mask FPR, latency/VRAM, original thresholds and aggregate manifests in [safe evidence](evidence/2026-11-11-convvae-lpips-validation.json). Producer aggregate SHA verified; private maps stay remote. Scoring1.514766s excludes metric/I-O/upload, peak allocated VRAM45882880B.
