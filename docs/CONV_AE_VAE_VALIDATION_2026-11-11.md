# ConvAE / ConvVAE validation results — 11 November 2026

Both 30-epoch z256 models evaluated using best normal-validation checkpoints: AE epoch13, VAE epoch15. Deterministic mean latent; fixed Residual/L1 and Local SSIM. Completed83/83images per model, no test access. Registered40calibration/43group-separated internal holdout. Thresholds maximize F1 on40calibration only and are persisted before43holdout metrics; no threshold refitting on holdout. One training seed17.

**Limitation:** normal-validation checkpoint selection used normal slices in both partitions. These are development-validation results, not independent model-holdout estimates. Repeatedly inspected validation; no statistical superiority claim.

## 43 validation images after calibration

Pixel-level: global micro240crop; F1 equals microDice.

| Model | Score | ROC-AUC | Precision | Recall | F1 | AP | Threshold |
|---|---|---:|---:|---:|---:|---:|---:|
| ConvAE | Residual / L1 | 0.893576 | 0.141326 | 0.361816 | 0.203259 | 0.117324 | 0.252432883 |
| ConvAE | Local SSIM | 0.879948 | 0.120914 | 0.230079 | 0.158520 | 0.102541 | 0.958538088 |
| ConvVAE | Residual / L1 | 0.907751 | 0.183201 | 0.312942 | 0.231108 | 0.148551 | 0.353132248 |
| ConvVAE | Local SSIM | 0.899503 | 0.171834 | 0.268377 | 0.209519 | 0.132878 | 0.998757358 |

Image-level: mean top576pixels; no foreground masking.

| Model | Score | ROC-AUC | Precision | Recall | F1 | AP | Threshold |
|---|---|---:|---:|---:|---:|---:|---:|
| ConvAE | Residual / L1 | 0.508772 | 0.558140 | 1.000000 | 0.716418 | 0.594447 | 0.450100017 |
| ConvAE | Local SSIM | 0.532895 | 0.600000 | 1.000000 | 0.750000 | 0.596156 | 1.141625242 |
| ConvVAE | Residual / L1 | 0.515351 | 0.585366 | 1.000000 | 0.738462 | 0.638061 | 0.569602118 |
| ConvVAE | Local SSIM | 0.548246 | 0.571429 | 1.000000 | 0.727273 | 0.622346 | 1.034944747 |

## 40 calibration images (threshold fitting)

Pixel-level: global micro240crop; F1 equals microDice.

| Model | Score | ROC-AUC | Precision | Recall | F1 | AP | Threshold |
|---|---|---:|---:|---:|---:|---:|---:|
| ConvAE | Residual / L1 | 0.906986 | 0.133810 | 0.394357 | 0.199819 | 0.107606 | 0.252432883 |
| ConvAE | Local SSIM | 0.915550 | 0.149347 | 0.357574 | 0.210694 | 0.125958 | 0.958538088 |
| ConvVAE | Residual / L1 | 0.943612 | 0.253746 | 0.488125 | 0.333911 | 0.213604 | 0.353132248 |
| ConvVAE | Local SSIM | 0.947505 | 0.233707 | 0.452998 | 0.308339 | 0.206173 | 0.998757358 |

Image-level: mean top576pixels; no foreground masking.

| Model | Score | ROC-AUC | Precision | Recall | F1 | AP | Threshold |
|---|---|---:|---:|---:|---:|---:|---:|
| ConvAE | Residual / L1 | 0.415000 | 0.512821 | 1.000000 | 0.677966 | 0.474590 | 0.450100017 |
| ConvAE | Local SSIM | 0.455000 | 0.513514 | 0.950000 | 0.666667 | 0.465487 | 1.141625242 |
| ConvVAE | Residual / L1 | 0.515000 | 0.527778 | 0.950000 | 0.678571 | 0.502456 | 0.569602118 |
| ConvVAE | Local SSIM | 0.510000 | 0.500000 | 1.000000 | 0.666667 | 0.552954 | 1.034944747 |

Image-level43validation AUROC0.509–0.548 is weak. All image recalls are1 with low precision0.558–0.600, so high F1 alone does not establish useful discrimination. For AE+Residual all43images are predicted abnormal (precision24/43); do not interpret recall1 as a strong detector. Among these fixed rows VAE+Residual has largest pixel AP/F1 on43validation, a descriptive observation only; no test configuration selected here.

Completed/uploaded [Modal](https://modal.com/apps/phutungcogioihaitan/main/ap-2IWmnHdXEkhrjA5ks7WOfU), [private immutable artifacts](https://huggingface.co/dacn-bmad-team/dacn-runs/tree/b3cbb757433195e32415ea178964cdb45ec66604/evaluations/conv-val-full83-20261111-a). Source snapshot `188fdda91965e09f176fbe4e8764e941f9512767d74a49d7e689eaef2ef3676a`; dataset `350c74f2313e4e73ca65e7ffc6e9dcf74040b3b7`. Safe producer aggregate/manifests in [evidence](evidence/2026-11-11-conv-validation.json), verified aggregate SHA against manifest. Private maps/reconstructions remain remote. Timing, macro lesion Dice, normal-pixel false positives, checkpoint hashes and all thresholds are in evidence. No LPIPS or score fusions evaluated in this run.
