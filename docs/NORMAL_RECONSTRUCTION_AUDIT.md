# Normal-foreground reconstruction audit — Gaussian EMA epoch30

Completed2026-11-02, offline CPU analysis of previously saved t100/200/300
ancestral reconstructions, sampling seed17. Uses all39 normal validation slices;
all corresponding annotation masks are empty. Normal here means no annotated
tumor in the slice, not proof of a clinically healthy patient. No new inference,
training or held-out test access. Source hashes verified; arrays align exactly
across timesteps. Dataset revision350c74f2313e4e73ca65e7ffc6e9dcf74040b3b7,
training source ddpm-partial-t4-b8-e30-cont-20261102T034644Z.

## Region and metric definitions

Crop padding to240x240; define foreground using input >-0.95 in the original
[-1,1] scale (approximately brightness>6.4/255). This input-only heuristic
removes black background; it is not an anatomical brain/tissue segmentation.
Interior is foreground more than5 pixels from its boundary. The same masks
are used at all timesteps, independent of reconstructions and their scores.

Foreground581,712 pixels /2,246,400 total (25.89%); background74.11%.
Interior482,183 and boundary99,529 pixels. MAE/RMSE use rescaled[0,1] images;
PSNR uses dynamic range1 and pooled ROI MSE. Local SSIM is the existing
Gaussian11x11/sigma1.5 map averaged at centers inside the ROI, equivalent to
[-1,1]/range2. Its windows can cross boundaries. Metrics are pooled over pixels,
not average patient quality; no uncertainty intervals.

## Measured reconstruction quality

| t* | Foreground MAE | Foreground RMSE | Foreground PSNR(dB) | Foreground mean local SSIM | Full-crop MAE | Full-crop mean local SSIM |
|---|---:|---:|---:|---:|---:|---:|
| 100 | 0.051991 | 0.070221 | 23.0706 | 0.532626 | 0.015741 | 0.869570 |
| 200 | 0.069675 | 0.097380 | 20.2306 | 0.374804 | 0.020983 | 0.823210 |
| 300 | 0.082587 | 0.117339 | 18.6112 | 0.292559 | 0.025143 | 0.797223 |

| t* | Interior MAE | Interior mean SSIM | Boundary MAE | Boundary mean SSIM | Background mean SSIM |
|---|---:|---:|---:|---:|---:|
| 100 | 0.048179 | 0.492650 | 0.070463 | 0.726300 | 0.987313 |
| 200 | 0.062742 | 0.340521 | 0.103267 | 0.540894 | 0.979902 |
| 300 | 0.072943 | 0.269191 | 0.129309 | 0.405767 | 0.973574 |

Per-slice foreground MAE median (IQR): t100=0.052005(0.049074–0.054210),
t200=0.069360(0.065690–0.075162),t300=0.082227(0.074317–0.090015).
The t300 foreground mean MAE corresponds to about21.1 grayscale levels out of255.
There is little global brightness bias (foreground signed recon-input at
t300=-0.000977); errors reflect local changes rather than a large uniform offset.

## Findings and limits

- The pipeline changes normal foreground substantially, increasingly with t*.
  At t300 foreground mean localSSIM is0.293 although full-crop meanSSIM is0.797;
  the large near-black background makes full-image averages look much better.
- Errors remain in the interior, not only along the brain outline. Together
  with the earlier false-positive analysis, this supports normal-structure
  distortion as a contributor to poor anomaly separation.
- Smaller t* preserves normal foreground better, but the earlier AP experiment
  found smaller t* worse for tumor localization. Reconstruction fidelity on
  normals and anomaly discrimination are different objectives/trade-offs.
- Noise-validation MSE~0.00736 is not reconstruction MSE. These measurements
  confirm that low noise loss did not ensure preservation of normal structure.
- These are descriptive values without a clinical quality threshold or a
  matched improved-model reference. They do not isolate weight/training quality,
  stochastic sampler variance, alignment or architecture as the root cause.
  SSIM is sensitive to fine texture and residual noise, so inspect the figures
  alongside the scores. Do not infer that more epochs will fix this.

Next diagnostic: audit train/inference contract, including weight/EMA loading,
normalization, t indexing, beta/variance and reverse-step equations. A controlled
sampler/weight comparison would be needed to separate causes. No run authorized
or launched by this offline audit alone.

## Artifacts and checks

Script `scripts/audit_normal_reconstruction.py`; local ignored directory
`runs/visualizations/ddpm-e30-t300/normal_tissue_audit_v1/` contains results.json
and normal_examples.png. Examples are lowest/median/highest t300 foreground
MAE, selected before viewing, shown paired at all three timesteps.
No patient identifiers in figures or logs. Results JSON SHA256
f4d6b9b43f0ae2f0f92c67a4bd7e2a2dbd372c7b9e5695357c0aed3cf6a95e86.
CPU40.586s; no GPU allocated. AP/AUROC/Precision/Recall/F1 are N/A because this
is an unthresholded reconstruction audit on exclusively normal slices.
Verified pooled regional MAE partitions exactly recover full-crop MAE and
MAE<=RMSE in every region; targeted Ruff/format/compile/diff checks passed.
