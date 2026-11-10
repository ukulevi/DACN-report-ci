# Diffusion Models for Anomaly Detection — Working Draft

Status as of 10 November 2026: this document contains evidence-backed research
material. It is not a completed submission or a held-out test report.
Bibliographic sources are maintained in [References](REFERENCES.md).

Protocol review and freeze have been requested from Hai Tuan. The configuration
remains a candidate, and the test split remains sealed. The
[readiness assessment and new ablations](REPORT_READINESS_2026-11-10.md) distinguish
completed work, producer-audited results and outstanding gates. Data, methods
and validation sections can be drafted before the final evaluation is completed.

## Objective and dataset

This project investigates diffusion-based anomaly detection on BMAD Brain FLAIR
2D images using normal-only training. A normal slice has no annotated tumor;
this does not imply that the patient is healthy. The PNG images rendered from
BraTS differ from the original NIfTI intensities.

The [data protocol](DATA_PROTOCOL.md) defines decoding, symmetric padding and
normalization. Patient-disjoint splitting prevents slices from the same patient
appearing across splits. The immutable dataset revision is
`350c74f2313e4e73ca65e7ffc6e9dcf74040b3b7`.

## Method

The DDPM learns to predict noise epsilon from normal slices. Epsilon-MSE measures
noise prediction error; it does not directly measure tumor segmentation quality.
Partial reconstruction adds noise up to a selected timestep and then denoises.
Differences between the input and reconstruction define anomaly maps. The
investigated scores include absolute pixel difference, local SSIM difference
and spatial perceptual LPIPS distance.

The current candidate uses a Gaussian EMA epoch-30 checkpoint. Phase 1 performs
20 DDIM evaluations starting at t=300. Phase 2 performs 30 evaluations starting
at t=50, with eta=0 and one resampling pass. Pseudo-masking, stitching and
resampling restrict the reconstructed region. Spatial SqueezeNet LPIPS produces
the final anomaly score. Parameters and hashes are documented in the
[final-evaluation candidate](FINAL_EVALUATION_CANDIDATE.md).
The 50 U-Net evaluations per image measure model-call count, not wall-clock
latency or peak GPU memory.

## Evaluation design and measured validation results

Validation contains 40 calibration slices and 43 group-holdout slices. Thresholds
are fitted on calibration only. The holdout has been examined across several
experiments and remains subject to selection bias.

Independent cached-score replay matched producer metrics within 1e-10. The
8 November Colab T4 run independently executed the U-Net and LPIPS pipeline;
its score maps matched the producer cache within an absolute tolerance of 1e-6.
The private ZIP was downloaded and checksum-verified. CPU metric replay matched
the Colab receipt. Every result below is validation-only, with one sampling seed
(base seed 17).

| DDIM20+30 method | Pixel AP | Pixel F1/Dice |
|---|---:|---:|
| Final LPIPS | 0.312330 | 0.371215 |
| Initial LPIPS | 0.157876 | 0.246871 |
| Naive-stitch LPIPS | 0.180593 | 0.296849 |
| Brightness control | 0.389506 | 0.268051 |

| Method | Pixel ROC-AUC | Pixel precision | Pixel recall |
|---|---:|---:|---:|
| Final LPIPS | 0.949036 | 0.297471 | 0.493572 |
| Initial LPIPS | 0.926110 | 0.194958 | 0.336464 |
| Naive-stitch LPIPS | 0.936299 | 0.218753 | 0.461666 |
| Brightness | 0.950197 | 0.161469 | 0.788577 |

| Method | Image ROC-AUC | Image precision | Image recall | Image F1 |
|---|---:|---:|---:|---:|
| Final LPIPS | 0.914474 | 0.687500 | 0.916667 | 0.785714 |
| Initial LPIPS | 0.787281 | 0.611111 | 0.916667 | 0.733333 |
| Naive-stitch LPIPS | 0.785088 | 0.617647 | 0.875000 | 0.724138 |
| Brightness | 0.679825 | 0.571429 | 1.000000 | 0.727273 |

Pixel metrics are global micro aggregates over the 240x240 crop. Pixel F1 equals
micro Dice. The image score is the mean of the top 576 pixel scores. Each method
uses its own thresholds from the original 40 calibration slices, retained when
scoring the 43 holdout slices.

The full 83-image run made 4,150 U-Net and 249 LPIPS calls. The sum of adapter
execution times was 164.659411 seconds, with peak allocated CUDA memory of
112,928,768 bytes (approximately 107.7 MiB). These are not end-to-end notebook
latency or reserved GPU memory. The
[8 November acceptance report](COLAB_ACCEPTANCE_2026-11-08.md) records artifact
hashes, environment details and the correction of inherited provenance labels.

A paired bootstrap over 25 patient groups in the 43-slice holdout produced
999 valid draws out of 1,000, using bootstrap seed 17. The 95% confidence interval
for final-minus-naive-stitch pixel AP was [0.026164, 0.177945]. For final-minus-
brightness AP, it was [-0.287385, 0.232914]. These conditional validation results
do not establish that the final method outperforms brightness on new data.
Ancestral-sampler LPIPS-only results must not be presented as compute-matched
DDIM controls.

Complete image/pixel ROC-AUC, precision, recall, F1 and provenance must be traced
to the replayed aggregate receipts in [Result tracking](RESULT_TRACKING.md).
Missing measurements must not be inferred from planned experiments.

## Limitations and remaining submission work

Generalization is limited by one training/sampling seed, a single MRI contrast,
2D slices, repeated examination of validation, and insufficient scanner/site
metadata. The strong brightness control motivates discussion of dataset bias
and anomaly-ranking quality. The experiments do not establish clinical utility.

Final evaluation requires protocol freeze review and verification of the live
runtime before test access. The adapter, bounded validation, runtime measurements
and image/pixel tables have supporting evidence. Static preflight does not
replace live GPU runtime binding.

The final report still requires artifact-derived figures, final-test results if
run under a frozen protocol, complete BMAD/BraTS/AnoDDPM/LPIPS attribution and
citations for methods actually used. Formatting must follow the university's
confirmed requirements. Slides and a documented reproducible demo remain part
of the final submission work.
