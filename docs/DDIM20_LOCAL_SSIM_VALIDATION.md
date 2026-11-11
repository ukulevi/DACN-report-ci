# DDIM20 Local SSIM — validation extension, 11 November 2026

User authorized scoring Local SSIM on validation to complete the DDIM20
Residual/Local SSIM/Spatial LPIPS comparison. This is an exploratory extension
after the completed frozen final-test study; it does not replace that protocol
or authorize a test rerun, model training or parameter sweep.

Use the immutable83sample validation cache from
`autoddpm-ddim2030-full83-t4-20261106T074200Z`, cache SHA256
22656e1f63a0047d3dbe32f7833eea8e303988414e5942dec1ba5a1d9ed1ef19.
Read only input images, initial DDIM20 reconstructions, masks and labels.
Gaussian EMA epoch30, DDIM20 fromt300, eta0, original seed17/UUID-derived sampling;
no new U-Net/LPIPS calls or GPU. Cache is already present on the team private
Volume and remains remote. Native dataset revision:
350c74f2313e4e73ca65e7ffc6e9dcf74040b3b7.

Verify cache/index/partition checksums before decoding. Reuse the exact registered
40calibration/43group-holdout membership accepted on10Nov; do not rederive folds.
Check cached labels/masks against validation index and patient-group disjointness.

Scoring uses existing `paired_score_maps` without changing its definition:
1−localSSIM, Gaussian11×11/sigma1.5, reflect padding, population moments,
K1=.01/K2=.03, data range2 on original signed[-1,1] input. Crop padding to240×240,
no clipping, foreground filtering or other postprocessing. Scores higher mean
more anomalous. Calculate all83maps with CPU. Fit separate pixel/image max-F1
thresholds only on40calibration, largest threshold on exact tie; write frozen
threshold receipt before43holdout metrics. Image score mean top576pixels;
pixel global micro, Dice=F1; report macro lesion Dice and normal pixel FPR too.

The historical DDIM20 L1 and Initial LPIPS rows were already evaluated on this
reconstruction/partition. New SSIM numbers must come from produced results.json;
no expected metrics are entered before execution. All83metrics are not used to
select thresholds. One original reconstruction seed, no uncertainty claim.

Implementation: `dacn.evaluation.ddim20_local_ssim`; CPU Modal entrypoint:
`src/dacn/cloud/modal_ddim20_local_ssim.py`. Local controller receives only safe
aggregate results and manifest. Private SSIM maps/masks stay on Volume and private
HF artifacts. Source snapshot, package versions and input hashes are recorded.

## Measured results — completed/uploaded
Computed all83validation score maps from the verified cached initial DDIM20 pairs. Thresholds fit on40calibration before43holdout metrics;25patient groups per partition. No test data or new model calls.
| DDIM20 score | Pixel ROC-AUC | Pixel AP | Precision | Recall | F1/Dice |
|---|---:|---:|---:|---:|---:|
| Residual / L1 | 0.874136 | 0.090936 | 0.101839 | 0.333698 | 0.156053 |
| Local SSIM | 0.880550 | 0.095772 | 0.105750 | 0.182641 | 0.133945 |
| Spatial LPIPS (Initial) | 0.926110 | 0.157876 | 0.194958 | 0.336464 | 0.246871 |

| Score | Image ROC-AUC | Image AP | Precision | Recall | F1 |
|---|---:|---:|---:|---:|---:|
| Residual / L1 | 0.250000 | 0.431879 | 0.558140 | 1.000000 | 0.716418 |
| Local SSIM | 0.368421 | 0.493883 | 0.558140 | 1.000000 | 0.716418 |
| Spatial LPIPS (Initial) | 0.787281 | 0.854501 | 0.611111 | 0.916667 | 0.733333 |

New Local SSIM thresholds: pixel0.9038662186681857, image1.108771815639135. Macro lesion-slice Dice0.17232036992102526; normal pixel FPR0.041466557017543856. At this image threshold all43holdout images are predicted positive; high recall does not establish useful image classification.

Local SSIM has slightly higher pixel AP but lower Dice than L1 on this cohort. Initial spatial LPIPS has higher pixel AP/Dice than both. This is one-seed exploratory validation evidence; no broad superiority or clinical claim.

CPU paired-score computation5.349924456s for83pairs, median0.064059484s, p950.066208280s/pair. Includes L1 computed inside paired_score_maps; excludes cache/hash/I-O/global metrics. Evaluation duration15.929608551s after start commit, before upload.2CPU,8192MiB; no GPU, VRAM N/A.

[New SSIM artifacts](https://huggingface.co/dacn-bmad-team/dacn-runs/tree/263e6e79711e85958c0dbce5d729577a45a05255/evaluations/ddim20-ssim-val83-20261111T062304Z) include source ZIP, results/thresholds, manifest and private maps. [Comparison evidence](evidence/2026-11-11-ddim20-validation-score-comparison.json) retains historical L1/LPIPS metrics and provenance; those were not rerun.
