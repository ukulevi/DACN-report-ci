# LPIPS comparison — 11 November 2026
All rows use Gaussian EMA30, original sampling base17/UUID policy, spatial LPIPS0.1 SqueezeNet, grayscale repeatedRGB[-1,1],normalizeFalse and full240crop. DDIM20 and DDIM50 start att300. Final20+30 uses masked t50 phase2 after initial20; naive-stitch changes the image only within the predicted mask.
Separate pixel/image thresholds fit on40calibration for each row. Validation below is the inspected43group holdout; test contains3665images. One model/sampling seed; no confidence intervals claimed. DDIM50 is an exploratory extension after primary-test inspection, not a new untouched test.

## Validation holdout — 43 images: Pixel global micro, F1=Dice
| Configuration | ROC-AUC | Precision | Recall | F1 | AP |
|---|---:|---:|---:|---:|---:|
| DDIM20 — Initial LPIPS | 0.926110 | 0.194958 | 0.336464 | 0.246871 | 0.157876 |
| DDIM50 — single-phase LPIPS | 0.934520 | 0.222473 | 0.418106 | 0.290416 | 0.186020 |
| Naive-stitch from DDIM20 — LPIPS | 0.936299 | 0.218753 | 0.461666 | 0.296849 | 0.180593 |
| DDIM20+30 — Final LPIPS | 0.949036 | 0.297471 | 0.493572 | 0.371215 | 0.312330 |

## Validation holdout — 43 images: Image mean top576pixels
| Configuration | ROC-AUC | Precision | Recall | F1 | AP |
|---|---:|---:|---:|---:|---:|
| DDIM20 — Initial LPIPS | 0.787281 | 0.611111 | 0.916667 | 0.733333 | 0.854501 |
| DDIM50 — single-phase LPIPS | 0.837719 | 0.628571 | 0.916667 | 0.745763 | 0.901438 |
| Naive-stitch from DDIM20 — LPIPS | 0.785088 | 0.617647 | 0.875000 | 0.724138 | 0.861003 |
| DDIM20+30 — Final LPIPS | 0.914474 | 0.687500 | 0.916667 | 0.785714 | 0.944386 |

## Test — 3665 images: Pixel global micro, F1=Dice
| Configuration | ROC-AUC | Precision | Recall | F1 | AP |
|---|---:|---:|---:|---:|---:|
| DDIM20 — Initial LPIPS | 0.920021 | 0.267751 | 0.262345 | 0.265020 | 0.224625 |
| DDIM50 — single-phase LPIPS | 0.928086 | 0.294173 | 0.327008 | 0.309722 | 0.257950 |
| Naive-stitch from DDIM20 — LPIPS | 0.931432 | 0.298738 | 0.376180 | 0.333016 | 0.269516 |
| DDIM20+30 — Final LPIPS | 0.938799 | 0.356107 | 0.372863 | 0.364293 | 0.318226 |

## Test — 3665 images: Image mean top576pixels
| Configuration | ROC-AUC | Precision | Recall | F1 | AP |
|---|---:|---:|---:|---:|---:|
| DDIM20 — Initial LPIPS | 0.721256 | 0.849675 | 0.951074 | 0.897520 | 0.913943 |
| DDIM50 — single-phase LPIPS | 0.756141 | 0.855993 | 0.949091 | 0.900141 | 0.928450 |
| Naive-stitch from DDIM20 — LPIPS | 0.719082 | 0.854599 | 0.952066 | 0.900704 | 0.912759 |
| DDIM20+30 — Final LPIPS | 0.805989 | 0.873994 | 0.933223 | 0.902638 | 0.945863 |

## Interpretation
On this cohort, DDIM50 improves pixel AP/F1 over DDIM20 Initial LPIPS; 20+30 Final LPIPS has higher pixel ROC-AUC/AP/F1 than the other LPIPS rows. These observations do not establish superiority over non-LPIPS controls; the brightness control remains in the complete final-test report. No new thresholds or model choices are made from this comparison.

## Evidence
- [Comparison JSON](evidence/2026-11-11-lpips-comparison.json)
- [Primary test report](FINAL_TEST_RESULTS_2026-11-10.md)
- [DDIM50 protocol and freeze](DDIM50_LPIPS_TEST.md)
- [DDIM50 test aggregate](evidence/2026-11-11-ddim50-lpips-test.json)
- [Private immutable DDIM50 artifacts](https://huggingface.co/dacn-bmad-team/dacn-runs/tree/cf288576d33ef49b8966a106d0473d4495950164/evaluations/ddim50-lpips-t300-20261111T064853Z-b-test)
