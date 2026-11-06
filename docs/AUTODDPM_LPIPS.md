# Gaussian EMA30: LPIPS pseudo-mask and re-sampling

User-requested2026-11-06 after the memory-bank experiment. Inference only:
reuse Gaussian EMA30 and the verified ancestral t*=300 cache; no training.
First run a bounded2image re-sampling smoke, then the requested validation
comparison if smoke passes. All83validation uses the existing40calibration/
43group-holdout fold, no final test. No hyperparameter or mask sweep.

## Sources and distinction from the previous experiment

Sources inspected2026-11-06:
[AutoDDPM paper](https://arxiv.org/abs/2305.19643),
[official model code](https://github.com/ci-ber/autoDDPM/blob/main/model_zoo/ddpm.py),
[official configuration](https://github.com/ci-ber/autoDDPM/blob/main/projects/autoddpm/autoddpm.yaml),
[official README threshold guidance](https://github.com/ci-ber/autoDDPM).
The official implementation uses spatial SqueezeNet LPIPS, residual weighting,
dilation and repeated stitching of noised known context. Thresholds must be
adapted to the dataset; do not copy the published0.13 to this rendered FLAIR.

The existing2026-11-02 t200 SSIM-proxy pilot already failed to improve overall
localization (finalSSIMAP0.093603 vs t300baseline0.102402). This new adaptation
uses LPIPS rather than that proxy, a normal-calibrated mask threshold, and the
t300cache. Several factors change together; differences cannot isolate LPIPS.
It is **not exact reproduction**: DACN FLAIR256/checkpoint/backbone/normalization,
fixed-large reverse variance, crop240, t300initial and own consistent indexing
are retained. It does not reuse the paper's T1/128 weights or preprocessing.

## Fixed protocol before GPU execution

- Source Gaussian EMA30 SHA
  `d3e06d2082e18234899b201718abd59078c8a800d66da3cc5d8b9e0d9ca9b946`.
  Cached t300/base17 reconstruction SHA
  `8278ad18543dd87f698c36b510564b9fcd4445d4a1b0ae5d7a859093e2a4e959`.
  Require byte checks and exact83sample image/mask/label/index ordering.
- LPIPS package0.1.4, network squeeze, version0.1, spatial=True, frozen/eval.
  Load official torchvision pretrained features and packaged learned LPIPS
  weights. Bake weights into the image on CPU; record versions and SHA of the
  full named LPIPS state in fitted parameters. No encoder training.
- Compute LPIPS on240crop, grayscale repeated toRGB, input[-1,1],
  `normalize=False`. Do not apply double range conversion. This is a deliberate
  DACN range adaptation rather than copying the upstream normalization call.
- Heatmap=`clip(abs residual / image95th percentile,0,1)*max(LPIPS,0)`.
  Per-image residual scaling is label-free. The mask threshold is the pooled
  95thpercentile over20normal calibration images only, frozen before holdout
  LPIPS/re-sampling. Use strictheatmap>threshold followed by3x3 dilation.
  Fivepercent refers to pre-dilation pooled calibration pixels; post-dilation
  coverage and holdout FPR need not equal5percent. Padding mask is false.
- Stitch initial reconstruction inside pseudo-mask and original input outside;
  forward-noise to one-based50/index49, reverse50steps with5repeats. Between
  repeats re-noise with the matching beta[k]. At each reverse step insert known
  original context at matching time; final context exactly equals input.
  Reuse tested `stitch_resample`, which differs from upstream loop details.
- Sampling base17, sample-UUID-derived isolated RNG, batch1,T4. Pseudo-masks
  never use lesion masks. Ground truth only enters evaluation/coverage metrics.
- **Primary score** final absolute residual times initial LPIPS heatmap.
  Fixed secondary final absolute/localSSIM; controls initial absolute/localSSIM,
  initial heatmap, naive-stitch absolute (no re-sampling), and brightness.
  Report all8arms. No holdout-based score selection or mask tuning.
- Select image/pixel maxF1 thresholds on40calibration, persist allthresholds,
  then evaluate43group-holdout. Full240crop micro metrics include background;
  image score mean top576pixels. PixelAP continuous; record ROC-AUC/P/R/F1/Dice,
  AP,normalFPR,image metrics. Baseline absolute/SSIM/brightnessAP must reproduce.
- Report pseudo-mask lesion recall/precision/ROI fraction and normal coverage
  after dilation, using masks only for post-hoc diagnostics. Exact outside-mask
  preservation is enforced by construction and cannot prove good restoration.
  Compare final residual with naive stitch to distinguish mask gating from
  benefits of re-sampling. Mask-missed lesions can have near-zero residual.

The smoke fits the same normal20calibration threshold but re-samples only first
normal and firstabnormal image. It checks frozen-model loading, real LPIPS,
finite outputs, consistent crop and zero outside-mask error, with no AP result.
The full run performs83x250 U-Net calls; no initial300step chains regenerated.
Private prefix checkpoints every5samples aid diagnosis, but this runner requires
fresh output IDs and does not claim exact prefix resume support.

## Entrypoints and outputs

```bash
.venv/bin/modal run --profile phutungcogioihaitan --detach \
  src/dacn/cloud/modal_autoddpm_lpips.py::run_autoddpm_lpips_remote \
  --evaluation-id autoddpm-lpips-smoke-UNIQUE \
  --code-revision GIT_HEAD_SHA --working-tree-dirty --mode smoke
```

Use `--mode full` and a fresh `autoddpm-lpips-full83-UNIQUE` ID after smoke.
Timeout1hour. Source snapshot, package versions, checkpoint/cache/index SHAs,
calibration parameters, fixed seeds and private derived arrays are recorded.
MRI/masks/maps remain on team Volumes and private HF; download aggregate JSON
only. Record every actual attempt in RESULT_TRACKING, including failures.
No improvement claim until measured. One model/seed on repeatedly inspected
validation is exploratory, not independently confirmed clinical performance.

## Actual attempts

- Smoke `autoddpm-lpips-smoke-t4-20261106T061700Z` completed/uploaded private
  revision `87d0943c3b1de7be211997c771c00edf495746b8`,2finite outputs,exact
  outside-mask preservation,error0. Maskthreshold0.13774138689041138 from20normal
  calibration;LPIPSstateSHA
  `18fed4d1fc5c04f2918f64df3a2463344108576d1888f0d75538ca6d5c4dffe6`.
  Total24.315s before upload,peak104408576B,no AP measured. Result SHA verified
  against manifest;source snapshot
  `b4a1fb55cfed8acb8aaa2004b04ae76e581e6f0b05d71e4bc07a7f7f4bea18b2`.
- Full `autoddpm-lpips-full83-t4-20261106T061833Z` launched detached after smoke:
  [Modal app](https://modal.com/apps/phutungcogioihaitan/main/ap-Q53OLMLNmmu3h6rZCVv2cV).
  Completed/uploaded private revision`bab00576bb8271ddb191c8fcd13840eff40f1919`,
  all83samples,06:31:00UTC on2026-11-06,total715.98s before upload,
  peak104408576B. Root result SHA verified against manifest:
  `6fb277491308c63b2cc52fa614fc912486266a9c3322f7f099aff850cb51a7e0`.

## Measured43group-holdout results

All40calibration thresholds frozen;normal20calibration mask threshold unchanged.
Pixel metrics are micro over240crop;no held-out test used.

| Method | Pixel AP | Micro Dice | Normal FPR | Image AP |
|---|---:|---:|---:|---:|
| Initial absolute | 0.101066 | 0.173678 | 0.058079 | 0.430006 |
| Initial SSIM | 0.102402 | 0.154654 | 0.055234 | 0.505258 |
| Initial LPIPS x normalized residual | 0.204298 | 0.274354 | 0.010727 | 0.887665 |
| Naive stitch absolute | 0.118781 | 0.204641 | 0.047041 | 0.481000 |
| Resampled absolute | 0.129407 | 0.227416 | 0.044342 | 0.534048 |
| Resampled SSIM | 0.149446 | 0.223469 | 0.045201 | 0.581114 |
| Weighted resampled absolute (primary) | 0.154785 | 0.226256 | 0.026640 | 0.814398 |
| Brightness | 0.389506 | 0.268051 | 0.134935 | 0.770309 |

Primary finalweighted residual improves over baseline,but does not beat the
initial LPIPS heatmap before re-sampling. Re-sampling adds0.010626 AP over naive
stitch when comparing absolute residuals;that modest incremental benefit does
not explain the larger gain from LPIPS-based scoring. LPIPS heatmap is an already
registered control,not a post-hoc unreported score choice;report primary and
allcontrols. No LPIPS-only unweighted arm was measured,so do not claim the
perceptual distance alone caused the gain. Brightness still has higherpixelAP.

Holdout pseudo-mask covers83.7541% of lesion pixels,precision13.3407%,overall
ROI14.1171%,normal-image ROI11.7060% after dilation. It misses16.2459% of lesion
pixels;preservation of unmasked lesion tissue can limit residual segmentation.
Final outside-mask error0 by construction. NormalFPRs in the table use separate
score thresholds,not the pseudo-mask threshold;they are not matched-recall
comparisons. Initial LPIPSheatmap recall0.280623 vs brightness0.788577 explains
why higherF1/lowerFPR need not imply better overall localization ranking.
Full pixel/image AUROC/P/R/F1,maskdiagnostics and provenance inRESULT_TRACKING.md.
No extra GPU run on check. One seed on repeatedly inspected validation remains
exploratory;no independent significance or clinical generalization claim.
