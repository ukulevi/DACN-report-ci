# DDPM + normal memory bank: implementation protocol

User-authorized implementation on 2026-11-06, following
[the method specification](fusionmemorybank.md). No improvement is assumed.
The user explicitly authorized full validation on 2026-11-06 after the smoke.

The DDPM branch reuses the byte-verified Gaussian EMA30, ancestral t*=300,
base17 cache. No DDPM training, new reverse chain, encoder conditioning or
spatial-conditioned checkpoint is used. Cache image/mask/label ordering is
verified against all83 validation entries and the registered index SHA before
extracting features. Existing DDPM and brightness holdout AP must reproduce.

The frozen encoder reuses the official ImageNet loader from
`models/diffusion/spatial_condition.py`, with full weights SHA
`f37072fd47e89c5e827621c5baffa7500819f7896bbacec160b1a16c560e07ec`.
RGB conv1 weights are averaged to grayscale. Input remains the DACN padded256
MRI in [-1,1], with no RGB normalization, resizing or per-image normalization.
This is a grayscale adaptation, not official RGB classification preprocessing.
BatchNorm remains in eval mode; no parameter is optimized.

Features layer2=(128,32,32), layer3=(256,16,16) are aligned by bilinear
interpolation (`align_corners=False`) to32x32 and concatenated into384D vectors.
No descriptor normalization, feature pooling, smoothing or projection is added.
Train cells enter the bank when at least half their corresponding8x8 input
pixels exceed -0.95. This is a foreground heuristic, not anatomical segmentation.
All inference cells are scored, including background, matching existing metrics.

The full bank uses all7500 normal train slices, no masks and no validation data.
Seed17 uniform random priorities retain10000 descriptors without replacement
from the eligible descriptor stream. This is **random sampling, not a PatchCore
coreset**. A private NPZ stores descriptors, train index checksum, encoder SHA,
dataset revision and extraction/sampling metadata; its byte checksum is recorded
and checked on reload. The smoke bank uses only16 train slices and128 vectors
and must never substitute for the full bank.

Exact Euclidean nearest-neighbor queries are chunked256queries x2048bank vectors.
The32x32 distance grid is upsampled to256 before cropping8pixels to240, preserving
the same coordinates as DDPM. No tumor mask is an inference input.

Each component score is normalized with `(score - median) / max(IQR,1e-6)`.
Median/IQR use **all240crop pixels of normal calibration slices only**. The
choice includes background to preserve the registered full-crop metric domain;
a near-zero IQR can amplify a branch and must be reported through fitted scales.
There is no clipping or per-image fitting. Alpha is fixed0.5 in two separate arms:
DDPM absolute + memory and DDPM SSIM + memory. No alpha search is implemented.

The existing first StratifiedGroupKFold(2,shuffle=True,random_state17) gives
40calibration/43group-holdout slices. Normalization, both pixel/image max-F1
thresholds and thresholds targeting calibration lesion recall0.5/0.8 are written
before calculating holdout metrics. Recall operating points are frozen from
calibration; actual holdout recall may differ, so they are not claims of exactly
matched holdout sensitivity. Report pixel AUROC/AP/P/R/F1/Dice, normal pixel FPR,
and image AUROC/AP/P/R/F1 for six arms including memory-only and brightness.
Image score is mean top576pixels; pixel metrics include the complete240crop.
Pixel AP uses continuous scores. The held-out test directory is never opened.

Implementation:

- `models/memory_bank.py`: frozen encoder, sampler, chunked neighbors, alignment,
  bank serialization/provenance checks.
- `training/build_memory_bank.py`: extraction only, no optimizer.
- `evaluation/ddpm_memory_fusion.py`: strict cache alignment, calibration, six-arm
  evaluation, private maps, binary predictions and normal/abnormal examples.
- `cloud/modal_memory_fusion.py`: bounded smoke/full T4 lifecycle and private upload.
- `configs/ddpm_memory_fusion.json`: documents the fixed initial protocol preset;
  the runner exposes smoke/full, not arbitrary tuning flags.

From the repository root, use a fresh ID and the actual HEAD SHA. The current
workspace is dirty; `--working-tree-dirty` and the exact source snapshot preserve
the implementation used by the run.

```bash
.venv/bin/modal run --profile phutungcogioihaitan --detach \
  src/dacn/cloud/modal_memory_fusion.py::run_memory_fusion_remote \
  --evaluation-id memory-fusion-smoke-UNIQUE_UTC_TIMESTAMP \
  --code-revision GIT_HEAD_SHA --working-tree-dirty --mode smoke
```

After explicit authorization, full validation command (replace both placeholders):

```bash
.venv/bin/modal run --profile phutungcogioihaitan --detach \
  src/dacn/cloud/modal_memory_fusion.py::run_memory_fusion_remote \
  --evaluation-id memory-fusion-full83-UNIQUE_UTC_TIMESTAMP \
  --code-revision GIT_HEAD_SHA --working-tree-dirty --mode full
```

Fresh IDs only; existing output folders are rejected rather than overwritten.
Timeout1hour, T4, shared Volumes and named Secret. All bank descriptors, MRI,
reconstructions, masks, scores, examples and predictions stay on the team Volume
and private HF model repo. Only aggregate JSON may be downloaded locally.
Run status/revisions belong in [RESULT_TRACKING](RESULT_TRACKING.md).
Smoke `memory-fusion-smoke-t4-20261106T054850Z` completed/uploaded at private
revision `56552b5792438971d008bb02eb1b4a72c6843902`:16normal train,128bank
vectors,2validation, finite aligned scores and bank checksum roundtrip passed.
Total8.23seconds before upload, peak117029888bytes. Smoke measures finite
output/provenance/alignment, not AP. Full local test suite:155passed.

Authorized full run `memory-fusion-full83-t4-20261106T055134Z` was launched detached
on the team T4 workspace:
[Modal app](https://modal.com/apps/phutungcogioihaitan/main/ap-UiDz4LoIR7Zey5Ld1cUuoA).
It completed/uploaded at private revision
`01c5995f6ec19092df7bb4ac302861399f2de868` on2026-11-06 at05:54:22UTC.
All7500normal train slices produced1929258eligible descriptors, with10000retained.
All83validation slices evaluated; main metrics are the43group holdout after
40calibration. Root result SHA verified against manifest:
`bfeb1f4573a6a0cef68cc68df6e8d86f0a3d7824080674cfb39e39547faaf476`.

| Method | Pixel AP | Pixel AUROC | Micro Dice | Normal FPR | Image AP |
|---|---:|---:|---:|---:|---:|
| Gaussian absolute | 0.101066 | 0.876854 | 0.173678 | 0.058079 | 0.430006 |
| Gaussian SSIM | 0.102402 | 0.888870 | 0.154654 | 0.055234 | 0.505258 |
| Memory only | 0.064212 | 0.824773 | 0.139943 | 0.134794 | 0.822745 |
| Absolute + memory | 0.102374 | 0.882715 | 0.180837 | 0.061204 | 0.434041 |
| SSIM + memory | 0.087033 | 0.881029 | 0.151198 | 0.079000 | 0.638276 |
| Brightness | 0.389506 | 0.950197 | 0.268051 | 0.134935 | 0.770309 |

Absolute fusion improves AP only0.001308 over its baseline; SSIM fusion worsens
AP by0.015370. F1/recall improvement of absolute fusion accompanies higher normal
FPR, so it is not a free sensitivity/specificity gain. Frozen calibration-recall
operating points do not establish a consistent FPR benefit either; actual holdout
recalls differ. No substantial tumor-localization improvement is established.
Memory-only image ranking is stronger than brightness in this fold, but its
calibrated image threshold predicts allholdout images positive and localization
is weak. The fusion does not beat this stronger image-ranking component.

Median/IQR fit20normal calibration slices: absolute0.010425895/0.019146100,
SSIM0.010653948/0.312484438,memory1.974215567/1.045310348. None uses the IQR floor.
No alpha tuning or post-hoc selection was performed. Total145.25seconds before
upload, bank phase46.91seconds including setup, peak118366208bytes. Private bank,
maps, masks, predictions and examples remain in private artifacts. Full required
P/R/F1 and image AUROC are recorded in RESULT_TRACKING. No extra GPU run on check.

Compare fusion against its stronger component before claiming benefit. One
model/bank seed on repeatedly inspected validation is exploratory and subject
to adaptive overfitting; independent confirmation is required.
