# Experiment and result tracking

## 2026-11-07 — Shared-trajectory controls and CPU benchmark

Attempt1 failed before inference: dependency text comparison treated Windows
CRLF and subprocess-normalized LF as different. Package versions unchanged;
fixed comparison to line sequences, preserving package/version checks.
Attempt2 completed: scripts/benchmark_ddim_controls.py, output
runs/ddim-controls-acceptance-20261107-attempt2. Same trained G30 checkpoint, immutable
model/dataset revisions and two validation samples as bounded acceptance.
Four score arms share a50-call U-Net trajectory per sample: final LPIPS, initial
LPIPS, naive-stitch LPIPS and brightness. Separate image/pixel thresholds copied
from measured40 calibration/replay, no new fit. Three LPIPS calls/sample.
CPU threads2/batch1. Windows process-lifetime peak working set and cold/warm
adapter-call timing measured separately; no GPU/VRAM or held-out test.
100 U-Net calls,6LPIPS calls, final maps match previous CPU acceptance exactly.
Cold sample24.398705s, warm sample21.706827s; peak process working set908849152bytes
through adapter completion before metrics. Diagnostic image/pixel metrics and
limitations in CONTROLS_ACCEPTANCE_2026-11-07.md; no model selection from2samples.

## 2026-11-07 — Trained G30 DDIM bounded CPU acceptance

Attempt 1 status: failed at preflight, before dataset/inference. Sampler config
artifact was incorrectly treated as trainer config. Checksum guard correctly
stopped execution. Fixed adapter to read trainer config from hashed checkpoint
and validate pinned sampler config against mask parameters. Original output
directory retained. Attempt 2 started in runs/ddim-acceptance-20261107-attempt2.
Two registered validation samples (one per class), fixed
DDIM20+30, checkpoint epoch30 SHA256
d3e06d2082e18234899b201718abd59078c8a800d66da3cc5d8b9e0d9ca9b946
from model revision fae9b50a4b78114fb51d87e803c77d019668f523.
Script accept_ddim_validation.py verifies all83 local validation image/mask/label
contents against checksum-pinned cache plus index hash, captures current source,
checks dependency freeze and all artifact hashes before inference. Planned
intentional interruption after first committed shard, fresh-model resume and
completed-cache replay. CPU only; no threshold fitting or held-out test.
Private output: runs/ddim-acceptance-20261107/aggregate_receipt.json.
Two-sample metrics are diagnostics, not efficacy estimates. Candidate is not
changed to frozen for this validation-only path.

Attempt2 completed: 2 samples, 100 trained U-Net calls; intentional interruption,
fresh-model resume, unchanged first shard and completed-cache resume all passed.
Wall 58.769387s including model load and I/O. Current-source snapshot SHA256
242e840d6134f81e8bb27083ad15bdb04a6622bf4dc1be3a881beb3c698de08e.
Additional uninterrupted CPU comparator completed: same two samples/seed/settings,
100 additional U-Net calls, 44.1546885 seconds. Every stored label/score array and
metric matches interrupted/resumed output exactly (maxabs0). Receipt
clean_compare_receipt.json. See ACCEPTANCE_2026-11-07.md for diagnostic image/pixel
AUROC/AP/P/R/F1, thresholds, scope and limitations. No parameter selection.

## 2026-11-07 — CPU LPIPS full cached-pair replay

Status: completed. After successful two-sample smoke, same pinned artifact cache,
fresh LPIPS CPU environment and exported states; all 83 validation initial/final
pairs, 166 calls. Compare maps against saved scores, no threshold fitting,
U-Net/GPU or test. Receipt: private validation_full_replay.json under
runs/lpips-environment-20261107. All 166 maps matched at atol 1e-6, max absolute
error 5.21540641784668e-7; pair-scoring CPU time 35.770264 seconds, excluding
model/cache loading. Cache SHA256
22656e1f63a0047d3dbe32f7833eea8e303988414e5942dec1ba5a1d9ed1ef19.
Output replay only, not DDIM reconstruction replay or end-to-end latency.
Reusable equivalent entry point: scripts/replay_lpips_pairs.py.

## 2026-11-07 — CPU LPIPS reconstruction-pair smoke

Status: completed. Fresh CPU environment with LPIPS 0.1.4, Torch 2.15.0+cpu,
torchvision 0.29.2+cpu. Two validation samples (one per class), four LPIPS calls
on cached initial/final DDIM pairs. Artifact revision
`fae9b50a4b78114fb51d87e803c77d019668f523`; exact cache checksum checked against
the run manifest before reading arrays. New native trunk/head exports and
dependency freeze stored privately in runs/lpips-environment-20261107.
No U-Net calls, threshold fitting, GPU or test. This attempts output replay,
not proof of historical environment identity. Receipt validation_smoke.json:
four maps match cached output at atol 1e-6, maximum absolute error
3.5762786865234375e-7, measured pair-scoring time 0.860262 seconds on CPU.
This is not end-to-end latency. No AUROC/AP estimated from this two-sample smoke.
State digest: 18fed4d1fc5c04f2918f64df3a2463344108576d1888f0d75538ca6d5c4dffe6.

## 2026-11-06 — Full image/pixel metric replay

Status: completed. CPU replay of cached DDIM final LPIPS, initial LPIPS, naive
stitch LPIPS and brightness. Same HF revision fae9b50, 40 calibration / 43
holdout validation slices, 25 holdout groups; seed 17 partition. Refit thresholds
on calibration only and compare image/pixel AUROC/AP/P/R/F1, Dice and normal FPR
with C's immutable results. All four candidates matched with absolute tolerance
1e-10, including calibration-derived thresholds, macro lesion Dice and normal FPR.
No GPU, inference or held-out test. Reused the prior verified data/index alignment.
[Aggregate receipt](evidence/2026-11-06-b-full-metrics-results.json), reproduced by
`python docs/evidence/2026-11-06-b-ddim-bootstrap-replay.py --metrics-only`.

## 2026-11-06 — B CPU replay / paired bootstrap

Status: **completed**. Validation only, source HF revision `fae9b50a4b78114fb51d87e803c77d019668f523`, dataset revision `350c74f2313e4e73ca65e7ffc6e9dcf74040b3b7`. DDIM final LPIPS, initial LPIPS, naive stitch LPIPS and brightness; fixed calibration thresholds, patient-group paired bootstrap seed 17, 1,000 draws, 999 valid (one lacked a class), 25 patient groups / 43 slices. No held-out test or GPU.

Verified both NPZ hashes against manifests, the source-cache linkage, index hash, each cached image/mask/label against the dataset, group order, disjoint partition and replayed AP/Dice against C's results. Final LPIPS AP .312330, Dice .371215. Paired delta AP (final minus initial) 95% percentile interval [.039686, .204806]; final minus naive stitch [.026164, .177945]; final minus brightness [-.287385, .232914]. These intervals are exploratory and conditional on this previously inspected validation cohort, selected model/protocol and one sampling seed; they do not quantify training-seed variability or fix selection bias.

[Aggregate output](evidence/2026-11-06-b-ddim-bootstrap-results.json); [replay script](evidence/2026-11-06-b-ddim-bootstrap-replay.py) (format/import cleanup after execution; paths refer to local private caches). No private images, group IDs, or per-sample output are published.

## 2026-11-06 — Audit receipt for C's completed runs

Đây là đối chiếu artifact đã được C tạo, không phải run mới của B. HF snapshot
`fae9b50a4b78114fb51d87e803c77d019668f523`: kiểm 94 references mới qua JSON bytes
hoặc LFS metadata, không mismatch; chưa tái tính metric từ NPZ trong lượt audit.
Source branch C được kiểm đến `c02addb`; không mặc định đó là source snapshot
của mọi run. Provenance từng run nằm trong manifest/evidence.

[Manifest bất biến của run DDIM20+30](https://huggingface.co/dacn-bmad-team/dacn-runs/blob/fae9b50a4b78114fb51d87e803c77d019668f523/evaluations/autoddpm-ddim2030-full83-t4-20261106T074200Z/run_manifest.json)
ghi code revision `e75469bedfe757ad9651e03ab6cfe2cfd2b5fd5a` và source snapshot
SHA256 `7b76a23f2712aa7f3bb26698aafdf6311ef26e088c4b62fadedd5bacb44869e1`.
[Results bất biến](https://huggingface.co/dacn-bmad-team/dacn-runs/blob/fae9b50a4b78114fb51d87e803c77d019668f523/evaluations/autoddpm-ddim2030-full83-t4-20261106T074200Z/results.json)
là nguồn các số bên dưới; cần quyền truy cập repo private.

Run `autoddpm-ddim2030-full83-t4-20261106T074200Z`: Gaussian EMA30,
DDIM20 + 30, LPIPS final; một sampling seed, 83 validation slices chia
40 calibration / 43 internal group holdout (25 groups). Pixel AP micro .312330,
AUROC .949036, Precision .297471, Recall .493572, F1/Dice .371215;
image AUROC .914474, Precision .687500, Recall .916667, F1 .785714.
Threshold chọn max F1 trên calibration, score >= threshold, hòa chọn threshold
lớn nhất; holdout giữ threshold đó. Đây là validation được khảo sát nhiều lần,
không phải kết quả held-out test cuối.

Naive-stitch LPIPS có AP .180593, brightness .389506 trên cùng holdout;
không kết luận diffusion vượt brightness từ Dice tại threshold riêng.
Đọc [audit và kế hoạch](PROJECT_AUDIT_2026-11-06.md) cùng
[aggregate evidence](evidence/2026-11-06-project-audit-latest.json) cho metric,
artifact paths, runtime và checksum checks. Không có GPU/MRI evaluation mới
được khởi chạy trong audit này.
## 2026-11-06 — RAD-DINO full83 frozen scoring

- Evaluation `rad-dino-full83-t4-20261106T084846Z` completed2026-11-06T08:50:32Z;
  [Modal app](https://modal.com/apps/phutungcogioihaitan/main/ap-MEBlm8GiyPkuor5st2SzkH).
  Status: completed, private upload revision `fae9b50a4b78114fb51d87e803c77d019668f523`.
- Identical [registered protocol](RAD_DINO_SCORING.md) and code snapshot to smoke:
  frozen CXR-pretrained RAD-DINO4taps raw primary/normal-calibrated secondary,
  fixed final DDIM20+30 cache and cachedLPIPS reference.83validation,
  40calibration/43patient-group holdout;0diffusion calls/no test.
- Dataset revision `350c74f2313e4e73ca65e7ffc6e9dcf74040b3b7`, encoder revision
  `110cbc18d5133582e320b43d53bf5c44e410c936`, code HEAD
  `e75469bedfe757ad9651e03ab6cfe2cfd2b5fd5a` plus dirty snapshot.
  [private Hub artifacts](https://huggingface.co/dacn-bmad-team/dacn-runs/tree/fae9b50a4b78114fb51d87e803c77d019668f523/evaluations/rad-dino-full83-t4-20261106T084846Z).
- Results SHA `34004333b348e467f6b0fefac502995440f84ede71787a4c4f1339f642289fef`
  matches manifest. 83encoder pairs,52.109713669s, peak VRAM874679296bytes;
  zero diffusion calls, no held-out test.

| Arm | Pixel AP | Pixel AUROC | P | R | F1/Dice | Normal pixel FPR | Image AP |
|---|---:|---:|---:|---:|---:|---:|---:|
| RAD-DINO raw mean | .289211335 | .942452048 | .185947150 | .638435020 | .288010044 | .030442251 | .948820197 |
| RAD-DINO normal-calibrated | .284123511 | .942153502 | .184587563 | .639620067 | .286495551 | .030927448 | .941350053 |
| SqueezeNet LPIPS reference | .312330164 | .949036058 | .297471026 | .493572019 | .371214628 | .005772113 | .944385917 |

- Thresholds: each arm's pixel/image maxF1 fit on40calibration, largest on ties;
  holdout43patient-disjoint validation. Pixel AP/AUROC/F1 use global micro
 240x240; image AP/AUROC/F1 use mean top576. RAD-DINO raw reduces PixelAP by
  .023118830 and Dice/F1 by .083204584 against LPIPS, raises recall but lowers
  precision, and has ~5.3x normal FPR. Image AP is similar (+.004434280).
  Normal calibration further lowers PixelAP. This transfer trial gives no
  improvement in pixel segmentation; RAD-DINO remains CXR-pretrained and this
  comparison also changes architecture/preprocessing/learned head. One fixed
  checkpoint/seed and inspected validation: exploratory only.

## 2026-11-06 — RAD-DINO frozen scoring smoke

- Attempt `rad-dino-smoke-t4-20261106T084704Z` submitted;
  [Modal app](https://modal.com/apps/phutungcogioihaitan/main/ap-B8POBcs4Gxjgalz71hjxnH).
  Status: completed, private upload confirmed, revision
  `751d189847802ac9dd3f68f2b764919761c7b75e`. No smoke AP claimed.
- User-requested [fixed protocol](RAD_DINO_SCORING.md): chest-X-ray pretrained
  RAD-DINO revision `110cbc18d5133582e320b43d53bf5c44e410c936`, frozen4taps,
  original finalDDIM20+30 cached pairs, raw primary/normal-calibrated secondary,
  cachedLPIPS control. No DDPM calls, encoder training or pseudo-mask change.
- Dataset `350c74f2313e4e73ca65e7ffc6e9dcf74040b3b7`, code HEAD
  `e75469bedfe757ad9651e03ab6cfe2cfd2b5fd5a` plus dirty snapshot.
  Team Volume manifest path `evaluations/rad-dino-smoke-t4-20261106T084704Z/run_manifest.json`
  once GPU callable starts. T4/30min,83validation ordering verified,20normalcal fit,
  2output pairs plus identical-pair sanity check; test remains sealed.
- Measured22encoder pair calls (21finalpairs including20normalcal + identical),
  identical-pair error0; all4IQR floors inactive.12.870362152s work,
  peak allocatedVRAM874679296bytes. Strict loading: no missing/unexpected/mismatched
  keys/errors. Source snapshot
  `1e1e4c4b57f21c94b7ca347fab42aa669b744f4a743b741ed0e6f771caa6f1bd`;
  encoder model.safetensors SHA `dbfb9f54459c38773505de64a6ab7807bdcb392610fe1e697166342e43fb91ae`.

## 2026-11-06 — Full83 normal-calibrated LPIPS

- Evaluation `lpips-normalcal-full83-t4-20261106T083649Z` started on T4;
  [Modal app](https://modal.com/apps/phutungcogioihaitan/main/ap-SsEzk6RQezELGdvji8sh4y).
  Status: completed2026-11-06T08:37:59Z; private upload confirmed.
- Same fixed [protocol](NORMAL_CALIBRATED_LPIPS.md) as passing smoke below,
  immutable DDIM20+30 finals, original LPIPS/equal mean/calibrated equal mean/
  naive-stitch LPIPS controls;83validation,40calibration/43group holdout.
  Normal statistics fit20normal calibration images; no test, no diffusion calls.
- Dataset revision `350c74f2313e4e73ca65e7ffc6e9dcf74040b3b7`; code
  `e75469bedfe757ad9651e03ab6cfe2cfd2b5fd5a` plus dirty source snapshot.
  Manifest/artifacts: team Volume `evaluations/lpips-normalcal-full83-t4-20261106T083649Z/`;
  [private Hub artifacts](https://huggingface.co/dacn-bmad-team/dacn-runs/tree/25c4868058a46e4a6e6b856c2c3be8f1c0825371/evaluations/lpips-normalcal-full83-t4-20261106T083649Z).
- Verified downloaded aggregate results SHA against manifest:
  `a715ead9b769058fb237ebaa301547d8a4816208c01caec80db59b9390c443ff`.
  Source snapshot `a358d099e96a85bca9409c703aca02d93a21c1eab881b71cfa6321265ebeb8a8`.
  166LPIPS pair calls,0diffusion calls;42.564381212s total function work,
  peak allocated VRAM45882880bytes. Cache replay and layer sum errors both0.
- Group-holdout43 metrics below; pixel global micro over240crop, image mean
  top576pixels. Each arm uses pixel/image maxF1 thresholds from40calibration,
  largest threshold on ties. One fixed source checkpoint/noise realization;
  this inspected validation is exploratory, not independent confirmation.

| Arm | Pixel AP | Pixel AUROC | Pixel P | Pixel R | Pixel F1/Dice | Normal pixel FPR |
|---|---:|---:|---:|---:|---:|---:|
| Original LPIPS | .312330164 | .949036058 | .297471026 | .493572019 | .371214628 | .005772113 |
| Uncalibrated aggregate/7 | .312330164 | .949036058 | .297471026 | .493572019 | .371214628 | .005772113 |
| Normal-calibrated equal mean | .307159662 | .947552281 | .312157458 | .465310446 | .373649189 | .005152595 |
| Naive stitch LPIPS | .180593333 | .936299322 | .218752925 | .461665530 | .296848753 | .018309576 |

| Arm | Image AP | Image AUROC | Image P | Image R | Image F1 |
|---|---:|---:|---:|---:|---:|
| Original LPIPS | .944385917 | .914473684 | .687500000 | .916666667 | .785714286 |
| Uncalibrated aggregate/7 | .944385917 | .914473684 | .687500000 | .916666667 | .785714286 |
| Normal-calibrated equal mean | .950431487 | .918859649 | .647058824 | .916666667 | .758620690 |
| Naive stitch LPIPS | .861002904 | .785087719 | .617647059 | .875000000 | .724137931 |

- Conclusion: primary Pixel AP did not improve (.312330→.307160).
  Tiny pixel F1 increase accompanies lower recall; lower normal FPR is not a
  matched-recall gain. Keep original LPIPS as reference. Final re-sampling with
  LPIPS exceeds naive-stitch LPIPS by .131736831AP here, supporting a contribution
  beyond simple stitching under this protocol; no general superiority claim.
  Medical encoders remain untested; no further GPU job started during check.

## 2026-11-06 — Normal-calibrated LPIPS smoke

- Evaluation `lpips-normalcal-smoke-t4-20261106T083504Z` submitted on T4;
  [Modal app](https://modal.com/apps/phutungcogioihaitan/main/ap-pZsyDu5Y4UNBI5QubstURo).
  Status: completed; private Hub upload confirmed by runner, revision
  `f0ee17da4f83de5c3bc0598839831ef27bf6154b`. No smoke AP/F1 claimed.
- Fixed [protocol](NORMAL_CALIBRATED_LPIPS.md), Gaussian EMA30 final DDIM20+30
  cache, seven existing SqueezeNet LPIPS layers, normal-only median/IQR fit.
  Two output pairs plus20normal calibration pairs; zero diffusion calls.
- Dataset revision `350c74f2313e4e73ca65e7ffc6e9dcf74040b3b7`, code HEAD
  `e75469bedfe757ad9651e03ab6cfe2cfd2b5fd5a`, dirty source snapshot retained.
  Manifest on team Volume at `evaluations/lpips-normalcal-smoke-t4-20261106T083504Z/run_manifest.json`.
  Test remains sealed; private arrays stay on Volume/private Hub.
- Aggregate result:2 output pairs,21 final pairs processed including20normal
  calibration,2naive pairs,23LPIPS calls,0diffusion calls;15.168606435s.
  Aggregate replay error0,seven-layer sum error0; all seven scale floors inactive,
  295786normal foreground pixels. Source snapshot
  `a358d099e96a85bca9409c703aca02d93a21c1eab881b71cfa6321265ebeb8a8`.

Use this file for runs that actually started, including failed, interrupted,
and smoke-test runs. Linear tracks task status; this file tracks experimental
provenance and results. The Modal `run_manifest.json` and saved artifacts are
the detailed run record; link them here and keep a compact comparison summary.

## Rules

- Register a run only after it starts. Mark planned configurations as planned,
  never as results.
- Preserve the patient-disjoint split. Train the diffusion model primarily on
  normal samples; do not use held-out test data for model, score, threshold, or
  hyperparameter selection.
- Record validation and test separately. State image-level versus pixel-level
  aggregation and the threshold-selection rule for Precision, Recall, F1, and
  Dice. Freeze the rule before test evaluation.
- Required brief metrics: ROC-AUC, Precision, Recall, and F1-score. Also include
  pixel AP, Dice, latency, peak VRAM, and uncertainty estimates where applicable.
- Record the number of seeds actually completed. Do not report a planned seed as
  completed or silently omit failed runs.
- Link immutable dataset/model revisions, code commit, config, run manifest, and
  artifacts. Never record credentials, patient IDs, patient paths, or private
  medical data.

## Comparison summary

### 2026-11-06 — Authorized full DDIM20+30 four-score ablation

- Run `autoddpm-ddim2030-full83-t4-20261106T074200Z`: **completed/uploaded**,
  [Modal app](https://modal.com/apps/phutungcogioihaitan/main/ap-98lEz64JkA8I5USQcgE4CR).
  User explicitly approved full83after successful50-callsmoke. This satisfies the
  suppliedtesting_Lpips.md authorizationgate;not the superseded10+10x5fullrun.
- Exact smoke-passed preset:EMA30,T1000linear,DDIM20fromt300 +DDIM30fromt50,
  eta0,onephase2pass,seed17. Expected4150U-Net/166LPIPScalls across83outputs.
  Fourrequired rows use sharedinitial/final reconstructions;also report initial
  LPIPS/product,naive-stitch and brightness,with historicalsamplers labeled.
- Same pinned dataset350c74f2313e4e73ca65e7ffc6e9dcf74040b3b7,40calibration/
  43group holdout,normal-only maskfit and calibration-only scorethresholds;
  no training or finaltest. Code parente75469bedfe757ad9651e03ab6cfe2cfd2b5fd5a,
  dirty source expected7b76a23f2712aa7f3bb26698aafdf6311ef26e088c4b62fadedd5bacb44869e1.
- Durablemanifest confirmsrunning at07:42:26UTC (14:42:26Vietnam),TeslaT4,
  modefull,20+30preset,repeats1,no-testflag,and exactsource identicaltosmoke.
  Completed07:45:33UTC (14:45:33Vietnam).
  [Private immutable revision](https://huggingface.co/dacn-bmad-team/dacn-runs/tree/ceba5af6b7cdc46b5404554575add3c5cb09d9c7/evaluations/autoddpm-ddim2030-full83-t4-20261106T074200Z).
  RootresultSHA`a8bae24bd7270b078d49f923461e43fb51aea3163ede5a2fe71c90a70b127730`
  verified against manifest;all83output callcounts50,4150U-Net/166LPIPS total,
  no extra calibrationcalls,onephase2pass,outside-maskerror0,no finaltest.
- Required fourmatched rows on43group holdout,40calibrationthresholds frozen.
  Time/image below is the mean of synchronized model/scoring component sums over
  shared computations,NOT independent end-to-end reruns. Scope excludes transfers,
  mask assembly,globalcalibration,metrics/cache/I-O andstartup.

| Method | Phase1 | Phase2 | Pixel AP | Micro Dice | Normal FPR | Component seconds/image |
|---|---|---|---:|---:|---:|---:|
| DDPM + L1 | DDIM20 | None | 0.090936 | 0.156053 | 0.074325 | 0.588357 |
| AutoDDPM + L1 | DDIM20 | DDIM30 | 0.099569 | 0.200077 | 0.069091 | 1.471042 |
| AutoDDPM + LPIPS | DDIM20 | DDIM30 | 0.312330 | 0.371215 | 0.005772 | 1.477092 |
| AutoDDPM + LPIPS x final residual | DDIM20 | DDIM30 | 0.196812 | 0.272453 | 0.006499 | 1.477929 |

| Current score/control | Pixel AUROC | Precision | Recall | F1/Dice | Pixel AP | Normal FPR |
|---|---:|---:|---:|---:|---:|---:|
| DDPM-DDIM20 L1 | 0.874136 | 0.101839 | 0.333698 | 0.156053 | 0.090936 | 0.074325 |
| AutoDDPM-DDIM20+30 L1 | 0.819276 | 0.128802 | 0.447966 | 0.200077 | 0.099569 | 0.069091 |
| AutoDDPM-DDIM20+30 LPIPS | 0.949036 | 0.297471 | 0.493572 | 0.371215 | 0.312330 | 0.005772 |
| AutoDDPM-DDIM20+30 final product | 0.831096 | 0.272722 | 0.272184 | 0.272453 | 0.196812 | 0.006499 |
| Initial-DDIM20 LPIPS only | 0.926110 | 0.194958 | 0.336464 | 0.246871 | 0.157876 | 0.015723 |
| Initial-DDIM20 product | 0.905685 | 0.193713 | 0.238554 | 0.213808 | 0.144068 | 0.016109 |
| Naive stitch L1 | 0.819259 | 0.128943 | 0.368783 | 0.191077 | 0.099308 | 0.057721 |
| Brightness | 0.950197 | 0.161469 | 0.788577 | 0.268051 | 0.389506 | 0.134935 |

| Current score/control | Image AUROC | Image AP | Precision | Recall | F1 |
|---|---:|---:|---:|---:|---:|
| DDPM-DDIM20 L1 | 0.250000 | 0.431879 | 0.558140 | 1.000000 | 0.716418 |
| AutoDDPM-DDIM20+30 L1 | 0.440789 | 0.501203 | 0.571429 | 1.000000 | 0.727273 |
| AutoDDPM-DDIM20+30 LPIPS | 0.914474 | 0.944386 | 0.687500 | 0.916667 | 0.785714 |
| AutoDDPM-DDIM20+30 final product | 0.912281 | 0.943103 | 0.687500 | 0.916667 | 0.785714 |
| Initial-DDIM20 LPIPS only | 0.787281 | 0.854501 | 0.611111 | 0.916667 | 0.733333 |
| Initial-DDIM20 product | 0.793860 | 0.858131 | 0.611111 | 0.916667 | 0.733333 |
| Naive stitch L1 | 0.410088 | 0.488216 | 0.571429 | 1.000000 | 0.727273 |
| Brightness | 0.679825 | 0.770309 | 0.571429 | 1.000000 | 0.727273 |

- Pixel globalmicro overfull240crop,continuousAP;image mean top576pixels.
  Historical different-sampler reference:ancestral300initial LPIPS-onlypixelAP
  0.266899/Dice0.339668,imageAP0.868242;not the matchedDDIM20baseline.
- SameLPIPS scoring before/after phase2:pixelAP0.157876->0.312330(+0.154454),
  Dice0.246871->0.371215,imageAP0.854501->0.944386. CalibrationLPIPSAP also
  increases0.238079->0.312952. On the SAMEfinal,LPIPSAP0.312330 exceedsL1
  0.099569 andfinalproduct0.196812. Report all fourarms,not a newholdout-selected
  winner or independently confirmed method. Oldweightedfinal0.154785 usedinitial
  heatmap and a different sampler/repeatbudget;do not attribute all differences
  from that historical score solely to fasterDDIM.
- Naive-stitch L1AP0.099308 vsfinalL1AP0.099569 is only+0.000262. No naive-stitch
  LPIPS control was measured,so the LPIPS gain cannot yet isolate learned
  inpainting from stitching effects in perceptual space. No proof of guaranteed
  tumor removal or superior normal restoration from exact-context preservation.
- Holdout pseudo-mask lesionrecall0.759256,precision0.123192,normalcoverage
  0.120276;calibration0.893122/0.106485/0.119341. Maskthreshold0.13344676792621613
  unchanged from smoke. It misses24.0744percent lesionpixels. LPIPS scores are
  spatial/nonlocal;do not treat maskcoverage as an anatomical segmentation quality
  claim. BrightnesspixelAP0.389506 still exceedsfinalLPIPS0.312330;Dice/FPR are
  at different recalls(0.788577 vs0.493572),not matched-sensitivity superiority.
- Meanphase1/phase2denoising0.588042/0.874035s across83images,sum1.462077s;
  fourcomponent medians0.577506/1.457580/1.463789/1.464696s. Historicalmatched
  TWO-image denoising reference gives9.187524xspeedup,priorT4container,not83image
  end-to-end speedup. Total186.571349s before upload includes setup,calibration,
  score metrics andI/O(2.247848s/image jobaverage,not isolatedinference latency).
  Peak104671232B. No new GPU run or unit rerun on this artifact-onlycheck.

### 2026-11-06 — Supplied DDIM20+30 single-pass ablation smoke

- Run `autoddpm-ddim2030-smoke-t4-20261106T073737Z`: **completed/uploaded**.
  [Modal app](https://modal.com/apps/phutungcogioihaitan/main/ap-mSM7Tlr1zK7LHFEEoS2fiu),
  [private immutable revision](https://huggingface.co/dacn-bmad-team/dacn-runs/tree/ac8aaaa082fa72363c53253e5db4317f4d4365ea/evaluations/autoddpm-ddim2030-smoke-t4-20261106T073737Z).
  User supplieddocs/testing_Lpips.md while the older10+10x5smoke finished;
  this is the new active contract,not a repeat of the older budget.
- Configconfigs/autoddpm_ddim20_30.json:20steps fromt300,30steps fromt50,
  eta0,onepass,seed17. Fourrequired scoring arms alluse the sameinitial/final;
  finalLPIPS andfinalproduct are actually computed againstfinal reconstruction.
  No hidden5repeat loop;U-Net hooks assert20+30 per output image.
- Only2outputimages,plus20normal calibrationinitials needed for propermaskfit;
  extra calibration calls explicitly counted. No smokeAP. Source/cache/LPIPS
  provenance retained;full83requires separate authorization per suppliedtask.
- Verified rootJSONSHAaeef3c54e055985edbd579afd3936ae27e098ca4b10a9aae0f729f98ce1f8955
  against manifest;source7b76a23f2712aa7f3bb26698aafdf6311ef26e088c4b62fadedd5bacb44869e1.
  Actual calls peroutput[50,50],phase2passes1,outside-maskerror0,finite maps and
  private2example panels saved. Total480U-Netcalls:100for2outputimages plus380
  for19additionalnormal-calibrationinitials. LPIPScalls23,not U-Net evaluations.
- Mask threshold0.13344676792621613 fit20normal calibration,not ancestral numeric
  reuse. Meanphase1/phase2denoising1.059434/0.885648s on2outputs;sum1.945082s.
  Historical matchedtwo-image ancestral denoisingmean16.975729s gives8.727513x
  denoising-only speedup,cross-containerT4 scope,not83image or end-to-end speedup.
  Score component latency means:M1=1.059616s,M2=2.007998s,M3=2.014264s,
  M4=2.015014s;shared-run component sums exclude transfers/mask assembly/global
  calibration,I/O andstartup. Total22.787917s before upload,peak104671232B.
- No AP/AUROC/F1 measured fromsmoke;historical metrics inartifact are references,
  not new20+30results. Full83 **not started**,awaiting explicit authorization as
  docs/testing_Lpips.md requires. Exactfullcommand inAUTODDPM_DDIM20_30.md.

### 2026-11-06 — AutoDDPM two-phase DDIM10 bounded smoke

- Run `autoddpm-ddim-smoke-t4-20261106T072722Z`: **completed/uploaded; superseded before full**,
  [Modal app](https://modal.com/apps/phutungcogioihaitan/main/ap-ghqmk2bcm1tRiBMyajKhEu).
  OriginalGaussianEMA30,LPIPS squeeze unchanged; fixed t300DDIM10 phase1 and
  t50DDIM10 x5phase2,eta0. Cumulative-alpha forward re-noising across skipped
  gaps;exact outside-mask context preservation. No model training or mask sweep.
- Smoke replays twofullancestral initial/final pairs with synchronized phase
  timings,then twoimages each forphase2-only and both-phase accelerated arms.
  Normal20calibration DDIMinitials are also generated tofit the unchangedmaskrule.
  No smoke AP;algorithmNFE550/350/60 is not measured wall speedup.
- Dataset/source/cache/LPIPS checks and masks/metric/timing protocol in
  AUTODDPM_DDIM.md. Code parente75469bedfe757ad9651e03ab6cfe2cfd2b5fd5a,
  dirty source`29f8804fdd554a3e11f892c145f6b17c291c9f4b827e4b1c468993cedea868b0`.
  Private revision
  [d3d088ed](https://huggingface.co/dacn-bmad-team/dacn-runs/tree/d3d088ed7cbf491f690db265eb7a16f3d308c184/evaluations/autoddpm-ddim-smoke-t4-20261106T072722Z).
  ResultSHA40ce933e3c848f9da8066f2235d5ebe72a7046a96c65e2cb4492b8314340a396
  matches manifest;1510actualnewU-Net calls(1100ancestralprobe+210initial+200inpaint),
  finite2imagearms;ancestralinitial/final replayerrors[0,0]. Total64.330359s,
  peak105523200B. No AP measured. No full10+10x5run launched after user supplied
  new20+30specification. Retained as two-image ancestral timing reference:
  normalphase1/2=10.550942/7.415179s,abnormal8.701458/7.283880s;not83image means.

### 2026-11-06 — Full83 LPIPS-only and pseudo-mask explanatory audit

- Run `perceptual-audit-full83-t4-20261106T065328Z`: **completed/uploaded**,
  [Modal app](https://modal.com/apps/phutungcogioihaitan/main/ap-VmGU48SYKuWh8W685WGCf6).
  Passed2image product/mask replay first;full83uses same frozen LPIPS and cached
  GaussianEMA30 initial images,zeroDDPM/re-sampling/training calls. No mask redesign.
- New twoarms LPIPS-only and normalized-absolute-only;report existing8controls.
  Thresholds fit40calibration,43group holdout unchanged. FP proxy regions,
  lesion boundary/core and8connected component bins fixed before run;gradient
  high threshold from20normal calibration only. Private4fixed example panels,
  no public MRI and no final-test access. SeePERCEPTUAL_MASK_ROADMAP.md.
- Code parente75469bedfe757ad9651e03ab6cfe2cfd2b5fd5a,dirtysource expected
  `4689409edc64b047b0822dfe966e377fda4e0fcd4c4f398eab19a11365f2e297`.
  [Private immutable revision](https://huggingface.co/dacn-bmad-team/dacn-runs/tree/3b82d8b2bbfa62d887dbb2f687a790905c351113/evaluations/perceptual-audit-full83-t4-20261106T065328Z).
  No method selection from the repeatedly inspectedholdout.
- Durable manifest confirms06:54:33UTC,TeslaT4,modefull,identicalsmoke source,
  diffusion_model_calls=0,mask_redesign=false,held_out_test_used=false.
- Completed06:55:00UTC (13:55Vietnam),83samples;total25.922446s before upload,
  peak45882880B. Root JSON SHA matches manifest:
  `224e17448d671bf96de619efe4196a1d0a2992bc07e29915a7c0499df63ba0fa`.
  Product replay maximumerror0 and pseudo-maskexactmatchTrue across83samples.
  Frozen diagnostic gradient threshold0.352968424559. Four private example PNGs
  and exact new maps persisted;not downloaded locally or anatomically reviewed.
- New score arms on43group holdout,thresholds selected40calibration only.
  Pixel globalmicro240crop includes background;normalFPR empty-mask normal images.

| Score | Pixel AUROC | Precision | Recall | F1/Dice | Pixel AP | Normal FPR |
|---|---:|---:|---:|---:|---:|---:|
| LPIPS only | 0.946559 | 0.284005 | 0.422469 | 0.339668 | 0.266899 | 0.007686 |
| Normalized absolute only | 0.874988 | 0.116323 | 0.299978 | 0.167640 | 0.102264 | 0.057595 |

| Score | Image AUROC | Image AP | Precision | Recall | F1 |
|---|---:|---:|---:|---:|---:|
| LPIPS only | 0.780702 | 0.868242 | 0.606061 | 0.833333 | 0.701754 |
| Normalized absolute only | 0.500000 | 0.558140 | 0.558140 | 1.000000 | 0.716418 |

- Pixel/image thresholds LPIPS0.266326934099/0.253568206500,normalized
  absolute0.916695535183/1.0. Per-image clipping saturates top5percent at1,
  so the top1percent image-score aggregation is constant1 for normalized residual;
  its image AUROC0.5/APclass-prevalence is expected,not evidence of detection.
- Controls unchanged: rawabsolute AP0.101066/Dice0.173678;initialLPIPSproduct
  0.204298/0.274354;weightedfinal0.154785/0.226256;brightness0.389506/0.268051.
  LPIPS-only AP exceeds product by0.062601 on holdout. CalibrationAP also shows
  LPIPS0.316582 > product0.285192 > weightedfinal0.196476. The ablation supports
  a strong perceptual-scoring contribution and residual multiplication reduces
  AP in this model/cohort;not an independently confirmed new-method claim.
  LPIPS-only imageAP0.868242 is below product0.887665;do not claim all metrics win.
- Pseudo-mask diagnostic onholdout:9048/55694 lesionpixels missed(16.2459%).
  Boundary<=3pixels misses2497/13693(18.2356%),core>3pixels misses6551/42001
  (15.5972%). Core contributes72.4027percent of allmisses due its larger area,
  not a higher within-core missrate. Not merely boundary undercoverage.
- Pseudo-mask false positives303007:background17424(5.7504percent),foreground
  boundary53260(17.5772percent),interior highgradient5613(1.8524percent),interior
  other226710(74.8201percent). Counts and geometric miss strata sum exactly.
  Within-region normal-image pseudo-mask rates:background0.006951,boundary
  0.432058,highgradient0.780517,interiorother0.418644. Highgradient is a small
  area despite high rate;mostFP counts are not there. These proxies are not
  anatomy labels and do not establish ventricle/cortex-specific errors.
- Connected lesion components(8connected,pixel areas) whollymissed bypseudo-mask:
  3/29 area1–25,3/6 area26–100,0/3 area101–400,0/22 area>400. LPIPS calibrated
  detection threshold whollymisses20/29 smallestcomponents despite better pixelAP;
  distinguish high-recall pseudo-mask from final segmentation threshold.
- LPIPS-only normalFPR0.007686 and Dice0.339668 vs brightness0.134935/0.268051
  occur at different sensitivity(LPIPSrecall0.422469 vsbrightness0.788577),not
  matched-recall superiority;brightnesspixelAP0.389506 remains higher. FP share
  depends on regionarea;report within-region rate alongside countshare.
- No new mask design chosen or tuned from this audit;no independenttest,
  training,reverse chain or extra GPU run on check. Freeze the later mask method
  and validate independently,as the user's roadmap requires.

### 2026-11-06 — Perceptual-only ablation and pseudo-mask audit smoke

- Run `perceptual-audit-smoke-t4-20261106T065148Z`: **completed/uploaded**,
  [Modal app](https://modal.com/apps/phutungcogioihaitan/main/ap-FvyVHTJjk0cuYi4Hso4zJw).
  TwoLPIPS image-pair calls,zeroDDPM/re-sampling/training calls;source artifacts
  fromcompleted AutoDDPM LPIPS t300 evaluation. No smoke AP.
- Replay same LPIPS state/product maps and exact pseudo-masks before scoring
  ablation. New LPIPS-only and normalized-absolute-only maps,normal-calibration
  gradient diagnostic threshold,unchanged40/43partition. Fixed diagnostic strata
  and independent-confirmation boundaries inPERCEPTUAL_MASK_ROADMAP.md.
- Code parente75469bedfe757ad9651e03ab6cfe2cfd2b5fd5a,dirty package snapshot
  `4689409edc64b047b0822dfe966e377fda4e0fcd4c4f398eab19a11365f2e297`.
  Private revision
  [402250a5](https://huggingface.co/dacn-bmad-team/dacn-runs/tree/402250a5f04037a03881be94cabff9ef8a82e09b/evaluations/perceptual-audit-smoke-t4-20261106T065148Z).
  Root result hash verified against manifest;product replay maximumerror0,
  pseudo-mask exactmatchTrue. Frozen diagnostic gradient threshold0.352968424559
  fromnormal calibration only. Total5.726332s before upload,peak45882880B.
  TwoLPIPS-only inputs,all outputs finite;no AP measured. Full comparative
  metrics are not smoke results.

### 2026-11-06 — LPIPS AutoDDPM t300 full83 comparison

- Run `autoddpm-lpips-full83-t4-20261106T061833Z`: **completed/uploaded**,
  [Modal app](https://modal.com/apps/phutungcogioihaitan/main/ap-Q53OLMLNmmu3h6rZCVv2cV).
  Same fixed source/config after successful2image LPIPS smoke; all83validation,
  40calibration/43group holdout, no final test and no training.
- Durable manifest started06:19:02UTC,TeslaT4,modefull;resampling1/83 confirmed.
  Exact source SHA matches smokeb4a1fb55cfed8acb8aaa2004b04ae76e581e6f0b05d71e4bc07a7f7f4bea18b2.
- Primary final absolute residual times initialLPIPSheatmap,8fixed score arms,
  including initialDDPM absolute/SSIM,naive stitch and brightness. Thresholds fit
  calibration only; report pseudo-mask lesion coverage and normal ROI fraction.
  Do not claim selective tumor removal from imposed exact-context preservation.
- Completed06:31:00UTC (13:31Vietnam);total715.978341s before upload,
  peak104408576B. Private immutable revision
  [bab00576](https://huggingface.co/dacn-bmad-team/dacn-runs/tree/bab00576bb8271ddb191c8fcd13840eff40f1919/evaluations/autoddpm-lpips-full83-t4-20261106T061833Z).
  Root aggregate result SHA verified against manifest:
  `6fb277491308c63b2cc52fa614fc912486266a9c3322f7f099aff850cb51a7e0`.
- Full83samples verified,no final test;existing40calibration/43group holdout,
  pooled20normal-calibration mask threshold0.13774138689041138 unchanged from
  smoke. LPIPSstate/source/checkpoint/cache hashes unchanged. Baseline and
  brightness AP reproduce. Score thresholds fit calibration and frozen.
- Allfollowing metrics are on43group holdout,pixel globalmicro240crop;
  normalFPR is positive-pixel fraction on empty-mask normal images. One seed.

| Method | Pixel AUROC | Precision | Recall | F1/Dice | Pixel AP | Normal FPR |
|---|---:|---:|---:|---:|---:|---:|
| Initial absolute | 0.876854 | 0.119934 | 0.314702 | 0.173678 | 0.101066 | 0.058079 |
| Initial SSIM | 0.888870 | 0.110476 | 0.257712 | 0.154654 | 0.102402 | 0.055234 |
| Initial LPIPS heatmap | 0.915842 | 0.268360 | 0.280623 | 0.274354 | 0.204298 | 0.010727 |
| Naive stitch absolute | 0.859269 | 0.145251 | 0.346195 | 0.204641 | 0.118781 | 0.047041 |
| Resampled absolute | 0.863330 | 0.161567 | 0.383865 | 0.227416 | 0.129407 | 0.044342 |
| Resampled SSIM | 0.922853 | 0.158391 | 0.379323 | 0.223469 | 0.149446 | 0.045201 |
| Weighted resampled absolute (primary) | 0.866490 | 0.185318 | 0.290408 | 0.226256 | 0.154785 | 0.026640 |
| Brightness | 0.950197 | 0.161469 | 0.788577 | 0.268051 | 0.389506 | 0.134935 |

| Method | Image AUROC | Image AP | Precision | Recall | F1 |
|---|---:|---:|---:|---:|---:|
| Initial absolute | 0.263158 | 0.430006 | 0.558140 | 1.000000 | 0.716418 |
| Initial SSIM | 0.372807 | 0.505258 | 0.558140 | 1.000000 | 0.716418 |
| Initial LPIPS heatmap | 0.807018 | 0.887665 | 0.606061 | 0.833333 | 0.701754 |
| Naive stitch absolute | 0.412281 | 0.481000 | 0.558140 | 1.000000 | 0.716418 |
| Resampled absolute | 0.491228 | 0.534048 | 0.585366 | 1.000000 | 0.738462 |
| Resampled SSIM | 0.554825 | 0.581114 | 0.600000 | 0.875000 | 0.711864 |
| Weighted resampled absolute (primary) | 0.739035 | 0.814398 | 0.633333 | 0.791667 | 0.703704 |
| Brightness | 0.679825 | 0.770309 | 0.571429 | 1.000000 | 0.727273 |

- Holdout pseudo-mask lesion recall83.7541%,precision13.3407%,ROI14.1171%;
  normal ROI after dilation11.7060%. Calibration lesion recall89.2360%,
  precision11.2842%,ROI13.0640%,normal ROI11.6789%. Mask misses16.2459% of
  holdout lesion pixels. Outside-mask reconstruction error exactly0 is imposed,
  not proof of better healthy-tissue restoration. Fivepercent calibration target
  is before dilation,not the final normal mask fraction or detection FPR.
- Conclusion:primary AP0.154785 and resampledSSIM0.149446 improve on baseline
  absolute0.101066/SSIM0.102402. Re-sampling adds0.010626 AP over naive stitch
  for unweighted absolute,but the initial LPIPS heatmap achieves higherpixelAP
  0.204298,imageAP0.887665,andF1/Dice0.274354 than any finalresampled score.
  Do not attribute the entire gain to inpainting or claim re-sampling is the best
  pipeline. Brightness remains higherpixelAP0.389506;LPIPSheatmap F1 is slightly
  higher at a different calibrated operating point with much lower recall.
- Exploratory inspected validation,one checkpoint/sampling seed;not significance,
  guaranteed tumor removal or exact official AutoDDPM reproduction. No further
  GPU job,mask tuning or final-test evaluation on this result check. Full protocol
  inAUTODDPM_LPIPS.md.

### 2026-11-06 — LPIPS AutoDDPM t300 bounded smoke

- Run `autoddpm-lpips-smoke-t4-20261106T061700Z`: **completed/uploaded**,
  [Modal app](https://modal.com/apps/phutungcogioihaitan/main/ap-852Pcx7xpt7qW71tJ7MVbD).
  CPU image preparation installs LPIPS0.1.4/torchvision and caches official
  SqueezeNet weights before GPU work. Private revision
  [87d0943c](https://huggingface.co/dacn-bmad-team/dacn-runs/tree/87d0943c3b1de7be211997c771c00edf495746b8/evaluations/autoddpm-lpips-smoke-t4-20261106T061700Z).
- Requested Gaussian EMA30 t300/cache -> LPIPS pseudo-mask -> stitch ->50steps
  x5repeat re-sampling. Threshold fit20normal calibration only, mask dilation3,
  seed17,batch1,T4. Only2images re-sampled;no smoke AP. Dataset/source/cache SHAs
  and full interpretation constraints in [protocol](AUTODDPM_LPIPS.md).
- Code parentd282f023afee1e33e4d7444ef842b10b69b6e649 with dirty source;
  exact snapshotb4a1fb55cfed8acb8aaa2004b04ae76e581e6f0b05d71e4bc07a7f7f4bea18b2.
  Root results checksum matches manifest. Prior t200 SSIM-proxy
  pilot remains separately recorded; this is an adaptation,not exact AutoDDPM.
- Started06:17:46UTC,completed06:18:18UTC. Result elapsed24.315037s before upload,
  peak104408576B;both2outputs finite,maximum outside-mask error0. No AP measured.
  Maskthreshold0.13774138689041138 fit20normal calibration only,dilation3,
  LPIPSstateSHA18fed4d1fc5c04f2918f64df3a2463344108576d1888f0d75538ca6d5c4dffe6.

### 2026-11-06 — Authorized full normal memory fusion on T4

- Run `memory-fusion-full83-t4-20261106T055134Z`: **completed/uploaded**.
  Started05:51:56UTC,completed05:54:22UTC (12:54:22Vietnam).
  [Modal app](https://modal.com/apps/phutungcogioihaitan/main/ap-UiDz4LoIR7Zey5Ld1cUuoA).
  User explicitly authorized full validation after the bounded smoke.
- Confirmed TeslaT4,modefull,bank budget10000,no train subset cap,
  held_out_test_used=false. All7500normal train slices supply the bank;
  existing83validation/40calibration/43group-holdout protocol,seed17/alpha0.5,
  normal-calibration median/IQR and frozen thresholds. Reuses baseline cached
  EMA30 ancestral t300 reconstructions; no DDPM/encoder optimization.
- Dataset `350c74f2313e4e73ca65e7ffc6e9dcf74040b3b7`; code parent
  `d282f023afee1e33e4d7444ef842b10b69b6e649`, dirty source snapshot
  `10ea0e671a1adeb29939896e94df9491db721bca622231939ec93026c629546e`
  matches successful smoke. Private output prefix
  `evaluations/memory-fusion-full83-t4-20261106T055134Z` on team run Volume;
  [private immutable revision](https://huggingface.co/dacn-bmad-team/dacn-runs/tree/01c5995f6ec19092df7bb4ac302861399f2de868/evaluations/memory-fusion-full83-t4-20261106T055134Z).
- Verified results SHA against manifest:
  `bfeb1f4573a6a0cef68cc68df6e8d86f0a3d7824080674cfb39e39547faaf476`.
  Bank10000vectors from1929258eligible descriptors/all7500normal training slices;
  bank SHA `458044bfaaa8c4a3ac8dbb71fe431ed23aef9d352e43e285b071b78207ad1452`.
  Total145.250545s before upload; bank-build phase46.909080s includes setup;
  peak118366208B. Full83 evaluated; thresholds calibrated40,frozen43group holdout.
- Pixel metrics below: global micro over240crop, AP from continuous maps,
  normal FPR on empty-mask holdout images; F1 is micro Dice. One model/bank seed.

| Method | Pixel AUROC | Precision | Recall | F1/Dice | Pixel AP | Normal FPR |
|---|---:|---:|---:|---:|---:|---:|
| DDPM absolute | 0.876854 | 0.119934 | 0.314702 | 0.173678 | 0.101066 | 0.058079 |
| DDPM SSIM | 0.888870 | 0.110476 | 0.257712 | 0.154654 | 0.102402 | 0.055234 |
| Memory only | 0.824773 | 0.080935 | 0.516555 | 0.139943 | 0.064212 | 0.134794 |
| Absolute + memory | 0.882715 | 0.122895 | 0.342155 | 0.180837 | 0.102374 | 0.061204 |
| SSIM + memory | 0.881029 | 0.097198 | 0.340198 | 0.151198 | 0.087033 | 0.079000 |
| Brightness | 0.950197 | 0.161469 | 0.788577 | 0.268051 | 0.389506 | 0.134935 |

| Method | Image AUROC | Image AP | Precision | Recall | F1 |
|---|---:|---:|---:|---:|---:|
| DDPM absolute | 0.263158 | 0.430006 | 0.558140 | 1.000000 | 0.716418 |
| DDPM SSIM | 0.372807 | 0.505258 | 0.558140 | 1.000000 | 0.716418 |
| Memory only | 0.703947 | 0.822745 | 0.558140 | 1.000000 | 0.716418 |
| Absolute + memory | 0.271930 | 0.434041 | 0.558140 | 1.000000 | 0.716418 |
| SSIM + memory | 0.574561 | 0.638276 | 0.615385 | 1.000000 | 0.761905 |
| Brightness | 0.679825 | 0.770309 | 0.571429 | 1.000000 | 0.727273 |

- Normalization fit20normal calibration slices only. Median/IQR:
  absolute0.010425895/0.019146100,SSIM0.010653948/0.312484438,
  memory1.974215567/1.045310348; IQR floor inactive. Alpha0.5 fixed independently
  for both arms. Baseline/brightness AP reproduce registered controls.
- Conclusion: absolute fusion AP gain only0.001308, F1 increases with higher
  normal FPR; SSIM fusion AP decreases0.015370. No meaningful localization
  improvement established, brightness much stronger. Memory imageAP0.822745
  exceeds brightness0.770309 in this fold, but pixelAP0.064212 is weak; its
  calibrated image threshold predicts every holdout image positive, so improved
  ranking is not useful thresholded detection at this operating point.
- Calibration recall operating points0.5/0.8 are frozen, not tuned on holdout.
  At target0.5 absolute baseline actual holdout recall0.397422/FPR0.074755,
  fusion0.389988/0.070659; at0.8 baseline0.744712/0.169766,
  fusion0.731731/0.172324. Not exactly matched sensitivity; no unconditional
  false-positive reduction claim. Repeatedly inspected validation; exploratory,
  no final-test access or extra GPU experiment during this result check.

### 2026-11-06 — Normal memory fusion bounded T4 smoke

- Run `memory-fusion-smoke-t4-20261106T054850Z`: **completed/uploaded**.
  [Modal app](https://modal.com/apps/phutungcogioihaitan/main/ap-VhtxgtRhs1AZl73YfYqZHp),
  [private immutable artifacts](https://huggingface.co/dacn-bmad-team/dacn-runs/tree/56552b5792438971d008bb02eb1b4a72c6843902/evaluations/memory-fusion-smoke-t4-20261106T054850Z).
- Dataset `350c74f2313e4e73ca65e7ffc6e9dcf74040b3b7`; code parent
  `d282f023afee1e33e4d7444ef842b10b69b6e649`, dirty source snapshot
  `10ea0e671a1adeb29939896e94df9491db721bca622231939ec93026c629546e`.
- Official frozen grayscale ImageNet ResNet18 layer2/layer3,384D descriptors,
  seed17 uniform random priorities,16 normal train slices,4165eligible vectors,
  bank budget128. No optimizer, no encoder/DDPM training, no final-test access.
  Bank SHA `cdd42ae60b8a374c870d1961d7e8f2a716ca04a2e2370b554c7ce82168f30f3d`.
- Verified original Gaussian EMA30/t300 baseline cache SHA
  `8278ad18543dd87f698c36b510564b9fcd4445d4a1b0ae5d7a859093e2a4e959`
  and full83image/mask/label ordering; scored only2validation images. Persisted
  bank reload/checksum and finite240crop memory/absolute-fusion/SSIM-fusion pass.
- Result total8.230089s before upload; bank-build phase6.018607s includes setup,
  peak117029888B. Manifest started05:48:20UTC,completed05:48:29UTC.
  **No AP/AUROC/P/R/F1 or threshold result measured on smoke.** Full7500normal/
  10000bank/83validation remains unrun, pending explicit authorization per
  [the specification](fusionmemorybank.md). See [protocol](DDPM_MEMORY_FUSION.md).

Add one row per completed evaluation protocol. Split and metric level must be
clear; do not combine metrics from different splits or score definitions.

| Run ID | Method | Dataset revision | Split | Seeds completed | ROC-AUC | Precision | Recall | F1 | Pixel AP / Dice | Latency / VRAM | Artifacts / manifest | Status |
|---|---|---|---|---:|---:|---:|---:|---:|---|---|---|---|
| `ddpm-val-e5-e10-t100-200-300-20261031T074028Z` | Partial DDPM epoch 10, `t*=300`, local one-minus-SSIM; pixel level | `350c74f2313e4e73ca65e7ffc6e9dcf74040b3b7` | validation, 83 slices | 1 evaluation seed | 0.892010 | 0.101571 | 0.395303 | 0.161616 | AP 0.095895 / macro lesion-slice Dice 0.241476 | 11.210 s/slice / 100,651,520 B | [private revision `51a9c22e`](https://huggingface.co/dacn-bmad-team/dacn-runs/tree/51a9c22ee71bdae7d9d78538fb8664a7b96c9e48/evaluations/ddpm-val-e5-e10-t100-200-300-20261031T074028Z) | validation-selected candidate; test sealed |
| `ddpm-val-e10-t300-posterior-mean-t4-b8-20261101T182333Z` | Partial DDPM epoch 10, `t*=300`, posterior mean, local one-minus-SSIM; pixel level | `350c74f2313e4e73ca65e7ffc6e9dcf74040b3b7` | validation, 83 slices | 1 evaluation seed | 0.861427 | 0.096832 | 0.254533 | 0.140292 | AP 0.081721 / macro lesion-slice Dice 0.207741 | 4.912 s/slice / 580,116,480 B | [private revision `3a91ec15`](https://huggingface.co/dacn-bmad-team/dacn-runs/tree/3a91ec15f28d4c0227d36c177a15373f09ff478b/evaluations/ddpm-val-e10-t300-posterior-mean-t4-b8-20261101T182333Z) | completed ablation; worse than ancestral; test sealed |

## Per-run record

### ddpm-uncertainty-k4-val83-t4-20261105T163529Z — Gaussian EMA30 K4 scoring83 — completed
- After bounded K4smoke passed,user-authorized full inference-only experiment,
  shared profile/Workspace phutungcogioihaitan,
  [Modal app](https://modal.com/apps/phutungcogioihaitan/main/ap-R0EC27PZnPnALYHMySkxkH).
  OneT4,batch1,t300,hardtimeout2h,no training or encoder.
- Original Gaussian EMA30 source d3e06d2082e18234899b201718abd59078c8a800d66da3cc5d8b9e0d9ca9b946;
  drawbases17/1017/2017/3017. Immutable cached draw1/base17 reused after
  2image replay;249new reconstructions across83images for remaining3draws.
  Atomic prefix caches/commit every5images,checkpoint/index/seed/source checked
  on resume. No Spatial checkpoint or extra training used.
- Dataset350c74f2313e4e73ca65e7ffc6e9dcf74040b3b7,index
  57ad2e31df3ed4e625da00efaecd6fddc1e2a132329dfba6e60867caf1802db3,
  code7134b105c0c4d4fb620715908f100103b1d0b0ab+dirty source
  15a90894fc8dcd6578a64256710af85604ae4fac3d1ef66e147ecd9d7f878583.
- Fixed6scores:single absolute/SSIM,mean4 absolute/SSIM,variance-normalized
  squared residual,pooled normal-calibration3x3patch Mahalanobis. Variance floor
  median normalcal variance/min1e-6;covariance shrinkage0.1/ridge fixed.
  Parameters normal-calibration only,persisted before holdout metrics.
  Thresholds40calibration/43group holdout,25groups each,globalmicro240crop,
  image mean top576. All83descriptive thresholds separate. No holdout tuning.
- Full AP/F1/FPR,image metrics,normal quality,lesion/nonlesion score/variance
  means,latency/VRAM and private ensemble/calibration artifacts planned. Six
  comparisons fixed before launch,no claims from smoke. One fixed ensemble/model
  on previously inspected validation,not4trainingseeds. Held-out test sealed.
  App created;metrics pending. Leave detached after start verification.
- Start verified:manifest2026-11-05T16:35:52UTC,running,
  reconstruction_draw_2_of4,TeslaT4,exact source snapshot. Cached baseline
  replay errors[0,0];draw2progress5/83 at67.945679s inclsetup/preflight.
  Detached local client after confirmation;no continuous monitoring.
- Completion checked06/11/2026local:all3newdraws83/83,full249newreconstructions,
  replayerrors[0,0],normal-calibration20images/1152000patch vectors. Completed
  2026-11-05T17:17:40UTC(06/11 00:17local),completed/uploaded manifest,
  [private revisionaf0bf4006db4c948292b9bc50748073e1b421bb4](https://huggingface.co/dacn-bmad-team/dacn-runs/tree/af0bf4006db4c948292b9bc50748073e1b421bb4/evaluations/ddpm-uncertainty-k4-val83-t4-20261105T163529Z).
  ResultsSHAfc12e2093045f9b6f7c0850e683a7c0258c05529d6a92580419101bf77f8ce16
  matches manifest artifact;sourcecheckpointSHA,83samples,sealed-test flag
  verified locally. Single-draw AP reproduced registered baseline exactly.
- Calibration frozen:variance_floor0.00005194913791110404,shrinkage0.1,
  ridge0.000000835616497628855. Covariance/mean from normal calibration only.
  All thresholds below independently calibrated on40images then applied to43
  group-holdout;globalmicro240crop,image mean top576,one ensemble/model.

| Score,43group holdout | Pixel ROC-AUC | Precision | Recall | F1/microDice | Pixel AP | Normal pixel FPR |
|---|---:|---:|---:|---:|---:|---:|
| single_absolute |0.876854|0.119934|0.314702|0.173678|0.101066|0.058079|
| single_ssim |0.888870|0.110476|0.257712|0.154654|0.102402|0.055234|
| mean4_absolute |0.872069|0.124609|0.284321|0.173276|0.103810|0.052835|
| mean4_ssim |0.879083|0.116795|0.179714|0.141579|0.099472|0.038697|
| variance_normalized |0.534862|0.096144|0.172352|0.123433|0.045729|0.040329|
| patch_mahalanobis |0.874171|0.087801|0.474827|0.148199|0.077657|0.137883|

| Score,43group holdout | Image ROC-AUC | Precision | Recall | F1 | Image AP |
|---|---:|---:|---:|---:|---:|
| single_absolute |0.263158|0.558140|1.000000|0.716418|0.430006|
| single_ssim |0.372807|0.558140|1.000000|0.716418|0.505258|
| mean4_absolute |0.307018|0.547619|0.958333|0.696970|0.453941|
| mean4_ssim |0.405702|0.558140|1.000000|0.716418|0.539785|
| variance_normalized |0.594298|0.536585|0.916667|0.676923|0.720711|
| patch_mahalanobis |0.416667|0.558140|1.000000|0.716418|0.542902|

- Pixel/image thresholds in score order above:
  0.253657147288/0.494910077574;0.956885287835/1.154324681719;
  0.207741984050/0.383705765685;0.909948174235/1.046867417355;
  4.402128372042/52.600949796141;14.172663102923/78.649962948236.
  All83descriptive PixelAP/F1 respectively:
  singleabs0.099834/0.182870,singleSSIM0.100741/0.167483,
  mean4abs0.101912/0.185849,mean4SSIM0.102312/0.161804,
  normalized0.042073/0.119290,Mahalanobis0.076216/0.156224.
- Main result:averaging helps absolute AP slightly(0.101066->0.103810),not
  SSIM AP;normalized and pooled-patch Mahalanobis do not beat mean4controls for
  pixel localization. Lower FPR in meanSSIM/normalized accompanies lower Recall;
  not an improvement at matched sensitivity. Mahalanobis FPR rises13.79%.
- Variance-normalized image AP0.720711/AUROC0.594298 is better than original
  image scores but not lesion localization. Existing brightness control image
  AP0.770309/AUROC0.679825 remains higher under sameholdout. One previously
  inspected validation ensemble,no significance/generalization claim.
- Normal tissue mean4 MAE[0,1]0.063042679/SSIM0.408378352 vs single
  0.082674084/0.292438059. Mean variance normalforeground0.032296945 and
  lesion0.041919858. Normalized score abnormal-slice macro lesion5.556999 vs
  nonlesionforeground11.326831,showing poor separation in this scoring design;
  consistent with suppressing lesion signals,not proof of one causal mechanism.
  Mahalanobis corresponding means24.278770/18.250254.
- End-to-end manifest walltime2507.975575s(~41.80min). Later draw elapsed
  counters restart(draw4final365.411s < draw3final1564.867s),consistent with
  prefix-cache resumption;do not sum those counters or call final-attempt timing
  full run latency. Final-attempt inference phase367.987923s,total469.895990s,
  peak101438464B. Private arrays/calibration/summary uploaded;no new GPU work
  during status check,no training or held-out-test access.

### ddpm-uncertainty-k4-smoke-t4-20261105T163114Z — Gaussian EMA30 K4 scoring smoke — completed
- Explicit user inference test,K4,t300,original GaussianEMA30 sourceSHA
  d3e06d2082e18234899b201718abd59078c8a800d66da3cc5d8b9e0d9ca9b946.
  Profile/Workspace phutungcogioihaitan,Secret name verified,
  [Modal app](https://modal.com/apps/phutungcogioihaitan/main/ap-XWKFboy5nw35LHkFTNev1O).
- Bounded1normal+1abnormal,4draws per image,bases17/1017/2017/3017. Cached
  originaldraw1 reused after two exact replay checks;6new reconstructions.
  Finite unbiased variance,normalized squared residual and normal-only pooled
  3x3patch Mahalanobis checked. No AP computed on this tiny sample,no training.
- Dataset350c74f2313e4e73ca65e7ffc6e9dcf74040b3b7,immutable baselinecache
  8278ad18543dd87f698c36b510564b9fcd4445d4a1b0ae5d7a859093e2a4e959,
  code7134b105c0c4d4fb620715908f100103b1d0b0ab+dirtysnapshot inmanifest.
 146tests passed27.89s,targeted Ruff/format/whitespace passed. Parameters
  and full protocol fixed in UNCERTAINTY_SCORING.md before GPU submission.
  Full83/249newdraws only after smoke success;no test access or claim of gain.
- Verified completed/uploaded501d5f86ee79ddd3d81b1cfa7b62c53a620670a8,
  replay maxerrors[0,0],draws2/3/4 each2/2completed,finite normalized/Mahalanobis
  smoke passed. Final draw progress84.248039s inclsetup/preflight/6newchains.
  Frozen source15a90894fc8dcd6578a64256710af85604ae4fac3d1ef66e147ecd9d7f878583.
  No AP result on2images. Independent full validation permitted after smoke.

### spatial-imagenet-gray-t4-b8-e10-20261105T141939Z — spatial ImageNet grayscale10 + validation — completed
- User-requested spatial feature trial with explicit1channel grayscale. After
  smoke3 passed,independent full run submitted to shared Workspace/profile
  phutungcogioihaitan,
  [Modal app](https://modal.com/apps/phutungcogioihaitan/main/ap-z896lVzq2nUac9XbgTsY4L).
- Config spatial_imagenet_grayscale_t4_epoch10.json,T4,batch8,seed17,
  7500normal train,10fine-tune epochs/9380updates. Original U-Net EMA30 strict
  source SHA d3e06d2082e18234899b201718abd59078c8a800d66da3cc5d8b9e0d9ca9b946,
  not smoke/A/B/control weights. ResNet18 official IMAGENET1K_V1 hash
  f37072fd47e89c5e827621c5baffa7500819f7896bbacec160b1a16c560e07ec,
  conv1 meanRGB->1channel,MRI[-1,1],noRGB repeat/norm/resize/crop.
- Four spatial maps64×64×64/128×32×32/256×16×16/512×8×8,no pooling;
  zero1x1 residual fusion into U-Net down terminal blocks[8,11,14,17],then
  bottleneck/skips. Jointtrain U-Net+encoderLR1e-5,fusion1e-4,freshAdamW
  betas0.9/0.999,wd0,clip1,EMA0.9992 with copied online BatchNorm buffers.
  Gaussian epsilonMSE,linearT1000. Best selected39normal-val/fixedseed1000017;
  save initial0,last each epoch,best,milestones1/5/10 andImageNet provenance.
- Dataset350c74f2313e4e73ca65e7ffc6e9dcf74040b3b7,code7134b105c0c4d4fb620715908f100103b1d0b0ab
  plus exact source snapshot in manifest.4hour timeout,private HF upload/durable
  checkpoints,source/config-checked resume;no held-out-test access.
- After training auto83slice t300 ancestral/batch1/base17 validation,shared
  40calibration/43group holdout/calibration-only thresholds,paired difference/
  SSIM,normal FPR,quality/latency/VRAM/private figures. Compare registered
  original30/A/B/unconditional control caches/results after SHA/protocol checks.
  Trial jointly changes spatial fusion/pretraining/encoderLR;not isolated
  spatial-only ablation,one exploratory seed. No AP improvement claim yet.
- Start verified:durable manifest2026-11-05T14:27:12UTC,running/training,
  TeslaT4,epoch0recovery checkpoint committed,exact source snapshot
  2f6c5005dd22618a4d49fa56707b9779143f12fab04467b580681cdd6c1118e8.
  Actual epoch1/10 step150/938,finite running epsilonMSE0.008948. Detached
  local client after start confirmation;job continues without monitoring.
- Completion verified:10epochs/9380updates,best normal-val EMA10,
  epsilonMSE0.004498155053,4507.506611s train+val,peak5883574272B.
  Training completed/uploaded29777bf14204bdc7b4450f2b9a0746b1a537911c at15:45:05UTC.
  No new training on user check.
- Validation83/83 completed16:00:37UTC,completed/uploaded revision
  [a6e21e5c2553e683fda3e51496fdc52a31662b60](https://huggingface.co/dacn-bmad-team/dacn-runs/tree/a6e21e5c2553e683fda3e51496fdc52a31662b60/evaluations/spatial-imagenet-gray-t4-b8-e10-20261105T141939Z-val).
  Bestcheckpoint SHA6a4103bb14937421fc79e595fb7fdd7d460774dc68e2db1ef3e8c75b04d307cb;
  resultsSHA0c9b75267a8d6f26a54c071ab26e263ea3fc0b292a438ff112acd2a15dd5299f
  matches manifest artifact. CheckpointSHA,samples83,sealed-test flag verified.
  SelectedEMA10,t300,base17,40calibration/43group holdout/25groups each,
  calibration-only thresholds,globalmicro240crop,image mean top576,one seed.

| Score,43group holdout | Level | ROC-AUC | Precision | Recall | F1/microDice | AP | Normal pixel FPR |
|---|---|---:|---:|---:|---:|---:|---:|
| Difference | pixel |0.795266|0.072812|0.337595|0.119789|0.063110|0.120666|
| Local1-SSIM | pixel |0.843081|0.074047|0.861924|0.136377|0.065459|0.260876|
| Difference | image |0.217105|0.512821|0.833333|0.634921|0.429099|n/a|
| Local1-SSIM | image |0.208333|0.472222|0.708333|0.566667|0.418773|n/a|

- Thresholds pixel/image:difference0.067786693573/0.227941763770;
  SSIM0.067449256461/0.731668359946. Macro lesion-slice Dice
  difference0.189223981/SSIM0.216837302. All83 descriptive AP/F1/FPR:
  difference0.062165556/0.130531382/0.170679309;
  SSIM0.066047096/0.121576622/0.271205929. All83thresholds descriptive only.
- Holdout PixelAP baseline30/control/A/B/spatial:difference
  0.101066/0.104391/0.104645/0.104464/0.063110;SSIM
  0.102402/0.104004/0.104247/0.104084/0.065459. Both threshold-independent AP
  scores lower;not solely a threshold problem. Brightness holdoutAP0.389506
  unchanged separate control. No significance/general-method claim from1seed.
- Normal39foreground MAE[0,1]0.039324712/SSIM0.688031178 versus control
  0.082240741/0.295440689. Abnormal44 macro lesion/nonlesion SSIMerror
  0.276705021/0.271836988;difference0.073481523/0.063075532. Improved
  normal reconstruction but weak lesion separation is consistent with original
  spatial condition retaining pathology;mechanism not proven by these means.
- Loop837.866527s,peak151107072B,total916.399822s. Private images/cache
  uploaded. Lower noiseMSE did not imply better anomaly detection. No new
  inference,extra epochs or test-set access initiated during status check.

### spatial-imagenet-gray-smoke3-t4-20261105T141647Z — grayscale spatial smoke3 — completed
- User-requested spatial ImageNet pilot with explicit grayscale1channel.
  App ap-Gkt1RTxe4nYswmOMDt2AnX,image im-HkuyXIq2wxtmzvaYr4i0PF built
  successfully with official pretrained hash check before GPU allocation.
  Required workspace/Secret verified;shared profile phutungcogioihaitan.
- Config spatial_imagenet_grayscale_smoke.json,1epoch,batch2,normal16train/
  4normal-val,seed17;U-NetEMA30originalSHA d3e06d2082e18234899b201718abd59078c8a800d66da3cc5d8b9e0d9ca9b946.
  ResNet18 IMAGENET1K_V1 conv1=mean RGB weights ->1channel,inputsMRI[-1,1],
  no RGB repeat/norm/crop. Four spatial maps,zero1x1 fusion at64/32/16/8,
  jointtrain U-Net/encoderLR1e-5,fusionLR1e-4,EMA0.9992.
- Dataset350c74f2313e4e73ca65e7ffc6e9dcf74040b3b7,code7134b105c0c4d4fb620715908f100103b1d0b0ab
  plus dirty snapshot in manifest.140tests passed17.19s;actual ImageNetfile
  hash/meanRGBadaptation/4map shapes checked locally. No AP or held-out-test result;
  weights from smoke are not reused.
- Remote smoke=passed,finite3step reconstruction,completed/uploaded private
  revision4d47de2f38c281e14401ed80a79ab315ef5186ba.15shard-aware updates,
  16normal train/4normal-val,best1,normal epsilonMSE0.000329594652,
  train+val3.862974s,peak1710858752B. Tiny subset MSE is not an AP result.
  Snapshot9ab3166e6a271edef902edb0aaaccbb92563344892bd75d43dd72114c2dc2be7.
  Smoke passed before full submission;subsequent Python edit only removes extra
  whitespace in cloud import block,not model/training behavior.

### spatial-imagenet-gray-smoke2-t4-20261105T141500Z — grayscale spatial smoke2 — image build failed
- App ap-0xrdUXDoZClwpR6h6KGhIb,image im-u7YjCDr3ssfButgK9qg8Tl. CPU build
  failed parsing Dockerfile because run_commands contained literal multiline
  Python source;no GPU training/data/checkpoint/AP started. Same bounded config
  and grayscale architecture as smoke1. Replaced with checked single-line
  stdlib Python download/checksum command;fresh smoke3 needed.

### spatial-imagenet-gray-smoke-t4-20261105T140929Z — grayscale spatial smoke1 — image build failed
- User requested spatial ImageNet+EMA30,then grayscale1channel. First sandbox
  request could not connect before app creation; retry with network permission
  initialized app ap-2XcnTEi9RUdSDrI79jCTJH.140local tests passed17.19s.
- Build im-tQ7GhEuXL1PQauMd6wRbNL failed before GPU training: run_function
  hydrated cloud module whose top-level imports required dacn before local source
  was mounted. ModuleNotFoundError,dacn. No training/checkpoint/AP or patient data
  access in this attempt. Intended bounded1epoch/16normal train/4normal-val,batch2.
- Fixed build to run self-contained stdlib download/checksum Python via
  run_commands,then mount local source. Official ImageNet weights hash already
  checked locally;meanRGB1channel adaptation and shapes tested. Use fresh smoke2
  ID;full10epoch pilot remains contingent on actual smoke success.

### ddpm-uncond-val-best36-t300-20261105T094934Z — matched control validation83 — completed
- After successful bounded validation smoke,user-authorized full evaluation on
  oneT4,ancestral300,batch1,base17 UUID sampling. Workspace/profile
  phutungcogioihaitan,
  [Modal app](https://modal.com/apps/phutungcogioihaitan/main/ap-2AEvfGt80K4eoxODBlGW4g).
- Fixed source ddpm-uncond-control-t4-b8-e40-20261105T082502Z/ddpm_best.pt,
  normal-val-selected epoch36(FT6),not milestone40,EMA,noencoder,
  SHA350791c48fd99babc38c329af5644a4c53a70f6628b6d751e85e3a6cab8cf0f7.
  Parent10epoch/9380update training completed/uploaded375bc6ba5578f5c27953f3306cecdabb531f073b.
- Dataset350c74f2313e4e73ca65e7ffc6e9dcf74040b3b7,83val39normal/44abnormal,
  existing40calibration/43group holdout(25groups each),calibration-only thresholds,
  difference/local1-SSIM and brightness control. Globalmicro240crop pixel
  metrics,image mean top576,normal tissue MAE/SSIM,lesion/nonlesion quality,
  latency/VRAM/private examples. Cached baseline30 plus immutable A/B results
  matched by protocol/index and exact image/mask/label/noisy arrays.
- Code7134b105c0c4d4fb620715908f100103b1d0b0ab+dirty source
  68092ef7fe5e470d833a1ce0909c84011438e7b41414e91f1761f5dc4968c93a.
  Repeated bounded replay preflight inside full worker;prefix cache every5images,
  source-checked resume,private HF upload,two-hour hard timeout. No new
  training,sweep,test access or AP-based selection.
  One exploratory seed on previously inspected validation; no significance claim.
- Start verified:durable manifest2026-11-05T09:50:15UTC,running,TeslaT4,
  selected36,exact frozen snapshot. Full preflight passed; actual reconstruction
  progress1/83 at11.104288s. Job subsequently completed83/83 before the local
  client disconnected; actual produced artifacts verified below.
- Completed2026-11-05T10:10:14UTC,manifest completed/uploaded at
  [private revision38491cba20168ac41a038e98cfaffe1bc3fb51b5](https://huggingface.co/dacn-bmad-team/dacn-runs/tree/38491cba20168ac41a038e98cfaffe1bc3fb51b5/evaluations/ddpm-uncond-val-best36-t300-20261105T094934Z).
  Full preflight original replay maxerrors[0,0],target36 smoke passed,exact
  input/mask/label/initial-noise pairing with B verified by worker. Result SHA
  babbd335d6a9c40e64d23b2e7dded4dd8f195e3522669586713488ced3f62c7e
  matches manifest artifact; actual checkpointSHA,83samples,sealed-test flag
  verified locally. Normal-val selection retained; no AP-based checkpoint search.

| Score,43group holdout | Level | ROC-AUC | Precision | Recall | F1/microDice | AP | Normal pixel FPR |
|---|---|---:|---:|---:|---:|---:|---:|
| Absolute difference | pixel |0.879392|0.125207|0.317036|0.179518|0.104391|0.056382|
| Local1-SSIM | pixel |0.889144|0.113874|0.260477|0.158469|0.104004|0.054682|
| Absolute difference | image |0.243421|0.558140|1.000000|0.716418|0.422203|n/a|
| Local1-SSIM | image |0.405702|0.558140|1.000000|0.716418|0.523079|n/a|

- Thresholds difference pixel0.257816046476/image0.483886012720,
  SSIMpixel0.957553517695/image1.139503239352 fitted on40calibration only,
  frozen on43holdout. Macro lesion-slice Dice difference0.234445353,
  SSIM0.207376961. Image thresholds predict all holdout images positive.
- Holdout PixelAP baseline30/control/A/B: difference
  0.101066/0.104391/0.104645/0.104464;SSIM
  0.102402/0.104004/0.104247/0.104084. Adding10epochs without encoder already
  produces nearly the same AP increase. A-minus-control AP only0.000253351
  difference/0.000243256SSIM;B-minus-control0.000072805/0.000079918.
  No clear incremental conditioning/pretraining benefit demonstrated. Small
  FPR/quality tradeoffs remain; one seed/inspected validation not significance.
- All83 descriptive control: differenceROC0.886424,Precision0.123099,
  Recall0.406822,F10.189007,AP0.103114,FPR0.061640;
  SSIMROC0.896166,Precision0.111455,Recall0.350228,F10.169098,
  AP0.102278,FPR0.063673. These thresholds are all-validation optimized and
  descriptive only. Brightness holdout AP0.389506/F10.268051 unchanged control.
- Normal foreground MAE[0,1]0.082240741/SSIM0.295440689 versus original
  0.082674084/0.292438059; A/B normal reconstruction quality somewhat better.
  Abnormal44 macro lesion/nonlesion difference0.232020910/0.147787429,
  localSSIMerror0.797625873/0.653556285.
- Evaluator loop1075.184828s,peak101438464B,total1196.861115s including preflight/
  quality/figures/artifacts before upload. Private reconstructions and examples
  uploaded; no medical images/checkpoints copied to Git.

### ddpm-uncond-val-smoke-t4-20261105T094736Z — best36 matched validation smoke — completed
- User requested validation after completed unconditional40epoch fine-tune.
  Normal-val-selected best36/FT6 SHA350791c48fd99babc38c329af5644a4c53a70f6628b6d751e85e3a6cab8cf0f7,
  selection MSE0.007353029596,source training uploaded375bc6ba5578f5c27953f3306cecdabb531f073b.
  No milestone/threshold selection by AP. Profile/Workspace phutungcogioihaitan,
  existing Secret name verified. OneT4,two-hour hard ceiling,no training/test use.
- [Modal app](https://modal.com/apps/phutungcogioihaitan/main/ap-7Hot55hkZx47D5XtcOaMrd)
  created. Bounded smoke replays original Gaussian30 EMA on1normal+1abnormal
  at t300,requires maxerror<=1e-6 versus registered original baseline and exact
  initial noisy arrays versus B,then finite3step best36 reconstruction on2images.
  No AP computed in smoke. Verify completed training/source/A/B/cache/index hashes.
- Dataset350c74f2313e4e73ca65e7ffc6e9dcf74040b3b7,seed17,code7134b105c0c4d4fb620715908f100103b1d0b0ab
  plus dirty snapshot in manifest.135local tests passed12.83s,targeted Ruff/format
  and whitespace passed. Exact original-chain equality,unconditional selected
  checkpoint,group thresholds and cache-prefix recovery tested before submission.
  Full83 validation only after smoke success; metrics pending.
- Verified completed/uploaded0ad3619e2aa19d0d04a37069e4c750f7d263f37f;
  smoke preflight passed,original Gaussian replay maxerrors[0,0],paired-noise
  checks passed,target36 finite2image/3steps.25.784308s evaluation,source snapshot
  68092ef7fe5e470d833a1ce0909c84011438e7b41414e91f1761f5dc4968c93a.
  This is a sampler/loader check,not an AP result; permits full83 validation.

### ddpm-uncond-control-t4-b8-e40-20261105T082502Z — unconditional EMA30 +10epoch control — completed
- User requested original Gaussian DDPM baseline fine-tune10epochs without
  encoder. After bounded smoke success, submitted independent full run to
  Workspace/profile phutungcogioihaitan,
  [Modal app](https://modal.com/apps/phutungcogioihaitan/main/ap-kpcgyltxZ55ANmBYWC1N2O).
- Config ddpm_unconditional_control_t4_epoch40.json,trainer
  dacn.training.ddpm_unconditional_control:train,oneT4,batch8,seed17,
  epoch31..40/9380new updates,all7500normal train,39normal-val selection.
  Source sameGaussian EMA30 as A/B with registered SHA; no encoder,condition
  projection or smoke/A/B weights. FreshAdamW LR1e-5,betas0.9/0.999,wd0,clip1,
  EMA0.9992,Gaussian epsilonMSE,linearT1000,unchanged U-Net architecture.
- Dataset350c74f2313e4e73ca65e7ffc6e9dcf74040b3b7,code7134b105c0c4d4fb620715908f100103b1d0b0ab
  plus frozen source03f0be1c2c2991db35952c492372b51e9cbad3afcc05307a3b60e80c0034ffd2.
  Four-hour hard timeout,private HF upload/shared checkpoint Volume. Save initial
  epoch30/0updates recovery,last each epoch,best among31..40 by normal EMA noise
  MSE fixed seed1000017,milestones31/35/40,initialization/history/config.
  Counter9380 is this fine-tune stage,not restored28140-step optimizer state.
- Training only;private2image/t100preview per original trainer contract,not
  full83 AP evaluation. No sweep/test access. Started08:26:49UTC;durable manifest
  running/training,TeslaT4,exact source hash,epoch30/0updates recovery checkpoint
  committed. Log epoch31/40 step50/938,finite running epsilonMSE0.008318 confirms
  training began. Full training metrics pending; client detached after this
  check, no continuous monitoring. Check on user request.
- This is a matched fine-tune control for A/B U-Net LR/update budget,not native
  full-state LR1e-4 continuation. Do not infer conditioning benefit until matched
  t300 grouped evaluation is actually run and uncertainty considered.
- Completion verified on user validation request: completed/uploaded at private
  revision375bc6ba5578f5c27953f3306cecdabb531f073b,2026-11-05T09:39:18UTC.
  All10fine-tune epochs31..40/9380updates,7500normal train/39normal-val.
  Best normal-val epoch36(FT6),epsilonMSE0.007353029596,training+normal-val
  4144.204522s,peak5465677824B. Best checkpoint SHA
  350791c48fd99babc38c329af5644a4c53a70f6628b6d751e85e3a6cab8cf0f7.
  Milestones31/35/40 and private2image/t100preview uploaded. EMA30/freshAdamW/
  encoder_added=false confirmed. This is training completion, not PixelAP evidence.

### ddpm-uncond-control-smoke-t4-20261105T082204Z — unconditional EMA30 fine-tune smoke — completed
- User-requested baseline fine-tune control. Profile/Workspace phutungcogioihaitan,
  existing Secret name checked. Initial sandbox submission could not connect
  before app creation; repeated with network permission.
  [Modal app](https://modal.com/apps/phutungcogioihaitan/main/ap-uW93yqQAyW93cmX0gdbFuY)
  initialized/functions created; training/checkpoint/private preview completed
  and manifest completed/uploaded verified after run.
- Config ddpm_unconditional_control_smoke.json,trainer
  dacn.training.ddpm_unconditional_control:train,T4,batch2,seed17,normal16train,
  atmost8val(subset normal for loss),one fine-tune epoch31,private2image/3step
  preview. Original Gaussian EMA30 source hash
  d3e06d2082e18234899b201718abd59078c8a800d66da3cc5d8b9e0d9ca9b946;
  freshAdamW LR1e-5,EMA0.9992,no encoder. Source manifest completed/uploaded
  at1ee5c4995623532e7ab51c043839555c1dd0b0db,source checkpoint SHA matched.
- Dataset350c74f2313e4e73ca65e7ffc6e9dcf74040b3b7,code7134b105c0c4d4fb620715908f100103b1d0b0ab
  plus dirty source snapshot in manifest.133tests passed12.79s,targeted
  Ruff/format and whitespace checks passed. No AP evaluation or held-out test.
  This checks initialization/training/checkpoint/reconstruction pipeline; full
  control must initialize independently from EMA30 after smoke succeeds.
- Completed1fine-tune epoch31/15shard-aware updates,16normal train,
  4normal-val out of8selected validation,2image/3step private preview artifact
  present. Normal epsilonMSE0.000374397350,4.466861s train+val,
  peak1466632704B. Not comparable to full-data validation MSE/AP.
- Private upload73552266d80393164a2a1dcf3fb3eacbe1d15a57;
  source snapshot03f0be1c2c2991db35952c492372b51e9cbad3afcc05307a3b60e80c0034ffd2.
  Manifest confirms sourceEMA30,freshoptimizer,encoder_added=false. Full133tests
  passed before launch. Smoke passed prerequisite for full10epoch control.

### taitrang2005 personal-workspace T4 startup — 2026-11-05 — completed
- User clarified the requested account is taitrang2005, not the old shared
  Workspace. Earlier modal-new probes used phutungcogioihaitan and do not prove
  startup in this new Workspace. Browser authentication created/activated CLI
  profile taitrang2005; Modal verified its Workspace as taihongtran2005.
- Explicit --profile taitrang2005,standalone modal_gpu_smoke.py::check_gpu,
  no Secrets/Volumes/Hugging Face/data/checkpoints. T4,60s function timeout.
  [New Workspace app](https://modal.com/apps/taihongtran2005/main/ap-HEmpKNsuPAviG0hKdP2KzN)
  completed/stopped,statuspassed,TeslaT4/15360MiB,driver610.57.04.
  Startup/driver check only; no training, tensor benchmark or credit estimate.

### Minimal T4 startup without Secret — 2026-11-05 — completed
- User explicitly requested another small test using profile modal-new without
  Hugging Face keys/Secret. Entrypoint src/dacn/cloud/modal_gpu_smoke.py::check_gpu
  uses standalone App, base Debian Python3.11, T4,60s function timeout, no Secrets,
  no Volumes, no Hugging Face calls, model or data access.
- [Modal app](https://modal.com/apps/phutungcogioihaitan/main/ap-t4QbkUQSa8RZ6Sc97R7us9)
  completed/stopped. Remote status=passed,nvidia-smi TeslaT4/15360MiB,
  driver610.57.04. Profile modal-new resolved to Workspace phutungcogioihaitan.
  This checks allocation/driver startup, not tensor computation or training.

### Modal profile modal-new access check — 2026-11-05 — completed
- User requested startup verification after changing account. Explicit CLI
  profile modal-new, Workspace phutungcogioihaitan; default local profile was
  still phutungcogioihaitan and was not changed. Secret names and both project
  Volumes visible to modal-new; no credentials printed or shared Secret changed.
- [Modal app](https://modal.com/apps/phutungcogioihaitan/main/ap-jGYAyb2PvLTaOey1nSFXb5),
  entrypoint src/dacn/cloud/modal_access_check.py::check_remote, short T4
  allocation with120s function timeout. Actual remote result gpu_startup=passed,
  TeslaT4/15360MiB. App completed and stopped after probe.
- Cached pinned validation index and B best checkpoint exist on mounted shared
  Volumes. Private HF dataset/model metadata both returnedHTTP200/private=true
  through existing shared Secret. This verifies read access, not upload/write
  permission or remaining credit balance. No data/checkpoint download, training,
  reconstruction, anomaly evaluation or test access; no AP/quality metrics.

### B-condition-full83-t4-20261105T072451Z — B condition diagnostic full83 — completed
- User-requested paired inference after B condition smoke passed. App
  [ap-nxZTHthEc7ebwJkuTlTOi0](https://modal.com/apps/phutungcogioihaitan/main/ap-nxZTHthEc7ebwJkuTlTOi0),
  oneTeslaT4,batch1,t300,2hour hard timeout, no training. Manifest started
  2026-11-05T07:25:23UTC,running/bounded_preflight. Source snapshot
  b5e4fccffe407d80914106884b49a41567a1fab134bfa025a89f77799d1f41c0,
  code7134b105c0c4d4fb620715908f100103b1d0b0ab+dirty; same pinned dataset/index,
  EMA6 checkpoint and original B reference hashes as smoke.
- Three arms: cached correct-image condition, zero injected condition including
  bias, and one fixed normal-training donor per evaluated image. Two new83-slice
  ancestral runs, same UUID-derived base17 noise/reverse RNG; donor mapping
  base77 independent of annotation/metric. Exact cached-image/noisy-array pairing
  required, full reverse replay preflight repeated before new runs.
- Reuse40calibration/43group holdout,calibration-only thresholds independently
  fitted per arm. AP,ROC-AUC,Precision/Recall/F1,normal pixel FPR,image metrics,
  240crop reconstruction changes versus self,latency/VRAM and private figures.
  Previous validation inspected,1seed,not final test. No held-out test access.
- Durable prefix caches every5images,source-checked resume; outputs under
  private evaluations/B-condition-full83-t4-20261105T072451Z. Self reference
  metrics reused from immutable B results17cea7ce87afc15332835ffdafa514b2758883d92510339f86bb45b521ac247d.
  Full worker passed preflight and began zero-condition reconstruction:
  completed1/83 at8.799s. Client detached after start; completion verified on
  subsequent user request. Both arms83/83,preflight errors[0,0],no failure.
- Completed2026-11-05T07:54:25UTC,manifest completed/uploaded at private revision
  [c22ee10da5f0193e99d3f3c097ab37e120cb7737](https://huggingface.co/dacn-bmad-team/dacn-runs/tree/c22ee10da5f0193e99d3f3c097ab37e120cb7737/evaluations/B-condition-full83-t4-20261105T072451Z).
  Aggregate results SHA17300018158205e715e2c186c6fa1ea713fe13abeebbdaec21ff2300f08df0b0
  matches manifest artifact hash; sample83,sealed-test flag and checkpoint SHA
  verified locally. No new GPU work initiated by result check.

| Condition,43group holdout | Pixel score | ROC-AUC | Precision | Recall | F1/microDice | AP | Normal pixel FPR |
|---|---|---:|---:|---:|---:|---:|---:|
| Correct image | Absolute difference |0.878343|0.126436|0.310393|0.179681|0.104464|0.055678|
| Zero injection | Absolute difference |0.879854|0.125348|0.312906|0.178993|0.104579|0.055271|
| Normal train donor | Absolute difference |0.880107|0.126232|0.319065|0.180896|0.105132|0.056058|
| Correct image | Local1-SSIM |0.887708|0.115097|0.251140|0.157851|0.104084|0.053144|
| Zero injection | Local1-SSIM |0.890025|0.114561|0.253636|0.157832|0.104416|0.052590|
| Normal train donor | Local1-SSIM |0.889813|0.115234|0.254911|0.158719|0.104699|0.052691|

- Aggregation globalmicro240crop; independent40calibration maxF1 thresholds
  frozen on43group holdout(25groups each); one previously inspected validation
  seed. Similar AP does not establish statistical equivalence or method failure.
  Correct-image condition has no demonstrated advantage under these interventions.
- Holdout image-level results(self/zero/donor): difference ROC-AUC
  0.228070/0.254386/0.243421,AP0.417369/0.426584/0.421661;
  SSIM ROC-AUC0.414474/0.405702/0.405702,AP0.528226/0.523079/0.523708.
  All image Precision0.558140,Recall1,F10.716418 at independently calibrated
  thresholds; all holdout images positive, not effective class separation.
- All83 descriptive PixelAP(self/zero/donor): difference
  0.104162/0.102956/0.103748;SSIM0.103002/0.102292/0.102624. Small gains for
  correct condition in all83 differ from holdout; no model selection from these.
- Reconstruction mean absolute change versus self on240crop,[0,1] scale:
  zero0.001952624(normal0.001514973,abnormal0.002340543),
  donor0.001710228(normal0.001585713,abnormal0.001820594).
  Max pixel changes0.535652/0.506981 respectively; crop means include background
  and do not exclude large localized effects. Condition affects reconstruction;
  this is not proof that encoder/injection is entirely ignored.
- Timings:zero813.079245s,peak146502656B;donor818.888632s,peak146765312B;
  total1739.938872s incl preflight/CPU scoring/artifacts before upload. Correct
  arm reused immutable B cache. Private paired arrays and figures uploaded.

### B-condition-smoke-t4-20261105T072319Z — B condition diagnostic smoke — completed
- User-authorized inference-only diagnostic, oneT4,batch1. Shared Workspace and
  Secret name checked. App ap-fuzCaXC71cYCTeIlbyiGtv completed; no training.
  B normal-val-selected EMA6 SHA0c4f21705830f5dd6054174fc6b2c1591263779413737ce87b4407e3ca8b01b7;
  immutable dataset350c74f2313e4e73ca65e7ffc6e9dcf74040b3b7 and index verified.
- Replay1normal+1abnormal image at300steps/base17 matched reference exactly:
  maximum absolute reconstruction errors[0,0],initial noisy tensors identical.
  Three-step zero-injection/donor controls finite. No AP evaluated on smoke.
  Zero bypasses projection plus bias; donor from normal training changes encoder
  input only. Cache/results/manifest checksums validated before replay.
- Manifest completed/uploaded,private HF revision
  92df7e4ef60628e019d7a2b85146b393f2d28450 under evaluations/B-condition-smoke-t4-20261105T072319Z.
  Code7134b105c0c4d4fb620715908f100103b1d0b0ab plus source snapshot
  b5e4fccffe407d80914106884b49a41567a1fab134bfa025a89f77799d1f41c0.
  129local tests, targeted Ruff/format and whitespace checks passed before smoke.
  No held-out test access. Passed prerequisite for paired full83 experiment.

### feature-B-pretrain5-joint10-t4-20261105T053132Z — B full chain — completed
- After successful bounded B smoke, submitted fresh full run on oneT4,
  [Modal app](https://modal.com/apps/phutungcogioihaitan/main/ap-UKIL1QqcsgKzhIzGNO0m50).
  Encoder pretrain5epochs/4690planned updates, then joint10epochs/9380planned
  updates with automatic83-slice ancestral t300 grouped evaluation. Four-hour
  hard timeout. No smoke or A fine-tune weights transferred to the full run.
- Configs encoder_pretrain_B_t4_epoch5.json/feature_condition_B_t4_epoch10.json;
  exact method/protocol in FEATURE_CONDITION_B.md. Dataset revision
  350c74f2313e4e73ca65e7ffc6e9dcf74040b3b7,seed17,batch8,7500normal train,
  39normal-val for checkpoint selection,code7134b105c0c4d4fb620715908f100103b1d0b0ab
  plus dirty source snapshot bcc65e702e23a2ca97179cf066c7d38c9a40d1bd1c3f3c58b8f726c9d67aadf2.
- Separate manifests/artifacts under private HF runs/RUN-enc and runs/RUN;
  auto evaluation under evaluations/RUN-val. Encoder best selected by fixed-mask
  normal reconstruction MSE; fine-tune best EMA by fixed-noise normal epsilon MSE.
  Encoder checkpoint checksum bound automatically and validated before loading.
- Validation will reuse A's40calibration/43group-holdout partition/seeds/scorers,
  calibration-only thresholds and immutable baseline/A artifacts. Reports pixel
  and image ROC-AUC/Precision/Recall/F1/AP, normal quality, runtime/VRAM and private
  examples. One exploratory seed, extra pretrain compute separate, test sealed.
  Durable manifest started2026-11-05T05:31:54UTC,statusrunning,stagetraining,
  TeslaT4,source hash verified. Log encoder epoch1/5 step150/938 finite running
  maskedMSE0.084553 confirms GPU training began; this is not an AP result.
  Detached after this start check. Completion verified on user request below.
- Encoder completed5epochs/4690updates,best5,normal maskedMSE0.055597168799,
  training+normal-val313.874214s,peak910336512B. Private uploaded revision
  626ec3ec07acdfc38274300c374045d09212f1c7,runs/RUN-enc.
- Joint completed10epochs/9380updates,best6,normal epsilonMSE0.007341488981,
  training+normal-val4436.784669s,peak5883660288B. Private uploaded revision
  352d96a26a3b9b056274bbeb7bc5864eb8e25562,runs/RUN. Best EMA SHA
  0c4f21705830f5dd6054174fc6b2c1591263779413737ce87b4407e3ca8b01b7.
- Validation completed83/83 at2026-11-05T07:10:59UTC; completed/uploaded manifest,
  [private evaluation artifacts](https://huggingface.co/dacn-bmad-team/dacn-runs/tree/348c47aa2528fcd48b1dc62f378e7078c7592571/evaluations/feature-B-pretrain5-joint10-t4-20261105T053132Z-val).
  Checkpointbest6/EMA,seed17,t300,index57ad2e31df3ed4e625da00efaecd6fddc1e2a132329dfba6e60867caf1802db3;
  40calibration/43holdout,25groups each,thresholds fitted only on calibration.
  Pixel global micro240crop, image mean top576pixels;1seed,held_out_test_used=false.
  Results JSON SHA17cea7ce87afc15332835ffdafa514b2758883d92510339f86bb45b521ac247d
  matches manifest artifact hash; checkpoint SHA/sample count verified locally.

| Score,43 grouped-holdout slices | Level | ROC-AUC | Precision | Recall | F1/microDice | AP | Normal pixel FPR |
|---|---|---:|---:|---:|---:|---:|---:|
| Absolute difference | pixel |0.878343|0.126436|0.310393|0.179681|0.104464|0.055678|
| Local1-SSIM | pixel |0.887708|0.115097|0.251140|0.157851|0.104084|0.053144|
| Absolute difference | image |0.228070|0.558140|1.000000|0.716418|0.417369|n/a|
| Local1-SSIM | image |0.414474|0.558140|1.000000|0.716418|0.528226|n/a|

- Holdout PixelAP baseline/A/B: difference0.101066/0.104645/0.104464;
  SSIM0.102402/0.104247/0.104084. B is marginally lower than A for both scores;
  no clear pretraining benefit with this pilot. PixelF1 A->B difference
  0.180757->0.179681,SSIM0.158039->0.157851. Normal FPR slightly lower than A.
  Brightness holdout pixelAP0.389506,F10.268051 remains substantially stronger
  at different normal FPR0.134935. Image classifier still predicts all holdout
  slices positive at calibration-selected threshold; F1 is not good separation.
- All83 descriptive metrics, independently validation-optimized thresholds:
  difference AP0.104162,F10.191531,FPR0.063298;SSIM AP0.103002,F10.169885,
  FPR0.061234. These tiny descriptive AP gains over A do not reverse the grouped
  holdout conclusion. No significance claim from one seed/inspected validation.
- Normal39 foreground(input>-0.95 heuristic) quality B:MAE[0,1]0.081059648,
  SSIM0.302334383 vs A0.081020562/0.302187646;essentially unchanged. Abnormal44
  macro SSIM error lesion/nonlesion0.791558521/0.640803810;absolute error
  0.230870375/0.144413748. Selected normal epsilonMSE also nearly identical A/B.
- Inference loop806.260116s/83images,eval total885.090154s,peak147551744B.
  Cached private reconstruction arrays/figures remain private. Additional encoder
  pretraining compute313.87s is separate from matched9380joint updates.

### feature-B-smoke-t4-20261104T202500Z — encoder pretrain + joint smoke — completed
- User-requested B, shared Workspace/Secret name verified. First local request
  failed to connect before app creation; repeated with network permission.
  [Modal app](https://modal.com/apps/phutungcogioihaitan/main/ap-y06ZcVgJDcUqeBm5WhIf0P)
  initialized and function objects created. Remote smoke=passed, finite3-step
  reconstruction and both manifests completed/uploaded verified.
- One T4, bounded1pretrain+1joint epoch,16normal train/4normal val,batch2,
  then3conditional reverse steps. Configs encoder_pretrain_B_smoke.json and
  feature_condition_B_smoke.json. Separate encoder run suffix-enc; fine-tune
  independently loads original Gaussian EMA30 and only the produced encoder.
- Dataset350c74f2313e4e73ca65e7ffc6e9dcf74040b3b7,code7134b105c0c4d4fb620715908f100103b1d0b0ab
  plus exact dirty source snapshot in manifests.124 local tests passed before
  submission; targeted Ruff/format and whitespace checks passed. Private HF
  artifacts will be under runs/feature-B-smoke-t4-20261104T202500Z[-enc].
- Encoder phase:15shard-aware updates,3.401343s,peak346722816B,normal-val masked
  MSE0.120478242636,best1;private revision60fd76a28f3a7bfe3b6feb336ef23339fedbd400.
  Joint phase:15updates,3.364274s,peak1709877760B,normal-val epsilonMSE
  0.000329598508,best1;private revision0f4de9f4bdc5549859f0fd03db1fb7fddfe34032.
  Both source snapshots bcc65e702e23a2ca97179cf066c7d38c9a40d1bd1c3f3c58b8f726c9d67aadf2.
  These different objectives/subsets are not comparable to full-data loss/AP.
- This is a pipeline check, not a Pixel AP evaluation. Successful smoke permits
  full B above. No held-out test read or external pretrained weights.

### feature-A-joint-t4-b8-e10-20261104T174552Z — ResNet condition A10 + validation — completed
- User-authorized experiment A after reading docs/newmethod.md; random grayscale
  ResNet-18(standard[2,2,2,2],512 pooled features),Linear512->128+SiLU,
  zero-initialized128->actual U-Net timestep embedding projection. Add condition
  embedding to timestep embedding through all original residual blocks. Original
 1-channel U-Net initialized strictly from Gaussian EMA epoch30, source hash
  d3e06d2082e18234899b201718abd59078c8a800d66da3cc5d8b9e0d9ca9b946.
  Fresh full-run initialization, not continued from smoke weights; no encoder
  pretraining/external weights/experiment B.
- [Modal app](https://modal.com/apps/phutungcogioihaitan/main/ap-nhqC8aXEv1sdC8orCC0nbi).
  Config configs/feature_condition_A_t4_epoch10.json,T4,batch8,seed17,10 fine-tune
  epochs/9,380 planned optimizer updates on7500 normal train slices. Gaussian
  linearT1000,epsilon-MSE,AdamW betas0.9/0.999,weight_decay0,gradientclip1;
  U-NetLR1e-5,encoder/projectionLR1e-4,EMA0.9992. All parameters trainable;
  EMA copies online BatchNorm buffers.39 normal-val fixed seed1,000,017 selects
  best. Initial zero-injection model output must exactly equal baseline.
- Epoch0 full-state recovery,last each epoch,best,and milestones1/5/10 use
  feature_condition_*.pt namespace. Wrapper detects preemption/repeated request,
  validates exact source/config and resumes full state rather than fresh-run
  collision. Preserves original baseline checkpoints. Dataset revision
 350c74f2313e4e73ca65e7ffc6e9dcf74040b3b7,code7134b105c0c4d4fb620715908f100103b1d0b0ab
  plus dirty package snapshot; source recorded in durable manifest.
- After training, automatically evaluate normal-val-selected best EMA on83
  validation slices at t300,ancestral,batch1,seed17. Encode original image once,
  same condition every reverse step; no tumor-mask input. Existing40 calibration
  /43 patient-group holdout,calibration-only thresholds,pixel difference/localSSIM,
  brightness control and separately descriptive all83 metrics. Reuse original
  Gaussian30 baseline cache after checksum and exact-array alignment checks.
  Also report normal tissue MAE/SSIM,lesion/nonlesion score means,private figures.
  Save private cache every5 images and resume prefixes after evaluation interruption.
- Smoke2 completed before full submission;119 local tests passed. Full training
  and AP results pending. Budget adds10epochs beyond baseline30, so a gain would
  not isolate conditioning from extra training without an unconditional matched
  fine-tune control. Whole-image condition may retain tumor information. Test
  sealed; no further training, B, external weights or sweeps. Run detached.
- Durable start2026-11-04 17:46:35UTC (2026-11-05 00:46:35local),TeslaT4;
  source snapshot SHA25672e43a42c623d123c29ec5ebb707de3de3b22c63a859148e662a7e4ceaef9ee1,
  identical package to successful smoke2. Epoch0 full-state checkpoint committed.
  Initial log confirms epoch1/10,step25/938,samples200/7500,finite epsilon
  MSE0.008372 (interim running mean only). No final normal-validation or AP
  metrics yet. Stream detached; remote chain continues training then evaluation.
- Completion check:10epochs/9,380updates completed2026-11-04 19:04:01UTC,
  uploaded19:04:20UTC at private training revision
  [10833f849f9a93bbfd5364336a2b920350b278b1](https://huggingface.co/dacn-bmad-team/dacn-runs/tree/10833f849f9a93bbfd5364336a2b920350b278b1/runs/feature-A-joint-t4-b8-e10-20261104T174552Z).
  Best normal-validation checkpoint is fine-tune epoch6, epsilon MSE0.007343506560.
  Normal-only selection, not AP selection. Milestones1/5/10 and best/last exist.
  Training/validation4449.683s (~74.16min), peak allocatedVRAM5,884,708,864B.
  Best checkpoint SHA2560a5939b2b9e1df31f635c411c67fb01ef46df42ed5dad9599992948d6b0ddd29;
  epoch10 milestonec6445e97e6fc4c44b9e7fc1349bb2f15b304cb08e43cdf4121e703cc511f626b.
- Automatic evaluation completed all83 slices at t300 using best EMA(epoch6),
  completed2026-11-04 19:20:08UTC, uploaded19:20:11UTC at private revision
  [56f13bc33a470858475058cb961b1e904f680946](https://huggingface.co/dacn-bmad-team/dacn-runs/tree/56f13bc33a470858475058cb961b1e904f680946/evaluations/feature-A-joint-t4-b8-e10-20261104T174552Z-val).
  App stopped2026-11-05 02:21:28local,zero tasks. Baseline cache SHA/content
  alignment verified;brightness control reproduces prior metrics.

  | Holdout43 / pixel score | AP | ROC-AUC | Precision | Recall | F1 / micro Dice | Normal pixel FPR |
  | --- | --- | --- | --- | --- | --- | --- |
  | Original Gaussian30 / difference | 0.101066 | 0.876854 | 0.119934 | 0.314702 | 0.173678 | 0.058079 |
  | Experiment A / difference | 0.104645 | 0.878721 | 0.126465 | 0.316731 | 0.180757 | 0.056641 |
  | Original Gaussian30 / SSIM | 0.102402 | 0.888870 | 0.110476 | 0.257712 | 0.154654 | 0.055234 |
  | Experiment A / SSIM | 0.104247 | 0.888060 | 0.115090 | 0.252128 | 0.158039 | 0.053224 |
  | Brightness control | 0.389506 | 0.950197 | 0.161469 | 0.788577 | 0.268051 | 0.134935 |

  Global micro over240crop;40calibration/43holdout,25 disjoint patient groups
  each. A thresholds difference0.25466160476207733 /SSIM0.9594686479174731,
  fit only on calibration. Pixel SSIM AP improves~1.80%relative on holdout;
  difference AP~3.54%. Gains small, no sampling-seed repeats or intervals.
  SSIM recall slightly declines; lower FPR is threshold-dependent.

  | All83 descriptive pixel score | AP | F1 / micro Dice | Normal pixel FPR |
  | --- | --- | --- | --- |
  | Original Gaussian30 / difference | 0.099834 | 0.182870 | 0.063527 |
  | Experiment A / difference | 0.104001 | 0.191032 | 0.059151 |
  | Original Gaussian30 / SSIM | 0.100741 | 0.167483 | 0.071126 |
  | Experiment A / SSIM | 0.102805 | 0.169561 | 0.060523 |

  Separate in-sample thresholds for all83; not interchangeable with group holdout.
  A all83 thresholds difference0.23569299280643463 /SSIM0.9217134888540419.

  | A holdout43 image score(top576 mean) | AP | ROC-AUC | Precision | Recall | F1 |
  | --- | --- | --- | --- | --- | --- |
  | Difference | 0.415001 | 0.221491 | 0.558140 | 1.000000 | 0.716418 |
  | Local SSIM | 0.527520 | 0.412281 | 0.558140 | 1.000000 | 0.716418 |

  Image thresholds0.4787318131986669(difference),1.139835233359465(SSIM),
  calibration-only;near-all-positive image predictions remain, no useful image
  discrimination established by high F1 alone.
- Normal39-slice macro foreground quality [0,1]MAE baseline0.082674->A0.081021;
  foregroundSSIM0.292438->0.302188. Foreground=input>-0.95,not an anatomical mask.
  Abnormal44-slice macro SSIM-error means lesion/nonlesion baseline0.798852/0.662143
  vs A0.791949/0.642771; absolute-error means baseline0.229169/0.150152 versus
  A0.230798/0.144902. Small normal-preservation/score gains, not complete tumor removal.
- Evaluation reconstruction/score/cache loop868.109s, full evaluation945.740s
  (includes baseline scoring/figures),peak allocatedVRAM147,551,744B. Produced
  private6-panel figures and cached images/noisy/reconstructions/masks. Results
  SHA256a8e57ed824c9e6a00404d82296f15fb208eabb9d8d911c06fcd983dfb8b5a962;
  NPZ329babd5232cb57b7c07c47ad0ed3f87ca013414d55017181c2e916f0f1d9c9f.
  Modest positive exploratory signal; extra fine-tune compute confounds attribution
  to conditioning alone. B not run; no new training or evaluation started by this check.

### feature-A-smoke2-t4-20261104T174325Z — ResNet condition smoke — completed
- Fresh retry after correcting image build order; [Modal app](https://modal.com/apps/phutungcogioihaitan/main/ap-1q7CzTsHLovLVuoq3X5tUi).
  Trainer dacn.training.feature_condition:train; configfeature_condition_A_smoke.json,
  random standard grayscale ResNet-18,zero condition injection,original Gaussian
  epoch30 EMA,all model parts trainable.16 normal train/4 normal val,1epoch,batch2,
  LR U-Net1e-5 /encoder+projections1e-4,EMA0.9992,seed17. Smoke only, no AP claim.
  Dataset350c74f2313e4e73ca65e7ffc6e9dcf74040b3b7; code7134b105c0c4d4fb620715908f100103b1d0b0ab
  plus dirty source snapshot. Image built successfully, dataset setup started;
  full10epoch run awaits smoke success. Source model SHA256d3e06d2082e18234899b201718abd59078c8a800d66da3cc5d8b9e0d9ca9b946.
  Checkpoint selection normal-val MSE; subsequent3-step normal-only reconstruction
  must be finite. No external weights, tumor training, test access or experiment B.
- Actual smoke start2026-11-04 17:43:56UTC, training completed17:45:02,
  uploaded17:45:18 at private revision
  [2e89ab1e4ad5469656bb53065fdccbfb5cd49b59](https://huggingface.co/dacn-bmad-team/dacn-runs/tree/2e89ab1e4ad5469656bb53065fdccbfb5cd49b59/runs/feature-A-smoke2-t4-20261104T174325Z).
 15 optimizer updates (subset spans shards; batch2 ceiling, mostly singleton
  shards),16 train/4normal val. Epsilon validation MSE0.000329598362,
  training+validation3.860s,peak allocatedVRAM1,710,402,048B. This is a tiny
  fixed subset and not comparable to full39-normal validation or anomaly AP.
  Remote function then returned smoke=passed, finite3-step reconstruction.
  Zero-injection real-backbone equality guard passed. Source snapshot SHA256
  72e43a42c623d123c29ec5ebb707de3de3b22c63a859148e662a7e4ceaef9ee1.

### feature-A-smoke-t4-20261104T174219Z — ResNet condition smoke — build failed
- User-authorized bounded smoke before experiment A10epoch training. Config
  configs/feature_condition_A_smoke.json,16 normal train/4 normal val,batch2,
  1epoch,Gaussian epoch30 EMA source. Random ResNet-18/no pretraining,
  zero-injection, joint training planned. No external weights or test data.
- [Modal app](https://modal.com/apps/phutungcogioihaitan/main/ap-dLWRfA9tkNS2g3UgBttvHe).
  Failed before remote training/image execution: image.pip_install was appended
  after add_local_python_source, which Modal disallows. No training checkpoint
  or metrics produced. Correct build order via separate base_image before
  local-source mount, then use a fresh smoke run ID. Full10epoch run not started.

### ddpm-val-simplex-e20-t300-20261104T105703Z — Simplex20 full validation — completed
- User requests validation after Simplex20 training completed/uploaded. Evaluate
  EMA ddpm_epoch_020.pt from ddpm-simplex-t4-b8-e20-cont-20261104T070341Z,
  SHA2561b9f50988f8a57634ddc9b3b9de1db93c4e324c5de7d42132db50525e4a36d07.
  [Modal app](https://modal.com/apps/phutungcogioihaitan/main/ap-OiSBEGf23BvlcdNTHlmIgh).
- Same83 validation slices (39 normal/44 abnormal), dataset revision
  350c74f2313e4e73ca65e7ffc6e9dcf74040b3b7,T4,ancestral batch1,t300,sampling
  seed17. Simplex forward and Gaussian fixed-large reverse as registered
  epoch10 adaptation; no changed noise settings or training. Code
  00d199fbe726922caaf6696b02addffe380f0d93, dirty source snapshot in manifest.
- Full240crop absolute difference/localSSIM scores; all83 validation max-F1
  thresholds, image score top576pixels. Compare same-protocol epoch10 AP/F1
  and normal pixel FPR. Save private full reconstruction cache and aggregate
  metrics, upload privately on completion; possible later group/fusion analysis
  requires fixed calibration protocol. Test sealed, no extra timestep sweep.
- Durable start2026-11-04 10:57:22UTC, stagevalidation_evaluation10:57:49UTC,
  TeslaT4. Checkpoint20 checksum verified, full validation cap=null,
  reconstruction cache enabled, testunused. Source snapshot SHA256
  a6a20656efda0b1094b27442f7429996ed3915a3804e33797404c835718b5651.
  Results pending. Estimated12–15min from prior t300 validation,
  not measured final latency. No continuous polling.
- Completed2026-11-04 11:10:39UTC, upload confirmed11:10:42UTC at private
  revision[1788b1232aa19b3c9ae96aa95a74e2975f4b592a](https://huggingface.co/dacn-bmad-team/dacn-runs/tree/1788b1232aa19b3c9ae96aa95a74e2975f4b592a/evaluations/ddpm-val-simplex-e20-t300-20261104T105703Z).
  All83 validation slices, full240crop global micro; independently fit max-F1
  thresholds on all83, descriptive validation metrics, not grouped holdout.

  | Pixel score | AP | ROC-AUC | Precision | Recall | F1 / micro Dice | Normal pixel FPR |
  | --- | --- | --- | --- | --- | --- | --- |
  | Local SSIM | 0.074791 | 0.743631 | 0.096629 | 0.599044 | 0.166414 | 0.131908 |
  | Absolute difference | 0.034863 | 0.691295 | 0.041214 | 0.336245 | 0.073428 | 0.167382 |

  Thresholds:SSIM0.9991420610921259 /difference0.939251720905304;
  macro lesion-slice Dice0.278893/0.121554 respectively.

  | Image score(top576 mean) | AP | ROC-AUC | Precision | Recall | F1 |
  | --- | --- | --- | --- | --- | --- |
  | Local SSIM | 0.385525 | 0.219697 | 0.530120 | 1.000000 | 0.692913 |
  | Absolute difference | 0.455751 | 0.363636 | 0.530120 | 1.000000 | 0.692913 |

  Both image thresholds predict all images positive; image F1 is not evidence
  of useful discrimination. Image thresholds1.0684869438444222(SSIM),
  1.8970544491377141(difference).
- Same-protocol epoch10->20: SSIM AP0.073910->0.074791 (~1.19%relative),
  F1 0.165806->0.166414,FPR0.133712->0.131908; difference AP0.032447->0.034863,
  F1 0.068113->0.073428,FPR0.189460->0.167382. Small anomaly-score gains despite
  ~14.33%normal-validation noise-MSE reduction. Gaussian epoch30 ancestral t300
  all83 SSIM AP0.100741 remains higher; budgets differ, not causal noise proof.
  One train/sampling seed; no intervals, testunused. No fusion update yet.
- Evaluation763.866s (~9.203s/image), cache compression1.667s, peak allocated
  VRAM100,913,664B. Results SHA256
  89b85f6008c79999e40f31906b53386ae4573c8dafe5ce1fb16571d4a371934f;
  reconstruction cache4dd1b3c5eac41f43638a2ccea5d97bdb53074f30ce3d21c8e84ef593fe8a28c0.

### ddpm-simplex-t4-b8-e20-cont-20261104T070341Z — Simplex continuation to20 — completed
- User requested continuing Simplex to20 total epochs, T4,batch8,seed17,
  LR1e-4,EMA0.9992. Config configs/ddpm_partial_simplex_t4_batch8_epoch20.json;
  only epochs/checkpoint_epochs differ from parent. Train11..20,938steps/epoch,
  9,380 additional /18,760 planned cumulative optimizer steps. Same7500 normal
  train slices,39 normal val slices; normal-val epsilon MSE selects best.
- [Modal app](https://modal.com/apps/phutungcogioihaitan/main/ap-qJD8Ddbo1XCuLciFJ6QJqS).
  Parent ddpm-simplex-t4-b8-e10-20261102T074257Z completed/uploaded at private
  revisione7b6ce4f167de1510b3111b6de9b10ab7d77492b. Continue from ddpm_last.pt,
  source hash e70883c98b3460484235b682d098dc8c942002ec7dd07753e4e03b875a4c0c4a;
  preserve model/EMA/optimizer/sampler/RNG/history. Parent best preserved,
  old epoch10 milestone remains available, new milestones15/20 saved separately.
- Dataset350c74f2313e4e73ca65e7ffc6e9dcf74040b3b7; code00d199fbe726922caaf6696b02addffe380f0d93
  plus dirty source snapshot to be pinned in manifest. Same noise settings:
  Simplex6 octaves,persistence0.8,frequency64,linearT1000. Private best/last and
  final milestone artifacts auto-upload on completion. Additional~75min estimate
  from parent timings; final metrics pending. No automatic AP inference or
  timestep sweep, no test access, no continuous monitoring.
- First local submission could not connect before app initialization; rerun
  with permitted network access initialized the app above. No duplicate GPU job
  or training checkpoint from the failed client connection.
- Durable start2026-11-04 07:05:50UTC, stage training07:06:19UTC on TeslaT4.
  Manifest verifies continuation parent epoch10, checkpoint/source-best hashes,
  config epochs20 and unchanged settings. Source snapshot SHA256
  a6a20656efda0b1094b27442f7429996ed3915a3804e33797404c835718b5651.
  Initial training log confirms epoch11/20,step50/938 with finite running
  epsilon MSE0.007772 (interim, not final epoch metric). Client detached;
  remote training continues, no continuous polling.
- Interruption/recovery2026-11-04: old app log confirms container preemption at
  15:18:43 Asia/Ho_Chi_Minh(08:18:43UTC) during epoch19,step625/938. Modal
  restarted same fresh-run input, which raised output-already-exists at15:18:55.
  App stopped15:20:20 local; last durable manifest remained stale running.
  Latest committed state is epoch18 /16,884steps, normal-val MSE0.002592801301,
  last checkpoint SHA256b28a410812ec1ee7456a0fa57284d7ec6c0bf71612e029ebe4c3eb16dca4b81c;
  milestone15 and best18 also remain. Epoch19 partial work is not checkpointed.
- Verified current exact source snapshot still equals saved hasha6a20656...
  before recovery. Reissued same run/config/code with --resume (no continuation
  argument); new [recovery app](https://modal.com/apps/phutungcogioihaitan/main/ap-9Ort1vmLBZ9Ki6Rp9n5RDi).
  Restore original model/EMA/optimizer/sampler/RNG/history from epoch18 and run
  epochs19..20. No restart from scratch or changed hyperparameters. Resume
  submission initially had local network failure before app initialization;
  approved network retry initialized this recovery app. Final metrics pending.
- Recovery accepted as attempt2, resumed2026-11-04 10:07:57UTC; source/config
  hash checks passed. Log confirms epoch19/20,step50/938 with running MSE0.006495.
  First25/50 running losses0.006732/0.006495 match the interrupted epoch19 log,
  consistent with restored training RNG/state. Client detached; job continues
  on T4 toward20, no continuous polling or extra inference.
- Completion verified: attempt2 completed2026-11-04 10:25:08UTC (17:25:08local),
  artifacts uploaded10:25:24UTC at private revision
  [0846480f8551c7e4b41a534c148320d61c906496](https://huggingface.co/dacn-bmad-team/dacn-runs/tree/0846480f8551c7e4b41a534c148320d61c906496/runs/ddpm-simplex-t4-b8-e20-cont-20261104T070341Z).
  App stopped17:27:05local,zero tasks.20 total epochs /18,760 optimizer steps;
  best normal-validation epsilon MSE0.002546022252108042 at epoch20, versus
  epoch10 0.002971782593820722 (~14.33% reduction). No AP evaluation on epoch20
  performed; noise loss improvement does not establish anomaly-score improvement.
- Produced ddpm_epoch_020.pt SHA256
  1b9f50988f8a57634ddc9b3b9de1db93c4e324c5de7d42132db50525e4a36d07;
  bestcbee7439b0eb7e408d7a85c0005f0661f471788093ddb3e79898e0829c9c211f;
  lastb03f765a4caec8aaaa25a37efd40acfe23ecce5010ba32a835fda10fa377d6a2.
  History SHA256a09125fedf8d18ceb510dc50fb2e07b5d28c1d57c854b82f9fbdc73c4cf11da8.
  Trainer-reported cumulative completed-epoch train/val9370.595s across1..20,
  incremental completed epochs11..20 ~4905.158s (~81.75min), excluding lost
  partial epoch19 work/setup/checkpoint writes/idle time; peak allocatedVRAM5,532,909,056B.

### gaussian-simplex-fusion-20261104T065916Z — cached SSIM fusion — completed
- User-requested CPU-only fusion, no GPU/training. Same83 validation inputs,
  Gaussian EMA epoch30 ancestral t300 and Simplex EMA epoch10 ancestral t300.
  Caches SHA256Gaussian8278ad18543dd87f698c36b510564b9fcd4445d4a1b0ae5d7a859093e2a4e959,
  Simplex5bf4cfddf144ac2ce88e3da6417df5e86cea91bcff4d39f97738ae70ab2d0218;
  pinned index57ad2e31df3ed4e625da00efaecd6fddc1e2a132329dfba6e60867caf1802db3,
  dataset350c74f2313e4e73ca65e7ffc6e9dcf74040b3b7.
- Module dacn.evaluation.noise_fusion, local dirty source on code00d199fbe726922caaf6696b02addffe380f0d93.
  Normal-calibration-only midrank CDFs, Gaussian weights0.25/0.5/0.75;
  raw/CDF controls for each model and brightness control. Same40 calibration /
  43 group holdout; select candidate and best-fusion weight solely by calibration
  pixel AP, fit pixel/image max-F1 thresholds there, save frozen plan before
  holdout scoring. Raw Gaussian preferred on exact selection ties; brightness
  excluded from diffusion-method selection. No per-image or holdout fitting.
- Ignored local output under runs/visualizations/ddpm-e30-t300/
  gaussian-simplex-fusion-20261104T065916Z, no patient identifiers in JSON.
  All input/mask/label arrays and registered hashes checked before evaluation.
  Previously inspected validation, one seed; different source training budgets
  mean this tests ensemble utility, not isolated noise effects. Test sealed.
  Completed local CPU evaluation in22.001s. Results SHA256
  f0d755d8415184eb3e42c80dc050176dca1edcad34edbeb1773b65cb666bed67.
  CDF reference:20 normal calibration images, full240crop pixels. Both cache
  arrays align exactly, hashes verified. No new GPU job or Hub upload.

  | Candidate | Calibration pixel AP | Holdout pixel AP | ROC-AUC | Precision | Recall | F1 / micro Dice | Normal pixel FPR |
  | --- | --- | --- | --- | --- | --- | --- | --- |
  | Gaussian raw | 0.101263 | 0.102402 | 0.888870 | 0.110476 | 0.257712 | 0.154654 | 0.055234 |
  | Simplex raw | 0.060600 | 0.086425 | 0.762165 | 0.108985 | 0.391838 | 0.170537 | 0.091722 |
  | Gaussian CDF | 0.101266 | 0.102400 | 0.888870 | 0.110474 | 0.257712 | 0.154653 | 0.055234 |
  | Simplex CDF | 0.060601 | 0.086425 | 0.762165 | 0.108984 | 0.391838 | 0.170536 | 0.091722 |
  | Fusion25% Gaussian /75% Simplex | 0.073585 | 0.096144 | 0.809289 | 0.116773 | 0.519841 | 0.190706 | 0.109873 |
  | Fusion50% Gaussian /50% Simplex | 0.086679 | 0.107577 | 0.873195 | 0.121213 | 0.381603 | 0.183985 | 0.076231 |
  | Fusion75% Gaussian /25% Simplex | 0.099191 | 0.116771 | 0.902634 | 0.122427 | 0.264176 | 0.167315 | 0.051199 |
  | Brightness control | 0.146595 | 0.389506 | 0.950197 | 0.161469 | 0.788577 | 0.268051 | 0.134935 |

  Pixel scores global micro over240crop. The highest-calibration-AP fusion is
 75/25, frozen before holdout, but the globally selected diffusion candidate
  is Gaussian CDF (calAP0.101266), not fusion (calAP0.099191).
  Holdout75/25 fusion AP increases ~14.03% relative to Gaussian raw, F1 also
  increases and normal FPR drops slightly at calibration-fitted thresholds.
  This is an exploratory heldout observation, not proof the calibration-based
  selector beats baseline: all fusion calibration APs remain below Gaussian.
  Do not select another weight from holdout outcomes or claim robust improvement.

  | Candidate image score(top576 mean), holdout43 | AP | ROC-AUC | Precision | Recall | F1 |
  | --- | --- | --- | --- | --- | --- |
  | Gaussian raw | 0.505258 | 0.372807 | 0.558140 | 1.000000 | 0.716418 |
  | Simplex raw | 0.445382 | 0.219298 | 0.536585 | 0.916667 | 0.676923 |
  | Gaussian CDF | 0.495125 | 0.350877 | 0.558140 | 1.000000 | 0.716418 |
  | Simplex CDF | 0.456416 | 0.217105 | 0.525000 | 0.875000 | 0.656250 |
  | Fusion25/75 | 0.448683 | 0.192982 | 0.486486 | 0.750000 | 0.590164 |
  | Fusion50/50 | 0.478039 | 0.271930 | 0.512821 | 0.833333 | 0.634921 |
  | Fusion75/25 | 0.505516 | 0.350877 | 0.547619 | 0.958333 | 0.696970 |
  | Brightness | 0.770309 | 0.679825 | 0.571429 | 1.000000 | 0.727273 |

  Pixel/image thresholds for every row are saved in frozen_selection.json,
  independently fit by calibration max-F1 before any holdout evaluation.
 75/25 pixel threshold0.9222649739583334 and image0.9687924167962723.
  Baseline and brightness control reproduce prior results. No consistent image
  discrimination gain. One sampling seed and group split, previously inspected
  validation, no intervals. Different checkpoint budgets; not causal noise study.

### gaussian-e30-t300-samplers-val83-20261104T062253Z — one-step/DDIM10/ancestral — completed
- User-authorized fixed t300 sampler comparison on all83 validation images,
  original Gaussian epoch30 EMA, T4; no training or test access.
  [Modal app](https://modal.com/apps/phutungcogioihaitan/main/ap-EwxBoeql7RQDq7PNhLiiwF).
  Checkpoint SHA256d3e06d2082e18234899b201718abd59078c8a800d66da3cc5d8b9e0d9ca9b946,
  source ddpm-partial-t4-b8-e30-cont-20261102T034644Z/ddpm_epoch_030.pt.
- Dataset350c74f2313e4e73ca65e7ffc6e9dcf74040b3b7; code00d199fbe726922caaf6696b02addffe380f0d93
  plus dirty source snapshot in manifest. One-step1 call /DDIM10 calls; shared
  initial x300 from same base17 UUID seed as original ancestral. DDIM eta0,
  indices299,269,239,209,179,149,119,89,59,29 ->clean-1. Clipped x0/re-derived
  epsilon consistent with guided-diffusion convention; no new reverse noise.
- Ancestral cache from ddpm-val-t4b8-e30-t300-20261102T071516Z,
  SHA2568278ad18543dd87f698c36b510564b9fcd4445d4a1b0ae5d7a859093e2a4e959;
  verify arrays/order and regenerate first ancestral sample as a control.
  No full ancestral rerun. Pixel difference/localSSIM for all three methods,
  brightness control. Same40 calibration/43 group holdout; calibration-only
  threshold fits, separately descriptive all83 metrics. AP, normal FPR and
  pixel/image ROC-AUC,precision,recall,F1/Dice recorded when produced.
- New sampler reconstruction-only CUDA-synchronized timings exclude noise
  generation, transfer, scores and cache; historical ancestral timing is not
  directly comparable pure latency. One sampling/train seed, exploratory validation.
- Durable start2026-11-04 06:23:13UTC, completed06:26:54UTC, uploaded06:27:00UTC
  at private revision
  [4aa62f3b961460928adfc84876a97f050f8f5af9](https://huggingface.co/dacn-bmad-team/dacn-runs/tree/4aa62f3b961460928adfc84876a97f050f8f5af9/evaluations/gaussian-e30-t300-samplers-val83-20261104T062253Z).
  All83 completed; ancestral control exactly matches cache(max error0).
  Source snapshot SHA256c6060d080cff1b0b9a400d11917a083c00830e1d3dbef0018259a8bb1b11c90c;
  results SHA2560a92aeb35f4a7e120476815803108983aa4e620945426b28dcca29c7c61b6658;
  private NPZ SHA2565e467c8f5c27df73d193d79fc7d9fc17b313ee479b3bae5ae9877ffe4d04d67a.
  Combined reconstruction/score/cache loop107.983s, peak allocatedVRAM101,700,096B.
  Synchronized reconstruction-only totals: one-step2.77831s(0.03347s/image),
  DDIM10 25.31150s(0.30496s/image). Ancestral cache reused, full-set model time
  not remeasured; do not compare old end-to-end time as pure model latency.

  | Holdout43 pixel score | AP | ROC-AUC | Precision | Recall | F1 / micro Dice | Normal pixel FPR |
  | --- | --- | --- | --- | --- | --- | --- |
  | One-step / difference | 0.100353 | 0.879683 | 0.121102 | 0.291091 | 0.171044 | 0.055528 |
  | One-step / SSIM | 0.098530 | 0.875884 | 0.124930 | 0.155475 | 0.138538 | 0.031443 |
  | DDIM10 / difference | 0.091722 | 0.874711 | 0.103228 | 0.334848 | 0.157807 | 0.073501 |
  | DDIM10 / SSIM | 0.095631 | 0.879376 | 0.106620 | 0.186304 | 0.135623 | 0.042014 |
  | Ancestral300 / difference | 0.101066 | 0.876854 | 0.119934 | 0.314702 | 0.173678 | 0.058079 |
  | Ancestral300 / SSIM | 0.102402 | 0.888870 | 0.110476 | 0.257712 | 0.154654 | 0.055234 |
  | Brightness control | 0.389506 | 0.950197 | 0.161469 | 0.788577 | 0.268051 | 0.134935 |

  Global micro240crop. FPR is predicted-positive pixels on normal slices;
  thresholds selected on40 calibration slices, independently per method/score:
  one-step difference0.19594386219978333 /SSIM0.9078656758048764;
  DDIM difference0.17309430241584778 /SSIM0.8886450301482306;
  ancestral difference0.25365714728832245 /SSIM0.9568852878352576.
  Lower FPR is threshold-dependent and accompanies lower SSIM recall here.

  | All83 descriptive pixel score | AP | F1 / micro Dice | Normal pixel FPR |
  | --- | --- | --- | --- |
  | One-step / difference | 0.100640 | 0.184844 | 0.062191 |
  | One-step / SSIM | 0.105883 | 0.167365 | 0.037852 |
  | DDIM10 / difference | 0.089527 | 0.164374 | 0.080129 |
  | DDIM10 / SSIM | 0.096631 | 0.155240 | 0.066641 |
  | Ancestral300 / difference | 0.099834 | 0.182870 | 0.063527 |
  | Ancestral300 / SSIM | 0.100741 | 0.167483 | 0.071126 |

  All83 thresholds fitted in-sample, separate from holdout. All83 one-step SSIM
  AP rises about5.1% relative to ancestral, but F1 is essentially unchanged and
  holdout AP/F1 are lower. No consistent AP gain established; no intervals or
  repeated train/sampling seeds. DDIM10 loses AP versus ancestral on both views.

  | Holdout43 image score(top576 mean) | AP | ROC-AUC | Precision | Recall | F1 |
  | --- | --- | --- | --- | --- | --- |
  | One-step / difference | 0.434727 | 0.267544 | 0.547619 | 0.958333 | 0.696970 |
  | One-step / SSIM | 0.524525 | 0.412281 | 0.571429 | 1.000000 | 0.727273 |
  | DDIM10 / difference | 0.431448 | 0.252193 | 0.558140 | 1.000000 | 0.716418 |
  | DDIM10 / SSIM | 0.508218 | 0.377193 | 0.558140 | 1.000000 | 0.716418 |
  | Ancestral300 / difference | 0.430006 | 0.263158 | 0.558140 | 1.000000 | 0.716418 |
  | Ancestral300 / SSIM | 0.505258 | 0.372807 | 0.558140 | 1.000000 | 0.716418 |

  Image thresholds fixed on calibration; near-all-positive classification and
  low AUROC persist. Ancestral and brightness metrics reproduce previous values.
  Results support faster sampling, not claims of complete tumor removal or
  preserved normal anatomy with zero false positives. Test remains unused.

### gaussian-e30-onestep-val83-t100-20261104T061451Z — original epoch30 one-step — completed
- User requested original DDPM epoch30 one-step validation on T4; all83 images,
  39 normal/44 abnormal. [Modal app](https://modal.com/apps/phutungcogioihaitan/main/ap-Ja7XAVFsO7Z9fhZYWfXAFo).
  Source ddpm-partial-t4-b8-e30-cont-20261102T034644Z/ddpm_epoch_030.pt, EMA,
  SHA256d3e06d2082e18234899b201718abd59078c8a800d66da3cc5d8b9e0d9ca9b946.
- Original1-channel model, full-image Gaussian noise,t100/index99, one epsilon
  forward pass and analytical/clamped x0; base17 UUID-derived seed identical to
  original ancestral baseline initial noise. No training/context/patch averaging.
  Dataset350c74f2313e4e73ca65e7ffc6e9dcf74040b3b7; code00d199fbe726922caaf6696b02addffe380f0d93
  with dirty source snapshot in manifest. Fresh run, prior reverse run stays stopped.
- Full240crop absolute difference/localSSIM and brightness. Established40
  calibration/43 grouped holdout; frozen calibration thresholds for holdout,
  separate descriptive all83 metrics. One seed, previously inspected validation;
  test unused. Prefix caches/outputs private.
- Durable start2026-11-04 06:15:21UTC, completed06:17:10UTC, uploaded06:17:13UTC
  at private revision
  [b9f33457a84860275d7caecdfd0438693bbd706a](https://huggingface.co/dacn-bmad-team/dacn-runs/tree/b9f33457a84860275d7caecdfd0438693bbd706a/evaluations/gaussian-e30-onestep-val83-t100-20261104T061451Z).
  Source snapshot5ab1f0a81b31d322073d000fd3ec6c7293e3af3ed34eae08369b1ea6dce00a9b;
  results SHA25638944ba464f949ba699ad780ae210308ac9634290b4359d3f985d15d0b966a9c;
  private NPZ SHA2568fbba855651f2da6fa6ff0221b5813e614b0738aa3f191ae0ec2a038ffbd511b.
  All83 completed; loop51.448s (~0.620s/image) includes score maps, prefix cache
  compression/commits, excludes metric fits/setup/upload. Peak allocatedVRAM100,388,864bytes.

  | Partition / pixel score | AP | ROC-AUC | Precision | Recall | F1 / micro Dice |
  | --- | --- | --- | --- | --- | --- |
  | Group holdout43 / absolute difference | 0.075539 | 0.859018 | 0.079675 | 0.356717 | 0.130257 |
  | Group holdout43 / local SSIM | 0.075919 | 0.859406 | 0.078218 | 0.843484 | 0.143160 |
  | Group holdout43 / brightness | 0.389506 | 0.950197 | 0.161469 | 0.788577 | 0.268051 |
  | All83 descriptive / absolute difference | 0.070311 | 0.862703 | 0.072853 | 0.697097 | 0.131919 |
  | All83 descriptive / local SSIM | 0.071103 | 0.859760 | 0.067374 | 0.899868 | 0.125361 |
  | All83 descriptive / brightness | 0.272325 | 0.943925 | 0.265567 | 0.302829 | 0.282976 |

  Global micro over240crop. Holdout thresholds from40 calibration images:
  difference0.07415133714675903,SSIM0.09172538083860027. Normal holdout FPR
  difference0.109500/SSIM0.237986. All83 threshold fits are descriptive in-sample.

  | Group holdout43 / image score | AP | ROC-AUC | Precision | Recall | F1 |
  | --- | --- | --- | --- | --- | --- |
  | Absolute difference | 0.428963 | 0.199561 | 0.547619 | 0.958333 | 0.696970 |
  | Local SSIM | 0.529576 | 0.372807 | 0.547619 | 0.958333 | 0.696970 |
  | Brightness | 0.770309 | 0.679825 | 0.571429 | 1.000000 | 0.727273 |

  Image score mean top576 pixels. High image F1 accompanies low AUROC and
  near-all-positive predictions, not evidence of useful discrimination.
- Comparison: all83 AP difference/SSIM original Gaussian one-step0.070311/0.071103
  exceeds context one-step0.060280/0.065432, but does not exceed same-checkpoint
  ancestral t100 baseline0.070977/0.076971. F1 original one-step0.131919/0.125361
  versus ancestral0.134604/0.128203. Original one-step is faster at measured
  loop51.448s versus ancestral263.089s, but loop timing includes score/cache work
  and is not pure model latency. No gain in anomaly AP established. Single seed,
  exploratory previously inspected validation, no uncertainty intervals.

### patch-context-e5-onestep-val83-t100-20261104T060533Z — one-step context validation — completed
- User-authorized T4 one-step on all83 validation slices (39 normal,44 abnormal);
  no training. [Modal app](https://modal.com/apps/phutungcogioihaitan/main/ap-UHmd9YbsYm2PchcTNbnqOa).
  Source patch-context-ft5-t4-b8-20261102T155628Z, best EMA epoch5, SHA256
  5d3dbaaf69e80502eab692b545917caf0fcbd609c5864c636a3d1c84098082b3.
- Dataset350c74f2313e4e73ca65e7ffc6e9dcf74040b3b7; code00d199fbe726922caaf6696b02addffe380f0d93
  plus dirty snapshot recorded by runner. t100(index99), patch64/stride48,
  same base17 sample/patch noise as reverse diagnostic; estimate/clamp x0 once
  per patch and average overlap.25 forward calls/image. No reverse chain or
  clean target leakage inside conditioning patch. Test sealed.
- Full240crop absolute difference/localSSIM and brightness control. Primary
  exploratory40 calibration/43 group holdout with frozen calibration-only
  thresholds; separate all83 descriptive metrics. Single seed; no timestep sweep.
  Private prefix caches and final Hub upload. Durable start2026-11-04
  06:06:14UTC; source snapshot SHA256
  384eddbfbddcb3c2ab0a1a8e17e8971091197245dc1761991724344c896c6d69.
  Completed06:11:08UTC, private upload confirmed06:11:18UTC at revision
  [243070769bc36adaba8e21fda4822ec697205399](https://huggingface.co/dacn-bmad-team/dacn-runs/tree/243070769bc36adaba8e21fda4822ec697205399/evaluations/patch-context-e5-onestep-val83-t100-20261104T060533Z).
  All83 reconstructions complete. Inference/score/cache loop219.687s
  (~2.647s/image; includes score maps, prefix compression and Volume commits,
  excludes metric fits/setup/upload), peak allocatedVRAM111,073,280bytes.
  Results SHA256422a6a7d735bc0106e1e284b8b26aff90e9dcb8052fd4242eaf60651b1e66c0d;
  private NPZ SHA2567b9595c0329da4fe13c0083ec667113b82955404e489a5c157d2b5a850a56d7b.

  | Partition / pixel score | AP | ROC-AUC | Precision | Recall | F1 / micro Dice |
  | --- | --- | --- | --- | --- | --- |
  | Group holdout43 / absolute difference | 0.065636 | 0.811354 | 0.078176 | 0.371853 | 0.129191 |
  | Group holdout43 / local SSIM | 0.071232 | 0.836876 | 0.076317 | 0.762470 | 0.138747 |
  | Group holdout43 / brightness | 0.389506 | 0.950197 | 0.161469 | 0.788577 | 0.268051 |
  | All83 descriptive / absolute difference | 0.060280 | 0.813209 | 0.070970 | 0.459885 | 0.122964 |
  | All83 descriptive / local SSIM | 0.065432 | 0.834194 | 0.063729 | 0.764740 | 0.117654 |
  | All83 descriptive / brightness | 0.272325 | 0.943925 | 0.265567 | 0.302829 | 0.282976 |

  Global micro over240x240. Holdout thresholdsdifference0.07077845931053162,
  SSIM0.13216713297477745; fixed on40 calibration slices. Normal-holdout FPR
  difference0.115223 /SSIM0.218042. All83 fits separate in-sample thresholds;
  do not confuse these with holdout estimates.

  | Group holdout43 / image score | AP | ROC-AUC | Precision | Recall | F1 |
  | --- | --- | --- | --- | --- | --- |
  | Absolute difference | 0.430020 | 0.243421 | 0.536585 | 0.916667 | 0.676923 |
  | Local SSIM | 0.540039 | 0.399123 | 0.538462 | 0.875000 | 0.666667 |
  | Brightness | 0.770309 | 0.679825 | 0.571429 | 1.000000 | 0.727273 |

  Image scores mean top576pixels; thresholds fixed on calibration. Image F1
  with high recall does not establish useful discrimination given low AUROC.
  Compared with Gaussian epoch30 ancestral baseline at same t100/all83:
  difference AP0.070977->0.060280,F1 0.134604->0.122964;
  SSIM AP0.076971->0.065432,F1 0.128203->0.117654. Different model/conditioning
  and patch averaging mean this is not an isolated sampler ablation. Group
  brightness control reproduces previous results exactly. Context one-step
  does not improve the registered baseline in this experiment. No full83
  context reverse comparison exists because that run was user-stopped.
  One seed, previously inspected validation, no uncertainty intervals; test unused.

### patch-context-e5-val83-t100-20261102T165901Z — full validation patch/context — user-stopped
- User explicitly requested all83 validation slices after the four-example
  pilot. Fixed source patch-context-ft5-t4-b8-20261102T155628Z, best EMA epoch5,
  SHA2565d3dbaaf69e80502eab692b545917caf0fcbd609c5864c636a3d1c84098082b3.
- [Modal app](https://modal.com/apps/phutungcogioihaitan/main/ap-5etvxh2YHgTHJhuu8M6JTe).
  T4,3-hour ceiling, detached. Dataset revision350c74f2313e4e73ca65e7ffc6e9dcf74040b3b7;
  code00d199fbe726922caaf6696b02addffe380f0d93 with dirty source snapshot pinned
  in durable manifest. T100, fixed-large stochastic chain, patch64/stride48,
  25 patches/image, sample/patch seeds identical to diagnostic protocol.
- Planned coverage: all83 slices (39 normal,44 abnormal). Global240crop pixel difference and
  localSSIM, plus brightness control; report pixel/image ROC-AUC, precision,
  recall,F1/AP/Dice and runtime/VRAM from produced results. Primary exploratory
  comparison uses the established40 calibration /43 patient-group-disjoint
  holdout slices, thresholds fitted only on calibration, frozen for holdout.
  All-validation descriptive metrics use separate in-sample thresholds.
- Persist private prefix NPZ after first and every5 completed samples; private
  final Hub upload. Estimated ~100min from pilot; no measured final latency or
  anomaly metrics yet. Test sealed, no extra training/timestep sweep. Durable
  start2026-11-02 16:59:21UTC; checkpoint hash confirmed, stagevalidate_dataset
  at16:59:24UTC. Source snapshot SHA256
  f90a3773e2d47b3a65238d5f392581550caebfdbf9b532955f6128dc83cfdc45.
- User requested pause; Modal app stop confirmed stopped with0 tasks at
  2026-11-03 00:00:34 Asia/Ho_Chi_Minh (2026-11-02 17:00:34UTC).
  Last durable manifest remains running at validation_evaluation because abrupt
  cancellation does not finalize it. No results/progress.json exists; no
  completed-sample prefix or full-validation metrics confirmed. Prior5-epoch
  checkpoint and four-example diagnostic remain available. Do not restart
  automatically.

### patch-context-e5-recon4-t100-20261102T163928Z — bounded reconstruction diagnostic — completed
- User-requested reconstruction trial, T4, two normal/two abnormal validation
  slices uniformly sampled per class with seed17. No test data. Source run
  patch-context-ft5-t4-b8-20261102T155628Z, best EMA epoch5, SHA256
  5d3dbaaf69e80502eab692b545917caf0fcbd609c5864c636a3d1c84098082b3.
- [Modal app](https://modal.com/apps/phutungcogioihaitan/main/ap-hd2NZeEANWnVk9F5Y7BAMr).
  Dataset350c74f2313e4e73ca65e7ffc6e9dcf74040b3b7; code00d199fbe726922caaf6696b02addffe380f0d93
  with dirty source snapshot recorded in durable manifest. T100, fixed-large
  stochastic reverse, patch64/stride48,25 patches per image, independent
  per-patch seeds derived from base17 and private sample UUID. No tumor-mask
  input to reconstruction. Save private input/output/mask/maps and aggregate
  foreground quality/timing; no population AP or threshold selection.
- Durable start2026-11-02 16:39:49UTC, evaluation stage16:40:11UTC on TeslaT4;
  source snapshot SHA256a05d85dd43225e8d438bb6983a4b3fdf339440901dbca220f1418d322ed8167a.
  Completed16:45:03UTC; uploaded16:45:06UTC at private revision
  [c08f4805a3a64fab9455be05815f949636e3d484](https://huggingface.co/dacn-bmad-team/dacn-runs/tree/c08f4805a3a64fab9455be05815f949636e3d484/evaluations/patch-context-e5-recon4-t100-20261102T163928Z).
  Inference289.624s (~72s/image), peak allocatedVRAM106,093,056bytes.
  Result SHA256b0cc838d741262b9e03781ccae3220909bde5b9cab09051fff812c6374cdf625;
  private NPZ SHA256c6a90246e4f29575d6f2d6d168fd95d0c2f941e18f6a0f6e5c86835d2adc9e30.

  | Example | Class | Foreground MAE [0,1] | Foreground SSIM | Tumor MAE [0,1] | Non-lesion foreground MAE [0,1] |
  | --- | --- | --- | --- | --- | --- |
  | 1 | Normal | 0.047572 | 0.568819 | n/a | 0.047572 |
  | 2 | Normal | 0.054103 | 0.563621 | n/a | 0.054103 |
  | 3 | Tumor | 0.041982 | 0.589217 | 0.044634 | 0.041322 |
  | 4 | Tumor | 0.037172 | 0.513111 | 0.033536 | 0.037772 |

  Foreground=input>-0.95 inside240crop, a heuristic tissue mask, not a clinical
  segmentation. Prior Gaussian epoch30 t100 cache on the same selected inputs
  gives normal-example MAE0.050346/0.060776 and SSIM0.537962/0.528152.
  Patch/context reduces normal reconstruction error in these two examples, but
  differs in fine-tuning, sampling noise and25-patch averaging; no causal
  attribution to conditioning alone. Tumor-vs-nonlesion MAE ratios1.080/0.888
  show weak/reversed separation; no improved anomaly AP claim. No full
  validation sweep or extra training.
  Local ignored comparison.png visually confirms retained bright lesion
  structures and broadly distributed tissue residuals; images and derived
  arrays remain private/ignored. Inputs and masks matched the prior Gaussian
  cache exactly for all four examples.

### patch-context-ft5-t4-b8-20261102T155628Z — patch/context fine-tune pilot — completed
- User-approved5-epoch pilot, durable start UTC2026-11-02 16:00:57 on Tesla T4.
  [Modal app](https://modal.com/apps/phutungcogioihaitan/main/ap-EBw2niCSb3IeS2iLs2DK5W).
  Trainer dacn.training.patch_context:train; config
  configs/patch_context_t4_finetune5.json; seed17, batch8, LR1e-5, EMA0.9992,
  patch64, all7,500 normal training slices,938 steps/epoch,4,690 planned new
  steps. Normal-val patch/noise/timestep fixed seed1,000,017 over39 slices.
- Initialization: Gaussian epoch30 EMA from
  ddpm-partial-t4-b8-e30-cont-20261102T034644Z; source checkpoint SHA256
  d3e06d2082e18234899b201718abd59078c8a800d66da3cc5d8b9e0d9ca9b946.
  Expand input1->3, copy original image kernels, new context/mask kernels0;
  output1; reset optimizer/RNG and EMA equal to expanded model. Training
  epochs1..5 are fine-tune epochs, not a continuation of the old objective.
- Source dataset revision350c74f2313e4e73ca65e7ffc6e9dcf74040b3b7; code revision
  00d199fbe726922caaf6696b02addffe380f0d93 with dirty source snapshot SHA256
  4ca43d18f31c1ace20ee0801f7cfa0307e8820558bd5032a98408aa550d4817b.
- Adaptation protocol: masked-patch epsilon loss, no clean target leakage through
  context, no tumor masks in training. Same U-Net backbone/linearGaussianT1000.
  Initial output reached finetune_epoch1,step25/938,200/7,500 samples;
  running mean patch epsilon MSE2.944604 (early log only, not final result).
  The input/noising distribution changes; do not compare raw patch MSE directly
  to the earlier global loss or infer anomaly performance from it.
- Best checkpoint selected by normal-validation patch MSE. Saves best/last plus
  milestones1/3/5 with optimizer/RNG/sampler/history, private final-EMA normal
  preview; private Hub upload confirmed at revision
  [62a2d58b5db5004cc6337626a026c76ba7ebf963](https://huggingface.co/dacn-bmad-team/dacn-runs/tree/62a2d58b5db5004cc6337626a026c76ba7ebf963/runs/patch-context-ft5-t4-b8-20261102T155628Z).
  Durable manifest: completed2026-11-02 16:35:49UTC, uploaded16:35:57UTC;
  all5 epochs /4,690 optimizer steps. Training+validation2008.7673s
  (33.48min, approximately6.70min/epoch); peakVRAM5,468,831,232bytes.
  Best normal-validation checkpoint is epoch5, patch_context_best.pt.
  Produced training_history.json SHA256
  ce34e4bb41e97e8bfc9bd5f426cff85b575b01f4a2b3ec92ef68c3eb61f2d43a:

  | Fine-tune epoch | Train patch epsilon MSE | Normal-validation EMA patch epsilon MSE |
  | --- | --- | --- |
  | 1 | 0.161592606 | 0.423785761 |
  | 2 | 0.034335899 | 0.151504139 |
  | 3 | 0.031836559 | 0.065445080 |
  | 4 | 0.029152573 | 0.043553216 |
  | 5 | 0.030230757 | 0.037765748 |

  Patch validation loss declined through epoch5; this is not evidence of
  improved tumor AP or reconstruction. No anomaly AP/F1/ROC-AUC/precision/recall
  evaluation was performed by this training run. Later sliding inference
  requires its own bounded cost estimate; no
  automatic full evaluation sweep. Test sealed; no continuous monitoring.

### gaussian-e30-contract-audit-20261102T153939Z — checkpoint/sampler diagnosis — completed
- User-authorized bounded T4 audit, durable start UTC2026-11-02 15:40:10;
  [Modal app](https://modal.com/apps/phutungcogioihaitan/main/ap-LOziRe7luRTbYyH7UZYu6y).
  Gaussian EMA epoch30, source run/ddataset as prior epoch30 experiments;
  explicit checkpoint SHA256d3e06d2082e18234899b201718abd59078c8a800d66da3cc5d8b9e0d9ca9b946.
- Fixed4 random normal-val images (seed17, no quality selection); t100/300,
  one-step EMA versus online, full chains fixed-large/fixed-small/mean sharing
  initial noise and stochastic draws. Verify EMA tensors, epoch30/28140 steps,
  train/val preprocessing samples, schedule/variance, oracle inversion and mean
  equations. Report foreground/interior reconstruction quality, not anomaly AP.
- Code revision00d199fbe726922caaf6696b02addffe380f0d93, dirty source snapshot
  SHA256d788fb7be26ba844b4fd782a4f07374f70751e362a8b050318724734ddf185d2.
  Dataset revision350c74f2313e4e73ca65e7ffc6e9dcf74040b3b7. Full source checks
  and method details in PIPELINE_CONTRACT_AUDIT. Completed15:43:10UTC;
  [private revision f69fc08d](https://huggingface.co/dacn-bmad-team/dacn-runs/tree/f69fc08d94816d6be82563d8a603a4291ea04b0a/evaluations/gaussian-e30-contract-audit-20261102T153939Z).
- Exact EMA load/epoch/steps, input padding/range/grid and schedule/variance
  verified. Training snapshot unet/shards/bmad modules identical. Oracle x0 and
  posterior-mean arithmetic checks pass; no contract bug found within scope.
- t300 foreground mean per-image MAE/SSIM: one-step EMA0.053872/0.527959;
  online0.054014/0.520815; fixed-large chain0.074614/0.358931;
  fixed-small0.074133/0.365356; mean chain0.091689/0.465686.
  t100 corresponding EMA0.033427/0.714672; large0.046668/0.571649;
  small0.045906/0.582003; mean0.034406/0.708926. No uniform win from variance
  changes. Full-chain normal distortion observed, not a proof of sole cause.
- Runtime150.763s, peakVRAM137,601,536B. Classification metrics N/A: four
  normal-only inputs, not AP comparison. No production sampler change,
  retraining or held-out test use. Further training optimization is a proposal,
  not a result; detailed limits and priorities in PIPELINE_CONTRACT_AUDIT.

### gaussian-e30-normal-foreground-audit-20261102 — CPU cache audit — completed
- User requested reconstruction quality on normal tissue only. Reused all39
  normal validation slices from Gaussian epoch30 caches at t100/200/300, original
  input-derived foreground >-0.95; same ROI at all timesteps, no GPU/test/train.
  Source hashes/content alignment checked. Region is a heuristic, not brain
  segmentation; no threshold-based anomaly metrics on this normal-only subset.
- Foreground[0,1] MAE t100/200/300=0.051991/0.069675/0.082587;
  mean localSSIM=0.532626/0.374804/0.292559. Corresponding full-crop meanSSIM
  =0.869570/0.823210/0.797223. Background comprises74.11% of pixels and inflates
  apparent reconstruction quality. Interior t300 meanSSIM0.269191.
- Full metrics/limits/checks in NORMAL_RECONSTRUCTION_AUDIT. Aggregate JSON hash
  f4d6b9b43f0ae2f0f92c67a4bd7e2a2dbd372c7b9e5695357c0aed3cf6a95e86;
  private paired examples/maps stay in ignored runs/. CPU40.586s, inference
  VRAM/latency N/A; AP/AUROC/Precision/Recall/F1 N/A for this unthresholded audit.
  Supports normal-structure error as a contributor, not a proof of root cause.

### ddpm-e30-multitimestep-rankmean-20261102T1430Z — CPU cached-map ensemble — completed
- User-requested combination of reconstruction anomaly maps at t100/200/300,
  Gaussian EMA epoch30, seed17, full83 validation caches from prior completed
  timestep evaluations. No new GPU inference or training; held-out test unused.
- Fixed protocol in RESEARCH before execution: separate calibration-only
  empirical midrank CDF normalization for each score/timestep, equal1/3 mean
  across timesteps; two ensembles (absolute difference and SSIM). Same grouped
  seed17 split40 calibration/43 holdout,25 groups each, full240x240 ROI,
  calibration max-F1 thresholds. Six raw single-timestep controls plus brightness.
  No weight search or holdout selection. Cache hashes and content alignment
  asserted by `scripts/evaluate_multitimestep_maps.py`.
- Holdout pixel difference ensemble: AUROC0.882247, AP0.091536,
  Precision0.098158, Recall0.372177, F1/Dice0.155345,
  threshold0.8950128761574074; macro lesion Dice0.224811,
  empty-normal FPR0.087952. SSIM ensemble: AUROC0.879846, AP0.092199,
  Precision0.097410, Recall0.206755, F1/Dice0.132428,
  threshold0.9361038049768517; macro0.170997, normal FPR0.049931.
- Both ensembles underperform the corresponding t300 raw score on the same
  holdout (AP0.101066 difference /0.102402 SSIM). Brightness AP0.389506
  reproduced exactly. Original t200/t300 metrics reproduced, confirming cache
  order/partition agreement. No improvement established for equal rank-mean.
- Aggregate results remain in ignored local runs/; JSON SHA256
  `ec19c5a6ad1cb6f7f0b2d1f66c9bdc9b65f9c98fad20b738573956520503ecd7`.
  CPU96.510s; inference VRAM/latency N/A. One seed/split, no uncertainty
  intervals; exploratory previously inspected validation, not final test.

### autoddpm-style-ssim-e30-20261102T140011Z — mask/stitch/resample adaptation — completed
- User-authorized T4 pilot, durable start UTC2026-11-02 14:01:03.
  [Modal app](https://modal.com/apps/phutungcogioihaitan/main/ap-pWoLoKKbxzgCa1RD9FoPOS).
  This is AutoDDPM-style with an SSIM proxy, not exact LPIPS-based AutoDDPM.
- Source Gaussian EMA epoch30 from
  `ddpm-partial-t4-b8-e30-cont-20261102T034644Z`; checkpoint SHA256 is checked
  against `d3e06d2082e18234899b201718abd59078c8a800d66da3cc5d8b9e0d9ca9b946`.
  Dataset revision `350c74f2313e4e73ca65e7ffc6e9dcf74040b3b7`.
- Initial t200 cache from `ddpm-val-e30-t100-200-diag-20261102T094115Z`, SHA256
  `cffefdc5d2cf599fc3903eb8a8fa9a192dddd2dabe4cffeb04ee28cfa60bc06a`;
  every image/mask array is checked against dataset order. Initial mask uses
  norm95 absolute residual times clipped local-SSIM difference, threshold0.5,
  dilation3; original input outside mask, initial reconstruction inside.
  Resample50 steps x5 repeats with matching-time noised context; fixed-large
  ancestral reverse, seed17, batch1. Ground-truth masks used only for metrics.
- All83 validation images,40 calibration/43 grouped holdout,25 groups each,
  calibration-only thresholds. Eight fixed score/control outputs; no holdout
  selection. Report naive stitch and initial raw scores alongside final scores
  and mask coverage, because exact context preservation alone is imposed by
  construction. Test unused, no retraining.
- Code revision `00d199fbe726922caaf6696b02addffe380f0d93` with dirty source
  snapshot SHA256 `af2d42ac4b06b1ad437be00b93bc5524175e3f30025e406bfd884189715c4a37`.
  Completed UTC2026-11-02 14:15:29; private upload revision
  [`2c8646c1`](https://huggingface.co/dacn-bmad-team/dacn-runs/tree/2c8646c1d8072c915e00df2c5cb7acf4bfbf87bd/evaluations/autoddpm-style-ssim-e30-20261102T140011Z).
- Holdout43-image pixel AP/F1: initial t200 difference0.085967/0.148511;
  initial SSIM0.090168/0.124833; initial heatmap0.097540/0.148673;
  naive stitch0.069851/0.153817; resampled difference0.070007/0.149523;
  resampled SSIM0.093603/0.138912; uncertainty-weighted difference
  0.072450/0.148187; brightness0.389506/0.268051.
- Resampled SSIM holdout pixel AUROC0.879478, Precision0.100619,
  Recall0.224261, threshold0.8825730735439271; macro lesion Dice0.185507,
  normal FPR0.053028. Resampled difference AUROC0.746651,
  Precision0.103225, Recall0.271124, threshold0.1666409969329834;
  macro0.207673, FPR0.062329. All thresholds fixed on calibration. Resampled
  SSIM image AUROC0.368421/F1 0.716418 is an all-positive image prediction.
- Predicted mask on holdout covers63.0463% of lesion pixels, Precision9.2048%,
  ROI fraction15.4016%. Thus36.9537% of lesion pixels are excluded from the
  unknown region; exact preservation outside mask constrains residual scoring.
  Calibration mask coverage76.9692%, precision8.1395%, ROI fraction15.6217%.
- Inference/cache scoring loop769.038s for83 images (excludes metric fits,
  setup/upload); peak VRAM101,765,120B. Private map SHA256
  `1190e602197f183931b8642b93dc53e5a477ade115fa9022f839339f70c7452c`.
  One seed/split, no uncertainty intervals; exploratory validation only.
- Interpretation: small SSIM gain over t200 initial but below t300 SSIM
  AP0.102402 on same holdout; resampling difference barely changes naive-stitch
  AP and loses F1. No overall AP improvement established. This negative result
  concerns the SSIM-proxy adaptation, not exact LPIPS-based AutoDDPM.

### ddpm-denoising-e30-3t4r-20261102T103159Z — fixed noise-error score — completed
- User-authorized Modal T4 experiment, durable start UTC2026-11-02 10:32:32.
  [Modal app](https://modal.com/apps/phutungcogioihaitan/main/ap-beD6gCpUwEv4VHg0Eu2krT).
- Gaussian EMA epoch30, source run
  `ddpm-partial-t4-b8-e30-cont-20261102T034644Z`, checkpoint SHA256
  `d3e06d2082e18234899b201718abd59078c8a800d66da3cc5d8b9e0d9ca9b946`;
  dataset revision `350c74f2313e4e73ca65e7ffc6e9dcf74040b3b7`.
- Fixed score: mean squared epsilon-prediction error at one-based100/200/300,
  four independent Gaussian noise draws per image; same draw across timesteps.
  Batch1, seed17, full240x240 ROI, no reverse chain/smoothing/weight search.
  All83 validation images; first grouped two-fold split seed17,40 calibration
  /43 internal holdout,25 disjoint patient groups each. Fit thresholds on
  calibration only, compare brightness plus earlier reconstruction holdout
  scores. Previously viewed validation; exploratory, no claim of final test.
- Code revision `00d199fbe726922caaf6696b02addffe380f0d93` with dirty snapshot
  SHA256 `2dd1b6166bcf1eeac9f414d6dd50c31a11cb1b5d0763468e0278568677315c74`.
  Completed UTC2026-11-02 10:33:39; uploaded private revision
  [`4713bd88`](https://huggingface.co/dacn-bmad-team/dacn-runs/tree/4713bd888ae70f3bf0c343c3dd4367837c88cbb4/evaluations/ddpm-denoising-e30-3t4r-20261102T103159Z).
- Denoising holdout pixel AUROC0.884733, AP0.085848, Precision0.096586,
  Recall0.573527, F1/Dice0.165330; threshold0.023040642716790034 frozen from
  calibration (calibration AP0.081437). Macro lesion Dice0.257572,
  empty-normal FPR0.141006. Image AUROC0.278509, AP0.455950,
  Precision0.547619, Recall0.958333, F1 0.696970;
  image threshold0.1847808777670508 frozen from calibration.
- Same holdout brightness control exactly reproduces prior metrics:
  pixel AUROC0.950197, AP0.389506, Precision0.161469, Recall0.788577,
  F1 0.268051. Original reconstruction AP difference0.101066 /SSIM0.102402
  on these same43 images. Thus simple averaged epsilon-error score does not
  improve AP over either reconstruction baseline or brightness.
- Inference/data-loop time28.876s for83 images (about0.348s/image), excludes
  metric computation/setup/upload; peak allocated VRAM101,437,440B.
  Private score-map hash79eea9e4bc13574b43044e6a2b2b046027e3db09aff390cff4cf7175d8c678ca.
  One training/sampling/partition seed; no uncertainty intervals. Negative
  evidence for this score only, not full ANDi/all diffusion methods. Test unused.

### ddpm-e30-score-fusion-group-holdout-20261102T1001Z — CPU score ablation — completed
- User-authorized nine-candidate experiment on existing Gaussian epoch30/t300
  cache; no new GPU/train/test. Protocol fixed in RESEARCH before execution:
  first grouped two-fold split seed17,40 calibration images/25 groups versus
  43 internal holdout images/25 disjoint groups. Fit normalization, weights and
  thresholds only on calibration; selected50% brightness+pixel difference by AP.
- Holdout pixel AP/AUROC/Precision/Recall/F1 respectively: selected fusion
  0.262977/0.934268/0.275226/0.377455/0.318334; brightness
  0.389506/0.950197/0.161469/0.788577/0.268051; original absolute difference
  0.101066/0.876854/0.119934/0.314702/0.173678; SSIM
  0.102402/0.888870/0.110476/0.257712/0.154654.
- All nine rows, macro Dice/FPR, limitations and artifact hashes are documented
  in `docs/SCORE_FUSION_RESULTS.md`. CPU20.628s; inference VRAM/latency N/A.
  No mixture beat brightness-only AP on this holdout. Do not use holdout to
  reselect weights or compare this43-image AP directly with full83-image AP.
  Validation was previously inspected; exploratory, one split, no intervals.

### ddpm-e30-t300-cache-diagnostic-20261102T0941Z — CPU cache analysis — completed
- User-requested diagnosis, all83 validation slices from the saved Gaussian
  epoch30 t*=300 cache. Cache SHA256
  `8278ad18543dd87f698c36b510564b9fcd4445d4a1b0ae5d7a859093e2a4e959`;
  source evaluation `ddpm-val-t4b8-e30-t300-20261102T071516Z`, same dataset,
  model, seed, full240x240 ROI and validation-max-F1 threshold rules.
- Script `scripts/diagnose_ddpm_cache.py`, CPU only, no new inference/test use.
  Region definitions, complete brief metrics and interpretation are in
  `docs/EPOCH30_DIAGNOSIS.md`; aggregate JSON stays in ignored runs/.
- Brightness-control pixel AUROC 0.943925, AP 0.272325, Precision 0.265567,
  Recall 0.302829, F1/Dice 0.282976. Foreground-gated difference AUROC 0.903959,
  AP 0.108038, Precision 0.124594, Recall 0.404091, F1 0.190463.
  Foreground-gated SSIM AUROC 0.900629, AP 0.103202, Precision 0.107484,
  Recall 0.403228, F1 0.169726. Baseline cache metrics exactly reproduced.
- 83.47% of baseline SSIM false positives are in nonlesion foreground interior;
  background accounts for only1.72%. Lesion prevalence1.9611%. One train/one
  sampling seed, no intervals. Latency/VRAM N/A: offline CPU diagnostic, not
  inference benchmark; no GPU allocated. No claims of clinical usefulness.

### ddpm-val-e30-t100-200-diag-20261102T094115Z — timestep diagnosis — completed
- User-authorized Gaussian EMA epoch30 validation at t*=100/200, reusing the
  previously completed t*=300 result. Same full83 validation slices, dataset
  revision `350c74f2313e4e73ca65e7ffc6e9dcf74040b3b7`, train source
  `ddpm-partial-t4-b8-e30-cont-20261102T034644Z`, explicit epoch30 milestone,
  T4, ancestral sampler, batch1, seed17, full240x240 ROI, both original score
  maps and validation-max-F1 thresholds. Held-out test unused; private cache on.
- Code revision `00d199fbe726922caaf6696b02addffe380f0d93` with dirty source
  snapshot; submitted UTC2026-11-02 approximately09:42.
  [Modal app](https://modal.com/apps/phutungcogioihaitan/main/ap-Yh8Bu3Qilehwpdxl5LDKkK).
  Completed UTC2026-11-02 09:56:00; uploaded private revision
  [`67626af0`](https://huggingface.co/dacn-bmad-team/dacn-runs/tree/67626af06afb1520ba0ab95b69eaf010b2ca24ed/evaluations/ddpm-val-e30-t100-200-diag-20261102T094115Z).
- t*=100: SSIM pixel AUROC0.868975, AP0.076971, Precision0.070057,
  Recall0.754000, F1/Dice0.128203, macro lesion Dice0.206457,
  empty-normal FPR0.199830, threshold0.2596774082491453. Absolute difference
  AUROC0.848351, AP0.070977, Precision0.075704, Recall0.606415,
  F1/Dice0.134604, macro0.219808, FPR0.155812,
  threshold0.061692699790000916. Runtime263.089s /3.170s per slice.
- t*=200: SSIM pixel AUROC0.880650, AP0.088429, Precision0.085001,
  Recall0.419131, F1/Dice0.141339, macro0.219956, FPR0.099155,
  threshold0.7046124709683635. Absolute difference AUROC0.867116,
  AP0.083184, Precision0.093829, Recall0.430916, F1/Dice0.154103,
  macro0.235807, FPR0.089256, threshold0.1503899097442627.
  Runtime533.091s /6.423s per slice. Peak allocated VRAM100,913,664B for each.
- Image AUROC remains poor: SSIM t100/200=0.366550/0.384033;
  difference=0.217949/0.207459. Both scores/timesteps select all-positive image
  predictions (Precision0.530120, Recall1, F1 0.692913). One train/one sampling
  seed, no uncertainty intervals. Held-out test unused.
- Interpretation: lowering t* from300 does not improve AP/F1 for this checkpoint;
  t*=300 remains best of100/200/300 for both original pixel scores. At separately
  optimized thresholds normal FPR is higher at smaller t*, despite less corruption;
  these threshold-dependent FPR values do not alone measure reconstruction quality.

### ddpm-val-simplex-e10-t300-20261102T090806Z — Simplex epoch-10 validation — completed
- User-requested evaluation, submitted UTC 2026-11-02 09:08; Modal created
  [app `ap-ju8ELLyWBFQIacpQs4fcxK`](https://modal.com/apps/phutungcogioihaitan/main/ap-ju8ELLyWBFQIacpQs4fcxK).
- Source run `ddpm-simplex-t4-b8-e10-20261102T074257Z`, EMA milestone
  `ddpm_epoch_010.pt`, SHA256
  `7b46c34ff1075102fe140d344621176988dd0f8b46506a912f014d3bf4f846aa`.
- Dataset revision `350c74f2313e4e73ca65e7ffc6e9dcf74040b3b7`, all 83 validation
  slices, t*=300 / T=1000, Simplex forward corruption, Gaussian fixed-large
  ancestral reverse, T4 batch1, sampling seed17. Paired absolute-difference and
  local-SSIM maps, full 240x240 ROI, validation-max-F1 thresholds with the
  largest-threshold tie break; image score mean top1%. Held-out test unused.
- Code revision `00d199fbe726922caaf6696b02addffe380f0d93` with dirty source
  snapshot and private reconstruction cache.
  Durable start 2026-11-02 09:08:35 UTC; Tesla T4 confirmed. Source snapshot
  SHA256 `6dd0e4821f071514f3c07c0409867a2594c4e63c3a768f53d2cb27affbf66c29`.
- Completed UTC 2026-11-02 09:21:53, uploaded private revision
  [`b4c408be`](https://huggingface.co/dacn-bmad-team/dacn-runs/tree/b4c408beaf7afd1cea81831e785d2333ce84d753/evaluations/ddpm-val-simplex-e10-t300-20261102T090806Z).
  One train seed and one sampling seed; no uncertainty estimates computed.
- Pixel global micro metrics: local SSIM AUROC 0.737657, AP 0.073910,
  Precision 0.096096, Recall 0.603844, F1/Dice 0.165806,
  macro lesion-slice Dice 0.278386, empty-normal FPR 0.133712;
  threshold 0.998853041422. Pixel difference AUROC 0.673577, AP 0.032447,
  Precision 0.037744, Recall 0.348607, F1/Dice 0.068113,
  macro lesion-slice Dice 0.114868, empty-normal FPR 0.189460;
  threshold 0.8980831503868103.
- Image metrics: SSIM AUROC 0.217366, AP 0.385214, Precision 0.530120,
  Recall 1, F1 0.692913, threshold 1.0678693280381035 (all images positive).
  Pixel difference AUROC 0.351981, AP 0.446235, Precision 0.5375,
  Recall 0.977273, F1 0.693548, threshold 1.903656475349433.
- Runtime 749.373 seconds / 9.029 seconds per slice, cache write 1.654 seconds,
  peak allocated VRAM 100,913,664 bytes. Reconstruction cache SHA256
  `5bf4cfddf144ac2ce88e3da6417df5e86cea91bcff4d39f97738ae70ab2d0218`.
- Interpretation: Simplex at t*=300 does not improve the main pixel AP over
  Gaussian baseline; SSIM Recall increases but false positives also increase.
  Gaussian epoch10 batch8 has not yet been evaluated; older Gaussian epoch10
  uses batch1/different EMA, and Gaussian epoch30 uses more optimization steps.
  These are contextual comparisons, not isolated causal noise ablations.

### ddpm-simplex-t4-b8-e10-20261102T074257Z — Simplex training ablation — completed
- Started (UTC): 2026-11-02 07:44:01. User requested a fresh 10-epoch Simplex
  experiment. Shared Workspace and existing Secret name verified before launch.
  [Modal app](https://modal.com/apps/phutungcogioihaitan/main/ap-ssVCXTam3wHn7EaSMJFMtS).
- Config: `configs/ddpm_partial_simplex_t4_batch8_epoch10.json`; trainer
  `dacn.training.ddpm_partial:train`, T4 batch 8, 7,500 normal train slices,
  938 steps/epoch, 9,380 planned total steps, seed 17, AdamW LR 1e-4,
  EMA 0.9992, base32 U-Net with unchanged architecture, linear T=1000.
  Simplex forward noise: six octaves, persistence 0.8, frequency divisor 64.
  Fresh random initialization; no Gaussian checkpoint resume. Saves full-state
  milestones 1/5/10 and normal-validation-MSE selected best/last checkpoints.
- Dataset revision `350c74f2313e4e73ca65e7ffc6e9dcf74040b3b7`;
  all 39 normal val slices for selection, held-out test unavailable. Noise MSE
  is not directly comparable between Gaussian and Simplex noise distributions.
- Code revision `00d199fbe726922caaf6696b02addffe380f0d93`, dirty source snapshot
  SHA256 `6dd0e4821f071514f3c07c0409867a2594c4e63c3a768f53d2cb27affbf66c29`.
  Durable manifest confirms completed/uploaded at 2026-11-02 09:00:14 UTC.
  Completed 10 epochs / 9,380 optimizer steps, best normal-validation epsilon
  MSE 0.002971782594 at epoch 10; completed epoch time 4,465.438 seconds;
  peak allocated VRAM 5,465,677,824 bytes. Milestones 1/5/10 exist.
  [Private artifacts revision `e7b6ce4f`](https://huggingface.co/dacn-bmad-team/dacn-runs/tree/e7b6ce4f167de1510b3111b6de9b10ab7d77492b/runs/ddpm-simplex-t4-b8-e10-20261102T074257Z).
- Evaluation protocol: milestone epoch10 at t*=300, ancestral
  reverse (Gaussian transition noise, Simplex forward corruption), sampling
  seed17, all83 validation slices, same paired pixel-difference/local-SSIM score
  and threshold protocol. Compare with Gaussian epoch10 batch8 when evaluated.

Copy this block for every actual run. Use `N/A` with a reason when a metric does
not apply. For a failed or interrupted run, include the error class and last
completed stage rather than fabricating metrics.

### ddpm-val-t4b8-e30-t300-20261102T071516Z — epoch-30 validation — completed
- User-authorized evaluation of EMA `ddpm_epoch_030.pt` from training run
  `ddpm-partial-t4-b8-e30-cont-20261102T034644Z`; checkpoint SHA256
  `d3e06d2082e18234899b201718abd59078c8a800d66da3cc5d8b9e0d9ca9b946`.
- Protocol: all 83 validation slices (39 normal / 44 anomalous), dataset revision
  `350c74f2313e4e73ca65e7ffc6e9dcf74040b3b7`, one-based `t*=300`, full schedule
  `T=1000`, Gaussian noise, ancestral reverse, T4, inference batch 1, sampling
  seed 17. Pixel absolute difference and local one-minus-SSIM share each
  reconstruction; original 240x240 ROI, no foreground filtering. Image score
  is mean top 1% pixels; pixel/image thresholds maximize validation F1 with
  highest-threshold tie break. Held-out test remains unused.
- Code revision `00d199fbe726922caaf6696b02addffe380f0d93` with dirty source
  snapshot; private reconstruction cache enabled. Completed UTC
  2026-11-02 07:28:35; uploaded private revision
  [`4472519e`](https://huggingface.co/dacn-bmad-team/dacn-runs/tree/4472519e6d3e86fc2bf7686e110408df27727571/evaluations/ddpm-val-t4b8-e30-t300-20261102T071516Z).
- Pixel metrics (global micro, one train seed / one evaluation seed): local SSIM
  AUROC 0.895647, AP 0.100741, Precision 0.107193, Recall 0.382770,
  F1/Dice 0.167483, macro lesion-slice Dice 0.245881, empty-normal FPR 0.071126,
  threshold 0.8922310462404459. Pixel difference AUROC 0.884132, AP 0.099834,
  Precision 0.118175, Recall 0.404081, F1/Dice 0.182870, macro lesion-slice Dice
  0.259971, empty-normal FPR 0.063527, threshold 0.22918590903282166.
- Image AUROC remains poor: SSIM 0.383450 / pixel difference 0.254079. Both
  validation-selected image thresholds predict all images positive (Precision
  0.530120, Recall 1, F1 0.692913); this F1 does not indicate useful separation.
- Runtime: 718.415 seconds, 8.656 seconds/slice; cache write 2.290 seconds;
  peak allocated VRAM 100,913,664 bytes. Cache SHA256
  `8278ad18543dd87f698c36b510564b9fcd4445d4a1b0ae5d7a859093e2a4e959`.
- Interpretation: SSIM AP rises from prior batch-1 epoch-10 baseline 0.095895
  to 0.100741 and F1 from 0.161616 to 0.167483, a small improvement. This is
  not an isolated epoch comparison: batch size, EMA and optimizer-step count
  differ between training runs. No patient-bootstrap uncertainty computed;
  held-out test remains unused.
  [Modal app](https://modal.com/apps/phutungcogioihaitan/main/ap-RWi4F9nKiAnh75BoUA42BG).

### ddpm-partial-t4-b8-e30-cont-20261102T034644Z — T4 epoch-2-to-30 continuation — completed
- Submitted (UTC): 2026-11-02 03:46:44. User-authorized Tesla T4 continuation
  from completed source run `ddpm-partial-t4-b8-e1-20261102T032740Z`; Modal
  reached training and logged epoch 2, confirming it did not restart epoch 1.
- Dataset/config: immutable private dataset revision
  `350c74f2313e4e73ca65e7ffc6e9dcf74040b3b7`; all 7,500 normal training slices;
  batch size 8, validation batch size 8, Gaussian 1,000-step linear diffusion,
  learning rate `1e-4`, EMA 0.9992, seed 17, final epoch 30 and new milestones
  5/10/20/30. Held-out test is unavailable to the trainer.
- Continuation/provenance: restores model, EMA, optimizer, shard sampler and RNG
  from the source `ddpm_last.pt` after epoch 1 / optimizer step 938. The runner
  allows only `epochs` and `checkpoint_epochs` to differ, uses a new immutable
  run directory, and records the parent checkpoint lineage. Code revision
  `00d199fbe726922caaf6696b02addffe380f0d93` with dirty working tree captured in
  the new source snapshot.
- Completed (UTC): 2026-11-02 07:13:43; 30 epochs / 28,140 optimizer steps.
  Best normal-validation epsilon MSE 0.00735903694 at epoch 30. Total completed
  epoch training/validation time including source epoch 1: 12,523.946 seconds;
  peak allocated VRAM 5,532,909,056 bytes. All four milestone checkpoints exist.
  Source snapshot SHA256
  `9ce78422ec1c46b3836925f1ff11826bd512aff84f22dae95b068dd9f0d344df`.
  [Private artifacts revision `1ee5c499`](https://huggingface.co/dacn-bmad-team/dacn-runs/tree/1ee5c4995623532e7ab51c043839555c1dd0b0db/runs/ddpm-partial-t4-b8-e30-cont-20261102T034644Z).
  [Modal app](https://modal.com/apps/phutungcogioihaitan/main/ap-Gau6wwuNQYP4U3UJBi6rrm).

### ddpm-partial-t4-b8-e1-20261102T032740Z — T4 batch-8 throughput pilot — completed
- Started/completed (UTC): 2026-11-02 03:34:02 / 03:42:09. User-authorized
  fresh Gaussian partial-DDPM run on Tesla T4; it did not resume or overwrite
  the batch-1 epoch-10 run.
- Dataset/config: immutable private revision
  `350c74f2313e4e73ca65e7ffc6e9dcf74040b3b7`; all 7,500 normal training slices;
  all validation slices with only the 39 normal slices used for checkpoint MSE;
  seed 17, one epoch, batch size 8, validation batch size 8, 938 expected
  optimizer steps, learning rate `1e-4`, EMA 0.9992, milestone epoch 1. Held-out
  test was not passed to the trainer.
- Provenance: code revision `00d199fbe726922caaf6696b02addffe380f0d93`
  with dirty working tree captured by source snapshot SHA256
  `0c0e193cf8b89be8120091626d0fa8ba156677bc65b6f5172f2be9d24b68fbbd`.
- Result/resources: epoch-1 train epsilon MSE 0.080065, normal-validation
  epsilon MSE 0.323197 on 39 normal slices, 938 optimizer steps, 412.506
  training/validation seconds and 5,465,677,824 bytes peak allocated VRAM.
  This is throughput/training evidence, not an anomaly-segmentation result.
- Artifacts: `ddpm_last.pt`, `ddpm_best.pt`, `ddpm_epoch_001.pt`, config,
  history, validation metrics/reconstruction and source snapshot were uploaded
  to [private revision `178c1de6`](https://huggingface.co/dacn-bmad-team/dacn-runs/tree/178c1de61cc6d873c8b16b590e19f77dd48050e1/runs/ddpm-partial-t4-b8-e1-20261102T032740Z).
  [Modal app](https://modal.com/apps/phutungcogioihaitan/main/ap-vIFbYSY0oC1V74Tj4DFJvV).

### ddpm-val-e10-t300-posterior-mean-t4-b8-20261101T182333Z — validation ablation — completed
- Started/completed (UTC): 2026-11-01 18:23:53 / 18:31:25. User-authorized
  Modal Tesla T4 evaluation using
  epoch-10 EMA, one-based `t*=300`, posterior-mean reverse, batch size 8,
  sampling base seed 17 and all 83 validation slices. The held-out test path is
  unavailable to the evaluator.
- Provenance: dataset revision
  `350c74f2313e4e73ca65e7ffc6e9dcf74040b3b7`; source training run
  `ddpm-partial-a100-s17-e30-milestones-rngfix-20261031T041542Z`; code revision
  `00d199fbe726922caaf6696b02addffe380f0d93` with a dirty working tree whose
  exact Python source snapshot SHA256 is
  `e64bb1b6efc41067134a9aa274da5f5ab274c5604540277e4dcd75cb0efb5dd6`.
  The private reconstruction cache SHA256 is
  `622a9c01cb2df8572ec4c87465ac768ecbbb3cd0de5327c9480708b792f1ca98`.
- Pixel result: local SSIM achieved AUROC 0.861427, Precision 0.096832,
  Recall 0.254533, F1/Dice 0.140292, AP 0.081721 and macro lesion-slice Dice
  0.207741. Empty-normal pixel FPR decreased to 0.054391, but relative to the
  ancestral baseline AP fell 14.8%, F1 fell 13.2% and Recall fell 35.6%.
  Pixel difference was worse still: AUROC 0.325965, AP 0.040250 and F1
  0.143712. Posterior mean is therefore rejected as a replacement sampler.
- Resources/artifacts: 407.712 seconds for 83 slices (4.912 s/slice), excluding
  2.139 seconds to write the cache; peak allocated VRAM 580,116,480 bytes.
  The manifest records status completed, artifact upload revision
  [`3a91ec15`](https://huggingface.co/dacn-bmad-team/dacn-runs/tree/3a91ec15f28d4c0227d36c177a15373f09ff478b/evaluations/ddpm-val-e10-t300-posterior-mean-t4-b8-20261101T182333Z),
  and held-out test use `false`. [Modal app](https://modal.com/apps/phutungcogioihaitan/main/ap-pr99lUwZxUq4OwhJf2rWXR).

### ddpm-posterior-mean-batch-smoke-e10-t3-cpu-20261101T1822Z — validation smoke — completed
- Started/completed (UTC): 2026-11-01 18:22 / 18:22. Local CPU-only smoke using
  the epoch-10 EMA checkpoint, two class-stratified validation slices,
  one-based `t*=3`, posterior-mean reverse and requested batch size 8.
- Result: both samples completed in one actual batch in 2.780 seconds and the
  private temporary reconstruction cache contained the expected image,
  reconstruction, mask and label arrays. This was a shape/device/cache smoke,
  not a research metric; no GPU or held-out test data was used.

### ddpm-inference-posterior-mean-audit-e10-t300-cpu-20261101T1813Z — validation diagnostic — completed
- Started/completed (UTC): 2026-11-01 18:13 / 18:15. Local CPU-only diagnostic
  using epoch-10 EMA, one-based `t*=300`, sampling base seed 17 and a
  posterior-mean reverse
  chain that omits stochastic noise at each reverse transition.
- Scope: one validation anomaly slice selected by largest lesion area. This is
  an inference-only exploratory ablation, not the registered ancestral sampler;
  no identifier is recorded, no GPU is used, and held-out test is not read.
- Result: reconstruction took 97.573 seconds. Within the input-derived brain
  ROI, local-SSIM AP was 0.4091 versus 0.3586 for the registered ancestral
  seed-17 reconstruction of the same slice, while AUROC decreased from 0.5755
  to 0.5473. Pixel-difference AP and AUROC were 0.2937 and 0.4912. This
  mixed, one-slice result is hypothesis-generating only; it does not justify
  replacing the registered sampler without a full validation ablation and
  validation-selected thresholds.

### ddpm-inference-ensemble-audit-e10-t300-cpu-20261101T1806Z — validation diagnostic — completed
- Started/completed (UTC): 2026-11-01 18:06 / 18:11. Local CPU-only diagnostic
  using epoch-10 EMA, one-based `t*=300`, and sampling base seeds 17, 18 and 19.
- Scope: three reconstructions of one validation anomaly slice selected by
  largest lesion area, plus their mean reconstruction. The diagnostic compares
  lesion-versus-normal-brain score separation and stochastic variation. No
  identifier is recorded, no GPU is used, and held-out test is not read.
- Result: reverse-sampling variation was material (mean per-pixel standard
  deviation 0.0406; 95th percentile 0.2021). For local SSIM within the brain,
  the three-seed mean reconstruction had AP 0.3778 versus 0.3586 for seed 17,
  but AUROC decreased from 0.5755 to 0.5495 and lesion/normal-brain mean-score
  ratio stayed approximately 1.10. The original single-reconstruction threshold
  is not calibrated for the averaged reconstruction. This one-slice result does
  not justify changing the registered validation protocol by itself.

### ddpm-inference-audit-e10-t300-cpu-20261101T1801Z — validation diagnostic — completed
- Started/completed (UTC): 2026-11-01 18:01 / 18:03. Local CPU-only
  reconstruction using the epoch-10 EMA checkpoint, one-based `t*=300`, and
  sampling base seed 17; reconstruction time was 91.262 seconds.
- Scope: one validation anomaly slice selected by largest lesion area solely for
  diagnostic score-region analysis; no sample identifier is recorded. Held-out
  test is not read. The diagnostic separates score behavior over lesion, normal
  brain tissue and background, and checks conservative input-derived brain ROIs.
- Result: for local SSIM at the validation-selected threshold, 2,982 of 3,087
  false-positive pixels were inside the input-derived brain ROI. Masking the
  background changed this slice's F1 only from 0.3476 to 0.3515; eroding the ROI
  by two pixels produced 0.3556 while retaining 99.85% of its lesion pixels.
  Mean local-SSIM anomaly score was 0.7730 in the lesion and 0.7061 in normal
  brain tissue, versus 0.0326 outside the brain. This is a one-slice diagnostic,
  not a validation metric or evidence sufficient to change the protocol.

### ddpm-val-e5-e10-t100-200-300-20261031T074028Z — validation sweep — completed
- Started/completed (UTC): 2026-10-31 07:40:44 / 08:43:45 from the durable
  evaluation manifest. Code revision
  `e39632d1ef8b81d663aae82b8787a42f37ab03ee`; working tree dirty; exact Python
  source snapshot SHA256
  `a4ae2b89a3c6a2254d29accf7f48d341960f9fa9e78dd82f617ea772194ab523`.
- Dataset and split: immutable private revision
  `350c74f2313e4e73ca65e7ffc6e9dcf74040b3b7`; all 83 validation slices (39
  normal, 44 anomalous). The manifest explicitly records held-out test use as
  `false`.
- Config: source checkpoints `ddpm_epoch_005.pt` and `ddpm_epoch_010.pt` from
  run `ddpm-partial-a100-s17-e30-milestones-rngfix-20261031T041542Z`, verified
  by their recorded SHA256 values; starting timesteps 100, 200 and 300;
  sampling base seed 17; pixel difference and local one-minus-SSIM scores.
  Thresholds were selected separately on validation by maximum F1, taking the
  largest threshold on a tie. Image scores are means over the top 576 pixels.
- Validation selection: local one-minus-SSIM at epoch 10, `t*=300` maximized
  the primary pixel AP across the six checkpoint/timestep combinations. Pixel
  AUROC 0.892010, Precision 0.101571, Recall 0.395303, micro F1/Dice 0.161616,
  AP 0.095895, and macro Dice over lesion slices 0.241476. Its image-level
  AUROC was 0.344406, Precision 0.530120, Recall 1.0, F1 0.692913, and AP
  0.454948. These are validation metrics from one inference seed, not test
  results or uncertainty estimates.
- Comparison: the best pixel-difference F1 was 0.165484 at epoch 5, `t*=300`,
  with pixel AP 0.086649 and AUROC 0.831706. The local-SSIM primary candidate
  was selected by pixel AP, not by image F1; all image-F1 values collapsed to
  the all-positive validation operating point and should not drive selection.
- Resource observation: `NVIDIA A100-SXM4-40GB`, batch size 1, 240x240 crop.
  The selected combination took 930.401 seconds total, 11.210 seconds per
  slice, and peak allocated VRAM was 100,651,520 bytes. No warmup-separated
  median or p95 latency was measured.
- [Private aggregate artifacts and manifest at immutable model revision
  `51a9c22ee71bdae7d9d78538fb8664a7b96c9e48`](https://huggingface.co/dacn-bmad-team/dacn-runs/tree/51a9c22ee71bdae7d9d78538fb8664a7b96c9e48/evaluations/ddpm-val-e5-e10-t100-200-300-20261031T074028Z).
  [Modal app](https://modal.com/apps/phutungcogioihaitan/main/ap-CvDDPrqHJRM8Js3jj0BuhS).
- Interpretation: all six requested validation combinations completed and were
  durably uploaded. Freeze the selected validation protocol and obtain more
  seeds before any one-time held-out test evaluation or research claim.

### ddpm-val-smoke-e5-t100-20261031T073859Z — validation evaluator — completed smoke
- Started/completed (UTC): 2026-10-31 07:39:16 / 07:39:49 from the durable
  evaluation manifest. Code revision
  `e39632d1ef8b81d663aae82b8787a42f37ab03ee`; working tree dirty; exact Python
  source snapshot SHA256
  `a4ae2b89a3c6a2254d29accf7f48d341960f9fa9e78dd82f617ea772194ab523`.
- Dataset: immutable private revision
  `350c74f2313e4e73ca65e7ffc6e9dcf74040b3b7`; validation split only. Held-out
  test use is explicitly `false` in the manifest.
- Config: epoch-5 checkpoint SHA256
  `8c1d957265858c70879f46273691be7d88e93c4a562d50168c2991716aed682f`;
  `t*=100`; sampling base seed 17; deterministic stratified limit of four
  validation slices (two normal, two anomalous).
- Diagnostic output: `NVIDIA A100-SXM4-40GB`; 10.897 seconds total, 2.724 seconds
  per slice, peak allocated VRAM 100,651,520 bytes. Pixel AP was 0.075746 for
  pixel difference and 0.106161 for local one-minus-SSIM. These four-slice
  metrics validate the pipeline only and are not research results.
- [Private aggregate artifacts at model revision
  `5e77e82dddfc423af2bf179282441978ee08d6f3`](https://huggingface.co/dacn-bmad-team/dacn-runs/tree/5e77e82dddfc423af2bf179282441978ee08d6f3/evaluations/ddpm-val-smoke-e5-t100-20261031T073859Z).
  [Modal app](https://modal.com/apps/phutungcogioihaitan/main/ap-wDhW800vAY4ZQv39mu6LW9).

### ddpm-partial-a100-s17-e30-milestones-rngfix-20261031T041542Z — partial DDPM — interrupted, resumable
- Started/interrupted (UTC): 2026-10-31 04:16:03 / 07:07:05. The start time is
  from the durable manifest; the stop time is from the Modal app record.
- Issue/PR: fresh run after correcting CUDA RNG-state restoration and replacing
  the client-bound controller with direct detached remote invocation; no issue or
  PR was provided. Code revision `e39632d1ef8b81d663aae82b8787a42f37ab03ee`;
  working tree dirty; exact Python source snapshot SHA256
  `5f04c494f975543e8146aa350e691bc33e331e197b5a95131d2b549d9ec1be75`.
- Dataset: private `dacn-bmad-team/bmad-flair-patient-disjoint-v1` at immutable
  revision `350c74f2313e4e73ca65e7ffc6e9dcf74040b3b7`; `patient-disjoint-v1`.
  The trainer receives train and validation only; held-out test is not passed.
- Config: normal-only partial DDPM; seed 17; 30 epochs; batch 1; all 7,500 normal
  training slices; 39 normal validation slices for checkpoint selection;
  Gaussian noise; `T=1000`; linear beta schedule; epsilon-prediction L2; AdamW
  `lr=1e-4`; EMA 0.9999; maximum 225,000 optimizer steps. Full-state milestone
  files are retained at epochs 5, 10, 20 and 30 in addition to `ddpm_best.pt`
  and `ddpm_last.pt`.
- Checkpoint/result status: the user intentionally stopped the app during epoch
  11 at step 1,800/7,500. Epoch 10 completed with 75,000 optimizer steps. The
  Volume retains `ddpm_epoch_005.pt`, `ddpm_epoch_010.pt`, `ddpm_last.pt`, and
  `ddpm_best.pt`; the validation-selected best is epoch 10 with normal-validation
  epsilon-MSE 0.0101424818. The last durable manifest remains at status
  `running`, stage `training`, but the Modal app record confirms the
  `NVIDIA A100-SXM4-40GB` job stopped. The run is resumable from epoch 11 with
  the exact source snapshot and config. Zero full seeds completed; no anomaly
  metric or held-out test evaluation exists.
- On 2026-11-01, the epoch-10 checkpoint and safe provenance files were synced
  from the Modal Volume to the private model repo at immutable revision
  `07ec8b455a3318a267df5568f7af3fa4f15d7a3c`. The checkpoint SHA256 is
  `b828b5889d4164c0d9cebbbe0bf7afb603f5bec9ad7c84508baa222a26bc4fc9`, matching
  the Modal manifest and local copy. This artifact sync used a CPU-only helper;
  it did not resume training or perform a new evaluation.
- [Private epoch-10 checkpoint and provenance at model revision
  `07ec8b45`](https://huggingface.co/dacn-bmad-team/dacn-runs/tree/07ec8b455a3318a267df5568f7af3fa4f15d7a3c/runs/ddpm-partial-a100-s17-e30-milestones-rngfix-20261031T041542Z).
- [Modal app](https://modal.com/apps/phutungcogioihaitan/main/ap-s5aqwTTko93RFtx7RHHocU).

### ddpm-partial-a100-s17-e30-milestones-20261030T165258Z — partial DDPM — interrupted; resume attempt failed
- Started/interrupted (UTC): 2026-10-30 17:12:59 / 18:01:07. The start time is
  from the durable manifest; the interruption time is from the Modal client log.
- Issue/PR: user-requested epoch 5, 10, 20 and 30 checkpoint retention plus the
  validation-selected best checkpoint; no issue or PR was provided. Code revision
  `e39632d1ef8b81d663aae82b8787a42f37ab03ee`; working tree dirty; exact Python
  source snapshot SHA256
  `a40f23113b3a954771c7c623dab147c29944f5e1750ca5cfcc6adcc09d5c5364`.
- Dataset: private `dacn-bmad-team/bmad-flair-patient-disjoint-v1` at immutable
  revision `350c74f2313e4e73ca65e7ffc6e9dcf74040b3b7`; `patient-disjoint-v1`.
  The trainer receives train and validation only; held-out test is not passed.
- Config: normal-only partial DDPM; seed 17; 30 epochs; batch 1; all 7,500 normal
  training slices; 39 normal validation slices for checkpoint selection;
  Gaussian noise; `T=1000`; linear beta schedule; epsilon-prediction L2; AdamW
  `lr=1e-4`; EMA 0.9999; maximum 225,000 optimizer steps. Full-state milestone
  files are retained at epochs 5, 10, 20 and 30 in addition to `ddpm_best.pt`
  and `ddpm_last.pt`.
- Checkpoint/result status: a Modal client cancellation stopped the function in
  epoch 5 at step 2,400/7,500. Epoch 4 completed with 30,000 optimizer steps;
  the Volume retains `ddpm_last.pt` and `ddpm_best.pt`, both at epoch 4. The best
  normal-validation epsilon-MSE is 0.0138888134. The first requested milestone
  is epoch 5, so no milestone file exists yet. The durable manifest remains at
  status `running`, stage `training`, but Modal has no active app/task. The run
  was eligible to resume from epoch 5 with the same source snapshot and config.
  A direct detached resume attempt passed provenance checks but failed before
  epoch 5 training because CUDA RNG tensors loaded onto the GPU were passed to
  an API requiring CPU ByteTensors (`TypeError`); the epoch-4 checkpoints were
  not overwritten. Zero full seeds completed; no anomaly metric or held-out test
  evaluation exists.
- [Modal app](https://modal.com/apps/phutungcogioihaitan/main/ap-gPqtekWaMpUbER5eHFc9CL).

### ddpm-partial-a100-s17-e30-20261030T150704Z — partial DDPM — interrupted, resumable
- Started/interrupted (UTC): 2026-10-30 15:07:21 / 15:29:27. The start time is
  from the durable Modal manifest; the stop time is from the Modal app record.
- Issue/PR: user-authorized 30-epoch replacement run; no issue or PR was
  provided. Code revision `e39632d1ef8b81d663aae82b8787a42f37ab03ee`;
  working tree dirty; exact Python source snapshot SHA256
  `b0e67ccbb051a24b81fca6a18805a49c748235d60ce96bcc6d6632f60d6d6e10`.
- Dataset: private `dacn-bmad-team/bmad-flair-patient-disjoint-v1` at immutable
  revision `350c74f2313e4e73ca65e7ffc6e9dcf74040b3b7`; `patient-disjoint-v1`.
  The trainer receives train and validation only; held-out test is not passed.
- Config: normal-only partial DDPM; seed 17; 30 epochs; batch 1; all 7,500 normal
  training slices; 39 normal validation slices for checkpoint selection;
  Gaussian noise; `T=1000`; linear beta schedule; epsilon-prediction L2; AdamW
  `lr=1e-4`; EMA 0.9999; maximum 225,000 optimizer steps; 10-hour function
  timeout.
- Checkpoint/result status: the user intentionally stopped the app during epoch
  3 at step 2,600/7,500. Epochs 1 and 2 completed (15,000 optimizer steps), and
  the Volume retains both `ddpm_last.pt` and `ddpm_best.pt`; both currently
  represent epoch 2, which is also the validation-selected best epoch with
  normal-validation epsilon-MSE 0.0652383983. The last durable manifest remains
  at status `running`, stage `training`, but the Modal app record confirms the
  `NVIDIA A100 80GB PCIe` job stopped. The run is resumable from epoch 3 with the
  same run ID, config and source snapshot. Zero full seeds completed; no anomaly
  metric or held-out test evaluation exists.
- [Modal app](https://modal.com/apps/phutungcogioihaitan/main/ap-vINvpDwCb8El3JTFjeC5el).

### ddpm-partial-a100-s17-e40-retry-20261030T145811Z — partial DDPM — interrupted
- Started/interrupted (UTC): 2026-10-30 14:58:33 / 15:02:43. The start time is
  from the durable Modal manifest; the stop time is from the Modal app record.
- Issue/PR: fresh retry after the preceding attempt was canceled before its first
  completed epoch; no issue or PR was provided. Code revision
  `e39632d1ef8b81d663aae82b8787a42f37ab03ee`; working tree dirty; exact Python
  source snapshot SHA256
  `cc8f607f274ba4d57d411ff08aaec7fb55637665a5f32972426de7e359d644b1`.
- Dataset: private `dacn-bmad-team/bmad-flair-patient-disjoint-v1` at immutable
  revision `350c74f2313e4e73ca65e7ffc6e9dcf74040b3b7`; `patient-disjoint-v1`.
  The trainer receives train and validation only; held-out test is not passed.
- Config: normal-only partial DDPM; seed 17; 40 epochs; batch 1; all 7,500 normal
  training slices; 39 normal validation slices for checkpoint selection;
  Gaussian noise; `T=1000`; linear beta schedule; epsilon-prediction L2; AdamW
  `lr=1e-4`; EMA 0.9999; maximum 300,000 optimizer steps.
- Checkpoint/result status: the job was intentionally stopped when the requested
  budget changed from 40 to 30 epochs. It reached epoch 1, step 2,300/7,500 with
  running mean epsilon-MSE 0.044026, before any completed epoch or checkpoint.
  Zero seeds completed; no anomaly metric or held-out test evaluation exists.
  The last durable manifest remains at status `running`, stage `training`, while
  the Modal app record confirms that the `NVIDIA A100-SXM4-40GB` job stopped.
- [Modal app](https://modal.com/apps/phutungcogioihaitan/main/ap-Ifrh5L4yHKikumhUHDPkCU).

### ddpm-partial-a100-s17-e40-20261030T144735Z — partial DDPM — interrupted
- Started/interrupted (UTC): 2026-10-30 14:48:46 / 14:56:29. The start time is
  from the durable manifest; the interruption time is from the Modal client log.
- Issue/PR: user-authorized full-data A100 training; no issue or PR was provided.
  Code revision `e39632d1ef8b81d663aae82b8787a42f37ab03ee`; the working tree was
  dirty and exact Python sources were captured in snapshot
  `cc8f607f274ba4d57d411ff08aaec7fb55637665a5f32972426de7e359d644b1`.
- Dataset: private `dacn-bmad-team/bmad-flair-patient-disjoint-v1` at immutable
  revision `350c74f2313e4e73ca65e7ffc6e9dcf74040b3b7`; split policy
  `patient-disjoint-v1`. The trainer receives train and validation only; held-out
  test is not passed to it.
- Training policy/config: normal-only partial DDPM; seed 17; 40 epochs; batch 1;
  all 7,500 normal training slices; all validation slices selected, with only 39
  normal slices used for fixed-noise epsilon-MSE checkpoint selection; Gaussian
  noise; `T=1000`; linear beta schedule; epsilon-prediction L2; AdamW
  `lr=1e-4`; EMA 0.9999. Expected maximum is 300,000 optimizer steps.
- Checkpoints/results: the client cancellation occurred before epoch 1 completed,
  at step 4,700/7,500 with running mean epsilon-MSE 0.027439. The Volume contains
  only `run_manifest.json` and `source_snapshot.zip`; no `ddpm_best.pt` or
  `ddpm_last.pt` exists, so this run cannot resume. Zero seeds completed. No
  anomaly threshold, score/timestep sweep, ROC-AUC, Precision, Recall, F1, AP,
  Dice or test metric was produced.
- Resource/status: requested and confirmed GPU `NVIDIA A100-SXM4-40GB`. The
  abrupt Modal cancellation left the last durable manifest at status `running`,
  stage `training`; the absence of post-cancellation logs and checkpoints is the
  evidence for interruption. This is not a completed training result.
- [Modal app](https://modal.com/apps/phutungcogioihaitan/main/ap-8EIIjwT7QoCK0YnNJl7OWD).
  No private Hub artifact upload/model revision was produced.

### ddpm-partial-smoke-20261030T044737Z-79f605 — partial DDPM — completed smoke
- Started/completed (UTC): 2026-10-30 04:48:42 / 04:49:54, manifest times;
  upload completed afterward according to the Modal CLI result.
- Issue: DAC-10/DAC-14. Code `a90b884`; working tree dirty, with exact sources
  captured by the runner. Dataset `dacn-bmad-team/bmad-flair-patient-disjoint-v1`
  at `350c74f2313e4e73ca65e7ffc6e9dcf74040b3b7`.
- Config: seed 17, one epoch, batch 1, 128 normal training slices and 16
  validation slices, four validation reconstructions at t*=100. Checkpoint
  selection used epsilon-MSE on 8 normal validation slices. Held-out test was
  not passed to the trainer. One smoke seed completed; no anomaly thresholds,
  ROC-AUC, Precision, Recall, F1, AP, Dice or test metrics were produced.
- Resource observation: Tesla T4; 128 optimizer steps; 18.367 seconds for epoch
  training plus normal validation, excluding checkpoint writes and preview;
  peak allocated VRAM 799,354,880 bytes. End-to-end latency per image and credit
  consumption were not measured. Normal-validation epsilon-MSE was
  0.9969820901751518; this is a training diagnostic, not anomaly performance.
- [Private artifact folder and manifest at immutable model revision
  `62c0f78c5f49d5baaa52b0204f90033892f0fd6e`](https://huggingface.co/dacn-bmad-team/dacn-runs/tree/62c0f78c5f49d5baaa52b0204f90033892f0fd6e/runs/ddpm-partial-smoke-20261030T044737Z-79f605):
  `ddpm_best.pt`, `ddpm_last.pt`, private validation reconstructions, history,
  metrics and source snapshot. [Modal app](https://modal.com/apps/tanh1c/main/ap-19UeEpUDpeaT9JlvN4TdX4).
- This sample-limited run verifies the pipeline, not model performance. The
  existing local notebook controls AnoDDPM and did not execute this DDPM run.
- DAC-8 review completion and B/C data validation were reported by the project
  lead; individual DAC-9 validator outputs have not yet been attached to Linear.

### anoddpm-local-smoke-20261029T055620Z-b71e — AnoDDPM-style BMAD adaptation — completed diagnostic smoke
- Started/completed (UTC): Modal app initialized at approximately 05:56 UTC;
  trainer started 2026-10-29 05:57:01 UTC and completed 05:57:45 UTC (manifest).
- Issue/PR: None linked; local notebook flow validation on branch
  feat/anoddpm-notebook-training.
- Code commit / dirty working tree: 82b4d2c01f4ea9f62c283f75eca17f114f562d3e;
  working tree was dirty for the notebook and documentation work. Source
  snapshot and manifest are in the run artifacts.
- Dataset repo and immutable revision:
  dacn-bmad-team/bmad-flair-patient-disjoint-v1 @
  350c74f2313e4e73ca65e7ffc6e9dcf74040b3b7.
- Split and patient-disjoint protocol: patient-disjoint-v1; diagnostic pipeline
  smoke; the held-out test split was not passed to the trainer.
- Training data policy (normal-only or other): normal-only; capped at 128
  training samples and 16 validation samples.
- Model / score formula / score direction: AnoDDPM-style U-Net prototype;
  simplex forward noise and squared reconstruction error; higher scores are
  intended to mean more anomalous. This is not the agreed primary partial-DDPM
  method.
- Training and scoring config: seed 17, one epoch, batch size 1, base channels
  32, maximum training timestep 100, 25 validation reconstruction steps, max
  train 128, max validation 16.
- Seed(s) attempted / completed: 1 requested / 1 completed; 128 optimizer steps.
- Threshold source and selection rule (validation only): image and pixel
  thresholds independently selected to maximize F1 on the 16-slice validation
  subset (0.08077009 image; 0.00685009 pixel). Do not reuse these smoke
  thresholds in the final protocol.
- Validation metrics (level + aggregation): one seed, 16 stratified validation
  slices. Image-level AUROC 0.6250, Precision 0.7500, Recall 0.7500, F1
  0.7500, AP 0.6747. Pixel-level AUROC 0.6439, micro Precision 0.0261, micro
  Recall 0.4258, micro F1 0.0492, AP 0.0254, macro Dice over anomalous slices
  0.0899. Higher scores indicate more anomalous.
- Test metrics (level + aggregation; only after freeze): N/A; no test evaluation
  was performed.
- Latency (device, batch, precision, resolution, warmup, sample count;
  median/p95): per-image latency was not measured; trainer-reported training
  time was 40.68 seconds on a Tesla T4.
- Peak VRAM: 799,354,880 bytes (manifest).
- Checkpoint, plots, and run_manifest.json links: [private run artifacts and
  manifest at model revision
  a95b40397aae6f11a7ee863aefbb2613f64a1f47](https://huggingface.co/dacn-bmad-team/dacn-runs/tree/a95b40397aae6f11a7ee863aefbb2613f64a1f47/runs/anoddpm-local-smoke-20261029T055620Z-b71e);
  checkpoint anoddpm_best.pt. [Modal app log](https://modal.com/apps/tanh1c/main/ap-649PQBlwoANkm7AtawNzmX).
  Aggregate manifest, history, validation metrics, and plot are also saved
  locally under runs/modal_notebook/anoddpm-local-smoke-20261029T055620Z-b71e/
  and displayed in notebooks/anoddpm_modal_local.ipynb.
- Failure, caveat, or interpretation: The actual Modal training and Hub upload
  succeeded. The first notebook launch then failed while parsing the
  human-oriented Modal progress stream, after training had completed. The
  completed artifacts were re-read by the notebook and rendered locally; a
  CPU-only Modal CLI probe verified the corrected write-result channel. This
  sample-limited AnoDDPM smoke is a workflow check, not a research result or the
  approved primary DDPM experiment. No test data was evaluated.

### anoddpm-smoke-cardfix-20261029T0524Z — AnoDDPM-style BMAD adaptation — completed diagnostic smoke
- Started/completed (UTC): Modal app initialized 2026-10-29 05:24 UTC; training started 05:26:14 UTC and completed 05:27:14 UTC (manifest timestamps).
- Issue/PR: W1 context [DAC-8](https://linear.app/dacn-khmt/issue/DAC-8/r01-chot-hop-djong-ddpm-partial-reconstruction-va-score); branch `feat/anoddpm-notebook-training` is not linked to an issue/PR.
- Code commit / dirty working tree: `36a56d4`; dirty for previously reviewed Modal smoke fixes and the HF dataset revision/card update.
- Dataset repo and immutable revision: `dacn-bmad-team/bmad-flair-patient-disjoint-v1` @ `350c74f2313e4e73ca65e7ffc6e9dcf74040b3b7`.
- Split and patient-disjoint protocol: `patient-disjoint-v1`; pipeline smoke only; held-out test path is not passed to the trainer.
- Training data policy (normal-only or other): normal-only; capped at 128 training samples and 16 validation samples.
- Model / score formula / score direction: AnoDDPM-style U-Net; simplex forward noise; squared reconstruction error; higher scores intended to mean more anomalous.
- Training and scoring config: seed 17, one epoch, batch size 1, base channels 32, maximum training timestep 100, 25 validation reconstruction steps, max train 128, max validation 16.
- Seed(s) attempted / completed: 1 requested / 1 completed; 128 optimizer steps.
- Threshold source and selection rule (validation only): image-level and pixel-level thresholds independently maximize F1 on the 16-slice validation subset (0.08077009 image; 0.00685009 pixel). Do not carry these smoke thresholds into a final protocol.
- Validation metrics (level + aggregation): 16 stratified validation slices, one seed. Image-level AUROC 0.6250, Precision 0.7500, Recall 0.7500, F1 0.7500, AP 0.6747. Pixel-level AUROC 0.6439, micro Precision 0.0261, micro Recall 0.4258, micro F1 0.0492, AP 0.0254, and macro Dice over anomalous slices 0.0899. Higher scores indicate more anomalous.
- Test metrics (level + aggregation; only after freeze): N/A; no test evaluation is planned for this smoke.
- Latency (device, batch, precision, resolution, warmup, sample count; median/p95): no per-image inference latency measured; total trainer-reported time 55.68 seconds on Tesla T4.
- Peak VRAM: 799,354,880 bytes (manifest).
- Checkpoint, plots, and run_manifest.json links: [private run artifacts and manifest at model revision `7bb2d87a3a88aff1c3f43d144f0e83276073194f`](https://huggingface.co/dacn-bmad-team/dacn-runs/tree/7bb2d87a3a88aff1c3f43d144f0e83276073194f/runs/anoddpm-smoke-cardfix-20261029T0524Z); checkpoint `anoddpm_best.pt`. [Modal app log](https://modal.com/apps/tanh1c/main/ap-ToI5KSTaTRoukNCGdzuSYK).
- Failure, caveat, or interpretation: The new pinned revision fetched all 95 files and passed `validate_hf_export`, then completed 128 training steps and validation reconstruction. The first local CLI launch failed before remote execution because `dacn` was not importable; rerunning with `PYTHONPATH=src` succeeded. This is a sample-limited AnoDDPM pipeline smoke, not the agreed primary DDPM method or a research result. No test data was evaluated.

### anoddpm-smoke-20261029T0518Z — AnoDDPM-style BMAD adaptation — failed before training
- Started/completed (UTC): Modal app initialized 2026-10-29 05:07:48 UTC; remote function failed 05:11:43 UTC.
- Issue/PR: W1 context [DAC-8](https://linear.app/dacn-khmt/issue/DAC-8/r01-chot-hop-djong-ddpm-partial-reconstruction-va-score); branch `feat/anoddpm-notebook-training` is not linked to an issue/PR.
- Code commit / dirty working tree: `36a56d4`; dirty for the local Modal Secret name and smoke-doc corrections in `.env.example`, `docs/HUGGINGFACE_MODAL.md`, and `src/dacn/cloud/modal_train.py`.
- Dataset repo and immutable revision: `dacn-bmad-team/bmad-flair-patient-disjoint-v1` @ `fbb884dfe144be5c71e205a73701507163557ea4`.
- Split and patient-disjoint protocol: `patient-disjoint-v1`; Modal fetched the pinned export and ran its validator; trainer was not called and no test path was passed to it.
- Training data policy (normal-only or other): normal-only requested; 128 training samples were configured, 0 were consumed.
- Model / score formula / score direction: AnoDDPM-style U-Net; simplex forward noise; squared reconstruction error; higher scores intended to mean more anomalous. Scoring was not reached.
- Training and scoring config: seed 17, one epoch, batch size 1, maximum training timestep 100, 25 reconstruction steps, at most 128 train and 16 stratified validation samples.
- Seed(s) attempted / completed: 1 requested / 0 completed; optimizer steps 0.
- Threshold source and selection rule (validation only): configured for validation only; threshold selection was not reached.
- Validation metrics (level + aggregation): N/A; trainer did not start.
- Test metrics (level + aggregation; only after freeze): N/A; no test evaluation was performed.
- Latency (device, batch, precision, resolution, warmup, sample count; median/p95): N/A; no model inference was performed.
- Peak VRAM: N/A; no model training was performed.
- Checkpoint, plots, and run_manifest.json links: none; no artifacts were written. [Modal app log](https://modal.com/apps/tanh1c/main/ap-Wx23GtpHBzeu1x601IkFOl).
- Failure, caveat, or interpretation: `validate_hf_export` failed after fetching all 95 files with `Hub export README differs from the generated privacy-safe dataset card`; last completed stage was dataset snapshot validation. No checkpoint, metrics, run manifest, or Hugging Face model upload was produced. This is a pipeline smoke failure, not a model result.

```text
### <run_id> — <method> — <status>
- Started/completed (UTC):
- Issue/PR:
- Code commit / dirty working tree:
- Dataset repo and immutable revision:
- Split and patient-disjoint protocol:
- Training data policy (normal-only or other):
- Model / score formula / score direction:
- Training and scoring config:
- Seed(s) attempted / completed:
- Threshold source and selection rule (validation only):
- Validation metrics (level + aggregation): ROC-AUC / Precision / Recall / F1 / AP / Dice
- Test metrics (level + aggregation; only after freeze): ROC-AUC / Precision / Recall / F1 / AP / Dice
- Latency (device, batch, precision, resolution, warmup, sample count; median/p95):
- Peak VRAM:
- Checkpoint, plots, and run_manifest.json links:
- Failure, caveat, or interpretation:
```

### anoddpm-a100-smoke-20261029-01 — AnoDDPM — failed before training

- Started/completed (UTC): 2026-10-29; exact times were not captured.
- Issue/PR: Not provided.
- Code commit / dirty working tree: `36a56d4`; dirty because the Modal GPU was
  changed from T4 to A100 in `modal_train.py`.
- Dataset repo and immutable revision:
  `dacn-bmad-team/bmad-flair-patient-disjoint-v1` at
  `fbb884dfe144be5c71e205a73701507163557ea4`.
- Split and patient-disjoint protocol: `patient-disjoint-v1`; validation stopped
  before any split was passed to the trainer.
- Training data policy (normal-only or other): Intended normal-only training;
  training did not start.
- Model / score formula / score direction: Intended AnoDDPM partial
  reconstruction with higher scores meaning more anomalous; scoring did not run.
- Training and scoring config: seed 17, 1 epoch, batch size 1, base channels 32,
  maximum training timestep 100, 25 validation reconstruction steps, at most 128
  training samples and 16 validation samples.
- Seed(s) attempted / completed: 17 attempted / 0 completed.
- Threshold source and selection rule (validation only): Not reached.
- Validation metrics (level + aggregation): N/A; training and validation did not run.
- Test metrics (level + aggregation; only after freeze): N/A; test was not used.
- Latency (device, batch, precision, resolution, warmup, sample count; median/p95):
  N/A; training did not start.
- Peak VRAM: N/A; not captured before failure.
- Checkpoint, plots, and run_manifest.json links: No artifacts or run manifest;
  [Modal app](https://modal.com/apps/phutungcogioihaitan/main/ap-DQeLi3yfypZyKTnIcfRgKq).
- Failure, caveat, or interpretation: `ValueError` from `validate_hf_export` after
  download and before the trainer call. The pinned schema-v1 export contains the
  original generated privacy-safe README, while commit `788ab72` expanded the
  schema-v1 card template; exact comparison against only the new template caused
  the compatibility failure.

### b-cached-reproduction-20261105 — CPU evaluation — completed; reviewed

- Actual CPU evaluations completed in preceding chat execution on 2026-11-05;
  this entry was added retrospectively during review. Exact start/end/runtime were
  not captured; do not infer latency from file timestamps. No new model seed,
  inference, training, GPU allocation or held-out test evaluation.
- Evaluator source `84088f9a1eb158dcafe472578179147f037db3e6`; same project
  score/metric implementation as C. This reproduces the artifact-to-evaluator
  path; it is not an independent mathematical implementation of the metrics.
- Dataset `350c74f2313e4e73ca65e7ffc6e9dcf74040b3b7`; model repository
  `dacn-bmad-team/dacn-runs` revision `1788b1232aa19b3c9ae96aa95a74e2975f4b592a`.
  Two full83 caches: Gaussian batch8 epoch30 and Simplex batch8 epoch20,
  t300/ancestral/EMA/sampling17,39 normal/44 abnormal, crop240.
- Input/mask/label equality against checked validation export was required before
  attaching UUIDs to separate private cache copies. Original caches preserved.
  Review rechecked source-cache bytes against manifest SHA and output UUID-cache
  bytes against evaluator receipt. Ground-truth arrays/UUIDs remain private.
- Review compares both score result dictionaries recursively to checked HF JSON:
  34 numeric fields per model, maximum absolute difference0; other fields match.
  Evidence: `evidence/2026-11-05-b-reproduction-review.json`; corrected aggregate
  outputs `evidence/2026-11-05-b-gaussian30-reproduction.json` and
  `evidence/2026-11-05-b-simplex20-reproduction.json`.
- Provenance correction: original local sidecars incorrectly declared repository
  commit7134b10 as producer code. Corrected copies use manifest code00d199f plus
  per-run dirty source snapshot SHA and separately identify CPU evaluator84088f9.
  Preserve old local files as historical records; use corrected evidence for reporting.

| Model / score | Pixel AP | Pixel AUROC | Precision | Recall | Micro F1/Dice | Normal pixel FPR |
|---|---:|---:|---:|---:|---:|---:|
| Gaussian30 / SSIM | 0.100741 | 0.895647 | 0.107193 | 0.382770 | 0.167483 | 0.071126 |
| Simplex20 / SSIM | 0.074791 | 0.743631 | 0.096629 | 0.599044 | 0.166414 | 0.131908 |

- Both scores for both models use independently fitted full83 validation-max-F1
  image/pixel thresholds. These are descriptive in-sample validation results.
  Image precision44/83 and recall1 imply TP44/FP39/TN0/FN0; specificity0 and
  imageF1 0.692913. This confusion is inferred from saved exact metrics/cohort,
  not from a new per-sample scoring run. Full outputs include both scores.
- Remaining: reusable guarded adapter, grouped calibration/holdout comparison,
  patient-group uncertainty and matched-budget Gaussian20 inference. Different
  budgets here preclude attributing score differences solely to noise family.

### b-spatial-g30-s20-20261105 — completed, CPU validation analysis

Both converted caches have 83 validation rows (39 normal, 44 abnormal), EMA,
ancestral t300 and seed 17. Dataset/model revisions, original checkpoint and
source-cache hashes are in the reproduction and spatial evidence files.
Both score dictionaries reproduce the prior baselines. FG-v1 uses input-only
foreground, the full 240 x 240 metric domain and independent validation max-F1
thresholds. It retains all 93,756 annotated lesion pixels.

| Model / score | AP baseline → FG-v1 | Pixel F1 baseline → FG-v1 | Baseline FP inside / outside FG |
|---|---:|---:|---:|
| Gaussian30 / difference | .099834 → .107474 | .182870 → .189722 | 267733 / 14965 |
| Gaussian30 / SSIM | .100741 → .103010 | .167483 → .169504 | 294303 / 4599 |
| Simplex20 / difference | .034863 → .127957 | .073428 → .183186 | 220116 / 513269 |
| Simplex20 / SSIM | .074791 → .094472 | .166414 → .172128 | 502832 / 22238 |

Full ROC-AUC, precision, recall, F1 and confusion/lesion strata are in
[evidence Gaussian30](evidence/2026-11-05-b-gaussian30-spatial.json) and
[evidence Simplex20](evidence/2026-11-05-b-simplex20-spatial.json).
Thresholds were fitted and evaluated on the same validation cohort. Runtime
was not captured. GPU and VRAM are inapplicable to these CPU computations.

### b-grouped-g30-s20-20261105 — completed, CPU validation analysis

Same source caches, one trained seed per model. First fold of the registered
patient-group split with seed 17. Calibration 40 rows from 25 groups and
holdout 43 rows from 25 disjoint groups. CDFs fit only on normal calibration
pixels. Thresholds and candidate choices fit only on calibration. The selected
Gaussian CDF had calibration AP .101266 and holdout AP .102400. The selected
fusion-only candidate (75% Gaussian CDF) had calibration AP .099565 and holdout
AP .117241. Raw Gaussian, raw Simplex and brightness control holdout AP were
.102402, .087416 and .389506. Brightness was excluded from model selection.
All pixel/image metrics, normal FPR, thresholds and provenance hashes are in
[grouped evidence](evidence/2026-11-05-b-grouped-g30-s20.json).
CPU elapsed 25.314 s. The validation set had already informed earlier analysis
and the checkpoints have different optimizer budgets. Held-out test remains sealed.

### b-group-bootstrap-g30-s20-20261105 — CPU uncertainty analysis

- Attempt1 failed before resampling: private UUID conversion used canonical UUIDs,
  while the checked CSV used a different textual case. Exact string comparison
  rejected the input. No metrics produced. Preserved local failed manifest.
- Attempt2 completed after canonicalizing CSV UUIDs. All83 UUIDs and their order
  match both byte-verified UUID artifacts, whose reconstructions exactly match
  the original checksum-pinned caches. Same registered40/43 patient-group split;
  holdout43/25groups, one train/sampling seed17, t300, full240crop/localSSIM.
- 1000 paired patient-group draws with replacement, seed17. Valid999;
  skipped1 missing-class draw for every candidate. Calibration normalizers and
  thresholds remain fixed; no candidate reselection. Percentile95% intervals.
  CPU28.394s including map computation/bootstrap; GPU/VRAM inapplicable.

| Candidate | Pixel AP [95% CI] | Pixel micro Dice [95% CI] | Delta AP CI versus Gaussian CDF |
|---|---|---|---|
| Gaussian CDF | .102400 [.036214,.148956] | .154653 [.067270,.208717] | reference |
| Simplex raw | .087416 [.028685,.147591] | .168550 [.063285,.259585] | [-.032808,.007927] |
| Fusion75% Gaussian | .117241 [.043010,.173240] | .166372 [.072881,.222934] | [.005152,.025822] |
| Brightness control | .389506 [.103778,.583293] | .268051 [.098454,.444018] | [.044397,.466302] |

- Aggregate evidence: [bootstrap JSON](evidence/2026-11-05-b-group-bootstrap-g30-s20.json),
  including source-cache/index/frozen-selection hashes and exact thresholds.
  Point AP/Dice reproduce the prior grouped output to1e-12 tolerance. Required
  ROC-AUC/Precision/Recall/F1 remain in the linked grouped evidence above;
  this run estimates uncertainty for AP/Dice only.
- Source: `dacn.evaluation.group_bootstrap`, driver `scripts/bootstrap_grouped_cache.py`
  on branch codex/bc-evaluation-colab following5361635. Gaussian30/Simplex20,
  pinned dataset/model revisions as above. Private inputs/identifiers not published.
- These conditional intervals omit model-training variability, selection bias
  from previously inspected validation and multiple comparisons. Fusion does not
  replace the calibration-selected Gaussian CDF based on these holdout findings.
  Matched-budget Gaussian20 inference still pending. No training or sealed test.

### Gaussian20 Colab preparation — 2026-11-05, inference not started

- Uploaded the output-free notebook5361635 to the user's private Google Colab;
  T4 runtime connected. Initial unauthenticated Git clone failed(exit128).
  Source initialization subsequently succeeded using a user-authorized private
  archive of source7a0f024, SHA256
  `16c9936c81fd4bb63286bc96dd591c42b104e14eae3986ec1ea372ef98086b66`.
- Temporary initialization edits also encountered SyntaxError and a canceled
  upload/StopIteration before the successful source installation. No model
  inference, checkpoint loading or metrics occurred in those initialization cells.
- Preparation is waiting for the user to enter their HF read credential directly
  into masked getpass. Token creation instructions supplied. No token stored in
  repository or chat. Do not count this as completed smoke/full83 evaluation.
- Subsequent authentication returned401 at whoami (invalid user token), before
  checkpoint/data download. User instructed to enter a valid read token directly
  in getpass and rerun that cell. No actual inference started.

### gaussian20-colab-20261105 — completed; artifact verified 2026-11-06

- User re-entered HF credential; checkpoint/dataset verification succeeded.
  Colab notebook5361635 with SHA-verified archive source7a0f024, T4 GPU.
  Model revision1788b1232aa19b3c9ae96aa95a74e2975f4b592a; dataset revision
  350c74f2313e4e73ca65e7ffc6e9dcf74040b3b7. Gaussian epoch20 checkpoint SHA
  b4fc177db493fce2143a6412161068c7af73a25e8ae7ef9d88c117863c7e37f1.
- Smoke4/4 completed at t3; notebook elapsed9.633s includes checkpoint/backend
  initialization, not pure reconstruction latency. Model epoch20/18,760steps/
  train seed17 checked against checkpoint state.
- Full83 validation inference started around15:37 Asia/Saigon,2026-11-05,
  ancestral t300, EMA, batch1, UUID-derived sampling base17. No training/test.
  Scope: same-budget comparison against Simplex20, not new checkpoint selection.
- Status: completed. Private UUID cache receipt:83samples, SHA256
  `e9138335394d86278badea560bbdbff2a98d67fa9328ec5680c2d2480c25d7df`.
  User downloaded bundle/checkpoint; moved to ignored runs/checkpoints folders
  and verified actual bytes. No weights, arrays or UUIDs committed.
- Full elapsed1049.814489406s includes reconstruction loop, dataset checks and
  compressed artifact writing, not isolated CUDA model latency. Manifest smoke
  timing5.237s differs from notebook wall9.633s. Peak allocated CUDA110219776bytes
  is since initialization across smoke/full, not total GPU reserved memory.
  Packages:torch2.11.0+cu130,numpy2.4.6,scipy1.16.3,sklearn1.6.1.
- Manifest and descriptive full83 spatial results:
  [manifest](evidence/2026-11-05-b-gaussian20-colab-manifest.json),
  [spatial report](evidence/2026-11-05-b-gaussian20-spatial.json).
  Baseline pixel AP difference .095659, localSSIM .098979; respective F1 .175333,
  .165446 with thresholds fitted/evaluated on all83(descriptive only).
  Colab runtime released after verified local handoff; disconnected state checked
  2026-11-06. No model training or held-out test evaluation.

### b-matched-g20-s20-20261105 — completed; recorded 2026-11-06

- Both checkpoints20epochs/18,760optimizer updates, batch8/150,000presentations;
  train/sampling seed17,EMA,ancestral t300,crop240. Pinned dataset/model revisions
  as above. UUID-aligned join and checked index reproduce the registered40/43
  fold; patient groups disjoint. Same budget does not remove all source,
  dependency or noise-path differences. This remains exploratory validation.
- Candidates in fixed order:Gaussian difference/SSIM,Simplex difference/SSIM.
  Brightness control excluded from selection. Calibration40 selected Gaussian20
  localSSIM by pixel AP. Thresholds and candidate choice persisted before scoring
  holdout43. No reselection from holdout. CPU48.042s including scoring/bootstrap.

| Holdout candidate | Pixel AP | Pixel AUROC | Precision | Recall | F1/Dice | Normal pixel FPR |
|---|---:|---:|---:|---:|---:|---:|
| Gaussian20 difference | .096786 | .872316 | .113693 | .310428 | .166431 | .060161 |
| Gaussian20 localSSIM | .099777 | .886228 | .107557 | .265253 | .153053 | .058109 |
| Simplex20 difference | .043395 | .701968 | .046950 | .478956 | .085517 | .232669 |
| Simplex20 localSSIM | .087416 | .768279 | .108888 | .372841 | .168550 | .087886 |
| Brightness control | .389506 | .950197 | .161469 | .788577 | .268051 | .134935 |

- Paired group bootstrap1000draws,seed17,25holdoutgroups:999valid/1missing-class.
  LocalSSIM delta AP(Simplex20 minus Gaussian20)95%CI[-.030666,.011189]; delta
  Dice CI[-.046426,.074146]. Neither supports a stable model advantage. Gaussian
  AP CI[.036366,.144901],Simplex AP CI[.028685,.147591]. Thresholds fixed.
- Gaussian localSSIM image AUROC .357456 and all-positive threshold(TP24,FP19);
  Simplex image AUROC .212719. Image-level classification remains unresolved.
- [Complete aggregate results](evidence/2026-11-05-b-matched-g20-s20.json)
  contain provenance/cache/source hashes, fixed selection, all83 descriptive,
  holdout pixel/image metrics and conditional CI. Existing full83 Simplex pixel
  AP/AUROC/Precision/Recall/F1 reproduce the saved baseline to1e-12.
- Driver: `scripts/evaluate_budget_pair.py` (executed as private runs copy before
  import/format cleanup; executed hash retained in evidence). No additional
  inference/training or held-out-test access for this CPU analysis.
