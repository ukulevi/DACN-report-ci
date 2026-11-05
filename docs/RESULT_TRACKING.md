# Experiment and result tracking

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

Add one row per completed evaluation protocol. Split and metric level must be
clear; do not combine metrics from different splits or score definitions.

| Run ID | Method | Dataset revision | Split | Seeds completed | ROC-AUC | Precision | Recall | F1 | Pixel AP / Dice | Latency / VRAM | Artifacts / manifest | Status |
|---|---|---|---|---:|---:|---:|---:|---:|---|---|---|---|
| `ddpm-val-e5-e10-t100-200-300-20261031T074028Z` | Partial DDPM epoch 10, `t*=300`, local one-minus-SSIM; pixel level | `350c74f2313e4e73ca65e7ffc6e9dcf74040b3b7` | validation, 83 slices | 1 evaluation seed | 0.892010 | 0.101571 | 0.395303 | 0.161616 | AP 0.095895 / macro lesion-slice Dice 0.241476 | 11.210 s/slice / 100,651,520 B | [private revision `51a9c22e`](https://huggingface.co/dacn-bmad-team/dacn-runs/tree/51a9c22ee71bdae7d9d78538fb8664a7b96c9e48/evaluations/ddpm-val-e5-e10-t100-200-300-20261031T074028Z) | validation-selected candidate; test sealed |
| `ddpm-val-e10-t300-posterior-mean-t4-b8-20261101T182333Z` | Partial DDPM epoch 10, `t*=300`, posterior mean, local one-minus-SSIM; pixel level | `350c74f2313e4e73ca65e7ffc6e9dcf74040b3b7` | validation, 83 slices | 1 evaluation seed | 0.861427 | 0.096832 | 0.254533 | 0.140292 | AP 0.081721 / macro lesion-slice Dice 0.207741 | 4.912 s/slice / 580,116,480 B | [private revision `3a91ec15`](https://huggingface.co/dacn-bmad-team/dacn-runs/tree/3a91ec15f28d4c0227d36c177a15373f09ff478b/evaluations/ddpm-val-e10-t300-posterior-mean-t4-b8-20261101T182333Z) | completed ablation; worse than ancestral; test sealed |

## Per-run record

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
