# Worklog

This log records completed repository work. Linear remains the source of task
status; this file records changes and evidence, not a second task board. Keep
entries concise and append new entries at the top.

## Entry format

```text
### YYYY-MM-DD — short task title
- Issue/PR:
- Changes:
- Evidence/checks:
- Next:
```

Record only checks that actually ran. Link relevant issues, PRs, commits, or
artifacts when known. Do not put credentials, patient identifiers, or private
source data in this file.

## Entries

### 2026-11-04 — Publish diffusion experiments on the existing feature branch
- Changes: Prepared accumulated T4 Gaussian/Simplex continuation, context training,
  one-step/DDIM/denoising/masked evaluation, CPU score-fusion/diagnostic scripts,
  configurations, tests and aggregate research records for the user-requested
  push to feat/anoddpm-notebook-training. Checkpoints, images and caches remain
  in ignored/private storage.
- Evidence/checks:111 tests passed11.84s, Ruff format check89 files passed,
  Ruff check all30 changed/new Python files passed. Publication scan of46
  changed/new files found no credential patterns, source patient identifiers,
  binary artifacts or oversized files. Git whitespace check passed; CLI demo
  ran successfully. Remote feature branch matches local HEAD before commit.
- Limitation: Full-tree Ruff check reports4 pre-existing findings in unchanged
  notebooks/anoddpm_bmad_colab.ipynb and src/dacn/models/diffusion/unet.py;
  these are outside the staged experiment changes. No GPU job started.

### 2026-11-04 — Read Simplex epoch20 validation results
- Changes: Recorded completed/private-uploaded validation at revision
  1788b1232aa19b3c9ae96aa95a74e2975f4b592a, full pixel/image metrics and thresholds,
  timing/cache hashes and same-protocol epoch10 comparison.
- Evidence: Durable manifest completed/uploaded; validation_results.json confirms
  all83 slices,t300,EMAepoch20,held_out_test_used=false. SSIM AP0.074791 versus
  epoch10 0.073910,F1 0.166414 versus0.165806; only small improvement despite
  noise-MSE decrease. No new GPU evaluation/training or fusion started.
- Checks: Git whitespace check passed; no tests rerun for status-only records.

### 2026-11-04 — Submit Simplex epoch20 full validation
- Changes: Recorded user-authorized83-slice T4 validation, original Simplex
  epoch20 EMA milestone,t300,ancestral,batch1,seed17,paired absolute/localSSIM
  maps and private reconstruction cache. Uses existing evaluator unchanged.
- Evidence/checks: Source training manifest confirms completed/uploaded
  checkpoint20 and hash. Shared Workspace/Secret name verified. Evaluator
  compiles; git whitespace check passed. No tests rerun for submission-only
  documentation; no training, timestep sweep or held-out-test access.
- Start verified: Durable manifest10:57:22UTC, validation_evaluation10:57:49UTC,
  TeslaT4, correct checkpoint20 SHA, cache enabled and held_out_test_used=false.
- Next: Leave detached and check on request.

### 2026-11-04 — Verify Simplex epoch20 completion and private upload
- Changes: Updated continuation result to completed with checkpoint20/best/last
  hashes, upload revision0846480f8551c7e4b41a534c148320d61c906496 and measured
  normal-validation MSE0.002546022252, bestepoch20,18,760 optimizer steps.
- Evidence/checks: Durable manifest statuscompleted/artifact_sync_statusuploaded,
  source artifacts include epoch15/20. App list confirms recovery app stopped
  withzero tasks. No tests rerun for status-only documentation; no GPU job or
  anomaly AP evaluation started. Git whitespace check passed.
- Next: Evaluate epoch20 anomaly maps before claiming AP gain or updating fusion.

### 2026-11-04 — Recover Simplex continuation after Modal preemption
- Changes: Resubmitted stopped run with --resume from durable epoch18 to finish
  epochs19/20, preserving exact source/config and full model/optimizer/RNG state.
  Recorded interrupted attempt, recovery app and checkpoint provenance.
- Evidence: Old app stopped with zero tasks. Logs explicitly report container
  preemption during epoch19 step625, followed by fresh-run directory collision;
  history/manifest confirm completed epoch18,16,884steps,normal-val MSE0.002592801301.
  Current source snapshot matches saved SHA256 exactly. No code changes or tests
  needed for runtime recovery. No additional AP inference or test access.
- Recovery verified: Manifest attempt2/resumed10:07:57UTC; log epoch19/20
  step50/938, running MSE0.006495. First25/50 losses match interrupted epoch19,
  consistent with restored RNG/state. Detached client; remote T4 keeps running.
- Checks: git diff whitespace check passed. No tests rerun for runtime recovery.

### 2026-11-04 — Continue Simplex epoch10 to20 on T4
- Changes: Added epoch20 continuation config with only horizon/milestone changes,
  kept parent run intact, submitted fresh detached T4 run restoring full state
  from epoch10. Save additional milestones15/20; retain batch8/LR1e-4/EMA/noise,
  training split and checkpoint selection. Recorded provenance and estimate.
- Evidence/checks: Read completed parent manifest and Volume listing confirming
  last/best/epoch10 files,9380 steps. Config extension validator accepts only
  epochs/checkpoint_epochs changes.10 DAC8 contract tests passed2.93s; git
  whitespace check passed. Shared Workspace and named Secret verified.
  Initial sandbox client network failure resolved via approved retry; Modal
  app ap-qJD8Ddbo1XCuLciFJ6QJqS initialized successfully.
- Start verification: Durable manifest confirms parent epoch10 and checkpoint
  hashes, TeslaT4, stage training07:06:19UTC, source snapshot
  a6a20656efda0b1094b27442f7429996ed3915a3804e33797404c835718b5651.
- Next: Leave detached; user will request status. No extra inference or test
  access authorized by this training run.

### 2026-11-04 — Test cached Gaussian/Simplex SSIM fusion
- Changes: Added CPU-only registered-cache fusion evaluator with normal-only
  calibration CDF, three frozen blend weights, raw/CDF/brightness controls,
  patient-group split and persisted calibration-only selection/thresholds before
  holdout scoring. Downloaded existing Simplex cache to ignored local runs/;
  no GPU inference/training. Actual evaluation started and tracked.
- Evidence/checks:2 focused tests passed2.97s, verifying CDF ties/holdout
  isolation and frozen thresholds/selection before holdout evaluation. Targeted
  Ruff check/format and git whitespace checks passed. Outputs remain private.
- Results: Completed in22.001 CPU seconds. Best-calibration fusion75%Gaussian /
  25%Simplex has holdout pixel AP0.116771,F1 0.167315,FPR0.051199 versus raw
  Gaussian0.102402/0.154654/0.055234. Overall calibration selector still chooses
  Gaussian CDF; all fusion calibration APs are lower than Gaussian. Recorded
  this distinction and full pixel/image metrics; no robust gain claimed.
- Evidence: Frozen selection persisted before holdout; produced JSON SHA256
  f0d755d8415184eb3e42c80dc050176dca1edcad34edbeb1773b65cb666bed67.
  Gaussian/brightness controls reproduce prior scores. No holdout retuning,
  extra inference, GPU, training or test access.

### 2026-11-04 — Test t300 one-step and DDIM10 against ancestral cache
- Changes: Added separate fixed sampler ablation with common noisy input,
  exact10 DDIM calls/eta0, clipped x0/re-derived epsilon, immutable cache hash/
  array alignment and first-sample ancestral recomputation control. Shared
  grouped evaluation, private prefix artifacts and per-sampler model timings.
  Submitted fresh user-requested T4 all83-validation job; no training/test use.
- Evidence/checks:3 focused tests passed4.39s; oracle inversion, independent
  eta0 transition formula, model-call sequence, RNG preservation, common input,
  frozen thresholds and83-sample coverage. Full109 tests passed9.28s; targeted
  Ruff check/format and git whitespace checks passed. Shared Workspace/Secret
  name verified without credential inspection. Formula compared to primary
  DDIM and guided-diffusion implementations linked in RESEARCH.
- Results: Completed83 samples; first ancestral control matches cache exactly.
  Uploaded private revision4aa62f3b961460928adfc84876a97f050f8f5af9.
  Holdout SSIM AP one-step0.098530/DDIM0.095631/ancestral0.102402; FPR
  0.031443/0.042014/0.055234 with lower recall for new samplers. All83 one-step
  AP increases slightly but not holdout AP; no consistent detection gain.
  Pure reconstruction one-step0.03347s/image,DDIM0.30496s/image, separately
  timed with CUDA synchronization. Recorded full pixel/image metrics and caveats.
- Evidence: Read completed/uploaded durable manifest and produced results.json,
  aggregate JSON hash verified locally; no additional GPU run started.

### 2026-11-04 — Run original Gaussian epoch30 one-step validation
- Changes: Added full-image one-step original1-channel EMA control to shared
  validation evaluator and Modal runner, preserving context paths. Loads original
  registered epoch30/config, Gaussian forward draw and analytical x0; one call
  per image. Submitted user-requested fresh T4 run on all83 validation slices.
- Evidence/checks:14 targeted tests passed4.39s, including original1-channel
  oracle noise inversion, one-call/RNG isolation, evaluator provenance and
  frozen thresholds; targeted Ruff checks passed. Shared workspace and Secret
  name verified, no credentials inspected. No training or test access.
- Results: Completed83/83 and private upload b9f33457a84860275d7caecdfd0438693bbd706a.
  Original one-step all83 pixel AP0.070311 difference /0.071103 SSIM;
  holdout0.075539/0.075919. Higher AP than context one-step, below same-checkpoint
  ancestral t100. Loop51.448s including score/cache work. Recorded pixel/image
  metrics, hashes, calibration-only thresholds and limitations.
- Additional evidence: Full106-test suite passed7.62s; durable manifest confirms
  completed/uploaded and83 samples. Aggregate JSON downloaded to ignored runs/.

### 2026-11-04 — Run one-step context reconstruction on full validation
- Changes: Added optional one-step x0 estimation to patch/context reconstruction,
  preserving ancestral default, coverage/context and derived seeds; extended
  full-validation evaluator/Modal method to distinguish one-step provenance.
  Submitted fresh user-authorized T4 all83-slice run, no training or test access.
- Evidence/checks:104 tests passed8.17s, including single model-call/formula,
  unchanged context and RNG isolation, both evaluator modes, prefix persistence
  and frozen calibration thresholds. Targeted Ruff checks passed. Verified
  shared Workspace and named Secret without reading credentials.
- Results: Completed83/83 and privately uploaded revision
  243070769bc36adaba8e21fda4822ec697205399. Loop219.687s including scores/cache;
  holdout pixel AP0.065636 difference /0.071232 SSIM, lower than brightness.
  All83 descriptive AP0.060280/0.065432 also below Gaussian epoch30 ancestral
  t100 reference. Recorded pixel/image metrics, thresholds, hashes and caveats.
- Evidence: Read completed/uploaded durable manifest and produced results.json;
  downloaded aggregate JSON into ignored runs/. No additional GPU job launched.

### 2026-11-03 — Stop full-validation job at user request
- Changes: Recorded user-stopped evaluation in RESULT_TRACKING; no automatic
  restart. Existing trained checkpoint and prior diagnostic are preserved.
- Evidence/checks: Modal app stop --yes succeeded; app list confirms stopped,
  zero tasks,00:00:34 local time. Durable manifest remains stale running;
  results/progress.json not found. No full-validation metrics produced or
  completed prefix confirmed. No tests run for this status-only update.

### 2026-11-02 — Submit full validation patch/context evaluation
- Changes: Added all-validation evaluator with same epoch5 EMA/t100/patch64
  reconstruction as the diagnostic, fixed patient-group calibration/holdout,
  calibration-only frozen thresholds, descriptive all83 metrics, brightness
  control and durable prefix caches. Raised Modal timeout to3hours based on
  measured ~72s/image; submitted detached user-authorized T4 run.
- Evidence/checks:15 targeted tests passed12.24s, including complete sample
  coverage, prefix persistence, and frozen-threshold propagation; targeted Ruff
  check/format and git diff whitespace checks passed. Verified shared Workspace
  and Secret name presence without reading credentials.
- Next: Confirm durable progress; report produced metrics after completion when
  requested. Do not poll continuously or access sealed test data.

### 2026-11-02 — Run bounded patch/context reconstruction diagnostic
- Changes: Added validation-only four-example sliding preview evaluator and
  Modal method with immutable epoch5 EMA checksum, plus local private comparison
  plotting script. Trial uses two normal/two abnormal examples, t100, patch64,
  stride48,25 patches/image; tumor masks are excluded from reconstruction.
- Evidence/checks: Eleven existing patch/context and contract-audit tests passed;
  new diagnostic test passed after correcting its synthetic UUID fixture.
  Targeted Ruff check/format and git whitespace checks passed. Durable manifest
  confirms completed/uploaded T4 app ap-hd2NZeEANWnVk9F5Y7BAMr, no test-set use;
  full suite101 tests passed20.11s. Four-example inference289.624s; private
  artifacts downloaded for local rendering, compared to the same cached t100
  Gaussian inputs/masks. Normal reconstruction improves on both examples,
  lesion separation remains weak/reversed on the two tumor examples.
- Next: Review diagnostic figures before choosing further experiments;
  this bounded diagnostic does not estimate population AP.

### 2026-11-02 — Check completed patch/context pilot
- Changes: Updated RESULT_TRACKING from running to completed with all five
  epoch losses, best epoch5 checkpoint, runtime/VRAM and immutable private Hub
  revision62a2d58b5db5004cc6337626a026c76ba7ebf963.
- Evidence/checks: Read durable Modal run manifest and training_history.json;
  confirmed5 epochs,4,690 steps and artifact_sync_status=uploaded. Git whitespace
  check passed. No new training or inference job started; no tests rerun for
  this documentation update.
- Next: Bounded patch/context reconstruction timing/quality pilot before a
  full validation evaluation; patch MSE alone cannot establish anomaly AP.

### 2026-11-02 — Implement and submit5-epoch patch/context fine-tune
- Issue/PR: none provided.
- Changes: Added independent3-channel patch/context trainer, verified EMA
  initialization from immutable Gaussian epoch30, zero-initialized extra input
  kernels, patch-only epsilon loss, random foreground-centered64px patches,
  new optimizer/EMA, milestone1/3/5 full-state checkpoints and private preview.
  Added conditioned patch/sliding inference and checkpoint loader for later use.
  Registered config patch_context_t4_finetune5 and adaptation protocol.
- Evidence/checks:100 tests passed, including8 patch-context tests covering
  target leakage, initialization, output shape, patch loss/gradient, crop
  coverage and end-to-end fresh-optimizer checkpointing. Actual base32 U-Net
  CPU smoke confirmed3 inputs/1 output at256px. Ruff/format/compile/diff passed.
  Shared Workspace/Secret name verified;5 epochs/4,690 new steps configured.
- Evidence/checks: Modal app ap-EBw2niCSb3IeS2iLs2DK5W durably started
  patch-context-ft5-t4-b8-20261102T155628Z at16:00:57UTC; source snapshot in
  tracking. Live output reached fine-tune epoch1,step25/938 with finite initial
  patch epsilon MSE2.944604. This is an early new-task loss, not an AP result.
- Next: Disconnect client; inspect epoch1/3/5 artifacts on request. No automatic full sliding
  evaluation or held-out test access; no epoch budget beyond5.

### 2026-11-02 — Audit checkpoint/preprocessing/diffusion contract
- Issue/PR: none provided.
- Changes: Added controlled read-only GPU diagnostic for exact EMA/epoch/input
  contract checks, one-step online/EMA predictions and paired large/small/mean
  reverse chains. Documented static audit in PIPELINE_CONTRACT_AUDIT and fixed
  protocol in RESEARCH; production model/sampling remains unchanged.
- Evidence/checks:18 targeted audit/contract/validation tests passed; diagnostic
  large chain exactly matches production in unit test, oracle inversion and
  common-noise controls verified. Targeted Ruff/format/compile/diff passed.
  T4 run gaussian-e30-contract-audit-20261102T153939Z started and recorded;
  T4 run completed and uploaded at f69fc08d: exactEMA/input/source/schedule/
  arithmetic checks pass. t300 one-stepEMA MAE0.053872/SSIM0.527959 versus
  full-large0.074614/0.358931. Small variance changes little; deterministic
  mean improves SSIM but worsens t300 MAE. No test data or retraining.
- Next: Retain baseline; investigate foreground/timestep loss breakdown and
  controlled training/conditioning changes, not claim AP gains from normal
  fidelity alone. Full findings in PIPELINE_CONTRACT_AUDIT.

### 2026-11-02 — Audit normal foreground reconstruction at t100/200/300
- Issue/PR: none provided.
- Changes: Added reproducible offline normal reconstruction audit, aggregate
  report NORMAL_RECONSTRUCTION_AUDIT and paired private example sheet.
- Evidence/checks: Verified39 normal empty-mask slices, all source cache hashes
  and cross-timestep alignment. Foreground meanSSIM drops0.533->0.375->0.293;
  full-crop SSIM is inflated by74.11% background. Interior errors remain high.
  Regional pooled-MAE identities/MAE<=RMSE verified; Ruff/format/compile/diff
  checks passed. CPU40.586s, no new inference/GPU/test/training.
- Next: Audit train/inference contract before blaming weights alone or
  allocating more epochs; this step measures the symptom, not its root cause.

### 2026-11-02 — Audit Gaussian batch8 loss convergence
- Issue/PR: none provided.
- Changes: Downloaded recorded30-epoch training history; added reproducible
  plotting script, PNG/PDF/CSV under ignored runs/, and TRAINING_CONVERGENCE
  report separating loss behavior from anomaly quality.
- Evidence/checks: History checksum matches manifest;30 sequential epochs and
 938 steps/epoch verified. Normal-val MSE10->20 improves3.46%,20->30 1.14%,
 25->30 0.36%; best epoch30. No sustained late validation-loss rise. Train loss
 uses online model/random noise, validation uses EMA/fixed noise; no direct
 train-versus-val gap interpretation. Targeted Ruff/format and plot generation
 passed; no retraining/new inference or held-out test access.
- Next: Audit normal-foreground reconstruction and train/inference contract;
 loss plateau alone does not explain low AP or justify more epochs.

### 2026-11-02 — Test reconstruction-map ensembles across t100/200/300
- Issue/PR: none provided.
- Changes: Added fixed equal-weight calibration-rank-mean evaluator using three
  existing Gaussian epoch30 caches. Prespecified protocol, same patient-group
  split and calibration thresholds; documented all comparison results.
- Evidence/checks: Source checksums and exact image/mask/label alignment
  asserted; baseline t200/t300 and brightness metrics reproduce earlier values.
  Ensemble difference AP0.091536 /F1 0.155345; SSIM AP0.092199 /F1 0.132428,
  both below single t300. Frozen pixel thresholds checked independently.
  Ruff/format/compile/diff passed. CPU96.510s; no GPU/test/training access.
- Next: Preserve negative result; do not adopt equal averaging as an improvement
  or retune weights on the already-inspected holdout.

### 2026-11-02 — Read completed AutoDDPM-style pilot
- Issue/PR: none provided.
- Changes: Updated run record from completed Modal manifest/results, private
  revision `2c8646c1`.
- Evidence/checks: Holdout resampled SSIM AP0.093603/F1 0.138912; difference
  AP0.070007/F1 0.149523. Mask covers63.05% of lesion pixels with9.20% precision;
  resampling gives no overall AP benefit over the best original DDPM baseline.
  Runtime769.038s/83 images; peakVRAM101,765,120B. Brightness control unchanged.
  No new GPU experiment, code change or local tests; test remains sealed.
- Next: Preserve negative evidence and review predicted mask quality before
  any further resampling/training. Do not generalize to exact LPIPS AutoDDPM.

### 2026-11-02 — Implement and submit AutoDDPM-style SSIM-proxy pilot
- Issue/PR: none provided.
- Changes: Added independently implemented automatic masking, stitching,
  matching-time context conditioning and resampling using the current Gaussian
  epoch30 checkpoint. Reuses verified t200 cache, fixed50x5 resampling schedule;
  SSIM substitutes LPIPS and the adaptation is explicitly labeled. Added initial
  score/naive-stitch controls, mask coverage diagnostics and grouped calibration
  holdout evaluation; no use of ground-truth masks for inference.
- Evidence/checks: Three sampler/mask tests passed,9 focused tests total;
  full suite89 tests passed. Targeted Ruff/format/compile/diff checks passed.
  Verified shared Workspace and Secret name. No clinical/improvement claims.
- Evidence/checks: Modal app `ap-pWoLoKKbxzgCa1RD9FoPOS` created a durable
  start manifest at14:01:03UTC for `autoddpm-style-ssim-e30-20261102T140011Z`;
  experiment recorded in RESULT_TRACKING.
- Next: Leave detached and inspect completed artifacts on request. Held-out
  test remains sealed.

### 2026-11-02 — Read completed denoising-score experiment
- Issue/PR: none provided.
- Changes: Updated run record from completed Modal manifest and results,
  private revision `4713bd88`.
- Evidence/checks: Denoising holdout pixel AP0.085848 /F1 0.165330 on43 images;
  below original reconstruction AP~0.10 and brightness0.389506. Brightness
  metrics/index hash reproduce prior grouped split. Inference loop28.876s for83
  images, peakVRAM101,437,440B. No new experiments, code edits or local tests.
- Next: Do not adopt this averaged epsilon-error score as an AP improvement.
  Preserve negative result and review a single justified alternative before
  spending further compute. Held-out test remains sealed.

### 2026-11-02 — Set up requested T4 denoising-score experiment
- Issue/PR: none provided.
- Changes: Added a fixed Modal T4 runner with checkpoint/dataset checksum
  verification, private Hub destination check, source snapshot and durable
  lifecycle/upload. Uses existing grouped denoising evaluator on epoch30.
- Evidence/checks: Six focused scoring tests passed; runner Ruff/format/diff
  checks passed. Shared Workspace and existing Secret name verified. No model
  retraining; experiment uses3 timesteps x4 noise repeats and original ROI.
- Evidence/checks: Modal app `ap-beD6gCpUwEv4VHg0Eu2krT` created a durable
  start manifest for `ddpm-denoising-e30-3t4r-20261102T103159Z` at10:32:32UTC;
  source checkpoint hash verified, exact source snapshot recorded in tracking.
- Next: Compare completed metrics on the same group holdout as brightness and
  reconstruction controls when requested; leave detached without continuous polls.

### 2026-11-02 — Add denoising-score pipeline alongside reconstruction
- Issue/PR: none provided.
- Changes: Added fixed multi-timestep epsilon-error scoring and validation-only
  CLI/evaluator. Reuses Gaussian checkpoints; averages3 timesteps x4 noise draws
  with isolated per-sample generators; same grouped calibration/holdout rule,
  brightness control and frozen thresholds as the preceding score study.
  Documented that this is an exploratory ablation, not a full ANDi reproduction.
- Evidence/checks: Six focused tests passed for exact-noise oracle, timestep
  indexing, averaging, global RNG isolation and invalid inputs. Full suite86
  tests passed; targeted Ruff/format/diff checks passed. No actual MRI experiment, AP claim, GPU job or test
  split access in this implementation task.
- Next: Run a registered validation pilot with epoch30 and compare against the
  existing brightness/reconstruction controls before adopting the new branch.

### 2026-11-02 — Test nine score options with patient-group calibration
- Issue/PR: none provided.
- Changes: Added reproducible CPU fusion evaluator and preregistered protocol;
  documented all nine candidates in SCORE_FUSION_RESULTS. Used existing epoch30
  t300 cache and private dataset index; group-disjoint40/43-image split.
- Evidence/checks: Cache checksum/labels/unique IDs/group disjointness asserted.
  Normalization and thresholds fit only calibration; selection JSON written
  before holdout evaluation. Independent check verified candidate selection and
  unchanged pixel/image thresholds. Ruff/format/compile/diff checks passed.
  Selected fusion holdout AP0.262977 versus brightness0.389506; reconstruction
  baselines~0.10. CPU20.628s; no new GPU inference or test access.
- Next: Keep the brightness control; no proven AP benefit of fusion over it.
  Confirm any chosen method with a separately registered validation procedure
  before final frozen test. No holdout-driven weight retuning.

### 2026-11-02 — Compare completed epoch30 t*=100/200/300 validation
- Issue/PR: none provided.
- Changes: Updated timestep run and diagnostic report from completed Modal
  manifest/results, private artifact revision `67626af0`.
- Evidence/checks: SSIM AP t100/200/300=0.076971/0.088429/0.100741;
  absolute-difference AP=0.070977/0.083184/0.099834. Lowering t* does not
  improve either score under the matched checkpoint/protocol. Recorded full
  pixel metrics, thresholds and runtime; no new training/evaluation or code edits.
- Next: Investigate score calibration/combination using existing validation
  caches rather than assume smaller t* solves the failure. Test stays sealed.

### 2026-11-02 — Diagnose Gaussian epoch30 low anomaly scores
- Issue/PR: none provided.
- Changes: Added checksum-verified validation cache diagnostics and an aggregate
  report at `docs/EPOCH30_DIAGNOSIS.md`. Measured score overlap, regional errors,
  brightness control and foreground-gating ablations. Submitted controlled
  epoch30 t*=100/200 evaluation on T4 with both original scores and private cache.
- Evidence/checks: CPU script reproduced original t*=300 metrics; brightness
  AP0.272325 exceeds reconstruction AP~0.10. Foreground-only gating gives small
  gains; 83.47% of SSIM false positives are in nonlesion foreground interior.
  Targeted Ruff/format passed; source cache checksum verified. GPU evaluation
  app `ap-Yh8Bu3Qilehwpdxl5LDKkK` created; new timestep metrics pending.
- Next: Read completed timestep artifacts on request; do not continuously poll.
  No retraining, held-out test use or silent baseline protocol changes.

### 2026-11-02 — Read completed Simplex epoch-10 validation
- Issue/PR: none provided.
- Changes: Updated evaluation record from completed/uploaded Modal manifest
  and validation_results.json, private revision `b4c408be`.
- Evidence/checks: All83 validation slices at t*=300; SSIM pixel AP 0.073910,
  F1 0.165806, Recall 0.603844, empty-normal FPR 0.133712; absolute-difference
  AP 0.032447 / F1 0.068113. Runtime 749.373 seconds. Main AP does not improve
  over existing Gaussian results; matched Gaussian epoch10 batch8 is unevaluated.
  No new GPU run, code changes or local tests; held-out test unused.
- Next: Inspect cached Simplex reconstructions before further training; compare
  matched training budgets before attributing changes solely to noise type.

### 2026-11-02 — Verify Simplex training and submit epoch-10 validation
- Issue/PR: none provided.
- Changes: Updated Simplex training record from the completed/uploaded manifest;
  submitted user-requested full validation at t*=300 with both registered scores.
- Evidence/checks: Simplex completed 10 epochs / 9,380 steps, normal-validation
  noise MSE 0.002971782594, training/validation time 4,465.438 seconds; private
  artifact revision `e7b6ce4f`. Shared Workspace and Secret name verified.
  Evaluation app `ap-ju8ELLyWBFQIacpQs4fcxK` created; metrics pending.
  No implementation changes or local tests in this task.
- Next: Read aggregate evaluation artifacts when requested after completion;
  avoid continuous polling. Held-out test remains sealed.

### 2026-11-02 — Launch user-requested 10-epoch Simplex ablation
- Issue/PR: none provided.
- Changes: Added explicit forward-noise/Simplex settings to partial-DDPM config
  and diffusion construction while keeping legacy checkpoints Gaussian by
  default. Registered a full-data T4 batch8 seed17 10-epoch Simplex config,
  retaining the same U-Net, optimizer/EMA and checkpoint-selection procedure.
  Documented noise semantics and planned validation comparison.
- Evidence/checks: Full suite passed with 80 tests; focused Ruff/format,
  compilation and diff checks passed. Full-shape CPU Simplex batch8 smoke was
  finite and exactly repeatable with fixed seed; warm noise generation took
  0.047 seconds using two CPU threads locally. No remote timing inferred.
  Modal app `ap-ssVCXTam3wHn7EaSMJFMtS` created a durable start manifest for
  `ddpm-simplex-t4-b8-e10-20261102T074257Z`; source snapshot recorded in tracking.
- Next: Leave detached without continuous polling. Read completion artifacts
  when requested, then evaluate final milestone on validation at t*=300.

### 2026-11-02 — Verify epoch-30 checkpoint on private Hugging Face Hub
- Issue/PR: none provided.
- Changes: Verified the existing upload instead of creating a duplicate commit.
- Evidence/checks: Team-authenticated Modal CPU check confirmed private model
  repository revision `1ee5c4995623532e7ab51c043839555c1dd0b0db` contains
  `ddpm_epoch_030.pt`, `ddpm_best.pt`, `ddpm_last.pt`, config and run manifest.
  Local Hub access returned 404; verification succeeded using the existing
  shared Secret without changing credentials. No training or evaluation run.
- Next: Download through the private repository with a team-authorized account.

### 2026-11-02 — Visualize saved epoch-30 validation reconstructions
- Issue/PR: none provided.
- Changes: Prepared four private local PNG comparison figures from the existing
  epoch-30 t*=300 reconstruction cache at the user's request: highest, median
  and lowest lesion-slice SSIM Dice, plus the normal slice with highest SSIM
  false-positive rate. Panels show input, reconstruction, both score maps,
  SSIM threshold prediction and ground truth. Figures and script stay under
  ignored `runs/visualizations/`; no patient identifiers are displayed.
- Evidence/checks: Downloaded cache checksum matches the completed evaluation
  manifest; plotting uses the existing score implementation and recorded SSIM
  threshold 0.8922310462404459. This is visualization of existing validation
  results, not a new inference run. No GPU job or held-out test access.
- Next: Review where predicted anomalies overlap lesions and normal anatomy.

### 2026-11-02 — Read completed epoch-30 t*=300 validation results
- Issue/PR: none provided.
- Changes: Updated RESULT_TRACKING from the completed Modal manifest and
  validation_results.json, private artifact revision `4472519e`.
- Evidence/checks: All 83 validation slices completed on T4; SSIM pixel AP
  0.100741 / F1 0.167483; pixel-difference AP 0.099834 / F1 0.182870.
  Image AUROC remains below 0.5 and max-F1 predicts every image positive.
  Runtime 718.415 seconds. No code changes, new GPU run or local tests.
- Next: Diagnose saved validation reconstructions/maps before increasing the
  training budget. Held-out test remains sealed.

### 2026-11-02 — Verify epoch-30 completion and launch t*=300 validation
- Issue/PR: none provided.
- Changes: Updated training status from the durable completed/uploaded manifest;
  submitted user-requested EMA epoch-30 evaluation on all 83 validation slices
  with ancestral sampling at t*=300, inference batch 1 and both registered scores.
- Evidence/checks: Epoch 30 / 28,140 optimizer steps completed; best normal-val
  epsilon MSE 0.00735903694; artifacts uploaded at revision `1ee5c499`.
  Evaluation app `ap-RWi4F9nKiAnh75BoUA42BG` created on the shared Workspace;
  results pending. No implementation changes or local tests in this task.
- Next: Inspect aggregate evaluation artifacts when complete; no continuous
  monitoring requested. Held-out test remains sealed.

### 2026-11-02 — Complete T4 epoch-1 pilot and prepare safe epoch extension
- Issue/PR: no issue or PR was provided.
- Changes: Recorded the completed batch-8 epoch-1 result and added a lineage-aware
  continuation path that permits only the epoch horizon/checkpoint schedule to
  change. Added the matching epoch-30 T4 config; model, EMA, optimizer, sampler
  and RNG state resume from the source checkpoint while the completed source run
  remains immutable.
- Evidence/checks: Epoch 1 completed in 412.506 seconds with train epsilon MSE
  0.080065, normal-validation epsilon MSE 0.323197, 938 optimizer steps and
  5,465,677,824 bytes peak VRAM; private revision `178c1de6`. Continuation unit
  tests verify epoch 2/history/optimizer behavior and rejection of batch-size
  changes. Full suite passed with 79 tests; targeted Ruff, format, compile and
  `git diff --check` passed. Full-tree Ruff still reports two pre-existing
  findings in `models/diffusion/unet.py` outside this change. The detached continuation
  `ddpm-partial-t4-b8-e30-cont-20261102T034644Z` was then created as Modal app
  `ap-Gau6wwuNQYP4U3UJBi6rrm`; live output reached epoch 2, step 25/938, proving
  that the source epoch was restored rather than repeated.
- Next: Do not poll continuously; inspect the durable manifest/artifacts when
  requested after the estimated completion window. Keep held-out test unavailable.

### 2026-11-02 — Prepare T4 batch-8 one-epoch DDPM pilot
- Issue/PR: `BLK-20261102-03`; no issue or PR was provided.
- Changes: Added a fresh full-data Gaussian DDPM pilot config with one epoch,
  batch size 8, validation batch size 8 and EMA 0.9992, plus a fixed T4 Modal
  remote entrypoint sharing the existing durable training lifecycle. Documented
  that it cannot resume the batch-1 run and is not a replacement checkpoint.
- Evidence/checks: Resolved config reports 7,500 training samples, 938 optimizer
  steps and checkpoint epoch 1. Full suite passed with 77 tests; targeted Ruff,
  format, compilation and `git diff --check` passed. Four sandboxed launch
  attempts failed at Modal connection before app/run creation. A final fixed
  entrypoint submission with direct network permission created Modal app
  `ap-vIFbYSY0oC1V74Tj4DFJvV`, durable start manifest and source snapshot; the
  running pilot is now registered in `RESULT_TRACKING`.
- Next: Read durable runtime/loss/checkpoint artifacts after completion; keep
  held-out test unavailable.

### 2026-11-02 — Evaluate posterior-mean DDPM inference on T4
- Issue/PR: `BLK-20261101-02`; no issue or PR was provided.
- Changes: Added a posterior-mean reverse sampler with batch-independent
  per-sample initial-noise seeds, T4 batch-8 validation support and private
  reconstruction caching. Ran a two-sample CPU shape/cache smoke, then evaluated
  epoch 10 at `t*=300` on all 83 validation slices. Held-out test was unavailable
  to the evaluator.
- Evidence/checks: Full local suite passed with 77 tests; targeted Ruff, format,
  compilation and `git diff --check` passed before launch. The Modal manifest
  completed on Tesla T4 and uploaded private revision `3a91ec15`. Local SSIM AP
  was 0.081721 and F1 0.140292, both below ancestral AP 0.095895 and F1 0.161616,
  despite lower empty-normal FPR. Posterior mean is rejected as the replacement
  sampler; reconstruction cache and exact source snapshot remain private.
- Next: Keep the ancestral baseline and review a controlled Simplex-noise
  training experiment before any held-out test evaluation.

### 2026-11-02 — Audit DDPM inference false positives without GPU
- Issue/PR: `BLK-20261101-02`; no issue or PR was provided.
- Changes: Audited checkpoint loading, EMA/config, one-based partial diffusion,
  deterministic per-sample seeding, crop, local SSIM/pixel-difference maps,
  threshold selection and validation-only split use. Ran bounded CPU diagnostics
  on one validation lesion slice for foreground ROI, three-reconstruction
  averaging and a posterior-mean reverse-chain hypothesis; no test data or GPU
  was used. Updated result tracking and blocker evidence without changing the
  registered inference implementation.
- Evidence/checks: Epoch-5/10 SHA256 values matched the durable manifest; 18
  targeted tests passed. The foreground diagnostic found most false positives
  inside normal brain tissue, not the padded/background region. Three-sample
  averaging and posterior-mean reverse produced mixed AP/AUROC changes on one
  slice, so neither is reported as a validation improvement.
- Next: If GPU budget is approved, run one registered full-validation sampler
  ablation with private reconstruction caching before spending budget on more
  training; keep held-out test sealed.

### 2026-11-01 — Back up the epoch-10 checkpoint to private Hugging Face
- Issue/PR: User-requested checkpoint upload; no issue or PR was provided.
- Changes: Used the shared Modal Secret in a temporary CPU-only helper to sync
  `ddpm_epoch_010.pt`, the durable run manifest, aggregate history/validation
  metrics and exact source snapshot from `dacn-bmad-runs` to the existing
  private `dacn-bmad-team/dacn-runs` model repo. No credential value, private
  image or per-sample identifier was read or logged.
- Evidence/checks: Hugging Face commit
  `07ec8b455a3318a267df5568f7af3fa4f15d7a3c` reports a private destination. The
  uploaded checkpoint SHA256
  `b828b5889d4164c0d9cebbbe0bf7afb603f5bec9ad7c84508baa222a26bc4fc9` matches
  the Modal manifest and prior local verification. No GPU, training or
  evaluation run was started.
- Next: Grant intended teammates private repo access or share the immutable
  artifact link; resume the original Modal run separately if epoch 20/30 is
  still required.

### 2026-11-01 — Add blocker tracking and team handoff rules
- Issue/PR: User-requested shared blocker log; no issue or PR was provided.
- Changes: Added `docs/ISSUES.md` with a stable blocker template, summary table,
  ownership/handoff fields, privacy rules, and two evidence-backed current
  blockers covering the incomplete primary DDPM run and validation false
  positives. Updated `AGENTS.md` so AI agents create or update this log before
  handing off unresolved blockers while keeping Linear and result tracking as
  their existing sources of truth.
- Evidence/checks: Reviewed the contributor workflow, project requirements,
  research protocol and recorded run results; `git diff --check` passed.
  Documentation-only change; no training, evaluation or test suite was run.
- Next: Confirm owners during team handoff, resume or close
  `BLK-20261101-01`, and use the fixed validation protocol to investigate
  `BLK-20261101-02` without opening held-out test data.

### 2026-10-31 — Add real-checkpoint validation evaluation runner
- Issue/PR: User-requested testing of epoch-5 and epoch-10 checkpoints; no issue
  or PR was provided.
- Changes: Added a validation-only checkpoint evaluator and Modal boundary for
  deterministic partial reconstruction, pixel-difference/local-SSIM scoring,
  threshold selection, aggregate metrics, incremental durable results and
  private artifact upload. The evaluator has no held-out test path.
- Evidence/checks: The complete local suite passed with 58 tests and 3
  PyTorch-dependent skips; targeted Ruff, Python compilation and
  `git diff --check` passed. The four-sample epoch-5 `t*=100` smoke completed on
  `NVIDIA A100-SXM4-40GB`, wrote aggregate results, explicitly recorded no
  held-out test use, and uploaded private artifacts at model revision
  `5e77e82dddfc423af2bf179282441978ee08d6f3`. The subsequent full sweep
  completed all six epoch/timestep combinations over 83 validation slices on
  the same A100 class. Local one-minus-SSIM with epoch 10 and `t*=300` had the
  best pixel AP (0.095895) and pixel AUROC (0.892010); its micro pixel F1 was
  0.161616 and macro lesion-slice Dice was 0.241476. The final manifest records
  six of six combinations complete, private artifact revision
  `51a9c22ee71bdae7d9d78538fb8664a7b96c9e48`, and no held-out test use.
- Next: Freeze the validation-selected epoch/timestep, score formula,
  aggregation and threshold protocol before authorizing one held-out test run;
  obtain additional seeds before drawing research conclusions.

### 2026-10-31 — Download epoch-5 and epoch-10 DDPM checkpoints
- Issue/PR: User-requested local checkpoint copies; no issue or PR was provided.
- Changes: Downloaded `ddpm_epoch_005.pt` and `ddpm_epoch_010.pt` from the
  stopped milestone run's Modal Volume into the ignored local `checkpoints/`
  run directory. No remote artifact was changed or deleted.
- Evidence/checks: Both files are approximately 126 MB. Local SHA-256 values
  `8c1d957265858c70879f46273691be7d88e93c4a562d50168c2991716aed682f`
  and `b828b5889d4164c0d9cebbbe0bf7afb603f5bec9ad7c84508baa222a26bc4fc9`
  match the durable Modal manifest for epochs 5 and 10 respectively.
- Next: Use these private local checkpoints for validation-only reconstruction
  and scoring; do not select parameters on the held-out test split.

### 2026-10-31 — Fix CUDA RNG restore and restart detached DDPM training
- Issue/PR: User-requested restart after a Modal client cancellation; no issue or
  PR was provided.
- Changes: Reproduced the epoch-4 resume failure, converted restored CUDA RNG
  states to CPU ByteTensors before calling PyTorch's CUDA RNG API, added a
  regression test, and invoked `train_remote` directly with Modal detach instead
  of the client-bound local controller. Preserved all earlier run artifacts.
- Evidence/checks: The resume attempt passed config/source provenance and then
  failed with `TypeError: RNG state must be a torch.ByteTensor` before epoch 5;
  its epoch-4 checkpoints remained intact. The local suite passed with 58 tests
  and 2 PyTorch-dependent skips; targeted Ruff, JSON validation, Python
  compilation and `git diff --check` passed. Fresh run
  `ddpm-partial-a100-s17-e30-milestones-rngfix-20261031T041542Z` has a durable
  manifest confirming the immutable dataset revision, milestones `[5,10,20,30]`,
  and `NVIDIA A100-SXM4-40GB`. At the user's request, the app stopped at
  07:07:05 UTC during epoch 11, step 1,800/7,500. Epoch 10 and 75,000 optimizer
  steps are durable; the Volume contains milestone checkpoints for epochs 5 and
  10 plus `ddpm_last.pt` and `ddpm_best.pt`. Best epoch is 10 with
  normal-validation epsilon-MSE 0.0101424818. No anomaly/test metric was
  produced.
- Next: Preserve the stopped artifacts or resume the exact run from epoch 11 if
  later milestones are required.

### 2026-10-31 — Retain DDPM milestone and best checkpoints
- Issue/PR: User-requested epoch 5, 10, 20 and 30 checkpoints; no issue or PR
  was provided.
- Changes: Added validated `checkpoint_epochs` configuration and atomic full-state
  checkpoint retention as `ddpm_epoch_005.pt`, `010.pt`, `020.pt`, and `030.pt`,
  while preserving validation-selected `ddpm_best.pt` and resumable
  `ddpm_last.pt`. Updated the full A100 config, guide, and resume contract test.
- Evidence/checks: The local suite passed with 58 tests and 2 PyTorch-dependent
  skips; targeted Ruff, JSON validation, Python compilation and
  `git diff --check` passed. Fresh run
  `ddpm-partial-a100-s17-e30-milestones-20261030T165258Z` has a durable manifest
  confirming the immutable dataset revision, milestone list `[5,10,20,30]`,
  status `running`, stage `training`, and `NVIDIA A100-SXM4-80GB`. A later Modal
  client cancellation stopped the function in epoch 5 at step 2,400/7,500.
  Epoch 4 and 30,000 optimizer steps are durable; `ddpm_last.pt` and
  `ddpm_best.pt` both represent epoch 4, with best normal-validation epsilon-MSE
  0.0138888134. No epoch-5 milestone or anomaly/test metric was produced. The
  interrupted epoch-2 run remains unchanged.
- Next: Resume the same milestone run from epoch 5 using its exact config and
  source snapshot, with a launch method that does not tie cancellation to the
  local controller lifecycle.

### 2026-10-30 — Change the full-data A100 budget to 30 epochs
- Issue/PR: User-authorized replacement of the running 40-epoch retry; no issue
  or PR was provided.
- Changes: Stopped `ddpm-partial-a100-s17-e40-retry-20261030T145811Z`, changed
  the reproducible seed-17 config and launch guide to 30 epochs (at most 225,000
  optimizer steps), raised the Modal function timeout from 6 to 10 hours for the
  observed A100 throughput, and retained the same full training data,
  validation-only checkpoint selection, `T=1000`, and held-out test isolation.
- Evidence/checks: Modal reports the replaced A100 app stopped at 15:02:43 UTC.
  Its last output was epoch 1, step 2,300/7,500 with no completed epoch or
  checkpoint; zero seeds completed and no anomaly/test metric was produced.
  JSON validation, Python compilation, `git diff --check`, and the local suite
  passed with 58 tests and 2 PyTorch-dependent skips. The replacement run
  `ddpm-partial-a100-s17-e30-20261030T150704Z` has a durable manifest confirming
  30 epochs, the immutable dataset revision, and `NVIDIA A100 80GB PCIe`. At the
  user's request, Modal stopped that run at 15:29:27 UTC during epoch 3, step
  2,600/7,500. The Volume retains `ddpm_last.pt` and `ddpm_best.pt` from epoch 2;
  the aggregate history confirms 15,000 optimizer steps and best normal-validation
  epsilon-MSE 0.0652383983. No anomaly or held-out test metric was produced.
- Next: Resume the same run ID with its exact config and source snapshot if the
  user wants to continue from epoch 3; otherwise preserve it as interrupted.

### 2026-10-30 — Start full-data partial-DDPM training on Modal A100
- Issue/PR: User-authorized run
  `ddpm-partial-a100-s17-e40-20261030T144735Z`; no issue or PR was provided.
- Changes: Synchronized the active Modal guide from T4 to A100, added the
  reproducible 40-epoch seed-17 full-data config, and registered the running
  training attempt in `RESULT_TRACKING.md`.
- Evidence/checks: Confirmed Modal Workspace `phutungcogioihaitan` and Secret
  `huggingface-secret-2`; validated the 25-key JSON config; Python compilation,
  `git diff --check`, and the local suite passed with 58 tests and 2
  PyTorch-dependent skips. The durable Modal manifest confirms immutable dataset
  revision `350c74f2313e4e73ca65e7ffc6e9dcf74040b3b7`, status `running`, stage
  `training`, and `NVIDIA A100-SXM4-40GB`. A later client cancellation stopped
  the run at epoch 1, step 4,700/7,500 before any checkpoint; the run Volume has
  only its manifest and source snapshot. No evaluation or held-out test run was
  started, and no model result was produced.
- Next: Stop the stale app and launch a fresh run ID with the same config; this
  attempt cannot resume because no completed-epoch checkpoint exists.

### 2026-10-30 — Retry full-data A100 run after client cancellation
- Issue/PR: User-authorized retry
  `ddpm-partial-a100-s17-e40-retry-20261030T145811Z`; no issue or PR was provided.
- Changes: Confirmed the previous function input was canceled before its first
  checkpoint, recorded that attempt as interrupted, and launched a fresh run ID
  with the identical source snapshot, immutable dataset revision and config.
- Evidence/checks: Modal reported the old app stopped at 21:57:55+07. The new
  durable manifest confirms status `running`, stage `training`, and
  `NVIDIA A100-SXM4-40GB`; initial output reached epoch 1, step 200/7,500. No
  evaluation or held-out test run was started.
- Next: Keep the local detached client alive, confirm the first completed-epoch
  checkpoint, and update both run records from durable artifacts.

### 2026-10-30 — Add a runnable paired-score evaluation demo
- Issue/PR: Not provided.
- Changes: Replaced the scalar metric example with a deterministic synthetic
  reconstruction pipeline that evaluates pixel difference and local SSIM,
  selects pixel/image thresholds on validation, and reuses those frozen
  thresholds on test. Added macOS/Linux setup commands and CLI regression
  assertions for the validation/test boundary.
- Evidence/checks: The synthetic CLI demo completed and wrote aggregate JSON;
  Ruff check/format, Python compilation, `git diff --check`, and the complete
  CPU suite passed with 58 tests and 2 PyTorch-dependent skips. No medical data,
  model training, or research evaluation run was used.
- Next: Install a compatible PyTorch build in the project Python environment
  before exercising DDPM training or reconstruction code locally.

### 2026-10-30 — Point Modal jobs to the rotated Hugging Face secret
- Issue/PR: Not provided.
- Changes: Updated the active Modal runner and setup guidance to use
  `huggingface-secret-2` while retaining `HF_TOKEN` as the injected environment
  key. No credential value was stored in the repository.
- Evidence/checks: `git diff --check`, Python compilation, and notebook JSON
  validation passed. Modal Workspace verification was not run because the Modal
  CLI is not installed in this environment. No Modal job or model training run
  was started.
- Next: Confirm `huggingface-secret-2` exists in the shared Modal Workspace and
  contains a read/write-capable `HF_TOKEN` before the next job.

### 2026-10-30 — Run bounded partial-DDPM Modal smoke
- Issue/PR: DAC-10, DAC-14; run `ddpm-partial-smoke-20261030T044737Z-79f605`.
- Changes: Recorded the completed smoke and its provenance in RESULT_TRACKING.
- Evidence/checks: The `tanh1c` Modal workspace and `huggingface-secret` Secret
  name were present. The pinned dataset passed validation; the job completed
  128 optimizer steps and uploaded checkpoints and four private validation previews.
  Manifest confirms Tesla T4 and completed status. No research evaluation or
  test selection started. Resume and credit consumption were not measured.
- Next: Attach B/C validator evidence to DAC-9, record the DAC-8 review decision,
  demonstrate resume and estimate the resource budget before a larger run.

### 2026-10-30 — Add paired scoring and evaluation primitives
- Issue/PR: DAC-12 and DAC-30; work in progress on `feat/anoddpm-notebook-training`.
- Changes: Added cropped pixel and local SSIM maps, image top-1% scores, validation
  F1 threshold selection with tie rule, and aggregate image/pixel metrics. Updated
  DAC-9 in Linear to distinguish the current pinned revision from the older export.
- Evidence/checks: The full CPU suite passed (66 tests before the final paired
  helper); the final focused suite passed (9 tests), covering score direction, crop,
  SSIM formula, threshold ties, metric output and test-threshold guard. A final
  full-suite rerun passed 67 tests; Ruff and formatting checks passed. No real-data
  evaluation, training, Modal job or sweep was started in this implementation task.
- Next: Collect B/C validator evidence, record DAC-8 review, integrate the evaluator
  with the DDPM reconstruction runner and obtain teammate review before the sweep.

### 2026-10-29 — Address DAC-8 review findings
- Issue/PR: [DAC-8](https://linear.app/dacn-khmt/issue/DAC-8/r01-chot-hop-djong-ddpm-partial-reconstruction-va-score); no PR linked.
- Changes: Specified a review-pending SSIM, seed, validation sweep and threshold
  contract and documented the DDPM/AnoDDPM adaptation. Added deterministic
  per-sample sampling seeds and single-sample reconstruction, exposed Hub sample
  UUIDs in memory, fixed AnoDDPM runner metadata compatibility and removed double
  counting of epoch time. Added focused regression tests and agent guidance;
  formatted the existing local notebook for the repository's Ruff CI check.
- Evidence/checks: Full suite passed (61 tests) with CPU and synthetic test data;
  Ruff and config parsing passed. Notebook output and execution-count hash was
  unchanged by formatting. The first full-suite invocation failed two CLI
  tests because the child Python process lacked `PYTHONPATH`; rerunning with
  `PYTHONPATH=src` passed. No real-data training, evaluation or Modal job started.
- Next: A/B review and agree the proposed DAC-8 contract and linked config;
  complete DAC-9 evidence before the bounded DDPM smoke, and DAC-15 before sweep.

### 2026-10-29 — Implement the partial-DDPM trainer and resumable Modal lifecycle
- Issue/PR: DAC-8, DAC-11, DAC-14; no PR opened yet.
- Changes: Added the normal-only Gaussian DDPM epsilon-prediction trainer, a
  review-pending method/config draft, one-based partial reconstruction, EMA
  checkpoint selection from fixed-noise normal validation loss, and private
  validation reconstruction artifacts. Modal now records durable run stages,
  failure class, checksums and epoch checkpoints; DDPM resume requires matching
  source, config, code, trainer and resolved dataset revision. Updated the team
  guide and agent instructions.
- Evidence/checks: Python compilation, Ruff check/format, and `git diff --check`
  passed. No unit test suite or DDPM training/GPU run was started. The method
  contract is still awaiting A/B review; DAC-9 data-access evidence is also a
  prerequisite for the bounded smoke.
- Next: Get A/B review on DAC-8, finish the teammate data gate, then run the
  documented bounded smoke and record its actual manifest.

### 2026-10-29 — Document team Modal workflow
- Issue/PR: No issue or PR linked; branch feat/anoddpm-notebook-training.
- Changes: Added teammate onboarding for the shared Modal Workspace and
  Hugging Face Secret, terminal/notebook smoke workflow guidance, and Modal run
  rules for AI coding agents.
- Evidence/checks: Cross-checked the documented commands and behavior against
  the repository notebook and current Modal CLI, Secrets, and Workspace docs.
  `git diff --check` passed. Documentation-only change; no training run or tests
  were started.
- Next: Share the guide with teammates and use a fresh run ID for each requested
  Modal attempt.

### 2026-10-29 — Add a local Jupyter controller for Modal training
- Issue/PR: No issue or PR linked; branch feat/anoddpm-notebook-training.
- Changes: Added a local Jupyter notebook that submits the existing Modal CLI
  job, streams logs to a cell, saves its result JSON locally, downloads only
  aggregate metrics and manifest from a pinned private Hub revision, and renders
  a local plot. Added notebook dependencies and README/Hugging Face + Modal
  documentation links. The Modal local entrypoint now returns its JSON result
  for the CLI write-result option.
- Evidence/checks: A bounded one-epoch Modal AnoDDPM diagnostic smoke completed
  128 optimizer steps on a Tesla T4 using 128 normal train samples and 16
  validation samples. Artifact revision
  a95b40397aae6f11a7ee863aefbb2613f64a1f47; the local notebook rendered the
  real manifest, metrics tables, and plot from that revision. The initial
  notebook launch trained and uploaded successfully but its post-run parser
  failed on interleaved progress output; a CPU-only Modal probe verified the
  write-result channel, and the saved notebook was rendered from the completed
  smoke artifacts without another GPU run. Notebook validation and code-cell
  execution succeeded; git diff --check passed. No unit test suite was run.
- Next: Use this notebook only for bounded pipeline checks until the agreed
  partial-DDPM trainer and score protocol are implemented.

### 2026-10-29 — Fix the Hugging Face card revision blocker
- Issue/PR: W1 context [DAC-8](https://linear.app/dacn-khmt/issue/DAC-8/r01-chot-hop-djong-ddpm-partial-reconstruction-va-score); branch `feat/anoddpm-notebook-training`.
- Changes: Updated active dataset pins in the handoff, Modal instructions, project tracking, and Colab notebook to the merged Hub card revision `350c74f2313e4e73ca65e7ffc6e9dcf74040b3b7`. Updated the notebook revision assertion, documented the card/revision lesson, and added the verified PowerShell `PYTHONPATH` fallback for Modal's local source mount. No dataset payload was re-uploaded.
- Evidence/checks: Compared the README at the new revision with the generated privacy-safe card; the bytes match. The metadata bytes are unchanged from the prior revision. Modal fetched all 95 files and `validate_hf_export` passed. A bounded diagnostic AnoDDPM smoke completed 128 steps on 128 normal train samples and validated 16 samples; artifacts and metrics are in `RESULT_TRACKING.md`. The first CLI attempt stopped before remote execution because `dacn` was not importable; the retry with `PYTHONPATH=src` succeeded. No unit test suite was run.
- Next: Role C should finish the approved partial-DDPM method contract; this exploratory AnoDDPM smoke does not close DAC-8.

### 2026-10-29 — Review and run the bounded AnoDDPM Modal smoke
- Issue/PR: W1 context [DAC-8](https://linear.app/dacn-khmt/issue/DAC-8/r01-chot-hop-djong-ddpm-partial-reconstruction-va-score); branch `feat/anoddpm-notebook-training` is not linked to an issue/PR.
- Changes: Aligned the Modal Secret name with the existing workspace Secret,
  capped the documented smoke at 128 train and 16 validation samples, clarified
  that Modal jobs use the terminal CLI after hosted Modal Notebooks shut down,
  and recorded the failed run in `RESULT_TRACKING.md`.
- Evidence/checks: Full suite passed (57 tests); Ruff check and format check
  passed; Modal fetched all 95 files from the pinned dataset, then failed in
  `validate_hf_export` because the README differed from the generated
  privacy-safe card. The trainer was not called; no metrics or artifacts were
  produced. The two earlier attempts stopped during image setup before a
  training function ran.
- Next: Reconcile the pinned Hub card with the generated card/validator, then
  rerun this bounded smoke before treating the branch as merge-ready.
### 2026-10-29 — Restore pinned dataset-card compatibility
- Issue/PR: Not provided.
- Changes: Preserved exact validation for the current generated dataset card and
  added the original reviewed schema-v1 card as the only legacy alternative, so
  the immutable pinned dataset remains usable without accepting arbitrary README
  edits. Added positive and tamper-rejection regression tests, documented the
  failed Modal smoke run, and recorded the reusable schema-versioning lesson.
- Evidence/checks: Full Ruff check and format check passed; the complete suite
  passed with 52 tests and 1 skip; `git diff --check` passed. No replacement
  training run was started.
- Next: Retry the bounded A100 smoke run with the same pinned dataset revision.

### 2026-10-29 — Add a Google Colab AnoDDPM training path
- Issue/PR: Not provided.
- Changes: Added a self-contained Colab notebook that clones the feature branch,
  reads `HF_TOKEN` from Colab Secrets, downloads and validates the pinned private
  dataset, runs a bounded smoke test or full pilot, saves to Drive, and renders
  aggregate metric tables and curves. Added deterministic sample limits and live
  trainer progress without changing the full-data defaults.
- Evidence/checks: Ruff check and format check, Python compilation, notebook JSON
  and code-cell validation, and the full test suite passed. No model training or
  measured result was produced; the earlier Modal attempt stopped before GPU
  scheduling because its workspace required a payment method.
- Next: Run `smoke` in a T4 Colab runtime, preserve its artifacts, and only then
  switch the notebook to `pilot` with the full train and validation splits.

### 2026-10-28 — Complete BMAD source and dataset card draft
- Issue/PR: [DAC-7](https://linear.app/dacn-khmt/issue/DAC-7/d01-chot-nguon-djieu-kien-su-dung-va-dataset-card-bmad); [GitHub PR #17](https://github.com/tanh1c/DACN/pull/17); [Hugging Face card PR #1](https://huggingface.co/datasets/dacn-bmad-team/bmad-flair-patient-disjoint-v1/discussions/1).
- Changes: Expanded the generated Hugging Face dataset card with intended use, sample meaning, split policy, preprocessing, limitations, source terms, and required attribution. Documented that BMAD states CC BY-NC-SA without a version and the BraTS2021 page requests three citations but does not specify re-hosting rights; kept license metadata unset and the repo private.
- Evidence/checks: Checked the BMAD repository and official BraTS2021 data/citation page; `git diff --check` passed. Opened and verified HF card PR #1 against the generated README; it changes documentation only. No tests or training run performed.
- Next: Obtain teammate review and merge both PRs; keep DAC-7 open until the accepted card is on the private HF repo.

### 2026-10-28 — Align the Modal Hugging Face secret name
- Issue/PR: Not provided.
- Changes: Updated the Modal runner and setup documentation to use the existing
  `huggingface-secret` secret while retaining `HF_TOKEN` as the environment key.
- Evidence/checks: Ruff check and format check, Python compilation, and the full
  test suite passed. No Modal job or model training run was started.
- Next: Authenticate the local Modal CLI and run the one-epoch AnoDDPM smoke test.

### 2026-10-28 — Add an AnoDDPM training prototype for BMAD
- Issue/PR: Not provided.
- Changes: Ported the attributed AnoDDPM U-Net, diffusion process, and simplex
  noise core; added a validated BMAD shard loader, normal-only full-split
  training, validation-only anomaly scoring and threshold selection, EMA
  checkpoints, a Modal-compatible trainer, and a pilot configuration. Kept this
  prototype explicitly separate from the planned ANDi method.
- Evidence/checks: Ruff check and format check, Python compilation, and the full
  55-test suite passed in an isolated Python 3.11 environment with PyTorch 2.15
  and Numba 0.68; six tests specifically exercised the AnoDDPM integration.
  No medical-data training, GPU job, or measured model result was produced.
- Next: Configure authorized Hugging Face access in Modal, run a one-epoch GPU
  smoke test, and record that actual run separately. ANDi remains pending.

### 2026-10-26 — Verify private Hugging Face dataset and handoff
- Issue/PR: [TRA-95](https://linear.app/traffic-agent-assistant/issue/TRA-95),
  [PR #14](https://github.com/tanh1c/DACN/pull/14).
- Changes: Synced current main and reused the reviewed HF/Modal NPZ exporter.
  Confirmed the team's private dataset already exists; retained its UUID mapping
  and revision `fbb884dfe144be5c71e205a73701507163557ea4`. Added loading instructions
  and safe aggregate provenance in `DATASET_HANDOFF.md`.
- Evidence/checks: Downloaded all 95 remote files, verified their Git/LFS checksums
  and the complete export validator; all 11,248 image/mask/label records matched
  prepared data exactly. Group distributions424/50/767 matched. Current-main suite:
  49 tests passed; Ruff check and format check passed (40 files). No new Hub upload
  or training run was performed; authentication was approved using browser OAuth.
- Follow-up checks: The first PR CI attempt flagged Python-fence formatting in
  the new Markdown guide. Corrected the snippet and reran Ruff check/format
  successfully (41 files); the data verification evidence is unchanged.
- Next: B/C verify the pinned dataset on their own machine and record acceptance
  in TRA-74 before the GPU pilot. Keep held-out test sealed for model selection.

### 2026-10-26 — Review fixes and merge Hugging Face/Modal pipeline
- Issue/PR: [PR #12](https://github.com/tanh1c/DACN/pull/12).
- Changes: Reject duplicate or unreferenced shard offsets; include a hashed
  snapshot of DACN Python sources in each Modal run; align HF examples with the
  group organization. Squash-merged as `c7a6f5f`.
- Evidence/checks: GitHub Actions CI passed: Ruff check/format, 49 pytest cases,
  and the CLI demo. An earlier CI attempt found formatting issues, which were
  corrected before the passing run. No new Modal training run was started.
- Next: Implement the ANDi diffusion trainer and track its actual runs/results.

### 2026-10-26 — Add project requirements and experiment tracking
- Issue/PR: [PR #13](https://github.com/tanh1c/DACN/pull/13).
- Changes: Recorded the Diffusion Models for Anomaly Detection brief; added
  worklog, lessons, and run-result tracking; added AI-agent instructions; linked
  the records from the existing project docs.
- Evidence/checks: `git diff --check` passed. GitHub Actions CI passed Ruff check,
  Ruff format check, `pytest -q`, and the CLI demo. No model training run was
  started.
- Next: Implement and evaluate the diffusion method, then record each actual run
  and its artifacts in `RESULT_TRACKING.md`.

### 2026-10-28 — Initialize the primary DACN Linear workspace
- Project: [DACN — Diffusion Models for Anomaly Detection](https://linear.app/dacn-khmt/project/dacn-diffusion-models-for-anomaly-detection-39ad392101f4).
- Changes: Created the project under DACN-KHMT, six dated milestones, seven workstream labels, 25 issues with acceptance criteria and 31 blocked-by links. Assigned the project lead and data/coordination tasks; left B/C implementation issues unassigned pending identity confirmation. Rebased unfinished first-week dates to 29–30 October while keeping the 4 December target. Synced repository tracking and dataset handoff links to the new DAC issue IDs.
- Evidence/checks: Read-only Linear queries verified the project, target date, six milestones, 25 issues, statuses, and milestone assignments. Current main history contains the merged PR #11 commit used to close F01. No tests, training, or evaluation runs were performed.
- Next: Confirm B/C identities in DAC-6, verify the pinned HF dataset on teammate machines in DAC-9, then run the bounded GPU pilot in DAC-10 and record actual provenance.

### 2026-10-28 — Switch research direction to DDPM reconstruction
- Project: [DACN Linear](https://linear.app/dacn-khmt/project/dacn-diffusion-models-for-anomaly-detection-39ad392101f4); scoring issue [DAC-30](https://linear.app/dacn-khmt/issue/DAC-30/s01-pixel-difference-va-local-ssim-anomaly-maps).
- Changes: Recorded the team decision to use DDPM partial reconstruction and compare starting timesteps with pixel/SSIM scores; moved ANDi and feature difference to optional work. Updated requirements application, plan, protocol, agent guidance, descriptive config and Linear roadmap/issues. Retained dataset split and validation/test controls.
- Evidence/checks: Read project requirements, existing data/evaluation protocol and implementation; queried the updated Linear project and 26 issues. Checked issue dependency graph: 34 links, no cycle or due-date inversion. Git diff whitespace check passed; CSV IDs/dependencies and descriptive YAML parsed successfully. No tests, training or evaluation runs were performed.
- Next: Confirm B/C owners, review DDPM/score and data contracts, prepare run lifecycle recording, then run a bounded pilot and record actual artifacts.

### 2026-10-28 — Confirm team roles and align issue ownership
- Issue/PR: [DAC-6 — Confirm B/C roles](https://linear.app/dacn-khmt/issue/DAC-6/c01-them-thanh-vien-va-chot-vai-tro-bc-tren-linear); [PR #18](https://github.com/tanh1c/DACN/pull/18).
- Changes: Recorded A = Chu Nguyen Tuan Anh (project/data lead), B = Lê Hoàng Chí Vĩ (evaluation), and C = Hai Tuan (diffusion). Confirmed B and C are Linear project members; aligned owner assignments and collaboration notes across the backlog. Updated `TEAM.md`, `PROJECT_PLAN.md`, `PROJECT_TRACKING.md`, and the Linear CSV snapshot.
- Evidence/checks: Linear user search matched both display names; the live project member list included both, and a fresh issue query confirmed the new owners. `git diff --check` passed; the CSV parsed as 26 issue rows. No tests, training, or evaluation runs were performed.
- Next: Review the role/ownership PR, verify private HF dataset access on B/C machines, and agree the weekly sync cadence; DAC-6 and DAC-9 remain In Progress until their criteria are met.

## 2026-11-05 — B integration branch

Created codex/bc-evaluation-colab from C7134b10 in a managed worktree. Integrated B evaluator, diagnostics, UUID artifact export and dataset-binding guards. Resolved runner conflict preserving C sampler/batch/cache options alongside B artifact_dir. Added reviewed aggregate reproduction evidence and retrospective run record; old worktree preserved. Targeted CPU tests:37 passed. Torch runner runtime/GPU remains unverified. No new inference or training. Legacy adapter and new Colab notebook remain pending.

## 2026-11-05 — B spatial diagnostics and grouped comparison

- Converted the checksum-pinned Gaussian30 and Simplex20 legacy caches using
  scripts/convert_legacy_validation_cache.py. Both 83-row UUID artifacts have the
  same SHA-256 as the previously reviewed conversions. Private files stay under runs/.
- Ran both baseline and FG-v1 diagnostics. Both baseline score dictionaries exactly
  reproduce the recorded metrics. Saved aggregate spatial and grouped evidence.
- Added an explicit Simplex20 preset to the grouped evaluator. Its original
  Simplex10 default remains available. Calibration selection is persisted before
  holdout scoring.
- Prepared notebooks/ddpm_validation_colab.ipynb for Gaussian20 at the registered
  checkpoint. Checked Python cell syntax locally. Colab execution remains pending.
- Remaining B work: paired patient-group bootstrap and the Gaussian20 Colab run
  followed by the budget-matched comparison.

## 2026-11-05 — B review: notebook lifecycle and verified cache snapshots

- Reviewed branch base `2df19bc`; details and remaining research limits in
  [B_REVIEW_2026-11-05.md](B_REVIEW_2026-11-05.md).
- Reproduced then fixed notebook smoke/full opt-in, missing failure manifests
  before checkpoint load, and manual-interrupt status. Full inference now has
  a separate default-off gate after successful smoke. Documented cooperative
  timeout and ephemeral Colab storage limits.
- Fusion now deserializes the exact hashed cache/index bytes. Regression tests
  replace producer files after hash verification for both Simplex presets.
  Added synthetic legacy-conversion validation tests, guarded optional Torch
  imports for data-only CI, repaired two invalid UTF-8 separators and lint-only
  notebook/U-Net issues. No model objective, weight or architecture change.
- Verification: full CPU suite with Torch **164 passed (29.84 s)**; minimal
  `[data,dev]` environment **113 passed, 10 skipped (16.27 s)**. Minimal suite
  first encountered a Windows sandbox PermissionError on atomic manifest
  replacement; full rerun outside sandbox passed. Ruff check/format passed;
  notebook cells compiled without saved outputs. Synthetic CLI demo passed.
- Checked 48 saved Precision/Recall/F1 values against aggregate confusion
  counts and confirmed candidate selection uses calibration AP. No new BMAD
  experiment, GPU job, held-out-test access or Linear update.
- Restored local Git metadata from remote while preserving source and backup
  of the missing-parent worktree pointer. Keep `BLK-20261101-02` Investigating;
  next is actual Gaussian20 Colab inference, matched-budget comparison and
  patient-group uncertainty. Tests do not constitute GPU/model acceptance.

## 2026-11-05 — B paired patient-group uncertainty and Colab preparation

- Added exact tied-score weighted AP and paired patient-group bootstrap with fixed
  thresholds/shared draws; tests compare against explicit group replication and
  check missing-class rejection. Added reproducible private-input CLI driver.
- Actual CPU bootstrap: first attempt rejected noncanonical CSV UUID strings;
  second verified canonical UUID/order/cache content and completed1000 draws,
  999valid/1skipped,28.394s. Added aggregate evidence and RESULT_TRACKING record.
  Conditional fusion-minus-Gaussian-CDF AP CI [.005152,.025822]; previously
  inspected validation/different training budgets prohibit a confirmatory claim.
- Colab T4 connected; source archive7a0f024 installed after Git clone access
  failure. User explicitly authorized source/checkpoint/validation transfer.
  Notebook waits for user-entered HF read credential; supplied token creation
  instructions. No actual Gaussian20 inference/training yet.
- Checks: full CPU suite166passed/28.38s; bootstrap CLI help runs. Colab HF
  authentication later returned401 before data/model download; handoff recorded
  as BLK-20261105-02. No credential copied into chat/repository.
