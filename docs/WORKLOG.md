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
