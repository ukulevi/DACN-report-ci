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
