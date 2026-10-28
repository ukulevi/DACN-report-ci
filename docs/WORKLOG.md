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
