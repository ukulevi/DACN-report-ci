# Lessons learned

### 2026-11-07 — Distinguish sampler configuration from trainer architecture

- Evidence: acceptance preflight stopped at config checksum when sampler JSON
  was treated as trainer settings; both files are valid JSON for different roles.
- Practice: validate sampler preset against mask parameters; load architecture
  and diffusion schedule from the hashed full-state checkpoint. Test the actual
  config passed into the EMA loader, not just whether loading was invoked.

### 2026-10-28 — Record run lifecycle before GPU work

- Observation/evidence: the Modal runner writes `run_manifest.json` after the
  trainer returns; an exception or abrupt stop can occur before that write.
- Practice: establish a durable started record and failure/interruption handling
  before pilots, with a documented manual fallback for abrupt termination.

Record reusable lessons supported by repository evidence or an actual run. Keep
plans and hypotheses out of this log unless they have become a lesson; link the
source or run when possible. Append new entries at the top.

## Entry format

```text
### YYYY-MM-DD — short lesson
- Observation/evidence:
- Lesson:
- Practice adopted:
- Source/run:
```

## Entries

### 2026-11-04 — Modal preemption can replay a fresh-run input
- Observation/evidence: Simplex continuation reached durable epoch18, then Modal
  preempted its container during epoch19 and replayed the original no-resume
  request. Directory protection correctly rejected the existing run; the app
  stopped while its last durable manifest still said running.
- Lesson: Detached execution and per-epoch checkpoints preserve progress, but
  do not make a fresh-run invocation automatically resumable after preemption.
- Practice adopted: Confirm the old app has stopped, inspect the latest durable
  checkpoint, verify identical source/config, and reissue same run with --resume.
  An initial continuation creates a new run; recovery of that run uses resume
  without continue-from-run-id. Keep the directory-protection check intact.
- Source/run: ddpm-simplex-t4-b8-e20-cont-20261104T070341Z; logs show preemption
  2026-11-04 08:18:43UTC and the resulting output-already-exists error.

### 2026-10-29 — Reserve runner metadata in trainer output directories
- Observation/evidence: the Modal runner writes its manifest and source snapshot
  before invoking the trainer, while AnoDDPM rejected every non-empty output
  directory. The focused regression test exercises this boundary.
- Lesson: A shared runner and trainer need an explicit agreement about files
  created before training.
- Practice adopted: Both diffusion trainers allow the runner's two reserved
  metadata files and reject other pre-existing artifacts for fresh runs.
- Source: `src/dacn/cloud/modal_train.py`, `src/dacn/training/anoddpm.py`, and
  `tests/test_dac8_contract.py`.

### 2026-10-29 — Handle shard batch tuple shape explicitly
- Observation/evidence: `ShardDataset` yields `(image, label)` when masks are
  disabled and `(image, mask, label)` when they are enabled. DDPM validation
  needs labels for stratified sample selection but its noise-prediction loss
  consumes only images.
- Lesson: A shared dataset can expose different batch shapes by split/options;
  trainer loops should select the intended tensor explicitly.
- Practice adopted: DDPM validation reads `batch[0]` for image input and keeps
  masks out of the objective.
- Source/run: `src/dacn/data/shards.py` and
  `src/dacn/training/ddpm_partial.py`; discovered by source inspection, not a
  training run.

### 2026-10-29 — Use a result file for notebook-launched Modal jobs
- Observation/evidence: Modal progress output interleaved with the local
  entrypoint's pretty-printed JSON, so parsing the combined terminal stream
  failed after GPU training and Hub upload had completed. The Modal CLI
  write-result option reliably wrote the local entrypoint's string return value
  in a CPU-only probe.
- Lesson: Human-readable progress output is not a stable machine-readable
  result channel.
- Practice adopted: Return structured JSON from the local entrypoint and use
  Modal CLI write-result for notebook orchestration; keep streaming logs
  separate from result parsing.
- Source/run: docs/HUGGINGFACE_MODAL.md, notebooks/anoddpm_modal_local.ipynb,
  and run anoddpm-local-smoke-20261029T055620Z-b71e.

### 2026-10-29 — Pin the dataset card revision used by the validator
- Observation/evidence: The old pinned Hub revision had the pre-review README,
  while the merged card update at `350c74f2313e4e73ca65e7ffc6e9dcf74040b3b7`
  matched the generated privacy-safe card exactly. With the old pin, the bounded
  Modal smoke stopped at card validation; after pinning the merged revision, all
  95 files validated and the bounded smoke completed.
- Lesson: A dataset revision includes its README; validating the same data with
  an older card revision can fail even after the corrected card is merged.
- Practice adopted: Pin the reviewed Hub commit in Modal, notebooks, and handoff
  instructions; keep historical run records on the revision they actually used.
- Source/run: `DATASET_HANDOFF.md`, `RESULT_TRACKING.md`, and Modal smoke
  `anoddpm-smoke-20261029T0518Z`.

### 2026-10-29 — Bound smoke runs by samples and inference steps
- Observation/evidence: The one-epoch Modal smoke command initially had no sample
  limits and would have used the full 7,500-slice train and 83-slice validation
  sets. The bounded 128/16 attempt stopped at dataset-card validation before
  training.
- Lesson: An epoch count alone does not bound smoke-test compute or cost.
- Practice adopted: Set explicit train and validation sample limits and a short
  reconstruction chain for pipeline smoke tests; label their metrics as
  diagnostics, not research results.
- Source/run: `docs/HUGGINGFACE_MODAL.md`, run
  `anoddpm-smoke-20261029T0518Z` in `RESULT_TRACKING.md`.
### 2026-10-29 — Preserve validators for immutable generated metadata

- Observation/evidence: The pinned dataset revision passed its original full
  validation, but a later documentation-only expansion of the generated card
  caused Modal validation to fail before training because both cards used schema
  version 1 and the validator recognized only the newest exact text.
- Lesson: Exact validation of generated metadata is safe only when historical
  generators remain recognized or the format version changes with the template.
- Practice adopted: Generate new exports with the current card, accept only the
  two exact reviewed schema-v1 templates for existing exports, and keep rejecting
  arbitrary README changes.
- Source/run: `anoddpm-a100-smoke-20261029-01`, `src/dacn/data/hub.py`, and
  `tests/test_hub.py`.

### 2026-10-28 — Port the algorithm core, not dataset-specific epoch semantics
- Observation/evidence: Upstream AnoDDPM loads NFBS/Edinburgh volume layouts
  and limits an epoch to 100 sampled images, while this project uses indexed
  BMAD NPZ shards and requires auditable full-split coverage.
- Lesson: Reusing a research implementation safely means preserving the model
  and diffusion mathematics while adapting its data and training boundaries to
  the target protocol.
- Practice adopted: Use the validated shared shard loader, define an epoch over
  the complete training split, keep test data out of the trainer, and name the
  prototype AnoDDPM rather than conflating it with ANDi.
- Source/run: `src/dacn/data/shards.py`, `src/dacn/training/anoddpm.py`, and the
  AnoDDPM integration tests in `tests/test_anoddpm.py`.

### 2026-10-26 — Pin and verify an existing Hub dataset before uploading again
- Observation/evidence: The organization already contained a private dataset with
  the same archive and canonical manifest hashes. All downloaded arrays matched
  local prepared data. Rebuilding the exporter generates fresh random UUIDs.
- Lesson: A second upload is unnecessary when the pinned existing dataset is
  correct; it would create another revision and change pseudonymous group IDs.
- Practice adopted: Inspect the target first, verify its full snapshot and hashes,
  and preserve its revision for handoff. Use canonical LF manifest hashes when
  comparing exporter metadata across platforms; keep file-byte hashes distinct.
- Source/run: `DATASET_HANDOFF.md`, HF revision
  `fbb884dfe144be5c71e205a73701507163557ea4`.

### 2026-10-26 — Keep plans separate from measured findings
- Observation/evidence: The repository documents an ANDi-FLAIR plan, but the
  README states that the diffusion trainer and measured model-performance
  results are not yet available.
- Lesson: A planned method, hypothesis, demo, or synthetic metric is not evidence
  of model performance.
- Practice adopted: Record real run metrics only from produced artifacts in
  `RESULT_TRACKING.md`; label smoke tests and synthetic outputs explicitly.
- Source/run: `README.md`, `docs/PROJECT_PLAN.md`, and `docs/RESEARCH.md`.

## 2026-11-10 — Reproduce the bytes of cross-platform config artifacts

Observed: candidate config pin1d731fe0… is the CRLF serialization from Windows;
Git checkout on macOS contains LF bytes with hashf1eaf46f…. Converting only line
endings reproduces the historical checksum exactly. Prepare correctly stopped
before inference. Practice: retain exact artifact bytes, and only recover LF/CRLF
variants when the historical hash matches; never repin a config merely to bypass
an integrity error. Regression verifies other content changes remain rejected.

## 2026-11-10 — Silent resume verification is not a hung inference job

Observed: resume verifies all existing sample identities/content and shard hashes
before its first new inference, but emits no progress for skipped shards. GPU
idle, absent logs and a PENDING call-graph response did not establish a hang:
the user subsequently supplied LPIPS initialization logs. An operator cancellation
was sent too early. Practice: distinguish silent integrity verification from GPU
inference; do not cancel solely from idle GPU or call-graph status. Check durable
receipt changes and actual terminal errors; allow the known verification phase.
