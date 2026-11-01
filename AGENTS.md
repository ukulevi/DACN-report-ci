# Instructions for AI coding agents

Read this file before changing the repository. For research, data, training, or
evaluation work, also read [the project requirements](docs/PROJECT_REQUIREMENTS.md)
and the relevant protocol in `docs/`.

## Project direction

- The project is **Diffusion Models for Anomaly Detection**. Keep diffusion
  modeling and anomaly scoring central to implementation and reporting.
- The current plan is partial DDPM reconstruction for BMAD Brain FLAIR 2D,
  trained on normal slices, with pixel difference and local SSIM anomaly scores.
  DAE is a comparison when feasible; feature difference, ANDi, VAE, and Isolation
  Forest are optional and must not silently expand the agreed scope.
- Higher anomaly scores must mean more anomalous. Keep held-out test data out of
  model selection, threshold selection, calibration, and hyperparameter tuning.
- Distinguish assignment requirements, hypotheses, plans, measured results, and
  limitations. Never describe a planned or synthetic result as a real experiment.

## Project records

- For each completed task that changes repository files, append a concise,
  date-stamped entry to `docs/WORKLOG.md`. Include the issue/PR when known, the
  change, evidence or checks actually performed, and the next step when useful.
  Do not claim checks that were not run.
- Add an entry to `docs/LESSONS_LEARNED.md` only when there is a reusable lesson.
  State the observation/evidence and the resulting practice; do not invent
  lessons to fill the log.
- For every training or evaluation run that actually starts, record its status
  and provenance in `docs/RESULT_TRACKING.md`, including failed and interrupted
  runs. Link the Modal/Hugging Face run manifest and artifacts when available.
  Planned runs are not results, and metric values must come from produced
  artifacts.
- Record ROC-AUC, Precision, Recall, and F1 as required by the project brief.
  Also record project-specific metrics such as pixel AP, Dice, latency, and VRAM
  when applicable. Identify whether each metric is image-level or pixel-level,
  its aggregation, threshold rule, split, and number of seeds.
- Never put tokens, credentials, patient identifiers, source patient paths, or
  private medical data in these records. Use dataset commit/revision and safe
  aggregate provenance only.

## Blockers and team handoff

- When progress is blocked by missing access, an external dependency, a failed
  prerequisite, a required team decision, or an artifact another member must
  provide, create or update the corresponding entry in `docs/ISSUES.md` before
  handing off the task. Do not use the blocker log for routine TODOs, expected
  experiment runtime, or an issue resolved during the same task.
- Use one stable blocker ID and update the existing entry instead of creating
  duplicates. Record its status, owner, affected work, aggregate evidence,
  attempts already made, the exact action needed from the next person, and the
  condition that will close it. Mention the blocker ID in the handoff.
- Keep resolved entries for history and add resolution evidence; do not silently
  delete them. Linear remains the source of task ownership/status, while
  `docs/RESULT_TRACKING.md` remains the source for actual run provenance and
  metrics.
- Apply the same privacy rules as all project records: never put credentials,
  patient identifiers, source patient paths, private images, or per-sample
  derived seeds in `docs/ISSUES.md`.

## Changes and validation

- Follow `CONTRIBUTING.md` and the data/evaluation protocols. Preserve dataset
  split rules and privacy restrictions.
- Keep changes focused. Do not fabricate experiment data or run a costly model
  training job unless the user asked for it.
- Report what changed and which checks were run. If no tests or experiments were
  run, say so explicitly.

## Modal workflow

- Before preparing or running a Modal job, read `docs/HUGGINGFACE_MODAL.md` and
  the relevant requirements and data protocol. Treat Modal as remote compute:
  the local Jupyter notebook submits and displays a job; its local kernel does
  not train the model.
- Use the shared team Modal Workspace so jobs can access the team's named
  Volumes and `huggingface-secret-2` Secret. Check the active Workspace and
  whether the Secret name is present; never inspect, print, log, or commit its
  value. Do not replace a shared Secret. If it is missing or needs rotation,
  explain that a Workspace owner must provision it from a private local `.env`
  file.
- Only start GPU work when the user explicitly asks. Start with the documented
  bounded smoke configuration unless the user specifically requests a larger
  run. Use a fresh run ID and immutable dataset revision for each attempt.
- Record every training/evaluation attempt in `docs/RESULT_TRACKING.md` once it
  starts, including failed and interrupted runs. Update the status from produced
  artifacts; never infer metrics from logs or plans. Keep the held-out test split
  out of trainer inputs and all model/threshold selection.
- The current Modal AnoDDPM path is a diagnostic prototype. Do not describe it
  as the agreed primary partial-DDPM research method. The local Jupyter workflow
  downloads only aggregate run artifacts; do not pull patient-level data into
  the notebook for display.
- The primary DDPM trainer is `dacn.training.ddpm_partial:train`; its method and
  checkpoint-selection contract are still a proposal until DAC-8 receives A/B
  review. Use `configs/ddpm_partial_smoke.json` only for the bounded smoke after
  DAC-8 review and DAC-9 data-access evidence. Do not run a timestep/score sweep
  until B's evaluator and DAC-15 review gate are complete.
- Follow the proposed score, sweep, threshold and seed contract in
  `docs/RESEARCH.md` when implementing B's evaluator; obtain A/B review before
  treating it as frozen. Derive a per-sample sampling seed from the Hub UUID,
  reconstruct one sample per call and reuse that tensor for both score maps.
  Do not publish sample UUIDs or per-sample derived seeds.
- Modal writes a durable start manifest and commits DDPM checkpoints after
  completed epochs. Resume only the same run ID when source snapshot, code,
  config, trainer, and resolved dataset revision match. A fresh attempt needs a
  new run ID; preserve failed/interrupted run records.
