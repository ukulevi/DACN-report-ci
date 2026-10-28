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

## Changes and validation

- Follow `CONTRIBUTING.md` and the data/evaluation protocols. Preserve dataset
  split rules and privacy restrictions.
- Keep changes focused. Do not fabricate experiment data or run a costly model
  training job unless the user asked for it.
- Report what changed and which checks were run. If no tests or experiments were
  run, say so explicitly.
