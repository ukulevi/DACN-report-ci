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
| No verified model result recorded yet. | | | | | | | | | | | | |

## Per-run record

Copy this block for every actual run. Use `N/A` with a reason when a metric does
not apply. For a failed or interrupted run, include the error class and last
completed stage rather than fabricating metrics.

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
