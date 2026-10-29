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
| No validated research model result recorded yet; diagnostic smoke runs are not research results. | | | | | | | | | | | | |

## Per-run record

Copy this block for every actual run. Use `N/A` with a reason when a metric does
not apply. For a failed or interrupted run, include the error class and last
completed stage rather than fabricating metrics.

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
