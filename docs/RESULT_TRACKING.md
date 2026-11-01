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

## Per-run record

Copy this block for every actual run. Use `N/A` with a reason when a metric does
not apply. For a failed or interrupted run, include the error class and last
completed stage rather than fabricating metrics.

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
