# Gaussian batch8 training: loss convergence audit

Read recorded artifacts on2026-11-02; no retraining, new inference or test use.
Source run: `ddpm-partial-t4-b8-e30-cont-20261102T034644Z`, continued from the
completed one-epoch batch8 pilot. The history contains30 sequential epochs and
938 optimizer steps per epoch, final28,140 steps. Training-history SHA256
`30908ee939a37470ba47549f9d92a607ad1bfeb21600842e40ec12f325888003`
matches the completed run manifest. Dataset/train counts remain7,500 normal
training slices /39 normal validation slices, seed17, EMA0.9992, LR1e-4.

## Recorded values

| Epoch | Train epsilon MSE | Normal-validation epsilon MSE (EMA) |
|---|---:|---:|
| 1 | 0.08006495 | 0.32319729 |
| 5 | 0.01048304 | 0.00885222 |
| 10 | 0.00964573 | 0.00771043 |
| 20 | 0.00925913 | 0.00744353 |
| 25 | 0.00917458 | 0.00738544 |
| 30 | 0.00922359 | 0.00735904 |

Validation MSE reduction: epoch10->20 is3.46%;20->30 is1.14%;25->30 is0.36%.
Train MSE20->30 decreases0.38% with fluctuations. Epoch30 has the best recorded
normal-validation MSE. Epoch29 has a tiny increase over28 (0.00736743 to
0.00736764); there is no sustained late validation-loss rise in this run.

## Interpretation

The curves show effective learning of the noise objective and diminishing
returns late in this30-epoch run. They are approaching a plateau by this metric;
this does not prove complete convergence or that extra training cannot help.
There is no clear overfitting pattern of persistently worsening validation MSE.

Train and validation curves are NOT directly paired generalization losses:
train averages an online changing model over randomly sampled t/noise across
all7,500 samples; validation uses the end-of-epoch EMA with fixed sampled
t/noise over39 samples. Thus validation below training need not indicate a
bug. Initial validation decrease is also consistent with EMA catching up; its
contribution has not been isolated by an online-versus-EMA validation ablation.
Late fluctuations in training loss alone are not evidence of instability.

The objective is noise MSE over the padded256x256 input, including background.
Anomaly metrics instead use cropped240x240 input and tumor annotations. Average
noise MSE can therefore conceal region/timestep-specific errors. No per-region
or per-timestep training loss breakdown is available in this history.

These curves do not establish good healthy reconstruction, tumor removal or
anomaly separation. Low pixel AP remains a separate observed outcome. Therefore
this audit does not justify attributing poor AP simply to insufficient epochs,
nor does it rule out limitations of training objective, data, capacity or sampler.
Next diagnostic: reconstruction error on normal foreground, plus train/inference
contract checks, before allocating substantial additional training budget.

## Artifacts

Generated local (ignored) artifacts:

- `runs/visualizations/ddpm-e30-t300/convergence_v1/loss_curves.png`
- Same directory `loss_curves.pdf` and full30-epoch `loss_history.csv`.

Reproduce using `scripts/plot_training_history.py`; it verifies source checksum,
epoch/step continuity and finite losses, and plots unsmoothed data.
