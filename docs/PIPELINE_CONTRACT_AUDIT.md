# Gaussian epoch30: train/inference contract and sampler audit

User-requested diagnosis2026-11-02. No production model, preprocessing, loss or
sampler is modified. Fixed four normal-val images at t*=100/300 with shared
initial noise; no tumor AP, training or held-out test experiment in this audit.

## Static checks

| Component | Inspected behavior | Finding |
|---|---|---|
| Weight loader | load_ema_checkpoint loads checkpoint['ema'] strictly, eval mode, frozen parameters | Correct intended path; runtime equality checked by diagnostic |
| Training data | ShardDataset train,1x256x256 float32, values[-1,1]; no augmentation/rescaling in loop | Same data contract as inference |
| Validation input | Same ShardDataset val; forward image sent without extra normalization | No double normalization found |
| Preparation | PNG grayscale, symmetric8px pad, uint8/127.5-1 | Matches documented input; runtime padding/grid checks included |
| Training timestep | torch.randint(0,1000) | Includes0..999 |
| Partial inference | one-based t* -> array index t*-1; iterate to0 inclusive | Consistent indexing |
| Schedule | np.linspace(0.0001,0.02,1000), float64 | Same build_diffusion config for train/inference |
| Target | epsilon MSE, full padded image | Not reconstruction or anomaly loss |
| Mean | x0 from epsilon, clip[-1,1], then posterior coefficients | Standard clipped-x0 parameterization |
| Variance | fixed-large: beta[k] for k>0; no noise added at k=0 | Valid DDPM variant, not inherently a bug |

Reference for fixed-large/small, x0 conversion and posterior coefficients:
[guided-diffusion GaussianDiffusion](https://github.com/openai/guided-diffusion/blob/main/guided_diffusion/gaussian_diffusion.py).
This comparison establishes algebraic consistency, not performance equivalence
to another model/backbone/dataset. Fixed-small is a controlled ablation, not an
automatic correction of an established error.

## Controlled runtime audit

Evaluation `gaussian-e30-contract-audit-20261102T153939Z` on T4,
source checkpoint hashd3e06d2082e18234899b201718abd59078c8a800d66da3cc5d8b9e0d9ca9b946,
dataset revision350c74f2313e4e73ca65e7ffc6e9dcf74040b3b7.
Choose4 of39 normal slices uniformly by seed17; no quality-based selection.
At t100/300 compare clipped noisy input, one-step clean prediction using EMA
and online weights, EMA full reverse with large/small/zero stochastic variance.
All chains use same initial noise and stochastic chains use same reverse draws.
Foreground=input>-0.95, interior>5px from boundary, metrics mean over4 per-image
values. MAE/RMSE on[0,1]; localSSIM uses original range2. This differs from
pooled39-image metrics in NORMAL_RECONSTRUCTION_AUDIT; compare within this run.

Runtime checks include exact tensor equality of loaded EMA versus checkpoint;
epoch/step verification, relative online/EMA parameter difference, training
snapshot equality for unet/data loaders, padding/shape/uint8-grid checks on
four train/four val samples, independent schedule/variance arrays, oracle-noise
x0 inversion and posterior-mean equation checks. Dataset export checksums are
also verified before GPU inference.

Run completed2026-11-02 15:43:10UTC and uploaded private revision
[`f69fc08d`](https://huggingface.co/dacn-bmad-team/dacn-runs/tree/f69fc08d94816d6be82563d8a603a4291ea04b0a/evaluations/gaussian-e30-contract-audit-20261102T153939Z).

## Runtime findings

- Loaded EMA tensors exactly match checkpoint EMA; epoch30/28,140 steps verified.
  Online/EMA relative parameter L2 distance0.031012, so they are distinct states.
  EMA has slightly better one-step SSIM than online, not a major quality gap.
- Training snapshot unet.py, shards.py and bmad.py are byte-identical to current
  modules. Four sampled train inputs/four normal-val inputs passed shape,
  range, padding and uint8-grid checks; export checksums verified remotely.
- Oracle x0 inversion error on the last sample at t100/300 is2.38e-7/9.54e-7,
  each below the asserted3e-6 tolerance on every sample. Direct posterior mean
  agrees exactly on the last sample; all sample checks passed. Linear beta and
  fixed-large/small variance arrays match the analytic reference.
- No weight-loading, input-contract or timestep/formula mismatch was found
  within these checks. This does not prove absence of every possible code/data
  issue or clinical validity of the rendered PNG data.

Mean per-image foreground reconstruction metrics on these same4 normal slices:

| Method | t100 MAE | t100 SSIM | t300 MAE | t300 SSIM |
|---|---:|---:|---:|---:|
| Clipped noisy input | 0.123871 | 0.243683 | 0.269699 | 0.043740 |
| One-step clean estimate, EMA | 0.033427 | 0.714672 | 0.053872 | 0.527959 |
| One-step clean estimate, online | 0.033745 | 0.707330 | 0.054014 | 0.520815 |
| Full fixed-large chain, EMA | 0.046668 | 0.571649 | 0.074614 | 0.358931 |
| Full fixed-small chain, EMA | 0.045906 | 0.582003 | 0.074133 | 0.365356 |
| Full posterior-mean chain, EMA | 0.034406 | 0.708926 | 0.091689 | 0.465686 |

The existing stochastic full chain worsens normal foreground MAE by39.61%
(t100) /38.50%(t300) relative to its own one-step estimate. It is not merely
an EMA mismatch: online one-step metrics are similar. Fixed-small changes MAE
only1.63%/0.65% versus fixed-large. Deterministic mean improves SSIM but at
t300 worsens MAE; it is not a universally better reconstruction choice.
These controlled comparisons support reverse-chain stochasticity/drift and
the denoising/restoration-objective mismatch as contributors, not a proven
single coding bug or an isolated causal explanation of anomaly AP.

At t300, one-step MAE remains0.0539/SSIM0.528, so even the initial learned clean
estimate is imperfect; sampling is not the only remaining limitation. Increasing
noise also amplifies epsilon error into clean-image error by sigma/sqrt(abar):
0.3388 at100 versus1.2339 at300. Last-sample foreground epsilon MSE0.0638 at100
and0.0122 at300 illustrates timestep-dependent error, but is NOT an aggregate
estimate or directly comparable to the39-slice random-t noise-val MSE.

Runtime150.763s, peak allocated VRAM137,601,536B. Results hash
a62f8becf27081f98ec41293cc9d37710a0239d7f98d6e6b76340b544a67ec36.
Private paired image archive hash
b19e4288f6eebccb3662fe52933408e1ef1cf7b42dd9b35240cd8f56615dfd2e.
AP/AUROC/Precision/Recall/F1 are N/A: only four normal images, no anomaly
classification experiment. One sample/seed configuration, no intervals.

## Research optimization priorities

1. Keep the present DDPM as the frozen baseline; do not silently switch its
   sampler/weight based only on normal-image metrics. All AP improvements need
   the same grouped calibration/holdout protocol and brightness control.
2. Before a large retrain, quantify noise/clean-estimate error by timestep and
   foreground/background on normals. Full-image epsilon MSE is dominated in
   pixel count by background and does not establish tissue fidelity.
3. A justified training ablation is foreground-weighted epsilon loss with
   nonzero background weight, using input-derived normal masks (not tumor GT).
   Hold architecture, initialization/optimizer, EMA, data and update budget fixed;
   compare foreground fidelity AND pixel AP. This is a hypothesis, not an
   implemented or proven optimization; no training has been launched.
4. For a larger method change, context-conditioned/patch diffusion targets the
   observed preservation problem more directly than merely increasing epochs.
   Avoid tuning many losses/samplers simultaneously. The weak-mask limitation
   of the preceding AutoDDPM-style adaptation remains a separate problem.

One-step clean prediction is not automatically an improved anomaly score: at
fixed t before clipping its residual is a scaled epsilon error, already a weak
anomaly signal in the denoising study. Better normal fidelity/latency alone must
not be reported as higher tumor AP. The result motivates a controlled change to
training/conditioning rather than another unbounded search over score formulas.

## Local verification

Diagnostic fixed-large chain reproduces the production sampler exactly in unit
tests. Additional tests verify shared noisy input across variance modes, oracle
noise inversion at indices0/99/299/999 and global RNG isolation.18 focused tests
passed across audit, DDPM contract and validation helpers. Ruff/format/compile
and diff checks passed for changed audit/runner files.

## Interpreting the experiment

- Bad one-step prediction on normal inputs suggests the learned denoiser may
  already be limited; inspect online versus EMA and region/timestep errors.
- Much worse full-chain reconstruction than one-step, with better deterministic
  or small-variance results, supports sampling stochasticity/drift as a contributor.
- Oracle inversion failure or loader/checkpoint mismatch would establish a
  concrete contract problem, requiring correction before further research.
- Improved normal reconstruction does not establish better tumor localization.
  Only a separately registered validation comparison can establish AP benefit.
