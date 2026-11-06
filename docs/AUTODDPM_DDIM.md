# AutoDDPM adaptation: DDIM acceleration in both phases

**Historical proposal, superseded before full evaluation.** The user subsequently
supplied [testing_Lpips.md](testing_Lpips.md), requiring DDIM20+30 with onepass and
four matched scoring arms. Its current implementation/protocol is
[AUTODDPM_DDIM20_30.md](AUTODDPM_DDIM20_30.md). The10+10x5smoke below completed;
no full run of this older budget was launched. Preserve it as a timing reference.

User-requested 2026-11-06. This is a sampler-speed experiment, not a new mask
design or retrained model. Keep Gaussian DDPM EMA30, Gaussian T1000 linear,
DACN grayscale/padding/crop, frozen LPIPS and existing40/43 group protocol.
No final test, parameter sweep or holdout-based mask redesign.

Sources checked before implementation:
[DDIM paper](https://arxiv.org/abs/2010.02502),
[official DDIM transitions](https://github.com/ermongroup/ddim/blob/main/functions/denoising.py),
[RePaint conditioning/re-sampling concept](https://arxiv.org/abs/2201.09865).
The inpainting composition below is a DACN adaptation, not exact RePaint or
official AutoDDPM reproduction. All measured claims need our own artifacts.

## Fixed sampler settings before the run

- Same original Gaussian EMA30 checkpoint SHA
  `d3e06d2082e18234899b201718abd59078c8a800d66da3cc5d8b9e0d9ca9b946`.
- Phase1 starts one-based300/index299 and uses the existing10-call DDIM control
  indices `[299,269,239,209,179,149,119,89,59,29]`, then clean terminal index-1.
  Initial forward noise matches the base17/sample-UUID-derived Gaussian noise
  of ancestral300. Clip predicted x0 and re-derive epsilon after clipping.
- Phase2 starts one-based50/index49 and uses10reverse indices
  `[49,44,39,34,29,24,19,14,9,4]`, then clean terminal-1. Five repeats per jump,
  so50U-Net calls. Both reverse phases use eta0. These10/10 counts are fixed,
  not selected by looking at holdout scores.
- For reverse `t -> s`, use
  `x_s=sqrt(alpha_bar_s)*clip(x0)+sqrt(1-alpha_bar_s)*epsilon_rederived`,
  with `alpha_bar_clean=1`. This is the eta0 jump; no added reverse variance.
- After stitching at s, repeating the same jump requires a forward transition
  `x_t=sqrt(rho)*x_s+sqrt(1-rho)*z`, where
  `rho=alpha_bar_t/alpha_bar_s`. Do not use a single adjacent beta for a multi-step
  gap. The same equation applies to s=-1 with alpha_bar_s=1.
- Each reverse jump stitches original known context at matching s, with clean
  original at s=-1. Fresh context Gaussian noise and forward re-noising between
  repeats remain stochastic. Eta0 therefore does **not** make the entire masked
  re-sampling chain deterministic without a fixed seed. Isolated base17-derived
  RNG makes repeated runs reproducible and preserves global RNG state.
- Mask=True stays the unknown region. Preserve original exactly outside mask.
  Do not omit repeats, add a new mask rule, change LPIPS or train/fine-tune.

## Fixed comparison arms

| Arm | Phase1 | Phase2 | Algorithm U-Net calls/image |
|---|---|---|---:|
| Ancestral reference | 300ancestral | 50ancestral x5 | 550 |
| Phase2-only acceleration | 300ancestral cached | 10DDIM jumps x5 | 350 |
| Both-phase acceleration | 10DDIM | 10DDIM jumps x5 | 60 |

Full ancestral accuracy comes from immutable completed
`autoddpm-lpips-full83-t4-20261106T061833Z`, rootresultSHA
`6fb277491308c63b2cc52fa614fc912486266a9c3322f7f099aff850cb51a7e0`,
private mapSHA
`f4b37638c5f6b2aca92ecaa072402847e4f18f371b4c4b178483d50ef1f4bce2`.
Verify cache/index SHA, full sample/image/initial/mask/label ordering and LPIPS
state. Timing probes freshly replay ancestral initial+final on firstnormal and
firstabnormal image, requiring maxerror<=1e-6 against those caches.

There is no fourth DDIMphase1+ancestralphase2 arm in this budgeted comparison;
do not describe it as a complete factorial study. Phase2-only vs reference keeps
initials and masks fixed. Changing phase1 changes both reconstruction and mask,
so both-phase vs phase2-only measures that coupled end-to-end change.

The mask remains `normalized95 absolute residual * spatial SqueezeNet LPIPS`,
strict pooled normal-calibration95percentile threshold then3x3dilation. Phase2-only
must reproduce the original numeric threshold and initialheatmapAP. DDIMphase1
gets its own threshold from the same20normal calibration slices/rule because
its reconstruction distribution changes. Do not reuse or tune a threshold on
holdout. All eight score arms/primaryweighted final residual from AUTODDPM_LPIPS
remain reported per sampler pair, with calibration-only score thresholds before
43group-holdout metrics. Input/tumor mask conventions and metrics unchanged.

## Timing and bounded smoke

Theoretical calls reduce550->60, a9.17xNFE ratio, **not measured wall-clock
speedup**. Phase2alone reduces250->50calls, a5xNFE ratio. End-to-end latency also
includes LPIPS, transfers and bookkeeping; report that separately.

Measure synchronized phase1(model+transitions+forward noise/RNG setup) and
phase2(inpainting,context/re-noise) times. Exclude LPIPS/scoring/cache/metric/I-O
from those phase counters. The newly replayed ancestral reference latency is
only2images on the currentT4, not a fresh83image latency estimate. The earlier
715.98s AutoDDPM job reused initial300step caches; it cannot be treated as the
full550call pipeline wall time. Report subset/median/mean/scope explicitly.

Smoke: replay2ancestral pairs, generate DDIMphase1 only for20normalcalibration
plus the selected2images, run both accelerated phase2arms on those2images. No
smoke AP. Fit masks with the same normalcalibration rules. Check finite results,
exact outside-mask preservation, cached ancestral replay, and U-Net hook count
`2*550 + 10*converted_initials + 100*selected_images`.
Full:83DDIM initials and2x83accelerated inpaintings plus the2ancestral timing
probes:10230newU-Net calls. Accuracy comparisons use the registered83images,
40calibration/43group holdout. No extra fullancestral rerun.

## Commands and artifacts

```bash
.venv/bin/modal run --profile phutungcogioihaitan --detach \
  src/dacn/cloud/modal_autoddpm_ddim.py::run_autoddpm_ddim_remote \
  --evaluation-id autoddpm-ddim-smoke-UNIQUE \
  --code-revision GIT_HEAD_SHA --working-tree-dirty --mode smoke
```

After smoke passes use a fresh`autoddpm-ddim-full83-UNIQUE` and`--mode full`.
T4,timeout1hour,team Secret/Volumes,exact source snapshot,private uploads. Store
converted initials,mask parameters,prefixes/maps,metrics,phase timings and actual
U-Net counts;only aggregate JSON downloads locally. Fresh IDs,no prefix resume
claim. Record every actual attempt inRESULT_TRACKING. Full detached run checked
on user request; no continuing sampler/step sweep.

Current request does not replace the independent-confirmation rule in
[PERCEPTUAL_MASK_ROADMAP.md](PERCEPTUAL_MASK_ROADMAP.md). The already inspected
validation remains exploratory; keep final test sealed and do not claim an AP
improvement or an independent mask-method result before measurement.
