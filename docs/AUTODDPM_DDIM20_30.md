# DDIM20 + DDIM30: matched four-score AutoDDPM ablation

The user supplied [testing_Lpips.md](testing_Lpips.md) while the earlier
10+10x5 smoke was finishing. That specification supersedes the earlier proposed
sampling budget. Preserve that completed smoke as historical timing/provenance;
do not launch its proposed full run. This experiment has **20+30=50 U-Net calls
per output image**, one phase2pass, no hidden five-repeat loop and no retraining.

## Fixed implementation

Config: [autoddpm_ddim20_30.json](../configs/autoddpm_ddim20_30.json), the existing
project-style flat JSON passed via `config_json`. Fixed starting levels300/50,
inference counts20/30, eta0, seed17/sample-UUID policy, onepass, masknormal95th
percentile and3dilation, residual epsilon1e-8. No new configuration framework.
The runner validates this fixed registered preset rather than exposing a sweep.

`ddim_indices(start,count)` rounds a linspace fromstart-1 to0. Exactlycount
strictly descending model indices, inclusive start/index0; clean terminal-1
does not add another model call. Reverse uses existing clipped-x0/re-derived
epsilon convention and the originalT1000linear Gaussian schedule. Phase2uses
the learned EMA30 U-Net at every one of30indices, inserting known original
context at the matching next time. Outside-mask final pixels exactly equalinput.
With onepass there is no repeat re-noising; initial forward noise and stochastic
context draws still follow the isolated fixed-seed policy.

Use same original EMA30/cache/index/model LPIPS checks as AUTODDPM_LPIPS. Phase1
DDIM20 reconstructions are new;mask threshold is refit with the unchanged pooled
95percentile strategy on20normal calibration slices,then frozen. No old ancestral
numeric threshold reuse and no ground-truth masks in pseudo-mask creation.
Heatmap=`LPIPS_initial * clip(L1_initial/(P95(L1_initial)+1e-8),0,1)`.
LPIPS squeeze0.1 weights/input adaptation unchanged,grayscale repeatedRGB on240
crop in[-1,1],normalize=False,numerical nonnegative floor only. No arbitrary
per-method rescaling of LPIPS. This epsilon-denominator spelling follows the
new supplied specification and is recorded explicitly.

Four required rows share the same phase1initial. Rows2–4 share the same single
phase2final tensor and pseudo-mask;the evaluator never regenerates final per score:

1. `ddpm_l1`: L1(input,DDIM20initial),no phase2 for this matchedbaseline.
2. `autoddpm_l1`: L1(input,DDIM30final).
3. `autoddpm_lpips`: LPIPS(input,DDIM30final),no L1 multiplier.
4. `autoddpm_lpips_residual`: LPIPS(input,DDIM30final) times **final** normalized
   residual. It never substitutes the initial pseudo-mask heatmap.

Diagnostic controls:initialDDIM20LPIPS-only,initialproduct,naive-stitchL1 and
brightness. Historical ancestral300LPIPS-only0.266899 and previousAutoDDPM scores
are labeled different-sampler references,not matched outcomes or expected newAP.
Report all four required rows and diagnostic controls,not a holdout-selectedwinner.

Identical83images,40calibration/43patient-group holdout,pixel240cropincluding
background,image top576aggregation,continuouspixelAP. Fit score thresholds on
40calibration,persist allthresholds,before43holdout metrics. Record pixel/image
AUROC/AP/P/R/F1/Dice,normalFPR,mask lesion recall/precision and normal coverage.
No final test,no method selection from repeatedly inspectedholdout. Exact same
reconstruction and scores retained privately for examples/review.

## Calls, timing, and smoke

Instrument both U-Net and LPIPS modules. Assert20phase1calls and30phase2calls
per output image and totalcount. LPIPS has two separate calls per output image
(initial andfinal),excluded from the explicitly named U-Net50budget and recorded
separately. Full83run uses4150U-Net calls and166LPIPS calls. No ancestral reference
chains are secretly run by this experiment.

The2image smoke must additionally generate phase1initials/LPIPS for20normal
calibration images so its mask threshold is fitted correctly. It therefore
reports the extra calibration U-Net calls separately;those are not hidden
phase2repeats. Onlyfirstnormal/firstabnormal getphase2 or outputsmoke scores,
with exactly50calls attributable to each output. No smoke AP.

Measure synchronized phase1 andphase2denoising times and score component times.
Per-arm latency values are **component sums from shared computations**,excluding
transfers,mask assembly,globalcalibration,metrics/cache/I-O and startup;they are
not measured independent end-to-end reruns. Root total includes setup/evaluation
before upload. Historical two-image ancestral550call timing from the completed
10+10x5smoke is byteverified and on the same firstnormal/abnormal pair/T4. Compare
denoising-only scopes and label it a two-image,cross-container timing reference,
not an83image wall-clock speedup. Theoretical550/50=11xNFE is not measuredspeed.

Create private panels showing input,DDIM20initial,DDIM30final,pseudo-mask,initial
LPIPS,finalLPIPS,finalL1,finalproduct. No MRI/masks/identifiers inGit or local
artifact downloads. Only aggregate JSON may be downloaded.

## Commands and authorization

Bounded smoke is authorized by the supplied task and is the next run:

```bash
.venv/bin/modal run --profile phutungcogioihaitan --detach \
  src/dacn/cloud/modal_autoddpm_ddim.py::run_ddim_ablation_remote \
  --evaluation-id autoddpm-ddim2030-smoke-UNIQUE \
  --code-revision GIT_HEAD_SHA --working-tree-dirty --mode smoke \
  --config-json "$(cat configs/autoddpm_ddim20_30.json)"
```

**Full83validation is not authorized by this document alone.** The supplied
task explicitly says: “After a successful smoke and explicit authorization, run
validation on all 83 images.” Wait for that separate user instruction after
reporting concrete smoke results. Then run:

```bash
.venv/bin/modal run --profile phutungcogioihaitan --detach \
  src/dacn/cloud/modal_autoddpm_ddim.py::run_ddim_ablation_remote \
  --evaluation-id autoddpm-ddim2030-full83-UNIQUE \
  --code-revision GIT_HEAD_SHA --working-tree-dirty --mode full \
  --config-json "$(cat configs/autoddpm_ddim20_30.json)"
```

Fresh IDs,teamT4/Volumes/Secret,timeout1hour,exact source and config inmanifest,
private HF uploads. Never overwrite previous baselines. Full metrics pending
until that authorized run actually completes. Record allattempts inRESULT_TRACKING.

## Actual bounded smoke

`autoddpm-ddim2030-smoke-t4-20261106T073737Z` completed/uploaded private revision
`ac8aaaa082fa72363c53253e5db4317f4d4365ea`;rootresultSHA
`aeef3c54e055985edbd579afd3936ae27e098ca4b10a9aae0f729f98ce1f8955`
verified against manifest. Exact dirty source snapshot
`7b76a23f2712aa7f3bb26698aafdf6311ef26e088c4b62fadedd5bacb44869e1`.

- Twooutputs both exactly50U-Netcalls(20+30),onephase2pass,finite scores,
  outside-maskerror0. Samefinal reused forMethods2–4;no AP measured.
- Total480U-Netcalls includes380for19extra normalcalibrationinitials;LPIPScalls23
  separately. Newmaskthreshold0.13344676792621613 fittednormalcalibration only.
- Meanphase1/phase2synchronized denoising1.059434/0.885648s;two-image summean
  1.945082s vs historicalmatchedancestral16.975729s:8.727513xdenoising-only.
  Cross-containerT4/two-image comparison,not full83or end-to-endwall speedup.
- Fourlatency component summeans:1.059616/2.007998/2.014264/2.015014seconds.
  Root total22.787917s before upload,peak104671232bytes. Two private diagnostic
  PNGs and maps saved. No local medical-image download or anatomical review.
- Full suite174tests passed17.66s;targeted Ruff/format/whitespace passed.
  Full83was not started at smoke handoff. The user subsequently authorized it;
  see the actual full-run entry below.

## Authorized full validation

User explicitly approved on2026-11-06 after reviewing successful smoke.
Launched detached`autoddpm-ddim2030-full83-t4-20261106T074200Z`:
[Modal app](https://modal.com/apps/phutungcogioihaitan/main/ap-98lEz64JkA8I5USQcgE4CR).
Same smoke-passed config/source,all83validation,40/43group partition,no training
or finaltest. Fourrequired rows plus diagnostics,exact50U-Net/output and166total
LPIPScalls expected. It completed/uploaded at07:45:33UTC,private revision
`ceba5af6b7cdc46b5404554575add3c5cb09d9c7`. RootresultSHA
`a8bae24bd7270b078d49f923461e43fb51aea3163ede5a2fe71c90a70b127730`
verified against manifest. All83outputs had50U-Netcalls,4150total,166LPIPScalls,
onephase2pass,noextra calibrationcalls,outside-maskerror0,no finaltest.

## Measured results:43group holdout

Thresholds fit on40calibration and frozen. Pixel240cropmicro and normalempty-mask
FPR follow the originalprotocol. Time values are synchronized shared-component
sums as defined above,not end-to-end reruns of each arm.

| Method | Phase1 | Phase2 | Pixel AP | Dice | Normal FPR | Component seconds/image |
|---|---|---|---:|---:|---:|---:|
| DDPM + L1 | DDIM20 | None | 0.090936 | 0.156053 | 0.074325 | 0.588357 |
| AutoDDPM + L1 | DDIM20 | DDIM30 | 0.099569 | 0.200077 | 0.069091 | 1.471042 |
| AutoDDPM + LPIPS | DDIM20 | DDIM30 | 0.312330 | 0.371215 | 0.005772 | 1.477092 |
| AutoDDPM + LPIPS x final residual | DDIM20 | DDIM30 | 0.196812 | 0.272453 | 0.006499 | 1.477929 |

Separate diagnostic and historical controls:

| Control | Sampler/role | Pixel AP | Dice | Image AP |
|---|---|---:|---:|---:|
| Initial LPIPS only | Matched DDIM20 beforephase2 | 0.157876 | 0.246871 | 0.854501 |
| Initial LPIPS x residual | Matched DDIM20 pseudo-mask heatmap | 0.144068 | 0.213808 | 0.858131 |
| Naive stitch L1 | Same mask,without learnedphase2 | 0.099308 | 0.191077 | 0.488216 |
| Brightness | Same images | 0.389506 | 0.268051 | 0.770309 |
| Historical initial LPIPS only | Ancestral300,different sampler | 0.266899 | 0.339668 | 0.868242 |

FinalLPIPSpixelAUROC0.949036,P0.297471,R0.493572,F1/Dice0.371215;
imageAUROC0.914474,AP0.944386,P0.687500,R0.916667,F1=0.785714.
Allother requiredpixel/image metrics are recorded inRESULT_TRACKING.md.

**Matched comparison:** LPIPS onDDIM20initial vsLPIPS on the sharedDDIM30final
increasespixelAP0.157876->0.312330 andDice0.246871->0.371215;calibrationAP also
increases0.238079->0.312952. On the samefinal,LPIPS beats L1andfinalproduct in
pixelAP. No arm was suppressed or reselected based onholdout. Historicalancestral
LPIPS-only0.266899 andoldAutoDDPMweightedscore0.154785 differ in sampler,budget
and/or scoring definition;do not treat those as a one-factor reconstruction test.

**Limits of re-sampling attribution:** naive-stitch L1AP0.099308 is almost the
same as finalL1AP0.099569. A naive-stitch LPIPS control was not part of this run;
the higherfinalLPIPS therefore does not yet separate learnedinpainting benefits
from the perceptual effect of stitching alone. No guarantee of healthy-tissue
replacement. Outside-mask equality is imposed,not a restoration-quality metric.

Mask threshold0.13344676792621613 fromnormalcalibration,unchanged fromsmoke.
Holdout lesionrecall75.9256percent,precision12.3192percent,normalcoverage
12.0276percent;calibration89.3122/10.6485/11.9341percent. Mask misses24.0744percent
of holdoutlesionpixels. NormalFPR for the finalLPIPSscore is0.5772percent atits
separate calibrated threshold,not12.0276percentmaskcoverage. Brightness still
has higherpixelAP0.389506;itsRecall0.788577 vsLPIPS0.493572 prevents claiming
matched-sensitivity superiority from the Dice/FPR figures.

Meanphase1/phase2denoising0.588042/0.874035seconds across83,sum1.462077s/image.
Matchedtwo-image historicalT4denoising comparison9.187524xfaster,not full83
end-to-end acceleration. Root total186.571349s before upload includes setup,
calibration,metrics andI/O;peak104671232B. Privateexamples/maps retained remotely.
Source snapshot unchanged fromsuccessfulsmoke. This is one seed on repeatedly
inspectedvalidation;independent confirmation remains necessary after methodfreeze.
No additional inference,training,mask redesign or held-out test access on check.
