# Single-phase DDIM50 + LPIPS — user-requested test extension, 11 November 2026

User explicitly requests DDIM50+LPIPS on the full test split. Interpret this as
one DDIM chain with50model calls from one-basedt300, eta0, using the same Gaussian
EMAepoch30 checkpoint and original base17/UUID-derived sampling as DDIM20.
This is a distinct experiment from20+30masked inpainting: no pseudo-mask,
stitching, second phase, residual multiplier, score fusion or newly trained model.
No timestep/checkpoint/LPIPS/threshold search on test.

Earlier test results have already been inspected. This secondary experiment is
therefore an exploratory comparison on the same test cohort, not a new untouched
confirmatory test or replacement of the previous frozen protocol. Record every
attempt/failure/resume; preserve earlier results. Do not tune from this test.

## Fixed configuration and inputs

Config:configs/ddim50_lpips_t300.json. Checkpoint SHA256
 d3e06d2082e18234899b201718abd59078c8a800d66da3cc5d8b9e0d9ca9b946.
Dataset revision350c74f2313e4e73ca65e7ffc6e9dcf74040b3b7, patient-disjoint-v1.
LPIPS0.1 SqueezeNet spatial with accepted native weights, grayscale repeatedRGB,
input[-1,1], normalizeFalse, original240crop. Score is raw LPIPS with the existing
numerical floor at0. Image score mean top576pixels. Batch1, fixed seed17 policy.
Initial forward Gaussian noise policy remains the same as DDIM20 at t300.

Capture/hash exact checkpoint, LPIPS trunk/head, sourceZIP, dependency inventory,
config, validation index/registered partition and dataset metadata. Runtime source
bytes/import root/dependency inventory must match captures before any dataset
factory. Use a fresh protocol directory. Keep previous protocol artifacts intact.

## Order before test

1. CPU prepare: capture artifacts and actual source/dependencies; check imports.
2. Bounded GPU smoke2validation images: exactly50U-Net/1LPIPS calls per image,
   finite score maps and reconstructions; store private outputs.
3. Full83validation images, same registered40calibration/43holdout groups. Fit
   separate image/pixel max-F1 thresholds only on40calibration, largest threshold
   on tie. Persist threshold file before43holdout metrics. No method selection
   from holdout; DDIM50 was fixed by the user before this validation.
4. Seal protocol frozen with threshold values and acceptance receipt checksum.
   Test gate requires this exact receipt and matching method/artifact contract.
5. Evaluate all3665test images with fixed thresholds. Report pixel/image ROC-AUC,
   Precision,Recall,F1/AP; pixel microDice,macro lesion Dice and normal pixel FPR.
   No test-data factory before the input/runtime/freeze gates succeed.

## Durability and progress

Private per-sample outputs contain LPIPS map, mask and reconstruction; retaining
reconstruction permits future CPU scoring without silently rerunning diffusion.
Inputs stay in the pinned private dataset cache. Private receipt checks sample
identity/content and output checksum on resume; verify existing samples without
new inference. Print verification progress at1/every100/last cached sample, plus
new-output progress at1/every25/lastsample. Missing logs alone never justify cancel.

Atomic per-image files/receipt; commit Volume every25new outputs and on normal
completion/handled interruption. An abrupt loss can require recomputing up to24
uncommitted images; deterministic sampling/output checks remain enforced. Timeout
8hours, avoiding the prior4hour ceiling. Resume same protocol/stage only with
identical source/config/runtime/thresholds/order/content, and only after prior
writer is terminal. No simultaneous writers for one stage.

Test aggregate ranking metrics use all test pixels with exact ties; never average
AUC/AP over shards. Display ROC/PR curves may be subsampled. Timings include cold
model load per continuation in adapter sums, exclude data/hash/shard/globalmetric
I/O; report scopes and last-process CUDA allocated peak explicitly. Completed
trajectory counts exclude uncommitted/wasted cancellation work.

## Entry point

src/dacn/cloud/modal_ddim50_lpips.py with --stage prepare|smoke|validation|test.
Protocol ID:ddim50-lpips-t300-20261111T064853Z. Test is submitted detached after
validation acceptance; no continuous tracker is re-enabled. Local controller
receives only safe manifests/aggregate metrics; private maps/reconstructions stay
on Volume. Safe aggregate/source/config/LPIPS export handoff uploads only to the
existing private HF model repo. No final metrics are claimed before produced
aggregate and terminal status.

### CPU prepare invocation correction

Initial file-path CLI startup could not import modal_ddim50_lpips with automatic
source mounting disabled; function body did not start and no inference/test data
was accessed. Stopped the repeatedly failing CPU app. Correct invocation uses
`modal run --detach -m dacn.cloud.modal_ddim50_lpips` so remote/local module names
agree while retaining the explicit /opt source root. Fresh actual protocol ID:
ddim50-lpips-t300-20261111T064853Z-b. No scientific config or source change needed.

## Validation acceptance and frozen inputs

Prepare-b and T4 smoke2/full83 completed. Exactly50U-Net/1LPIPS calls per image;
validation153.337264689adapter seconds, median1.754961053,p952.043427624s.
On43holdout:pixel ROC-AUC.934520,AP.186020,Precision.222473,Recall.418106,
F1/Dice.290416; image ROC-AUC.837719,Precision.628571,Recall.916667,F1.745763.
These are validation results, not the new test result.

New thresholds from40calibration:pixel0.23461146652698517,
image0.22034396567485398. Frozen manifest saved in
configs/ddim50_lpips_frozen_20261111.json. Validation aggregate hash matches both
producer manifest and frozen acceptance SHA; actual new source pin
bdc2cfcda87c266cf0092e1c7fdd676021af31245a72b65c9ec913b03086277c.
[Private validation artifacts](https://huggingface.co/dacn-bmad-team/dacn-runs/tree/d496f1c51103d47cf64e897a84308bee11091627/evaluations/ddim50-lpips-t300-20261111T064853Z-b-validation).
No test-based selection or new training. Actual test will use these fixed inputs.

## Terminal test result

Test completed3665/3665 and uploadedcf288576d33ef49b8966a106d0473d4495950164,
terminal2026-11-11T09:01:47.076599Z. No resume/interruption or test-threshold change.
Pixel ROC-AUC.928086,Precision.294173,Recall.327008,F1/Dice.309722,AP.257950.
Image ROC-AUC.756141,Precision.855993,Recall.949091,F1.900141,AP.928450.
Source/threshold provenance and full-precision metrics in terminal evidence;
[LPIPS comparison](LPIPS_COMPARISON_2026-11-11.md) includes old20/20+30 controls.
