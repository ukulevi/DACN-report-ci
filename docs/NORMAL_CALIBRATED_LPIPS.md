# Normal-calibrated LPIPS layers on cached final DDIM20+30

User authorized running the prioritized direction on2026-11-06 after
[the encoder review](LPIPS_ENCODER_RESEARCH.md). This first implementation tests
statistical layer calibration with the same SqueezeNet, not RAD-DINO or a new
MRI encoder. No training, changed diffusion weights, reverse calls or new masks.
Start bounded smoke, then full83 with the same fixed settings if smoke passes.

## Fixed protocol before execution

- Immutable source: `autoddpm-ddim2030-full83-t4-20261106T074200Z`, private
  revision`ceba5af6b7cdc46b5404554575add3c5cb09d9c7`, rootresultsSHA
  `a8bae24bd7270b078d49f923461e43fb51aea3163ede5a2fe71c90a70b127730`;
  comparison/private_ablation.npz SHA
  `22656e1f63a0047d3dbe32f7833eea8e303988414e5942dec1ba5a1d9ed1ef19`.
  Verify full83ordered image/mask/label content against pinned validation/index.
- Reuse input,final,initial and originalpseudo-mask tensors from that cache.
  LPIPS0.1.4,net=squeeze,version0.1,frozen/eval,spatial=True,identical240crop,
  grayscale repeatedRGB in[-1,1],normalize=False. StateSHA
  `18fed4d1fc5c04f2918f64df3a2463344108576d1888f0d75538ca6d5c4dffe6`.
- Request `retPerLayer=True`:capture all7pretrained per-layer spatial
  contributions including their learned perceptual heads. Require sum replay
  <=1e-6 and returned aggregate replay against cachedfinalLPIPS<=1e-7.
  Baseline score is returned aggregate with the established nonnegative floor.
- First extract the20normal calibration finalpairs. For eachlayer,fit medianm
  andIQRs on input foreground (input > -0.95 heuristic), pooled over
  normalcalibration pixels. No tumor annotations in fitting. Scale=max(IQR,1e-6),
  record rawIQR/floorflags and normalforeground count. Persist statistics before
  extracting remainingselected finalpairs. No layer selection or learned weights.
- Primarynewscore `mean_l((d_l-m_l)/s_l)`,all7weights=1/7. No positive clamp,
  per-image normalization,hardbackground gating or new anatomic prior. Apply to
  entire240crop including background. Foreground is only a fitting heuristic.

Four fixed scoring arms:

| Arm | Purpose |
|---|---|
| original_lpips | Reproduce existing finalDDIM20+30LPIPS PixelAP0.312330 |
| uncalibrated_equal_mean | Originalaggregate/7;positive-scale ranking control |
| normal_calibrated_equal_mean | Primarynormal-only per-layer median/IQR calibration |
| naive_stitch_lpips | Diagnostic of inpainting vs simple original-mask stitching |

Naive stitching uses originalinput outside the unchangedmask and originalDDIM20
initial inside it. Compute its rawLPIPS againstinput,without calling U-Net or
altering the finalpairs used by the calibration comparison. This closes a missing
perceptual control identified in the preceding report;do not tune masks from it.
No per-layer AP sweep or holdout-based winner selection.

## Evaluation, privacy, and scope

Same patient-disjoint dataset revision
`350c74f2313e4e73ca65e7ffc6e9dcf74040b3b7`,83validationimages,firstSGKF2 with
shuffle/random_state17,40calibration/43group holdout. Scorethresholds pixel/image
maxF1 fit on40calibration for eacharm,stored before holdoutmetrics. Normal layer
statistics use20normalcalibration only. Held-out test never opened. Continuous
pixelAP/globalmicro240crop,image mean top576,record AUROC/AP/P/R/F1/Dice andnormal
pixelFPR. Different calibratedrecalls prevent an automatic matched-FPR claim.

Fullrun must reproducebaselineAP and meancontrolAP within1e-10 of immutable
reference. Allfinals/pseudo-masks/weights remain fixed;any accuracy change in
the calibrationarm concerns scorecomposition,not improved reconstruction.
Layer normalization may amplify nearlyconstant layers or suppress useful lesion
signals. An inspected validation gain is exploratory,not independentconfirmation.

Smoke:2outputpairs(firstnormal/firstabnormal),plus20normalcalibration finalpairs
tofit statistics. Report exact replay,finitecalibrated maps,layerfloors/calls;
no AP or F1 on smoke. Full:83finalpairs+83naivestitchpairs=166LPIPSpair calls,
zeroDDPMcalls. No new modelweights downloaded. Fitparameters/arrays/maps and
4fixednormal/abnormalcalibration/holdout panels remain on teamVolumes/privateHF;
only aggregateJSON downloaded locally. Panel color scales are visual aids,
not equal score ranges or anatomical diagnoses.

Implementation: evaluation/calibrated_lpips.py and cloud/modal_calibrated_lpips.py,
using the existing metric/split/data/LPIPS infrastructure. Fresh IDs only,T4,
30minute timeout,source snapshot and private artifactchecksums.

```bash
.venv/bin/modal run --profile phutungcogioihaitan --detach \
  src/dacn/cloud/modal_calibrated_lpips.py::run_calibrated_lpips_remote \
  --evaluation-id lpips-normalcal-smoke-UNIQUE \
  --code-revision GIT_HEAD_SHA --working-tree-dirty --mode smoke
```

After successfulsmoke,use fresh`lpips-normalcal-full83-UNIQUE` and`--mode full`.
The user authorized this scoring trial; no extra training or other encoder arms
are included. Record allactual attempts inRESULT_TRACKING and check the detached
fullrun on user request rather than continuous monitoring.

## Observed outcome — 2026-11-06

Full83 completed and uploaded; see [run record and metrics](RESULT_TRACKING.md).
On43group-holdout images, original LPIPS PixelAP .312330164/F1 .371214628;
normal-calibrated score AP .307159662/F1 .373649189. The primary AP hypothesis
was not supported. Lower normal FPR (.005772113→.005152595) accompanies lower
lesion recall (.493572019→.465310446), so this is not a matched-recall improvement.
Naive-stitch LPIPS AP .180593333 is below final re-sampling LPIPS .312330164,
supporting a re-sampling contribution in this fixed pipeline. Keep original
LPIPS as reference; no layer/weight sweep or new encoder was run after inspection.
