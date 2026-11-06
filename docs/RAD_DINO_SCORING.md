# Frozen RAD-DINO spatial distance — fixed trial 2026-11-06

User explicitly requested RAD-DINO after the negative normal-calibrated LPIPS
trial. Run bounded smoke then full83 with unchanged settings if smoke passes.
No encoder/DDPM training or new reconstruction; use the same Gaussian EMA30
final DDIM20+30 pairs as [the LPIPS calibration trial](NORMAL_CALIBRATED_LPIPS.md).

## Registered design before GPU execution

- Encoder: [microsoft/rad-dino](https://huggingface.co/microsoft/rad-dino), MIT,
  revision `110cbc18d5133582e320b43d53bf5c44e410c936`. Official public model is
  chest-X-ray pretrained, not MRI. Download only model.safetensors, config,
  processor, README/license; retain hashes in run manifest. No remote code.
- Transformers4.46.3 AutoModel/Dinov2Model, strict loading-info check; frozen/eval,
  float32 eager attention, batch of input/reconstruction pair, no AMP/TF32.
- Crop padding8 → full240square. Both input and reconstruction independently
  round fixed `127.5*(x+1)` to uint8, repeat grayscaleRGB, official Bit processor:
  bicubic shortest-edge518 and center-crop518 (no field-of-view loss for square
  input), rescale1/255, mean .5307/std .2583 all channels. No per-image min-max.
  Quantization/preprocessing differs from float LPIPS and is a declared confound.
- Fixed blocks3/6/9/12 hidden states, apply shared encoder final LayerNorm at each
  tap, discard CLS,37x37 patch tokens,768channels. Channel unit vectors using
  `f/sqrt(sum(f²)+1e-10)`; per-patch mean-channel squared vector difference,
  bilinear align_corners=False upsample to240. Equal mean across four layers.
- Primary score: raw equal mean. Secondary fixed diagnostic: per-layer
  median/IQR fit on20normal calibration foreground pixels (`input > -.95`),
  scale floor1e-6, equal mean without clipping/gating. Freeze parameters before
  remaining extraction. Do not choose layers/weights based on AP.
- Unchanged cached final SqueezeNet LPIPS control, expected holdout PixelAP
  .3123301640701585. New distance is not original LPIPS/no learned perceptual head.
  No DINOv2 control run in this request: cannot attribute differences separately
  to architecture, medical pretraining, preprocessing or perceptual calibration.

## Data, checks and reporting

Same pinned dataset/index/source cache hashes and exact image/mask/label ordering
checks as the calibration trial.83validation, SGKF2 shuffle17 first40calibration/
43patient-group holdout. Thresholds maxF1 on40, largest on ties, persisted before
holdout metrics. Pixel global micro over full240crop, image mean top576pixels;
record AP, AUROC, P/R/F1/Dice and normal FPR. Held-out test not accessed.

Smoke: firstnormal+firstabnormal pair and20normalcalibration, plus one identical
pair sanity check (zero feature distance); no smoke AP. Full83pair calls,0DDPM.
Shapes/finite values/unit-distance alignment checked; scale floor activation
reported. Private feature maps on Volume/privateHF only. Aggregate JSON local.
T4,30min timeout, fresh evaluation IDs and source snapshot for every attempt.
No automatic additional encoder/layer sweeps after inspecting results.

Official sources: [processor](https://huggingface.co/microsoft/rad-dino/blob/110cbc18d5133582e320b43d53bf5c44e410c936/preprocessor_config.json),
[config](https://huggingface.co/microsoft/rad-dino/blob/110cbc18d5133582e320b43d53bf5c44e410c936/config.json).

## Outcome — 2026-11-06

Full83 succeeded, strict checkpoint load, zero identical-pair error, zero DDPM
calls, and private artifact upload. On43group-holdout images, raw RAD-DINO
PixelAP .289211335/Dice .288010044; normal-calibrated PixelAP .284123511;
SqueezeNet LPIPS control .312330164/Dice .371214628. Raw RAD-DINO image AP
.948820 vs LPIPS .944386, but pixel FPR .030442 vs .005772. Do not claim
segmentation improvement. Full metrics and threshold details are in RESULT_TRACKING.
