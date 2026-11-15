# Method selected for the CO4029 report

On 15 November 2026 the group agreed to report **Final LPIPS × brightness
with sampling seed 17** as the primary method and demo configuration.
The **three-sampling-seed ensemble** is the higher-compute variant. The
original four-arm protocol remains the historical baseline comparison.

- Primary: Gaussian DDPM EMA epoch 30; DDIM20 starting at level 300 and
  masked DDIM30 starting at level 50; eta 0; one phase-two pass. Keep the
  LPIPS × normalized residual pseudo-mask with the normal-calibration
  95th percentile and 3 × 3 dilation. Weight the final spatial SqueezeNet
  LPIPS map by input brightness in [0, 1]. The primary uses 50 U-Net calls
  per slice and the published pixel/image thresholds in
  [postprocess_hybrid.py](../src/dacn/evaluation/postprocess_hybrid.py).
- Variant: average Final LPIPS score maps from sampling seeds 17, 18 and
  19 of the same checkpoint and weight by the same brightness map.
  Use the published per-seed mask thresholds and variant-specific decision
  thresholds in [seed_ensemble.py](../src/dacn/evaluation/seed_ensemble.py).
  This variant uses 150 U-Net calls per slice.
- Evidence: [test results and access history](FINAL_TEST_RESULTS_2026-11-10.md),
  [ablation and retraining evidence](ABLATION_SUMMARY_2026-11-13.md) and
  [handoff](HANDOFF_2026-11-14.md). Existing test values describe an already
  inspected cohort. The decision concerns report and demo presentation.
  Additional experiments, including test reuse, are outside this update.
- BLK-20261113-01 concerns a deferred research direction. Its historical
  record remains open and it is outside the scope of this report update.
