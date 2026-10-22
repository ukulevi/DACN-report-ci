# Research plan: diffusion anomaly detection

## Foundational reading
- DDPM: https://arxiv.org/abs/2006.11239
- DDIM: https://arxiv.org/abs/2010.02502
- Score matching: https://arxiv.org/abs/1907.05600
- Score-based SDEs: https://arxiv.org/abs/2011.13456

## Candidate hypothesis, not a validated novelty claim
**Region-consistency / noise-scale agreement for medical anomaly localization**: learn a healthy distribution and compare residual / denoising-score consistency at multiple timestep bands; optionally restrict updates using anatomy-aware features. Hypothesis: anomalous regions yield unusually unstable score trajectories relative to normal tissue, improving robustness over single-time reconstruction error.

Do not begin by assuming novelty. Conduct a literature review of medical diffusion anomaly detection, multiscale residual scoring, masked restoration and score variance. Define mathematical score, train/test sampling, calibration and ablation independently of code.

## Experimental matrix
- Baselines: Isolation Forest (fixed non-leaking embeddings), AE, VAE, GAN (clearly define GAN anomaly scoring).
- Diffusion: DDPM denoising error; DDIM inversion/reconstruction at matched compute budgets; multiple timestep aggregation.
- Proposed method: consistent backbone and training samples; score-only change vs architectural change; ablate each component, steps, seeds and compute.
- Negative controls: random masks, noise levels, intensity augmentation, image size and score calibration.

## Decision gates
Gate A: verify data licensing/splits. Gate B: reproduce reliable baselines. Gate C: confirm diffusion signal and failure cases. Gate D: freeze new algorithm design and ablation. Gate E: evaluate once on sealed test set, write findings including negative results.
