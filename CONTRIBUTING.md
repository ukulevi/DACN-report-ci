# Contributor workflow

1. Work on one issue per PR. Branch naming: `feat/baselines-ae`, `feat/diffusion-ddpm`, `feat/data-loader`, etc.
2. Do not directly edit `main`; open PR and obtain one teammate review. Merge via squash after CI passes.
3. Shared API: a model scores a batch of data with higher scores meaning more anomalous; pixel anomaly maps are optional. Dataset provides stable anonymous IDs, patient/group IDs, split, image and optional labels/masks.
4. All randomness must be seeded. Save config, git SHA, package versions, selection criterion, epoch and checkpoint identifiers alongside metrics.
5. Select thresholds only using training/validation under a documented protocol; never tune with the test data. Report sample-level and patient-level metrics separately.
6. Do not commit real patient data. Record source permissions and de-identification process. Only use authorized, de-identified datasets.
7. Keep implementation changes and scientific claims separate: improvements require multiple seeds, strong baselines, uncertainty estimates and ablations.
