# Medical data and evaluation protocol (must finalize before experiments)

## Dataset decision gate
Choose **one primary modality** (e.g. brain MRI) and one benchmark for the main result; do not mix incompatible modalities into one headline score. Record URL, paper, permission/license, acquisition center, patient count, healthy/anomalous definition, labels, masks and inclusion/exclusion rules. Public access does not imply redistribution rights.

Potential starting points to verify: MOOD challenge brain MRI or abdominal CT for realistic out-of-distribution detection; other appropriately licensed medical anomaly datasets as secondary external validation. Verify latest access rules, split availability and annotations directly with providers.

## Splitting
Split by patient *before* augmentations, patches or slices; group by site/acquisition where possible. No subject, scan, near-duplicate or transformed descendant can cross splits. Keep the test set sealed. The nominal training set must be healthy/normal; document any expected contamination. Select thresholds and hyperparameters only on approved validation data. If no labeled validation anomalies are allowed, specify a clean normal-data calibration procedure before evaluation.

## Reporting
- Main: image/patient AUROC + AUPRC with prevalence, and 95% patient-bootstrap CI.
- With localization masks: pixel AUROC, pixel AUPRC, Dice and/or PRO with a precisely specified aggregation strategy.
- Precision, Recall, F1 at an explicitly locked threshold; report its origin.
- At least three independent seeds if compute permits; runtime, GPU memory, sampling steps and reconstruction variance.
- Report stratified performance by scanner/site/demographics when metadata and approvals permit.
- Fit all normalizations on train only; explicitly control resized resolution and modality/windowing.

**Warning:** near-perfect scores on easy synthetic corruptions do not establish genuine clinical anomaly detection performance. This system is never diagnostic.
