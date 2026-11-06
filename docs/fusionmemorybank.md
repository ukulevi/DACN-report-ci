# Task: Implement DDPM + Normal Memory Bank Fusion for Brain MRI Anomaly Detection

## 1. Project Context

Repository: https://github.com/tanh1c/DACN

Reference baseline commit: `7134b105c0c4d4fb620715908f100103b1d0b0ab`

Inspect the actual current repository state before changing code.

The project studies unsupervised brain MRI anomaly detection using diffusion models.

Existing Gaussian DDPM:

- Trained on 7,500 normal MRI slices.
- Single-channel grayscale input, 256×256.
- Normalization [-1, 1].
- Gaussian epsilon-prediction DDPM.
- Linear noise schedule, T=1000.
- Baseline: Gaussian DDPM EMA epoch 30.
- Reconstruction inference starting timestep: t*=300.
- Anomaly maps: absolute difference and local 1-SSIM.

Current performance on the 43-image validation holdout:

- Gaussian DDPM absolute Pixel AP: 0.101066.
- Gaussian DDPM local SSIM Pixel AP: 0.102402.
- Brightness-only Pixel AP: 0.389506.

Previous experiments with global ResNet-18 conditioning, spatial conditioning, Simplex noise and uncertainty scoring have not substantially improved tumor localization.

## 2. Research Objective

**Implement a combined DDPM + Normal Memory Bank anomaly detection pipeline.**

Research question:

Can a feature-based normal memory bank provide complementary information to DDPM reconstruction error and improve brain tumor segmentation?

The proposed method combines:

1. DDPM reconstruction anomaly score.
2. Normal Memory Bank feature-distance anomaly score.

The final output is a fused pixel-level anomaly map.

Do NOT build a standalone PatchCore research project.

Do NOT modify DDPM training or introduce a new diffusion model.

Do NOT use the memory-bank features as conditional inputs to the DDPM U-Net.

The Memory Bank is a separate inference branch whose output is fused with the DDPM anomaly score.

## 3. Required Architecture

Implement the following pipeline:

```text
                     Input Brain MRI
                            |
              +-------------+-------------+
              |                           |
              v                           v
       Gaussian DDPM                ResNet-18
       EMA Epoch 30            Pretrained Encoder
              |                           |
              v                           v
       Reconstruction              Spatial Features
              |                           |
              v                           v
       Residual Score              Normal Memory Bank
              |                           |
              v                           v
      DDPM Anomaly Map             Feature Distance Map
              |                           |
              +-------------+-------------+
                            |
                            v
                  Score Normalization
                            |
                            v
                     Score Fusion
                            |
                            v
                    Final Anomaly Map
                            |
                            v
                   Pixel AP / Dice
```

## 4. Branch A — Existing DDPM

Reuse the existing Gaussian DDPM EMA epoch-30 checkpoint.

Do not retrain or fine-tune it.

Use the existing inference configuration:

- Starting timestep: 300.
- Ancestral reverse diffusion.
- Existing preprocessing, normalization and checkpoint logic.
- Existing deterministic sampling policy.

Prefer reusing the verified cached baseline reconstructions for all 83 validation images.

Do not regenerate DDPM reconstructions when an exact matching cache is available.

Calculate two baseline anomaly scores:

**Absolute difference**

\[
A_D^{abs}(p)=|x(p)-\hat{x}_0(p)|
\]

**Local SSIM difference**

\[
A_D^{ssim}(p)=1-\operatorname{SSIM}_{local}(x,\hat{x}_0)(p)
\]

Preserve the current implementation and score convention.

## 5. Branch B — Normal Memory Bank

### 5.1 Feature Encoder

Use pretrained ResNet-18 with official ImageNet weights.

Adapt ResNet-18 to grayscale MRI:

```python
old_weight = resnet.conv1.weight

new_conv = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)

with torch.no_grad():
    new_conv.weight.copy_(old_weight.mean(dim=1, keepdim=True))

resnet.conv1 = new_conv
```

Use intermediate spatial feature maps rather than the final global pooled feature vector.

Initial feature layers:

- `layer2`
- `layer3`

Verify their actual tensor shapes.

Align feature-map resolutions and concatenate the resulting patch descriptors.

Keep the encoder frozen and in evaluation mode.

Do not train ResNet-18 during this initial fusion experiment.

Do not reuse the previously trained spatial-conditioned DDPM encoder without an explicit new experimental justification.

### 5.2 Build Normal Memory Bank

Use only the 7,500 normal training MRI slices.

```text
Normal MRI
   ↓
Pretrained ResNet-18
   ↓
Spatial Feature Maps
   ↓
Patch Feature Vectors
   ↓
Foreground Filtering
   ↓
Representative Feature Selection
   ↓
Normal Memory Bank
```

Use the existing DACN preprocessing contract.

Apply a documented foreground heuristic, initially `input > -0.95`, to reduce irrelevant black-background descriptors.

This is not an anatomical segmentation mask.

Start with a fixed bank-size budget of 10,000 feature vectors if feasible.

Use deterministic representative sampling with seed 17. Clearly distinguish random sampling from a true PatchCore coreset algorithm.

Save the Memory Bank and all feature-extraction metadata.

Do not use abnormal validation images, tumor masks or test images when building the bank.

### 5.3 Memory Bank Inference

For an MRI test image:

1. Extract spatial features with frozen ResNet-18.
2. Convert the features into patch descriptors.
3. Compare each descriptor with the normal Memory Bank.
4. Calculate the nearest-neighbor Euclidean distance.
5. Convert the patch scores into a spatial anomaly map.
6. Upsample and align this map with the DDPM anomaly map.

The score is:

\[
A_M(p)=\min_{m\in M_{normal}}\|f(x,p)-m\|_2
\]

Higher distance means more anomalous.

Use chunked nearest-neighbor calculations to avoid excessive GPU memory allocation.

Do not require tumor ground-truth masks during inference.

## 6. Fuse DDPM and Memory Bank

Both branches must produce aligned pixel-level anomaly maps for the same MRI.

DDPM score:

\[
A_D(p)
\]

Memory Bank score:

\[
A_M(p)
\]

Their score ranges are different, so do not simply add their raw values.

### Step 1 — Score Normalization

Fit the normalization parameters using normal calibration data only.

Use a fixed, documented robust normalization method, such as calibration-derived median and IQR.

Apply the fitted parameters unchanged to all holdout images.

Do not normalize each test image independently using its tumor labels or use holdout results to fit parameters.

### Step 2 — Fusion

Define:

\[
\boxed{
A_{fusion}(p)=
\alpha\widetilde A_D(p)
+(1-\alpha)\widetilde A_M(p)
}
\]

Where:

- \(\widetilde A_D\): normalized DDPM anomaly map.
- \(\widetilde A_M\): normalized Memory Bank anomaly map.
- \(\alpha\): fusion weight.

Start with a fixed equal-weight fusion:

\[
\alpha=0.5
\]

If fusion-weight optimization is desired, use a small predefined candidate set, for example:

`[0.25, 0.50, 0.75]`

Select the weight only on the 40-image calibration partition.

Do not select alpha based on the 43-image holdout.

Use separate, documented fusion arms for DDPM absolute and DDPM SSIM. Do not silently optimize over both using holdout results.

### Step 3 — Final Anomaly Map

Return:

- Original MRI.
- DDPM reconstruction.
- DDPM anomaly map.
- Memory Bank anomaly map.
- Fused anomaly map.
- Binary predicted mask using a calibration-selected threshold.

## 7. Evaluation Contract

Use the pinned DACN dataset:

`dacn-bmad-team/bmad-flair-patient-disjoint-v1`

Revision:

`350c74f2313e4e73ca65e7ffc6e9dcf74040b3b7`

Evaluation data:

- 83 validation images.
- 39 normal, 44 abnormal.
- Existing 40-image calibration partition.
- Existing 43-image patient-group holdout.
- Preserve the existing patient-group split.

Image preprocessing:

- Original crop: 240×240.
- Padded image: 256×256.
- Pixel metric domain: original 240×240 crop.

Do not access the final held-out test set.

### Metrics

Measure:

- Pixel AUROC.
- Pixel Average Precision.
- Pixel Precision.
- Pixel Recall.
- Pixel F1/Dice.
- Normal pixel false-positive rate.
- Image-level AUROC/AP.

Use calibration-only thresholds.

Also compare false-positive rates at matched recall values where feasible.

Pixel AP must be calculated from continuous scores, not thresholded binary masks.

All anomaly scores must follow:

**Higher score = more anomalous.**

## 8. Required Comparison

Although the deliverable is a combined DDPM + Memory Bank system, evaluate its component scores as controls.

| Method | Purpose |
|---|---|
| Gaussian DDPM absolute | Existing baseline |
| Gaussian DDPM SSIM | Existing baseline |
| Normal Memory Bank score | Measure feature contribution |
| DDPM absolute + Memory Bank | Proposed fusion |
| DDPM SSIM + Memory Bank | Proposed fusion |
| Brightness-only | Existing strong control |

The Memory Bank-only row is a necessary ablation to establish whether fusion provides any benefit beyond its stronger component. It is not a request to develop a separate PatchCore project.

The main result must be the performance of the two fused methods.

Do not claim DDPM fusion is better if the Memory Bank alone performs equally well or better.

## 9. Checks Before Full Evaluation

First inspect the repository and reuse existing components where possible.

Implement unit tests and a small smoke test to verify:

- Correct grayscale encoder weights.
- Correct feature-map dimensions.
- Memory Bank construction from normal training samples only.
- Correct nearest-neighbor distance.
- Correct upsampling and spatial alignment.
- Same normalization and crop contract as DDPM.
- Identical sample ordering across branches.
- Valid and finite anomaly scores.
- Deterministic memory-bank sampling.
- Calibration-only fitting of normalization, fusion weights and thresholds.
- No leakage from lesion masks or final test data.
- Exact baseline-cache provenance and consistency.

Use a bounded smoke test with a small normal-training subset.

Do not select methods or parameters based on smoke-test AP.

Do not run the complete GPU evaluation without explicit authorization.

## 10. Implementation Requirements

Inspect the current codebase before choosing file locations.

Reuse the existing DDPM evaluation pipeline and stored baseline reconstructions.

Implement only the necessary Memory Bank extraction, storage, inference and score fusion modules.

Avoid rewriting existing DDPM architecture, training or reverse sampling code.

Suggested components:

```text
src/dacn/models/memory_bank.py
src/dacn/evaluation/ddpm_memory_fusion.py
src/dacn/training/build_memory_bank.py
configs/ddpm_memory_fusion.json
tests/test_ddpm_memory_fusion.py
```

Adapt these names to actual repository conventions.

Do not overwrite existing baseline artifacts.

Save configuration, dataset revision, encoder provenance, bank checksum, fitted normalization, fusion weight, thresholds and evaluation metrics.

Keep private MRI images, masks and identifiers out of public Git repositories.

## 11. Deliverables

1. Review the current DACN source and identify reusable modules.
2. Implement the two-branch inference pipeline.
3. Build and persist the normal feature memory bank.
4. Implement normalized DDPM + Memory Bank score fusion.
5. Run unit tests and a bounded smoke test.
6. Provide an exact command for full validation.
7. After approval, run full validation and produce comparison results.
8. Save visualization examples showing original MRI, DDPM reconstruction, both individual maps, fused map and ground truth for evaluation only.

Report the following final results:

| Method | Pixel AP | Pixel AUROC | Dice | Normal FPR |
|---|---|---|---|---|
| DDPM | Measured | Measured | Measured | Measured |
| Memory Bank | Measured | Measured | Measured | Measured |
| DDPM + Memory Bank | Measured | Measured | Measured | Measured |

## 12. Important Research Constraints

The goal is to determine whether feature distances complement diffusion reconstruction errors.

Do not claim an improvement before measuring it.

Do not introduce additional encoders, train a new DDPM, perform extensive parameter sweeps or modify the final test set.

The validation set has already been examined repeatedly, so report potential adaptive overfitting and treat improvements as exploratory until independently confirmed.

**Priority: Implement DDPM + Memory Bank fusion only, using existing Gaussian DDPM EMA30 and a frozen pretrained ResNet-18 feature extractor.**

Begin by inspecting the repository and implementing the smallest reliable end-to-end fusion pipeline.