# Task: Implement ResNet-18 Feature-Conditioned DDPM for Brain MRI Anomaly Detection

## 1. Project context

Repository: https://github.com/tanh1c/DACN

Reference commit: `7134b105c0c4d4fb620715908f100103b1d0b0ab`

This project investigates unsupervised brain MRI anomaly detection using diffusion models.

The core idea is to train DDPM on normal brain MRI slices, then reconstruct potentially abnormal MRI images during inference.

The anomaly map is obtained by comparing the original MRI with its reconstruction.

Our main problem is that the existing Gaussian DDPM produces substantial reconstruction errors in normal brain tissue, resulting in many false-positive pixels.

**Research objective:** Investigate whether feature conditioning using a ResNet-18 encoder can improve normal-tissue preservation and tumor anomaly localization.

Do not assume that adding an encoder automatically improves anomaly detection.

## 2. Existing implementation

The repository already contains:

- `src/dacn/models/diffusion/unet.py`: Current diffusion U-Net.
- `src/dacn/models/diffusion/process.py`: GaussianDiffusion implementation.
- `src/dacn/training/ddpm_partial.py`: Normal-only DDPM training.
- `src/dacn/training/patch_context.py`: Previous patch/context-conditioned experiment.
- `src/dacn/evaluation/ddpm_validation.py`: Reconstruction evaluation.
- `src/dacn/evaluation/validation.py`: Pixel and image anomaly metrics.

Current Gaussian DDPM configuration:

| Parameter | Value |
|---|---|
| Image size | 256 × 256 |
| Image channels | 1 |
| Image normalization | [-1, 1] |
| Training images | 7,500 normal slices |
| Normal validation images | 39 |
| Full validation | 83 slices |
| Diffusion timesteps | 1,000 |
| Beta schedule | Linear |
| Training objective | Epsilon prediction |
| Loss | MSE |
| Backbone | Existing diffusion U-Net |
| Baseline checkpoint | Gaussian DDPM EMA epoch 30 |
| Main inference timestep | 300 |

The baseline Gaussian DDPM at epoch 30 achieved approximately:

- Pixel AUROC: 0.8956
- Pixel AP: 0.1007
- Pixel F1/Dice: 0.1675

These are local-SSIM anomaly scores evaluated on the existing 83-slice validation set, not test results.

Do not modify or overwrite existing baseline checkpoints.

## 3. Problem with the previous Patch/Context DDPM

The repository previously attempted patch-conditioned DDPM.

That method expanded the U-Net input from 1 channel to 3 channels:

1. Noisy image.
2. Clean context outside a patch.
3. Binary patch mask.

The model was fine-tuned for 5 epochs.

The previous method did not improve validation Pixel AP.

One possible explanation is that tumor information remains in the noisy patch or surrounding context, allowing the model to reconstruct abnormal tissue.

This explanation is a hypothesis, not a confirmed root cause.

**The new experiment must use encoder-derived feature conditioning, not the previous three-channel patch-conditioning design.**

## 4. Proposed architecture: ResNet-18 Conditional DDPM

Implement a separate model variant with two branches.

### Branch A — ResNet-18 feature encoder

Input:

`x0: [B, 1, 256, 256]`

Use a ResNet-18 encoder adapted for one-channel grayscale MRI.

Replace its classification head with an appropriate feature extraction and projection module.

Suggested shapes:

- ResNet-18 pooled features: `[B, 512]`
- Linear projection: `[B, 128]`
- Final condition: `c = f_phi(x0)`

The encoder must not be used as a tumor classifier.

The condition vector represents learned MRI features.

### Branch B — Existing Gaussian DDPM

Keep the original DDPM forward process:

`x_t = sqrt(alpha_bar_t) * x0 + sqrt(1-alpha_bar_t) * epsilon`

where epsilon is standard Gaussian noise.

Modify the U-Net to accept:

`predicted_noise = unet(x_t, timestep, condition)`

The U-Net must continue to output:

`[B, 1, 256, 256]`

Do not change the output to a tumor mask.

## 5. How conditioning should work

Inspect the current timestep embedding and residual blocks in:

`src/dacn/models/diffusion/unet.py`

Implement feature conditioning using an embedding projection.

Conceptually:

- `time_embedding = TimeEmbedding(t)`
- `condition_embedding = ConditionProjection(c)`
- `combined_embedding = time_embedding + condition_embedding`

Pass the combined embedding through the existing time-conditioned residual blocks.

The projection output must match the dimension of the U-Net timestep embedding; do not assume 128 is the final embedding dimension.

Initialize the condition injection so that the initial conditional U-Net behaves identically to the original unconditional checkpoint.

Preserve the original image input channel.

Prefer a minimal, backward-compatible architecture change rather than rewriting the U-Net.

## 6. Model initialization and encoder pretraining

Initialize the diffusion U-Net using the existing Gaussian DDPM epoch-30 EMA checkpoint.

Load and verify the original pretrained weights.

Create a new conditional model variant and separate checkpoint namespace.

For the ResNet-18 encoder:

- Support a self-supervised pretrained encoder using only normal MRI data.
- A masked-image reconstruction pretraining objective is a possible implementation.
- Do not claim that a simple masked ResNet-18 implementation exactly reproduces SparK.
- Do not freeze a randomly initialized encoder and assume its features are meaningful.
- Clearly document whether the encoder is pretrained, jointly trained or frozen.

For the initial experiment, prefer normal-only encoder pretraining followed by a controlled conditional DDPM fine-tuning run.

Do not download external pretrained weights unless explicitly authorized.

## 7. Training pipeline

For each normal MRI image:

1. Load the image using the existing preprocessing contract.
2. Extract the condition vector using ResNet-18.
3. Sample a random timestep from the existing 1,000-step schedule.
4. Sample Gaussian noise.
5. Generate the noisy image.
6. Predict noise using the condition-aware U-Net.
7. Compute epsilon-MSE.
8. Backpropagate according to the chosen encoder fine-tuning strategy.

Loss:

`L = mean((epsilon - predicted_noise)^2)`

Do not introduce tumor labels, lesion masks or supervised segmentation loss into training.

Keep the existing noise schedule, normalization and diffusion training objective for the first comparison.

Use a new configuration with explicitly documented optimizer settings, learning rates, update budget, EMA and seed.

Do not compare different methods using unequal compute budgets without reporting the difference.

Checkpoint selection should use normal-validation data only.

## 8. Inference pipeline

Input: an unseen MRI image that may contain a tumor.

Perform:

1. Load and normalize the MRI.
2. Extract condition `c` from the original test image using ResNet-18.
3. Add Gaussian noise at the selected starting timestep.
4. Run reverse diffusion while passing the same condition to the U-Net at every denoising step.
5. Obtain the reconstructed image.
6. Compute anomaly maps.

Required scores:

- Absolute pixel difference.
- Local `1 - SSIM`.

Do not use tumor ground-truth masks during inference.

Investigate the risk of condition leakage: because the encoder receives the entire test image, it may encode tumor information and allow the U-Net to reconstruct the lesion.

## 9. Evaluation protocol

Reuse the existing DACN validation pipeline.

For the first controlled comparison:

- Keep `t*=300`.
- Use the original Gaussian DDPM checkpoint as baseline.
- Preserve the dataset revision, preprocessing and sampling seed policy.
- Use the same validation images.
- Use the existing 40-slice calibration / 43-slice patient-group holdout procedure.
- Select thresholds only on calibration.
- Keep held-out test data sealed.

Compare:

A. Original Gaussian DDPM.

B. ResNet-18 Conditional DDPM.

Evaluate:

- Pixel AUROC.
- Pixel Average Precision.
- Pixel Precision.
- Pixel Recall.
- Pixel F1/Dice.
- Normal-slice false-positive rate.
- Foreground reconstruction MAE and SSIM.
- Tumor versus non-tumor anomaly-score separation on validation.

Also retain the existing brightness-only control.

Include visualizations of original MRI, noisy image, reconstruction, absolute-difference map, SSIM map and ground-truth mask for evaluation only.

## 10. Required tests

Before any full GPU training:

1. Verify that original Gaussian DDPM checkpoint loading is unchanged.
2. Verify that disabled or zero-initialized conditioning produces the same output as the original U-Net for identical inputs and weights.
3. Verify tensor dimensions and condition projection.
4. Verify gradients reach the intended trainable parameters.
5. Verify that the diffusion loss is computed correctly.
6. Verify inference never uses tumor masks as inputs.
7. Verify deterministic sampling under the existing seed policy.
8. Verify conditional checkpoints cannot be silently loaded through the original unconditional checkpoint loader.
9. Run a bounded smoke test using synthetic tensors and a small normal-only dataset subset.

## 11. Expected deliverables

First inspect the existing repository and report the minimal implementation plan.

Then implement:

- ResNet-18 encoder module.
- Feature-conditioned U-Net variant.
- Conditional DDPM model and initialization logic.
- Separate training configuration and entrypoint.
- Conditional reconstruction/evaluation entrypoint.
- Unit tests.
- Documentation explaining the architecture, training and inference.

Keep existing unconditional DDPM and patch/context experiments functional.

Do not run expensive full training, deploy remote jobs or access held-out test data without explicit approval.

Finally, explain:

1. Which files were added or modified.
2. How the encoder condition enters the U-Net.
3. How the original epoch-30 checkpoint is reused.
4. Which model weights are frozen or trainable.
5. Whether the implementation passed the tests.
6. What exact experiment should be run next to measure improvements in Pixel AP/Dice.

**Success criterion:** A correct, reproducible conditional DDPM implementation with a fair validation protocol. An improvement in anomaly detection must be measured, not assumed.