# ConvAE and ConvVAE training baselines — 11 November 2026

User explicitly requests matched CNN encoder-decoder baselines, global latent
vector256,30epochs each. This authorizes a bounded smoke then the two full runs;
it does not authorize changing diffusion results, training on anomalies or
opening test for model/threshold selection. Diffusion remains the primary study.
These are project-defined CNN baselines, not an asserted paper reproduction.

## Architecture and objective fixed before training

Both use grayscale256×256 input in the existing[-1,1] preprocessing, six
Conv2d(kernel4,stride2,padding1) stages, channels32/64/128/256/256/256, GroupNorm8
and SiLU; spatial256→128→64→32→16→8→4. Flatten4096→latent256. Decoder linear256→4096,
reshape256×4×4 and mirrored6ConvTranspose stages; Tanh output. No skip connections,
pretrained weights, image corruption, augmentation or foreground filtering.
Shared encoder/decoder initialization matches under seed17; VAE adds a logvar
head alongside mean. ConvAE has7,676,545parameters, ConvVAE8,725,377; the extra
1,048,832parameters are the VAE logvar head, not a different CNN backbone.

AE uses mean latent directly, minimizing reconstruction MSE on the original240crop.
VAE samples z=mu+exp(.5logvar)*epsilon during training, epsilon~N(0,I), logvar
clamped[-20,10]. Fixed objective:

loss = crop240_reconstruction_MSE + beta * mean_batch(KL summed over256latents)/57600,
beta=1. This states the normalization explicitly; no beta/window/latent sweep.
VAE validation decodes mu deterministically; KL is logged but checkpoint selection
uses reconstruction MSE, the same criterion as AE. No early stopping: full runs
complete30epochs even if an earlier epoch has the smallest normal-validation MSE.

## Data and training contract

Immutable dataset350c74f2313e4e73ca65e7ffc6e9dcf74040b3b7, patient-disjoint-v1,
7,500normal training slices and39normal validation slices. Trainer accepts only
train_dir and validation_dir; abnormal validation slices are excluded from model
inputs/objective/selection and no masks are used for training. The shared runner
may perform original full-export data QC; test is never passed to this trainer
or used for model/threshold selection. Normal means no tumor annotation per slice,
not necessarily a healthy patient. No test anomaly metrics produced by training.

Same settings:seed17,batch cap32 (partial batches at shard boundaries),AdamLR1e-4,weight_decay0,gradient_norm_clip1,float32,
noAMP,no scheduler. Shard-aware deterministic order keeps same per-epoch batches
for both models. Each full run consumes all7500samples every epoch, no drop_last.
Fixed30epochs, one training seed per model; no multi-seed significance claim.
Best epoch=min normal-validation crop240MSE, first on exact tie; retain epoch30
last checkpoint separately. Report training/normal-validation losses as such,
never as ROC-AUC/AP/F1 or proof of anomaly detection performance.

## Durability and launch

T4,4CPU/8192MiB,eight-hour timeout in modal_conv_autoencoder.py, shared team
Workspace/Volumes/huggingface-secret-2. Use existing source-snapshot/dataset-pin/
private-Hub lifecycle. Commit checkpoints/history after each completed epoch.
Save autoencoder_best.pt,autoencoder_last.pt and epoch001/005/010/020/030 snapshots.
Checkpoints include optimizer,RNG,sampler epoch,config and cumulative updates for
same-run resume. Require identical code/source/config/trainer/dataset provenance;
a fresh attempt uses a fresh run ID. Never allow simultaneous writers per run.

Configs:conv_ae_z256_smoke.json/conv_vae_z256_smoke.json use one epoch,64normal
train/4normal validation samples, same width32/latent256/batch32 as full runs.
After both pass, start conv_ae_z256_t4_epoch30.json and
conv_vae_z256_t4_epoch30.json from fresh random initialization, not smoke weights.
No GPU/model result is claimed until actual produced checkpoints/manifest exist.

Actual smoke subsets span35shards and produced35updates for64samples. They validate GPU execution/checkpointing at variable small batches, not full-batch32 memory. Full runs use unchanged batch cap32 and shard-local sampling. See RESULT_TRACKING for actual statuses and immutable evidence.

## Authorized validation anomaly evaluation — 2026-11-11

Evaluate both fixed best checkpoints (AE epoch13 SHAfc34f511e4910c688d9e7c5f4f075957f291cb801ced3f1d77850294f271bfbb; VAE epoch15 SHAb85619af1f53f74f09136c26c92e5ac960f9ebdea7d84329cc582c1d56f0ab8e), deterministic mean latent. Use original pinned83validation cache/index, registered40calibration/43group holdout, exactly as cached DDIM scoring. Fixed Residual/L1 and Local SSIM signed[-1,1], Gaussian11 sigma1.5,data_range2; full240crop,no foreground or postprocessing, image top576mean. Separate max-F1 calibration pixel/image thresholds, largest tie; persist before holdout. Report all four model/score combinations, no winner selection/sweep, one training seed. Normal-validation checkpoint selection overlaps both partitions: development validation, not an independent holdout claim. No test access. Bounded2image smoke per model before full83. Private maps/reconstructions/checkpoints remain remote; download aggregate results only.

## Authorized ConvVAE LPIPS validation — 2026-11-11

User requests additional ConvVAE validation after LPIPS discussion. Fixed raw spatial LPIPS0.1 squeeze, pretrained state18fed4d1fc5c04f2918f64df3a2463344108576d1888f0d75538ca6d5c4dffe6, lpips0.1.4. Reuse SHA-verified83deterministic reconstructions from full83evaluation (ConvVAE bestepoch15), original83input cache and registered40/43partition. No retraining or new VAE calls. Crop240, grayscale repeatedRGB[-1,1], normalizeFalse, clamp0, no residual product/fusion/gating. Separate pixel/image max-F1 thresholds fitted only on40calibration, then43validation. Image top576mean, pixel globalmicro. First smoke2images then83, fresh run IDs, T4. Both partitions participated in normal-validation checkpoint selection: exploratory validation only, no independent holdout or test claim; one training seed17. No ConvAE extension or held-out test inference authorized here.
