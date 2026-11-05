# ResNet-18 feature-conditioned DDPM: experiment A

User authorized joint training for10 epochs on2026-11-05, following newmethod.md.
Experiment B (normal-only encoder pretraining) is not run or claimed here.

The original grayscale U-Net and state-dict keys remain unchanged. Its forward
optionally accepts an embedding shaped like its timestep embedding. ResNet-18
uses a1-channel7x7 stem and standard BasicBlock stage depths[2,2,2,2], followed by
global pooling512 -> Linear512/128 -> SiLU -> zero-initialized Linear128/time_dim.
The injection is added to the timestep embedding used by every residual block;
time_dim is read from the U-Net rather than hard-coded. No patch mask, spatial
conditioning map, external encoder checkpoint or ImageNet weights are used.

Initialize the U-Net strictly from Gaussian epoch30 EMA with immutable SHA256
d3e06d2082e18234899b201718abd59078c8a800d66da3cc5d8b9e0d9ca9b946.
Fresh encoder/features, zero condition injection, fresh optimizer and EMA equal
to initialized composite model. All parameters trainable. At the first backward
pass encoder gradients are zero because injection is zero; injection learns
first, and gradients reach the encoder after it opens. This is expected, not
encoder freezing. EMA averages all parameters and copies online BatchNorm buffers.

Train7,500 normal slices, Gaussian epsilon MSE with uniform index0..999 and
unchanged linearT1000, padding/normalization. T4,batch8,seed17,10 epochs,
9,380 planned updates; AdamW betas0.9/0.999,weight_decay0,clip_norm1,
U-NetLR1e-5,encoder/projectionsLR1e-4,EMA0.9992. Normal39-slice validation
EMA epsilon MSE uses fixed seed1,000,017 and selects best. Save last each epoch,
full-state milestones1/5/10, initial epoch0 recovery checkpoint, best, config,
source provenance and history. Namespace feature_condition_*.pt, method
ddpm_resnet18_joint_condition_v1; unconditional loader rejects it.

Before full training, synthetic unit tests and a T4 smoke use16 normal training
and4 normal validation slices,1epoch,batch2, plus3-step normal reconstruction.
Smoke is not a metric result. No full training starts until smoke passes.

Full job automatically evaluates the normal-validation-selected best EMA on all83
validation images at t300, ancestral batch1,seed17. Encode original input once;
bind the same128-vector at every reverse step. No lesion-mask input. Preserve
pixel difference/local1-SSIM, existing40 calibration /43 grouped holdout with
calibration-only thresholds, brightness control, separately descriptive all83.
Compare identical registered baseline cache with SHA/content checks. Report
foreground MAE[0,1]/SSIM on normal images (input>-0.95 heuristic), lesion vs
nonlesion score means on abnormal validation slices and private four-example
figures(input/noisy/reconstruction/difference/SSIM/annotation). Test remains sealed.

Whole-image features can encode tumors; improved anatomy or AP is not assumed.
Original baseline has30 epochs, while A adds10fine-tune epochs; a gain cannot be
attributed uniquely to conditioning without a matched unconditional fine-tune
control. B must start independently from the same Gaussian EMA, with the same
fine-tuning update budget; its encoder pretraining compute must be reported.

Modal entrypoint: src/dacn/cloud/modal_feature_condition.py::run_remote.
Mode smoke uses configs/feature_condition_A_smoke.json. Mode train_eval uses
configs/feature_condition_A_t4_epoch10.json, training then full validation and
private Hub uploads. Fresh output ID for each experiment. On preemption the same
request checks exact config/source and resumes feature_condition_last.pt;
completed/uploaded training is skipped, evaluation resumes cached prefix images.
Resuming an old experiment after changing Python sources is rejected. No automatic
additional experiments, encoder pretraining, sweeps or test access are authorized.
