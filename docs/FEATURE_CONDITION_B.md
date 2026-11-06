# Experiment B: normal-only encoder pretraining

User authorized B on2026-11-05, following completed A. This is an implementation
comparison, not a claim of a new diffusion method or an AP improvement.

Pretrain the same grayscale ResNet-18 for5 epochs on7500 normal training slices.
Use dense masked reconstruction: randomly hide32 of64 image-grid patches,32px
each, fill hidden pixels with0 in[-1,1], keep8px padding intact. Force at least
one hidden patch to intersect input>-0.95. A five-stage upsampling decoder
predicts the image; MSE covers only hidden foreground pixels, normalized per
image. Foreground is an intensity heuristic, not anatomical segmentation. No
tumor annotations are read. AdamW LR1e-4,betas0.9/0.999,weight_decay0,clip1,
batch8,seed17. Select best encoder by masked-reconstruction MSE on39 normal
validation slices with fixed masks(seed1000017). This is not SparK or sparse
masked-convolution pretraining. Five epochs is a bounded pilot, not evidence
that the encoder has converged.

Discard the decoder and optimizer. Initialize the U-Net independently from the
same original Gaussian EMA epoch30/SHA as A. Load only the pretraining encoder;
retain A's random512->128 projection and zero-initialized condition injection.
Jointly update encoder, projections and U-Net with exactly A's fine-tuning
configuration:10epochs/9380updates, U-NetLR1e-5,encoder/projectionLR1e-4,
EMA0.9992, copied BN buffers, normal-validation epsilon-MSE checkpoint selection.
Pretraining adds4690 updates and is reported separately. Neither A's trained
weights nor smoke weights initialize full B. Encoder pretraining and B encoder
initialization are checksum/method/dataset-revision validated.

Run bounded smoke first:1pretrain+1joint epoch,16normal training samples,
4normal validation samples,batch2,then3-step conditional reverse. Full worker
then chains5pretrain+10joint epochs and automatic83-slice t300 ancestral
validation. Reuse A's40calibration/43group-holdout partition, sampling seeds,
scorers and calibration-only thresholds. Preserve separate all83 descriptive
metrics, brightness control, normal reconstruction quality and private examples.
Compare against registered original Gaussian cache and immutable A results
JSON(hash a8e57ed824c9e6a00404d82296f15fb208eabb9d8d911c06fcd983dfb8b5a962).
Previously inspected validation remains exploratory; held-out test is sealed.
The encoder sees original images at inference, so encoding pathology remains
a possible failure mode. Improvement over original30 cannot isolate conditioning
from additional training; A/B isolates this pretraining intervention at equal
joint-update budgets but unequal total compute. One seed does not establish
statistical significance.

Entrypoint: `src/dacn/cloud/modal_feature_condition_b.py::run_b_remote`.
Use the paired `encoder_pretrain_B_*`/`feature_condition_B_*` JSON configs.
The worker binds the produced encoder checksum automatically. It reserves one
T4 for at most4h and writes separate pretraining(`RUN-enc`),joint(`RUN`) and
evaluation(`RUN-val`) manifests. Checkpoints commit each epoch, uploads are
private, repeated requests require identical source/config and resume full
states. Run detached; check on request rather than monitoring the whole job.

Both conditional methods share `load_feature_checkpoint`; original Gaussian
loaders continue rejecting their checkpoint schema. Pretraining files use
`encoder_pretrain_*.pt`, fine-tune uses `feature_condition_*.pt`.
