# Diffusion Models for Anomaly Detection

## Abstract {.unnumbered}

This project studies normal-only diffusion models for detecting and localizing
abnormalities in two-dimensional brain FLAIR images. The implementation combines
a Gaussian denoising diffusion probabilistic model, partial DDIM reconstruction,
and perceptual anomaly scoring. A data audit identified patient and exact-pixel
overlap in the downloaded source archive. A documented patient-disjoint derived
split was therefore fixed before training. The final validation candidate uses
an epoch-30 exponential-moving-average checkpoint and a two-phase reconstruction
budget of 20 plus 30 U-Net evaluations per image. Its initial reconstruction
produces a pseudo-mask; masked stitching and resampling produce the final image,
which is compared with the input using spatial LPIPS.

On 43 validation holdout slices, with thresholds calibrated separately on 40
slices, the accepted final score obtained pixel AP 0.312330, ROC-AUC 0.949036,
and micro Dice 0.371215. Image-level ROC-AUC was 0.914474. An independent Colab
T4 execution reproduced the registered score maps within an absolute tolerance
of $10^{-6}$, and CPU metric replay agreed within $10^{-10}$. Brightness alone
obtained higher pixel AP, 0.389506, despite lower image-level ROC-AUC and Dice
at its own calibrated operating point. The results support a reproducible
validation pipeline, but do not establish superiority over simple controls or
clinical generalization. This report includes one sampling seed, repeatedly
examined validation data, and no held-out test results. Protocol freeze and
live final-runtime verification remain prerequisites for final evaluation.

**Keywords:** anomaly detection; diffusion models; brain MRI; DDIM; LPIPS;
patient-disjoint evaluation; reproducibility.

## Introduction

### Problem and motivation

Anomaly detection learns patterns of normal data and assigns high scores to
deviations. In this project, the model is trained on slices without annotated
tumor and evaluated against image-level labels and binary lesion masks. A
normal-labeled slice may still belong to a patient with a tumor elsewhere;
normal-only slices must not be equated with an independently healthy cohort.
The objective is an experimental anomaly detector, not a diagnostic system.

The attraction of a generative model is that it may reconstruct an input
toward the learned normal distribution. The difficulty is preserving legitimate
anatomy while changing pathological content. A good noise-prediction loss does
not guarantee that the reconstruction removes a tumor, and a large residual can
also be caused by ordinary structure or preprocessing artifacts. The project
therefore treats training quality, reconstruction quality and localization
quality as separate questions.

### Research questions and scope

The original questions concern the starting noise level and the choice of
anomaly score on the same reconstructed input. Subsequent experiments examined
perceptual scoring and a bounded mask-stitch-resample adaptation. The report
focuses on three questions: whether the complete validation pipeline can be
independently reproduced; whether final perceptual scoring improves upon
initial and naive-stitch controls; and whether its advantage survives comparison
with a non-diffusion brightness control. These questions are evaluated within
the available validation cohort rather than answered by the training loss.

The contributions are an audited BMAD-derived dataset contract, an implemented
normal-only diffusion/reconstruction pipeline, and an artifact-backed evaluation
workflow with explicit thresholds, computation counts and reproducibility checks.
The project adapts established methods; it does not claim a new diffusion
algorithm, an exact reproduction of AnoDDPM/AutoDDPM, or a clinical result.

## Background and related work

### Denoising diffusion probabilistic models

DDPM learns a reverse denoising process from progressively noised data
[@ho2020]. Let $x_0$ be a normalized training image, $\beta_t$ a noise schedule,
$\alpha_t=1-\beta_t$, and $\bar\alpha_t=\prod_{s=1}^{t}\alpha_s$.
The forward marginal is

$$x_t=\sqrt{\bar\alpha_t}x_0+\sqrt{1-\bar\alpha_t}\epsilon,
\qquad \epsilon\sim\mathcal N(0,I).$$

The implemented noise-prediction objective has the form

$$\mathcal L_\epsilon=\mathbb E_{x_0,t,\epsilon}
\left[\|\epsilon-\epsilon_\theta(x_t,t)\|_2^2\right].$$

This objective supervises noise prediction. It is neither a lesion segmentation
loss nor reconstruction MSE. Since each training example can be corrupted
directly to a sampled timestep, a schedule with 1,000 levels does not imply
training only at $t=1000$. Aggregate epsilon-MSE can conceal timestep-dependent
behavior and cannot by itself establish quality at inference level $t^*=300$.

### Partial reconstruction and DDIM

Partial reconstruction starts from a corrupted observation rather than pure
noise. Low corruption can preserve both anatomy and lesions; stronger corruption
can alter both. AnoDDPM motivates diffusion-based anomaly detection and studies
noise structure [@wyatt2022]. The current accepted candidate uses Gaussian
noise and must not be described as the original Simplex-noise method.

DDIM allows reverse evaluation on a reduced timestep grid [@song2021]. For
$\eta=0$, the transition from a model index $t$ to an earlier index $s$ uses
the predicted clean image and predicted noise in the schematic form

$$\widehat x_0=(x_t-\sqrt{1-\bar\alpha_t}\,
\epsilon_\theta(x_t,t))/\sqrt{\bar\alpha_t},$$
$$x_s=\sqrt{\bar\alpha_s}\widehat x_0+
\sqrt{1-\bar\alpha_s}\epsilon_\theta(x_t,t).$$

The implementation applies its registered clipped-clean-image and re-derived
noise convention. Reduced model-call count is not automatically a measured
wall-clock speedup. Moreover, deterministic reverse transitions do not remove
randomness from initial corruption or the sampled context used for inpainting.

### Structural and perceptual discrepancy

Absolute pixel difference is transparent but sensitive to small intensity
changes. SSIM compares local luminance, contrast and structure [@wang2004].
The project's paired implementation computes $1-\mathrm{SSIM}$ using an
11-by-11 Gaussian window with standard deviation 1.5, reflected boundaries,
and data range 2 on normalized inputs. It does not silently rescale each image.

LPIPS compares learned feature representations [@zhang2018]. This project uses
the registered spatial SqueezeNet LPIPS configuration. Grayscale images are
repeated across RGB channels; the metric receives inputs in $[-1,1]$ with
`normalize=False`. These pretrained features were not trained here on tumor
masks. Their perceptual usefulness on natural images does not establish MRI
specificity. AutoDDPM supplies the mask, stitch and resample idea [@bercea2023];
the two-phase DDIM configuration below is a project-specific adaptation.

## Dataset and preprocessing

### Provenance, labels and split audit

BMAD provides a medical anomaly-detection benchmark [@bao2024]. The brain
images used here are rendered 2D FLAIR PNGs derived from BraTS2021, rather than
the original volumetric MRI intensities. BraTS attribution includes the 2021
benchmark, the multimodal benchmark and the annotated TCGA collections
[@baid2021; @menze2015; @bakas2017]. No external benchmark result is copied into
the project's measured result tables.

The downloaded archive audit found ten overlapping patient identifiers and ten
exact-pixel duplicate images between validation and test. This finding applies
to the specific audited archive, not to every BMAD distribution. The derived
`patient-disjoint-v1` policy retains training and validation, then removes every
test slice belonging to a validation patient. It neither moves examples between
splits nor selects exclusions using model outputs. Fifty test slices belonging
to ten patients were excluded. Counts below are dataset inventory statistics;
they do not mean the test set was evaluated by this report.

| Derived split | Normal slices | Abnormal slices | Total | Patient groups |
|---|---:|---:|---:|---:|
| Training | 7,500 | 0 | 7,500 | 424 |
| Validation | 39 | 44 | 83 | 50 |
| Sealed test inventory | 640 | 3,025 | 3,665 | 767 |

The immutable processed-dataset revision is:

```text
350c74f2313e4e73ca65e7ffc6e9dcf74040b3b7
```

Results refer to this
BMAD-derived split and are not directly comparable with reported numbers on
another distribution. Original provenance and exclusion rules are retained in
the [data protocol](../docs/DATA_PROTOCOL.md).

### Decoding and tensor contract

The 240-by-240 source PNGs use rendered RGB/RGBA intensities. Image decoding
converts them to one displayed-luminance channel; it does not invert the source
colormap into physical MRI intensity. Opaque alpha is checked explicitly.
Color masks are positive where any RGB channel is positive, excluding alpha:
thresholding RGBA indiscriminately would label the opaque background as tumor.
The evaluation target is a whole-lesion binary mask, not separate tumor classes.

Each image is symmetrically padded by eight pixels per side to 256-by-256 and
normalized using $x=\mathrm{uint8}/127.5-1$. There is no resizing, histogram
equalization, test-fitted normalization, or mask input to the normal-only model.
The tensor contract is one-channel float32 image and binary evaluation mask.
Scores and masks are cropped back to 240-by-240 before metrics, excluding the
artificial border. Missing masks, shape disagreements, inconsistent labels,
invalid encodings and cross-split duplicates are checked during data preparation.

### Validation partition and information boundaries

The 83 validation slices are partitioned into 40 calibration and 43 holdout
slices using a recorded patient-group partition. The holdout contains 25
patient groups. Its membership is loaded from the pinned partition rather than
recreated with a different software version. Twenty normal calibration slices
support pseudo-mask calibration. The 39 normal slices in the entire validation
set are not the same subset as those twenty mask-calibration slices.

Lesion masks are not used to optimize the normal-only diffusion model or to
construct a pseudo-mask for an individual inference image. However, labeled
calibration data are used to choose detection thresholds. The complete
evaluation procedure is therefore not label-free. Repeated examination of the
43-slice holdout also limits its independence for exploratory comparisons.

## Methodology

![Conceptual project pipeline. Training uses normal slices; inference and evaluation remain separate. This diagram contains no patient images.](figures/pipeline.pdf){width=100%}

### Training and reconstruction contract

The accepted validation candidate uses the Gaussian DDPM EMA epoch-30
checkpoint. EMA parameters provide the inference weights; a checkpoint name
alone is insufficient provenance, so the candidate pins checkpoint, source,
configuration, dataset and dependency evidence. Training state and model
selection are documented separately from anomaly validation. The immutable
execution-source archive for the accepted T4 run is `d5889d5`; the commit used
to build this report can be newer and does not imply that the model was retrained.

The documented Gaussian training configuration uses a one-channel U-Net with
base width 32, channel multipliers [1, 1, 2, 2, 4, 4], two residual blocks,
two attention heads at resolutions 16 and 8, and no dropout. The registered
30-epoch configuration uses batch size 8, learning rate $10^{-4}$, zero weight
decay, a 1,000-level linear beta schedule, L2 noise loss, gradient clipping at
norm 1, and EMA decay 0.9992. These configuration values describe the documented
training path; actual run identity remains tied to its checkpoint and manifest,
not to a later edited JSON file. Aggregate epsilon-MSE is kept separate from
the anomaly metrics below.

Phase 1 performs 20 U-Net evaluations from starting noise level 300. Its
reconstruction is denoted $\widehat x^{(1)}$. The timestep grid uses distinct
descending model indices; the clean terminal state does not add another model
call. A per-sample sampling seed is deterministically derived from the private
sample identity and base seed 17, allowing paired reconstruction replay without
publishing identifiers or derived seeds.

### Pseudo-mask, stitch and resample

Let $P(x,\widehat x^{(1)})$ be spatial LPIPS, and let
$R=|x-\widehat x^{(1)}|$. The initial combined map is

$$H=P(x,\widehat x^{(1)})\,
\operatorname{clip}\left(\frac{R}{Q_{0.95}(R)+10^{-8}},0,1\right).$$

The pseudo-mask threshold is fitted by the registered pooled 95th-percentile
rule on normal calibration data, then kept fixed. The accepted numerical mask
threshold is 0.13344676792621613, with strict comparison and configured dilation
3. No ground-truth lesion mask is supplied to this operation. This mask is a
proposal for where to reconstruct, not a segmentation ground truth or a
guarantee that lesions are covered.

Stitching combines reconstructed content inside the pseudo-mask with the
original context outside it. Phase 2 performs 30 DDIM U-Net evaluations from
noise level 50, retaining original context at corresponding noise levels. There
is exactly one phase-2 pass, not an implicit five-repeat loop. The final image
equals the input outside the registered mask. Spatial LPIPS between the input
and the actual final reconstruction defines the main final anomaly map.

### Paired controls and image aggregation

The accepted comparison contains four arms: initial LPIPS, naive-stitch LPIPS,
final LPIPS and brightness. The first three isolate different reconstruction
outputs within the registered pipeline. Brightness uses displayed intensity
scaled to $[0,1]$ and is a deliberately simple non-diffusion control. The
initial reconstruction, pseudo-mask and final tensor are shared where required;
the evaluator does not generate a different random final image for each score.

All scores increase with anomalousness. The image score is the mean of the top
576 pixel scores, corresponding to 1% of the evaluated 240-by-240 crop. A
thresholded prediction uses $S\geq\tau$. Each arm has separately calibrated
pixel and image thresholds, retained unchanged for holdout reporting.

| Arm | Pixel threshold | Image threshold |
|---|---:|---:|
| Final LPIPS | 0.266176343 | 0.247193718 |
| Initial LPIPS | 0.250182927 | 0.226208439 |
| Naive-stitch LPIPS | 0.216492772 | 0.208568546 |
| Brightness | 0.482352942 | 0.499775335 |

Displayed thresholds are rounded for readability; evaluation uses full-precision
values in the registered control contracts. Absolute difference and local SSIM
remain implemented paired scores. Their results from other samplers must not be
substituted into this compute-matched four-arm table.

## Experimental design and verification

### Metrics and aggregation

ROC-AUC measures score ranking across thresholds. Pixel average precision (AP)
summarizes precision-recall ranking under substantial lesion imbalance. Neither
metric specifies a single deployable binary mask. At a fixed threshold,

$$\mathrm{Precision}=\frac{TP}{TP+FP},\qquad
\mathrm{Recall}=\frac{TP}{TP+FN},$$
$$F_1=\frac{2TP}{2TP+FP+FN}=\mathrm{Dice}_{\mathrm{micro}}.$$

Pixel counts are pooled across the evaluated crops, giving global micro
aggregation. Macro Dice on lesion slices gives each positive slice equal
weight; it is a different summary and is not interchangeable with micro Dice.
Normal-empty-mask false-positive rate measures predicted-positive pixels on
normal slices. Image metrics use one aggregated score and label per slice,
not one decision per patient. Correlation between slices motivates group-level
uncertainty estimation rather than an independent-pixel confidence interval.

Thresholds maximize F1 on calibration, preferring the largest threshold on an
exact tie. Pixel and image thresholds are optimized separately and then fixed.
AP and ROC-AUC do not require choosing those thresholds. A lower binary Dice
with higher AP need not be contradictory: ranking and the calibrated operating
point measure different behavior. Final-test data must never select a model,
mask threshold, score threshold, checkpoint or noise level.

### Independent evidence and runtime scope

The accepted 8 November Colab T4 execution first passed a two-image smoke, then
completed all 83 validation slices. All ten registered private artifact hashes
were verified after download. Inputs, masks and labels were aligned with the
producer cache; the maximum final-map discrepancy was approximately
$4.77\times10^{-7}$, below the $10^{-6}$ acceptance tolerance. Independent CPU
replay reproduced holdout numeric metrics to $10^{-10}$. Cache/resume checks
verify durable shards and replay behavior; they do not alone prove recovery
from every remote GPU-service interruption.

The complete 83-image run recorded 4,150 U-Net and 249 LPIPS calls. The extra
LPIPS call per image includes the naive-stitch control. The sum of adapter
execution times was 164.659411 seconds; peak allocated CUDA memory was
112,928,768 bytes, approximately 107.7 MiB. These values do not describe
end-to-end notebook wall time, GPU reserved memory or total device occupancy.
The captured environment used Torch 2.14.1+cu130, torchvision 0.29.1+cu130,
LPIPS 0.1.4 and NumPy 2.4.6. A version inventory is not a complete binary lock.

Raw manifests contained inherited source/environment labels. The accepted
evidence sidecar corrects them using the actual source archive and execution
receipt while preserving the raw bundle. The report therefore distinguishes
verified execution provenance from stale labels rather than silently rewriting
the experimental record.

## Results

### Accepted four-arm validation comparison

The tables in the generated evidence appendix are read directly from the
verified aggregate receipt. Figure 2 visualizes its pixel AP and Dice; all
numbers use 43 holdout slices and the original calibration thresholds.

![Accepted validation metrics, one sampling seed. Bars show aggregate values, not confidence intervals. Brightness has the highest pixel AP; final LPIPS has the highest micro Dice among these four arms.](figures/validation.pdf){width=100%}

Final LPIPS obtained pixel AP 0.312330, ROC-AUC 0.949036, precision 0.297471,
recall 0.493572 and micro Dice 0.371215. Initial LPIPS obtained AP 0.157876
and Dice 0.246871; naive stitching obtained AP 0.180593 and Dice 0.296849.
The observed final-minus-initial AP difference is approximately 0.154454.
These paired outcomes are consistent with a benefit from the complete final
pipeline on this cohort, but do not isolate a universal resampling effect.

Brightness obtained pixel AP 0.389506 and ROC-AUC 0.950197, both higher than
the final method. Its calibrated precision was 0.161469, recall 0.788577 and
Dice 0.268051. Final LPIPS achieved image ROC-AUC 0.914474 and F1 0.785714,
compared with brightness image ROC-AUC 0.679825 and F1 0.727273. Thus the
ranking depends on metric and aggregation; reporting Dice alone would hide
the strong brightness AP result.

Final macro Dice on lesion slices was 0.313571 and normal-empty-mask pixel
false-positive rate was 0.005772. These aggregate values suggest that binary
mask quality remains limited. They do not locate false positives anatomically,
nor prove that every tumor is removed while normal tissue is preserved.

### Group bootstrap and exploratory ablations

A paired bootstrap over the 25 holdout patient groups used seed 17 and 1,000
draws, of which 999 were valid. The 95% interval for final-minus-naive-stitch
pixel AP was [0.026164, 0.177945]. The interval for final-minus-brightness AP
was [-0.287385, 0.232914]. These are conditional validation comparisons;
they do not quantify variability across independent training runs, new sites,
or future clinical cohorts.

![Producer ablations with a fixed 50-call budget. Lines connect reported aggregate observations; they do not imply interpolation or uncertainty estimates. Calibration and holdout must not be combined to select a new method.](figures/ablations.pdf){width=100%}

The 9 November producer ablations explore phase allocations and starting noise
levels. Aggregate JSON and hashes were audited for eleven full inference runs
and eleven smoke runs. Not every run's private arrays has been independently
replayed. The highest observed holdout AP in that table was 0.328809 for
30+20 steps at starting level 500; the highest calibration AP was 0.320219 for
20+30 at level 500. Reporting these rows does not authorize replacing the
accepted 20+30/300 candidate using repeatedly inspected holdout outcomes.

## Discussion

### What improved, and what remains unresolved

The final perceptual score improves upon initial and naive-stitch LPIPS in the
accepted comparison, while brightness remains stronger in pixel AP. One
plausible explanation is that FLAIR lesions correlate with brightness, but the
present experiment does not establish a causal mechanism. Differences in
reconstruction, mask coverage, perceptual features and threshold calibration
can all affect the final result. New score fusion or postprocessing should be
preregistered and tested on independent evidence rather than optimized against
the most favorable observed holdout row.

Early low epsilon-MSE and weak localization are compatible: training optimizes
noise prediction, while inference must remove abnormal content and preserve
normal structure. Increasing epochs alone need not repair the score map. The
validated approach separates data integrity, checkpoint provenance, trajectory
correctness, reconstruction discrepancy and calibration before considering
additional expensive training. This is particularly relevant under a limited
student GPU budget.

### Limitations and responsible interpretation

There is one training/sampling seed, one rendered MRI contrast and a small
validation cohort. Multiple slices per patient are correlated, and repeated
validation inspection creates selection bias. The processed PNG intensity
domain, incomplete scanner/site metadata and lack of an independent healthy
cohort restrict generalization. The original archive audit does not establish
that the official BMAD benchmark is defective. The derived split prevents the
identified overlap, but does not prove absence of all near-duplicates or hidden
confounding.

No supervised tumor-segmentation baseline or DAE result is claimed without an
accepted artifact. No cross-method state-of-the-art ranking is justified across
different datasets, splits, supervision or inference budgets. Raw medical images
and sample identities remain private. Research-use permission and attribution
do not automatically grant rights to redistribute derived medical data.

## Engineering reproducibility and project delivery

Source, data revision, checkpoint hash, configuration, sampler, sampling seed,
thresholds, environment inventory and output hashes form the reproducibility
contract. Sampling and metrics are separate stages, so cached maps can be
replayed without retraining. Dataset, model and score contracts reject mismatched
inputs rather than silently adapting them. Failed and interrupted runs remain
in result tracking, separate from planned experiments.

The group now works jointly; A/B/C are historical responsibility labels.
Tuấn Anh coordinates project/data work, Chí Vĩ has led evaluation and data
verification, and Hải Tuấn has led model training and reconstruction development.
This description is not a quantitative workload allocation or a claim of
exclusive ownership. The final contribution statement requires team review.

The English manuscript in `report/thesis.md`, bibliography, layout and aggregate
figures are version-controlled. Pandoc generates LaTeX and Tectonic compiles the
PDF without shell escape. CI rebuilds each satellite push, attaches provenance,
and publishes a labeled private validation-draft prerelease on main. The
private build host is independent of the original owner's payment restriction;
source synchronization currently uses an explicit Git push, with no broad PAT
or automatic merge. Successful document CI is not evidence that a new model
experiment was run or that the scientific protocol is frozen.

## Conclusion and final-evaluation boundary

The project delivers a reproducible normal-only diffusion anomaly-detection
pipeline and a transparent validation comparison. The accepted DDIM20+30 final
LPIPS score improves over its initial and naive-stitch controls on the measured
cohort, but does not outperform brightness in pixel AP. The strongest supported
claim is successful independent validation reproduction with documented
limitations, not a clinically validated detector or a universally superior
segmentation method.

Before final-test execution, the team must record protocol approval, verify the
live runtime and immutable inputs, and run the completed launcher against the
fixed validation contract. Only then may the sealed test be evaluated once
under the approved rules, without subsequent test-based tuning. Until those
steps have evidence, this document is a complete account of the available
validation study and remains a submission draft. The final submission also
requires team-reviewed contributions, any required institution-specific template
checks, presentation materials and the reproducible demonstration.
