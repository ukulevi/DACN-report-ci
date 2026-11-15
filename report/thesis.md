# Diffusion Models for Anomaly Detection

## Abstract {.unnumbered}

A Gaussian DDPM trained on 7,500 normal brain FLAIR slices supports
reconstruction-based anomaly detection on a patient-disjoint BMAD-derived
split. The pipeline runs DDIM20, constructs a pseudo-mask from initial LPIPS
and normalized residual, and applies masked DDIM30. The selected primary
method multiplies the final spatial LPIPS map by input brightness. On the
3,665-slice test cohort, pixel average precision (AP) increased from 0.3182
for raw Final LPIPS to 0.6505 for the primary method at 50 U-Net calls per
slice. Pixel AUROC and micro Dice were 0.9759 and 0.5943, respectively.
Averaging three sampling seeds before brightness weighting yielded AP
0.7009, AUROC 0.9805 and Dice 0.6290 at 150 calls per slice. Validation
experiments across four trained checkpoints showed consistent AP ordering.
The test data informed the brightness hypothesis through earlier comparisons.
The hybrid and ensemble results are post-development evaluations on that
previously inspected cohort. Small lesions remain challenging.

**Keywords:** anomaly detection, diffusion models, brain MRI, DDIM, LPIPS,
brightness weighting, patient-group evaluation.

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
noise structure [@wyatt2022]. The selected method uses Gaussian
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
the test analyses reported below use this same cohort.

| Derived split | Normal slices | Abnormal slices | Total | Patient groups |
|---|---:|---:|---:|---:|
| Training | 7,500 | 0 | 7,500 | 424 |
| Validation | 39 | 44 | 83 | 50 |
| Test | 640 | 3,025 | 3,665 | 767 |

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

The original and improved methods use the Gaussian DDPM EMA epoch-30
checkpoint. EMA parameters provide the inference weights; a checkpoint name
alone is insufficient provenance, so the experimental record pins checkpoint, source,
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

### Selected scoring method and sampling ensemble

The primary method weights the final spatial LPIPS map by slice brightness.
For cropped input $x$ in $[-1,1]$, define $B=(x+1)/2$. The primary pixel
score is $S(p)=B(p)A_{17}(p)$, where $A_{17}$ is the Final LPIPS map at
sampling seed 17. This continuous weight reduces the score at dark pixels.
The original pseudo-mask, checkpoint, two DDIM stages and spatial SqueezeNet
LPIPS stay fixed. No extra denoising call is required. Brightness alone
serves as a zero-call control.

The higher-compute variant uses the same checkpoint with sampling seeds
17, 18 and 19 and averages their score maps before weighting:
$$S_{\mathrm{ens}}(p)=B(p)\frac{A_{17}(p)+A_{18}(p)+A_{19}(p)}{3}.$$
Each seed uses a pseudo-mask threshold fitted on normal calibration slices
with the same p95/dilation rule. The three mask thresholds are approximately
0.133447, 0.125878 and 0.129472. The primary pixel and image decision
thresholds are 0.135423 and 0.176694. The ensemble thresholds are 0.135428
and 0.153793. Published full-precision values are applied unchanged.
The team selected the primary method and the higher-compute variant for the
report and demo on 15 November. See the [method record](../docs/REPORT_METHOD_DECISION_2026-11-15.md).

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

### Original test and post-development follow-ups

The original four-arm protocol completed on 10 November. Hybrid and
ensemble follow-ups used 3,665 slices (640 normal, 3,025 abnormal) from
767 patient groups. Decision thresholds were selected on the 40-slice
calibration split and applied unchanged. Pixel metrics pool the full
240-by-240 crop, and pixel F1 equals micro Dice. Image metrics use the mean
of the top 576 pixel scores and a slice-level label. The 10 November
pixel table is the frozen comparison. The 14 November table repeats the
Final LPIPS and brightness controls alongside the two follow-up methods.
The intervals come from 1,000 paired patient-group bootstrap replicates.

<!-- TEST_RESULTS -->

The primary method improves pixel AP by 0.3323 over raw Final LPIPS at
the same U-Net budget and by 0.1391 over brightness. The ensemble adds
0.0504 AP over the primary (paired 95% CI [0.0454, 0.0551]) at three
times the model-call budget. Image AUROC also improves, while at the
calibrated image thresholds both improved methods have lower recall and
F1 than raw Final LPIPS. On normal slices, the ensemble's pixel
false-positive rate is 0.72% versus 1.01% for the primary. This rate
counts pixels within normal slices. The call counts do not measure
end-to-end latency.

### Test-access history

| Access | Date (November 2026) | Purpose |
|---:|---|---|
| 1 | 10 | Original frozen four-arm evaluation |
| 2 | 11 | Exploratory DDIM50 comparison |
| 3 | 12 | Exploratory LPIPS-only pseudo-mask comparison |
| 4 | 14 | Replay of the original protocol for uncertainty intervals |
| 5 | 14 | Evaluation of the validation-selected brightness hybrid |
| 6 | 14 | Evaluation of the validation-selected three-seed ensemble |

Earlier test comparisons suggested that brightness could complement
Final LPIPS. The hybrid and ensemble configurations and operating
thresholds were subsequently fixed from validation before their respective
test runs. These are post-development evaluations on the previously
inspected test cohort. Run provenance and original metrics are available
in the [test record](../docs/FINAL_TEST_RESULTS_2026-11-10.md).

## Discussion

### Component contributions and remaining errors

In the original test comparison, Initial LPIPS reached AP 0.224625 and
naive stitching reached 0.269516. Full masked resampling reached 0.318226.
Initial and naive-stitch use 20 U-Net calls. Final uses 50, so the comparison
against stitching measures the benefit of adding phase two at extra
compute. The submitted report also included the exploratory LPIPS-only mask
variant at the same 50-call budget (AP 0.289999, AUROC 0.925694, Dice
0.347241). This comparison isolates the mask recipe more closely. The
variant uses separately calibrated mask and decision thresholds, and its
test result has no paired interval. Adding the residual to the final score
instead reduced validation holdout AP from 0.3123 to 0.1968.

On validation holdout, brightness alone reached AP 0.390 and brightness
weighted Initial LPIPS reached 0.530. Weighting Final LPIPS reached 0.656,
while brightness weighted L1 variants reached about 0.17 to 0.18. These
paired comparisons support a contribution from perceptual scoring and the
second reconstruction phase within the tried configurations. Changing the
pseudo-mask to use brightness gave worse results at all three sampled
seeds. AlexNet and VGG16 also underperformed the SqueezeNet hybrid in
the comparative run. Rank fusion with the residual, the normal-tissue CDF
gate and a within-slice brightness rank did not yield a consistent holdout
advantage. See the [controlled ablation results](../docs/ABLATION_SUMMARY_2026-11-13.md).

The noise sweep fixed the remaining pipeline settings. AP was lower at
levels 50 and 800 than at 300. Differences between levels 200 and 500 were
less clear, and level 400 did not give a consistent calibration advantage.
Changing the DDIM budget, phase-two start or mask dilation showed no
reliable basis for replacing the selected configuration in these trials.

At the hybrid's calibration threshold, about 22% of the 32 annotated
components smaller than 50 pixels on validation holdout overlapped at least
one predicted pixel. Large lesions can be detected while still being
undersegmented. About 99% of hybrid false-positive pixels were inside the
estimated brain region on this holdout. These descriptive figures give a
clearer account of remaining errors than selecting a single visual example.

### Robustness across trained checkpoints

The original T4 checkpoint and three EMA epoch-30 checkpoints trained on an
A100 under the same training configuration were evaluated on validation.
The A100 runs used training seeds 17, 18 and 19. For each checkpoint the
method was applied with the same sampling-seed protocol and recalibrated
on its validation calibration partition. Across the four checkpoints,
43-slice holdout AP was 0.324 ± 0.027 for Final LPIPS, 0.653 ± 0.007 for
the primary hybrid, and 0.725 ± 0.012 for the ensemble (mean ± sample SD).
The prespecified rules used pooled-83 validation AP and favored the
hybrid over its controls and the ensemble over the hybrid for all four
checkpoints. The 43-slice holdout and pooled-83 summaries answer different
questions. The latter includes calibration examples. The cross-GPU
contrast did not resolve a hardware effect, and the follow-up test results
use the original trained checkpoint. See the [retraining comparison](../docs/ABLATION_SUMMARY_2026-11-13.md).

### Limits of the evidence

Patient-group bootstrap intervals capture resampling variation within
the measured cohort, conditional on the evaluated checkpoint and score maps.
Earlier test results contributed to the brightness hypothesis, so the
follow-up test estimates have an adaptive development history even though
their thresholds were fixed on validation. The repeatedly inspected
validation split, rendered 2D intensity domain and limited scanner/site
metadata restrict generalization. In particular, brightness weighting can
suppress a lesion with low displayed intensity. Independent patient data
would be needed to establish transfer across cohorts.

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
and publishes a private report prerelease on main. The
private build host is independent of the original owner's payment restriction;
source synchronization currently uses an explicit Git push, with no broad PAT
or automatic merge. Successful document CI is not evidence that a new model
experiment was run or that the scientific protocol is frozen.

## Conclusion

Final LPIPS weighted by input brightness is the team's primary method.
It improves pixel AP from 0.3182 to 0.6505 at 50 U-Net calls per slice on
the reported test cohort. A three-sampling-seed ensemble reaches 0.7009
at 150 calls. Ablations support retaining the residual-weighted
pseudo-mask and the second reconstruction stage. The validation
retraining comparison supports the ordering across four checkpoints.
Further confirmation requires an independent cohort. The present
evidence includes six recorded accesses to the same test split and
shows a remaining performance gap for small lesions.
