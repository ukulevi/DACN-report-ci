# Project requirements

## Project title

**Diffusion Models cho anomaly detection**
**Diffusion Models for Anomaly Detection**

## Project brief

Anomaly detection aims to identify data samples that significantly deviate from
the normal patterns observed in a dataset. It has important applications in
industrial inspection, medical diagnosis, cybersecurity, and financial fraud
detection. Traditional anomaly detection methods often rely on statistical
assumptions or learn a representation of normal data using techniques such as
autoencoders and generative models.

Diffusion Models have recently emerged as powerful generative models capable of
learning complex data distributions. Their ability to model the distribution of
normal data provides a promising approach to anomaly detection. The key idea of
this project is to train a Diffusion Model primarily on normal data and use its
denoising or reconstruction behavior to identify anomalous samples. Samples
that are poorly reconstructed or have a large discrepancy from the learned
normal distribution can be assigned higher anomaly scores.

In this project, students will implement a Diffusion Model and investigate
different strategies for computing anomaly scores. The performance will be
evaluated on benchmark datasets using metrics such as ROC-AUC, Precision,
Recall, and F1-score. The project may also compare Diffusion Models with
traditional approaches such as Autoencoders, Variational Autoencoders (VAEs), or
Isolation Forest.

Through this project, students will gain practical experience with Generative
AI, Diffusion Models, unsupervised learning, and anomaly detection, while
developing an understanding of how generative models can be used to learn and
characterize complex data distributions.

## How this repository applies the brief

- **Primary method:** a diffusion model trained primarily on normal samples;
  anomaly scores come from a documented denoising/reconstruction discrepancy
  or another explicitly justified diffusion-based strategy.
- **Current benchmark scope:** BMAD Brain FLAIR 2D, using the patient-disjoint
  split and data protocol documented in this repository. The current plan is an
  ANDi-FLAIR adaptation; it is not claimed to be an exact reproduction of the
  source paper.
- **Score convention:** larger scores indicate more anomalous samples/pixels.
- **Required report metrics:** ROC-AUC, Precision, Recall, and F1-score. State
  whether each is image-level or pixel-level, its aggregation and threshold
  protocol. Select thresholds on validation only and freeze them before test.
- **Project-specific metrics:** retain pixel AP, image-level metrics, Dice,
  latency, and peak VRAM from `docs/RESEARCH.md` where applicable.
- **Baselines:** the current plan uses a denoising autoencoder as the practical
  baseline. VAE and Isolation Forest comparisons are optional ideas from the
  brief, not mandatory parallel implementations; check project scope and time
  before adding them.

The assignment brief defines the goal. `docs/PROJECT_PLAN.md` and
`docs/RESEARCH.md` define the current implementation and evaluation scope. If a
future decision changes that scope, record the decision and rationale before
changing the plan.
