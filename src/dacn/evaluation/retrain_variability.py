"""Training-seed variability of the final method (AF-11); rules fixed before results.

Fixed on 15 Nov while the A100 retraining runs, before any retrained model is
evaluated. Validation only; the test split is not used.

Models (all with the same config, data and EMA epoch-30 checkpoint):
  t4_s17    the original T4 model (producer and variants maps)
  a100_s17  same training seed on an A100 (isolates the hardware effect)
  a100_s18, a100_s19   new training seeds on an A100

For every model, the frozen pipeline (t300 DDIM20+30, producer pseudo-mask with the
p95/d3 rule refit on that model's normal calibration slices) runs with sampling seeds
17/18/19. Arms per model: final (Final LPIPS, sampling seed 17), hybrid (final x
brightness) and ensemble (mean Final LPIPS over the 3 sampling seeds x brightness).
Brightness is shared. Thresholds are max-F1 on calibration; AP is compared on the
43-image holdout and on all 83 slices (pooled) with paired patient-group bootstraps.

Rules
  V1 (method holds after retraining): for every model, the hybrid beats brightness and
     that model's Final LPIPS on pooled-83 AP (delta CI excludes 0).
  V2 (ensemble holds): for every model, the ensemble's pooled-83 AP point estimate is
     at or above that model's hybrid.
  Reported: the mean and SD across the 4 models of holdout and pooled AP for final,
  hybrid and ensemble; the hardware effect a100_s17 - t4_s17 (pooled delta CI); and the
  across-model SD next to the sampling-seed SD (0.027 for the hybrid).
"""

import numpy as np

MODELS = ("t4_s17", "a100_s17", "a100_s18", "a100_s19")
SAMPLING_SEEDS = (17, 18, 19)
KINDS = ("final", "hybrid", "ensemble")
# Training runs (Workspace tanh1c). Checkpoint SHA and Hub revision are read from each run
# manifest and checked by the GPU stage.
TRAINING_RUNS = {
    "a100_s17": "ddpm-partial-a100-b8-e30-s17-20261115T020858Z",
    "a100_s18": "ddpm-partial-a100-b8-e30-s18-20261115T020858Z",
    "a100_s19": "ddpm-partial-a100-b8-e30-s19-20261115T020858Z",
}
CHECKPOINT_NAME = "ddpm_epoch_030.pt"
# GPU map runs per retrained model: (evaluation run ID, Hub revision); filled in after
# they complete. Their maps are SHA-checked against each run's own manifest.
RETRAIN_EVAL_RUNS = {}


def arm(model, kind):
    return f"{model}__{kind}"


def model_arms(finals_by_seed, brightness, model):
    """final/hybrid/ensemble maps for one model; ``finals_by_seed`` keyed by sampling seed."""
    brightness = np.asarray(brightness, dtype=np.float64)
    final = np.asarray(finals_by_seed[17], dtype=np.float64)
    stacked = [np.asarray(finals_by_seed[s], dtype=np.float64) for s in SAMPLING_SEEDS]
    return {
        arm(model, "final"): final,
        arm(model, "hybrid"): final * brightness,
        arm(model, "ensemble"): np.mean(stacked, axis=0) * brightness,
    }


def verdict(summary, pooled_by_reference, pooled_points):
    """Apply V1/V2; ``pooled_by_reference[m]`` is a pooled bootstrap with reference m's hybrid."""
    v1, v2 = {}, {}
    for model in MODELS:
        deltas = pooled_by_reference[model]["candidates"]
        v1[model] = bool(
            deltas["brightness"]["delta_ap_ci95"][1] < 0
            and deltas[arm(model, "final")]["delta_ap_ci95"][1] < 0
        )
        v2[model] = bool(
            pooled_points[arm(model, "ensemble")] >= pooled_points[arm(model, "hybrid")]
        )
    spread = {}
    for kind in KINDS:
        holdout = [summary[arm(m, kind)]["holdout_ap"] for m in MODELS]
        pooled = [pooled_points[arm(m, kind)] for m in MODELS]
        spread[kind] = {
            "holdout_mean": float(np.mean(holdout)),
            "holdout_sd": float(np.std(holdout, ddof=1)),
            "pooled_mean": float(np.mean(pooled)),
            "pooled_sd": float(np.std(pooled, ddof=1)),
        }
    return {"v1_method_holds": v1, "v2_ensemble_holds": v2, "across_model": spread}
