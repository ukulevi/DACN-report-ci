"""Brain-mask post-processing and brightness hybrids on the Final reconstruction.

Validation only; cached producer maps of the t300 DDIM20+30 reference run; no model
calls and no test access. Fixed before running:

Hypotheses
  H1 (post-processing): metrics use the full 240 crop with no foreground filter, so
     scores outside the brain and on its rim are false positives. Zeroing scores
     outside an eroded brain mask, then a 5x5 median filter (common in brain-MRI
     UAD post-processing), should raise pixel AP for both Final LPIPS and brightness.
  H2 (hybrid): FLAIR lesions are hyperintense. Brightness wins pixel AP on test,
     Final LPIPS wins image AUROC. Combining them should keep Final's image-level
     signal and gain brightness's pixel-level signal.

Arms (all reported; nothing selected): base map x post-processing, where
  base = final_lpips, brightness, lpips_x_brightness (product),
         lpips_bright_cdf_w{025,050,075} (midrank-CDF rank fusion fitted on pooled
         normal calibration pixels, as in AF-03: (1-w)*CDF_lpips + w*CDF_brightness)
  post = none, brain (score x eroded brain mask), brain_median5 (then 5x5 median).
Brain mask per slice: input > -0.95 inside the 240 crop, holes filled, 3 iterations
of 3x3 binary erosion. It uses only the input image, never labels.

Thresholds are max-F1 on the 40-image calibration half; intervals are paired
patient-group bootstraps on the 43-image holdout against both Final LPIPS and
brightness. Decision rule for proposing one configuration for a single later test
run: holdout pixel AP above BOTH references with both delta CIs excluding 0, image
AUROC not lower than Final beyond uncertainty, and calibration AP not lower than
Final. With 18 arms some intervals will exclude 0 by chance, so a pass is a
candidate, not a confirmed result.
"""

import numpy as np
from scipy import ndimage

from dacn.evaluation.final_fusion import apply_midrank_cdf, fit_midrank_cdf

FOREGROUND_LEVEL = -0.95
ERODE_ITERATIONS = 3
MEDIAN_SIZE = 5
BRIGHTNESS_WEIGHTS = (0.25, 0.5, 0.75)
POSTPROCESS = ("none", "brain", "brain_median5")
REFERENCE = "final_lpips__none"
BASELINE = "brightness__none"


def brain_masks(images):
    """Eroded brain masks (N,240,240) from (N,256,256) inputs in [-1, 1]."""
    images = np.asarray(images)
    if images.ndim != 3 or images.shape[1:] != (256, 256):
        raise ValueError("images must have shape (N, 256, 256)")
    out = np.empty((len(images), 240, 240), dtype=bool)
    for i, image in enumerate(images[:, 8:-8, 8:-8]):
        filled = ndimage.binary_fill_holes(image > FOREGROUND_LEVEL)
        out[i] = ndimage.binary_erosion(filled, iterations=ERODE_ITERATIONS)
    return out


def postprocess(maps, brain, kind):
    maps = np.asarray(maps, dtype=np.float64)
    if kind == "none":
        return maps
    masked = maps * brain
    if kind == "brain":
        return masked
    if kind == "brain_median5":
        return ndimage.median_filter(masked, size=(1, MEDIAN_SIZE, MEDIAN_SIZE), mode="nearest")
    raise ValueError("unknown post-processing")


def base_maps(lpips, brightness, labels, calibration, weights=BRIGHTNESS_WEIGHTS):
    lpips = np.asarray(lpips, dtype=np.float64)
    brightness = np.asarray(brightness, dtype=np.float64)
    maps = {
        "final_lpips": lpips,
        "brightness": brightness,
        "lpips_x_brightness": lpips * brightness,
    }
    lpips_cdf = apply_midrank_cdf(fit_midrank_cdf(lpips, labels, calibration), lpips)
    bright_cdf = apply_midrank_cdf(fit_midrank_cdf(brightness, labels, calibration), brightness)
    for weight in weights:
        name = f"lpips_bright_cdf_w{round(weight * 100):03d}"
        maps[name] = (1 - weight) * lpips_cdf + weight * bright_cdf
    return maps


def all_arms(lpips, brightness, images, labels, calibration):
    """Every base map under every post-processing, keyed ``base__post``."""
    brain = brain_masks(images)
    arms = {}
    for base, values in base_maps(lpips, brightness, labels, calibration).items():
        for kind in POSTPROCESS:
            arms[f"{base}__{kind}"] = postprocess(values, brain, kind)
    return arms, brain


def brain_mask_coverage(brain, truth):
    """Share of ground-truth lesion pixels kept inside the eroded brain mask."""
    lesion = np.asarray(truth).astype(bool)
    return float((lesion & brain).sum() / max(lesion.sum(), 1))


# Gated-reconstruction ablation (fixed before running, after the hybrid run):
# Question: is the gain of Final LPIPS x brightness due to the diffusion re-sampling,
# or would any reconstruction error gated by brightness do as well? Each cached
# producer map is multiplied by the same brightness map; no post-processing.
# Credit to the re-sampling requires the Final LPIPS x brightness minus
# Initial LPIPS x brightness holdout pixel-AP delta CI to exclude 0 in Final's
# favour, with calibration AP agreeing in sign. Other gates are reported for context.
GATE_SOURCES = {
    "final_lpips": "autoddpm_lpips",
    "initial_lpips": "initial_lpips_only",
    "final_l1": "autoddpm_l1",
    "initial_l1": "ddpm_l1",
    "naive_stitch_l1": "naive_stitch_l1",
}
GATE_REFERENCE = "final_lpips_x_brightness"
GATE_KEY_CONTRAST = "initial_lpips_x_brightness"
# hybrid-full-20261114T103245Z, lpips_x_brightness__none holdout (pixel AP, Dice, image AUROC).
HYBRID_REPLAY = (0.6558091692470411, 0.5879735387280586, 0.9320175438596492)


# Frozen 14 Nov for one test evaluation: Final LPIPS x brightness, no post-processing.
# Thresholds are max-F1 on the 40-image validation calibration half
# (hybrid-full-20261114T103245Z, arm lpips_x_brightness__none; score >= threshold).
FROZEN_HYBRID_ARM = "final_lpips_x_brightness"
FROZEN_HYBRID_THRESHOLDS = {"pixel": 0.1354231256334515, "image": 0.17669394359404414}


def frozen_hybrid_map(final_lpips, brightness):
    return np.asarray(final_lpips, dtype=np.float64) * np.asarray(brightness, dtype=np.float64)


def gated_arms(arrays):
    """Each cached producer map x brightness, plus the two unprocessed references."""
    brightness = np.asarray(arrays["brightness"], dtype=np.float64)
    arms = {
        REFERENCE: np.asarray(arrays["autoddpm_lpips"], dtype=np.float64),
        BASELINE: brightness,
    }
    for name, key in GATE_SOURCES.items():
        arms[f"{name}_x_brightness"] = np.asarray(arrays[key], dtype=np.float64) * brightness
    return arms


def credits_resampling(summary, pixel_vs_reference):
    """Pre-registered gate-ablation rule (see GATE_* comment)."""
    delta = pixel_vs_reference["candidates"][GATE_KEY_CONTRAST]["delta_ap_ci95"]
    calibration = (
        summary[GATE_REFERENCE]["calibration_ap"] - summary[GATE_KEY_CONTRAST]["calibration_ap"]
    )
    # Deltas are contrast - reference, so the reference wins when the CI is below 0.
    return bool(delta[1] < 0 and calibration > 0)


def passes_rule(name, holdout, pixel_vs_final, pixel_vs_brightness, image_vs_final):
    """The pre-registered candidate rule for one arm (see module docstring)."""
    if name in (REFERENCE, BASELINE):
        return False
    return bool(
        pixel_vs_final["candidates"][name]["delta_ap_ci95"][0] > 0
        and pixel_vs_brightness["candidates"][name]["delta_ap_ci95"][0] > 0
        and image_vs_final["candidates"][name]["delta_auroc_ci95"][1] >= 0
        and holdout[name]["calibration_ap"] >= holdout[REFERENCE]["calibration_ap"]
    )


# Sampling-seed robustness of the frozen hybrid (fixed before running, 14 Nov).
# Maps: variants-full-20261114T041258Z, t300 DDIM20+30, sampling seed bases 17/18/19
# (seed 17 is the producer control). Brightness has no seed. Thresholds are refit on
# calibration per seed, as the protocol would for any rerun. Rules, per seed s:
#   beats_brightness[s]: CI of (brightness - final_x_brightness_s) holdout AP below 0
#   beats_final[s]:      CI of (final_lpips_s - final_x_brightness_s) holdout AP below 0
#   resampling[s]:       CI of (initial_x_brightness_s - final_x_brightness_s) below 0
#                        and calibration AP of final_x_brightness_s above initial's
# The hybrid is called seed-robust if beats_brightness and beats_final hold for all
# three seeds; the re-sampling credit is reported as a count out of three.
SEED_VARIANTS = {
    17: (
        "private_control_t300_20_30_maps.npz",
        "eb9d56805fb4cadddc354dd700b803cfa4f69fae9ea361b7de1e96ff194cdae3",
    ),
    18: (
        "private_seed18_t300_20_30_maps.npz",
        "c980d9c3b517f3e1dbab978c0e969b13e32ba2cabe84e3df703afbdfac96e133",
    ),
    19: (
        "private_seed19_t300_20_30_maps.npz",
        "ea40d38506f2ce4366ec56e15ca1406c76c11d96d3bee0384fda7d52c9f3e16c",
    ),
}
SEED_SOURCE_RUN = "variants-full-20261114T041258Z"
SEED_SOURCE_REVISION = "c699b85d3025fb559e5fb68e9a8669c6f94a0b86"


def seed_arms(seed_maps, brightness):
    """Brightness plus Final LPIPS, Final x brightness and Initial x brightness per seed."""
    brightness = np.asarray(brightness, dtype=np.float64)
    arms = {"brightness": brightness}
    for seed, maps in sorted(seed_maps.items()):
        final = np.asarray(maps["autoddpm_lpips"], dtype=np.float64)
        arms[f"final_lpips_s{seed}"] = final
        arms[f"final_x_brightness_s{seed}"] = final * brightness
        arms[f"initial_x_brightness_s{seed}"] = (
            np.asarray(maps["initial_lpips_only"], dtype=np.float64) * brightness
        )
    return arms


def seed_verdict(summary, pixel_by_reference, seeds):
    """Apply the SEED_* rules; ``pixel_by_reference[s]`` uses final_x_brightness_s."""
    out = {"beats_brightness": {}, "beats_final": {}, "resampling": {}}
    for seed in seeds:
        hybrid = f"final_x_brightness_s{seed}"
        deltas = pixel_by_reference[seed]["candidates"]
        out["beats_brightness"][seed] = bool(deltas["brightness"]["delta_ap_ci95"][1] < 0)
        out["beats_final"][seed] = bool(deltas[f"final_lpips_s{seed}"]["delta_ap_ci95"][1] < 0)
        initial = f"initial_x_brightness_s{seed}"
        out["resampling"][seed] = bool(
            deltas[initial]["delta_ap_ci95"][1] < 0
            and summary[hybrid]["calibration_ap"] > summary[initial]["calibration_ap"]
        )
    values = [summary[f"final_x_brightness_s{s}"]["holdout_ap"] for s in seeds]
    out["seed_robust"] = all(out["beats_brightness"].values()) and all(out["beats_final"].values())
    out["resampling_credit_count"] = sum(out["resampling"].values())
    out["hybrid_holdout_ap_mean"] = float(np.mean(values))
    out["hybrid_holdout_ap_sd"] = float(np.std(values, ddof=1))
    return out
