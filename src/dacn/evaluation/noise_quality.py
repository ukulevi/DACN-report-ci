"""Descriptive reconstruction-quality aggregates for the DDIM noise/step sweep.

Links noise level to reconstruction fidelity, lesion "healing", LPIPS contrast and
pseudo-mask coverage. Uses fixed producer arrays only: no thresholds are fitted
and nothing is selected. Ground-truth masks enter these post-hoc aggregates only.
"""

import numpy as np

FOREGROUND_THRESHOLD = -0.95  # input <= -0.95 is treated as background, as in prior audits
CROP = slice(8, 248)


def _mean(values, select):
    count = int(select.sum())
    return float(values[select].sum(dtype=np.float64) / count) if count else None


def _ratio(numerator, denominator):
    if numerator is None or denominator in (None, 0):
        return None
    return float(numerator / denominator)


def reconstruction_quality(arrays, positions):
    """Pixel-micro aggregates over ``positions`` of one producer sweep run.

    ``arrays`` holds the producer NPZ members: images/initials/finals (N,256,256),
    masks (N,256,256), labels (N,), predicted_masks and LPIPS maps (N,240,240).
    """
    positions = np.asarray(positions)
    image = np.asarray(arrays["images"])[positions][:, CROP, CROP]
    lesion = np.asarray(arrays["masks"])[positions][:, CROP, CROP] > 0
    labels = np.asarray(arrays["labels"])[positions].astype(bool)
    predicted = np.asarray(arrays["predicted_masks"])[positions].astype(bool)
    if image.shape != predicted.shape or labels.shape != (len(positions),):
        raise ValueError("Arrays must share N and the 240x240 crop")
    foreground = image > FOREGROUND_THRESHOLD
    normal_slice = ~labels[:, None, None] & np.ones_like(foreground)
    abnormal_slice = labels[:, None, None] & np.ones_like(foreground)
    normal_tissue = foreground & normal_slice
    lesion_tissue = lesion & abnormal_slice
    peri_tissue = foreground & abnormal_slice & ~lesion

    phases = {}
    for phase, reconstruction_key, lpips_key in (
        ("initial", "initials", "initial_lpips_only"),
        ("final", "finals", "autoddpm_lpips"),
    ):
        error = np.abs(image - np.asarray(arrays[reconstruction_key])[positions][:, CROP, CROP])
        lpips = np.asarray(arrays[lpips_key])[positions]
        normal_error = _mean(error, normal_tissue)
        lesion_error = _mean(error, lesion_tissue)
        normal_lpips = _mean(lpips, normal_tissue)
        lesion_lpips = _mean(lpips, lesion_tissue)
        phases[phase] = {
            "normal_slice_foreground_abs_error": normal_error,
            "abnormal_slice_nonlesion_foreground_abs_error": _mean(error, peri_tissue),
            "lesion_abs_error": lesion_error,
            "lesion_to_normal_abs_error_ratio": _ratio(lesion_error, normal_error),
            "normal_slice_foreground_lpips_mean": normal_lpips,
            "normal_slice_foreground_lpips_p99": (
                float(np.quantile(lpips[normal_tissue], 0.99)) if normal_tissue.any() else None
            ),
            "abnormal_slice_nonlesion_foreground_lpips_mean": _mean(lpips, peri_tissue),
            "lesion_lpips_mean": lesion_lpips,
            "lesion_to_normal_lpips_ratio": _ratio(lesion_lpips, normal_lpips),
        }
    lesion_pixels = int(lesion_tissue.sum())
    masked_lesion = int((predicted & lesion_tissue).sum())
    masked = int(predicted.sum())
    return {
        "samples": len(positions),
        "normal_slices": int((~labels).sum()),
        "abnormal_slices": int(labels.sum()),
        "lesion_pixels": lesion_pixels,
        "phases": phases,
        "pseudo_mask": {
            "lesion_pixel_recall": _ratio(masked_lesion, lesion_pixels),
            "pixel_precision": _ratio(masked_lesion, masked),
            "normal_slice_foreground_fraction_masked": _mean(predicted, normal_tissue),
            "abnormal_slice_nonlesion_foreground_fraction_masked": _mean(predicted, peri_tissue),
            "mean_masked_fraction_of_crop": float(predicted.mean()),
        },
    }


def producer_detection(results, arms=("initial_lpips_only", "autoddpm_lpips")):
    """Copy the producer's fixed-threshold holdout pixel/image metrics for linking."""
    out = {}
    for arm in arms:
        holdout = results["metrics"][arm]["group_holdout"]
        out[arm] = {
            "pixel_ap": holdout["pixel"]["auprc"],
            "pixel_auroc": holdout["pixel"]["auroc"],
            "pixel_dice_micro": holdout["pixel"]["f1"],
            "image_auroc": holdout["image"]["auroc"],
            "normal_pixel_fpr": holdout["empty_mask_pixel_false_positive_rate"],
        }
    return out
