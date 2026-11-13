"""Validation-only pseudo-mask rule ablation for Final LPIPS (advisor item AF-07).

Hypothesis fixed before running: the p95/dilation3 pseudo-mask re-samples about
45% of normal brain foreground at pixel precision about 0.11. A tighter mask
(higher percentile and/or no dilation) re-samples less normal tissue, lowering
normal-tissue LPIPS and FPR after phase 2; AP should be retained only while
lesion recall stays high. All arms reuse the cached DDIM20(t300) initials and
LPIPS x normalized-residual heatmaps; only the mask rule changes. No arm is
selected here, and the held-out test split is never used.
"""

import numpy as np
from scipy.ndimage import maximum_filter

from dacn.evaluation.noise_quality import FOREGROUND_THRESHOLD

CONTROL = (95, 3)
MASK_RULES = ((95, 3), (90, 3), (90, 1), (95, 1), (98, 3), (98, 1), (99, 3), (99, 1))
SMOKE_RULES = ((95, 3), (99, 1))
REPLAY_ATOL = 1e-5


def rule_name(percentile, dilation):
    return f"p{percentile}_d{dilation}"


def fit_percentile_threshold(heatmaps, labels, calibration, percentile):
    """Pooled percentile over normal calibration slices (producer rule at p95)."""
    calibration = np.asarray(calibration)
    normal = calibration[np.asarray(labels)[calibration] == 0]
    if not len(normal) or not 0 < percentile < 100:
        raise ValueError("Normal calibration slices and a percentile in (0,100) are required")
    values = np.asarray(heatmaps)[normal]
    if not np.isfinite(values).all() or (values < 0).any():
        raise ValueError("Invalid pseudo-mask heatmaps")
    return float(np.quantile(values, percentile / 100))


def rule_mask(heatmap, threshold, dilation):
    """Strict-greater threshold then square maximum filter; dilation1 means none."""
    if heatmap.shape != (240, 240) or not np.isfinite(heatmap).all():
        raise ValueError("Expected finite 240x240 heatmap")
    if dilation not in (1, 3) or not np.isfinite(threshold) or threshold < 0:
        raise ValueError("Invalid mask rule")
    return maximum_filter(heatmap > threshold, size=dilation, mode="constant")


def mask_statistics(predicted, images, masks, labels, positions):
    """Lesion recall/precision and normal-foreground coverage for one mask rule."""
    positions = np.asarray(positions)
    predicted = np.asarray(predicted)[positions].astype(bool)
    lesion = np.asarray(masks)[positions][:, 8:-8, 8:-8] > 0
    foreground = np.asarray(images)[positions][:, 8:-8, 8:-8] > FOREGROUND_THRESHOLD
    normal = (np.asarray(labels)[positions] == 0)[:, None, None]
    normal_foreground = foreground & normal
    hits = int((predicted & lesion).sum())
    return {
        "lesion_pixel_recall": hits / int(lesion.sum()) if lesion.any() else None,
        "pixel_precision": hits / int(predicted.sum()) if predicted.any() else None,
        "normal_slice_foreground_fraction_masked": (
            float(predicted[normal_foreground].mean()) if normal_foreground.any() else None
        ),
        "mean_masked_fraction_of_crop": float(predicted.mean()),
    }


def check_control_replay(threshold, masks, finals, lpips_maps, cached, *, atol=REPLAY_ATOL):
    """Require the p95/d3 control to reproduce the producer threshold, masks and maps."""
    if abs(threshold - cached["mask_threshold"]) > 1e-12:
        raise ValueError("Control mask threshold does not replay the producer")
    if not np.array_equal(np.asarray(masks, bool), np.asarray(cached["predicted_masks"], bool)):
        raise ValueError("Control pseudo-masks do not replay the producer")
    final_error = float(np.max(np.abs(np.asarray(finals) - cached["finals"])))
    lpips_error = float(np.max(np.abs(np.asarray(lpips_maps) - cached["autoddpm_lpips"])))
    if final_error > atol or lpips_error > atol:
        raise ValueError(
            f"Control phase-2 replay exceeds tolerance: final {final_error}, LPIPS {lpips_error}"
        )
    return {"final_max_abs_error": final_error, "lpips_max_abs_error": lpips_error}
