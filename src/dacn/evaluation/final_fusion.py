"""Weighted LPIPS + residual fusion on the Final reconstruction (advisor AF-03).

Hypothesis fixed before running: LPIPS measures structural/texture change and
L1 measures intensity change; FLAIR lesions are hyperintense, so intensity carries
complementary signal. The LPIPS x residual product failed because a small
residual suppresses LPIPS. An additive fusion in a common rank scale does not
suppress either term, so a small residual weight (0.25) should keep or raise
pixel AP, while large weights drift toward L1 (AP about 0.10).

Each score is mapped through its midrank empirical CDF fitted on pooled pixels of
the normal calibration slices only (no lesion labels), as in the 11 Nov DDIM20
fusion. fused = (1 - w) * CDF_lpips + w * CDF_l1. Weights are fixed; nothing is
selected and the held-out test split is never used. Values above the normal
reference maximum saturate at 1, so ``lpips_l1_cdf_w000`` (CDF of LPIPS alone) is reported next to raw
LPIPS to separate the transform's effect from the fusion's effect.
"""

import numpy as np

RESIDUAL_WEIGHTS = (0.0, 0.25, 0.5, 0.75, 1.0)
REFERENCE = "final_lpips_raw"


def fusion_name(weight):
    return f"lpips_l1_cdf_w{round(weight * 100):03d}"


def fit_midrank_cdf(maps, labels, calibration):
    """Sorted pooled pixels of normal calibration slices (the CDF reference)."""
    calibration = np.asarray(calibration)
    normal = calibration[np.asarray(labels)[calibration] == 0]
    if not len(normal):
        raise ValueError("Normal calibration slices are required")
    reference = np.sort(np.asarray(maps, dtype=np.float64)[normal].ravel())
    if not np.isfinite(reference).all():
        raise ValueError("Non-finite reference scores")
    return reference


def apply_midrank_cdf(reference, values):
    """(count below + half count equal) / N for every value; result in [0, 1]."""
    values = np.asarray(values, dtype=np.float64)
    flat = values.ravel()
    below = np.searchsorted(reference, flat, side="left")
    through = np.searchsorted(reference, flat, side="right")
    return ((below + through) / (2 * len(reference))).reshape(values.shape)


def fused_maps(lpips, residual, labels, calibration, weights=RESIDUAL_WEIGHTS):
    """Raw LPIPS reference plus fixed-weight rank fusions for all slices."""
    lpips_cdf = apply_midrank_cdf(fit_midrank_cdf(lpips, labels, calibration), lpips)
    residual_cdf = apply_midrank_cdf(fit_midrank_cdf(residual, labels, calibration), residual)
    maps = {REFERENCE: np.asarray(lpips, dtype=np.float64)}
    for weight in weights:
        maps[fusion_name(weight)] = (1 - weight) * lpips_cdf + weight * residual_cdf
    return maps
