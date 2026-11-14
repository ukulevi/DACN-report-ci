"""Error analysis, pooled-83 comparisons and reconstruction ensembles (validation).

Fixed before running (14 Nov). Uses cached producer maps only: no model calls, no
test access, no threshold fitting beyond the usual calibration max-F1.

A. Error analysis (descriptive, no selection). At the calibration-fitted pixel
   thresholds of the frozen hybrid, Final LPIPS and brightness:
   - lesion detection by connected ground-truth component (8-connectivity), binned
     by size in pixels: [1, 50), [50, 200), [200, 1000), >= 1000; a lesion counts as
     detected if any predicted pixel touches it; pixel recall per bin;
   - false-positive pixels by region: background (input <= -0.95), brain rim (filled
     foreground minus 3-pixel erosion), interior; by brightness quarter; on normal vs
     abnormal slices; and the share in predicted components under 10 pixels.
   Reported separately for calibration and holdout.

B. Pooled-83 comparison. AP needs no threshold, so AP is also compared on all 83
   validation slices with the paired patient-group bootstrap. Caveat: the 20 normal
   calibration slices also fitted the phase-2 mask threshold and the CDF gate. A
   contrast counts as resolved when its pooled CI excludes 0. Pre-specified
   contrasts: Final t400 - t300; hybrid t400 - t300; slice-rank gate - frozen hybrid;
   normal-CDF gate - frozen hybrid.

C. Ensembles (H: averaging stochastic reconstructions removes sampling noise).
   seed_ensemble   mean of t300 Final LPIPS over sampling seeds 17/18/19, x brightness
   noise_ensemble  mean of Final LPIPS over t200/t300/t400/t500 (DDIM20+30, seed 17),
                   x brightness
   An ensemble improves on the frozen hybrid only if all of the following hold:
   the pooled-83 AP delta CI is above 0; the holdout and calibration AP point
   estimates are both higher; and the pooled image AUROC is not lower beyond
   uncertainty (delta CI upper bound >= 0).
"""

import numpy as np
from scipy import ndimage

from dacn.evaluation.postprocess_hybrid import ERODE_ITERATIONS, FOREGROUND_LEVEL

SIZE_BINS = ((1, 50), (50, 200), (200, 1000), (1000, None))
SPECKLE_PIXELS = 10
EIGHT = np.ones((3, 3), dtype=bool)
ENSEMBLE_NOISE_LEVELS = (200, 300, 400, 500)
ENSEMBLE_SEEDS = (17, 18, 19)
CONTRASTS = (
    ("final_t400", "final_t300"),
    ("hybrid_t400", "frozen_hybrid"),
    ("slice_rank", "frozen_hybrid"),
    ("normal_cdf", "frozen_hybrid"),
)


def _bin_name(low, high):
    return f"{low}-{high - 1}" if high else f">={low}"


def lesion_detection(truth, predicted):
    """Per-size-bin lesion counts, detection rate and pixel recall."""
    truth, predicted = np.asarray(truth, bool), np.asarray(predicted, bool)
    stats = {_bin_name(*b): {"lesions": 0, "detected": 0, "pixels": 0, "hit": 0} for b in SIZE_BINS}
    for t, p in zip(truth, predicted, strict=True):
        labelled, count = ndimage.label(t, structure=EIGHT)
        if not count:
            continue
        sizes = np.bincount(labelled.ravel())[1:]
        hits = np.bincount(labelled[p].ravel(), minlength=count + 1)[1:]
        for size, hit in zip(sizes, hits, strict=True):
            for low, high in SIZE_BINS:
                if size >= low and (high is None or size < high):
                    row = stats[_bin_name(low, high)]
                    row["lesions"] += 1
                    row["detected"] += int(hit > 0)
                    row["pixels"] += int(size)
                    row["hit"] += int(hit)
    for row in stats.values():
        row["detection_rate"] = row["detected"] / row["lesions"] if row["lesions"] else None
        row["pixel_recall"] = row["hit"] / row["pixels"] if row["pixels"] else None
        del row["hit"]
    return stats


def regions(images):
    """Background, rim and interior masks of the 240 crop."""
    crop = np.asarray(images)[:, 8:-8, 8:-8]
    fg = np.stack([ndimage.binary_fill_holes(s > FOREGROUND_LEVEL) for s in crop])
    interior = np.stack([ndimage.binary_erosion(s, iterations=ERODE_ITERATIONS) for s in fg])
    return {"background": ~fg, "rim": fg & ~interior, "interior": interior}


def false_positive_breakdown(truth, predicted, images, brightness, labels):
    truth, predicted = np.asarray(truth, bool), np.asarray(predicted, bool)
    fp = predicted & ~truth
    total = int(fp.sum())

    def share(mask):
        return float((fp & mask).sum() / total) if total else None

    out = {"fp_pixels": total, "predicted_pixels": int(predicted.sum())}
    out["by_region"] = {name: share(mask) for name, mask in regions(images).items()}
    brightness = np.asarray(brightness)
    out["by_brightness_quarter"] = {
        f"{q / 4:.2f}-{(q + 1) / 4:.2f}": share(
            (brightness >= q / 4) & ((brightness < (q + 1) / 4) if q < 3 else True)
        )
        for q in range(4)
    }
    normal = np.asarray(labels) == 0
    out["on_normal_slices"] = share(np.broadcast_to(normal[:, None, None], fp.shape))
    speckle = 0
    for p in predicted:
        labelled, count = ndimage.label(p, structure=EIGHT)
        if count:
            sizes = np.bincount(labelled.ravel())
            small = np.flatnonzero(sizes < SPECKLE_PIXELS)
            small = small[small > 0]
            speckle += int(np.isin(labelled, small).sum())
    out["predicted_in_components_under_10px"] = (
        speckle / out["predicted_pixels"] if out["predicted_pixels"] else None
    )
    return out


def ensemble_mean(maps):
    stacked = [np.asarray(m, dtype=np.float64) for m in maps]
    if len({m.shape for m in stacked}) != 1:
        raise ValueError("Ensemble members must share one shape")
    return np.mean(stacked, axis=0)


def ensemble_improves(name, summary, pooled_pixel, pooled_image):
    """Rule C; ``pooled_*`` are bootstraps on all 83 slices against the frozen hybrid."""
    return bool(
        pooled_pixel["candidates"][name]["delta_ap_ci95"][0] > 0
        and summary[name]["holdout_ap"] > summary["frozen_hybrid"]["holdout_ap"]
        and summary[name]["calibration_ap"] > summary["frozen_hybrid"]["calibration_ap"]
        and pooled_image["candidates"][name]["delta_auroc_ci95"][1] >= 0
    )


def contrast_status(pooled_by_reference, contrasts=CONTRASTS):
    """Pooled-83 delta CI for each pre-specified contrast and whether it excludes 0."""
    out = {}
    for arm, reference in contrasts:
        ci = pooled_by_reference[reference]["candidates"][arm]["delta_ap_ci95"]
        out[f"{arm} - {reference}"] = {
            "delta_ap_ci95": ci,
            "resolved": bool(ci[0] > 0 or ci[1] < 0),
        }
    return out
