"""Patient-group confidence intervals for the frozen four-arm test (AF-09/AF-10).

Re-computes the frozen 10 Nov four-arm maps with the unchanged frozen adapter and
thresholds; nothing is fitted or selected on the test split. Intervals are only
reported when the recomputed point metrics reproduce the published aggregate.
"""

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

ARMS = ("final_lpips", "initial_lpips", "naive_stitch_lpips", "brightness")
REFERENCE = "final_lpips"
TEST_SAMPLES = 3665
IMAGE_TOP_PIXELS = 576
# Frozen mask threshold (reference run mask_parameters; four-arm acceptance 10 Nov).
MASK_THRESHOLD = 0.13344676792621613
# configs/final_controls_frozen_20261110.json (score >= threshold).
FROZEN_THRESHOLDS = {
    "brightness": {"image": 0.4997753345200585, "pixel": 0.48235294222831726},
    "final_lpips": {"image": 0.24719371760470998, "pixel": 0.26617634296417236},
    "initial_lpips": {"image": 0.2262084392261588, "pixel": 0.2501829266548157},
    "naive_stitch_lpips": {"image": 0.20856854640361336, "pixel": 0.21649277210235596},
}
REPLAY_ATOL = 1e-6


def image_scores(maps):
    """Mean of the top 576 pixels per slice for (N,240,240) maps."""
    flat = np.asarray(maps, dtype=np.float64).reshape(len(maps), -1)
    return np.partition(flat, -IMAGE_TOP_PIXELS, axis=1)[:, -IMAGE_TOP_PIXELS:].mean(axis=1)


def check_published(pixel, image, published, *, atol=REPLAY_ATOL):
    """Compare recomputed point metrics with the published aggregate per arm."""
    errors = {}
    for arm in ARMS:
        reference = published["arms"][arm]
        errors[arm] = {
            "pixel_ap": abs(pixel[arm]["pixel_ap"] - reference["pixel"]["auprc"]),
            "pixel_dice": abs(pixel[arm]["pixel_dice_micro"] - reference["pixel"]["dice_micro"]),
            "image_auroc": abs(image[arm]["auroc"] - reference["image"]["auroc"]),
            "image_ap": abs(image[arm]["ap"] - reference["image"]["auprc"]),
        }
    worst = max(max(values.values()) for values in errors.values())
    if worst > atol:
        raise ValueError(f"Recomputed test metrics do not reproduce the publication: {worst}")
    return {"max_abs_error": worst, "per_arm": errors}


def image_group_bootstrap(labels, scores, groups, *, reference, replicates=1000, seed=17):
    """Paired patient-group percentile bootstrap of image ROC-AUC and AP."""
    labels, groups = np.asarray(labels), np.asarray(groups)
    _, row_groups = np.unique(groups, return_inverse=True)
    members = [np.flatnonzero(row_groups == g) for g in range(row_groups.max() + 1)]
    rng = np.random.default_rng(seed)
    samples = {arm: [] for arm in scores}
    skipped = 0
    for _ in range(replicates):
        rows = np.concatenate([members[g] for g in rng.integers(0, len(members), len(members))])
        if len(np.unique(labels[rows])) < 2:
            skipped += 1
            continue
        for arm, values in scores.items():
            values = np.asarray(values)[rows]
            samples[arm].append(
                [roc_auc_score(labels[rows], values), average_precision_score(labels[rows], values)]
            )
    base = np.asarray(samples[reference])
    out = {}
    for arm, values in scores.items():
        array = np.asarray(samples[arm])
        out[arm] = {
            "auroc": float(roc_auc_score(labels, values)),
            "ap": float(average_precision_score(labels, values)),
            "auroc_ci95": np.percentile(array[:, 0], [2.5, 97.5]).tolist(),
            "ap_ci95": np.percentile(array[:, 1], [2.5, 97.5]).tolist(),
            "delta_auroc_ci95": np.percentile(array[:, 0] - base[:, 0], [2.5, 97.5]).tolist(),
            "delta_ap_ci95": np.percentile(array[:, 1] - base[:, 1], [2.5, 97.5]).tolist(),
        }
    return {
        "method": "paired patient-group percentile bootstrap; image level; fixed scores",
        "reference": reference,
        "seed": seed,
        "requested_replicates": replicates,
        "valid_replicates": replicates - skipped,
        "groups": len(members),
        "candidates": out,
    }
