"""Common image and pixel metrics with validation-only threshold selection."""

import numpy as np
from sklearn.metrics import precision_recall_curve

from dacn.evaluation.metrics import compute_binary_metrics
from dacn.evaluation.scores import paired_score_maps


def select_validation_threshold(labels, scores):
    """Choose max-F1 threshold, preferring the largest on an exact tie."""
    labels = np.asarray(labels)
    scores = np.asarray(scores, dtype=np.float64)
    if labels.ndim != 1 or scores.ndim != 1 or labels.shape != scores.shape:
        raise ValueError("labels and scores must be same-length 1D arrays")
    if not len(labels) or not np.isin(labels, [0, 1]).all() or len(np.unique(labels)) != 2:
        raise ValueError("validation labels must contain both binary classes")
    if not np.isfinite(scores).all():
        raise ValueError("scores must be finite")
    precision, recall, thresholds = precision_recall_curve(labels, scores)
    denominator = precision[:-1] + recall[:-1]
    f1 = np.divide(
        2 * precision[:-1] * recall[:-1],
        denominator,
        out=np.zeros_like(denominator),
        where=denominator != 0,
    )
    return float(thresholds[np.flatnonzero(f1 == f1.max())[-1]])


def evaluate_maps(score_maps, masks, labels, *, split, thresholds=None):
    """Evaluate cropped 240x240 maps; only validation may select thresholds.

    For test, pass thresholds selected on validation. Input arrays and returned
    aggregate metrics do not contain identifiers or image artifacts.
    """
    scores = np.asarray(score_maps, dtype=np.float64)
    masks = np.asarray(masks)
    labels = np.asarray(labels)
    if scores.ndim != 3 or scores.shape[1:] != (240, 240) or masks.shape != scores.shape:
        raise ValueError("score maps and masks must have shape (N, 240, 240)")
    if not len(scores) or labels.shape != (len(scores),):
        raise ValueError("labels must have one value per image")
    if not np.isfinite(scores).all() or not np.isin(masks, [0, 1]).all():
        raise ValueError("scores must be finite and masks binary")
    if not np.isin(labels, [0, 1]).all() or not np.array_equal(
        labels.astype(bool), masks.astype(bool).any(axis=(1, 2))
    ):
        raise ValueError("image labels must agree with binary masks")
    if split not in {"validation", "test"}:
        raise ValueError("split must be validation or test")
    flat_scores = scores.reshape(len(scores), -1)
    image_scores = np.partition(flat_scores, -576, axis=1)[:, -576:].mean(axis=1)
    pixel_labels, pixel_scores = masks.ravel(), scores.ravel()
    selected_here = thresholds is None
    if selected_here:
        if split != "validation":
            raise ValueError("test requires frozen validation thresholds")
        thresholds = {
            "pixel": select_validation_threshold(pixel_labels, pixel_scores),
            "image": select_validation_threshold(labels, image_scores),
        }
    if set(thresholds) != {"pixel", "image"} or not all(
        np.isfinite(float(thresholds[k])) for k in ("pixel", "image")
    ):
        raise ValueError("thresholds must contain finite pixel and image values")
    pixel = compute_binary_metrics(pixel_labels, pixel_scores, thresholds["pixel"])
    image = compute_binary_metrics(labels, image_scores, thresholds["image"])
    predicted = scores >= float(thresholds["pixel"])
    mask_bool = masks.astype(bool)
    lesion = labels == 1
    intersections = (predicted & mask_bool).sum(axis=(1, 2))
    denominators = predicted.sum(axis=(1, 2)) + mask_bool.sum(axis=(1, 2))
    macro_dice = (
        float(np.mean(2 * intersections[lesion] / denominators[lesion])) if lesion.any() else None
    )
    empty = ~mask_bool.any(axis=(1, 2))
    empty_fpr = float(predicted[empty].mean()) if empty.any() else None
    return {
        "split": split,
        "samples": len(scores),
        "pixel": {**pixel, "average_precision": pixel["auprc"], "dice_micro": pixel["f1"]},
        "image": image,
        "pixel_dice_macro_lesion_slices": macro_dice,
        "empty_mask_pixel_false_positive_rate": empty_fpr,
        "threshold_selection": (
            "validation max F1, score >= threshold, largest threshold on tie"
            if selected_here
            else "provided by caller; provenance must be recorded separately"
        ),
        "pixel_aggregation": "global micro over 240x240 crop",
        "image_aggregation": "mean of top 576 pixels",
    }


def evaluate_reconstructions(images, reconstructions, masks, labels, *, split, thresholds=None):
    """Evaluate both scores on the same 256x256 reconstruction pairs.

    Test requires separate frozen pixel/image thresholds for each scoring method.
    The caller supplies and records model, seed and threshold provenance.
    """
    images = np.asarray(images)
    reconstructions = np.asarray(reconstructions)
    masks = np.asarray(masks)
    if images.ndim != 3 or images.shape[1:] != (256, 256):
        raise ValueError("images must have shape (N, 256, 256)")
    if reconstructions.shape != images.shape or masks.shape != images.shape or not len(images):
        raise ValueError("reconstructions and masks must match nonempty images")
    methods = ("pixel_difference", "local_ssim")
    if thresholds is not None and set(thresholds) != set(methods):
        raise ValueError("thresholds must contain both score methods")
    if split == "test" and thresholds is None:
        raise ValueError("test requires frozen validation thresholds for both scores")
    maps = {method: [] for method in methods}
    for image, reconstruction in zip(images, reconstructions):
        paired = paired_score_maps(image, reconstruction)
        for method in methods:
            maps[method].append(paired[method])
    return {
        method: evaluate_maps(
            np.stack(maps[method]),
            masks[:, 8:-8, 8:-8],
            labels,
            split=split,
            thresholds=None if thresholds is None else thresholds[method],
        )
        for method in methods
    }
