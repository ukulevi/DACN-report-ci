"""Paired cluster bootstrap with fixed scores and thresholds; aggregate output only."""

import numpy as np


def _weighted_ap(y, group, ends, multiplicity):
    tp = 0.0
    total = 0.0
    previous_tp = 0.0
    numerator = 0.0
    for i in range(len(y)):
        weight = multiplicity[group[i]]
        tp += weight * y[i]
        total += weight
        if ends[i]:
            if total > 0:
                numerator += (tp - previous_tp) * tp / total
            previous_tp = tp
    return numerator / tp if tp > 0 else np.nan


def paired_group_bootstrap(truth, maps, groups, thresholds, *, reference, replicates=1000, seed=17):
    """Resample whole groups with replacement; identical draws for all candidates.

    Scores, candidate names, thresholds and reference must be frozen beforehand.
    Percentile intervals are conditional on the fitted models/protocol and cohort.
    Optional numba accelerates the exact tied-score AP loop for medical pixel maps.
    """
    y = np.asarray(truth)
    groups = np.asarray(groups)
    if y.ndim < 2 or len(groups) != len(y) or not np.isin(y, [0, 1]).all():
        raise ValueError("Expected binary per-sample truth and one group per sample")
    if reference not in maps or set(maps) != set(thresholds) or replicates < 1:
        raise ValueError("Require a reference, fixed thresholds and positive replicate count")
    _, row_groups = np.unique(groups, return_inverse=True)
    n_groups = int(row_groups.max()) + 1
    flat_groups = np.repeat(row_groups, y[0].size)
    positives = np.bincount(flat_groups, weights=y.ravel(), minlength=n_groups)
    negatives = np.bincount(flat_groups, weights=1 - y.ravel(), minlength=n_groups)
    ap_fn = _weighted_ap
    try:
        from numba import njit

        ap_fn = njit(_weighted_ap)
    except ImportError:
        pass
    prepared = {}
    for name, values in maps.items():
        values = np.asarray(values)
        if (
            values.shape != y.shape
            or not np.isfinite(values).all()
            or not np.isfinite(thresholds[name])
        ):
            raise ValueError("Expected finite score maps and fixed finite thresholds")
        order = np.argsort(-values.ravel(), kind="stable")
        ordered = values.ravel()[order]
        ends = np.r_[ordered[1:] != ordered[:-1], True]
        pred = values.ravel() >= thresholds[name]
        tp = np.bincount(flat_groups, weights=pred * y.ravel(), minlength=n_groups)
        fp = np.bincount(flat_groups, weights=pred * (1 - y.ravel()), minlength=n_groups)
        prepared[name] = (y.ravel()[order], flat_groups[order], ends, tp, fp, positives - tp)
    draws = np.random.default_rng(seed)
    samples = {name: [] for name in maps}
    skipped = 0
    for _ in range(replicates):
        weights = np.bincount(draws.integers(0, n_groups, size=n_groups), minlength=n_groups)
        if np.dot(weights, positives) == 0 or np.dot(weights, negatives) == 0:
            skipped += 1
            continue
        for name, (ordered_y, ordered_groups, ends, tp, fp, fn) in prepared.items():
            ap = ap_fn(ordered_y, ordered_groups, ends, weights)
            a, b, c = (float(np.dot(weights, v)) for v in (tp, fp, fn))
            samples[name].append([float(ap), 2 * a / (2 * a + b + c)])
    if skipped == replicates:
        raise ValueError("No bootstrap replicate contains both classes")
    baseline = np.asarray(samples[reference])
    results = {}
    ones = np.ones(n_groups, dtype=int)
    for name, values in samples.items():
        array = np.asarray(values)
        ordered_y, ordered_groups, ends, tp, fp, fn = prepared[name]
        a, b, c = tp.sum(), fp.sum(), fn.sum()
        results[name] = {
            "pixel_ap": float(ap_fn(ordered_y, ordered_groups, ends, ones)),
            "pixel_dice_micro": float(2 * a / (2 * a + b + c)),
            "threshold": float(thresholds[name]),
            "ap_ci95": np.percentile(array[:, 0], [2.5, 97.5]).tolist(),
            "dice_ci95": np.percentile(array[:, 1], [2.5, 97.5]).tolist(),
            "delta_ap_ci95": np.percentile(array[:, 0] - baseline[:, 0], [2.5, 97.5]).tolist(),
            "delta_dice_ci95": np.percentile(array[:, 1] - baseline[:, 1], [2.5, 97.5]).tolist(),
        }
    return {
        "method": "paired patient-group percentile bootstrap; fixed scores and thresholds",
        "reference": reference,
        "seed": seed,
        "requested_replicates": replicates,
        "valid_replicates": replicates - skipped,
        "skipped_missing_class": skipped,
        "groups": n_groups,
        "samples": len(y),
        "candidates": results,
    }
