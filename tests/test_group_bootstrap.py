import numpy as np
import pytest
from sklearn.metrics import average_precision_score

from dacn.evaluation.group_bootstrap import paired_group_bootstrap


def test_group_bootstrap_matches_explicit_group_replication_with_ties():
    truth = np.array([[0, 1, 0], [1, 0, 0], [0, 0, 1]], dtype=np.uint8)
    scores = np.array([[0.1, 0.5, 0.5], [0.5, 0.1, 0.2], [0.2, 0.2, 0.1]])
    groups = np.array(["a", "a", "b"])
    result = paired_group_bootstrap(
        truth,
        {"base": scores, "copy": scores.copy()},
        groups,
        {"base": 0.5, "copy": 0.5},
        reference="base",
        replicates=12,
        seed=17,
    )
    rng = np.random.default_rng(17)
    aps, dices = [], []
    for _ in range(12):
        draw = rng.integers(0, 2, size=2)
        rows = np.concatenate([np.flatnonzero(groups == ["a", "b"][i]) for i in draw])
        y, s = truth[rows].ravel(), scores[rows].ravel()
        aps.append(average_precision_score(y, s))
        pred = s >= 0.5
        tp = (pred & (y == 1)).sum()
        dices.append(2 * tp / (pred.sum() + y.sum()))
    assert result["candidates"]["base"]["ap_ci95"] == pytest.approx(np.percentile(aps, [2.5, 97.5]))
    assert result["candidates"]["base"]["dice_ci95"] == pytest.approx(
        np.percentile(dices, [2.5, 97.5])
    )
    assert result["candidates"]["copy"]["delta_ap_ci95"] == [0, 0]
    assert result["candidates"]["copy"]["delta_dice_ci95"] == [0, 0]


def test_missing_class_replicates_are_skipped_for_every_candidate():
    y = np.array([[0, 0], [1, 1]])
    s = y.astype(float)
    result = paired_group_bootstrap(
        y, {"a": s}, ["normal", "lesion"], {"a": 0.5}, reference="a", replicates=20
    )
    assert result["skipped_missing_class"] > 0
    assert result["valid_replicates"] + result["skipped_missing_class"] == 20
    assert result["candidates"]["a"]["ap_ci95"] == [1, 1]
