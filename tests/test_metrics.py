import pytest

from dacn.evaluation.metrics import compute_binary_metrics


def test_perfect_ranking():
    result = compute_binary_metrics([0, 0, 1, 1], [0.1, 0.2, 0.7, 0.9], 0.5)
    assert result["auroc"] == result["auprc"] == result["f1"] == 1.0


def test_only_one_class_is_invalid():
    with pytest.raises(ValueError, match="both"):
        compute_binary_metrics([0, 0], [0.1, 0.2])


def test_nan_is_invalid():
    with pytest.raises(ValueError, match="finite"):
        compute_binary_metrics([0, 1], [float("nan"), 0.2])
