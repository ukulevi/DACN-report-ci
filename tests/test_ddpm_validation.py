"""Tests for validation-only partial-DDPM evaluation helpers."""

import numpy as np
import pytest

pytest.importorskip("torch", reason="DDPM validation requires the train extra")

from dacn.evaluation.ddpm_validation import select_validation_indices


def test_validation_subset_is_deterministic_and_contains_both_classes():
    labels = np.asarray([0] * 5 + [1] * 7, dtype=np.uint8)
    first = select_validation_indices(labels, 6, seed=17)
    second = select_validation_indices(labels, 6, seed=17)

    assert first == second
    assert len(first) == 6
    assert set(labels[first]) == {0, 1}


@pytest.mark.parametrize("limit", [0, 1, 13])
def test_validation_subset_rejects_invalid_limits(limit):
    labels = np.asarray([0, 0, 1, 1], dtype=np.uint8)
    with pytest.raises(ValueError, match="sample limit"):
        select_validation_indices(labels, limit, seed=17)
