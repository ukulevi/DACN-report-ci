import json

import numpy as np
import pytest

from dacn.evaluation import uncertainty_validation as evaluation
from dacn.evaluation.uncertainty_scores import (
    ensemble_statistics,
    fit_normal_calibration,
    mahalanobis_score,
    normalized_score,
    residual_vectors,
)


def test_k4_mean_and_unbiased_variance():
    draws = np.broadcast_to(np.arange(4)[:, None, None, None], (4, 2, 3, 3))
    mean, variance = ensemble_statistics(draws)
    np.testing.assert_allclose(mean, 1.5)
    np.testing.assert_allclose(variance, 5 / 3)
    with pytest.raises(ValueError):
        ensemble_statistics(draws[:3])


def test_fit_ignores_abnormal_calibration_and_all_holdout_including_normal():
    rng = np.random.default_rng(17)
    residual = rng.normal(size=(4, 5, 5))
    variance = np.abs(rng.normal(size=residual.shape))
    labels = np.array([0, 1, 0, 1])
    first = fit_normal_calibration(residual, variance, labels, [0, 1])
    residual[1:] += 1000
    variance[1:] += 1000
    second = fit_normal_calibration(residual, variance, labels, [0, 1])
    for key in first:
        np.testing.assert_equal(first[key], second[key])
    assert first["normal_calibration_samples"] == 1
    assert first["normal_calibration_vectors"] == 25


def test_singular_covariance_and_zero_variance_are_finite():
    residual = np.zeros((2, 5, 5))
    variance = np.zeros_like(residual)
    fitted = fit_normal_calibration(residual, variance, [0, 1], [0, 1])
    assert fitted["variance_floor"] == 1e-6
    assert np.linalg.eigvalsh(fitted["mahalanobis_precision"]).min() > 0
    np.testing.assert_equal(mahalanobis_score(residual[0], fitted), 0)
    assert np.isfinite(normalized_score(residual, variance, fitted)).all()
    assert (
        normalized_score(np.ones((2, 2)), np.full((2, 2), 9), {"variance_floor": 1}).mean() == 0.1
    )
    with pytest.raises(ValueError, match="Normal calibration"):
        fit_normal_calibration(residual, variance, [1, 1], [0])


def test_mahalanobis_uses_full_patch_covariance():
    residual = np.arange(25).reshape(5, 5) / 25
    fitted = {"mahalanobis_mean": np.zeros(9), "mahalanobis_precision": np.eye(9)}
    expected = (residual_vectors(residual) ** 2).sum(1).reshape(5, 5)
    np.testing.assert_allclose(mahalanobis_score(residual, fitted), expected)
    fitted["mahalanobis_precision"][0, 1] = fitted["mahalanobis_precision"][1, 0] = 0.3
    assert not np.allclose(mahalanobis_score(residual, fitted), expected)


def test_private_draw_prefix_requires_exact_provenance(tmp_path):
    path = tmp_path / "draw1.npz"
    expected = {
        "base_seed": 1017,
        "checkpoint_sha": "synthetic-checkpoint",
        "index_sha": "synthetic-index",
    }
    values = np.zeros((3, 256, 256), dtype=np.float32)
    evaluation.save_draw_prefix(path, values, **expected)
    np.testing.assert_array_equal(
        evaluation.load_draw_prefix(path, **expected, max_samples=8), values
    )
    with pytest.raises(ValueError, match="provenance"):
        evaluation.load_draw_prefix(path, **{**expected, "base_seed": 2017}, max_samples=8)
    with pytest.raises(ValueError, match="arrays"):
        evaluation.load_draw_prefix(path, **expected, max_samples=2)


def test_ensemble_thresholds_fit_only_calibration_and_save_parameters_first(tmp_path, monkeypatch):
    rng = np.random.default_rng(17)
    images = np.full((8, 256, 256), 0.2, dtype=np.float32)
    labels = np.array([0, 1] * 4, dtype=np.uint8)
    masks = np.zeros_like(images, dtype=np.uint8)
    masks[labels == 1, 100:120, 100:120] = 1
    draws = np.clip(images[None] + rng.normal(0, 0.05, (4, *images.shape)), -1, 1)
    calls = []

    def measure(scores, mask, y, *, split, thresholds=None):
        assert split == "validation"
        fitted = json.loads((tmp_path / "calibration_parameters.json").read_text())
        assert fitted["normal_calibration_samples"] == 2
        calls.append(thresholds)
        return {"pixel": {"threshold": 0.2}, "image": {"threshold": 0.3}}

    monkeypatch.setattr(evaluation, "evaluate_maps", measure)
    result = evaluation.evaluate_ensemble(
        images, draws, masks, labels, np.repeat(np.arange(4), 2), tmp_path
    )
    assert calls == [None, {"pixel": 0.2, "image": 0.3}, None] * 6
    assert len(result["metrics"]) == 6 and not result["held_out_test_used"]
    assert result["partition"]["holdout"] == {"samples": 4, "groups": 2}
    assert (tmp_path / "private_ensemble_summary.npz").exists()
