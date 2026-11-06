"""Fixed K4 scoring; all learned statistics use normal calibration residuals only."""

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view


def ensemble_statistics(draws):
    draws = np.asarray(draws, dtype=np.float64)
    if draws.ndim != 4 or draws.shape[0] != 4 or not np.isfinite(draws).all():
        raise ValueError("Expected finite four draws with shape K,N,H,W")
    return draws.mean(axis=0), draws.var(axis=0, ddof=1)


def residual_vectors(residual):
    residual = np.asarray(residual, dtype=np.float64)
    if residual.ndim != 2 or min(residual.shape) < 2 or not np.isfinite(residual).all():
        raise ValueError("Expected finite2D signed residual")
    return sliding_window_view(np.pad(residual, 1, mode="reflect"), (3, 3)).reshape(-1, 9)


def fit_normal_calibration(residuals, variances, labels, calibration_indices):
    residuals = np.asarray(residuals, dtype=np.float64)
    variances = np.asarray(variances, dtype=np.float64)
    labels = np.asarray(labels)
    calibration_indices = np.asarray(calibration_indices, dtype=int)
    if (
        residuals.ndim != 3
        or residuals.shape != variances.shape
        or labels.shape != (len(residuals),)
        or not np.isfinite(residuals).all()
        or not np.isfinite(variances).all()
        or (variances < 0).any()
        or not np.isin(labels, [0, 1]).all()
        or calibration_indices.ndim != 1
        or not len(calibration_indices)
        or (calibration_indices < 0).any()
        or (calibration_indices >= len(residuals)).any()
    ):
        raise ValueError("Invalid calibration residuals/variances/indices")
    normal = calibration_indices[labels[calibration_indices] == 0]
    if not len(normal):
        raise ValueError("Normal calibration images required")
    floor = max(1e-6, float(np.median(variances[normal])))
    total, sum_vector, sum_outer = 0, np.zeros(9), np.zeros((9, 9))
    for index in normal:
        vectors = residual_vectors(residuals[index])
        total += len(vectors)
        sum_vector += vectors.sum(axis=0)
        sum_outer += vectors.T @ vectors
    mean = sum_vector / total
    covariance = sum_outer / total - np.outer(mean, mean)
    covariance = (covariance + covariance.T) / 2
    shrinkage = 0.1
    ridge = max(1e-8, 1e-4 * float(np.trace(covariance)) / 9)
    regularized = (
        (1 - shrinkage) * covariance + shrinkage * np.diag(np.diag(covariance)) + ridge * np.eye(9)
    )
    values, vectors = np.linalg.eigh(regularized)
    precision = (vectors * (1 / np.maximum(values, ridge))) @ vectors.T
    return {
        "variance_floor": floor,
        "mahalanobis_mean": mean,
        "mahalanobis_covariance": regularized,
        "mahalanobis_precision": precision,
        "shrinkage": shrinkage,
        "ridge": ridge,
        "normal_calibration_samples": len(normal),
        "normal_calibration_vectors": int(total),
    }


def normalized_score(residual, variance, fitted):
    residual, variance = np.asarray(residual), np.asarray(variance)
    if (
        residual.shape != variance.shape
        or not np.isfinite(residual).all()
        or not np.isfinite(variance).all()
        or (variance < 0).any()
    ):
        raise ValueError("Invalid residual/variance score inputs")
    return residual**2 / (variance + fitted["variance_floor"])


def mahalanobis_score(residual, fitted):
    vectors = residual_vectors(residual) - fitted["mahalanobis_mean"]
    squared = np.einsum(
        "ni,ij,nj->n", vectors, fitted["mahalanobis_precision"], vectors, optimize=True
    )
    return np.maximum(squared, 0).reshape(np.asarray(residual).shape)
