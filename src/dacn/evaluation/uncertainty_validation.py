"""K4 ensemble controls and normal-calibrated scoring with grouped holdout."""

import json
import os
from pathlib import Path

import numpy as np
from sklearn.model_selection import StratifiedGroupKFold

from dacn.evaluation.scores import paired_score_maps
from dacn.evaluation.uncertainty_scores import (
    ensemble_statistics,
    fit_normal_calibration,
    mahalanobis_score,
    normalized_score,
)
from dacn.evaluation.validation import evaluate_maps

BASE_SEEDS = (17, 1017, 2017, 3017)


def save_draw_prefix(path, values, *, base_seed, checkpoint_sha, index_sha):
    path = Path(path)
    temporary = path.with_suffix(".npz.tmp")
    with temporary.open("wb") as stream:
        np.savez_compressed(
            stream,
            reconstructions=np.asarray(values, dtype=np.float32),
            base_seed=base_seed,
            checkpoint_sha=checkpoint_sha,
            index_sha=index_sha,
        )
    os.replace(temporary, path)


def load_draw_prefix(path, *, base_seed, checkpoint_sha, index_sha, max_samples):
    path = Path(path)
    if not path.exists():
        return np.empty((0, 256, 256), dtype=np.float32)
    with np.load(path, allow_pickle=False) as cached:
        if (
            int(cached["base_seed"]) != base_seed
            or str(cached["checkpoint_sha"]) != checkpoint_sha
            or str(cached["index_sha"]) != index_sha
        ):
            raise ValueError("Draw-prefix provenance differs")
        values = cached["reconstructions"].copy()
    if (
        values.ndim != 3
        or values.shape[1:] != (256, 256)
        or len(values) > max_samples
        or not np.isfinite(values).all()
        or (np.abs(values) > 1).any()
    ):
        raise ValueError("Invalid draw-prefix arrays")
    return values


def evaluate_ensemble(images, draws, masks, labels, groups, output_dir, on_calibrated=None):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    images, draws, masks, labels, groups = [
        np.asarray(x) for x in (images, draws, masks, labels, groups)
    ]
    if (
        images.ndim != 3
        or images.shape[1:] != (256, 256)
        or draws.shape != (4, *images.shape)
        or masks.shape != images.shape
        or labels.shape != (len(images),)
        or groups.shape != labels.shape
    ):
        raise ValueError("Invalid ensemble evaluation arrays")
    calibration, holdout = next(
        StratifiedGroupKFold(2, shuffle=True, random_state=17).split(
            np.zeros(len(labels)), labels, groups
        )
    )
    if not set(groups[calibration]).isdisjoint(groups[holdout]):
        raise ValueError("Patient-group leakage")
    mean, variance = ensemble_statistics(draws)
    residual = (images - mean)[:, 8:-8, 8:-8]
    variance_crop = variance[:, 8:-8, 8:-8]
    fitted = fit_normal_calibration(residual, variance_crop, labels, calibration)
    # Persist all calibration parameters before accessing any holdout metric.
    fit_json = {k: v.tolist() if isinstance(v, np.ndarray) else v for k, v in fitted.items()}
    (output_dir / "calibration_parameters.json").write_text(
        json.dumps(fit_json, indent=2, allow_nan=False) + "\n"
    )
    if on_calibrated:
        on_calibrated({"normal_calibration_samples": fitted["normal_calibration_samples"]})
    maps = {
        name: []
        for name in (
            "single_absolute",
            "single_ssim",
            "mean4_absolute",
            "mean4_ssim",
            "variance_normalized",
            "patch_mahalanobis",
        )
    }
    quality = {"single": [], "mean4": []}
    for i in range(len(images)):
        foreground = images[i, 8:-8, 8:-8] > -0.95
        for name, prediction in (("single", draws[0, i]), ("mean4", mean[i])):
            paired = paired_score_maps(images[i], prediction)
            maps[f"{name}_absolute"].append(paired["pixel_difference"])
            maps[f"{name}_ssim"].append(paired["local_ssim"])
            if labels[i] == 0:
                quality[name].append(
                    (
                        float((paired["pixel_difference"][foreground] / 2).mean()),
                        float((1 - paired["local_ssim"][foreground]).mean()),
                    )
                )
        maps["variance_normalized"].append(normalized_score(residual[i], variance_crop[i], fitted))
        maps["patch_mahalanobis"].append(mahalanobis_score(residual[i], fitted))
    result = {
        "method": "gaussian_EMA30_K4_normal_calibrated_scoring",
        "samples": len(images),
        "split": "validation",
        "held_out_test_used": False,
        "ensemble_size": 4,
        "draw_base_seeds": list(BASE_SEEDS),
        "timesteps_one_based": [300],
        "partition_seed": 17,
        "partition": {
            name: {"samples": len(idx), "groups": len(np.unique(groups[idx]))}
            for name, idx in (("calibration", calibration), ("holdout", holdout))
        },
        "calibration": fit_json,
        "metrics": {},
        "score_diagnostics": {},
        "normal_reconstruction_quality": {
            name: {
                "foreground_mae_01": float(np.mean([v[0] for v in values])),
                "foreground_ssim": float(np.mean([v[1] for v in values])),
            }
            for name, values in quality.items()
        },
        "caveat": "One fixed ensemble/model on previously inspected validation. Variance is stochastic reconstruction variation, not calibrated epistemic uncertainty. Normalization may suppress lesion scores. Mahalanobis covariance is pooled normal-calibration signed3x3 residuals, not rank-deficient per-image covariance fromK4. No holdout tuning/test use.",
    }
    cropped_masks = masks[:, 8:-8, 8:-8]
    foreground = images[:, 8:-8, 8:-8] > -0.95
    for name, values in maps.items():
        values = np.asarray(values)
        fit = evaluate_maps(
            values[calibration], cropped_masks[calibration], labels[calibration], split="validation"
        )
        thresholds = {level: fit[level]["threshold"] for level in ("pixel", "image")}
        result["metrics"][name] = {
            "calibration": fit,
            "group_holdout": evaluate_maps(
                values[holdout],
                cropped_masks[holdout],
                labels[holdout],
                split="validation",
                thresholds=thresholds,
            ),
            "all_validation_descriptive": evaluate_maps(
                values, cropped_masks, labels, split="validation"
            ),
        }
        lesion_means, nonlesion_means = [], []
        for index in np.flatnonzero(labels == 1):
            lesion = cropped_masks[index].astype(bool)
            lesion_means.append(float(values[index][lesion].mean()))
            nonlesion_means.append(float(values[index][foreground[index] & ~lesion].mean()))
        result["score_diagnostics"][name] = {
            "abnormal_slice_macro_lesion": float(np.mean(lesion_means)),
            "abnormal_slice_macro_nonlesion_foreground": float(np.mean(nonlesion_means)),
        }
    result["variance_diagnostics"] = {
        "normal_foreground_mean": float(
            variance_crop[(labels == 0)[:, None, None] & foreground].mean()
        ),
        "lesion_mean": float(variance_crop[cropped_masks.astype(bool)].mean()),
        "definition": "Unbiased K4 variance, original[-1,1] units squared; descriptions on all validation, never used to retune calibration",
    }
    np.savez_compressed(
        output_dir / "private_ensemble_summary.npz",
        images=images,
        mean=mean.astype(np.float32),
        variance=variance.astype(np.float32),
        masks=masks,
        labels=labels,
    )
    (output_dir / "results.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    return result
