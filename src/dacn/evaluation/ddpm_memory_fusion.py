"""Normal-calibrated equal-weight fusion on the existing grouped validation fold."""

import hashlib
import json
from pathlib import Path

import numpy as np
from sklearn.model_selection import StratifiedGroupKFold

from dacn.evaluation.scores import paired_score_maps
from dacn.evaluation.validation import evaluate_maps

ALPHA = 0.5


def verified_cache(path, checksum, dataset, index_checksum):
    """Require byte-exact registered cache and full ordered image/mask/label match."""
    path = Path(path)
    if hashlib.sha256(path.read_bytes()).hexdigest() != checksum:
        raise ValueError("Baseline cache checksum differs")
    index_path = Path(dataset.split_dir) / "index.csv"
    if hashlib.sha256(index_path.read_bytes()).hexdigest() != index_checksum:
        raise ValueError("Validation index checksum differs")
    if dataset.split != "val" or not dataset.include_masks or len(dataset) != 83:
        raise ValueError("Expected full validation dataset")
    with np.load(path, allow_pickle=False) as archive:
        cached = {
            key: archive[key].copy() for key in ("images", "reconstructions", "masks", "labels")
        }
    expected = (83, 256, 256)
    if any(cached[key].shape != expected for key in ("images", "reconstructions", "masks")):
        raise ValueError("Baseline cache shapes differ")
    if cached["labels"].shape != (83,) or not all(np.isfinite(v).all() for v in cached.values()):
        raise ValueError("Invalid baseline arrays")
    for index in range(len(dataset)):
        image, mask, label = dataset[index]
        if (
            not np.array_equal(image[0].numpy(), cached["images"][index])
            or not np.array_equal(mask.numpy(), cached["masks"][index])
            or int(label) != int(cached["labels"][index])
        ):
            raise ValueError("Baseline cache/data ordering differs")
    return cached


def fit_robust_normalization(maps, labels, calibration):
    normal = np.asarray(calibration)[np.asarray(labels)[calibration] == 0]
    if not len(normal):
        raise ValueError("Normal calibration samples required")
    result = {}
    for name, values in maps.items():
        values = np.asarray(values)
        if not np.isfinite(values).all():
            raise ValueError("Nonfinite score maps")
        q25, median, q75 = np.quantile(values[normal], [0.25, 0.5, 0.75])
        result[name] = {"median": float(median), "scale": float(max(q75 - q25, 1e-6))}
    return result


def fuse_maps(maps, fitted):
    normalized = {
        name: (values - fitted[name]["median"]) / fitted[name]["scale"]
        for name, values in maps.items()
    }
    return {
        **maps,
        "absolute_memory_fusion": ALPHA * normalized["ddpm_absolute"]
        + (1 - ALPHA) * normalized["memory_bank"],
        "ssim_memory_fusion": ALPHA * normalized["ddpm_ssim"]
        + (1 - ALPHA) * normalized["memory_bank"],
    }


def recall_threshold(scores, masks, target):
    positives = np.asarray(scores)[np.asarray(masks).astype(bool)]
    if not len(positives) or not 0 < target <= 1:
        raise ValueError("Positive calibration pixels and valid recall required")
    return float(np.quantile(positives, 1 - target, method="lower"))


def evaluate_fusion(cached, memory_scores, groups, output_dir, *, provenance, on_calibrated=None):
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    images, reconstructions, masks, labels = [
        cached[k] for k in ("images", "reconstructions", "masks", "labels")
    ]
    if memory_scores.shape != (len(images), 240, 240) or not np.isfinite(memory_scores).all():
        raise ValueError("Memory score alignment/values differ")
    calibration, holdout = next(
        StratifiedGroupKFold(2, shuffle=True, random_state=17).split(
            np.zeros(len(labels)), labels, groups
        )
    )
    if len(images) != 83 or len(calibration) != 40 or len(holdout) != 43:
        raise ValueError("Registered83/40/43 partition required")
    if not set(np.asarray(groups)[calibration]).isdisjoint(np.asarray(groups)[holdout]):
        raise ValueError("Group leakage")
    paired = [
        paired_score_maps(image, reconstruction)
        for image, reconstruction in zip(images, reconstructions, strict=True)
    ]
    maps = {
        "ddpm_absolute": np.stack([v["pixel_difference"] for v in paired]),
        "ddpm_ssim": np.stack([v["local_ssim"] for v in paired]),
        "memory_bank": memory_scores,
    }
    fitted = fit_robust_normalization(maps, labels, calibration)
    maps = fuse_maps(maps, fitted)
    maps["brightness"] = (images[:, 8:-8, 8:-8] + 1) / 2
    cropped = masks[:, 8:-8, 8:-8]
    calibration_metrics, thresholds, recalls = {}, {}, {}
    for name, values in maps.items():
        fit = evaluate_maps(
            values[calibration], cropped[calibration], labels[calibration], split="validation"
        )
        calibration_metrics[name] = fit
        thresholds[name] = {level: fit[level]["threshold"] for level in ("pixel", "image")}
        recalls[name] = {
            str(r): recall_threshold(values[calibration], cropped[calibration], r)
            for r in (0.5, 0.8)
        }
    parameters = {
        "alpha": ALPHA,
        "normalization": fitted,
        "thresholds": thresholds,
        "matched_recall_thresholds": recalls,
        "normal_calibration_samples": int((labels[calibration] == 0).sum()),
        "normalization_domain": "all pixels of normal calibration240crop; IQR floor1e-6",
    }
    # Freeze every fitted value before any holdout metric is calculated.
    (output / "calibration_parameters.json").write_text(
        json.dumps(parameters, indent=2, allow_nan=False) + "\n"
    )
    if on_calibrated:
        on_calibrated(parameters)
    metrics = {}
    for name, values in maps.items():
        hold = evaluate_maps(
            values[holdout],
            cropped[holdout],
            labels[holdout],
            split="validation",
            thresholds=thresholds[name],
        )
        matched = {}
        for target, threshold in recalls[name].items():
            matched[target] = evaluate_maps(
                values[holdout],
                cropped[holdout],
                labels[holdout],
                split="validation",
                thresholds={**thresholds[name], "pixel": threshold},
            )
        metrics[name] = {
            "calibration": calibration_metrics[name],
            "group_holdout": hold,
            "calibration_recall_operating_points": matched,
        }
    result = {
        "method": "Gaussian_EMA30_normal_memory_fusion_v1",
        "samples": len(images),
        "split": "validation",
        "held_out_test_used": False,
        "partition_seed": 17,
        "partition": {"calibration": 40, "holdout": 43},
        "parameters": parameters,
        "provenance": provenance,
        "metrics": metrics,
        "caveat": "Exploratory on repeatedly inspected validation; one model/bank seed. Recall targets fitted on calibration, not matched by retuning on holdout. Compare both fusion arms with memory-only and brightness controls.",
    }
    private = {
        "images": images,
        "reconstructions": reconstructions,
        "masks": masks,
        "labels": labels,
        **maps,
    }
    for name in ("absolute_memory_fusion", "ssim_memory_fusion"):
        private[name + "_predicted_mask"] = maps[name] >= thresholds[name]["pixel"]
    np.savez_compressed(output / "private_maps.npz", **private)
    save_examples(images, reconstructions, maps, cropped, labels, output)
    (output / "results.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    return result


def save_examples(images, reconstructions, maps, masks, labels, output):
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    selected = [int(np.flatnonzero(labels == label)[0]) for label in (0, 1)]
    for number, index in enumerate(selected):
        fig, axes = plt.subplots(1, 8, figsize=(24, 3))
        values = [
            images[index, 8:-8, 8:-8],
            reconstructions[index, 8:-8, 8:-8],
            maps["ddpm_absolute"][index],
            maps["ddpm_ssim"][index],
            maps["memory_bank"][index],
            maps["absolute_memory_fusion"][index],
            maps["ssim_memory_fusion"][index],
            masks[index],
        ]
        titles = [
            "Input",
            "DDPM",
            "Absolute",
            "SSIM",
            "Memory",
            "Abs fusion",
            "SSIM fusion",
            "Ground truth",
        ]
        for axis, value, title in zip(axes, values, titles, strict=True):
            axis.imshow(
                value, cmap="gray" if title in {"Input", "DDPM", "Ground truth"} else "inferno"
            )
            axis.set_title(title)
            axis.axis("off")
        fig.tight_layout()
        fig.savefig(Path(output) / f"private_example_{number}.png")
        plt.close(fig)
