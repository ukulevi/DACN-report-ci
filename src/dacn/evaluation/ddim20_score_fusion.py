"""Fixed equal-weight sums of calibration-normalized DDIM20 validation scores."""

import csv
import hashlib
import io
import time
from pathlib import Path

import numpy as np

from dacn.evaluation.ddim20_local_ssim import CACHE_SHA256, DATASET_REVISION, PARAMETERS, write_json
from dacn.evaluation.registered_partition import registered_partition
from dacn.evaluation.scores import paired_score_maps
from dacn.evaluation.validation import evaluate_maps

FUSIONS = {
    "lpips_residual": ("lpips", "residual"),
    "lpips_ssim": ("lpips", "ssim"),
    "residual_ssim": ("residual", "ssim"),
    "lpips_ssim_residual": ("lpips", "ssim", "residual"),
}


def normalize_scores(raw_maps, normal_calibration):
    """Midrank empirical CDF fitted only on normal calibration pixels."""
    if set(raw_maps) != {"lpips", "residual", "ssim"} or not len(normal_calibration):
        raise ValueError("Require three raw maps and normal calibration examples")
    normalized, references = {}, {}
    shapes = {np.asarray(v).shape for v in raw_maps.values()}
    if len(shapes) != 1:
        raise ValueError("Raw score map shapes differ")
    for name, raw in raw_maps.items():
        raw = np.asarray(raw, dtype=np.float64)
        if not np.isfinite(raw).all():
            raise ValueError("Raw score maps must be finite")
        reference = np.sort(raw[normal_calibration].ravel())
        normalized[name] = (
            np.searchsorted(reference, raw, side="left")
            + np.searchsorted(reference, raw, side="right")
        ) / (2 * len(reference))
        references[name] = reference
    return normalized, references


def combine_normalized(normalized):
    """A '+' means a fixed arithmetic mean, never a product or learned weights."""
    if set(normalized) != {"lpips", "residual", "ssim"}:
        raise ValueError("Require exactly the three normalized score maps")
    values = {k: np.asarray(v, dtype=np.float64) for k, v in normalized.items()}
    if len({v.shape for v in values.values()}) != 1 or any(
        not np.isfinite(v).all() or (v < 0).any() or (v > 1).any() for v in values.values()
    ):
        raise ValueError("Normalized maps must align and be finite within [0,1]")
    return {
        name: sum(values[key] for key in members) / len(members)
        for name, members in FUSIONS.items()
    }


def evaluate_fusions(raw_maps, masks, labels, calibration, holdout, output, on_progress=None):
    """Freeze all normalizers/thresholds before reporting any holdout outcomes."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    calibration, holdout = np.asarray(calibration), np.asarray(holdout)
    if sorted(np.concatenate([calibration, holdout]).tolist()) != list(range(len(labels))):
        raise ValueError("Calibration and holdout must partition the samples without overlap")
    normal = calibration[np.asarray(labels)[calibration] == 0]
    tick = time.perf_counter()
    normalized, references = normalize_scores(raw_maps, normal)
    fusion_maps = combine_normalized(normalized)
    normalization_seconds = time.perf_counter() - tick
    reference_path = output / "private_normal_cdf.npz"
    np.savez_compressed(reference_path, **references)
    definition = {
        "normalization": "midrank empirical CDF on normal calibration pixels only; not per-image",
        "normalizer_fit_samples": len(normal),
        "normalizer_fit_pixels_per_score": len(next(iter(references.values()))),
        "normalizer_sha256": hashlib.sha256(reference_path.read_bytes()).hexdigest(),
        "fusions": {
            name: {key: 1 / len(members) for key in members} for name, members in FUSIONS.items()
        },
        "threshold_rule": PARAMETERS["threshold_rule"],
        "calibration_samples": len(calibration),
        "holdout_samples": len(holdout),
        "candidate_selection": "none; all four prespecified fusions reported",
        "score_comparison": ">=",
        "image_score": PARAMETERS["image_score"],
        "ssim_definition": PARAMETERS,
    }
    candidates = {**{f"raw_{k}": v for k, v in raw_maps.items()}, **fusion_maps}
    calibrated = {}
    for name, values in candidates.items():
        measured = evaluate_maps(
            values[calibration], masks[calibration], labels[calibration], split="validation"
        )
        calibrated[name] = {
            "thresholds": {level: measured[level]["threshold"] for level in ("pixel", "image")},
            "metrics": measured,
        }
        if on_progress:
            on_progress({"calibrated_score": name})
    write_json(output / "frozen_calibration.json", {**definition, "scores": calibrated})
    holdout_results = {}
    for name, values in candidates.items():
        holdout_results[name] = evaluate_maps(
            values[holdout],
            masks[holdout],
            labels[holdout],
            split="validation",
            thresholds=calibrated[name]["thresholds"],
        )
        if on_progress:
            on_progress({"evaluated_holdout_score": name})
    # Keep maps and fitted pixel distributions private for replay, not in Git.
    np.savez_compressed(
        output / "private_fusion_maps.npz", masks=masks, labels=labels, **fusion_maps
    )
    return {
        "definition": definition,
        "calibration": calibrated,
        "holdout_results": holdout_results,
        "normalization_seconds": normalization_seconds,
    }


def evaluate_cached_fusions(*, cache, manifest, index, partition, output, on_progress=None):
    """Reuse LPIPS/L1 maps from the pinned DDIM20 cache; recompute CPU SSIM."""
    if manifest.get("dataset_revision") != DATASET_REVISION:
        raise ValueError("Require the registered dataset revision")
    paths = {"cache": cache, "index": index, "partition": partition}
    pins = {
        "cache": CACHE_SHA256,
        "index": manifest["artifacts"]["validation_index"],
        "partition": manifest["artifacts"]["validation_partition"],
    }
    captured = {}
    for name, path in paths.items():
        if Path(path).is_symlink():
            raise ValueError("Symlinked input is forbidden")
        payload = Path(path).read_bytes()
        if hashlib.sha256(payload).hexdigest() != pins[name]:
            raise ValueError("Input checksum differs: " + name)
        captured[name] = payload
    rows = list(csv.DictReader(io.StringIO(captured["index"].decode())))
    if len(rows) != 83 or any(row["split"] != "val" for row in rows):
        raise ValueError("Only the registered83 validation samples are allowed")
    groups = np.array([row["patient_group_id"] for row in rows])
    calibration, holdout = registered_partition(captured["partition"], captured["index"], groups)
    with np.load(io.BytesIO(captured.pop("cache")), allow_pickle=False) as archive:
        images, initial = archive["images"], archive["initials"]
        masks, labels = archive["masks"], archive["labels"]
        raw = {"residual": archive["ddpm_l1"], "lpips": archive["initial_lpips_only"]}
        if (
            any(v.shape != (83, 256, 256) for v in (images, initial, masks))
            or labels.shape != (83,)
            or any(v.shape != (83, 240, 240) for v in raw.values())
            or not np.isin(masks, [0, 1]).all()
            or not np.isin(labels, [0, 1]).all()
            or not np.array_equal(masks.any(axis=(1, 2)), labels.astype(bool))
            or not np.array_equal(labels, [int(row["label"]) for row in rows])
        ):
            raise ValueError("Cached score/image/label/mask alignment differs")
        tick = time.perf_counter()
        raw["ssim"] = np.stack(
            [paired_score_maps(x, y)["local_ssim"] for x, y in zip(images, initial, strict=True)]
        )
        ssim_seconds = time.perf_counter() - tick
        result = evaluate_fusions(
            raw, masks[:, 8:-8, 8:-8], labels, calibration, holdout, output, on_progress
        )
    result.update(
        status="completed",
        split="validation",
        samples_scored=83,
        calibration_groups=len(set(groups[calibration])),
        holdout_groups=len(set(groups[holdout])),
        dataset_revision=DATASET_REVISION,
        input_checksums=pins,
        checkpoint_sha256=manifest["artifacts"]["checkpoint"],
        method="Gaussian EMA30; initial DDIM20(t300), eta0; shared cached reconstruction",
        training_seeds=1,
        sampling_seeds=1,
        original_sampling_base_seed=17,
        new_unet_calls=0,
        new_lpips_calls=0,
        test_accessed=False,
        training_performed=False,
        gpu=None,
        vram_bytes=None,
        cpu_ssim_seconds=ssim_seconds,
        limitations="Post-final-test exploratory validation extension; repeatedly inspected validation; one seed, no test tuning or final-protocol replacement",
    )
    write_json(Path(output) / "results.json", result)
    return result
