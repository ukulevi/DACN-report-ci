"""Validation-only Local SSIM scoring of the immutable cached DDIM20 outputs."""

import csv
import hashlib
import io
import json
import time
from pathlib import Path

import numpy as np

from dacn.evaluation.registered_partition import registered_partition
from dacn.evaluation.scores import paired_score_maps
from dacn.evaluation.validation import evaluate_maps

CACHE_SHA256 = "22656e1f63a0047d3dbe32f7833eea8e303988414e5942dec1ba5a1d9ed1ef19"
DATASET_REVISION = "350c74f2313e4e73ca65e7ffc6e9dcf74040b3b7"
PARAMETERS = {
    "score": "1-local_SSIM(input, initial_DDIM20_reconstruction)",
    "window": "11x11 Gaussian, sigma=1.5, reflect padding, population moments",
    "data_range": 2,
    "k1": 0.01,
    "k2": 0.03,
    "input_range": "[-1,1]; original signed-input contract, no [0,1] remapping",
    "metric_region": "full240x240 crop; no foreground filtering",
    "postprocessing": "none; no score clipping",
    "threshold_rule": "max F1 on original40 calibration; largest threshold on exact tie",
    "score_comparison": ">=",
    "image_score": "mean of top576 pixels",
}


def sha256_file(path):
    hasher = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            hasher.update(block)
    return hasher.hexdigest()


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(".partial")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def evaluate_partitions(maps, masks, labels, calibration, holdout, output):
    """Persist calibration-only thresholds before computing holdout metrics."""
    fit = evaluate_maps(
        maps[calibration], masks[calibration], labels[calibration], split="validation"
    )
    thresholds = {level: fit[level]["threshold"] for level in ("pixel", "image")}
    write_json(
        output / "thresholds.json",
        {
            "thresholds": thresholds,
            "fit_samples": len(calibration),
            "rule": PARAMETERS["threshold_rule"],
            "parameters": PARAMETERS,
        },
    )
    measured = evaluate_maps(
        maps[holdout], masks[holdout], labels[holdout], split="validation", thresholds=thresholds
    )
    return {"thresholds": thresholds, "calibration": fit, "group_holdout": measured}


def evaluate_cached_validation(*, cache, manifest, index, partition, output, on_progress=None):
    """No model, GPU, test split, refitting on holdout or per-sample public output."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    if manifest.get("dataset_revision") != DATASET_REVISION:
        raise ValueError("Cached validation requires the registered dataset revision")
    pins = {
        "private_reference": CACHE_SHA256,
        "validation_index": manifest["artifacts"]["validation_index"],
        "validation_partition": manifest["artifacts"]["validation_partition"],
    }
    for key, path in (
        ("private_reference", cache),
        ("validation_index", index),
        ("validation_partition", partition),
    ):
        if Path(path).is_symlink() or sha256_file(path) != pins[key]:
            raise ValueError("Cached validation checksum differs: " + key)
    index_bytes = Path(index).read_bytes()
    rows = list(csv.DictReader(io.StringIO(index_bytes.decode())))
    if len(rows) != 83 or any(row["split"] != "val" for row in rows):
        raise ValueError("Only the original83 validation samples are allowed")
    groups = np.array([row["patient_group_id"] for row in rows])
    calibration, holdout = registered_partition(Path(partition).read_bytes(), index_bytes, groups)
    with np.load(cache, allow_pickle=False) as archive:
        images, initial = archive["images"], archive["initials"]
        masks, labels = archive["masks"], archive["labels"]
        if (
            any(value.shape != (83, 256, 256) for value in (images, initial, masks))
            or labels.shape != (83,)
            or not np.isin(masks, [0, 1]).all()
            or not np.isin(labels, [0, 1]).all()
            or not np.array_equal(masks.any(axis=(1, 2)), labels.astype(bool))
            or not np.array_equal(labels, np.array([int(row["label"]) for row in rows]))
        ):
            raise ValueError("Cached validation shape, label or mask alignment differs")
        maps, timings = [], []
        for position in range(83):
            tick = time.perf_counter()
            maps.append(paired_score_maps(images[position], initial[position])["local_ssim"])
            timings.append(time.perf_counter() - tick)
            if on_progress and ((position + 1) % 10 == 0 or position == 82):
                on_progress({"score_samples": position + 1, "expected_samples": 83})
        maps = np.stack(maps)
        cropped_masks = masks[:, 8:-8, 8:-8]
        result = evaluate_partitions(maps, cropped_masks, labels, calibration, holdout, output)
        # Private medical-derived maps remain on the remote private Volume.
        np.savez_compressed(
            output / "private_local_ssim_maps.npz", scores=maps, masks=cropped_masks, labels=labels
        )
    result.update(
        status="completed",
        split="validation",
        samples_scored=83,
        calibration_samples=40,
        holdout_samples=43,
        calibration_groups=len(set(groups[calibration])),
        holdout_groups=len(set(groups[holdout])),
        dataset_revision=DATASET_REVISION,
        input_checksums=pins,
        parameters=PARAMETERS,
        method="Gaussian EMA30; initial DDIM20(t300), eta0",
        checkpoint_sha256=manifest["artifacts"]["checkpoint"],
        original_sampling_base_seed=17,
        training_seeds=1,
        sampling_seeds=1,
        new_unet_calls=0,
        new_lpips_calls=0,
        training_performed=False,
        test_accessed=False,
        scoring_seconds_sum=sum(timings),
        scoring_seconds_median=float(np.median(timings)),
        scoring_seconds_p95=float(np.quantile(timings, 0.95)),
        timing_scope="CPU paired_score_maps includes both L1 and SSIM; excludes cache/hash/I-O/global metrics",
        vram_bytes=None,
        limitations="Post-final-test exploratory validation extension; already inspected validation, one seed. No test rerun or frozen final-protocol change.",
    )
    write_json(output / "results.json", result)
    return result
