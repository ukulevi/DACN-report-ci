"""CPU-only validation diagnostics and one predeclared foreground candidate."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy import ndimage

from dacn.evaluation.artifacts import _validation_data, load_validation_pairs
from dacn.evaluation.scores import paired_score_maps
from dacn.evaluation.validation import evaluate_maps

POLICY = "FG-v1: crop8; input>-1; largest8-connected component; fill holes; no erosion"


def foreground_v1(image):
    image = np.asarray(image)
    if image.shape != (256, 256) or not np.isfinite(image).all() or np.any(np.abs(image) > 1):
        raise ValueError("expected finite normalized 256x256 input")
    components, count = ndimage.label(image[8:-8, 8:-8] > -1, structure=np.ones((3, 3)))
    if not count:
        return np.zeros((240, 240), dtype=bool)
    sizes = np.bincount(components.ravel())
    sizes[0] = 0
    return ndimage.binary_fill_holes(components == sizes.argmax())


def coverage(images, masks):
    fg = np.stack([foreground_v1(image) for image in images])
    lesions = masks[:, 8:-8, 8:-8].astype(bool)
    removed = lesions & ~fg
    total = int(lesions.sum())
    return fg, {
        "policy": POLICY,
        "samples": len(images),
        "lesion_pixels": total,
        "excluded_lesion_pixels": int(removed.sum()),
        "excluded_lesion_fraction": float(removed.sum() / total) if total else None,
        "affected_lesion_slices": int(removed.any(axis=(1, 2)).sum()),
        "fully_excluded_lesion_slices": int(
            (lesions.any(axis=(1, 2)) & ~(lesions & fg).any(axis=(1, 2))).sum()
        ),
        "empty_foreground_slices": int((~fg.any(axis=(1, 2))).sum()),
        "foreground_below_top576_slices": int((fg.sum(axis=(1, 2)) < 576).sum()),
        "foreground_fraction": float(fg.mean()),
    }


def confusion(predicted, truth):
    truth = np.asarray(truth, dtype=bool)
    return {
        "tp": int((predicted & truth).sum()),
        "fp": int((predicted & ~truth).sum()),
        "fn": int((~predicted & truth).sum()),
        "tn": int((~predicted & ~truth).sum()),
    }


def summarize(scores, masks, labels, foreground, metrics):
    predicted = scores >= metrics["pixel"]["threshold"]
    truth = masks.astype(bool)
    flat = scores.reshape(len(scores), -1)
    top = np.argpartition(flat, -576, axis=1)[:, -576:]
    top_values = np.take_along_axis(flat, top, axis=1)
    cutoff = top_values.min(axis=1)
    ties_total = (flat == cutoff[:, None]).sum(axis=1)
    ties_selected = (top_values == cutoff[:, None]).sum(axis=1)
    ambiguous = ties_total > ties_selected
    image_scores = top_values.mean(axis=1)
    image_predicted = image_scores >= metrics["image"]["threshold"]
    normal = labels == 0
    fp = predicted & ~truth
    strata = {}
    lesion_sizes = truth.sum(axis=(1, 2))
    for name, lo, hi in [
        ("1_to_100", 1, 100),
        ("101_to_1000", 101, 1000),
        ("over_1000", 1001, 57600),
    ]:
        selected = (lesion_sizes >= lo) & (lesion_sizes <= hi)
        strata[name] = {
            "slices": int(selected.sum()),
            **confusion(predicted[selected], truth[selected]),
        }
    return {
        "top576_boundary_tie_slices": int(ambiguous.sum()),
        "top576_location_caveat": "location ambiguous on slices with a cutoff tie",
        "pixel_confusion": confusion(predicted, truth),
        "image_confusion": confusion(image_predicted, labels),
        "image_all_positive": bool(image_predicted.all()),
        "normal_image_false_positive_rate": float(image_predicted[normal].mean())
        if normal.any()
        else None,
        "fp_outside_foreground": int((fp & ~foreground).sum()),
        "fp_inside_foreground": int((fp & foreground).sum()),
        "top576_outside_foreground_fraction": float(
            (~np.take_along_axis(foreground.reshape(len(scores), -1), top, axis=1)).mean()
        ),
        "image_score_quantiles_q0_q25_q50_q75_q100": {
            name: np.quantile(image_scores[labels == label], [0, 0.25, 0.5, 0.75, 1]).tolist()
            for name, label in [("normal", 0), ("anomaly", 1)]
            if (labels == label).any()
        },
        "lesion_size_strata": strata,
    }


def compare(artifact, dataset, provenance):
    images, recons, masks, labels, provenance, digest = load_validation_pairs(
        artifact, dataset, provenance
    )
    foreground, audit = coverage(images, masks)
    cropped = masks[:, 8:-8, 8:-8]
    maps = {name: [] for name in ("pixel_difference", "local_ssim")}
    for image, reconstruction in zip(images, recons):
        for name, scores in paired_score_maps(image, reconstruction).items():
            maps[name].append(scores)
    results = {}
    for name, values in maps.items():
        scores = np.stack(values).astype(np.float64)
        floor = float(scores.min() - max(1.0, abs(scores.min()) * 0.1))
        candidate = np.where(foreground, scores, floor)
        results[name] = {}
        for variant, values in [("baseline", scores), ("foreground_floor", candidate)]:
            metrics = evaluate_maps(values, cropped, labels, split="validation")
            results[name][variant] = {
                "metrics": metrics,
                "diagnostics": summarize(values, cropped, labels, foreground, metrics),
            }
        results[name]["floor"] = floor
        results[name]["candidate_threshold_at_floor"] = bool(
            results[name]["foreground_floor"]["metrics"]["pixel"]["threshold"] <= floor
        )
    return {
        "split": "validation",
        "scope": "exploratory CPU comparison; test not used",
        "samples": len(images),
        "thresholds_frozen": False,
        "provenance_declared_by_producer": provenance,
        "reconstruction_artifact_sha256": digest,
        "coverage": audit,
        "results": results,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--artifact", type=Path)
    parser.add_argument("--provenance", type=Path)
    parser.add_argument("--metadata-sha256")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("choose a fresh report path")
    if args.artifact:
        if not args.provenance:
            parser.error("artifact requires provenance")
        result = compare(
            args.artifact, args.dataset, json.loads(args.provenance.read_text(encoding="utf-8"))
        )
    else:
        if not args.metadata_sha256:
            parser.error("coverage-only audit requires metadata-sha256")
        _, images, masks, _ = _validation_data(args.dataset, args.metadata_sha256)
        _, result = coverage(images, masks)
        result.update(
            split="validation",
            scope="input-derived coverage audit; no reconstruction metrics",
            dataset_metadata_sha256=args.metadata_sha256,
        )
    result["diagnostic_source_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")


if __name__ == "__main__":
    main()
