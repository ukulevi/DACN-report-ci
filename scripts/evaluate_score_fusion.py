"""Exploratory score fusion on one fixed patient-group split of validation."""

import argparse
import csv
import hashlib
import json
import time
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score
from sklearn.model_selection import StratifiedGroupKFold

from dacn.evaluation.scores import paired_score_maps
from dacn.evaluation.validation import evaluate_maps, select_validation_threshold


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("Use a fresh output directory")
    args.output.mkdir(parents=True)
    started = time.perf_counter()
    digest = hashlib.sha256(args.cache.read_bytes()).hexdigest()
    if digest != "8278ad18543dd87f698c36b510564b9fcd4445d4a1b0ae5d7a859093e2a4e959":
        raise ValueError("Expected the registered full validation epoch30 t300 cache")
    with np.load(args.cache, allow_pickle=False) as data:
        images, recon, masks, labels = [
            data[key] for key in ("images", "reconstructions", "masks", "labels")
        ]
    with args.index.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == len(images) == 83
    assert all(row["split"] == "val" for row in rows)
    assert np.array_equal(labels, [int(row["label"]) for row in rows])
    assert len({row["sample_id"] for row in rows}) == len(rows)
    groups = np.asarray([row["patient_group_id"] for row in rows])
    splitter = StratifiedGroupKFold(n_splits=2, shuffle=True, random_state=17)
    calibration, holdout = next(splitter.split(np.zeros(len(labels)), labels, groups))
    assert set(groups[calibration]).isdisjoint(groups[holdout])
    assert all(len(np.unique(labels[idx])) == 2 for idx in (calibration, holdout))
    truth = masks[:, 8:-8, 8:-8]
    paired = [paired_score_maps(x, y) for x, y in zip(images, recon, strict=True)]
    maps = {
        "intensity": images[:, 8:-8, 8:-8].astype(np.float64),
        **{
            name: np.stack([p[name] for p in paired]) for name in ("pixel_difference", "local_ssim")
        },
    }
    # Fit a monotone empirical midrank CDF using calibration pixels only.
    # Equal-valued pixels share the same rank; no holdout fitting or labels.
    normalized = {}
    for name, values in maps.items():
        reference = np.sort(values[calibration].ravel())
        normalized[name] = (
            np.searchsorted(reference, values, side="left")
            + np.searchsorted(reference, values, side="right")
        ) / (2 * len(reference))
    candidates = {name: (name, None) for name in maps}
    for family in ("pixel_difference", "local_ssim"):
        for weight in (0.25, 0.5, 0.75):
            candidates[f"intensity_{weight:.2f}+{family}"] = (family, weight)

    def candidate_map(spec):
        family, weight = spec
        if weight is None:
            return maps[family]
        return weight * normalized["intensity"] + (1 - weight) * normalized[family]

    selection = {}
    for name, spec in candidates.items():
        scores = candidate_map(spec)[calibration]
        pixels = scores.reshape(len(scores), -1)
        image_scores = np.partition(pixels, -576, axis=1)[:, -576:].mean(axis=1)
        selection[name] = {
            "calibration_pixel_ap": float(
                average_precision_score(truth[calibration].ravel(), scores.ravel())
            ),
            "thresholds": {
                "pixel": select_validation_threshold(truth[calibration].ravel(), scores.ravel()),
                "image": select_validation_threshold(labels[calibration], image_scores),
            },
        }
        print(f"calibrated {name}", flush=True)
    # Intensity comes first, giving a conservative exact-tie preference.
    selected = max(selection, key=lambda name: selection[name]["calibration_pixel_ap"])
    plan = {
        "split": "validation_patient_group_holdout",
        "held_out_test_used": False,
        "exploratory": True,
        "caveat": "Full validation was examined in earlier experiments; not a pristine test set",
        "seed": 17,
        "cache_sha256": digest,
        "index_sha256": hashlib.sha256(args.index.read_bytes()).hexdigest(),
        "split_rule": "first fold StratifiedGroupKFold(n_splits=2,shuffle=True,random_state=17)",
        "partition": {
            tag: {
                "samples": len(idx),
                "groups": len(np.unique(groups[idx])),
                "normal": int((labels[idx] == 0).sum()),
                "anomalous": int((labels[idx] == 1).sum()),
            }
            for tag, idx in (("calibration", calibration), ("holdout", holdout))
        },
        "normalization": "empirical midrank CDF fit only on all calibration pixels",
        "selection": selection,
        "selected_by_calibration_ap": selected,
    }
    (args.output / "frozen_selection.json").write_text(json.dumps(plan, indent=2) + "\n")
    results = {}
    for name, spec in candidates.items():
        result = evaluate_maps(
            candidate_map(spec)[holdout],
            truth[holdout],
            labels[holdout],
            split="validation",
            thresholds=selection[name]["thresholds"],
        )
        results[name] = result
        print(json.dumps({"candidate": name, "holdout_pixel": result["pixel"]}), flush=True)
    output = {**plan, "holdout_results": results, "cpu_seconds": time.perf_counter() - started}
    (args.output / "results.json").write_text(json.dumps(output, indent=2, allow_nan=False) + "\n")
    print(
        json.dumps(
            {
                "selected": selected,
                "partition": plan["partition"],
                "cpu_seconds": output["cpu_seconds"],
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
