"""CPU-only Gaussian/Simplex SSIM-map fusion on a fixed validation group split."""

import argparse
import csv
import hashlib
import json
import platform
import time
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score
from sklearn.model_selection import StratifiedGroupKFold

from dacn.evaluation.scores import paired_score_maps
from dacn.evaluation.validation import evaluate_maps, select_validation_threshold

GAUSSIAN_SHA = "8278ad18543dd87f698c36b510564b9fcd4445d4a1b0ae5d7a859093e2a4e959"
SIMPLEX_SHA = "5bf4cfddf144ac2ce88e3da6417df5e86cea91bcff4d39f97738ae70ab2d0218"
SIMPLEX20_SHA = "4dd1b3c5eac41f43638a2ccea5d97bdb53074f30ce3d21c8e84ef593fe8a28c0"
INDEX_SHA = "57ad2e31df3ed4e625da00efaecd6fddc1e2a132329dfba6e60867caf1802db3"


def normal_cdf(values, normal_indices):
    """Fit midrank empirical CDF on normal calibration pixels, never per image."""
    if not len(normal_indices) or not np.isfinite(values).all():
        raise ValueError("Finite maps and normal calibration examples required")
    reference = np.sort(np.asarray(values)[normal_indices].ravel())
    return (
        np.searchsorted(reference, values, side="left")
        + np.searchsorted(reference, values, side="right")
    ) / (2 * len(reference))


def evaluate_noise_fusion(
    *, gaussian_cache, simplex_cache, index_path, output_dir, simplex_epoch=10
):
    if output_dir.exists():
        raise FileExistsError("Use a fresh output directory")
    paths = {"gaussian": gaussian_cache, "simplex": simplex_cache, "index": index_path}
    hashes = {key: hashlib.sha256(path.read_bytes()).hexdigest() for key, path in paths.items()}
    simplex_hash = {10: SIMPLEX_SHA, 20: SIMPLEX20_SHA}[simplex_epoch]
    if hashes != {"gaussian": GAUSSIAN_SHA, "simplex": simplex_hash, "index": INDEX_SHA}:
        raise ValueError("Registered cache/index checksum mismatch")
    started = time.perf_counter()
    with np.load(gaussian_cache, allow_pickle=False) as cache:
        images, gaussian, masks, labels = [
            cache[key] for key in ("images", "reconstructions", "masks", "labels")
        ]
    with np.load(simplex_cache, allow_pickle=False) as cache:
        for key, expected in (("images", images), ("masks", masks), ("labels", labels)):
            if not np.array_equal(cache[key], expected):
                raise ValueError("Gaussian/Simplex cache images, masks or label order differ")
        simplex = cache["reconstructions"].copy()
    with index_path.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    if len(rows) != 83 or len(images) != 83 or any(row["split"] != "val" for row in rows):
        raise ValueError("Expected all83 validation examples")
    if not np.array_equal(labels, [int(row["label"]) for row in rows]):
        raise ValueError("Index label order differs from caches")
    groups = np.asarray([row["patient_group_id"] for row in rows])
    calibration, holdout = next(
        StratifiedGroupKFold(2, shuffle=True, random_state=17).split(
            np.zeros(len(labels)), labels, groups
        )
    )
    if not set(groups[calibration]).isdisjoint(groups[holdout]):
        raise ValueError("Patient-group leakage")
    if any(len(np.unique(labels[idx])) != 2 for idx in (calibration, holdout)):
        raise ValueError("Each partition needs both classes")
    normal = calibration[labels[calibration] == 0]
    truth = masks[:, 8:-8, 8:-8]
    maps = {}
    for name, reconstructions in (("gaussian_raw", gaussian), ("simplex_raw", simplex)):
        maps[name] = np.stack(
            [
                paired_score_maps(x, y)["local_ssim"]
                for x, y in zip(images, reconstructions, strict=True)
            ]
        )
        print(f"Computed {name} SSIM maps", flush=True)
    maps["gaussian_cdf"] = normal_cdf(maps["gaussian_raw"], normal)
    maps["simplex_cdf"] = normal_cdf(maps["simplex_raw"], normal)
    candidates = list(maps) + [f"fusion_gaussian_{weight:.2f}" for weight in (0.25, 0.5, 0.75)]

    def candidate_map(name):
        if name in maps:
            return maps[name]
        if name == "brightness_control":
            return images[:, 8:-8, 8:-8].astype(np.float64)
        weight = float(name.rsplit("_", 1)[1])
        return weight * maps["gaussian_cdf"] + (1 - weight) * maps["simplex_cdf"]

    selection = {}
    for name in [*candidates, "brightness_control"]:
        scores = candidate_map(name)[calibration]
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
        print(f"Calibrated {name}", flush=True)
    selected = max(candidates, key=lambda name: selection[name]["calibration_pixel_ap"])
    selected_fusion = max(candidates[4:], key=lambda name: selection[name]["calibration_pixel_ap"])
    plan = {
        "method": f"gaussian30_simplex{simplex_epoch}_normal_calibration_cdf_ssim_fusion",
        "split": "validation_exploratory_patient_group_holdout",
        "held_out_test_used": False,
        "seed": 17,
        "starting_timestep": 300,
        "hashes": hashes,
        "normalization": "midrank CDF fit on all pixels of normal calibration images only",
        "normalizer_fit_samples": len(normal),
        "gaussian_weights": [0.25, 0.5, 0.75],
        "selection": selection,
        "selected_by_calibration_ap": selected,
        "best_fusion_by_calibration_ap": selected_fusion,
        "selection_tie_rule": "first candidate; raw Gaussian comes first",
        "brightness_control_used_for_selection": False,
        "partition": {
            name: {
                "samples": len(idx),
                "groups": len(np.unique(groups[idx])),
                "normal": int((labels[idx] == 0).sum()),
                "abnormal": int((labels[idx] == 1).sum()),
            }
            for name, idx in (("calibration", calibration), ("holdout", holdout))
        },
        "caveat": (
            f"Previously examined validation. Gaussian epoch30 and Simplex epoch{simplex_epoch} "
            "differ in training budget. The analysis is exploratory."
        ),
    }
    output_dir.mkdir(parents=True)
    # Persist model/normalizer/threshold choice BEFORE reading any holdout outcomes.
    (output_dir / "frozen_selection.json").write_text(
        json.dumps(plan, indent=2, allow_nan=False) + "\n"
    )
    print(f"Frozen selection: {selected}; best fusion: {selected_fusion}", flush=True)
    results = {}
    for name in [*candidates, "brightness_control"]:
        results[name] = evaluate_maps(
            candidate_map(name)[holdout],
            truth[holdout],
            labels[holdout],
            split="validation",
            thresholds=selection[name]["thresholds"],
        )
        print(json.dumps({"candidate": name, "holdout_pixel": results[name]["pixel"]}), flush=True)
    result = {
        **plan,
        "holdout_results": results,
        "cpu_seconds": time.perf_counter() - started,
        "execution": "local CPU, no new inference or training",
        "python_version": platform.python_version(),
    }
    (output_dir / "results.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(
        json.dumps(
            {
                "selected": selected,
                "selected_fusion": selected_fusion,
                "cpu_seconds": result["cpu_seconds"],
            }
        ),
        flush=True,
    )
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gaussian-cache", type=Path, required=True)
    parser.add_argument("--simplex-cache", type=Path, required=True)
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--simplex-epoch", type=int, choices=(10, 20), default=10)
    args = parser.parse_args()
    evaluate_noise_fusion(
        gaussian_cache=args.gaussian_cache,
        simplex_cache=args.simplex_cache,
        index_path=args.index,
        output_dir=args.output,
        simplex_epoch=args.simplex_epoch,
    )


if __name__ == "__main__":
    main()
