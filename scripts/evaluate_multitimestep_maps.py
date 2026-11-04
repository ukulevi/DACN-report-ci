"""Fixed equal-weight rank-mean fusion of cached epoch30 reconstruction maps."""

import csv
import hashlib
import json
import time
from pathlib import Path

import numpy as np
from sklearn.model_selection import StratifiedGroupKFold

from dacn.evaluation.scores import paired_score_maps
from dacn.evaluation.validation import evaluate_maps


def main():
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("Use a fresh output directory")
    args.output.mkdir(parents=True)
    start = time.perf_counter()
    files = {
        100: (
            "reconstructions_t100.npz",
            "c1aa679db880a13a860109dee028db80d979eca684fad161c4726d5c4000db6d",
        ),
        200: (
            "reconstructions_t200.npz",
            "cffefdc5d2cf599fc3903eb8a8fa9a192dddd2dabe4cffeb04ee28cfa60bc06a",
        ),
        300: (
            "reconstructions.npz",
            "8278ad18543dd87f698c36b510564b9fcd4445d4a1b0ae5d7a859093e2a4e959",
        ),
    }
    maps = {}
    reference = None
    for timestep, (name, digest) in files.items():
        path = args.cache_dir / name
        assert hashlib.sha256(path.read_bytes()).hexdigest() == digest
        with np.load(path, allow_pickle=False) as cache:
            images, recon, masks, labels = [
                cache[k] for k in ("images", "reconstructions", "masks", "labels")
            ]
        if reference is None:
            reference = (images, masks, labels)
        else:
            assert all(
                np.array_equal(a, b)
                for a, b in zip(reference, (images, masks, labels), strict=True)
            )
        paired = [paired_score_maps(a, b) for a, b in zip(images, recon, strict=True)]
        for method in ("pixel_difference", "local_ssim"):
            maps[f"{method}_t{timestep}"] = np.stack([p[method] for p in paired])
    with (args.cache_dir / "validation_index.csv").open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == len(labels) == 83
    assert all(row["split"] == "val" for row in rows)
    assert np.array_equal(labels, [int(row["label"]) for row in rows])
    groups = np.asarray([row["patient_group_id"] for row in rows])
    calibration, holdout = next(
        StratifiedGroupKFold(2, shuffle=True, random_state=17).split(
            np.zeros(len(labels)), labels, groups
        )
    )
    assert set(groups[calibration]).isdisjoint(groups[holdout])
    for method in ("pixel_difference", "local_ssim"):
        combined = np.zeros_like(maps[f"{method}_t100"])
        for t in (100, 200, 300):
            values = maps[f"{method}_t{t}"]
            fitted = np.sort(values[calibration].ravel())
            ranks = (
                np.searchsorted(fitted, values, side="left")
                + np.searchsorted(fitted, values, side="right")
            ) / (2 * len(fitted))
            combined += ranks / 3
        maps[f"{method}_rank_mean_100_200_300"] = combined
    maps["brightness_control"] = images[:, 8:-8, 8:-8].astype(np.float64)
    truth = masks[:, 8:-8, 8:-8]
    result = {
        "split": "validation_group_holdout",
        "held_out_test_used": False,
        "exploratory": True,
        "seed": 17,
        "timesteps": [100, 200, 300],
        "cache_hashes": {str(t): digest for t, (_, digest) in files.items()},
        "normalization": "calibration-only empirical midrank CDF; equal1/3 mean",
        "partition": {
            name: {"samples": len(idx), "groups": len(np.unique(groups[idx]))}
            for name, idx in (("calibration", calibration), ("holdout", holdout))
        },
        "metrics": {},
    }
    # Fixed two ensembles; no parameter selection or weight search on holdout.
    for name, scores in maps.items():
        cal = evaluate_maps(
            scores[calibration], truth[calibration], labels[calibration], split="validation"
        )
        thresholds = {k: cal[k]["threshold"] for k in ("pixel", "image")}
        measured = evaluate_maps(
            scores[holdout],
            truth[holdout],
            labels[holdout],
            split="validation",
            thresholds=thresholds,
        )
        result["metrics"][name] = {"calibration": cal, "group_holdout": measured}
        print(json.dumps({"method": name, "holdout_pixel": measured["pixel"]}), flush=True)
    result["cpu_seconds"] = time.perf_counter() - start
    (args.output / "results.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
