import csv
import hashlib
import json

import numpy as np

from dacn.evaluation import noise_fusion


def test_normal_cdf_uses_only_reference_normals_and_preserves_ties():
    maps = np.array([[[0.0, 1.0]], [[0.0, 1.0]], [[0.5, 100.0]]])
    result = noise_fusion.normal_cdf(maps, np.array([0, 1]))
    assert np.array_equal(result[0], [[0.25, 0.75]])
    assert np.array_equal(result[2], [[0.5, 1.0]])
    maps[2] = [-100.0, -50.0]
    assert np.array_equal(noise_fusion.normal_cdf(maps, np.array([0, 1]))[:2], result[:2])


def test_fusion_freezes_calibration_selection_before_holdout(tmp_path, monkeypatch):
    images = np.zeros((83, 256, 256), dtype=np.float32)
    masks = np.zeros_like(images, dtype=np.uint8)
    labels = np.array([i % 2 for i in range(83)], dtype=np.uint8)
    masks[labels == 1, 100:120, 100:120] = 1
    paths = {name: tmp_path / f"{name}.npz" for name in ("gaussian", "simplex")}
    for name, path in paths.items():
        np.savez_compressed(
            path,
            images=images,
            reconstructions=images + (0.1 if name == "gaussian" else 0.2),
            masks=masks,
            labels=labels,
        )
        monkeypatch.setattr(
            noise_fusion, f"{name.upper()}_SHA", hashlib.sha256(path.read_bytes()).hexdigest()
        )
    index = tmp_path / "index.csv"
    with index.open("w", newline="") as stream:
        writer = csv.DictWriter(
            stream, fieldnames=["sample_id", "patient_group_id", "label", "split"]
        )
        writer.writeheader()
        writer.writerows(
            {
                "sample_id": str(i),
                "patient_group_id": str(i // 2),
                "label": int(labels[i]),
                "split": "val",
            }
            for i in range(83)
        )
    monkeypatch.setattr(noise_fusion, "INDEX_SHA", hashlib.sha256(index.read_bytes()).hexdigest())
    monkeypatch.setattr(
        noise_fusion,
        "paired_score_maps",
        lambda x, y: {"local_ssim": np.full((240, 240), float(y[0, 0]))},
    )
    fit_sizes = []

    def ap(y, scores):
        fit_sizes.append(len(y))
        assert len(y) < 83 * 240 * 240
        return 0.5

    monkeypatch.setattr(noise_fusion, "average_precision_score", ap)
    monkeypatch.setattr(noise_fusion, "select_validation_threshold", lambda *a: 0.123)
    output = tmp_path / "output"
    evaluated = []

    def evaluate(scores, y, labels, *, split, thresholds):
        frozen = json.loads((output / "frozen_selection.json").read_text())
        assert frozen["selected_by_calibration_ap"] == "gaussian_raw"
        assert thresholds == {"pixel": 0.123, "image": 0.123}
        evaluated.append(len(labels))
        return {"pixel": {"average_precision": 0.4}}

    monkeypatch.setattr(noise_fusion, "evaluate_maps", evaluate)
    result = noise_fusion.evaluate_noise_fusion(
        gaussian_cache=paths["gaussian"],
        simplex_cache=paths["simplex"],
        index_path=index,
        output_dir=output,
    )
    assert len(fit_sizes) == len(evaluated) == 8
    assert result["selected_by_calibration_ap"] == "gaussian_raw"
    assert not result["held_out_test_used"]
    assert result["brightness_control_used_for_selection"] is False
    assert (output / "results.json").is_file()
