import json

import numpy as np
import pytest

from dacn.evaluation import ddim20_local_ssim


def test_thresholds_are_saved_before_holdout_and_ignore_holdout_scores(tmp_path, monkeypatch):
    maps = np.stack([np.full((240, 240), score) for score in (0.2, 0.8, 0.9, 0.1)])
    masks = np.stack([np.full((240, 240), label, dtype=np.uint8) for label in (0, 1, 0, 1)])
    labels = np.array([0, 1, 0, 1], dtype=np.uint8)
    original = ddim20_local_ssim.evaluate_maps
    output = tmp_path / "first"
    output.mkdir()

    def checked(*args, **kwargs):
        if kwargs.get("thresholds") is not None:
            frozen = json.loads((output / "thresholds.json").read_bytes())
            assert frozen["thresholds"] == kwargs["thresholds"]
        return original(*args, **kwargs)

    monkeypatch.setattr(ddim20_local_ssim, "evaluate_maps", checked)
    first = ddim20_local_ssim.evaluate_partitions(
        maps, masks, labels, np.array([0, 1]), np.array([2, 3]), output
    )
    assert first["calibration"]["pixel"]["f1"] == 1
    assert first["group_holdout"]["pixel"]["f1"] == 0
    second_output = tmp_path / "second"
    second_output.mkdir()
    output = second_output
    maps[2:] = 0.99
    second = ddim20_local_ssim.evaluate_partitions(
        maps, masks, labels, np.array([0, 1]), np.array([2, 3]), second_output
    )
    assert second["thresholds"] == first["thresholds"]
    assert second["group_holdout"]["pixel"]["f1"] != first["group_holdout"]["pixel"]["f1"]


def test_changed_cache_rejected_before_numpy_load(tmp_path, monkeypatch):
    cache = tmp_path / "cache.npz"
    cache.write_bytes(b"changed cache")
    monkeypatch.setattr(np, "load", lambda *_a, **_kw: pytest.fail("Read unverified cache"))
    with pytest.raises(ValueError, match="checksum differs: private_reference"):
        ddim20_local_ssim.evaluate_cached_validation(
            cache=cache,
            manifest={
                "dataset_revision": ddim20_local_ssim.DATASET_REVISION,
                "artifacts": {"validation_index": "a" * 64, "validation_partition": "b" * 64},
            },
            index=tmp_path / "missing-index",
            partition=tmp_path / "missing-partition",
            output=tmp_path / "out",
        )


def test_other_dataset_revision_cannot_open_cache(tmp_path):
    with pytest.raises(ValueError, match="registered dataset revision"):
        ddim20_local_ssim.evaluate_cached_validation(
            cache=tmp_path / "absent",
            manifest={"dataset_revision": "0" * 40},
            index=tmp_path / "absent-index",
            partition=tmp_path / "absent-partition",
            output=tmp_path / "out",
        )
