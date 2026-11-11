import json

import numpy as np
import pytest

from dacn.evaluation import ddim20_score_fusion as fusion


def test_normal_cdf_ties_and_holdout_changes_do_not_affect_fit():
    maps = np.array([[[0.0, 1.0, 1.0, 2.0]], [[-10.0, 1.0, 2.0, 100.0]]])
    raw = {name: maps.copy() for name in ("lpips", "residual", "ssim")}
    normalized, references = fusion.normalize_scores(raw, np.array([0]))
    np.testing.assert_array_equal(normalized["lpips"][0], [[0.125, 0.5, 0.5, 0.875]])
    np.testing.assert_array_equal(normalized["lpips"][1], [[0.0, 0.5, 0.875, 1.0]])
    raw["lpips"][1] = 1000
    again, new_references = fusion.normalize_scores(raw, np.array([0]))
    np.testing.assert_array_equal(new_references["lpips"], references["lpips"])
    np.testing.assert_array_equal(again["lpips"][0], normalized["lpips"][0])


def test_four_fusions_use_fixed_equal_weights():
    normalized = {"lpips": np.array([0.2]), "residual": np.array([0.4]), "ssim": np.array([0.6])}
    result = fusion.combine_normalized(normalized)
    assert set(result) == set(fusion.FUSIONS)
    expected = {
        "lpips_residual": 0.3,
        "lpips_ssim": 0.4,
        "residual_ssim": 0.5,
        "lpips_ssim_residual": 0.4,
    }
    for key, value in expected.items():
        assert result[key][0] == pytest.approx(value)


def test_all_thresholds_saved_before_holdout_and_ignore_holdout_scores(tmp_path, monkeypatch):
    labels = np.array([0, 1, 0, 1], dtype=np.uint8)
    masks = np.stack([np.full((240, 240), label, dtype=np.uint8) for label in labels])
    raw = {
        key: np.stack([np.full((240, 240), v) for v in (0.2, 0.8, 0.9, 0.1)])
        for key in ("lpips", "residual", "ssim")
    }
    original = fusion.evaluate_maps
    output = tmp_path / "first"

    def guarded(*args, **kwargs):
        if kwargs.get("thresholds") is not None:
            record = json.loads((output / "frozen_calibration.json").read_bytes())
            assert set(record["scores"]) == {
                "raw_lpips",
                "raw_residual",
                "raw_ssim",
                *fusion.FUSIONS,
            }
        return original(*args, **kwargs)

    monkeypatch.setattr(fusion, "evaluate_maps", guarded)
    first = fusion.evaluate_fusions(raw, masks, labels, np.array([0, 1]), np.array([2, 3]), output)
    assert first["definition"]["normalizer_fit_samples"] == 1
    for values in raw.values():
        values[2:] = 100
    output = tmp_path / "second"
    second = fusion.evaluate_fusions(raw, masks, labels, np.array([0, 1]), np.array([2, 3]), output)
    for key in first["calibration"]:
        assert first["calibration"][key]["thresholds"] == second["calibration"][key]["thresholds"]
    for key in fusion.FUSIONS:
        assert (
            first["holdout_results"][key]["pixel"]["f1"]
            != second["holdout_results"][key]["pixel"]["f1"]
        )


def test_altered_cache_rejected_before_deserialization(tmp_path, monkeypatch):
    cache = tmp_path / "cache.npz"
    cache.write_bytes(b"different")
    monkeypatch.setattr(np, "load", lambda *_a, **_kw: pytest.fail("Unverified cache read"))
    with pytest.raises(ValueError, match="checksum differs: cache"):
        fusion.evaluate_cached_fusions(
            cache=cache,
            manifest={
                "dataset_revision": fusion.DATASET_REVISION,
                "artifacts": {"validation_index": "a" * 64, "validation_partition": "b" * 64},
            },
            index=tmp_path / "absent-index",
            partition=tmp_path / "absent-partition",
            output=tmp_path / "output",
        )
