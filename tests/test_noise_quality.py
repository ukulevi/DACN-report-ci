import numpy as np
import pytest

from dacn.evaluation.noise_quality import producer_detection, reconstruction_quality


def _arrays():
    """Two slices: 0 normal, 1 abnormal with a 10x10 lesion; known constant errors."""
    n = 2
    images = np.full((n, 256, 256), -1.0, dtype=np.float32)
    images[:, 8:248, 8:248] = 0.0  # whole crop is foreground
    images[:, 8:18, 8:248] = -1.0  # ten background rows inside the crop
    masks = np.zeros((n, 256, 256), dtype=np.uint8)
    masks[1, 100:110, 100:110] = 1
    initials = images.copy()
    initials[0, 8:248, 8:248] += 0.1  # normal slice error 0.1 everywhere in crop
    initials[1, 8:248, 8:248] += 0.2
    initials[1, 100:110, 100:110] = images[1, 100:110, 100:110] + 0.8  # lesion "healed"
    finals = images.copy()
    finals[1, 100:110, 100:110] += 0.6
    lpips_initial = np.full((n, 240, 240), 0.1, dtype=np.float32)
    lpips_initial[1, 92:102, 92:102] = 0.5  # lesion region in crop coordinates
    lpips_final = lpips_initial * 2
    predicted = np.zeros((n, 240, 240), dtype=bool)
    predicted[1, 92:97, 92:102] = True  # half the lesion
    predicted[0, 200:210, 200:210] = True  # 100 false-positive normal pixels
    return {
        "images": images,
        "initials": initials,
        "finals": finals,
        "masks": masks,
        "labels": np.array([0, 1], dtype=np.uint8),
        "predicted_masks": predicted,
        "initial_lpips_only": lpips_initial,
        "autoddpm_lpips": lpips_final,
    }


def test_reconstruction_quality_matches_hand_computed_aggregates():
    result = reconstruction_quality(_arrays(), [0, 1])
    initial, final = result["phases"]["initial"], result["phases"]["final"]
    assert result["samples"] == 2 and result["lesion_pixels"] == 100
    assert initial["normal_slice_foreground_abs_error"] == pytest.approx(0.1, rel=1e-6)
    assert initial["abnormal_slice_nonlesion_foreground_abs_error"] == pytest.approx(0.2, rel=1e-6)
    assert initial["lesion_abs_error"] == pytest.approx(0.8, rel=1e-6)
    assert initial["lesion_to_normal_abs_error_ratio"] == pytest.approx(8.0, rel=1e-6)
    assert initial["lesion_lpips_mean"] == pytest.approx(0.5, rel=1e-6)
    assert initial["lesion_to_normal_lpips_ratio"] == pytest.approx(5.0, rel=1e-6)
    assert final["normal_slice_foreground_abs_error"] == 0
    assert final["lesion_abs_error"] == pytest.approx(0.6, rel=1e-6)
    assert final["lesion_lpips_mean"] == pytest.approx(1.0, rel=1e-6)
    mask = result["pseudo_mask"]
    foreground_per_slice = 230 * 240
    assert mask["lesion_pixel_recall"] == pytest.approx(0.5)
    assert mask["pixel_precision"] == pytest.approx(50 / 150)
    assert mask["normal_slice_foreground_fraction_masked"] == pytest.approx(
        100 / foreground_per_slice
    )
    assert mask["abnormal_slice_nonlesion_foreground_fraction_masked"] == 0


def test_reconstruction_quality_subset_and_missing_class_are_explicit():
    result = reconstruction_quality(_arrays(), [0])
    assert result["normal_slices"] == 1 and result["abnormal_slices"] == 0
    assert result["phases"]["initial"]["lesion_abs_error"] is None
    assert result["phases"]["initial"]["lesion_to_normal_abs_error_ratio"] is None
    assert result["pseudo_mask"]["lesion_pixel_recall"] is None


def test_reconstruction_quality_rejects_misaligned_arrays():
    arrays = _arrays()
    arrays["predicted_masks"] = arrays["predicted_masks"][:, :200]
    with pytest.raises(ValueError, match="240x240"):
        reconstruction_quality(arrays, [0, 1])


def test_producer_detection_copies_holdout_metrics():
    block = {
        "group_holdout": {
            "pixel": {"auprc": 0.3, "auroc": 0.9, "f1": 0.4},
            "image": {"auroc": 0.8},
            "empty_mask_pixel_false_positive_rate": 0.01,
        }
    }
    out = producer_detection({"metrics": {"a": block}}, arms=("a",))
    assert out == {
        "a": {
            "pixel_ap": 0.3,
            "pixel_auroc": 0.9,
            "pixel_dice_micro": 0.4,
            "image_auroc": 0.8,
            "normal_pixel_fpr": 0.01,
        }
    }
