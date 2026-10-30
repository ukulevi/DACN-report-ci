import numpy as np
import pytest

from dacn.evaluation.scores import paired_score_maps
from dacn.evaluation.validation import (
    evaluate_maps,
    evaluate_reconstructions,
    select_validation_threshold,
)


def test_score_maps_use_crop_and_shared_pair():
    image = np.zeros((256, 256))
    reconstruction = image.copy()
    reconstruction[:8] = 1  # padding is excluded before SSIM convolution
    maps = paired_score_maps(image, reconstruction)
    assert maps["pixel_difference"].shape == (240, 240)
    assert np.max(maps["pixel_difference"]) == 0
    assert np.allclose(maps["local_ssim"], 0, atol=1e-12)
    reconstruction[100:110, 100:110] = 1
    maps = paired_score_maps(image, reconstruction)
    assert maps["pixel_difference"][92, 92] == 1
    assert maps["local_ssim"][92, 92] > 0


def test_ssim_constant_pair_matches_formula():
    x = np.full((256, 256), 0.5)
    y = np.full((256, 256), -0.25)
    expected = 1 - (2 * 0.5 * -0.25 + 0.02**2) / (0.5**2 + 0.25**2 + 0.02**2)
    assert np.allclose(paired_score_maps(x, y)["local_ssim"], expected)


def test_validation_tie_chooses_larger_threshold():
    assert select_validation_threshold([1, 0, 0, 1], [0.3, 0.2, 0.1, 0.0]) == 0.3


def test_validation_thresholds_are_required_for_test_and_metrics_are_aggregate():
    masks = np.zeros((2, 240, 240), dtype=np.uint8)
    masks[1, :24, :24] = 1
    scores = masks.astype(float)
    with pytest.raises(ValueError, match="frozen"):
        evaluate_maps(scores, masks, [0, 1], split="test")
    val = evaluate_maps(scores, masks, [0, 1], split="validation")
    assert val["pixel"]["average_precision"] == 1
    assert val["pixel"]["dice_micro"] == 1
    assert val["image"]["f1"] == 1
    assert val["pixel_dice_macro_lesion_slices"] == 1
    assert val["empty_mask_pixel_false_positive_rate"] == 0
    thresholds = {level: val[level]["threshold"] for level in ("pixel", "image")}
    test = evaluate_maps(scores, masks, [0, 1], split="test", thresholds=thresholds)
    assert test["image"]["f1"] == 1


def test_rejects_bad_labels_and_values():
    with pytest.raises(ValueError, match="range|\\[-1, 1\\]"):
        paired_score_maps(np.full((256, 256), 2), np.zeros((256, 256)))
    with pytest.raises(ValueError, match="agree"):
        evaluate_maps(np.zeros((1, 240, 240)), np.zeros((1, 240, 240)), [1], split="validation")


def test_paired_pipeline_uses_one_reconstruction_for_both_scores():
    images = np.zeros((2, 256, 256))
    reconstructions = images.copy()
    reconstructions[1, 100:124, 100:124] = 1
    masks = np.zeros_like(images, dtype=np.uint8)
    masks[1, 100:124, 100:124] = 1
    result = evaluate_reconstructions(images, reconstructions, masks, [0, 1], split="validation")
    assert set(result) == {"pixel_difference", "local_ssim"}
    assert result["pixel_difference"]["pixel"]["average_precision"] == 1
    assert result["local_ssim"]["image"]["auroc"] == 1
    with pytest.raises(ValueError, match="frozen"):
        evaluate_reconstructions(images, reconstructions, masks, [0, 1], split="test")
