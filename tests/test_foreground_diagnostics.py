import numpy as np
import pytest

from dacn.evaluation import diagnostics


def test_foreground_keeps_largest_component_fills_hole_and_ignores_padding():
    image = np.full((256, 256), -1, dtype=np.float32)
    image[:8] = 1
    image[30:60, 30:60] = 0
    image[40:45, 40:45] = -1
    image[90:92, 90:92] = 0
    mask = diagnostics.foreground_v1(image)
    assert mask.shape == (240, 240)
    assert mask.sum() == 900
    assert mask[34, 34]
    assert not mask[83, 83]


def test_blank_and_invalid_images():
    assert not diagnostics.foreground_v1(np.full((256, 256), -1)).any()
    with pytest.raises(ValueError):
        diagnostics.foreground_v1(np.zeros((240, 240)))


def test_fp_attribution_and_all_positive_image_detection():
    scores = np.zeros((2, 240, 240))
    masks = np.zeros_like(scores, dtype=np.uint8)
    masks[1, 10, 10] = 1
    foreground = np.zeros_like(scores, dtype=bool)
    foreground[:, 10:20, 10:20] = True
    scores[:, 0, 0] = 2
    scores[1, 10, 10] = 2
    result = diagnostics.summarize(
        scores,
        masks,
        np.array([0, 1]),
        foreground,
        {"pixel": {"threshold": 1}, "image": {"threshold": 0}},
    )
    assert result["pixel_confusion"]["tp"] == 1
    assert result["fp_outside_foreground"] == 2
    assert result["fp_inside_foreground"] == 0
    assert result["image_confusion"] == {"tp": 1, "fp": 1, "fn": 0, "tn": 0}
    assert result["image_all_positive"]


def test_top576_tie_location_is_flagged_ambiguous():
    scores = np.zeros((2, 240, 240))
    masks = np.zeros_like(scores, dtype=np.uint8)
    result = diagnostics.summarize(
        scores,
        masks,
        np.array([0, 1]),
        np.ones_like(scores, dtype=bool),
        {"pixel": {"threshold": 1}, "image": {"threshold": 1}},
    )
    assert result["top576_boundary_tie_slices"] == 2
