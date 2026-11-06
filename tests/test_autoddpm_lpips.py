import numpy as np
import pytest
import torch

from dacn.evaluation.autoddpm_lpips import fit_mask_threshold, lpips_heatmap, pseudo_mask


def test_lpips_input_range_channels_crop_and_residual_weighting():
    class Perceptual(torch.nn.Module):
        def forward(self, a, b, normalize):
            assert not normalize
            assert a.shape == b.shape == (1, 3, 240, 240)
            assert a.min() == a.max() == -1
            assert b.min() == b.max() == 1
            return torch.full((1, 1, 240, 240), 0.25)

    image = -np.ones((256, 256), dtype=np.float32)
    value = lpips_heatmap(Perceptual(), image, -image, "cpu")
    assert np.all(value == 0.25)
    assert not lpips_heatmap(PerceptualZero(), image, image, "cpu").any()


class PerceptualZero(torch.nn.Module):
    def forward(self, a, b, normalize):
        return torch.zeros(1, 1, 240, 240)


def test_mask_threshold_fit_excludes_abnormal_calibration_and_all_holdout():
    heatmaps = np.arange(4 * 16, dtype=float).reshape(4, 4, 4)
    labels, calibration = np.array([0, 1, 0, 1]), np.array([0, 1])
    threshold = fit_mask_threshold(heatmaps, labels, calibration)
    heatmaps[1:] = 1e12
    assert fit_mask_threshold(heatmaps, labels, calibration) == threshold
    assert threshold == pytest.approx(14.25)
    with pytest.raises(ValueError, match="Normal calibration"):
        fit_mask_threshold(heatmaps, labels, np.array([1]))


def test_pseudo_mask_fixed_threshold_and_dilation_without_ground_truth():
    heatmap = np.zeros((240, 240))
    heatmap[100, 100] = 1
    mask = pseudo_mask(heatmap, 0.5)
    assert mask.sum() == 9 and mask[99:102, 99:102].all()
    assert not pseudo_mask(heatmap, 1).any()
    with pytest.raises(ValueError, match="threshold"):
        pseudo_mask(heatmap, float("nan"))
