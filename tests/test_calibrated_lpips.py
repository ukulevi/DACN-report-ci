import numpy as np
import pytest

torch = pytest.importorskip("torch")

from dacn.evaluation.calibrated_lpips import (
    apply_layer_calibration,
    fit_layer_calibration,
    layer_contributions,
)


def test_seven_layer_capture_reproduces_aggregate_and_input_contract():
    class Model(torch.nn.Module):
        def forward(self, x, y, normalize, retPerLayer):
            assert not normalize and retPerLayer
            assert x.shape == y.shape == (1, 3, 240, 240)
            assert x.min() == -0.5 and y.max() == 0.5
            layers = [torch.full((1, 1, 240, 240), i / 100) for i in range(7)]
            return sum(layers), layers

    image = np.full((256, 256), -0.5, np.float32)
    total, layers, error = layer_contributions(Model(), image, -image, "cpu")
    assert total.shape == (240, 240) and layers.shape == (7, 240, 240)
    assert error == 0 and np.allclose(total, 0.21)


def test_layer_calibration_only_normal_calibration_foreground_and_no_tumor_masks():
    images = np.zeros((4, 4, 4))
    images[0, 0] = -1
    values = np.arange(2 * 4 * 4 * 4, dtype=float).reshape(2, 4, 4, 4)
    labels, calibration = np.array([0, 1, 0, 1]), np.array([0, 1])
    fitted = fit_layer_calibration(values, images, labels, calibration)
    values[:, 1:] = 1e12  # abnormal calibration and both holdout classes
    values[:, 0, 0] = 1e10  # background in the only normal-calibration image
    assert fit_layer_calibration(values, images, labels, calibration) == fitted
    assert fitted["normal_calibration_samples"] == 1
    assert fitted["normal_foreground_pixels"] == 12
    assert fitted["medians"] == [9.5, 73.5]


def test_normal_calibration_can_have_unprocessed_holdout_without_leaking():
    values = np.full((7, 3, 2, 2), np.nan)
    values[:, 0] = 0.1
    images, labels = np.zeros((3, 2, 2)), np.array([0, 0, 1])
    fitted = fit_layer_calibration(values, images, labels, np.array([0]))
    assert fitted["floor_active"] == [True] * 7
    assert np.isfinite(apply_layer_calibration(values[:, [0]], fitted)).all()
    with pytest.raises(ValueError, match="Nonfinite"):
        apply_layer_calibration(values, fitted)


def test_fixed_equal_layer_normalization_formula_without_background_gating():
    layers = np.array([[[1.0, 3.0]], [[2.0, 6.0]]])
    fitted = {"medians": [1.0, 2.0], "scales": [2.0, 4.0]}
    result = apply_layer_calibration(layers, fitted)
    assert np.array_equal(result, np.array([[0.0, 1.0]]))
    assert np.array_equal(apply_layer_calibration(layers * 0, fitted), np.array([[-0.5, -0.5]]))


def test_reject_missing_normal_foreground_and_duplicate_fit_indices():
    images, values, labels = np.full((2, 3, 3), -1.0), np.zeros((7, 2, 3, 3)), np.array([0, 1])
    with pytest.raises(ValueError, match="Normal foreground"):
        fit_layer_calibration(values, images, labels, np.array([0]))
    with pytest.raises(ValueError, match="Normal calibration"):
        fit_layer_calibration(values, images, labels, np.array([1]))
    with pytest.raises(ValueError, match="indices"):
        fit_layer_calibration(values, images, labels, np.array([0, 0]))
