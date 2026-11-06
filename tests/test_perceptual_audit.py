import numpy as np
import pytest

torch = pytest.importorskip("torch")

from dacn.evaluation.autoddpm_lpips import lpips_heatmap
from dacn.evaluation.perceptual_audit import (
    fit_diagnostic_gradient,
    mask_error_report,
    perceptual_components,
    tissue_proxy_regions,
)


class FixedPerceptual(torch.nn.Module):
    def forward(self, a, b, normalize):
        assert a.shape == b.shape == (1, 3, 240, 240) and not normalize
        return torch.linspace(0, 1, 240).repeat(240, 1)[None, None]


def test_perceptual_terms_reproduce_existing_product_without_lesion_masks():
    rng = np.random.default_rng(17)
    image = rng.uniform(-1, 1, (256, 256)).astype(np.float32)
    initial = np.clip(image + 0.2, -1, 1)
    parts = perceptual_components(FixedPerceptual(), image, initial, "cpu")
    assert np.array_equal(parts["product"], lpips_heatmap(FixedPerceptual(), image, initial, "cpu"))
    assert np.isfinite(parts["lpips_only"]).all()
    assert parts["lpips_only"].shape == (240, 240)
    # Identical inputs remove the product even if a mock LPIPS map is nonzero.
    identical = perceptual_components(FixedPerceptual(), image, image, "cpu")
    assert not identical["normalized_absolute_only"].any() and not identical["product"].any()


def test_diagnostic_threshold_ignores_abnormal_calibration_and_holdout():
    images = np.zeros((4, 20, 20))
    images[0, 5:15] = 0.25
    labels, calibration = np.array([0, 1, 0, 1]), np.array([0, 1])
    before = fit_diagnostic_gradient(images, labels, calibration)
    images[1:] = np.random.default_rng(3).normal(size=images[1:].shape) * 1e6
    assert fit_diagnostic_gradient(images, labels, calibration) == before


def test_proxy_regions_exclusive_exhaustive_and_edge_width():
    image = np.full((30, 30), -1.0)
    image[5:25, 5:25] = 0
    regions = tissue_proxy_regions(image, 0.01)
    assert np.array_equal(sum(v.astype(int) for v in regions.values()), np.ones_like(image))
    assert regions["background"][0, 0]
    assert regions["foreground_boundary_5px"][5, 5]
    assert not regions["foreground_boundary_5px"][15, 15]


def test_mask_error_counts_lesion_misses_and_disjoint_false_positive_strata():
    image = np.zeros((1, 30, 30))
    truth = np.zeros_like(image, dtype=bool)
    truth[0, 10:20, 10:20] = True
    pred = truth.copy()
    pred[0, 10:12, 10:20] = False
    pred[0, 0, 0] = True
    report = mask_error_report(image, truth, np.array([1]), pred, [0], 0.1)
    assert report["lesion_pixels"] == 100 and report["missed_lesion_pixels"] == 20
    assert report["lesion_miss_rate"] == pytest.approx(0.2)
    assert report["false_positive_pixels"] == 1
    assert sum(v["false_positive_pixels"] for v in report["false_positive_regions"].values()) == 1
    assert sum(v["missed_pixels"] for v in report["missed_lesion_geometry"].values()) == 20
    assert report["lesion_components_8_connected"]["area26_100"]["component_macro_recall"] == 0.8
    assert report["lesion_components_8_connected"]["area26_100"]["wholly_missed"] == 0


def test_wholly_missed_small_component_and_normal_only_strata():
    images = np.zeros((2, 20, 20))
    masks = np.zeros_like(images, dtype=bool)
    masks[1, 10, 10] = True
    pred = np.zeros_like(masks)
    pred[0, 0, 0] = True
    report = mask_error_report(images, masks, np.array([0, 1]), pred, [0, 1], 0.1)
    assert report["lesion_components_8_connected"]["area1_25"]["wholly_missed"] == 1
    assert (
        sum(v["normal_false_positive_pixels"] for v in report["false_positive_regions"].values())
        == 1
    )
    assert report["missed_lesion_geometry"]["core_over3px"]["miss_rate"] is None
