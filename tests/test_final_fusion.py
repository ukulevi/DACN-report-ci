import numpy as np
import pytest

from dacn.evaluation import final_fusion as ff


def test_midrank_cdf_hand_values_with_ties_and_saturation():
    reference = np.sort(np.array([0.0, 1.0, 1.0, 2.0]))
    values = np.array([-1.0, 0.0, 1.0, 1.5, 2.0, 9.0])
    expected = [0.0, 0.125, 0.5, 0.75, 0.875, 1.0]
    assert ff.apply_midrank_cdf(reference, values).tolist() == pytest.approx(expected)


def test_reference_uses_only_normal_calibration_slices():
    maps = np.stack([np.full((2, 2), v) for v in (1.0, 2.0, 100.0, 3.0)])
    labels = np.array([0, 0, 1, 0])
    reference = ff.fit_midrank_cdf(maps, labels, [0, 1, 2])
    assert reference.tolist() == [1.0] * 4 + [2.0] * 4
    with pytest.raises(ValueError, match="Normal calibration"):
        ff.fit_midrank_cdf(maps, np.ones(4, int), [0, 1])


def test_fused_maps_weights_and_endpoints():
    rng = np.random.default_rng(0)
    lpips = rng.random((4, 8, 8))
    residual = rng.random((4, 8, 8))
    labels = np.array([0, 0, 1, 1])
    maps = ff.fused_maps(lpips, residual, labels, [0, 1, 2, 3])
    assert list(maps) == [ff.REFERENCE] + [ff.fusion_name(w) for w in ff.RESIDUAL_WEIGHTS]
    assert ff.fusion_name(0.25) == "lpips_l1_cdf_w025"
    np.testing.assert_array_equal(maps[ff.REFERENCE], lpips)
    lpips_cdf = ff.apply_midrank_cdf(ff.fit_midrank_cdf(lpips, labels, [0, 1]), lpips)
    residual_cdf = ff.apply_midrank_cdf(ff.fit_midrank_cdf(residual, labels, [0, 1]), residual)
    np.testing.assert_allclose(maps["lpips_l1_cdf_w000"], lpips_cdf)
    np.testing.assert_allclose(maps["lpips_l1_cdf_w100"], residual_cdf)
    np.testing.assert_allclose(maps["lpips_l1_cdf_w050"], 0.5 * lpips_cdf + 0.5 * residual_cdf)
    # Inside the normal reference range the CDF preserves LPIPS ordering.
    inside = (lpips[:2] > lpips[:2].min()) & (lpips[:2] < lpips[:2].max())
    order = np.argsort(lpips[:2][inside])
    assert np.all(np.diff(maps["lpips_l1_cdf_w000"][:2][inside][order]) >= 0)
