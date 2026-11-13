import importlib.util
import sys
import types

import numpy as np
import pytest

if importlib.util.find_spec("numba") is None:
    # The producer import chain loads the simplex noise module, which needs numba
    # at import time only; the mask functions under test never call it.
    stub = types.ModuleType("numba")
    stub.njit = lambda *a, **k: a[0] if a and callable(a[0]) else (lambda f: f)
    stub.prange = range
    sys.modules["numba"] = stub

from dacn.evaluation import mask_rule_ablation as mra  # noqa: E402
from dacn.evaluation.autoddpm_lpips import fit_mask_threshold, pseudo_mask  # noqa: E402


def _heatmaps(seed=0, n=6):
    rng = np.random.default_rng(seed)
    return rng.gamma(2.0, 0.05, size=(n, 240, 240)).astype(np.float32)


def test_control_rule_is_identical_to_producer_threshold_and_mask():
    heatmaps = _heatmaps()
    labels = np.array([0, 0, 1, 0, 1, 1])
    calibration = np.array([0, 1, 2, 3])
    threshold = mra.fit_percentile_threshold(heatmaps, labels, calibration, 95)
    assert threshold == fit_mask_threshold(heatmaps, labels, calibration)
    for heatmap in heatmaps:
        assert np.array_equal(mra.rule_mask(heatmap, threshold, 3), pseudo_mask(heatmap, threshold))


def test_tighter_rules_mask_fewer_pixels_and_dilation1_is_identity():
    heatmaps = _heatmaps(1)
    labels = np.array([0, 0, 1, 0, 1, 1])
    calibration = np.arange(6)
    sizes = {}
    for percentile, dilation in mra.MASK_RULES:
        threshold = mra.fit_percentile_threshold(heatmaps, labels, calibration, percentile)
        sizes[(percentile, dilation)] = int(mra.rule_mask(heatmaps[2], threshold, dilation).sum())
        if dilation == 1:
            assert np.array_equal(mra.rule_mask(heatmaps[2], threshold, 1), heatmaps[2] > threshold)
    for dilation in (1, 3):
        assert sizes[(90, dilation)] >= sizes[(95, dilation)] >= sizes[(98, dilation)]
        assert sizes[(98, dilation)] >= sizes[(99, dilation)]
    assert all(sizes[(p, 3)] >= sizes[(p, 1)] for p in (90, 95, 98, 99))
    assert mra.MASK_RULES[0] == mra.SMOKE_RULES[0] == mra.CONTROL


def test_invalid_rules_and_calibration_are_rejected():
    heatmaps = _heatmaps()
    with pytest.raises(ValueError, match="Normal calibration"):
        mra.fit_percentile_threshold(heatmaps, np.ones(6, int), np.arange(6), 95)
    with pytest.raises(ValueError, match="Invalid mask rule"):
        mra.rule_mask(heatmaps[0], 0.1, 5)


def test_mask_statistics_hand_computed():
    images = np.full((2, 256, 256), 0.0, dtype=np.float32)
    images[:, 8:18, :] = -1.0  # ten background rows in the crop
    masks = np.zeros((2, 256, 256), dtype=np.uint8)
    masks[1, 108:118, 108:118] = 1  # crop rows/cols 100:110
    predicted = np.zeros((2, 240, 240), dtype=bool)
    predicted[1, 100:105, 100:110] = True  # 50 lesion pixels
    predicted[1, 0:5, 0:10] = True  # 50 background pixels on the abnormal slice
    predicted[0, 50:60, 50:60] = True  # 100 normal foreground pixels
    stats = mra.mask_statistics(predicted, images, masks, np.array([0, 1]), [0, 1])
    assert stats["lesion_pixel_recall"] == pytest.approx(0.5)
    assert stats["pixel_precision"] == pytest.approx(50 / 200)
    assert stats["normal_slice_foreground_fraction_masked"] == pytest.approx(100 / (230 * 240))
    assert stats["mean_masked_fraction_of_crop"] == pytest.approx(200 / (2 * 240 * 240))


def test_control_replay_accepts_tiny_error_and_rejects_drift():
    rng = np.random.default_rng(3)
    cached = {
        "mask_threshold": 0.25,
        "predicted_masks": rng.random((2, 240, 240)) > 0.9,
        "finals": rng.uniform(-1, 1, (2, 256, 256)).astype(np.float32),
        "autoddpm_lpips": rng.random((2, 240, 240)).astype(np.float32),
    }
    ok = mra.check_control_replay(
        0.25,
        cached["predicted_masks"],
        cached["finals"] + 1e-7,
        cached["autoddpm_lpips"],
        cached,
    )
    assert ok["final_max_abs_error"] < 1e-6 and ok["lpips_max_abs_error"] == 0
    with pytest.raises(ValueError, match="threshold"):
        mra.check_control_replay(
            0.2501, cached["predicted_masks"], cached["finals"], cached["autoddpm_lpips"], cached
        )
    flipped = cached["predicted_masks"].copy()
    flipped[0, 0, 0] = ~flipped[0, 0, 0]
    with pytest.raises(ValueError, match="pseudo-masks"):
        mra.check_control_replay(0.25, flipped, cached["finals"], cached["autoddpm_lpips"], cached)
    with pytest.raises(ValueError, match="tolerance"):
        mra.check_control_replay(
            0.25,
            cached["predicted_masks"],
            cached["finals"] + 1e-3,
            cached["autoddpm_lpips"],
            cached,
        )
