import numpy as np
import pytest

from dacn.evaluation import hybrid_diagnostics as dg


def test_lesion_detection_bins_by_size_and_counts_touching_hits():
    truth = np.zeros((1, 240, 240), bool)
    truth[0, 10:15, 10:15] = True  # 25 px -> small bin, detected
    truth[0, 100:120, 100:120] = True  # 400 px -> 200-999 bin, missed
    predicted = np.zeros_like(truth)
    predicted[0, 12, 12] = True
    stats = dg.lesion_detection(truth, predicted)
    assert stats["1-49"] == {
        "lesions": 1,
        "detected": 1,
        "pixels": 25,
        "detection_rate": 1.0,
        "pixel_recall": 0.04,
    }
    assert stats["200-999"]["detected"] == 0 and stats["200-999"]["detection_rate"] == 0.0
    assert stats[">=1000"]["detection_rate"] is None


def test_false_positive_breakdown_by_region_brightness_and_speckle():
    images = np.full((2, 256, 256), -1.0)
    images[:, 40:216, 40:216] = 0.0
    truth = np.zeros((2, 240, 240), bool)
    predicted = np.zeros_like(truth)
    predicted[0, 0:5, 0:5] = True  # background, 25 px component
    predicted[1, 100, 100] = True  # interior speckle on the normal slice
    brightness = (images[:, 8:-8, 8:-8] + 1) / 2
    out = dg.false_positive_breakdown(truth, predicted, images, brightness, np.array([1, 0]))
    assert out["fp_pixels"] == 26
    assert out["by_region"]["background"] == pytest.approx(25 / 26)
    assert out["by_region"]["interior"] == pytest.approx(1 / 26)
    assert out["on_normal_slices"] == pytest.approx(1 / 26)
    assert out["predicted_in_components_under_10px"] == pytest.approx(1 / 26)
    assert sum(out["by_brightness_quarter"].values()) == pytest.approx(1.0)


def test_ensemble_mean_and_shape_check():
    a, b = np.zeros((1, 2, 2)), np.ones((1, 2, 2))
    assert np.array_equal(dg.ensemble_mean([a, b]), np.full((1, 2, 2), 0.5))
    assert np.array_equal(dg.ensemble_mean(x for x in (a, b)), np.full((1, 2, 2), 0.5))
    with pytest.raises(ValueError):
        dg.ensemble_mean([a, np.ones((1, 3, 3))])


def test_ensemble_rule_and_contrast_status():
    def boot(name, ap, auroc=(-0.1, 0.1)):
        return {"candidates": {name: {"delta_ap_ci95": ap, "delta_auroc_ci95": auroc}}}

    summary = {
        "e": {"holdout_ap": 0.7, "calibration_ap": 0.6},
        "frozen_hybrid": {"holdout_ap": 0.65, "calibration_ap": 0.5},
    }
    assert dg.ensemble_improves("e", summary, boot("e", [0.01, 0.1]), boot("e", [0, 0]))
    assert not dg.ensemble_improves("e", summary, boot("e", [-0.01, 0.1]), boot("e", [0, 0]))
    summary["e"]["holdout_ap"] = 0.6
    assert not dg.ensemble_improves("e", summary, boot("e", [0.01, 0.1]), boot("e", [0, 0]))
    pooled = {
        "final_t300": boot("final_t400", [0.01, 0.05]),
        "frozen_hybrid": {
            "candidates": {
                "hybrid_t400": {"delta_ap_ci95": [-0.1, 0.1]},
                "slice_rank": {"delta_ap_ci95": [-0.2, -0.01]},
                "normal_cdf": {"delta_ap_ci95": [-0.1, 0.0]},
            }
        },
    }
    status = dg.contrast_status(pooled)
    assert status["final_t400 - final_t300"]["resolved"]
    assert not status["hybrid_t400 - frozen_hybrid"]["resolved"]
    assert status["slice_rank - frozen_hybrid"]["resolved"]
    assert not status["normal_cdf - frozen_hybrid"]["resolved"]
