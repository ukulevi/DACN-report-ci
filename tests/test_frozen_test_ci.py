import json
from pathlib import Path

import numpy as np
import pytest
from sklearn.metrics import roc_auc_score

from dacn.evaluation import frozen_test_ci as tc

ROOT = Path(__file__).resolve().parents[1]


def test_frozen_thresholds_match_the_frozen_controls_config():
    controls = json.loads((ROOT / "configs/final_controls_frozen_20261110.json").read_text())
    assert controls["status"] == "frozen"
    assert controls["arms"] == tc.FROZEN_THRESHOLDS
    frozen = json.loads((ROOT / "configs/final_evaluation_frozen_20261110.json").read_text())
    assert frozen["thresholds"] == tc.FROZEN_THRESHOLDS["final_lpips"]


def test_image_scores_mean_of_top_576_pixels():
    maps = np.zeros((2, 240, 240), dtype=np.float32)
    maps[0].flat[:576] = 2.0
    maps[1].flat[:288] = 4.0
    assert tc.image_scores(maps).tolist() == pytest.approx([2.0, 2.0])


def _published():
    evidence = json.loads((ROOT / "docs/evidence/2026-11-10-fourarm-test.json").read_text())
    pixel = {
        arm: {
            "pixel_ap": evidence["arms"][arm]["pixel"]["auprc"],
            "pixel_dice_micro": evidence["arms"][arm]["pixel"]["dice_micro"],
        }
        for arm in tc.ARMS
    }
    image = {
        arm: {
            "auroc": evidence["arms"][arm]["image"]["auroc"],
            "ap": evidence["arms"][arm]["image"]["auprc"],
        }
        for arm in tc.ARMS
    }
    return evidence, pixel, image


def test_check_published_accepts_exact_and_rejects_drift():
    evidence, pixel, image = _published()
    assert tc.check_published(pixel, image, evidence)["max_abs_error"] == 0
    pixel["brightness"]["pixel_ap"] += 1e-4
    with pytest.raises(ValueError, match="do not reproduce"):
        tc.check_published(pixel, image, evidence)


def test_image_group_bootstrap_is_paired_and_matches_point_estimates():
    rng = np.random.default_rng(0)
    labels = np.array([0, 1] * 20)
    groups = np.repeat(np.arange(10), 4)
    scores = {
        "final_lpips": labels + rng.normal(0, 0.8, 40),
        "brightness": labels + rng.normal(0, 1.5, 40),
    }
    report = tc.image_group_bootstrap(
        labels, scores, groups, reference="final_lpips", replicates=50
    )
    final = report["candidates"]["final_lpips"]
    assert final["auroc"] == pytest.approx(roc_auc_score(labels, scores["final_lpips"]))
    assert final["delta_auroc_ci95"] == [0, 0] and final["delta_ap_ci95"] == [0, 0]
    assert report["groups"] == 10 and report["valid_replicates"] == 50
    low, high = final["auroc_ci95"]
    assert 0 <= low <= final["auroc"] <= high <= 1
