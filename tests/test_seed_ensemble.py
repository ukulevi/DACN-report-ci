import json
from pathlib import Path

import numpy as np
import pytest

from dacn.evaluation import seed_ensemble as se

EVIDENCE = Path(__file__).parents[1] / "docs" / "evidence"


def _load(name):
    return json.loads((EVIDENCE / name).read_text(encoding="utf-8"))


def test_frozen_values_match_validation_evidence():
    arm = _load("2026-11-14-maskgate-results.json")["arms"]["product_ensemble"]
    assert arm["score_thresholds"] == se.FROZEN_ENSEMBLE_THRESHOLDS
    assert arm["group_holdout"]["pixel"]["auprc"] == se.VALIDATION_HOLDOUT_AP
    points = _load("2026-11-14-variants-results.json")["points"]
    assert points["control_t300_20_30"]["mask_threshold"] == se.MASK_THRESHOLDS[17]
    assert points["seed18_t300_20_30"]["mask_threshold"] == se.MASK_THRESHOLDS[18]
    assert points["seed19_t300_20_30"]["mask_threshold"] == se.MASK_THRESHOLDS[19]
    hybrid = _load("2026-11-14-hybrid-test-results.json")["frozen_hybrid_test"]
    assert hybrid["pixel"]["auprc"] == se.HYBRID_TEST_AP


def test_ensemble_map_is_mean_times_brightness():
    a, b, c = (np.full((1, 2, 2), v, np.float32) for v in (0.1, 0.2, 0.6))
    bright = np.full((1, 2, 2), 0.5)
    assert np.allclose(se.ensemble_map([a, b, c], bright), 0.15)
    with pytest.raises(ValueError):
        se.ensemble_map([a, b], bright)
