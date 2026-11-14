import importlib.util
import sys
import types

import numpy as np
import pytest

pytest.importorskip("torch", reason="Producer pipeline imports require the train extra")

if "numba" not in sys.modules and importlib.util.find_spec("numba") is None:
    # Import-time dependency of the simplex module only; not used by these tests.
    stub = types.ModuleType("numba")
    stub.njit = lambda *a, **k: a[0] if a and callable(a[0]) else (lambda f: f)
    stub.prange = range
    sys.modules["numba"] = stub

from dacn.evaluation import noise_extension as ne


def test_points_keep_budget_and_control_runs_first():
    assert ne.SMOKE_POINTS[0] == ne.CONTROL_POINT == (300, 20, 30)
    assert {p[0] for p in ne.NEW_POINTS} == {50, 400}
    assert all(s1 + s2 == 50 for _, s1, s2 in (ne.CONTROL_POINT, *ne.NEW_POINTS))
    assert ne.point_name(50, 30, 20) == "t50_30_20"


def _cached(rng, n=3):
    cached = {
        "mask_threshold": 0.2,
        "predicted_masks": rng.random((n, 240, 240)) > 0.9,
        "initials": rng.uniform(-1, 1, (n, 256, 256)).astype(np.float32),
        "finals": rng.uniform(-1, 1, (n, 256, 256)).astype(np.float32),
    }
    for name in ne.SCORE_NAMES:
        cached[name] = rng.random((n, 240, 240)).astype(np.float32)
    return cached


def _result_from(cached, selected):
    return {
        "mask_threshold": cached["mask_threshold"],
        "predicted_masks": cached["predicted_masks"][selected].copy(),
        "initials": cached["initials"][selected].copy(),
        "finals": cached["finals"][selected].copy(),
        "maps": {name: cached[name][selected].copy() for name in ne.SCORE_NAMES},
    }


def test_point_replay_accepts_exact_subset_and_rejects_drift():
    cached = _cached(np.random.default_rng(0))
    selected = [0, 2]
    result = _result_from(cached, selected)
    ok = ne.check_point_replay(result, cached, selected)
    assert max(ok["max_abs_errors"].values()) == 0 and ok["threshold_abs_error"] == 0
    drift = _result_from(cached, selected)
    drift["maps"]["autoddpm_lpips"] = drift["maps"]["autoddpm_lpips"] + 1e-3
    with pytest.raises(ValueError, match="tolerance"):
        ne.check_point_replay(drift, cached, selected)
    moved = _result_from(cached, selected)
    moved["mask_threshold"] += 1e-6
    with pytest.raises(ValueError, match="threshold"):
        ne.check_point_replay(moved, cached, selected)


def test_variants_are_valid_single_component_changes_with_control_first():
    from dacn.evaluation.ddim_inpainting import ddim_indices

    names = list(ne.VARIANTS)
    assert names[0] == ne.SMOKE_VARIANTS[0] == "control_t300_20_30"
    assert ne.VARIANTS["control_t300_20_30"] == (ne.CONTROL_POINT, 50, 17, "product")
    control = ne.VARIANTS["control_t300_20_30"]
    for name, (point, phase2_start, seed_base, mask_score) in ne.VARIANTS.items():
        start, steps1, steps2 = point
        assert len(ddim_indices(start, steps1)) == steps1
        assert len(ddim_indices(phase2_start, steps2)) == steps2
        assert mask_score in ("product", "lpips")
        if name.startswith(("seed", "budget", "phase2", "lpips_mask")):
            base = (point, phase2_start, seed_base, mask_score)
            changed = sum(
                a != b
                for a, b in zip(
                    (point[0], (steps1, steps2), phase2_start, seed_base, mask_score),
                    (
                        control[0][0] if not name.startswith("seed") or "t300" in name else 400,
                        control[0][1:],
                        control[1],
                        control[2],
                        control[3],
                    ),
                    strict=True,
                )
            )
            assert changed == 1, (name, base)
    assert set(ne.SMOKE_VARIANTS) <= set(names)
    assert set(ne.PUBLISHED_HOLDOUT_AP) <= set(names)


def test_maskgate_variants_change_only_mask_and_seed_and_control_runs_first():
    names = list(ne.MASKGATE_VARIANTS)
    assert names[0] == ne.SMOKE_MASKGATE[0] == "control_t300_20_30"
    assert ne.MASKGATE_VARIANTS["control_t300_20_30"] == ne.VARIANTS["control_t300_20_30"]
    for name in names[1:]:
        point, phase2_start, seed_base, mask_score = ne.MASKGATE_VARIANTS[name]
        assert (point, phase2_start, mask_score) == (ne.CONTROL_POINT, 50, "lpips_brightness")
        assert f"_s{seed_base}_" in name
    assert "lpips_brightness" in ne.MASK_SCORES
