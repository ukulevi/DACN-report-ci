import numpy as np

from dacn.evaluation import retrain_variability as rv


def test_model_arms_use_seed17_for_hybrid_and_mean_for_ensemble():
    f = {s: np.full((1, 2, 2), v) for s, v in zip(rv.SAMPLING_SEEDS, (0.3, 0.6, 0.9), strict=True)}
    bright = np.full((1, 2, 2), 0.5)
    arms = rv.model_arms(f, bright, "a100_s18")
    assert np.allclose(arms["a100_s18__final"], 0.3)
    assert np.allclose(arms["a100_s18__hybrid"], 0.15)
    assert np.allclose(arms["a100_s18__ensemble"], 0.3)


def test_verdict_rules_and_spread():
    summary, points, pooled = {}, {}, {}
    for i, model in enumerate(rv.MODELS):
        for kind, base in (("final", 0.3), ("hybrid", 0.6), ("ensemble", 0.65)):
            summary[rv.arm(model, kind)] = {"holdout_ap": base + 0.01 * i}
            points[rv.arm(model, kind)] = base + 0.01 * i
        pooled[model] = {
            "candidates": {
                "brightness": {"delta_ap_ci95": [-0.3, -0.1]},
                rv.arm(model, "final"): {"delta_ap_ci95": [-0.4, -0.2]},
            }
        }
    out = rv.verdict(summary, pooled, points)
    assert all(out["v1_method_holds"].values()) and all(out["v2_ensemble_holds"].values())
    assert abs(out["across_model"]["hybrid"]["pooled_mean"] - 0.615) < 1e-12
    pooled["a100_s19"]["candidates"]["brightness"]["delta_ap_ci95"] = [-0.1, 0.02]
    points["a100_s18__ensemble"] = 0.5
    out = rv.verdict(summary, pooled, points)
    assert out["v1_method_holds"]["a100_s19"] is False
    assert out["v2_ensemble_holds"]["a100_s18"] is False
