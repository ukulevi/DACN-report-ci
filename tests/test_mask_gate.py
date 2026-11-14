from dacn.evaluation import mask_gate as mg


def _summary(bmask=(0.70, 0.62, 0.66), product=(0.656, 0.610, 0.659), cal=(0.55, 0.50)):
    out = {}
    for seed, b, p in zip(mg.SEEDS, bmask, product, strict=True):
        out[f"bmask_s{seed}"] = {"holdout_ap": b, "calibration_ap": cal[0]}
        out[f"product_s{seed}"] = {"holdout_ap": p, "calibration_ap": cal[1]}
    out["bmask_ensemble"] = {"holdout_ap": 0.75, "calibration_ap": 0.6}
    out["product_ensemble"] = {"holdout_ap": 0.711, "calibration_ap": 0.561}
    return out


def _boot(name, ap, auroc=(-0.1, 0.1)):
    return {"candidates": {name: {"delta_ap_ci95": list(ap), "delta_auroc_ci95": list(auroc)}}}


def test_r1_needs_pooled_ci_points_image_and_two_seed_wins():
    s = _summary()
    assert mg.seed_wins(s) == 3
    assert mg.r1(s, _boot("bmask_s17", (0.01, 0.1)), _boot("bmask_s17", (0, 0)))
    assert not mg.r1(s, _boot("bmask_s17", (-0.01, 0.1)), _boot("bmask_s17", (0, 0)))
    assert not mg.r1(
        s, _boot("bmask_s17", (0.01, 0.1)), _boot("bmask_s17", (0, 0), auroc=(-0.2, -0.01))
    )
    few = _summary(bmask=(0.70, 0.60, 0.65))
    assert mg.seed_wins(few) == 1
    assert not mg.r1(few, _boot("bmask_s17", (0.01, 0.1)), _boot("bmask_s17", (0, 0)))


def test_test_candidate_falls_back_to_product_ensemble():
    s = _summary()
    good = _boot("bmask_ensemble", (0.01, 0.1))
    image = _boot("bmask_ensemble", (0, 0))
    assert mg.choose_test_candidate(s, good, image) == "bmask_ensemble"
    bad = _boot("bmask_ensemble", (-0.01, 0.1))
    assert mg.choose_test_candidate(s, bad, image) == "product_ensemble"
