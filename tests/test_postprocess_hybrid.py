import numpy as np

from dacn.evaluation import postprocess_hybrid as ph


def _images(n=2):
    images = np.full((n, 256, 256), -1.0, dtype=np.float32)
    images[:, 40:216, 40:216] = 0.2  # brain block inside the 240 crop
    images[:, 100:110, 100:110] = -1.0  # dark hole that must be filled
    return images


def test_brain_mask_fills_holes_erodes_rim_and_uses_crop():
    brain = ph.brain_masks(_images())
    assert brain.shape == (2, 240, 240)
    # Crop offset 8: block spans 32..207 in crop coordinates; eroded by 3 pixels.
    assert brain[0, 35:205, 35:205].all()
    assert not brain[0, 32:35].any() and not brain[0, :, 205:].any()
    assert brain[0, 92:102, 92:102].all()  # filled hole


def test_postprocess_zeroes_outside_brain_and_median_smooths():
    brain = ph.brain_masks(_images(1))
    maps = np.ones((1, 240, 240))
    maps[0, 120, 120] = 50.0  # single-pixel spike inside the brain
    assert np.array_equal(ph.postprocess(maps, brain, "none"), maps)
    masked = ph.postprocess(maps, brain, "brain")
    assert masked[0, 0, 0] == 0 and masked[0, 120, 120] == 50.0
    smooth = ph.postprocess(maps, brain, "brain_median5")
    assert smooth[0, 120, 120] == 1.0 and smooth[0, 0, 0] == 0


def test_arms_cover_every_base_and_postprocess_and_references_are_raw():
    rng = np.random.default_rng(0)
    lpips = rng.random((4, 240, 240))
    bright = rng.random((4, 240, 240))
    labels = np.array([0, 0, 1, 1])
    arms, _ = ph.all_arms(lpips, bright, _images(4), labels, np.arange(4))
    assert len(arms) == 6 * len(ph.POSTPROCESS)
    assert np.array_equal(arms[ph.REFERENCE], lpips)
    assert np.array_equal(arms[ph.BASELINE], bright)
    w050 = arms["lpips_bright_cdf_w050__none"]
    assert w050.min() >= 0 and w050.max() <= 1


def test_rule_requires_both_references_beaten_and_no_image_or_calibration_loss():
    def boot(delta):
        return {"candidates": {"x": {"delta_ap_ci95": delta, "delta_auroc_ci95": delta}}}

    summary = {"x": {"calibration_ap": 0.4}, ph.REFERENCE: {"calibration_ap": 0.3}}
    good, bad = [0.01, 0.05], [-0.01, 0.05]
    assert ph.passes_rule("x", summary, boot(good), boot(good), boot([-0.02, 0.01]))
    assert not ph.passes_rule("x", summary, boot(good), boot(bad), boot(good))
    assert not ph.passes_rule("x", summary, boot(good), boot(good), boot([-0.05, -0.01]))
    summary["x"]["calibration_ap"] = 0.2
    assert not ph.passes_rule("x", summary, boot(good), boot(good), boot(good))
    assert not ph.passes_rule(ph.REFERENCE, summary, boot(good), boot(good), boot(good))


def test_coverage_counts_lesion_pixels_inside_brain():
    brain = np.zeros((1, 240, 240), bool)
    brain[0, :, :120] = True
    truth = np.zeros((1, 240, 240), np.uint8)
    truth[0, 0, 118:122] = 1
    assert ph.brain_mask_coverage(brain, truth) == 0.5


def test_gated_arms_multiply_each_source_by_the_same_brightness():
    rng = np.random.default_rng(1)
    arrays = {key: rng.random((2, 240, 240)) for key in {*ph.GATE_SOURCES.values(), "brightness"}}
    arms = ph.gated_arms(arrays)
    assert set(arms) == {ph.REFERENCE, ph.BASELINE} | {f"{n}_x_brightness" for n in ph.GATE_SOURCES}
    assert ph.GATE_REFERENCE in arms and ph.GATE_KEY_CONTRAST in arms
    assert np.array_equal(
        arms["initial_lpips_x_brightness"],
        arrays["initial_lpips_only"] * arrays["brightness"],
    )


def test_resampling_credit_requires_holdout_and_calibration_agreement():
    def boot(delta):
        return {"candidates": {ph.GATE_KEY_CONTRAST: {"delta_ap_ci95": delta}}}

    summary = {
        ph.GATE_REFERENCE: {"calibration_ap": 0.5},
        ph.GATE_KEY_CONTRAST: {"calibration_ap": 0.4},
    }
    assert ph.credits_resampling(summary, boot([-0.2, -0.05]))
    assert not ph.credits_resampling(summary, boot([-0.2, 0.01]))
    summary[ph.GATE_KEY_CONTRAST]["calibration_ap"] = 0.6
    assert not ph.credits_resampling(summary, boot([-0.2, -0.05]))


def test_frozen_hybrid_thresholds_match_validation_evidence():
    import json
    from pathlib import Path

    evidence = Path(__file__).parents[1] / "docs/evidence/2026-11-14-hybrid-results.json"
    arm = json.loads(evidence.read_text(encoding="utf-8"))["arms"]["lpips_x_brightness__none"]
    assert arm["score_thresholds"] == ph.FROZEN_HYBRID_THRESHOLDS
    a, b = np.random.default_rng(2).random((2, 1, 240, 240)).astype(np.float32)
    assert np.array_equal(
        ph.frozen_hybrid_map(a, b),
        ph.gated_arms({**{k: a for k in ph.GATE_SOURCES.values()}, "brightness": b})[
            ph.FROZEN_HYBRID_ARM
        ],
    )


def test_seed_arms_share_brightness_and_cover_each_seed():
    rng = np.random.default_rng(3)
    seed_maps = {
        s: {
            "autoddpm_lpips": rng.random((2, 240, 240)),
            "initial_lpips_only": rng.random((2, 240, 240)),
        }
        for s in (17, 18)
    }
    bright = rng.random((2, 240, 240))
    arms = ph.seed_arms(seed_maps, bright)
    assert set(arms) == {"brightness"} | {
        f"{kind}_s{s}"
        for s in (17, 18)
        for kind in ("final_lpips", "final_x_brightness", "initial_x_brightness")
    }
    assert np.array_equal(arms["final_x_brightness_s18"], seed_maps[18]["autoddpm_lpips"] * bright)


def test_seed_verdict_applies_rules_per_seed():
    def boot(seed, bright, final, initial):
        return {
            "candidates": {
                "brightness": {"delta_ap_ci95": bright},
                f"final_lpips_s{seed}": {"delta_ap_ci95": final},
                f"initial_x_brightness_s{seed}": {"delta_ap_ci95": initial},
            }
        }

    win, tie = [-0.3, -0.1], [-0.1, 0.02]
    summary = {}
    for seed, ap in ((17, 0.65), (18, 0.60)):
        summary[f"final_x_brightness_s{seed}"] = {"holdout_ap": ap, "calibration_ap": 0.5}
        summary[f"initial_x_brightness_s{seed}"] = {"calibration_ap": 0.4}
    verdict = ph.seed_verdict(
        summary, {17: boot(17, win, win, win), 18: boot(18, win, win, tie)}, (17, 18)
    )
    assert verdict["seed_robust"] and verdict["resampling_credit_count"] == 1
    assert abs(verdict["hybrid_holdout_ap_mean"] - 0.625) < 1e-12
    worse = ph.seed_verdict(
        summary, {17: boot(17, win, win, win), 18: boot(18, tie, win, win)}, (17, 18)
    )
    assert not worse["seed_robust"] and worse["beats_brightness"] == {17: True, 18: False}
