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
