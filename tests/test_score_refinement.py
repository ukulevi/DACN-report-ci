import numpy as np
import pytest

from dacn.evaluation import score_refinement as sr


def _images(n=3, seed=0):
    rng = np.random.default_rng(seed)
    images = np.full((n, 256, 256), -1.0)
    images[:, 50:200, 50:200] = rng.uniform(-0.5, 1.0, (n, 150, 150))
    return images


def test_gates_are_zero_on_background_and_bounded():
    images = _images()
    brightness = (images[:, 8:-8, 8:-8] + 1) / 2
    labels = np.array([0, 0, 1])
    out = sr.gates(brightness, images, labels, np.arange(3))
    assert set(out) == set(sr.GATES)
    for name in ("normal_cdf", "slice_rank"):
        assert out[name][0, 0, 0] == 0 and 0 <= out[name].min() and out[name].max() <= 1
    assert np.array_equal(out["none"], np.ones_like(brightness))


def test_slice_rank_is_within_slice_and_normal_cdf_uses_normal_calibration_only():
    images = _images(2)
    brightness = (images[:, 8:-8, 8:-8] + 1) / 2
    ranks = sr.slice_rank_gate(brightness, images)
    fg = sr.foreground(images)
    # Same ordering inside a slice after shifting that slice's brightness.
    shifted = brightness.copy()
    shifted[1][fg[1]] += 0.01
    assert np.allclose(sr.slice_rank_gate(shifted, images)[1], ranks[1])
    cdf = sr.normal_cdf_gate(brightness, images, np.array([0, 1]), np.array([0, 1]))
    changed = brightness.copy()
    changed[1] = changed[1] * 0.5  # abnormal slice must not move the reference
    again = sr.normal_cdf_gate(changed, images, np.array([0, 1]), np.array([0, 1]))
    assert np.array_equal(cdf[0], again[0])
    with pytest.raises(ValueError):
        sr.normal_cdf_gate(brightness, images, np.array([1, 1]), np.array([0, 1]))


def test_arms_cover_grid_and_reference_is_producer_product():
    rng = np.random.default_rng(1)
    lp = {b: rng.random((2, 240, 240)) for b in sr.BACKBONES}
    g = {k: rng.random((2, 240, 240)) for k in sr.GATES}
    producer, bright = rng.random((2, 240, 240)), rng.random((2, 240, 240))
    arms = sr.refinement_arms(lp, g, producer, bright)
    assert len(arms) == 1 + len(sr.BACKBONES) * len(sr.GATES)
    assert np.array_equal(arms[sr.REFERENCE], producer * bright)
    assert np.array_equal(arms["vgg__slice_rank"], lp["vgg"] * g["slice_rank"])


def test_improvement_rule_needs_holdout_calibration_and_no_image_loss():
    def boot(ap, auroc):
        return {"candidates": {"x": {"delta_ap_ci95": ap, "delta_auroc_ci95": auroc}}}

    summary = {"x": {"calibration_ap": 0.6}, sr.REFERENCE: {"calibration_ap": 0.5}}
    assert sr.improves("x", summary, boot([0.01, 0.1], [-0.1, 0.1]), boot([0.01, 0.1], [-0.1, 0.1]))
    assert not sr.improves("x", summary, boot([-0.01, 0.1], [0, 0]), boot([0, 0], [-0.1, 0.1]))
    assert not sr.improves("x", summary, boot([0.01, 0.1], [0, 0]), boot([0, 0], [-0.2, -0.01]))
    summary["x"]["calibration_ap"] = 0.4
    assert not sr.improves("x", summary, boot([0.01, 0.1], [0, 0]), boot([0, 0], [-0.1, 0.1]))


def test_lpips_maps_follow_producer_contract():
    torch = pytest.importorskip("torch")

    class Fake(torch.nn.Module):
        def forward(self, a, b, normalize):
            assert not normalize and a.shape == (1, 3, 240, 240)
            return (a - b)[:, :1] - 0.5  # negative values must be floored at 0

    images = np.zeros((1, 256, 256), np.float32)
    recon = np.full((1, 256, 256), -1.0, np.float32)
    out = sr.lpips_maps(Fake(), images, recon)
    assert out.shape == (1, 240, 240) and np.allclose(out, 0.5)
    out = sr.lpips_maps(Fake(), recon, images)
    assert np.all(out == 0)
