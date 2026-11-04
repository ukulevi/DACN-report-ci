"""Sampler invariants for the AutoDDPM-style SSIM-proxy adaptation."""

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from dacn.evaluation.masked_diffusion import automatic_mask, stitch_resample


class Model(torch.nn.Module):
    def forward(self, x, t):
        return x


class Diffusion:
    noise_type = "gaussian"
    timesteps = 1000
    betas = np.full(1000, 0.01)

    def __init__(self):
        self.seen = []

    def q_sample(self, x, t, noise):
        self.seen.append(("q", int(t.item())))
        return x + noise * 0.1

    def p_sample(self, model, x, t):
        self.seen.append(("p", int(t.item())))
        return x * 0.5


def test_empty_predicted_mask_preserves_input_and_global_rng():
    x = torch.full((1, 1, 256, 256), 0.25)
    state = torch.get_rng_state().clone()
    d = Diffusion()
    final = stitch_resample(
        Model(),
        x,
        x * 0,
        torch.zeros_like(x, dtype=torch.bool),
        diffusion=d,
        seed=17,
        steps=2,
        repeats=2,
    )
    assert torch.equal(final, x)
    assert torch.equal(state, torch.get_rng_state())
    assert [t for tag, t in d.seen if tag == "p"] == [1, 1, 0, 0]
    assert all(t >= 0 for _, t in d.seen)


def test_masked_region_changes_while_context_remains_exact():
    x = torch.full((1, 1, 256, 256), 0.5)
    mask = torch.zeros_like(x, dtype=torch.bool)
    mask[:, :, 100:120, 100:120] = True
    kwargs = {"diffusion": Diffusion(), "seed": 17, "steps": 2, "repeats": 2}
    final = stitch_resample(Model(), x, x * 0, mask, **kwargs)
    again = stitch_resample(Model(), x, x * 0, mask, **kwargs)
    assert torch.equal(final[~mask], x[~mask])
    assert not torch.equal(final[mask], x[mask])
    assert torch.equal(final, again)


def test_automatic_mask_uses_only_image_pair_and_handles_identical_pair():
    x = np.zeros((256, 256))
    heatmap, mask = automatic_mask(x, x)
    assert heatmap.shape == mask.shape == (240, 240)
    assert not mask.any()
    assert np.isfinite(heatmap).all()
