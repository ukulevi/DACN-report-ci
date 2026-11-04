"""Verify the denoising score's indexing, aggregation and isolated randomness."""

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from dacn.evaluation.denoising import denoising_score_map


class IdentityForward:
    noise_type = "gaussian"
    timesteps = 1000

    def q_sample(self, image, t, noise):
        # Oracle model can recover exact noise from this test forward process.
        return noise


class Oracle(torch.nn.Module):
    def forward(self, noisy, t):
        return noisy


def test_exact_noise_predictor_gives_zero_anomaly_map():
    result = denoising_score_map(Oracle(), torch.zeros(1, 1, 256, 256), diffusion=IdentityForward())
    assert result.shape == (240, 240)
    assert torch.count_nonzero(result) == 0


def test_score_uses_one_based_steps_and_means_error_over_repeats():
    seen = []

    class ShiftedOracle(torch.nn.Module):
        def forward(self, noisy, t):
            seen.append(int(t.item()))
            return noisy + t.float().reshape(1, 1, 1, 1) / 1000

    global_state = torch.get_rng_state().clone()
    result = denoising_score_map(
        ShiftedOracle(), torch.zeros(1, 1, 256, 256), diffusion=IdentityForward(), repeats=2
    )
    assert seen == [99, 199, 299] * 2
    assert np.allclose(result.numpy(), np.mean(np.square([0.099, 0.199, 0.299])), atol=1e-7)
    assert torch.equal(global_state, torch.get_rng_state())


@pytest.mark.parametrize(
    "options", [{"timesteps": (0,)}, {"timesteps": (100, 100)}, {"repeats": 0}, {"seed": -1}]
)
def test_invalid_scoring_contract_is_rejected(options):
    with pytest.raises(ValueError):
        denoising_score_map(
            Oracle(), torch.zeros(1, 1, 256, 256), diffusion=IdentityForward(), **options
        )
