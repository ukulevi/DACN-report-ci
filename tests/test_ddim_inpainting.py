import pytest
import torch

from dacn.evaluation.ddim_inpainting import (
    INPAINT_INDICES,
    ddim_initial_reconstruction,
    ddim_step,
    ddim_stitch_resample,
    forward_gap,
)
from dacn.evaluation.sampler_ablation import DDIM_INDICES, deterministic_from_noisy
from dacn.training.ddpm_partial import _fixed_rng, build_diffusion


class ZeroNoise(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.calls = []

    def forward(self, image, timestep):
        self.calls.append(int(timestep.item()))
        return torch.zeros_like(image)


def test_general_forward_gap_uses_cumulative_ratio_including_clean_terminal():
    diffusion = build_diffusion({})
    value, noise = torch.full((1, 1, 4, 4), 0.3), torch.full((1, 1, 4, 4), 0.7)
    for lower, upper in ((44, 49), (-1, 4)):
        a = 1 if lower == -1 else diffusion.alphas_cumprod[lower]
        ratio = diffusion.alphas_cumprod[upper] / a
        expected = ratio**0.5 * value + (1 - ratio) ** 0.5 * noise
        assert torch.equal(forward_gap(diffusion, value, lower, upper, noise), expected)
    wrong_beta = (1 - diffusion.betas[49]) ** 0.5 * value + diffusion.betas[49] ** 0.5 * noise
    assert not torch.allclose(forward_gap(diffusion, value, 44, 49, noise), wrong_beta)
    with pytest.raises(ValueError, match="forward gap"):
        forward_gap(diffusion, value, 49, 44, noise)


def test_single_ddim_jumps_match_existing_clipped_eta0_chain():
    diffusion, model = build_diffusion({}), ZeroNoise()
    value = torch.full((1, 1, 8, 8), 0.95)
    expected = deterministic_from_noisy(model, diffusion, value, indices=(299, 149, 49))
    actual = value
    for current, following in ((299, 149), (149, 49), (49, -1)):
        actual = ddim_step(model, diffusion, actual, current, following)
    assert torch.equal(actual, expected)


def test_initial_ddim_shares_gaussian_forward_noise_and_ten_model_calls():
    diffusion, model = build_diffusion({}), ZeroNoise()
    image = torch.full((1, 1, 16, 16), 0.1)
    rng_before = torch.get_rng_state().clone()
    actual = ddim_initial_reconstruction(model, diffusion, image, seed=17)
    assert model.calls == list(DDIM_INDICES) and len(model.calls) == 10
    assert torch.equal(torch.get_rng_state(), rng_before)
    with _fixed_rng(17, image.device):
        t = torch.tensor([299])
        noisy = diffusion.q_sample(image, t, diffusion.noise(image, t))
    expected = deterministic_from_noisy(ZeroNoise(), diffusion, noisy, indices=DDIM_INDICES)
    assert torch.equal(actual, expected)


def test_ddim_inpaint_fifty_calls_exact_context_seed_determinism_and_rng_isolation():
    diffusion, model = build_diffusion({}), ZeroNoise()
    image = torch.full((1, 1, 256, 256), 0.2)
    mask = torch.zeros_like(image, dtype=torch.bool)
    mask[:, :, 100:120, 100:120] = True
    kwargs = {"diffusion": diffusion, "seed": 17}
    before = torch.get_rng_state().clone()
    output = ddim_stitch_resample(model, image, image * 0, mask, **kwargs)
    assert model.calls == [k for k in INPAINT_INDICES for _ in range(5)]
    assert len(model.calls) == 50
    assert torch.equal(output[~mask], image[~mask]) and not torch.equal(output[mask], image[mask])
    assert torch.equal(torch.get_rng_state(), before)
    again = ddim_stitch_resample(ZeroNoise(), image, image * 0, mask, **kwargs)
    assert torch.equal(output, again) and torch.isfinite(output).all()


def test_full_mask_oracle_recovers_clean_with_nonadjacent_repeated_jumps():
    diffusion = build_diffusion({})
    clean = torch.full((1, 1, 256, 256), 0.1)

    class Oracle(torch.nn.Module):
        def forward(self, value, timestep):
            alpha = float(diffusion.alphas_cumprod[int(timestep.item())])
            return (value - alpha**0.5 * clean) / (1 - alpha) ** 0.5

    output = ddim_stitch_resample(
        Oracle(),
        clean * 2,
        clean * -3,
        torch.ones_like(clean, dtype=torch.bool),
        diffusion=diffusion,
        seed=17,
        indices=(49, 24, 4),
        repeats=3,
    )
    assert torch.allclose(output, clean, atol=1e-6)


def test_ddim_empty_mask_preserves_input_and_rejects_invalid_schedules():
    diffusion, model = build_diffusion({}), ZeroNoise()
    image = torch.full((1, 1, 256, 256), 0.2)
    mask = torch.zeros_like(image, dtype=torch.bool)
    output = ddim_stitch_resample(
        model, image, image, mask, diffusion=diffusion, seed=17, indices=(49, 4), repeats=2
    )
    assert torch.equal(output, image)
    for indices in ((), (4, 49), (49, 49), (True,), (1000,)):
        with pytest.raises(ValueError, match="Indices"):
            ddim_stitch_resample(
                model, image, image, mask, diffusion=diffusion, seed=17, indices=indices
            )
    with pytest.raises(ValueError, match="repeat"):
        ddim_stitch_resample(model, image, image, mask, diffusion=diffusion, seed=17, repeats=0)
