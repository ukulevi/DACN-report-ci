"""Controlled audit must reproduce production sampling before comparing variants."""

import pytest

torch = pytest.importorskip("torch")

from dacn.evaluation.contract_audit import controlled_chain
from dacn.training.ddpm_partial import _fixed_rng, build_diffusion


class ZeroModel(torch.nn.Module):
    def forward(self, x, t):
        return torch.zeros_like(x)


def test_controlled_large_chain_exactly_matches_production():
    diffusion = build_diffusion({})
    model = ZeroModel()
    image = torch.full((1, 1, 8, 8), 0.2)
    global_state = torch.get_rng_state().clone()
    audited = controlled_chain(
        model, image, diffusion=diffusion, seed=17, steps=10, variance="large"
    )
    with _fixed_rng(17, image.device):
        reference = diffusion.reconstruct(model, image, steps=10)
    assert torch.equal(audited, reference)
    assert torch.equal(global_state, torch.get_rng_state())


def test_variance_controls_share_initial_noisy_input():
    seen = []

    class Recorder(ZeroModel):
        def forward(self, x, t):
            if int(t.item()) == 9:
                seen.append(x.clone())
            return super().forward(x, t)

    for mode in ("large", "small", "mean"):
        controlled_chain(
            Recorder(),
            torch.zeros(1, 1, 8, 8),
            diffusion=build_diffusion({}),
            seed=17,
            steps=10,
            variance=mode,
        )
    assert len(seen) == 3
    assert torch.equal(seen[0], seen[1]) and torch.equal(seen[1], seen[2])


def test_oracle_noise_inversion_and_posterior_mean_equations():
    diffusion = build_diffusion({})
    torch.manual_seed(17)
    clean = torch.rand(1, 1, 8, 8) * 2 - 1
    noise = torch.randn_like(clean)
    for k in (0, 99, 299, 999):
        t = torch.tensor([k])
        noisy = diffusion.q_sample(clean, t, noise)
        recovered = diffusion._predict_clean(noisy, t, noise)
        assert torch.allclose(recovered, clean, atol=3e-5)
