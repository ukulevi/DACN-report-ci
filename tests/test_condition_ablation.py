import numpy as np
import pytest
import torch
from torch import nn

from dacn.evaluation.condition_ablation import (
    ConditionIntervention,
    compare_reconstructions,
    donor_index,
)
from dacn.training.ddpm_partial import build_diffusion
from dacn.training.feature_condition import feature_reconstruct


class TinyConditioned(nn.Module):
    def __init__(self):
        super().__init__()
        self.condition_projection = nn.Linear(1, 1)
        with torch.no_grad():
            self.condition_projection.weight.fill_(0.02)
            self.condition_projection.bias.fill_(0.05)
        self.encoded_inputs = []
        self.noisy_inputs = []

    def encode(self, image):
        self.encoded_inputs.append(image.clone())
        return image.mean((2, 3))

    def unet(self, noisy, timesteps, condition_embedding=None):
        self.noisy_inputs.append(noisy.clone())
        return noisy * 0.01 + (
            0 if condition_embedding is None else condition_embedding[:, :, None, None]
        )

    def forward(self, noisy, timesteps, condition):
        return self.unet(noisy, timesteps, self.condition_projection(condition))


def test_zero_intervention_bypasses_learned_bias_and_never_encodes():
    model = TinyConditioned()
    x = torch.zeros(1, 1, 32, 32)
    zero = ConditionIntervention(model, "zero")
    assert not zero.encode(x).any()
    assert not zero(x, torch.tensor([299]), zero.encode(x)).any()
    assert not model.encoded_inputs
    assert model(x, torch.tensor([299]), model.encode(x)).mean() > 0


def test_donor_changes_encoder_input_only_and_reverse_noise_is_paired():
    model, diffusion = TinyConditioned(), build_diffusion({})
    image, donor = torch.full((1, 1, 32, 32), 0.3), torch.full((1, 1, 32, 32), -0.2)
    before = torch.random.get_rng_state().clone()
    outputs, noisy_tensors, first_reverse_inputs = [], [], []
    for selected in (
        model,
        ConditionIntervention(model, "zero"),
        ConditionIntervention(model, "normal_donor", donor),
    ):
        model.noisy_inputs.clear()
        output, noisy = feature_reconstruct(
            selected, diffusion, image, steps=3, seed=17, return_noisy=True
        )
        outputs.append(output)
        noisy_tensors.append(noisy)
        first_reverse_inputs.append(model.noisy_inputs[0])
    assert torch.equal(before, torch.random.get_rng_state())
    assert len(model.encoded_inputs) == 2
    assert torch.equal(model.encoded_inputs[0], image) and torch.equal(
        model.encoded_inputs[1], donor
    )
    assert all(torch.equal(noisy_tensors[0], x) for x in noisy_tensors)
    assert all(torch.equal(first_reverse_inputs[0], x) for x in first_reverse_inputs)
    assert not torch.equal(outputs[0], outputs[1]) and not torch.equal(outputs[0], outputs[2])
    with pytest.raises(TypeError):
        ConditionIntervention(model, "normal_donor", donor, mask=image)


def test_donor_mapping_is_fixed_and_reconstruction_delta_uses_original_crop():
    identifier = "00000000-0000-0000-0000-000000000017"
    assert donor_index(identifier, 7500) == donor_index(identifier, 7500)
    assert 0 <= donor_index(identifier, 7500) < 7500
    with pytest.raises(ValueError):
        donor_index(identifier, 0)
    reference = np.zeros((2, 256, 256))
    observed = np.full_like(reference, 0.2)
    observed[:, :8] = 500
    result = compare_reconstructions(reference, observed, np.array([0, 1]))
    assert result["mean_absolute_change_01"] == pytest.approx(0.1)
    assert result["normal_mean_absolute_change_01"] == pytest.approx(0.1)
    with pytest.raises(ValueError):
        compare_reconstructions(reference, observed[:1], np.array([0, 1]))
