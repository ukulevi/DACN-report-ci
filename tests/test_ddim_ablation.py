import json
from itertools import pairwise
from pathlib import Path

import numpy as np
import pytest
import torch

from dacn.evaluation.ddim_ablation import (
    DEFAULT_CONFIG,
    ablation_pair_scores,
    four_ablation_scores,
    resolve_ablation_config,
)
from dacn.evaluation.ddim_inpainting import (
    ddim_indices,
    ddim_initial_reconstruction,
    ddim_stitch_resample,
)
from dacn.training.ddpm_partial import build_diffusion


def test_explicit_indices_budget_and_clean_endpoint_without_extra_call():
    for start, steps in ((300, 20), (50, 30)):
        values = ddim_indices(start, steps)
        assert len(values) == steps and values[0] == start - 1 and values[-1] == 0
        assert len(set(values)) == steps and all(a > b for a, b in pairwise(values))
    for start, steps in ((50, 51), (300, 0), (50, True), (50.0, 30)):
        with pytest.raises(ValueError):
            ddim_indices(start, steps)


def test_actual_twenty_plus_thirty_unet_calls_and_context_preservation():
    calls = []

    class Model(torch.nn.Module):
        def forward(self, image, timestep):
            calls.append(int(timestep.item()))
            return torch.zeros_like(image)

    model, diffusion = Model(), build_diffusion({})
    image = torch.full((1, 1, 256, 256), 0.2)
    initial = ddim_initial_reconstruction(
        model, diffusion, image, seed=17, indices=ddim_indices(300, 20)
    )
    assert len(calls) == 20
    mask = torch.zeros_like(image, dtype=torch.bool)
    mask[:, :, 100:120, 100:120] = True
    final = ddim_stitch_resample(
        model,
        image,
        initial,
        mask,
        diffusion=diffusion,
        seed=17,
        indices=ddim_indices(50, 30),
        repeats=1,
    )
    assert len(calls) == 50 and calls[:20] == list(ddim_indices(300, 20))
    assert calls[20:] == list(ddim_indices(50, 30))
    assert torch.equal(final[~mask], image[~mask]) and torch.isfinite(final).all()


def test_four_score_arms_use_same_final_and_actual_final_lpips():
    initial = {"l1": np.array([3.0]), "lpips": np.array([7.0]), "product": np.array([21.0])}
    final = {"l1": np.array([2.0]), "lpips": np.array([5.0]), "product": np.array([10.0])}
    values = four_ablation_scores(initial, final)
    assert values["ddpm_l1"] is initial["l1"]
    assert values["autoddpm_l1"] is final["l1"]
    assert values["autoddpm_lpips"] is final["lpips"]
    assert values["autoddpm_lpips_residual"] is final["product"]


def test_real_pair_score_calls_lpips_on_the_supplied_final_reconstruction():
    seen = []

    class Perceptual(torch.nn.Module):
        def forward(self, x, y, normalize):
            assert x.shape == y.shape == (1, 3, 240, 240) and not normalize
            seen.append(float(y.mean()))
            return (x - y).abs().mean(1, keepdim=True)

    image = np.zeros((256, 256), dtype=np.float32)
    initial, _ = ablation_pair_scores(Perceptual(), image, image + 0.8, torch.device("cpu"))
    final, _ = ablation_pair_scores(Perceptual(), image, image + 0.2, torch.device("cpu"))
    scores = four_ablation_scores(initial, final)
    assert seen == pytest.approx([0.8, 0.2])
    assert np.allclose(scores["autoddpm_lpips"], 0.2)
    assert np.allclose(scores["autoddpm_lpips_residual"], 0.2, atol=1e-6)
    assert not np.allclose(scores["autoddpm_lpips_residual"], 0.16)


def test_fixed_json_config_has_one_pass_and_exact_fifty_call_preset():
    values = json.loads(
        (Path(__file__).resolve().parents[1] / "configs/autoddpm_ddim20_30.json").read_text()
    )
    assert resolve_ablation_config(values) == DEFAULT_CONFIG
    assert values["phase1_inference_steps"] + values["phase2_inference_steps"] == 50
    assert values["resampling_passes"] == 1
    for overrides in ({"resampling_passes": 5}, {"phase1_inference_steps": 10}, {"eta": False}):
        with pytest.raises(ValueError):
            resolve_ablation_config(overrides)
