"""Eta-zero DDIM reverse jumps with matching forward gaps for masked resampling."""

from itertools import pairwise

import numpy as np
import torch

from dacn.evaluation.sampler_ablation import DDIM_INDICES, deterministic_from_noisy
from dacn.training.ddpm_partial import _fixed_rng

INPAINT_INDICES = tuple(range(49, -1, -5))


def ddim_indices(starting_timestep, inference_steps):
    """Exactly N model indices, inclusive start/index0; terminal clean is not a call."""
    if (
        any(
            isinstance(v, bool) or not isinstance(v, int)
            for v in (starting_timestep, inference_steps)
        )
        or not 1 <= inference_steps <= starting_timestep
    ):
        raise ValueError("Require integer1 <= inference_steps <= starting_timestep")
    return tuple(int(v) for v in np.rint(np.linspace(starting_timestep - 1, 0, inference_steps)))


def _alpha(diffusion, index):
    if (
        isinstance(index, bool)
        or not isinstance(index, int)
        or not -1 <= index < diffusion.timesteps
    ):
        raise ValueError("Invalid diffusion index")
    return 1.0 if index == -1 else float(diffusion.alphas_cumprod[index])


@torch.inference_mode()
def ddim_step(model, diffusion, value, current, following):
    """One model call for arbitrary decreasing indices; -1 denotes clean alpha=1."""
    if diffusion.noise_type != "gaussian" or not current > following >= -1:
        raise ValueError("DDIM requires Gaussian decreasing indices")
    alpha, next_alpha = _alpha(diffusion, current), _alpha(diffusion, following)
    t = torch.full((len(value),), current, device=value.device, dtype=torch.long)
    clean = diffusion._predict_clean(value, t, model(value, t))
    if following == -1:
        return clean
    # Epsilon is re-derived after x0 clipping, consistent with the existing DDIM control.
    epsilon = (value - alpha**0.5 * clean) / (1 - alpha) ** 0.5
    return next_alpha**0.5 * clean + (1 - next_alpha) ** 0.5 * epsilon


def forward_gap(diffusion, value, lower, upper, noise):
    """q(x_upper|x_lower), including clean lower=-1; never substitute beta[upper]."""
    if not upper > lower >= -1 or noise.shape != value.shape:
        raise ValueError("Invalid forward gap")
    ratio = _alpha(diffusion, upper) / _alpha(diffusion, lower)
    return ratio**0.5 * value + (1 - ratio) ** 0.5 * noise


@torch.inference_mode()
def ddim_initial_reconstruction(model, diffusion, image, *, seed, indices=DDIM_INDICES):
    if diffusion.noise_type != "gaussian":
        raise ValueError("Gaussian DDIM required")
    with _fixed_rng(seed, image.device):
        t = torch.full((len(image),), indices[0], device=image.device, dtype=torch.long)
        noisy = diffusion.q_sample(image, t, diffusion.noise(image, t))
    return deterministic_from_noisy(model, diffusion, noisy, indices=indices)


@torch.inference_mode()
def ddim_stitch_resample(
    model, image, initial, mask, *, diffusion, seed, indices=INPAINT_INDICES, repeats=5
):
    """Mask=True is unknown. Reverse eta=0, with stochastic context and repeat jumps."""
    if image.shape != (1, 1, 256, 256) or initial.shape != image.shape:
        raise ValueError("image/initial must match grayscale256 batch1")
    if mask.shape != image.shape or mask.dtype != torch.bool or mask.device != image.device:
        raise ValueError("mask must match image as boolean tensor")
    if (
        not indices
        or any(
            isinstance(k, bool) or not isinstance(k, int) or not 0 <= k < diffusion.timesteps
            for k in indices
        )
        or any(a <= b for a, b in pairwise(indices))
    ):
        raise ValueError("Indices must strictly decrease in schedule")
    if (
        diffusion.noise_type != "gaussian"
        or isinstance(repeats, bool)
        or not isinstance(repeats, int)
        or repeats < 1
    ):
        raise ValueError("Gaussian diffusion and positive repeat count required")
    if any(not torch.isfinite(v).all() or (v.abs() > 1).any() for v in (image, initial)):
        raise ValueError("Expected finite clean images in[-1,1]")
    model.eval()
    with _fixed_rng(seed, image.device):
        stitched = torch.where(mask, initial, image)
        t = torch.full((1,), indices[0], device=image.device, dtype=torch.long)
        value = diffusion.q_sample(stitched, t, torch.randn_like(image))
        for position, current in enumerate(indices):
            following = indices[position + 1] if position + 1 < len(indices) else -1
            for repeat in range(repeats):
                unknown = ddim_step(model, diffusion, value, current, following)
                if following == -1:
                    context = image
                else:
                    t.fill_(following)
                    context = diffusion.q_sample(image, t, torch.randn_like(image))
                value = torch.where(mask, unknown, context)
                if repeat + 1 < repeats:
                    value = forward_gap(
                        diffusion, value, following, current, torch.randn_like(value)
                    )
    return value.clamp(-1, 1)
