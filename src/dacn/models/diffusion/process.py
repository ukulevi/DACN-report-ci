"""AnoDDPM-style diffusion training and reconstruction for BMAD FLAIR.

The equations and simplex-noise strategy are adapted from Julian Wyatt's
AnoDDPM implementation (MIT). This module intentionally excludes its
dataset-specific detection scripts, plotting, and hard-coded paths.
"""

import math

import numpy as np
import torch

from dacn.models.diffusion.simplex import SimplexNoise


def get_beta_schedule(timesteps: int, name: str = "linear") -> np.ndarray:
    """Return the fixed DDPM beta schedule used by AnoDDPM."""
    if isinstance(timesteps, bool) or not isinstance(timesteps, int) or timesteps < 2:
        raise ValueError("timesteps must be an integer greater than one")
    if name == "linear":
        scale = 1000 / timesteps
        betas = np.linspace(scale * 0.0001, scale * 0.02, timesteps, dtype=np.float64)
    elif name == "cosine":
        maximum = 0.999

        def alpha_bar(value: float) -> float:
            return math.cos((value + 0.008) / 1.008 * math.pi / 2) ** 2

        betas = np.asarray(
            [
                min(
                    1 - alpha_bar((index + 1) / timesteps) / alpha_bar(index / timesteps),
                    maximum,
                )
                for index in range(timesteps)
            ],
            dtype=np.float64,
        )
    else:
        raise ValueError(f"Unsupported beta schedule: {name}")
    if not np.isfinite(betas).all() or np.any(betas <= 0) or np.any(betas >= 1):
        raise ValueError("Beta schedule must stay strictly between zero and one")
    return betas


def _extract(values: np.ndarray, timesteps: torch.Tensor, shape: torch.Size) -> torch.Tensor:
    result = torch.as_tensor(values, device=timesteps.device, dtype=torch.float32)[timesteps]
    while result.ndim < len(shape):
        result = result[..., None]
    return result.expand(shape)


def _mean_flat(value: torch.Tensor) -> torch.Tensor:
    return value.mean(dim=tuple(range(1, value.ndim)))


class GaussianDiffusion:
    """Fixed-variance DDPM with Gaussian or AnoDDPM simplex forward noise."""

    def __init__(
        self,
        *,
        timesteps: int = 1000,
        beta_schedule: str = "linear",
        noise_type: str = "simplex",
        loss_type: str = "l2",
        simplex_octaves: int = 6,
        simplex_persistence: float = 0.8,
        simplex_frequency: float = 64,
    ):
        if noise_type not in {"gaussian", "simplex"}:
            raise ValueError("noise_type must be 'gaussian' or 'simplex'")
        if loss_type not in {"l1", "l2"}:
            raise ValueError("loss_type must be 'l1' or 'l2'")
        if simplex_octaves < 1 or simplex_frequency <= 0:
            raise ValueError("Simplex octave and frequency settings must be positive")
        if not 0 < simplex_persistence <= 1:
            raise ValueError("simplex_persistence must be in (0, 1]")

        betas = get_beta_schedule(timesteps, beta_schedule)
        alphas = 1.0 - betas
        alphas_cumprod = np.cumprod(alphas)
        alphas_cumprod_previous = np.append(1.0, alphas_cumprod[:-1])

        self.timesteps = timesteps
        self.noise_type = noise_type
        self.loss_type = loss_type
        self.simplex_octaves = simplex_octaves
        self.simplex_persistence = simplex_persistence
        self.simplex_frequency = simplex_frequency
        self.betas = betas
        self.alphas_cumprod = alphas_cumprod
        self.sqrt_alphas_cumprod = np.sqrt(alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = np.sqrt(1.0 - alphas_cumprod)
        self.sqrt_recip_alphas_cumprod = np.sqrt(1.0 / alphas_cumprod)
        self.sqrt_recipm1_alphas_cumprod = np.sqrt(1.0 / alphas_cumprod - 1)
        self.posterior_variance = betas * (1.0 - alphas_cumprod_previous) / (1.0 - alphas_cumprod)
        # AnoDDPM follows guided-diffusion's fixed-large model variance: use the
        # posterior value only at t=0, then beta_t for the rest of the chain.
        self.model_log_variance = np.log(np.append(self.posterior_variance[1], betas[1:]))
        self.posterior_mean_coef1 = (
            betas * np.sqrt(alphas_cumprod_previous) / (1.0 - alphas_cumprod)
        )
        self.posterior_mean_coef2 = (
            (1.0 - alphas_cumprod_previous) * np.sqrt(alphas) / (1.0 - alphas_cumprod)
        )

    def noise(self, inputs: torch.Tensor, timesteps: torch.Tensor) -> torch.Tensor:
        """Generate training/sampling noise with the same shape as ``inputs``."""
        if self.noise_type == "gaussian":
            return torch.randn_like(inputs)
        if inputs.ndim != 4:
            raise ValueError("Simplex noise expects NCHW inputs")
        timestep_values = timesteps.detach().cpu().numpy().astype(np.float64)
        result = torch.empty_like(inputs)
        for channel in range(inputs.shape[1]):
            simplex = SimplexNoise()
            values = simplex.rand_3d_fixed_T_octaves(
                inputs.shape[-2:],
                timestep_values,
                octaves=self.simplex_octaves,
                persistence=self.simplex_persistence,
                frequency=self.simplex_frequency,
            )
            result[:, channel] = torch.from_numpy(values).to(
                device=inputs.device, dtype=inputs.dtype
            )
        return result

    def q_sample(
        self, clean: torch.Tensor, timesteps: torch.Tensor, noise: torch.Tensor
    ) -> torch.Tensor:
        return (
            _extract(self.sqrt_alphas_cumprod, timesteps, clean.shape) * clean
            + _extract(self.sqrt_one_minus_alphas_cumprod, timesteps, clean.shape) * noise
        )

    def training_loss(
        self,
        model: torch.nn.Module,
        clean: torch.Tensor,
        *,
        maximum_timestep: int | None = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Train the U-Net to predict the sampled noise at one random timestep."""
        upper = self.timesteps if maximum_timestep is None else maximum_timestep
        if not 1 <= upper <= self.timesteps:
            raise ValueError("maximum_timestep must be between one and timesteps")
        timesteps = torch.randint(0, upper, (len(clean),), device=clean.device)
        target_noise = self.noise(clean, timesteps).float()
        noisy = self.q_sample(clean, timesteps, target_noise)
        predicted_noise = model(noisy, timesteps)
        if predicted_noise.shape != target_noise.shape:
            raise ValueError("U-Net output shape must match the input noise shape")
        difference = predicted_noise - target_noise
        per_sample = _mean_flat(difference.abs() if self.loss_type == "l1" else difference.square())
        return per_sample.mean(), {
            "timesteps": timesteps.detach(),
            "noisy": noisy.detach(),
            "predicted_noise": predicted_noise.detach(),
        }

    def _predict_clean(
        self, noisy: torch.Tensor, timesteps: torch.Tensor, predicted_noise: torch.Tensor
    ) -> torch.Tensor:
        return (
            _extract(self.sqrt_recip_alphas_cumprod, timesteps, noisy.shape) * noisy
            - _extract(self.sqrt_recipm1_alphas_cumprod, timesteps, noisy.shape) * predicted_noise
        ).clamp(-1, 1)

    def _p_mean_variance(
        self, model: torch.nn.Module, noisy: torch.Tensor, timesteps: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        predicted_noise = model(noisy, timesteps)
        predicted_clean = self._predict_clean(noisy, timesteps, predicted_noise)
        mean = (
            _extract(self.posterior_mean_coef1, timesteps, noisy.shape) * predicted_clean
            + _extract(self.posterior_mean_coef2, timesteps, noisy.shape) * noisy
        )
        log_variance = _extract(self.model_log_variance, timesteps, noisy.shape)
        return mean, log_variance

    def p_sample(
        self, model: torch.nn.Module, noisy: torch.Tensor, timesteps: torch.Tensor
    ) -> torch.Tensor:
        mean, log_variance = self._p_mean_variance(model, noisy, timesteps)
        # Upstream AnoDDPM diffuses the input with its configured training noise,
        # but its default reverse chain samples Gaussian noise.
        sampled_noise = torch.randn_like(noisy)
        nonzero = (timesteps != 0).float().view(-1, 1, 1, 1)
        return mean + nonzero * torch.exp(0.5 * log_variance) * sampled_noise

    def p_mean_sample(
        self, model: torch.nn.Module, noisy: torch.Tensor, timesteps: torch.Tensor
    ) -> torch.Tensor:
        """Take the deterministic posterior-mean reverse transition."""
        mean, _ = self._p_mean_variance(model, noisy, timesteps)
        return mean

    @torch.inference_mode()
    def reconstruct(
        self,
        model: torch.nn.Module,
        clean: torch.Tensor,
        *,
        steps: int,
        initial_noise: torch.Tensor | None = None,
        reverse_sampler: str = "ancestral",
    ) -> torch.Tensor:
        """Diffuse to ``steps - 1`` and run the reverse chain back to zero."""
        if not 1 <= steps <= self.timesteps:
            raise ValueError("steps must be between one and timesteps")
        if reverse_sampler not in {"ancestral", "posterior_mean"}:
            raise ValueError("reverse_sampler must be ancestral or posterior_mean")
        start = torch.full((len(clean),), steps - 1, device=clean.device, dtype=torch.long)
        if initial_noise is None:
            initial_noise = self.noise(clean, start)
        elif (
            initial_noise.shape != clean.shape
            or initial_noise.device != clean.device
            or initial_noise.dtype != clean.dtype
        ):
            raise ValueError("initial_noise must match clean shape, device and dtype")
        value = self.q_sample(clean, start, initial_noise)
        for timestep in range(steps - 1, -1, -1):
            batch_timesteps = torch.full(
                (len(clean),), timestep, device=clean.device, dtype=torch.long
            )
            if reverse_sampler == "ancestral":
                value = self.p_sample(model, value, batch_timesteps)
            else:
                value = self.p_mean_sample(model, value, batch_timesteps)
        return value.clamp(-1, 1)
