"""Matched convolutional AE/VAE backbones with a global256-dimensional bottleneck."""

import torch
from torch import nn


class ConvAutoencoder(nn.Module):
    """Six stride2levels; no skip connections around the latent vector.

    Common encoder/decoder are initialized before heads so the same seed gives
    the same shared initial weights in AE and VAE. Only VAE adds a logvar head.
    """

    def __init__(self, model_type="conv_ae", latent_dim=256, width=32):
        super().__init__()
        if model_type not in {"conv_ae", "conv_vae"}:
            raise ValueError("Require conv_ae or conv_vae")
        if type(latent_dim) is not int or latent_dim != 256:
            raise ValueError("Registered global latent dimension is256")
        if type(width) is not int or width < 8 or width % 8:
            raise ValueError("Width must be a positive multiple of8")
        self.model_type, self.latent_dim = model_type, latent_dim
        channels = [width, 2 * width, 4 * width, 8 * width, 8 * width, 8 * width]
        self.channels = channels
        blocks = []
        previous = 1
        for current in channels:
            blocks.extend(
                [
                    nn.Conv2d(previous, current, 4, stride=2, padding=1, bias=False),
                    nn.GroupNorm(8, current),
                    nn.SiLU(),
                ]
            )
            previous = current
        self.encoder = nn.Sequential(*blocks)
        self.flat_dim = channels[-1] * 4 * 4
        self.decoder_input = nn.Linear(latent_dim, self.flat_dim)
        blocks = []
        decoded_channels = list(reversed(channels[:-1])) + [1]
        previous = channels[-1]
        for current in decoded_channels:
            blocks.append(
                nn.ConvTranspose2d(previous, current, 4, stride=2, padding=1, bias=current == 1)
            )
            if current == 1:
                blocks.append(nn.Tanh())
            else:
                blocks.extend([nn.GroupNorm(8, current), nn.SiLU()])
            previous = current
        self.decoder = nn.Sequential(*blocks)
        self.mean = nn.Linear(self.flat_dim, latent_dim)
        self.logvar = nn.Linear(self.flat_dim, latent_dim) if model_type == "conv_vae" else None

    def forward(self, images, *, sample=None):
        if images.ndim != 4 or images.shape[1:] != (1, 256, 256):
            raise ValueError("Expected grayscale batch(B,1,256,256)")
        encoded = self.encoder(images).flatten(1)
        mean = self.mean(encoded)
        logvar = (
            self.logvar(encoded).clamp(-20, 10)
            if self.logvar is not None
            else torch.zeros_like(mean)
        )
        stochastic = self.training if sample is None else sample
        if self.model_type == "conv_vae" and stochastic:
            latent = mean + torch.exp(0.5 * logvar) * torch.randn_like(mean)
        else:
            latent = mean
        decoded = self.decoder_input(latent).reshape(-1, self.channels[-1], 4, 4)
        return self.decoder(decoded), mean, logvar


def loss_terms(reconstruction, target, mean, logvar, *, model_type, beta=1.0):
    """Crop240 MSE plus beta*KL per crop pixel; KL sums256latent dimensions."""
    if model_type not in {"conv_ae", "conv_vae"} or reconstruction.shape != target.shape:
        raise ValueError("Invalid autoencoder loss inputs")
    residual = reconstruction[:, :, 8:-8, 8:-8] - target[:, :, 8:-8, 8:-8]
    mse = residual.square().mean()
    kl = (
        -0.5 * (1 + logvar - mean.square() - logvar.exp()).sum(dim=1).mean()
        if model_type == "conv_vae"
        else mse.new_zeros(())
    )
    objective = mse + beta * kl / (240 * 240)
    return objective, mse, kl
