"""Validation-only evaluation for partial-DDPM checkpoints."""

from __future__ import annotations

import time
from collections.abc import Callable
from pathlib import Path

import numpy as np
import torch

from dacn.data.shards import ShardDataset
from dacn.evaluation.validation import evaluate_reconstructions
from dacn.training.ddpm_partial import (
    build_diffusion,
    derive_sampling_seed,
    load_ema_checkpoint,
    partial_reconstruct,
)


def select_validation_indices(labels: np.ndarray, limit: int | None, *, seed: int) -> list[int]:
    """Select a deterministic, class-stratified validation subset."""
    labels = np.asarray(labels)
    if labels.ndim != 1 or not len(labels) or set(np.unique(labels)) != {0, 1}:
        raise ValueError("validation labels must be a nonempty binary two-class vector")
    if limit is None:
        return list(range(len(labels)))
    if isinstance(limit, bool) or not isinstance(limit, int) or not 2 <= limit <= len(labels):
        raise ValueError("validation sample limit must be between 2 and the split size")

    rng = np.random.default_rng(seed)
    selected: list[int] = []
    base, remainder = divmod(limit, 2)
    for class_index, label in enumerate((0, 1)):
        candidates = np.flatnonzero(labels == label)
        requested = base + int(class_index < remainder)
        count = min(requested, len(candidates))
        selected.extend(rng.choice(candidates, size=count, replace=False).tolist())
    if len(selected) < limit:
        remaining = np.setdiff1d(np.arange(len(labels)), np.asarray(selected), assume_unique=True)
        selected.extend(rng.choice(remaining, size=limit - len(selected), replace=False).tolist())
    return sorted(selected)


def evaluate_checkpoint(
    *,
    validation_dir: Path,
    checkpoint_path: Path,
    starting_timesteps: list[int],
    device: torch.device,
    max_validation_samples: int | None = None,
    sampling_base_seed: int = 17,
    log_every_samples: int = 10,
    on_timestep_complete: Callable[[dict], None] | None = None,
) -> dict:
    """Evaluate one checkpoint without exposing validation sample identifiers."""
    if not starting_timesteps or len(set(starting_timesteps)) != len(starting_timesteps):
        raise ValueError("starting_timesteps must be a nonempty unique list")
    if any(
        isinstance(timestep, bool) or not isinstance(timestep, int) or not 1 <= timestep <= 1000
        for timestep in starting_timesteps
    ):
        raise ValueError("starting_timesteps must contain integers in [1, 1000]")
    if (
        isinstance(sampling_base_seed, bool)
        or not isinstance(sampling_base_seed, int)
        or not 0 <= sampling_base_seed < 2**32
    ):
        raise ValueError("sampling_base_seed must be an integer in [0, 2**32)")
    if isinstance(log_every_samples, bool) or not isinstance(log_every_samples, int):
        raise TypeError("log_every_samples must be an integer")
    if log_every_samples < 1:
        raise ValueError("log_every_samples must be positive")

    checkpoint_path = Path(checkpoint_path)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    config = checkpoint.get("config")
    epoch = checkpoint.get("epoch")
    if not isinstance(config, dict):
        raise TypeError("checkpoint resolved config must be a dictionary")
    if isinstance(epoch, bool) or not isinstance(epoch, int):
        raise TypeError("checkpoint epoch must be an integer")

    validation_data = ShardDataset(validation_dir, "val", include_masks=True)
    indices = select_validation_indices(
        validation_data.labels,
        max_validation_samples,
        seed=sampling_base_seed,
    )
    images = np.empty((len(indices), 256, 256), dtype=np.float32)
    masks = np.empty((len(indices), 256, 256), dtype=np.uint8)
    labels = np.empty((len(indices),), dtype=np.uint8)
    for position, index in enumerate(indices):
        image, mask, label = validation_data[index]
        images[position] = image[0].numpy()
        masks[position] = mask.numpy()
        labels[position] = int(label)

    model = load_ema_checkpoint(checkpoint_path, config, device)
    diffusion = build_diffusion(config)
    results = {}
    for timestep in starting_timesteps:
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
            torch.cuda.synchronize(device)
        started = time.perf_counter()
        reconstructions = np.empty_like(images)
        for position, index in enumerate(indices, start=1):
            image, _, _ = validation_data[index]
            image_batch = image.unsqueeze(0).to(device, non_blocking=True)
            sample_seed = derive_sampling_seed(
                sampling_base_seed,
                validation_data.sample_ids[index],
            )
            reconstruction = partial_reconstruct(
                model,
                image_batch,
                starting_timestep=timestep,
                seed=sample_seed,
                diffusion=diffusion,
            )
            reconstructions[position - 1] = reconstruction[0, 0].cpu().numpy()
            if position % log_every_samples == 0 or position == len(indices):
                print(
                    f"checkpoint_epoch={epoch} t={timestep} samples={position}/{len(indices)}",
                    flush=True,
                )
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        metrics = evaluate_reconstructions(
            images,
            reconstructions,
            masks,
            labels,
            split="validation",
        )
        elapsed = time.perf_counter() - started
        result = {
            "starting_timestep": timestep,
            "validation_samples": len(indices),
            "normal_samples": int((labels == 0).sum()),
            "anomaly_samples": int((labels == 1).sum()),
            "sampling_base_seed": sampling_base_seed,
            "seconds": elapsed,
            "seconds_per_sample": elapsed / len(indices),
            "peak_vram_bytes": (
                int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0
            ),
            "metrics": metrics,
        }
        results[str(timestep)] = result
        if on_timestep_complete is not None:
            on_timestep_complete(result)

    return {
        "checkpoint": checkpoint_path.name,
        "checkpoint_epoch": epoch,
        "validation_samples": len(indices),
        "timesteps": results,
    }
