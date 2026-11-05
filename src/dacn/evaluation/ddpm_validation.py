"""Validation-only evaluation for partial-DDPM checkpoints."""

from __future__ import annotations

import os
import time
from collections.abc import Callable
from pathlib import Path

import numpy as np
import torch

from dacn.data.shards import ShardDataset
from dacn.evaluation.handoff import write_pairs
from dacn.evaluation.validation import evaluate_reconstructions
from dacn.training.ddpm_partial import (
    build_diffusion,
    derive_sampling_seed,
    load_ema_checkpoint,
    partial_reconstruct,
    posterior_mean_reconstruct_batch,
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
    reverse_sampler: str = "ancestral",
    batch_size: int = 1,
    reconstruction_cache_dir: Path | None = None,
    log_every_samples: int = 10,
    artifact_dir: Path | None = None,
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
    if reverse_sampler not in {"ancestral", "posterior_mean"}:
        raise ValueError("reverse_sampler must be ancestral or posterior_mean")
    if isinstance(batch_size, bool) or not isinstance(batch_size, int):
        raise TypeError("batch_size must be an integer")
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    if reverse_sampler == "ancestral" and batch_size != 1:
        raise ValueError("ancestral evaluation requires batch_size=1 for per-sample RNG isolation")
    if reconstruction_cache_dir is not None:
        reconstruction_cache_dir = Path(reconstruction_cache_dir)
        reconstruction_cache_dir.mkdir(parents=True, exist_ok=True)

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
        for start in range(0, len(indices), batch_size):
            batch_indices = indices[start : start + batch_size]
            image_batch = torch.stack([validation_data[index][0] for index in batch_indices]).to(
                device,
                non_blocking=True,
            )
            sample_seeds = [
                derive_sampling_seed(sampling_base_seed, validation_data.sample_ids[index])
                for index in batch_indices
            ]
            if reverse_sampler == "posterior_mean":
                reconstruction = posterior_mean_reconstruct_batch(
                    model,
                    image_batch,
                    starting_timestep=timestep,
                    seeds=sample_seeds,
                    diffusion=diffusion,
                )
            else:
                reconstruction = partial_reconstruct(
                    model,
                    image_batch,
                    starting_timestep=timestep,
                    seed=sample_seeds[0],
                    diffusion=diffusion,
                )
            stop = start + len(batch_indices)
            reconstructions[start:stop] = reconstruction[:, 0].cpu().numpy()
            if stop // log_every_samples > start // log_every_samples or stop == len(indices):
                print(
                    f"checkpoint_epoch={epoch} t={timestep} samples={stop}/{len(indices)}",
                    flush=True,
                )
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        artifact_name = None
        if artifact_dir is not None:
            artifact_name = f"{checkpoint_path.stem}_t{timestep:03d}_pairs.npz"
            write_pairs(
                Path(artifact_dir) / artifact_name,
                [validation_data.sample_ids[index] for index in indices],
                reconstructions,
            )
        metrics = evaluate_reconstructions(
            images,
            reconstructions,
            masks,
            labels,
            split="validation",
        )
        elapsed = time.perf_counter() - started
        cache_name = None
        cache_seconds = 0.0
        if reconstruction_cache_dir is not None:
            cache_started = time.perf_counter()
            cache_name = (
                f"{checkpoint_path.stem}_t{timestep:04d}_{reverse_sampler}_reconstructions.npz"
            )
            cache_path = reconstruction_cache_dir / cache_name
            temporary = cache_path.with_suffix(cache_path.suffix + ".tmp")
            with temporary.open("wb") as stream:
                np.savez_compressed(
                    stream,
                    images=images,
                    reconstructions=reconstructions,
                    masks=masks,
                    labels=labels,
                )
            os.replace(temporary, cache_path)
            cache_seconds = time.perf_counter() - cache_started
        result = {
            "starting_timestep": timestep,
            "reverse_sampler": reverse_sampler,
            "batch_size": batch_size,
            "validation_samples": len(indices),
            "normal_samples": int((labels == 0).sum()),
            "anomaly_samples": int((labels == 1).sum()),
            "sampling_base_seed": sampling_base_seed,
            "timing_scope": "reconstruction, transfer, private artifact write and scoring",
            "seconds": elapsed,
            "seconds_per_sample": elapsed / len(indices),
            "reconstruction_cache": cache_name,
            "reconstruction_cache_seconds": cache_seconds,
            "peak_vram_bytes": (
                int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0
            ),
            "metrics": metrics,
            "private_reconstruction_artifact": artifact_name,
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
