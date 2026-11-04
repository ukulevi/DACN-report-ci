"""Normal-only DDPM training and partial reconstruction for BMAD Brain FLAIR."""

from __future__ import annotations

import hashlib
import json
import math
import os
import random
import time
import uuid
from collections.abc import Callable
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from dacn.data.shards import ShardBatchSampler, ShardDataset
from dacn.models.diffusion.process import GaussianDiffusion
from dacn.models.diffusion.unet import UNetModel, update_ema_params

DEFAULTS = {
    "seed": 17,
    "epochs": 1,
    "batch_size": 1,
    "validation_batch_size": 1,
    "learning_rate": 1e-4,
    "weight_decay": 0.0,
    "timesteps": 1000,
    "beta_schedule": "linear",
    "loss_type": "l2",
    "noise_type": "gaussian",
    "simplex_octaves": 6,
    "simplex_persistence": 0.8,
    "simplex_frequency": 64.0,
    "base_channels": 32,
    "channel_mults": (1, 1, 2, 2, 4, 4),
    "num_res_blocks": 2,
    "attention_resolutions": "16,8",
    "num_heads": 2,
    "num_head_channels": -1,
    "dropout": 0.0,
    "ema_decay": 0.9999,
    "gradient_clip_norm": 1.0,
    "log_every_steps": 25,
    "max_train_samples": 128,
    "max_validation_samples": 16,
    "validation_seed_offset": 1_000_000,
    "preview_starting_timestep": 100,
    "preview_samples": 4,
    "preview_seed_offset": 2_000_000,
    "checkpoint_epochs": (),
}

METHOD = "ddpm_partial_reconstruction_bmad_flair"


def _resolved_config(config: dict) -> dict:
    if not isinstance(config, dict):
        raise TypeError("config must be a JSON object")
    unknown = set(config) - set(DEFAULTS)
    if unknown:
        raise ValueError(f"Unknown partial DDPM config keys: {', '.join(sorted(unknown))}")
    result = {**DEFAULTS, **config}

    positive_integers = (
        "seed",
        "epochs",
        "batch_size",
        "validation_batch_size",
        "timesteps",
        "base_channels",
        "num_res_blocks",
        "num_heads",
        "log_every_steps",
        "validation_seed_offset",
        "preview_starting_timestep",
        "preview_samples",
        "preview_seed_offset",
        "simplex_octaves",
    )
    for key in positive_integers:
        value = result[key]
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"{key} must be a positive integer")
    for key in ("max_train_samples", "max_validation_samples"):
        value = result[key]
        if value is not None and (
            isinstance(value, bool) or not isinstance(value, int) or value < 1
        ):
            raise ValueError(f"{key} must be null or a positive integer")

    numeric = (
        "learning_rate",
        "weight_decay",
        "dropout",
        "ema_decay",
        "gradient_clip_norm",
        "simplex_persistence",
        "simplex_frequency",
    )
    for key in numeric:
        value = result[key]
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TypeError(f"{key} must be numeric")
        result[key] = float(value)
        if not math.isfinite(result[key]):
            raise ValueError(f"{key} must be finite")
    if result["learning_rate"] <= 0 or result["gradient_clip_norm"] <= 0:
        raise ValueError("learning_rate and gradient_clip_norm must be positive")
    if result["weight_decay"] < 0:
        raise ValueError("weight_decay must be non-negative")
    if not 0 <= result["dropout"] < 1:
        raise ValueError("dropout must be in [0, 1)")
    if not 0 < result["ema_decay"] < 1:
        raise ValueError("ema_decay must be in (0, 1)")
    if result["timesteps"] != 1000 or result["beta_schedule"] != "linear":
        raise ValueError("The reviewed-method candidate fixes a 1000-step linear beta schedule")
    if result["loss_type"] != "l2":
        raise ValueError("The reviewed-method candidate uses epsilon-prediction MSE")
    if result["noise_type"] not in {"gaussian", "simplex"}:
        raise ValueError("noise_type must be gaussian or simplex")
    if not 0 < result["simplex_persistence"] <= 1 or result["simplex_frequency"] <= 0:
        raise ValueError("Invalid simplex persistence or frequency")
    if result["base_channels"] % 32:
        raise ValueError("base_channels must be divisible by 32 for the U-Net GroupNorm layers")
    if result["preview_starting_timestep"] > result["timesteps"]:
        raise ValueError("preview_starting_timestep cannot exceed timesteps")
    head_channels = result["num_head_channels"]
    if isinstance(head_channels, bool) or not isinstance(head_channels, int):
        raise TypeError("num_head_channels must be an integer")
    if head_channels == 0 or head_channels < -1:
        raise ValueError("num_head_channels must be -1 or a positive integer")
    if not isinstance(result["attention_resolutions"], str):
        raise TypeError("attention_resolutions must be a comma-separated string")
    for value in result["attention_resolutions"].split(","):
        if not value.isdigit() or int(value) < 1 or 256 % int(value):
            raise ValueError("attention_resolutions must divide the 256-pixel image size")

    channel_mults = result["channel_mults"]
    if not isinstance(channel_mults, (tuple, list)) or not channel_mults:
        raise TypeError("channel_mults must be a non-empty list of positive integers")
    if any(
        isinstance(value, bool) or not isinstance(value, int) or value < 1
        for value in channel_mults
    ):
        raise ValueError("channel_mults must contain positive integers")
    if len(channel_mults) > 9:
        raise ValueError("channel_mults cannot downsample a 256-pixel image more than nine levels")
    result["channel_mults"] = tuple(channel_mults)
    if result["max_validation_samples"] is not None and result["max_validation_samples"] < 2:
        raise ValueError("max_validation_samples must include both validation classes")
    checkpoint_epochs = result["checkpoint_epochs"]
    if not isinstance(checkpoint_epochs, (tuple, list)):
        raise TypeError("checkpoint_epochs must be a list of epoch numbers")
    if any(
        isinstance(epoch, bool) or not isinstance(epoch, int) or epoch < 1
        for epoch in checkpoint_epochs
    ):
        raise ValueError("checkpoint_epochs must contain positive integers")
    if len(set(checkpoint_epochs)) != len(checkpoint_epochs):
        raise ValueError("checkpoint_epochs must not contain duplicates")
    if any(epoch > result["epochs"] for epoch in checkpoint_epochs):
        raise ValueError("checkpoint_epochs cannot exceed epochs")
    result["checkpoint_epochs"] = tuple(sorted(checkpoint_epochs))
    return result


def _select_indices(
    labels: np.ndarray,
    limit: int | None,
    *,
    seed: int,
    stratified: bool,
) -> list[int]:
    if limit is None:
        return list(range(len(labels)))
    if limit > len(labels):
        raise ValueError(f"Sample limit {limit} exceeds split size {len(labels)}")
    rng = np.random.default_rng(seed)
    if not stratified:
        return sorted(rng.choice(len(labels), size=limit, replace=False).tolist())

    classes = sorted(np.unique(labels).tolist())
    if limit < len(classes):
        raise ValueError("Stratified validation sample must include both classes")
    selected: list[int] = []
    base, remainder = divmod(limit, len(classes))
    for class_index, label in enumerate(classes):
        candidates = np.flatnonzero(labels == label)
        count = min(base + int(class_index < remainder), len(candidates))
        selected.extend(rng.choice(candidates, size=count, replace=False).tolist())
    if len(selected) < limit:
        remaining = np.setdiff1d(np.arange(len(labels)), np.asarray(selected), assume_unique=True)
        selected.extend(rng.choice(remaining, size=limit - len(selected), replace=False).tolist())
    return sorted(selected)


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@contextmanager
def _fixed_rng(seed: int, device: torch.device):
    numpy_state = np.random.get_state()
    devices = []
    if device.type == "cuda":
        devices = [device.index if device.index is not None else torch.cuda.current_device()]
    try:
        with torch.random.fork_rng(devices=devices):
            np.random.seed(seed)
            torch.manual_seed(seed)
            if device.type == "cuda":
                torch.cuda.manual_seed_all(seed)
            yield
    finally:
        np.random.set_state(numpy_state)


@torch.inference_mode()
def _validation_noise_mse(
    model: torch.nn.Module,
    diffusion: GaussianDiffusion,
    loader: DataLoader,
    device: torch.device,
    *,
    seed: int,
) -> tuple[float, int]:
    model.eval()
    total_loss = 0.0
    total_samples = 0
    with _fixed_rng(seed, device):
        for batch in loader:
            # ShardDataset returns (image, label) without masks and
            # (image, mask, label) with masks. Validation keeps masks enabled
            # for stratified sampling, but they are not part of this objective.
            images = batch[0]
            images = images.to(device, non_blocking=True)
            loss, _ = diffusion.training_loss(model, images)
            count = len(images)
            total_loss += float(loss) * count
            total_samples += count
    if not total_samples:
        raise ValueError("Normal validation subset is empty")
    return total_loss / total_samples, total_samples


def _write_json(path: Path, value: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def _rng_state() -> dict:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
    }


def _restore_rng_state(state: dict) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"].cpu())
    if torch.cuda.is_available() and state["cuda"]:
        torch.cuda.set_rng_state_all([cuda_state.cpu() for cuda_state in state["cuda"]])


def _save_checkpoint(path: Path, state: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(state, temporary)
    os.replace(temporary, path)


def _checkpoint_state(
    *,
    model: torch.nn.Module,
    ema: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    optimizer_steps: int,
    sampler_epoch: int,
    config: dict,
    history: list[dict],
    best_epoch: int,
    best_validation_noise_mse: float,
    total_training_seconds: float,
) -> dict:
    return {
        "schema_version": 1,
        "method": METHOD,
        "model": model.state_dict(),
        "ema": ema.state_dict(),
        "optimizer": optimizer.state_dict(),
        "epoch": epoch,
        "optimizer_steps": optimizer_steps,
        "sampler_epoch": sampler_epoch,
        "config": config,
        "history": history,
        "best_epoch": best_epoch,
        "best_validation_noise_mse": best_validation_noise_mse,
        "total_training_seconds": total_training_seconds,
        "rng_state": _rng_state(),
    }


def build_model(config: dict, device: torch.device | str = "cpu") -> UNetModel:
    """Create the configured one-channel BMAD U-Net."""
    config = _resolved_config(config)
    return UNetModel(
        256,
        config["base_channels"],
        channel_mults=config["channel_mults"],
        num_res_blocks=config["num_res_blocks"],
        dropout=config["dropout"],
        attention_resolutions=config["attention_resolutions"],
        n_heads=config["num_heads"],
        n_head_channels=config["num_head_channels"],
        in_channels=1,
    ).to(device)


def build_diffusion(config: dict) -> GaussianDiffusion:
    """Create the fixed schedule with the checkpoint's configured forward noise."""
    config = _resolved_config(config)
    return GaussianDiffusion(
        timesteps=config["timesteps"],
        beta_schedule=config["beta_schedule"],
        noise_type=config["noise_type"],
        loss_type=config["loss_type"],
        simplex_octaves=config["simplex_octaves"],
        simplex_persistence=config["simplex_persistence"],
        simplex_frequency=config["simplex_frequency"],
    )


def _validate_resume_config(
    checkpoint_config: dict,
    requested_config: dict,
    *,
    completed_epoch: int,
    allow_epoch_extension: bool,
) -> None:
    """Require an exact resume config unless only the run horizon is extended."""
    checkpoint_config = _resolved_config(checkpoint_config)
    requested_config = _resolved_config(requested_config)
    if checkpoint_config == requested_config:
        return
    if not allow_epoch_extension:
        raise ValueError("Resume config differs from the checkpoint config")

    changed = {key for key in checkpoint_config if checkpoint_config[key] != requested_config[key]}
    allowed_changes = {"epochs", "checkpoint_epochs"}
    disallowed = changed - allowed_changes
    if disallowed:
        raise ValueError(
            "Epoch extension cannot change training settings: " + ", ".join(sorted(disallowed))
        )
    if requested_config["epochs"] <= checkpoint_config["epochs"]:
        raise ValueError("Epoch extension must increase the configured epoch horizon")
    if requested_config["epochs"] <= completed_epoch:
        raise ValueError("Epoch extension must end after the completed checkpoint epoch")


def load_ema_checkpoint(
    checkpoint_path: Path,
    config: dict,
    device: torch.device | str = "cpu",
) -> UNetModel:
    """Load the validation-selected EMA U-Net for reconstruction/evaluation."""
    resolved = _resolved_config(config)
    checkpoint = torch.load(Path(checkpoint_path), map_location="cpu", weights_only=False)
    if checkpoint.get("schema_version") != 1 or checkpoint.get("method") != METHOD:
        raise ValueError("Checkpoint schema or method does not match partial DDPM")
    _validate_resume_config(
        checkpoint.get("config"),
        resolved,
        completed_epoch=checkpoint.get("epoch", 0),
        allow_epoch_extension=True,
    )
    model = build_model(resolved, device)
    model.load_state_dict(checkpoint["ema"])
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model


def derive_sampling_seed(base_seed: int, sample_id: str) -> int:
    """Derive a stable 32-bit seed from a run seed and Hub sample UUID.

    Keep the UUID private; the derived seed is not an anonymization mechanism.
    """
    if isinstance(base_seed, bool) or not isinstance(base_seed, int) or base_seed < 0:
        raise ValueError("base_seed must be a non-negative integer")
    if not isinstance(sample_id, str):
        raise TypeError("sample_id must be a Hub sample UUID string")
    try:
        canonical_id = str(uuid.UUID(sample_id))
    except ValueError as exc:
        raise ValueError("sample_id must be a Hub sample UUID string") from exc
    payload = f"dacn-ddpm-seed-v1:{base_seed}:{canonical_id}".encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:4], "big")


@torch.inference_mode()
def partial_reconstruct(
    model: torch.nn.Module,
    images: torch.Tensor,
    *,
    starting_timestep: int,
    seed: int,
    diffusion: GaussianDiffusion,
) -> torch.Tensor:
    """Noisify to one-based t* and run every ancestral DDPM step back to t=0."""
    if isinstance(starting_timestep, bool) or not isinstance(starting_timestep, int):
        raise TypeError("starting_timestep must be an integer")
    if not 1 <= starting_timestep <= diffusion.timesteps:
        raise ValueError("starting_timestep must be in [1, total_timesteps]")
    if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed < 2**32:
        raise ValueError("seed must be an integer in [0, 2**32)")
    if images.ndim != 4 or images.shape != (1, 1, 256, 256):
        raise ValueError("images must have shape [1, 1, 256, 256]; reconstruct one sample per call")
    model.eval()
    with _fixed_rng(seed, images.device):
        return diffusion.reconstruct(model, images, steps=starting_timestep).clamp(-1, 1)


@torch.inference_mode()
def posterior_mean_reconstruct_batch(
    model: torch.nn.Module,
    images: torch.Tensor,
    *,
    starting_timestep: int,
    seeds: list[int],
    diffusion: GaussianDiffusion,
) -> torch.Tensor:
    """Batch deterministic reverse chains with per-sample initial-noise seeds."""
    if isinstance(starting_timestep, bool) or not isinstance(starting_timestep, int):
        raise TypeError("starting_timestep must be an integer")
    if not 1 <= starting_timestep <= diffusion.timesteps:
        raise ValueError("starting_timestep must be in [1, total_timesteps]")
    if images.ndim != 4 or images.shape[1:] != (1, 256, 256) or not len(images):
        raise ValueError("images must have nonempty shape [N, 1, 256, 256]")
    if len(seeds) != len(images) or any(
        isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed < 2**32
        for seed in seeds
    ):
        raise ValueError("seeds must contain one 32-bit integer per image")
    if diffusion.noise_type != "gaussian":
        raise ValueError("batched posterior-mean reconstruction currently requires Gaussian noise")

    initial_noise = []
    for image, seed in zip(images, seeds, strict=True):
        generator = torch.Generator(device=images.device).manual_seed(seed)
        initial_noise.append(
            torch.randn(
                image.shape,
                device=images.device,
                dtype=images.dtype,
                generator=generator,
            )
        )
    model.eval()
    return diffusion.reconstruct(
        model,
        images,
        steps=starting_timestep,
        initial_noise=torch.stack(initial_noise),
        reverse_sampler="posterior_mean",
    ).clamp(-1, 1)


def train(
    *,
    train_dir: Path,
    validation_dir: Path,
    output_dir: Path,
    config: dict,
    device: torch.device,
    resume_from_checkpoint: Path | None = None,
    allow_epoch_extension: bool = False,
    on_checkpoint: Callable[[dict], None] | None = None,
) -> dict:
    """Train epsilon-prediction DDPM on normal train slices; select by normal val loss."""
    config = _resolved_config(config)
    if not isinstance(allow_epoch_extension, bool):
        raise TypeError("allow_epoch_extension must be a boolean")
    _seed_everything(config["seed"])
    train_data = ShardDataset(train_dir, "train", include_masks=False)
    validation_data = ShardDataset(validation_dir, "val", include_masks=True)
    if np.any(train_data.labels != 0):
        raise ValueError("DDPM training split must contain normal slices only")
    if set(np.unique(validation_data.labels)) != {0, 1}:
        raise ValueError("Validation split must contain normal and anomalous slices")

    train_indices = _select_indices(
        train_data.labels,
        config["max_train_samples"],
        seed=config["seed"],
        stratified=False,
    )
    validation_indices = _select_indices(
        validation_data.labels,
        config["max_validation_samples"],
        seed=config["seed"] + 1,
        stratified=True,
    )
    normal_validation_indices = [
        index for index in validation_indices if validation_data.labels[index] == 0
    ]
    if not normal_validation_indices:
        raise ValueError("Normal validation samples are required for checkpoint selection")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    train_sampler = ShardBatchSampler(
        train_data,
        config["batch_size"],
        config["seed"],
        indices=train_indices,
    )
    train_loader = DataLoader(
        train_data,
        batch_sampler=train_sampler,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )
    validation_loader = DataLoader(
        Subset(validation_data, normal_validation_indices),
        batch_size=config["validation_batch_size"],
        shuffle=False,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )

    model = build_model(config, device)
    ema = build_model(config, device)
    ema.load_state_dict(model.state_dict())
    ema.eval()
    for parameter in ema.parameters():
        parameter.requires_grad_(False)
    diffusion = build_diffusion(config)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config["learning_rate"],
        weight_decay=config["weight_decay"],
        betas=(0.9, 0.999),
    )

    history: list[dict] = []
    best_epoch = 0
    best_validation_noise_mse = float("inf")
    optimizer_steps = 0
    total_training_seconds = 0.0
    start_epoch = 1
    if resume_from_checkpoint is not None:
        checkpoint_path = Path(resume_from_checkpoint)
        if not checkpoint_path.is_file():
            raise FileNotFoundError("Resume checkpoint is not available in the run Volume")
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        if checkpoint.get("schema_version") != 1 or checkpoint.get("method") != METHOD:
            raise ValueError("Resume checkpoint schema or method does not match")
        checkpoint_epoch = checkpoint.get("epoch")
        if isinstance(checkpoint_epoch, bool) or not isinstance(checkpoint_epoch, int):
            raise ValueError("Resume checkpoint epoch is invalid")
        _validate_resume_config(
            checkpoint.get("config"),
            config,
            completed_epoch=checkpoint_epoch,
            allow_epoch_extension=allow_epoch_extension,
        )
        model.load_state_dict(checkpoint["model"])
        ema.load_state_dict(checkpoint["ema"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        history = checkpoint["history"]
        best_epoch = checkpoint["best_epoch"]
        best_validation_noise_mse = checkpoint["best_validation_noise_mse"]
        optimizer_steps = checkpoint["optimizer_steps"]
        total_training_seconds = checkpoint["total_training_seconds"]
        start_epoch = checkpoint_epoch + 1
        train_sampler.epoch = checkpoint["sampler_epoch"]
        _restore_rng_state(checkpoint["rng_state"])
    else:
        reserved = {"run_manifest.json", "source_snapshot.zip"}
        unexpected = [path.name for path in output_dir.iterdir() if path.name not in reserved]
        if unexpected:
            raise ValueError("DDPM output directory contains artifacts from another run")

    best_checkpoint_path = output_dir / "ddpm_best.pt"
    last_checkpoint_path = output_dir / "ddpm_last.pt"
    for epoch in range(start_epoch, config["epochs"] + 1):
        epoch_started = time.perf_counter()
        model.train()
        loss_total = 0.0
        sample_total = 0
        for batch_index, (clean_images, _) in enumerate(train_loader, start=1):
            clean_images = clean_images.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            loss, _ = diffusion.training_loss(model, clean_images)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config["gradient_clip_norm"])
            optimizer.step()
            update_ema_params(ema, model, decay_rate=config["ema_decay"])
            count = len(clean_images)
            loss_total += float(loss.detach()) * count
            sample_total += count
            optimizer_steps += 1
            if batch_index % config["log_every_steps"] == 0 or batch_index == len(train_loader):
                print(
                    f"epoch={epoch}/{config['epochs']} step={batch_index}/{len(train_loader)} "
                    f"samples={sample_total}/{len(train_indices)} "
                    f"mean_epsilon_mse={loss_total / max(sample_total, 1):.6f}",
                    flush=True,
                )

        validation_mse, validation_samples = _validation_noise_mse(
            ema,
            diffusion,
            validation_loader,
            device,
            seed=config["seed"] + config["validation_seed_offset"],
        )
        epoch_seconds = time.perf_counter() - epoch_started
        total_training_seconds += epoch_seconds
        epoch_record = {
            "epoch": epoch,
            "optimizer_steps": optimizer_steps,
            "train_samples": sample_total,
            "train_epsilon_mse": loss_total / max(sample_total, 1),
            "normal_validation_epsilon_mse": validation_mse,
            "normal_validation_samples": validation_samples,
            "seconds": epoch_seconds,
        }
        history.append(epoch_record)
        best_improved = validation_mse < best_validation_noise_mse
        if best_improved:
            best_validation_noise_mse = validation_mse
            best_epoch = epoch
        _write_json(output_dir / "training_history.json", {"epochs": history})

        state = _checkpoint_state(
            model=model,
            ema=ema,
            optimizer=optimizer,
            epoch=epoch,
            optimizer_steps=optimizer_steps,
            sampler_epoch=train_sampler.epoch,
            config=config,
            history=history,
            best_epoch=best_epoch,
            best_validation_noise_mse=best_validation_noise_mse,
            total_training_seconds=total_training_seconds,
        )
        _save_checkpoint(last_checkpoint_path, state)
        if epoch in config["checkpoint_epochs"]:
            _save_checkpoint(output_dir / f"ddpm_epoch_{epoch:03d}.pt", state)
        if best_improved:
            _save_checkpoint(best_checkpoint_path, state)
        _write_json(
            output_dir / "validation_metrics.json",
            {
                "selection_metric": "normal_validation_epsilon_mse",
                "best_epoch": best_epoch,
                "best_normal_validation_epsilon_mse": best_validation_noise_mse,
                "normal_validation_samples": validation_samples,
            },
        )
        print(
            f"epoch={epoch} complete train_epsilon_mse={epoch_record['train_epsilon_mse']:.6f} "
            f"normal_validation_epsilon_mse={validation_mse:.6f} "
            f"seconds={epoch_seconds:.1f}",
            flush=True,
        )
        if on_checkpoint is not None:
            on_checkpoint(
                {
                    "epoch": epoch,
                    "optimizer_steps": optimizer_steps,
                    "checkpoint": last_checkpoint_path.name,
                    "best_epoch": best_epoch,
                }
            )

    if best_epoch == 0 or not best_checkpoint_path.is_file():
        raise RuntimeError("DDPM run has no validation-selected checkpoint")
    best_state = torch.load(best_checkpoint_path, map_location=device, weights_only=False)
    ema.load_state_dict(best_state["ema"])
    ema.eval()

    _write_json(output_dir / "config.json", config)
    preview_count = min(config["preview_samples"], len(validation_indices))
    if preview_count >= 2:
        preview_relative_indices = _select_indices(
            validation_data.labels[validation_indices],
            preview_count,
            seed=config["seed"] + config["preview_seed_offset"] + 1,
            stratified=True,
        )
        preview_indices = [validation_indices[index] for index in preview_relative_indices]
    else:
        preview_indices = validation_indices[:preview_count]
    preview_loader = DataLoader(
        Subset(validation_data, preview_indices),
        batch_size=preview_count,
        shuffle=False,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )
    preview_batch = next(iter(preview_loader))
    preview_images = preview_batch[0]
    preview_images = preview_images.to(device, non_blocking=True)
    with _fixed_rng(config["seed"] + config["preview_seed_offset"], device):
        preview_reconstructions = diffusion.reconstruct(
            ema,
            preview_images,
            steps=config["preview_starting_timestep"],
        ).clamp(-1, 1)
    original = preview_images[:, 0, 8:248, 8:248].cpu().numpy().astype(np.float32)
    reconstructed = preview_reconstructions[:, 0, 8:248, 8:248].cpu().numpy().astype(np.float32)
    temporary = output_dir / "validation_reconstruction.npz.tmp"
    with temporary.open("wb") as stream:
        np.savez_compressed(
            stream,
            images=original,
            reconstructions=reconstructed,
            pixel_absolute_difference=np.abs(original - reconstructed).astype(np.float32),
        )
    os.replace(temporary, output_dir / "validation_reconstruction.npz")

    peak_vram = torch.cuda.max_memory_allocated(device) if device.type == "cuda" else 0
    return {
        "method": METHOD,
        "selection_metric": "normal_validation_epsilon_mse",
        "best_epoch": best_epoch,
        "best_normal_validation_epsilon_mse": best_validation_noise_mse,
        "epochs_completed": history[-1]["epoch"] if history else 0,
        "optimizer_steps": optimizer_steps,
        "train_samples_per_epoch": len(train_indices),
        "normal_validation_samples": len(normal_validation_indices),
        "validation_samples_selected": len(validation_indices),
        "preview_starting_timestep_one_based": config["preview_starting_timestep"],
        "preview_samples": len(original),
        "checkpoint": best_checkpoint_path.name,
        "resume_checkpoint": last_checkpoint_path.name,
        "milestone_checkpoints": [
            f"ddpm_epoch_{epoch:03d}.pt"
            for epoch in config["checkpoint_epochs"]
            if (output_dir / f"ddpm_epoch_{epoch:03d}.pt").is_file()
        ],
        "validation_reconstruction": "validation_reconstruction.npz",
        "training_history": "training_history.json",
        "training_seconds": total_training_seconds,
        "training_seconds_scope": "sum of completed epoch training and validation durations",
        "peak_vram_bytes": peak_vram,
    }
