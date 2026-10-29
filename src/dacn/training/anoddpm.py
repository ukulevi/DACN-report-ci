"""AnoDDPM-style normal-only trainer for the BMAD Brain FLAIR Hub export.

This is an explicit BMAD adaptation, not an ANDi implementation and not a
claim of reproducing the AnoDDPM paper. Held-out test data is not accepted by
the trainer; validation alone selects checkpoints and thresholds.
"""

import copy
import json
import math
import os
import random
import time
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import precision_recall_curve
from torch.utils.data import DataLoader, Subset

from dacn.data.shards import ShardBatchSampler, ShardDataset
from dacn.evaluation.metrics import compute_binary_metrics
from dacn.models.diffusion.process import GaussianDiffusion
from dacn.models.diffusion.unet import UNetModel, update_ema_params

DEFAULTS = {
    "seed": 17,
    "epochs": 3,
    "batch_size": 1,
    "validation_batch_size": 1,
    "learning_rate": 1e-4,
    "weight_decay": 0.0,
    "timesteps": 1000,
    "maximum_training_timestep": 800,
    "validation_reconstruction_steps": 100,
    "beta_schedule": "linear",
    "noise_type": "simplex",
    "loss_type": "l2",
    "base_channels": 32,
    "channel_mults": "",
    "dropout": 0.0,
    "attention_resolutions": "16,8",
    "num_heads": 2,
    "num_head_channels": -1,
    "simplex_octaves": 6,
    "simplex_persistence": 0.8,
    "simplex_frequency": 64.0,
    "ema_decay": 0.9999,
    "gradient_clip_norm": 1.0,
    "early_stopping_patience": 5,
    "log_every_steps": 100,
    "max_train_samples": None,
    "max_validation_samples": None,
}


def _resolved_config(config: dict) -> dict:
    if not isinstance(config, dict):
        raise TypeError("config must be a JSON object")
    unknown = set(config) - set(DEFAULTS)
    if unknown:
        raise ValueError(f"Unknown AnoDDPM config keys: {', '.join(sorted(unknown))}")
    result = {**DEFAULTS, **config}

    positive_integers = (
        "seed",
        "epochs",
        "batch_size",
        "validation_batch_size",
        "timesteps",
        "maximum_training_timestep",
        "validation_reconstruction_steps",
        "base_channels",
        "num_heads",
        "simplex_octaves",
        "early_stopping_patience",
        "log_every_steps",
    )
    for key in positive_integers:
        if isinstance(result[key], bool) or not isinstance(result[key], int) or result[key] < 1:
            raise ValueError(f"{key} must be a positive integer")
    for key in ("max_train_samples", "max_validation_samples"):
        value = result[key]
        if value is not None and (
            isinstance(value, bool) or not isinstance(value, int) or value < 1
        ):
            raise ValueError(f"{key} must be null or a positive integer")
    if not isinstance(result["num_head_channels"], int):
        raise TypeError("num_head_channels must be an integer")
    if result["num_head_channels"] == 0 or result["num_head_channels"] < -1:
        raise ValueError("num_head_channels must be -1 or a positive integer")

    numeric = (
        "learning_rate",
        "weight_decay",
        "dropout",
        "simplex_persistence",
        "simplex_frequency",
        "ema_decay",
        "gradient_clip_norm",
    )
    for key in numeric:
        if isinstance(result[key], bool) or not isinstance(result[key], (int, float)):
            raise TypeError(f"{key} must be numeric")
        result[key] = float(result[key])
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
    if not 0 < result["simplex_persistence"] <= 1 or result["simplex_frequency"] <= 0:
        raise ValueError("Invalid simplex persistence or frequency")
    if result["base_channels"] % 32:
        raise ValueError("base_channels must be divisible by 32 for the AnoDDPM U-Net")
    if result["maximum_training_timestep"] > result["timesteps"]:
        raise ValueError("maximum_training_timestep cannot exceed timesteps")
    if result["validation_reconstruction_steps"] > result["timesteps"]:
        raise ValueError("validation_reconstruction_steps cannot exceed timesteps")
    if result["beta_schedule"] not in {"linear", "cosine"}:
        raise ValueError("beta_schedule must be linear or cosine")
    if result["noise_type"] not in {"gaussian", "simplex"}:
        raise ValueError("noise_type must be gaussian or simplex")
    if result["loss_type"] not in {"l1", "l2"}:
        raise ValueError("loss_type must be l1 or l2")
    if not isinstance(result["attention_resolutions"], str):
        raise TypeError("attention_resolutions must be a comma-separated string")
    for value in result["attention_resolutions"].split(","):
        if not value.isdigit() or int(value) < 1 or 256 % int(value):
            raise ValueError("attention_resolutions must divide the 256-pixel image size")

    channel_mults = result["channel_mults"]
    if channel_mults != "":
        if not isinstance(channel_mults, (list, tuple)) or not channel_mults:
            raise TypeError("channel_mults must be empty or a non-empty numeric list")
        if any(
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
            or value <= 0
            for value in channel_mults
        ):
            raise ValueError("channel_mults must contain positive finite numbers")
        result["channel_mults"] = tuple(channel_mults)
    return result


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _select_indices(
    labels: np.ndarray,
    limit: int | None,
    *,
    seed: int,
    stratified: bool,
) -> list[int]:
    """Select a deterministic smoke-test subset without changing full-run defaults."""
    if limit is None:
        return list(range(len(labels)))
    if limit > len(labels):
        raise ValueError(f"Sample limit {limit} exceeds split size {len(labels)}")
    generator = np.random.default_rng(seed)
    if not stratified:
        return sorted(generator.choice(len(labels), size=limit, replace=False).tolist())

    classes = sorted(np.unique(labels).tolist())
    if limit < len(classes):
        raise ValueError("Stratified sample limit must include at least one item per class")
    selected: list[int] = []
    base, remainder = divmod(limit, len(classes))
    for class_index, label in enumerate(classes):
        candidates = np.flatnonzero(labels == label)
        requested = base + int(class_index < remainder)
        take = min(requested, len(candidates))
        selected.extend(generator.choice(candidates, size=take, replace=False).tolist())
    if len(selected) < limit:
        remaining = np.setdiff1d(np.arange(len(labels)), np.asarray(selected), assume_unique=True)
        selected.extend(
            generator.choice(remaining, size=limit - len(selected), replace=False).tolist()
        )
    return sorted(selected)


@contextmanager
def _fixed_validation_rng(seed: int, device: torch.device):
    """Make validation reconstruction repeatable without changing training RNG state."""
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


def _best_f1_threshold(labels: np.ndarray, scores: np.ndarray) -> float:
    precision, recall, thresholds = precision_recall_curve(labels, scores)
    if not len(thresholds):
        raise ValueError("Cannot select a threshold from constant validation scores")
    f1 = 2 * precision[:-1] * recall[:-1] / (precision[:-1] + recall[:-1] + 1e-12)
    return float(thresholds[int(np.argmax(f1))])


def _evaluate(
    model: torch.nn.Module,
    diffusion: GaussianDiffusion,
    loader: DataLoader,
    device: torch.device,
    *,
    reconstruction_steps: int,
    validation_seed: int,
) -> dict:
    model.eval()
    all_scores, all_masks, all_labels = [], [], []
    with _fixed_validation_rng(validation_seed, device), torch.inference_mode():
        for batch_index, (images, masks, labels) in enumerate(loader, start=1):
            images = images.to(device, non_blocking=True)
            reconstruction = diffusion.reconstruct(model, images, steps=reconstruction_steps)
            scores = (reconstruction - images).square()[:, 0, 8:248, 8:248]
            all_scores.append(scores.cpu().numpy())
            all_masks.append(masks[:, 8:248, 8:248].numpy())
            all_labels.append(labels.numpy())
            if batch_index % 10 == 0 or batch_index == len(loader):
                print(
                    f"validation_batch={batch_index}/{len(loader)} "
                    f"reconstruction_steps={reconstruction_steps}",
                    flush=True,
                )

    scores = np.concatenate(all_scores).astype(np.float64)
    masks = np.concatenate(all_masks).astype(bool)
    labels = np.concatenate(all_labels).astype(np.uint8)
    if set(np.unique(labels)) != {0, 1}:
        raise ValueError("Validation split must contain normal and anomalous samples")

    pixel_labels = masks.reshape(-1).astype(np.uint8)
    pixel_scores = scores.reshape(-1)
    pixel_threshold = _best_f1_threshold(pixel_labels, pixel_scores)
    pixel_metrics = compute_binary_metrics(pixel_labels, pixel_scores, pixel_threshold)
    predicted_masks = scores >= pixel_threshold
    anomalous_dice = []
    for prediction, target, label in zip(predicted_masks, masks, labels, strict=True):
        if label:
            denominator = int(prediction.sum() + target.sum())
            anomalous_dice.append(
                2 * int(np.logical_and(prediction, target).sum()) / max(denominator, 1)
            )

    top_k = max(1, math.ceil(scores.shape[1] * scores.shape[2] * 0.01))
    flattened = scores.reshape(len(scores), -1)
    image_scores = np.partition(flattened, -top_k, axis=1)[:, -top_k:].mean(axis=1)
    image_threshold = _best_f1_threshold(labels, image_scores)
    image_metrics = compute_binary_metrics(labels, image_scores, image_threshold)
    return {
        "score_direction": "higher_is_more_anomalous",
        "reconstruction_steps": reconstruction_steps,
        "pixel_ap": pixel_metrics["auprc"],
        "pixel_auroc": pixel_metrics["auroc"],
        "pixel_precision": pixel_metrics["precision"],
        "pixel_recall": pixel_metrics["recall"],
        "pixel_f1_micro": pixel_metrics["f1"],
        "pixel_threshold_selected_on_validation": pixel_threshold,
        "pixel_dice_macro_anomalous_slices": float(np.mean(anomalous_dice)),
        "image_ap": image_metrics["auprc"],
        "image_auroc": image_metrics["auroc"],
        "image_precision": image_metrics["precision"],
        "image_recall": image_metrics["recall"],
        "image_f1": image_metrics["f1"],
        "image_threshold_selected_on_validation": image_threshold,
        "image_score": "mean_top_1_percent_of_240x240_squared_reconstruction_error",
        "validation_slices": len(labels),
    }


def _write_json(path: Path, value: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def _save_checkpoint(
    path: Path,
    *,
    model: torch.nn.Module,
    ema: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    optimizer_steps: int,
    config: dict,
    metrics: dict,
) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(
        {
            "schema_version": 1,
            "method": "anoddpm_bmad_flair_adaptation",
            "model": model.state_dict(),
            "ema": ema.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch,
            "optimizer_steps": optimizer_steps,
            "config": config,
            "validation_metrics": metrics,
        },
        temporary,
    )
    os.replace(temporary, path)


def train(
    *,
    train_dir: Path,
    validation_dir: Path,
    output_dir: Path,
    config: dict,
    device: torch.device,
) -> dict:
    """Train an AnoDDPM-style model on normal BMAD slices and select on validation AP."""
    config = _resolved_config(config)
    _seed_everything(config["seed"])
    train_data = ShardDataset(train_dir, "train", include_masks=False)
    validation_data = ShardDataset(validation_dir, "val", include_masks=True)
    if np.any(train_data.labels != 0):
        raise ValueError("AnoDDPM training split must contain normal samples only")
    if set(np.unique(validation_data.labels)) != {0, 1}:
        raise ValueError("Validation split must contain normal and anomalous samples")
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

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    reserved = {"run_manifest.json", "source_snapshot.zip"}
    if any(path.name not in reserved for path in output_dir.iterdir()):
        raise ValueError("AnoDDPM output directory contains artifacts from another run")
    device = torch.device(device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    train_loader = DataLoader(
        train_data,
        batch_sampler=ShardBatchSampler(
            train_data,
            config["batch_size"],
            config["seed"],
            indices=train_indices,
        ),
        num_workers=0,
        pin_memory=device.type == "cuda",
    )
    validation_loader = DataLoader(
        Subset(validation_data, validation_indices),
        batch_size=config["validation_batch_size"],
        shuffle=False,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )
    model = UNetModel(
        256,
        config["base_channels"],
        channel_mults=config["channel_mults"],
        dropout=config["dropout"],
        attention_resolutions=config["attention_resolutions"],
        n_heads=config["num_heads"],
        n_head_channels=config["num_head_channels"],
        in_channels=1,
    ).to(device)
    ema = copy.deepcopy(model).to(device)
    ema.eval()
    for parameter in ema.parameters():
        parameter.requires_grad_(False)
    diffusion = GaussianDiffusion(
        timesteps=config["timesteps"],
        beta_schedule=config["beta_schedule"],
        noise_type=config["noise_type"],
        loss_type=config["loss_type"],
        simplex_octaves=config["simplex_octaves"],
        simplex_persistence=config["simplex_persistence"],
        simplex_frequency=config["simplex_frequency"],
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config["learning_rate"],
        weight_decay=config["weight_decay"],
        betas=(0.9, 0.999),
    )

    history = []
    best_ap = -1.0
    best_epoch = 0
    best_metrics = None
    optimizer_steps = 0
    stale_epochs = 0
    checkpoint_path = output_dir / "anoddpm_best.pt"
    started_training = time.perf_counter()
    print(
        json.dumps(
            {
                "event": "training_started",
                "device": str(device),
                "epochs": config["epochs"],
                "train_samples_per_epoch": len(train_indices),
                "validation_samples": len(validation_indices),
                "steps_per_epoch": len(train_loader),
            },
            sort_keys=True,
        ),
        flush=True,
    )

    for epoch in range(1, config["epochs"] + 1):
        started_epoch = time.perf_counter()
        model.train()
        loss_total = 0.0
        sample_total = 0
        for batch_index, (clean_images, _) in enumerate(train_loader, start=1):
            clean_images = clean_images.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            loss, _ = diffusion.training_loss(
                model,
                clean_images,
                maximum_timestep=config["maximum_training_timestep"],
            )
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
                    f"mean_loss={loss_total / sample_total:.6f}",
                    flush=True,
                )

        print(
            f"epoch={epoch} validation_started samples={len(validation_indices)} "
            f"reconstruction_steps={config['validation_reconstruction_steps']}",
            flush=True,
        )
        validation_metrics = _evaluate(
            ema,
            diffusion,
            validation_loader,
            device,
            reconstruction_steps=config["validation_reconstruction_steps"],
            validation_seed=config["seed"] + 1_000_000,
        )
        epoch_record = {
            "epoch": epoch,
            "optimizer_steps": optimizer_steps,
            "train_samples": sample_total,
            "train_loss": loss_total / max(sample_total, 1),
            "validation": validation_metrics,
            "seconds": time.perf_counter() - started_epoch,
        }
        history.append(epoch_record)
        _write_json(output_dir / "training_history.json", {"epochs": history})
        print(
            f"epoch={epoch} complete train_loss={epoch_record['train_loss']:.6f} "
            f"pixel_ap={validation_metrics['pixel_ap']:.6f} "
            f"pixel_auroc={validation_metrics['pixel_auroc']:.6f} "
            f"image_ap={validation_metrics['image_ap']:.6f} "
            f"seconds={epoch_record['seconds']:.1f}",
            flush=True,
        )

        current_ap = validation_metrics["pixel_ap"]
        if current_ap > best_ap:
            best_ap = current_ap
            best_epoch = epoch
            best_metrics = validation_metrics
            stale_epochs = 0
            _save_checkpoint(
                checkpoint_path,
                model=model,
                ema=ema,
                optimizer=optimizer,
                epoch=epoch,
                optimizer_steps=optimizer_steps,
                config=config,
                metrics=validation_metrics,
            )
            _write_json(
                output_dir / "validation_metrics.json",
                {"best_epoch": best_epoch, "metrics": best_metrics},
            )
        else:
            stale_epochs += 1
        if stale_epochs >= config["early_stopping_patience"]:
            break

    if best_metrics is None:
        raise RuntimeError("Training completed without a validation checkpoint")
    _write_json(output_dir / "config.json", config)
    peak_vram = torch.cuda.max_memory_allocated(device) if device.type == "cuda" else 0
    return {
        "method": "anoddpm_bmad_flair_adaptation",
        "best_epoch": best_epoch,
        "epochs_completed": len(history),
        "optimizer_steps": optimizer_steps,
        "train_samples_per_epoch": len(train_indices),
        "validation_samples": len(validation_indices),
        "best_validation_pixel_ap": best_ap,
        "validation_metrics": best_metrics,
        "checkpoint": checkpoint_path.name,
        "training_history": "training_history.json",
        "training_seconds": time.perf_counter() - started_training,
        "peak_vram_bytes": peak_vram,
    }
