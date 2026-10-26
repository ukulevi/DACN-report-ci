"""Coarse-noise denoising autoencoder baseline for BMAD FLAIR shards.

The implementation adapts the single-channel BMAD export to the DAE setup of
Kascenas et al. It trains only on normal samples and does not accept a test path.
"""

import csv
import json
import math
import os
import random
import time
from collections import defaultdict
from pathlib import Path, PurePosixPath

import numpy as np
import torch
from sklearn.metrics import precision_recall_curve
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset, Sampler

from dacn.evaluation.metrics import compute_binary_metrics

DEFAULTS = {
    "seed": 17,
    "epochs": 3,
    "batch_size": 16,
    "learning_rate": 1e-4,
    "weight_decay": 1e-5,
    "noise_resolution": 16,
    "noise_std": 0.2,
    "width": 32,
    "early_stopping_patience": 5,
}


class _StandardizedConv2d(nn.Conv2d):
    """Conv2d with per-output-channel weight standardization."""

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        weight = self.weight
        weight = weight - weight.mean(dim=(1, 2, 3), keepdim=True)
        variance = weight.flatten(1).var(dim=1, unbiased=False).view(-1, 1, 1, 1)
        weight = weight * torch.rsqrt(variance + 1e-5)
        return F.conv2d(
            inputs,
            weight,
            self.bias,
            self.stride,
            self.padding,
            self.dilation,
            self.groups,
        )


class _ConvBlock(nn.Module):
    def __init__(self, input_channels: int, output_channels: int):
        super().__init__()
        groups = min(8, output_channels)
        while output_channels % groups:
            groups -= 1
        self.layers = nn.Sequential(
            _StandardizedConv2d(input_channels, output_channels, 3, padding=1, bias=False),
            nn.GroupNorm(groups, output_channels),
            nn.SiLU(),
            _StandardizedConv2d(output_channels, output_channels, 3, padding=1, bias=False),
            nn.GroupNorm(groups, output_channels),
            nn.SiLU(),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.layers(inputs)


class CoarseNoiseDAE(nn.Module):
    """Four-level U-Net trained to denoise smoothly upsampled Gaussian noise."""

    def __init__(self, width: int = 32):
        super().__init__()
        channels = (width, width * 2, width * 4, width * 8)
        self.pool = nn.AvgPool2d(2)
        self.encoders = nn.ModuleList(
            [
                _ConvBlock(1, channels[0]),
                _ConvBlock(channels[0], channels[1]),
                _ConvBlock(channels[1], channels[2]),
                _ConvBlock(channels[2], channels[3]),
            ]
        )
        self.middle = _ConvBlock(channels[3], channels[3] * 2)
        decoder_inputs = (channels[3] * 2, channels[3], channels[2], channels[1])
        decoder_outputs = (channels[3], channels[2], channels[1], channels[0])
        self.upconvs = nn.ModuleList(
            [
                nn.ConvTranspose2d(source, target, kernel_size=2, stride=2)
                for source, target in zip(decoder_inputs, decoder_outputs, strict=True)
            ]
        )
        self.decoders = nn.ModuleList(
            [_ConvBlock(target * 2, target) for target in decoder_outputs]
        )
        self.output = nn.Conv2d(channels[0], 1, kernel_size=1)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        skips = []
        value = inputs
        for encoder in self.encoders:
            value = encoder(value)
            skips.append(value)
            value = self.pool(value)
        value = self.middle(value)
        for upconv, decoder, skip in zip(self.upconvs, self.decoders, reversed(skips), strict=True):
            value = upconv(value)
            if value.shape[-2:] != skip.shape[-2:]:
                value = F.interpolate(
                    value, size=skip.shape[-2:], mode="bilinear", align_corners=False
                )
            value = decoder(torch.cat((skip, value), dim=1))
        return self.output(value)


class _ShardDataset(Dataset):
    """Read one compressed NPZ shard at a time, reusing it across its batches."""

    def __init__(self, split_dir: Path, split: str, *, include_masks: bool):
        self.split_dir = Path(split_dir).resolve()
        self.split = split
        self.include_masks = include_masks
        self.records = []
        self.indices_by_shard: dict[str, list[int]] = defaultdict(list)
        index_path = self.split_dir / "index.csv"
        if not index_path.is_file():
            raise ValueError(f"Dataset split is missing index.csv: {split}")

        with index_path.open(newline="", encoding="utf-8") as stream:
            reader = csv.DictReader(stream)
            expected = ("sample_id", "patient_group_id", "split", "label", "shard", "offset")
            if tuple(reader.fieldnames or ()) != expected:
                raise ValueError(f"Unexpected index.csv schema for split={split}")
            for row in reader:
                if row["split"] != split or row["label"] not in ("0", "1"):
                    raise ValueError(f"Invalid split or label in {split}/index.csv")
                relative = PurePosixPath(row["shard"])
                if (
                    len(relative.parts) != 3
                    or relative.parts[:2] != ("data", split)
                    or not relative.name.startswith("shard-")
                    or relative.suffix != ".npz"
                    or ".." in relative.parts
                ):
                    raise ValueError(f"Invalid shard path in {split}/index.csv")
                shard_name = relative.name
                shard_path = self.split_dir / shard_name
                if not shard_path.is_file():
                    raise ValueError(f"Missing data shard: {shard_path.name}")
                try:
                    offset = int(row["offset"])
                except ValueError as exc:
                    raise ValueError(f"Invalid shard offset in {split}/index.csv") from exc
                if offset < 0:
                    raise ValueError(f"Invalid shard offset in {split}/index.csv")
                self.indices_by_shard[shard_name].append(len(self.records))
                self.records.append((shard_name, offset, int(row["label"])))

        if not self.records:
            raise ValueError(f"Dataset split is empty: {split}")
        self.labels = np.asarray([record[2] for record in self.records], dtype=np.uint8)
        self._cached_name = None
        self._cached_images = None
        self._cached_labels = None
        self._cached_masks = None

    def __len__(self) -> int:
        return len(self.records)

    def _load_shard(self, shard_name: str) -> None:
        if shard_name == self._cached_name:
            return
        with np.load(self.split_dir / shard_name, allow_pickle=False) as archive:
            expected = {"images", "masks", "labels"}
            if set(archive.files) != expected:
                raise ValueError(f"Unexpected fields in {shard_name}")
            self._cached_images = np.array(archive["images"], copy=True)
            self._cached_labels = np.array(archive["labels"], copy=True)
            self._cached_masks = (
                np.array(archive["masks"], copy=True) if self.include_masks else None
            )
        self._cached_name = shard_name

    def __getitem__(self, index: int):
        shard_name, offset, expected_label = self.records[index]
        self._load_shard(shard_name)
        if offset >= len(self._cached_images):
            raise ValueError(f"Shard offset is out of range: {shard_name}")
        image = self._cached_images[offset]
        label = int(self._cached_labels[offset])
        if image.shape != (1, 256, 256) or image.dtype != np.float32:
            raise ValueError(f"Unexpected image tensor in {shard_name}")
        if label != expected_label:
            raise ValueError(f"Index label disagrees with shard label in {shard_name}")
        image_tensor = torch.from_numpy(image.copy())
        label_tensor = torch.tensor(label, dtype=torch.uint8)
        if not self.include_masks:
            return image_tensor, label_tensor
        mask = self._cached_masks[offset]
        if mask.shape != (256, 256) or mask.dtype != np.uint8:
            raise ValueError(f"Unexpected mask tensor in {shard_name}")
        return image_tensor, torch.from_numpy(mask.copy()), label_tensor


class _ShardBatchSampler(Sampler[list[int]]):
    """Shuffle shards and samples while keeping each shard's batches together."""

    def __init__(self, dataset: _ShardDataset, batch_size: int, seed: int):
        self.dataset = dataset
        self.batch_size = batch_size
        self.seed = seed
        self.epoch = 0

    def __iter__(self):
        generator = torch.Generator().manual_seed(self.seed + self.epoch)
        self.epoch += 1
        shard_names = list(self.dataset.indices_by_shard)
        shard_order = torch.randperm(len(shard_names), generator=generator).tolist()
        for shard_index in shard_order:
            records = self.dataset.indices_by_shard[shard_names[shard_index]]
            order = torch.randperm(len(records), generator=generator).tolist()
            for start in range(0, len(order), self.batch_size):
                yield [records[position] for position in order[start : start + self.batch_size]]

    def __len__(self) -> int:
        return sum(
            math.ceil(len(records) / self.batch_size)
            for records in self.dataset.indices_by_shard.values()
        )


def _resolved_config(config: dict) -> dict:
    if not isinstance(config, dict):
        raise TypeError("config must be a JSON object")
    unknown = set(config) - set(DEFAULTS)
    if unknown:
        raise ValueError(f"Unknown DAE config keys: {', '.join(sorted(unknown))}")
    result = {**DEFAULTS, **config}
    for key in (
        "seed",
        "epochs",
        "batch_size",
        "noise_resolution",
        "width",
        "early_stopping_patience",
    ):
        if isinstance(result[key], bool) or not isinstance(result[key], int) or result[key] < 1:
            raise ValueError(f"{key} must be a positive integer")
    for key in ("learning_rate", "weight_decay", "noise_std"):
        if isinstance(result[key], bool) or not isinstance(result[key], (int, float)):
            raise TypeError(f"{key} must be numeric")
        result[key] = float(result[key])
        if not math.isfinite(result[key]) or result[key] < 0:
            raise ValueError(f"{key} must be finite and non-negative")
    if result["learning_rate"] == 0 or result["noise_std"] == 0:
        raise ValueError("learning_rate and noise_std must be greater than zero")
    if result["width"] < 8 or result["width"] % 8:
        raise ValueError("width must be at least 8 and divisible by 8")
    if result["noise_resolution"] > 256:
        raise ValueError("noise_resolution cannot exceed the 256-pixel image size")
    return result


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _corrupt(images: torch.Tensor, resolution: int, noise_std: float) -> torch.Tensor:
    batch, channels, height, width = images.shape
    noise = torch.randn(batch, channels, resolution, resolution, device=images.device)
    noise = noise * noise_std
    noise = F.interpolate(noise, size=(height, width), mode="bilinear", align_corners=False)
    shift_y = int(torch.randint(height, (1,), device=images.device).item())
    shift_x = int(torch.randint(width, (1,), device=images.device).item())
    noise = torch.roll(noise, shifts=(shift_y, shift_x), dims=(-2, -1))
    foreground = images > 0.01
    return images + noise * foreground


def _foreground(images: torch.Tensor) -> torch.Tensor:
    foreground = (images > 0.01).float()
    # Erode the region to avoid scoring the sharp edge between brain and background.
    return (F.avg_pool2d(foreground, kernel_size=5, stride=1, padding=2) > 0.95).float()


def _evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> dict:
    model.eval()
    all_scores, all_masks, all_labels = [], [], []
    with torch.inference_mode():
        for images, masks, labels in loader:
            # Undo the fixed dataset normalization to match the source DAE's [0, 1] range.
            clean = ((images.to(device, non_blocking=True) + 1.0) / 2.0).clamp(0.0, 1.0)
            reconstruction = model(clean)
            scores = (reconstruction - clean).abs() * _foreground(clean)
            all_scores.append(scores[:, 0].cpu().numpy())
            all_masks.append(masks.numpy())
            all_labels.append(labels.numpy())

    scores = np.concatenate(all_scores, axis=0)[:, 8:248, 8:248]
    masks = np.concatenate(all_masks, axis=0)[:, 8:248, 8:248].astype(bool)
    labels = np.concatenate(all_labels).astype(np.uint8)
    if set(np.unique(labels)) != {0, 1}:
        raise ValueError("Validation split must contain normal and anomalous samples")
    pixel_targets = masks.reshape(-1).astype(np.uint8)
    pixel_scores = scores.reshape(-1).astype(np.float64)
    precision, recall, thresholds = precision_recall_curve(pixel_targets, pixel_scores)
    dice_values = 2 * precision[:-1] * recall[:-1] / (precision[:-1] + recall[:-1] + 1e-12)
    best_threshold = float(thresholds[int(np.argmax(dice_values))])
    pixel_metrics = compute_binary_metrics(pixel_targets, pixel_scores, best_threshold)
    predicted = scores >= best_threshold

    dice_by_anomaly = []
    for prediction, target, label in zip(predicted, masks, labels, strict=True):
        if label:
            denominator = int(prediction.sum() + target.sum())
            dice_by_anomaly.append(2 * int(np.logical_and(prediction, target).sum()) / denominator)
    empty_mask_pixels = int(np.count_nonzero(labels == 0) * masks.shape[1] * masks.shape[2])
    false_positive_pixels = int(
        np.logical_and(predicted, ~masks).sum(axis=(1, 2))[labels == 0].sum()
    )
    k = max(1, math.ceil(scores.shape[1] * scores.shape[2] * 0.01))
    image_scores = np.partition(scores.reshape(len(scores), -1), -k, axis=1)[:, -k:].mean(axis=1)
    image_metrics = compute_binary_metrics(labels, image_scores)

    return {
        "pixel_ap": pixel_metrics["auprc"],
        "pixel_auroc": pixel_metrics["auroc"],
        "pixel_dice_micro": pixel_metrics["f1"],
        "pixel_precision": pixel_metrics["precision"],
        "pixel_recall": pixel_metrics["recall"],
        "pixel_threshold_selected_on_validation": best_threshold,
        "pixel_dice_macro_anomalous_slices": float(np.mean(dice_by_anomaly)),
        "false_positive_pixel_rate_empty_mask_slices": (
            false_positive_pixels / empty_mask_pixels if empty_mask_pixels else 0.0
        ),
        "image_ap": image_metrics["auprc"],
        "image_auroc": image_metrics["auroc"],
        "image_score": "mean_top_1_percent_of_240x240_pixel_scores",
        "validation_slices": len(labels),
    }


def _write_json(path: Path, value: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    os.replace(temporary, path)


def _save_checkpoint(
    path: Path, model: nn.Module, optimizer, epoch: int, config: dict, metrics: dict
):
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(
        {
            "schema_version": 1,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch,
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
    """Train a normal-only coarse-noise DAE and select it using validation AP."""
    config = _resolved_config(config)
    _seed_everything(config["seed"])
    train_data = _ShardDataset(train_dir, "train", include_masks=False)
    validation_data = _ShardDataset(validation_dir, "val", include_masks=True)
    if np.any(train_data.labels != 0):
        raise ValueError("DAE training split must contain normal samples only")
    if not np.any(validation_data.labels == 0) or not np.any(validation_data.labels == 1):
        raise ValueError("Validation split must contain both normal and anomalous samples")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(device)
    train_loader = DataLoader(
        train_data,
        batch_sampler=_ShardBatchSampler(train_data, config["batch_size"], config["seed"]),
        num_workers=0,
        pin_memory=device.type == "cuda",
    )
    validation_loader = DataLoader(
        validation_data,
        batch_size=config["batch_size"],
        shuffle=False,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )
    model = CoarseNoiseDAE(width=config["width"]).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config["learning_rate"],
        amsgrad=True,
        weight_decay=config["weight_decay"],
    )

    history = []
    best_ap = -1.0
    best_epoch = 0
    best_metrics = None
    stale_epochs = 0
    checkpoint_path = output_dir / "dae_best.pt"

    for epoch in range(1, config["epochs"] + 1):
        started = time.perf_counter()
        model.train()
        loss_total = 0.0
        sample_total = 0
        for clean_images, _ in train_loader:
            # Store format is [-1, 1]; source DAE noise is defined on [0, 1].
            clean = ((clean_images.to(device, non_blocking=True) + 1.0) / 2.0).clamp(0.0, 1.0)
            foreground = (clean > 0.01).float()
            noisy = _corrupt(clean, config["noise_resolution"], config["noise_std"])
            optimizer.zero_grad(set_to_none=True)
            reconstruction = model(noisy)
            squared_error = (reconstruction - clean).square() * foreground
            loss = squared_error.sum() / foreground.sum().clamp_min(1.0)
            loss.backward()
            optimizer.step()
            count = len(clean_images)
            loss_total += float(loss.detach()) * count
            sample_total += count

        validation_metrics = _evaluate(model, validation_loader, device)
        epoch_record = {
            "epoch": epoch,
            "train_loss": loss_total / max(sample_total, 1),
            "validation": validation_metrics,
            "seconds": time.perf_counter() - started,
        }
        history.append(epoch_record)
        _write_json(output_dir / "training_history.json", {"epochs": history})
        current_ap = validation_metrics["pixel_ap"]
        if current_ap > best_ap:
            best_ap = current_ap
            best_epoch = epoch
            best_metrics = validation_metrics
            stale_epochs = 0
            _save_checkpoint(checkpoint_path, model, optimizer, epoch, config, validation_metrics)
            _write_json(
                output_dir / "validation_metrics.json",
                {"best_epoch": best_epoch, "metrics": best_metrics},
            )
        else:
            stale_epochs += 1
        if stale_epochs >= config["early_stopping_patience"]:
            break

    if best_metrics is None:
        raise RuntimeError("Training completed without producing a validation checkpoint")
    _write_json(output_dir / "config.json", config)
    return {
        "best_epoch": best_epoch,
        "epochs_completed": len(history),
        "best_validation_pixel_ap": best_ap,
        "validation_metrics": best_metrics,
        "checkpoint": checkpoint_path.name,
        "training_history": "training_history.json",
    }
