"""Bounded normal-only masked reconstruction pretraining for experiment B.

Dense ResNet-18 plus a small decoder, not a reproduction of SparK.
Only the encoder is transferred to conditional DDPM fine-tuning.
"""

import hashlib
import math
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Subset

from dacn.data.shards import ShardBatchSampler, ShardDataset
from dacn.models.diffusion.feature_condition import FeatureConditionedDDPM
from dacn.training.ddpm_partial import (
    _fixed_rng,
    _restore_rng_state,
    _rng_state,
    _save_checkpoint,
    _seed_everything,
    _write_json,
    build_model,
)

METHOD = "resnet18_normal_masked_pretrain_v1"
REVISION = "350c74f2313e4e73ca65e7ffc6e9dcf74040b3b7"
DEFAULTS = {
    "epochs": 5,
    "batch_size": 8,
    "seed": 17,
    "learning_rate": 1e-4,
    "mask_ratio": 0.5,
    "patch_size": 32,
    "checkpoint_epochs": [1, 5],
    "max_train_samples": None,
    "max_normal_validation_samples": None,
    "log_every_steps": 50,
    "source_run_id": "ddpm-partial-t4-b8-e30-cont-20261102T034644Z",
    "source_sha256": "d3e06d2082e18234899b201718abd59078c8a800d66da3cc5d8b9e0d9ca9b946",
}


def resolved_config(config):
    if set(config) - DEFAULTS.keys():
        raise ValueError("Unknown pretraining config keys")
    result = {**DEFAULTS, **config}
    for key, maximum in (("epochs", 5), ("batch_size", 8)):
        if (
            isinstance(result[key], bool)
            or not isinstance(result[key], int)
            or not 1 <= result[key] <= maximum
        ):
            raise ValueError("Encoder pretraining budget exceeded")
    if result["patch_size"] != 32 or result["mask_ratio"] != 0.5:
        raise ValueError("B pilot fixes32pixel patches and50%masking")
    if not math.isfinite(result["learning_rate"]) or not 0 < result["learning_rate"] <= 1e-3:
        raise ValueError("Invalid learning rate")
    if result["seed"] < 0 or result["log_every_steps"] < 1:
        raise ValueError("Invalid seed/log frequency")
    for key in ("max_train_samples", "max_normal_validation_samples"):
        if result[key] is not None and (
            isinstance(result[key], bool) or not isinstance(result[key], int) or result[key] < 1
        ):
            raise ValueError("Invalid sample cap")
    if any(
        isinstance(k, bool) or not isinstance(k, int) or not 1 <= k <= result["epochs"]
        for k in result["checkpoint_epochs"]
    ):
        raise ValueError("Invalid checkpoint milestones")
    return result


def image_patch_mask(images):
    """Mask32/64 grid patches, force one to intersect foreground, preserve padding."""
    if images.ndim != 4 or images.shape[1:] != (1, 256, 256):
        raise ValueError("Expected grayscale256 images")
    scores = torch.rand(len(images), 64, device=images.device)
    for index in range(len(images)):
        coordinates = torch.nonzero(images[index, 0] > -0.95)
        if not len(coordinates):
            raise ValueError("Normal slice has no foreground")
        y, x = coordinates[torch.randint(len(coordinates), (), device=images.device)]
        scores[index, (y // 32) * 8 + x // 32] = 2
    selected = scores.topk(32, dim=1).indices
    grid = (
        torch.zeros_like(scores, dtype=torch.bool).scatter_(1, selected, True).reshape(-1, 1, 8, 8)
    )
    mask = F.interpolate(grid.float(), scale_factor=32, mode="nearest").bool()
    mask[:, :, :8] = mask[:, :, -8:] = False
    mask[:, :, :, :8] = mask[:, :, :, -8:] = False
    return mask


class MaskedResNet(nn.Module):
    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder
        blocks, channels = [], 512
        for outgoing in (256, 128, 64, 32, 16):
            blocks.extend(
                [
                    nn.Upsample(scale_factor=2, mode="nearest"),
                    nn.Conv2d(channels, outgoing, 3, padding=1),
                    nn.GroupNorm(min(32, outgoing), outgoing),
                    nn.SiLU(),
                ]
            )
            channels = outgoing
        self.decoder = nn.Sequential(*blocks, nn.Conv2d(16, 1, 1), nn.Tanh())

    def forward(self, masked):
        return self.decoder(self.encoder.layers(self.encoder.stem(masked)))


def masked_reconstruction_loss(model, clean, mask):
    if mask.shape != clean.shape or mask.dtype != torch.bool:
        raise ValueError("Mask must match image shape and be boolean")
    prediction = model(torch.where(mask, torch.zeros_like(clean), clean))
    if prediction.shape != clean.shape:
        raise ValueError("Decoder output shape mismatch")
    scored = mask & (clean > -0.95)
    counts = scored.sum((1, 2, 3))
    if (counts == 0).any():
        raise ValueError("Mask must intersect foreground in every image")
    return (((prediction - clean).square() * scored).sum((1, 2, 3)) / counts).mean()


def train(
    *,
    train_dir,
    validation_dir,
    output_dir,
    config,
    device,
    on_checkpoint=None,
    resume_from_checkpoint=None,
    initial_checkpoint=None,
):
    config = resolved_config(config)
    output_dir, device = Path(output_dir), torch.device(device)
    output_dir.mkdir(parents=True, exist_ok=True)
    if resume_from_checkpoint is None and any(
        p.name not in {"run_manifest.json", "source_snapshot.zip"} for p in output_dir.iterdir()
    ):
        raise ValueError("Use a fresh pretraining directory")
    source = (
        Path(initial_checkpoint)
        if initial_checkpoint
        else output_dir.parent / config["source_run_id"] / "ddpm_epoch_030.pt"
    )
    if hashlib.sha256(source.read_bytes()).hexdigest() != config["source_sha256"]:
        raise ValueError("Original Gaussian source hash mismatch")
    parent = torch.load(source, map_location="cpu", weights_only=False)
    if parent.get("method") != "ddpm_partial_reconstruction_bmad_flair" or parent["epoch"] != 30:
        raise ValueError("Expected registered Gaussian epoch30 source")
    _seed_everything(config["seed"])
    # Match A's random encoder initialization: construct the same U-Net/feature
    # wrapper in the same RNG order; no U-Net weights are trained in pretraining.
    original = build_model(parent["config"])
    original.load_state_dict(parent["ema"], strict=True)
    composite = FeatureConditionedDDPM(original)
    model = MaskedResNet(composite.encoder).to(device)
    del parent, composite, original
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config["learning_rate"], weight_decay=0, betas=(0.9, 0.999)
    )
    data = ShardDataset(train_dir, "train", include_masks=False)
    val_data = ShardDataset(validation_dir, "val", include_masks=False)
    if np.any(data.labels != 0):
        raise ValueError("Encoder pretraining only permits normal training slices")
    indices = list(range(len(data)))
    if config["max_train_samples"]:
        indices = sorted(
            np.random.default_rng(config["seed"])
            .choice(indices, min(len(indices), config["max_train_samples"]), replace=False)
            .tolist()
        )
    normal = np.flatnonzero(val_data.labels == 0).tolist()
    if config["max_normal_validation_samples"]:
        normal = normal[: config["max_normal_validation_samples"]]
    if not normal:
        raise ValueError("Normal validation missing")
    sampler = ShardBatchSampler(data, config["batch_size"], config["seed"], indices=indices)
    loader = DataLoader(
        data, batch_sampler=sampler, num_workers=0, pin_memory=device.type == "cuda"
    )
    val_loader = DataLoader(
        Subset(val_data, normal), batch_size=config["batch_size"], num_workers=0
    )
    history, best, best_epoch, steps, start_epoch = [], float("inf"), 0, 0, 1
    if resume_from_checkpoint:
        saved = torch.load(resume_from_checkpoint, map_location=device, weights_only=False)
        if saved.get("method") != METHOD or saved["config"] != config:
            raise ValueError("Pretraining resume method/config mismatch")
        model.load_state_dict(saved["model"], strict=True)
        optimizer.load_state_dict(saved["optimizer"])
        history, best, best_epoch, steps = (
            saved["history"],
            saved["best_masked_normal_mse"],
            saved["best_epoch"],
            saved["optimizer_steps"],
        )
        sampler.epoch = saved["sampler_epoch"]
        _restore_rng_state(saved["rng_state"])
        start_epoch = saved["epoch"] + 1
    _write_json(output_dir / "config.json", config)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    def state(epoch):
        return {
            "schema_version": 1,
            "method": METHOD,
            "normal_only": True,
            "dataset_revision": REVISION,
            "epoch": epoch,
            "config": config,
            "model": model.state_dict(),
            "encoder": model.encoder.state_dict(),
            "optimizer": optimizer.state_dict(),
            "rng_state": _rng_state(),
            "sampler_epoch": sampler.epoch,
            "history": history,
            "optimizer_steps": steps,
            "best_epoch": best_epoch,
            "best_masked_normal_mse": best,
        }

    if not resume_from_checkpoint:
        _save_checkpoint(output_dir / "encoder_pretrain_last.pt", state(0))
        if on_checkpoint:
            on_checkpoint(
                {
                    "epoch": 0,
                    "optimizer_steps": 0,
                    "checkpoint": "encoder_pretrain_last.pt",
                    "best_epoch": 0,
                }
            )
    for epoch in range(start_epoch, config["epochs"] + 1):
        started, total, count = time.perf_counter(), 0.0, 0
        model.train()
        for step, (images, _) in enumerate(loader, 1):
            images = images.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = masked_reconstruction_loss(model, images, image_patch_mask(images))
            if not torch.isfinite(loss):
                raise ValueError("Nonfinite encoder pretraining loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total += float(loss.detach()) * len(images)
            count += len(images)
            steps += 1
            if step % config["log_every_steps"] == 0 or step == len(loader):
                print(
                    f"encoder_pretrain epoch={epoch}/{config['epochs']} step={step}/{len(loader)} masked_normal_mse={total / count:.6f}",
                    flush=True,
                )
        val_total, val_count = 0.0, 0
        model.eval()
        with _fixed_rng(config["seed"] + 1_000_000, device), torch.inference_mode():
            for images, _ in val_loader:
                images = images.to(device)
                val_total += float(
                    masked_reconstruction_loss(model, images, image_patch_mask(images))
                ) * len(images)
                val_count += len(images)
        validation = val_total / val_count
        if not math.isfinite(validation):
            raise ValueError("Nonfinite masked normal validation")
        improved = validation < best
        if improved:
            best, best_epoch = validation, epoch
        history.append(
            {
                "epoch": epoch,
                "optimizer_steps": steps,
                "train_samples": count,
                "train_masked_normal_mse": total / count,
                "normal_validation_masked_mse": validation,
                "normal_validation_samples": val_count,
                "seconds": time.perf_counter() - started,
            }
        )
        snapshot = state(epoch)
        _save_checkpoint(output_dir / "encoder_pretrain_last.pt", snapshot)
        if improved:
            _save_checkpoint(output_dir / "encoder_pretrain_best.pt", snapshot)
        if epoch in config["checkpoint_epochs"]:
            _save_checkpoint(output_dir / f"encoder_pretrain_epoch_{epoch:03d}.pt", snapshot)
        _write_json(output_dir / "training_history.json", {"epochs": history})
        print(
            f"encoder_pretrain epoch={epoch} complete normal_validation_masked_mse={validation:.6f}",
            flush=True,
        )
        if on_checkpoint:
            on_checkpoint(
                {
                    "epoch": epoch,
                    "optimizer_steps": steps,
                    "checkpoint": "encoder_pretrain_last.pt",
                    "best_epoch": best_epoch,
                }
            )
    return {
        "method": METHOD,
        "epochs_completed": config["epochs"],
        "optimizer_steps": steps,
        "best_epoch": best_epoch,
        "best_normal_validation_masked_mse": best,
        "checkpoint": "encoder_pretrain_best.pt",
        "resume_checkpoint": "encoder_pretrain_last.pt",
        "train_samples_per_epoch": len(indices),
        "normal_validation_samples": len(normal),
        "training_seconds": sum(row["seconds"] for row in history),
        "peak_vram_bytes": torch.cuda.max_memory_allocated(device) if device.type == "cuda" else 0,
    }
