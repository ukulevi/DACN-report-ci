"""Three-channel patch/context DDPM fine-tuning; an adaptation, not exact pDDPM."""

import copy
import hashlib
import math
import re
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from dacn.data.shards import ShardBatchSampler, ShardDataset
from dacn.models.diffusion.unet import update_ema_params
from dacn.training.ddpm_partial import (
    _fixed_rng,
    _rng_state,
    _save_checkpoint,
    _seed_everything,
    _write_json,
    build_diffusion,
    build_model,
)

METHOD = "ddpm_patch_context_finetune_v1"
DEFAULTS = {
    "seed": 17,
    "epochs": 5,
    "batch_size": 8,
    "learning_rate": 1e-5,
    "ema_decay": 0.9992,
    "patch_size": 64,
    "log_every_steps": 25,
    "checkpoint_epochs": [1, 3, 5],
    "initial_run_id": "ddpm-partial-t4-b8-e30-cont-20261102T034644Z",
    "initial_checkpoint_sha256": "d3e06d2082e18234899b201718abd59078c8a800d66da3cc5d8b9e0d9ca9b946",
}


def _resolved_config(config):
    if not isinstance(config, dict) or set(config) - set(DEFAULTS):
        raise ValueError("Unknown patch-context config")
    config = {**DEFAULTS, **config}
    for k in ("seed", "epochs", "batch_size", "patch_size", "log_every_steps"):
        if isinstance(config[k], bool) or not isinstance(config[k], int) or config[k] < 1:
            raise ValueError(f"Invalid {k}")
    if config["epochs"] > 5 or config["batch_size"] > 8:
        raise ValueError("Pilot is limited to5 epochs and batch8")
    if config["patch_size"] > 240:
        raise ValueError("Patch cannot exceed original crop")
    for k in ("learning_rate", "ema_decay"):
        if (
            isinstance(config[k], bool)
            or not isinstance(config[k], (int, float))
            or not math.isfinite(config[k])
        ):
            raise ValueError(f"Invalid {k}")
    if not 0 < config["learning_rate"] <= 1e-4 or not 0 < config["ema_decay"] < 1:
        raise ValueError("Invalid learning rate or EMA")
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,63}", config["initial_run_id"]):
        raise ValueError("Invalid parent run ID")
    if not re.fullmatch(r"[0-9a-f]{64}", config["initial_checkpoint_sha256"]):
        raise ValueError("Invalid parent checksum")
    milestones = config["checkpoint_epochs"]
    if not isinstance(milestones, list) or any(
        isinstance(e, bool) or not isinstance(e, int) or not 1 <= e <= config["epochs"]
        for e in milestones
    ):
        raise ValueError("Invalid milestones")
    return config


def expand_input(model):
    """Preserve old image-channel kernel exactly; initialize context/mask kernels0."""
    old = model.down[0][0]
    if old.in_channels != 1 or model.out_channels != 1:
        raise ValueError("Expected original one-channel U-Net")
    new = torch.nn.Conv2d(
        3,
        old.out_channels,
        old.kernel_size,
        stride=old.stride,
        padding=old.padding,
        bias=old.bias is not None,
    ).to(old.weight.device)
    with torch.no_grad():
        new.weight.zero_()
        new.weight[:, :1].copy_(old.weight)
        if old.bias is not None:
            new.bias.copy_(old.bias)
    model.down[0][0] = new
    model.in_channels = 3
    return model


def random_patch_mask(images, size):
    """One square per image, centered on an input-derived foreground pixel."""
    mask = torch.zeros_like(images, dtype=torch.bool)
    if not 1 <= size <= 240:
        raise ValueError("Invalid patch size")
    for i in range(len(images)):
        candidates = torch.nonzero(images[i, 0, 8:-8, 8:-8] > -0.95)
        if not len(candidates):
            raise ValueError("Empty image foreground")
        pixel = candidates[torch.randint(len(candidates), (1,), device=images.device).item()] + 8
        y = int(torch.clamp(pixel[0] - size // 2, 8, 248 - size).item())
        x = int(torch.clamp(pixel[1] - size // 2, 8, 248 - size).item())
        mask[i, :, y : y + size, x : x + size] = True
    return mask


def conditioned_input(noisy, clean, mask):
    if noisy.shape != clean.shape or mask.shape != clean.shape or mask.dtype != torch.bool:
        raise ValueError("Condition inputs must have matching shape and bool mask")
    context = torch.where(mask, torch.zeros_like(clean), clean)
    first = torch.where(mask, noisy, context)
    return torch.cat((first, context, mask.to(clean.dtype)), dim=1)


def patch_noise_loss(model, diffusion, clean, mask, *, noise=None, timesteps=None):
    if noise is None:
        noise = torch.randn_like(clean)
    if timesteps is None:
        timesteps = torch.randint(diffusion.timesteps, (len(clean),), device=clean.device)
    noisy = diffusion.q_sample(clean, timesteps, noise)
    predicted = model(conditioned_input(noisy, clean, mask), timesteps)
    if predicted.shape != clean.shape:
        raise ValueError("Context model must predict one noise channel")
    counts = mask.sum(dim=(1, 2, 3))
    if (counts == 0).any():
        raise ValueError("Each sample needs a nonempty patch")
    per_sample = ((predicted - noise).square() * mask).sum(dim=(1, 2, 3)) / counts
    return per_sample.mean()


@torch.inference_mode()
def reconstruct_patch(model, diffusion, image, mask, *, steps=100, seed=17, one_step=False):
    """Condition throughout the reverse chain; return unchanged context outside mask."""

    class Adapter(torch.nn.Module):
        def forward(self, noisy, t):
            return model(conditioned_input(noisy, image, mask), t)

    model.eval()
    with _fixed_rng(seed, image.device):
        if one_step:
            if not 1 <= steps <= diffusion.timesteps:
                raise ValueError("steps must lie within the diffusion schedule")
            t = torch.full((len(image),), steps - 1, device=image.device, dtype=torch.long)
            noisy = diffusion.q_sample(image, t, diffusion.noise(image, t))
            predicted = model(conditioned_input(noisy, image, mask), t)
            result = diffusion._predict_clean(noisy, t, predicted).clamp(-1, 1)
        else:
            result = diffusion.reconstruct(Adapter(), image, steps=steps)
    return torch.where(mask, result, image)


@torch.inference_mode()
def sliding_reconstruct(
    model, diffusion, image, *, steps=100, seed=17, patch_size=64, stride=48, one_step=False
):
    """Full original crop, equal overlap averaging; no anomaly mask is required."""
    if image.shape != (1, 1, 256, 256) or not 1 <= stride <= patch_size <= 240:
        raise ValueError("Invalid sliding reconstruction contract")
    positions = list(range(8, 249 - patch_size, stride))
    if positions[-1] != 248 - patch_size:
        positions.append(248 - patch_size)
    total = torch.zeros_like(image)
    counts = torch.zeros_like(image)
    for row, y in enumerate(positions):
        for col, x in enumerate(positions):
            mask = torch.zeros_like(image, dtype=torch.bool)
            mask[:, :, y : y + patch_size, x : x + patch_size] = True
            value = reconstruct_patch(
                model,
                diffusion,
                image,
                mask,
                steps=steps,
                seed=(seed + row * len(positions) + col) % 2**32,
                one_step=one_step,
            )
            total += value * mask
            counts += mask
    assert (counts[:, :, 8:-8, 8:-8] > 0).all()
    return torch.where(counts > 0, total / counts.clamp_min(1), image)


def load_context_checkpoint(path, device="cpu"):
    state = torch.load(path, map_location="cpu", weights_only=False)
    if state.get("method") != METHOD:
        raise ValueError("Wrong checkpoint method")
    model = expand_input(build_model(state["backbone_config"], device))
    model.load_state_dict(state["ema"], strict=True)
    model.eval()
    return model, build_diffusion(state["backbone_config"])


def train(
    *,
    train_dir,
    validation_dir,
    output_dir,
    config,
    device,
    on_checkpoint=None,
    initial_checkpoint=None,
):
    config = _resolved_config(config)
    device = torch.device(device)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if any(
        p.name not in {"run_manifest.json", "source_snapshot.zip"} for p in output_dir.iterdir()
    ):
        raise ValueError("Training output already contains artifacts")
    source = (
        Path(initial_checkpoint)
        if initial_checkpoint is not None
        else output_dir.parent / config["initial_run_id"] / "ddpm_epoch_030.pt"
    )
    if hashlib.sha256(source.read_bytes()).hexdigest() != config["initial_checkpoint_sha256"]:
        raise ValueError("Parent checkpoint checksum mismatch")
    parent = torch.load(source, map_location="cpu", weights_only=False)
    if (
        parent.get("epoch") != 30
        or parent.get("method") != "ddpm_partial_reconstruction_bmad_flair"
    ):
        raise ValueError("Expected Gaussian epoch30 source")
    diffusion = build_diffusion(parent["config"])
    if diffusion.noise_type != "gaussian":
        raise ValueError("Pilot requires Gaussian source")
    _seed_everything(config["seed"])
    model = build_model(parent["config"], device)
    model.load_state_dict(parent["ema"], strict=True)
    model = expand_input(model)
    ema = copy.deepcopy(model).eval()
    for p in ema.parameters():
        p.requires_grad_(False)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config["learning_rate"], weight_decay=0, betas=(0.9, 0.999)
    )
    train_data = ShardDataset(train_dir, "train", include_masks=False)
    val_data = ShardDataset(validation_dir, "val", include_masks=False)
    if np.any(train_data.labels != 0):
        raise ValueError("Training requires normal slices only")
    normal = np.flatnonzero(val_data.labels == 0).tolist()
    if not normal:
        raise ValueError("Normal validation is empty")
    sampler = ShardBatchSampler(train_data, config["batch_size"], config["seed"])
    loader = DataLoader(
        train_data, batch_sampler=sampler, num_workers=0, pin_memory=device.type == "cuda"
    )
    val_loader = DataLoader(
        Subset(val_data, normal), batch_size=config["batch_size"], num_workers=0
    )
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    history = []
    best = float("inf")
    best_epoch = 0
    steps = 0
    _write_json(output_dir / "config.json", config)
    _write_json(
        output_dir / "initialization.json",
        {
            "source_run_id": config["initial_run_id"],
            "source_checkpoint": "ddpm_epoch_030.pt",
            "source_sha256": config["initial_checkpoint_sha256"],
            "source_weight": "ema",
            "source_epoch": 30,
            "optimizer_reset": True,
            "input_channels": 3,
            "output_channels": 1,
            "new_channel_initialization": "zero",
        },
    )
    for epoch in range(1, config["epochs"] + 1):
        start = time.perf_counter()
        model.train()
        total = 0.0
        count = 0
        for batch_index, (images, _) in enumerate(loader, 1):
            images = images.to(device)
            mask = random_patch_mask(images, config["patch_size"])
            optimizer.zero_grad(set_to_none=True)
            loss = patch_noise_loss(model, diffusion, images, mask)
            if not torch.isfinite(loss):
                raise ValueError("Nonfinite training loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            update_ema_params(ema, model, decay_rate=config["ema_decay"])
            total += float(loss.detach()) * len(images)
            count += len(images)
            steps += 1
            if batch_index % config["log_every_steps"] == 0 or batch_index == len(loader):
                print(
                    f"finetune_epoch={epoch}/{config['epochs']} step={batch_index}/{len(loader)} samples={count}/{len(train_data)} patch_epsilon_mse={total / count:.6f}",
                    flush=True,
                )
        val_total = 0.0
        val_count = 0
        with _fixed_rng(config["seed"] + 1_000_000, device), torch.inference_mode():
            ema.eval()
            for images, _ in val_loader:
                images = images.to(device)
                mask = random_patch_mask(images, config["patch_size"])
                loss = patch_noise_loss(ema, diffusion, images, mask)
                val_total += float(loss) * len(images)
                val_count += len(images)
        val = val_total / val_count
        if not math.isfinite(val):
            raise ValueError("Nonfinite validation loss")
        improved = val < best
        if improved:
            best, best_epoch = val, epoch
        history.append(
            {
                "epoch": epoch,
                "optimizer_steps": steps,
                "train_samples": count,
                "train_patch_epsilon_mse": total / count,
                "normal_validation_patch_epsilon_mse": val,
                "normal_validation_samples": val_count,
                "seconds": time.perf_counter() - start,
            }
        )
        state = {
            "schema_version": 1,
            "method": METHOD,
            "epoch": epoch,
            "optimizer_steps": steps,
            "model": model.state_dict(),
            "ema": ema.state_dict(),
            "optimizer": optimizer.state_dict(),
            "rng_state": _rng_state(),
            "sampler_epoch": sampler.epoch,
            "history": history,
            "config": config,
            "backbone_config": parent["config"],
            "source_epoch": 30,
            "best_epoch": best_epoch,
            "best_normal_validation_patch_epsilon_mse": best,
        }
        _save_checkpoint(output_dir / "patch_context_last.pt", state)
        if improved:
            _save_checkpoint(output_dir / "patch_context_best.pt", state)
        if epoch in config["checkpoint_epochs"]:
            _save_checkpoint(output_dir / f"patch_context_epoch_{epoch:03d}.pt", state)
        _write_json(output_dir / "training_history.json", {"epochs": history})
        print(
            f"finetune_epoch={epoch} complete normal_validation_patch_epsilon_mse={val:.6f}",
            flush=True,
        )
        if on_checkpoint:
            on_checkpoint(
                {
                    "epoch": epoch,
                    "optimizer_steps": steps,
                    "checkpoint": "patch_context_last.pt",
                    "best_epoch": best_epoch,
                }
            )
    # Bounded normal-only conditional one-step preview, not an anomaly evaluation.
    with _fixed_rng(config["seed"] + 2_000_000, device), torch.inference_mode():
        images = torch.stack([val_data[i][0] for i in normal[:4]]).to(device)
        mask = random_patch_mask(images, config["patch_size"])
        t = torch.full((len(images),), 99, device=device, dtype=torch.long)
        noise = torch.randn_like(images)
        noisy = diffusion.q_sample(images, t, noise)
        predicted = ema(conditioned_input(noisy, images, mask), t)
        preview = torch.where(mask, diffusion._predict_clean(noisy, t, predicted), images)
        np.savez_compressed(
            output_dir / "private_normal_preview.npz",
            images=images.cpu().numpy(),
            reconstructions=preview.cpu().numpy(),
            patch_mask=mask.cpu().numpy(),
        )
    return {
        "method": METHOD,
        "epochs_completed": config["epochs"],
        "optimizer_steps": steps,
        "source_epoch": 30,
        "best_epoch": best_epoch,
        "best_normal_validation_patch_epsilon_mse": best,
        "checkpoint": "patch_context_best.pt",
        "resume_checkpoint": "patch_context_last.pt",
        "training_seconds": sum(r["seconds"] for r in history),
        "peak_vram_bytes": int(torch.cuda.max_memory_allocated(device))
        if device.type == "cuda"
        else 0,
        "train_samples_per_epoch": len(train_data),
        "normal_validation_samples": val_count,
    }
