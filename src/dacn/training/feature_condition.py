"""Experiment A: joint random ResNet-18 + zero-injection DDPM fine-tuning."""

import copy
import hashlib
import math
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from dacn.data.shards import ShardBatchSampler, ShardDataset
from dacn.models.diffusion.feature_condition import (
    BoundCondition,
    FeatureConditionedDDPM,
    update_feature_ema,
)
from dacn.training.ddpm_partial import (
    _fixed_rng,
    _restore_rng_state,
    _rng_state,
    _save_checkpoint,
    _seed_everything,
    _write_json,
    build_diffusion,
    build_model,
)

METHOD = "ddpm_resnet18_joint_condition_v1"
B_METHOD = "ddpm_resnet18_pretrained_condition_v1"
DEFAULTS = {
    "epochs": 10,
    "batch_size": 8,
    "seed": 17,
    "unet_lr": 1e-5,
    "encoder_lr": 1e-4,
    "projection_lr": 1e-4,
    "ema_decay": 0.9992,
    "checkpoint_epochs": [1, 5, 10],
    "log_every_steps": 25,
    "max_train_samples": None,
    "max_normal_validation_samples": None,
    "source_run_id": "ddpm-partial-t4-b8-e30-cont-20261102T034644Z",
    "source_sha256": "d3e06d2082e18234899b201718abd59078c8a800d66da3cc5d8b9e0d9ca9b946",
    "encoder_pretrain_run_id": None,
    "encoder_pretrain_sha256": None,
    "condition_architecture": "global",
}


def resolved_config(config):
    if set(config) - DEFAULTS.keys():
        raise ValueError("Unknown feature-condition config keys")
    result = {**DEFAULTS, **config}
    if result["condition_architecture"] not in {"global", "spatial_imagenet"}:
        raise ValueError("Unknown conditioning architecture")
    if result["condition_architecture"] == "spatial_imagenet" and result["encoder_pretrain_run_id"]:
        raise ValueError("Spatial pilot uses ImageNet weights, not B encoder pretraining")
    for key in ("epochs", "batch_size", "log_every_steps"):
        if isinstance(result[key], bool) or not isinstance(result[key], int) or result[key] < 1:
            raise ValueError("Invalid training budget")
    if result["epochs"] > 10 or result["batch_size"] > 8:
        raise ValueError("Feature fine-tuning budget is at most10 epochs/batch8")
    if bool(result["encoder_pretrain_run_id"]) != bool(result["encoder_pretrain_sha256"]):
        raise ValueError("Encoder pretraining requires both source run and checksum")
    if result["encoder_pretrain_run_id"]:
        import re

        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,63}", result["encoder_pretrain_run_id"]):
            raise ValueError("Invalid encoder pretraining run ID")
    for key in ("unet_lr", "encoder_lr", "projection_lr"):
        if not math.isfinite(result[key]) or not 0 < result[key] <= 1e-3:
            raise ValueError("Invalid learning rate")
    if not 0 <= result["ema_decay"] < 1 or result["seed"] < 0:
        raise ValueError("Invalid EMA/seed")
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


def feature_noise_loss(model, diffusion, images):
    condition = model.encode(images)
    return diffusion.training_loss(BoundCondition(model, condition), images)[0]


@torch.inference_mode()
def feature_reconstruct(model, diffusion, image, *, seed, steps=300, return_noisy=False):
    model.eval()
    condition = model.encode(image)  # fixed features of original input, computed once
    with _fixed_rng(seed, image.device):
        t = torch.full((len(image),), steps - 1, device=image.device, dtype=torch.long)
        noise = diffusion.noise(image, t)
        noisy = diffusion.q_sample(image, t, noise)
        result = diffusion.reconstruct(
            BoundCondition(model, condition), image, steps=steps, initial_noise=noise
        )
    return (result, noisy) if return_noisy else result


def load_feature_checkpoint(path, device="cpu"):
    state = torch.load(path, map_location="cpu", weights_only=False)
    from dacn.models.diffusion.spatial_condition import (
        METHOD as SPATIAL_METHOD,
    )
    from dacn.models.diffusion.spatial_condition import (
        SpatialConditionedDDPM,
    )

    if state.get("method") not in {METHOD, B_METHOD, SPATIAL_METHOD}:
        raise ValueError("Expected feature-conditioned checkpoint")
    factory = (
        SpatialConditionedDDPM if state["method"] == SPATIAL_METHOD else FeatureConditionedDDPM
    )
    model = factory(build_model(state["backbone_config"])).to(device)
    model.load_state_dict(state["ema"], strict=True)
    return model.eval(), build_diffusion(state["backbone_config"])


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
    method = B_METHOD if config["encoder_pretrain_run_id"] else METHOD
    experiment = "B_pretrained" if config["encoder_pretrain_run_id"] else "A_joint"
    spatial = config["condition_architecture"] == "spatial_imagenet"
    if spatial:
        from dacn.models.diffusion.spatial_condition import (
            METHOD as SPATIAL_METHOD,
        )
        from dacn.models.diffusion.spatial_condition import (
            SpatialConditionedDDPM,
        )

        method, experiment = SPATIAL_METHOD, "spatial_imagenet"
    output_dir, device = Path(output_dir), torch.device(device)
    output_dir.mkdir(parents=True, exist_ok=True)
    if resume_from_checkpoint is None and any(
        p.name not in {"run_manifest.json", "source_snapshot.zip"} for p in output_dir.iterdir()
    ):
        raise ValueError("Use a fresh feature training directory")
    source = (
        Path(initial_checkpoint)
        if initial_checkpoint
        else output_dir.parent / config["source_run_id"] / "ddpm_epoch_030.pt"
    )
    if hashlib.sha256(source.read_bytes()).hexdigest() != config["source_sha256"]:
        raise ValueError("Original Gaussian epoch30 SHA mismatch")
    parent = torch.load(source, map_location="cpu", weights_only=False)
    if parent.get("method") != "ddpm_partial_reconstruction_bmad_flair" or parent["epoch"] != 30:
        raise ValueError("Expected original DDPM epoch30")
    diffusion = build_diffusion(parent["config"])
    if diffusion.noise_type != "gaussian":
        raise ValueError("Gaussian source required")
    _seed_everything(config["seed"])
    unet = build_model(parent["config"])
    unet.load_state_dict(parent["ema"], strict=True)
    if not all(torch.equal(unet.state_dict()[key], value) for key, value in parent["ema"].items()):
        raise ValueError("EMA source loading mismatch")
    backbone_config = parent["config"]
    del parent
    model = (
        SpatialConditionedDDPM(unet, pretrained=True) if spatial else FeatureConditionedDDPM(unet)
    ).to(device)
    if config["encoder_pretrain_run_id"]:
        encoder_path = (
            output_dir.parent / config["encoder_pretrain_run_id"] / "encoder_pretrain_best.pt"
        )
        if (
            hashlib.sha256(encoder_path.read_bytes()).hexdigest()
            != config["encoder_pretrain_sha256"]
        ):
            raise ValueError("Pretrained encoder checksum mismatch")
        pretrained = torch.load(encoder_path, map_location="cpu", weights_only=False)
        if (
            pretrained.get("method") != "resnet18_normal_masked_pretrain_v1"
            or pretrained.get("normal_only") is not True
        ):
            raise ValueError("Expected normal-only MRI encoder pretraining")
        if pretrained.get("dataset_revision") != "350c74f2313e4e73ca65e7ffc6e9dcf74040b3b7":
            raise ValueError("Encoder pretraining dataset differs")
        model.encoder.load_state_dict(pretrained["encoder"], strict=True)
        del pretrained
    model.eval()
    with _fixed_rng(config["seed"] + 3_000_000, device), torch.inference_mode():
        sample = torch.zeros(1, 1, 256, 256, device=device)
        t = torch.tensor([299], device=device)
        equivalence_error = float(
            (model.unet(sample, t) - model(sample, t, model.encode(sample))).abs().max()
        )
        if equivalence_error != 0:
            raise ValueError("Zero condition does not preserve original output")
    ema = copy.deepcopy(model).eval().requires_grad_(False)
    optimizer = torch.optim.AdamW(
        [
            {"params": model.unet.parameters(), "lr": config["unet_lr"]},
            {"params": model.encoder.parameters(), "lr": config["encoder_lr"]},
            {
                "params": model.fusion_layers.parameters()
                if spatial
                else list(model.feature_projection.parameters())
                + list(model.condition_projection.parameters()),
                "lr": config["projection_lr"],
            },
        ],
        weight_decay=0,
        betas=(0.9, 0.999),
    )
    data = ShardDataset(train_dir, "train", include_masks=False)
    normal_data = ShardDataset(validation_dir, "val", include_masks=False)
    if np.any(data.labels != 0):
        raise ValueError("Training must use only normal slices")
    indices = list(range(len(data)))
    if config["max_train_samples"]:
        indices = sorted(
            np.random.default_rng(config["seed"])
            .choice(indices, min(len(indices), config["max_train_samples"]), replace=False)
            .tolist()
        )
    normal = np.flatnonzero(normal_data.labels == 0).tolist()
    if config["max_normal_validation_samples"]:
        normal = normal[: config["max_normal_validation_samples"]]
    if not normal:
        raise ValueError("Normal validation missing")
    # ShardBatchSampler retains the original dataset and a fixed explicit subset.
    sampler = ShardBatchSampler(data, config["batch_size"], config["seed"], indices=indices)
    loader = DataLoader(
        data, batch_sampler=sampler, num_workers=0, pin_memory=device.type == "cuda"
    )
    val_loader = DataLoader(
        Subset(normal_data, normal), batch_size=config["batch_size"], num_workers=0
    )
    history, best, best_epoch, steps, start_epoch = [], float("inf"), 0, 0, 1
    if resume_from_checkpoint:
        saved = torch.load(resume_from_checkpoint, map_location=device, weights_only=False)
        if saved.get("method") != method or saved["config"] != config:
            raise ValueError("Conditional resume method/config mismatch")
        model.load_state_dict(saved["model"], strict=True)
        ema.load_state_dict(saved["ema"], strict=True)
        optimizer.load_state_dict(saved["optimizer"])
        history, best, best_epoch, steps = (
            saved["history"],
            saved["best_normal_validation_epsilon_mse"],
            saved["best_epoch"],
            saved["optimizer_steps"],
        )
        sampler.epoch = saved["sampler_epoch"]
        _restore_rng_state(saved["rng_state"])
        start_epoch = saved["epoch"] + 1
    _write_json(output_dir / "config.json", config)
    _write_json(
        output_dir / "initialization.json",
        {
            "source_weight": "ema",
            "source_epoch": 30,
            "source_sha256": config["source_sha256"],
            "encoder_initialization": "ResNet18 IMAGENET1K_V1 spatial, no pooling"
            if spatial
            else "normal-only masked MRI pretraining"
            if config["encoder_pretrain_run_id"]
            else "random, no external weights or pretraining",
            "encoder_pretrain_run_id": config["encoder_pretrain_run_id"],
            "encoder_pretrain_sha256": config["encoder_pretrain_sha256"],
            "experiment": experiment,
            "zero_injection_max_output_error": equivalence_error,
            "trainable": ["unet", "encoder", "fusion_layers"]
            if spatial
            else ["unet", "encoder", "feature_projection", "condition_projection"],
            "batchnorm_ema_buffers": "copy online buffers after each update",
        },
    )
    if spatial:
        from dacn.models.diffusion.spatial_condition import WEIGHTS_SHA256, WEIGHTS_URL

        _write_json(
            output_dir / "imagenet_provenance.json",
            {
                "weights": "ResNet18_Weights.IMAGENET1K_V1",
                "url": WEIGHTS_URL,
                "sha256": WEIGHTS_SHA256,
                "input": "1-channel MRI[-1,1]; conv1=mean RGB ImageNet kernels; no RGB repeat/normalization/resize/crop",
                "spatial_features": [[64, 64, 64], [128, 32, 32], [256, 16, 16], [512, 8, 8]],
                "fusion": "zero1x1 residual at last encoder block of each scale; propagated through skips",
                "fusion_indices": model.fusion_indices,
            },
        )
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    def state(epoch):
        return {
            "schema_version": 1,
            "method": method,
            "epoch": epoch,
            "source_epoch": 30,
            "config": config,
            "backbone_config": backbone_config,
            "model": model.state_dict(),
            "ema": ema.state_dict(),
            "optimizer": optimizer.state_dict(),
            "rng_state": _rng_state(),
            "sampler_epoch": sampler.epoch,
            "optimizer_steps": steps,
            "history": history,
            "best_epoch": best_epoch,
            "best_normal_validation_epsilon_mse": best,
        }

    if not resume_from_checkpoint:
        _save_checkpoint(output_dir / "feature_condition_last.pt", state(0))
        if on_checkpoint:
            on_checkpoint(
                {
                    "epoch": 0,
                    "optimizer_steps": 0,
                    "checkpoint": "feature_condition_last.pt",
                    "best_epoch": 0,
                }
            )
    for epoch in range(start_epoch, config["epochs"] + 1):
        started, total, count = time.perf_counter(), 0.0, 0
        model.train()
        for step, (images, _) in enumerate(loader, 1):
            images = images.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = feature_noise_loss(model, diffusion, images)
            if not torch.isfinite(loss):
                raise ValueError("Nonfinite conditional loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            update_feature_ema(ema, model, config["ema_decay"])
            total += float(loss.detach()) * len(images)
            count += len(images)
            steps += 1
            if step % config["log_every_steps"] == 0 or step == len(loader):
                print(
                    f"feature_{experiment} epoch={epoch}/{config['epochs']} step={step}/{len(loader)} samples={count}/{len(indices)} epsilon_mse={total / count:.6f}",
                    flush=True,
                )
        val_total, val_count = 0.0, 0
        with _fixed_rng(config["seed"] + 1_000_000, device), torch.inference_mode():
            ema.eval()
            for images, _ in val_loader:
                images = images.to(device)
                val_total += float(feature_noise_loss(ema, diffusion, images)) * len(images)
                val_count += len(images)
        validation = val_total / val_count
        if not math.isfinite(validation):
            raise ValueError("Nonfinite conditional normal-validation loss")
        improved = validation < best
        if improved:
            best, best_epoch = validation, epoch
        history.append(
            {
                "epoch": epoch,
                "optimizer_steps": steps,
                "train_samples": count,
                "train_epsilon_mse": total / count,
                "normal_validation_epsilon_mse": validation,
                "normal_validation_samples": val_count,
                "seconds": time.perf_counter() - started,
            }
        )
        snapshot = state(epoch)
        _save_checkpoint(output_dir / "feature_condition_last.pt", snapshot)
        if improved:
            _save_checkpoint(output_dir / "feature_condition_best.pt", snapshot)
        if epoch in config["checkpoint_epochs"]:
            _save_checkpoint(output_dir / f"feature_condition_epoch_{epoch:03d}.pt", snapshot)
        _write_json(output_dir / "training_history.json", {"epochs": history})
        print(
            f"feature_{experiment} epoch={epoch} complete normal_validation_epsilon_mse={validation:.8f}",
            flush=True,
        )
        if on_checkpoint:
            on_checkpoint(
                {
                    "epoch": epoch,
                    "optimizer_steps": steps,
                    "checkpoint": "feature_condition_last.pt",
                    "best_epoch": best_epoch,
                }
            )
    return {
        "method": method,
        "experiment": experiment,
        "epochs_completed": config["epochs"],
        "optimizer_steps": steps,
        "best_epoch": best_epoch,
        "best_normal_validation_epsilon_mse": best,
        "checkpoint": "feature_condition_best.pt",
        "resume_checkpoint": "feature_condition_last.pt",
        "train_samples_per_epoch": len(indices),
        "normal_validation_samples": len(normal),
        "training_seconds": sum(row["seconds"] for row in history),
        "peak_vram_bytes": torch.cuda.max_memory_allocated(device) if device.type == "cuda" else 0,
    }
