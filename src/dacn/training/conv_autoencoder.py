"""Normal-only ConvAE/ConvVAE training; checkpoint selection uses normal validation MSE."""

import math
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from dacn.data.shards import ShardBatchSampler, ShardDataset
from dacn.models.conv_autoencoder import ConvAutoencoder, loss_terms
from dacn.training.ddpm_partial import (
    _restore_rng_state,
    _rng_state,
    _save_checkpoint,
    _seed_everything,
    _write_json,
)

DEFAULTS = {
    "model_type": "conv_ae",
    "latent_dim": 256,
    "width": 32,
    "epochs": 30,
    "batch_size": 32,
    "learning_rate": 1e-4,
    "weight_decay": 0.0,
    "beta": 1.0,
    "gradient_clip_norm": 1.0,
    "seed": 17,
    "max_train_samples": None,
    "max_validation_samples": None,
}


def resolve_config(values):
    if not isinstance(values, dict) or set(values) - set(DEFAULTS):
        raise ValueError("Unknown autoencoder configuration keys")
    result = {**DEFAULTS, **values}
    if result["model_type"] not in {"conv_ae", "conv_vae"}:
        raise ValueError("Require conv_ae or conv_vae")
    for key in ("latent_dim", "width", "epochs", "batch_size", "seed"):
        if type(result[key]) is not int or result[key] < 1:
            raise ValueError("Require positive integer config field: " + key)
    if result["latent_dim"] != 256 or result["width"] < 8 or result["width"] % 8:
        raise ValueError("Require latent256 and width multiple of8")
    for key in ("max_train_samples", "max_validation_samples"):
        if result[key] is not None and (type(result[key]) is not int or result[key] < 1):
            raise ValueError("Invalid bounded subset size")
    for key in ("learning_rate", "weight_decay", "beta", "gradient_clip_norm"):
        value = result[key]
        if type(value) not in (float, int) or not math.isfinite(value) or value < 0:
            raise ValueError("Require finite nonnegative config field: " + key)
        result[key] = float(value)
    if result["learning_rate"] <= 0 or result["gradient_clip_norm"] <= 0 or result["beta"] != 1:
        raise ValueError("Require positive LR/clip and registered beta1")
    return result


def selected_indices(labels, limit, seed):
    selected = np.flatnonzero(np.asarray(labels) == 0)
    if limit is not None and len(selected) > limit:
        selected = np.sort(np.random.default_rng(seed).choice(selected, limit, replace=False))
    return selected.tolist()


@torch.inference_mode()
def normal_validation(model, loader, device):
    model.eval()
    mse_sum, kl_sum, count = 0.0, 0.0, 0
    for images, labels in loader:
        if (labels != 0).any():
            raise ValueError("Only normal validation allowed for checkpoint selection")
        images = images.to(device)
        reconstructed, mean, logvar = model(images, sample=False)
        _, mse, kl = loss_terms(reconstructed, images, mean, logvar, model_type=model.model_type)
        mse_sum += float(mse) * len(images)
        kl_sum += float(kl) * len(images)
        count += len(images)
    if not count:
        raise ValueError("No normal validation examples")
    return {
        "reconstruction_mse": mse_sum / count,
        "kl_per_sample": kl_sum / count,
        "normal_samples": count,
        "latent_mode": "mean; deterministic",
    }


def train(
    *,
    train_dir,
    validation_dir,
    output_dir,
    config,
    device,
    resume_from_checkpoint=None,
    on_checkpoint=None,
):
    config = resolve_config(config)
    _seed_everything(config["seed"])
    device, output = torch.device(device), Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    data = ShardDataset(train_dir, "train", include_masks=False)
    validation = ShardDataset(validation_dir, "val", include_masks=False)
    if np.any(data.labels != 0):
        raise ValueError("Autoencoder training split must be normal-only")
    indices = selected_indices(data.labels, config["max_train_samples"], config["seed"])
    val_indices = selected_indices(
        validation.labels, config["max_validation_samples"], config["seed"] + 1
    )
    if not indices or not val_indices:
        raise ValueError("Require normal training and validation examples")
    sampler = ShardBatchSampler(data, config["batch_size"], config["seed"], indices=indices)
    loader = DataLoader(
        data, batch_sampler=sampler, num_workers=0, pin_memory=device.type == "cuda"
    )
    val_loader = DataLoader(
        Subset(validation, val_indices),
        batch_size=config["batch_size"],
        shuffle=False,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )
    model = ConvAutoencoder(config["model_type"], config["latent_dim"], config["width"]).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=config["learning_rate"], weight_decay=config["weight_decay"]
    )
    history, best, best_epoch, completed, updates = [], float("inf"), 0, 0, 0
    training_seconds = 0.0
    if resume_from_checkpoint is not None:
        saved = torch.load(resume_from_checkpoint, map_location="cpu", weights_only=False)
        if saved.get("config") != config or saved.get("trainer") != "conv_autoencoder_v1":
            raise ValueError("Resume config/trainer differs")
        model.load_state_dict(saved["model"], strict=True)
        optimizer.load_state_dict(saved["optimizer"])
        completed, updates = saved["epoch"], saved["optimizer_steps"]
        history, best, best_epoch = (
            saved["history"],
            saved["best_normal_validation_mse"],
            saved["best_epoch"],
        )
        training_seconds = saved["training_seconds"]
        sampler.epoch = saved["sampler_epoch"]
        _restore_rng_state(saved["rng_state"])
    architecture = {
        "model_type": config["model_type"],
        "latent_dim": 256,
        "encoder_channels": [1, *model.channels],
        "spatial_levels": [256, 128, 64, 32, 16, 8, 4],
        "skip_connections": False,
        "output_activation": "tanh",
        "normalization": "GroupNorm8",
        "parameters": sum(p.numel() for p in model.parameters()),
        "objective": "crop240MSE + beta*sum_latent_KL/(batch*57600); beta1; AE has noKL",
        "vae_logvar_clamp": [-20, 10],
        "optimizer": "Adam, noAMP",
        "checkpoint_selection": "lowest normal-validation crop240 reconstruction MSE; first on ties",
        "training_samples": len(indices),
        "normal_validation_samples": len(val_indices),
        "early_stopping": False,
        "test_path_accepted": False,
    }
    _write_json(output / "architecture.json", architecture)
    _write_json(output / "config.json", config)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    for epoch in range(completed + 1, config["epochs"] + 1):
        tick = time.perf_counter()
        model.train()
        objective_sum, mse_sum, kl_sum, count = 0.0, 0.0, 0.0, 0
        for images, labels in loader:
            if (labels != 0).any():
                raise ValueError("Unexpected anomalous training sample")
            images = images.to(device)
            optimizer.zero_grad(set_to_none=True)
            reconstruction, mean, logvar = model(images)
            loss, mse, kl = loss_terms(
                reconstruction,
                images,
                mean,
                logvar,
                model_type=model.model_type,
                beta=config["beta"],
            )
            if not all(torch.isfinite(v) for v in (loss, mse, kl)):
                raise ValueError("Non-finite training objective")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), config["gradient_clip_norm"], error_if_nonfinite=True
            )
            optimizer.step()
            updates += 1
            objective_sum += float(loss.detach()) * len(images)
            mse_sum += float(mse.detach()) * len(images)
            kl_sum += float(kl.detach()) * len(images)
            count += len(images)
        if count != len(indices):
            raise ValueError("Epoch did not consume every selected normal training sample")
        measured = normal_validation(model, val_loader, device)
        if not all(math.isfinite(measured[key]) for key in ("reconstruction_mse", "kl_per_sample")):
            raise ValueError("Non-finite normal-validation objective")
        seconds = time.perf_counter() - tick
        training_seconds += seconds
        history.append(
            {
                "epoch": epoch,
                "optimizer_steps": updates,
                "training_samples": count,
                "train_objective": objective_sum / count,
                "train_reconstruction_mse": mse_sum / count,
                "train_kl_per_sample": kl_sum / count,
                "normal_validation": measured,
                "seconds": seconds,
            }
        )
        improved = measured["reconstruction_mse"] < best
        if improved:
            best, best_epoch = measured["reconstruction_mse"], epoch
        _write_json(output / "training_history.json", {"epochs": history})
        state = {
            "schema_version": 1,
            "trainer": "conv_autoencoder_v1",
            "config": config,
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "optimizer_steps": updates,
            "sampler_epoch": sampler.epoch,
            "rng_state": _rng_state(),
            "history": history,
            "training_seconds": training_seconds,
            "best_epoch": best_epoch,
            "best_normal_validation_mse": best,
        }
        _save_checkpoint(output / "autoencoder_last.pt", state)
        if improved:
            _save_checkpoint(output / "autoencoder_best.pt", state)
        if epoch in {1, 5, 10, 20, 30}:
            _save_checkpoint(output / f"autoencoder_epoch_{epoch:03d}.pt", state)
        if on_checkpoint:
            on_checkpoint(
                {
                    "epoch": epoch,
                    "optimizer_steps": updates,
                    "checkpoint": "autoencoder_last.pt",
                    "best_epoch": best_epoch,
                }
            )
        print(
            f"model={config['model_type']} epoch={epoch}/{config['epochs']} "
            f"train_mse={mse_sum / count:.8f} normal_val_mse={measured['reconstruction_mse']:.8f} "
            f"kl_per_sample={kl_sum / count:.6f}",
            flush=True,
        )
    result = {
        "model_type": config["model_type"],
        "epochs_completed": len(history),
        "latent_dim": 256,
        "best_epoch": best_epoch,
        "best_normal_validation_mse": best,
        "optimizer_steps": updates,
        "training_samples_per_epoch": len(indices),
        "normal_validation_samples": len(val_indices),
        "training_seconds": training_seconds,
        "checkpoint": "autoencoder_best.pt",
        "last_checkpoint": "autoencoder_last.pt",
        "training_history": "training_history.json",
        "selection_metric": "normal_validation_reconstruction_mse",
        "early_stopping": False,
        "anomaly_metrics_produced": False,
        "test_used": False,
        "peak_vram_allocated_bytes": (
            int(torch.cuda.max_memory_allocated()) if device.type == "cuda" else None
        ),
    }
    _write_json(output / "training_results.json", result)
    return result
