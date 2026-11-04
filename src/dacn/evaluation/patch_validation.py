"""Fixed validation scoring for patch/context and original DDPM one-step controls."""

import csv
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.model_selection import StratifiedGroupKFold

from dacn.data.shards import ShardDataset
from dacn.evaluation.scores import paired_score_maps
from dacn.evaluation.validation import evaluate_maps
from dacn.training.ddpm_partial import (
    _fixed_rng,
    build_diffusion,
    derive_sampling_seed,
    load_ema_checkpoint,
)
from dacn.training.patch_context import load_context_checkpoint, sliding_reconstruct


@torch.inference_mode()
def gaussian_one_step(model, diffusion, image, *, steps=100, seed=17):
    """Predict x0 once from the same full-image forward draw as the DDPM baseline."""
    if image.shape != (1, 1, 256, 256) or not 1 <= steps <= diffusion.timesteps:
        raise ValueError("Invalid input shape or timestep")
    if diffusion.noise_type != "gaussian":
        raise ValueError("This control requires Gaussian noise")
    model.eval()
    with _fixed_rng(seed, image.device):
        t = torch.full((1,), steps - 1, device=image.device, dtype=torch.long)
        noisy = diffusion.q_sample(image, t, diffusion.noise(image, t))
        return diffusion._predict_clean(noisy, t, model(noisy, t)).clamp(-1, 1)


def evaluate_patch_validation(
    *,
    validation_dir,
    checkpoint_path,
    output_dir,
    device,
    on_progress=None,
    one_step=False,
    gaussian_baseline=False,
):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)
    if gaussian_baseline:
        if not one_step:
            raise ValueError("Gaussian control is one-step only")
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        if (
            checkpoint["epoch"] != 30
            or checkpoint["method"] != "ddpm_partial_reconstruction_bmad_flair"
        ):
            raise ValueError("Expected registered Gaussian epoch30 checkpoint")
        model = load_ema_checkpoint(checkpoint_path, checkpoint["config"], device)
        diffusion = build_diffusion(checkpoint["config"])
        del checkpoint
    else:
        model, diffusion = load_context_checkpoint(checkpoint_path, device)
    data = ShardDataset(validation_dir, "val", include_masks=True)
    with (validation_dir / "index.csv").open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    if [row["sample_id"] for row in rows] != data.sample_ids:
        raise ValueError("Group index order differs from validation data")
    groups = np.asarray([row["patient_group_id"] for row in rows])
    calibration, holdout = next(
        StratifiedGroupKFold(2, shuffle=True, random_state=17).split(
            np.zeros(len(data)), data.labels, groups
        )
    )
    if not set(groups[calibration]).isdisjoint(groups[holdout]):
        raise ValueError("Patient-group leakage")
    if any(len(np.unique(data.labels[idx])) != 2 for idx in (calibration, holdout)):
        raise ValueError("Both partitions require both classes")
    images, reconstructions, masks = [], [], []
    maps = {name: [] for name in ("pixel_difference", "local_ssim", "brightness_control")}
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)
    started = time.perf_counter()
    reconstruction_kwargs = {"one_step": True} if one_step else {}
    reconstruct = gaussian_one_step if gaussian_baseline else sliding_reconstruct
    if gaussian_baseline:
        reconstruction_kwargs = {}
    for index in range(len(data)):
        image, mask, _ = data[index]
        reconstructed = (
            reconstruct(
                model,
                diffusion,
                image[None].to(device),
                steps=100,
                seed=derive_sampling_seed(17, data.sample_ids[index]),
                **reconstruction_kwargs,
            )[0, 0]
            .cpu()
            .numpy()
        )
        x = image[0].numpy()
        scores = paired_score_maps(x, reconstructed)
        images.append(x)
        reconstructions.append(reconstructed)
        masks.append(mask.numpy())
        for name in ("pixel_difference", "local_ssim"):
            maps[name].append(scores[name])
        maps["brightness_control"].append(x[8:-8, 8:-8])
        completed = index + 1
        if completed == 1 or completed % 5 == 0 or completed == len(data):
            # Durable prefix cache prevents losing all reconstruction work on interruption.
            np.savez_compressed(
                output_dir / "private_reconstructions.npz",
                images=np.asarray(images),
                reconstructions=np.asarray(reconstructions),
                masks=np.asarray(masks),
                labels=data.labels[:completed],
            )
            progress = {
                "completed_samples": completed,
                "total_samples": len(data),
                "elapsed_seconds": time.perf_counter() - started,
            }
            (output_dir / "progress.json").write_text(json.dumps(progress) + "\n")
            if on_progress is not None:
                on_progress(progress)
            print(json.dumps(progress), flush=True)
    inference_seconds = time.perf_counter() - started
    cropped_masks = np.asarray(masks)[:, 8:-8, 8:-8]
    result = {
        "method": "gaussian_ddpm_full_image_one_step_x0"
        if gaussian_baseline
        else "patch_context_sliding_one_step_x0"
        if one_step
        else "patch_context_sliding_reconstruction",
        "reconstruction_mode": "one_step_x0" if one_step else "ancestral_reverse_chain",
        "model_calls_per_image": 1 if gaussian_baseline else 25 if one_step else 2500,
        "split": "validation",
        "held_out_test_used": False,
        "timesteps_one_based": [100],
        "sampling_base_seed": 17,
        "partition_seed": 17,
        "checkpoint_epoch": 30 if gaussian_baseline else 5,
        "ema": True,
        "patch_size": None if gaussian_baseline else 64,
        "stride": None if gaussian_baseline else 48,
        "patches_per_image": None if gaussian_baseline else 25,
        "samples": len(data),
        "checkpoint_sha256": hashlib.sha256(checkpoint_path.read_bytes()).hexdigest(),
        "index_sha256": hashlib.sha256((validation_dir / "index.csv").read_bytes()).hexdigest(),
        "inference_seconds": inference_seconds,
        "peak_vram_bytes": torch.cuda.max_memory_allocated(device) if device.type == "cuda" else 0,
        "partition": {
            name: {"samples": len(idx), "groups": len(np.unique(groups[idx]))}
            for name, idx in (("calibration", calibration), ("holdout", holdout))
        },
        "caveat": "Previously inspected validation; exploratory internal group holdout, not final test. All-validation thresholds are descriptive only.",
        "metrics": {},
    }
    for name, values in maps.items():
        values = np.asarray(values)
        fitted = evaluate_maps(
            values[calibration],
            cropped_masks[calibration],
            data.labels[calibration],
            split="validation",
        )
        thresholds = {level: fitted[level]["threshold"] for level in ("pixel", "image")}
        measured = evaluate_maps(
            values[holdout],
            cropped_masks[holdout],
            data.labels[holdout],
            split="validation",
            thresholds=thresholds,
        )
        result["metrics"][name] = {
            "calibration": fitted,
            "group_holdout": measured,
            "all_validation_descriptive": evaluate_maps(
                values, cropped_masks, data.labels, split="validation"
            ),
        }
    (output_dir / "results.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    return result
