"""Fixed validation scoring for patch/context and original DDPM one-step controls."""

import csv
import hashlib
import io
import json
import os
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


@torch.inference_mode()
def gaussian_ancestral_with_noisy(model, diffusion, image, *, steps=300, seed=17):
    """Original one-channel chain with explicit paired forward noise for audit."""
    if image.shape != (1, 1, 256, 256) or not 1 <= steps <= diffusion.timesteps:
        raise ValueError("Invalid input shape or timestep")
    if diffusion.noise_type != "gaussian":
        raise ValueError("This control requires Gaussian noise")
    model.eval()
    with _fixed_rng(seed, image.device):
        t = torch.full((1,), steps - 1, device=image.device, dtype=torch.long)
        noise = diffusion.noise(image, t)
        noisy = diffusion.q_sample(image, t, noise)
        result = diffusion.reconstruct(model, image, steps=steps, initial_noise=noise)
    return result, noisy


def evaluate_patch_validation(
    *,
    validation_dir,
    checkpoint_path,
    output_dir,
    device,
    on_progress=None,
    one_step=False,
    gaussian_baseline=False,
    feature_condition=False,
    starting_timestep=100,
    resume_partial=False,
    condition_mode="self",
    donor_train_dir=None,
    unconditional_finetune=False,
):
    output_dir = Path(output_dir)
    if unconditional_finetune and (feature_condition or gaussian_baseline or one_step):
        raise ValueError("Unconditional fine-tune cannot mix evaluation modes")
    stores_noisy = feature_condition or unconditional_finetune
    if condition_mode not in {"self", "zero", "normal_donor"} or (
        condition_mode != "self" and not feature_condition
    ):
        raise ValueError("Invalid conditional intervention")
    output_dir.mkdir(parents=True, exist_ok=stores_noisy and resume_partial)
    if unconditional_finetune:
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        initialization = checkpoint.get("ema_initialization", {})
        if (
            checkpoint.get("method") != "ddpm_partial_reconstruction_bmad_flair"
            or not 31 <= checkpoint["epoch"] <= 40
            or initialization.get("source_epoch") != 30
            or initialization.get("encoder_added") is not False
        ):
            raise ValueError("Expected unconditional EMA30 fine-tune checkpoint")
        selected_epoch = checkpoint["epoch"]
        model = load_ema_checkpoint(checkpoint_path, checkpoint["config"], device)
        diffusion = build_diffusion(checkpoint["config"])
        del checkpoint
    elif feature_condition:
        from dacn.training.feature_condition import load_feature_checkpoint

        model, diffusion = load_feature_checkpoint(checkpoint_path, device)
        selected_epoch = torch.load(checkpoint_path, map_location="cpu", weights_only=False)[
            "epoch"
        ]
    elif gaussian_baseline:
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
    donors = None
    if condition_mode == "normal_donor":
        if donor_train_dir is None:
            raise ValueError("Normal training donor directory required")
        donors = ShardDataset(donor_train_dir, "train", include_masks=False)
        if not len(donors) or np.any(donors.labels != 0):
            raise ValueError("Condition donors must be normal training slices")
    with (validation_dir / "index.csv").open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    if [row["sample_id"] for row in rows] != data.sample_ids:
        raise ValueError("Group index order differs from validation data")
    groups = np.asarray([row["patient_group_id"] for row in rows])
    checkpoint_bytes = Path(checkpoint_path).read_bytes()
    checkpoint_sha = hashlib.sha256(checkpoint_bytes).hexdigest()
    index_bytes = (Path(validation_dir) / "index.csv").read_bytes()
    prefix_contract = {
        "schema_version": 1,
        "checkpoint_sha256": checkpoint_sha,
        "index_sha256": hashlib.sha256(index_bytes).hexdigest(),
        "sample_count": len(data),
        "starting_timestep": starting_timestep,
        "sampling_base_seed": 17,
        "one_step": one_step,
        "gaussian_baseline": gaussian_baseline,
        "feature_condition": feature_condition,
        "condition_mode": condition_mode,
        "unconditional_finetune": unconditional_finetune,
    }
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
    noisy_images = []
    if feature_condition:
        from dacn.training.feature_condition import feature_reconstruct

        reconstruct = feature_reconstruct
        reconstruction_kwargs = {"return_noisy": True}
    elif unconditional_finetune:
        reconstruct, reconstruction_kwargs = gaussian_ancestral_with_noisy, {}
    previous = None
    if stores_noisy and resume_partial and (output_dir / "private_reconstructions.npz").is_file():
        contract_path = output_dir / "prefix_contract.json"
        if not contract_path.is_file() or json.loads(contract_path.read_text()) != prefix_contract:
            raise ValueError("Partial evaluation cache contract differs or is missing")
        cache_bytes = (output_dir / "private_reconstructions.npz").read_bytes()
        with np.load(io.BytesIO(cache_bytes), allow_pickle=False) as cache:
            previous = {key: cache[key].copy() for key in cache.files}
        if (
            len(previous["images"]) > len(data)
            or len(previous["labels"]) != len(previous["images"])
            or not np.array_equal(previous["labels"], data.labels[: len(previous["labels"])])
        ):
            raise ValueError("Partial evaluation cache labels differ")
    for index in range(len(data)):
        image, mask, _ = data[index]
        if previous is not None and index < len(previous["images"]):
            if not np.array_equal(
                previous["images"][index], image[0].numpy()
            ) or not np.array_equal(previous["masks"][index], mask.numpy()):
                raise ValueError("Partial evaluation cache sample differs")
            reconstructed = previous["reconstructions"][index]
            noisy_images.append(previous["noisy_images"][index])
        else:
            selected_model = model
            if condition_mode != "self":
                from dacn.evaluation.condition_ablation import ConditionIntervention, donor_index

                donor = (
                    donors[donor_index(data.sample_ids[index], len(donors))][0][None].to(device)
                    if donors is not None
                    else None
                )
                selected_model = ConditionIntervention(model, condition_mode, donor)
            value = reconstruct(
                selected_model,
                diffusion,
                image[None].to(device),
                steps=starting_timestep,
                seed=derive_sampling_seed(17, data.sample_ids[index]),
                **reconstruction_kwargs,
            )
            if stores_noisy:
                value, noisy = value
                noisy_images.append(noisy[0, 0].cpu().numpy())
            reconstructed = value[0, 0].cpu().numpy()
        x = image[0].numpy()
        scores = paired_score_maps(x, reconstructed)
        images.append(x)
        reconstructions.append(reconstructed)
        masks.append(mask.numpy())
        for name in ("pixel_difference", "local_ssim"):
            maps[name].append(scores[name])
        maps["brightness_control"].append(x[8:-8, 8:-8])
        completed = index + 1
        if (completed == 1 or completed % 5 == 0 or completed == len(data)) and (
            previous is None or completed >= len(previous["images"])
        ):
            # Durable prefix cache prevents losing all reconstruction work on interruption.
            cache_path = output_dir / "private_reconstructions.npz"
            temporary = cache_path.with_suffix(".npz.tmp")
            with temporary.open("wb") as stream:
                np.savez_compressed(
                    stream,
                    images=np.asarray(images),
                    reconstructions=np.asarray(reconstructions),
                    masks=np.asarray(masks),
                    labels=data.labels[:completed],
                    **({"noisy_images": np.asarray(noisy_images)} if stores_noisy else {}),
                )
            os.replace(temporary, cache_path)
            contract_tmp = output_dir / "prefix_contract.json.tmp"
            contract_tmp.write_text(json.dumps(prefix_contract, sort_keys=True) + "\n")
            os.replace(contract_tmp, output_dir / "prefix_contract.json")
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
        "method": "unconditional_ddpm_EMA30_finetune"
        if unconditional_finetune
        else "resnet18_feature_conditioned_ddpm"
        if feature_condition
        else "gaussian_ddpm_full_image_one_step_x0"
        if gaussian_baseline
        else "patch_context_sliding_one_step_x0"
        if one_step
        else "patch_context_sliding_reconstruction",
        "reconstruction_mode": "one_step_x0" if one_step else "ancestral_reverse_chain",
        "condition_mode": "none" if unconditional_finetune else condition_mode,
        "model_calls_per_image": starting_timestep
        if stores_noisy
        else 1
        if gaussian_baseline
        else 25
        if one_step
        else 2500,
        "split": "validation",
        "held_out_test_used": False,
        "timesteps_one_based": [starting_timestep],
        "sampling_base_seed": 17,
        "partition_seed": 17,
        "checkpoint_epoch": selected_epoch if stores_noisy else 30 if gaussian_baseline else 5,
        "ema": True,
        "patch_size": None if gaussian_baseline or stores_noisy else 64,
        "stride": None if gaussian_baseline or stores_noisy else 48,
        "patches_per_image": None if gaussian_baseline or stores_noisy else 25,
        "samples": len(data),
        "checkpoint_sha256": checkpoint_sha,
        "index_sha256": prefix_contract["index_sha256"],
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
