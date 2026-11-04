"""Exploratory Gaussian noise-prediction anomaly scores (not full ANDi)."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.model_selection import StratifiedGroupKFold

from dacn.data.shards import ShardDataset
from dacn.evaluation.validation import evaluate_maps
from dacn.training.ddpm_partial import (
    build_diffusion,
    derive_sampling_seed,
    load_ema_checkpoint,
)


@torch.inference_mode()
def denoising_score_map(
    model,
    image: torch.Tensor,
    *,
    diffusion,
    timesteps: tuple[int, ...] = (100, 200, 300),
    repeats: int = 4,
    seed: int = 17,
) -> torch.Tensor:
    """Mean squared epsilon error, sharing a noise draw across timesteps.

    Timesteps are one-based. Each repeat draws independent Gaussian noise with
    an isolated generator. No reverse chain, reconstruction target or mask is
    used. Returns a 240x240 cropped score map; larger means larger noise error.
    """
    if image.shape != (1, 1, 256, 256) or not image.is_floating_point():
        raise ValueError("image must be floating [1,1,256,256]")
    if not torch.isfinite(image).all() or image.abs().max() > 1:
        raise ValueError("image must be finite and in [-1,1]")
    if diffusion.noise_type != "gaussian":
        raise ValueError("This exploratory denoising score requires Gaussian noise")
    if (
        not timesteps
        or len(set(timesteps)) != len(timesteps)
        or any(
            isinstance(t, bool) or not isinstance(t, int) or not 1 <= t <= diffusion.timesteps
            for t in timesteps
        )
    ):
        raise ValueError("timesteps must be distinct one-based integers in the schedule")
    if isinstance(repeats, bool) or not isinstance(repeats, int) or repeats < 1:
        raise ValueError("repeats must be a positive integer")
    if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed < 2**32:
        raise ValueError("seed must be a 32-bit nonnegative integer")
    generator = torch.Generator(device=image.device).manual_seed(seed)
    score = torch.zeros_like(image, dtype=torch.float64)
    model.eval()
    for _ in range(repeats):
        noise = torch.randn(
            image.shape, device=image.device, dtype=image.dtype, generator=generator
        )
        for timestep in timesteps:
            t = torch.tensor([timestep - 1], device=image.device, dtype=torch.long)
            predicted = model(diffusion.q_sample(image, t, noise), t)
            if predicted.shape != noise.shape or not torch.isfinite(predicted).all():
                raise ValueError("Model output must be finite and match noise shape")
            score += (noise.double() - predicted.double()).square()
    score /= repeats * len(timesteps)
    return score[0, 0, 8:-8, 8:-8]


def evaluate_denoising(
    *,
    validation_dir: Path,
    checkpoint_path: Path,
    output_dir: Path,
    device: torch.device,
) -> dict:
    """Fixed 100/200/300 x4 validation experiment with group calibration.

    Writes aggregate metrics plus a private score-map NPZ. No configurable
    search over timestep/weights. Uses the same group split as score fusion.
    """
    if output_dir.exists():
        raise FileExistsError("Choose a fresh experiment output directory")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    config = checkpoint["config"]
    diffusion = build_diffusion(config)
    if diffusion.noise_type != "gaussian":
        raise ValueError("Use a Gaussian checkpoint for this experiment")
    data = ShardDataset(validation_dir, "val", include_masks=True)
    with (validation_dir / "index.csv").open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    if [row["sample_id"] for row in rows] != data.sample_ids:
        raise ValueError("Patient-group order does not match dataset")
    groups = np.asarray([row["patient_group_id"] for row in rows])
    calibration, holdout = next(
        StratifiedGroupKFold(2, shuffle=True, random_state=17).split(
            np.zeros(len(data)), data.labels, groups
        )
    )
    if not set(groups[calibration]).isdisjoint(groups[holdout]):
        raise ValueError("Patient-group leakage")
    if any(len(np.unique(data.labels[idx])) != 2 for idx in (calibration, holdout)):
        raise ValueError("Each partition needs both classes")
    model = load_ema_checkpoint(checkpoint_path, config, device)
    scores = np.empty((len(data), 240, 240), dtype=np.float64)
    brightness = np.empty_like(scores)
    masks = np.empty_like(scores, dtype=np.uint8)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)
    started = time.perf_counter()
    for index in range(len(data)):
        image, mask, _ = data[index]
        scores[index] = (
            denoising_score_map(
                model,
                image.unsqueeze(0).to(device),
                diffusion=diffusion,
                seed=derive_sampling_seed(17, data.sample_ids[index]),
            )
            .cpu()
            .numpy()
        )
        brightness[index] = image[0, 8:-8, 8:-8].numpy()
        masks[index] = mask[8:-8, 8:-8].numpy()
        if (index + 1) % 10 == 0 or index + 1 == len(data):
            print(f"denoising samples={index + 1}/{len(data)}", flush=True)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    result = {
        "method": "mean_squared_epsilon_error_3timesteps_4repeats",
        "split": "validation",
        "held_out_test_used": False,
        "timesteps_one_based": [100, 200, 300],
        "noise_repeats": 4,
        "sampling_base_seed": 17,
        "partition_seed": 17,
        "checkpoint_epoch": checkpoint["epoch"],
        "checkpoint_sha256": hashlib.sha256(checkpoint_path.read_bytes()).hexdigest(),
        "index_sha256": hashlib.sha256((validation_dir / "index.csv").read_bytes()).hexdigest(),
        "inference_seconds": time.perf_counter() - started,
        "peak_vram_bytes": int(torch.cuda.max_memory_allocated(device))
        if device.type == "cuda"
        else 0,
        "partition": {
            name: {"samples": len(idx), "groups": len(np.unique(groups[idx]))}
            for name, idx in (("calibration", calibration), ("holdout", holdout))
        },
        "caveat": "Previously inspected validation; exploratory internal group holdout, not final test",
        "metrics": {},
    }
    for name, values in (("denoising", scores), ("brightness_control", brightness)):
        fitted = evaluate_maps(
            values[calibration], masks[calibration], data.labels[calibration], split="validation"
        )
        thresholds = {level: fitted[level]["threshold"] for level in ("pixel", "image")}
        measured = evaluate_maps(
            values[holdout],
            masks[holdout],
            data.labels[holdout],
            split="validation",
            thresholds=thresholds,
        )
        result["metrics"][name] = {"calibration": fitted, "group_holdout": measured}
    output_dir.mkdir(parents=True)
    np.savez_compressed(
        output_dir / "private_score_maps.npz",
        denoising=scores,
        brightness=brightness,
        masks=masks,
        labels=data.labels,
    )
    (output_dir / "results.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--validation-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    args = parser.parse_args()
    result = evaluate_denoising(
        validation_dir=args.validation_dir,
        checkpoint_path=args.checkpoint,
        output_dir=args.output_dir,
        device=torch.device(args.device),
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
