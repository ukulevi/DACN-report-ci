"""Fixed t300 one-step/DDIM10 comparison against a verified ancestral cache."""

import csv
import hashlib
import json
import time
from itertools import pairwise

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
    partial_reconstruct,
)

DDIM_INDICES = tuple(range(299, -1, -30))
BASELINE_RUN = "ddpm-val-t4b8-e30-t300-20261102T071516Z"
BASELINE_SHA = "8278ad18543dd87f698c36b510564b9fcd4445d4a1b0ae5d7a859093e2a4e959"


@torch.inference_mode()
def deterministic_from_noisy(model, diffusion, noisy, *, indices):
    """Eta0 DDIM with clipped x0/re-derived epsilon, as in guided-diffusion.

    Zero-based schedule indices; the final transition targets clean (alpha=1).
    A singleton sequence is the analytical one-step x0 estimate. No RNG draws.
    """
    if (
        not indices
        or any(not 0 <= k < diffusion.timesteps for k in indices)
        or any(a <= b for a, b in pairwise(indices))
    ):
        raise ValueError("Indices must strictly decrease inside the diffusion schedule")
    if diffusion.noise_type != "gaussian":
        raise ValueError("Gaussian control only")
    model.eval()
    value = noisy.clone()
    for position, k in enumerate(indices):
        t = torch.full((len(value),), k, device=value.device, dtype=torch.long)
        clean = diffusion._predict_clean(value, t, model(value, t))
        next_k = indices[position + 1] if position + 1 < len(indices) else -1
        if next_k == -1:
            value = clean
        else:
            alpha = float(diffusion.alphas_cumprod[k])
            next_alpha = float(diffusion.alphas_cumprod[next_k])
            epsilon = (value - alpha**0.5 * clean) / (1 - alpha) ** 0.5
            value = next_alpha**0.5 * clean + (1 - next_alpha) ** 0.5 * epsilon
    return value.clamp(-1, 1)


def evaluate_sampler_ablation(
    *, validation_dir, checkpoint_path, output_dir, device, on_progress=None
):
    cache_path = (
        checkpoint_path.parent.parent
        / "evaluations"
        / BASELINE_RUN
        / "ddpm_epoch_030_t0300_ancestral_reconstructions.npz"
    )
    if hashlib.sha256(cache_path.read_bytes()).hexdigest() != BASELINE_SHA:
        raise ValueError("Registered ancestral cache hash mismatch")
    with np.load(cache_path, allow_pickle=False) as cache:
        images, ancestral, masks, labels = [
            cache[k] for k in ("images", "reconstructions", "masks", "labels")
        ]
    data = ShardDataset(validation_dir, "val", include_masks=True)
    if len(data) != 83 or len(images) != len(data) or not np.array_equal(labels, data.labels):
        raise ValueError("Expected aligned complete83 validation cache")
    with (validation_dir / "index.csv").open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    if [row["sample_id"] for row in rows] != data.sample_ids:
        raise ValueError("Group index order mismatch")
    groups = np.asarray([row["patient_group_id"] for row in rows])
    calibration, holdout = next(
        StratifiedGroupKFold(2, shuffle=True, random_state=17).split(
            np.zeros(len(data)), labels, groups
        )
    )
    if not set(groups[calibration]).isdisjoint(groups[holdout]):
        raise ValueError("Patient-group leakage")
    if any(len(np.unique(labels[idx])) != 2 for idx in (calibration, holdout)):
        raise ValueError("Each partition must contain both classes")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if (
        checkpoint["epoch"] != 30
        or checkpoint["method"] != "ddpm_partial_reconstruction_bmad_flair"
    ):
        raise ValueError("Expected original epoch30 DDPM")
    model = load_ema_checkpoint(checkpoint_path, checkpoint["config"], device)
    diffusion = build_diffusion(checkpoint["config"])
    del checkpoint
    output_dir.mkdir(parents=True, exist_ok=False)
    predictions = {name: np.empty_like(images) for name in ("one_step", "ddim10")}
    methods = ("one_step", "ddim10", "ancestral300")
    maps = {
        f"{method}_{score}": []
        for method in methods
        for score in ("pixel_difference", "local_ssim")
    }
    maps["brightness_control"] = []
    model_seconds = {name: 0.0 for name in predictions}
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    control_max_error = None
    for index in range(len(data)):
        image, mask, _ = data[index]
        if not np.array_equal(image[0].numpy(), images[index]) or not np.array_equal(
            mask.numpy(), masks[index]
        ):
            raise ValueError("Ancestral cache sample content/order differs")
        seed = derive_sampling_seed(17, data.sample_ids[index])
        image = image[None].to(device)
        with _fixed_rng(seed, device):
            t = torch.tensor([299], device=device)
            noisy = diffusion.q_sample(image, t, diffusion.noise(image, t))
        if index == 0:
            control = partial_reconstruct(
                model, image, starting_timestep=300, seed=seed, diffusion=diffusion
            )
            control_max_error = float(np.abs(control[0, 0].cpu().numpy() - ancestral[index]).max())
            if not np.allclose(control[0, 0].cpu().numpy(), ancestral[index], atol=2e-5, rtol=1e-5):
                raise ValueError("Recomputed ancestral control differs from cache")
        for name, indices in (("one_step", (299,)), ("ddim10", DDIM_INDICES)):
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            sample_start = time.perf_counter()
            prediction = deterministic_from_noisy(model, diffusion, noisy, indices=indices)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            model_seconds[name] += time.perf_counter() - sample_start
            predictions[name][index] = prediction[0, 0].cpu().numpy()
        for method in methods:
            reconstructed = (
                ancestral[index] if method == "ancestral300" else predictions[method][index]
            )
            scores = paired_score_maps(images[index], reconstructed)
            for score, values in scores.items():
                maps[f"{method}_{score}"].append(values)
        maps["brightness_control"].append(images[index, 8:-8, 8:-8])
        completed = index + 1
        if completed == 1 or completed % 10 == 0 or completed == len(data):
            np.savez_compressed(
                output_dir / "private_reconstructions.npz",
                images=images[:completed],
                masks=masks[:completed],
                labels=labels[:completed],
                one_step=predictions["one_step"][:completed],
                ddim10=predictions["ddim10"][:completed],
                ancestral300=ancestral[:completed],
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
    result = {
        "method": "gaussian_e30_t300_one_step_ddim10_cached_ancestral",
        "samples": len(data),
        "split": "validation",
        "held_out_test_used": False,
        "checkpoint_epoch": 30,
        "ema": True,
        "timesteps_one_based": [300],
        "sampling_base_seed": 17,
        "partition_seed": 17,
        "ddim_indices_zero_based": list(DDIM_INDICES),
        "ddim_terminal_index": -1,
        "eta": 0,
        "clip_x0": True,
        "ddim_rederive_epsilon_after_clipping": True,
        "model_calls_per_image": {"one_step": 1, "ddim10": 10, "ancestral300": 300},
        "shared_noisy_input": True,
        "ancestral_cache_run_id": BASELINE_RUN,
        "ancestral_cache_sha256": BASELINE_SHA,
        "ancestral_control_max_abs_error": control_max_error,
        "reconstruction_only_seconds": model_seconds,
        "timing_note": "Synchronized CUDA model/transition time excluding forward noise, transfer, scores, caches. Ancestral full-set time not remeasured.",
        "inference_seconds": time.perf_counter() - started,
        "peak_vram_bytes": torch.cuda.max_memory_allocated(device) if device.type == "cuda" else 0,
        "partition": {
            name: {"samples": len(idx), "groups": len(np.unique(groups[idx]))}
            for name, idx in (("calibration", calibration), ("holdout", holdout))
        },
        "caveat": "One seed, previously inspected validation, exploratory group holdout; all83 metrics descriptive, not final test.",
        "metrics": {},
    }
    cropped = masks[:, 8:-8, 8:-8]
    for name, values in maps.items():
        values = np.asarray(values)
        fitted = evaluate_maps(
            values[calibration], cropped[calibration], labels[calibration], split="validation"
        )
        thresholds = {level: fitted[level]["threshold"] for level in ("pixel", "image")}
        result["metrics"][name] = {
            "calibration": fitted,
            "group_holdout": evaluate_maps(
                values[holdout],
                cropped[holdout],
                labels[holdout],
                split="validation",
                thresholds=thresholds,
            ),
            "all_validation_descriptive": evaluate_maps(
                values, cropped, labels, split="validation"
            ),
        }
    (output_dir / "results.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    return result
