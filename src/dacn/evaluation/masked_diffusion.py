"""AutoDDPM-style pilot: SSIM mask proxy, stitch and resample (not exact reproduction)."""

import csv
import hashlib
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
from scipy.ndimage import maximum_filter
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


def automatic_mask(image, initial):
    """Image-only norm95(abs residual)*positive SSIM error, threshold0.5, dilation3."""
    paired = paired_score_maps(image, initial)
    absolute = paired["pixel_difference"]
    scale = max(float(np.quantile(absolute, 0.95)), 1e-8)
    heatmap = np.clip(absolute / scale, 0, 1) * np.clip(paired["local_ssim"], 0, 1)
    mask = maximum_filter(heatmap >= 0.5, size=3, mode="constant")
    return heatmap, mask


@torch.inference_mode()
def stitch_resample(model, image, initial, mask, *, diffusion, seed, steps=50, repeats=5):
    """Mask=True is unknown; each reverse step stitches context at matching time.

    Reverse index k maps noisy state x[k] to x[k-1], with k=0 giving clean.
    Before repeating at k, forward-noise the stitched x[k-1] with beta[k].
    Final context is exactly the input. No ground-truth mask is accepted.
    """
    if image.shape != initial.shape or image.shape != (1, 1, 256, 256):
        raise ValueError("image/initial must be [1,1,256,256]")
    if mask.shape != image.shape or mask.dtype != torch.bool or mask.device != image.device:
        raise ValueError("mask must be a matching boolean tensor")
    if diffusion.noise_type != "gaussian":
        raise ValueError("This pilot requires Gaussian diffusion")
    if (
        isinstance(steps, bool)
        or not isinstance(steps, int)
        or not 1 <= steps <= diffusion.timesteps
    ):
        raise ValueError("Invalid resampling timestep")
    if isinstance(repeats, bool) or not isinstance(repeats, int) or repeats < 1:
        raise ValueError("Invalid resampling repeats")
    model.eval()
    with _fixed_rng(seed, image.device):
        stitched = torch.where(mask, initial, image)
        t = torch.full((1,), steps - 1, device=image.device, dtype=torch.long)
        value = diffusion.q_sample(stitched, t, torch.randn_like(image))
        for k in range(steps - 1, -1, -1):
            for repeat in range(repeats):
                t.fill_(k)
                unknown = diffusion.p_sample(model, value, t)
                if k == 0:
                    context = image
                else:
                    context = diffusion.q_sample(image, t - 1, torch.randn_like(image))
                value = torch.where(mask, unknown, context)
                if repeat + 1 < repeats:
                    beta = float(diffusion.betas[k])
                    value = math.sqrt(1 - beta) * value + math.sqrt(beta) * torch.randn_like(value)
        return value.clamp(-1, 1)


def evaluate_masked(*, validation_dir: Path, checkpoint_path: Path, output_dir: Path, device):
    if output_dir.exists():
        raise FileExistsError("Use a fresh output directory")
    # Existing full-validation t200 initial reconstruction; validate array alignment.
    cache_path = (
        checkpoint_path.parent.parent
        / "evaluations"
        / "ddpm-val-e30-t100-200-diag-20261102T094115Z"
        / "ddpm_epoch_030_t0200_ancestral_reconstructions.npz"
    )
    if (
        hashlib.sha256(cache_path.read_bytes()).hexdigest()
        != "cffefdc5d2cf599fc3903eb8a8fa9a192dddd2dabe4cffeb04ee28cfa60bc06a"
    ):
        raise ValueError("Initial reconstruction cache checksum mismatch")
    with np.load(cache_path, allow_pickle=False) as cache:
        images, initials, masks, labels = [
            cache[k] for k in ("images", "reconstructions", "masks", "labels")
        ]
    data = ShardDataset(validation_dir, "val", include_masks=True)
    with (validation_dir / "index.csv").open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    if len(data) != len(images) or not np.array_equal(data.labels, labels):
        raise ValueError("Initial cache/index alignment mismatch")
    groups = np.asarray([row["patient_group_id"] for row in rows])
    calibration, holdout = next(
        StratifiedGroupKFold(2, shuffle=True, random_state=17).split(
            np.zeros(len(labels)), labels, groups
        )
    )
    assert set(groups[calibration]).isdisjoint(groups[holdout])
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model = load_ema_checkpoint(checkpoint_path, checkpoint["config"], device)
    diffusion = build_diffusion(checkpoint["config"])
    names = (
        "initial_difference",
        "initial_ssim",
        "initial_heatmap",
        "stitch_only_difference",
        "masked_difference",
        "masked_ssim",
        "uncertainty_weighted_difference",
        "brightness_control",
    )
    maps = {name: np.empty((len(data), 240, 240), dtype=np.float64) for name in names}
    finals = np.empty_like(images)
    predicted_masks = np.zeros_like(masks, dtype=np.uint8)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)
    started = time.perf_counter()
    for i in range(len(data)):
        image, truth, _ = data[i]
        if not np.array_equal(image[0].numpy(), images[i]) or not np.array_equal(
            truth.numpy(), masks[i]
        ):
            raise ValueError("Cache sample order/content differs from dataset")
        heatmap, predicted = automatic_mask(images[i], initials[i])
        initial_scores = paired_score_maps(images[i], initials[i])
        maps["initial_difference"][i] = initial_scores["pixel_difference"]
        maps["initial_ssim"][i] = initial_scores["local_ssim"]
        predicted_masks[i, 8:-8, 8:-8] = predicted
        mask_tensor = torch.from_numpy(predicted_masks[i].astype(bool)).to(device)[None, None]
        final = stitch_resample(
            model,
            image[None].to(device),
            torch.from_numpy(initials[i])[None, None].to(device),
            mask_tensor,
            diffusion=diffusion,
            seed=derive_sampling_seed(17, data.sample_ids[i]),
        )
        finals[i] = final[0, 0].cpu().numpy()
        paired = paired_score_maps(images[i], finals[i])
        maps["initial_heatmap"][i] = heatmap
        maps["stitch_only_difference"][i] = (
            np.abs(images[i, 8:-8, 8:-8] - initials[i, 8:-8, 8:-8]) * predicted
        )
        maps["masked_difference"][i] = paired["pixel_difference"]
        maps["masked_ssim"][i] = paired["local_ssim"]
        maps["uncertainty_weighted_difference"][i] = paired["pixel_difference"] * heatmap
        maps["brightness_control"][i] = images[i, 8:-8, 8:-8]
        if (i + 1) % 10 == 0 or i + 1 == len(data):
            print(f"masked_resampling samples={i + 1}/{len(data)}", flush=True)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    result = {
        "method": "autoddpm_style_ssim_proxy_not_exact_reproduction",
        "split": "validation",
        "held_out_test_used": False,
        "initial_timestep": 200,
        "resampling_timestep": 50,
        "resamples_per_step": 5,
        "mask_threshold": 0.5,
        "dilation_size": 3,
        "seed": 17,
        "inference_seconds": time.perf_counter() - started,
        "peak_vram_bytes": int(torch.cuda.max_memory_allocated(device))
        if device.type == "cuda"
        else 0,
        "metrics": {},
        "mask_coverage": {},
        "partition": {
            name: {"samples": len(idx), "groups": len(np.unique(groups[idx]))}
            for name, idx in (("calibration", calibration), ("holdout", holdout))
        },
    }
    truth = masks[:, 8:-8, 8:-8]
    for name, scores in maps.items():
        cal = evaluate_maps(
            scores[calibration], truth[calibration], labels[calibration], split="validation"
        )
        thresholds = {k: cal[k]["threshold"] for k in ("pixel", "image")}
        measured = evaluate_maps(
            scores[holdout],
            truth[holdout],
            labels[holdout],
            split="validation",
            thresholds=thresholds,
        )
        result["metrics"][name] = {"calibration": cal, "group_holdout": measured}
    for name, idx in (("calibration", calibration), ("holdout", holdout)):
        pred = predicted_masks[idx, 8:-8, 8:-8].astype(bool)
        gt = truth[idx].astype(bool)
        result["mask_coverage"][name] = {
            "lesion_recall": float((pred & gt).sum() / gt.sum()),
            "precision": float((pred & gt).sum() / max(pred.sum(), 1)),
            "roi_fraction": float(pred.mean()),
        }
    output_dir.mkdir(parents=True)
    np.savez_compressed(
        output_dir / "private_maps.npz",
        images=images,
        reconstructions=finals,
        predicted_masks=predicted_masks,
        masks=masks,
        labels=labels,
        **maps,
    )
    (output_dir / "results.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    return result
