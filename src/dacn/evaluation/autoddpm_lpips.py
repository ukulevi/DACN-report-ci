"""LPIPS mask/stitch/resample adaptation using the existing Gaussian EMA30."""

import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch
from scipy.ndimage import maximum_filter
from sklearn.model_selection import StratifiedGroupKFold

from dacn.evaluation.masked_diffusion import stitch_resample
from dacn.evaluation.scores import paired_score_maps
from dacn.evaluation.validation import evaluate_maps
from dacn.training.ddpm_partial import derive_sampling_seed


def perceptual_provenance(model):
    digest = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        digest.update(name.encode())
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return {
        "network": "LPIPS0.1_squeeze_spatial",
        "state_sha256": digest.hexdigest(),
        "input": "240crop grayscale repeatedRGB [-1,1]; normalize=False",
    }


@torch.inference_mode()
def lpips_heatmap(model, image, reconstruction, device):
    image, reconstruction = np.asarray(image), np.asarray(reconstruction)
    if image.shape != (256, 256) or reconstruction.shape != image.shape:
        raise ValueError("Expected padded256 image pair")
    if not all(np.isfinite(v).all() and np.max(np.abs(v)) <= 1 for v in (image, reconstruction)):
        raise ValueError("Expected finite image pair in[-1,1]")
    pair = [
        torch.from_numpy(v[8:-8, 8:-8].copy()).float()[None, None].repeat(1, 3, 1, 1).to(device)
        for v in (image, reconstruction)
    ]
    # Inputs already satisfy LPIPS[-1,1]; do not apply a second range conversion.
    perceptual = model(pair[0], pair[1], normalize=False)[0, 0].cpu().numpy()
    if perceptual.shape != (240, 240) or not np.isfinite(perceptual).all():
        raise ValueError("Invalid spatial LPIPS output")
    residual = np.abs(image[8:-8, 8:-8] - reconstruction[8:-8, 8:-8])
    normalized = np.clip(residual / max(float(np.quantile(residual, 0.95)), 1e-8), 0, 1)
    return normalized * np.maximum(perceptual, 0)


def fit_mask_threshold(heatmaps, labels, calibration):
    normal = np.asarray(calibration)[np.asarray(labels)[calibration] == 0]
    if not len(normal):
        raise ValueError("Normal calibration required for pseudo-mask threshold")
    values = np.asarray(heatmaps)[normal]
    if not np.isfinite(values).all() or (values < 0).any():
        raise ValueError("Invalid pseudo-mask heatmaps")
    return float(np.quantile(values, 0.95))


def pseudo_mask(heatmap, threshold):
    if heatmap.shape != (240, 240) or not np.isfinite(heatmap).all():
        raise ValueError("Expected finite240crop heatmap")
    if not np.isfinite(threshold) or threshold < 0:
        raise ValueError("Invalid pseudo-mask threshold")
    return maximum_filter(heatmap > threshold, size=3, mode="constant")


def evaluate_lpips_resampling(
    *,
    cached,
    groups,
    sample_ids,
    model,
    diffusion,
    perceptual_model,
    output_dir,
    device,
    smoke=False,
    on_progress=None,
    initial_sampler="ancestral300",
    inpainting_sampler="ancestral50",
):
    if initial_sampler not in {"ancestral300", "ddim10"} or inpainting_sampler not in {
        "ancestral50",
        "ddim10",
    }:
        raise ValueError("Unsupported AutoDDPM sampler pair")
    if inpainting_sampler == "ddim10":
        from dacn.evaluation.ddim_inpainting import ddim_stitch_resample

        resample = ddim_stitch_resample
        resample_kwargs = {"repeats": 5}
    else:
        resample = stitch_resample
        resample_kwargs = {"steps": 50, "repeats": 5}
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=False)
    images, initials, masks, labels = [
        cached[k] for k in ("images", "reconstructions", "masks", "labels")
    ]
    if images.shape != (83, 256, 256) or len(sample_ids) != 83:
        raise ValueError("Registered83sample cache required")
    calibration, holdout = next(
        StratifiedGroupKFold(2, shuffle=True, random_state=17).split(
            np.zeros(len(labels)), labels, groups
        )
    )
    if (
        len(calibration) != 40
        or len(holdout) != 43
        or not set(np.asarray(groups)[calibration]).isdisjoint(np.asarray(groups)[holdout])
    ):
        raise ValueError("Registered40/43 group partition required")
    # Normal-calibration heatmaps alone fit the threshold before any holdout work.
    heatmaps = np.empty((83, 240, 240), dtype=np.float32)
    normal_calibration = calibration[labels[calibration] == 0]
    for index in normal_calibration:
        heatmaps[index] = lpips_heatmap(perceptual_model, images[index], initials[index], device)
    mask_threshold = fit_mask_threshold(heatmaps, labels, calibration)
    parameters = {
        "mask_threshold": mask_threshold,
        "fit": "pooled95percentile normal calibration",
        "normal_calibration_samples": len(normal_calibration),
        "dilation": 3,
        "initial_timestep_one_based": 300,
        "resampling_steps": 50,
        "repeats": 5,
        "sampling_base_seed": 17,
        "perceptual": perceptual_provenance(perceptual_model),
        "initial_sampler": initial_sampler,
        "inpainting_sampler": inpainting_sampler,
        "inpainting_model_calls_per_image": 50 if inpainting_sampler == "ddim10" else 250,
        "ddim_eta": 0 if "ddim10" in {initial_sampler, inpainting_sampler} else None,
    }
    (output / "mask_parameters.json").write_text(json.dumps(parameters, indent=2) + "\n")
    if on_progress:
        on_progress({"stage": "pseudo_mask_calibration_frozen", "parameters": parameters})
    selected = (
        [int(np.flatnonzero(labels == label)[0]) for label in (0, 1)] if smoke else list(range(83))
    )
    names = (
        "initial_absolute",
        "initial_ssim",
        "initial_heatmap",
        "stitch_only_absolute",
        "resampled_absolute",
        "resampled_ssim",
        "weighted_resampled_absolute",
        "brightness",
    )
    maps = {name: [] for name in names}
    finals, predicted_masks = [], []
    normal_set = set(normal_calibration)
    max_context_error = 0.0
    inpainting_seconds = []
    for position, index in enumerate(selected):
        if index not in normal_set:
            heatmaps[index] = lpips_heatmap(
                perceptual_model, images[index], initials[index], device
            )
        predicted = pseudo_mask(heatmaps[index], mask_threshold)
        padded = np.pad(predicted, 8)
        mask_tensor = torch.from_numpy(padded)[None, None].to(device)
        image_tensor = torch.from_numpy(images[index])[None, None].to(device)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        sample_started = time.perf_counter()
        final = resample(
            model,
            image_tensor,
            torch.from_numpy(initials[index])[None, None].to(device),
            mask_tensor,
            diffusion=diffusion,
            seed=derive_sampling_seed(17, sample_ids[index]),
            **resample_kwargs,
        )
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        inpainting_seconds.append(time.perf_counter() - sample_started)
        if not torch.isfinite(final).all():
            raise ValueError("Nonfinite inpainting output")
        context_error = float((final - image_tensor)[~mask_tensor].abs().max())
        max_context_error = max(max_context_error, context_error)
        if context_error != 0:
            raise ValueError("Outside-mask context changed")
        final = final[0, 0].cpu().numpy()
        before, after = (
            paired_score_maps(images[index], initials[index]),
            paired_score_maps(images[index], final),
        )
        values = (
            before["pixel_difference"],
            before["local_ssim"],
            heatmaps[index],
            before["pixel_difference"] * predicted,
            after["pixel_difference"],
            after["local_ssim"],
            after["pixel_difference"] * heatmaps[index],
            (images[index, 8:-8, 8:-8] + 1) / 2,
        )
        for name, value in zip(names, values, strict=True):
            maps[name].append(value)
        finals.append(final)
        predicted_masks.append(predicted)
        if position == 0 or (position + 1) % 5 == 0 or position + 1 == len(selected):
            np.savez_compressed(
                output / "private_prefix.npz",
                reconstructions=finals,
                predicted_masks=predicted_masks,
            )
            if on_progress:
                on_progress(
                    {
                        "stage": "resampling",
                        "completed_samples": position + 1,
                        "total_samples": len(selected),
                    }
                )
    maps = {k: np.stack(v) for k, v in maps.items()}
    if not all(np.isfinite(v).all() for v in maps.values()):
        raise ValueError("Nonfinite anomaly scores")
    result = {
        "method": f"autoddpm_lpips_EMA30_t300_{initial_sampler}_{inpainting_sampler}",
        "smoke": smoke,
        "samples": len(selected),
        "held_out_test_used": False,
        "partition": {"calibration": 40, "holdout": 43},
        "parameters": parameters,
        "max_outside_mask_error": max_context_error,
        "metrics": {},
        "phase2_synchronized_seconds": {
            "total": float(sum(inpainting_seconds)),
            "mean_per_image": float(np.mean(inpainting_seconds)),
            "median_per_image": float(np.median(inpainting_seconds)),
            "scope": "inpainting model/transitions incl forward noise and context; excludes LPIPS/scoring/transfers/cache/metrics",
        },
    }
    if not smoke:
        truth = masks[:, 8:-8, 8:-8]
        fits, thresholds = {}, {}
        for name, values in maps.items():
            fits[name] = evaluate_maps(
                values[calibration], truth[calibration], labels[calibration], split="validation"
            )
            thresholds[name] = {
                level: fits[name][level]["threshold"] for level in ("pixel", "image")
            }
        (output / "score_thresholds.json").write_text(json.dumps(thresholds, indent=2) + "\n")
        if on_progress:
            on_progress({"stage": "score_thresholds_frozen"})
        for name, values in maps.items():
            result["metrics"][name] = {
                "calibration": fits[name],
                "group_holdout": evaluate_maps(
                    values[holdout],
                    truth[holdout],
                    labels[holdout],
                    split="validation",
                    thresholds=thresholds[name],
                ),
            }
        pred = np.stack(predicted_masks)
        result["mask_diagnostics"] = {}
        for name, indices in (("calibration", calibration), ("holdout", holdout)):
            roi, gt = pred[indices], truth[indices].astype(bool)
            result["mask_diagnostics"][name] = {
                "lesion_recall": float((roi & gt).sum() / gt.sum()),
                "precision": float((roi & gt).sum() / max(roi.sum(), 1)),
                "roi_fraction": float(roi.mean()),
                "normal_roi_fraction_after_dilation": float(roi[labels[indices] == 0].mean()),
            }
        references = (
            (
                ("initial_absolute", 0.101065919667),
                ("initial_ssim", 0.102402212970),
                ("brightness", 0.389505655716),
            )
            if initial_sampler == "ancestral300"
            else (("brightness", 0.389505655716),)
        )
        for name, expected in references:
            if not np.isclose(
                result["metrics"][name]["group_holdout"]["pixel"]["average_precision"],
                expected,
                rtol=0,
                atol=1e-10,
            ):
                raise ValueError("Registered baseline AP differs")
    np.savez_compressed(
        output / "private_maps.npz",
        images=images[selected],
        initials=initials[selected],
        reconstructions=finals,
        predicted_masks=predicted_masks,
        masks=masks[selected],
        labels=labels[selected],
        **maps,
    )
    (output / "results.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    return result
