"""Matched DDIM20+30, single-pass inpainting and final-reconstruction score ablation."""

import json
import time
from functools import partial
from pathlib import Path

import numpy as np
import torch
from sklearn.model_selection import StratifiedGroupKFold

from dacn.evaluation.autoddpm_ddim import _timed
from dacn.evaluation.autoddpm_lpips import fit_mask_threshold, pseudo_mask
from dacn.evaluation.ddim_inpainting import (
    ddim_indices,
    ddim_initial_reconstruction,
    ddim_stitch_resample,
)
from dacn.evaluation.validation import evaluate_maps

DEFAULT_CONFIG = {
    "seed": 17,
    "phase1_starting_timestep": 300,
    "phase1_inference_steps": 20,
    "phase2_starting_timestep": 50,
    "phase2_inference_steps": 30,
    "resampling_passes": 1,
    "eta": 0,
    "mask_percentile": 95,
    "mask_dilation": 3,
    "residual_epsilon": 1e-8,
}


def resolve_ablation_config(values):
    config = {**DEFAULT_CONFIG, **values}
    if config != DEFAULT_CONFIG or set(values) - set(DEFAULT_CONFIG):
        raise ValueError(
            "This registered experiment requires the fixed DDIM20+30 single-pass preset"
        )
    if any(isinstance(v, bool) for v in values.values()):
        raise ValueError("Boolean config values are not valid counts/settings")
    if any(
        isinstance(v, int) and not isinstance(config[k], int) for k, v in DEFAULT_CONFIG.items()
    ):
        raise ValueError("Counts and seeds must be integers")
    return config


@torch.inference_mode()
def ablation_pair_scores(perceptual_model, image, reconstruction, device, epsilon=1e-8):
    """All three scores use THIS reconstruction; LPIPS gets no residual multiplier."""
    if image.shape != (256, 256) or reconstruction.shape != image.shape:
        raise ValueError("Expected padded256 image pair")
    if not all(np.isfinite(v).all() and (np.abs(v) <= 1).all() for v in (image, reconstruction)):
        raise ValueError("Expected finite[-1,1] image pair")
    started = time.perf_counter()
    residual = np.abs(image[8:-8, 8:-8] - reconstruction[8:-8, 8:-8])
    l1_seconds = time.perf_counter() - started
    tensors = [
        torch.from_numpy(v[8:-8, 8:-8].copy()).float()[None, None].repeat(1, 3, 1, 1).to(device)
        for v in (image, reconstruction)
    ]
    lpips, lpips_seconds = _timed(partial(perceptual_model, *tensors, normalize=False), device)
    lpips = lpips[0, 0].cpu().numpy()
    if lpips.shape != (240, 240) or not np.isfinite(lpips).all():
        raise ValueError("Invalid spatial LPIPS map")
    # LPIPS weights define a nonnegative distance; retain the established numerical floor.
    lpips = np.maximum(lpips, 0)
    started = time.perf_counter()
    normalized = np.clip(residual / (float(np.quantile(residual, 0.95)) + epsilon), 0, 1)
    product = lpips * normalized
    product_seconds = time.perf_counter() - started
    return {"l1": residual, "lpips": lpips, "product": product}, {
        "l1_seconds": l1_seconds,
        "lpips_seconds": lpips_seconds,
        "normalization_product_seconds": product_seconds,
    }


def four_ablation_scores(initial, final):
    return {
        "ddpm_l1": initial["l1"],
        "autoddpm_l1": final["l1"],
        "autoddpm_lpips": final["lpips"],
        "autoddpm_lpips_residual": final["product"],
    }


def evaluate_ddim20_30(
    *,
    cached,
    groups,
    sample_ids,
    model,
    diffusion,
    perceptual_model,
    output_dir,
    device,
    config,
    smoke=False,
    on_progress=None,
    timing_reference=None,
    historical_metrics=None,
):
    config = resolve_ablation_config(config)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=False)
    images, masks, labels = cached["images"], cached["masks"], cached["labels"]
    calibration, holdout = next(
        StratifiedGroupKFold(2, shuffle=True, random_state=17).split(
            np.zeros(len(labels)), labels, groups
        )
    )
    if (
        len(images) != 83
        or len(calibration) != 40
        or len(holdout) != 43
        or not set(np.asarray(groups)[calibration]).isdisjoint(np.asarray(groups)[holdout])
    ):
        raise ValueError("Registered83/40/43 group partition required")
    selected = (
        [int(np.flatnonzero(labels == target)[0]) for target in (0, 1)]
        if smoke
        else list(range(83))
    )
    converted = (
        sorted(set(calibration[labels[calibration] == 0]) | set(selected))
        if smoke
        else list(range(83))
    )
    phase1_indices = ddim_indices(
        config["phase1_starting_timestep"], config["phase1_inference_steps"]
    )
    phase2_indices = ddim_indices(
        config["phase2_starting_timestep"], config["phase2_inference_steps"]
    )
    from dacn.training.ddpm_partial import derive_sampling_seed

    counter = {"unet": 0, "lpips": 0}

    def count_unet(*_args):
        counter["unet"] += 1

    def count_lpips(*_args):
        counter["lpips"] += 1

    handles = [
        model.register_forward_hook(count_unet),
        perceptual_model.register_forward_hook(count_lpips),
    ]
    try:
        initials, initial_scores, phase1_times, initial_score_times = {}, {}, {}, {}
        # Fit mask parameters from normal calibration before any phase2 image is processed.
        heatmaps = np.empty((83, 240, 240), dtype=np.float32)
        for index in converted:
            image = torch.from_numpy(images[index])[None, None].to(device)
            seed = derive_sampling_seed(config["seed"], sample_ids[index])
            before = counter["unet"]
            prediction, elapsed = _timed(
                partial(
                    ddim_initial_reconstruction,
                    model,
                    diffusion,
                    image,
                    seed=seed,
                    indices=phase1_indices,
                ),
                device,
            )
            if counter["unet"] - before != 20 or not torch.isfinite(prediction).all():
                raise ValueError("Phase1 requires exactly20 finite U-Net evaluations")
            initials[index] = prediction[0, 0].cpu().numpy()
            phase1_times[index] = elapsed
            initial_scores[index], initial_score_times[index] = ablation_pair_scores(
                perceptual_model, images[index], initials[index], device, config["residual_epsilon"]
            )
            heatmaps[index] = initial_scores[index]["product"]
        mask_threshold = fit_mask_threshold(heatmaps, labels, calibration)
        parameters = {
            "config": config,
            "mask_threshold": mask_threshold,
            "phase1_indices_zero_based": list(phase1_indices),
            "phase2_indices_zero_based": list(phase2_indices),
            "terminal_index": -1,
            "mask_fit": "pooled95percentile20normal calibration; strict>; dilation3",
            "score_thresholds": {},
        }
        (output / "mask_parameters.json").write_text(json.dumps(parameters, indent=2) + "\n")
        if on_progress:
            on_progress({"stage": "mask_parameters_frozen", "mask_threshold": mask_threshold})
        names = (
            "ddpm_l1",
            "autoddpm_l1",
            "autoddpm_lpips",
            "autoddpm_lpips_residual",
            "initial_lpips_only",
            "initial_product",
            "naive_stitch_l1",
            "brightness",
        )
        maps = {name: [] for name in names}
        finals, predicted_masks, phase2_times, score_times, per_image_calls = [], [], [], [], []
        max_context_error = 0.0
        for position, index in enumerate(selected):
            predicted = pseudo_mask(heatmaps[index], mask_threshold)
            mask = torch.from_numpy(np.pad(predicted, 8))[None, None].to(device)
            image = torch.from_numpy(images[index])[None, None].to(device)
            initial = torch.from_numpy(initials[index])[None, None].to(device)
            seed = derive_sampling_seed(config["seed"], sample_ids[index])
            before = counter["unet"]
            final, elapsed = _timed(
                partial(
                    ddim_stitch_resample,
                    model,
                    image,
                    initial,
                    mask,
                    diffusion=diffusion,
                    seed=seed,
                    indices=phase2_indices,
                    repeats=config["resampling_passes"],
                ),
                device,
            )
            phase2_calls = counter["unet"] - before
            if phase2_calls != 30 or not torch.isfinite(final).all():
                raise ValueError("Phase2 requires exactly30 finite U-Net evaluations")
            context_error = float((final - image)[~mask].abs().max())
            max_context_error = max(max_context_error, context_error)
            if context_error > 1e-7:
                raise ValueError("Outside-mask context changed")
            final = final[0, 0].cpu().numpy()
            final_scores, measured_score_times = ablation_pair_scores(
                perceptual_model, images[index], final, device, config["residual_epsilon"]
            )
            scores = {
                **four_ablation_scores(initial_scores[index], final_scores),
                "initial_lpips_only": initial_scores[index]["lpips"],
                "initial_product": initial_scores[index]["product"],
                "naive_stitch_l1": initial_scores[index]["l1"] * predicted,
                "brightness": (images[index, 8:-8, 8:-8] + 1) / 2,
            }
            for name, values in maps.items():
                values.append(scores[name])
            finals.append(final)
            predicted_masks.append(predicted)
            phase2_times.append(elapsed)
            score_times.append(measured_score_times)
            per_image_calls.append(20 + phase2_calls)
            if on_progress and (
                position == 0 or (position + 1) % 10 == 0 or position + 1 == len(selected)
            ):
                on_progress(
                    {
                        "stage": "single_pass_inpainting",
                        "completed_samples": position + 1,
                        "total_samples": len(selected),
                        "unet_calls_per_output_image": 50,
                    }
                )
        if counter["unet"] != 20 * len(converted) + 30 * len(selected):
            raise ValueError("Unexpected total U-Net count")
        if counter["lpips"] != len(converted) + len(selected):
            raise ValueError("Expected one initial and one final LPIPS call per output image")
        maps = {k: np.stack(v) for k, v in maps.items()}
        if not all(np.isfinite(v).all() for v in maps.values()):
            raise ValueError("Nonfinite ablation scores")
        latency = {
            name: []
            for name in ("ddpm_l1", "autoddpm_l1", "autoddpm_lpips", "autoddpm_lpips_residual")
        }
        for position, index in enumerate(selected):
            initial_time = initial_score_times[index]
            final_time = score_times[position]
            # Component sums over shared reconstructions, not four separate sampler reruns.
            prefix = phase1_times[index] + sum(initial_time.values()) + phase2_times[position]
            latency["ddpm_l1"].append(phase1_times[index] + initial_time["l1_seconds"])
            latency["autoddpm_l1"].append(prefix + final_time["l1_seconds"])
            latency["autoddpm_lpips"].append(prefix + final_time["lpips_seconds"])
            latency["autoddpm_lpips_residual"].append(prefix + sum(final_time.values()))
        fits = {}
        if not smoke:
            for name, values in maps.items():
                fits[name] = evaluate_maps(
                    values[calibration],
                    masks[calibration, 8:-8, 8:-8],
                    labels[calibration],
                    split="validation",
                )
                parameters["score_thresholds"][name] = {
                    level: fits[name][level]["threshold"] for level in ("pixel", "image")
                }
        (output / "score_parameters.json").write_text(json.dumps(parameters, indent=2) + "\n")
        if on_progress:
            on_progress({"stage": "score_thresholds_frozen"})
        result = {
            "method": "AutoDDPM_DDIM20_DDIM30_single_pass_four_score_ablation",
            "smoke": smoke,
            "samples": len(selected),
            "config": config,
            "held_out_test_used": False,
            "parameters": parameters,
            "unet_calls_per_output_image": per_image_calls,
            "unet_calls_total": counter["unet"],
            "lpips_calls_total": counter["lpips"],
            "extra_smoke_normal_calibration_unet_calls": 20 * (len(converted) - len(selected)),
            "max_outside_mask_error": max_context_error,
            "phase2_passes": 1,
            "phase1_mean_seconds_selected": float(np.mean([phase1_times[i] for i in selected])),
            "phase2_mean_seconds": float(np.mean(phase2_times)),
            "score_latency_component_sums": {
                name: {"mean_seconds": float(np.mean(v)), "median_seconds": float(np.median(v))}
                for name, v in latency.items()
            },
            "latency_scope": "Synchronized denoising plus measured scoring component sums; shared runs, not separate reruns. Excludes transfers, mask assembly, global calibration, cache/metric I-O and startup. Not end-to-end wall latency.",
            "metrics": {},
            "historical_different_sampler_controls": historical_metrics or {},
        }
        if timing_reference:
            result["historical_ancestral_two_image_probe"] = timing_reference
            old_mean = float(
                np.mean([v["phase1_seconds"] + v["phase2_seconds"] for v in timing_reference])
            )
            matched = [int(np.flatnonzero(labels == target)[0]) for target in (0, 1)]
            new_mean = float(
                np.mean([phase1_times[i] + phase2_times[selected.index(i)] for i in matched])
            )
            result["matched_two_image_denoising_speedup"] = old_mean / new_mean
            result["timing_reference_caveat"] = (
                "Two matched normal/abnormal samples, prior T4 container; excludes LPIPS/I-O, not83image end-to-end speedup. No new ancestral calls in this run."
            )
        if not smoke:
            for name, values in maps.items():
                result["metrics"][name] = {
                    "calibration": fits[name],
                    "group_holdout": evaluate_maps(
                        values[holdout],
                        masks[holdout, 8:-8, 8:-8],
                        labels[holdout],
                        split="validation",
                        thresholds=parameters["score_thresholds"][name],
                    ),
                }
            pred, truth = np.stack(predicted_masks), masks[:, 8:-8, 8:-8].astype(bool)
            result["mask_diagnostics"] = {
                name: {
                    "lesion_recall": float((pred[idx] & truth[idx]).sum() / truth[idx].sum()),
                    "lesion_precision": float(
                        (pred[idx] & truth[idx]).sum() / max(pred[idx].sum(), 1)
                    ),
                    "normal_coverage": float(pred[idx][labels[idx] == 0].mean()),
                }
                for name, idx in (("calibration", calibration), ("holdout", holdout))
            }
        np.savez_compressed(
            output / "private_ablation.npz",
            images=images[selected],
            initials=np.stack([initials[i] for i in selected]),
            finals=finals,
            masks=masks[selected],
            labels=labels[selected],
            predicted_masks=predicted_masks,
            **maps,
        )
        save_ablation_examples(
            output, images, initials, finals, predicted_masks, maps, labels, selected
        )
        (output / "results.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
        return result
    finally:
        for handle in handles:
            handle.remove()


def save_ablation_examples(
    output, images, initials, finals, predicted_masks, maps, labels, selected
):
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    for target in (0, 1):
        index = next(i for i in selected if labels[i] == target)
        position = selected.index(index)
        values = [
            images[index, 8:-8, 8:-8],
            initials[index][8:-8, 8:-8],
            finals[position][8:-8, 8:-8],
            predicted_masks[position],
            maps["initial_lpips_only"][position],
            maps["autoddpm_lpips"][position],
            maps["autoddpm_l1"][position],
            maps["autoddpm_lpips_residual"][position],
        ]
        titles = [
            "Input",
            "DDIM20 initial",
            "DDIM30 final",
            "Pseudo-mask",
            "Initial LPIPS",
            "Final LPIPS",
            "Final L1",
            "Final LPIPS x final residual",
        ]
        fig, axes = plt.subplots(1, 8, figsize=(24, 3))
        for axis, value, title in zip(axes, values, titles, strict=True):
            axis.imshow(value, cmap="gray" if title in titles[:4] else "inferno")
            axis.set_title(title, fontsize=8)
            axis.axis("off")
        fig.tight_layout()
        fig.savefig(Path(output) / f"private_example_class{target}.png")
        plt.close(fig)
