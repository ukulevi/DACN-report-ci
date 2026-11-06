"""Fixed normal-calibration of seven pretrained LPIPS layer contributions."""

import json
from pathlib import Path

import numpy as np
import torch
from sklearn.model_selection import StratifiedGroupKFold

from dacn.evaluation.autoddpm_lpips import perceptual_provenance
from dacn.evaluation.validation import evaluate_maps

LPIPS_STATE_SHA = "18fed4d1fc5c04f2918f64df3a2463344108576d1888f0d75538ca6d5c4dffe6"


@torch.inference_mode()
def layer_contributions(model, image, reconstruction, device):
    if image.shape != (256, 256) or reconstruction.shape != image.shape:
        raise ValueError("Expected padded256 image pair")
    if not all(np.isfinite(v).all() and (np.abs(v) <= 1).all() for v in (image, reconstruction)):
        raise ValueError("Expected finite images in[-1,1]")
    inputs = [
        torch.from_numpy(v[8:-8, 8:-8].copy()).float()[None, None].repeat(1, 3, 1, 1).to(device)
        for v in (image, reconstruction)
    ]
    total, layers = model(*inputs, normalize=False, retPerLayer=True)
    if len(layers) != 7 or total.shape != (1, 1, 240, 240):
        raise ValueError("Expected seven spatial SqueezeNet LPIPS contributions")
    contributions = torch.stack(layers)[:, 0, 0].cpu().numpy()
    total = total[0, 0].cpu().numpy()
    if contributions.shape != (7, 240, 240) or not np.isfinite(contributions).all():
        raise ValueError("Invalid layer maps")
    summed = contributions[0].copy()
    for value in contributions[1:]:
        summed += value
    error = float(np.abs(summed - total).max())
    if not np.isfinite(total).all() or error > 1e-6:
        raise ValueError("LPIPS layer sum does not reproduce aggregate")
    return np.maximum(total, 0), contributions, error


def fit_layer_calibration(layers, images, labels, calibration, *, scale_floor=1e-6):
    """Only normal-calibration foreground enters fitting, never tumor annotations."""
    layers, images, labels, calibration = [
        np.asarray(v) for v in (layers, images, labels, calibration)
    ]
    if layers.ndim != 4 or layers.shape[1:] != images.shape or labels.shape != (len(images),):
        raise ValueError("Layer/image/label alignment differs")
    if not np.isin(labels, [0, 1]).all() or not np.isfinite(scale_floor) or scale_floor <= 0:
        raise ValueError("Invalid labels or scale floor")
    if (
        calibration.ndim != 1
        or not np.issubdtype(calibration.dtype, np.integer)
        or (
            (calibration < 0).any()
            or (calibration >= len(images)).any()
            or len(np.unique(calibration)) != len(calibration)
        )
    ):
        raise ValueError("Invalid calibration indices")
    normal = calibration[labels[calibration] == 0]
    if not len(normal):
        raise ValueError("Normal calibration required")
    foreground = images[normal] > -0.95
    if not foreground.any():
        raise ValueError("Normal foreground required")
    medians, iqrs = [], []
    for values in layers:
        observed = values[normal][foreground]
        if not np.isfinite(observed).all():
            raise ValueError("Nonfinite normal calibration contributions")
        q25, median, q75 = np.quantile(observed, [0.25, 0.5, 0.75])
        medians.append(float(median))
        iqrs.append(float(q75 - q25))
    return {
        "medians": medians,
        "raw_iqrs": iqrs,
        "scales": [max(value, scale_floor) for value in iqrs],
        "floor_active": [value < scale_floor for value in iqrs],
        "scale_floor": scale_floor,
        "normal_calibration_samples": len(normal),
        "normal_foreground_pixels": int(foreground.sum()),
        "foreground_threshold": -0.95,
        "layer_weights": [1 / len(layers)] * len(layers),
        "definition": "per-layer (d-median)/max(IQR,1e-6), equal mean; no clipping or spatial gating",
    }


def apply_layer_calibration(layers, fitted):
    values = np.asarray(layers, dtype=np.float64)
    if values.ndim < 3 or values.shape[0] != len(fitted["medians"]):
        raise ValueError("Layer count differs from calibration")
    if not np.isfinite(values).all():
        raise ValueError("Nonfinite layer maps")
    shape = (len(values),) + (1,) * (values.ndim - 1)
    medians = np.asarray(fitted["medians"]).reshape(shape)
    scales = np.asarray(fitted["scales"]).reshape(shape)
    if not np.isfinite(scales).all() or (scales <= 0).any():
        raise ValueError("Invalid fitted scales")
    return ((values - medians) / scales).mean(axis=0)


def evaluate_calibrated_lpips(
    *,
    cached,
    groups,
    perceptual_model,
    original_results,
    output_dir,
    device,
    smoke=False,
    on_progress=None,
):
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=False)
    images, finals, masks, labels = [cached[k] for k in ("images", "finals", "masks", "labels")]
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
    provenance = perceptual_provenance(perceptual_model)
    if provenance["state_sha256"] != LPIPS_STATE_SHA:
        raise ValueError("Registered LPIPS weights differ")
    selected = (
        [int(np.flatnonzero(labels == target)[0]) for target in (0, 1)]
        if smoke
        else list(range(83))
    )
    normal = list(calibration[labels[calibration] == 0])
    converted = sorted(set(normal) | set(selected))
    layers = np.full((7, 83, 240, 240), np.nan, dtype=np.float32)
    totals = np.full((83, 240, 240), np.nan, dtype=np.float32)
    max_replay_error, max_sum_error = 0.0, 0.0
    calls = 0

    def extract(index):
        nonlocal calls, max_replay_error, max_sum_error
        total, contribution, error = layer_contributions(
            perceptual_model, images[index], finals[index], device
        )
        calls += 1
        replay_error = float(np.abs(total - cached["autoddpm_lpips"][index]).max())
        max_replay_error = max(max_replay_error, replay_error)
        max_sum_error = max(max_sum_error, error)
        if replay_error > 1e-7:
            raise ValueError("Cached final LPIPS aggregate replay differs")
        totals[index], layers[:, index] = total, contribution

    for index in normal:
        extract(index)
    fitted = fit_layer_calibration(layers, images[:, 8:-8, 8:-8], labels, calibration)
    (output / "normal_layer_parameters.json").write_text(
        json.dumps(fitted, indent=2, allow_nan=False) + "\n"
    )
    if on_progress:
        on_progress({"stage": "normal_layer_statistics_frozen", "parameters": fitted})
    for index in selected:
        if index not in normal:
            extract(index)
    maps = {
        "original_lpips": totals[selected].astype(np.float64),
        "uncalibrated_equal_mean": totals[selected].astype(np.float64) / 7,
        "normal_calibrated_equal_mean": apply_layer_calibration(layers[:, selected], fitted),
    }
    naive_maps = []
    for index in selected:
        # Diagnostic only: identical original pseudo-mask; never create a new mask.
        roi = np.pad(cached["predicted_masks"][index].astype(bool), 8)
        stitched = np.where(roi, cached["initials"][index], images[index])
        total, _, error = layer_contributions(perceptual_model, images[index], stitched, device)
        calls += 1
        max_sum_error = max(max_sum_error, error)
        naive_maps.append(total)
    maps["naive_stitch_lpips"] = np.stack(naive_maps)
    fits, thresholds = {}, {}
    if not smoke:
        for name, values in maps.items():
            fits[name] = evaluate_maps(
                values[calibration],
                masks[calibration, 8:-8, 8:-8],
                labels[calibration],
                split="validation",
            )
            thresholds[name] = {
                level: fits[name][level]["threshold"] for level in ("pixel", "image")
            }
    (output / "score_thresholds.json").write_text(json.dumps(thresholds, indent=2) + "\n")
    if on_progress:
        on_progress({"stage": "score_thresholds_frozen"})
    result = {
        "method": "normal_calibrated_seven_layer_SqueezeNet_LPIPS_on_DDIM20_30_final",
        "smoke": smoke,
        "samples": len(selected),
        "held_out_test_used": False,
        "diffusion_model_calls": 0,
        "encoder_changed": False,
        "mask_changed": False,
        "perceptual": provenance,
        "partition": {"calibration": 40, "holdout": 43},
        "layer_parameters": fitted,
        "lpips_pair_calls": calls,
        "final_pairs_processed": len(converted),
        "naive_pairs_processed": len(selected),
        "aggregate_replay_max_error": max_replay_error,
        "layer_sum_max_error": max_sum_error,
        "metrics": {},
        "caveat": "Fixed normal-only layer calibration on cached final images. Naive stitching is a diagnostic, no learned reverse calls. Inspected validation/one model, no holdout design selection or independent confirmation.",
    }
    if not smoke:
        for name, values in maps.items():
            result["metrics"][name] = {
                "calibration": fits[name],
                "group_holdout": evaluate_maps(
                    values[holdout],
                    masks[holdout, 8:-8, 8:-8],
                    labels[holdout],
                    split="validation",
                    thresholds=thresholds[name],
                ),
            }
        baseline = result["metrics"]["original_lpips"]["group_holdout"]["pixel"][
            "average_precision"
        ]
        reference = original_results["metrics"]["autoddpm_lpips"]["group_holdout"]["pixel"][
            "average_precision"
        ]
        mean_ap = result["metrics"]["uncalibrated_equal_mean"]["group_holdout"]["pixel"][
            "average_precision"
        ]
        if not np.isclose(baseline, reference, atol=1e-10, rtol=0) or not np.isclose(
            mean_ap, baseline, atol=1e-10, rtol=0
        ):
            raise ValueError("Baseline replay or positive-scaling AP control differs")
        save_examples(output, cached, maps, selected, calibration, holdout)
    np.savez_compressed(
        output / "private_layer_maps.npz",
        layers=layers[:, converted],
        converted_positions=converted,
        **maps,
    )
    (output / "results.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    return result


def save_examples(output, cached, maps, selected, calibration, holdout):
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    for partition, indices in (("calibration", calibration), ("holdout", holdout)):
        for target in (0, 1):
            index = next(int(i) for i in indices if cached["labels"][i] == target)
            position = selected.index(index)
            values = [
                cached["images"][index, 8:-8, 8:-8],
                cached["finals"][index, 8:-8, 8:-8],
                maps["original_lpips"][position],
                maps["normal_calibrated_equal_mean"][position],
                maps["naive_stitch_lpips"][position],
                cached["masks"][index, 8:-8, 8:-8],
            ]
            titles = [
                "Input",
                "Final DDIM20+30",
                "Original LPIPS",
                "Normal-calibrated layers",
                "Naive stitch LPIPS",
                "Ground truth",
            ]
            fig, axes = plt.subplots(1, 6, figsize=(18, 3))
            for axis, value, title in zip(axes, values, titles, strict=True):
                axis.imshow(value, cmap="gray" if title in titles[:2] + titles[-1:] else "inferno")
                axis.set_title(title, fontsize=8)
                axis.axis("off")
            fig.tight_layout()
            fig.savefig(Path(output) / f"private_{partition}_class{target}.png")
            plt.close(fig)
