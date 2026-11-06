"""Fixed LPIPS scoring ablation and descriptive pseudo-mask error strata."""

import json
from pathlib import Path

import numpy as np
import torch
from scipy.ndimage import distance_transform_edt, label, sobel
from sklearn.model_selection import StratifiedGroupKFold

from dacn.evaluation.autoddpm_lpips import perceptual_provenance, pseudo_mask
from dacn.evaluation.validation import evaluate_maps


@torch.inference_mode()
def perceptual_components(model, image, initial, device):
    """Separate terms of the previous heatmap; no labels, fitting or resampling."""
    image, initial = np.asarray(image), np.asarray(initial)
    if image.shape != (256, 256) or initial.shape != image.shape:
        raise ValueError("Expected padded256 image pairs")
    if not all(np.isfinite(v).all() and (np.abs(v) <= 1).all() for v in (image, initial)):
        raise ValueError("Expected finite image values in[-1,1]")
    tensors = [
        torch.from_numpy(v[8:-8, 8:-8].copy()).float()[None, None].repeat(1, 3, 1, 1).to(device)
        for v in (image, initial)
    ]
    lpips = model(*tensors, normalize=False)[0, 0].cpu().numpy()
    if lpips.shape != (240, 240) or not np.isfinite(lpips).all():
        raise ValueError("Invalid spatial LPIPS map")
    lpips = np.maximum(lpips, 0)
    residual = np.abs(image[8:-8, 8:-8] - initial[8:-8, 8:-8])
    normalized = np.clip(residual / max(float(np.quantile(residual, 0.95)), 1e-8), 0, 1)
    return {
        "lpips_only": lpips,
        "normalized_absolute_only": normalized,
        "product": lpips * normalized,
    }


def image_gradient(image):
    return np.hypot(sobel(image, axis=0, mode="reflect"), sobel(image, axis=1, mode="reflect")) / 8


def fit_diagnostic_gradient(images, labels, calibration):
    indices = np.asarray(calibration)[np.asarray(labels)[calibration] == 0]
    values = [image_gradient(images[i])[images[i] > -0.95] for i in indices]
    values = [v for v in values if len(v)]
    if not values:
        raise ValueError("Normal foreground calibration required")
    return float(np.quantile(np.concatenate(values), 0.95))


def tissue_proxy_regions(image, gradient_threshold):
    """Mutually exclusive image-derived proxies, NOT named anatomical structures."""
    foreground = image > -0.95
    # Pad to define the crop edge as outside foreground, even for all-foreground inputs.
    distance = distance_transform_edt(np.pad(foreground, 1))[1:-1, 1:-1]
    boundary = foreground & (distance <= 5)
    high_gradient = foreground & ~boundary & (image_gradient(image) > gradient_threshold)
    return {
        "background": ~foreground,
        "foreground_boundary_5px": boundary,
        "interior_high_gradient": high_gradient,
        "interior_other": foreground & ~boundary & ~high_gradient,
    }


def _ratio(numerator, denominator):
    return float(numerator / denominator) if denominator else None


def mask_error_report(images, masks, labels, predictions, indices, gradient_threshold):
    """Ground truth is accepted only here, for post-hoc diagnostic counts."""
    regions = {
        name: {
            "eligible_nonlesion_pixels": 0,
            "false_positive_pixels": 0,
            "normal_eligible_pixels": 0,
            "normal_false_positive_pixels": 0,
        }
        for name in (
            "background",
            "foreground_boundary_5px",
            "interior_high_gradient",
            "interior_other",
        )
    }
    lesions = {
        name: {"lesion_pixels": 0, "missed_pixels": 0}
        for name in ("boundary_within3px", "core_over3px")
    }
    bins = {
        name: {"components": 0, "wholly_missed": 0, "recalls": []}
        for name in ("area1_25", "area26_100", "area101_400", "area_over400")
    }
    total_fp, lesion_pixels, missed_pixels = 0, 0, 0
    for i in indices:
        image, gt, pred = images[i], masks[i].astype(bool), predictions[i].astype(bool)
        fp, missed = pred & ~gt, gt & ~pred
        total_fp += int(fp.sum())
        lesion_pixels += int(gt.sum())
        missed_pixels += int(missed.sum())
        for name, region in tissue_proxy_regions(image, gradient_threshold).items():
            values = regions[name]
            values["eligible_nonlesion_pixels"] += int((region & ~gt).sum())
            values["false_positive_pixels"] += int((fp & region).sum())
            if labels[i] == 0:
                values["normal_eligible_pixels"] += int(region.sum())
                values["normal_false_positive_pixels"] += int((fp & region).sum())
        distance = distance_transform_edt(gt)
        for name, region in (
            ("boundary_within3px", gt & (distance <= 3)),
            ("core_over3px", gt & (distance > 3)),
        ):
            lesions[name]["lesion_pixels"] += int(region.sum())
            lesions[name]["missed_pixels"] += int((missed & region).sum())
        components, count = label(gt, structure=np.ones((3, 3), dtype=int))
        for component in range(1, count + 1):
            region = components == component
            area, hit = int(region.sum()), int((region & pred).sum())
            key = (
                "area1_25"
                if area <= 25
                else "area26_100"
                if area <= 100
                else "area101_400"
                if area <= 400
                else "area_over400"
            )
            bins[key]["components"] += 1
            bins[key]["wholly_missed"] += int(hit == 0)
            bins[key]["recalls"].append(hit / area)
    for values in regions.values():
        values["false_positive_rate_within_region"] = _ratio(
            values["false_positive_pixels"], values["eligible_nonlesion_pixels"]
        )
        values["share_of_all_false_positives"] = _ratio(values["false_positive_pixels"], total_fp)
        values["normal_false_positive_rate_within_region"] = _ratio(
            values["normal_false_positive_pixels"], values["normal_eligible_pixels"]
        )
    for values in lesions.values():
        values["miss_rate"] = _ratio(values["missed_pixels"], values["lesion_pixels"])
        values["share_of_all_missed_lesion_pixels"] = _ratio(values["missed_pixels"], missed_pixels)
    for values in bins.values():
        recalls = values.pop("recalls")
        values["component_macro_recall"] = float(np.mean(recalls)) if recalls else None
    return {
        "lesion_pixels": lesion_pixels,
        "missed_lesion_pixels": missed_pixels,
        "lesion_miss_rate": _ratio(missed_pixels, lesion_pixels),
        "false_positive_pixels": total_fp,
        "false_positive_regions": regions,
        "missed_lesion_geometry": lesions,
        "lesion_components_8_connected": bins,
    }


def save_audit_examples(output, cached, components, calibration, holdout):
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    for partition, indices in (("calibration", calibration), ("holdout", holdout)):
        for binary_class in (0, 1):
            index = next(int(i) for i in indices if cached["labels"][i] == binary_class)
            gt, mask = (
                cached["masks"][index, 8:-8, 8:-8].astype(bool),
                cached["predicted_masks"][index],
            )
            overlay = np.zeros((240, 240, 3))
            overlay[mask & ~gt] = (1, 0.5, 0)  # false positives
            overlay[gt & ~mask] = (1, 0, 1)  # missed lesion pixels
            overlay[gt & mask] = (0, 1, 0)  # true positive pixels
            values = [
                cached["images"][index, 8:-8, 8:-8],
                cached["initials"][index, 8:-8, 8:-8],
                components["lpips_only"][index],
                components["normalized_absolute_only"][index],
                cached["initial_heatmap"][index],
                mask,
                overlay,
                cached["weighted_resampled_absolute"][index],
            ]
            titles = [
                "Input",
                "Initial DDPM",
                "LPIPS only",
                "Normalized residual",
                "LPIPS x residual",
                "Pseudo-mask",
                "TP green / FP orange / FN magenta",
                "Weighted final residual",
            ]
            fig, axes = plt.subplots(1, 8, figsize=(26, 3))
            for axis, value, title in zip(axes, values, titles, strict=True):
                axis.imshow(
                    value,
                    cmap="gray" if title in {"Input", "Initial DDPM", "Pseudo-mask"} else "inferno",
                )
                axis.set_title(title, fontsize=8)
                axis.axis("off")
            fig.tight_layout()
            fig.savefig(Path(output) / f"private_{partition}_class{binary_class}.png")
            plt.close(fig)


def evaluate_perceptual_audit(
    *,
    cached,
    groups,
    perceptual_model,
    original_results,
    output_dir,
    device,
    smoke=False,
    on_calibrated=None,
):
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
    provenance = perceptual_provenance(perceptual_model)
    if provenance != original_results["parameters"]["perceptual"]:
        raise ValueError("LPIPS model state/preprocessing differs from original run")
    crop, truth = images[:, 8:-8, 8:-8], masks[:, 8:-8, 8:-8]
    gradient_threshold = fit_diagnostic_gradient(crop, labels, calibration)
    selected = (
        [int(np.flatnonzero(labels == target)[0]) for target in (0, 1)]
        if smoke
        else list(range(83))
    )
    maps = {name: [] for name in ("lpips_only", "normalized_absolute_only")}
    maximum_error = 0.0
    for i in selected:
        parts = perceptual_components(perceptual_model, images[i], cached["initials"][i], device)
        error = float(np.abs(parts["product"] - cached["initial_heatmap"][i]).max())
        maximum_error = max(maximum_error, error)
        if error > 1e-7:
            raise ValueError("Original LPIPS product replay differs")
        predicted = pseudo_mask(parts["product"], original_results["parameters"]["mask_threshold"])
        if not np.array_equal(predicted, cached["predicted_masks"][i]):
            raise ValueError("Original pseudo-mask replay differs")
        for name, values in maps.items():
            values.append(parts[name])
    maps = {k: np.stack(v) for k, v in maps.items()}
    parameters = {
        "gradient_threshold_normal95": gradient_threshold,
        "foreground_boundary_width_pixels": 5,
        "lesion_boundary_width_pixels": 3,
        "perceptual": provenance,
        "score_thresholds": {},
    }
    fits = {}
    if not smoke:
        for name, values in maps.items():
            fits[name] = evaluate_maps(
                values[calibration], truth[calibration], labels[calibration], split="validation"
            )
            parameters["score_thresholds"][name] = {
                level: fits[name][level]["threshold"] for level in ("pixel", "image")
            }
    (output / "calibration_parameters.json").write_text(
        json.dumps(parameters, indent=2, allow_nan=False) + "\n"
    )
    if on_calibrated:
        on_calibrated(parameters)
    result = {
        "method": "LPIPS_only_and_pseudo_mask_explanatory_audit",
        "smoke": smoke,
        "samples": len(selected),
        "held_out_test_used": False,
        "partition": {"calibration": 40, "holdout": 43},
        "parameters": parameters,
        "product_replay_max_error": maximum_error,
        "pseudo_mask_replay_exact": True,
        "new_metrics": {},
        "original_metrics": original_results["metrics"] if not smoke else {},
        "caveat": "Explanatory ablation on previously inspected validation, not independent method confirmation. Region names are image-derived proxies, not anatomical labels. No DDPM/encoder training, reverse calls or mask redesign. No holdout-based design selection.",
    }
    if not smoke:
        predictions = {"pseudo_mask": cached["predicted_masks"].astype(bool)}
        for name, values in maps.items():
            result["new_metrics"][name] = {
                "calibration": fits[name],
                "group_holdout": evaluate_maps(
                    values[holdout],
                    truth[holdout],
                    labels[holdout],
                    split="validation",
                    thresholds=parameters["score_thresholds"][name],
                ),
            }
            predictions[name] = values >= parameters["score_thresholds"][name]["pixel"]
        for name in ("initial_heatmap", "weighted_resampled_absolute"):
            threshold = original_results["metrics"][name]["calibration"]["pixel"]["threshold"]
            predictions[name] = cached[name] >= threshold
        result["error_diagnostics"] = {
            partition: {
                name: mask_error_report(crop, truth, labels, pred, indices, gradient_threshold)
                for name, pred in predictions.items()
            }
            for partition, indices in (("calibration", calibration), ("holdout", holdout))
        }
        save_audit_examples(output, cached, maps, calibration, holdout)
    np.savez_compressed(output / "private_new_maps.npz", **maps)
    (output / "results.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    return result
