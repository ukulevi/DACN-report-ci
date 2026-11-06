"""Frozen RAD-DINO spatial feature distance on fixed MRI reconstruction pairs."""

import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from sklearn.model_selection import StratifiedGroupKFold

from dacn.evaluation.calibrated_lpips import apply_layer_calibration, fit_layer_calibration
from dacn.evaluation.validation import evaluate_maps

ENCODER_REPO = "microsoft/rad-dino"
ENCODER_REVISION = "110cbc18d5133582e320b43d53bf5c44e410c936"
TAPS = (3, 6, 9, 12)


def prepare_pair(processor, image, reconstruction):
    """Crop padding, round both sides to RGB uint8, then official processor."""
    pair = []
    for value in (image, reconstruction):
        if value.shape != (256, 256) or not np.isfinite(value).all():
            raise ValueError("Expected finite padded256 images")
        if (np.abs(value) > 1).any():
            raise ValueError("Expected images in[-1,1]")
        pixels = np.rint((value[8:-8, 8:-8] + 1) * 127.5).astype(np.uint8)
        pair.append(Image.fromarray(pixels).convert("RGB"))
    values = processor(images=pair, return_tensors="pt")["pixel_values"]
    if values.shape != (2, 3, 518, 518) or not torch.isfinite(values).all():
        raise ValueError("Official processor must preserve full square ROI at518")
    return values


def token_distances(hidden_states, layernorm):
    """Exclude CLS; shared final LayerNorm then per-token channel L2 normalization."""
    layers = []
    for tap in TAPS:
        tokens = layernorm(hidden_states[tap])
        if tokens.shape != (2, 1370, 768):
            raise ValueError("Expected CLS+37x37 patch tokens,768channels")
        tokens = tokens[:, 1:]
        tokens = tokens / torch.sqrt(tokens.square().sum(dim=-1, keepdim=True) + 1e-10)
        distances = (tokens[0] - tokens[1]).square().mean(dim=-1).reshape(1, 1, 37, 37)
        layers.append(
            F.interpolate(distances, size=(240, 240), mode="bilinear", align_corners=False)[0, 0]
        )
    result = torch.stack(layers).float().cpu().numpy()
    if not np.isfinite(result).all():
        raise ValueError("Nonfinite spatial distances")
    return result


@torch.inference_mode()
def pair_distance(model, processor, image, reconstruction, device):
    pixels = prepare_pair(processor, image, reconstruction).to(device)
    result = model(pixel_values=pixels, output_hidden_states=True, return_dict=True)
    return token_distances(result.hidden_states, model.layernorm)


def evaluate_rad_dino(*, cached, groups, model, processor, output_dir, smoke, progress):
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=False)
    images, labels = cached["images"], cached["labels"]
    calibration, holdout = next(
        StratifiedGroupKFold(2, shuffle=True, random_state=17).split(
            np.zeros(len(labels)), labels, groups
        )
    )
    if len(images) != 83 or len(calibration) != 40 or len(holdout) != 43:
        raise ValueError("Registered83/40/43 partition required")
    if not set(groups[calibration]).isdisjoint(groups[holdout]):
        raise ValueError("Patient groups overlap")
    selected = [int(np.flatnonzero(labels == v)[0]) for v in (0, 1)] if smoke else list(range(83))
    normal = list(calibration[labels[calibration] == 0])
    layers = np.full((4, 83, 240, 240), np.nan, dtype=np.float32)
    identical_error = 0.0
    if smoke:
        identical_error = float(
            np.abs(
                pair_distance(model, processor, images[selected[0]], images[selected[0]], "cuda")
            ).max()
        )
        if identical_error > 1e-8:
            raise ValueError("Identical image pair must produce zero distance")
    for i in normal:
        layers[:, i] = pair_distance(model, processor, images[i], cached["finals"][i], "cuda")
    fitted = fit_layer_calibration(layers, images[:, 8:-8, 8:-8], labels, calibration)
    (output / "normal_parameters.json").write_text(
        json.dumps(fitted, indent=2, allow_nan=False) + "\n"
    )
    progress("normal_parameters_frozen")
    for i in selected:
        if i not in normal:
            layers[:, i] = pair_distance(model, processor, images[i], cached["finals"][i], "cuda")
    maps = {
        "rad_dino_raw": layers[:, selected].astype(np.float64).mean(axis=0),
        "rad_dino_normal_calibrated": apply_layer_calibration(layers[:, selected], fitted),
        "original_lpips": cached["autoddpm_lpips"][selected].astype(np.float64),
    }
    fits, thresholds, metrics = {}, {}, {}
    truth = cached["masks"][:, 8:-8, 8:-8]
    if not smoke:
        for name, values in maps.items():
            fits[name] = evaluate_maps(
                values[calibration], truth[calibration], labels[calibration], split="validation"
            )
            thresholds[name] = {
                level: fits[name][level]["threshold"] for level in ("pixel", "image")
            }
    (output / "thresholds.json").write_text(json.dumps(thresholds, indent=2) + "\n")
    progress("score_thresholds_frozen")
    if not smoke:
        for name, values in maps.items():
            metrics[name] = {
                "calibration": fits[name],
                "group_holdout": evaluate_maps(
                    values[holdout],
                    truth[holdout],
                    labels[holdout],
                    split="validation",
                    thresholds=thresholds[name],
                ),
            }
        if (
            abs(
                metrics["original_lpips"]["group_holdout"]["pixel"]["average_precision"]
                - 0.3123301640701585
            )
            > 1e-10
        ):
            raise ValueError("Original LPIPS control differs")
    converted = sorted(set(normal) | set(selected))
    np.savez_compressed(
        output / "private_maps.npz",
        layers=layers[:, converted],
        converted_positions=converted,
        **maps,
    )
    return {
        "method": "frozen_RAD_DINO_spatial_feature_distance_on_DDIM20_30_final",
        "samples": len(selected),
        "smoke": smoke,
        "held_out_test_used": False,
        "diffusion_model_calls": 0,
        "encoder_training_steps": 0,
        "mask_changed": False,
        "partition": {"calibration": 40, "holdout": 43},
        "taps": list(TAPS),
        "patch_grid": [37, 37],
        "feature_channels": 768,
        "layer_parameters": fitted,
        "identical_pair_max_error": identical_error if smoke else None,
        "encoder_pair_calls": len(converted) + int(smoke),
        "metrics": metrics,
        "primary_score": "rad_dino_raw",
        "secondary_score": "rad_dino_normal_calibrated",
        "caveat": "Chest-X-ray pretrained encoder transferred to MRI, not MRI-pretrained or original LPIPS. Architecture/preprocessing/head differ from SqueezeNet. Inspected validation, no independent confirmation, no layer/weight search.",
    }
