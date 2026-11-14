"""Extend the DDIM20+30 noise sweep to the advisor's t50/t400 points (AF-05).

Re-implements the producer DDIM20+30 single-pass pipeline for any phase-1 start:
DDIM phase 1 from t, LPIPS x clip(L1/P95) heatmap, pooled-95th-percentile mask
threshold on normal calibration slices with dilation3, masked DDIM30 from t50, and
the producer's eight score maps. The t300 DDIM20+30 control must reproduce the
producer arrays before new points run. All other components stay fixed; the
held-out test split is never used and nothing is selected.
"""

from functools import partial

import numpy as np
import torch

from dacn.evaluation.autoddpm_ddim import _timed
from dacn.evaluation.autoddpm_lpips import fit_mask_threshold, pseudo_mask
from dacn.evaluation.ddim_ablation import ablation_pair_scores, four_ablation_scores
from dacn.evaluation.ddim_inpainting import (
    ddim_indices,
    ddim_initial_reconstruction,
    ddim_stitch_resample,
)
from dacn.training.ddpm_partial import derive_sampling_seed

CONTROL_POINT = (300, 20, 30)
NEW_POINTS = ((50, 20, 30), (50, 30, 20), (400, 20, 30), (400, 30, 20))
SMOKE_POINTS = (CONTROL_POINT, (400, 20, 30))
PHASE2_START = 50
SCORE_NAMES = (
    "ddpm_l1",
    "autoddpm_l1",
    "autoddpm_lpips",
    "autoddpm_lpips_residual",
    "initial_lpips_only",
    "initial_product",
    "naive_stitch_l1",
    "brightness",
)
REPLAY_ATOL = 1e-5


def point_name(start, steps1, steps2):
    return f"t{start}_{steps1}_{steps2}"


def run_point(
    *,
    model,
    diffusion,
    perceptual,
    device,
    images,
    labels,
    sample_ids,
    calibration,
    selected,
    point,
    counter,
    phase2_start=PHASE2_START,
    seed_base=17,
    mask_score="product",
):
    """Producer-equivalent DDIM phase 1 + masked phase 2 for one sweep point.

    Defaults reproduce the producer. ``seed_base`` changes only the per-sample
    sampling seeds; ``mask_score="lpips"`` builds the pseudo-mask heatmap from raw
    initial LPIPS instead of LPIPS x normalized residual (same threshold rule).
    """
    start, steps1, steps2 = point
    if mask_score not in ("product", "lpips"):
        raise ValueError("mask_score must be 'product' or 'lpips'")
    phase1 = ddim_indices(start, steps1)
    phase2 = ddim_indices(phase2_start, steps2)
    normal_calibration = [int(i) for i in calibration if labels[i] == 0]
    converted = sorted(set(normal_calibration) | set(selected))
    initials, initial_scores, heatmaps, phase1_seconds = {}, {}, {}, []
    for index in converted:
        image = torch.from_numpy(images[index])[None, None].to(device)
        before = counter["unet"]
        prediction, elapsed = _timed(
            partial(
                ddim_initial_reconstruction,
                model,
                diffusion,
                image,
                seed=derive_sampling_seed(seed_base, sample_ids[index]),
                indices=phase1,
            ),
            device,
        )
        if counter["unet"] - before != steps1 or not torch.isfinite(prediction).all():
            raise ValueError("Phase 1 call count or values invalid")
        initials[index] = prediction[0, 0].cpu().numpy()
        initial_scores[index], _ = ablation_pair_scores(
            perceptual, images[index], initials[index], device, 1e-8
        )
        heatmaps[index] = initial_scores[index][mask_score]
        phase1_seconds.append(elapsed)
    stacked = np.zeros((len(images), 240, 240), dtype=np.float32)
    for index, value in heatmaps.items():
        stacked[index] = value
    mask_threshold = fit_mask_threshold(stacked, labels, calibration)
    maps = {name: [] for name in SCORE_NAMES}
    finals, predicted_masks, phase2_seconds = [], [], []
    for index in selected:
        # Producer thresholds the float32 heatmap array, not the per-image product.
        predicted = pseudo_mask(stacked[index], mask_threshold)
        mask = torch.from_numpy(np.pad(predicted, 8))[None, None].to(device)
        image = torch.from_numpy(images[index])[None, None].to(device)
        initial = torch.from_numpy(initials[index])[None, None].to(device)
        before = counter["unet"]
        final, elapsed = _timed(
            partial(
                ddim_stitch_resample,
                model,
                image,
                initial,
                mask,
                diffusion=diffusion,
                seed=derive_sampling_seed(seed_base, sample_ids[index]),
                indices=phase2,
                repeats=1,
            ),
            device,
        )
        if counter["unet"] - before != steps2 or not torch.isfinite(final).all():
            raise ValueError("Phase 2 call count or values invalid")
        if float((final - image)[~mask].abs().max()) > 1e-7:
            raise ValueError("Outside-mask context changed")
        final = final[0, 0].cpu().numpy()
        final_scores, _ = ablation_pair_scores(perceptual, images[index], final, device, 1e-8)
        scores = {
            **four_ablation_scores(initial_scores[index], final_scores),
            "initial_lpips_only": initial_scores[index]["lpips"],
            "initial_product": initial_scores[index]["product"],
            "naive_stitch_l1": initial_scores[index]["l1"] * predicted,
            "brightness": (images[index, 8:-8, 8:-8] + 1) / 2,
        }
        for name in SCORE_NAMES:
            maps[name].append(scores[name])
        finals.append(final)
        predicted_masks.append(predicted)
        phase2_seconds.append(elapsed)
    return {
        "mask_threshold": mask_threshold,
        "phase1_indices": list(phase1),
        "phase2_indices": list(phase2),
        "initials": np.stack([initials[i] for i in selected]),
        "finals": np.stack(finals),
        "predicted_masks": np.stack(predicted_masks),
        "maps": {name: np.stack(values) for name, values in maps.items()},
        "phase1_median_seconds": float(np.median(phase1_seconds)),
        "phase2_median_seconds": float(np.median(phase2_seconds)),
    }


def check_point_replay(result, cached, selected, *, atol=REPLAY_ATOL):
    """Require the control point to reproduce the producer reference arrays."""
    threshold_error = abs(result["mask_threshold"] - cached["mask_threshold"])
    if threshold_error > 1e-9:
        raise ValueError("Control mask threshold does not replay the producer")
    if not np.array_equal(result["predicted_masks"], cached["predicted_masks"][selected]):
        raise ValueError("Control pseudo-masks do not replay the producer")
    errors = {
        "initials": float(np.max(np.abs(result["initials"] - cached["initials"][selected]))),
        "finals": float(np.max(np.abs(result["finals"] - cached["finals"][selected]))),
    }
    for name in SCORE_NAMES:
        errors[name] = float(np.max(np.abs(result["maps"][name] - cached[name][selected])))
    worst = max(errors.values())
    if worst > atol:
        raise ValueError(f"Control replay exceeds tolerance: {worst}")
    return {"max_abs_errors": errors, "threshold_abs_error": threshold_error}


# Validation-only variants for the advisor's remaining items; t300 control first.
# name: (point, phase2_start, seed_base, mask_score)
VARIANTS = {
    "control_t300_20_30": ((300, 20, 30), 50, 17, "product"),
    "t400_20_30": ((400, 20, 30), 50, 17, "product"),
    "seed18_t300_20_30": ((300, 20, 30), 50, 18, "product"),
    "seed19_t300_20_30": ((300, 20, 30), 50, 19, "product"),
    "seed18_t400_20_30": ((400, 20, 30), 50, 18, "product"),
    "seed19_t400_20_30": ((400, 20, 30), 50, 19, "product"),
    "budget_t300_10_15": ((300, 10, 15), 50, 17, "product"),
    "budget_t300_40_50": ((300, 40, 50), 50, 17, "product"),
    "phase2_t30_t300_20_30": ((300, 20, 30), 30, 17, "product"),
    "phase2_t100_t300_20_30": ((300, 20, 30), 100, 17, "product"),
    "lpips_mask_t300_20_30": ((300, 20, 30), 50, 17, "lpips"),
}
SMOKE_VARIANTS = ("control_t300_20_30", "lpips_mask_t300_20_30", "budget_t300_40_50")
# Published aggregates the matching variants must reproduce (43-holdout pixel AP).
PUBLISHED_HOLDOUT_AP = {
    "t400_20_30": ("noise-ext-full-20261114T034843Z", 0.34794637897971714),
    "lpips_mask_t300_20_30": ("lpips-only-mask-20261112T154126Z-d-validation", 0.2872849863588635),
}
PUBLISHED_LPIPS_MASK_THRESHOLD = 0.1989728808403015
