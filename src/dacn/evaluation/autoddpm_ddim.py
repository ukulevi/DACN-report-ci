"""Fixed two-phase DDIM acceleration with a cached ancestral control."""

import json
import time
from functools import partial
from pathlib import Path

import numpy as np
import torch
from sklearn.model_selection import StratifiedGroupKFold

from dacn.evaluation.autoddpm_lpips import evaluate_lpips_resampling
from dacn.evaluation.ddim_inpainting import (
    INPAINT_INDICES,
    ddim_initial_reconstruction,
)
from dacn.evaluation.masked_diffusion import stitch_resample
from dacn.evaluation.sampler_ablation import DDIM_INDICES
from dacn.training.ddpm_partial import derive_sampling_seed, partial_reconstruct


def _timed(function, device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    started = time.perf_counter()
    value = function()
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    return value, time.perf_counter() - started


def evaluate_ddim_autoddpm(
    *,
    cached,
    original_private,
    original_results,
    groups,
    sample_ids,
    model,
    diffusion,
    perceptual_model,
    output_dir,
    device,
    smoke=False,
    on_progress=None,
):
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=False)
    images, labels = cached["images"], cached["labels"]
    calibration, holdout = next(
        StratifiedGroupKFold(2, shuffle=True, random_state=17).split(
            np.zeros(len(labels)), labels, groups
        )
    )
    if len(images) != 83 or len(calibration) != 40 or len(holdout) != 43:
        raise ValueError("Registered83/40/43 dataset required")
    selected = (
        [int(np.flatnonzero(labels == target)[0]) for target in (0, 1)]
        if smoke
        else list(range(83))
    )
    paired = [int(np.flatnonzero(labels == target)[0]) for target in (0, 1)]
    counter = {"calls": 0}

    def counted(*_args):
        counter["calls"] += 1

    handle = model.register_forward_hook(counted)
    try:
        # Fresh two-image reference measures both phases; do not compare new runtime
        # with old cache-inclusive scoring runtime as if it were end-to-end sampling.
        probe = []
        for index in paired:
            image = torch.from_numpy(images[index])[None, None].to(device)
            seed = derive_sampling_seed(17, sample_ids[index])
            initial, phase1 = _timed(
                partial(
                    partial_reconstruct,
                    model,
                    image,
                    starting_timestep=300,
                    seed=seed,
                    diffusion=diffusion,
                ),
                device,
            )
            initial_error = float(
                np.abs(initial[0, 0].cpu().numpy() - cached["reconstructions"][index]).max()
            )
            if initial_error > 1e-6:
                raise ValueError("Ancestral initial replay differs")
            mask = torch.from_numpy(np.pad(original_private["predicted_masks"][index], 8))
            mask = mask.to(device=device, dtype=torch.bool)[None, None]
            final, phase2 = _timed(
                partial(
                    stitch_resample,
                    model,
                    image,
                    initial,
                    mask,
                    diffusion=diffusion,
                    seed=seed,
                    steps=50,
                    repeats=5,
                ),
                device,
            )
            final_error = float(
                np.abs(final[0, 0].cpu().numpy() - original_private["reconstructions"][index]).max()
            )
            if final_error > 1e-6:
                raise ValueError("Ancestral inpainting replay differs")
            probe.append(
                {
                    "phase1_seconds": phase1,
                    "phase2_seconds": phase2,
                    "initial_replay_error": initial_error,
                    "final_replay_error": final_error,
                }
            )
        if on_progress:
            on_progress({"stage": "ancestral_two_image_timing_replay_passed"})
        # Smoke converts normal calibration and the two selected images only;
        # unconverted placeholder entries are never used by smoke scoring/inpainting.
        convert = (
            sorted(set(calibration[labels[calibration] == 0]) | set(selected))
            if smoke
            else list(range(83))
        )
        ddim_cache = {**cached, "reconstructions": cached["reconstructions"].copy()}
        phase1_seconds = {}
        for position, index in enumerate(convert):
            image = torch.from_numpy(images[index])[None, None].to(device)
            seed = derive_sampling_seed(17, sample_ids[index])
            prediction, elapsed = _timed(
                partial(ddim_initial_reconstruction, model, diffusion, image, seed=seed), device
            )
            if not torch.isfinite(prediction).all():
                raise ValueError("Nonfinite DDIM phase1")
            ddim_cache["reconstructions"][index] = prediction[0, 0].cpu().numpy()
            phase1_seconds[index] = elapsed
            if on_progress and (position + 1 == len(convert) or (position + 1) % 20 == 0):
                on_progress(
                    {
                        "stage": "ddim_phase1",
                        "completed_samples": position + 1,
                        "total_samples": len(convert),
                    }
                )
        np.savez_compressed(
            output / "private_ddim_initials.npz",
            converted_positions=np.asarray(convert),
            reconstructions=ddim_cache["reconstructions"][convert],
        )
        arms = {}
        for name, values, initial_sampler in (
            ("ancestral_initial_ddim_inpaint", cached, "ancestral300"),
            ("ddim_both", ddim_cache, "ddim10"),
        ):

            def progress(value, arm=name):
                if on_progress:
                    on_progress({"arm": arm, **value})

            arms[name] = evaluate_lpips_resampling(
                cached=values,
                groups=groups,
                sample_ids=sample_ids,
                model=model,
                diffusion=diffusion,
                perceptual_model=perceptual_model,
                output_dir=output / name,
                device=device,
                smoke=smoke,
                on_progress=progress,
                initial_sampler=initial_sampler,
                inpainting_sampler="ddim10",
            )
            if name == "ancestral_initial_ddim_inpaint":
                if not np.isclose(
                    arms[name]["parameters"]["mask_threshold"],
                    original_results["parameters"]["mask_threshold"],
                    atol=1e-8,
                    rtol=0,
                ):
                    raise ValueError("Unchanged phase1 mask threshold differs")
                if not smoke and not np.isclose(
                    arms[name]["metrics"]["initial_heatmap"]["group_holdout"]["pixel"][
                        "average_precision"
                    ],
                    original_results["metrics"]["initial_heatmap"]["group_holdout"]["pixel"][
                        "average_precision"
                    ],
                    atol=1e-10,
                    rtol=0,
                ):
                    raise ValueError("Unchanged phase1 heatmap AP differs")
        expected_calls = len(paired) * 550 + len(convert) * 10 + len(selected) * 100
        if counter["calls"] != expected_calls:
            raise ValueError("Unexpected U-Net call count")
        result = {
            "method": "AutoDDPM_two_phase_DDIM10_eta0_acceleration",
            "smoke": smoke,
            "samples": len(selected),
            "held_out_test_used": False,
            "eta": 0,
            "repeats": 5,
            "phase1_indices_zero_based": list(DDIM_INDICES),
            "phase2_indices_zero_based": list(INPAINT_INDICES),
            "terminal_index": -1,
            "forward_repeat_jump": "alpha_bar[current]/alpha_bar[following], clean alpha=1",
            "new_unet_calls_this_run": counter["calls"],
            "algorithm_unet_calls_per_image": {
                "ancestral_both": 550,
                "ancestral_initial_ddim_inpaint": 350,
                "ddim_both": 60,
            },
            "phase1_ddim_generated_samples": len(convert),
            "phase1_ddim_synchronized_seconds": {
                "total_generated": float(sum(phase1_seconds.values())),
                "mean_selected": float(np.mean([phase1_seconds[i] for i in selected])),
                "median_selected": float(np.median([phase1_seconds[i] for i in selected])),
            },
            "ancestral_two_image_timing_probe": probe,
            "cached_ancestral_reference": original_results,
            "arms": arms,
            "timing_note": "Probe is two images only on current GPU. DDIM phase1 and phase2 timings exclude LPIPS/metrics/I-O. Full ancestral accuracy reused from immutable cache; not freshly timed on83. Pipeline remains stochastic due to context/forward re-noising even with eta0. NFE ratio550/60 is not a measured wall speedup.",
            "caveat": "Fixed sampler acceleration, no mask-design or holdout parameter search. DDIM phase1 changes reconstructions and masks; its threshold is independently fit by the unchanged normal-calibration rule. One seed/inspected validation, not independent method confirmation.",
        }
        (output / "results.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
        return result
    finally:
        handle.remove()
