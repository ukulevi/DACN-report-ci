"""Bounded patch/context sliding reconstruction diagnostic on validation only."""

import json
import time
from pathlib import Path

import numpy as np
import torch

from dacn.data.shards import ShardDataset
from dacn.evaluation.scores import paired_score_maps
from dacn.training.ddpm_partial import derive_sampling_seed
from dacn.training.patch_context import load_context_checkpoint, sliding_reconstruct


def evaluate_patch_preview(*, validation_dir, checkpoint_path, output_dir, device):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)
    model, diffusion = load_context_checkpoint(checkpoint_path, device)
    data = ShardDataset(validation_dir, "val", include_masks=True)
    rng = np.random.default_rng(17)
    selected = [
        int(index)
        for label in (0, 1)
        for index in np.sort(rng.choice(np.flatnonzero(data.labels == label), 2, replace=False))
    ]
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)
    started = time.perf_counter()
    saved = {
        key: [] for key in ("images", "reconstructions", "masks", "labels", "difference", "ssim")
    }
    records = []
    for ordinal, index in enumerate(selected, 1):
        image, mask, label = data[index]
        image = image[None].to(device)
        sample_start = time.perf_counter()
        reconstruction = sliding_reconstruct(
            model,
            diffusion,
            image,
            steps=100,
            seed=derive_sampling_seed(17, data.sample_ids[index]),
        )
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        elapsed = time.perf_counter() - sample_start
        x, y = image[0, 0].cpu().numpy(), reconstruction[0, 0].cpu().numpy()
        scores = paired_score_maps(x, y)
        foreground = x[8:-8, 8:-8] > -0.95
        lesion = mask[8:-8, 8:-8].numpy().astype(bool)
        assert np.isfinite(y).all()
        if not label:
            assert not lesion.any()
        # [0,1] MAE; SSIM uses data_range2 on the original [-1,1] representation.
        difference = scores["pixel_difference"] / 2
        record = {
            "ordinal": ordinal,
            "label": int(label),
            "seconds": elapsed,
            "foreground_mae_01": float(difference[foreground].mean()),
            "foreground_ssim": float((1 - scores["local_ssim"])[foreground].mean()),
            "normal_tissue_mae_01": float(difference[foreground & ~lesion].mean()),
            "lesion_mae_01": float(difference[lesion].mean()) if lesion.any() else None,
        }
        records.append(record)
        for key, value in {
            "images": x,
            "reconstructions": y,
            "masks": mask.numpy(),
            "labels": int(label),
            "difference": difference,
            "ssim": scores["local_ssim"],
        }.items():
            saved[key].append(value)
        print(json.dumps({"patch_preview_completed": ordinal, "total": 4, **record}), flush=True)
    result = {
        "split": "validation_diagnostic",
        "held_out_test_used": False,
        "selection": "seed17 uniform two per class; labels only select examples; masks never condition inference",
        "partition": {"diagnostic": {"normal": 2, "abnormal": 2}},
        "checkpoint_epoch": 5,
        "ema": True,
        "timesteps_one_based": [100],
        "patch_size": 64,
        "stride": 48,
        "patches_per_image": 25,
        "sampling": "fixed_large stochastic DDPM, independent derived seed per patch",
        "inference_seconds": time.perf_counter() - started,
        "peak_vram_bytes": torch.cuda.max_memory_allocated(device) if device.type == "cuda" else 0,
        "metrics": {"per_example": records},
        "limitation": "Four selected diagnostic examples; no AP, thresholds or population performance claim",
    }
    np.savez_compressed(
        output_dir / "private_reconstructions.npz",
        **{key: np.asarray(value) for key, value in saved.items()},
    )
    (output_dir / "results.json").write_text(json.dumps(result, indent=2) + "\n")
    return result
