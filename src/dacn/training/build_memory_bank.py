"""Build a frozen feature bank using normal training images only; no optimization."""

import hashlib
from pathlib import Path

import torch

from dacn.models.diffusion.spatial_condition import WEIGHTS_SHA256
from dacn.models.memory_bank import RandomPriorityBank, foreground_descriptors, save_bank


def build_memory_bank(
    dataset,
    encoder,
    output,
    *,
    revision,
    budget=10000,
    seed=17,
    max_samples=None,
    device="cuda",
    on_progress=None,
):
    if dataset.split != "train" or dataset.include_masks or (dataset.labels != 0).any():
        raise ValueError("Bank requires mask-free normal training split only")
    if max_samples is None and len(dataset) != 7500:
        raise ValueError("Full bank requires7500 normal training slices")
    count = len(dataset) if max_samples is None else min(max_samples, len(dataset))
    if count < 1:
        raise ValueError("Empty training subset")
    encoder.eval()
    sampler = RandomPriorityBank(budget, seed)
    with torch.inference_mode():
        for start in range(0, count, 8):
            batch = [dataset[i] for i in range(start, min(start + 8, count))]
            if any(int(label) != 0 for _, label in batch):
                raise ValueError("Non-normal training sample")
            images = torch.stack([image for image, _ in batch]).to(device)
            descriptors = foreground_descriptors(encoder(images), images)
            sampler.update(descriptors.cpu().numpy())
            if on_progress and (start == 0 or (start // 8) % 50 == 0):
                on_progress(min(start + 8, count))
    if sampler.values is None or not len(sampler.values):
        raise ValueError("No foreground descriptors")
    metadata = {
        "dataset_revision": revision,
        "train_index_sha256": hashlib.sha256(
            (Path(dataset.split_dir) / "index.csv").read_bytes()
        ).hexdigest(),
        "encoder_sha256": WEIGHTS_SHA256,
        "layers": ["layer2", "layer3"],
        "descriptor_shape": [384, 32, 32],
        "preprocessing": "DACN grayscale [-1,1], padded256; meanRGB conv1",
        "foreground": "input>-0.95; >=0.5 fraction per8x8 cell",
        "sampling": "uniform random priorities without replacement; not coreset",
        "seed": seed,
        "budget": budget,
        "training_samples": count,
        "eligible_descriptors": sampler.seen,
        "bank_vectors": len(sampler.values),
        "normal_only": True,
        "split": "train",
        "smoke": max_samples is not None,
    }
    checksum = save_bank(output, sampler.values, metadata)
    return sampler.values, metadata, checksum
