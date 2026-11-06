import hashlib

import numpy as np
import pytest
import torch

from dacn.evaluation.ddpm_memory_fusion import (
    fit_robust_normalization,
    fuse_maps,
    recall_threshold,
    verified_cache,
)
from dacn.models.diffusion.spatial_condition import WEIGHTS_SHA256, SpatialResNet18
from dacn.models.memory_bank import (
    MemoryFeatureEncoder,
    RandomPriorityBank,
    foreground_descriptors,
    load_bank,
    memory_maps,
    nearest_distances,
    save_bank,
    upsample_patch_scores,
)
from dacn.training.build_memory_bank import build_memory_bank


def test_frozen_encoder_shape_and_batchnorm_do_not_update(monkeypatch):
    # Official RGB->gray/checksum mapping is also covered in test_spatial_condition.
    monkeypatch.setattr(SpatialResNet18, "load_imagenet", lambda *a: None)
    encoder = MemoryFeatureEncoder().train()
    before = {k: v.clone() for k, v in encoder.state_dict().items() if "running_" in k}
    features = encoder(torch.zeros(2, 1, 256, 256))
    assert features.shape == (2, 384, 32, 32)
    assert not encoder.training and not any(p.requires_grad for p in encoder.parameters())
    assert not features.requires_grad
    assert all(torch.equal(value, encoder.state_dict()[key]) for key, value in before.items())


def test_priority_sampling_deterministic_across_chunk_boundaries():
    values = np.arange(100 * 4, dtype=np.float32).reshape(100, 4)
    first, second = RandomPriorityBank(10, 17), RandomPriorityBank(10, 17)
    first.update(values)
    for chunk in np.array_split(values, 7):
        second.update(chunk)
    assert first.seen == second.seen == 100
    assert np.array_equal(first.values, second.values)
    assert len(np.unique(first.values[:, 0])) == 10


def test_foreground_fraction_and_padding_alignment():
    image = torch.full((1, 1, 256, 256), -1.0)
    image[:, :, 8:16, 8:16] = 0
    features = torch.arange(1024.0).reshape(1, 1, 32, 32)
    assert foreground_descriptors(features, image).tolist() == [[33.0]]
    ramp = torch.arange(32.0).repeat(32, 1)[None]
    aligned = upsample_patch_scores(ramp)
    assert aligned.shape == (1, 240, 240)
    assert aligned[0, 0, 0].item() == pytest.approx(0.5625)
    assert aligned[0, -1, -1].item() == pytest.approx(30.4375)


def test_chunked_euclidean_matches_known_neighbors_and_inference_has_no_masks():
    query = torch.tensor([[0.0, 0.0], [3.0, 4.0], [-3.0, -4.0]])
    bank = torch.tensor([[0.0, 0.0], [3.0, 0.0]])
    assert torch.equal(nearest_distances(query, bank, 1, 1), torch.tensor([0.0, 4.0, 5.0]))
    encoder = lambda x: torch.zeros(len(x), 2, 32, 32)
    score = memory_maps(encoder, torch.zeros(1, 1, 256, 256), bank)
    assert score.shape == (1, 240, 240) and not score.any()


def test_bank_roundtrip_checks_metadata_and_bytes(tmp_path):
    path = tmp_path / "bank.npz"
    metadata = {"encoder_sha256": WEIGHTS_SHA256, "split": "train"}
    values = np.zeros((3, 384), dtype=np.float32)
    checksum = save_bank(path, values, metadata)
    assert np.array_equal(load_bank(path, checksum=checksum, expected_metadata=metadata), values)
    with pytest.raises(ValueError, match="provenance"):
        load_bank(path, checksum=checksum, expected_metadata={**metadata, "split": "val"})
    with pytest.raises(ValueError, match="checksum"):
        load_bank(path, checksum="0" * 64, expected_metadata=metadata)


def test_build_rejects_validation_abnormal_and_masks_before_encoder(tmp_path):
    class Dataset:
        split = "val"
        include_masks = False
        labels = np.array([0])

    for split, masks, label in [("val", False, 0), ("train", True, 0), ("train", False, 1)]:
        data = Dataset()
        data.split, data.include_masks, data.labels = split, masks, np.array([label])
        with pytest.raises(ValueError, match="normal training"):
            build_memory_bank(data, None, tmp_path / "bad.npz", revision="pinned")


def test_normalization_excludes_abnormal_calibration_and_every_holdout():
    values = np.arange(4 * 16, dtype=float).reshape(4, 4, 4)
    labels, calibration = np.array([0, 1, 0, 1]), np.array([0, 1])
    maps = {name: values.copy() for name in ("ddpm_absolute", "ddpm_ssim", "memory_bank")}
    original = fit_robust_normalization(maps, labels, calibration)
    for value in maps.values():
        value[1:] = 1e9
    assert original == fit_robust_normalization(maps, labels, calibration)
    assert original["memory_bank"] == {"median": 7.5, "scale": 7.5}
    constant = {k: np.ones_like(v) for k, v in maps.items()}
    fitted = fit_robust_normalization(constant, labels, calibration)
    assert all(np.isfinite(v).all() for v in fuse_maps(constant, fitted).values())


def test_recall_threshold_is_fixed_from_calibration_lesion_scores():
    scores, masks = np.array([[[0.0, 1.0, 2.0, 3.0]]]), np.ones((1, 1, 4))
    threshold = recall_threshold(scores, masks, 0.5)
    assert (scores >= threshold).mean() >= 0.5
    with pytest.raises(ValueError, match="Positive"):
        recall_threshold(scores, np.zeros_like(masks), 0.8)


def test_verified_cache_rejects_reordered_branch_inputs(tmp_path):
    images = np.zeros((83, 256, 256), np.float32)
    masks, labels = np.zeros_like(images, dtype=np.uint8), np.zeros(83, dtype=np.uint8)
    path = tmp_path / "baseline.npz"
    np.savez(path, images=images, reconstructions=images, masks=masks, labels=labels)
    (tmp_path / "index.csv").write_text("synthetic ordering")

    class Dataset:
        split, include_masks, split_dir = "val", True, tmp_path

        def __len__(self):
            return 83

        def __getitem__(self, index):
            return torch.ones(1, 256, 256), torch.from_numpy(masks[index]), 0

    checksum = hashlib.sha256(path.read_bytes()).hexdigest()
    index_sha = hashlib.sha256((tmp_path / "index.csv").read_bytes()).hexdigest()
    with pytest.raises(ValueError, match="ordering"):
        verified_cache(path, checksum, Dataset(), index_sha)
