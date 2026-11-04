"""Tests for validation-only partial-DDPM evaluation helpers."""

import numpy as np
import pytest

torch = pytest.importorskip("torch", reason="DDPM validation requires the train extra")

from dacn.evaluation import ddpm_validation
from dacn.evaluation.ddpm_validation import select_validation_indices


def test_validation_subset_is_deterministic_and_contains_both_classes():
    labels = np.asarray([0] * 5 + [1] * 7, dtype=np.uint8)
    first = select_validation_indices(labels, 6, seed=17)
    second = select_validation_indices(labels, 6, seed=17)

    assert first == second
    assert len(first) == 6
    assert set(labels[first]) == {0, 1}


@pytest.mark.parametrize("limit", [0, 1, 13])
def test_validation_subset_rejects_invalid_limits(limit):
    labels = np.asarray([0, 0, 1, 1], dtype=np.uint8)
    with pytest.raises(ValueError, match="sample limit"):
        select_validation_indices(labels, limit, seed=17)


def test_posterior_mean_evaluation_batches_and_writes_private_cache(tmp_path, monkeypatch):
    class FakeDataset:
        def __init__(self, directory, split, *, include_masks):
            assert split == "val"
            assert include_masks
            self.labels = np.asarray([0, 1], dtype=np.uint8)
            self.sample_ids = [
                "12345678-1234-5678-1234-567812345678",
                "abcdefab-1234-5678-1234-567812345678",
            ]

        def __getitem__(self, index):
            image = torch.zeros(1, 256, 256)
            mask = torch.zeros(256, 256, dtype=torch.uint8)
            if index == 1:
                image[:, 100:124, 100:124] = 0.5
                mask[100:124, 100:124] = 1
            return image, mask, torch.tensor(index, dtype=torch.uint8)

        def __len__(self):
            return 2

    class FakeDiffusion:
        timesteps = 1000

    checkpoint = tmp_path / "ddpm_epoch_010.pt"
    torch.save({"config": {}, "epoch": 10}, checkpoint)
    seen = {}

    def reconstruct(model, images, *, starting_timestep, seeds, diffusion):
        seen["batch"] = len(images)
        seen["seeds"] = seeds
        return images

    monkeypatch.setattr(ddpm_validation, "ShardDataset", FakeDataset)
    monkeypatch.setattr(ddpm_validation, "load_ema_checkpoint", lambda *args: object())
    monkeypatch.setattr(ddpm_validation, "build_diffusion", lambda config: FakeDiffusion())
    monkeypatch.setattr(ddpm_validation, "posterior_mean_reconstruct_batch", reconstruct)
    cache_dir = tmp_path / "private-cache"

    result = ddpm_validation.evaluate_checkpoint(
        validation_dir=tmp_path,
        checkpoint_path=checkpoint,
        starting_timesteps=[300],
        device=torch.device("cpu"),
        reverse_sampler="posterior_mean",
        batch_size=8,
        reconstruction_cache_dir=cache_dir,
    )

    timestep = result["timesteps"]["300"]
    assert seen["batch"] == 2
    assert len(seen["seeds"]) == 2
    assert timestep["reverse_sampler"] == "posterior_mean"
    assert timestep["batch_size"] == 8
    cache_path = cache_dir / timestep["reconstruction_cache"]
    with np.load(cache_path, allow_pickle=False) as cache:
        assert set(cache.files) == {"images", "reconstructions", "masks", "labels"}
        assert cache["images"].shape == (2, 256, 256)
