import csv

import numpy as np
import pytest

torch = pytest.importorskip(
    "torch", reason="AnoDDPM tests require the optional train extra", exc_type=ImportError
)
pytest.importorskip(
    "numba", reason="Simplex tests require the optional train extra", exc_type=ImportError
)


def _write_split(root, split, labels):
    folder = root / split
    folder.mkdir(parents=True)
    images = np.zeros((len(labels), 1, 256, 256), dtype=np.float32)
    masks = np.zeros((len(labels), 256, 256), dtype=np.uint8)
    for index, label in enumerate(labels):
        if label:
            masks[index, 120:124, 120:124] = 1
    labels_array = np.asarray(labels, dtype=np.uint8)
    np.savez_compressed(folder / "shard-00000.npz", images=images, masks=masks, labels=labels_array)
    with (folder / "index.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=(
                "sample_id",
                "patient_group_id",
                "split",
                "label",
                "shard",
                "offset",
            ),
        )
        writer.writeheader()
        for index, label in enumerate(labels):
            writer.writerow(
                {
                    "sample_id": f"sample-{index}",
                    "patient_group_id": f"group-{index}",
                    "split": split,
                    "label": label,
                    "shard": f"data/{split}/shard-00000.npz",
                    "offset": index,
                }
            )


def test_shard_dataset_and_sampler_cover_full_split(tmp_path):
    from dacn.data.shards import ShardBatchSampler, ShardDataset

    _write_split(tmp_path, "train", [0, 0, 0])
    dataset = ShardDataset(tmp_path / "train", "train", include_masks=False)
    assert dataset.sample_ids == ["sample-0", "sample-1", "sample-2"]
    sampler = ShardBatchSampler(dataset, batch_size=2, seed=17)
    indices = [index for batch in sampler for index in batch]
    assert sorted(indices) == [0, 1, 2]
    limited = ShardBatchSampler(dataset, batch_size=2, seed=17, indices=[0, 2])
    limited_indices = [index for batch in limited for index in batch]
    assert sorted(limited_indices) == [0, 2]
    image, label = dataset[0]
    assert image.shape == (1, 256, 256)
    assert label.item() == 0


def test_beta_schedules_are_valid():
    from dacn.models.diffusion.process import get_beta_schedule

    for name in ("linear", "cosine"):
        betas = get_beta_schedule(100, name)
        assert betas.shape == (100,)
        assert np.all((0 < betas) & (betas < 1))


def test_unet_and_diffusion_loss_preserve_image_shape():
    from dacn.models.diffusion.process import GaussianDiffusion
    from dacn.models.diffusion.unet import UNetModel

    model = UNetModel(
        32,
        32,
        channel_mults=(1, 2),
        attention_resolutions="16,8",
        n_heads=1,
        in_channels=1,
    )
    images = torch.zeros(2, 1, 32, 32)
    diffusion = GaussianDiffusion(
        timesteps=100,
        beta_schedule="linear",
        noise_type="gaussian",
        loss_type="l2",
    )
    loss, details = diffusion.training_loss(model, images, maximum_timestep=20)
    assert torch.isfinite(loss)
    assert details["noisy"].shape == images.shape
    assert details["predicted_noise"].shape == images.shape


def test_simplex_noise_supports_more_than_one_image():
    from dacn.models.diffusion.process import GaussianDiffusion

    diffusion = GaussianDiffusion(
        timesteps=100,
        beta_schedule="linear",
        noise_type="simplex",
        simplex_octaves=1,
        simplex_frequency=8,
    )
    images = torch.zeros(2, 1, 8, 8)
    noise = diffusion.noise(images, torch.tensor([5, 10]))
    assert noise.shape == images.shape
    assert torch.isfinite(noise).all()


def test_reverse_chain_uses_gaussian_noise_with_fixed_large_variance():
    from dacn.models.diffusion.process import GaussianDiffusion

    diffusion = GaussianDiffusion(timesteps=100, beta_schedule="linear", noise_type="simplex")
    assert np.allclose(np.exp(diffusion.model_log_variance[1:]), diffusion.betas[1:])

    def reject_forward_noise(*_args, **_kwargs):
        raise AssertionError("reverse sampling must not call the configured forward-noise function")

    diffusion.noise = reject_forward_noise
    inputs = torch.zeros(1, 1, 4, 4)
    result = diffusion.p_sample(
        lambda noisy, _timesteps: torch.zeros_like(noisy),
        inputs,
        torch.tensor([1]),
    )
    assert result.shape == inputs.shape


def test_config_rejects_training_timestep_beyond_schedule():
    from dacn.training.anoddpm import _resolved_config

    with pytest.raises(ValueError, match="maximum_training_timestep"):
        _resolved_config({"timesteps": 100, "maximum_training_timestep": 101})


def test_smoke_sample_selection_is_deterministic_and_stratified():
    from dacn.training.anoddpm import _resolved_config, _select_indices

    labels = np.asarray([0] * 8 + [1] * 4, dtype=np.uint8)
    first = _select_indices(labels, 6, seed=17, stratified=True)
    second = _select_indices(labels, 6, seed=17, stratified=True)
    assert first == second
    assert set(labels[first]) == {0, 1}
    assert _resolved_config({"max_train_samples": 128})["max_train_samples"] == 128
    with pytest.raises(ValueError, match="max_validation_samples"):
        _resolved_config({"max_validation_samples": 0})
