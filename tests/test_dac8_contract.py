"""Focused checks for DAC-8's runner boundary and reconstruction contract."""

from types import SimpleNamespace

import numpy as np
import pytest

torch = pytest.importorskip("torch", reason="DAC-8 requires the train extra")
pytest.importorskip("numba", reason="DAC-8 requires the train extra")

anoddpm = pytest.importorskip("dacn.training.anoddpm")
ddpm_partial = pytest.importorskip("dacn.training.ddpm_partial")


class FakeDataset(torch.utils.data.Dataset):
    def __init__(self, _directory, split, *, include_masks):
        self.labels = np.asarray([0, 0] if split == "train" else [0, 1], dtype=np.uint8)
        self.records = [("shard-00000.npz", i, int(label)) for i, label in enumerate(self.labels)]
        self.include_masks = include_masks

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        image = torch.zeros(1, 256, 256)
        label = torch.tensor(int(self.labels[index]), dtype=torch.uint8)
        if self.include_masks:
            return image, torch.zeros(256, 256, dtype=torch.uint8), label
        return image, label


class TinyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(0.0))


class FakeDiffusion:
    timesteps = 1000

    def training_loss(self, model, images):
        return (model.weight - 1).square(), {}

    def reconstruct(self, model, images, *, steps):
        return torch.randn_like(images) * 0.1


def test_restore_rng_state_moves_cuda_states_to_cpu(monkeypatch):
    class FakeCudaState:
        def cpu(self):
            return torch.tensor([1, 2, 3], dtype=torch.uint8)

    state = ddpm_partial._rng_state()
    state["cuda"] = [FakeCudaState()]
    restored = []
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "set_rng_state_all", restored.extend)

    ddpm_partial._restore_rng_state(state)

    assert len(restored) == 1
    assert restored[0].device.type == "cpu"
    assert restored[0].dtype == torch.uint8


def test_sampling_seeds_are_stable_per_uuid_and_reconstruction_is_batch_independent():
    first_id = "12345678-1234-5678-1234-567812345678"
    second_id = "abcdefab-1234-5678-1234-567812345678"
    first_seed = ddpm_partial.derive_sampling_seed(17, first_id)
    second_seed = ddpm_partial.derive_sampling_seed(17, second_id)
    assert first_seed == 1674000798
    assert first_seed == ddpm_partial.derive_sampling_seed(17, first_id.upper())
    assert first_seed != second_seed
    model = TinyModel()
    images = torch.zeros(1, 1, 256, 256)
    diffusion = FakeDiffusion()

    def reconstruct(seed):
        return ddpm_partial.partial_reconstruct(
            model, images, starting_timestep=100, seed=seed, diffusion=diffusion
        )

    first = reconstruct(first_seed)
    other = reconstruct(second_seed)
    assert torch.equal(first, reconstruct(first_seed))
    assert not torch.equal(first, other)
    with pytest.raises(ValueError, match="one sample per call"):
        ddpm_partial.partial_reconstruct(
            model,
            images.repeat(2, 1, 1, 1),
            starting_timestep=100,
            seed=first_seed,
            diffusion=diffusion,
        )


def test_partial_ddpm_uses_one_based_timestep_and_full_reverse_chain(monkeypatch):
    diffusion = ddpm_partial.build_diffusion({})
    seen = []

    def q_sample(clean, timesteps, noise):
        seen.append(("forward", int(timesteps[0])))
        return clean

    def p_sample(model, noisy, timesteps):
        seen.append(("reverse", int(timesteps[0])))
        return noisy

    monkeypatch.setattr(diffusion, "q_sample", q_sample)
    monkeypatch.setattr(diffusion, "p_sample", p_sample)
    diffusion.reconstruct(None, torch.zeros(1, 1, 4, 4), steps=3)
    assert seen == [("forward", 2), ("reverse", 2), ("reverse", 1), ("reverse", 0)]


def test_anoddpm_accepts_runner_metadata_but_rejects_existing_artifacts(tmp_path, monkeypatch):
    class SamplerReached(Exception):
        pass

    def stop_before_training(*args, **kwargs):
        raise SamplerReached

    monkeypatch.setattr(anoddpm, "ShardDataset", FakeDataset)
    monkeypatch.setattr(anoddpm, "ShardBatchSampler", stop_before_training)
    (tmp_path / "run_manifest.json").write_text("{}", encoding="utf-8")
    (tmp_path / "source_snapshot.zip").write_bytes(b"snapshot")
    kwargs = {
        "train_dir": tmp_path,
        "validation_dir": tmp_path,
        "output_dir": tmp_path,
        "config": {"max_train_samples": 2, "max_validation_samples": 2},
        "device": torch.device("cpu"),
    }
    with pytest.raises(SamplerReached):
        anoddpm.train(**kwargs)
    (tmp_path / "anoddpm_best.pt").write_bytes(b"existing")
    with pytest.raises(ValueError, match="artifacts from another run"):
        anoddpm.train(**kwargs)


def test_epoch_time_is_not_double_counted_after_resume(tmp_path, monkeypatch):
    monkeypatch.setattr(ddpm_partial, "ShardDataset", FakeDataset)
    monkeypatch.setattr(ddpm_partial, "build_model", lambda config, device: TinyModel())
    monkeypatch.setattr(ddpm_partial, "build_diffusion", lambda config: FakeDiffusion())
    ticks = iter([100.0, 103.0, 200.0, 202.0])
    monkeypatch.setattr(ddpm_partial, "time", SimpleNamespace(perf_counter=lambda: next(ticks)))
    config = {
        "epochs": 2,
        "checkpoint_epochs": [1, 2],
        "max_train_samples": 2,
        "max_validation_samples": 2,
        "preview_samples": 2,
        "batch_size": 1,
    }
    kwargs = {
        "train_dir": tmp_path,
        "validation_dir": tmp_path,
        "output_dir": tmp_path,
        "config": config,
        "device": torch.device("cpu"),
    }

    class Interrupted(Exception):
        pass

    def interrupt_after_checkpoint(progress):
        assert progress["epoch"] == 1
        raise Interrupted

    with pytest.raises(Interrupted):
        ddpm_partial.train(**kwargs, on_checkpoint=interrupt_after_checkpoint)
    assert (tmp_path / "ddpm_epoch_001.pt").is_file()
    result = ddpm_partial.train(**kwargs, resume_from_checkpoint=tmp_path / "ddpm_last.pt")
    assert result["epochs_completed"] == 2
    assert result["training_seconds"] == 5.0
    assert result["optimizer_steps"] == 4
    assert result["milestone_checkpoints"] == ["ddpm_epoch_001.pt", "ddpm_epoch_002.pt"]
    assert (tmp_path / "ddpm_epoch_002.pt").is_file()
