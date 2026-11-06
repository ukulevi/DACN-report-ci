import hashlib

import numpy as np
import pytest

torch = pytest.importorskip("torch")
from torch import nn

from dacn.training import ddpm_partial, ddpm_unconditional_control


def test_ema30_finetune_has_no_encoder_resets_optimizer_and_resumes(tmp_path, monkeypatch):
    class TinyUNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = nn.Conv2d(1, 1, 1)

        def forward(self, x, t):
            return self.conv(x)

    class Data:
        def __init__(self, split):
            self.split = split
            self.labels = np.array([0, 0] if split == "train" else [0, 1, 0, 1], dtype=np.uint8)
            self.records = tuple(("shard-00000.npz", i, int(y)) for i, y in enumerate(self.labels))

        def __len__(self):
            return len(self.labels)

        def __getitem__(self, i):
            image = torch.full((1, 256, 256), i / 10)
            label = torch.tensor(self.labels[i])
            if self.split == "train":
                return image, label
            return image, torch.zeros(256, 256, dtype=torch.uint8), label

    def dataset(path, split, *, include_masks):
        assert include_masks == (split == "val")
        return Data(split)

    monkeypatch.setattr(ddpm_partial, "ShardDataset", dataset)
    monkeypatch.setattr(
        ddpm_partial, "build_model", lambda config, device="cpu": TinyUNet().to(device)
    )
    source = tmp_path / "source.pt"
    config = {
        "epochs": 30,
        "batch_size": 2,
        "validation_batch_size": 2,
        "max_train_samples": None,
        "max_validation_samples": None,
        "preview_samples": 2,
        "preview_starting_timestep": 3,
        "ema_decay": 0.9992,
    }
    original = TinyUNet()
    with torch.no_grad():
        for value in original.parameters():
            value.fill_(0.07)
    original_ema = original.state_dict()
    torch.save(
        {
            "schema_version": 1,
            "method": ddpm_partial.METHOD,
            "epoch": 30,
            "config": config,
            "ema": original_ema,
            "model": {k: torch.zeros_like(v) for k, v in original_ema.items()},
            "optimizer": {"invalid_source_optimizer": True},
        },
        source,
    )
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    requested = {**config, "epochs": 32, "learning_rate": 1e-5, "checkpoint_epochs": [31, 32]}
    output = tmp_path / "control"
    args = {
        "train_dir": tmp_path,
        "validation_dir": tmp_path,
        "output_dir": output,
        "config": requested,
        "device": "cpu",
        "initial_checkpoint": source,
        "initial_checkpoint_sha256": digest,
    }

    def interrupt(progress):
        if progress["epoch"] == 30:
            state = torch.load(output / "ddpm_last.pt", weights_only=False)
            assert state["optimizer_steps"] == 0 and not state["optimizer"]["state"]
            assert state["optimizer"]["param_groups"][0]["lr"] == 1e-5
            assert set(state["model"]) == set(original_ema)
            for key, value in original_ema.items():
                assert torch.equal(state["model"][key], value)
                assert torch.equal(state["ema"][key], value)
        if progress["epoch"] == 31:
            raise RuntimeError("synthetic preemption")

    with pytest.raises(RuntimeError, match="synthetic preemption"):
        ddpm_partial.train(**args, on_checkpoint=interrupt)
    result = ddpm_partial.train(**args, resume_from_checkpoint=output / "ddpm_last.pt")
    assert result["epochs_completed"] == 32 and result["optimizer_steps"] == 2
    assert result["ema_initialization"]["encoder_added"] is False
    state = torch.load(output / "ddpm_last.pt", weights_only=False)
    assert [entry["epoch"] for entry in state["history"]] == [31, 32]
    assert set(state["model"]) == set(original_ema)
    loaded = ddpm_partial.load_ema_checkpoint(output / "ddpm_best.pt", requested)
    assert not loaded.training
    assert hashlib.sha256(source.read_bytes()).hexdigest() == digest
    with pytest.raises(ValueError, match="resume source differs"):
        ddpm_partial.train(
            **{**args, "initial_checkpoint_sha256": "0" * 64},
            resume_from_checkpoint=output / "ddpm_last.pt",
        )


@pytest.mark.parametrize(
    "config",
    [
        {"epochs": 40, "learning_rate": 1e-4, "ema_decay": 0.9992},
        {"epochs": 40, "learning_rate": 1e-5, "ema_decay": 0.9992},
        {"epochs": 31, "learning_rate": 1e-5, "ema_decay": 0.9992},
    ],
)
def test_control_rejects_unmatched_or_unbounded_configs(tmp_path, config):
    with pytest.raises(ValueError):
        ddpm_unconditional_control.train(
            train_dir=tmp_path,
            validation_dir=tmp_path,
            output_dir=tmp_path,
            config=config,
            device="cpu",
        )
