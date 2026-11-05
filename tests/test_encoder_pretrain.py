import hashlib

import numpy as np
import pytest
import torch
from torch import nn

from dacn.models.diffusion.feature_condition import MRIResNet18
from dacn.training import encoder_pretrain as training


def test_mask_hides_half_grid_preserves_padding_and_intersects_foreground():
    image = torch.full((2, 1, 256, 256), -1.0)
    image[:, :, 120:125, 120:125] = 0.7
    mask = training.image_patch_mask(image)
    assert mask.dtype == torch.bool and mask.shape == image.shape
    assert not mask[:, :, :8].any() and not mask[:, :, -8:].any()
    assert not mask[:, :, :, :8].any() and not mask[:, :, :, -8:].any()
    assert torch.all(mask[:, :, 16::32, 16::32].sum((1, 2, 3)) == 32)
    assert torch.all((mask & (image > -0.95)).sum((1, 2, 3)) > 0)


def test_loss_hides_target_and_scores_only_masked_foreground():
    class Capture(nn.Module):
        def forward(self, value):
            self.input = value.clone()
            return torch.zeros_like(value)

    model = Capture()
    clean = torch.tensor([[[[0.5, 0.8], [-1.0, 0.9]]]])
    mask = torch.tensor([[[[True, False], [True, False]]]])
    assert training.masked_reconstruction_loss(model, clean, mask) == 0.25
    assert torch.equal(model.input, torch.tensor([[[[0.0, 0.8], [0.0, 0.9]]]]))
    with pytest.raises(ValueError, match="foreground"):
        training.masked_reconstruction_loss(model, clean, clean < -0.95)


def test_resnet_decoder_outputs_image_and_receives_encoder_gradients():
    model = training.MaskedResNet(MRIResNet18()).train()
    clean = torch.rand(2, 1, 256, 256) * 2 - 1
    loss = training.masked_reconstruction_loss(model, clean, training.image_patch_mask(clean))
    loss.backward()
    for module in (model.encoder, model.decoder):
        assert sum(float(p.grad.abs().sum()) for p in module.parameters() if p.grad is not None) > 0
    assert torch.isfinite(loss)


def test_normal_only_pretrain_checkpoint_and_resume(tmp_path, monkeypatch):
    from dacn.models.diffusion import feature_condition as architecture

    class Encoder(nn.Module):
        def __init__(self):
            super().__init__()
            self.stem = nn.Conv2d(1, 512, 1, stride=32)
            self.layers = nn.Identity()

        def forward(self, x):
            return self.layers(self.stem(x)).mean((2, 3))

    class Backbone(nn.Module):
        def __init__(self):
            super().__init__()
            self.time_embedding = nn.Sequential(nn.Linear(1, 128))

    class Decoder(nn.Module):
        def __init__(self, encoder):
            super().__init__()
            self.encoder = encoder
            self.decoder = nn.Conv2d(512, 1, 1)

        def forward(self, x):
            return nn.functional.interpolate(self.decoder(self.encoder.stem(x)), size=(256, 256))

    class Data:
        labels = np.array([0, 0, 1], dtype=np.uint8)
        records = tuple(("shard-00000.npz", i, 0) for i in range(3))

        def __len__(self):
            return len(self.labels)

        def __getitem__(self, i):
            assert i < 2  # tumor validation is never opened by pretraining
            return torch.full((1, 256, 256), 0.3 + i / 10), torch.tensor(0)

    def dataset(path, split, *, include_masks):
        assert not include_masks
        data = Data()
        if split == "train":
            data.labels = np.array([0, 0], dtype=np.uint8)
            data.records = data.records[:2]
        return data

    monkeypatch.setattr(architecture, "MRIResNet18", Encoder)
    monkeypatch.setattr(training, "build_model", lambda *a: Backbone())
    monkeypatch.setattr(training, "MaskedResNet", Decoder)
    monkeypatch.setattr(training, "ShardDataset", dataset)
    source = tmp_path / "source.pt"
    torch.save(
        {
            "method": "ddpm_partial_reconstruction_bmad_flair",
            "epoch": 30,
            "config": {},
            "ema": Backbone().state_dict(),
        },
        source,
    )
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    config = {"epochs": 2, "batch_size": 2, "checkpoint_epochs": [1, 2], "source_sha256": digest}
    output = tmp_path / "encoder"

    def interrupt(progress):
        if progress["epoch"] == 1:
            raise RuntimeError("preemption")

    args = {
        "train_dir": tmp_path,
        "validation_dir": tmp_path,
        "output_dir": output,
        "config": config,
        "device": "cpu",
        "initial_checkpoint": source,
    }
    with pytest.raises(RuntimeError, match="preemption"):
        training.train(**args, on_checkpoint=interrupt)
    result = training.train(**args, resume_from_checkpoint=output / "encoder_pretrain_last.pt")
    assert result["epochs_completed"] == 2 and result["optimizer_steps"] == 2
    assert result["normal_validation_samples"] == 2
    state = torch.load(output / "encoder_pretrain_best.pt", weights_only=False)
    assert state["normal_only"] and state["dataset_revision"] == training.REVISION
    for key, value in state["encoder"].items():
        assert torch.equal(value, state["model"][f"encoder.{key}"])
    assert hashlib.sha256(source.read_bytes()).hexdigest() == digest
