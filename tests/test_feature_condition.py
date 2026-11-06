import copy
import hashlib

import numpy as np
import pytest

torch = pytest.importorskip("torch")
from torch import nn

from dacn.models.diffusion.feature_condition import (
    FeatureConditionedDDPM,
    MRIResNet18,
    update_feature_ema,
)
from dacn.models.diffusion.unet import UNetModel
from dacn.training import feature_condition as training
from dacn.training.ddpm_partial import _fixed_rng, build_diffusion, load_ema_checkpoint


def test_experiment_a_uses_verified_bytes_without_reopening(monkeypatch):
    import json

    from dacn.evaluation import feature_condition as evaluation

    keys = (
        "index_sha256",
        "partition",
        "timesteps_one_based",
        "sampling_base_seed",
        "partition_seed",
        "samples",
        "ema",
        "reconstruction_mode",
        "baseline_cache_sha256",
    )
    reference = dict.fromkeys(keys, "same")
    reference.update(
        checkpoint_sha256="checkpoint",
        checkpoint_epoch=1,
        metrics={},
        reconstruction_quality={"feature_condition": {}},
    )
    payload = json.dumps(reference).encode()

    class ChangingPath:
        def read_bytes(self):
            return payload

        def read_text(self):
            raise AssertionError("verified file reopened")

    class Digest:
        def hexdigest(self):
            return "a8e57ed824c9e6a00404d82296f15fb208eabb9d8d911c06fcd983dfb8b5a962"

    monkeypatch.setattr(evaluation.hashlib, "sha256", lambda value: Digest())
    assert (
        evaluation.compare_experiment_A(reference, ChangingPath())["A_checkpoint_sha256"]
        == "checkpoint"
    )


def trained_backbone():
    model = UNetModel(
        img_size=32,
        base_channels=32,
        channel_mults=(1, 2),
        num_res_blocks=1,
        attention_resolutions="8",
    )
    # Simulate trained (nonzero) residual/output convolutions, so condition
    # gradients can propagate, unlike an all-zero freshly initialized output.
    with torch.no_grad():
        for layer in model.modules():
            if isinstance(layer, nn.Conv2d) and not layer.weight.any():
                layer.weight.normal_(0, 0.01)
    return model


def test_standard_resnet18_dimensions_and_zero_injection_preserve_original():
    backbone = trained_backbone().eval()
    original = copy.deepcopy(backbone.state_dict())
    model = FeatureConditionedDDPM(backbone).eval()
    x, t = torch.randn(2, 1, 32, 32), torch.tensor([99, 299])
    with torch.inference_mode():
        assert model.encoder(x).shape == (2, 512)
        condition = model.encode(x)
        assert condition.shape == (2, 128)
        assert model.condition_projection(condition).shape == (
            2,
            backbone.time_embedding[-1].out_features,
        )
        assert torch.equal(model(x, t, condition), backbone(x, t))
        assert torch.equal(
            model(x, t, condition), backbone(x, t, condition_embedding=torch.zeros(2, 128))
        )
    assert all(torch.equal(value, backbone.state_dict()[key]) for key, value in original.items())
    assert sum(p.numel() for p in MRIResNet18().parameters()) == 11_170_240
    with torch.inference_mode():
        assert model(x, t, condition).shape == x.shape


def test_joint_gradients_reach_encoder_after_zero_gate_first_update():
    model = FeatureConditionedDDPM(trained_backbone()).train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    diffusion = build_diffusion({})
    images = torch.rand(2, 1, 32, 32) * 2 - 1
    first = training.feature_noise_loss(model, diffusion, images)
    first.backward()
    assert model.condition_projection.weight.grad.abs().sum() > 0
    assert all(p.grad is None or not p.grad.any() for p in model.encoder.parameters())
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    loss = training.feature_noise_loss(model, diffusion, images)
    loss.backward()
    for module in (model.unet, model.encoder, model.feature_projection, model.condition_projection):
        gradients = [p.grad for p in module.parameters() if p.grad is not None]
        assert gradients and all(torch.isfinite(grad).all() for grad in gradients)
        assert sum(float(grad.abs().sum()) for grad in gradients) > 0


def test_feature_loss_is_original_epsilon_mse():
    class Zero(nn.Module):
        def encode(self, image):
            return torch.zeros(len(image), 128)

        def forward(self, noisy, t, condition):
            return torch.zeros_like(noisy)

    diffusion, images = build_diffusion({}), torch.zeros(2, 1, 32, 32)
    with _fixed_rng(17, images.device):
        t = torch.randint(0, 1000, (2,))
        expected = diffusion.noise(images, t).square().mean()
    with _fixed_rng(17, images.device):
        observed = training.feature_noise_loss(Zero(), diffusion, images)
    torch.testing.assert_close(observed, expected, rtol=2e-7, atol=0)


def test_conditional_reverse_encodes_once_uses_fixed_condition_and_preserves_rng():
    class Zero(nn.Module):
        def __init__(self):
            super().__init__()
            self.encodes, self.conditions = 0, []

        def encode(self, image):
            self.encodes += 1
            return image.mean((2, 3))

        def forward(self, noisy, t, condition):
            self.conditions.append(condition)
            return torch.zeros_like(noisy)

    model, diffusion, images = Zero(), build_diffusion({}), torch.zeros(1, 1, 32, 32)
    before = torch.random.get_rng_state().clone()
    first, noisy = training.feature_reconstruct(
        model, diffusion, images, steps=3, seed=17, return_noisy=True
    )
    assert model.encodes == 1 and len(model.conditions) == 3
    assert all(value is model.conditions[0] for value in model.conditions)
    assert torch.equal(before, torch.random.get_rng_state())
    assert torch.equal(
        first, training.feature_reconstruct(model, diffusion, images, steps=3, seed=17)
    )
    assert first.shape == noisy.shape == images.shape
    # The public signature accepts only image/seed/steps, never lesion masks.
    with pytest.raises(TypeError):
        training.feature_reconstruct(model, diffusion, images, seed=17, mask=images)


def test_ema_copies_batchnorm_buffers_and_averages_parameters():
    model = FeatureConditionedDDPM(trained_backbone())
    ema = copy.deepcopy(model)
    with torch.no_grad():
        model.encoder.stem[1].running_mean.fill_(3)
        model.encoder.stem[1].num_batches_tracked.fill_(5)
        model.condition_projection.weight.fill_(2)
    update_feature_ema(ema, model, 0.5)
    assert torch.all(ema.encoder.stem[1].running_mean == 3)
    assert ema.encoder.stem[1].num_batches_tracked == 5
    assert torch.all(ema.condition_projection.weight == 1)


def test_conditional_checkpoints_rejected_by_unconditional_loader(tmp_path):
    path = tmp_path / "conditional.pt"
    torch.save({"method": training.METHOD, "schema_version": 1}, path)
    with pytest.raises(ValueError, match="schema or method"):
        load_ema_checkpoint(path, {}, "cpu")
    torch.save({"method": "ddpm_partial_reconstruction_bmad_flair"}, path)
    with pytest.raises(ValueError, match="feature-conditioned"):
        training.load_feature_checkpoint(path)


@pytest.mark.parametrize("encoder_kind", ["random", "normal_pretrained", "spatial"])
def test_training_checkpoint_resume_and_source_immutability(tmp_path, monkeypatch, encoder_kind):
    from dacn.models.diffusion import feature_condition as architecture
    from dacn.models.diffusion import spatial_condition as spatial_architecture

    pretrained, spatial = encoder_kind == "normal_pretrained", encoder_kind == "spatial"

    class TinyEncoder(nn.Sequential):
        def __init__(self):
            super().__init__(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(1, 512))

    class TinyUNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.time_embedding = nn.Sequential(nn.Linear(1, 128))
            self.embed = nn.Linear(128, 1)
            self.conv = nn.Conv2d(1, 1, 1)

        def forward(self, x, t, condition_embedding=None):
            embedding = self.time_embedding(t.float()[:, None] / 1000)
            if condition_embedding is not None:
                embedding = embedding + condition_embedding
            return self.conv(x) + self.embed(embedding)[:, :, None, None]

    class Data:
        labels = np.array([0, 0], dtype=np.uint8)
        records = (("shard-00000.npz", 0, 0), ("shard-00000.npz", 1, 0))

        def __len__(self):
            return 2

        def __getitem__(self, i):
            return torch.full((1, 256, 256), i / 10), torch.tensor(0)

    def dataset(*a, include_masks):
        assert include_masks is False
        return Data()

    monkeypatch.setattr(architecture, "MRIResNet18", TinyEncoder)

    class TinySpatial(nn.Module):
        def __init__(self, unet, *, pretrained=False):
            super().__init__()
            self.unet, self.encoder = unet, TinyEncoder()
            fusion = nn.Linear(512, 128)
            nn.init.zeros_(fusion.weight)
            nn.init.zeros_(fusion.bias)
            self.fusion_layers = nn.ModuleDict({"0": fusion})
            self.fusion_indices = [0]

        def encode(self, image):
            return self.encoder(image)

        def forward(self, image, t, condition):
            return self.unet(image, t, condition_embedding=self.fusion_layers["0"](condition))

    monkeypatch.setattr(spatial_architecture, "SpatialConditionedDDPM", TinySpatial)
    monkeypatch.setattr(training, "build_model", lambda *a: TinyUNet())
    monkeypatch.setattr(training, "ShardDataset", dataset)
    source = tmp_path / "source.pt"
    torch.save(
        {
            "method": "ddpm_partial_reconstruction_bmad_flair",
            "epoch": 30,
            "config": {},
            "ema": TinyUNet().state_dict(),
        },
        source,
    )
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    config = {"epochs": 2, "batch_size": 2, "checkpoint_epochs": [1, 2], "source_sha256": digest}
    if spatial:
        config["condition_architecture"] = "spatial_imagenet"
    if pretrained:
        encoder = TinyEncoder()
        with torch.no_grad():
            for parameter in encoder.parameters():
                parameter.fill_(0.02)
        encoder_dir = tmp_path / "encoder"
        encoder_dir.mkdir()
        path = encoder_dir / "encoder_pretrain_best.pt"
        torch.save(
            {
                "method": "resnet18_normal_masked_pretrain_v1",
                "normal_only": True,
                "dataset_revision": "350c74f2313e4e73ca65e7ffc6e9dcf74040b3b7",
                "encoder": encoder.state_dict(),
            },
            path,
        )
        config.update(
            encoder_pretrain_run_id="encoder",
            encoder_pretrain_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        )

    def interrupt(progress):
        if progress["epoch"] == 0:
            initial = torch.load(output / "feature_condition_last.pt", weights_only=False)
            assert not initial["optimizer"]["state"]
            assert not initial["model"][
                "fusion_layers.0.weight" if spatial else "condition_projection.weight"
            ].any()
            source_state = torch.load(source, weights_only=False)["ema"]
            for key, value in source_state.items():
                assert torch.equal(initial["model"][f"unet.{key}"], value)
            if pretrained:
                for key, value in encoder.state_dict().items():
                    assert torch.equal(initial["model"][f"encoder.{key}"], value)
        if progress["epoch"] == 1:
            raise RuntimeError("synthetic preemption")

    output = tmp_path / "new"
    with pytest.raises(RuntimeError, match="synthetic preemption"):
        training.train(
            train_dir=tmp_path,
            validation_dir=tmp_path,
            output_dir=output,
            config=config,
            device="cpu",
            initial_checkpoint=source,
            on_checkpoint=interrupt,
        )
    result = training.train(
        train_dir=tmp_path,
        validation_dir=tmp_path,
        output_dir=output,
        config=config,
        device="cpu",
        initial_checkpoint=source,
        resume_from_checkpoint=output / "feature_condition_last.pt",
    )
    assert result["epochs_completed"] == 2 and result["optimizer_steps"] == 2
    state = torch.load(output / "feature_condition_last.pt", weights_only=False)
    assert state["source_epoch"] == 30
    assert state["method"] == (
        spatial_architecture.METHOD
        if spatial
        else training.B_METHOD
        if pretrained
        else training.METHOD
    )
    assert state["config"]["encoder_lr"] == 1e-4
    assert (output / "feature_condition_epoch_002.pt").is_file()
    assert hashlib.sha256(source.read_bytes()).hexdigest() == digest
    loaded, _ = training.load_feature_checkpoint(output / "feature_condition_best.pt")
    assert not loaded.training
