import hashlib

import pytest

torch = pytest.importorskip("torch")
from torch import nn

from dacn.models.diffusion import spatial_condition as architecture
from dacn.models.diffusion.spatial_condition import SpatialConditionedDDPM, SpatialResNet18
from dacn.models.diffusion.unet import UNetModel
from dacn.training.ddpm_partial import build_diffusion
from dacn.training.feature_condition import (
    feature_noise_loss,
    feature_reconstruct,
    load_feature_checkpoint,
)


def small_unet():
    model = UNetModel(
        256, 32, channel_mults=(1, 1, 1, 1, 1, 1), num_res_blocks=1, attention_resolutions="8"
    )
    with torch.no_grad():
        for layer in model.modules():
            if isinstance(layer, nn.Conv2d) and not layer.weight.any():
                layer.weight.normal_(0, 0.01)
    return model


def test_grayscale_imagenet_mapping_averages_conv1_and_requires_checksum(tmp_path, monkeypatch):
    encoder = SpatialResNet18()
    official = {}
    for key, value in encoder.state_dict().items():
        if key.startswith("stem.0."):
            target = key.replace("stem.0.", "conv1.", 1)
            value = torch.cat([value, value + 0.2, value - 0.4], dim=1)
        elif key.startswith("stem.1."):
            target = key.replace("stem.1.", "bn1.", 1)
        elif key.startswith("layers."):
            target = f"layer{int(key[7]) + 1}." + key[9:]
        else:
            raise AssertionError(key)
        official[target] = value
    official["fc.weight"] = torch.zeros(2, 512)
    official["fc.bias"] = torch.zeros(2)
    path = tmp_path / "synthetic_imagenet.pt"
    torch.save(official, path)
    monkeypatch.setattr(
        architecture, "WEIGHTS_SHA256", hashlib.sha256(path.read_bytes()).hexdigest()
    )
    encoder.load_imagenet(path)
    assert torch.equal(encoder.stem[0].weight, official["conv1.weight"].mean(1, keepdim=True))
    assert torch.equal(encoder.layers[3][1].conv2.weight, official["layer4.1.conv2.weight"])
    assert encoder.stem[0].in_channels == 1
    assert not any(isinstance(m, nn.AdaptiveAvgPool2d) for m in encoder.modules())
    monkeypatch.setattr(architecture, "WEIGHTS_SHA256", "0" * 64)
    with pytest.raises(ValueError, match="checksum"):
        encoder.load_imagenet(path)


def test_four_maps_zero_fusion_preserves_baseline_and_spatial_injection_changes_output():
    model = SpatialConditionedDDPM(small_unet()).eval()
    image = torch.randn(1, 1, 256, 256).clamp(-1, 1)
    t = torch.tensor([299])
    with torch.inference_mode():
        features = model.encode(image)
        assert [tuple(x.shape) for x in features] == [
            (1, 64, 64, 64),
            (1, 128, 32, 32),
            (1, 256, 16, 16),
            (1, 512, 8, 8),
        ]
        baseline = model.unet(image, t)
        assert torch.equal(model(image, t, features), baseline)
        for fusion in model.fusion_layers.values():
            fusion.weight.fill_(0.001)
        assert not torch.equal(model(image, t, features), baseline)
    with pytest.raises(ValueError, match="grayscale"):
        model.encoder(torch.zeros(1, 3, 256, 256))


def test_spatial_gradients_open_after_zero_gate_and_reverse_encodes_once():
    model = SpatialConditionedDDPM(small_unet()).train()
    diffusion, image = build_diffusion({}), torch.rand(1, 1, 256, 256) * 2 - 1
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    feature_noise_loss(model, diffusion, image).backward()
    assert sum(float(p.grad.abs().sum()) for p in model.fusion_layers.parameters()) > 0
    assert all(p.grad is None or not p.grad.any() for p in model.encoder.parameters())
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    feature_noise_loss(model, diffusion, image).backward()
    assert (
        sum(float(p.grad.abs().sum()) for p in model.encoder.parameters() if p.grad is not None) > 0
    )
    counts = []
    handle = model.encoder.register_forward_hook(lambda *args: counts.append(1))
    before = torch.random.get_rng_state().clone()
    output = feature_reconstruct(model, diffusion, image, seed=17, steps=3)
    handle.remove()
    assert len(counts) == 1 and torch.equal(before, torch.random.get_rng_state())
    assert output.shape == image.shape and torch.isfinite(output).all()


def test_spatial_checkpoint_inference_does_not_download_imagenet(tmp_path, monkeypatch):
    model = SpatialConditionedDDPM(small_unet())
    config = {
        "base_channels": 32,
        "channel_mults": [1, 1, 1, 1, 1, 1],
        "num_res_blocks": 1,
        "attention_resolutions": "8",
    }
    path = tmp_path / "spatial.pt"
    torch.save(
        {"method": architecture.METHOD, "ema": model.state_dict(), "backbone_config": config}, path
    )

    def no_download(*a, **kw):
        raise AssertionError("Inference must load checkpoint encoder, not ImageNet")

    monkeypatch.setattr(SpatialResNet18, "load_imagenet", no_download)
    loaded, _ = load_feature_checkpoint(path)
    assert not loaded.training
    assert torch.equal(loaded.encoder.stem[0].weight, model.encoder.stem[0].weight)
