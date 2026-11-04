"""Patch/context training must preserve initialization and prevent target leakage."""

import copy
import hashlib

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from dacn.training.ddpm_partial import build_diffusion
from dacn.training.patch_context import (
    _resolved_config,
    conditioned_input,
    expand_input,
    patch_noise_loss,
    random_patch_mask,
    sliding_reconstruct,
)


class ToyUNet(torch.nn.Module):
    out_channels = 1

    def __init__(self):
        super().__init__()
        self.down = torch.nn.ModuleList([torch.nn.Sequential(torch.nn.Conv2d(1, 2, 3, padding=1))])
        self.out = torch.nn.Conv2d(2, 1, 1)

    def forward(self, x, t):
        return self.out(self.down[0](x))


def test_extra_input_kernels_zero_and_original_prediction_preserved():
    source = ToyUNet()
    model = expand_input(copy.deepcopy(source))
    assert torch.equal(model.down[0][0].weight[:, :1], source.down[0][0].weight)
    assert not model.down[0][0].weight[:, 1:].any()
    x = torch.randn(2, 3, 16, 16)
    t = torch.tensor([1, 2])
    assert torch.allclose(model(x, t), source(x[:, :1], t), atol=1e-6)
    assert model(x, t).shape == (2, 1, 16, 16)


def test_clean_patch_does_not_leak_into_conditioning():
    clean = torch.zeros(1, 1, 256, 256)
    mask = torch.zeros_like(clean, dtype=torch.bool)
    mask[:, :, 100:164, 100:164] = True
    noisy = torch.randn_like(clean)
    first = conditioned_input(noisy, clean, mask)
    changed = clean.clone()
    changed[mask] = 0.9
    assert torch.equal(first, conditioned_input(noisy, changed, mask))
    assert not first[:, 1:2][mask].any()
    assert torch.equal(first[:, :1][mask], noisy[mask])


def test_patch_loss_ignores_targets_outside_mask():
    class Zero(torch.nn.Module):
        def forward(self, x, t):
            return torch.zeros_like(x[:, :1])

    clean = torch.zeros(2, 1, 256, 256)
    mask = torch.zeros_like(clean, dtype=torch.bool)
    mask[:, :, 100:164, 100:164] = True
    noise = torch.full_like(clean, 100)
    noise[mask] = 1
    loss = patch_noise_loss(
        Zero(), build_diffusion({}), clean, mask, noise=noise, timesteps=torch.tensor([99, 299])
    )
    assert loss.item() == 1


def test_mask_covers_foreground_and_never_padding():
    clean = torch.full((8, 1, 256, 256), -1.0)
    clean[:, :, 120:130, 120:130] = 0.5
    mask = random_patch_mask(clean, 64)
    assert torch.all(mask.sum(dim=(1, 2, 3)) == 64 * 64)
    assert torch.all((mask & (clean > -0.95)).sum(dim=(1, 2, 3)) > 0)
    assert not mask[:, :, :8].any() and not mask[:, :, :, :8].any()
    assert not mask[:, :, -8:].any() and not mask[:, :, :, -8:].any()


def test_masked_loss_backpropagates_to_new_channels():
    model = expand_input(ToyUNet())
    clean = torch.rand(2, 1, 256, 256) * 2 - 1
    mask = random_patch_mask(clean, 64)
    loss = patch_noise_loss(model, build_diffusion({}), clean, mask)
    loss.backward()
    grad = model.down[0][0].weight.grad
    assert torch.isfinite(loss) and torch.isfinite(grad).all()
    assert grad[:, 2:].abs().sum() > 0


def test_sliding_inference_covers_entire_crop_without_anomaly_mask(monkeypatch):
    from dacn.training import patch_context

    def fake_patch(model, diffusion, image, mask, **kwargs):
        return torch.where(mask, torch.ones_like(image), image)

    monkeypatch.setattr(patch_context, "reconstruct_patch", fake_patch)
    x = torch.zeros(1, 1, 256, 256)
    result = sliding_reconstruct(None, None, x)
    assert torch.all(result[:, :, 8:-8, 8:-8] == 1)
    assert not result[:, :, :8].any()


def test_pilot_budget_guard():
    with pytest.raises(ValueError):
        _resolved_config({"epochs": 30})


def test_one_step_matches_x0_formula_uses_one_call_and_preserves_context():
    from dacn.training.ddpm_partial import _fixed_rng
    from dacn.training.patch_context import reconstruct_patch

    class Zero(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.calls = 0

        def forward(self, conditioned, t):
            self.calls += 1
            assert t.tolist() == [99]
            assert torch.equal(conditioned[:, 2:3], mask.float())
            assert not conditioned[:, 1:2][mask].any()
            return torch.zeros_like(conditioned[:, :1])

    model = Zero()
    diffusion = build_diffusion({})
    image = torch.full((1, 1, 256, 256), 0.2)
    mask = torch.zeros_like(image, dtype=torch.bool)
    mask[:, :, 80:144, 80:144] = True
    t = torch.tensor([99])
    with _fixed_rng(17, image.device):
        noisy = diffusion.q_sample(image, t, diffusion.noise(image, t))
    alpha = float(np.cumprod(1 - diffusion.betas)[99])
    expected = (noisy / alpha**0.5).clamp(-1, 1)
    before = torch.random.get_rng_state().clone()
    result = reconstruct_patch(model, diffusion, image, mask, steps=100, seed=17, one_step=True)
    assert model.calls == 1
    assert torch.allclose(result[mask], expected[mask], atol=1e-6)
    assert torch.equal(result[~mask], image[~mask])
    assert torch.equal(before, torch.random.get_rng_state())


def test_finetune_saves_new_optimizer_and_separate_checkpoints(tmp_path, monkeypatch):
    from dacn.training import patch_context

    class TinyDataset(torch.utils.data.Dataset):
        def __init__(self, directory, split, *, include_masks):
            self.labels = np.zeros(2, dtype=np.uint8)
            self.records = [("shard-00000.npz", i, 0) for i in range(2)]

        def __len__(self):
            return 2

        def __getitem__(self, i):
            return torch.zeros(1, 256, 256), torch.tensor(0)

    monkeypatch.setattr(patch_context, "ShardDataset", TinyDataset)
    monkeypatch.setattr(patch_context, "build_model", lambda config, device: ToyUNet().to(device))
    initial = tmp_path / "source.pt"
    torch.save(
        {
            "epoch": 30,
            "method": "ddpm_partial_reconstruction_bmad_flair",
            "config": {},
            "ema": ToyUNet().state_dict(),
        },
        initial,
    )
    digest = hashlib.sha256(initial.read_bytes()).hexdigest()
    result = patch_context.train(
        train_dir=tmp_path,
        validation_dir=tmp_path,
        output_dir=tmp_path / "new_run",
        config={"epochs": 1, "checkpoint_epochs": [1], "initial_checkpoint_sha256": digest},
        device="cpu",
        initial_checkpoint=initial,
    )
    state = torch.load(tmp_path / "new_run/patch_context_last.pt", weights_only=False)
    assert result["optimizer_steps"] == 1 and state["source_epoch"] == 30
    assert state["model"]["down.0.0.weight"].shape[1] == 3
    assert state["optimizer"]["param_groups"][0]["lr"] == 1e-5
    assert all(value["step"].item() == 1 for value in state["optimizer"]["state"].values())
    assert (tmp_path / "new_run/patch_context_epoch_001.pt").is_file()
    assert (tmp_path / "new_run/private_normal_preview.npz").is_file()
    assert hashlib.sha256(initial.read_bytes()).hexdigest() == digest
